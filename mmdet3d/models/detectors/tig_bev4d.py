import torch
from torch.nn import functional as F
from mmcv.runner import force_fp32
from .. import builder
from ..builder import DETECTORS
from .centerpoint import CenterPoint
import torch.nn as nn


class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2norm = nn.BatchNorm2d(dim_out)      
    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x


@DETECTORS.register_module()
class TiG_BEV4D(CenterPoint):
    r"""TiG-BEV: Multi-view BEV 3D Object Detection via Target Inner-Geometry Learning.
    
    Please refer to the `paper <https://arxiv.org/abs/2212.13979>`
    
    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
        code_mode (str): The value must be 'bevdepth' or 'bevdet'. Used to determine the specific paradigm. The main difference is extracting features.
        x_sample_num (int): Sampling points in x-direction of the target bounding box. 
        y_sample_num (int): Sampling points in y-direction of the target bounding box. The total number N of key points is x_sample_num * y_sample_num.
        embed_channels (list[int]): Use to align channel dimension when transfering knowledge.
            (1) 'embed_channels = []' means not using this module.
            (2) 'embed_channels = [input_dim,output_dim]' means aligning input feature dims with output dims.
        inter_keypoint_weight (float): The weight of inter_keypoint distillation loss.
        inter_channel_weight (float): The weight of inter_channel distillation loss.
        inner_depth_weight (float): The weight of inner_depth loss.
        enlarge_width (float): The value of the enlarging size for the target bounding box. If the value < 0, it means keeping the origin size.
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        pre_process_neck (dict | None): Configuration dict of BEV pre-process neck net.
        detach (bool): Whether to back propagation the other frame feature's gradient.
        before (bool): Pre-process before aligning features or after.
        interpolation_mode (str): The interpolation method for grid sampling when aligning features. Default is 'bilinear'.
    """
    def __init__(self, 
                 img_view_transformer, 
                 img_bev_encoder_backbone, 
                 img_bev_encoder_neck, 
                 code_mode = 'bevdepth',
                 x_sample_num=16,
                 y_sample_num=16,
                 embed_channels = [],
                 inter_keypoint_weight = 0.0,
                 inter_channel_weight=0.0,
                 inner_depth_weight = 0.0,
                 enlarge_width = -1,
                 pre_process=None,
                 pre_process_neck=None, 
                 detach=True, 
                 before=False, 
                 interpolation_mode='bilinear',
                 **kwargs):
        super(TiG_BEV4D, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.before = before
        self.code_mode = code_mode
        self.interpolation_mode = interpolation_mode
        self.inter_keypoint_weight = inter_keypoint_weight
        self.inter_channel_weight = inter_channel_weight
        self.inner_depth_weight = inner_depth_weight
        self.enlarge_width = enlarge_width
        if len(embed_channels) == 2:
            self.embed = nn.Conv2d(embed_channels[0], embed_channels[1], kernel_size=1, padding=0)
        self.x_sample_num = x_sample_num
        self.y_sample_num = y_sample_num
        
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.pre_process_neck = pre_process_neck is not None
        if self.pre_process_neck:
            self.pre_process_neck_net = builder.build_neck(pre_process_neck)
        self.detach = detach
        
        if self.pts_middle_encoder:
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
            self.pts_middle_encoder.eval()

        if self.pts_backbone:
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
            self.pts_backbone.eval()

        if self.pts_neck:
            for param in self.pts_neck.parameters():
                param.requires_grad = False
            self.pts_neck.eval()
        
    def image_encoder(self,img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
        if type(x) == list:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        return x
    
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.pts_middle_encoder:
            return None
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def extract_img_feat_bevdet(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans,
                                                          intrins, post_rots,
                                                          post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            #BEVDet
            x = self.img_view_transformer.depthnet(x)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)
            depth_digit = x[:, :self.img_view_transformer.D]
            depth = self.img_view_transformer.get_depth_dist(depth_digit)
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)
            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                              rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return x, depth_digit_list[0]
    
    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans,
                                                          intrins, post_rots,
                                                          post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            # BEVDepth
            img_feat = self.img_view_transformer.featnet(x)
            depth_feat = x
            cam_params = torch.cat([intrin.reshape(B * N, -1),
                                   post_rot.reshape(B * N, -1),
                                   post_tran.reshape(B * N, -1),
                                   rot.reshape(B * N, -1),
                                   tran.reshape(B * N, -1)], dim=1)
            depth_feat = self.img_view_transformer.se(depth_feat,
                                                      cam_params)
            depth_feat = self.img_view_transformer.extra_depthnet(depth_feat)[0]
            depth_feat = self.img_view_transformer.dcn(depth_feat)
            depth_digit = self.img_view_transformer.depthnet(depth_feat)
            depth = self.img_view_transformer.get_depth_dist(depth_digit)
            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans,
                                 self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin,
                                                          post_rot, post_tran)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                              rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        
        return x, depth_digit_list[0]
    
    @force_fp32()
    def shift_feature(self, input, trans, rots):
        n, c, h, w = input.shape
        _,v,_ =trans[0].shape

        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)
        grid = grid

        # get transformation from current lidar frame to adjacent lidar frame
        # transformation from current camera frame to current lidar frame
        c02l0 = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        c02l0[:,:,:3,:3] = rots[0]
        c02l0[:,:,:3,3] = trans[0]
        c02l0[:,:,3,3] = 1

        # transformation from adjacent camera frame to current lidar frame
        c12l0 = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)
        c12l0[:,:,:3,:3] = rots[1]
        c12l0[:,:,:3,3] = trans[1]
        c12l0[:,:,3,3] =1

        # transformation from current lidar frame to adjacent lidar frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:,0,:,:].view(n,1,1,4,4)

        l02l1 = l02l1[:,:,:,[True,True,False,True],:][:,:,:,:,[True,True,False,True]]

        feat2bev = torch.zeros((3,3),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 2] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 2] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1,3,3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
        return output
    
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        if self.code_mode == 'bevdepth': 
            img_feats, depth = self.extract_img_feat(img, img_metas)
        elif self.code_mode == 'bevdet': 
            img_feats, depth = self.extract_img_feat_bevdet(img, img_metas)
        if points:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = None
        return (img_feats, pts_feats, depth)
    
    @force_fp32()
    def get_depth_loss(self, depth_gt, depth):
        B, N, H, W = depth_gt.shape
        loss_weight = (~(depth_gt == 0)).reshape(B, N, 1, H, W).expand(B, N,
                                                                       self.img_view_transformer.D,
                                                                       H, W)
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                   /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                              self.img_view_transformer.D).to(torch.long)
        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32)
        depth = depth.sigmoid().view(B, N, self.img_view_transformer.D, H, W)

        loss_depth = F.binary_cross_entropy(depth, depth_gt_logit,
                                            weight=loss_weight)
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth
    
    @force_fp32()
    def get_inner_depth_loss(self,fg_gt, depth_gt,depth):
        """Calculate the relative depth values within the foreground area of each target object, and supervise the inner-depth prediction of the student detector."""
        B, N, H, W = fg_gt.shape
        depth_prob = depth.softmax(dim=1) #B*N,D,H,W
        discrete_depth = torch.arange(*self.img_view_transformer.grid_config['dbound'], dtype=torch.float,device=depth_prob.device).view(1,-1, 1, 1).expand(B*N,-1, H,W)
        depth_estimate = torch.sum(depth_prob * discrete_depth,dim=1) #B*N,H,W 
        fg_gt = fg_gt.reshape(B*N,H,W)
        depth_gt = depth_gt.reshape(B*N,H,W) 
        pos_mask = (fg_gt>0)
        num_pos = torch.clip(pos_mask.sum(),min=1.0)
        
        fg_gt = F.one_hot(fg_gt.long()).permute(0,3,1,2)[:,1:,:,:]
         
        depth_estimate_fg = fg_gt*(depth_estimate.unsqueeze(1)) #B*N,num_bbox,H,W
        depth_fg_gt = fg_gt*(depth_gt.unsqueeze(1)) #B*N,num_bbox,H,W
        
        depth_dis = torch.abs(depth_estimate_fg - depth_fg_gt) + 99 * (1 - fg_gt)
        depth_min_dis_ind = depth_dis.view(B*N,-1,H*W).argmin(-1,keepdim=True)
        
        depth_estimate_fg_min = fg_gt * (depth_estimate_fg.view(B*N,-1,H*W).gather(-1,depth_min_dis_ind)).view(B*N,-1,1,1)
        depth_fg_gt_min = fg_gt * (depth_fg_gt.view(B*N,-1,H*W).gather(-1,depth_min_dis_ind)).view(B*N,-1,1,1)

        
        diff_pred = depth_estimate - depth_estimate_fg_min.sum(1).detach() #B*N,H,W 
        diff_gt = depth_gt - depth_fg_gt_min.sum(1)
        loss_inner_depth = torch.sum(F.mse_loss(diff_pred,diff_gt.detach(),reduction='none') * pos_mask) / num_pos * self.inner_depth_weight
        
        return loss_inner_depth
   
    def get_gt_sample_grid(self,corner_points2d):
        """Use corners to generate the grid for sampling key points inside the target bounding box
        
                corner2
                dW_x dH_x
                _ _ _ _                      
                |   /\ | dH_y                      
            dW_y|  /H \|                           
                | /   /|corner1
        corner4 |/  W/ |
                |\  /  | dW_y
            dH_y|_\/___|
                dH_x dW_x
                corner3
                
        """
        dH_x, dH_y = corner_points2d[0] - corner_points2d[1] 
        dW_x, dW_y = corner_points2d[0] - corner_points2d[2]
        raw_grid_x = torch.linspace(corner_points2d[0][0], corner_points2d[1][0], self.x_sample_num).view(1,-1).repeat(self.y_sample_num,1)
        raw_grid_y = torch.linspace(corner_points2d[0][1],corner_points2d[2][1], self.y_sample_num).view(-1,1).repeat(1,self.x_sample_num)
        raw_grid = torch.cat((raw_grid_x.unsqueeze(2),raw_grid_y.unsqueeze(2)), dim=2)
        raw_grid_x_offset = torch.linspace(0,-dW_x,self.x_sample_num).view(-1,1).repeat(1,self.y_sample_num)
        raw_grid_y_offset = torch.linspace(0,-dH_y,self.y_sample_num).view(1,-1).repeat(self.x_sample_num,1)
        raw_grid_offset = torch.cat((raw_grid_x_offset.unsqueeze(2),raw_grid_y_offset.unsqueeze(2)),dim=2)
        grid = raw_grid + raw_grid_offset #X_sample,Y_sample,2
        grid[:,:,0] = torch.clip(((grid[:,:,0] - (self.img_view_transformer.bx[0].to(grid.device) - self.img_view_transformer.dx[0].to(grid.device) / 2.)
                       ) / self.img_view_transformer.dx[0].to(grid.device) / (self.img_view_transformer.nx[0].to(grid.device)-1))*2.0 - 1.0 ,min=-1.0,max=1.0)
        grid[:,:,1] = torch.clip(((grid[:,:,1] - (self.img_view_transformer.bx[1].to(grid.device) - self.img_view_transformer.dx[1].to(grid.device) / 2.)
                       ) / self.img_view_transformer.dx[1].to(grid.device) / (self.img_view_transformer.nx[1].to(grid.device)-1))*2.0 - 1.0 ,min=-1.0,max=1.0)
        
        return grid.unsqueeze(0)  
    
    def get_inner_feat(self,gt_bboxes_3d,img_feats,pts_feats):
        """Use grid to sample features of key points"""
        device = img_feats.device
        dtype = img_feats[0].dtype
        
        img_feats_sampled_list = []
        pts_feats_sampled_list = []
        
        for sample_ind in torch.arange(len(gt_bboxes_3d)):
            img_feat = img_feats[sample_ind].unsqueeze(0)   #1,C,H,W
            pts_feat = pts_feats[sample_ind].unsqueeze(0)   #1,C,H,W

            bbox_num, corner_num, point_num = gt_bboxes_3d[sample_ind].corners.shape
            
            for bbox_ind in torch.arange(bbox_num):
                if self.enlarge_width>0:
                    gt_sample_grid = self.get_gt_sample_grid(gt_bboxes_3d[sample_ind].enlarged_box(self.enlarge_width).corners[bbox_ind][[0,2,4,6],:-1]).to(device)
                else:
                    gt_sample_grid = self.get_gt_sample_grid(gt_bboxes_3d[sample_ind].corners[bbox_ind][[0,2,4,6],:-1]).to(device)  #1,sample_y,sample_x,2
                
                img_feats_sampled_list.append(F.grid_sample(img_feat, grid=gt_sample_grid, align_corners=False, mode='bilinear'))#'bilinear')) #all_bbox_num,C,y_sample,x_sample
                pts_feats_sampled_list.append(F.grid_sample(pts_feat, grid=gt_sample_grid, align_corners=False, mode='bilinear'))#'bilinear')) #all_bbox_num,C,y_sample,x_sample

        return torch.cat(img_feats_sampled_list,dim=0), torch.cat(pts_feats_sampled_list,dim=0)
   
    @force_fp32()
    def get_inter_keypoint_loss(self,img_feats_kd, pts_feats_kd):
        """Calculate the inter-keypoint similarities, guide the student keypoint features to mimic the feature relationships between different N keypoints of the teacher’s"""
        C_img = img_feats_kd.shape[1] 
        C_pts = pts_feats_kd.shape[1]
        N = self.x_sample_num*self.y_sample_num
        
        img_feats_kd = img_feats_kd.view(-1,C_img,N).permute(0,2,1).matmul(
            img_feats_kd.view(-1,C_img,N)) #-1,N,N
        pts_feats_kd = pts_feats_kd.view(-1,C_pts,N).permute(0,2,1).matmul(
            pts_feats_kd.view(-1,C_pts,N))
        
        img_feats_kd = F.normalize(img_feats_kd,dim=2)
        pts_feats_kd = F.normalize(pts_feats_kd,dim=2)
        
        loss_inter_keypoint = F.mse_loss(img_feats_kd, pts_feats_kd, reduction='none')
        loss_inter_keypoint = loss_inter_keypoint.sum(-1)
        loss_inter_keypoint = loss_inter_keypoint.mean()
        loss_inter_keypoint = self.inter_keypoint_weight * loss_inter_keypoint
        return loss_inter_keypoint 
    
    @force_fp32()
    def get_inter_channel_loss(self,img_feats_kd, pts_feats_kd):
        """Calculate the inter-channel similarities, guide the student keypoint features to mimic the channel-wise relationships of the teacher’s"""
        img_feats_kd = self.embed(img_feats_kd)
        C_img = img_feats_kd.shape[1] 
        C_pts = pts_feats_kd.shape[1]
        N = self.x_sample_num*self.y_sample_num
        
        img_feats_kd = img_feats_kd.view(-1,C_img,N).matmul(
            img_feats_kd.view(-1,C_img,N).permute(0,2,1)) #-1,N,N
        pts_feats_kd = pts_feats_kd.view(-1,C_pts,N).matmul(
            pts_feats_kd.view(-1,C_pts,N).permute(0,2,1))
        
        img_feats_kd = F.normalize(img_feats_kd,dim=2)
        pts_feats_kd = F.normalize(pts_feats_kd,dim=2)
        
        loss_inter_channel = F.mse_loss(img_feats_kd, pts_feats_kd, reduction='none')
        loss_inter_channel = loss_inter_channel.sum(-1)
        loss_inter_channel = loss_inter_channel.mean()
        loss_inter_channel = self.inter_channel_weight * loss_inter_channel
        return loss_inter_channel 
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """   
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        depth_gt = img_inputs[6]
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        losses = dict()
        if self.img_view_transformer.loss_depth_weight > 0:
            loss_depth = self.get_depth_loss(depth_gt, depth)
            losses.update({'loss_depth':loss_depth})
        if self.inner_depth_weight > 0:
            fg_gt = img_inputs[7]
            fg_gt = torch.split(fg_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
            loss_inner_depth = self.get_inner_depth_loss(fg_gt,depth_gt, depth)
            losses.update({'loss_inner_depth':loss_inner_depth})
            
        img_feats_kd = img_feats[0]
        if pts_feats:
            pts_feats_kd = pts_feats[0]
        

        losses_pts = self.forward_pts_train([img_feats[0]], gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
       
        if self.inter_keypoint_weight > 0 or self.inter_channel_weight > 0:
            img_feats_kd, pts_feats_kd = self.get_inner_feat(gt_bboxes_3d,img_feats_kd, pts_feats_kd)
        
        if self.inter_keypoint_weight > 0:
            loss_inter_keypoint = self.get_inter_keypoint_loss(img_feats_kd, pts_feats_kd)
            losses.update({'loss_inter_keypoint':loss_inter_keypoint})
            
        if self.inter_channel_weight > 0:
            loss_inter_channel = self.get_inter_channel_loss(img_feats_kd, pts_feats_kd)
            losses.update({'loss_inter_channel':loss_inter_channel})
       
        losses.update(losses_pts)
        return losses
    
    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0],list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],**kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)
            
    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_metas = [img_metas] * len(imgs)
        points = [None] * len(imgs)
        pts_feats = []
        for point,img_meta,img in zip(points, img_metas, imgs):
            pts_feat, _, _ = self.extract_feat(point, img=img, img_metas=img_meta)
            pts_feats.append(pts_feat)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def simple_test(self, points, img_metas, img=None, rescale=False,gt_bboxes_3d=None,gt_labels_3d=None):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        img_metas=[dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
