_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/default_runtime.py']
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
downsample = 16
data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}

# Model
grid_config={
        'xbound': [-51.2, 51.2, 0.8],
        'ybound': [-51.2, 51.2, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans=64

model = dict(
    type='TiG_BEV4D',
    detach=True,
    before=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(type='ViewTransformerLSSBEVDepth',
                              loss_depth_weight=100.0,
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=numC_Trans,
                              extra_depth_net=dict(type='ResNetForBEVDet', numC_input=256,
                                                   num_layer=[3,], num_channels=[256,], stride=[1,],)),
    img_bev_encoder_backbone = dict(type='ResNetForBEVDet',  numC_input=128,
                                    num_channels=[128, 256, 512]),
    img_bev_encoder_neck = dict(type='FPN_LSS',
                                in_channels=numC_Trans*8+numC_Trans*2,
                                out_channels=256,
                                ),
    pre_process = dict(type='ResNetForBEVDet',numC_input=numC_Trans,
                      num_layer=[2,], num_channels=[64,], stride=[1,],
                      backbone_output_ids=[0,]),
    
    #############################################################################################
    
    x_sample_num=24,
    y_sample_num=24,
    embed_channels = [256,512],
    inter_keypoint_weight = 100.0,
    enlarge_width = 1.6,
    inner_depth_weight = 1.0, 
    inter_channel_weight = 10.0, 
    
    #############################################################################################
    
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,max_num_points=10, voxel_size=voxel_size, max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    #############################################################################################
    
    pts_bbox_head=dict(
        type='CenterHead',
        task_specific=True,
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,

            # Scale-NMS
            nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0,1.0], [4.5, 9.0]]
        )))


# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
         sequential=True, aligned=True, trans_only=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),

    dict(type='PointToMultiViewDepthWithGTIndex', grid_config=grid_config,downsample=downsample),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points','img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config,
         sequential=True, aligned=True, trans_only=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='PointToMultiViewDepth', grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'],
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'adjacent', 'adjacent_type',)
                 )
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config,
         sequential=True, aligned=True, trans_only=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='PointToMultiViewDepth', grid_config=grid_config),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train_4d_interval3_max60.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            box_type_3d='LiDAR',
            img_info_prototype='bevdet_sequential',
            speed_mode='abs_dis',
            max_interval=9,
            min_interval=2,
            prev_only=True,
            fix_direction=True)),
        val=dict(
            pipeline=test_pipeline, 
            classes=class_names,
            ann_file=data_root + 'nuscenes_infos_val_4d_interval3_max60.pkl',
            modality=input_modality, img_info_prototype='bevdet_sequential',),
        test=dict(
            pipeline=test_pipeline, 
            classes=class_names,
            ann_file=data_root + 'nuscenes_infos_val_4d_interval3_max60.pkl',
            modality=input_modality,
            img_info_prototype='bevdet_sequential',
            speed_mode='abs_dis',
            max_interval=10,
            fix_direction=True,))


# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[16,19])
runner = dict(type='EpochBasedRunner', max_epochs=20)
load_from = 'data/nuscenes/model_pth/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20210815_085857-9ba7f3a5.pth'
evaluation = dict(interval=25, pipeline=eval_pipeline)


