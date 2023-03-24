
# TiG-BEV: Multi-view BEV 3D Object Detection via Target Inner-Geometry Learning

Official implementation of ['TiG-BEV: Multi-view BEV 3D Object Detection via Target Inner-Geometry Learning'](https://arxiv.org/pdf/2212.13979.pdf).


## Introduction

We propose TiG-BEV, a learning scheme of Target Inner-Geometry from the LiDAR modality into camera-based BEV detectors for both dense depth and BEV features. First, we introduce an inner-depth supervision module to learn the low-level relative depth relations between different foreground pixels. This enables the camerabased detector to better understand the object-wise spatial structures. Second, we design an inner-feature BEV distillation module to imitate the high-level semantics of different keypoints within foreground targets. To further alleviate the BEV feature gap between two modalities, we adopt both inter-channel and inter-keypoint distillation for feature-similarity modeling. With our target inner-geometry distillation, TiG-BEV can effectively boost BEVDepth by +2.3% NDS and +2.4% mAP, along with BEVDet by +9.1% NDS and +10.3% mAP on nuScenes val set.
![pipeline](figures/framework.png)


## Main Results
| Method | mAP      | NDS     | 
|--------|----------|---------|
| [**TiG-BEV-R50**](configs/tig_bev/tig_bev-r50.py)  | 33.8     | 37.5     |
| [**TiG-BEV4D-R50**](configs/tig_bev/tig_bev4d-r50.py) | 36.6     | 46.1    | 

We provide the model and log of TiG-BEV4D-R101-CBGS.

| Method | mAP      | NDS     |  Model | Log
|--------|----------|---------|--------|-------
| [**TiG-BEV4D-R101-CBGS**](configs/tig_bev/tig_bev4d-r101-CBGS.py) | 44.0   | 54.4   |[Google](https://drive.google.com/file/d/1sJFeI9byxHmNWrTCzeRcKOYzgGKXrgq2/view?usp=sharing)| [Google](https://drive.google.com/file/d/1sJFeI9byxHmNWrTCzeRcKOYzgGKXrgq2/view?usp=sharing) 


## Get Started
#### Installation and Data Preparation
Please see [getting_started.md](https://github.com/HuangJunJie2017/BEVDet/blob/master/docs/getting_started.md) in BEVDet.

## Acknowledgement
We sincerely thank these great open-sourced work below:
* [open-mmlab](https://github.com/open-mmlab) 
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
* [BEVDet](https://github.com/HuangJunJie2017/BEVDet/tree/master)
* [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
* [BEVFusion](https://github.com/mit-han-lab/bevfusion)  

## Bibtex
If you find this project useful, please cite:
```
@article{huang2022tig,
  title={TiG-BEV: Multi-view BEV 3D Object Detection via Target Inner-Geometry Learning},
  author={Huang, Peixiang and Liu, Li and Zhang, Renrui and Zhang, Song and Xu, Xinli and Wang, Baichao and Liu, Guoyi},
  journal={arXiv preprint arXiv:2212.13979},
  year={2022}
}
```
