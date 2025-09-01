在 OmniScene 数据集上与 SVF-GS 进行对比

* 训练
```shell
python -W ignore train.py --config_file ./configs/nuscenes/comp_svfgs.yaml --novel_view_mode SF
python -W ignore train.py --config_file ./configs/nuscenes/comp_svfgs.yaml --novel_view_mode SF --weight_path ./results/main/models/weights_0_n  # 断点续训
```

* 推理
```shell
python -W ignore eval.py --config_file ./configs/nuscenes/comp_svfgs.yaml --weight_path ./results/main/models/weights_0_100000 --novel_view_mode SF
```

------

<p align="center">
  <h1 align="center">DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction from Flexible Surround-view Input</h1>
  <p align="center">
    <a href="https://fangzhou2000.github.io/">Qijian Tian</a><sup>1</sup>
    &nbsp;·&nbsp;
    <a href="https://tanxincs.github.io/">Xin Tan</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a href="https://scholar.google.com/citations?user=RN1QMPgAAAAJ">Yuan Xie</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a href="https://dmcv.sjtu.edu.cn/people/">Lizhuang Ma</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Shanghai Jiao Tong University
    <br>
    <sup>2</sup>East China Normal University
  </p>
  <h3 align="center">AAAI 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2409.12753">Paper</a> | <a href="https://fangzhou2000.github.io/projects/drivingforward/">Project Page</a> | <a href="https://drive.google.com/drive/folders/1IASOPK1RQeP-nLQvJUn7WQUtb_fwGlVS">Pretrained Models</a> </h3>
</p>

## Introduction

We propose a feed-forward Gaussian Splatting model that reconstructs driving scenes from flexible sparse surround-view input.

<img src=".\assets\framework.png">

Given sparse surround-view input from vehicle-mounted cameras, our model learns
scale-aware localization for Gaussian primitives from the small overlap of spatial and temporal context views. A Gaussian
network predicts other parameters from each image individually. This feed-forward pipeline enables the real-time reconstruction
of driving scenes and the independent prediction from single-frame images supports flexible input modes. At the inference stage,
we include only the depth network and the Gaussian network, as shown in the lower part of the figure.

## Installation

To get started, clone this project, create a conda virtual environment using Python 3.8, and install the requirements:

```bash
git clone git@github.com:dzp62442/DrivingForward.git
cd DrivingForward
git submodule update --init --recursive
conda create -n DrivingForward python=3.8
conda activate DrivingForward
pip install numpy==1.24.4 torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd models/gaussian/gaussian-splatting
pip install submodules/diff-gaussian-rasterization
cd ../../..
```

## Datasets

> 训练/推理过程中均需要加载尺度深度图，见 dataset/nuscenes_dataset.py 中的 generate_depth_map 函数

### nuScenes 
* Download [nuScenes](https://www.nuscenes.org/nuscenes) official dataset
* Place the dataset in `input_data/nuscenes/`

Data should be as follows:
```
├── input_data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
│   │   ├── v1.0-trainval
```

## Running the Code

> 作者发布的预训练权重是在 352x640 的分辨率下训练的
> 在 4090 主机上使用 112x200 的分辨率重新训练，取得了与论文中报告指标基本一致的结果

### Evaluation

Get the [pretrained models](https://drive.google.com/drive/folders/1IASOPK1RQeP-nLQvJUn7WQUtb_fwGlVS), save them to the root directory of the project, and unzip them.

For SF mode, run the following:
```shell
python -W ignore eval.py --weight_path ./weights_SF --novel_view_mode SF
```


For MF mode, run the following:
```shell
python -W ignore eval.py --weight_path ./weights_MF --novel_view_mode MF
```

### Training

For SF mode, run the following:
```shell
python -W ignore train.py --novel_view_mode SF
python -W ignore train.py --novel_view_mode SF --weight_path ./results/main/models/weights_n/  # 断点续训
```

For MF mode, run the following:
```shell
python -W ignore train.py --novel_view_mode MF
python -W ignore train.py --novel_view_mode MF --weight_path ./results/main/models/weights_n/  # 断点续训
```

## BibTeX
```
@inproceedings{tian2025drivingforward,
      title={DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction from Flexible Surround-view Input}, 
      author={Qijian Tian and Xin Tan and Yuan Xie and Lizhuang Ma},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2025}
}
```

## Acknowledgements

The project is partially based on some awesome repos: [MVSplat](https://github.com/donydchen/mvsplat), [GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian), and [VFDepth](https://github.com/42dot/VFDepth). Many thanks to these projects for their excellent contributions!
