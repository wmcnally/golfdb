# GolfDB: A Video Database for Golf Swing Sequencing

## Introduction
GolfDB is a high-quality video dataset created for general recognition applications 
in the sport of golf, and specifically for the task of golf swing sequencing. 

This repo contains a simple PyTorch implemention of the SwingNet baseline model presented in the 
[paper](https://arxiv.org/abs/1903.06528).
The model was trained on split 1 **without any data augmentation** and achieved an average PCE of 71.5% (PCE
of 76.1% reported in the paper is credited to data augmentation including horizontal flipping and affine 
transformations). 

If you use this repo please cite the GolfDB paper:
```
@InProceedings{McNally_2019_CVPR_Workshops,
author = {McNally, William and Vats, Kanav and Pinto, Tyler and Dulhanty, Chris and McPhee, John and Wong, Alexander},
title = {GolfDB: A Video Database for Golf Swing Sequencing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```

## Dependencies
* [PyTorch 1.3](https://pytorch.org/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)

This code is tested under Ubuntu 18.04, CUDA 10.0, cuDNN 7.2 environment with an NVIDIA Titan Xp GPU

Python 3.6.5 version with Anaconda 3 is used for development.

## Getting Started
Run [generate_splits.py](./data/generate_splits.py) to convert the .mat dataset file to a dataframe and 
generate the 4 splits.

### Train
* I have provided the preprocessed video clips for a frame size of 160x160 (download 
[here](https://drive.google.com/file/d/1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ/view?usp=sharing)). 
Place 'videos_160' in the [data](./data/) directory. 
If you wish to use a different input configuration you must download the YouTube videos (URLs provided in 
dataset) and preprocess the videos yourself. I have provided [preprocess_videos.py](./data/preprocess_videos.py) to
help with that.

* Download the MobileNetV2 pretrained weights from this [repository](https://github.com/tonylins/pytorch-mobilenet-v2) 
and place 'mobilenet_v2.pth.tar' in the root directory. 

* Run [train.py](train.py)

### Evaluate
* Train your own model by following the steps above or download the pre-trained weights 
[here](https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view?usp=sharing). Create a 'models' directory
if not already created and place 'swingnet_1800.pth.tar' in this directory.

* Run [eval.py](eval.py). If using the pre-trained weights provided, the PCE should be 0.715.  

### Test your own video
* Follow steps above to download pre-trained weights. Then in the terminal: `python3 test_video.py -p test_video.mp4`

* **Note:** This code requires the sample video to be cropped and cut to bound a single golf swing. 
I used online video [cropping](https://ezgif.com/crop-video) and [cutting](https://online-video-cutter.com/) 
tools for my golf swing video. See test_video.mp4 for reference.

Good luck!
