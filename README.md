This repository is a PyTorch implementation of the paper "Learning to Extract a Video Sequence from a Single Motion-Blurred Image" from CVPR 2018 [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jin_Learning_to_Extract_CVPR_2018_paper.pdf)[[arxiv]](https://arxiv.org/pdf/1804.04065.pdf)[[supplementary]](https://github.com/MeiguangJin/test/blob/master/supplementary.pdf)[[video examples]](https://github.com/MeiguangJin/test/blob/master/video_demo.zip).

If you find our work useful in your research or publication, please cite our work:

@InProceedings{Jin_2018_CVPR,  
author = {Jin, Meiguang and Meishvili, Givi and Favaro, Paolo},  
title = {Learning to Extract a Video Sequence From a Single Motion-Blurred Image},  
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {June},  
year = {2018}  
}  
## **Requirements**
The code has been updated to work with Python 3.8+ and PyTorch 1.9+.

### Setup
```bash
conda create -n deblur_py3 python=3.8 pytorch=1.9.0 torchvision=0.10.0 cpuonly pillow numpy -c pytorch -y
conda activate deblur_py3
```

For GPU support, replace `cpuonly` with the appropriate CUDA version (e.g., `cudatoolkit=11.1`).

## **Test**
Download pretrained models from [here](https://www.dropbox.com/sh/r0n9x6uz1ke8iuy/AADJBQBf9E2UMzG4Gt2Az-Qza?dl=0) and place them in the `models/` directory.
```bash
python demo.py --input examples/image_1_blurry.png        # CPU
python demo.py --cuda --input examples/image_1_blurry.png # GPU
```

The output will be 7 extracted video frames saved with suffixes `-esti1.png` through `-esti7.png`.

## **Updates (2025)**
The original code was written for Python 2.7 and PyTorch 0.3.0, which are no longer available. The following changes were made for compatibility with modern Python/PyTorch:

- **demo.py**: Replaced deprecated `Variable(volatile=True)` with `torch.no_grad()`, added `map_location` for CPU support, use `strict=False` for checkpoint loading
- **model.py**: Added `track_running_stats=True` to all `InstanceNorm2d` layers (default changed from `True` to `False` in PyTorch 0.4)
- **utils.py**: Replaced deprecated `Image.ANTIALIAS` with `Image.Resampling.LANCZOS`, removed `Variable` wrapper usage  
## **Training**  
To be updated  

## **Contact**
If you have any suggestions and questions, please send an email to jinmeiguang@gmail.com
