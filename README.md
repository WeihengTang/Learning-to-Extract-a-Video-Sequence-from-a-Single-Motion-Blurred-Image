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
The code has been updated to work with Python 3.10+ and PyTorch 2.x.

### Setup (GPU with CUDA 12.1)
```bash
conda create -n deblur_py3 python=3.10 pillow numpy -y
conda activate deblur_py3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Setup (CPU only)
```bash
conda create -n deblur_py3 python=3.10 pillow numpy -y
conda activate deblur_py3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

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

## **Benchmarking on Adobe240**

### Step 1: Generate the benchmark set

This creates deterministic blurred inputs and ground-truth frames from the Adobe240 dataset.
For each sequence it selects the middle 17 frames, synthesises one motion-blurred image via gamma-aware averaging (`build_blur`), and saves the 7 evenly-spaced GT frames that correspond to the model's 7 outputs.

```bash
python generate_benchmark_set.py \
    --data_root /depot/chan129/users/harshana/Datasets/Adobe240/Adobe_240fps_dataset/Adobe_240fps_blur/full_sharp/ \
    --output_dir ./benchmark_data \
    --blur_len 17
```

### Step 2: Run inference on the benchmark set

```bash
for img in ./benchmark_data/input/*.png; do
    python demo.py --cuda --input "$img" --output_dir ./benchmark_data/output
done
```

### Output structure

```
benchmark_data/
  input/<seq_name>.png                      # blurred input
  gt/<seq_name>/frame_00.png … frame_06.png # 7 ground-truth frames
  output/<seq_name>/frame_00.png … frame_06.png # 7 inferred frames
```

GT and output folders share the same naming so they can be compared frame-by-frame directly.
Errors during data generation are logged to `benchmark_generation_error.log`.

## **Contact**
If you have any suggestions and questions, please send an email to jinmeiguang@gmail.com
