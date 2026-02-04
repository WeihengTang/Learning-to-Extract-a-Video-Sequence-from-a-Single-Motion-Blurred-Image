#!/usr/bin/env python
"""Evaluate benchmark results by comparing output frames against ground truth.

Computes PSNR and SSIM per frame, per sequence, and overall averages.
LPIPS is computed if the 'lpips' package is installed.

Expected layout:
  benchmark_data/
    gt/<seq_name>/frame_00.png … frame_06.png
    output/<seq_name>/frame_00.png … frame_06.png

Usage:
  python evaluate_benchmark.py --data_dir ./benchmark_data
"""

import os
import glob
import argparse
import math
import csv

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# PSNR (numpy only)
# ---------------------------------------------------------------------------
def compute_psnr(img1, img2):
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(255.0 * 255.0 / mse)


# ---------------------------------------------------------------------------
# SSIM (numpy only, per-channel then averaged)
# ---------------------------------------------------------------------------
def _ssim_channel(ch1, ch2, k1=0.01, k2=0.03, L=255):
    """SSIM on a single channel (float64 arrays)."""
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu1 = ch1.mean()
    mu2 = ch2.mean()
    sigma1_sq = ch1.var()
    sigma2_sq = ch2.var()
    sigma12 = np.mean((ch1 - mu1) * (ch2 - mu2))

    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return num / den


def compute_ssim(img1, img2):
    """Compute mean SSIM across RGB channels."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    ssim_vals = []
    for c in range(img1.shape[2]):
        ssim_vals.append(_ssim_channel(img1[:, :, c], img2[:, :, c]))
    return np.mean(ssim_vals)


# ---------------------------------------------------------------------------
# LPIPS (optional — requires 'pip install lpips')
# ---------------------------------------------------------------------------
_lpips_model = None

def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex')
        _lpips_model.eval()
    return _lpips_model


def compute_lpips(img1, img2):
    """Compute LPIPS between two uint8 numpy images (H,W,3)."""
    import torch
    model = _get_lpips_model()
    # uint8 [0,255] -> float [-1,1], shape (1,3,H,W)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    with torch.no_grad():
        return model(t1, t2).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmark metrics')
    parser.add_argument(
        '--data_dir', type=str, default='./benchmark_data',
        help='Root benchmark directory (default: ./benchmark_data)'
    )
    parser.add_argument(
        '--csv', type=str, default=None,
        help='Optional path to save per-sequence results as CSV'
    )
    args = parser.parse_args()

    gt_root = os.path.join(args.data_dir, 'gt')
    out_root = os.path.join(args.data_dir, 'output')

    # Check for LPIPS availability
    try:
        import lpips
        use_lpips = True
        print('LPIPS available — will compute LPIPS (alex).')
    except ImportError:
        use_lpips = False
        print('LPIPS not installed — skipping. Install with: pip install lpips')

    # Discover sequences
    seq_names = sorted([
        d for d in os.listdir(gt_root)
        if os.path.isdir(os.path.join(gt_root, d))
           and os.path.isdir(os.path.join(out_root, d))
    ])

    if not seq_names:
        print(f'No matching sequences found in {gt_root} and {out_root}')
        return

    print(f'Found {len(seq_names)} sequences to evaluate.\n')

    all_psnr, all_ssim, all_lpips = [], [], []
    seq_results = []

    for seq_name in seq_names:
        gt_dir = os.path.join(gt_root, seq_name)
        out_dir = os.path.join(out_root, seq_name)

        gt_frames = sorted(glob.glob(os.path.join(gt_dir, 'frame_*.png')))
        out_frames = sorted(glob.glob(os.path.join(out_dir, 'frame_*.png')))

        if len(gt_frames) != len(out_frames):
            print(f'[SKIP] {seq_name}: GT has {len(gt_frames)} frames, '
                  f'output has {len(out_frames)} frames')
            continue

        seq_psnr, seq_ssim, seq_lpips = [], [], []

        for gt_path, out_path in zip(gt_frames, out_frames):
            gt_img = np.array(Image.open(gt_path).convert('RGB'))
            out_img = np.array(Image.open(out_path).convert('RGB'))

            # Crop to match if sizes differ (demo.py crops to multiples of 20)
            min_h = min(gt_img.shape[0], out_img.shape[0])
            min_w = min(gt_img.shape[1], out_img.shape[1])
            gt_img = gt_img[:min_h, :min_w]
            out_img = out_img[:min_h, :min_w]

            seq_psnr.append(compute_psnr(gt_img, out_img))
            seq_ssim.append(compute_ssim(gt_img, out_img))
            if use_lpips:
                seq_lpips.append(compute_lpips(gt_img, out_img))

        avg_psnr = np.mean(seq_psnr)
        avg_ssim = np.mean(seq_ssim)
        all_psnr.extend(seq_psnr)
        all_ssim.extend(seq_ssim)

        row = {'sequence': seq_name, 'psnr': avg_psnr, 'ssim': avg_ssim}

        if use_lpips:
            avg_lpips = np.mean(seq_lpips)
            all_lpips.extend(seq_lpips)
            row['lpips'] = avg_lpips
            print(f'{seq_name:40s}  PSNR: {avg_psnr:6.2f}  '
                  f'SSIM: {avg_ssim:.4f}  LPIPS: {avg_lpips:.4f}')
        else:
            print(f'{seq_name:40s}  PSNR: {avg_psnr:6.2f}  SSIM: {avg_ssim:.4f}')

        seq_results.append(row)

    # Overall averages
    print('-' * 70)
    overall = (f'{"OVERALL":40s}  PSNR: {np.mean(all_psnr):6.2f}  '
               f'SSIM: {np.mean(all_ssim):.4f}')
    if use_lpips:
        overall += f'  LPIPS: {np.mean(all_lpips):.4f}'
    print(overall)

    # Save CSV if requested
    if args.csv:
        fieldnames = ['sequence', 'psnr', 'ssim']
        if use_lpips:
            fieldnames.append('lpips')
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(seq_results)
            writer.writerow({
                'sequence': 'OVERALL',
                'psnr': np.mean(all_psnr),
                'ssim': np.mean(all_ssim),
                **({'lpips': np.mean(all_lpips)} if use_lpips else {}),
            })
        print(f'\nResults saved to {args.csv}')


if __name__ == '__main__':
    main()
