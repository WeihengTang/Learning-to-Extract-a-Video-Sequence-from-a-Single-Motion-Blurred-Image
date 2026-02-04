#!/usr/bin/env python
"""Generate a deterministic benchmark set from the Adobe240 dataset.

For each sequence found under full_sharp/:
  1. Select the middle 17 frames and synthesize one motion-blurred image
  2. Pick the 7 evenly-spaced GT frames that correspond to the model's
     7 outputs (esti1..esti7)
  3. Save blurred input to ./benchmark_data/input/<seq_name>.png
  4. Save the 7 GT frames to ./benchmark_data/gt/<seq_name>/frame_00..06.png
"""

import os
import sys
import glob
import shutil
import logging
import argparse

import numpy as np
from PIL import Image

from motionblur_dataset import build_blur

DEFAULT_DATA_ROOT = (
    '/depot/chan129/users/harshana/Datasets/Adobe240/'
    'Adobe_240fps_dataset/Adobe_240fps_blur/full_sharp/'
)

NUM_MODEL_OUTPUTS = 7  # the model always produces 7 frames


def find_sequence_dirs(root):
    """Recursively find leaf directories containing .png images."""
    seq_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        png_files = [f for f in filenames if f.lower().endswith('.png')]
        if png_files and not dirnames:
            seq_dirs.append(dirpath)
    return sorted(seq_dirs)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a deterministic Adobe240 benchmark set'
    )
    parser.add_argument(
        '--data_root', type=str, default=DEFAULT_DATA_ROOT,
        help='Path to the full_sharp directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./benchmark_data',
        help='Output directory (default: ./benchmark_data)'
    )
    parser.add_argument(
        '--blur_len', type=int, default=17,
        help='Number of frames used to synthesize the blur (default: 17)'
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Error logging: all tracebacks go to a file, stdout gets short msgs
    # ------------------------------------------------------------------
    logging.basicConfig(
        filename='benchmark_generation_error.log',
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    input_dir = os.path.join(args.output_dir, 'input')
    gt_dir = os.path.join(args.output_dir, 'gt')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Indices of the 7 evenly-spaced GT frames within the blur window
    # e.g. for blur_len=17: [0, 3, 5, 8, 11, 13, 16]
    gt_indices = np.round(
        np.linspace(0, args.blur_len - 1, NUM_MODEL_OUTPUTS)
    ).astype(int).tolist()

    seq_dirs = find_sequence_dirs(args.data_root)
    print(f'Found {len(seq_dirs)} sequences in {args.data_root}')
    print(f'Blur length: {args.blur_len}, GT indices: {gt_indices}')

    success_count = 0
    fail_count = 0

    for i, seq_dir in enumerate(seq_dirs):
        seq_name = os.path.relpath(seq_dir, args.data_root).replace(os.sep, '_')

        try:
            frame_paths = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
            n = len(frame_paths)
            if n < args.blur_len:
                raise ValueError(
                    f'Sequence has {n} frames, need at least {args.blur_len}'
                )

            # Deterministic: select the middle blur_len frames
            mid = n // 2
            half = args.blur_len // 2
            start = mid - half
            blur_paths = frame_paths[start:start + args.blur_len]

            # Generate motion-blurred input image
            blur_img = build_blur(blur_paths)
            blur_path = os.path.join(input_dir, f'{seq_name}.png')
            Image.fromarray(blur_img).save(blur_path)

            # Save the 7 evenly-spaced ground-truth frames
            seq_gt_dir = os.path.join(gt_dir, seq_name)
            os.makedirs(seq_gt_dir, exist_ok=True)
            for j, idx in enumerate(gt_indices):
                shutil.copy2(
                    blur_paths[idx],
                    os.path.join(seq_gt_dir, f'frame_{j:02d}.png')
                )

            success_count += 1
            print(f'[{i + 1}/{len(seq_dirs)}] OK: {seq_name} ({n} frames)')

        except Exception:
            fail_count += 1
            logging.exception(f'Failed to process sequence: {seq_dir}')
            print(
                f'[{i + 1}/{len(seq_dirs)}] FAILED: {seq_name} '
                f'-- see benchmark_generation_error.log'
            )

    print(f'\nDone. Success: {success_count}, Failed: {fail_count}')
    print(f'Blurred inputs saved to:      {input_dir}')
    print(f'Ground truth frames saved to:  {gt_dir}')


if __name__ == '__main__':
    main()
