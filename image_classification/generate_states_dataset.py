#!/usr/bin/env python3
"""
Generate synthetic 16x128 RGB PNG images for 8-state block detection training.

Image layout: 16 pixels tall, 128 pixels wide, divided into 8 blocks of 16x16.
Block N occupies columns [N*16, (N+1)*16-1].
Active state = white block (255,255,255), inactive = black block (0,0,0).

States 0-6 can be active in any combination -> 128 classes (named by active states).
State 7 (Normal) is active only when all states 0-6 are inactive -> 1 class.

Total: 129 classes. Output is organized as class subdirectories with a labels.txt
file, compatible with the STM32 model zoo image classification training pipeline.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def class_name_from_mask(mask: int) -> str:
    """Convert a 7-bit mask to class name. Bit i = state i active."""
    active = [i for i in range(7) if mask & (1 << i)]
    if not active:
        return "none"
    return "s" + "_s".join(str(s) for s in active)


def class_name_to_mask(name: str) -> int:
    """Convert class name back to 7-bit mask."""
    if name == "none":
        return 0
    mask = 0
    for part in name.split("_"):
        mask |= 1 << int(part[1:])
    return mask


def generate_image(mask: int, normal: bool = False) -> np.ndarray:
    """
    Generate a 16x128 RGB image as a numpy array (height=16, width=128, channels=3).

    Args:
        mask: 7-bit mask for states 0-6 (ignored if normal=True).
        normal: If True, only block 7 is white (Normal state).

    Returns:
        uint8 numpy array of shape (16, 128, 3).
    """
    img = np.zeros((16, 128, 3), dtype=np.uint8)

    if normal:
        img[:, 112:128, :] = 255
    else:
        for state in range(7):
            if mask & (1 << state):
                start_col = state * 16
                end_col = start_col + 16
                img[:, start_col:end_col, :] = 255

    return img


def generate_split(output_dir: Path, split_name: str, total_count: int,
                   class_names: list, seed: int):
    """Generate one split (train/val/test) of the dataset."""
    rng = np.random.default_rng(seed)
    n_classes = len(class_names)

    per_class = total_count // n_classes
    remainder = total_count % n_classes

    split_dir = output_dir / split_name
    print(f"  Generating '{split_name}' split: {total_count} images "
          f"({per_class}-{per_class + 1} per class)")

    for class_idx, class_name in enumerate(class_names):
        count = per_class + (1 if class_idx < remainder else 0)
        if count == 0:
            continue

        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(count):
            if class_name == "normal":
                img = generate_image(0, normal=True)
            else:
                mask = class_name_to_mask(class_name)
                img = generate_image(mask)

            img_path = class_dir / f"img_{img_idx:04d}.png"
            Image.fromarray(img).save(img_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic 8-state block images for image classification training."
    )
    parser.add_argument("--train-count", type=int, default=200,
                        help="Total number of training images (default: 200)")
    parser.add_argument("--val-count", type=int, default=50,
                        help="Total number of validation images (default: 50)")
    parser.add_argument("--test-count", type=int, default=50,
                        help="Total number of test images (default: 50)")
    parser.add_argument("--output-dir", type=str, default="./datasets/states_dataset",
                        help="Output directory (default: ./datasets/states_dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    # Build class list: 128 combinations of states 0-6 + normal
    combo_names = [class_name_from_mask(m) for m in range(128)]
    all_class_names = combo_names + ["normal"]  # 129 classes

    # Sort alphabetically for consistent label ordering (labels.txt and dirs)
    sorted_class_names = sorted(all_class_names)

    print(f"Output directory: {output_dir}")
    print(f"Classes: {len(sorted_class_names)} (128 combos + normal)")
    print(f"Image size: 16x128 RGB PNG")

    # Generate splits
    splits = [
        ("train", args.train_count),
        ("val", args.val_count),
        ("test", args.test_count),
    ]

    for split_name, total_count in splits:
        generate_split(output_dir, split_name, total_count,
                       sorted_class_names, args.seed)

    # Write labels.txt (alphabetically sorted)
    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for name in sorted_class_names:
            f.write(f"{name}\n")

    print(f"\nDone. Labels written to: {labels_path}")
    print(f"Directory structure:")
    print(f"  {output_dir}/")
    print(f"    labels.txt")
    for split_name, _ in splits:
        print(f"    {split_name}/")
        print(f"      <class_name>/")
        print(f"        img_XXXX.png")


if __name__ == "__main__":
    main()
