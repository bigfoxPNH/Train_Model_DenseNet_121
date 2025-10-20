#!/usr/bin/env python3
"""
Enhance images exported by skull_crop.py (or any folder of images).

Operations, in order:
1) Light Gaussian blur (sigma ~0.6) to suppress noise without losing details
2) CLAHE to boost local contrast (clipLimit ~2.0, tileGridSize 8x8)

The tool preserves the input folder structure under the output root and
works with both a single image path or a directory (processed recursively).

Usage examples:
  python enhance_quality.py -i cropped_out -o enhanced_images
  python enhance_quality.py -i cropped_out/train -o enhanced_images
  python enhance_quality.py -i cropped_out/train/Qualified_PNG/processed_1.png -o enhanced_images

Dependencies: opencv-python, numpy
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff',
              '.PNG', '.JPG', '.JPEG', '.BMP', '.TIF', '.TIFF'}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_image(path: Path):
    # Safe read allowing unicode paths
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return img


def apply_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    if img is None:
        return img
    # Compute kernel size from sigma
    sigma = max(0.05, float(sigma))
    k = max(3, int(2 * np.ceil(2 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_clahe(img: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    if img is None:
        return img
    clip_limit = float(clip_limit)
    tile_grid = int(tile_grid)
    if img.ndim == 3:
        # Use LAB space: apply CLAHE on L channel only
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        return clahe.apply(img)


def enhance(img: np.ndarray, sigma: float, clip_limit: float, tile_grid: int) -> np.ndarray:
    # Convert to uint8 if needed
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    blurred = apply_gaussian(img, sigma)
    enhanced = apply_clahe(blurred, clip_limit, tile_grid)
    return enhanced


def collect_files(input_path: Path):
    files = []
    if input_path.is_dir():
        for ext in VALID_EXTS:
            files.extend(input_path.rglob(f'*{ext}'))
    else:
        if input_path.suffix in VALID_EXTS:
            files = [input_path]
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description='Enhance images: Gaussian blur + CLAHE, preserving folder structure.')
    ap.add_argument('-i', '--input', required=True, help='Input image or folder (processed recursively if folder)')
    ap.add_argument('-o', '--output', required=True, help='Output root folder')
    ap.add_argument('--sigma', type=float, default=0.6, help='Gaussian blur sigma (default: 0.6)')
    ap.add_argument('--clip_limit', type=float, default=2.0, help='CLAHE clip limit (default: 2.0)')
    ap.add_argument('--tile_grid', type=int, default=8, help='CLAHE tile grid size (default: 8)')
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output)
    ensure_dir(out_root)

    files = collect_files(in_path)
    if not files:
        print(f'No images found in: {args.input}')
        return

    # If input is folder, base output is output/input_name; else direct to output
    base_out = out_root / in_path.name if in_path.is_dir() else out_root
    ensure_dir(base_out)

    count = 0
    for f in files:
        # Preserve folder structure if input is a directory
        out_dir = base_out / f.parent.relative_to(in_path) if in_path.is_dir() else base_out
        ensure_dir(out_dir)
        # Read, enhance, save
        img = read_image(f)
        if img is None or getattr(img, 'size', 0) == 0:
            print(f'[WARN] Cannot read: {f}')
            continue
        out_img = enhance(img, args.sigma, args.clip_limit, args.tile_grid)
        # Use same filename or prefix enhanced_
        out_name = f.stem
        if not out_name.lower().startswith('enhanced_'):
            out_name = 'enhanced_' + out_name
        out_file = out_dir / f'{out_name}.png'
        ok, buf = cv2.imencode('.png', out_img)
        if ok:
            buf.tofile(str(out_file))
            count += 1
        else:
            print(f'[ERROR] Failed to save: {out_file}')

    print(f'Done. {count}/{len(files)} images enhanced. Output base: {base_out}')


if __name__ == '__main__':
    main()

