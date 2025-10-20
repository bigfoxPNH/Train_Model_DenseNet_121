#!/usr/bin/env python3
"""
Skull-centric cropper for obstetric ultrasound.

Usage (examples):
  python skull_crop.py -i input_dir -o cropped_out --save_debug --debug_dir debug_skull \
      --intermediate 256 --final 224

It will:
  1) Detect the bright skull ring using HoughCircles first, then fallback to edge+ellipse.
  2) Build a circular/elliptical mask from the best ellipse.
  3) Take tight bounding box around the mask, crop, pad to square, resize to --intermediate,
     then center-crop to --final (if given).
  4) Save final images and optional debug visualizations.

Dependencies: opencv-python, numpy
"""
import argparse
import os
import glob
import math
from typing import Tuple, Optional, List

import cv2
import numpy as np

# ------------------------ Utility helpers ------------------------

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img


# ------------------------ Geometry + masks ------------------------

def ellipse_to_mask(shape: Tuple[int, int], center: Tuple[float, float], axes: Tuple[float, float], angle: float) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(axes[0] / 2), int(axes[1] / 2)), angle, 0, 360, 255, -1)
    return mask


def bbox_from_mask(mask: np.ndarray, pad_ratio: float = 0.10) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    pad = int(max(w, h) * pad_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask.shape[1] - 1, x2 + pad)
    y2 = min(mask.shape[0] - 1, y2 + pad)
    return x1, y1, x2, y2


def crop_pad_square(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    crop = img[y1:y2 + 1, x1:x2 + 1]
    h, w = crop.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    if len(crop.shape) == 2:
        pad_val = int(np.median(crop))
        padded = cv2.copyMakeBorder(crop, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=pad_val)
    else:
        pad_val = [int(np.median(crop[..., c])) for c in range(3)]
        padded = cv2.copyMakeBorder(crop, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=pad_val)
    return padded


# ------------------------ Detection core ------------------------

def try_hough_circle(gray: np.ndarray, dp: float = 1.2, min_dist_ratio: float = 0.35,
                     canny_high: int = 150, acc_thresh: int = 24) -> Optional[Tuple[Tuple[float, float], float]]:
    h, w = gray.shape[:2]
    min_dist = int(min(h, w) * min_dist_ratio)
    # Pre-smooth a bit to stabilize Hough
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=canny_high, param2=acc_thresh,
                               minRadius=int(min(h, w) * 0.18), maxRadius=int(min(h, w) * 0.6))
    if circles is None:
        return None
    # Pick circle closest to center
    cy, cx = h / 2, w / 2
    best = None
    best_d = 1e9
    for c in circles[0, :]:
        x, y, r = c
        d = math.hypot(x - cx, y - cy)
        if d < best_d:
            best_d = d
            best = ((float(x), float(y)), float(r))
    return best


def edges_and_ellipse(gray: np.ndarray, canny_low: int = 40, canny_high: int = 120,
                      dilate_iter: int = 3, min_arc_len: int = 80) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float, float]]:
    h, w = gray.shape[:2]
    # Enhance edges: CLAHE then bilateral to keep edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.bilateralFilter(eq, 9, 50, 50)
    edges = cv2.Canny(blur, canny_low, canny_high)
    if dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, k, iterations=dilate_iter)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    H, W = gray.shape
    cx, cy = W / 2, H / 2
    best = None
    best_score = -1e9
    for c in cnts:
        if len(c) < min_arc_len:
            continue
        if len(c) < 5:
            continue
        ellipse = cv2.fitEllipse(c)
        (ex, ey), (MA, ma), angle = ellipse
        # cv2 returns axes lengths (major, minor) as diameters
        a, b = max(MA, ma), min(MA, ma)
        axis_ratio = b / (a + 1e-6)
        # scoring: prefer near center, reasonable size, roundness, large support
        center_dist = math.hypot(ex - cx, ey - cy) / (0.5 * min(H, W))
        size_norm = a / min(H, W)  # 0..1-ish
        # Compute how many contour points fall near the ellipse perimeter
        # Approximate by distance to ellipse implicit equation normalized
        pts = c.reshape(-1, 2)
        cos_t, sin_t = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
        pts_c = (pts - np.array([[ex, ey]], dtype=np.float32)) @ R.T
        val = (pts_c[:, 0] / (a / 2 + 1e-6)) ** 2 + (pts_c[:, 1] / (b / 2 + 1e-6)) ** 2
        peri_err = np.mean(np.abs(val - 1.0))
        support = -peri_err  # smaller error better
        score = (
            3.0 * axis_ratio  # prefer round
            - 1.5 * center_dist
            + 1.0 * size_norm
            + 2.5 * support
            + 0.002 * len(c)  # longer contour slightly better
        )
        if score > best_score:
            best_score = score
            best = ((ex, ey), (a, b), angle, axis_ratio)
    return best


# ------------------------ Pipeline ------------------------

def detect_skull_ellipse(gray: np.ndarray,
                          use_hough: bool = True,
                          canny_low: int = 40,
                          canny_high: int = 120,
                          dilate_iter: int = 3,
                          axis_ratio_min: float = 0.55,
                          axis_ratio_max: float = 0.98) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    h, w = gray.shape[:2]
    # 1) Hough circle
    if use_hough:
        hc = try_hough_circle(gray)
        if hc is not None:
            (x, y), r = hc
            axes = (2 * r, 2 * r)
            return (x, y), axes, 0.0
    # 2) Fallback ellipse
    res = edges_and_ellipse(gray, canny_low, canny_high, dilate_iter)
    if res is None:
        return None
    (ex, ey), (a, b), angle, ar = res
    if not (axis_ratio_min <= ar <= axis_ratio_max):
        # If too elongated, try a softer dilation to see if another contour fits
        res2 = edges_and_ellipse(gray, canny_low, canny_high, max(1, dilate_iter - 1))
        if res2 is not None:
            (ex, ey), (a, b), angle, ar = res2
    return (ex, ey), (a, b), angle


def process_image(img_path: str, out_dir: str, args) -> Tuple[bool, Optional[str]]:
    try:
        gray = imread_gray(img_path)
        h, w = gray.shape[:2]
        det = detect_skull_ellipse(gray, use_hough=args.hough, canny_low=args.canny_low,
                                   canny_high=args.canny_high, dilate_iter=args.dilate_iter,
                                   axis_ratio_min=args.axis_ratio_min, axis_ratio_max=args.axis_ratio_max)
        if det is None:
            # Fallback: take central square crop to avoid crash
            cx, cy = w // 2, h // 2
            side = int(0.8 * min(h, w))
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w - 1, x1 + side)
            y2 = min(h - 1, y1 + side)
            cropped = crop_pad_square(gray, (x1, y1, x2, y2))
            note = "fallback_center"
            ellipse = None
        else:
            (ex, ey), (a, b), angle = det
            # Scale ellipse axes if requested to crop larger
            a_scaled = a * float(max(0.1, args.scale_axes))
            b_scaled = b * float(max(0.1, args.scale_axes))
            mask = ellipse_to_mask(gray.shape, (ex, ey), (a_scaled, b_scaled), angle)
            # Add extra padding around bbox if requested
            x1, y1, x2, y2 = bbox_from_mask(mask, pad_ratio=float(max(0.0, args.pad_ratio)))
            cropped = crop_pad_square(gray, (x1, y1, x2, y2))
            note = "ok"
            ellipse = ((ex, ey), (a_scaled, b_scaled), angle)
        # Resize steps
        inter_sz = args.intermediate if args.intermediate else None
        final_sz = args.final if args.final else None
        if inter_sz:
            cropped = cv2.resize(cropped, (inter_sz, inter_sz), interpolation=cv2.INTER_AREA)
        if final_sz and final_sz < (cropped.shape[0]):
            # center-crop
            s = final_sz
            y0 = (cropped.shape[0] - s) // 2
            x0 = (cropped.shape[1] - s) // 2
            cropped = cropped[y0:y0 + s, x0:x0 + s]
        # Save outputs
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"processed_{base}.png")
        ensure_dir(out_dir)
        cv2.imwrite(out_path, cropped)
        if args.save_debug:
            dbg_dir = args.debug_dir if args.debug_dir else os.path.join(out_dir, "debug")
            ensure_dir(dbg_dir)
            # Save edges/ellipse overlay minimal set
            vis = to_bgr(gray)
            if ellipse is not None:
                (ex, ey), (a, b), angle = ellipse
                cv2.ellipse(vis, (int(ex), int(ey)), (int(a / 2), int(b / 2)), angle, 0, 360, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(dbg_dir, f"ellipse_{base}.png"), vis)
            cv2.imwrite(os.path.join(dbg_dir, f"crop_{base}.png"), cropped)
        return True, note
    except Exception as e:
        return False, str(e)


def collect_images(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff",
                "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF"]
        files = []
        for ex in exts:
            files.extend(glob.glob(os.path.join(input_path, "**", ex), recursive=True))
        return sorted(files)
    else:
        return [input_path]


def main():
    parser = argparse.ArgumentParser(description="Skull ring cropper for ultrasound")
    parser.add_argument("-i", "--input", required=True, help="Input image or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory for processed images")
    parser.add_argument("--save_debug", action="store_true", help="Save debug overlays")
    parser.add_argument("--debug_dir", type=str, default=None, help="Debug directory (optional)")
    parser.add_argument("--intermediate", type=int, default=256, help="Resize side before final crop (0 to skip)")
    parser.add_argument("--final", type=int, default=224, help="Final center-crop size (0 to skip)")
    # Detection params
    parser.add_argument("--hough", action="store_true", help="Try Hough circle first")
    parser.add_argument("--canny_low", type=int, default=40)
    parser.add_argument("--canny_high", type=int, default=120)
    parser.add_argument("--dilate_iter", type=int, default=3)
    parser.add_argument("--axis_ratio_min", type=float, default=0.55, help="b/a minimum accepted")
    parser.add_argument("--axis_ratio_max", type=float, default=0.98, help="b/a maximum accepted")
    # Crop expansion controls (defaults tuned per your preference)
    parser.add_argument("--pad_ratio", type=float, default=0.10, help="extra padding around bbox as fraction of max(w,h)")
    parser.add_argument("--scale_axes", type=float, default=1.15, help="scale ellipse axes before cropping, e.g. 1.1 = 10% larger")

    args = parser.parse_args()

    images = collect_images(args.input)
    if not images:
        raise SystemExit("No input images found.")

    # If input is a folder, create a subfolder under output with the same name
    in_is_dir = os.path.isdir(args.input)
    if in_is_dir:
        in_name = os.path.basename(os.path.normpath(args.input))
        effective_out = os.path.join(args.output, in_name)
    else:
        effective_out = args.output

    ensure_dir(effective_out)

    ok = 0
    for p in images:
        # Nếu input là thư mục, bảo toàn cấu trúc thư mục con khi lưu output
        if in_is_dir:
            rel_dir = os.path.relpath(os.path.dirname(p), args.input)
            out_dir = os.path.join(effective_out, rel_dir)
        else:
            out_dir = effective_out
        ensure_dir(out_dir)
        success, note = process_image(p, out_dir, args)
        if success:
            ok += 1
        else:
            print(f"[ERROR] {p}: {note}")
    print(f"Done. {ok}/{len(images)} images processed. Base output: {effective_out}")


if __name__ == "__main__":
    main()

