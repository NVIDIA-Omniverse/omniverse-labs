#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
compare_viewers.py — Compare screenshots from different viewers.

Loads two PPM images and computes:
  - Per-channel RMSE (root mean square error)
  - SSIM-like structural similarity (simplified luminance comparison)
  - Generates a visual diff image highlighting differences

Usage:
  python3 compare_viewers.py reference.ppm test.ppm [--output diff.png] [--strict]
  python3 compare_viewers.py ref_dir/ test_dir/ [--output diff_dir/] [--strict]

Exit codes:
  0  all comparisons within PASS or OK threshold
  1  one or more comparisons reached WARN or FAIL threshold
  2  invalid invocation / IO error / dimension mismatch in --strict mode
"""

import sys
import os
import math
from PIL import Image
import numpy as np

# RMSE thresholds — anything below PASS is treated as success; FAIL/WARN cause
# a non-zero exit. Keep in sync with the labels used in print().
PASS_THRESH = 0.02
OK_THRESH = 0.05
WARN_THRESH = 0.10


def load_image(path):
    """Load PPM/PNG/JPG image as numpy float32 array [0,1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def rmse(a, b):
    """Root mean square error per channel and total."""
    diff = a - b
    mse_r = np.mean(diff[:, :, 0] ** 2)
    mse_g = np.mean(diff[:, :, 1] ** 2)
    mse_b = np.mean(diff[:, :, 2] ** 2)
    mse_total = np.mean(diff ** 2)
    return {
        "r": math.sqrt(mse_r),
        "g": math.sqrt(mse_g),
        "b": math.sqrt(mse_b),
        "total": math.sqrt(mse_total),
    }


def ssim_simple(a, b, window=11):
    """Simplified SSIM on luminance channel."""
    # Convert to grayscale
    la = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
    lb = 0.2126 * b[:, :, 0] + 0.7152 * b[:, :, 1] + 0.0722 * b[:, :, 2]

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_a = la.mean()
    mu_b = lb.mean()
    sigma_a = la.std()
    sigma_b = lb.std()
    sigma_ab = np.mean((la - mu_a) * (lb - mu_b))

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2)
    return num / den


def generate_diff_image(a, b, amplify=5.0):
    """Generate amplified absolute difference image."""
    diff = np.abs(a - b) * amplify
    diff = np.clip(diff, 0.0, 1.0)
    return (diff * 255).astype(np.uint8)


def compare_pair(ref_path, test_path, output_path=None, strict=False):
    """Compare two images and print metrics. Returns (errors, similarity, status)
    where status is one of "PASS"|"OK"|"WARN"|"FAIL"|"DIM_MISMATCH"."""
    ref = load_image(ref_path)
    test = load_image(test_path)

    ref_name = os.path.basename(ref_path)
    test_name = os.path.basename(test_path)

    if ref.shape != test.shape:
        msg = (f"  Result:     DIM_MISMATCH (ref={ref.shape[1]}x{ref.shape[0]}, "
               f"test={test.shape[1]}x{test.shape[0]})")
        if strict:
            print(f"--- {ref_name} vs {test_name} ---")
            print(msg)
            print()
            return None, None, "DIM_MISMATCH"
        # Lenient mode: resize test to match reference and warn loudly.
        print(f"--- {ref_name} vs {test_name} ---")
        print(f"  WARNING: {msg.strip()} — resizing test to ref (use --strict to fail)")
        test_img = Image.fromarray((test * 255).astype(np.uint8))
        test_img = test_img.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
        test = np.array(test_img, dtype=np.float32) / 255.0
    else:
        print(f"--- {ref_name} vs {test_name} ---")

    errors = rmse(ref, test)
    similarity = ssim_simple(ref, test)

    print(f"  RMSE total: {errors['total']:.4f}  (R:{errors['r']:.4f} G:{errors['g']:.4f} B:{errors['b']:.4f})")
    print(f"  SSIM:       {similarity:.4f}")

    if errors["total"] < PASS_THRESH:
        status = "PASS"
        print(f"  Result:     PASS (very close)")
    elif errors["total"] < OK_THRESH:
        status = "OK"
        print(f"  Result:     OK (minor differences)")
    elif errors["total"] < WARN_THRESH:
        status = "WARN"
        print(f"  Result:     WARN (noticeable differences)")
    else:
        status = "FAIL"
        print(f"  Result:     FAIL (large differences)")

    if output_path:
        diff_img = generate_diff_image(ref, test)
        Image.fromarray(diff_img).save(output_path)
        print(f"  Diff saved: {output_path}")

    print()
    return errors, similarity, status


def compare_directories(ref_dir, test_dir, output_dir=None, strict=False):
    """Compare matching files in two directories. Returns the worst status seen."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ref_files = {}
    for f in os.listdir(ref_dir):
        if f.lower().endswith((".ppm", ".png", ".jpg")):
            # Key by stripping viewer-specific prefixes
            key = f.rsplit(".", 1)[0]
            ref_files[key] = os.path.join(ref_dir, f)

    test_files = {}
    for f in os.listdir(test_dir):
        if f.lower().endswith((".ppm", ".png", ".jpg")):
            key = f.rsplit(".", 1)[0]
            test_files[key] = os.path.join(test_dir, f)

    matched = set(ref_files.keys()) & set(test_files.keys())
    if not matched:
        print(f"No exact key matches. Ref has {len(ref_files)} files, test has {len(test_files)}.")
        print(f"  Ref keys:  {sorted(ref_files.keys())[:5]}")
        print(f"  Test keys: {sorted(test_files.keys())[:5]}")
        return "FAIL"

    results = []
    for key in sorted(matched):
        out = os.path.join(output_dir, f"diff_{key}.png") if output_dir else None
        errors, sim, status = compare_pair(ref_files[key], test_files[key], out, strict=strict)
        rmse_val = errors["total"] if errors is not None else float("nan")
        ssim_val = sim if sim is not None else float("nan")
        results.append((key, rmse_val, ssim_val, status))

    # Summary
    print("=== SUMMARY ===")
    print(f"{'Scene':<40} {'RMSE':>8} {'SSIM':>8} {'Status':>14}")
    print("-" * 76)
    for key, rmse_val, ssim_val, status in results:
        rmse_s = f"{rmse_val:>8.4f}" if not math.isnan(rmse_val) else f"{'n/a':>8}"
        ssim_s = f"{ssim_val:>8.4f}" if not math.isnan(ssim_val) else f"{'n/a':>8}"
        print(f"{key:<40} {rmse_s} {ssim_s} {status:>14}")

    severity = {"PASS": 0, "OK": 1, "WARN": 2, "FAIL": 3, "DIM_MISMATCH": 3}
    worst = max((severity.get(s, 3) for _, _, _, s in results), default=0)
    return ["PASS", "OK", "WARN", "FAIL"][worst]


def main():
    args = sys.argv[1:]
    output = None
    strict = False

    if "--output" in args:
        idx = args.index("--output")
        output = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--strict" in args:
        strict = True
        args = [a for a in args if a != "--strict"]

    if len(args) != 2:
        print(__doc__)
        sys.exit(2)

    path1, path2 = args
    try:
        if os.path.isdir(path1) and os.path.isdir(path2):
            status = compare_directories(path1, path2, output, strict=strict)
        else:
            _, _, status = compare_pair(path1, path2, output, strict=strict)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    if status in ("PASS", "OK"):
        sys.exit(0)
    if status == "DIM_MISMATCH":
        sys.exit(2)
    sys.exit(1)


if __name__ == "__main__":
    main()
