#!/usr/bin/env python3
"""Apply hipfire's preprocessing pipeline IN PYTHON with BICUBIC resize and
2x2-block reorder, then compare to HF reference. If byte-identical, we've
confirmed (1) the patch reorder is the primary bug, and (2) the Triangle
resize is the secondary 9% residual."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def hipfire_preprocess(image_path, patch_size=16, sms=2, temporal=2, bicubic=True, channel_order="RGB"):
    """Replicate hipfire's preprocessor in Python with toggles."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # smart_resize, factor = patch_size * sms = 32
    factor = patch_size * sms
    min_pixels = 56 * 56
    max_pixels = 14 * 14 * 4 * 1280

    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = ((h * w) / max_pixels) ** 0.5
        h_bar = max(factor, int(h / beta / factor) * factor)
        w_bar = max(factor, int(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (h * w)) ** 0.5
        h_bar = max(factor, -(-int(h * beta) // factor) * factor)
        w_bar = max(factor, -(-int(w * beta) // factor) * factor)

    resample = Image.Resampling.BICUBIC if bicubic else Image.Resampling.BILINEAR
    img_r = img.resize((w_bar, h_bar), resample)
    arr = np.array(img_r, dtype=np.float32) / 127.5 - 1.0   # HWC, [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))  # CHW

    if channel_order == "RBG":
        arr = arr[[0, 2, 1], :, :]  # R, B, G — hipfire's deliberate swap

    # extract_patches → per_patch[temporal, channel, ph, pw] flat
    C, H, W = arr.shape
    ph, pw = H // patch_size, W // patch_size
    n = ph * pw
    elems = temporal * C * patch_size * patch_size  # 1536

    patches = np.zeros((n, elems), dtype=np.float32)
    for py in range(ph):
        for px in range(pw):
            patch = arr[:, py*patch_size:(py+1)*patch_size,
                          px*patch_size:(px+1)*patch_size]  # (C, ps, ps)
            # Duplicate frame for temporal=2
            patch_t = np.stack([patch, patch], axis=0)  # (T, C, ps, ps)
            patches[py*pw + px] = patch_t.flatten()
    return patches, (ph, pw)


def reorder_2x2(patches, ph, pw):
    """Reorder (ph*pw, elems) row-major → 2x2-block-grouped."""
    g = patches.reshape(ph, pw, -1)
    g = g.reshape(ph // 2, 2, pw // 2, 2, -1).transpose(0, 2, 1, 3, 4)
    return g.reshape(-1, patches.shape[1])


def main():
    img_path = Path(sys.argv[1])
    hf_pixels = np.load(Path(sys.argv[2]) / "pixel_values.npy")

    for filter_label, bicubic in [("triangle/bilinear", False), ("bicubic", True)]:
        for ch_label, ch in [("RGB", "RGB"), ("RBG (hipfire)", "RBG")]:
            patches, (ph, pw) = hipfire_preprocess(img_path, bicubic=bicubic, channel_order=ch)
            reordered = reorder_2x2(patches, ph, pw)
            d = np.abs(hf_pixels - reordered)
            rel = d.sum() / (np.abs(hf_pixels).sum() + 1e-9)
            label = f"  filter={filter_label:18s} ch={ch_label:13s}"
            verdict = "  >>> BYTE-IDENTICAL" if d.max() < 1e-4 else (
                      "  >>> tiny residual" if rel < 1e-3 else "")
            print(f"{label}  rel-L1={rel:.4e}  max|Δ|={d.max():.4e}{verdict}")


if __name__ == "__main__":
    main()
