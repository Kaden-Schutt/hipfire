#!/usr/bin/env python3
"""
Dump HuggingFace Qwen3.5-0.8B-VL vision-tower intermediate activations on a
single image, for hipfire numerical-diff debugging.

Outputs are written to hf-ref/<image_stem>/ as .npy files:
  pixel_values.npy        # post-preprocessor, shape (n_patches, 1536)
  patch_embed.npy         # post patch_embed conv, shape (n_patches, 1152)
  post_pos_embed.npy      # patch_embed + pos_embed, shape (n_patches, 1152)
  block_{nn}.npy          # post-block-nn output, shape (n_patches, 1152), for all 27 blocks
  pre_merger.npy          # final ViT output before merger (= block_26)
  post_merger.npy         # post merger output, shape (n_patches / 4, lm_hidden)
  grid_thw.npy            # (1, 3) [temporal, h_patches, w_patches]
  meta.json               # image size, patch_size, merge_size, dims

The 0.8B variant has identical vision-tower architecture (n_embd=1152, 27 blocks)
to the 9B that hipfire is shipping; vision-tower dumps match across LM sizes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, Qwen2VLImageProcessor


HF_MODEL_PATH = "Qwen/Qwen3.5-0.8B"  # override with --model for fine-tunes


def dump_one(image_path: Path, out_dir: Path, model, processor, device: str, model_path: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    print(f"  image: {image.size} ({image.width}x{image.height})")

    # Vision-only reference: bypass AutoProcessor's video component so this
    # diagnostic does not require torchvision. The slow Qwen2-VL image
    # processor implements the same patch extraction contract.
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    print(f"  pixel_values: shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw.tolist()}")

    # --- save preprocessor outputs ---
    np.save(out_dir / "pixel_values.npy", pixel_values.cpu().float().numpy())
    np.save(out_dir / "grid_thw.npy", grid_thw.cpu().numpy())

    # --- find vision tower module + hook all sub-modules we care about ---
    visual = model.model.visual

    # Hooks
    captured = {}

    def make_hook(name):
        def _h(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach().cpu().float().numpy()
        return _h

    handles = []
    handles.append(visual.patch_embed.register_forward_hook(make_hook("patch_embed")))
    if hasattr(visual, "pos_embed"):
        # pos_embed is a Parameter, not a Module — capture after block 0 input instead.
        pass
    for i, blk in enumerate(visual.blocks):
        handles.append(blk.register_forward_hook(make_hook(f"block_{i:02d}")))
    handles.append(visual.merger.register_forward_hook(make_hook("post_merger")))

    # --- run vision encoder ---
    with torch.no_grad():
        # The visual module accepts (pixel_values, grid_thw) and returns either
        # a Tensor (older transformers) or BaseModelOutputWithPooling (newer).
        out = visual(pixel_values, grid_thw=grid_thw)
        if hasattr(out, "last_hidden_state"):
            image_features = out.last_hidden_state
        elif isinstance(out, tuple):
            image_features = out[0]
        else:
            image_features = out
        print(f"  image_features: shape={tuple(image_features.shape)} dtype={image_features.dtype}")

        # This addition happens between patch_embed and block 0. It is not a
        # module, so derive it explicitly for a direct Hipfire stage diff.
        pos_embed = visual.fast_pos_embed_interpolate(grid_thw)
        captured["post_pos_embed"] = (
            torch.from_numpy(captured["patch_embed"]).to(pos_embed)
            + pos_embed
        ).cpu().float().numpy()

    for h in handles:
        h.remove()

    # --- save captures ---
    for name, arr in captured.items():
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  saved {name}.npy  shape={arr.shape}  mean={arr.mean():+.4f} std={arr.std():.4f}")

    # Convenience aliases
    if "block_26" in captured:
        np.save(out_dir / "pre_merger.npy", captured["block_26"])
    np.save(out_dir / "image_features.npy", image_features.cpu().float().numpy())

    # Also dump the raw pos_embed table (for direct hipfire compare).
    if hasattr(visual, "pos_embed"):
        pe = visual.pos_embed
        pe_t = pe.weight if hasattr(pe, "weight") else pe
        pos_embed = pe_t.detach().cpu().float().numpy()
        np.save(out_dir / "pos_embed_full.npy", pos_embed)
        print(f"  saved pos_embed_full.npy  shape={pos_embed.shape}")

    meta = {
        "image_path": str(image_path),
        "image_size": image.size,
        "pixel_values_shape": list(pixel_values.shape),
        "grid_thw": grid_thw.tolist(),
        "image_features_shape": list(image_features.shape),
        "model": model_path,
        "captured_keys": sorted(captured.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta written to {out_dir}/meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="image paths")
    ap.add_argument("--out", default="hf-ref", help="output directory root")
    ap.add_argument("--model", default=HF_MODEL_PATH, help="HF model id or local checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="reference compute dtype",
    )
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"loading {args.model} on {args.device} ({args.dtype})...")
    processor = Qwen2VLImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
    )
    model.eval()
    print(f"loaded. visual: {type(model.model.visual).__name__}, "
          f"n_blocks={len(model.model.visual.blocks)}")

    out_root = Path(args.out)
    for img_path in args.images:
        img_path = Path(img_path)
        print(f"\n== {img_path.name} ==")
        dump_one(
            img_path,
            out_root / img_path.stem,
            model,
            processor,
            args.device,
            args.model,
        )

    print(f"\nDone. Dumps at: {out_root.absolute()}")


if __name__ == "__main__":
    main()
