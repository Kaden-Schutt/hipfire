#!/usr/bin/env python3
"""
Dump a HuggingFace vision-tower's per-stage activations on an image, for
hipfire numerical-diff debugging (`diff_dumps.py`).

Multi-family: the vision-tower module layout differs per model family, so a
small registry (keyed by `config.model_type`) describes how to locate the
patch-embed, encoder blocks, final norm, and projector/merger for each. Add a
new family by adding one `VisionFamily` entry — see `FAMILIES` below.

Currently supported families:
  * qwen3_vl / qwen2_5_vl / qwen2_vl  — `model.model.visual` (blocks + merger,
    grid_thw-driven).
  * gemma3 / gemma3_vl                — `model.model.vision_tower.vision_model`
    (SigLIP: patch_embedding conv, position_embedding, encoder.layers,
    post_layernorm) + `model.model.multi_modal_projector`.

Outputs (per image) to <out>/<image_stem>/ as .npy:
  pixel_values.npy        # post-preprocessor
  patch_embed.npy         # post patch-embed
  block_{nn}.npy          # per encoder block output
  pre_merger.npy          # final pre-projector features (post final-norm)
  post_merger.npy         # projector/merger output (image embeddings)
  pos_embed_full.npy      # raw learned position-embedding table (when present)
  image_features.npy      # model.get_image_features(...) (splice-ready rows)
  meta.json

Usage:
  dump_hf_reference.py IMAGE... --model google/medgemma-1.5-4b-it
  dump_hf_reference.py IMAGE... --model medgemma-4b          # alias (see ALIASES)
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

# Convenience aliases → HF id or local snapshot path. Any HF id / path also works
# directly. Local snapshots avoid re-downloading the (large) weights.
ALIASES = {
    "qwen-0.8b": "Qwen/Qwen3.5-0.8B",
    "medgemma-4b": "/srv/huggingface/models--google--medgemma-1.5-4b-it/snapshots/91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
    "medgemma-27b": "/srv/huggingface/models--google--medgemma-27b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
}


@dataclass
class VisionFamily:
    """How to navigate one model family's vision tower for activation capture."""

    name: str
    # model_type strings this family matches.
    model_types: tuple
    # model -> the vision transformer module whose submodules we hook.
    vision_tower: Callable
    # vision_tower module -> the patch-embed submodule (hooked as "patch_embed").
    patch_embed: Callable
    # vision_tower -> iterable of encoder block modules (hooked block_NN).
    blocks: Callable
    # vision_tower -> final norm module, or None (hooked "pre_merger").
    final_norm: Optional[Callable]
    # model -> the projector/merger module (hooked "post_merger"), or None.
    projector: Callable
    # vision_tower -> raw position-embedding tensor, or None.
    pos_embed: Optional[Callable] = None
    # processor-output keys to persist + pass to get_image_features (e.g. grids).
    extra_inputs: tuple = field(default_factory=tuple)


def _getattr_path(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _as_tensor(x):
    """Coerce a model output (tensor, tuple, or *Output dataclass) to a tensor."""
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("last_hidden_state", "image_embeds", "image_features", "pooler_output"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    return None


FAMILIES = [
    VisionFamily(
        name="qwen-vl",
        model_types=("qwen3_vl", "qwen2_5_vl", "qwen2_vl"),
        vision_tower=lambda m: _getattr_path(m, "model.visual"),
        patch_embed=lambda v: v.patch_embed,
        blocks=lambda v: v.blocks,
        final_norm=None,
        projector=lambda m: _getattr_path(m, "model.visual.merger"),
        pos_embed=lambda v: getattr(getattr(v, "pos_embed", None), "weight", None),
        extra_inputs=("image_grid_thw",),
    ),
    VisionFamily(
        name="gemma3-siglip",
        model_types=("gemma3", "gemma3_vl", "gemma3_text"),
        # SiglipVisionModel -> SiglipVisionTransformer
        vision_tower=lambda m: _getattr_path(m, "model.vision_tower.vision_model"),
        patch_embed=lambda v: v.embeddings,  # patch + position embedding
        blocks=lambda v: v.encoder.layers,
        final_norm=lambda v: v.post_layernorm,
        projector=lambda m: _getattr_path(m, "model.multi_modal_projector"),
        # SigLIP position embedding is an nn.Embedding inside .embeddings.
        pos_embed=lambda v: getattr(
            getattr(v.embeddings, "position_embedding", None), "weight", None
        ),
    ),
]


def pick_family(model_type: str) -> VisionFamily:
    for fam in FAMILIES:
        if model_type in fam.model_types:
            return fam
    known = sorted({mt for f in FAMILIES for mt in f.model_types})
    raise SystemExit(
        f"unsupported model_type {model_type!r}; known: {known}. "
        f"Add a VisionFamily entry to dump_hf_reference.py."
    )


def dump_one(image_path, out_dir, model, processor, family, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    print(f"  image: {image.width}x{image.height}")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    np.save(out_dir / "pixel_values.npy", inputs["pixel_values"].cpu().float().numpy())
    for key in family.extra_inputs:
        if key in inputs:
            np.save(out_dir / f"{key}.npy", inputs[key].cpu().numpy())

    visual = family.vision_tower(model)
    captured = {}

    def make_hook(name):
        def _h(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            t = getattr(t, "last_hidden_state", t)
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach().cpu().float().numpy()
        return _h

    handles = [family.patch_embed(visual).register_forward_hook(make_hook("patch_embed"))]
    for i, blk in enumerate(family.blocks(visual)):
        handles.append(blk.register_forward_hook(make_hook(f"block_{i:02d}")))
    if family.final_norm is not None:
        handles.append(family.final_norm(visual).register_forward_hook(make_hook("pre_merger")))
    proj = family.projector(model)
    if proj is not None:
        handles.append(proj.register_forward_hook(make_hook("post_merger")))

    # Drive the full image-feature path so the projector/merger hook fires.
    with torch.no_grad():
        feats = model.get_image_features(**{
            k: inputs[k] for k in ("pixel_values", *family.extra_inputs) if k in inputs
        })
    feats = _as_tensor(feats)
    feats_shape = list(feats.shape) if feats is not None else None
    if feats is not None:
        np.save(out_dir / "image_features.npy", feats.detach().cpu().float().numpy())

    for h in handles:
        h.remove()

    for name, arr in sorted(captured.items()):
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  {name:14s} shape={str(arr.shape):22s} mean={arr.mean():+.4f} std={arr.std():.4f}")

    if family.pos_embed is not None:
        pe = family.pos_embed(visual)
        if pe is not None:
            np.save(out_dir / "pos_embed_full.npy", pe.detach().cpu().float().numpy())
            print(f"  pos_embed_full shape={tuple(pe.shape)}")

    meta = {
        "image_path": str(image_path),
        "image_size": list(image.size),
        "family": family.name,
        "image_features_shape": feats_shape,
        "captured_keys": sorted(captured.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def dump_lm(prompt, out_dir, model, processor, device):
    """Dump the LANGUAGE-MODEL forward for a text-only prompt: per-decoder-layer
    hidden states + final logits + input_ids, to validate hipfire's gemma3
    decoder prefill against HF ground truth (no image, so token alignment is
    trivial)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = getattr(processor, "tokenizer", processor)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # (embed, layer0_out, ..., layerN_out), each [1, M, dim]
    np.save(out_dir / "lm_input_ids.npy", ids.cpu().numpy())
    np.save(out_dir / "lm_embed.npy", hs[0].detach().cpu().float().numpy())
    for i, h in enumerate(hs[1:]):
        np.save(out_dir / f"lm_block_{i:02d}.npy", h.detach().cpu().float().numpy())
    np.save(out_dir / "lm_logits.npy", out.logits.detach().cpu().float().numpy())
    print(f"  prompt M={ids.shape[1]} tokens, {len(hs)-1} layers, hidden={hs[0].shape[-1]}")
    print(f"  last-pos top-5 token ids: {out.logits[0, -1].topk(5).indices.tolist()}")
    (out_dir / "lm_meta.json").write_text(json.dumps({
        "prompt": prompt, "n_tokens": int(ids.shape[1]),
        "n_layers": len(hs) - 1, "hidden": int(hs[0].shape[-1]),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image paths (vision dump)")
    ap.add_argument("--model", required=True,
                    help="HF id, local snapshot path, or alias: " + ", ".join(ALIASES))
    ap.add_argument("--out", default="hf-ref")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lm-prompt", default=None,
                    help="dump the LM forward (hidden states + logits) for this text prompt")
    args = ap.parse_args()

    model_path = ALIASES.get(args.model, args.model)
    cfg = AutoConfig.from_pretrained(model_path)
    print(f"model_type={cfg.model_type}; loading on {args.device}...")

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=args.device,
    ).eval()

    out_root = Path(args.out)
    if args.lm_prompt is not None:
        print("\n== LM forward ==")
        dump_lm(args.lm_prompt, out_root / "lm", model, processor, args.device)
    if args.images:
        family = pick_family(cfg.model_type)
        for img in args.images:
            p = Path(img)
            print(f"\n== {p.name} ({family.name}) ==")
            dump_one(p, out_root / p.stem, model, processor, family, args.device)
    print(f"\nDone. Dumps at: {out_root.absolute()}")


if __name__ == "__main__":
    main()
