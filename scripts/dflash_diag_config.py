#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""
dflash_diag_config.py — Hermes Hypothesis D diagnostic (2026-04-19).

Audit the DFlashDraftModel config/instantiation path. Qwen/Qwen3.5-* returns
a composite `Qwen3_5Config` (text_config + vision_config). Our original
`build_draft_config` cloned the composite and set num_hidden_layers=5 at the
top level, relying on attribute delegation to text_config. Attribute
delegation is fragile — if hidden_size or head_dim silently returns a wrong
value, the DFlashDraftModel instantiates with tensors of the wrong shape.
Training converges on SOMETHING, but that something isn't a valid draft.

This diagnostic:
  1. Loads target config (probably composite) and prints its flat vs nested
     structure.
  2. Builds draft config via our patched `build_draft_config` (flat Qwen3Config).
  3. Instantiates DFlashDraftModel, prints key tensor shapes.
  4. Loads any provided safetensors (ours or z-lab's) and reports which keys
     match vs diverge, and whether shapes align.
  5. Optionally compares our training output to z-lab's HF draft side-by-side
     (key × shape).

Usage (no GPU needed; uses meta device if CUDA absent):
  python3 scripts/dflash_diag_config.py \
      --target-repo Qwen/Qwen3.5-4B \
      --safetensors /root/dflash_4b_agentic/model.safetensors \
      --compare-safetensors ~/.hipfire/reference_drafts/Qwen3.5-4B-DFlash/model.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".dflash-reference"))

from dflash.model import DFlashDraftModel  # type: ignore[import-not-found]  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--draft-layers", type=int, default=5)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--safetensors", default=None,
                   help="Path to our trained safetensors to audit.")
    p.add_argument("--compare-safetensors", default=None,
                   help="Path to z-lab's (or any reference) safetensors for diff.")
    return p.parse_args()


def print_config_flat_vs_nested(tc):
    print("=" * 72)
    print(f"Target config type: {type(tc).__name__}")
    has_text = hasattr(tc, "text_config")
    has_vision = hasattr(tc, "vision_config")
    print(f"  composite? text_config={has_text}  vision_config={has_vision}")
    attrs = [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size",
        "vocab_size", "max_position_embeddings", "sliding_window",
        "rms_norm_eps", "rope_theta", "attention_bias", "attention_dropout",
    ]
    print("\nattribute  → top-level vs text_config")
    for a in attrs:
        top = getattr(tc, a, "MISSING")
        txt = getattr(tc.text_config, a, "MISSING") if has_text else "—"
        same = "==" if str(top) == str(txt) else "≠≠"
        print(f"  {a:30s} {str(top):20s} {same} {str(txt):20s}")
    # layer_types deserves extra attention
    lt_top = getattr(tc, "layer_types", None)
    lt_txt = getattr(tc.text_config, "layer_types", None) if has_text else None
    print(f"\n  layer_types top-level: "
          f"{'len=%d, %s' % (len(lt_top), list(dict.fromkeys(lt_top))[:3]) if lt_top else 'None'}")
    print(f"  layer_types text_cfg:  "
          f"{'len=%d, %s' % (len(lt_txt), list(dict.fromkeys(lt_txt))[:3]) if lt_txt else 'None'}")


def instantiate_and_dump(args, tc):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import importlib
    mod = importlib.import_module("dflash_train_poc")
    cfg = mod.build_draft_config(tc, args.draft_layers, args.block_size,
                                 mask_token_id=151935)
    print("\n" + "=" * 72)
    print("Draft config (from build_draft_config):")
    print(f"  type               = {type(cfg).__name__}")
    print(f"  num_hidden_layers  = {cfg.num_hidden_layers}")
    print(f"  hidden_size        = {cfg.hidden_size}")
    print(f"  num_attention_heads= {cfg.num_attention_heads}")
    print(f"  num_key_value_heads= {cfg.num_key_value_heads}")
    print(f"  head_dim           = {cfg.head_dim}")
    print(f"  intermediate_size  = {cfg.intermediate_size}")
    print(f"  vocab_size         = {cfg.vocab_size}")
    print(f"  layer_types        = {cfg.layer_types[:5]}... (len {len(cfg.layer_types)})")
    print(f"  num_target_layers  = {cfg.num_target_layers}")
    print(f"  block_size         = {cfg.block_size}")
    print(f"  dflash_config      = {cfg.dflash_config}")

    assert cfg.num_hidden_layers == len(cfg.layer_types), "asserting invariant"

    # Instantiate on meta device if no GPU — we only need shapes, not values.
    try:
        draft = DFlashDraftModel(cfg)
        print(f"\n✓ DFlashDraftModel instantiated ({sum(p.numel() for p in draft.parameters()) / 1e6:.1f}M params)")
    except Exception as e:
        print(f"\n✗ DFlashDraftModel instantiation FAILED: {e}")
        return None

    key_shapes = {k: tuple(v.shape) for k, v in draft.state_dict().items()}
    print(f"\nstate_dict keys: {len(key_shapes)}")
    # Print a few representative layer-0 tensors
    repr_keys = [k for k in key_shapes if k.startswith("layers.0.")][:8]
    for k in repr_keys + ["fc.weight", "hidden_norm.weight", "norm.weight"]:
        if k in key_shapes:
            print(f"  {k:50s} {key_shapes[k]}")
    return key_shapes


def load_and_compare(our_path, other_path, ref_shapes):
    print("\n" + "=" * 72)
    if not our_path and not other_path:
        print("(no safetensors provided — skipping shape-diff)")
        return
    for label, path in (("ours", our_path), ("z-lab", other_path)):
        if not path:
            continue
        if not Path(path).exists():
            print(f"[{label}] {path!r}: FILE NOT FOUND")
            continue
        sd = load_file(path)
        print(f"\n[{label}] {path}")
        print(f"  keys: {len(sd)}")
        mismatch = 0
        extra_theirs = []
        missing_theirs = []
        shape_mismatch = []
        for k, v in sd.items():
            if k not in ref_shapes:
                extra_theirs.append(k)
            elif tuple(v.shape) != ref_shapes[k]:
                shape_mismatch.append((k, ref_shapes[k], tuple(v.shape)))
                mismatch += 1
        for k in ref_shapes:
            if k not in sd:
                missing_theirs.append(k)

        print(f"  matching keys+shapes: {len(sd) - len(extra_theirs) - mismatch}")
        if shape_mismatch:
            print(f"  SHAPE MISMATCHES ({len(shape_mismatch)}):")
            for k, want, got in shape_mismatch[:10]:
                print(f"    {k}: model_expects={want}  ckpt_has={got}")
            if len(shape_mismatch) > 10:
                print(f"    ... +{len(shape_mismatch) - 10} more")
        if extra_theirs:
            print(f"  extra in checkpoint ({len(extra_theirs)}): {extra_theirs[:5]}")
        if missing_theirs:
            print(f"  missing in checkpoint ({len(missing_theirs)}): {missing_theirs[:5]}")
        if not shape_mismatch and not extra_theirs and not missing_theirs:
            print("  ✓ keys + shapes match model → draft loadable")


def main():
    args = parse_args()
    print(f"Loading target config from {args.target_repo} ...")
    from transformers import AutoConfig
    tc = AutoConfig.from_pretrained(args.target_repo)
    print_config_flat_vs_nested(tc)
    ref_shapes = instantiate_and_dump(args, tc)
    if ref_shapes is None:
        return 2
    load_and_compare(args.safetensors, args.compare_safetensors, ref_shapes)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
