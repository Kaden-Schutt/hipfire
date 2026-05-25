#!/usr/bin/env python3
"""Capture intermediate vision-tower activations from dots.ocr under HF.

Phase 2c-1 of the dots.ocr bring-up plan (see
docs/plans/dots-ocr-prd.md).

What this captures, for the page at
benchmarks/images/dots_ocr_smoke_001.jpg under the prompt_layout_all_en
template (the same image + prompt as the phase-0 item-5 captures):

- patch_embed output      [N_patches, embed_dim=1536]   after Conv2d
                                                        proj + bias +
                                                        patch_embed
                                                        RMSNorm
- block 0 output          [N_patches, embed_dim]        post first
                                                        DotsVisionBlock
                                                        (RMSNorm → attn →
                                                        residual →
                                                        RMSNorm → SwiGLU
                                                        → residual)
- block 21 output         [N_patches, embed_dim]        mid-stack
- block 41 output         [N_patches, embed_dim]        final block,
                                                        pre-merger
- post_trunk_norm output  [N_patches, embed_dim]        after final RMSNorm
- merger output           [N_merged, out_hidden=1536]   merged visual
                                                        tokens — what
                                                        gets spliced into
                                                        the text embed
                                                        stream

Saved as .npy under
`benchmarks/references/dots_ocr_smoke_001_activations/<name>.npy`
with a sibling `index.json` recording shapes + dtypes + module paths
+ hook layer numbers.

Run from repo root:
    .venv/bin/python scripts/capture_dots_ocr_activations.py

Idempotent — overwrites the output directory.

# Why activations not logits?

Phase 0 item 5 captured the top-100 logit dump at four prompt
positions; that validates hipfire's text-side prefill end-to-end.
Phase 2c validates the *vision tower* — comparing logits alone there
turns "the visual embed is wrong" into a 42-block bisect via print.
With per-layer activation refs the diff is a few-second cosine check
against the right .npy.

# Runtime

Forward-pass-only — no generate loop. ~3.5 min on Strix Halo CPU at
bf16, same as the existing capture's forward stage. The hook
overhead is free.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

REPO = Path(__file__).resolve().parent.parent
IMAGE_PATH = REPO / "benchmarks" / "images" / "dots_ocr_smoke_001.jpg"
OUT_DIR = REPO / "benchmarks" / "references" / "dots_ocr_smoke_001_activations"
SNAPSHOT = (
    "/data/cache/huggingface/hub/models--rednote-hilab--dots.ocr/"
    "snapshots/c0111ce6bc07803dbc267932ffef0ae3a51dc951"
)

# Same prompt as the phase-0 item-5 capture — keep byte-identical so
# the two refs are directly comparable.
PROMPT_LAYOUT_ALL_EN = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

# Indices of blocks to capture (out of 42 total). Block 0 establishes
# the first-block-output reference; mid-stack catches drift accumulation;
# final block is the pre-merger reference.
BLOCK_INDICES_TO_CAPTURE = [0, 1, 2, 4, 8, 12, 16, 21, 41]   # bisection of 1..21 to locate per-block divergence

# Patch-row sampling. The full activation tensors are [N_patches, 1536]
# (this image: 19520 patches → 120 MB per stage × 5 stages = 600 MB).
# Too large to commit to git. We save a deterministic sample (~256
# patches per stage = 1.5 MB) so the per-stage diff is committable;
# the full tensors are written to a side path for local-only use.
#
# Sample indices are linspace across the full sequence — gives uniform
# coverage including corner patches (idx 0..3, the top-left 2×2 block)
# and the boundary at N_patches-1. Consumers diff at these same indices
# from the hipfire-side forward pass.
N_SAMPLE_PATCHES = 256
N_SAMPLE_MERGED = 64

LOCAL_FULL_DIR = Path("/data/cache/hipfire/dots_ocr_activations_full")


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not IMAGE_PATH.exists():
        print(f"error: image not found: {IMAGE_PATH}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_md5 = md5(IMAGE_PATH)
    print(f"image: {IMAGE_PATH.relative_to(REPO)}  md5={image_md5}")

    print(f"loading processor + model (bf16, eager) ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(SNAPSHOT, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # ── Hook setup ────────────────────────────────────────────────────
    #
    # The vision tower lives at `model.vision_tower`. Its structure (per
    # modeling_dots_vision.py:412-427):
    #   - patch_embed         DotsViTPreprocessor (Conv2d + RMSNorm + bias)
    #   - blocks              ModuleList of 42 DotsVisionBlock
    #   - post_trunk_norm     RMSNorm
    #   - merger              PatchMerger (LayerNorm + 2 linears + GELU)
    #
    # Forward hooks return the output of each module — exactly what we
    # want as the per-stage ref. Save the bf16 tensor as f32 .npy so the
    # consumer side (numpy / hipfire-rs) doesn't need bf16 support.
    vt = model.vision_tower
    captures: dict[str, torch.Tensor] = {}
    hook_handles = []

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            # `out` is a Tensor for the modules we're hooking. Detach and
            # move to CPU at full f32 for the save.
            t = out
            if isinstance(out, tuple):
                t = out[0]
            captures[name] = t.detach().to(torch.float32).cpu()
        return hook

    hook_handles.append(vt.patch_embed.register_forward_hook(make_hook("patch_embed")))
    for idx in BLOCK_INDICES_TO_CAPTURE:
        # `vt.blocks[idx]` returns `x + attn(norm1(x)) + mlp(norm2(...))`.
        # `vt.blocks[idx].attn` returns the FULL attention output including the
        # `self.proj` projection — i.e. what gets added to the residual on the
        # attention side of the block.
        hook_handles.append(
            vt.blocks[idx].register_forward_hook(make_hook(f"block_{idx:02d}"))
        )
        hook_handles.append(
            vt.blocks[idx].attn.register_forward_hook(make_hook(f"block_{idx:02d}_attn_out"))
        )
        # Hook the qkv linear's OUTPUT (= pre-reshape, pre-permute, pre-RoPE,
        # pre-attention). Shape is [seq, 3*num_heads*head_dim]. Lets us
        # diff our QKV linear output directly vs HF, isolating whether
        # the bug is in QKV linear or downstream (RoPE / attention compute).
        hook_handles.append(
            vt.blocks[idx].attn.qkv.register_forward_hook(make_hook(f"block_{idx:02d}_qkv"))
        )
    hook_handles.append(vt.post_trunk_norm.register_forward_hook(make_hook("post_trunk_norm")))
    hook_handles.append(vt.merger.register_forward_hook(make_hook("merger")))
    print(f"  registered {len(hook_handles)} hooks")

    # ── Build inputs (same as phase-0 item-5) ─────────────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(IMAGE_PATH)},
                {"type": "text", "text": PROMPT_LAYOUT_ALL_EN},
            ],
        }
    ]
    print("applying chat template + processing image...")
    t0 = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    image_grid_thw = inputs["image_grid_thw"].tolist()
    print(f"  preprocessing done in {time.time() - t0:.1f}s")
    print(f"  input_ids shape: {tuple(inputs['input_ids'].shape)}")
    print(f"  image_grid_thw: {image_grid_thw}")

    # ── Forward pass to populate hooks ────────────────────────────────
    print("forward pass over prompt (hooks active)...")
    t0 = time.time()
    with torch.no_grad():
        # The forward only needs to reach the vision tower; we don't
        # need the LM logits here. Run the full forward anyway so the
        # capture is reproducible end-to-end.
        out = model(
            **{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            use_cache=False,
        )
        _ = out  # discard text-side logits; we have them in the phase-0 item-5 capture
    print(f"  forward done in {time.time() - t0:.1f}s")
    print(f"  captured stages: {list(captures.keys())}")

    # ── Detach hooks + save ───────────────────────────────────────────
    for h in hook_handles:
        h.remove()

    LOCAL_FULL_DIR.mkdir(parents=True, exist_ok=True)
    index_entries = []
    for name, tensor in captures.items():
        arr = tensor.numpy()
        # Write the FULL tensor to a side path on /data/cache — not in
        # the repo (too large to commit, ~120 MB per stage).
        full_path = LOCAL_FULL_DIR / f"{name}.npy"
        np.save(full_path, arr)

        # Write the SAMPLED tensor to the repo path. Sample size depends
        # on whether this is a patch-shape tensor (~19520 rows) or the
        # merger output (~4880 rows).
        n_rows = arr.shape[0]
        n_sample = N_SAMPLE_MERGED if name == "merger" else N_SAMPLE_PATCHES
        n_sample = min(n_sample, n_rows)
        # Deterministic linspace — gives uniform coverage including
        # boundary indices 0 and N-1. Cast to int via np.linspace's
        # `dtype=int` for exact reproducibility across machines.
        sample_indices = np.linspace(0, n_rows - 1, n_sample, dtype=np.int64)
        sampled = arr[sample_indices].copy()
        sample_path = OUT_DIR / f"{name}.npy"
        np.save(sample_path, sampled)
        index_entries.append({
            "name": name,
            "file": sample_path.name,
            "sampled_shape": list(sampled.shape),
            "full_shape": list(arr.shape),
            "sample_indices": sample_indices.tolist(),
            "dtype": str(arr.dtype),
            "n_sampled_rows": int(sampled.shape[0]),
            "full_local_path": str(full_path),
        })
        print(
            f"  {name}: full {arr.shape} → /data/cache; "
            f"sampled [{n_sample}, {arr.shape[1]}] → repo"
        )

    # Sort entries to match forward-order (patch_embed → blocks → post_norm → merger).
    order_rank = {
        "patch_embed": 0,
        "post_trunk_norm": 2,
        "merger": 3,
    }
    def rank(e):
        name = e["name"]
        if name.startswith("block_"):
            return (1, int(name.split("_")[1]))
        return (order_rank.get(name, 99), 0)
    index_entries.sort(key=rank)

    index = {
        "source": "hf_transformers",
        "model_id": "rednote-hilab/dots.ocr",
        "snapshot": SNAPSHOT,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "torch_dtype": "bfloat16",
        "saved_dtype": "float32",
        "device": "cpu",
        "attn_implementation": "eager",
        "image_path": str(IMAGE_PATH.relative_to(REPO)),
        "image_md5": image_md5,
        "image_grid_thw": image_grid_thw,
        "n_blocks_total": 42,
        "block_indices_captured": BLOCK_INDICES_TO_CAPTURE,
        "captures": index_entries,
        "sampling": {
            "strategy": "linspace_uniform",
            "n_sample_patch_stages": N_SAMPLE_PATCHES,
            "n_sample_merger": N_SAMPLE_MERGED,
            "note": (
                "Patch-shape stages and the merger output are too large "
                "to commit in full (~600 MB total). We save a uniform "
                "linspace sample for the committed reference and stash "
                "the full tensors at `full_local_path` for local-only "
                "use. Sample indices are recorded per-stage so the "
                "hipfire-side diff can read from its own full output "
                "at the matching rows."
            ),
        },
        "consumer_note": (
            "Saved at float32 from the bf16 forward. hipfire-side "
            "comparison should expect bf16→f16 cast loss at deeper "
            "blocks (per plan §5 phase 2: absolute < 1e-2 or cosine "
            "> 0.999). The merger output is what splices into the "
            "text embed stream as visual tokens."
        ),
    }
    index_path = OUT_DIR / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"  wrote index: {index_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
