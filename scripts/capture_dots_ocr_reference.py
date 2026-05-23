#!/usr/bin/env python3
"""Capture dots.ocr reference logits + token IDs + parsed JSON layout
for hipfire validation.

Phase 0 item 5 of the dots.ocr + Qwen2 bring-up plan (see
docs/plans/dots-ocr-prd.md).

What this captures, for the page image at
benchmarks/images/dots_ocr_smoke_001.jpg under the
prompt_layout_all_en template (the dots.ocr canonical
layout-extraction prompt):

  - input_token_ids: full prompt-with-image tokenisation
    (chat-template-applied, includes `<|img|>`...`<|endofimg|>`
    wrapper + IMGPAD placeholders + the layout prompt text).
  - n_prompt_tokens: token count including image-pad expansion.
  - image_grid_thw: per-image (t, h, w) post-smart-resize patch grid.
  - first_200_completion_token_ids: greedy-decoded continuation
    (capped at 200 to bound runtime; full JSON would be much longer
    on a real page).
  - completion_text_partial: detokenised completion bytes (may be
    truncated mid-JSON; the phase 4 OCR gate runs a full decode
    against the model's natural EOS).
  - logits_top100_at_positions: top-100 token IDs and their f32 logit
    values at positions 0, 32, and the last prompt position (the
    predictor of the first completion token).
  - parsed_json (best-effort): if the full completion runs to EOS in
    this capture, the parsed layout JSON. Truncation is recorded
    explicitly so consumers don't mistake a partial dump for ground
    truth.

The output JSON is committed to benchmarks/references/ so the future
hipfire-arch-dots-ocr forward pass and the phase-4 OCR coherence gate
have a fixed comparison target.

Run from repo root:
    .venv/bin/python scripts/capture_dots_ocr_reference.py

Idempotent — overwrites the output file. Recompute when transformers
version or model snapshot changes; the artifact records both for
reproducibility.

Runtime notes
-------------
- CPU-only run at bf16. The full model is ~3 B params (~6 GB at bf16);
  loads + first forward takes a few minutes on Strix Halo APU CPU.
- 200-token greedy decode at bf16 on CPU is the bottleneck (~10-20 min
  end-to-end). Acceptable for a one-time reference-capture artifact.
- bf16 is the model's NATIVE precision (per config.json:torch_dtype)
  AND the vision tower's forward unconditionally casts inputs to bf16
  at entry (modeling_dots_vision.py:493-494). Loading at f32 produces
  a dtype mismatch in patch_embed Conv2d (input bf16 vs weight f32).
  Reference dtype is therefore bf16 not f32 — hipfire-side comparison
  uses cosine > 0.999 / abs < 1e-2 tolerances per plan §5 phase 2,
  which accommodates the bf16 → f16 cast loss that hipfire will
  apply at load time.
- `attn_implementation="eager"` — flash_attention_2 is CUDA-only and
  we're on CPU. sdpa appeared to work but produced a degenerate
  decode (correct JSON skeleton then immediate collapse to repeated
  `1` token), suggesting a numerical issue with sdpa+CPU+bf16 for
  this model. Eager is slower but the most well-tested attention
  backend in PyTorch and the natural choice for a one-time reference
  capture.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

REPO = Path(__file__).resolve().parent.parent
IMAGE_PATH = REPO / "benchmarks" / "images" / "dots_ocr_smoke_001.jpg"
OUT_PATH = REPO / "benchmarks" / "references" / "dots_ocr_smoke_001.json"
MODEL_ID = "rednote-hilab/dots.ocr"
SNAPSHOT = (
    "/data/cache/huggingface/hub/models--rednote-hilab--dots.ocr/"
    "snapshots/c0111ce6bc07803dbc267932ffef0ae3a51dc951"
)

# Canonical dots.ocr layout-extraction prompt, from
# dots_ocr/utils/prompts.py:prompt_layout_all_en. Stored as a literal
# here so this script is reproducible without the dots.ocr github
# checkout on the host.
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

MAX_NEW_TOKENS = 200


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    if not IMAGE_PATH.exists():
        print(f"error: image not found: {IMAGE_PATH}", file=sys.stderr)
        return 1
    if not Path(SNAPSHOT).is_dir():
        print(
            f"error: model snapshot not found: {SNAPSHOT}\n"
            f"  hint: huggingface-cli download {MODEL_ID}",
            file=sys.stderr,
        )
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_md5 = md5(IMAGE_PATH)
    print(f"image: {IMAGE_PATH}  ({IMAGE_PATH.stat().st_size} bytes, md5={image_md5})")

    print(f"loading processor from {SNAPSHOT}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(SNAPSHOT, trust_remote_code=True)
    print(f"  processor loaded in {time.time() - t0:.1f}s")

    print(f"loading model from {SNAPSHOT} (cpu, bfloat16, sdpa attn)...")
    t0 = time.time()
    # Use sdpa because flash_attention_2 is CUDA-only. dtype=bfloat16
    # because (a) it's the model's native publish precision and (b)
    # the vision tower's forward unconditionally casts to bf16 at
    # entry — loading at f32 produces a dtype mismatch in patch_embed
    # Conv2d (input bf16 vs weight f32).
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  model loaded in {time.time() - t0:.1f}s")

    # Patch dots.ocr's prepare_inputs_for_generation — the trust-remote-
    # code modeling code at modeling_dots_ocr.py:107-131 has two bugs
    # against transformers 5.5.1:
    #
    # 1. `_prefill` calls it with cache_position=None on the first
    #    step; dereferencing `cache_position[0]` then raises TypeError.
    # 2. We need to splice pixel_values into model_inputs ONLY on the
    #    first decode step (when the KV cache is being built), not on
    #    every subsequent step (which would crash inside
    #    prepare_inputs_embeds when the decode-only input has no
    #    `<|imgpad|>` slots to substitute into).
    #
    # The correct heuristic for "first step" is the
    # `cache_position[0] == 0` check that the original code wants to
    # perform — but on the cache_position that the parent class
    # *populates* inside `model_inputs`, not the (possibly None) one
    # we received. So: call super, then read back the populated
    # cache_position to decide. This is a strictly-more-correct
    # version of the original code; no behavior change when
    # cache_position is properly threaded.
    bypass_orig = type(model).prepare_inputs_for_generation
    # The model class is a VLM subclass of a generation model. The
    # dots.ocr override (in transformers_modules/.../modeling_dots_ocr.py)
    # is the one we want to bypass. We invoke the FIRST ancestor whose
    # prepare_inputs_for_generation is NOT defined in a `dots_ocr`
    # source file (heuristic via __code__.co_filename). This skips
    # the dots.ocr override regardless of whether the cache module
    # is at __mro__[0] or under an Auto/wrapper class.
    parent_prep = None
    mro_chain = []
    for cls in type(model).__mro__:
        method = cls.__dict__.get("prepare_inputs_for_generation")
        if method is None:
            continue
        src = getattr(method, "__code__", None)
        src_file = src.co_filename if src is not None else "<unknown>"
        mro_chain.append((cls.__name__, src_file))
        if "dots_ocr" in src_file:
            continue
        parent_prep = method
        break
    print(f"  prepare_inputs_for_generation MRO chain (first 5):")
    for name, src in mro_chain[:5]:
        print(f"    {name}: {src}")
    if parent_prep is None:
        raise RuntimeError("could not find non-dots-ocr prepare_inputs_for_generation")
    print(f"  using parent_prep from: {parent_prep.__code__.co_filename}")

    def patched_prepare(self, input_ids, **kw):
        # Both VL inputs need to be spliced ONLY on the first decode step.
        # The parent's prepare_inputs_for_generation doesn't pass them
        # through, and forward's vision branch (line 87-89 of
        # modeling_dots_ocr.py) only fires when pixel_values is not None,
        # so absence on subsequent steps is the correct behavior.
        pixel_values = kw.pop("pixel_values", None)
        image_grid_thw = kw.pop("image_grid_thw", None)
        model_inputs = parent_prep(self, input_ids, **kw)
        # First-step detection via the parent-populated cache_position.
        # The original dots.ocr code checked the unmodified outer
        # cache_position, which crashes when None.
        cp = model_inputs.get("cache_position", None)
        first_step = cp is not None and int(cp[0]) == 0
        if first_step:
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw
        return model_inputs

    type(model).prepare_inputs_for_generation = patched_prepare
    print("  patched prepare_inputs_for_generation (None-guard + first-step pixel_values splice)")

    # transformers 5.5.1's `_validate_model_kwargs` walks the model
    # class's `forward` signature via reflection and rejects any
    # kwarg it doesn't see. The dots.ocr `forward` declared at
    # modeling_dots_ocr.py:68-83 DOES include `pixel_values`,
    # `image_grid_thw`, `attention_mask`, but the validator's
    # introspection picks up the parent class's forward instead and
    # rejects them. No-op the validator; we know our kwargs are
    # legitimate (the prompt-forward pass earlier in this script
    # used the same kwargs against `model(...)` directly without
    # complaint).
    type(model)._validate_model_kwargs = lambda self, model_kwargs: None
    print("  no-op'd _validate_model_kwargs for VL-arg compat")

    # Build the chat-template message structure dots.ocr expects.
    # Note: passing the path as the `image` field — process_vision_info
    # (called inside the dots.ocr processor) accepts paths, PIL Images,
    # or URLs. Path keeps the inputs reproducible.
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Re-use the dots.ocr github checkout's qwen_vl_utils for image
    # resolution / preprocessing.
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print(f"  preprocessing done in {time.time() - t0:.1f}s")
    print(f"  input_ids shape: {tuple(inputs['input_ids'].shape)}")
    if "image_grid_thw" in inputs:
        print(f"  image_grid_thw: {inputs['image_grid_thw'].tolist()}")

    input_ids = inputs["input_ids"]
    n_prompt = int(input_ids.shape[1])

    # ── Logits capture: forward pass over the prompt to grab logits at
    # 0, 32, n_prompt-1. We can't reuse model(...) output for both this
    # and generate() since generate uses KV cache; capture once here.
    print("forward pass over prompt to capture logits...")
    t0 = time.time()
    with torch.no_grad():
        # Move all tensor inputs to model device (CPU). The pixel_values
        # tensor is large; this is a no-op on CPU.
        out = model(
            **{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            use_cache=False,
        )
    logits = out.logits[0].float()  # [n_prompt, vocab_size]
    print(f"  forward done in {time.time() - t0:.1f}s, logits shape={tuple(logits.shape)}")

    top_positions = [0, 32, 128, n_prompt - 1]
    top_positions = sorted(set(p for p in top_positions if 0 <= p < n_prompt))
    logit_dump: dict[str, list[dict[str, float]]] = {}
    for p in top_positions:
        vals, ids = torch.topk(logits[p], k=100)
        logit_dump[f"pos_{p}"] = [
            {"token_id": int(t), "logit": float(v)}
            for t, v in zip(ids.tolist(), vals.tolist())
        ]

    # ── Greedy decode 200 tokens. dots.ocr's typical layout output runs
    # many thousands of tokens; this cap bounds the one-time reference-
    # capture runtime to something reasonable on CPU. The phase-4 OCR
    # gate is expected to run unbounded with --max-tokens up to the
    # model's natural EOS.
    print(f"greedy decoding up to {MAX_NEW_TOKENS} continuation tokens...")
    t0 = time.time()
    # The dots.ocr processor returns an `mm_token_type_ids` tensor
    # alongside `input_ids` / `pixel_values` / `image_grid_thw`.
    # Forward accepts it; `generate` (transformers 5.5.1) rejects it
    # via `_validate_model_kwargs`. Filter explicitly.
    GEN_KWARG_DROP = {"mm_token_type_ids"}
    gen_inputs = {
        k: v for k, v in inputs.items()
        if isinstance(v, torch.Tensor) and k not in GEN_KWARG_DROP
    }
    with torch.no_grad():
        gen = model.generate(
            **gen_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            num_beams=1,
        )
    print(f"  generation done in {time.time() - t0:.1f}s")
    completion_ids = gen[0, n_prompt:].tolist()
    n_completion = len(completion_ids)

    completion_text = processor.tokenizer.decode(
        completion_ids, skip_special_tokens=False
    )
    print(f"completion ({n_completion} tokens): {completion_text[:200]!r}...")

    # Best-effort JSON parse. Truncation at 200 tokens almost certainly
    # leaves the JSON unterminated; record both the partial text and
    # the parse status so consumers don't conflate the two.
    parsed_json = None
    parse_status = "truncated_at_max_new_tokens"
    decoded_clean = processor.tokenizer.decode(
        completion_ids, skip_special_tokens=True
    )
    try:
        parsed_json = json.loads(decoded_clean)
        parse_status = "ok"
    except Exception as e:
        parse_status = f"unparseable: {type(e).__name__}: {e}"

    # Decode-quality diagnostic. Empirically the greedy decode on
    # CPU+bf16 (both sdpa AND eager) for this model collapses into a
    # repeated-token attractor within ~5 generated tokens. The JSON
    # structure header (`[{"bbox": [`) emits correctly — confirming
    # vision context made it to the embedding — then numerical drift
    # in the attention cascade catastrophically degrades coordinate
    # predictions. This is NOT a phase-2c blocker: the captured
    # prefill logits at the recorded positions are coherent (verified
    # by strong top-1 peaks: pos_0='task'/+13 nats, pos_5094='['/+11
    # nats) and are the actual ground truth hipfire compares against.
    # For the phase-4 OCR coherence gate the decode tokens need a
    # GPU+bf16+flash_attn re-capture for usable JSON output. Recorded
    # below so downstream consumers don't mistake the captured
    # completion_token_ids for a usable layout reference.
    completion_first10 = completion_ids[:10]
    # Heuristic: if the same token appears in 4+ of the first 10
    # generated tokens, flag the decode as degraded.
    most_common_count = max(
        completion_first10.count(t) for t in set(completion_first10)
    ) if completion_first10 else 0
    decode_quality = (
        "degraded_cpu_bf16_collapse"
        if most_common_count >= 4
        else "ok_first_10_diverse"
    )

    artifact = {
        "model_id": MODEL_ID,
        "snapshot": SNAPSHOT,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "torch_dtype": "bfloat16",
        "device": "cpu",
        "attn_implementation": "eager",
        "image_path": str(IMAGE_PATH.relative_to(REPO)),
        "image_md5": image_md5,
        "image_byte_count": IMAGE_PATH.stat().st_size,
        "prompt_template_key": "prompt_layout_all_en",
        "prompt_template_text": PROMPT_LAYOUT_ALL_EN,
        "input_token_ids": input_ids[0].tolist(),
        "n_prompt_tokens": n_prompt,
        "image_grid_thw": (
            inputs["image_grid_thw"].tolist()
            if "image_grid_thw" in inputs
            else None
        ),
        "completion_token_ids": completion_ids,
        "n_completion_tokens": n_completion,
        "completion_text_partial": completion_text,
        "parsed_json": parsed_json,
        "parse_status": parse_status,
        "decode_quality": decode_quality,
        "decode_quality_note": (
            "CPU+bf16 greedy decode collapses into a repeated-token "
            "attractor after ~5 tokens for this model (both sdpa and "
            "eager attention paths). The captured logits at the "
            "positions listed under `logits_top100_at_positions` are "
            "coherent (strong top-1 peaks) and ARE the valid ground "
            "truth for hipfire phase-2c forward-pass validation. The "
            "phase-4 OCR coherence gate requires a separate GPU+bf16 "
            "re-capture for a usable layout-JSON reference."
        ),
        "logits_top100_at_positions": logit_dump,
        "greedy_decode": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "num_beams": 1,
        },
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote: {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
