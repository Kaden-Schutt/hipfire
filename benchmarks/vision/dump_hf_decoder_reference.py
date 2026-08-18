#!/usr/bin/env python3
"""Dump Qwen3.5-VL decoder hidden rows for a Hipfire vision embedding.

This deliberately bypasses the HF vision tower. Feeding the exact post-merger
rows emitted by Hipfire into the source decoder makes any subsequent
divergence attributable to decoder semantics rather than image preprocessing
or vision-tower numerical drift.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def fake_q8f16_(tensor: torch.Tensor) -> None:
    """Apply Hipfire's 32-value symmetric Q8/F16-scale round trip in place."""
    flat = tensor.detach().float().contiguous().view(-1)
    padded = ((flat.numel() + 31) // 32) * 32
    if padded != flat.numel():
        work = torch.zeros(padded, dtype=torch.float32)
        work[: flat.numel()] = flat
    else:
        work = flat.clone()
    blocks = work.view(-1, 32)
    scale = (blocks.abs().amax(dim=1) / 127.0).to(torch.float16).float()
    inv = torch.where(scale > 0, scale.reciprocal(), torch.zeros_like(scale))
    quant = torch.round(blocks * inv[:, None]).clamp(-128, 127)
    dequant = (quant * scale[:, None]).view(-1)[: flat.numel()]
    tensor.copy_(dequant.reshape(tensor.shape).to(tensor.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--visual", required=True, help="raw little-endian f32 [N, hidden]")
    parser.add_argument("--grid-h", required=True, type=int)
    parser.add_argument("--grid-w", required=True, type=int)
    parser.add_argument("--prompt", default="Read the page.")
    parser.add_argument(
        "--suffix-file",
        help="append tokenized text after the assistant prefix for teacher-forced decode parity",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--fake-q8",
        action="store_true",
        help="round-trip decoder matmul weights through Hipfire Q8F16",
    )
    parser.add_argument(
        "--positions",
        default="3,4,199,200,-1",
        help="comma-separated token positions; -1 means the final prompt token",
    )
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.fake_q8:
        seen = set()
        quantized = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if (
                    name.startswith("model.visual.")
                    or not name.endswith("weight")
                    or "norm" in name
                ):
                    continue
                fake_q8f16_(parameter)
                quantized += parameter.numel()
        print(f"fake-q8: round-tripped {quantized} decoder parameters")

    merge = model.config.vision_config.spatial_merge_size
    hidden = model.config.text_config.hidden_size
    n_visual = (args.grid_h // merge) * (args.grid_w // merge)
    visual = np.fromfile(args.visual, dtype="<f4")
    if visual.size != n_visual * hidden:
        raise ValueError(
            f"{args.visual}: got {visual.size} floats, expected "
            f"{n_visual} * {hidden} = {n_visual * hidden}"
        )
    visual_t = torch.from_numpy(visual.reshape(n_visual, hidden).copy())

    # Match ChatFrame::build_with_user_tokens exactly. Encoding the whole
    # rendered string at once is subtly different because BPE can merge across
    # component boundaries; Hipfire intentionally encodes each scaffold/body
    # component independently and concatenates token IDs.
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    image_id = model.config.image_token_id
    ids = []
    ids += encode("<|im_start|>")
    ids += encode("user")
    ids += encode("\n")
    ids += encode("<|vision_start|>")
    ids += [image_id] * n_visual
    ids += encode("<|vision_end|>")
    ids += encode("\n")
    ids += encode(args.prompt)
    ids += encode("<|im_end|>")
    ids += encode("\n")
    ids += encode("<|im_start|>")
    ids += encode("assistant")
    ids += encode("\n")
    ids += [tokenizer.convert_tokens_to_ids("<think>")]
    ids += encode("\n")
    ids += encode("\n")
    ids += [tokenizer.convert_tokens_to_ids("</think>")]
    ids += encode("\n")
    ids += encode("\n")
    prompt_tokens = len(ids)
    if args.suffix_file:
        ids += encode(Path(args.suffix_file).read_text())
    input_ids = torch.tensor([ids], dtype=torch.long)
    image_mask = input_ids == image_id
    if int(image_mask.sum()) != n_visual:
        raise ValueError(
            f"framing produced {int(image_mask.sum())} image tokens, expected {n_visual}"
        )

    layer_outputs = [None] * len(model.model.language_model.layers)
    hooks = []
    for layer_index, layer in enumerate(model.model.language_model.layers):
        def capture(_module, _inputs, output, index=layer_index):
            value = output[0] if isinstance(output, (tuple, list)) else output
            layer_outputs[index] = value.detach().float().cpu()

        hooks.append(layer.register_forward_hook(capture))

    with torch.no_grad():
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        inputs_embeds[image_mask] = visual_t
        mm_types = image_mask.to(torch.int32)
        attention_mask = torch.ones_like(input_ids)
        grid_thw = torch.tensor([[1, args.grid_h, args.grid_w]], dtype=torch.long)
        position_ids, rope_delta = model.model.get_rope_index(
            input_ids,
            mm_types,
            image_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )
        output = model.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()

    requested = [int(x) for x in args.positions.split(",") if x.strip()]
    positions = [input_ids.shape[1] - 1 if p == -1 else p for p in requested]
    for p in positions:
        if not 0 <= p < input_ids.shape[1]:
            raise ValueError(f"position {p} outside prompt length {input_ids.shape[1]}")

    # Preserve RAW post-layer residuals. Transformers' output_hidden_states
    # replaces the final raw layer result with the post-final-norm state, which
    # makes a layer-23 comparison against Hipfire's post-layer hook misleading.
    rows = torch.stack(
        [inputs_embeds[0, positions].float().cpu()]
        + [state[0, positions] for state in layer_outputs]
        + [output.last_hidden_state[0, positions].float().cpu()],
        dim=0,
    ).numpy()
    final_logits = (
        model.lm_head(output.last_hidden_state[:, positions])
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, rows)
    np.save(out.with_name(out.stem + "_logits.npy"), final_logits)
    meta = {
        "model": args.model,
        "visual": args.visual,
        "grid_thw": [1, args.grid_h, args.grid_w],
        "n_visual": n_visual,
        "prompt_tokens": int(input_ids.shape[1]),
        "assistant_prefix_tokens": prompt_tokens,
        "image_start": int(torch.where(image_mask[0])[0][0]),
        "positions": positions,
        "token_ids": [int(input_ids[0, p]) for p in positions],
        "position_ids": [
            [int(v) for v in position_ids[:, 0, p]]
            for p in positions
        ],
        "rope_delta": int(rope_delta[0, 0]),
        "rows_shape": list(rows.shape),
        "rows_layout": ["input"] + [f"post_layer_{i}" for i in range(24)] + ["final_norm"],
        "logits_shape": list(final_logits.shape),
        "fake_q8": args.fake_q8,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
