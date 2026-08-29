# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.
#!/usr/bin/env python3
"""Gemma-4 26B-A4B per-layer oracle.

Runs HF reference on CPU float32 with forward hooks to capture per-layer
intermediate tensors: post-attn hidden, post-FFN hidden, MoE branch
intermediates (router logits, topk indices, expert outputs).

Compares against hipfire's HIPFIRE_GEMMA4_DUMP=1 output.

Usage:
  # Quick test with short prompt (uses the historical default checkpoint)
  .venv-rocm/bin/python3 scripts/oracle_gemma4_26b.py --ids 2,105,2364,107 --out findings/oracle_26b_short.json

  # Portable checkpoint path via CLI
  .venv-rocm/bin/python3 scripts/oracle_gemma4_26b.py --model /path/to/gemma-4-26B-A4B-it --ids-file /tmp/ids.txt --out findings/oracle_26b.json

  # Or set GEMMA4_26B_MODEL to avoid repeating --model
  GEMMA4_26B_MODEL=/path/to/gemma-4-26B-A4B-it .venv-rocm/bin/python3 scripts/oracle_gemma4_26b.py --ids-file /tmp/ids.txt --out findings/oracle_26b.json
"""
import argparse, json, sys, os, math

MODEL = "/local/models/google/gemma-4-26B-A4B-it"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF checkpoint path (overrides GEMMA4_26B_MODEL)")
    ap.add_argument("--ids", help="Comma-separated token IDs (e.g. 2,105,2364)")
    ap.add_argument("--ids-file", help="File with space-separated token IDs")
    ap.add_argument("--out", help="Output JSON path")
    ap.add_argument("--layers", help="Layers to dump (e.g. 0,1,5) or 'all'", default="0,1,5")
    ap.add_argument(
        "--boundaries",
        action="store_true",
        help="Also capture attention, dense-FFN, router, and MoE branch boundaries",
    )
    ap.add_argument(
        "--position",
        type=int,
        help="Absolute sequence position to capture (default: last position)",
    )
    return ap


def resolve_capture_position(position, n_ids):
    capture_position = n_ids - 1 if position is None else position
    if not 0 <= capture_position < n_ids:
        raise ValueError(
            f"capture position {capture_position} is outside sequence of length {n_ids}"
        )
    return capture_position


def finite_round(value, decimals):
    value = float(value)
    return round(value, decimals) if math.isfinite(value) else None
def format_stat(value, spec):
    return "None" if value is None else format(value, spec)


def tensor_at_position(tensor, position):
    if tensor.ndim >= 3:
        return tensor[0, position]
    if tensor.ndim == 2:
        return tensor[position]
    return tensor


def main():
    args = build_parser().parse_args()
    # Resolve in precedence order: CLI, environment, historical default.
    model_path = args.model if args.model is not None else os.environ.get("GEMMA4_26B_MODEL", MODEL)

    # Parse IDs
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    elif args.ids_file:
        ids = [int(x) for x in open(args.ids_file).read().split()]
    else:
        print("Need --ids or --ids-file", file=sys.stderr)
        sys.exit(1)
    try:
        capture_position = resolve_capture_position(args.position, len(ids))
    except ValueError as error:
        ap.error(str(error))

    # Parse layers to dump
    if args.layers == "all":
        dump_layers = None  # all
    else:
        dump_layers = set(int(x) for x in args.layers.split(","))

    print(f"IDs: {ids[:10]}{'...' if len(ids)>10 else ''} ({len(ids)} tokens)", file=sys.stderr)
    print(f"Loading model from {model_path} (float32 CPU)...", file=sys.stderr)

    import torch
    from transformers import AutoModelForCausalLM
    # Loading without `device_map` keeps this oracle usable in the lightweight
    # torch environment used on the validation host; the default device is CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).eval()

    # Get model config
    config = model.config
    text_config = config.text_config
    print(f"hidden_size={text_config.hidden_size} n_layers={text_config.num_hidden_layers} "
          f"n_heads={text_config.num_attention_heads} n_kv={text_config.num_key_value_heads} "
          f"head_dim={text_config.head_dim} global_n_kv={text_config.num_global_key_value_heads} "
          f"global_head_dim={text_config.global_head_dim} vocab={text_config.vocab_size} "
          f"moe={text_config.enable_moe_block} n_experts={text_config.num_experts} "
          f"top_k={text_config.top_k_experts}", file=sys.stderr)
    # Hook into model layers to capture intermediates.  Every record follows
    # the same position-indexed schema as the layer_out/embedding_last records.
    captured = {}


    def capture_tensor(key, tensor):
        t = tensor[0] if isinstance(tensor, tuple) else tensor
        v = tensor_at_position(t, capture_position).detach().float()
        captured[key] = {
            "first8": [finite_round(x, 5) for x in v[:8]],
            "sum": finite_round(v.sum(), 4),
            "norm": finite_round(v.norm(), 4),
            "min": finite_round(v.min(), 4),
            "max": finite_round(v.max(), 4),
        }

    def make_hook(layer_idx, name):
        def hook(module, input, output):
            capture_tensor(f"L{layer_idx}_{name}", output)

        return hook

    def make_pre_hook(layer_idx, name):
        def hook(module, input):
            capture_tensor(f"L{layer_idx}_{name}", input[0])

        return hook

    # Register hooks on decoder layers
    layers = model.model.language_model.layers
    hooks = []
    for li, layer in enumerate(layers):
        if dump_layers is not None and li not in dump_layers:
            continue
        hooks.append(layer.register_forward_hook(make_hook(li, "layer_out")))
        if args.boundaries:
            hooks.extend(
                [
                    layer.self_attn.register_forward_hook(make_hook(li, "attention_output")),
                    layer.post_attention_layernorm.register_forward_hook(
                        make_hook(li, "attention_norm")
                    ),
                    layer.pre_feedforward_layernorm.register_forward_pre_hook(
                        make_pre_hook(li, "attention_residual")
                    ),
                    layer.mlp.register_forward_pre_hook(
                        make_pre_hook(li, "pre_ffn_norm")
                    ),
                    layer.mlp.register_forward_hook(make_hook(li, "dense_ffn")),
                ]
            )
            if layer.enable_moe_block:
                hooks.extend(
                    [
                        layer.post_feedforward_layernorm_1.register_forward_hook(
                            make_hook(li, "dense_branch_norm")
                        ),
                        layer.pre_feedforward_layernorm_2.register_forward_hook(
                            make_hook(li, "moe_pre2")
                        ),
                        layer.router.proj.register_forward_hook(
                            make_hook(li, "router_logits")
                        ),
                        layer.experts.register_forward_hook(make_hook(li, "moe_branch")),
                        layer.post_feedforward_layernorm_2.register_forward_hook(
                            make_hook(li, "moe_branch_norm")
                        ),
                        layer.post_feedforward_layernorm.register_forward_pre_hook(
                            make_pre_hook(li, "moe_combined")
                        ),
                        layer.post_feedforward_layernorm.register_forward_hook(
                            make_hook(li, "outer_norm")
                        ),
                    ]
                )
            else:
                hooks.append(
                    layer.post_feedforward_layernorm.register_forward_hook(
                        make_hook(li, "dense_ffn_norm")
                    )
                )

    # Also capture embedding output
    def embed_hook(module, input, output):
        # output is the embedded tensor [1, seq, hidden].
        capture_tensor("embedding_last", output)
    hooks.append(model.model.language_model.embed_tokens.register_forward_hook(embed_hook))

    # Forward pass
    input_ids = torch.tensor([ids], device="cpu")
    print("Running forward pass...", file=sys.stderr)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Collect results at the same absolute position used by the lowered dump.
    logits = tensor_at_position(out.logits, capture_position).float()

    # Apply logit softcapping (Gemma4 does this inside the model, but verify)
    # model already applies it via Gemma4ForCausalLM → logit softcap
    topv, topi = torch.topk(logits, 20)
    result = {
        "model": model_path,
        "n_ids": len(ids),
        "position": capture_position,
        "ids_first10": ids[:10],
        "logits_top5": [
            [int(i), finite_round(x, 4)]
            for i, x in zip(topi[:5].tolist(), topv[:5].tolist())
        ],
        "logit_argmax": (
            int(topi[0]) if math.isfinite(float(topv[0])) else None
        ),
        "captured": captured,
        "hidden_states_layers": [],
    }

    # Also dump per-layer hidden states from the model's output_hidden_states.
    for li, hs in enumerate(out.hidden_states):
        v = tensor_at_position(hs, capture_position).float()
        result["hidden_states_layers"].append(
            {
                "layer": li,  # 0 = embeddings, 1..n = after decoder layer i-1
                "first8": [finite_round(x, 5) for x in v[:8]],
                "sum": finite_round(v.sum(), 4),
                "norm": finite_round(v.norm(), 4),
                "min": finite_round(v.min(), 4),
                "max": finite_round(v.max(), 4),
            }
        )

    if args.out:
        with open(args.out, "w") as output_file:
            json.dump(result, output_file, indent=2, allow_nan=False)
        print(f"Wrote {args.out}", file=sys.stderr)

    # Print summary
    print(f"\nargmax: {result['logit_argmax']}")
    print(f"top5: {result['logits_top5']}")
    print(f"\nPer-layer hidden states (position {capture_position}):")
    for l in result["hidden_states_layers"]:
        li = l["layer"]
        tag = "embed" if li == 0 else f"L{li-1}"
        print(
            f"  {tag:6s}: first4={l['first8'][:4]} "
            f"sum={format_stat(l['sum'], '+.2e')} "
            f"norm={format_stat(l['norm'], '.2f')}"
        )

    print(f"\nCaptured layer outputs (position {capture_position}):")
    for k in sorted(captured.keys()):
        c = captured[k]
        print(
            f"  {k:25s}: first4={c.get('first8', ['?'])[:4]} "
            f"sum={format_stat(c.get('sum'), '+.2e')}"
        )


if __name__ == "__main__":
    main()
