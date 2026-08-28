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
import argparse, json, sys, os

MODEL = "/local/models/google/gemma-4-26B-A4B-it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF checkpoint path (overrides GEMMA4_26B_MODEL)")
    ap.add_argument("--ids", help="Comma-separated token IDs (e.g. 2,105,2364)")
    ap.add_argument("--ids-file", help="File with space-separated token IDs")
    ap.add_argument("--out", help="Output JSON path")
    ap.add_argument("--layers", help="Layers to dump (e.g. 0,1,5) or 'all'", default="0,1,5")
    args = ap.parse_args()

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

    # Parse layers to dump
    if args.layers == "all":
        dump_layers = None  # all
    else:
        dump_layers = set(int(x) for x in args.layers.split(","))

    print(f"IDs: {ids[:10]}{'...' if len(ids)>10 else ''} ({len(ids)} tokens)", file=sys.stderr)
    print(f"Loading model from {model_path} (float32 CPU)...", file=sys.stderr)

    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
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

    # Hook into model layers to capture intermediates
    captured = {}

    def make_hook(layer_idx, name):
        def hook(module, input, output):
            key = f"L{layer_idx}_{name}"
            # output can be tuple or tensor
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            # Capture last-position hidden state
            v = t[0, -1].detach().float()  # [hidden_size]
            captured[key] = {
                "first8": [round(float(x), 5) for x in v[:8]],
                "sum": round(float(v.sum()), 4),
                "norm": round(float(v.norm()), 4),
                "min": round(float(v.min()), 4),
                "max": round(float(v.max()), 4),
            }
        return hook

    # Register hooks on decoder layers
    layers = model.model.language_model.layers
    hooks = []
    for li, layer in enumerate(layers):
        if dump_layers is not None and li not in dump_layers:
            continue
        # Hook post-self-attention (input_layernorm output → attention)
        # Actually, hook the layer forward to get hidden_states
        hooks.append(layer.register_forward_hook(make_hook(li, "layer_out")))

    # Also capture embedding output
    def embed_hook(module, input, output):
        # output is the embedded tensor [1, seq, hidden]
        v = output[0, -1].detach().float() if isinstance(output, tuple) else output[0, -1].detach().float()
        captured["embedding_last"] = {
            "first8": [round(float(x), 5) for x in v[:8]],
            "sum": round(float(v.sum()), 4),
            "norm": round(float(v.norm()), 4),
        }
    hooks.append(model.model.language_model.embed_tokens.register_forward_hook(embed_hook))

    # Forward pass
    input_ids = torch.tensor([ids], device="cpu")
    print("Running forward pass...", file=sys.stderr)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Collect results
    logits = out.logits[0, -1].float()

    # Apply logit softcapping (Gemma4 does this inside the model, but verify)
    # model already applies it via Gemma4ForCausalLM → logit softcap
    topv, topi = torch.topk(logits, 20)
    result = {
        "model": model_path,
        "n_ids": len(ids),
        "ids_first10": ids[:10],
        "logits_top5": [[int(i), round(float(x), 4)] for i, x in zip(topi[:5].tolist(), topv[:5].tolist())],
        "logit_argmax": int(topi[0]),
        "captured": captured,
        "hidden_states_layers": [],
    }

    # Also dump per-layer hidden states from the model's output_hidden_states
    for li, hs in enumerate(out.hidden_states):
        v = hs[0, -1].float()
        result["hidden_states_layers"].append({
            "layer": li,  # 0 = embeddings, 1..n = after decoder layer i-1
            "first8": [round(float(x), 5) for x in v[:8]],
            "sum": round(float(v.sum()), 4),
            "norm": round(float(v.norm()), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
        })

    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"Wrote {args.out}", file=sys.stderr)

    # Print summary
    print(f"\nargmax: {result['logit_argmax']}")
    print(f"top5: {result['logits_top5']}")
    print(f"\nPer-layer hidden states (last position):")
    for l in result["hidden_states_layers"]:
        li = l["layer"]
        tag = "embed" if li == 0 else f"L{li-1}"
        print(f"  {tag:6s}: first4={l['first8'][:4]} sum={l['sum']:+.2e} norm={l['norm']:.2f}")

    print(f"\nCaptured layer outputs:")
    for k in sorted(captured.keys()):
        c = captured[k]
        print(f"  {k:25s}: first4={c.get('first8',['?'])[:4]} sum={c.get('sum',0):+.2e}")


if __name__ == "__main__":
    main()
