#!/usr/bin/env python3
"""Dump ZAYA1 PyTorch reference activations for hipfire arch-intake.

Phase 1 of the ZAYA1 port intake (see
docs/investigations/2026-05-07-zaya1-port-intake/). Loads the HF model
via Zyphra's transformers fork (trust_remote_code=True), registers
forward hooks at every meaningful submodule, runs ONE forward on a
fixed canonical prompt, and dumps each captured tensor as a
single-tensor safetensors file plus a manifest.

Layout (matches the methodology in CLAUDE.md / scripts/arch-intake/README.md):

    <output>/manifest.json           # input_ids, prompt md5, shapes, dtypes
    <output>/layer_NN/<step>.<side>.safetensors

Side names follow the ZayaBlock structure (modeling_zaya.py:1197):
  - pre_norm                (block input post-norm, pre-CCA)
  - cca_q / cca_k / cca_v   (CCA outputs feeding ZayaAttention)
  - post_attn               (attention block output)
  - post_attn_norm          (norm output before MLP/MoE)
  - moe_route_logits        (MLP router logits)
  - moe_out                 (MoE block output)
  - post_residual           (block output post-residual-add)

Phase 1 runs only PREFILL (no decode). Decode hooks land alongside the
hipfire-side decode_step implementation (Phase 2+).

Usage:
    HIP_VISIBLE_DEVICES=2 python scripts/arch-intake/dump_zaya_reference.py \\
        --model Zyphra/ZAYA1-8B \\
        --prompt scripts/arch-intake/prompts/zaya_canonical.txt \\
        --output /tmp/zaya-port/refs/canonical/
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Zyphra/ZAYA1-8B",
                   help="HF model id or local path (default: Zyphra/ZAYA1-8B)")
    p.add_argument("--prompt", required=True,
                   help="Path to a text file containing the canonical prompt (committed for repeatability)")
    p.add_argument("--output", required=True,
                   help="Directory under which manifest.json and layer_NN/ subdirs are written")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Model dtype for the reference forward (default: bf16)")
    p.add_argument("--max-layers", type=int, default=None,
                   help="If set, hook only the first N layers (smoke-test mode)")
    return p.parse_args()


def lazy_import_torch():
    """Defer torch / transformers imports so --help works without a venv."""
    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.stderr.write(
            f"ImportError: {e}\n\n"
            "This script requires:\n"
            "  pip install --user safetensors\n"
            "  pip install --user 'transformers @ git+https://github.com/Zyphra/transformers.git@zaya1'\n"
            "  pip install --user torch  # bf16-capable build\n"
        )
        sys.exit(2)
    return torch, save_file, AutoModelForCausalLM, AutoTokenizer


def hook_target_modules(model, max_layers=None):
    """Yield (label, module) for every module we want to hook.

    The decoder layer naming follows Zyphra's ZayaModel:
      model.model.layers[i] is a ZayaBlock
      .layers[i].pre_attn_norm / .self_attn / .pre_mlp_norm / .mlp / etc
    Concrete attribute names differ between modeling_zaya.py and
    modular_zaya.py — this hook list uses defensive `getattr` lookups
    so it works against both.
    """
    layers = model.model.layers
    if max_layers is not None:
        layers = layers[:max_layers]

    for i, layer in enumerate(layers):
        # Names below follow modeling_zaya.py:1197 (ZayaBlock). Each
        # getattr is wrapped so a missing attr emits a None which we
        # filter at registration time.
        candidates = [
            ("pre_norm", getattr(layer, "input_layernorm", None) or getattr(layer, "pre_attn_norm", None)),
            ("cca", getattr(getattr(layer, "self_attn", layer), "cca", None)),
            ("self_attn", getattr(layer, "self_attn", None)),
            ("post_attn_norm", getattr(layer, "post_attention_layernorm", None) or getattr(layer, "pre_mlp_norm", None)),
            ("mlp_router", getattr(getattr(layer, "mlp", layer), "router", None)),
            ("mlp", getattr(layer, "mlp", None)),
            ("block_out", layer),  # Hooks the ZayaBlock's own forward output
        ]
        for side, mod in candidates:
            if mod is not None:
                yield (i, side, mod)

    # Final norm + lm_head
    yield (-1, "final_norm", getattr(model.model, "norm", None) or getattr(model.model, "final_layernorm", None))
    yield (-1, "lm_head", model.lm_head)


def main():
    args = parse_args()
    torch, save_file, AutoModelForCausalLM, AutoTokenizer = lazy_import_torch()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt)
    prompt_bytes = prompt_path.read_bytes()
    prompt_md5 = hashlib.md5(prompt_bytes).hexdigest()
    prompt_text = prompt_bytes.decode("utf-8").rstrip("\n")

    print(f"[zaya-ref] prompt md5: {prompt_md5}")
    print(f"[zaya-ref] prompt    : {prompt_text!r}")
    print(f"[zaya-ref] model     : {args.model}")
    print(f"[zaya-ref] dtype     : {args.dtype}")
    print(f"[zaya-ref] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")

    visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if visible.strip() in ("", "0", "0,1", "1"):
        print(
            "[zaya-ref] WARNING: HIP_VISIBLE_DEVICES is empty or includes lane 0/1.\n"
            "[zaya-ref]          The contract reserves GPUs 2 and 3 for the zaya port;\n"
            "[zaya-ref]          0 and 1 belong to the concurrent gemma-eseries agent.\n"
            "[zaya-ref]          Set HIP_VISIBLE_DEVICES=2 before running.",
            file=sys.stderr,
        )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("[zaya-ref] loading tokenizer + model (trust_remote_code=True) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cuda:0",  # HIP_VISIBLE_DEVICES already remapped to lane-local 0
        trust_remote_code=True,
    )
    model.eval()

    print(f"[zaya-ref] tokenizing ...")
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda:0")
    print(f"[zaya-ref] input_ids shape: {tuple(input_ids.shape)}, tokens: {input_ids[0].tolist()[:20]}{'...' if input_ids.shape[1] > 20 else ''}")

    captures = {}  # (layer_idx, side) -> tensor (cpu, fp32)

    def make_hook(layer_idx, side):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            if not hasattr(t, "detach"):
                return  # non-tensor output (e.g. cache obj) - skip
            captures[(layer_idx, side)] = t.detach().to("cpu", dtype=torch.float32).contiguous()
        return hook

    handles = []
    for layer_idx, side, mod in hook_target_modules(model, max_layers=args.max_layers):
        handles.append(mod.register_forward_hook(make_hook(layer_idx, side)))
    print(f"[zaya-ref] registered {len(handles)} hooks")

    print("[zaya-ref] running prefill forward ...")
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)

    for h in handles:
        h.remove()

    print(f"[zaya-ref] captured {len(captures)} tensors; writing safetensors ...")
    manifest_entries = []
    for (layer_idx, side), tensor in sorted(captures.items()):
        layer_name = "final" if layer_idx < 0 else f"layer_{layer_idx:02d}"
        sub = out / layer_name
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"prefill.{side}.safetensors"
        save_file({"x": tensor}, str(path))
        manifest_entries.append({
            "layer": layer_idx,
            "side": side,
            "shape": list(tensor.shape),
            "dtype": "float32",
            "path": str(path.relative_to(out)),
        })

    manifest = {
        "model": args.model,
        "dtype": args.dtype,
        "prompt_md5": prompt_md5,
        "prompt_path": str(prompt_path),
        "input_ids": input_ids[0].tolist(),
        "tensors": manifest_entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[zaya-ref] wrote manifest.json + {len(manifest_entries)} tensors under {out}")


if __name__ == "__main__":
    main()
