#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# HF/PyTorch reference dump for the nemotron_h numeric and generation-boundary
# bisect. Runs NemotronHForCausalLM on either raw text, explicit token IDs, or the
# same single-turn Jinja ChatML prompt shape Hipfire serves by default, then saves
# per-layer hidden states plus logits to an .npz consumed by compare_bisect.py.
#
# The default `--mamba-import stub --mamba-reference native` disables the
# checkpoint's CUDA/Triton Mamba imports and monkeypatches each Nemotron Mamba
# mixer to call Transformers' maintained Mamba2Mixer.torch_forward. Use
# `--mamba-import real --mamba-reference remote` when a known-good mamba_ssm
# install is available and the checkpoint's remote-code fast path is the thing
# being checked.
#
#   python3 benchmarks/nemotron/dump_hf_reference.py \
#       --model /srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/<snap> \
#       --mode jinja --thinking off \
#       --text 'Answer in one short sentence: What is 2+2?' \
#       --max-new-tokens 1 \
#       --out /tmp/nemo_hf_ref.npz
#
# Output npz keys:
#   input_ids [T]
#   hidden_<L> [T,H] for L in 0..=num_layers (0 = embeddings, L>0 = after block L-1)
#   final_norm_hidden [T,H]
#   logits [T,V]
#   generated_ids [G]
#   step_top_ids [G,top_k]
#   step_top_logits [G,top_k]

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import struct
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_MODEL = (
    "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
)
DEFAULT_TEXT = "The capital of France is"


def _install_mamba_ssm_stub() -> None:
    """Let the remote Nemotron code import without CUDA-only mamba-ssm.

    The installed modules intentionally leave the SSD/conv fast-path kernels as
    None, so the model falls back to Python/PyTorch. `rmsnorm_fn` is provided
    because the remote module imports it unconditionally.
    """

    def rmsnorm_fn(
        x,
        weight,
        bias=None,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        **_kw,
    ):
        dtype = x.dtype
        x = x.float()
        if z is not None and not norm_before_gate:
            x = x * F.silu(z.float())
        if group_size is None:
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + eps)
        else:
            shape = x.shape
            xg = x.view(*shape[:-1], shape[-1] // group_size, group_size)
            var = xg.pow(2).mean(-1, keepdim=True)
            x = (xg * torch.rsqrt(var + eps)).view(shape)
        out = x * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None and norm_before_gate:
            out = out * F.silu(z.float())
        return out.to(dtype)

    import importlib.util

    def mod(name: str, **attrs) -> None:
        m = types.ModuleType(name)
        m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        m.__path__ = []
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    mod("mamba_ssm", __version__="2.2.2")
    mod("mamba_ssm.ops")
    mod("mamba_ssm.ops.triton")
    mod("mamba_ssm.ops.triton.layernorm_gated", rmsnorm_fn=rmsnorm_fn)
    mod("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
    mod(
        "mamba_ssm.ops.triton.ssd_combined",
        mamba_chunk_scan_combined=None,
        mamba_split_conv1d_scan_combined=None,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dump a Nemotron-H HF/PyTorch reference for Hipfire comparison.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="/tmp/nemo_hf_ref.npz")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--prompt-file")
    ap.add_argument("--system", default=None)
    ap.add_argument("--token-ids", default="", help="comma-separated ids; overrides --text")
    ap.add_argument(
        "--mode",
        choices=["raw", "jinja", "plain"],
        default="raw",
        help="raw encodes text directly; jinja uses tokenizer.apply_chat_template; plain hand-rolls ChatML.",
    )
    ap.add_argument(
        "--thinking",
        choices=["off", "on"],
        default="off",
        help="For --mode jinja: off passes enable_thinking=False, matching Hipfire max_think_tokens=1.",
    )
    ap.add_argument(
        "--assistant-prefix",
        choices=["plain", "open_think", "closed_think"],
        default="closed_think",
        help="For --mode plain only; mirrors hipfire-prompt AssistantPrefix.",
    )
    ap.add_argument(
        "--mamba-reference",
        choices=["native", "remote"],
        default="native",
        help="native patches Nemotron Mamba mixers to Transformers Mamba2Mixer.torch_forward.",
    )
    ap.add_argument(
        "--mamba-import",
        choices=["stub", "real"],
        default="stub",
        help="stub disables CUDA/Triton mamba-ssm imports; real uses installed mamba-ssm kernels.",
    )
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Model dtype for the HF reference.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-stop-on-eos", action="store_true")
    ap.add_argument("--max-layers-print", type=int, default=0)
    ap.add_argument("--print-rendered", action="store_true")
    return ap.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def encode_no_special(tok, text: str) -> list[int]:
    return [int(x) for x in tok.encode(text, add_special_tokens=False)]


def prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    return args.text


def render_prompt(tok, args: argparse.Namespace) -> tuple[list[int], str, list[dict[str, str]]]:
    if args.token_ids:
        ids = [int(x) for x in args.token_ids.split(",") if x.strip()]
        return ids, "<explicit token ids>", []

    text = prompt_text(args)
    if args.mode == "raw":
        return encode_no_special(tok, text), text, []

    messages: list[dict[str, str]] = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": text})

    if args.mode == "jinja":
        rendered = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking == "on",
        )
        return encode_no_special(tok, rendered), rendered, messages

    parts: list[str] = []
    if args.system is not None:
        parts.append(f"<|im_start|>system\n{args.system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if args.assistant_prefix == "open_think":
        parts.append("<think>\n")
    elif args.assistant_prefix == "closed_think":
        parts.append("<think>\n\n</think>\n\n")
    rendered = "".join(parts)
    return encode_no_special(tok, rendered), rendered, messages


def restore_dt_bias(model, model_dir: Path) -> int:
    """Restore trained dt_bias tensors from safetensors after HF re-init."""

    files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {model_dir}")

    restored = 0
    # Nano-4B is one safetensors file today; loop keeps the helper shard-safe.
    for sfile in files:
        with open(sfile, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
            data_base = 8 + header_len
            for name, param in model.named_parameters():
                if not name.endswith(".dt_bias") or name not in header:
                    continue
                o0, o1 = header[name]["data_offsets"]
                fh.seek(data_base + o0)
                raw = fh.read(o1 - o0)
                dtype = header[name]["dtype"]
                if dtype == "BF16":
                    vals = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                    vals = vals.view(np.float32)
                elif dtype == "F32":
                    vals = np.frombuffer(raw, np.float32)
                else:
                    raise ValueError(f"unsupported dt_bias dtype {dtype} in {sfile}")
                with torch.no_grad():
                    param.copy_(torch.from_numpy(vals.copy()).to(param.device, param.dtype))
                restored += 1
    return restored


def patch_native_mamba2(model) -> int:
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    native_params = inspect.signature(Mamba2Mixer.torch_forward).parameters
    accepts_cache_position = "cache_position" in native_params

    def native_torch_forward(
        self,
        input_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        kwargs = {
            "cache_params": cache_params,
            "attention_mask": attention_mask,
        }
        if accepts_cache_position:
            kwargs["cache_position"] = cache_position
        return Mamba2Mixer.torch_forward(
            self,
            input_states,
            **kwargs,
        )

    patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "NemotronHMamba2Mixer":
            module.torch_forward = types.MethodType(native_torch_forward, module)
            patched += 1
    return patched


def topk(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    vals, ids = torch.topk(logits.float(), k=k)
    return [int(x) for x in ids.tolist()], [float(x) for x in vals.tolist()]


def install_block_capture_hooks(model):
    layers = model.backbone.layers
    captures = [None] * len(layers)
    handles = []

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            captures[idx] = out.detach().float().cpu()

        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))
    return handles, captures


def generate_greedy(
    model,
    input_ids: list[int],
    tok,
    args: argparse.Namespace,
) -> tuple[list[int], list[list[int]], list[list[float]]]:
    generated: list[int] = []
    all_top_ids: list[list[int]] = []
    all_top_logits: list[list[float]] = []
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        eos_ids.add(im_end)

    cur = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            t0 = time.time()
            out = model(cur, use_cache=False)
            logits = out.logits[0, -1, :]
            ids, vals = topk(logits, args.top_k)
            next_id = ids[0]
            generated.append(next_id)
            all_top_ids.append(ids)
            all_top_logits.append(vals)
            print(
                f"step {step}: token={next_id} text={tok.decode([next_id])!r} "
                f"top2_margin={vals[0] - vals[1] if len(vals) > 1 else float('nan'):.5f} "
                f"elapsed={time.time() - t0:.2f}s",
                flush=True,
            )
            if not args.no_stop_on_eos and next_id in eos_ids:
                break
            cur = torch.cat(
                [cur, torch.tensor([[next_id]], dtype=torch.long, device=args.device)],
                dim=1,
            )
    return generated, all_top_ids, all_top_logits


def main() -> int:
    args = parse_args()
    if args.mamba_import == "stub":
        _install_mamba_ssm_stub()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        print(f"error: model dir not found: {model_dir}", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    input_ids, rendered, messages = render_prompt(tok, args)
    if not input_ids:
        print("error: prompt rendered to zero tokens", file=sys.stderr)
        return 2

    rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    print(f"prompt mode={args.mode} thinking={args.thinking} tokens={len(input_ids)}")
    print(f"input_ids: {input_ids}")
    print(f"rendered_sha256: {rendered_sha}")
    if args.print_rendered:
        print("rendered_prompt:")
        print(rendered)

    print(
        f"loading model dtype={args.dtype} device={args.device} mamba_reference={args.mamba_reference}",
        flush=True,
    )
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
    ).to(args.device)
    model.eval()
    print(f"loaded model in {time.time() - t_load:.2f}s", flush=True)

    restored = restore_dt_bias(model, model_dir)
    print(f"restored {restored} stored dt_bias tensors")
    patched = 0
    if args.mamba_reference == "native":
        patched = patch_native_mamba2(model)
        print(f"patched {patched} Mamba mixers to native Transformers Mamba2")

    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=args.device)
    handles, block_captures = install_block_capture_hooks(model)
    with torch.no_grad():
        out = model(ids_tensor, output_hidden_states=True, use_cache=False)
    for h in handles:
        h.remove()

    # HF's `output_hidden_states` for this remote model contains:
    #   hidden_states[0] = embeddings before block 0
    #   hidden_states[-1] = after final norm_f
    # It does NOT contain "after final block, before norm_f", which is what the
    # Hipfire bisect dump records as hidden_42. The hooks above capture every
    # block output so hidden_<L> is aligned to Hipfire's dump layout.
    hs = [out.hidden_states[0].detach().float().cpu()]
    for idx, cap in enumerate(block_captures):
        if cap is None:
            raise RuntimeError(f"missing block capture for layer {idx}")
        hs.append(cap)
    final_norm_hidden = out.hidden_states[-1].detach().float().cpu()
    logits = out.logits
    final_ids, final_vals = topk(logits[0, -1], args.top_k)
    print("final-pos top ids:", final_ids)
    print("final-pos top text:", [tok.decode([i]) for i in final_ids])
    print("final-pos top logits:", [round(v, 6) for v in final_vals])

    generated, step_top_ids, step_top_logits = generate_greedy(model, input_ids, tok, args)

    save = {"input_ids": np.array(input_ids, dtype=np.int64)}
    for i, h in enumerate(hs):
        save[f"hidden_{i}"] = h[0].numpy()
    save["final_norm_hidden"] = final_norm_hidden[0].numpy()
    save["logits"] = logits[0].float().cpu().numpy()
    save["generated_ids"] = np.array(generated, dtype=np.int64)
    save["step_top_ids"] = np.array(step_top_ids, dtype=np.int64)
    save["step_top_logits"] = np.array(step_top_logits, dtype=np.float32)
    np.savez(args.out, **save)

    meta = {
        "model": str(model_dir),
        "mode": args.mode,
        "thinking": args.thinking,
        "assistant_prefix": args.assistant_prefix,
        "mamba_reference": args.mamba_reference,
        "mamba_import": args.mamba_import,
        "dtype": args.dtype,
        "device": args.device,
        "input_ids": input_ids,
        "rendered_sha256": rendered_sha,
        "messages": messages,
        "generated_ids": generated,
        "generated_text": tok.decode(generated) if generated else "",
        "final_top_ids": final_ids,
        "final_top_text": [tok.decode([i]) for i in final_ids],
        "final_top_logits": final_vals,
        "dt_bias_restored": restored,
        "mamba_mixers_patched": patched,
        "torch": torch.__version__,
    }
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {len(hs)} hidden states + logits to {args.out}")
    print(f"saved metadata to {meta_path}")

    if args.max_layers_print:
        for i in range(min(args.max_layers_print, len(hs))):
            h = hs[i][0, -1]
            print(f"  hidden_{i}[last]: mean={h.float().mean():.4f} std={h.float().std():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
