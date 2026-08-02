#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Reference per-position residual L2 — stream one Block at a time (GPU).

Imports model.py verbatim. Builds Transformer(n_layers=0) for globals+embed,
then constructs Block(layer_id) one at a time, loads that layer's weights,
runs forward, captures dense per-row HC residual L2, frees the block.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks

sys.modules["kernel"] = _ks
import torch
from model import Block, ModelArgs, Transformer
from weight_loader import _load_tensor, build_tensor_index, dequant_fp8_block128

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = (
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin"
)


def log(m=""):
    print(m, flush=True)


def load_tokens(path, n):
    raw = Path(path).read_bytes()
    ids = struct.unpack("<" + "i" * n, raw[: n * 4])
    return torch.tensor(ids, dtype=torch.long).view(1, n)


def make_args(config_path, max_seq_len):
    cfg = json.loads(Path(config_path).read_text())
    fields = ModelArgs.__dataclass_fields__
    kwargs = {}
    for k, v in cfg.items():
        if k not in fields:
            continue
        kwargs[k] = (
            tuple(v) if k in ("compress_ratios", "dspark_target_layer_ids") else v
        )
    a = ModelArgs(**kwargs)
    a.max_batch_size = 1
    a.max_seq_len = max_seq_len
    return a


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_embed_only(model, model_dir, device):
    index = build_tensor_index(Path(model_dir))
    w = _load_tensor(index, "embed.weight", device="cpu")
    with torch.no_grad():
        model.embed.weight.copy_(w.to(model.embed.weight.dtype))
    model.to(device)
    return ["embed.weight"]


def load_block_weights(block, model_dir, layer_id, device, verbose=True):
    index = build_tensor_index(Path(model_dir))
    prefix = f"layers.{layer_id}."
    sd = block.state_dict()
    assigned = {}
    loaded = []
    for key in sd.keys():
        src = prefix + key
        if src not in index:
            continue
        tensor = _load_tensor(index, src, device="cpu")
        if src.endswith("wo_a.weight") and tensor.dtype == torch.float8_e4m3fn:
            scale = _load_tensor(
                index, src.replace(".weight", ".scale"), device="cpu"
            )
            tensor = dequant_fp8_block128(tensor, scale).to(torch.bfloat16)
            if verbose:
                log(f"  dequant wo_a -> bf16 {tuple(tensor.shape)}")
        elif tensor.dtype == torch.int8 and ".ffn.experts." in src:
            tensor = tensor.view(torch.float4_e2m1fn_x2)
        elif tensor.dtype == torch.int64 and src.endswith("tid2eid"):
            tensor = tensor.to(torch.int32)
        dst = sd[key]
        if tuple(tensor.shape) != tuple(dst.shape):
            if not (
                tensor.dtype == torch.float4_e2m1fn_x2
                and tuple(tensor.shape) == tuple(dst.shape)
            ):
                raise RuntimeError(
                    f"shape mismatch {src}: ckpt {tuple(tensor.shape)} "
                    f"vs model {tuple(dst.shape)}"
                )
        if tensor.dtype != dst.dtype and dst.dtype != torch.float4_e2m1fn_x2:
            tensor = tensor.to(dst.dtype)
        assigned[key] = tensor
        loaded.append(src)
    incompat = block.load_state_dict(assigned, strict=False)
    if verbose:
        log(
            f"  L{layer_id} assigned {len(assigned)} tensors "
            f"missing={len(incompat.missing_keys)}"
        )

    for name, module in block.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight
        scale_key = f"{prefix}{name}.scale" if name else f"{prefix}scale"
        if scale_key not in index:
            continue
        if w.dtype not in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
            continue
        scale = _load_tensor(index, scale_key, device="cpu")
        if scale.dtype != torch.float8_e8m0fnu and scale.element_size() == 1:
            scale = scale.view(torch.float8_e8m0fnu)
        # keep on CPU for now; moved with block below
        w.scale = scale
        if hasattr(module, "scale"):
            try:
                module.scale = torch.nn.Parameter(scale, requires_grad=False)
            except Exception:
                module.scale = scale
        loaded.append(scale_key)

    block.to(device)
    # weight.scale is a plain attribute — block.to does not move it
    for name, module in block.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight
        if hasattr(w, "scale") and torch.is_tensor(w.scale):
            w.scale = w.scale.to(device)
        if hasattr(module, "scale") and torch.is_tensor(module.scale):
            if isinstance(module.scale, torch.nn.Parameter):
                module.scale.data = module.scale.data.to(device)
            else:
                module.scale = module.scale.to(device)
    return loaded


def row_l2_hc(h):
    x = h.detach().float()[0].reshape(h.size(1), -1)
    return x.norm(dim=-1).cpu()


def summarize_rows(rr, seq):
    early = float(rr[:128].mean()) if seq >= 128 else float(rr.mean())
    late = float(rr[-128:].mean()) if seq >= 128 else float(rr.mean())
    mid = float(rr[400:512].mean()) if seq >= 512 else None
    xs = list(range(0, seq, 32))
    ys = [math.log(max(float(rr[p]), 1e-12)) for p in xs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = num / den
    series = {str(p): float(rr[p]) for p in range(0, seq, 32)}
    series[str(seq - 1)] = float(rr[seq - 1])
    buckets = {}
    for start in range(0, seq, 64):
        end = min(start + 64, seq)
        buckets[f"[{start},{end})"] = float(rr[start:end].mean())
    return {
        "row_mean": float(rr.mean()),
        "row_std": float(rr.std()),
        "row_min": float(rr.min()),
        "row_max": float(rr.max()),
        "early128_mean": early,
        "late128_mean": late,
        "late_over_early": float(late / max(early, 1e-12)),
        "mid400_512_mean": mid,
        "log_l2_vs_pos_slope": slope,
        "approx_l2_ratio_per_512_tok": math.exp(slope * 512),
        "series_every32": series,
        "buckets_64": buckets,
        "dense_row_l2": [float(v) for v in rr.tolist()],
    }


@torch.inference_mode()
def run(args):
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    log(f"torch={torch.__version__} device={device}")
    if device.type == "cuda":
        log(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"vram={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GiB"
        )

    model_dir = Path(args.model)
    cfgp = model_dir / "inference" / "config.json"
    if not cfgp.exists():
        cfgp = HERE / "config.json"
    margs = make_args(cfgp, max(args.seq, 256))
    n_total = margs.n_layers
    if args.layers.strip() == "all":
        layers_keep = list(range(n_total))
    else:
        layers_keep = [int(x) for x in args.layers.split(",")]
    last_needed = max(layers_keep)
    log(f"seq={args.seq} stream_layers=0..{last_needed} capture={args.layers}")

    full_n = margs.n_layers
    margs.n_layers = 0
    margs.n_mtp_layers = 0
    margs.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        shell = Transformer(margs)
    margs.n_layers = full_n
    load_embed_only(shell, model_dir, device)
    shell.eval()
    if device.type == "cuda":
        log(f"embed loaded; vram_alloc={torch.cuda.memory_allocated()/1024**3:.2f}GiB")
    else:
        log("embed loaded cpu")

    tokens = load_tokens(args.tokens, args.seq).to(device)
    h = shell.embed(tokens).unsqueeze(2).repeat(1, 1, shell.hc_mult, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log(f"h0 global_L2={float(h.float().norm()):.6f}")

    del shell.head, shell.norm, shell.hc_head_fn, shell.hc_head_base, shell.hc_head_scale
    free_cuda()

    summary = {
        "seq": args.seq,
        "device": str(device),
        "torch": torch.__version__,
        "n_layers_run": last_needed + 1,
        "mode": "stream_one_block",
        "h0_global_l2": float(h.float().norm()),
        "embed": summarize_rows(row_l2_hc(h), args.seq),
        "global_l2_curve": {"embed": float(h.float().norm())},
        "layers": {},
        "notes": [
            "HC residual per-row L2 after each layer (model.py Block.forward)",
            "Streamed one Block at a time; model.py imported verbatim",
            "late_over_early = mean(row_l2[-128:]) / mean(row_l2[:128])",
        ],
    }
    log(f"embed late/early={summary['embed']['late_over_early']:.4f}")

    want = set(layers_keep)
    for i in range(last_needed + 1):
        t0 = time.time()
        with torch.device("cpu"):
            block = Block(i, margs)
        load_block_weights(block, model_dir, i, device, verbose=True)
        block.eval()
        t_load = time.time() - t0

        t1 = time.time()
        # model.py Indexer uses bare torch.arange; match __main__ default device
        if device.type == "cuda":
            torch.set_default_device("cuda")
        try:
            h = block(h, 0, tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
        finally:
            torch.set_default_device("cpu")
        t_fwd = time.time() - t1
        g = float(h.float().norm())
        ratio = (
            int(margs.compress_ratios[i]) if i < len(margs.compress_ratios) else -1
        )
        vram = (
            torch.cuda.memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        )
        log(
            f"L{i:02d} ratio={ratio} load={t_load:.1f}s fwd={t_fwd:.2f}s "
            f"global_L2={g:.4f} vram={vram:.2f}GiB"
        )

        summary["global_l2_curve"][str(i)] = g
        if i in want:
            rr = row_l2_hc(h)
            entry = summarize_rows(rr, args.seq)
            entry["layer"] = i
            entry["ratio"] = ratio
            entry["global_l2"] = g
            entry["load_s"] = t_load
            entry["fwd_s"] = t_fwd
            probes = [
                p
                for p in [
                    0,
                    64,
                    128,
                    200,
                    256,
                    400,
                    448,
                    512,
                    600,
                    768,
                    800,
                    1000,
                    1023,
                ]
                if p < args.seq
            ]
            entry["probes"] = {str(p): float(rr[p]) for p in probes}
            summary["layers"][str(i)] = entry
            log(
                f"     early={entry['early128_mean']:.4f} "
                f"late={entry['late128_mean']:.4f} "
                f"late/early={entry['late_over_early']:.4f} "
                f"mid400-512={entry['mid400_512_mean']}"
            )
            log(
                "     probes "
                + " ".join(f"{p}:{entry['probes'][str(p)]:.2f}" for p in probes)
            )

        del block
        free_cuda()

    keys = sorted(summary["layers"], key=int)
    growth = []
    for a, b in zip(keys, keys[1:]):
        ga = summary["layers"][a]["global_l2"]
        gb = summary["layers"][b]["global_l2"]
        ea = summary["layers"][a]["late_over_early"]
        eb = summary["layers"][b]["late_over_early"]
        growth.append(
            {
                "from": int(a),
                "to": int(b),
                "global_ratio": gb / max(ga, 1e-12),
                "late_over_early_from": ea,
                "late_over_early_to": eb,
                "late_over_early_delta": eb - ea,
            }
        )
    summary["growth_between_captured"] = growth
    if keys:
        last = keys[-1]
        summary["embed_to_last_global_growth"] = summary["layers"][last][
            "global_l2"
        ] / max(summary["h0_global_l2"], 1e-12)
        summary["last_late_over_early"] = summary["layers"][last]["late_over_early"]
        log(
            f"\nembed->L{last} global x{summary['embed_to_last_global_growth']:.3f}"
        )
        log(f"L{last} late/early={summary['last_late_over_early']:.4f}")

    compact = {k: v for k, v in summary.items() if k != "layers"}
    compact["layers"] = {}
    for Lk, e in summary["layers"].items():
        compact["layers"][Lk] = {
            kk: vv for kk, vv in e.items() if kk != "dense_row_l2"
        }
    Path(args.out).write_text(json.dumps(summary))
    Path(args.out + ".compact.json").write_text(json.dumps(compact, indent=2))
    log(f"wrote {args.out} and compact")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out", default="/tmp/residual_pos_traj.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
