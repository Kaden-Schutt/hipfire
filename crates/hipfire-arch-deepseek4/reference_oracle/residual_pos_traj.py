#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Reference per-position residual L2 trajectory (model.py verbatim).

For layers L0,L10,L20,L30,L38,L42 at seq=1024, report:
  - per-row HC residual L2 after each named layer (shape [S])
  - ratio late/early (mean L2 of last 128 vs first 128)
  - geo growth of global L2 across those layers
CPU-only; experts loaded so residual chain is real.
"""
from __future__ import annotations
import argparse, json, os, struct, sys, time
from pathlib import Path
from typing import Dict, List
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks
sys.modules["kernel"] = _ks
import torch
from model import ModelArgs, Transformer
from weight_loader import load_state_into_model

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin"
DEFAULT_LAYERS = [0,10,20,30,38,42]
PROBES = [0,32,64,128,200,256,400,448,512,600,768,800,1000,1023]

def log(m=""): print(m, flush=True)

def load_tokens(path, n):
    raw = Path(path).read_bytes()
    ids = struct.unpack("<"+"i"*n, raw[:n*4])
    return torch.tensor(ids, dtype=torch.long).view(1,n)

def make_args(config_path, max_seq_len):
    cfg = json.loads(Path(config_path).read_text())
    fields = ModelArgs.__dataclass_fields__
    kwargs = {}
    for k,v in cfg.items():
        if k not in fields: continue
        kwargs[k] = tuple(v) if k in ("compress_ratios","dspark_target_layer_ids") else v
    a = ModelArgs(**kwargs)
    a.max_batch_size = 1
    a.max_seq_len = max_seq_len
    return a

def build_model(args, model_dir, n_build):
    full = args.n_layers
    args.n_layers = n_build
    args.n_mtp_layers = 0
    args.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        model = Transformer(args)
    args.n_layers = full
    load_state_into_model(model, model_dir, layers=range(n_build), load_experts=True, device="cpu", verbose=True)
    model.eval()
    return model

def reset_caches(model):
    for layer in model.layers:
        a = layer.attn
        a.kv_cache.zero_()
        c = getattr(a, "compressor", None)
        if c is not None:
            c.kv_state.zero_(); c.score_state.fill_(float("-inf")); c.kv_cache = None
        ix = getattr(a, "indexer", None)
        if ix is not None:
            ix.kv_cache.zero_()
            ix.compressor.kv_state.zero_(); ix.compressor.score_state.fill_(float("-inf"))
            ix.compressor.kv_cache = None

def row_l2_hc(h):
    # h: [B,S,hc,D] -> [S] L2 over hc*D
    x = h.float()[0].reshape(h.size(1), -1)
    return x.norm(dim=-1)

@torch.inference_mode()
def run(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try: torch.set_default_device("cpu")
    except Exception: pass
    model_dir = Path(args.model)
    cfgp = model_dir/"inference"/"config.json"
    if not cfgp.exists(): cfgp = HERE/"config.json"
    layers = [int(x) for x in args.layers.split(",")]
    margs = make_args(cfgp, max(args.seq, 256))
    n_build = max(layers) + 1
    log(f"seq={args.seq} layers={layers} n_build={n_build}")
    log(f"compress_ratios snapshot={[margs.compress_ratios[i] for i in layers]}")

    t0=time.time()
    model = build_model(margs, model_dir, n_build)
    log(f"load {time.time()-t0:.1f}s")
    tokens = load_tokens(args.tokens, args.seq)
    reset_caches(model)
    h = model.embed(tokens).unsqueeze(2).repeat(1,1,model.hc_mult,1)
    log(f"h0 global_L2={float(h.float().norm()):.6f}")

    want = set(layers)
    probes = [p for p in PROBES if p < args.seq]
    # also every 32 for series
    series_pos = sorted(set(list(range(0, args.seq, 32)) + probes + [args.seq-1]))

    summary = {
        "seq": args.seq,
        "layers_requested": layers,
        "h0_global_l2": float(h.float().norm()),
        "layers": {},
        "global_l2_curve": {"-1_embed": float(h.float().norm())},
    }

    # dump row L2 after embed
    r0 = row_l2_hc(h)
    summary["embed_row"] = {
        "global_l2": float(h.float().norm()),
        "row_mean": float(r0.mean()),
        "row_std": float(r0.std()),
        "early128_mean": float(r0[:128].mean()),
        "late128_mean": float(r0[-128:].mean()),
        "late_over_early": float(r0[-128:].mean() / r0[:128].mean().clamp(min=1e-12)),
        "probes": {str(p): float(r0[p]) for p in probes},
        "series": {str(p): float(r0[p]) for p in series_pos},
    }
    log(f"embed early128={summary['embed_row']['early128_mean']:.4f} late128={summary['embed_row']['late128_mean']:.4f} late/early={summary['embed_row']['late_over_early']:.4f}")

    for i in range(n_build):
        t1=time.time()
        h = model.layers[i](h, 0, tokens)
        g = float(h.float().norm())
        summary["global_l2_curve"][str(i)] = g
        log(f"  L{i} {time.time()-t1:.1f}s global_L2={g:.4f} ratio={margs.compress_ratios[i]}")
        if i in want:
            rr = row_l2_hc(h)
            early = float(rr[:128].mean())
            late = float(rr[-128:].mean())
            mid = float(rr[400:512].mean()) if args.seq >= 512 else None
            entry = {
                "layer": i,
                "ratio": int(margs.compress_ratios[i]),
                "global_l2": g,
                "row_mean": float(rr.mean()),
                "row_std": float(rr.std()),
                "row_min": float(rr.min()),
                "row_max": float(rr.max()),
                "early128_mean": early,
                "late128_mean": late,
                "late_over_early": float(late / max(early, 1e-12)),
                "mid400_512_mean": mid,
                "probes": {str(p): float(rr[p]) for p in probes},
                "series_every32": {str(p): float(rr[p]) for p in series_pos},
            }
            # local slope: linear fit log(row_l2) vs position (rough)
            import math
            xs = list(range(0, args.seq, 32))
            ys = [math.log(max(float(rr[p]), 1e-12)) for p in xs]
            n = len(xs)
            mx = sum(xs)/n; my = sum(ys)/n
            num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
            den = sum((x-mx)**2 for x in xs) or 1.0
            slope = num/den  # d log(l2) / d pos
            entry["log_l2_vs_pos_slope"] = slope
            entry["approx_l2_ratio_per_512_tok"] = math.exp(slope * 512)
            summary["layers"][str(i)] = entry
            log(f"    early128={early:.4f} late128={late:.4f} late/early={entry['late_over_early']:.4f} mid400-512={mid} slope_log={slope:.6e} per512={entry['approx_l2_ratio_per_512_tok']:.4f}")
            log("    probes: " + " ".join(f"{p}:{entry['probes'][str(p)]:.3f}" for p in probes))

    # growth table across captured layers
    keys = [str(i) for i in layers if str(i) in summary["layers"]]
    growth = []
    for a,b in zip(keys, keys[1:]):
        ga = summary["layers"][a]["global_l2"]
        gb = summary["layers"][b]["global_l2"]
        ea = summary["layers"][a]["late_over_early"]
        eb = summary["layers"][b]["late_over_early"]
        growth.append({
            "from": int(a), "to": int(b),
            "global_ratio": gb/max(ga,1e-12),
            "late_over_early_from": ea,
            "late_over_early_to": eb,
            "late_over_early_delta": eb-ea,
        })
        log(f"growth L{a}->L{b}: global×{gb/max(ga,1e-12):.4f} late/early {ea:.4f}->{eb:.4f} (Δ{eb-ea:+.4f})")
    summary["growth_between_captured"] = growth

    # overall embed->last
    last = str(layers[-1])
    if last in summary["layers"]:
        summary["embed_to_last_global_growth"] = summary["layers"][last]["global_l2"] / max(summary["h0_global_l2"], 1e-12)
        summary["last_late_over_early"] = summary["layers"][last]["late_over_early"]
        log(f"\nembed->L{last} global growth ×{summary['embed_to_last_global_growth']:.3f}")
        log(f"L{last} late/early = {summary['last_late_over_early']:.4f}")

    Path(args.out).write_text(json.dumps(summary, indent=2))
    log(f"wrote {args.out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--out", default="/tmp/residual_pos_traj.json")
    args = ap.parse_args(); run(args)
if __name__ == "__main__":
    main()
