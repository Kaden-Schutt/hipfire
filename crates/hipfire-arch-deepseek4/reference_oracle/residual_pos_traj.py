#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Reference per-position residual L2 trajectory — full stack, GPU if available."""
from __future__ import annotations
import argparse, json, math, os, struct, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks
sys.modules["kernel"] = _ks
import torch
from model import ModelArgs, Transformer
from weight_loader import load_state_into_model

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin"

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
        kwargs[k] = tuple(v) if k in ("compress_ratios", "dspark_target_layer_ids") else v
    a = ModelArgs(**kwargs)
    a.max_batch_size = 1
    a.max_seq_len = max_seq_len
    return a

def build_model(args, model_dir, n_build, device):
    full = args.n_layers
    args.n_layers = n_build
    args.n_mtp_layers = 0
    args.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        model = Transformer(args)
    args.n_layers = full
    load_state_into_model(
        model, model_dir, layers=range(n_build), load_experts=True,
        device="cpu", verbose=True,
    )
    if device.type == "cuda":
        log(f"moving model to {device} ...")
        t0 = time.time()
        model = model.to(device)
        torch.cuda.synchronize()
        log(f"move done in {time.time()-t0:.1f}s vram_alloc={torch.cuda.memory_allocated()/1024**3:.2f}GiB")
    model.eval()
    return model

def reset_caches(model):
    for layer in model.layers:
        a = layer.attn
        a.kv_cache.zero_()
        c = getattr(a, "compressor", None)
        if c is not None:
            c.kv_state.zero_()
            c.score_state.fill_(float("-inf"))
            c.kv_cache = None
        ix = getattr(a, "indexer", None)
        if ix is not None:
            ix.kv_cache.zero_()
            ix.compressor.kv_state.zero_()
            ix.compressor.score_state.fill_(float("-inf"))
            ix.compressor.kv_cache = None

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
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = sum((x-mx)**2 for x in xs) or 1.0
    slope = num/den
    series = {str(p): float(rr[p]) for p in range(0, seq, 32)}
    series[str(seq-1)] = float(rr[seq-1])
    buckets = {}
    for start in range(0, seq, 64):
        end = min(start+64, seq)
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
        log(f"gpu={torch.cuda.get_device_name(0)}")

    model_dir = Path(args.model)
    cfgp = model_dir / "inference" / "config.json"
    if not cfgp.exists():
        cfgp = HERE / "config.json"
    margs = make_args(cfgp, max(args.seq, 256))
    if args.layers.strip() == "all":
        layers_keep = list(range(margs.n_layers))
        n_build = margs.n_layers
    else:
        layers_keep = [int(x) for x in args.layers.split(",")]
        n_build = max(layers_keep) + 1
    log(f"seq={args.seq} n_build={n_build} capture={args.layers}")

    t0 = time.time()
    model = build_model(margs, model_dir, n_build, device)
    log(f"build+load {time.time()-t0:.1f}s")

    tokens = load_tokens(args.tokens, args.seq).to(device)
    reset_caches(model)
    h = model.embed(tokens).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log(f"h0 global_L2={float(h.float().norm()):.6f}")

    summary = {
        "seq": args.seq,
        "device": str(device),
        "torch": torch.__version__,
        "n_layers_run": n_build,
        "h0_global_l2": float(h.float().norm()),
        "embed": summarize_rows(row_l2_hc(h), args.seq),
        "global_l2_curve": {"embed": float(h.float().norm())},
        "layers": {},
        "notes": [
            "HC residual per-row L2 after each layer",
            "late_over_early = mean(row_l2[-128:]) / mean(row_l2[:128])",
        ],
    }
    log(f"embed late/early={summary['embed']['late_over_early']:.4f}")

    want = set(layers_keep)
    for i in range(n_build):
        t1 = time.time()
        h = model.layers[i](h, 0, tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        g = float(h.float().norm())
        dt = time.time() - t1
        summary["global_l2_curve"][str(i)] = g
        ratio = int(margs.compress_ratios[i]) if i < len(margs.compress_ratios) else -1
        log(f"L{i:02d} ratio={ratio} {dt:.2f}s global_L2={g:.4f}")
        if i in want:
            rr = row_l2_hc(h)
            entry = summarize_rows(rr, args.seq)
            entry["layer"] = i
            entry["ratio"] = ratio
            entry["global_l2"] = g
            entry["wall_s"] = dt
            probes = [p for p in [0,64,128,200,256,400,448,512,600,768,800,1000,1023] if p < args.seq]
            entry["probes"] = {str(p): float(rr[p]) for p in probes}
            summary["layers"][str(i)] = entry
            log(f"     early={entry['early128_mean']:.4f} late={entry['late128_mean']:.4f} late/early={entry['late_over_early']:.4f} mid400-512={entry['mid400_512_mean']}")
            log("     probes " + " ".join(f"{p}:{entry['probes'][str(p)]:.2f}" for p in probes))

    keys = sorted(summary["layers"], key=int)
    growth = []
    for a, b in zip(keys, keys[1:]):
        ga = summary["layers"][a]["global_l2"]
        gb = summary["layers"][b]["global_l2"]
        ea = summary["layers"][a]["late_over_early"]
        eb = summary["layers"][b]["late_over_early"]
        growth.append({"from": int(a), "to": int(b), "global_ratio": gb/max(ga,1e-12),
                       "late_over_early_from": ea, "late_over_early_to": eb,
                       "late_over_early_delta": eb-ea})
    summary["growth_between_captured"] = growth
    if keys:
        last = keys[-1]
        summary["embed_to_last_global_growth"] = summary["layers"][last]["global_l2"] / max(summary["h0_global_l2"], 1e-12)
        summary["last_late_over_early"] = summary["layers"][last]["late_over_early"]
        log(f"\nembed->L{last} global x{summary['embed_to_last_global_growth']:.3f}")
        log(f"L{last} late/early={summary['last_late_over_early']:.4f}")

    compact = {k: v for k, v in summary.items() if k != "layers"}
    compact["layers"] = {}
    for Lk, e in summary["layers"].items():
        compact["layers"][Lk] = {kk: vv for kk, vv in e.items() if kk != "dense_row_l2"}
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
