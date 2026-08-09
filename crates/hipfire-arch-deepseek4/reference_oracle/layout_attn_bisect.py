#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Reference layout + per-row attn trajectory (selection already ruled out).

Dumps at probes 64,200,400,600,800,1000:
  - offset / unified index ranges
  - ring write vs token_kv consistency
  - compress KV row L2 series
  - attn_o (pre-wo) per-row L2 series (every 32)
  - wo_out per-row L2 at probes
"""
from __future__ import annotations
import argparse, json, os, struct, sys, time
from pathlib import Path
from typing import Any, Dict, List
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks
sys.modules["kernel"] = _ks
import torch
import torch.nn.functional as F
from model import ModelArgs, Transformer, Attention, apply_rotary_emb, get_window_topk_idxs, get_compress_topk_idxs
import model as model_mod
from weight_loader import load_state_into_model
from kernel_shim import act_quant, sparse_attn

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin"
PROBES = [64, 200, 400, 600, 800, 1000]

def log(m=""): print(m, flush=True)

def load_tokens(path, n):
    raw = Path(path).read_bytes()
    ids = struct.unpack("<"+"i"*n, raw[:n*4])
    return torch.tensor(ids, dtype=torch.long).view(1, n)

def make_args(config_path, max_seq_len):
    cfg = json.loads(Path(config_path).read_text())
    fields = ModelArgs.__dataclass_fields__
    kwargs = {}
    for k,v in cfg.items():
        if k not in fields: continue
        kwargs[k] = tuple(v) if k in ("compress_ratios","dspark_target_layer_ids") else v
    args = ModelArgs(**kwargs)
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    return args

def build_model(args, model_dir, n_build, with_experts):
    full = args.n_layers
    args.n_layers = n_build
    args.n_mtp_layers = 0
    args.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        model = Transformer(args)
    args.n_layers = full
    load_state_into_model(model, model_dir, layers=range(n_build), load_experts=with_experts, device="cpu", verbose=True)
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

@torch.inference_mode()
def capture_attn(attn: Attention, x, probes):
    bsz, seqlen, _ = x.size()
    win = attn.window_size
    ratio = int(attn.compress_ratio)
    rd = attn.rope_head_dim
    freqs = attn.freqs_cis[:seqlen]
    indexer = getattr(attn, "indexer", None)
    if ratio and attn.compressor.kv_cache is None:
        attn.compressor.kv_cache = attn.kv_cache[:, win:]
        attn.compressor.freqs_cis = attn.freqs_cis
        if indexer is not None:
            indexer.freqs_cis = attn.freqs_cis

    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
    apply_rotary_emb(q[..., -rd:], freqs)

    kv = attn.kv_norm(attn.wkv(x))
    apply_rotary_emb(kv[..., -rd:], freqs)
    act_quant(kv[..., :-rd], 64, model_mod.scale_fmt, model_mod.scale_dtype, True)
    token_kv = kv.clone()

    topk = get_window_topk_idxs(win, bsz, seqlen, 0)
    offset = None
    if ratio:
        offset = int(kv.size(1))  # seqlen on prefill
        if indexer is not None:
            ctopk = indexer(x, qr, 0, offset).int()
        else:
            ctopk = get_compress_topk_idxs(ratio, bsz, seqlen, 0, offset)
        topk = torch.cat([topk, ctopk], dim=-1)

    # ring write (decode prime only)
    if seqlen <= win:
        attn.kv_cache[:bsz, :seqlen] = kv
    else:
        cutoff = seqlen % win
        attn.kv_cache[:bsz, cutoff:win], attn.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win-cutoff, cutoff], dim=1)
    ring = attn.kv_cache[:bsz, :win].clone()

    kv_attn = token_kv
    kv_comp = None
    if ratio:
        kv_comp = attn.compressor(x, 0)
        if kv_comp is not None:
            kv_attn = torch.cat([token_kv, kv_comp], dim=1)

    o = sparse_attn(q, kv_attn, attn.attn_sink, topk, attn.softmax_scale)
    apply_rotary_emb(o[..., -rd:], freqs, True)
    o_pre = o
    og = o.view(bsz, seqlen, attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o2 = torch.einsum("bsgd,grd->bsgr", og, wo_a)
    x_out = attn.wo_b(o2.flatten(2))

    of = o_pre.float()[0]  # [S,H,D]
    xf = x_out.float()[0]  # [S,D]
    row_o = of.reshape(seqlen, -1).norm(dim=-1)
    row_x = xf.norm(dim=-1)

    # ring recon check
    cutoff = seqlen % win
    last = token_kv[:, -win:]
    recon = torch.zeros_like(ring)
    recon[:, cutoff:win] = last[:, :win-cutoff]
    recon[:, :cutoff] = last[:, win-cutoff:]
    ring_err = float((ring.float() - recon.float()).abs().max())

    layout = {
        "seqlen": seqlen, "ratio": ratio, "offset": offset,
        "kv_attn_shape": list(kv_attn.shape),
        "n_comp": 0 if kv_comp is None else int(kv_comp.size(1)),
        "token_region": [0, seqlen-1],
        "compress_region": None if kv_comp is None else [seqlen, seqlen + kv_comp.size(1) - 1],
        "ring_matches_rotated_lastW_maxabs": ring_err,
        "q_l2": float(q.float().norm()),
        "token_kv_l2": float(token_kv.float().norm()),
        "compress_kv_l2": None if kv_comp is None else float(kv_comp.float().norm()),
        "attn_o_all_l2": float(of.norm()),
        "wo_all_l2": float(xf.norm()),
        "topk_shape": list(topk.shape),
    }
    if kv_comp is not None:
        ck = kv_comp.float()[0]  # [n_comp, D]
        layout["compress_row_l2_series"] = {str(i): float(ck[i].norm()) for i in range(0, ck.size(0), max(1, ck.size(0)//32))}
        layout["compress_row_l2_first"] = float(ck[0].norm())
        layout["compress_row_l2_last"] = float(ck[-1].norm())

    # series every 32
    series_o = {str(r): float(row_o[r]) for r in list(range(0, seqlen, 32)) + probes + [seqlen-1] if r < seqlen}
    series_x = {str(r): float(row_x[r]) for r in probes if r < seqlen}
    layout["attn_o_row_l2_series"] = dict(sorted(((int(k),v) for k,v in series_o.items())))
    # convert keys back to str for json
    layout["attn_o_row_l2_series"] = {str(k): v for k,v in sorted(((int(k),v) for k,v in series_o.items()))}

    # growth ratios between consecutive probes
    growth = []
    prev_r, prev_v = None, None
    for r in probes:
        if r >= seqlen: continue
        v = float(row_o[r])
        if prev_v is not None and prev_v > 0:
            growth.append({"from": prev_r, "to": r, "l2_ratio": v/prev_v, "l2": v})
        prev_r, prev_v = r, v
    layout["attn_o_probe_growth"] = growth

    # detect step: large ratio jump between adjacent probe pairs vs others
    if len(growth) >= 2:
        ratios = [g["l2_ratio"] for g in growth]
        layout["growth_ratios"] = ratios
        layout["growth_max_jump"] = max(ratios)
        layout["growth_argmax_pair"] = growth[int(max(range(len(ratios)), key=lambda i: ratios[i]))]

    probes_out = {}
    ref = of[probes[0]].reshape(-1)
    for r in probes:
        if r >= seqlen: continue
        ov = of[r].reshape(-1)
        xv = xf[r]
        ti = [int(v) for v in topk[0,r].tolist() if v >= 0]
        n_tok = sum(1 for v in ti if v < seqlen)
        entry = {
            "attn_o_l2": float(ov.norm()),
            "attn_o_mean": float(ov.mean()),
            "attn_o_absmax": float(ov.abs().max()),
            "wo_l2": float(xv.norm()),
            "cos_to_first_probe": float(F.cosine_similarity(ov.unsqueeze(0), ref.unsqueeze(0)).item()),
            "joint_valid": len(ti),
            "n_idx_token": n_tok,
            "n_idx_comp": len(ti)-n_tok,
            "head_l2_mean": float(of[r].norm(dim=-1).mean()),
        }
        probes_out[str(r)] = entry
    return layout, probes_out

@torch.inference_mode()
def run(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try: torch.set_default_device("cpu")
    except Exception: pass
    model_dir = Path(args.model)
    cfgp = model_dir/"inference"/"config.json"
    if not cfgp.exists(): cfgp = HERE/"config.json"
    margs = make_args(cfgp, max(args.seq,256))
    layers = [int(x) for x in args.layers.split(",")]
    probes = [int(x) for x in args.probes.split(",") if int(x) < args.seq]
    log(f"seq={args.seq} layers={layers} probes={probes}")
    log(f"window={margs.window_size} index_topk={margs.index_topk} compress_rope_theta={margs.compress_rope_theta}")

    t0=time.time()
    model = build_model(margs, model_dir, max(layers)+1, args.with_experts)
    log(f"load {time.time()-t0:.1f}s")
    tokens = load_tokens(args.tokens, args.seq)
    h0 = model.embed(tokens).unsqueeze(2).repeat(1,1,model.hc_mult,1)
    log(f"h0 L2={float(h0.float().norm()):.4f}")

    summary = {"seq": args.seq, "probes": probes, "layers": {},
               "conventions_doc": "INDEX_SPACE_CONVENTIONS.md"}
    for L in layers:
        log(f"\n######## L{L} ratio={margs.compress_ratios[L]} ########")
        reset_caches(model)
        hh = h0.clone()
        for i in range(L):
            t1=time.time(); hh = model.layers[i](hh, 0, tokens)
            log(f"  L{i} {time.time()-t1:.1f}s res={float(hh.float().norm()):.3f}")
        block = model.layers[L]
        y, post, comb = block.hc_pre(hh, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
        x = block.attn_norm(y)
        lay, pr = capture_attn(block.attn, x, probes)
        log(f"  offset={lay['offset']} kv_attn={lay['kv_attn_shape']} n_comp={lay['n_comp']} ring_err={lay['ring_matches_rotated_lastW_maxabs']}")
        log(f"  attn_o_all={lay['attn_o_all_l2']:.4f} wo_all={lay['wo_all_l2']:.4f} comp_kv={lay['compress_kv_l2']}")
        for r in probes:
            e = pr.get(str(r))
            if e: log(f"  r={r:4d} o_l2={e['attn_o_l2']:.4f} wo={e['wo_l2']:.4f} cos0={e['cos_to_first_probe']:.4f} joint={e['joint_valid']} tok/c={e['n_idx_token']}/{e['n_idx_comp']}")
        log("  growth:" + str(lay.get("attn_o_probe_growth")))
        # print series at 64-aligned
        ser = lay["attn_o_row_l2_series"]
        log("  series:")
        for r in sorted(int(k) for k in ser):
            if r % 64 == 0 or r in probes or r == args.seq-1:
                log(f"    r={r:4d} {ser[str(r)]:.6f}")
        summary["layers"][str(L)] = {"ratio": int(margs.compress_ratios[L]), "layout": lay, "probes": pr}

    Path(args.out).write_text(json.dumps(summary, indent=2))
    log(f"wrote {args.out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--layers", default="0,2")
    ap.add_argument("--probes", default=",".join(map(str,PROBES)))
    ap.add_argument("--out", default="/tmp/layout_attn_bisect.json")
    ap.add_argument("--with-experts", action="store_true")
    args = ap.parse_args(); run(args)
if __name__ == "__main__":
    main()
