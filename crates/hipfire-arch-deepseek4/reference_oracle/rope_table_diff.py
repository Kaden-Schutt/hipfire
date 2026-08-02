#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Element-wise f64 RoPE frequency table diff: model.py vs parent Rust formulas.

Three configs:
  A) ratio-0 SWA:      base=10000, original_seq_len=0  (YaRN OFF)
  B) ratio>0 main:     base=160000, original_seq_len=65536, factor=16, beta_fast=32, beta_slow=1 (YaRN ON)
  C) indexer plain:    base=160000, original_seq_len=0  (YaRN OFF)

Reports per-dim relative |p-r|/|r|, max, argmax, and phase error at pos=1000 = pos*freq*rel.
Also cross-checks parent yarn helpers against model.py find_correction_*.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks
sys.modules["kernel"] = _ks

import torch
from model import precompute_freqs_cis

DIM = 64           # rope_head_dim
N_FREQ = DIM // 2  # 32 complex pairs / frequency entries
POS = 1000

def log(m=""):
    print(m, flush=True)

# ── Parent Rust formulas transcribed (attention.rs:511-575) ──────────────

def yarn_correction_dim(num_rotations, dim, base, max_seq_len):
    # model.py: dim * log(max_seq_len / (num_rotations * 2 * pi)) / (2 * log(base))
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

def yarn_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    low = math.floor(yarn_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(yarn_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0.0), min(high, float(dim - 1))

def yarn_linear_ramp(min_v, max_v, x):
    if min_v == max_v:
        max_v = max_v + 0.001
    t = (x - min_v) / (max_v - min_v)
    return max(0.0, min(1.0, t))

def parent_precompute_freqs(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    """Angles-only table length dim//2 — attention.rs precompute_rope_freqs."""
    n = dim // 2
    freqs = []
    for i in range(n):
        # freqs = 1 / base**(arange(0,dim,2)/dim)  → i maps to 2i
        f = 1.0 / (base ** ((2 * i) / dim))
        freqs.append(f)
    if original_seq_len > 0:
        low, high = yarn_correction_range(beta_fast, beta_slow, dim, base, float(original_seq_len))
        out = []
        for i, f in enumerate(freqs):
            ramp = yarn_linear_ramp(low, high, float(i))
            smooth = 1.0 - ramp
            # freqs = freqs/factor*(1-smooth) + freqs*smooth
            out.append(f / factor * (1.0 - smooth) + f * smooth)
        freqs = out
    return freqs

def ref_freqs_base(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    """Extract base frequency vector (pos=1 angles) from model.py precompute_freqs_cis."""
    # precompute returns complex cis of shape [seqlen, dim//2]
    # angle at pos p, dim i = p * freq[i]; so freq[i] = angle(cis[1,i]) for p>=1
    cis = precompute_freqs_cis(dim, 2, original_seq_len, base, factor, beta_fast, beta_slow)
    # cis is complex64/complex128
    c = cis[1].to(torch.complex128)
    # angle = atan2(imag, real); also == freq since polar(1, freq)
    ang = torch.atan2(c.imag, c.real).tolist()
    # also compute from real formula mirroring model.py exactly in f64
    return ang

def ref_freqs_f64(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    """Pure f64 reimplementation of model.py precompute_freqs_cis base freqs."""
    # match torch.arange(0, dim, 2, dtype=float32) then promote — but use f64
    # Reference uses float32 arange then ops; for oracle we also report f32 path
    freqs_f32 = (1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))).tolist()
    freqs = [float(f) for f in freqs_f32]
    if original_seq_len > 0:
        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        if low == high:
            high = high + 0.001
        ramp = []
        for i in range(dim // 2):
            t = (i - low) / (high - low)
            ramp.append(max(0.0, min(1.0, t)))
        smooth = [1.0 - r for r in ramp]
        freqs = [f / factor * (1 - s) + f * s for f, s in zip(freqs, smooth)]
    return freqs, (low, high) if original_seq_len > 0 else None

def compare(name, original_seq_len, base, factor, beta_fast, beta_slow):
    log(f"\n======== {name} ========")
    log(f"  base={base} original_seq_len={original_seq_len} factor={factor} beta_fast={beta_fast} beta_slow={beta_slow}")
    ref, corr = ref_freqs_f64(DIM, original_seq_len, base, factor, beta_fast, beta_slow)
    par = parent_precompute_freqs(DIM, original_seq_len, base, factor, beta_fast, beta_slow)
    # also from actual model.py tensor
    ref_from_cis = ref_freqs_base(DIM, original_seq_len, base, factor, beta_fast, beta_slow)
    if corr is not None:
        log(f"  yarn correction range (low,high)={corr}")
        pl, ph = yarn_correction_range(beta_fast, beta_slow, DIM, base, float(original_seq_len))
        log(f"  parent yarn range=({pl},{ph}) match={abs(pl-corr[0])<1e-12 and abs(ph-corr[1])<1e-12}")

    assert len(ref) == len(par) == N_FREQ
    rels = []
    absdiffs = []
    phase1000 = []
    for i in range(N_FREQ):
        r, p = ref[i], par[i]
        ad = abs(p - r)
        rd = ad / max(abs(r), 1e-30)
        # phase error at pos: pos * |p-r| radians
        pe = POS * ad
        rels.append(rd); absdiffs.append(ad); phase1000.append(pe)

    # also parent vs cis
    rels_cis = []
    for i in range(N_FREQ):
        r, p = ref_from_cis[i], par[i]
        # cis angle may wrap; for small freqs at pos=1 angle≈freq
        ad = abs(p - r)
        # handle possible 2pi wrap (shouldn't for pos=1 small freqs)
        rels_cis.append(ad / max(abs(r), 1e-30))

    imax = max(range(N_FREQ), key=lambda i: rels[i])
    imax_pe = max(range(N_FREQ), key=lambda i: phase1000[i])
    log(f"  max_rel_err={rels[imax]:.6e} at dim_pair={imax} ref={ref[imax]:.12e} par={par[imax]:.12e}")
    log(f"  max_abs_err={max(absdiffs):.6e}")
    log(f"  max_phase_err_at_pos{POS}={phase1000[imax_pe]:.6e} rad ({phase1000[imax_pe]*180/math.pi:.4f} deg) at dim_pair={imax_pe}")
    log(f"  mean_rel_err={sum(rels)/len(rels):.6e}")
    log(f"  max_rel_vs_model_cis={max(rels_cis):.6e}")
    # print all dims with rel > 1e-12
    hot = [(i, rels[i], phase1000[i], ref[i], par[i]) for i in range(N_FREQ) if rels[i] > 1e-12]
    if not hot:
        log("  ALL dims match to rel <= 1e-12 (bit-level for practical purposes)")
    else:
        log(f"  dims with rel>1e-12: {len(hot)}")
        for i, rd, pe, r, p in hot[:16]:
            log(f"    i={i:2d} rel={rd:.3e} phase@{POS}={pe:.3e} rad ref={r:.10e} par={p:.10e}")

    return {
        "name": name,
        "base": base,
        "original_seq_len": original_seq_len,
        "factor": factor,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
        "yarn_range": corr,
        "max_rel_err": rels[imax],
        "argmax_rel_dim": imax,
        "max_abs_err": max(absdiffs),
        "max_phase_err_rad_at_pos": {str(POS): phase1000[imax_pe]},
        "argmax_phase_dim": imax_pe,
        "mean_rel_err": sum(rels)/len(rels),
        "max_rel_vs_model_cis": max(rels_cis),
        "n_hot_rel_gt_1e12": len(hot),
        "per_dim": [
            {
                "i": i,
                "ref": ref[i],
                "parent": par[i],
                "rel": rels[i],
                "abs": absdiffs[i],
                "phase_err_pos1000": phase1000[i],
            }
            for i in range(N_FREQ)
        ],
        "verdict": (
            "MATCH" if rels[imax] < 1e-12 else
            ("NEAR" if rels[imax] < 1e-6 else "MISMATCH")
        ),
    }

def main():
    log("RoPE table f64 diff: model.py vs parent Rust formulas")
    log(f"rope_head_dim={DIM} n_freq={N_FREQ} phase_probe_pos={POS}")

    # Confirm three configs from model.py Attention.__init__ 482-487
    log("\n=== CONFIG BINDING (model.py Attention.__init__) ===")
    log("ratio==0: original_seq_len=0, rope_theta=10000  (YaRN disabled)")
    log("ratio>0:  original_seq_len=65536, compress_rope_theta=160000, factor=16, beta_fast=32, beta_slow=1")
    log("indexer:  uses attn.freqs_cis for Q (same as main layer) — BUT indexer compressor")
    log("          rotate path: plain compress_rope_theta via its own freqs? Check Indexer.")
    log("  Indexer.forward: apply_rotary_emb(q, self.freqs_cis[start:start+seq])")
    log("  Attention sets indexer.freqs_cis = self.freqs_cis (ratio>0 YaRN table)")
    log("  Indexer compressor: Compressor(..., rotate=True) uses same freqs_cis alias")
    log("  → indexer Q RoPE is YaRN table when attached to ratio-4 layer, NOT plain!")
    log("  Parent comment claims indexer plain compress_rope_theta — VERIFY against this.")

    results = {}
    results["A_ratio0_swa"] = compare(
        "A_ratio0_swa_plain_theta10k", original_seq_len=0, base=10000.0,
        factor=16.0, beta_fast=32.0, beta_slow=1.0,
    )
    results["B_ratio_gt0_yarn"] = compare(
        "B_ratio_gt0_yarn_theta160k", original_seq_len=65536, base=160000.0,
        factor=16.0, beta_fast=32.0, beta_slow=1.0,
    )
    results["C_indexer_plain_claimed"] = compare(
        "C_indexer_plain_theta160k_NO_yarn", original_seq_len=0, base=160000.0,
        factor=16.0, beta_fast=32.0, beta_slow=1.0,
    )
    # Also: what reference ACTUALLY gives indexer (alias of B)
    results["C_indexer_actual_ref_is_B"] = {
        "note": "model.py sets indexer.freqs_cis = attn.freqs_cis (YaRN B table). "
                "Parent claiming plain-no-YaRN for indexer Q is a DEFECT if true.",
        "ref_table": "B_ratio_gt0_yarn",
        "parent_claimed": "C_indexer_plain",
    }

    # Diff B vs C to show YaRN effect magnitude
    b = results["B_ratio_gt0_yarn"]["per_dim"]
    c = results["C_indexer_plain_claimed"]["per_dim"]
    bc_rel = [abs(b[i]["ref"]-c[i]["ref"])/max(abs(b[i]["ref"]),1e-30) for i in range(N_FREQ)]
    im = max(range(N_FREQ), key=lambda i: bc_rel[i])
    log(f"\n=== YaRN effect B vs plain-C (same base 160k) ===")
    log(f"  max_rel_diff_yarn_vs_plain={bc_rel[im]:.6e} at i={im}")
    log(f"  phase_err_if_parent_uses_plain_at_pos{POS} = {POS*abs(b[im]['ref']-c[im]['ref']):.6e} rad")
    for i in range(N_FREQ):
        if bc_rel[i] > 1e-6:
            pe = POS * abs(b[i]["ref"] - c[i]["ref"])
            log(f"  i={i:2d} yarn={b[i]['ref']:.8e} plain={c[i]['ref']:.8e} rel={bc_rel[i]:.3e} phase@{POS}={pe:.3e} rad")

    results["yarn_vs_plain_theta160k"] = {
        "max_rel": bc_rel[im],
        "argmax": im,
        "phase_err_if_swapped_at_1000": POS * abs(b[im]["ref"] - c[im]["ref"]),
        "per_dim_rel": bc_rel,
    }

    # Summary verdict
    log("\n=== SUMMARY ===")
    for k in ("A_ratio0_swa", "B_ratio_gt0_yarn", "C_indexer_plain_claimed"):
        r = results[k]
        log(f"  {k}: {r['verdict']} max_rel={r['max_rel_err']:.3e} max_phase@1000={list(r['max_phase_err_rad_at_pos'].values())[0]:.3e} rad")

    out = Path("/tmp/rope_table_diff.json")
    # strip bulky per_dim from A if match for readability — keep all
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")

if __name__ == "__main__":
    main()
