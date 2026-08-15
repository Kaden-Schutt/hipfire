#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Compare apply_rotary_emb (model.py) vs parent apply_rope_interleaved math."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks
sys.modules["kernel"] = _ks
import torch
from model import precompute_freqs_cis, apply_rotary_emb

DIM_ROPE = 64
HEAD_DIM = 512
N_HEADS = 2

def parent_freqs(dim, original_seq_len, base, factor):
    n = dim // 2
    freqs = [1.0 / (base ** ((2 * i) / dim)) for i in range(n)]
    if original_seq_len > 0:
        def fcd(nr):
            return dim * math.log(original_seq_len / (nr * 2 * math.pi)) / (2 * math.log(base))
        low = math.floor(fcd(32.0))
        high = math.ceil(fcd(1.0))
        low = max(low, 0)
        high = min(high, dim - 1)
        if low == high:
            high += 0.001
        out = []
        for i, f in enumerate(freqs):
            ramp = max(0.0, min(1.0, (i - low) / (high - low)))
            smooth = 1.0 - ramp
            out.append(f / factor * (1.0 - smooth) + f * smooth)
        freqs = out
    return freqs

def parent_apply(x, positions, freqs, inverse=False):
    # x [rows, nh, hd] float64
    out = x.copy()
    rows, nh, hd = out.shape
    n_rot = len(freqs) * 2
    base = hd - n_rot
    for r in range(rows):
        pos = float(positions[r])
        for h in range(nh):
            for p, f in enumerate(freqs):
                ang = pos * f
                if inverse:
                    ang = -ang
                c = math.cos(ang)
                s = math.sin(ang)
                i0 = base + 2 * p
                i1 = i0 + 1
                a = out[r, h, i0]
                b = out[r, h, i1]
                out[r, h, i0] = a * c - b * s
                out[r, h, i1] = a * s + b * c
    return out

def compare(name, original_seq_len, base, factor, positions, head_dim=HEAD_DIM):
    torch.manual_seed(0)
    rows = len(positions)
    # contiguous positions required by apply_rotary_emb slicing convention;
    # so we build full table and index, then apply on a tensor whose seq dim
    # equals len(positions) with matching cis rows.
    maxp = max(positions) + 1
    freqs_cis_full = precompute_freqs_cis(
        DIM_ROPE, maxp, original_seq_len, base, factor, 32.0, 1.0
    )
    cis = freqs_cis_full[list(positions)]  # [rows, 32] complex

    x = torch.randn(1, rows, N_HEADS, head_dim, dtype=torch.float32)
    xref = x.clone()
    tail = xref[..., -DIM_ROPE:]
    apply_rotary_emb(tail, cis)

    freqs = parent_freqs(DIM_ROPE, original_seq_len, base, factor)
    xnp = x[0].double().numpy()
    xpar = parent_apply(xnp, positions, freqs, False)

    ref = xref[0].numpy()
    rt = ref[..., -DIM_ROPE:]
    pt = xpar[..., -DIM_ROPE:].astype(np.float32)
    ad = np.abs(rt - pt)
    rd = ad / np.maximum(np.abs(rt), 1e-6)
    nope_ad = float(np.max(np.abs(ref[..., :-DIM_ROPE] - x[0].numpy()[..., :-DIM_ROPE])))

    # inverse roundtrip
    inv = tail.clone()
    apply_rotary_emb(inv, cis, inverse=True)
    xpar_inv = parent_apply(xpar, positions, freqs, True)
    inv_ref = float(torch.max(torch.abs(inv - x[..., -DIM_ROPE:])).item())
    inv_par = float(np.max(np.abs(xpar_inv[..., -DIM_ROPE:] - xnp[..., -DIM_ROPE:])))

    per_pos = ad.reshape(rows, -1).max(axis=1)
    im = int(per_pos.argmax())
    print(f"=== {name} npos={rows} head_dim={head_dim}")
    print(f"  rope max_abs={ad.max():.6e} mean_abs={ad.mean():.6e} max_rel={rd.max():.6e}")
    print(f"  nope_max_abs={nope_ad:.6e} inv_ref={inv_ref:.6e} inv_par={inv_par:.6e}")
    print(f"  worst_pos={positions[im]} max_abs={per_pos[im]:.6e}")
    # also check growth of abs err with pos
    if rows >= 4:
        early = float(per_pos[: max(1, rows//8)].mean())
        late = float(per_pos[-max(1, rows//8):].mean())
        print(f"  err_early_mean={early:.6e} err_late_mean={late:.6e} late/early={late/max(early,1e-30):.3f}")
    return {
        "name": name,
        "max_abs": float(ad.max()),
        "mean_abs": float(ad.mean()),
        "max_rel": float(rd.max()),
        "nope_max_abs": nope_ad,
        "inv_ref": inv_ref,
        "inv_par": inv_par,
        "worst_pos": int(positions[im]),
        "worst_abs": float(per_pos[im]),
        "per_pos_max_abs": {str(int(positions[i])): float(per_pos[i]) for i in range(rows)},
    }

def main():
    results = {}
    probes = sorted(set(list(range(0, 1024, 64)) + [1, 2, 3, 127, 128, 255, 256, 511, 512, 999, 1000, 1023]))
    results["A_swa"] = compare("A_swa_theta10k", 0, 10000.0, 16.0, probes, HEAD_DIM)
    results["B_yarn"] = compare("B_yarn_theta160k", 65536, 160000.0, 16.0, probes, HEAD_DIM)
    comp_pos = list(range(0, 1024, 4))
    results["B_yarn_comp_pos"] = compare("B_yarn_comp_stride4", 65536, 160000.0, 16.0, comp_pos, HEAD_DIM)
    results["B_yarn_idx128"] = compare("B_yarn_indexer_hd128", 65536, 160000.0, 16.0, probes, 128)
    results["B_yarn_pos1000"] = compare("B_yarn_only1000", 65536, 160000.0, 16.0, [1000], HEAD_DIM)
    # contiguous full 0..1023
    results["B_yarn_full1024"] = compare("B_yarn_full0_1023", 65536, 160000.0, 16.0, list(range(1024)), HEAD_DIM)
    print("\nSUMMARY")
    for k, v in results.items():
        print(f"  {k}: max_abs={v['max_abs']:.3e} max_rel={v['max_rel']:.3e} worst_pos={v['worst_pos']}")
    Path("/tmp/apply_rope_diff.json").write_text(json.dumps(results, indent=2))
    print("wrote /tmp/apply_rope_diff.json")

if __name__ == "__main__":
    main()
