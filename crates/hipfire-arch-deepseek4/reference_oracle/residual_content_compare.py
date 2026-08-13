#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Compare parent vs reference residual CONTENT by cosine + rel L2 + norm ratio.

Inputs:
  --ref-dir   artifacts from residual_content_dump.py  (NPZ + JSON)
  --parent-dir artifacts from ds4_parent_residual_content (raw f32 + JSON)

Reports per (layer, position):
  cosine, relative L2 error, norm_parent/norm_ref

Floor: layer=-1 embed, known bit-identical earlier. Domain named in output.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

HC_MULT = 4
DIM = 4096
HC_DIM = HC_MULT * DIM


def log(m=""):
    print(m, flush=True)


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rel_l2(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b)
    if den == 0:
        return float(num)
    return float(num / den)


def norm_ratio(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    nb = np.linalg.norm(b)
    na = np.linalg.norm(a)
    if nb == 0:
        return float("nan") if na == 0 else float("inf")
    return float(na / nb)


def load_ref(ref_dir: Path):
    meta = json.loads((ref_dir / "residual_content_ref.json").read_text())
    npz = np.load(ref_dir / "residual_content_ref.npz")
    return meta, npz


def load_parent_layer(parent_dir: Path, name: str, n_pos: int):
    path = parent_dir / f"{name}.f32"
    raw = path.read_bytes()
    n_f = len(raw) // 4
    arr = np.array(struct.unpack("<" + "f" * n_f, raw), dtype=np.float32)
    expect = n_pos * HC_DIM
    if arr.size != expect:
        raise RuntimeError(f"{path}: got {arr.size} f32, expect {expect}")
    return arr.reshape(n_pos, HC_MULT, DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--parent-dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir)
    parent_dir = Path(args.parent_dir)
    meta_ref, npz = load_ref(ref_dir)
    meta_par = json.loads((parent_dir / "residual_content_parent.json").read_text())

    positions = meta_ref["positions"]
    assert positions == meta_par["positions"], (positions, meta_par["positions"])
    n_pos = len(positions)

    layers = sorted(set([-1] + list(meta_ref["layers"])))
    rows = []
    log(f"positions={positions}")
    log(f"layers={layers}")
    log(f"ref domain notes: {meta_ref.get('notes')}")
    log(f"parent domain notes: {meta_par.get('notes')}")
    log("")
    hdr = f"{'layer':>6} {'pos':>6} {'cosine':>12} {'rel_l2':>12} {'norm_ratio':>12} {'n_ref':>12} {'n_par':>12}"
    log(hdr)

    for layer in layers:
        if layer == -1:
            ref_key = "layer_-1_embed"
            par_name = "layer_-1_embed"
        else:
            ref_key = f"layer_{layer}"
            par_name = f"layer_{layer}"
        if ref_key not in npz:
            log(f"skip L{layer}: missing in ref npz")
            continue
        ref = npz[ref_key]
        try:
            par = load_parent_layer(parent_dir, par_name, n_pos)
        except Exception as e:
            log(f"skip L{layer}: parent load failed: {e}")
            continue
        if ref.shape != par.shape:
            log(f"skip L{layer}: shape ref {ref.shape} par {par.shape}")
            continue

        for i, pos in enumerate(positions):
            r = ref[i]
            p = par[i]
            c = cosine(p, r)
            rl = rel_l2(p, r)
            nr = norm_ratio(p, r)
            n_r = float(np.linalg.norm(r.astype(np.float64)))
            n_p = float(np.linalg.norm(p.astype(np.float64)))
            rows.append({
                "layer": layer,
                "pos": pos,
                "cosine": c,
                "rel_l2": rl,
                "norm_ratio": nr,
                "norm_ref": n_r,
                "norm_parent": n_p,
            })
            log(f"{layer:6d} {pos:6d} {c:12.8f} {rl:12.6e} {nr:12.8f} {n_r:12.4f} {n_p:12.4f}")

        sub = [x for x in rows if x["layer"] == layer and x["pos"] != 0]
        if sub:
            mean_c = sum(x["cosine"] for x in sub) / len(sub)
            min_c = min(x["cosine"] for x in sub)
            mean_rl = sum(x["rel_l2"] for x in sub) / len(sub)
            log(f"  L{layer} excl-pos0: mean_cos={mean_c:.8f} min_cos={min_c:.8f} mean_rel_l2={mean_rl:.6e}")

    floor = [x for x in rows if x["layer"] == -1]
    out = {
        "positions": positions,
        "layers": layers,
        "floor_embed": floor,
        "rows": rows,
        "domain": {
            "ref": "bf16 block forward, dump cast to f32; layout [n_pos,4,4096]",
            "parent": "F32 residual internal; dump f32 LE; layout [n_pos,4,4096]",
            "compare": "cosine + rel_l2 + norm_ratio on flattened hc*dim vectors per position",
        },
    }
    if floor:
        mean_c = sum(x["cosine"] for x in floor) / len(floor)
        max_rl = max(x["rel_l2"] for x in floor)
        out["floor_summary"] = {
            "mean_cosine": mean_c,
            "max_rel_l2": max_rl,
            "n": len(floor),
            "statement": (
                f"Embed floor (layer=-1): mean_cosine={mean_c:.10f} max_rel_l2={max_rl:.6e} "
                f"over {len(floor)} positions. Domain: parent F32 residual vs ref bf16-embed"
                f"-widened-to-f32. Earlier probe claimed n_diff=0 bit-identical."
            ),
        }
        log("")
        log(out["floor_summary"]["statement"])

    non_floor = [x for x in rows if x["layer"] != -1]
    if non_floor and floor:
        floor_c = min(x["cosine"] for x in floor)
        floor_rl = max(x["rel_l2"] for x in floor)
        thresh_c = min(floor_c - 1e-3, 0.999)
        first = None
        for x in non_floor:
            if x["cosine"] < thresh_c or (floor_rl > 0 and x["rel_l2"] > max(1e-3, 100 * floor_rl)):
                first = x
                break
        if first is None:
            min_c = min(x["cosine"] for x in non_floor)
            verdict = (
                f"COSINE HOLDS near 1 across dumped layers/positions "
                f"(min_cos={min_c:.8f}, floor_min={floor_c:.8f}). "
                f"Residual directions agree; defect is downstream of last residual "
                f"(final norm / hc_head / head)."
            )
            falls = False
        else:
            verdict = (
                f"COSINE FALLS first at layer={first['layer']} pos={first['pos']} "
                f"cosine={first['cosine']:.8f} rel_l2={first['rel_l2']:.6e} "
                f"norm_ratio={first['norm_ratio']:.6f} "
                f"(floor_min_cos={floor_c:.8f})."
            )
            falls = True
        out["verdict"] = verdict
        out["cosine_falls"] = falls
        out["first_departure"] = first
        log("")
        log("VERDICT: " + verdict)

    log("")
    log("=== cosine-by-position tables ===")
    for layer in layers:
        sub = [x for x in rows if x["layer"] == layer]
        if not sub:
            continue
        log(f"--- L{layer} ---")
        log(f"{'pos':>6} {'cosine':>12} {'rel_l2':>12} {'norm_ratio':>12}")
        for x in sub:
            log(f"{x['pos']:6d} {x['cosine']:12.8f} {x['rel_l2']:12.6e} {x['norm_ratio']:12.8f}")

    out_path = Path(args.out) if args.out else parent_dir / "residual_content_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
