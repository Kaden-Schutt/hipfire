#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# GPTQ cross-reference harness.
#
# Stages:
#   0. Floor calibration (paper ref vs itself under two alg. equivalent paths)
#   1. Hessian accumulation (paper vs hipfire HFHS / E8H1 conventions)
#   2. Damping + Cholesky + inverse-Cholesky upper
#   3. Full GPTQ pass (quantized weights + reconstruction error)
#   4. Fault injection (transpose H, drop damp, no invperm, no error feedback)
#   5. Format round-trip (HFHS + E8H1)
#
# CPU only. See README.md for venv recipe.

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Local imports (run from this directory or with PYTHONPATH=.).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from formats import (  # noqa: E402
    E8H1_MAGIC,
    HFHS_MAGIC,
    accumulate_block_diagonal_xxT,
    hessian_key,
    read_hblk,
    read_hfhs,
    write_hblk,
    write_hfhs,
)
from gptq_paper_ref import (  # noqa: E402
    GptqConfig,
    accumulate_hessian_official_running,
    accumulate_hessian_xxT,
    damp_and_inv_cholesky_upper,
    frozen_minmax_int4_quantize_fn,
    gptq_fasterquant,
    inverse_perm,
    metrics,
    weight_mode_actorder_perm,
)
from hipfire_shadow import (  # noqa: E402
    hipfire_damped_inv_cholesky_upper,
    hipfire_e8h1_accumulate,
    hipfire_gptq_column_sequential,
    hipfire_hfhs_accumulate,
)

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

SEED = 0xC0FFEE
torch.manual_seed(SEED)
np.random.seed(SEED % (2**32 - 1))


def fmt_m(m: dict[str, float]) -> str:
    return (
        f"max_abs={m['max_abs']:.6e}  rel_frob={m['rel_frob']:.6e}  "
        f"norm_ratio={m['norm_ratio']:.6e}  cosine={m['cosine']:.10f}"
    )


def verdict(m: dict[str, float], floor_max_abs: float, floor_rel: float, tag: str) -> str:
    """PASS only if within a small multiple of the established floor.

    err==0 is reported as INCONCLUSIVE floor-wise if the floor itself is 0 and
    we have not established a non-trivial noise floor — but for f64 Cholesky on
    tiny well-conditioned problems the floor is ~1e-15 and zero is fine.
    """
    # Scaling mismatch: cosine ~1, norm_ratio away from 1.
    if m["cosine"] > 0.999 and abs(m["norm_ratio"] - 1.0) > 0.05:
        return f"FAIL_SCALE ({tag})"
    # Transpose / permutation: cosine near 0, norms similar.
    if abs(m["cosine"]) < 0.5 and abs(m["norm_ratio"] - 1.0) < 0.25:
        return f"FAIL_PERM_OR_TRANSPOSE ({tag})"
    if m["max_abs"] <= max(50.0 * floor_max_abs, 1e-9) and m["rel_frob"] <= max(
        50.0 * floor_rel, 1e-9
    ):
        return f"PASS ({tag})"
    return f"FAIL ({tag})"


# ---------------------------------------------------------------------------
# Stage 0 — floor
# ---------------------------------------------------------------------------

def stage0_floor() -> dict:
    print("\n=== Stage 0: floor calibration (paper ref vs algebraically equivalent path) ===")
    print("Arithmetic domain: float64 throughout. Device: CPU.")
    m, k, n = 8, 32, 64
    w = torch.randn(m, k, dtype=torch.float64)
    x = torch.randn(n, k, dtype=torch.float64)

    # Two algebraically equivalent Hessian paths (paper factor-2 normalized).
    h_a = accumulate_hessian_xxT(x, normalize=True, factor_two=True)
    h_b = accumulate_hessian_official_running([x[: n // 2], x[n // 2 :]])
    mh = metrics(h_a, h_b)
    print(f"  Hessian path-A vs path-B (both paper 2/N XX^T): {fmt_m(mh)}")

    # Two Cholesky paths: damp_and_inv_cholesky_upper vs manual torch
    u_a, damp_a, _ = damp_and_inv_cholesky_upper(h_a, percdamp=0.01)
    # Manual replay
    hd = h_a.clone()
    dmean = float(torch.diagonal(hd).mean())
    damp_b = 0.01 * dmean
    idx = torch.arange(k)
    hd[idx, idx] += damp_b
    l = torch.linalg.cholesky(hd)
    hinv = torch.cholesky_inverse(l)
    u_b = torch.linalg.cholesky(hinv, upper=True)
    mu = metrics(u_a, u_b)
    print(f"  inv-Cholesky U helper vs manual:              {fmt_m(mu)}")
    print(f"  damp_a={damp_a:.6e} damp_b={damp_b:.6e}")

    # Full GPTQ twice with identical inputs → bit-identical expected.
    qfn = frozen_minmax_int4_quantize_fn(w, group_size=0)
    cfg = GptqConfig(percdamp=0.01, blocksize=128, actorder=True)
    r1 = gptq_fasterquant(w, h_a, quantize_fn=qfn, cfg=cfg)
    r2 = gptq_fasterquant(w, h_a, quantize_fn=qfn, cfg=cfg)
    mq = metrics(r1.qweight, r2.qweight)
    print(f"  full GPTQ self-repeat:                       {fmt_m(mq)}")

    # Non-trivial floor: compare blocksize=8 vs blocksize=128 (alg. equivalent
    # for the OBS math; only blocking of the trailing update differs numerically).
    r_bs = gptq_fasterquant(
        w, h_a, quantize_fn=qfn, cfg=GptqConfig(percdamp=0.01, blocksize=8, actorder=True)
    )
    m_bs = metrics(r1.qweight, r_bs.qweight)
    print(f"  full GPTQ blocksize=128 vs 8:                {fmt_m(m_bs)}")

    floor = {
        "domain": "float64_cpu",
        "hessian_max_abs": mh["max_abs"],
        "hessian_rel_frob": mh["rel_frob"],
        "u_max_abs": mu["max_abs"],
        "u_rel_frob": mu["rel_frob"],
        "gptq_self_max_abs": mq["max_abs"],
        "gptq_blocksize_max_abs": m_bs["max_abs"],
        "gptq_blocksize_rel_frob": m_bs["rel_frob"],
        # Conservative floors used by later verdicts:
        "floor_max_abs": max(m_bs["max_abs"], mu["max_abs"], 1e-14),
        "floor_rel_frob": max(m_bs["rel_frob"], mu["rel_frob"], 1e-14),
    }
    print(
        f"  STATED FLOOR: max_abs <= {floor['floor_max_abs']:.6e}, "
        f"rel_frob <= {floor['floor_rel_frob']:.6e}  [{floor['domain']}]"
    )
    if mq["max_abs"] != 0.0:
        print("  NOTE: self-repeat max_abs != 0 — non-determinism in ref; investigate.")
    else:
        print("  self-repeat is exact zero (expected for pure torch f64 on CPU).")
    return floor


# ---------------------------------------------------------------------------
# Stage 1 — Hessian
# ---------------------------------------------------------------------------

def stage1_hessian(floor: dict) -> dict:
    print("\n=== Stage 1: Hessian accumulation conventions ===")
    n, k = 48, 32
    x = torch.randn(n, k, dtype=torch.float64)

    h_paper = accumulate_hessian_xxT(x, normalize=True, factor_two=True)
    h_hfhs = hipfire_hfhs_accumulate(x)  # (1/N) X^T X
    h_raw = accumulate_hessian_xxT(x, normalize=False, factor_two=False)

    # paper = 2 * hfhs  (both normalized)
    m_scale = metrics(h_paper, 2.0 * h_hfhs)
    print(f"  paper (2/N XX^T) vs 2 * hipfire_hfhs (1/N): {fmt_m(m_scale)}  "
          f"{verdict(m_scale, floor['floor_max_abs'], floor['floor_rel_frob'], 'scale-bridge')}")

    m_direct = metrics(h_paper, h_hfhs)
    print(f"  paper vs hipfire_hfhs DIRECT (expect scale 2): {fmt_m(m_direct)}")
    if m_direct["cosine"] > 0.999 and abs(m_direct["norm_ratio"] - 2.0) < 0.05:
        print("  FINDING: hipfire HFHS omits paper factor 2; cosine≈1, norm_ratio≈2.")
        print("           OBS ratios Hinv[j,k]/Hinv[j,j] are scale-invariant, so GPTQ")
        print("           weights match iff damping is also proportional to mean(diag).")
    elif m_direct["max_abs"] <= 50 * floor["floor_max_abs"]:
        print("  unexpected: direct match without scale bridge")
    else:
        print("  FINDING: paper vs HFHS differ by more than pure scale — investigate.")

    # E8H1: raw sum, block diagonal. Compare full-K raw diag-blocks.
    x256 = torch.randn(24, 512, dtype=torch.float64)
    blocks = hipfire_e8h1_accumulate(x256)
    h_full_raw = accumulate_hessian_xxT(x256, normalize=False, factor_two=False)
    # Extract block 0 from full raw and compare.
    b0_from_full = h_full_raw[:256, :256]
    m_b0 = metrics(blocks[0], b0_from_full)
    print(f"  E8H1 block0 vs full-raw[:256,:256]:         {fmt_m(m_b0)}  "
          f"{verdict(m_b0, floor['floor_max_abs'], floor['floor_rel_frob'], 'e8-block')}")
    # Off-block must be nonzero in full raw — E8H1 drops cross-block terms.
    cross = h_full_raw[:256, 256:512]
    cross_norm = float(torch.linalg.vector_norm(cross))
    print(f"  full-raw cross-block Frobenius (E8H1 drops this): {cross_norm:.6e}")
    print("  FINDING: E8H1 is block-diagonal-256 approximation of full XX^T;")
    print("           cross-block Hessian mass is discarded by construction.")

    return {
        "paper_vs_2hfhs": m_scale,
        "paper_vs_hfhs_direct": m_direct,
        "e8_block0": m_b0,
        "e8_cross_block_frob": cross_norm,
        "h_raw_norm": float(torch.linalg.vector_norm(h_raw)),
    }


# ---------------------------------------------------------------------------
# Stage 2 — damp + Cholesky + inv-Cholesky
# ---------------------------------------------------------------------------

def stage2_cholesky(floor: dict) -> dict:
    print("\n=== Stage 2: damping + Cholesky + inverse-Cholesky upper ===")
    k = 24
    x = torch.randn(80, k, dtype=torch.float64)
    # Use paper Hessian; damp fraction 0.01 both sides.
    h = accumulate_hessian_xxT(x, normalize=True, factor_two=True)
    # Make SPD well-conditioned.
    h = h + 0.1 * torch.eye(k, dtype=torch.float64)

    u_paper, damp_p, hd = damp_and_inv_cholesky_upper(h, percdamp=0.01)
    u_hip, damp_h = hipfire_damped_inv_cholesky_upper(
        h, initial_damp=0.01, max_damp_multiplier=1.0, damp_is_absolute=False
    )
    mu = metrics(u_paper, u_hip)
    print(f"  U paper vs hipfire_shadow (frac damp): {fmt_m(mu)}")
    print(f"  damp paper={damp_p:.6e}  hipfire_shadow={damp_h:.6e}")
    print(f"  {verdict(mu, floor['floor_max_abs'], floor['floor_rel_frob'], 'U-frac-damp')}")

    # Invariant check: U^T U ≈ H_damped^{-1}
    hinv = torch.linalg.inv(hd)
    utu = u_paper.transpose(0, 1) @ u_paper
    minv = metrics(utu, hinv)
    print(f"  invariant U^T U vs H_damped^{{-1}}:     {fmt_m(minv)}  "
          f"{verdict(minv, floor['floor_max_abs'], floor['floor_rel_frob'], 'invariant')}")

    # Wrong invariant U U^T (the 2026-05-14 hipfire bug class)
    uut = u_paper @ u_paper.transpose(0, 1)
    mwrong = metrics(uut, hinv)
    print(f"  WRONG invariant U U^T vs H_inv (bug class): {fmt_m(mwrong)}")
    if mwrong["max_abs"] > 100 * floor["floor_max_abs"]:
        print("  good: harness distinguishes U^T U from U U^T on this H.")
    else:
        print("  BLIND SPOT: U U^T ≈ U^T U on this H (near-diagonal); not a general miss.")

    # Absolute-damp mis-wiring: pass 0.01 as absolute when mean(diag)>>1
    u_abs, damp_abs = hipfire_damped_inv_cholesky_upper(
        h, initial_damp=0.01, max_damp_multiplier=100.0, damp_is_absolute=True
    )
    m_abs = metrics(u_paper, u_abs)
    print(f"  U paper vs hipfire ABSOLUTE damp=0.01: {fmt_m(m_abs)}")
    print(f"  damp_abs={damp_abs:.6e} (paper damp={damp_p:.6e})")
    if m_abs["max_abs"] > 50 * floor["floor_max_abs"]:
        print("  FINDING: absolute vs fractional damp wiring changes U when mean(diag) != 1.")
        print("           gptq.rs takes absolute damp; callers must pre-multiply by mean(diag).")
        print("           e8_gptq.rs uses LAMBDA*mean(diag) internally (paper-correct).")

    return {
        "u_frac": mu,
        "invariant": minv,
        "wrong_invariant_uut": mwrong,
        "u_absolute_damp": m_abs,
        "damp_paper": damp_p,
        "damp_abs": damp_abs,
    }


# ---------------------------------------------------------------------------
# Stage 3 — full GPTQ
# ---------------------------------------------------------------------------

def stage3_full_gptq(floor: dict) -> dict:
    print("\n=== Stage 3: full GPTQ pass (paper ref vs hipfire_shadow) ===")
    m, k, n = 16, 64, 128
    w = torch.randn(m, k, dtype=torch.float64) * 0.5
    x = torch.randn(n, k, dtype=torch.float64)
    h_paper = accumulate_hessian_xxT(x, normalize=True, factor_two=True)
    h_paper = h_paper + 1e-3 * torch.eye(k, dtype=torch.float64)
    # HFHS-convention H (= half of paper). With fractional damp both should
    # agree on quantized weights (scale invariance of OBS ratios).
    h_hfhs = hipfire_hfhs_accumulate(x)
    h_hfhs = h_hfhs + 0.5e-3 * torch.eye(k, dtype=torch.float64)

    qfn = frozen_minmax_int4_quantize_fn(w, group_size=0)
    cfg = GptqConfig(percdamp=0.01, blocksize=32, actorder=True)
    r_paper = gptq_fasterquant(w, h_paper, quantize_fn=qfn, cfg=cfg)

    # hipfire shadow with fractional damp on HFHS hessian
    r_hip = hipfire_gptq_column_sequential(
        w,
        h_hfhs,
        quantize_fn=qfn,
        initial_damp=0.01,
        max_damp_multiplier=1.0,
        damp_is_absolute=False,
        actorder=True,
    )

    mq = metrics(r_paper.qweight, r_hip.qweight)
    print(f"  Q paper(H_2/N) vs hipfire(H_1/N, frac damp): {fmt_m(mq)}")
    print(f"  {verdict(mq, floor['floor_max_abs'], floor['floor_rel_frob'], 'Q-scale-invariant')}")

    # Same H (paper) both sides
    r_hip2 = hipfire_gptq_column_sequential(
        w,
        h_paper,
        quantize_fn=qfn,
        initial_damp=0.01,
        max_damp_multiplier=1.0,
        damp_is_absolute=False,
        actorder=True,
    )
    mq2 = metrics(r_paper.qweight, r_hip2.qweight)
    print(f"  Q paper vs hipfire SAME H_paper:             {fmt_m(mq2)}")
    print(f"  {verdict(mq2, floor['floor_max_abs'], floor['floor_rel_frob'], 'Q-same-H')}")

    # Reconstruction error under activation Gram (paper objective proxy)
    # E = ||(W-Q) X||_F^2  with X from calibration
    def recon_err(q: torch.Tensor) -> float:
        err = (w - q) @ x.transpose(0, 1)  # [M, N]
        return float((err * err).sum())

    e_paper = recon_err(r_paper.qweight)
    e_hip = recon_err(r_hip.qweight)
    e_rtn_qfn = frozen_minmax_int4_quantize_fn(w, group_size=0)
    # RTN baseline
    rtn = torch.stack([e_rtn_qfn(w[:, j], j) for j in range(k)], dim=1)
    e_rtn = recon_err(rtn)
    print(f"  recon_err paper={e_paper:.6e}  hipfire={e_hip:.6e}  rtn={e_rtn:.6e}")
    if e_paper <= e_rtn * 1.01:
        print("  paper GPTQ <= RTN recon (expected).")
    else:
        print("  WARNING: paper GPTQ worse than RTN on this draw — rare but possible.")

    # Actorder: confirm perm descending diag
    perm = weight_mode_actorder_perm(h_paper)
    diag = torch.diagonal(h_paper)
    assert torch.all(diag[perm][:-1] >= diag[perm][1:] - 1e-15)
    print(f"  actorder perm (first 8): {perm[:8].tolist()}")

    return {
        "q_scale_invariant": mq,
        "q_same_h": mq2,
        "recon_paper": e_paper,
        "recon_hip": e_hip,
        "recon_rtn": e_rtn,
    }


# ---------------------------------------------------------------------------
# Stage 4 — fault injection
# ---------------------------------------------------------------------------

def stage4_faults(floor: dict) -> dict:
    print("\n=== Stage 4: fault injection (harness teeth) ===")
    m, k, n = 12, 48, 96
    w = torch.randn(m, k, dtype=torch.float64) * 0.4
    x = torch.randn(n, k, dtype=torch.float64)
    h = accumulate_hessian_xxT(x, normalize=True, factor_two=True)
    h = h + 1e-2 * torch.eye(k, dtype=torch.float64)
    qfn = frozen_minmax_int4_quantize_fn(w, group_size=0)
    cfg = GptqConfig(percdamp=0.01, blocksize=16, actorder=True)

    clean = gptq_fasterquant(w, h, quantize_fn=qfn, cfg=cfg)
    results = {}

    injections = [
        "transpose_h",
        "no_damp",
        "no_invperm",
        "no_error_feedback",
    ]
    for inj in injections:
        try:
            bad = gptq_fasterquant(w, h, quantize_fn=qfn, cfg=cfg, inject=inj)
            mm = metrics(clean.qweight, bad.qweight)
            caught = mm["max_abs"] > max(100.0 * floor["floor_max_abs"], 1e-6)
            # Also flag pure scale / perm signatures.
            sig = ""
            if mm["cosine"] > 0.999 and abs(mm["norm_ratio"] - 1.0) > 0.05:
                sig = " scale-signature"
            if abs(mm["cosine"]) < 0.5 and abs(mm["norm_ratio"] - 1.0) < 0.25:
                sig = " perm/transpose-signature"
            status = "CAUGHT" if caught else "MISSED"
            print(f"  inject={inj:20s}  {fmt_m(mm)}  => {status}{sig}")
            results[inj] = {"metrics": mm, "caught": caught, "signature": sig.strip()}
        except Exception as e:  # noqa: BLE001
            # Loud failure (e.g. Cholesky of transposed broken H) also counts as caught.
            print(f"  inject={inj:20s}  RAISED {type(e).__name__}: {e}  => CAUGHT (exception)")
            results[inj] = {"metrics": None, "caught": True, "exception": repr(e)}

    missed = [k for k, v in results.items() if not v["caught"]]
    if missed:
        print(f"  BLIND SPOTS (injection not caught): {missed}")
        print("  Documented so Gate 9 does not trust this oracle past its teeth.")
    else:
        print("  All four injections caught.")

    # Extra: wrong inverse-Cholesky invariant used in OBS (U U^T vs U^T U)
    # Implemented ad-hoc: take clean Hinv_U and replace with L factor wrongly.
    print("  --- extra: wrong U form (U U^T = H_inv instead of U^T U) ---")
    u_right, damp, hd = damp_and_inv_cholesky_upper(h, percdamp=0.01)
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(hd))
    # Wrong upper-looking factor: take lower chol of Hinv and pretend it's U.
    l_hi = torch.linalg.cholesky(hinv)  # lower, L L^T = Hinv
    # If someone returns L^T thinking any factor works... actually L^T is correct U.
    # The bug was returning L_H^{-T} from chol(H) directly without second chol.
    l_h = torch.linalg.cholesky(hd)
    l_h_inv_t = torch.linalg.solve_triangular(
        l_h, torch.eye(k, dtype=torch.float64), upper=False
    ).transpose(0, 1)  # L_H^{-T}, satisfies U U^T = H_inv (WRONG form for OBS)
    # Run OBS manually with wrong U
    def run_with_u(u_mat: torch.Tensor, do_actorder: bool = True) -> torch.Tensor:
        ww = w.clone()
        hh = h.clone()
        if do_actorder:
            perm = weight_mode_actorder_perm(hh)
            ww = ww[:, perm]
            # u_mat already in some order — rebuild properly below
        # Simpler: no actorder for this probe
        ww = w.clone()
        q_out = torch.zeros_like(ww)
        residual = ww.clone()
        for j in range(k):
            d = u_mat[j, j]
            q = qfn(residual[:, j], j)
            q_out[:, j] = q
            err = (residual[:, j] - q) / d
            residual[:, j:] = residual[:, j:] - err.unsqueeze(1) * u_mat[j, j:].unsqueeze(0)
        return q_out

    # Right U without actorder for fair compare
    u_r, _, hd2 = damp_and_inv_cholesky_upper(h, percdamp=0.01, perm=None)
    q_right = run_with_u(u_r)
    q_wrong = run_with_u(l_h_inv_t)
    m_form = metrics(q_right, q_wrong)
    caught_form = m_form["max_abs"] > max(100.0 * floor["floor_max_abs"], 1e-6)
    print(f"  wrong U form (L_H^{{-T}}) vs right U: {fmt_m(m_form)}  => "
          f"{'CAUGHT' if caught_form else 'MISSED'}")
    results["wrong_u_form_LHmT"] = {"metrics": m_form, "caught": caught_form}
    if not caught_form:
        print("  BLIND SPOT: wrong U form not visible on this well-conditioned draw.")

    return results


# ---------------------------------------------------------------------------
# Stage 5 — format round-trip
# ---------------------------------------------------------------------------

def stage5_formats() -> dict:
    print("\n=== Stage 5: HFHS + E8H1 format round-trip ===")
    out: dict = {}
    with tempfile.TemporaryDirectory(prefix="gptq_xref_") as td:
        td_path = Path(td)

        # --- HFHS ---
        h1 = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.float64)
        h2 = np.array([[3.0, -0.5, 0.1], [-0.5, 4.0, 0.2], [0.1, 0.2, 5.0]], dtype=np.float64)
        hfhs_path = td_path / "test.hfhs"
        write_hfhs(
            hfhs_path,
            [
                ("model.layers.0.q_proj", 0, h1.astype(np.float32)),
                ("model.layers.1.mlp.down_proj", 3, h2),  # f64 path below
            ],
            dtype="f32",
        )
        # Rewrite with mixed dtypes manually for f64 second tensor
        write_hfhs(
            hfhs_path,
            [
                ("model.layers.0.q_proj", 0, h1),
                ("model.layers.1.mlp.down_proj", 3, h2),
            ],
            dtype="f32",
        )
        parsed = read_hfhs(hfhs_path)
        assert parsed.version == 1
        assert len(parsed.tensors) == 2
        t0 = parsed.get("model.layers.0.q_proj", 0)
        assert t0 is not None and t0.k == 2
        err0 = float(np.max(np.abs(t0.h - h1)))
        print(f"  HFHS t0 max_abs roundtrip: {err0:.6e}  magic_ok={hfhs_path.read_bytes()[:4]==HFHS_MAGIC}")
        t1 = parsed.get("model.layers.1.mlp.down_proj", 3)
        assert t1 is not None and t1.k == 3
        err1 = float(np.max(np.abs(t1.h - h2.astype(np.float32).astype(np.float64))))
        print(f"  HFHS t1 max_abs roundtrip (via f32): {err1:.6e}")
        assert parsed.get("model.layers.1.mlp.down_proj", 0) is None
        out["hfhs"] = {"t0_max_abs": err0, "t1_max_abs": err1, "n_tensors": len(parsed.tensors)}

        # Also f64 dtype flag path
        hfhs64 = td_path / "test64.hfhs"
        write_hfhs(hfhs64, [("tB", 0, h2)], dtype="f64")
        p64 = read_hfhs(hfhs64)
        err64 = float(np.max(np.abs(p64.tensors[0].h - h2)))
        print(f"  HFHS f64 dtype roundtrip max_abs: {err64:.6e}  dtype_flag={p64.tensors[0].dtype_flag}")
        out["hfhs_f64_max_abs"] = err64

        # --- E8H1 ---
        k = 512
        rng = np.random.default_rng(0)
        acts = rng.standard_normal((10, k)).astype(np.float64)
        blocks = accumulate_block_diagonal_xxT(acts)
        name = "layers.0.mlp.experts.3.gate_up_proj.weight"
        hblk_path = td_path / f"{hessian_key(name)}.hblk"
        write_hblk(hblk_path, k, blocks)
        raw = hblk_path.read_bytes()
        magic = int.from_bytes(raw[0:4], "little")
        n_blocks = int.from_bytes(raw[4:8], "little")
        k_stored = int.from_bytes(raw[8:12], "little")
        print(f"  E8H1 header magic=0x{magic:08x} (expect 0x{E8H1_MAGIC:08x}) "
              f"n_blocks={n_blocks} K={k_stored} nbytes={len(raw)}")
        assert magic == E8H1_MAGIC
        assert n_blocks == 2 and k_stored == 512
        assert len(raw) == 12 + 2 * 256 * 256 * 4
        parsed_h = read_hblk(hblk_path)
        # f64 -> f32 -> f64 roundtrip error
        err_h = float(np.max(np.abs(parsed_h.blocks - blocks.astype(np.float32).astype(np.float64))))
        print(f"  E8H1 payload max_abs roundtrip (f64 via f32): {err_h:.6e}")
        # Independent re-accumulate and compare block 1 entry (3,5)
        ref35 = float(np.sum(acts[:, 256 + 3] * acts[:, 256 + 5]))
        got35 = float(parsed_h.blocks[1][3, 5])
        print(f"  E8H1 block1[3,5] got={got35:.8f} ref={ref35:.8f} diff={abs(got35-ref35):.6e}")
        out["e8h1"] = {
            "magic": magic,
            "n_blocks": n_blocks,
            "k": k_stored,
            "nbytes": len(raw),
            "payload_max_abs": err_h,
            "entry35_diff": abs(got35 - ref35),
            "key": hessian_key(name),
        }
        assert hessian_key("a/b\\c..d") == "a_b_c_d"
        print(f"  hessian_key sanitization ok: a/b\\c..d -> {hessian_key('a/b\\c..d')}")

    print("  Format round-trips: PASS")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("GPTQ cross-reference harness")
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  device=cpu")
    print(f"seed={SEED:#x}")

    floor = stage0_floor()
    s1 = stage1_hessian(floor)
    s2 = stage2_cholesky(floor)
    s3 = stage3_full_gptq(floor)
    s4 = stage4_faults(floor)
    s5 = stage5_formats()

    summary = {
        "floor": floor,
        "stage1_hessian": {
            "paper_vs_2hfhs": s1["paper_vs_2hfhs"],
            "paper_vs_hfhs_direct": s1["paper_vs_hfhs_direct"],
            "e8_block0": s1["e8_block0"],
            "e8_cross_block_frob": s1["e8_cross_block_frob"],
        },
        "stage2_cholesky": {
            "u_frac": s2["u_frac"],
            "invariant": s2["invariant"],
            "wrong_invariant_uut": s2["wrong_invariant_uut"],
            "u_absolute_damp": s2["u_absolute_damp"],
            "damp_paper": s2["damp_paper"],
            "damp_abs": s2["damp_abs"],
        },
        "stage3_gptq": {
            "q_scale_invariant": s3["q_scale_invariant"],
            "q_same_h": s3["q_same_h"],
            "recon_paper": s3["recon_paper"],
            "recon_hip": s3["recon_hip"],
            "recon_rtn": s3["recon_rtn"],
        },
        "stage4_faults": {
            k: {
                "caught": v["caught"],
                "metrics": v.get("metrics"),
                "exception": v.get("exception"),
            }
            for k, v in s4.items()
        },
        "stage5_formats": s5,
    }

    out_json = HERE / "last_run.json"
    out_json.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {out_json}")

    # Exit non-zero if any required injection was missed or format broke.
    missed = [k for k, v in s4.items() if not v["caught"]]
    # wrong_u_form is extra; required four:
    required_miss = [k for k in ("transpose_h", "no_damp", "no_invperm", "no_error_feedback") if k in missed]
    fmt_ok = (
        s5["hfhs"]["t0_max_abs"] == 0.0
        and s5["e8h1"]["magic"] == E8H1_MAGIC
        and s5["e8h1"]["entry35_diff"] < 1e-5
    )
    # Stage 3 same-H should pass
    q_ok = s3["q_same_h"]["max_abs"] <= max(50.0 * floor["floor_max_abs"], 1e-6)

    print("\n=== SUMMARY ===")
    print(f"  floor_max_abs={floor['floor_max_abs']:.6e}  domain={floor['domain']}")
    print(f"  format_ok={fmt_ok}  q_same_h_ok={q_ok}  required_injections_missed={required_miss}")
    if required_miss:
        print("  HARNESS INCOMPLETE: blind spots remain on required injections.")
        return 2
    if not fmt_ok or not q_ok:
        print("  HARNESS FAILED a required equivalence/format check.")
        return 1
    print("  HARNESS OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
