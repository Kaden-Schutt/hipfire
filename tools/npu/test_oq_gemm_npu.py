#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""OQ8 (W8A8) / OQ+ (W4A8) int8 grouped GEMM — NPU oracle + bench.

Validates that the XDNA1 (AIE2) NPU reproduces the hipfire Opus-Quant grouped
matmul, and benchmarks it. The compute is the int8·int8→int32 contraction the
GPU runs on v_wmma_i32_16x16x16_iu8 (kernels/src/gemm_oq8_grouped_wmma.hip),
plus the per-group (G=256) f32 rescale:

    Y[b,m] = Σ_g scale_w[m,g]·scale_x[b,g] · Σ_{k∈g} q_w[m,k]·q_x[b,k]

The int32 contraction runs on the NPU (per 256-group); the f32 rescale is done
on the host (in production it fuses into the kernel epilogue — that's a follow-up
if the spike says go). For --wbits 4 (OQ+ W4A8) the weights are quantized to
signed int4 then unpacked to int8 host-side before the same matmul; on-tile
nibble unpack (the real DMA win) is a follow-up.

FWHT rotation is omitted: it's an orthonormal pre-rotation applied identically
to W and X offline and does not affect whether the NPU reproduces the
post-rotation int8 matmul, which is what this validates.

Env is bootstrapped by oq_gemm_design (pyxrt path, coreutil preload, boost
LD_LIBRARY_PATH) — no external env vars needed. NPU must be free (no daemon
holding the npu-0 resource lease).

Usage:
    python tools/npu/test_oq_gemm_npu.py --wbits 8 --M 64 --K 256 --B 64
    python tools/npu/test_oq_gemm_npu.py --wbits 4 --M 64 --K 256 --B 64
    python tools/npu/test_oq_gemm_npu.py --wbits 8 --M 512 --K 1024 --B 256 --bench
"""

import argparse
import sys

import numpy as np

# oq_gemm_design must import first — it sets up sys.path/LD_LIBRARY_PATH/pyxrt.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import oq_gemm_design as design  # noqa: E402

GROUP = 256


def quantize_group_symmetric(x_f32, bits):
    """Per-256-group symmetric quant. x_f32: [rows, K] (K % 256 == 0).
    Returns (q int8 [rows,K], scale f32 [rows, K/256]). Matches the GPU codec:
    scale = absmax(group)/qmax; q = round(x/scale).clip(-qmax, qmax)."""
    qmax = (1 << (bits - 1)) - 1  # 127 for int8, 7 for int4
    rows, K = x_f32.shape
    ng = K // GROUP
    xg = x_f32.reshape(rows, ng, GROUP)
    absmax = np.abs(xg).max(axis=2)                      # [rows, ng]
    scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
    q = np.round(xg / scale[:, :, None]).clip(-qmax, qmax).astype(np.int8)
    return q.reshape(rows, K), scale


def oq_reference(qw, sw, qx, sx):
    """Host f32 oracle: Σ_g sw[m,g]·sx[b,g]·Σ_{k∈g} qw[m,k]·qx[b,k]. Returns Y[B,M]."""
    M, K = qw.shape
    B = qx.shape[0]
    ng = K // GROUP
    Y = np.zeros((B, M), dtype=np.float32)
    qwg = qw.astype(np.int64).reshape(M, ng, GROUP)
    qxg = qx.astype(np.int64).reshape(B, ng, GROUP)
    for g in range(ng):
        P = qxg[:, g, :] @ qwg[:, g, :].T                # [B, M] int64
        Y += (sx[:, g][:, None] * sw[:, g][None, :]) * P.astype(np.float32)
    return Y


def npu_grouped(qw, qx, run):
    """Run the per-group int8 contraction on the NPU. Returns int32 partials
    P[ng][M,B] stacked as [ng, M, B], plus the chosen (m,k,n) tile."""
    M, K = qw.shape
    B = qx.shape[0]
    ng = K // GROUP
    parts = np.empty((ng, M, B), dtype=np.int32)
    tile_used = None
    for g in range(ng):
        Wg = qw[:, g * GROUP:(g + 1) * GROUP]            # [M, 256] int8
        Xg = qx[:, g * GROUP:(g + 1) * GROUP]            # [B, 256] int8
        C, tile_used = run(Wg, Xg)                       # C[M,B] = Wg · Xg^T
        parts[g] = C
    return parts, tile_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbits", type=int, choices=[8, 4], default=8)
    ap.add_argument("--M", type=int, default=64, help="output rows (weight rows)")
    ap.add_argument("--K", type=int, default=256, help="contraction dim (mult of 256)")
    ap.add_argument("--B", type=int, default=64, help="batch / tokens (the N dim)")
    ap.add_argument("--bench", action="store_true", help="also benchmark one group matmul")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert args.K % GROUP == 0, "K must be a multiple of 256"
    fmt = "OQ8 (W8A8)" if args.wbits == 8 else "OQ+ (W4A8)"
    print(f"[test_oq_gemm] {fmt}  M={args.M} K={args.K} B={args.B}  groups={args.K // GROUP}")

    rng = np.random.default_rng(args.seed)
    Wf = (rng.standard_normal((args.M, args.K)) * 0.7).astype(np.float32)
    Xf = (rng.standard_normal((args.B, args.K)) * 1.0).astype(np.float32)

    # Activations are always int8 (A8); weights are int8 (OQ8) or int4 (OQ+).
    qw, sw = quantize_group_symmetric(Wf, args.wbits)
    qx, sx = quantize_group_symmetric(Xf, 8)
    # OQ+ weights quantize to int4 but the matmul consumes int8 — values already
    # fit [-7,7], so the int8 array IS the unpacked form (host-side unpack).
    print(f"  weight q range [{qw.min()},{qw.max()}]  act q range [{qx.min()},{qx.max()}]")

    # ── NPU int32 contraction (per group) ────────────────────────────────────
    parts, tile = npu_grouped(qw, qx, lambda Wg, Xg: design.matmul_npu(Wg, Xg))
    print(f"  npu tile (m,k,n) = {tile}")

    # ── Bit-exact int32 check vs numpy int64 contraction (per group) ─────────
    M, K, B = args.M, args.K, args.B
    ng = K // GROUP
    qwg = qw.astype(np.int64).reshape(M, ng, GROUP)
    qxg = qx.astype(np.int64).reshape(B, ng, GROUP)
    int_ok = True
    for g in range(ng):
        ref_i = (qwg[:, g, :] @ qxg[:, g, :].T).astype(np.int64)   # [M,B]
        if not np.array_equal(parts[g].astype(np.int64), ref_i):
            int_ok = False
            diff = np.abs(parts[g].astype(np.int64) - ref_i)
            print(f"  GROUP {g}: int32 MISMATCH  max|Δ|={diff.max()} "
                  f"({np.count_nonzero(diff)}/{diff.size} elems)")
            break
    print(f"  int32 contraction bit-exact: {'PASS' if int_ok else 'FAIL'}")

    # ── f32 rescale + compare to oracle ──────────────────────────────────────
    Y_npu = np.zeros((B, M), dtype=np.float32)
    for g in range(ng):
        Y_npu += (sx[:, g][:, None] * sw[:, g][None, :]) * parts[g].T.astype(np.float32)
    Y_ref = oq_reference(qw, sw, qx, sx)
    abs_err = np.abs(Y_npu - Y_ref)
    rel = abs_err / (np.abs(Y_ref) + 1e-6)
    atol = 1e-3
    f32_ok = bool((abs_err <= atol + 1e-3 * np.abs(Y_ref)).all())
    print(f"  f32 rescale: max_abs={abs_err.max():.2e}  mean_abs={abs_err.mean():.2e}  "
          f"max_rel={rel.max():.2e}  {'PASS' if f32_ok else 'FAIL'} (atol={atol})")

    ok = int_ok and f32_ok
    if args.bench:
        Wg = qw[:, :GROUP]
        Xg = qx[:, :GROUP]
        _, bench, btile = design.bench_npu(Wg, Xg)
        npu_us = getattr(bench, "npu_time_us", None) or getattr(bench, "npu_time", None)
        e2e_us = getattr(bench, "e2e_time_us", None) or getattr(bench, "e2e_time", None)
        print(f"  bench (one 256-group, M={M} N={B}, tile={btile}): "
              f"npu={npu_us} e2e={e2e_us} (us)")

    print(f"[test_oq_gemm] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
