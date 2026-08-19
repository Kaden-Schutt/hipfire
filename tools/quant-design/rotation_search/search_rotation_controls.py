#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""search_rotation_controls.py — offline rotation-control upper-bound experiments.

Deterministic NumPy CLI that screens three control families against production
MQ4V2 plus hypothetical matched-header 2-bit/3-bit V2 research layouts (two
G128 grids, truncating f32→f16 headers, floor(x+0.5)). Canonical rotation is

    R = D2 · H_256 · D1
    D1 seed = 42, D2 seed = 1042
    H_256 = unnormalized 8-stage FWHT, then ×1/16

CONTROL-ONLY STATUS
  d1-seeds     D1 LCG seed is already a runtime dial; trace established.
  partitions   Offline control: output-group partition is not a deployed
               runtime path. runtime_trace_established=false.
  butterfly    Offline control: signed FWHT butterflies are not a deployed
               runtime path. runtime_trace_established=false.

  Partition and butterfly results are offline upper bounds only — not
  deployable winners. They may change runtime topology; do not treat a
  winning CSV row as a shippable kernel config.

USAGE
  search_rotation_controls.py --help
  search_rotation_controls.py d1-seeds   INPUT COUNT OUT.csv
  search_rotation_controls.py partitions INPUT OUT.csv
  search_rotation_controls.py butterfly  INPUT COUNT RNG_SEED OUT.csv

INPUT
  Headerless little-endian f32 [nblocks, 256] raw unrotated weight groups.
  Byte length must be a positive multiple of 1024. Non-finite values refused.

MODES
  d1-seeds INPUT COUNT OUT.csv
    Screen D1 LCG seeds in [0, COUNT) with D2 fixed at 1042.
    Always emits canonical D1=42 rows (even if 42 not in [0, COUNT)).
    Contiguous G128 affine grids. Fixed top-1% |U| tail mask from the
    canonical pre-D2 transform U (D2-invariant magnitudes).

  partitions INPUT OUT.csv
    Evaluate all 255 nonzero GF(2)^8 parity masks m in {1..255} as balanced
    128/128 output-group partitions after canonical R (D1=42, D2=1042):
      half(i) = parity(i & m)          # popcount mod 2
      groups  = indices sorted ascending within parity-0 then parity-1
    Each group is one asymmetric affine grid (replaces contiguous halves).
    Canonical mask = 128 (0x80) = contiguous high-bit split.
    OFFLINE CONTROL — runtime_trace_established=false.

  butterfly INPUT COUNT RNG_SEED OUT.csv
    Screen COUNT signed-butterfly FWHT networks. Canonical D1=42 and D2=1042
    are preserved around the butterfly. Each of 8 stages (stride 1..128) has
    one +/- orientation bit per butterfly pair (128 bits/stage, 1024 total):
      a,b = x[i+j], x[i+j+stride]
      x[i+j]         = a + b
      x[i+j+stride]  = s * (a - b)     # s in {+1,-1}; canonical all +
    Networks i=0..COUNT-1 drawn from LCG seed (RNG_SEED + i + 1).
    Always emits the all-+ canonical network (candidate_id=-1).
    OFFLINE CONTROL — runtime_trace_established=false.

CODEC (production MQ4V2 arithmetic; 2/3-bit layouts are hypothetical)
  lo,hi = min/max of the 128-group
  st,z  = f16_to_f32(f32_to_f16_TRUNC(.))   # crates/hipfire-quantize float16.rs
  q     = clamp(floor((v-z)/st + 0.5), 0, 2^b-1)
  recon = q*st + z
  Degenerate (hi==lo or st==0 after trunc): q=0, recon=z.
  Bits in {2,3,4}.

TAIL MASK
  Fixed pre-candidate top-1% |U| mask from the mode's canonical transformed
  coefficients (post-FWHT index space; |D2·U|=|U| so D2 does not move the
  mask). Same mask for every candidate in a run.

METRICS (one CSV row per candidate x bits)
  mse            mean squared recon error over all coeffs
  tail_mse       mean squared error on the fixed top-1% |U| mask
  abs_p99        99th percentile of |recon-u| (linear interp on sorted abs err)
  abs_p999       99.9th percentile of |recon-u|
  abs_max        max |recon-u|
  rel_mse        mse / mse_canonical(bits)
  rel_tail       tail_mse / tail_mse_canonical(bits)
  rel_p999       abs_p999 / abs_p999_canonical(bits)
  Macro relatives (equal weight over bits) are the mean of the per-bit rel_*
  columns across the three rows of a candidate.

CSV SCHEMA (header row, then data)
  control,candidate_id,bits,mse,tail_mse,abs_p99,abs_p999,abs_max,
  rel_mse,rel_tail,rel_p999,runtime_trace_established,is_canonical

  control                      d1-seeds | partitions | butterfly
  candidate_id                 d1 seed | mask | network ordinal
                               (canonical butterfly uses candidate_id=-1)
  bits                         2 | 3 | 4
  runtime_trace_established    true for d1-seeds; false for partitions/butterfly
  is_canonical                 true for the explicit canonical reference rows

OUTPUT
  Deterministic. Candidates processed in bounded batches. Atomic CSV write.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

G = 256
HALF = 128
BITS = (2, 3, 4)
QMAX = {2: 3, 3: 7, 4: 15}
D1_CANON_SEED = 42
D2_CANON_SEED = 1042
FWHT_SCALE = np.float32(0.0625)
BATCH_CAND = 32  # bounded candidate processing
CSV_HEADER = (
    "control,candidate_id,bits,mse,tail_mse,abs_p99,abs_p999,abs_max,"
    "rel_mse,rel_tail,rel_p999,runtime_trace_established,is_canonical\n"
)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def usage(argv0: str, code: int = 1) -> None:
    text = f"""search_rotation_controls — offline rotation-control upper-bound experiments

USAGE
  {argv0} --help
  {argv0} d1-seeds   INPUT COUNT OUT.csv
  {argv0} partitions INPUT OUT.csv
  {argv0} butterfly  INPUT COUNT RNG_SEED OUT.csv

INPUT
  Headerless little-endian f32 [nblocks,256] raw unrotated weight groups.
  Size must be a positive multiple of 1024 bytes. Non-finite values rejected.

MODES
  d1-seeds INPUT COUNT OUT.csv
    Screen D1 LCG seeds [0, COUNT) with D2=1042. Contiguous G128 affine grids.
    Canonical D1=42 always emitted. runtime_trace_established=true.

  partitions INPUT OUT.csv
    All 255 nonzero GF(2)^8 parity masks as balanced 128/128 output partitions
    after canonical R (D1=42,D2=1042). half(i)=parity(i&m); groups = indices
    ascending within parity-0 then parity-1. Canonical mask=128 (0x80).
    OFFLINE CONTROL — runtime trace NOT established; not a deployable winner.

  butterfly INPUT COUNT RNG_SEED OUT.csv
    COUNT signed-butterfly networks; each of 8 FWHT stages has one +/- orientation
    bit per butterfly (1024 bits). Pair: (a+b, s*(a-b)); canonical all +.
    Canonical D1=42 and D2=1042 preserved around the butterfly.
    OFFLINE CONTROL — runtime trace NOT established; not a deployable winner.

CODEC
  Two affine grids per G256 (contiguous halves, or partition groups).
  st,z = f16_to_f32(f32_to_f16_TRUNC(.))  # hipfire float16.rs truncation, NOT RNE
  q = clamp(floor((v-z)/st + 0.5), 0, 2^b-1); bits in {{2,3,4}}

TAIL
  Fixed pre-candidate top-1% |U| mask from the mode canonical transform.

CSV SCHEMA (one row per candidate x bits; deterministic)
  control,candidate_id,bits,mse,tail_mse,abs_p99,abs_p999,abs_max,
  rel_mse,rel_tail,rel_p999,runtime_trace_established,is_canonical

  mse/tail_mse     mean squared error (all coeffs / tail mask)
  abs_p99/p999/max absolute recon error percentiles / max
  rel_*            candidate / canonical (same bits); macro = mean over bits
  runtime_trace_established
                   true only for d1-seeds; false for partitions and butterfly
  is_canonical     true on explicit canonical reference rows

CONTROL-ONLY
  partitions and butterfly change runtime topology. CSV hits are offline
  upper-bound diagnostics, not shippable kernel configs.
"""
    print(text, file=sys.stderr if code else sys.stdout)
    raise SystemExit(code)


# ── production f16 truncation (float16.rs) ───────────────────────────────────


def f32_to_f16_trunc_bits(val: np.ndarray) -> np.ndarray:
    """Vectorized Hipfire truncating f32→f16 bit patterns (not IEEE RNE)."""
    bits = val.astype(np.float32, copy=False).view(np.uint32)
    sign = (bits >> np.uint32(31)) & np.uint32(1)
    exp = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    frac = bits & np.uint32(0x7FFFFF)

    out = np.zeros(val.shape, dtype=np.uint32)

    nan_inf = exp == 0xFF
    f16_frac = np.where(
        frac == 0, np.uint32(0), (frac >> np.uint32(13)) | np.uint32(1)
    )
    out = np.where(
        nan_inf,
        (sign << np.uint32(15)) | (np.uint32(0x1F) << np.uint32(10)) | f16_frac,
        out,
    )

    new_exp = exp - np.int32(127) + np.int32(15)
    overflow = (~nan_inf) & (new_exp >= 31)
    out = np.where(
        overflow,
        (sign << np.uint32(15)) | (np.uint32(0x1F) << np.uint32(10)),
        out,
    )

    under_hard = (~nan_inf) & (~overflow) & (new_exp < -10)
    out = np.where(under_hard, sign << np.uint32(15), out)

    sub = (~nan_inf) & (~overflow) & (~under_hard) & (new_exp <= 0)
    if np.any(sub):
        f = frac | np.uint32(0x800000)
        shift = (np.int32(1) - new_exp + np.int32(13)).astype(np.uint32)
        shifted = np.zeros_like(f)
        shifted[sub] = f[sub] >> shift[sub]
        out = np.where(sub, (sign << np.uint32(15)) | shifted, out)

    normal = (~nan_inf) & (~overflow) & (new_exp > 0) & (new_exp < 31)
    out = np.where(
        normal,
        (sign << np.uint32(15))
        | (new_exp.astype(np.uint32) << np.uint32(10))
        | (frac >> np.uint32(13)),
        out,
    )
    return out.astype(np.uint16)


def f16_bits_to_f32(h: np.ndarray) -> np.ndarray:
    """IEEE binary16 bits → f32 (finite-faithful; matches half crate)."""
    h = h.astype(np.uint16, copy=False)
    sign = (h.astype(np.uint32) >> np.uint32(15)) << np.uint32(31)
    exp = (h.astype(np.uint32) >> np.uint32(10)) & np.uint32(0x1F)
    mant = h.astype(np.uint32) & np.uint32(0x3FF)

    out = np.empty(h.shape, dtype=np.uint32)

    zero = (exp == 0) & (mant == 0)
    sub = (exp == 0) & (mant != 0)
    infnan = exp == 31
    normal = ~(zero | sub | infnan)

    out[zero] = sign[zero]

    if np.any(sub):
        m = mant[sub].copy()
        e = np.full(m.shape, -1, dtype=np.int32)
        for _ in range(10):
            need = (m & np.uint32(0x400)) == 0
            if not np.any(need):
                break
            m = np.where(need, m << np.uint32(1), m)
            e = np.where(need, e + 1, e)
        out[sub] = (
            sign[sub]
            | ((np.int32(127 - 15) - e).astype(np.uint32) << np.uint32(23))
            | ((m & np.uint32(0x3FF)) << np.uint32(13))
        )

    out[infnan] = sign[infnan] | np.uint32(0x7F800000) | (mant[infnan] << np.uint32(13))
    out[normal] = (
        sign[normal]
        | ((exp[normal] + np.uint32(127 - 15)) << np.uint32(23))
        | (mant[normal] << np.uint32(13))
    )
    return out.view(np.float32)


def f16_trunc_rt(x: np.ndarray | float) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    return f16_bits_to_f32(f32_to_f16_trunc_bits(a))


# ── LCG signs / FWHT ─────────────────────────────────────────────────────────


def lcg_next(state: int) -> int:
    """Match search_affine_signs.hip: (state*1103515245+12345) & 0x7fffffff."""
    return (state * 1103515245 + 12345) & 0x7FFFFFFF


def gen_fwht_signs(seed: int, n: int = G) -> np.ndarray:
    """LCG +/-1 table; seed loaded as uint32 then advanced with lcg_next."""
    state = int(seed) & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = lcg_next(state)
        out[i] = np.float32(1.0 if ((state >> 16) & 1) else -1.0)
    return out


def fwht_unnormalized_256(x: np.ndarray) -> None:
    """In-place unnormalized FWHT-256; x shape (..., 256), float32."""
    for stride in (1, 2, 4, 8, 16, 32, 64, 128):
        y = x.reshape(*x.shape[:-1], G // (stride * 2), 2, stride)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        x[...] = y.reshape(x.shape)


def fwht_signed_butterfly_256(x: np.ndarray, ori: np.ndarray) -> None:
    """In-place signed-butterfly FWHT.

    ori: shape (8, 128) float32 +/-1 — stage s, butterfly k orientation.
    Pair update: (a+b, s*(a-b)).
    """
    if ori.shape != (8, HALF):
        raise ValueError(f"ori shape {ori.shape}, expected (8, 128)")
    for si, stride in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
        y = x.reshape(*x.shape[:-1], G // (stride * 2), 2, stride)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        s = ori[si].reshape(G // (stride * 2), stride).astype(np.float32, copy=False)
        while s.ndim < a.ndim:
            s = s.reshape(1, *s.shape)
        y[..., 0, :] = a + b
        y[..., 1, :] = s * (a - b)
        x[...] = y.reshape(x.shape)


def load_corpus(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        die(f"cannot open INPUT {path}")
    raw = p.read_bytes()
    if len(raw) == 0:
        die("INPUT is empty")
    if len(raw) % (G * 4) != 0:
        die(f"INPUT size {len(raw)} is not a multiple of {G * 4} bytes")
    nblocks = len(raw) // (G * 4)
    if nblocks <= 0:
        die("INPUT has zero blocks")
    data = (
        np.frombuffer(raw, dtype="<f4").reshape(nblocks, G).astype(np.float32, copy=True)
    )
    if not np.isfinite(data).all():
        bad = int(np.argmin(np.isfinite(data.ravel())))
        die(f"non-finite value at element {bad}")
    return data


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── affine codec + metrics ───────────────────────────────────────────────────


def _affine_recon_group(v: np.ndarray, qmax: int) -> np.ndarray:
    """v: (nblocks, 128) float32 → recon same shape float32."""
    lo = v.min(axis=1)
    hi = v.max(axis=1)
    z = f16_trunc_rt(lo)
    degenerate = hi == lo
    span = (hi - lo) / np.float32(qmax)
    st_full = f16_trunc_rt(span)
    st = np.where(degenerate, np.float32(0.0), st_full)
    degenerate = degenerate | (st == 0.0)

    z_b = z[:, None]
    st_b = st[:, None]
    st_safe = np.where(degenerate[:, None], np.float32(1.0), st_b)
    q = np.floor((v - z_b) / st_safe + np.float32(0.5))
    q = np.clip(q, 0.0, float(qmax))
    recon = q * st_b + z_b
    if np.any(degenerate):
        recon[degenerate] = z_b[degenerate]
    return recon.astype(np.float32, copy=False)


def affine_recon_contiguous(v: np.ndarray, bits: int) -> np.ndarray:
    """v (nblocks, 256) → recon; two contiguous G128 grids."""
    qmax = QMAX[bits]
    out = np.empty_like(v)
    out[:, :HALF] = _affine_recon_group(v[:, :HALF], qmax)
    out[:, HALF:] = _affine_recon_group(v[:, HALF:], qmax)
    return out


def partition_index_groups(mask: int) -> tuple[np.ndarray, np.ndarray]:
    """Ascending indices for parity-0 and parity-1 of (i & mask)."""
    if mask <= 0 or mask > 255:
        raise ValueError(mask)
    idx = np.arange(G, dtype=np.int32)
    bits = (idx & np.int32(mask)).astype(np.uint32)
    # parity = popcount mod 2 via repeated xor-fold
    parity = np.zeros(G, dtype=np.uint32)
    b = bits
    for _ in range(8):
        parity ^= b & np.uint32(1)
        b >>= np.uint32(1)
    g0 = idx[parity == 0]
    g1 = idx[parity == 1]
    if g0.size != HALF or g1.size != HALF:
        raise RuntimeError(f"mask {mask} unbalanced: {g0.size}/{g1.size}")
    return g0, g1


_PARTITION_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] | None = None


def all_partitions() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    global _PARTITION_CACHE
    if _PARTITION_CACHE is None:
        _PARTITION_CACHE = {m: partition_index_groups(m) for m in range(1, 256)}
    return _PARTITION_CACHE


def affine_recon_partition(
    v: np.ndarray, bits: int, g0: np.ndarray, g1: np.ndarray
) -> np.ndarray:
    """Quantize two parity groups; write recon back to original indices."""
    qmax = QMAX[bits]
    out = np.empty_like(v)
    out[:, g0] = _affine_recon_group(v[:, g0], qmax)
    out[:, g1] = _affine_recon_group(v[:, g1], qmax)
    return out


def percentile_linear(sorted_abs: np.ndarray, p: float) -> float:
    """sorted_abs: 1d ascending; linear interpolation like HIP host_eval."""
    n = sorted_abs.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_abs[0])
    idx = p * float(n - 1)
    i0 = int(math.floor(idx))
    i1 = min(i0 + 1, n - 1)
    t = idx - float(i0)
    return float((1.0 - t) * sorted_abs[i0] + t * sorted_abs[i1])


def build_tail_mask(U: np.ndarray) -> np.ndarray:
    """Fixed top-1% |U| mask; U shape (nblocks, 256). Returns bool ravel mask."""
    abs_u = np.abs(U.ravel())
    n = abs_u.size
    if n == 0:
        die("empty U")
    p99_idx = int(round(0.99 * float(n - 1)))
    part = np.partition(abs_u, p99_idx)
    thr = float(part[p99_idx])
    mask = abs_u >= thr
    if not np.any(mask):
        mask = mask.copy()
        mask[int(np.argmax(abs_u))] = True
    return mask


def eval_metrics(
    v: np.ndarray,
    bits: int,
    tail_mask: np.ndarray,
    groups: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute mse/tail_mse/abs percentiles for one bit-width.

    v: (nblocks, 256) rotated values to quantize.
    tail_mask: bool ravel length nblocks*256 on the same index space as v.
    groups: optional (g0, g1) partition; None → contiguous halves.
    """
    if groups is None:
        recon = affine_recon_contiguous(v, bits)
    else:
        recon = affine_recon_partition(v, bits, groups[0], groups[1])

    err = recon.astype(np.float64) - v.astype(np.float64)
    abs_e = np.abs(err).ravel()
    e2 = (err * err).ravel()
    n = e2.size
    mse = float(e2.mean()) if n else 0.0
    tcount = int(tail_mask.sum())
    tail_mse = float(e2[tail_mask].mean()) if tcount else 0.0
    sorted_abs = np.sort(abs_e)
    return {
        "mse": mse,
        "tail_mse": tail_mse,
        "abs_p99": percentile_linear(sorted_abs, 0.99),
        "abs_p999": percentile_linear(sorted_abs, 0.999),
        "abs_max": float(abs_e.max()) if n else 0.0,
    }


def rel(num: float, den: float) -> float:
    d = den if den > 0.0 else 1e-300
    return float(num / d)


def fmt_row(
    control: str,
    candidate_id: int,
    bits: int,
    m: dict[str, float],
    canon: dict[str, float],
    runtime_trace: bool,
    is_canonical: bool,
) -> str:
    return (
        f"{control},{candidate_id},{bits},"
        f"{m['mse']:.17g},{m['tail_mse']:.17g},"
        f"{m['abs_p99']:.17g},{m['abs_p999']:.17g},{m['abs_max']:.17g},"
        f"{rel(m['mse'], canon['mse']):.17g},"
        f"{rel(m['tail_mse'], canon['tail_mse']):.17g},"
        f"{rel(m['abs_p999'], canon['abs_p999']):.17g},"
        f"{str(runtime_trace).lower()},{str(is_canonical).lower()}\n"
    )


# ── transforms ───────────────────────────────────────────────────────────────


def apply_d1_fwht_scale(raw: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """U = (H · D1 · w) / 16 ; raw (nblocks,256), d1 (256,)."""
    U = (raw * d1[None, :]).astype(np.float32, copy=False)
    fwht_unnormalized_256(U)
    U *= FWHT_SCALE
    return U


def apply_d2(U: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return (U * d2[None, :]).astype(np.float32, copy=False)


def transform_d1_seed(
    raw: np.ndarray, d1_seed: int, d2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (V=D2·U, U_pre_d2) for contiguous metrics / tail."""
    d1 = gen_fwht_signs(d1_seed, G)
    U = apply_d1_fwht_scale(raw, d1)
    return apply_d2(U, d2), U


def transform_canonical(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return V, U_pre_d2, d2_signs for canonical D1=42 D2=1042."""
    d2 = gen_fwht_signs(D2_CANON_SEED, G)
    V, U = transform_d1_seed(raw, D1_CANON_SEED, d2)
    return V, U, d2


def transform_butterfly(
    raw: np.ndarray, ori: np.ndarray, d1: np.ndarray, d2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """V = D2 · (H_ori · D1 · w)/16 ; returns V, U_pre_d2."""
    U = (raw * d1[None, :]).astype(np.float32, copy=False)
    fwht_signed_butterfly_256(U, ori)
    U *= FWHT_SCALE
    return apply_d2(U, d2), U


def gen_butterfly_ori(seed: int) -> np.ndarray:
    """(8, 128) +/-1 from LCG seed — one orientation per butterfly per stage."""
    state = int(seed) & 0xFFFFFFFF
    out = np.empty((8, HALF), dtype=np.float32)
    for s in range(8):
        for k in range(HALF):
            state = lcg_next(state)
            out[s, k] = np.float32(1.0 if ((state >> 16) & 1) else -1.0)
    return out


def canonical_ori() -> np.ndarray:
    return np.ones((8, HALF), dtype=np.float32)


# ── modes ────────────────────────────────────────────────────────────────────


def _canon_bit_metrics(
    V: np.ndarray,
    tail: np.ndarray,
    groups: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[int, dict[str, float]]:
    return {b: eval_metrics(V, b, tail, groups) for b in BITS}


def mode_d1_seeds(input_path: str, count: int, out_csv: str) -> None:
    if count < 0:
        die("COUNT must be >= 0")
    raw = load_corpus(input_path)
    d2 = gen_fwht_signs(D2_CANON_SEED, G)

    V_c, U_c = transform_d1_seed(raw, D1_CANON_SEED, d2)
    tail = build_tail_mask(U_c)  # pre-D2; |D2 U|=|U|
    canon_m = _canon_bit_metrics(V_c, tail)

    lines = [CSV_HEADER]
    for b in BITS:
        lines.append(
            fmt_row("d1-seeds", D1_CANON_SEED, b, canon_m[b], canon_m[b], True, True)
        )

    seeds = [s for s in range(count) if s != D1_CANON_SEED]
    for base in range(0, len(seeds), BATCH_CAND):
        batch = seeds[base : base + BATCH_CAND]
        for seed in batch:
            V, _U = transform_d1_seed(raw, seed, d2)
            for b in BITS:
                m = eval_metrics(V, b, tail, None)
                lines.append(fmt_row("d1-seeds", seed, b, m, canon_m[b], True, False))

    atomic_write_text(out_csv, "".join(lines))
    print(
        f"d1-seeds: count={count} canon_seed={D1_CANON_SEED} "
        f"nblocks={raw.shape[0]} rows={len(lines) - 1} -> {out_csv}",
        flush=True,
    )


def mode_partitions(input_path: str, out_csv: str) -> None:
    raw = load_corpus(input_path)
    V_c, U_c, _d2 = transform_canonical(raw)
    tail = build_tail_mask(U_c)
    parts = all_partitions()
    g_canon = parts[128]
    # Canonical mask 128 == contiguous high-bit split.
    canon_m = _canon_bit_metrics(V_c, tail, g_canon)

    lines = [CSV_HEADER]
    for b in BITS:
        lines.append(
            fmt_row("partitions", 128, b, canon_m[b], canon_m[b], False, True)
        )

    masks = [m for m in range(1, 256) if m != 128]
    for base in range(0, len(masks), BATCH_CAND):
        batch = masks[base : base + BATCH_CAND]
        for mask in batch:
            g = parts[mask]
            for b in BITS:
                m = eval_metrics(V_c, b, tail, g)
                lines.append(
                    fmt_row("partitions", mask, b, m, canon_m[b], False, False)
                )

    atomic_write_text(out_csv, "".join(lines))
    print(
        f"partitions: masks=255 nblocks={raw.shape[0]} "
        f"rows={len(lines) - 1} runtime_trace_established=false -> {out_csv}",
        flush=True,
    )
    print(
        "NOTE: partitions is an OFFLINE CONTROL; runtime trace is not established.",
        flush=True,
    )


def mode_butterfly(
    input_path: str, count: int, rng_seed: int, out_csv: str
) -> None:
    if count < 0:
        die("COUNT must be >= 0")
    raw = load_corpus(input_path)
    d1 = gen_fwht_signs(D1_CANON_SEED, G)
    d2 = gen_fwht_signs(D2_CANON_SEED, G)
    ori_c = canonical_ori()
    V_c, U_c = transform_butterfly(raw, ori_c, d1, d2)
    tail = build_tail_mask(U_c)
    canon_m = _canon_bit_metrics(V_c, tail, None)

    lines = [CSV_HEADER]
    # canonical network id = -1
    for b in BITS:
        lines.append(
            fmt_row("butterfly", -1, b, canon_m[b], canon_m[b], False, True)
        )

    for base in range(0, count, BATCH_CAND):
        batch_n = min(BATCH_CAND, count - base)
        for j in range(batch_n):
            i = base + j
            net_seed = (int(rng_seed) + i + 1) & 0xFFFFFFFF
            ori = gen_butterfly_ori(net_seed)
            V, _U = transform_butterfly(raw, ori, d1, d2)
            for b in BITS:
                m = eval_metrics(V, b, tail, None)
                lines.append(
                    fmt_row("butterfly", i, b, m, canon_m[b], False, False)
                )

    atomic_write_text(out_csv, "".join(lines))
    print(
        f"butterfly: count={count} rng_seed={rng_seed} nblocks={raw.shape[0]} "
        f"rows={len(lines) - 1} runtime_trace_established=false -> {out_csv}",
        flush=True,
    )
    print(
        "NOTE: butterfly is an OFFLINE CONTROL; runtime trace is not established.",
        flush=True,
    )


def parse_nonneg_int(name: str, s: str) -> int:
    try:
        v = int(s, 0)
    except ValueError:
        die(f"{name} must be a non-negative integer, got {s!r}")
    if v < 0:
        die(f"{name} must be >= 0, got {v}")
    return v


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        usage(argv[0], 1)
    mode = argv[1]
    if mode in ("--help", "-h", "help"):
        usage(argv[0], 0)

    if mode == "d1-seeds":
        if len(argv) != 5:
            usage(argv[0], 1)
        mode_d1_seeds(argv[2], parse_nonneg_int("COUNT", argv[3]), argv[4])
        return 0
    if mode == "partitions":
        if len(argv) != 4:
            usage(argv[0], 1)
        mode_partitions(argv[2], argv[3])
        return 0
    if mode == "butterfly":
        if len(argv) != 6:
            usage(argv[0], 1)
        mode_butterfly(
            argv[2],
            parse_nonneg_int("COUNT", argv[3]),
            parse_nonneg_int("RNG_SEED", argv[4]),
            argv[5],
        )
        return 0

    print(f"error: unknown mode {mode!r}", file=sys.stderr)
    usage(argv[0], 1)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
