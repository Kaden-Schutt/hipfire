# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# Independent GPTQ reference written from Frantar et al. arXiv:2210.17323
# and the official DASLab implementation
# (https://github.com/IST-DASLab/gptq/blob/main/gptq.py).
#
# NOT transcribed from hipfire Rust. Disagreements with hipfire are findings.
#
# Paper Algorithm 1 (column-wise OBS / GPTQ):
#   Given W (rows x cols) and Hessian H (cols x cols):
#     1. damp = percdamp * mean(diag(H));  H <- H + damp*I
#     2. Hinv_U = chol( inv( chol(H) ) )   # upper Cholesky of H^{-1}
#        i.e. Hinv_U^T @ Hinv_U = H^{-1}
#     3. for j in 0..cols (optionally act-ordered):
#          q_j = quantize(W[:, j])
#          err = (W[:, j] - q_j) / Hinv_U[j, j]
#          W[:, j:] -= err[:, None] * Hinv_U[j, j:]
#     4. if actorder: unpermute quantized columns
#
# Hessian accumulation (official GPTQ add_batch, simplified without the
# running-mean rescaling trick — equivalent end state):
#   H = (2 / N) * X @ X.T
# where X is [cols, N] (features x samples). The factor of 2 comes from the
# squared-error Hessian of ||W X - Q X||_F^2 = 2 X X^T (paper eq. after
# dropping the constant that cancels in the OBS ratios).
#
# Hipfire HFHS / E8H1 store sum(x x^T) or mean(x x^T) WITHOUT the factor 2;
# see CONVENTIONS.md. The reference exposes both modes explicitly.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class GptqConfig:
    """Knobs matching the official GPTQ reference defaults."""

    percdamp: float = 0.01
    blocksize: int = 128
    actorder: bool = True
    # If True, Hessian is treated as (2/N) X X^T (paper/official).
    # If False, Hessian is treated as (1/N) X X^T (hipfire HFHS finalize).
    # The inverse-Cholesky ratios are scale-invariant under a global scale of H,
    # so both conventions yield identical quantized weights when damping is
    # also scaled consistently (damp ∝ mean(diag(H))).
    hessian_has_factor_two: bool = True


def accumulate_hessian_xxT(
    activations: torch.Tensor,
    *,
    normalize: bool = True,
    factor_two: bool = False,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Accumulate H from activation rows.

    activations: [N, K] — N tokens, K input features (matches hipfire collector
    and nn.Linear input layout after flattening batch/seq).

    Returns KxK Hessian on `dtype`.

    Conventions (caller must pick one and stick to it):
      - hipfire HFHS finalize: normalize=True,  factor_two=False → H = (1/N) X^T X
      - paper / DASLab GPTQ:   normalize=True,  factor_two=True  → H = (2/N) X^T X
      - hipfire E8H1 .hblk:    normalize=False, factor_two=False → H = X^T X (raw sum)
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be [N, K], got {tuple(activations.shape)}")
    x = activations.to(dtype=dtype)
    n, k = x.shape
    if n == 0:
        return torch.zeros((k, k), dtype=dtype)
    # H = X^T @ X  (K x K). activations are row-vectors of features.
    h = x.transpose(0, 1) @ x
    if normalize:
        h = h / float(n)
    if factor_two:
        h = h * 2.0
    # Force symmetry (kills FP drift on the off-diagonals).
    h = 0.5 * (h + h.transpose(0, 1))
    return h


def accumulate_hessian_official_running(
    batches: list[torch.Tensor],
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """DASLab GPTQ add_batch running accumulation, bit-for-bit algorithm.

    Each batch is [N_b, K] (tokens x features). Internally the official code
    transposes to [K, N_b] then does:
        H *= nsamples / (nsamples + tmp)
        nsamples += tmp
        inp = sqrt(2 / nsamples) * inp
        H += inp @ inp.T
    which is algebraically H_final = (2/N) X_all X_all^T.
    """
    h: Optional[torch.Tensor] = None
    nsamples = 0
    k = None
    for batch in batches:
        if batch.ndim != 2:
            raise ValueError(f"batch must be [N, K], got {tuple(batch.shape)}")
        inp = batch.to(dtype=dtype).transpose(0, 1).contiguous()  # [K, N_b]
        tmp = inp.shape[1]
        if h is None:
            k = inp.shape[0]
            h = torch.zeros((k, k), dtype=dtype)
        elif inp.shape[0] != k:
            raise ValueError(f"K mismatch: {inp.shape[0]} vs {k}")
        h *= nsamples / (nsamples + tmp) if (nsamples + tmp) else 1.0
        nsamples += tmp
        scale = (2.0 / nsamples) ** 0.5
        inp = scale * inp
        h = h + inp @ inp.transpose(0, 1)
    assert h is not None
    return 0.5 * (h + h.transpose(0, 1))


def damp_and_inv_cholesky_upper(
    h: torch.Tensor,
    *,
    percdamp: float = 0.01,
    perm: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Paper/DASLab: damp, Cholesky, inverse, upper-Cholesky of inverse.

    Returns (Hinv_U, damp_value, H_damped) where
        Hinv_U is upper-triangular and Hinv_U.T @ Hinv_U = (H_damped)^{-1}
        (equivalently Hinv_U @ Hinv_U.T is NOT the invariant — see paper).

    Steps match gptq.py fasterquant exactly:
        damp = percdamp * mean(diag(H))
        H[diag] += damp
        H = cholesky(H)                 # lower L, L @ L.T = H_damped
        H = cholesky_inverse(H)         # H_inv
        H = cholesky(H, upper=True)     # upper U, U.T @ U = H_inv
    """
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"H must be square, got {tuple(h.shape)}")
    h_work = h.to(dtype=torch.float64).clone()
    cols = h_work.shape[0]

    if perm is not None:
        # P^T H P via advanced indexing (row then col).
        h_work = h_work[perm][:, perm].contiguous()

    # Dead columns: official sets H[d,d]=1 and W[:,d]=0. We only touch H here.
    diag = torch.diagonal(h_work)
    dead = diag == 0
    if dead.any():
        h_work[dead, dead] = 1.0

    damp = float(percdamp) * float(torch.mean(torch.diagonal(h_work)))
    idx = torch.arange(cols, device=h_work.device)
    h_work[idx, idx] += damp

    # Lower Cholesky of damped H.
    l = torch.linalg.cholesky(h_work)  # lower
    h_inv = torch.cholesky_inverse(l)
    # Upper Cholesky of H_inv: U^T U = H_inv.
    u = torch.linalg.cholesky(h_inv, upper=True)
    return u, damp, h_work


def weight_mode_actorder_perm(h: torch.Tensor) -> torch.Tensor:
    """Descending diag(H) permutation (stable for ties via argsort stability)."""
    diag = torch.diagonal(h.to(dtype=torch.float64))
    # torch.argsort is stable.
    return torch.argsort(diag, descending=True, stable=True)


def inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def quantize_asymmetric_int4(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int = 15,
) -> torch.Tensor:
    """Per-row (or broadcast) asymmetric INT4: q = clamp(round((w-zero)/scale), 0, maxq).

    Returns dequantized values in the same shape as w.
    scale/zero broadcast over the last dim of w.
    """
    # w: [rows] or [rows, 1]; scale/zero: scalar or [rows]
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.clamp(torch.round((w - zero) * inv_scale), 0, maxq)
    return q * scale + zero


def quantize_symmetric_grid(
    w: torch.Tensor,
    scale: float,
    maxq: int = 15,
) -> torch.Tensor:
    """Simple symmetric mid-rise grid used by synthetic tests (min_val=0)."""
    s = torch.as_tensor(scale, dtype=w.dtype, device=w.device)
    return quantize_asymmetric_int4(w, s, torch.zeros_like(s), maxq=maxq)


@dataclass
class GptqResult:
    qweight: torch.Tensor  # quantized weights, original column order
    hinv_u: torch.Tensor  # upper Cholesky of H_inv in processing order
    damp: float
    perm: Optional[torch.Tensor]
    losses: torch.Tensor


def gptq_fasterquant(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    quantize_fn: Callable[[torch.Tensor, int], torch.Tensor],
    cfg: Optional[GptqConfig] = None,
    inject: Optional[str] = None,
) -> GptqResult:
    """Full GPTQ pass over one tensor (paper Algorithm 1 / DASLab fasterquant).

    weight:   [M, K] row-major (out_features x in_features) — nn.Linear layout.
    hessian:  [K, K] in the SAME basis as weight columns.
    quantize_fn(column_vec[M], col_index) -> quantized column_vec[M]

    inject (for fault-injection only; default None):
      'transpose_h'       — use H.T before solving (no-op if H symmetric; we
                            break symmetry first by scaling upper triangle).
      'no_damp'           — percdamp forced to 0 (and no dead-column fix damp).
      'no_invperm'        — actorder permute without final inverse permute.
      'no_error_feedback' — skip W[:, j:] -= err * Hinv[j, j:]
    """
    cfg = cfg or GptqConfig()
    w = weight.to(dtype=torch.float64).clone()
    h = hessian.to(dtype=torch.float64).clone()
    if w.ndim != 2:
        raise ValueError(f"weight must be [M, K], got {tuple(w.shape)}")
    m, k = w.shape
    if tuple(h.shape) != (k, k):
        raise ValueError(f"H shape {tuple(h.shape)} != ({k}, {k})")

    if inject == "transpose_h":
        # Break symmetry then transpose so the defect is observable even when
        # the input H was exactly symmetric.
        h = h.clone()
        h = h + torch.triu(torch.ones_like(h), diagonal=1) * (0.17 * h.abs().mean().clamp_min(1e-12))
        h = h.transpose(0, 1).contiguous()

    percdamp = 0.0 if inject == "no_damp" else cfg.percdamp

    perm = None
    invperm = None
    if cfg.actorder:
        perm = weight_mode_actorder_perm(h)
        w = w[:, perm].contiguous()
        h = h[perm][:, perm].contiguous()
        invperm = inverse_perm(perm)

    # Dead columns
    dead = torch.diagonal(h) == 0
    if dead.any():
        h = h.clone()
        h[dead, dead] = 1.0
        w = w.clone()
        w[:, dead] = 0.0

    damp = float(percdamp) * float(torch.mean(torch.diagonal(h)))
    h_d = h.clone()
    idx = torch.arange(k, device=h.device)
    h_d[idx, idx] += damp

    l = torch.linalg.cholesky(h_d)
    h_inv = torch.cholesky_inverse(l)
    hinv_u = torch.linalg.cholesky(h_inv, upper=True)

    losses = torch.zeros_like(w)
    q_out = torch.zeros_like(w)
    blocksize = cfg.blocksize

    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        count = i2 - i1
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        losses1 = torch.zeros_like(w1)
        hinv1 = hinv_u[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            w_col = w1[:, i]
            d = hinv1[i, i]
            q = quantize_fn(w_col, col if perm is None else int(perm[col].item()))
            q1[:, i] = q
            losses1[:, i] = (w_col - q) ** 2 / (d**2)
            err = (w_col - q) / d
            if inject != "no_error_feedback":
                # W1[:, i:] -= err[:, None] @ Hinv1[i, i:][None, :]
                w1[:, i:] = w1[:, i:] - err.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            err1[:, i] = err

        q_out[:, i1:i2] = q1
        losses[:, i1:i2] = losses1 / 2.0
        if inject != "no_error_feedback":
            w[:, i2:] = w[:, i2:] - err1 @ hinv_u[i1:i2, i2:]

    if cfg.actorder and invperm is not None and inject != "no_invperm":
        q_out = q_out[:, invperm].contiguous()
        losses = losses[:, invperm].contiguous()
        if perm is not None:
            # Return Hinv_U in original order? Keep processing-order U; document it.
            pass

    return GptqResult(
        qweight=q_out,
        hinv_u=hinv_u,
        damp=damp,
        perm=perm,
        losses=losses,
    )


def frozen_minmax_int4_quantize_fn(
    weight: torch.Tensor,
    group_size: int = 0,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Build a quantize_fn with scales/zeros frozen from the ORIGINAL weight.

    group_size <= 0 → one scale/zero per output row over all K columns
    (tensor-wide per-channel affine).
    group_size > 0  → per-row groups along K (must divide K).

    This matches the 'find_params once before the loop' behaviour of the
    official quantizer when static_groups/groupsize defaults apply, and is
    the right contract for cross-checking hipfire's frozen BlockGrid path
    (hipfire freezes per-256 flat blocks; for pure GPTQ math tests we use
    per-row affine which is the paper default).
    """
    w = weight.to(dtype=torch.float64)
    m, k = w.shape
    if group_size <= 0:
        w_min = w.min(dim=1).values
        w_max = w.max(dim=1).values
        scale = (w_max - w_min).clamp_min(0.0) / 15.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = w_min

        def qfn(col: torch.Tensor, _j: int) -> torch.Tensor:
            return quantize_asymmetric_int4(col, scale, zero)

        return qfn

    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    n_g = k // group_size
    scales = torch.empty((m, n_g), dtype=torch.float64)
    zeros = torch.empty((m, n_g), dtype=torch.float64)
    for g in range(n_g):
        sl = w[:, g * group_size : (g + 1) * group_size]
        w_min = sl.min(dim=1).values
        w_max = sl.max(dim=1).values
        sc = (w_max - w_min).clamp_min(0.0) / 15.0
        sc = torch.where(sc > 0, sc, torch.ones_like(sc))
        scales[:, g] = sc
        zeros[:, g] = w_min

    def qfn_g(col: torch.Tensor, j: int) -> torch.Tensor:
        g = j // group_size
        return quantize_asymmetric_int4(col, scales[:, g], zeros[:, g])

    return qfn_g


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """max abs, relative Frobenius, L2-norm ratio, cosine."""
    a64 = a.detach().to(dtype=torch.float64).reshape(-1)
    b64 = b.detach().to(dtype=torch.float64).reshape(-1)
    if a64.numel() != b64.numel():
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = a64 - b64
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    na = float(torch.linalg.vector_norm(a64))
    nb = float(torch.linalg.vector_norm(b64))
    nd = float(torch.linalg.vector_norm(diff))
    rel_frob = nd / max(na, 1e-30)
    norm_ratio = na / max(nb, 1e-30)
    cosine = float(torch.dot(a64, b64) / max(na * nb, 1e-30)) if (na > 0 and nb > 0) else float("nan")
    return {
        "max_abs": max_abs,
        "rel_frob": rel_frob,
        "norm_ratio": norm_ratio,
        "cosine": cosine,
    }


__all__ = [
    "GptqConfig",
    "GptqResult",
    "accumulate_hessian_xxT",
    "accumulate_hessian_official_running",
    "damp_and_inv_cholesky_upper",
    "weight_mode_actorder_perm",
    "inverse_perm",
    "quantize_asymmetric_int4",
    "gptq_fasterquant",
    "frozen_minmax_int4_quantize_fn",
    "metrics",
]
