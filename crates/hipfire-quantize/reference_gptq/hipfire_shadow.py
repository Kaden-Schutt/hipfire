# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
#
# Shadow of hipfire GPTQ *conventions* for staged A/B against the paper
# reference. This is intentionally a SEPARATE module from gptq_paper_ref.py.
#
# WARNING: this file encodes what hipfire COMMITS TO (see CONVENTIONS.md),
# re-derived from the documented behaviour of:
#   - scripts/collect_hessian.py HessianAccumulator
#   - crates/hipfire-quantize/src/gptq.rs (compute_damped_inv_cholesky_upper,
#     gptq_column_sequential, weight_mode_actorder)
#   - crates/hipfire-quantize/src/e8_gptq.rs (block_feedback, LAMBDA=0.01)
#   - E8H1 writers (raw sum, no /N)
#
# It is NOT the oracle. The oracle is gptq_paper_ref.py. When this shadow
# and the paper ref disagree, that is a FINDING about hipfire — do not
# "fix" the paper ref to match this file.
#
# The inverse-Cholesky construction here follows the SAME mathematical
# invariant as the paper (U^T U = H_inv) because hipfire was corrected to
# that form on 2026-05-14; the remaining convention deltas are Hessian
# scaling, damping absolute-vs-fractional call-site wiring, actorder
# storage layout, and block-diagonal-256 vs full-K.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from gptq_paper_ref import metrics  # shared metric helper only


# ---------------------------------------------------------------------------
# Hessian accumulation — hipfire HFHS path
# ---------------------------------------------------------------------------

def hipfire_hfhs_accumulate(activations: torch.Tensor) -> torch.Tensor:
    """scripts/collect_hessian.py: H += x.T @ x; finalize H/n_tokens.

    No factor of 2. FP32 accumulation in production; we use f64 here for the
    cross-check floor measurement.
    """
    x = activations.to(dtype=torch.float64)
    n = x.shape[0]
    h = x.transpose(0, 1) @ x
    if n > 0:
        h = h / float(n)
    return 0.5 * (h + h.transpose(0, 1))


def hipfire_e8h1_accumulate(activations: torch.Tensor) -> list[torch.Tensor]:
    """E8H1 path: per-256-block raw sum x_b x_b^T, NO /N, NO factor 2."""
    x = activations.to(dtype=torch.float64)
    n, k = x.shape
    assert k % 256 == 0
    blocks = []
    for b in range(k // 256):
        xb = x[:, b * 256 : (b + 1) * 256]
        hb = xb.transpose(0, 1) @ xb
        blocks.append(0.5 * (hb + hb.transpose(0, 1)))
    return blocks


# ---------------------------------------------------------------------------
# Damping + inverse Cholesky — hipfire gptq.rs form
# ---------------------------------------------------------------------------

def hipfire_clamped_initial_damp(initial_damp: float, diag_mean: float) -> float:
    """gptq.rs clamped_initial_damp: max(initial_damp, eps * max(diag_mean, 1))."""
    return max(float(initial_damp), torch.finfo(torch.float64).eps * max(diag_mean, 1.0))


def hipfire_damped_inv_cholesky_upper(
    h: torch.Tensor,
    *,
    initial_damp: float,
    max_damp_multiplier: float = 1.0,
    perm: Optional[torch.Tensor] = None,
    damp_is_absolute: bool = True,
) -> tuple[torch.Tensor, float]:
    """Mirror of compute_damped_inv_cholesky_upper.

    Hipfire call sites pass `initial_damp` in two different ways:
      - gptq_column_sequential tests often pass 1e-6 or 0.0 as an ABSOLUTE
        addend (then clamped_initial_damp floors it).
      - e8_gptq block_feedback uses LAMBDA * mean(diag) with LAMBDA=0.01,
        i.e. the paper fractional form.

    Set damp_is_absolute=True to match gptq.rs (initial_damp added as-is after
    clamp). Set False to treat initial_damp as a fraction of mean(diag) like
    the paper / e8_gptq.

    Adaptive schedule: damp *= 10 until success or damp >= max_damp_multiplier
    * diag_mean.
    """
    h0 = h.to(dtype=torch.float64)
    k = h0.shape[0]
    if perm is not None:
        h_eff = h0[perm][:, perm].contiguous()
    else:
        h_eff = h0.clone()

    diag_mean = float(torch.diagonal(h_eff).mean())
    if damp_is_absolute:
        damp = hipfire_clamped_initial_damp(initial_damp, diag_mean)
    else:
        damp = hipfire_clamped_initial_damp(initial_damp * diag_mean, diag_mean)
    damp_cap = max_damp_multiplier * diag_mean if diag_mean > 0 else max_damp_multiplier

    eye = torch.eye(k, dtype=torch.float64, device=h_eff.device)
    last_err: Optional[Exception] = None
    while True:
        try:
            hd = h_eff + damp * eye
            l = torch.linalg.cholesky(hd)  # lower
            # Invert L by forward-sub columns (match gptq.rs structure).
            l_inv = torch.zeros_like(l)
            for j in range(k):
                # solve L x = e_j
                e = torch.zeros(k, dtype=torch.float64, device=h_eff.device)
                e[j] = 1.0
                l_inv[:, j] = torch.linalg.solve_triangular(l, e.unsqueeze(1), upper=False).squeeze(1)
            h_inv = l_inv.transpose(0, 1) @ l_inv
            # Second Cholesky lower on H_inv, then U = L_HI^T.
            l_hi = torch.linalg.cholesky(h_inv)
            u = l_hi.transpose(0, 1).contiguous()  # upper, U^T U = H_inv
            return u, damp
        except Exception as e:  # noqa: BLE001 — mirror adaptive cascade
            last_err = e
            if damp >= damp_cap and damp_cap > 0:
                raise RuntimeError(
                    f"hipfire_shadow: Cholesky failed at damp={damp} cap={damp_cap}: {last_err}"
                ) from e
            if damp == 0.0:
                damp = max(torch.finfo(torch.float64).eps * max(diag_mean, 1.0), 1e-12)
            else:
                damp = min(damp * 10.0, damp_cap if damp_cap > 0 else damp * 10.0)


def hipfire_weight_mode_actorder(h: torch.Tensor) -> torch.Tensor:
    """Descending diag(H), stable — weight_mode_actorder."""
    diag = torch.diagonal(h.to(dtype=torch.float64))
    return torch.argsort(diag, descending=True, stable=True)


@dataclass
class HipfireGptqResult:
    qweight: torch.Tensor
    hinv_u: torch.Tensor
    damp: float
    perm: torch.Tensor


def hipfire_gptq_column_sequential(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    quantize_fn: Callable[[torch.Tensor, int], torch.Tensor],
    initial_damp: float = 0.01,
    max_damp_multiplier: float = 1.0,
    damp_is_absolute: bool = False,
    actorder: bool = True,
) -> HipfireGptqResult:
    """Mirror gptq_column_sequential storage convention.

    Key layout difference vs paper/DASLab:
      - Hipfire does NOT permute the weight storage for the loop.
      - perm[step] = original column processed at `step`.
      - U is computed on P^T H P (processing order).
      - err uses U[step,step]; residual update uses U[step, next] into
        original-index residual columns.
      - Output columns stay in original order (no final invperm needed).

    damp_is_absolute=False treats initial_damp as fraction of mean(diag),
    matching e8_gptq LAMBDA and the paper. gptq.rs itself takes an absolute
    damp and relies on the caller to pre-multiply; production call sites vary.
    """
    w = weight.to(dtype=torch.float64).clone()
    h = hessian.to(dtype=torch.float64).clone()
    m, k = w.shape
    assert h.shape == (k, k)

    if actorder:
        perm = hipfire_weight_mode_actorder(h)
    else:
        perm = torch.arange(k, dtype=torch.long)

    u, damp = hipfire_damped_inv_cholesky_upper(
        h,
        initial_damp=initial_damp,
        max_damp_multiplier=max_damp_multiplier,
        perm=perm,
        damp_is_absolute=damp_is_absolute,
    )

    residual = w.clone()
    q_out = torch.zeros_like(w)

    for step in range(k):
        j_orig = int(perm[step].item())
        u_ss = u[step, step]
        if float(u_ss) <= 0.0:
            continue
        w_col = residual[:, j_orig]
        q = quantize_fn(w_col, j_orig)
        q_out[:, j_orig] = q
        err = (w_col - q) / u_ss
        # OBS into remaining original columns via U[step, next_step]
        if step + 1 < k:
            # residual[:, perm[step+1:]] -= err[:, None] * U[step, step+1:]
            rest_perm = perm[step + 1 :]
            u_row = u[step, step + 1 :]
            residual[:, rest_perm] = residual[:, rest_perm] - err.unsqueeze(1) * u_row.unsqueeze(0)

    return HipfireGptqResult(qweight=q_out, hinv_u=u, damp=damp, perm=perm)


__all__ = [
    "hipfire_hfhs_accumulate",
    "hipfire_e8h1_accumulate",
    "hipfire_damped_inv_cholesky_upper",
    "hipfire_weight_mode_actorder",
    "hipfire_gptq_column_sequential",
    "HipfireGptqResult",
    "metrics",
]
