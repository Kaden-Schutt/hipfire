# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Pure-Python stand-in for parent::hc::parent_hc_post (post-fix orientation).

Does NOT import parent Rust code. Implements the same formula documented in
`src/parent/hc.rs` after the comb-axis fix, so this oracle can prove:
  1. agreement with model.py Block.hc_post on identical inputs
  2. loud disagreement when comb is deliberately transposed

Formula (reference orientation, model.py:692):
  out[r,B,d] = post[r,B] * x[r,d] + sum_A comb[r,A,B] * residual[r,A,d]
"""
from __future__ import annotations
import torch

def parent_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    transpose_comb: bool = False,
) -> torch.Tensor:
    """Match model.py hc_post; if transpose_comb, contract the wrong axis."""
    # x: [B,S,D] or [rows,D]; residual [...,hc,D]; post [...,hc]; comb [...,hc,hc]
    if transpose_comb:
        # Deliberate bug: swap the two hc axes before the reference contraction.
        comb_use = comb.transpose(-1, -2).contiguous()
    else:
        comb_use = comb
    # identical composition to model.py:690-693
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb_use.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
    )
    # dim=-3 is the first hc axis of comb after unsqueeze(-1) makes comb [...,A,B,1]
    # and residual.unsqueeze(-2) is [...,A,1,D] → product [...,A,B,D], sum A.
    # Wait: residual.unsqueeze(-2) on [...,hc,D] → [...,hc,1,D]
    # comb.unsqueeze(-1) on [...,A,B] → [...,A,B,1]
    # product broadcasts to [...,A,B,D]; sum over dim corresponding to A.
    #
    # model.py uses dim=2 assuming rank-4 [b,s,hc,hc] layout after unsqueezes
    # on [b,s,hc,d] residual. For general rank use the axis of A explicitly:
    return y.type_as(x)


def parent_hc_post_explicit(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    transpose_comb: bool = False,
) -> torch.Tensor:
    """Loop form matching hc_post_ref / parent_hc_post docs (clearer axis)."""
    # Normalize to [rows, ...] 
    assert x.shape[-1] == residual.shape[-1]
    d = x.shape[-1]
    hc = post.shape[-1]
    assert residual.shape[-2] == hc
    assert comb.shape[-2:] == (hc, hc)
    prefix = x.shape[:-1]
    rows = 1
    for s in prefix:
        rows *= s
    x_ = x.reshape(rows, d).float()
    res = residual.reshape(rows, hc, d).float()
    p = post.reshape(rows, hc).float()
    c = comb.reshape(rows, hc, hc).float()
    if transpose_comb:
        c = c.transpose(-1, -2)
    out = torch.empty(rows, hc, d, dtype=torch.float32, device=x.device)
    for r in range(rows):
        for b in range(hc):
            # out[B,d] = post[B]*x[d] + sum_A comb[A,B]*res[A,d]
            out[r, b] = p[r, b] * x_[r] + torch.einsum("a,ad->d", c[r, :, b], res[r])
    return out.reshape(*prefix, hc, d).type_as(x)
