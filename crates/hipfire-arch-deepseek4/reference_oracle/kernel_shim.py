# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Eager replacements for every `kernel.py` entry point `model.py` imports.

Transcribed from `.codeinsight+research/ds4-parent-ref/inference/kernel.py`
and kept deliberately naive: plain torch ops, no fusion, no tilelang.
Speed is irrelevant; independence from the Rust parent path is the point.

Imported symbols (model.py:12):
  act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

# OCP float4_e2m1fn LUT (convert.py / kernel packing). Low-nibble-first.
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fast_log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) via IEEE-754 bit ops — kernel.py:22-27."""
    bits = x.float().contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    return (exp - 127 + (man != 0).to(torch.int32)).to(torch.int32)


def _fast_pow2(n: torch.Tensor) -> torch.Tensor:
    """2^n for integer n via bit insert — kernel.py:30-33."""
    bits = ((n.to(torch.int32) + 127) << 23)
    return bits.view(torch.float32)


def _fast_round_scale(amax: torch.Tensor, fp_max_inv: float) -> torch.Tensor:
    return _fast_pow2(_fast_log2_ceil(amax * fp_max_inv))


def _ue8m0_to_f32(s: torch.Tensor) -> torch.Tensor:
    """float8_e8m0fnu / uint8 → f32. value = 2^(byte-127)."""
    if s.dtype == torch.float8_e8m0fnu:
        b = s.view(torch.uint8)
    elif s.dtype == torch.uint8:
        b = s
    else:
        # already float scales (non-ue8m0 path)
        return s.float()
    bb = b.to(torch.int64)
    # 2^(b-127); handle 0 and 255 specially
    out = torch.ldexp(torch.ones(bb.shape, dtype=torch.float32, device=s.device), bb.to(torch.int32) - 127)
    out = torch.where(b == 0, torch.full_like(out, 2.0 ** -127), out)
    out = torch.where(b == 255, torch.full_like(out, float("nan")), out)
    return out


def _unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    """Unpack float4_e2m1fn_x2 or uint8 [N, K//2] → f32 [N, K], low-nibble-first."""
    if b.dtype == torch.float4_e2m1fn_x2:
        raw = b.view(torch.uint8)
    else:
        raw = b.view(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    lut = _E2M1_LUT.to(device=raw.device)
    lo_f = lut[low]
    hi_f = lut[high]
    return torch.stack([lo_f, hi_f], dim=-1).flatten(-2)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise FP8 quant. inplace=True: fused quant+dequant back to BF16.

    When scale_fmt is set, scales are rounded to power-of-2 (MXFP / ue8m0).
    """
    N = x.size(-1)
    assert N % block_size == 0, (tuple(x.shape), block_size)
    # kernel in_dtype=BF16 — enter BF16 domain before amax
    z = x.contiguous()
    flat_bf = z.to(torch.bfloat16).float().reshape(-1, N)
    M = flat_bf.size(0)
    groups = N // block_size
    fp8_max, fp8_min = 448.0, -448.0
    grouped = flat_bf.view(M, groups, block_size)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s = _fast_round_scale(amax, 1.0 / fp8_max)
    else:
        s = amax * (1.0 / fp8_max)
    q = (grouped / s.unsqueeze(-1)).clamp(fp8_min, fp8_max)
    q_fp8 = q.to(torch.float8_e4m3fn)
    if inplace:
        deq = q_fp8.float() * s.unsqueeze(-1)
        y = deq.reshape(z.shape).to(torch.bfloat16)
        x.copy_(y)
        return x
    y = q_fp8.reshape(*z.shape[:-1], N)
    if scale_dtype == torch.float8_e8m0fnu:
        # s is pure power-of-two; encode exp field as UE8M0
        s_bits = s.float().contiguous().view(torch.int32)
        exp = ((s_bits >> 23) & 0xFF).to(torch.uint8)
        s_out = exp.view(torch.float8_e8m0fnu).reshape(*z.shape[:-1], groups)
    else:
        s_out = s.to(scale_dtype).reshape(*z.shape[:-1], groups)
    return y, s_out


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    """Block-wise FP4 quant. inplace=True: fused quant+dequant to BF16."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    flat_bf = z.to(torch.bfloat16).float().reshape(-1, N)
    M = flat_bf.size(0)
    groups = N // block_size
    fp4_max = 6.0
    amax_floor = 6.0 * (2.0 ** -126)
    grouped = flat_bf.view(M, groups, block_size)
    amax = grouped.abs().amax(dim=-1).clamp_min(amax_floor)
    s = _fast_round_scale(amax, 1.0 / fp4_max)
    q = (grouped / s.unsqueeze(-1)).clamp(-fp4_max, fp4_max)
    # nearest E2M1 magnitude (README notes RNE-tie judgement)
    lut = _E2M1_LUT[:8].to(device=q.device)
    abs_q = q.abs()
    idx = (abs_q.unsqueeze(-1) - lut).abs().argmin(dim=-1)
    mag = lut[idx]
    signed = torch.copysign(mag, q)
    if inplace:
        deq = signed * s.unsqueeze(-1)
        y = deq.reshape(z.shape).to(torch.bfloat16)
        x.copy_(y)
        return x
    # pack low-then-high along K
    sign_bit = (q < 0).to(torch.uint8) << 3
    nib = idx.to(torch.uint8) | sign_bit
    nib_flat = nib.reshape(M, N)
    packed = (nib_flat[:, 0::2] | (nib_flat[:, 1::2] << 4)).to(torch.uint8)
    y = packed.view(torch.float4_e2m1fn_x2).reshape(*z.shape[:-1], N // 2)
    s_bits = s.float().contiguous().view(torch.int32)
    exp = ((s_bits >> 23) & 0xFF).to(torch.uint8)
    s_out = exp.view(torch.float8_e8m0fnu).reshape(*z.shape[:-1], groups)
    return y, s_out


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-128 block FP8 scales on A and B."""
    assert a.is_contiguous() and b.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    group = 128
    assert K % group == 0
    A = a.to(torch.float32).reshape(M, K)
    B = b.to(torch.float32).reshape(N, K)
    As = _ue8m0_to_f32(a_s).reshape(M, K // group)
    Bs = _ue8m0_to_f32(b_s)
    # b_s layout: [ceil(N/128), K/128]
    if Bs.dim() == 2 and Bs.size(0) == (N + group - 1) // group:
        Bs_nk = Bs.repeat_interleave(group, dim=0)[:N].repeat_interleave(group, dim=-1)
    else:
        Bs_nk = Bs.reshape(N, K // group).repeat_interleave(group, dim=-1)
    A_deq = A * As.repeat_interleave(group, dim=-1)
    B_deq = B * Bs_nk
    C = A_deq @ B_deq.t()
    return C.to(torch.get_default_dtype()).reshape(*a.shape[:-1], N)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T. A per-128; B per-32 E8M0; B packed K//2."""
    assert a.is_contiguous() and b.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    act_group, wgt_group = 128, 32
    assert K % act_group == 0
    A = a.to(torch.float32).reshape(M, K)
    B = _unpack_fp4(b).reshape(N, K).to(device=A.device)
    As = _ue8m0_to_f32(a_s).reshape(M, K // act_group)
    Bs = _ue8m0_to_f32(b_s).reshape(N, K // wgt_group)
    A_deq = A * As.repeat_interleave(act_group, dim=-1)
    B_deq = B * Bs.repeat_interleave(wgt_group, dim=-1)
    C = A_deq @ B_deq.t()
    return C.to(torch.get_default_dtype()).reshape(*a.shape[:-1], N)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse MHA via index gather + softmax with learnable sink (kernel.py:351-366)."""
    bsz, seqlen, n_heads, d = q.shape
    qf = q.float()
    kvf = kv.float()
    sink = attn_sink.float()
    idx = topk_idxs.long()
    topk = idx.size(-1)
    valid = idx >= 0
    idx_c = idx.clamp(min=0)
    # gather kv → [B,S,T,D]
    gather_idx = idx_c.unsqueeze(-1).expand(bsz, seqlen, topk, d)
    kv_exp = kvf.unsqueeze(1).expand(bsz, seqlen, kvf.size(1), d)
    kv_g = torch.gather(kv_exp, 2, gather_idx)
    kv_g = torch.where(valid.unsqueeze(-1), kv_g, torch.zeros_like(kv_g))
    scores = torch.einsum("bshd,bstd->bsht", qf, kv_g) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink_b = sink.view(1, 1, n_heads, 1).expand(bsz, seqlen, n_heads, 1)
    m = torch.maximum(scores.amax(dim=-1, keepdim=True), sink_b)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    exp_s = torch.exp(scores - m) * valid.unsqueeze(2).to(scores.dtype)
    exp_sink = torch.exp(sink_b - m)
    attn = exp_s / (exp_s.sum(dim=-1, keepdim=True) + exp_sink)
    out = torch.einsum("bsht,bstd->bshd", attn, kv_g)
    return out.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """pre/post/comb split. Row-softmax then alternating col/row norm, ending on col.

    post = 2 * sigmoid(...)  (kernel.py:394) — NOT an env-var scale.
    """
    bsz, seqlen, _ = mixes.shape
    hc = hc_mult
    m = mixes.float()
    scale = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (
        m[..., 2 * hc : 2 * hc + hc * hc].reshape(bsz, seqlen, hc, hc) * scale[2]
        + base[2 * hc : 2 * hc + hc * hc].reshape(hc, hc)
    )
    # row-softmax + eps
    comb = torch.softmax(comb, dim=-1) + eps
    # col normalize
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)  # row
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # col
    return pre, post, comb
