#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Isolate-mixer validation (FU1/FU2): compare hipfire's exact Mamba-2 decode
# math (block.rs / mamba2_ssd_decode.hip conventions) against transformers'
# CANONICAL native Mamba2Mixer.torch_forward — the correct, maintained
# reference (uses repeat_interleave = interleave group mapping, no mamba-ssm
# needed). Confirms hipfire's mixer is correct independent of the broken
# nemotron custom torch_forward and the generation/cache path.
#
#   python3 benchmarks/nemotron/validate_mixer_vs_native.py --model <snap>

import argparse, sys, types, importlib.util, json, glob, struct
import numpy as np, torch, torch.nn.functional as F


def _stub():
    def m(n, **a):
        mod = types.ModuleType(n)
        mod.__spec__ = importlib.util.spec_from_loader(n, loader=None)
        mod.__path__ = []
        for k, v in a.items():
            setattr(mod, k, v)
        sys.modules[n] = mod

    def rn(*A, **K):
        x = A[0] if A else K["x"]
        w = A[1] if len(A) > 1 else K["weight"]
        z = K.get("z")
        eps = K.get("eps", 1e-6)
        gs = K.get("group_size")
        nbg = K.get("norm_before_gate", True)
        d = x.dtype
        x = x.float()
        if z is not None and not nbg:
            x = x * F.silu(z.float())
        if gs is None:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        else:
            s = x.shape
            xg = x.view(*s[:-1], s[-1] // gs, gs)
            x = (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)).view(s)
        return (x * w.float()).to(d)

    m("mamba_ssm", __version__="2.2.2")
    m("mamba_ssm.ops")
    m("mamba_ssm.ops.triton")
    m("mamba_ssm.ops.triton.layernorm_gated", rmsnorm_fn=rn)
    m("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
    m("mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=None, mamba_split_conv1d_scan_combined=None)


def main():
    _stub()
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--len", type=int, default=6)
    args = ap.parse_args()
    dev = "cuda"
    mdl = (
        AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32)
        .to(dev)
        .eval()
    )
    mixer = mdl.backbone.layers[0].mixer
    cfg = mdl.config
    nh, hd, ns, ng = cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size, cfg.n_groups
    d_inner, conv_dim, hidden = nh * hd, mixer.conv1d.weight.shape[0], cfg.hidden_size
    K = mixer.conv1d.weight.shape[-1]
    eps = cfg.rms_norm_eps
    L = args.len

    torch.manual_seed(0)
    x = torch.randn(1, L, hidden, device=dev) * 0.4

    # Reference: canonical native Mamba2 torch_forward (prefill, no cache).
    with torch.no_grad():
        native = Mamba2Mixer.torch_forward(
            mixer, x.clone(), cache_params=None, cache_position=None, attention_mask=None
        )
    native = native[0].float().cpu().numpy().astype(np.float64)  # [L, hidden]

    # hipfire conventions (block.rs / kernel), per-token decode in numpy.
    g = lambda k: k.detach().float().cpu().numpy().astype(np.float64)
    in_proj = g(mixer.in_proj.weight)
    cw = g(mixer.conv1d.weight)[:, 0, :]
    cb = g(mixer.conv1d.bias)
    A_log = g(mixer.A_log)
    D = g(mixer.D)
    dt_bias = g(mixer.dt_bias)
    mix_norm = g(mixer.norm.weight)
    out_proj = g(mixer.out_proj.weight)
    gs = d_inner // ng  # gated-norm group size
    xn = x[0].float().cpu().numpy().astype(np.float64)
    conv_state = np.zeros((conv_dim, K - 1))
    ssm = np.zeros((nh, hd, ns))
    my = np.zeros((L, hidden))
    for t in range(L):
        proj = in_proj @ xn[t]
        z = proj[:d_inner]
        xbc = proj[d_inner : d_inner + conv_dim]
        dt_raw = proj[d_inner + conv_dim :]
        acc = cb.copy()
        for k in range(K - 1):
            acc += conv_state[:, k] * cw[:, k]
        acc += xbc * cw[:, K - 1]
        a = acc / (1.0 + np.exp(-acc))  # silu
        conv_state[:, : K - 2] = conv_state[:, 1 : K - 1] if K > 2 else conv_state[:, :0]
        conv_state[:, K - 2] = xbc
        xx = a[:d_inner].reshape(nh, hd)
        B = a[d_inner : d_inner + ng * ns].reshape(ng, ns)
        C = a[d_inner + ng * ns :].reshape(ng, ns)
        dt = np.log1p(np.exp(dt_raw + dt_bias))  # no clamp
        Av = -np.exp(A_log)
        y = np.zeros(d_inner)
        for h in range(nh):
            grp = h // (nh // ng)  # INTERLEAVE (hipfire / native)
            dA = np.exp(dt[h] * Av[h])
            for p in range(hd):
                ssm[h, p] = dA * ssm[h, p] + dt[h] * B[grp] * xx[h, p]
                y[h * hd + p] = (C[grp] * ssm[h, p]).sum() + D[h] * xx[h, p]
        gated = y * (z / (1.0 + np.exp(-z)))
        yn = np.zeros(d_inner)
        for grp in range(d_inner // gs):
            sl = slice(grp * gs, grp * gs + gs)
            yn[sl] = gated[sl] * (1.0 / np.sqrt((gated[sl] ** 2).mean() + eps)) * mix_norm[sl]
        my[t] = out_proj @ yn

    for t in range(L):
        d = np.abs(native[t] - my[t])
        rel = d.mean() / (np.abs(native[t]).mean() + 1e-9)
        print(
            f"  pos {t}: max|Δ|={d.max():.3e} rel={rel:.5f}  (native std={native[t].std():.4f} hipfire std={my[t].std():.4f})"
        )
    worst = max(np.abs(native[t] - my[t]).mean() / (np.abs(native[t]).mean() + 1e-9) for t in range(L))
    print(
        f"\nRESULT: {'MATCH — hipfire Mamba-2 mixer == canonical native Mamba2' if worst < 0.02 else 'DIVERGES'} (worst rel {worst:.4f})"
    )


if __name__ == "__main__":
    main()
