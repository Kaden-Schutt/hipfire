#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FU2 sub-step debug: hook the HF block-0 Mamba-2 mixer submodules (in_proj /
# conv1d / gated norm) and compare each sub-step against hipfire's block.rs
# conventions, recomputed in numpy, to pinpoint the divergent step.
#
#   python3 benchmarks/nemotron/debug_mamba_block0.py --model <snap>

import argparse
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F


def _stub():
    import importlib.util

    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, **kw):
        dtype = x.dtype
        x = x.float()
        if z is not None and not norm_before_gate:
            x = x * F.silu(z.float())
        if group_size is None:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        else:
            s = x.shape
            xg = x.view(*s[:-1], s[-1] // group_size, group_size)
            x = (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)).view(s)
        out = x * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None and norm_before_gate:
            out = out * F.silu(z.float())
        return out.to(dtype)

    def m(n, **a):
        mod = types.ModuleType(n)
        mod.__spec__ = importlib.util.spec_from_loader(n, loader=None)
        mod.__path__ = []
        for k, v in a.items():
            setattr(mod, k, v)
        sys.modules[n] = mod

    m("mamba_ssm", __version__="2.2.2")
    m("mamba_ssm.ops")
    m("mamba_ssm.ops.triton")
    m("mamba_ssm.ops.triton.layernorm_gated", rmsnorm_fn=rmsnorm_fn)
    m("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
    m("mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=None, mamba_split_conv1d_scan_combined=None)


def cmp(label, a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    rel = d.mean() / (np.abs(b).mean() + 1e-9)
    print(f"  {label:28s} max|Δ|={d.max():.3e} rel={rel:.4f}  (mine std={a.std():.4f} hf std={b.std():.4f})")


def main():
    _stub()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="The capital of France is")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32).to(
        "cuda"
    )
    model.eval()
    ids = tok(args.text, return_tensors="pt").input_ids.to("cuda")

    layer0 = model.backbone.layers[0]
    mixer = layer0.mixer
    caps = {}
    h = []
    h.append(mixer.in_proj.register_forward_hook(lambda m, i, o: caps.__setitem__("in_proj", o.detach().cpu())))
    h.append(mixer.conv1d.register_forward_hook(lambda m, i, o: caps.__setitem__("conv1d", o.detach().cpu())))
    h.append(
        mixer.norm.register_forward_hook(
            lambda m, i, o: (
                caps.__setitem__("norm_in", i[0].detach().cpu())
                or caps.__setitem__("norm_gate", i[1].detach().cpu())
                or caps.__setitem__("norm_out", o.detach().cpu())
            )
        )
    )
    h.append(layer0.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("block_norm_out", o.detach().cpu())))
    h.append(mixer.norm.register_forward_hook(lambda m, i, o: None))  # placeholder (norm hooked above)
    h.append(mixer.out_proj.register_forward_hook(lambda m, i, o: caps.__setitem__("mixer_out", o.detach().cpu())))
    with torch.no_grad():
        model(ids, use_cache=False)
    for hk in h:
        hk.remove()

    cfg = model.config
    nh, hd, ns, ng = cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size, cfg.n_groups
    d_inner = nh * hd
    eps = cfg.rms_norm_eps
    conv_dim = mixer.conv1d.weight.shape[0]
    K = mixer.conv1d.weight.shape[-1]
    proj = caps["in_proj"][0, 0].numpy().astype(np.float64)  # pos 0
    print(
        "in_proj.shape[0] =",
        mixer.in_proj.weight.shape[0],
        "expected(d_mlp=0) =",
        d_inner + conv_dim + nh,
        "=> d_mlp =",
        (mixer.in_proj.weight.shape[0] - d_inner - conv_dim - nh) // 2,
    )

    # My split [z | xBC | dt] at offset 0:
    z = proj[:d_inner]
    xBC = proj[d_inner : d_inner + conv_dim]
    dt_raw = proj[d_inner + conv_dim :]
    # HF gate / conv input come from the hook directly:
    # conv1d output is [1, conv_dim, seq]; pos 0 = [:,0]; pre-act.
    hf_conv = caps["conv1d"][0, :, 0].numpy().astype(np.float64)
    my_conv = mixer.conv1d.weight[:, 0, K - 1].detach().cpu().numpy().astype(
        np.float64
    ) * xBC + mixer.conv1d.bias.detach().cpu().numpy().astype(np.float64)
    cmp("conv_out (pre-act)", my_conv, hf_conv)

    hf_norm_in = caps["norm_in"][0, 0].numpy().astype(np.float64)  # SSD output y
    hf_norm_gate = caps["norm_gate"][0, 0].numpy().astype(np.float64)
    cmp("gate z (=norm gate in)", z, hf_norm_gate)

    # my SSD at pos 0 (zero state), with the two group mappings:
    xBC_act = my_conv / (1.0 + np.exp(-my_conv))  # silu (use my conv; if conv matches, fine)
    # but use HF's post-act xBC for an apples-to-apples SSD check:
    hf_xBC_act = caps["conv1d"][0, :, 0]
    hf_xBC_act = (hf_xBC_act * torch.sigmoid(hf_xBC_act)).numpy().astype(np.float64)
    x = hf_xBC_act[:d_inner]
    B = hf_xBC_act[d_inner : d_inner + ng * ns].reshape(ng, ns)
    C = hf_xBC_act[d_inner + ng * ns :].reshape(ng, ns)
    A_log = mixer.A_log.detach().cpu().numpy().astype(np.float64)
    D = mixer.D.detach().cpu().numpy().astype(np.float64)
    dt_bias = mixer.dt_bias.detach().cpu().numpy().astype(np.float64)
    dt = np.log1p(np.exp(dt_raw + dt_bias))  # NO clamp (time_step_limit=(0,inf))
    xr = x.reshape(nh, hd)
    for label, gm in [("ssd h//12", lambda hh: hh // (nh // ng)), ("ssd h%8", lambda hh: hh % ng)]:
        y = np.zeros(d_inner)
        for head in range(nh):
            grp = gm(head)
            for p in range(hd):
                hstate = dt[head] * B[grp] * xr[head, p]
                y[head * hd + p] = (C[grp] * hstate).sum() + D[head] * xr[head, p]
        cmp(label + " vs HF y", y, hf_norm_in)

    # gated norm: my gate-then-group-RMSNorm of HF's y + HF's gate vs HF norm_out
    mix_norm = mixer.norm.weight.detach().cpu().numpy().astype(np.float64)
    gs = mixer.norm.group_size
    print("  gated-norm group_size =", gs, " (d_inner/n_groups =", d_inner // ng, ")")
    gate = hf_norm_gate
    gated = hf_norm_in * (gate / (1.0 + np.exp(-gate)))
    yn = np.zeros(d_inner)
    for grp in range(d_inner // gs):
        sl = slice(grp * gs, grp * gs + gs)
        inv = 1.0 / np.sqrt((gated[sl] ** 2).mean() + eps)
        yn[sl] = gated[sl] * inv * mix_norm[sl]
    cmp("gated_norm vs HF norm_out", yn, caps["norm_out"][0, 0].numpy())

    # out_proj of HF norm_out vs HF mixer_out
    out_proj = mixer.out_proj.weight.detach().cpu().numpy().astype(np.float64)
    my_out = out_proj @ caps["norm_out"][0, 0].numpy().astype(np.float64)
    cmp("out_proj vs HF mixer_out", my_out, caps["mixer_out"][0, 0].numpy())

    # FULL block from embeddings (my rmsnorm + in_proj), all fixes, vs HF mixer_out
    ref = np.load("/tmp/nemo_hf_ref.npz")
    emb = ref["hidden_0"][0].astype(np.float64)
    block_norm = layer0.norm.weight.detach().cpu().numpy().astype(np.float64)
    hn = emb * (1.0 / np.sqrt((emb * emb).mean() + eps)) * block_norm
    in_proj_w = mixer.in_proj.weight.detach().cpu().numpy().astype(np.float64)
    cmp("my rmsnorm vs HF block_norm_out", hn, caps["block_norm_out"][0, 0].numpy())
    myproj = in_proj_w @ hn
    cmp("my in_proj vs HF in_proj", myproj, caps["in_proj"][0, 0].numpy())

    # FULL CHAIN from embeddings (block.rs path), vs HF mixer_out:
    mz = myproj[:d_inner]
    mxBC = myproj[d_inner : d_inner + conv_dim]
    mdt_raw = myproj[d_inner + conv_dim :]
    cw = mixer.conv1d.weight[:, 0, K - 1].detach().cpu().numpy().astype(np.float64)
    cb = mixer.conv1d.bias.detach().cpu().numpy().astype(np.float64)
    mconv = cw * mxBC + cb
    mact = mconv / (1.0 + np.exp(-mconv))
    mx = mact[:d_inner]
    mB = mact[d_inner : d_inner + ng * ns].reshape(ng, ns)
    mC = mact[d_inner + ng * ns :].reshape(ng, ns)
    mdt = np.log1p(np.exp(mdt_raw + dt_bias))
    xr = mx.reshape(nh, hd)
    for label, gm in [("FULL h%8", lambda hh: hh % ng), ("FULL h//12", lambda hh: hh // (nh // ng))]:
        y = np.zeros(d_inner)
        for head in range(nh):
            grp = gm(head)
            for p in range(hd):
                y[head * hd + p] = (mC[grp] * (mdt[head] * mB[grp] * xr[head, p])).sum() + D[head] * xr[head, p]
        gated = y * (mz / (1.0 + np.exp(-mz)))
        yn = np.zeros(d_inner)
        for grp in range(d_inner // gs):
            sl = slice(grp * gs, grp * gs + gs)
            yn[sl] = gated[sl] * (1.0 / np.sqrt((gated[sl] ** 2).mean() + eps)) * mix_norm[sl]
        mout = out_proj @ yn
        cmp(label + " block vs HF", mout, caps["mixer_out"][0, 0].numpy())


if __name__ == "__main__":
    main()
