#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# HF reference dump for the nemotron_h numeric bisect (FU2). Runs the real
# NemotronHForCausalLM on a fixed token sequence and saves per-layer hidden
# states (embeddings + each block output) and final logits to an .npz, so the
# hipfire forward can be compared layer-by-layer to pinpoint any divergence.
#
#   python3 benchmarks/nemotron/dump_hf_reference.py \
#       --model /srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/<snap> \
#       --out /tmp/nemo_hf_ref.npz
#
# Output npz keys: input_ids [T], hidden_<L> [T, H] for L in 0..=num_layers
# (0 = embeddings, k = after block k-1), logits [T, V].

import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F


def _install_mamba_ssm_stub():
    """The HF modeling hard-imports `rmsnorm_fn` from mamba-ssm (CUDA-only).
    We only need the pure-torch reference path (`torch_forward`), which runs
    when the fast-path kernels are absent. Inject a correct pure-torch
    `rmsnorm_fn` (gate-then-group-RMSNorm, the MambaRMSNormGated convention)
    into sys.modules so the import succeeds; the SSD/conv fast paths stay None
    (mamba_ssm find_spec still fails), so the model uses torch_forward."""

    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None,
                   norm_before_gate=True, **kw):
        dtype = x.dtype
        x = x.float()
        if z is not None and not norm_before_gate:
            x = x * F.silu(z.float())
        if group_size is None:
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + eps)
        else:
            shape = x.shape
            xg = x.view(*shape[:-1], shape[-1] // group_size, group_size)
            var = xg.pow(2).mean(-1, keepdim=True)
            x = (xg * torch.rsqrt(var + eps)).view(shape)
        out = x * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None and norm_before_gate:
            out = out * F.silu(z.float())
        return out.to(dtype)

    import importlib.util

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        m.__path__ = []  # mark as package so submodule imports resolve
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    mod("mamba_ssm", __version__="2.2.2")
    mod("mamba_ssm.ops")
    mod("mamba_ssm.ops.triton")
    mod("mamba_ssm.ops.triton.layernorm_gated", rmsnorm_fn=rmsnorm_fn)
    # SSD/conv fast-path kernels left as None → is_fast_path_available False →
    # the model uses the pure-torch torch_forward reference path.
    mod("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
    mod(
        "mamba_ssm.ops.triton.ssd_combined",
        mamba_chunk_scan_combined=None,
        mamba_split_conv1d_scan_combined=None,
    )


_install_mamba_ssm_stub()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/nemo_hf_ref.npz")
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--max-layers-print", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(dev)
    model.eval()

    # HF's _init_weights RE-RANDOMIZES dt_bias post-load (ungated copy_(inv_dt)
    # with torch.rand → non-deterministic). Restore the STORED (trained) dt_bias
    # from the checkpoint so the reference is deterministic and correct.
    import glob
    import struct

    sfile = sorted(glob.glob(os.path.join(args.model, "*.safetensors")))[0]
    with open(sfile, "rb") as fh:
        hn = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hn))
        hbase = 8 + hn
        n_restored = 0
        for name, p in model.named_parameters():
            if name.endswith(".dt_bias") and name in hdr:
                o0, o1 = hdr[name]["data_offsets"]
                fh.seek(hbase + o0)
                raw = fh.read(o1 - o0)
                vals = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
                vals = vals.view(np.float32)
                with torch.no_grad():
                    p.copy_(torch.from_numpy(vals.copy()).to(p.device, p.dtype))
                n_restored += 1
    print(f"restored {n_restored} stored dt_bias tensors (HF randomizes them at load)")

    ids = tok(args.text, return_tensors="pt").input_ids.to(dev)
    print("input_ids:", ids.tolist())

    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)

    hs = out.hidden_states  # tuple len num_layers+1, each [1, T, H]
    logits = out.logits  # [1, T, V]

    save = {"input_ids": ids[0].cpu().numpy().astype(np.int64)}
    for i, h in enumerate(hs):
        save[f"hidden_{i}"] = h[0].float().cpu().numpy()
    save["logits"] = logits[0].float().cpu().numpy()
    np.savez(args.out, **save)
    print(f"saved {len(hs)} hidden states + logits to {args.out}")
    print("final-pos top5:", torch.topk(logits[0, -1], 5).indices.tolist())

    if args.max_layers_print:
        for i in range(min(args.max_layers_print, len(hs))):
            h = hs[i][0, -1]
            print(f"  hidden_{i}[last]: mean={h.float().mean():.4f} std={h.float().std():.4f}")


if __name__ == "__main__":
    main()
