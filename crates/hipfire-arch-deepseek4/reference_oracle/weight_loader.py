# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Load HF parent safetensors into an unmodified model.py Transformer.

Checkpoint keys already match convert.py naming (layers.N.*, embed.weight, …).
Experts stay packed float4_e2m1fn_x2; dense FP8 keeps e4m3 + ue8m0 scale on
`weight.scale`. `wo_a` is dequantized to BF16 on load (convert.py:123-127),
because Attention.forward einsums it in BF16.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open


def build_tensor_index(model_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for st in sorted(model_dir.glob("*.safetensors")):
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k in hdr:
            if k == "__metadata__":
                continue
            idx[k] = st
    return idx


def _load_tensor(index: Dict[str, Path], name: str, device="cpu") -> torch.Tensor:
    path = index[name]
    with safe_open(path, framework="pt", device=str(device)) as f:
        return f.get_tensor(name)


def _ue8m0_to_f32(s: torch.Tensor) -> torch.Tensor:
    b = s.view(torch.uint8) if s.dtype == torch.float8_e8m0fnu else s.to(torch.uint8)
    bb = b.to(torch.int64)
    out = torch.ldexp(torch.ones(bb.shape, dtype=torch.float32), bb.to(torch.int32) - 127)
    out = torch.where(b == 0, torch.full_like(out, 2.0 ** -127), out)
    return out


def dequant_fp8_block128(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 [M,K] + UE8M0 [ceil(M/128), ceil(K/128)] → bf16/f32."""
    w = weight.to(torch.float32)
    m, k = w.shape
    s = _ue8m0_to_f32(scale)
    bm, bk = s.shape
    block_m, block_k = (m + bm - 1) // bm, (k + bk - 1) // bk
    # typically 128x128
    out = torch.empty_like(w)
    for i in range(bm):
        for j in range(bk):
            rs, re = i * block_m, min((i + 1) * block_m, m)
            cs, ce = j * block_k, min((j + 1) * block_k, k)
            out[rs:re, cs:ce] = w[rs:re, cs:ce] * s[i, j]
    return out


def load_state_into_model(
    model: torch.nn.Module,
    model_dir: Path,
    *,
    layers: Optional[Iterable[int]] = None,
    load_experts: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> List[str]:
    """Copy checkpoint tensors into `model` in place. Returns list of loaded names."""
    index = build_tensor_index(model_dir)
    layer_set = set(layers) if layers is not None else None
    loaded: List[str] = []

    def want(name: str) -> bool:
        if name.startswith("mtp."):
            return False
        if layer_set is not None and name.startswith("layers."):
            # layers.N....
            rest = name[len("layers."):]
            lid = int(rest.split(".", 1)[0])
            if lid not in layer_set:
                return False
            if (not load_experts) and ".ffn.experts." in name:
                return False
        return name in index

    # Walk named parameters + buffers that exist on the module tree.
    # We assign by matching convert/HF names to module attributes.
    # Strategy: materialize a flat state dict of what model expects, then load.

    # 1) Collect module parameter destinations via state_dict keys
    sd = model.state_dict()
    # model state keys look like layers.0.attn.wq_a.weight, etc.

    assigned = {}
    missing = []
    for key in sd.keys():
        if key.startswith("mtp."):
            continue
        if layer_set is not None and key.startswith("layers."):
            lid = int(key.split(".")[1])
            if lid not in layer_set:
                continue
            if (not load_experts) and ".ffn.experts." in key:
                continue

        src_name = key
        # gate tid2eid is a parameter named tid2eid
        if src_name not in index:
            # try without trailing quirks
            missing.append(src_name)
            continue

        tensor = _load_tensor(index, src_name, device="cpu")

        # dtype fixes
        if src_name.endswith("wo_a.weight") and tensor.dtype == torch.float8_e4m3fn:
            scale = _load_tensor(index, src_name.replace(".weight", ".scale"), device="cpu")
            tensor = dequant_fp8_block128(tensor, scale).to(torch.bfloat16)
            if verbose:
                print(f"  dequant wo_a → bf16 {tuple(tensor.shape)}")
        elif tensor.dtype == torch.int8 and ".ffn.experts." in src_name:
            # packed E2M1 → float4_e2m1fn_x2 view
            tensor = tensor.view(torch.float4_e2m1fn_x2)
        elif tensor.dtype == torch.int64 and src_name.endswith("tid2eid"):
            tensor = tensor.to(torch.int32)  # model Parameter is int32

        # shape / dtype must match destination buffer
        dst = sd[key]
        if tuple(tensor.shape) != tuple(dst.shape):
            # expert fp4: dst is [out, in//2] float4; ok
            if not (tensor.dtype == torch.float4_e2m1fn_x2 and tuple(tensor.shape) == tuple(dst.shape)):
                raise RuntimeError(
                    f"shape mismatch {src_name}: ckpt {tuple(tensor.shape)} vs model {tuple(dst.shape)} dtype {tensor.dtype} vs {dst.dtype}"
                )
        assigned[key] = tensor.to(dst.dtype) if tensor.dtype != dst.dtype and dst.dtype != torch.float4_e2m1fn_x2 else tensor
        loaded.append(src_name)

    # load_state_dict strict=False for skipped layers
    incompat = model.load_state_dict(assigned, strict=False)
    if verbose:
        print(f"assigned {len(assigned)} tensors; missing_keys={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")

    # Attach .scale onto Linear weights for fp8/fp4 parameters we kept quantized
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        w = module.weight
        if w is None:
            continue
        scale_key = f"{name}.scale" if name else "scale"
        # module name like layers.0.attn.wq_a → scale layers.0.attn.wq_a.scale
        if scale_key not in index:
            continue
        if w.dtype not in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
            # wo_a dequantized — skip
            continue
        if layer_set is not None and scale_key.startswith("layers."):
            lid = int(scale_key.split(".")[1])
            if lid not in layer_set:
                continue
        scale = _load_tensor(index, scale_key, device="cpu")
        if scale.dtype == torch.float8_e8m0fnu or scale.element_size() == 1:
            # keep as float8_e8m0fnu
            if scale.dtype != torch.float8_e8m0fnu:
                scale = scale.view(torch.float8_e8m0fnu)
        # register as attribute on weight and module (model.py Linear sets both)
        w.scale = scale
        if hasattr(module, "scale"):
            try:
                module.scale = torch.nn.Parameter(scale, requires_grad=False)
            except Exception:
                module.scale = scale
        loaded.append(scale_key)

    # Move to device
    model.to(device)
    return loaded
