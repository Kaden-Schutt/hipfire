#!/usr/bin/env python3
"""Masked flat-MQ4 calibration for Qwen3.5 experiments.

This is a PoC for the "Paro/QAT-style quality at flat MQ4 runtime cost" lane.
It keeps the output HFQ structurally identical to the base MQ4 file and only
re-packs selected MQ4 tensors. The main calibration signal is the diagonal of
the *rotated* activation covariance, which matches MQ4's FWHT-at-runtime dot
product better than weighting original input columns.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


SCHEMA_MASK = "hipfire.astrea.mq4_masked.mask.v0"
SCHEMA_STATS = "hipfire.astrea.mq4_masked.stats.v0"
SCHEMA_CANDIDATE = "hipfire.astrea.mq4_masked.candidate.v0"
SCHEMA_TENSOR_SWEEP = "hipfire.astrea.mq4_masked.tensor_sweep.v0"
SCHEMA_COMPOSE = "hipfire.astrea.mq4_masked.compose.v0"
SCHEMA_ITERATE = "hipfire.astrea.mq4_masked.iterate.v0"


SCRIPT_DIR = Path(__file__).resolve().parent


def load_astrea():
    spec = importlib.util.spec_from_file_location("hipfire_astrea_for_mq4_masked", SCRIPT_DIR / "astrea.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import scripts/astrea.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ASTREA = load_astrea()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hfq_to_module_name(hfq_name: str) -> str:
    name = hfq_name
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return name


def safe_key(name: str) -> str:
    return name.replace("/", "__slash__").replace(".", "__dot__")


def unsafekey(name: str) -> str:
    return name.replace("__dot__", ".").replace("__slash__", "/")


def tensor_numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def extract_mask(args):
    _, base_tensors = ASTREA.read_hfq_index(args.base, max_tensors=0)
    _, target_tensors = ASTREA.read_hfq_index(args.target, max_tensors=0)
    rows = []
    by_transition = {}
    for name, base in base_tensors.items():
        target = target_tensors.get(name)
        if target is None:
            continue
        changed = (
            base["quant_type"] != target["quant_type"]
            or base["group_size"] != target["group_size"]
            or base["data_size"] != target["data_size"]
        )
        if not changed or base["quant_type_name"] != "MQ4G256":
            continue
        shape = [int(x) for x in base["shape"]]
        numel = tensor_numel(shape)
        packable_flat_mq4 = numel % 256 == 0 and int(base["data_size"]) == (numel // 256) * 136
        item = {
            "hfq_name": name,
            "module_name": hfq_to_module_name(name),
            "base_quant_type": base["quant_type_name"],
            "target_quant_type": target["quant_type_name"],
            "base_data_size": int(base["data_size"]),
            "target_data_size": int(target["data_size"]),
            "shape": shape,
            "numel": numel,
            "packable_flat_mq4": packable_flat_mq4,
            "kind": "linear_2d" if len(shape) == 2 and shape[1] % 256 == 0 else "flat",
        }
        rows.append(item)
        key = f"{base['quant_type_name']}->{target['quant_type_name']}"
        by_transition[key] = by_transition.get(key, 0) + 1

    result = {
        "schema": SCHEMA_MASK,
        "captured_at_utc": utc_now(),
        "base": str(Path(args.base).expanduser()),
        "target": str(Path(args.target).expanduser()),
        "selected_count": len(rows),
        "packable_flat_mq4_count": sum(1 for row in rows if row["packable_flat_mq4"]),
        "by_transition": by_transition,
        "tensors": rows,
    }
    write_json(args.out, result, pretty=args.pretty)


def write_json(path, payload, *, pretty=True):
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=pretty)
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")
    else:
        print(text)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_stats_counts(stats_npz_path: str | None, stats_json_path: str | None = None) -> dict[str, int]:
    if stats_json_path:
        path = Path(stats_json_path)
    elif stats_npz_path:
        path = Path(stats_npz_path).resolve().parent / "stats.json"
    else:
        return {}
    if not path.exists():
        return {}
    data = read_json(path)
    counts = data.get("counts") or {}
    return {str(name): int(count) for name, count in counts.items()}


def awq_hessian_sidecar_name(tensor_name: str) -> str:
    stem = tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") else tensor_name
    return f"{stem}.awq_scale.weight"


def apply_awq_hessian_transform(hessian, scales):
    import numpy as np

    h = np.asarray(hessian)
    s = np.asarray(scales, dtype=np.float32)
    if h.ndim != 3 or h.shape[1] != h.shape[2]:
        raise ValueError(f"AWQ Hessian transform expects [G,K,K], got {h.shape}")
    if s.ndim != 1:
        raise ValueError(f"AWQ scale vector must be 1D, got {s.shape}")
    n_groups, group_k, _ = h.shape
    expected_k = n_groups * group_k
    if s.shape[0] != expected_k:
        raise ValueError(f"AWQ scale length mismatch: {s.shape[0]} vs Hessian K={expected_k}")
    if np.array_equal(s, np.ones_like(s)):
        return h.copy()
    if not np.all(np.isfinite(s) & (s > 0.0)):
        raise ValueError("AWQ Hessian scales must be finite and positive")
    work_dtype = h.dtype if np.issubdtype(h.dtype, np.floating) else np.float32
    grouped = s.reshape(n_groups, group_k).astype(work_dtype, copy=False)
    denom = grouped[:, :, None] * grouped[:, None, :]
    return (h.astype(work_dtype, copy=False) / denom).astype(work_dtype, copy=False)


def apply_awq_hessian_transform_torch(hessian, scales, *, device=None):
    torch = require_torch_for_gptq()

    if device is None and hasattr(hessian, "device"):
        device = hessian.device
    if device is None:
        device = torch.device("cpu")
    h = torch.as_tensor(hessian, dtype=torch.float32, device=device)
    s = torch.as_tensor(scales, dtype=torch.float32, device=device)
    if h.dim() != 3 or h.shape[1] != h.shape[2]:
        raise ValueError(f"AWQ Hessian transform expects [G,K,K], got {tuple(h.shape)}")
    if s.dim() != 1:
        raise ValueError(f"AWQ scale vector must be 1D, got {tuple(s.shape)}")
    n_groups, group_k, _ = h.shape
    expected_k = n_groups * group_k
    if int(s.shape[0]) != expected_k:
        raise ValueError(f"AWQ scale length mismatch: {int(s.shape[0])} vs Hessian K={expected_k}")
    if bool(torch.all(s == 1.0).detach().cpu()):
        return h.clone()
    if not bool(torch.all(torch.isfinite(s) & (s > 0.0)).detach().cpu()):
        raise ValueError("AWQ Hessian scales must be finite and positive")
    grouped = s.reshape(n_groups, group_k)
    return h / (grouped[:, :, None] * grouped[:, None, :])


def read_awq_hessian_scales(path) -> dict[str, object]:
    import numpy as np

    _, tensors = ASTREA.read_hfq_index(path, max_tensors=0)
    scales = {}
    for name, info in tensors.items():
        if not name.endswith(".awq_scale.weight"):
            continue
        if int(info["quant_type"]) != 1 or len(info["shape"]) != 1:
            continue
        payload = read_hfq_payload(path, info)
        expected_bytes = int(info["shape"][0]) * 2
        if len(payload) != expected_bytes:
            continue
        scales[name] = np.frombuffer(payload, dtype="<f2").astype(np.float32)
    return scales


def compute_awq_scales(in_sum2, alpha: float):
    """Python twin of hipfire-quantize's paper-formula AWQ scale helper."""
    import numpy as np

    values = np.asarray(in_sum2, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("empty imatrix vector")
    half_alpha = float(alpha) * 0.5
    log_s = half_alpha * np.log(np.maximum(values, 1.0e-12))
    log_s -= np.mean(log_s)
    return np.exp(log_s).astype(np.float32)


def compute_awq_scales_autoawq(in_sum2, w_rms, alpha: float):
    """Python twin of hipfire-quantize's AutoAWQ scale helper."""
    import numpy as np

    values = np.asarray(in_sum2, dtype=np.float64).reshape(-1)
    weights = np.asarray(w_rms, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("empty imatrix vector")
    if weights.shape != values.shape:
        raise ValueError(f"w_rms length {weights.size} does not match imatrix length {values.size}")
    alpha64 = float(alpha)
    log_s = (
        (alpha64 * 0.5) * np.log(np.maximum(values, 1.0e-12))
        + (1.0 - alpha64) * np.log(np.maximum(weights, 1.0e-12))
    )
    log_s -= np.mean(log_s)
    return np.exp(log_s).astype(np.float32)


def compute_awq_scales_from_hessian(hessian, alpha: float):
    import numpy as np

    h = np.asarray(hessian)
    if h.ndim == 3:
        if h.shape[1] != h.shape[2]:
            raise ValueError(f"hessian groups must be square, got {h.shape}")
        diag = np.diagonal(h, axis1=1, axis2=2).reshape(-1)
    elif h.ndim == 1:
        diag = h.reshape(-1)
    else:
        raise ValueError(f"AWQ scale computation expects [G,K,K] or [K], got {h.shape}")
    return compute_awq_scales(diag, alpha)


def relative_l2_delta(prev_scales: dict[str, object] | None, scales: dict[str, object]) -> float:
    import numpy as np

    if not prev_scales:
        return 0.0
    numer = 0.0
    denom = 0.0
    for name, current in scales.items():
        if name not in prev_scales:
            continue
        cur = np.asarray(current, dtype=np.float64).reshape(-1)
        prev = np.asarray(prev_scales[name], dtype=np.float64).reshape(-1)
        if cur.shape != prev.shape:
            raise ValueError(f"scale shape mismatch for {name}: {cur.shape} vs {prev.shape}")
        diff = cur - prev
        numer += float(diff @ diff)
        denom += float(prev @ prev)
    if denom <= 0.0:
        return 0.0
    return float((numer / denom) ** 0.5)


def load_round_hessians(stats_npz_path: str, selected: list[dict[str, object]]) -> dict[str, object]:
    import numpy as np

    data = np.load(stats_npz_path)
    hessians = {}
    for item in selected:
        name = item["hfq_name"]
        key = safe_key(name)
        if key not in data.files:
            raise ValueError(f"missing imatrix stats for {name} in {stats_npz_path}")
        hessians[name] = data[key].astype(np.float64)
    return hessians


def compute_awq_scale_dict(hessians: dict[str, object], *, alpha: float) -> dict[str, object]:
    return {name: compute_awq_scales_from_hessian(h, alpha) for name, h in hessians.items()}


def load_raw_sumsq_dict(
    npz_path: str, selected: list[dict[str, object]]
) -> tuple[dict[str, "object"], list[str]]:
    """Load raw per-channel sum² vectors for the iterate-selected tensors.

    Returns (mapping, missing) where ``mapping[hfq_name]`` is a 1D
    float64 array of length K (input channels) and ``missing`` lists
    the hfq names that had no entry in the npz.

    This is the "AWQ-side" stats source. It exists to bypass the
    structural bug in ``compute_awq_scales_from_hessian`` where the
    rotated Hessian's diagonal is fed to the AWQ paper formula:
    ``diag(R · H · R.T)`` mixes channels under the FWHT and is not the
    raw ``sum(x_j²)`` AWQ actually needs.
    """
    import numpy as np

    data = np.load(npz_path)
    mapping: dict[str, object] = {}
    missing: list[str] = []
    for item in selected:
        name = item["hfq_name"]
        key = safe_key(name)
        if key in data.files:
            mapping[name] = data[key].astype(np.float64).reshape(-1)
        else:
            missing.append(name)
    return mapping, missing


def compute_awq_scale_dict_with_raw(
    hessians: dict[str, object],
    raw_sumsq: dict[str, object],
    *,
    alpha: float,
) -> dict[str, object]:
    """Per-tensor AWQ scales, preferring raw sumsq over Hessian-diagonal.

    ``raw_sumsq[name]`` (if present) is consumed directly by the
    1D-vector branch of ``compute_awq_scales``. Tensors absent from
    ``raw_sumsq`` fall back to the legacy Hessian-diagonal path so the
    flag is incremental and safe to deploy with partial coverage.
    """
    out: dict[str, object] = {}
    for name, h in hessians.items():
        raw = raw_sumsq.get(name)
        if raw is not None:
            out[name] = compute_awq_scales(raw, alpha)
        else:
            out[name] = compute_awq_scales_from_hessian(h, alpha)
    return out


def damp_awq_scale_dict(
    previous: dict[str, object] | None,
    raw: dict[str, object],
    *,
    damping: float,
) -> dict[str, object]:
    import numpy as np

    beta = float(damping)
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"damping must be in [0, 1], got {damping}")
    if previous is None:
        return {name: np.asarray(scale, dtype=np.float32).copy() for name, scale in raw.items()}
    out = {}
    for name, raw_scale in raw.items():
        if name not in previous:
            out[name] = np.asarray(raw_scale, dtype=np.float32).copy()
            continue
        prev = np.asarray(previous[name], dtype=np.float32)
        cur = np.asarray(raw_scale, dtype=np.float32)
        if prev.shape != cur.shape:
            raise ValueError(f"scale shape mismatch for {name}: {prev.shape} vs {cur.shape}")
        out[name] = ((np.float32(1.0 - beta) * prev) + (np.float32(beta) * cur)).astype(np.float32)
    return out


def save_awq_scales_npz(path, raw: dict[str, object], damped: dict[str, object]) -> None:
    import numpy as np

    payload = {}
    for name, scale in raw.items():
        payload[f"raw__{safe_key(name)}"] = np.asarray(scale, dtype=np.float32)
    for name, scale in damped.items():
        payload[f"damped__{safe_key(name)}"] = np.asarray(scale, dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def awq_scales_to_f16_bytes(scales) -> bytes:
    import numpy as np

    return np.asarray(scales, dtype="<f2").reshape(-1).tobytes()


def write_awq_sidecar_hfq(base_hfq, output_hfq, scales_by_weight: dict[str, object]) -> None:
    layout = ASTREA.read_hfq_layout(base_hfq)
    records = layout["records"]
    by_name = {record["name"]: i for i, record in enumerate(records)}
    inserts = []
    for weight_name, scales in scales_by_weight.items():
        payload = awq_scales_to_f16_bytes(scales)
        sidecar_name = awq_hessian_sidecar_name(weight_name)
        record = {
            "name": sidecar_name,
            "quant_type": 1,
            "quant_type_name": "F16",
            "shape": [len(payload) // 2],
            "group_size": 0,
            "data_size": len(payload),
            "data": payload,
        }
        if sidecar_name in by_name:
            records[by_name[sidecar_name]].update(record)
            continue
        parent_index = by_name.get(weight_name)
        if parent_index is None:
            inserts.append((len(records), record))
        else:
            inserts.append((parent_index + 1, record))
    for index, record in sorted(inserts, key=lambda item: item[0], reverse=True):
        records.insert(index, record)
    ASTREA.write_hfq_layout(output_hfq, layout)


def parse_max_memory(value: str | None):
    if not value:
        return None
    out: dict[int | str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"max-memory item must be DEVICE=VALUE, got {item!r}")
        key, mem = item.split("=", 1)
        key = key.strip()
        mem = mem.strip()
        if not key or not mem:
            raise ValueError(f"invalid max-memory item: {item!r}")
        out[int(key) if key.isdigit() else key] = mem
    return out


def model_input_device(model, fallback):
    try:
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "weight", None) is not None:
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def torch_fwht_256(x, signs1, signs2):
    import torch

    y = x.reshape(-1, 256).to(torch.float32) * signs1
    stride = 1
    while stride < 256:
        y = y.reshape(-1, 256 // (stride * 2), stride * 2)
        a = y[:, :, :stride].clone()
        b = y[:, :, stride:].clone()
        y[:, :, :stride] = a + b
        y[:, :, stride:] = a - b
        y = y.reshape(-1, 256)
        stride *= 2
    y = y * 0.0625 * signs2
    return y.reshape(x.shape)


def linear_rotated_sumsq(x, signs1, signs2):
    import torch

    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    else:
        x = x.reshape(-1, x.shape[-1])
    k = int(x.shape[-1])
    if k % 256 != 0:
        return None, 0
    xr = torch_fwht_256(x.reshape(-1, k // 256, 256), signs1, signs2).reshape(-1, k)
    return xr.square().sum(dim=0).detach().cpu().to(torch.float64).numpy(), int(xr.shape[0])


def linear_rotated_hessian(x, signs1, signs2):
    import torch

    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    else:
        x = x.reshape(-1, x.shape[-1])
    k = int(x.shape[-1])
    if k % 256 != 0:
        return None, 0
    xr = torch_fwht_256(x.reshape(-1, k // 256, 256), signs1, signs2).to(torch.float32)
    h = torch.einsum("ngi,ngj->gij", xr, xr)
    return h.detach().cpu().to(torch.float64).numpy(), int(xr.shape[0])


def conv1d_flat_sumsq(x, kernel_size: int):
    import torch

    if x.dim() != 3:
        return None, 0
    if x.shape[1] < x.shape[2]:
        # Conv1d normally sees [B, C, T]. Keep a defensive fallback for [B, T, C].
        b, c, t = x.shape
    else:
        x = x.transpose(1, 2).contiguous()
        b, c, t = x.shape
    per_channel = x.to(torch.float32).square().sum(dim=(0, 2)).detach().cpu().to(torch.float64).numpy()
    flat = per_channel.repeat(kernel_size)
    return flat, int(b * t)


def dequantize_mq4g256_payload(payload: bytes, shape: list[int]):
    import numpy as np

    if len(shape) != 2:
        raise ValueError(f"MQ4G256 dequant expects 2D tensor shape, got {shape}")
    m, k = int(shape[0]), int(shape[1])
    if k % 256 != 0:
        raise ValueError(f"MQ4G256 dequant requires K%256==0, got K={k}")
    n_groups = k // 256
    expected_blocks = m * n_groups
    expected_bytes = expected_blocks * 136
    if len(payload) != expected_bytes:
        raise ValueError(f"MQ4G256 payload size mismatch: {len(payload)} vs {expected_bytes}")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(expected_blocks, 136)
    scale = blocks[:, 0:4].copy().view("<f4").reshape(expected_blocks)
    zero = blocks[:, 4:8].copy().view("<f4").reshape(expected_blocks)
    packed = blocks[:, 8:]
    q = np.empty((expected_blocks, 256), dtype=np.uint8)
    q[:, 0::2] = packed & np.uint8(0x0F)
    q[:, 1::2] = (packed >> np.uint8(4)) & np.uint8(0x0F)
    rotated = q.astype(np.float32) * scale[:, None].astype(np.float32) + zero[:, None].astype(np.float32)
    unrotated = ASTREA.inverse_fwht_256_numpy(rotated.reshape(m, n_groups, 256)).reshape(m, k)
    return unrotated.astype(np.float32, copy=False)


def read_hfq_f16_vector(path, tensor_info):
    import numpy as np

    payload = read_hfq_payload(path, tensor_info)
    return np.frombuffer(payload, dtype="<f2").astype(np.float32)


def load_awq_scales_by_weight(path) -> dict[str, object]:
    _, tensors = ASTREA.read_hfq_index(path, max_tensors=0)
    out = {}
    for name, info in tensors.items():
        if not name.endswith(".awq_scale.weight"):
            continue
        weight_name = name[: -len(".awq_scale.weight")] + ".weight"
        out[weight_name] = read_hfq_f16_vector(path, info)
    return out


def install_candidate_mq4_weights(model, candidate_mq4):
    import torch

    _, tensors = ASTREA.read_hfq_index(candidate_mq4, max_tensors=0)
    module_map = dict(model.named_modules())
    awq_scales = load_awq_scales_by_weight(candidate_mq4)
    installed = []
    missing_modules = []
    hooks = []

    for name, info in tensors.items():
        if not name.endswith(".weight") or name.endswith(".awq_scale.weight"):
            continue
        if int(info["quant_type"]) != 13 or len(info["shape"]) != 2:
            continue
        module_name = hfq_to_module_name(name)
        module = module_map.get(module_name)
        if module is None or not hasattr(module, "weight"):
            missing_modules.append(module_name)
            continue
        deq = dequantize_mq4g256_payload(read_hfq_payload(candidate_mq4, info), [int(x) for x in info["shape"]])
        weight = torch.as_tensor(deq, dtype=torch.float32, device=module.weight.device)
        if tuple(module.weight.shape) != tuple(weight.shape):
            missing_modules.append(f"{module_name}:shape {tuple(module.weight.shape)}!={tuple(weight.shape)}")
            continue
        module.weight.data.copy_(weight.to(dtype=module.weight.dtype))
        installed.append(name)
        scale = awq_scales.get(name)
        if scale is not None:
            scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=module.weight.device)

            def pre_hook(_module, inputs, scale_tensor=scale_tensor):
                if not inputs:
                    return inputs
                x = inputs[0]
                local_scale = scale_tensor.to(device=x.device, dtype=x.dtype)
                return (x / local_scale,) + tuple(inputs[1:])

            hooks.append(module.register_forward_pre_hook(pre_hook))

    return {"installed": installed, "missing_modules": missing_modules, "hooks": hooks}


def collect_stats_candidate_mq4(args, targets, chunks, run_dir):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    install_summary = install_candidate_mq4_weights(model, args.candidate_mq4)
    input_device = model_input_device(model, device)
    module_map = dict(model.named_modules())
    sign_cache = {}
    stats = {}
    counts = {}
    handles = []

    def signs_for(tensor_device):
        key = str(tensor_device)
        if key not in sign_cache:
            sign_cache[key] = (
                torch.tensor(ASTREA.FWHT_SIGNS1, dtype=torch.float32, device=tensor_device).reshape(1, 256),
                torch.tensor(ASTREA.FWHT_SIGNS2, dtype=torch.float32, device=tensor_device).reshape(1, 256),
            )
        return sign_cache[key]

    def make_hook(item):
        hfq_name = item["hfq_name"]
        shape = [int(x) for x in item["shape"]]
        is_conv = len(shape) == 3 and "conv1d" in item["module_name"]
        kernel_size = int(shape[-1]) if is_conv else 0

        def hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0].detach()
            if is_conv:
                delta, n = conv1d_flat_sumsq(x, kernel_size)
            elif args.stats_mode == "hessian":
                signs1, signs2 = signs_for(x.device)
                delta, n = linear_rotated_hessian(x, signs1, signs2)
            else:
                signs1, signs2 = signs_for(x.device)
                delta, n = linear_rotated_sumsq(x, signs1, signs2)
            if delta is None or n <= 0:
                return
            stats[hfq_name] = delta if hfq_name not in stats else stats[hfq_name] + delta
            counts[hfq_name] = counts.get(hfq_name, 0) + n

        return hook

    missing_modules = []
    for item in targets:
        module = module_map.get(item["module_name"])
        if module is None:
            missing_modules.append(item["module_name"])
            continue
        handles.append(module.register_forward_hook(make_hook(item)))

    for input_ids in chunks:
        ids = torch.tensor([input_ids], dtype=torch.long, device=input_device)
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False)

    for handle in handles:
        handle.remove()
    for handle in install_summary["hooks"]:
        handle.remove()

    stats_npz = run_dir / "stats-merged.npz"
    np.savez_compressed(stats_npz, **{safe_key(k): v for k, v in stats.items()})
    result = {
        "schema": SCHEMA_STATS,
        "captured_at_utc": utc_now(),
        "stats_mode": args.stats_mode,
        "hf_model": args.hf_model,
        "candidate_mq4": str(Path(args.candidate_mq4).expanduser()),
        "mask": str(Path(args.mask)),
        "calib_text": str(Path(args.calib_text)),
        "ctx": args.ctx,
        "requested_chunks": args.chunks,
        "actual_chunks": len(chunks),
        "devices": ["cpu"],
        "device_map": "candidate-mq4-cpu",
        "max_memory": None,
        "target_count": len(targets),
        "stat_count": len(stats),
        "counts": counts,
        "rank_summaries": [
            {
                "device": "cpu",
                "input_device": str(input_device),
                "chunk_count": len(chunks),
                "target_count": len(targets),
                "stat_count": len(stats),
                "missing_modules": missing_modules,
                "candidate_install": {
                    "installed_count": len(install_summary["installed"]),
                    "missing_modules": install_summary["missing_modules"],
                },
                "counts": counts,
            }
        ],
        "stats_npz": str(stats_npz),
    }
    write_json(run_dir / "stats.json", result, pretty=True)
    return result


def stats_worker(worker):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    device_index = int(worker["device"])
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    sign_cache = {}

    def signs_for(tensor_device):
        key = str(tensor_device)
        if key not in sign_cache:
            sign_cache[key] = (
                torch.tensor(ASTREA.FWHT_SIGNS1, dtype=torch.float32, device=tensor_device).reshape(1, 256),
                torch.tensor(ASTREA.FWHT_SIGNS2, dtype=torch.float32, device=tensor_device).reshape(1, 256),
            )
        return sign_cache[key]

    device_map = worker.get("device_map")
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if device_map and device_map != "none":
        load_kwargs["device_map"] = device_map
        max_memory = parse_max_memory(worker.get("max_memory"))
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(worker["hf_model"], **load_kwargs)
    if not device_map or device_map == "none":
        model = model.to(device)
    model.eval()
    input_device = model_input_device(model, device)
    module_map = dict(model.named_modules())

    targets = worker["targets"]
    stats = {}
    counts = {}
    handles = []

    def make_hook(item):
        hfq_name = item["hfq_name"]
        shape = [int(x) for x in item["shape"]]
        is_conv = len(shape) == 3 and "conv1d" in item["module_name"]
        kernel_size = int(shape[-1]) if is_conv else 0

        def hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0].detach()
            if is_conv:
                delta, n = conv1d_flat_sumsq(x, kernel_size)
            elif worker.get("stats_mode") == "hessian":
                signs1, signs2 = signs_for(x.device)
                delta, n = linear_rotated_hessian(x, signs1, signs2)
            else:
                signs1, signs2 = signs_for(x.device)
                delta, n = linear_rotated_sumsq(x, signs1, signs2)
            if delta is None or n <= 0:
                return
            if hfq_name not in stats:
                stats[hfq_name] = delta
                counts[hfq_name] = n
            else:
                stats[hfq_name] += delta
                counts[hfq_name] += n

        return hook

    missing_modules = []
    for item in targets:
        module = module_map.get(item["module_name"])
        if module is None:
            missing_modules.append(item["module_name"])
            continue
        handles.append(module.register_forward_hook(make_hook(item)))

    chunks = worker["chunks"]
    for input_ids in chunks:
        ids = torch.tensor([input_ids], dtype=torch.long, device=input_device)
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False)

    for handle in handles:
        handle.remove()

    out_npz = Path(worker["out_npz"])
    out_json = Path(worker["out_json"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **{safe_key(k): v for k, v in stats.items()})
    summary = {
        "device": device_index,
        "input_device": str(input_device),
        "device_map": device_map or "none",
        "chunk_count": len(chunks),
        "target_count": len(targets),
        "stat_count": len(stats),
        "missing_modules": missing_modules,
        "counts": counts,
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def chunk_token_ids(token_ids: list[int], *, ctx: int, chunks: int, offset: int = 0) -> list[list[int]]:
    out = []
    pos = max(0, int(offset))
    while len(out) < chunks and pos + ctx <= len(token_ids):
        out.append(token_ids[pos : pos + ctx])
        pos += ctx
    return out


def collect_stats(args):
    import numpy as np
    from transformers import AutoTokenizer

    mask = read_json(args.mask)
    targets = [row for row in mask["tensors"] if row.get("packable_flat_mq4")]
    if args.tensor_filter:
        filters = [x.strip() for x in args.tensor_filter.split(",") if x.strip()]
        targets = [row for row in targets if any(f in row["hfq_name"] for f in filters)]
    if args.max_tensors:
        targets = targets[: args.max_tensors]
    if not targets:
        raise SystemExit("no targets selected")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True, trust_remote_code=True)
    text = Path(args.calib_text).read_text(errors="replace")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = chunk_token_ids(token_ids, ctx=args.ctx, chunks=args.chunks, offset=args.offset)
    if not chunks:
        raise SystemExit("calibration text did not yield any chunks")

    device_map = args.device_map if args.device_map != "none" else None
    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    if device_map:
        if len(devices) < 1:
            raise SystemExit("--device-map requires at least one device in --devices for the input device")
        devices = [devices[0]]
    shard_chunks = [chunks[i:: len(devices)] for i in range(len(devices))]
    run_dir = Path(args.out_dir)
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "candidate_mq4", None):
        collect_stats_candidate_mq4(args, targets, chunks, run_dir)
        return

    workers = []
    for rank, device in enumerate(devices):
        workers.append(
            {
                "device": device,
                "hf_model": args.hf_model,
                "stats_mode": args.stats_mode,
                "device_map": device_map,
                "max_memory": args.max_memory,
                "targets": targets,
                "chunks": shard_chunks[rank],
                "out_npz": str(run_dir / f"stats-rank{rank}-gpu{device}.npz"),
                "out_json": str(run_dir / f"stats-rank{rank}-gpu{device}.json"),
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(workers)) as pool:
        summaries = pool.map(stats_worker, workers)

    merged = {}
    counts = {}
    for worker, summary in zip(workers, summaries):
        data = np.load(worker["out_npz"])
        for key in data.files:
            name = unsafekey(key)
            arr = data[key].astype(np.float64)
            merged[name] = arr if name not in merged else merged[name] + arr
        for name, count in summary.get("counts", {}).items():
            counts[name] = counts.get(name, 0) + int(count)

    stats_npz = run_dir / "stats-merged.npz"
    np.savez_compressed(stats_npz, **{safe_key(k): v for k, v in merged.items()})
    result = {
        "schema": SCHEMA_STATS,
        "captured_at_utc": utc_now(),
        "stats_mode": args.stats_mode,
        "hf_model": args.hf_model,
        "mask": str(Path(args.mask)),
        "calib_text": str(Path(args.calib_text)),
        "ctx": args.ctx,
        "requested_chunks": args.chunks,
        "actual_chunks": len(chunks),
        "devices": devices,
        "device_map": device_map or "none",
        "max_memory": (
            {str(k): v for k, v in parse_max_memory(args.max_memory).items()}
            if args.max_memory
            else None
        ),
        "target_count": len(targets),
        "stat_count": len(merged),
        "counts": counts,
        "rank_summaries": summaries,
        "stats_npz": str(stats_npz),
    }
    write_json(run_dir / "stats.json", result, pretty=True)


def weighted_ls_refit(rotated, q, scale, zero, weights, *, iters: int):
    import numpy as np

    r = np.asarray(rotated, dtype=np.float32)
    best_q = np.asarray(q, dtype=np.uint8).copy()
    best_scale = np.asarray(scale, dtype=np.float32).copy()
    best_zero = np.asarray(zero, dtype=np.float32).copy()
    if weights is None:
        return ASTREA.mq4_affine_ls_refit_numpy(r, best_q, best_scale, best_zero, iters=iters)
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != r.shape:
        raise ValueError(f"weights shape {w.shape} does not match rotated shape {r.shape}")
    w = np.where(np.isfinite(w) & (w > 0.0), w, np.float32(0.0))
    mean_w = np.mean(w, axis=1, keepdims=True)
    w = np.where(mean_w > 0.0, w / mean_w, np.float32(1.0))
    sw = np.sum(w, axis=1)
    valid_sw = sw > np.float32(1.0e-12)
    for _ in range(max(1, int(iters))):
        qf = best_q.astype(np.float32)
        mean_q = np.where(valid_sw, np.sum(w * qf, axis=1) / sw, np.mean(qf, axis=1))
        mean_r = np.where(valid_sw, np.sum(w * r, axis=1) / sw, np.mean(r, axis=1))
        dq = qf - mean_q[:, None]
        dr = r - mean_r[:, None]
        var_q = np.where(valid_sw, np.sum(w * dq * dq, axis=1) / sw, np.mean(dq * dq, axis=1))
        cov_qr = np.where(valid_sw, np.sum(w * dq * dr, axis=1) / sw, np.mean(dq * dr, axis=1))
        cand_scale = np.where(var_q > np.float32(1.0e-12), cov_qr / var_q, best_scale)
        cand_zero = mean_r - cand_scale * mean_q
        valid = np.isfinite(cand_scale) & np.isfinite(cand_zero) & (cand_scale > 0.0)
        best_scale = np.where(valid, cand_scale, best_scale).astype(np.float32)
        best_zero = np.where(valid, cand_zero, best_zero).astype(np.float32)
        inv = np.where(best_scale > 0.0, np.float32(1.0) / best_scale, np.float32(0.0))
        best_q = np.clip(np.floor((r - best_zero[:, None]) * inv[:, None] + np.float32(0.5)), 0, 15).astype(np.uint8)
    return best_q, best_scale, best_zero


def quantize_fixed_affine_gptq(w, h_inv, scale, zero):
    import numpy as np

    work = np.asarray(w, dtype=np.float32).copy()
    h_inv = np.asarray(h_inv, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float32)
    zero = np.asarray(zero, dtype=np.float32)
    m, k = work.shape
    q = np.zeros((m, k), dtype=np.uint8)
    inv_scale = np.where(scale > 0.0, np.float32(1.0) / scale, np.float32(0.0))
    for i in range(k):
        qi = np.clip(
            np.floor((work[:, i] - zero) * inv_scale + np.float32(0.5)),
            0,
            15,
        ).astype(np.uint8)
        q[:, i] = qi
        qv = qi.astype(np.float32) * scale + zero
        denom = float(h_inv[i, i])
        if abs(denom) < 1.0e-12 or not np.isfinite(denom):
            denom = 1.0
        err = (work[:, i] - qv) / np.float32(denom)
        if i + 1 < k:
            work[:, i + 1 :] -= err[:, None] * h_inv[i, i + 1 :].astype(np.float32)[None, :]
    return q


def weighted_objective_for_codes(w, q, scale, zero, h):
    import numpy as np

    err = (
        w.astype(np.float64)
        - q.astype(np.float64) * scale[:, None].astype(np.float64)
        - zero[:, None].astype(np.float64)
    )
    return np.sum((err @ h) * err, axis=1)


def refit_affine_full_h(w, q, h, scale, zero):
    import numpy as np

    qf = q.astype(np.float64)
    wf = w.astype(np.float64)
    h = h.astype(np.float64)
    ones = np.ones(h.shape[0], dtype=np.float64)
    h1 = h @ ones
    c = float(ones @ h1)
    qh = qf @ h
    a = np.sum(qh * qf, axis=1)
    b = qf @ h1
    d = np.sum(qh * wf, axis=1)
    e = wf @ h1
    det = a * c - b * b
    denom = np.where(np.abs(det) > 1.0e-12, det, 1.0)
    new_scale = (d * c - b * e) / denom
    new_zero = (a * e - b * d) / denom
    valid = np.isfinite(new_scale) & np.isfinite(new_zero) & (new_scale > 0.0) & (np.abs(det) > 1.0e-12)
    old_obj = weighted_objective_for_codes(w, q, scale, zero, h)
    cand_scale = np.where(valid, new_scale, scale).astype(np.float32)
    cand_zero = np.where(valid, new_zero, zero).astype(np.float32)
    new_obj = weighted_objective_for_codes(w, q, cand_scale, cand_zero, h)
    keep = valid & (new_obj <= old_obj)
    out_scale = np.where(keep, cand_scale, scale).astype(np.float32)
    out_zero = np.where(keep, cand_zero, zero).astype(np.float32)
    return out_scale, out_zero


def damped_inverse_hessian(h, damp):
    import numpy as np

    h = np.asarray(h, dtype=np.float64)
    h = (h + h.T) * 0.5
    diag = np.diag(h)
    positive = diag[np.isfinite(diag) & (diag > 0.0)]
    mean_diag = float(np.mean(positive)) if positive.size else 1.0
    eye = np.eye(h.shape[0], dtype=np.float64)
    for mult in (1.0, 3.0, 10.0, 30.0, 100.0):
        hd = h + eye * (float(damp) * mult * mean_diag + 1.0e-8)
        try:
            return hd, np.linalg.inv(hd), float(damp) * mult
        except np.linalg.LinAlgError:
            pass
    hd = h + eye * (float(damp) * 1000.0 * mean_diag + 1.0e-6)
    return hd, np.linalg.pinv(hd), float(damp) * 1000.0


def require_torch_for_gptq():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for --gpu GPTQ quantization; install torch or omit --gpu") from exc
    return torch


def damped_inverse_hessian_torch(h, damp, *, device):
    torch = require_torch_for_gptq()

    h = torch.as_tensor(h, dtype=torch.float32, device=device)
    h = (h + h.T) * torch.tensor(0.5, dtype=torch.float32, device=device)
    diag = torch.diagonal(h)
    positive = diag[torch.isfinite(diag) & (diag > 0.0)]
    mean_diag = torch.mean(positive) if positive.numel() else torch.tensor(1.0, dtype=torch.float32, device=device)
    eye = torch.eye(h.shape[0], dtype=torch.float32, device=device)
    for mult in (1.0, 3.0, 10.0, 30.0, 100.0):
        used_damp = torch.tensor(float(damp) * mult, dtype=torch.float32, device=device)
        hd = h + eye * (used_damp * mean_diag + torch.tensor(1.0e-8, dtype=torch.float32, device=device))
        try:
            chol = torch.linalg.cholesky(hd)
            return hd, torch.cholesky_inverse(chol), float(damp) * mult
        except RuntimeError:
            pass
    hd = h + eye * (
        torch.tensor(float(damp) * 1000.0, dtype=torch.float32, device=device) * mean_diag
        + torch.tensor(1.0e-6, dtype=torch.float32, device=device)
    )
    try:
        chol = torch.linalg.cholesky(hd)
        h_inv = torch.cholesky_inverse(chol)
    except RuntimeError:
        h_inv = torch.linalg.pinv(hd)
    return hd, h_inv.to(torch.float32), float(damp) * 1000.0


def quantize_fixed_affine_gptq_torch(w, h_inv, scale, zero, *, device):
    torch = require_torch_for_gptq()

    work = torch.as_tensor(w, dtype=torch.float32, device=device).clone()
    h_inv = torch.as_tensor(h_inv, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    zero = torch.as_tensor(zero, dtype=torch.float32, device=device)
    m, k = work.shape
    q = torch.empty((m, k), dtype=torch.uint8, device=device)
    inv_scale = torch.where(scale > 0.0, torch.reciprocal(scale), torch.zeros_like(scale))
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    eps = torch.tensor(1.0e-12, dtype=torch.float32, device=device)
    for i in range(k):
        qi = torch.clamp(torch.floor((work[:, i] - zero) * inv_scale + 0.5), 0, 15).to(torch.uint8)
        q[:, i] = qi
        qv = qi.to(torch.float32) * scale + zero
        denom = h_inv[i, i]
        denom = torch.where(torch.isfinite(denom) & (torch.abs(denom) >= eps), denom, one)
        err = (work[:, i] - qv) / denom
        if i + 1 < k:
            work[:, i + 1 :] -= err[:, None] * h_inv[i, i + 1 :][None, :]
    return q


def weighted_objective_for_codes_torch(w, q, scale, zero, h, *, device):
    torch = require_torch_for_gptq()

    w = torch.as_tensor(w, dtype=torch.float32, device=device)
    qf = torch.as_tensor(q, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    zero = torch.as_tensor(zero, dtype=torch.float32, device=device)
    h = torch.as_tensor(h, dtype=torch.float32, device=device)
    err = w - qf * scale[:, None] - zero[:, None]
    return torch.sum((err @ h) * err, dim=1)


def refit_affine_full_h_torch(w, q, h, scale, zero, *, device):
    torch = require_torch_for_gptq()

    qf = torch.as_tensor(q, dtype=torch.float32, device=device)
    wf = torch.as_tensor(w, dtype=torch.float32, device=device)
    h = torch.as_tensor(h, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    zero = torch.as_tensor(zero, dtype=torch.float32, device=device)
    ones = torch.ones(h.shape[0], dtype=torch.float32, device=device)
    h1 = h @ ones
    c = ones @ h1
    qh = qf @ h
    a = torch.sum(qh * qf, dim=1)
    b = qf @ h1
    d = torch.sum(qh * wf, dim=1)
    e = wf @ h1
    det = a * c - b * b
    denom = torch.where(torch.abs(det) > 1.0e-12, det, torch.ones_like(det))
    new_scale = (d * c - b * e) / denom
    new_zero = (a * e - b * d) / denom
    valid = torch.isfinite(new_scale) & torch.isfinite(new_zero) & (new_scale > 0.0) & (torch.abs(det) > 1.0e-12)
    old_obj = weighted_objective_for_codes_torch(wf, qf, scale, zero, h, device=device)
    cand_scale = torch.where(valid, new_scale, scale).to(torch.float32)
    cand_zero = torch.where(valid, new_zero, zero).to(torch.float32)
    new_obj = weighted_objective_for_codes_torch(wf, qf, cand_scale, cand_zero, h, device=device)
    keep = valid & (new_obj <= old_obj)
    out_scale = torch.where(keep, cand_scale, scale).to(torch.float32)
    out_zero = torch.where(keep, cand_zero, zero).to(torch.float32)
    return out_scale, out_zero


def solve_mq4_gptq_group_torch(w_fit, h, h_inv, *, refit_iters, device):
    torch = require_torch_for_gptq()

    w = torch.as_tensor(w_fit, dtype=torch.float32, device=device)
    h = torch.as_tensor(h, dtype=torch.float32, device=device)
    h_inv = torch.as_tensor(h_inv, dtype=torch.float32, device=device)
    min_val = torch.min(w, dim=1).values.to(torch.float32)
    max_val = torch.max(w, dim=1).values.to(torch.float32)
    value_range = (max_val - min_val).to(torch.float32)
    scale = torch.where(value_range > 0.0, value_range / 15.0, torch.ones_like(value_range)).to(torch.float32)
    zero = min_val
    q = None
    for _ in range(max(1, int(refit_iters))):
        q = quantize_fixed_affine_gptq_torch(w, h_inv, scale, zero, device=device)
        scale, zero = refit_affine_full_h_torch(w, q, h, scale, zero, device=device)
    q = quantize_fixed_affine_gptq_torch(w, h_inv, scale, zero, device=device)
    scale, zero = refit_affine_full_h_torch(w, q, h, scale, zero, device=device)
    obj = weighted_objective_for_codes_torch(w, q, scale, zero, h, device=device)
    return q, scale, zero, obj


def pack_mq4_gptq_blocks(q, scale, zero):
    import numpy as np

    q = np.asarray(q, dtype=np.uint8)
    scale = np.asarray(scale, dtype=np.float32)
    zero = np.asarray(zero, dtype=np.float32)
    block = np.zeros((q.shape[0], 136), dtype=np.uint8)
    block[:, 0:4] = scale.astype("<f4").view(np.uint8).reshape(q.shape[0], 4)
    block[:, 4:8] = zero.astype("<f4").view(np.uint8).reshape(q.shape[0], 4)
    block[:, 8:] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return block


def quantize_mq4_gptq_torch(
    values,
    *,
    hessian,
    shape,
    damp=0.01,
    refit_iters=2,
    clip_ratio=1.0,
    clip_grid=None,
    device=None,
):
    import numpy as np

    torch = require_torch_for_gptq()
    if device is None:
        device = torch.device("cpu")
    if clip_ratio != 1.0 or clip_grid is not None:
        raise ValueError("torch GPTQ path does not support clip_ratio or clip_grid; match the numpy GPTQ path")
    if len(shape) != 2:
        raise ValueError("GPTQ MQ4 PoC only supports 2D linear tensors")
    m, k = int(shape[0]), int(shape[1])
    if k % 256 != 0:
        raise ValueError(f"GPTQ MQ4 requires K%256==0, got K={k}")
    n_groups = k // 256
    hessian_t = torch.as_tensor(hessian, dtype=torch.float32, device=device)
    if tuple(hessian_t.shape) != (n_groups, 256, 256):
        raise ValueError(f"hessian shape mismatch: {tuple(hessian_t.shape)} vs {(n_groups, 256, 256)}")
    rows = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(m, k)
    signs1 = torch.tensor(ASTREA.FWHT_SIGNS1, dtype=torch.float32, device=device).reshape(1, 256)
    signs2 = torch.tensor(ASTREA.FWHT_SIGNS2, dtype=torch.float32, device=device).reshape(1, 256)
    rotated = torch_fwht_256(rows.reshape(m, n_groups, 256), signs1, signs2).reshape(m, n_groups, 256)
    out = np.zeros((m * n_groups, 136), dtype=np.uint8)
    objectives = []
    damp_values = []
    for group in range(n_groups):
        h, h_inv, used_damp = damped_inverse_hessian_torch(hessian_t[group], damp, device=device)
        damp_values.append(used_damp)
        q, scale, zero, obj = solve_mq4_gptq_group_torch(
            rotated[:, group, :],
            h,
            h_inv,
            refit_iters=refit_iters,
            device=device,
        )
        objectives.append(float(torch.mean(obj).detach().cpu()))
        block = pack_mq4_gptq_blocks(
            q.detach().cpu().numpy(),
            scale.detach().cpu().numpy(),
            zero.detach().cpu().numpy(),
        )
        out[group::n_groups, :] = block
    return out.tobytes(), {
        "mean_group_objective": float(np.mean(objectives)) if objectives else 0.0,
        "max_group_objective": float(np.max(objectives)) if objectives else 0.0,
        "mean_effective_damp": float(np.mean(damp_values)) if damp_values else float(damp),
    }


def quantize_mq4_gptq(values, *, hessian, shape, damp=0.01, refit_iters=2):
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    if len(shape) != 2:
        raise ValueError("GPTQ MQ4 PoC only supports 2D linear tensors")
    m, k = int(shape[0]), int(shape[1])
    if k % 256 != 0:
        raise ValueError(f"GPTQ MQ4 requires K%256==0, got K={k}")
    rows = arr.reshape(m, k)
    n_groups = k // 256
    hessian = np.asarray(hessian, dtype=np.float64)
    if hessian.shape != (n_groups, 256, 256):
        raise ValueError(f"hessian shape mismatch: {hessian.shape} vs {(n_groups, 256, 256)}")
    rotated = ASTREA.fwht_256_numpy(rows.reshape(m, n_groups, 256)).reshape(m, n_groups, 256)
    out = np.zeros((m * n_groups, 136), dtype=np.uint8)
    objectives = []
    damp_values = []
    for group in range(n_groups):
        w = rotated[:, group, :].astype(np.float32, copy=False)
        h, h_inv, used_damp = damped_inverse_hessian(hessian[group], damp)
        damp_values.append(used_damp)
        min_val = np.min(w, axis=1).astype(np.float32)
        max_val = np.max(w, axis=1).astype(np.float32)
        value_range = (max_val - min_val).astype(np.float32)
        scale = np.where(value_range > 0.0, value_range / np.float32(15.0), np.float32(1.0)).astype(np.float32)
        zero = min_val
        q = None
        for _ in range(max(1, int(refit_iters))):
            q = quantize_fixed_affine_gptq(w, h_inv, scale, zero)
            scale, zero = refit_affine_full_h(w, q, h, scale, zero)
        q = quantize_fixed_affine_gptq(w, h_inv, scale, zero)
        scale, zero = refit_affine_full_h(w, q, h, scale, zero)
        obj = weighted_objective_for_codes(w, q, scale, zero, h)
        objectives.append(float(np.mean(obj)))
        block = np.zeros((m, 136), dtype=np.uint8)
        block[:, 0:4] = scale.astype("<f4").view(np.uint8).reshape(m, 4)
        block[:, 4:8] = zero.astype("<f4").view(np.uint8).reshape(m, 4)
        block[:, 8:] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
        out[group::n_groups, :] = block
    return out.tobytes(), {
        "mean_group_objective": float(np.mean(objectives)) if objectives else 0.0,
        "max_group_objective": float(np.max(objectives)) if objectives else 0.0,
        "mean_effective_damp": float(np.mean(damp_values)) if damp_values else float(damp),
    }


def quantize_mq4_weighted(values, *, hrot=None, shape, clip_ratio=1.0, ls_iters=5):
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size % 256 != 0:
        raise ValueError(f"MQ4 flat pack requires numel%256==0, got {flat.size}")
    if len(shape) == 2 and int(shape[1]) % 256 == 0:
        rows = arr.reshape(int(shape[0]), int(shape[1]))
        m, k = rows.shape
        clipped = ASTREA.clipped_rows_for_ratio(rows, clip_ratio).reshape(m, k // 256, 256)
        rotated = ASTREA.fwht_256_numpy(clipped).reshape(-1, 256)
        if hrot is not None and len(hrot) == k:
            hblocks = np.asarray(hrot, dtype=np.float32).reshape(k // 256, 256)
            weights = np.tile(hblocks, (m, 1))
        else:
            weights = None
    else:
        rows = flat.reshape(1, flat.size)
        clipped = ASTREA.clipped_rows_for_ratio(rows, clip_ratio).reshape(-1, 256)
        rotated = ASTREA.fwht_256_numpy(clipped).reshape(-1, 256)
        weights = None
        if hrot is not None and len(hrot) == flat.size:
            # For non-linear flat tensors this is an approximation. Diagonal
            # activation weights become nearly uniform under FWHT if cross terms
            # are unknown, so use them only as a per-element prior.
            w = np.asarray(hrot, dtype=np.float32).reshape(-1, 256)
            weights = np.where(np.isfinite(w) & (w > 0.0), w, np.float32(1.0))

    min_val = np.min(rotated, axis=1).astype(np.float32)
    max_val = np.max(rotated, axis=1).astype(np.float32)
    value_range = (max_val - min_val).astype(np.float32)
    scale = np.where(value_range > 0.0, value_range / np.float32(15.0), np.float32(1.0)).astype(np.float32)
    inv_scale = np.where(value_range > 0.0, np.float32(1.0) / scale, np.float32(0.0)).astype(np.float32)
    q = np.floor((rotated - min_val[:, None]) * inv_scale[:, None] + np.float32(0.5))
    q = np.clip(q, 0, 15).astype(np.uint8)
    q, scale, min_val = weighted_ls_refit(rotated, q, scale, min_val, weights, iters=ls_iters)
    out = np.zeros((q.shape[0], 136), dtype=np.uint8)
    out[:, 0:4] = scale.astype("<f4").view(np.uint8).reshape(-1, 4)
    out[:, 4:8] = min_val.astype("<f4").view(np.uint8).reshape(-1, 4)
    out[:, 8:] = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    deq_rot = q.astype(np.float32) * scale[:, None] + min_val[:, None]
    err = (deq_rot - rotated) ** 2
    if weights is not None:
        denom = float(np.mean(weights)) if float(np.mean(weights)) > 0.0 else 1.0
        mse = float(np.mean(err * (weights / np.float32(denom))))
    else:
        mse = float(np.mean(err))
    return out.tobytes(), mse


def quantize_candidate(args):
    import numpy as np

    gptq_device = None
    if args.gpu is not None:
        torch = require_torch_for_gptq()
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was requested, but torch.cuda.is_available() is false")
        torch.cuda.set_device(int(args.gpu))
        gptq_device = torch.device(f"cuda:{int(args.gpu)}")

    mask = read_json(args.mask)
    stats = np.load(args.stats_npz) if args.stats_npz else None
    stats_counts = read_stats_counts(args.stats_npz, args.stats_json)
    awq_hessian_scales = read_awq_hessian_scales(args.awq_aware_hessian) if args.awq_aware_hessian else None
    _, base_tensors = ASTREA.read_hfq_index(args.base, max_tensors=0)
    source_summary, source_tensors = ASTREA.read_safetensors_dir_index(args.source_dir)
    ASTREA.copy_candidate_file(args.base, args.output)
    selected = [row for row in mask["tensors"] if row.get("packable_flat_mq4")]
    if args.tensor_name or args.tensor_list:
        requested = set(load_tensor_names_from_args(args))
        selected = [row for row in selected if row["hfq_name"] in requested]
    if args.tensor_filter:
        filters = [x.strip() for x in args.tensor_filter.split(",") if x.strip()]
        selected = [row for row in selected if any(f in row["hfq_name"] for f in filters)]
    if args.exclude_tensor_filter:
        filters = [x.strip() for x in args.exclude_tensor_filter.split(",") if x.strip()]
        selected = [row for row in selected if not any(f in row["hfq_name"] for f in filters)]
    if args.sort_by == "name":
        selected = sorted(selected, key=lambda row: row["hfq_name"])
    elif args.sort_by == "numel-asc":
        selected = sorted(selected, key=lambda row: int(row["numel"]))
    elif args.sort_by == "numel-desc":
        selected = sorted(selected, key=lambda row: int(row["numel"]), reverse=True)
    elif args.sort_by == "bytes-asc":
        selected = sorted(selected, key=lambda row: int(row["base_data_size"]))
    elif args.sort_by == "bytes-desc":
        selected = sorted(selected, key=lambda row: int(row["base_data_size"]), reverse=True)
    if args.max_tensors:
        selected = selected[: args.max_tensors]

    mutations = []
    total = len(selected)
    started = time.time()
    for index, item in enumerate(selected, start=1):
        name = item["hfq_name"]
        tensor_started = time.time()
        if args.progress_every and (
            index == 1 or index == total or (index - 1) % args.progress_every == 0
        ):
            print(
                f"[mq4_masked_calib] quantize {index}/{total} {name} "
                f"shape={item['shape']} bytes={item['base_data_size']}",
                file=sys.stderr,
                flush=True,
            )
        if name not in source_tensors:
            raise ValueError(f"source tensor missing: {name}")
        tensor_info = base_tensors[name]
        values = ASTREA.load_safetensors_array(source_tensors, name)
        hrot = None
        if stats is not None:
            key = safe_key(name)
            if key in stats.files:
                hrot = stats[key].astype(np.float32)
                count = max(1, int(stats_counts.get(name, item.get("count", 1))))
                hrot = hrot / np.float32(count)
        if args.method == "gptq":
            if len(item["shape"]) != 2:
                if args.skip_unsupported:
                    continue
                raise ValueError(f"GPTQ currently supports only 2D linear tensors: {name}")
            if hrot is None or hrot.ndim != 3:
                raise ValueError(f"GPTQ requires full hessian stats for {name}")
            awq_sidecar_name = None
            if awq_hessian_scales is not None:
                awq_sidecar_name = awq_hessian_sidecar_name(name)
                scale = awq_hessian_scales.get(awq_sidecar_name)
                if scale is not None:
                    hrot = apply_awq_hessian_transform(hrot, scale)
                    # AWQ-aware GPTQ also needs source weights pre-scaled by s
                    # so the GPTQ solve targets W·diag(s), matching the runtime
                    # invariant: (W·s)·(x/s) = W·x. Without this, GPTQ aims at
                    # raw W and the runtime x/s divide produces per-channel
                    # corruption (KLD ~1.75 vs expected ~0.10–0.13).
                    import numpy as _np
                    values = _np.asarray(values, dtype=_np.float32) * scale[_np.newaxis, :]
            if gptq_device is not None:
                packed, method_stats = quantize_mq4_gptq_torch(
                    values,
                    hessian=hrot,
                    shape=item["shape"],
                    damp=args.gptq_damp,
                    refit_iters=args.gptq_refit_iters,
                    device=gptq_device,
                )
            else:
                packed, method_stats = quantize_mq4_gptq(
                    values,
                    hessian=hrot,
                    shape=item["shape"],
                    damp=args.gptq_damp,
                    refit_iters=args.gptq_refit_iters,
                )
            if awq_hessian_scales is not None:
                method_stats["awq_hessian_sidecar"] = awq_sidecar_name
                method_stats["awq_hessian_applied"] = awq_sidecar_name in awq_hessian_scales
            metric = method_stats["mean_group_objective"]
        else:
            packed, metric = quantize_mq4_weighted(
                values,
                hrot=hrot,
                shape=item["shape"],
                clip_ratio=args.clip_ratio,
                ls_iters=args.ls_iters,
            )
            method_stats = {"weighted_rotated_mse": metric}
        if len(packed) != int(tensor_info["data_size"]):
            raise ValueError(f"packed size mismatch for {name}: {len(packed)} vs {tensor_info['data_size']}")
        ASTREA.patch_hfq_tensor(args.output, tensor_info, packed)
        elapsed = time.time() - tensor_started
        if args.progress_every and (
            index == total or index % args.progress_every == 0
        ):
            rate = index / max(1.0e-9, time.time() - started)
            eta = (total - index) / rate if rate > 0.0 else 0.0
            print(
                f"[mq4_masked_calib] done {index}/{total} {name} "
                f"seconds={elapsed:.1f} eta={eta:.1f}s metric={metric:.6g}",
                file=sys.stderr,
                flush=True,
            )
        mutations.append(
            {
                "hfq_name": name,
                "shape": item["shape"],
                "kind": item["kind"],
                "used_stats": hrot is not None,
                "method": args.method,
                "clip_ratio": args.clip_ratio,
                "ls_iters": args.ls_iters,
                "metric": metric,
                **method_stats,
            }
        )

    result = {
        "schema": SCHEMA_CANDIDATE,
        "captured_at_utc": utc_now(),
        "base": str(Path(args.base).expanduser()),
        "output": str(Path(args.output).expanduser()),
        "output_bytes": Path(args.output).expanduser().stat().st_size,
        "source": source_summary,
        "mask": str(Path(args.mask)),
        "stats_npz": str(Path(args.stats_npz)) if args.stats_npz else None,
        "stats_json": str(Path(args.stats_json)) if args.stats_json else (
            str(Path(args.stats_npz).resolve().parent / "stats.json") if args.stats_npz else None
        ),
        "selected_count": len(selected),
        "mutated_tensor_count": len(mutations),
        "used_stats_count": sum(1 for m in mutations if m["used_stats"]),
        "sort_by": args.sort_by,
        "method": args.method,
        "clip_ratio": args.clip_ratio,
        "ls_iters": args.ls_iters,
        "gptq_damp": args.gptq_damp,
        "gptq_refit_iters": args.gptq_refit_iters,
        "mutations": mutations,
        "next_step": "run test_inference, KLD/PPL, then Atlas AR perf if quality improves",
    }
    if awq_hessian_scales is not None:
        result["awq_aware_hessian"] = str(Path(args.awq_aware_hessian).expanduser())
        result["awq_hessian_scale_count"] = len(awq_hessian_scales)
    write_json(args.out, result, pretty=True)


def read_hfq_payload(path, tensor_info):
    with Path(path).expanduser().open("rb") as f:
        f.seek(int(tensor_info["data_offset"]))
        data = f.read(int(tensor_info["data_size"]))
    if len(data) != int(tensor_info["data_size"]):
        raise ValueError(f"short tensor payload for {tensor_info['name']}")
    return data


def patch_from_candidate(base, source, output, tensor_names):
    base = Path(base).expanduser()
    source = Path(source).expanduser()
    output = Path(output).expanduser()
    ASTREA.copy_candidate_file(base, output)
    _, base_tensors = ASTREA.read_hfq_index(base, max_tensors=0)
    _, source_tensors = ASTREA.read_hfq_index(source, max_tensors=0)
    patched = []
    for name in tensor_names:
        if name not in base_tensors:
            raise ValueError(f"tensor missing from base: {name}")
        if name not in source_tensors:
            raise ValueError(f"tensor missing from source: {name}")
        base_info = base_tensors[name]
        source_info = source_tensors[name]
        if int(base_info["data_size"]) != int(source_info["data_size"]):
            raise ValueError(
                f"tensor data size mismatch for {name}: {base_info['data_size']} vs {source_info['data_size']}"
            )
        payload = read_hfq_payload(source, source_info)
        ASTREA.patch_hfq_tensor(output, base_info, payload)
        patched.append(
            {
                "hfq_name": name,
                "data_offset": int(base_info["data_offset"]),
                "data_size": int(base_info["data_size"]),
            }
        )
    return patched


def load_tensor_names_from_args(args):
    names = []
    if args.tensor_name:
        names.extend(args.tensor_name)
    if args.tensor_list:
        data = read_json(args.tensor_list)
        if isinstance(data, list):
            names.extend(str(x) for x in data)
        elif isinstance(data, dict):
            for key in ("tensor_names", "selected", "tensors"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        names.append(item["hfq_name"] if isinstance(item, dict) and "hfq_name" in item else str(item))
                    break
        else:
            raise ValueError(f"unsupported tensor list payload in {args.tensor_list}")
    deduped = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def compose_candidate(args):
    names = load_tensor_names_from_args(args)
    if not names:
        raise ValueError("compose needs at least one tensor name")
    patched = patch_from_candidate(args.base, args.source, args.output, names)
    result = {
        "schema": SCHEMA_COMPOSE,
        "captured_at_utc": utc_now(),
        "base": str(Path(args.base).expanduser()),
        "source": str(Path(args.source).expanduser()),
        "output": str(Path(args.output).expanduser()),
        "output_bytes": Path(args.output).expanduser().stat().st_size,
        "patched_tensor_count": len(patched),
        "patched": patched,
        "next_step": "run test_inference and KLD/PPL against BF16 reference",
    }
    write_json(args.out, result, pretty=True)


def tensor_variant(index, name):
    stem = name
    for prefix in ("model.language_model.layers.", "model.language_model."):
        stem = stem.replace(prefix, "")
    stem = stem.replace(".weight", "")
    safe = []
    for ch in stem:
        safe.append(ch if ch.isalnum() else "-")
    compact = "-".join("".join(safe).split("-"))
    return f"t{index:03d}-{compact[:88]}"


def tensor_sweep_worker(worker):
    base = worker["base"]
    source = worker["source"]
    eval_bin = worker["eval_bin"]
    ref = worker["ref"]
    kv_mode = worker["kv_mode"]
    scoring_mode = worker["scoring_mode"]
    max_chunks = str(worker["max_chunks"])
    device = str(worker["device"])
    temp_dir = Path(worker["temp_dir"])
    result_dir = Path(worker["result_dir"])
    log_dir = Path(worker["log_dir"])
    records = []
    env_base = os.environ.copy()
    env_base["HIP_VISIBLE_DEVICES"] = device
    env_base["HIPFIRE_GRAPH"] = "0"
    for task in worker["tasks"]:
        variant = task["variant"]
        name = task["hfq_name"]
        model_path = temp_dir / f"{variant}.hfq"
        out_path = result_dir / f"{variant}__gfx1201__{scoring_mode}.kldseq"
        log_path = log_dir / f"{variant}.log"
        status = "unknown"
        started = time.time()
        try:
            patch_from_candidate(base, source, model_path, [name])
            cmd = [
                eval_bin,
                "--model",
                str(model_path),
                "--ref",
                ref,
                "--output",
                str(out_path),
                "--kv-mode",
                kv_mode,
                "--scoring-mode",
                scoring_mode,
                "--max-chunks",
                max_chunks,
            ]
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env_base)
            log_path.write_text(proc.stdout + proc.stderr)
            status = "ok" if proc.returncode == 0 else f"failed_{proc.returncode}"
        except Exception as exc:
            log_path.write_text(f"{type(exc).__name__}: {exc}\n")
            status = "exception"
        finally:
            try:
                model_path.unlink()
            except FileNotFoundError:
                pass
        records.append(
            {
                "variant": variant,
                "hfq_name": name,
                "device": int(device),
                "status": status,
                "seconds": time.time() - started,
                "result": str(out_path),
                "log": str(log_path),
            }
        )
    return records


def tensor_sweep(args):
    candidate = read_json(args.candidate_json)
    mutations = list(candidate.get("mutations") or [])
    if args.tensor_filter:
        filters = [x.strip() for x in args.tensor_filter.split(",") if x.strip()]
        mutations = [m for m in mutations if any(f in m["hfq_name"] for f in filters)]
    if args.max_tensors:
        mutations = mutations[: args.max_tensors]
    if not mutations:
        raise ValueError("no candidate mutations selected for tensor sweep")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    temp_dir = out_dir / "tmp-models"
    result_dir = out_dir / "results"
    log_dir = out_dir / "logs"
    for path in (temp_dir, result_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    tasks = []
    for index, mutation in enumerate(mutations):
        name = mutation["hfq_name"]
        tasks.append({"index": index, "variant": tensor_variant(index, name), "hfq_name": name})

    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    shards = [tasks[i:: len(devices)] for i in range(len(devices))]
    workers = [
        {
            "device": device,
            "tasks": shards[rank],
            "base": str(Path(args.base).expanduser()),
            "source": str(Path(args.source).expanduser()),
            "eval_bin": args.eval_bin,
            "ref": args.ref,
            "kv_mode": args.kv_mode,
            "scoring_mode": args.scoring_mode,
            "max_chunks": args.max_chunks,
            "temp_dir": str(temp_dir),
            "result_dir": str(result_dir),
            "log_dir": str(log_dir),
        }
        for rank, device in enumerate(devices)
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(workers)) as pool:
        nested = pool.map(tensor_sweep_worker, workers)
    records = [item for group in nested for item in group]
    records.sort(key=lambda x: x["variant"])
    result = {
        "schema": SCHEMA_TENSOR_SWEEP,
        "captured_at_utc": utc_now(),
        "base": str(Path(args.base).expanduser()),
        "source": str(Path(args.source).expanduser()),
        "candidate_json": str(Path(args.candidate_json)),
        "ref": str(Path(args.ref)),
        "devices": devices,
        "scoring_mode": args.scoring_mode,
        "kv_mode": args.kv_mode,
        "max_chunks": args.max_chunks,
        "tensor_count": len(tasks),
        "ok_count": sum(1 for r in records if r["status"] == "ok"),
        "records": records,
        "result_dir": str(result_dir),
        "next_step": "run kld_reduce on result_dir and join variants back to hfq_name using this JSON",
    }
    write_json(out_dir / "tensor-sweep.json", result, pretty=True)


def _is_awq_eligible_f1(hfq_name: str) -> bool:
    """F1-only mirror of crates/hipfire-quantize/src/main.rs::awq_eligible.

    Returns True iff the tensor name matches hipfire's F1 AWQ-eligible
    whitelist (input-side projections only — q/k/v, gate/up, in_proj_*,
    router). Output-side projections (lm_head, o_proj, down_proj, etc.)
    have no runtime x/s inverse on master without F2, so pre-scaling them
    produces (W·s)·x ≠ W·x corruption (KLD 0.7 → 1.7 measured on path 1).
    """
    F1_SUFFIXES = (
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "qkv_proj.weight", "wqkv.weight",
        "gate_proj.weight", "up_proj.weight",
        "w_gate.weight", "w_up.weight",
        "gate_up_proj.weight",
        "mlp.gate.weight", "router.weight",
    )
    if any(hfq_name.endswith(s) for s in F1_SUFFIXES):
        return True
    if ".in_proj_" in hfq_name:
        return True
    return False


def selected_iterate_targets(mask: dict[str, object]) -> list[dict[str, object]]:
    """Pick iterate targets: packable_flat_mq4 AND F1 AWQ-eligible.

    The AWQ-eligibility filter avoids corrupting tensors whose runtime
    path lacks an AWQ inverse divide (lm_head, o_proj, down_proj on
    master without F2). Without this filter, iterate's round 0 produces
    KLD ~0.70 vs v3's 0.13 — see investigation 2026-05-18-awq-gptq-sub-0.10-kld.
    """
    targets = [row for row in mask["tensors"] if row.get("packable_flat_mq4")]
    targets = [row for row in targets if _is_awq_eligible_f1(row["hfq_name"])]
    if not targets:
        raise ValueError("iterate requires at least one packable F1-AWQ-eligible MQ4 tensor")
    return targets


def materialize_round_stats(round_dir: Path, stats_npz, stats_json) -> tuple[Path, Path]:
    dst_npz = round_dir / "imatrix.npz"
    dst_json = round_dir / "stats.json"
    src_npz = Path(stats_npz)
    src_json = Path(stats_json) if stats_json else src_npz.resolve().parent / "stats.json"
    if src_npz.resolve() != dst_npz.resolve():
        shutil.copy2(src_npz, dst_npz)
    if src_json.exists() and src_json.resolve() != dst_json.resolve():
        shutil.copy2(src_json, dst_json)
    elif not dst_json.exists():
        dst_json.write_text(json.dumps({"counts": {}}, indent=2, sort_keys=True) + "\n")
    return dst_npz, dst_json


def collect_iterate_round_stats(args, round_index: int, previous_model: str | None, round_dir: Path):
    initial_npz = getattr(args, "initial_stats_npz", None)
    if round_index == 0 and initial_npz:
        initial_json = getattr(args, "initial_stats_json", None) or ""
        print(f"[iterate] round 0: using --initial-stats-npz {initial_npz}", flush=True)
        return materialize_round_stats(round_dir, initial_npz, initial_json)
    collect_dir = round_dir / "collect"
    collect_args = SimpleNamespace(
        hf_model=args.hf_model,
        mask=args.imatrix_mask,
        calib_text=args.calib_text,
        out_dir=str(collect_dir),
        devices=args.collect_devices,
        device_map="none",
        max_memory=None,
        ctx=args.ctx,
        chunks=args.chunks,
        offset=args.offset,
        stats_mode="hessian",
        max_tensors=None,
        tensor_filter=None,
        overwrite=True,
        candidate_mq4=previous_model if round_index > 0 else None,
    )
    collect_stats(collect_args)
    return materialize_round_stats(round_dir, collect_dir / "stats-merged.npz", collect_dir / "stats.json")


def run_round_bench(args, round_dir: Path, model_path: Path):
    if not args.bench_each_round:
        return None
    if not args.bench_ref:
        raise ValueError("--bench-each-round requires --bench-ref")
    output = round_dir / "bench.kldseq"
    log = round_dir / "bench.log"
    cmd = [
        args.eval_bin,
        "--model",
        str(model_path),
        "--ref",
        args.bench_ref,
        "--output",
        str(output),
        "--kv-mode",
        args.kv_mode,
        "--scoring-mode",
        args.scoring_mode,
        "--max-chunks",
        str(args.bench_max_chunks),
    ]
    env = os.environ.copy()
    if args.gpu is not None:
        env["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    log.write_text(proc.stdout + proc.stderr)
    metrics = reduce_kldseq_metrics(output) if proc.returncode == 0 and output.exists() else None
    return {
        "status": "ok" if proc.returncode == 0 else f"failed_{proc.returncode}",
        "command": cmd,
        "output": str(output),
        "log": str(log),
        "metrics": metrics,
    }


def reduce_kldseq_metrics(path):
    try:
        import numpy as np

        harness = SCRIPT_DIR.parent / "benchmarks" / "quality-baselines" / "harness"
        sys.path.insert(0, str(harness))
        from kldref_format import read_per_seq_kld

        means, p99s, nlls = read_per_seq_kld(Path(path))
        means_arr = np.asarray(means, dtype=np.float64)
        p99s_arr = np.asarray(p99s, dtype=np.float64)
        nlls_arr = np.asarray(nlls, dtype=np.float64)
        finite_nll = nlls_arr[np.isfinite(nlls_arr)]
        return {
            "kld_mean": float(means_arr.mean()) if means_arr.size else None,
            "kld_p99": float(np.percentile(p99s_arr, 99)) if p99s_arr.size else None,
            "ppl": float(np.exp(finite_nll.mean())) if finite_nll.size else None,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def write_round_summary(round_dir: Path, record: dict[str, object]) -> None:
    bench = record.get("bench")
    metrics = bench.get("metrics") if bench else None
    lines = [
        f"# Iterative AWQ+GPTQ round {record['round']}",
        "",
        f"- scale_delta: {record['scale_delta']:.8g}",
        f"- kld_mean: {metrics.get('kld_mean') if metrics and 'kld_mean' in metrics else 'not_run'}",
        f"- kld_p99: {metrics.get('kld_p99') if metrics and 'kld_p99' in metrics else 'not_run'}",
        f"- ppl: {metrics.get('ppl') if metrics and 'ppl' in metrics else 'not_run'}",
        f"- imatrix: {record['imatrix_npz']}",
        f"- awq_scales: {record['awq_scales_npz']}",
        f"- model: {record['model']}",
        f"- elapsed_seconds: {record['elapsed_seconds']:.3f}",
    ]
    if bench:
        lines.append(f"- bench_status: {bench['status']}")
        lines.append(f"- bench_output: {bench['output']}")
    else:
        lines.append("- bench_status: not_run")
    round_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n")


def quantize_iterate_round(
    *,
    round_dir: Path,
    base_hfq: str,
    hf_model: str,
    mask_path: str,
    stats_npz: Path,
    stats_json: Path,
    scales: dict[str, object],
    gpu: int | None,
    gptq_damp: float,
    gptq_refit_iters: int,
) -> Path:
    sidecar_base = round_dir / "awq_sidecar_base.hfq"
    output = round_dir / "model.hfq"
    candidate_json = round_dir / "candidate.json"
    write_awq_sidecar_hfq(base_hfq, sidecar_base, scales)
    quantize_candidate(
        SimpleNamespace(
            base=str(sidecar_base),
            source_dir=hf_model,
            mask=mask_path,
            stats_npz=str(stats_npz),
            stats_json=str(stats_json),
            output=str(output),
            out=str(candidate_json),
            clip_ratio=1.0,
            ls_iters=5,
            method="gptq",
            gpu=gpu,
            awq_aware_hessian=str(sidecar_base),
            gptq_damp=gptq_damp,
            gptq_refit_iters=gptq_refit_iters,
            skip_unsupported=True,
            max_tensors=None,
            tensor_filter="lm_head,in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj,mlp.,self_attn.",
            exclude_tensor_filter=None,
            tensor_name=None,
            tensor_list=None,
            sort_by="mask-order",
            progress_every=0,
        )
    )
    return output


def run_iterative_awq_gptq(args, *, stats_provider=None):
    mask = read_json(args.imatrix_mask)
    selected = selected_iterate_targets(mask)
    base_hfq = str(Path(mask.get("base") or "").expanduser())
    if not base_hfq or not Path(base_hfq).exists():
        raise ValueError("iterate mask must contain an existing 'base' HFQ path")
    if int(args.max_rounds) <= 0:
        raise ValueError("--max-rounds must be positive")
    if float(args.damping) < 0.0 or float(args.damping) > 1.0:
        raise ValueError("--damping must be in [0, 1]")

    out_dir = Path(args.base_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_scales = None
    previous_model = None
    trace = []
    started_all = time.time()
    provider = stats_provider or collect_iterate_round_stats

    raw_sumsq_path = getattr(args, "awq_raw_sumsq_npz", None)
    raw_sumsq_mapping: dict[str, object] = {}
    if raw_sumsq_path:
        raw_sumsq_mapping, raw_missing = load_raw_sumsq_dict(str(raw_sumsq_path), selected)
        print(
            f"[iterate] AWQ raw-sumsq override: {len(raw_sumsq_mapping)}/{len(selected)} "
            f"tensors keyed from {raw_sumsq_path}",
            flush=True,
        )
        if raw_missing:
            print(
                f"[iterate] {len(raw_missing)} F1 tensors not in raw-sumsq npz; "
                f"falling back to Hessian-diagonal for those. First 3: {raw_missing[:3]}",
                flush=True,
            )

    for round_index in range(int(args.max_rounds)):
        round_started = time.time()
        round_dir = out_dir / f"round_{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)
        stats_npz, stats_json = provider(args, round_index, previous_model, round_dir)
        hessians = load_round_hessians(str(stats_npz), selected)
        if raw_sumsq_mapping:
            raw_scales = compute_awq_scale_dict_with_raw(
                hessians, raw_sumsq_mapping, alpha=args.awq_alpha
            )
        else:
            raw_scales = compute_awq_scale_dict(hessians, alpha=args.awq_alpha)
        damped_scales = damp_awq_scale_dict(previous_scales, raw_scales, damping=args.damping)
        scale_delta = relative_l2_delta(previous_scales, damped_scales)
        scales_npz = round_dir / "awq_scales.npz"
        save_awq_scales_npz(scales_npz, raw_scales, damped_scales)
        write_json(
            round_dir / "scales_delta.json",
            {
                "round": round_index,
                "relative_l2_vs_prev": scale_delta,
                "epsilon": float(args.epsilon),
                "converged": round_index > 0 and scale_delta < float(args.epsilon),
            },
            pretty=True,
        )
        model_path = quantize_iterate_round(
            round_dir=round_dir,
            base_hfq=base_hfq,
            hf_model=args.hf_model,
            mask_path=args.imatrix_mask,
            stats_npz=stats_npz,
            stats_json=stats_json,
            scales=damped_scales,
            gpu=args.gpu,
            gptq_damp=args.gptq_damp,
            gptq_refit_iters=args.gptq_refit_iters,
        )
        bench = run_round_bench(args, round_dir, model_path)
        elapsed = time.time() - round_started
        record = {
            "round": round_index,
            "imatrix_npz": str(stats_npz),
            "awq_scales_npz": str(scales_npz),
            "model": str(model_path),
            "scale_delta": scale_delta,
            "elapsed_seconds": elapsed,
            "bench": bench,
        }
        write_round_summary(round_dir, record)
        trace.append(record)
        previous_scales = damped_scales
        previous_model = str(model_path)
        if round_index > 0 and scale_delta < float(args.epsilon):
            break

    result = {
        "schema": SCHEMA_ITERATE,
        "captured_at_utc": utc_now(),
        "hf_model": args.hf_model,
        "mask": args.imatrix_mask,
        "base_hfq": base_hfq,
        "base_output_dir": str(out_dir),
        "awq_alpha": float(args.awq_alpha),
        "damping": float(args.damping),
        "epsilon": float(args.epsilon),
        "max_rounds": int(args.max_rounds),
        "elapsed_seconds": time.time() - started_all,
        "rounds": trace,
    }
    write_json(out_dir / "iterate-summary.json", result, pretty=True)
    print("round\tscale_delta\tmodel")
    for record in trace:
        print(f"{record['round']}\t{record['scale_delta']:.8g}\t{record['model']}")
    return result


def run_iterative_awq_gptq_with_stats_sequence(
    *,
    hf_model: str,
    mask_path: str,
    base_output_dir: str,
    stats_sequence: list[dict[str, object]],
    awq_alpha: float = 0.5,
    damping: float = 0.5,
    epsilon: float = 0.01,
    max_rounds: int = 6,
    gpu: int | None = None,
    gptq_damp: float = 0.01,
    gptq_refit_iters: int = 2,
):
    if not stats_sequence:
        raise ValueError("stats_sequence must contain at least round 0")

    def provider(_args, round_index, _previous_model, round_dir):
        import numpy as np

        stats = stats_sequence[min(round_index, len(stats_sequence) - 1)]
        payload = {safe_key(name): np.asarray(value, dtype=np.float64) for name, value in stats.items()}
        stats_npz = round_dir / "imatrix.npz"
        stats_json = round_dir / "stats.json"
        np.savez_compressed(stats_npz, **payload)
        stats_json.write_text(
            json.dumps({"counts": {name: 1 for name in stats}}, indent=2, sort_keys=True) + "\n"
        )
        return stats_npz, stats_json

    args = SimpleNamespace(
        hf_model=hf_model,
        calib_text="",
        imatrix_mask=mask_path,
        base_output_dir=base_output_dir,
        awq_alpha=awq_alpha,
        damping=damping,
        epsilon=epsilon,
        max_rounds=max_rounds,
        bench_each_round=False,
        bench_ref=None,
        gpu=gpu,
        collect_devices="",
        ctx=256,
        chunks=1,
        offset=0,
        eval_bin="target/release/examples/eval_hipfire",
        kv_mode="q8",
        scoring_mode="prefill",
        bench_max_chunks=20,
        gptq_damp=gptq_damp,
        gptq_refit_iters=gptq_refit_iters,
    )
    return run_iterative_awq_gptq(args, stats_provider=provider)


def iterate_awq_gptq(args):
    run_iterative_awq_gptq(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mask", help="extract KMD2-sensitive tensor mask from two HFQ files")
    p.add_argument("--base", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pretty", action="store_true")
    p.set_defaults(func=extract_mask)

    p = sub.add_parser("collect-stats", help="collect rotated activation stats with PyTorch")
    p.add_argument("--hf-model", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--calib-text", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--device-map", choices=("none", "auto", "balanced", "balanced_low_0", "sequential"), default="none")
    p.add_argument("--max-memory", help="Comma-separated Accelerate max_memory map, e.g. 0=30GiB,1=30GiB,cpu=64GiB")
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--chunks", type=int, default=32)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--stats-mode", choices=("diag", "hessian"), default="diag")
    p.add_argument("--candidate-mq4", help="Collect stats from a quantized MQ4 HFQ candidate via CPU dequantized forward")
    p.add_argument("--max-tensors", type=int)
    p.add_argument("--tensor-filter")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=collect_stats)

    p = sub.add_parser("quantize", help="write same-size flat MQ4 candidate from BF16 source and stats")
    p.add_argument("--base", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--stats-npz")
    p.add_argument("--stats-json", help="Optional stats.json with activation counts; defaults to stats-npz sibling")
    p.add_argument("--output", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clip-ratio", type=float, default=1.0)
    p.add_argument("--ls-iters", type=int, default=5)
    p.add_argument("--method", choices=("wls", "gptq"), default="wls")
    p.add_argument("--gpu", type=int, help="Run GPTQ solve on CUDA device N; omit for CPU numpy path")
    p.add_argument("--awq-aware-hessian", help="HFQ model containing AWQ .awq_scale.weight sidecars for GPTQ Hessian scaling")
    p.add_argument("--gptq-damp", type=float, default=0.01)
    p.add_argument("--gptq-refit-iters", type=int, default=2)
    p.add_argument("--skip-unsupported", action="store_true")
    p.add_argument("--max-tensors", type=int)
    p.add_argument("--tensor-filter")
    p.add_argument("--exclude-tensor-filter")
    p.add_argument("--tensor-name", action="append")
    p.add_argument("--tensor-list")
    p.add_argument(
        "--sort-by",
        choices=("mask-order", "name", "numel-asc", "numel-desc", "bytes-asc", "bytes-desc"),
        default="mask-order",
    )
    p.add_argument("--progress-every", type=int, default=1, help="Print quantize progress every N tensors; 0 disables")
    p.set_defaults(func=quantize_candidate)

    p = sub.add_parser("compose", help="copy base HFQ and patch selected tensor payloads from a source HFQ")
    p.add_argument("--base", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tensor-name", action="append")
    p.add_argument("--tensor-list")
    p.set_defaults(func=compose_candidate)

    p = sub.add_parser("tensor-sweep", help="evaluate one selected tensor patch per temporary candidate")
    p.add_argument("--base", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--candidate-json", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--eval-bin", default="target/release/examples/eval_hipfire")
    p.add_argument("--kv-mode", default="q8")
    p.add_argument("--scoring-mode", default="prefill")
    p.add_argument("--max-chunks", type=int, default=20)
    p.add_argument("--max-tensors", type=int)
    p.add_argument("--tensor-filter")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=tensor_sweep)

    p = sub.add_parser("iterate", help="run iterative AWQ+GPTQ fixed-point refinement")
    p.add_argument("--hf-model", required=True)
    p.add_argument("--calib-text", required=True)
    p.add_argument("--imatrix-mask", required=True)
    p.add_argument("--base-output-dir", required=True)
    p.add_argument("--awq-alpha", type=float, default=0.5)
    p.add_argument("--damping", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--bench-each-round", action="store_true")
    p.add_argument("--bench-ref")
    p.add_argument("--gpu", type=int)
    p.add_argument("--collect-devices", default="0,1,2,3")
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--chunks", type=int, default=32)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--eval-bin", default="target/release/examples/eval_hipfire")
    p.add_argument("--kv-mode", default="q8")
    p.add_argument("--scoring-mode", default="prefill")
    p.add_argument("--bench-max-chunks", type=int, default=20)
    p.add_argument("--gptq-damp", type=float, default=0.01)
    p.add_argument("--gptq-refit-iters", type=int, default=2)
    p.add_argument(
        "--initial-stats-npz",
        default=None,
        help="Use this stats-merged.npz as round-0 imatrix; bypasses in-process collection.",
    )
    p.add_argument(
        "--initial-stats-json",
        default=None,
        help="Optional stats.json paired with --initial-stats-npz.",
    )
    p.add_argument(
        "--awq-raw-sumsq-npz",
        default=None,
        help=(
            "Per-input-channel raw sum(x²) npz keyed by safe_key(hfq_name). "
            "When given, AWQ scale derivation reads from this directly "
            "instead of taking np.diagonal of the rotated Hessian — "
            "the latter mixes channels under the FWHT and yields wrong "
            "per-channel statistics (see investigation 2026-05-18). "
            "Produce one via scripts/convert_gguf_imatrix_to_npz.py from "
            "a llama.cpp GGUF imatrix file."
        ),
    )
    p.set_defaults(func=iterate_awq_gptq)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
