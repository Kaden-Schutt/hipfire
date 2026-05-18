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


SCHEMA_MASK = "hipfire.astrea.mq4_masked.mask.v0"
SCHEMA_STATS = "hipfire.astrea.mq4_masked.stats.v0"
SCHEMA_CANDIDATE = "hipfire.astrea.mq4_masked.candidate.v0"
SCHEMA_TENSOR_SWEEP = "hipfire.astrea.mq4_masked.tensor_sweep.v0"
SCHEMA_COMPOSE = "hipfire.astrea.mq4_masked.compose.v0"


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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
