#!/usr/bin/env python3
"""Probe/import ParoQuant checkpoints into hipfire HFQ.

This is a runtime-enablement bridge, not a tuned quantizer. It consumes the
`RotateQuantizedLinear` safetensors layout emitted by z-lab/paroquant:

  base.qweight, base.qzeros, base.scales, base.pairs, base.theta,
  base.channel_scales

and writes `base.weight` as HFQ quant_type=28 (PARO4G128). The runtime kernel
then applies ParoQuant's activation channel scaling + pairwise rotations and
performs a slow-correct W4 GEMV. The qtype-28 payload intentionally preserves
the native Paro/AWQ tensor layout instead of lowering to hipfire's HFQ row-major
layout:

  qweight:int32[K, M/8], qzeros:int32[K/128, M/8], scales:f16[K/128, M],
  pairs:int16[8, K], theta:f16[8, K/2], channel_scales:f16[K]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path


PARO_PROBE_SCHEMA = "hipfire.astrea.paro_probe.v0"
PARO_IMPORT_SCHEMA = "hipfire.astrea.paro_import.v0"

GROUP_SIZE = 128
BITS = 4
PACK = 32 // BITS
KROT = 8
AWQ_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_INV_REORDER = tuple(AWQ_REORDER.index(i) for i in range(PACK))

PARO_SUFFIXES = (
    ".qweight",
    ".qzeros",
    ".scales",
    ".pairs",
    ".theta",
    ".channel_scales",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_deps():
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        from safetensors import safe_open  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "paro import requires numpy, torch, and safetensors in the active Python environment"
        ) from exc


def resolve_model(model: str, *, local_only: bool = False) -> Path:
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if local_only:
        raise FileNotFoundError(f"model path is not a directory and --local-only was set: {model}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            repo_type="model",
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json", "*.model"],
        )
    )


def safetensor_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*.safetensors") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no .safetensors files found under {root}")
    return files


def build_tensor_index(root: Path) -> dict[str, Path]:
    from safetensors import safe_open

    index: dict[str, Path] = {}
    for path in safetensor_files(root):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in index:
                    raise ValueError(f"duplicate safetensors key {key}: {index[key]} and {path}")
                index[key] = path
    return index


def load_tensor(index: dict[str, Path], name: str):
    from safetensors import safe_open

    path = index.get(name)
    if path is None:
        raise KeyError(name)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def tensor_meta(index: dict[str, Path], name: str) -> dict:
    from safetensors import safe_open

    path = index[name]
    with safe_open(str(path), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(name)
    return {
        "name": name,
        "file": str(path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def paro_base_from_key(name: str) -> str | None:
    return name.removesuffix(".qweight") if name.endswith(".qweight") else None


def companion_name(base: str, suffix: str) -> str:
    return f"{base}.{suffix}"


def is_paro_companion(name: str, quant_bases: set[str]) -> bool:
    return any(name == f"{base}{suffix}" for base in quant_bases for suffix in PARO_SUFFIXES)


def discover_paro_modules(index: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    modules = []
    incomplete = []
    for name in sorted(index):
        base = paro_base_from_key(name)
        if base is None:
            continue
        required = [f"{base}{suffix}" for suffix in PARO_SUFFIXES]
        missing = [key for key in required if key not in index]
        if missing:
            incomplete.append({"base": base, "missing": missing})
            continue
        qweight = load_tensor(index, f"{base}.qweight")
        qzeros = load_tensor(index, f"{base}.qzeros")
        scales = load_tensor(index, f"{base}.scales")
        pairs = load_tensor(index, f"{base}.pairs")
        theta = load_tensor(index, f"{base}.theta")
        channel_scales = load_tensor(index, f"{base}.channel_scales")
        in_features = int(qweight.shape[0])
        out_features = int(qweight.shape[1]) * PACK
        groups = int(qzeros.shape[0])
        group_size = in_features // groups if groups else None
        modules.append(
            {
                "base": base,
                "hfq_name": f"{base}.weight",
                "in_features": in_features,
                "out_features": out_features,
                "group_size": group_size,
                "bits": BITS,
                "krot": int(theta.shape[0]) if len(theta.shape) else None,
                "qweight_shape": list(qweight.shape),
                "qzeros_shape": list(qzeros.shape),
                "scales_shape": list(scales.shape),
                "pairs_shape": list(pairs.shape),
                "theta_shape": list(theta.shape),
                "channel_scales_shape": list(channel_scales.shape),
            }
        )
    return modules, incomplete


def read_json_if_exists(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def arch_id_from_config(config: dict | None) -> tuple[int, str]:
    config = config or {}
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    arch_str = config.get("model_type") or text_config.get("model_type") or "llama"
    arch_id = {
        "llama": 0,
        "qwen2": 1,
        "qwen3": 1,
        "qwen3_5": 5,
        "qwen3_5_text": 5,
        "qwen3_5_moe": 6,
        "qwen3_5_moe_text": 6,
    }.get(str(arch_str), 0)
    return arch_id, str(arch_str)


def paro_quant_contract(layout: str) -> tuple[int, str]:
    if layout == "native":
        return 28, "PARO4G128"
    if layout == "engine":
        return 29, "PARO4G128T"
    raise ValueError(f"unknown Paro payload layout: {layout}")


def metadata_bytes(source_dir: Path, *, layout: str = "native") -> tuple[bytes, int, str]:
    config = read_json_if_exists(source_dir / "config.json") or {}
    arch_id, arch_str = arch_id_from_config(config)
    tokenizer_path = source_dir / "tokenizer.json"
    tokenizer = tokenizer_path.read_text(encoding="utf-8") if tokenizer_path.is_file() else "{}"
    tokenizer_config = read_json_if_exists(source_dir / "tokenizer_config.json")
    quant_type, quant_type_name = paro_quant_contract(layout)
    metadata = {
        "architecture": arch_str,
        "config": config,
        "tokenizer": tokenizer,
        "tokenizer_config": tokenizer_config,
        "paroquant_import": {
            "format": quant_type_name,
            "quant_type": quant_type,
            "layout": layout,
            "theta_encoding": "sincos_f32" if layout == "engine" else "theta_f16",
            "group_size": GROUP_SIZE,
            "bits": BITS,
            "krot": KROT,
        },
    }
    return json.dumps(metadata, separators=(",", ":")).encode("utf-8"), arch_id, arch_str


def validate_module(base: str, qweight, qzeros, scales, pairs, theta, channel_scales) -> tuple[int, int, int]:
    in_features = int(qweight.shape[0])
    out_features = int(qweight.shape[1]) * PACK
    groups = int(qzeros.shape[0])
    if qweight.ndim != 2:
        raise ValueError(f"{base}.qweight must be rank-2, got {tuple(qweight.shape)}")
    if qzeros.ndim != 2:
        raise ValueError(f"{base}.qzeros must be rank-2, got {tuple(qzeros.shape)}")
    if scales.ndim != 2:
        raise ValueError(f"{base}.scales must be rank-2, got {tuple(scales.shape)}")
    if in_features % GROUP_SIZE != 0:
        raise ValueError(f"{base} in_features={in_features} is not divisible by {GROUP_SIZE}")
    if groups != in_features // GROUP_SIZE:
        raise ValueError(
            f"{base}.qzeros groups={groups} does not match in_features/group_size={in_features // GROUP_SIZE}"
        )
    if list(qzeros.shape) != [groups, out_features // PACK]:
        raise ValueError(f"{base}.qzeros shape mismatch: {tuple(qzeros.shape)}")
    if list(scales.shape) != [groups, out_features]:
        raise ValueError(f"{base}.scales shape mismatch: {tuple(scales.shape)}")
    if list(pairs.shape) != [KROT, in_features]:
        raise ValueError(f"{base}.pairs shape mismatch: {tuple(pairs.shape)}")
    if list(theta.shape) != [KROT, in_features // 2]:
        raise ValueError(f"{base}.theta shape mismatch: {tuple(theta.shape)}")
    if channel_scales.numel() != in_features:
        raise ValueError(f"{base}.channel_scales numel mismatch: {channel_scales.numel()} != {in_features}")
    return out_features, in_features, groups


def pack_paro_payload(index: dict[str, Path], base: str, out_file, *, layout: str = "native") -> tuple[int, list[int]]:
    import torch

    qweight_t = load_tensor(index, f"{base}.qweight")
    qzeros_t = load_tensor(index, f"{base}.qzeros")
    scales_t = load_tensor(index, f"{base}.scales")
    pairs_t = load_tensor(index, f"{base}.pairs")
    theta_t = load_tensor(index, f"{base}.theta")
    channel_scales_t = load_tensor(index, f"{base}.channel_scales")

    out_features, in_features, groups = validate_module(
        base, qweight_t, qzeros_t, scales_t, pairs_t, theta_t, channel_scales_t
    )

    pairs_min = int(pairs_t.min().item())
    pairs_max = int(pairs_t.max().item())
    if pairs_min < 0 or pairs_max >= GROUP_SIZE:
        raise ValueError(f"{base}.pairs contain indices outside local 0..{GROUP_SIZE - 1}")

    qweight = qweight_t.to(dtype=torch.int32)
    if layout == "engine":
        qweight = qweight.t().contiguous()
    elif layout == "native":
        qweight = qweight.contiguous()
    else:
        raise ValueError(f"unknown Paro payload layout: {layout}")
    qzeros = qzeros_t.to(dtype=torch.int32).contiguous()
    scales = scales_t.to(dtype=torch.float16).contiguous()
    pairs = pairs_t.to(dtype=torch.int16).contiguous()
    if layout == "engine":
        theta_f32 = theta_t.to(dtype=torch.float32)
        theta = torch.stack((torch.sin(theta_f32), torch.cos(theta_f32)), dim=-1).contiguous()
    else:
        theta = theta_t.to(dtype=torch.float16).contiguous()
    channel_scales = channel_scales_t.reshape(-1).to(dtype=torch.float16).contiguous()

    payload_start = out_file.tell()
    for tensor in (qweight, qzeros, scales, pairs, theta, channel_scales):
        out_file.write(tensor.numpy().tobytes(order="C"))
    payload_size = out_file.tell() - payload_start
    return payload_size, [out_features, in_features]


def quantize_q8f16_payload(tensor) -> bytes:
    import numpy as np
    import torch

    arr = tensor.to(dtype=torch.float32).contiguous().cpu().numpy()
    if arr.shape[-1] % 32 != 0:
        raise ValueError(f"Q8F16 copy requires last dimension divisible by 32, got shape={arr.shape}")
    groups = arr.reshape(-1, 32)
    max_abs = np.max(np.abs(groups), axis=1)
    scale = (max_abs / 127.0).astype(np.float32)
    inv = np.zeros_like(scale)
    np.divide(1.0, scale, out=inv, where=scale > 0.0)
    q = np.rint(groups * inv[:, None]).clip(-128, 127).astype(np.int8)
    out = np.empty((groups.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes(order="C")


def tensor_to_copy_payload(index: dict[str, Path], name: str, *, copy_floats: str = "f16") -> tuple[bytes, list[int], int, int]:
    import torch

    tensor = load_tensor(index, name)
    if not tensor.is_floating_point():
        raise ValueError(f"non-floating safetensors entry cannot be copied into HFQ F16: {name} ({tensor.dtype})")
    shape = [int(x) for x in tensor.shape]
    if copy_floats == "q8" and tensor.ndim >= 2 and shape[-1] % 32 == 0:
        return quantize_q8f16_payload(tensor), shape, 3, 32
    if copy_floats != "f16" and copy_floats != "q8":
        raise ValueError(f"unknown copied-float mode: {copy_floats}")
    tensor = tensor.to(dtype=torch.float16).contiguous()
    return tensor.numpy().tobytes(order="C"), shape, 1, 0


def write_hfq_from_spill(
    output: Path,
    *,
    version: int,
    arch_id: int,
    metadata: bytes,
    records: list[dict],
    spill_path: Path,
):
    index = bytearray()
    index += struct.pack("<I", len(records))
    for record in records:
        raw_name = record["name"].encode("utf-8")
        index += struct.pack("<H", len(raw_name))
        index += raw_name
        index += struct.pack("<B", int(record["quant_type"]))
        index += struct.pack("<B", len(record["shape"]))
        for dim in record["shape"]:
            index += struct.pack("<I", int(dim))
        index += struct.pack("<I", int(record["group_size"]))
        index += struct.pack("<Q", int(record["data_size"]))

    metadata_offset = 32
    data_start_unaligned = metadata_offset + len(metadata) + len(index)
    data_offset = (data_start_unaligned + 4095) & ~4095
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as dst:
        dst.write(b"HFQM")
        dst.write(struct.pack("<I", version))
        dst.write(struct.pack("<I", arch_id))
        dst.write(struct.pack("<I", len(records)))
        dst.write(struct.pack("<Q", metadata_offset))
        dst.write(struct.pack("<Q", data_offset))
        dst.write(metadata)
        dst.write(index)
        dst.write(bytes(data_offset - data_start_unaligned))
        with spill_path.open("rb") as src:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def probe_model(model: str, *, local_only: bool = False, max_modules: int | None = None) -> dict:
    require_deps()
    source_dir = resolve_model(model, local_only=local_only)
    index = build_tensor_index(source_dir)
    modules, incomplete = discover_paro_modules(index)
    complete = modules[:max_modules] if max_modules else modules
    config = read_json_if_exists(source_dir / "config.json") or {}
    arch_id, arch_str = arch_id_from_config(config)
    compatibility_warnings = []
    if "vision_config" in config:
        compatibility_warnings.append(
            "checkpoint has vision_config; text-only hipfire smoke may still need Qwen3.5-VL prompt/runtime parity before quality data is meaningful"
        )
    quant_types = {}
    for name in index:
        suffix = next((s for s in PARO_SUFFIXES if name.endswith(s)), "other")
        quant_types[suffix.removeprefix(".")] = quant_types.get(suffix.removeprefix("."), 0) + 1
    return {
        "schema": PARO_PROBE_SCHEMA,
        "captured_at_utc": utc_now(),
        "model": model,
        "resolved_path": str(source_dir),
        "tensor_count": len(index),
        "arch_id": arch_id,
        "architecture": arch_str,
        "paro": {
            "complete_module_count": len(modules),
            "incomplete_module_count": len(incomplete),
            "suffix_counts": dict(sorted(quant_types.items())),
            "modules": complete,
            "modules_truncated": max_modules is not None and len(modules) > max_modules,
            "incomplete": incomplete[:max_modules] if max_modules else incomplete,
        },
        "runtime_contract": {
            "hfq_quant_type": 28,
            "hfq_quant_type_name": "PARO4G128",
            "group_size": GROUP_SIZE,
            "bits": BITS,
            "krot": KROT,
            "has_vision_config": "vision_config" in config,
            "compatibility_warnings": compatibility_warnings,
            "engine_support_required": [
                "DType::PARO4G128",
                "quant_type 28 loader mapping",
                "gemv_paro4g128 and gemv_paro4g128_residual kernels",
            ],
        },
    }


def import_model(
    model: str,
    output: str,
    *,
    local_only: bool = False,
    max_modules: int | None = None,
    layout: str = "native",
    copy_floats: str = "f16",
) -> dict:
    require_deps()
    quant_type, quant_type_name = paro_quant_contract(layout)
    source_dir = resolve_model(model, local_only=local_only)
    index = build_tensor_index(source_dir)
    modules, incomplete = discover_paro_modules(index)
    if incomplete:
        raise ValueError(f"{len(incomplete)} Paro qweight tensors are missing required companions")
    if not modules:
        raise ValueError(f"no complete ParoQuant modules found in {source_dir}")
    selected_bases = {m["base"] for m in (modules[:max_modules] if max_modules else modules)}
    all_quant_bases = {m["base"] for m in modules}
    metadata, arch_id, arch_str = metadata_bytes(source_dir, layout=layout)
    config = read_json_if_exists(source_dir / "config.json") or {}
    compatibility_warnings = []
    if "vision_config" in config:
        compatibility_warnings.append(
            "checkpoint has vision_config; successful HFQ import/load does not prove text-generation quality until Qwen3.5-VL parity is checked"
        )

    output_path = Path(output).expanduser()
    records: list[dict] = []
    imported_modules = []
    copied_tensors = 0
    copied_q8_tensors = 0
    skipped_tensors = []

    with tempfile.NamedTemporaryFile(prefix="paroquant-import-", suffix=".spill", delete=False) as tmp:
        spill_path = Path(tmp.name)
        try:
            for name in sorted(index):
                base = paro_base_from_key(name)
                if base is not None:
                    if base not in selected_bases:
                        skipped_tensors.append({"name": name, "reason": "outside max_modules selection"})
                        continue
                    payload_size, shape = pack_paro_payload(index, base, tmp, layout=layout)
                    records.append(
                        {
                            "name": f"{base}.weight",
                            "quant_type": quant_type,
                            "shape": shape,
                            "group_size": GROUP_SIZE,
                            "data_size": payload_size,
                        }
                    )
                    imported_modules.append({"base": base, "hfq_name": f"{base}.weight", "shape": shape, "data_size": payload_size})
                    continue
                if is_paro_companion(name, all_quant_bases):
                    continue
                try:
                    payload, shape, copied_qt, copied_group_size = tensor_to_copy_payload(index, name, copy_floats=copy_floats)
                except ValueError as exc:
                    skipped_tensors.append({"name": name, "reason": str(exc)})
                    continue
                tmp.write(payload)
                records.append(
                    {
                        "name": name,
                        "quant_type": copied_qt,
                        "shape": shape,
                        "group_size": copied_group_size,
                        "data_size": len(payload),
                    }
                )
                copied_tensors += 1
                if copied_qt == 3:
                    copied_q8_tensors += 1
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                spill_path.unlink()
            raise

    try:
        write_hfq_from_spill(
            output_path,
            version=1,
            arch_id=arch_id,
            metadata=metadata,
            records=records,
            spill_path=spill_path,
        )
    finally:
        try:
            spill_path.unlink()
        except FileNotFoundError:
            pass

    return {
        "schema": PARO_IMPORT_SCHEMA,
        "captured_at_utc": utc_now(),
        "model": model,
        "resolved_path": str(source_dir),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_md5": md5_file(output_path),
        "arch_id": arch_id,
        "architecture": arch_str,
        "hfq_quant_type": quant_type,
        "hfq_quant_type_name": quant_type_name,
        "layout": layout,
        "copied_float_mode": copy_floats,
        "has_vision_config": "vision_config" in config,
        "compatibility_warnings": compatibility_warnings,
        "imported_module_count": len(imported_modules),
        "copied_f16_tensor_count": copied_tensors,
        "copied_q8_tensor_count": copied_q8_tensors,
        "skipped_tensor_count": len(skipped_tensors),
        "imported_modules": imported_modules[:32],
        "imported_modules_truncated": len(imported_modules) > 32,
        "skipped_tensors": skipped_tensors[:32],
        "skipped_tensors_truncated": len(skipped_tensors) > 32,
        "next_steps": [
            f"build the runtime with {quant_type_name} enabled",
            "run a short AR coherence smoke before any quality/perf claim",
            "send successful AR rows to Atlas before optimizing fused PARO kernels",
        ],
    }


def write_json(payload: dict, *, pretty: bool = False, out: str | None = None):
    text = json.dumps(payload, indent=2, sort_keys=True) if pretty else json.dumps(payload, sort_keys=True)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe/import ParoQuant safetensors into hipfire HFQ.")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="Inspect a ParoQuant checkpoint.")
    probe.add_argument("--model", required=True)
    probe.add_argument("--local-only", action="store_true")
    probe.add_argument("--max-modules", type=int)
    probe.add_argument("--pretty", action="store_true")
    probe.add_argument("--out")
    imp = sub.add_parser("import", help="Write a runtime-loadable PARO4G128 HFQ file.")
    imp.add_argument("--model", required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--local-only", action="store_true")
    imp.add_argument("--max-modules", type=int)
    imp.add_argument("--layout", choices=("native", "engine"), default="native")
    imp.add_argument("--copy-floats", choices=("f16", "q8"), default="f16")
    imp.add_argument("--pretty", action="store_true")
    imp.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        write_json(
            probe_model(args.model, local_only=args.local_only, max_modules=args.max_modules),
            pretty=args.pretty,
            out=args.out,
        )
    elif args.command == "import":
        write_json(
            import_model(
                args.model,
                args.output,
                local_only=args.local_only,
                max_modules=args.max_modules,
                layout=args.layout,
                copy_floats=args.copy_floats,
            ),
            pretty=args.pretty,
            out=args.out,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
