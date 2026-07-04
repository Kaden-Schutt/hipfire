#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire - see LICENSE and NOTICE in the project root.

"""Extract a PEFT-style LoRA adapter from two aligned safetensors models.

Given a target/fine-tuned model and a base model with matching tensor names,
this computes delta = target - base for selected rank-2 tensors, approximates
each delta with a rank-r randomized SVD, and writes:

  adapter_model.safetensors
  adapter_config.json
  extraction_report.json

The tool is CPU-only and processes one tensor at a time. Embeddings and lm_head
and vision tower tensors are excluded by default because they are very large
or less portable across PEFT loaders; pass --include-embeddings,
--include-lm-head, or --include-vision to opt in.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:
    raise SystemExit(
        "error: extract_lora_from_safetensor_delta.py requires torch and safetensors "
        "(install them in the active Python environment)"
    ) from exc


INDEX_NAME = "model.safetensors.index.json"
DEFAULT_EXCLUDE_RE = r"(?:embed_tokens|lm_head)"


@dataclass
class TensorRef:
    name: str
    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass
class TensorReport:
    name: str
    shape: tuple[int, int]
    rank: int
    delta_l2: float
    residual_l2: float
    relative_residual_l2: float
    captured_energy: float
    rmse: float
    lora_a: str
    lora_b: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only extraction of a PEFT LoRA adapter from target-base safetensors deltas."
    )
    parser.add_argument("target", help="Target/fine-tuned model directory, index file, or safetensors file.")
    parser.add_argument("base", help="Base model directory, index file, or safetensors file.")
    parser.add_argument("out_dir", help="Output adapter directory.")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank to extract (default: 64).")
    parser.add_argument(
        "--alpha",
        type=float,
        help="LoRA alpha. Defaults to rank, making PEFT scale alpha/r equal 1.",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=16,
        help="Randomized SVD oversampling dimension (default: 16).",
    )
    parser.add_argument(
        "--power-iters",
        type=int,
        default=1,
        help="Randomized SVD power iterations; 0 is fastest, 1-2 improves quality (default: 1).",
    )
    parser.add_argument("--include", help="Regex; only matching tensor names are considered.")
    parser.add_argument("--exclude", help="Regex; matching tensor names are skipped after default exclusions.")
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Allow embed_tokens tensors. Off by default.",
    )
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="Allow lm_head tensors. Off by default.",
    )
    parser.add_argument(
        "--include-vision",
        action="store_true",
        help="Allow model.visual tensors. Off by default for text LoRA extraction.",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="Debug aid: process at most this many selected tensors.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Adapter tensor dtype to write (default: bfloat16).",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="CPU computation dtype (default: float32).",
    )
    parser.add_argument(
        "--base-model-name-or-path",
        default=None,
        help="Value for adapter_config.json base_model_name_or_path. Defaults to the base argument.",
    )
    parser.add_argument(
        "--peft-prefix",
        default="base_model.model.",
        help="Prefix for adapter state-dict keys (default: base_model.model.).",
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Fail if out_dir already contains adapter_model.safetensors or adapter_config.json.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for randomized SVD.")
    parser.add_argument("--threads", type=int, default=0, help="Set torch CPU thread count.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list selected tensors; do not compute SVD or write adapter files.",
    )
    return parser.parse_args()


def resolve_model(source: str) -> dict[str, TensorRef]:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        index = path / INDEX_NAME
        if index.exists():
            return refs_from_index(index)
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise SystemExit(f"error: no {INDEX_NAME} or *.safetensors files found in {path}")
        return refs_from_files(shards)
    if path.name == INDEX_NAME:
        return refs_from_index(path)
    if path.is_file() and path.suffix == ".safetensors":
        return refs_from_files([path])
    raise SystemExit(f"error: unsupported model source: {source}")


def refs_from_index(index: Path) -> dict[str, TensorRef]:
    with index.open("r", encoding="utf-8") as f:
        data = json.load(f)
    root = index.parent
    refs: dict[str, TensorRef] = {}
    for name, shard_name in data.get("weight_map", {}).items():
        refs[name] = tensor_ref_from_file(name, root / shard_name)
    return refs


def refs_from_files(paths: list[Path]) -> dict[str, TensorRef]:
    refs: dict[str, TensorRef] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                refs[name] = tensor_ref_from_handle(name, path, handle)
    return refs


def tensor_ref_from_file(name: str, path: Path) -> TensorRef:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return tensor_ref_from_handle(name, path, handle)


def tensor_ref_from_handle(name: str, path: Path, handle: Any) -> TensorRef:
    tensor_slice = handle.get_slice(name)
    return TensorRef(
        name=name,
        path=path,
        shape=tuple(int(v) for v in tensor_slice.get_shape()),
        dtype=str(tensor_slice.get_dtype()),
    )


def read_tensor(ref: TensorRef, dtype: torch.dtype) -> torch.Tensor:
    with safe_open(ref.path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(ref.name)
    return tensor.to(dtype=dtype)


def selected_tensor_names(
    args: argparse.Namespace, target_refs: dict[str, TensorRef], base_refs: dict[str, TensorRef]
) -> list[str]:
    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None
    default_excludes = []
    if not args.include_embeddings:
        default_excludes.append("embed_tokens")
    if not args.include_lm_head:
        default_excludes.append("lm_head")
    if not args.include_vision:
        default_excludes.append("model.visual.")

    names = []
    for name in sorted(set(target_refs) & set(base_refs)):
        target = target_refs[name]
        base = base_refs[name]
        if len(target.shape) != 2:
            continue
        if target.shape != base.shape:
            continue
        if default_excludes and any(token in name for token in default_excludes):
            continue
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        names.append(name)
    return names[: args.max_tensors] if args.max_tensors else names


def randomized_svd(
    delta: torch.Tensor, rank: int, oversample: int, power_iters: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = delta.shape
    k = min(rank + oversample, m, n)
    if rank > k:
        raise ValueError(f"rank {rank} cannot exceed smallest tensor dimension {min(m, n)}")
    omega = torch.randn(n, k, dtype=delta.dtype)
    q, _ = torch.linalg.qr(delta @ omega, mode="reduced")
    for _ in range(power_iters):
        z, _ = torch.linalg.qr(delta.T @ q, mode="reduced")
        q, _ = torch.linalg.qr(delta @ z, mode="reduced")
    small = q.T @ delta
    u_hat, s, vh = torch.linalg.svd(small, full_matrices=False)
    u = q @ u_hat
    return u[:, :rank].contiguous(), s[:rank].contiguous(), vh[:rank, :].contiguous()


def factors_from_svd(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_s = torch.sqrt(torch.clamp(s, min=0.0))
    # PEFT applies delta = (alpha / r) * (B @ A). We write unscaled factors and
    # default alpha=r, so the nominal scale is 1.
    lora_b = u * sqrt_s.unsqueeze(0)
    lora_a = sqrt_s.unsqueeze(1) * vh
    return lora_a.contiguous(), lora_b.contiguous()


def adapter_key(prefix: str, tensor_name: str, suffix: str) -> str:
    if not tensor_name.endswith(".weight"):
        raise ValueError(f"selected tensor is not a weight tensor: {tensor_name}")
    module = tensor_name[: -len(".weight")]
    return f"{prefix}{module}.{suffix}.weight"


def leaf_module_name(tensor_name: str) -> str:
    if not tensor_name.endswith(".weight"):
        return tensor_name.rsplit(".", 1)[-1]
    return tensor_name[: -len(".weight")].rsplit(".", 1)[-1]


def output_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def compute_tensor_report(
    name: str,
    target_ref: TensorRef,
    base_ref: TensorRef,
    args: argparse.Namespace,
    compute_dtype: torch.dtype,
    write_dtype: torch.dtype,
) -> tuple[TensorReport, dict[str, torch.Tensor]]:
    target = read_tensor(target_ref, compute_dtype)
    base = read_tensor(base_ref, compute_dtype)
    delta = target - base
    delta_l2_sq = float(torch.sum(delta * delta).item())
    delta_l2 = math.sqrt(delta_l2_sq)
    del target
    del base

    rank = min(args.rank, min(delta.shape))
    u, s, vh = randomized_svd(delta, rank, args.oversample, args.power_iters)
    lora_a, lora_b = factors_from_svd(u, s, vh)
    approx = lora_b @ lora_a
    residual = delta - approx
    residual_l2 = math.sqrt(float(torch.sum(residual * residual).item()))
    captured = float(torch.sum(s * s).item()) / delta_l2_sq if delta_l2_sq else 1.0
    rmse = residual_l2 / math.sqrt(delta.numel()) if delta.numel() else 0.0

    a_key = adapter_key(args.peft_prefix, name, "lora_A")
    b_key = adapter_key(args.peft_prefix, name, "lora_B")
    report = TensorReport(
        name=name,
        shape=(int(delta.shape[0]), int(delta.shape[1])),
        rank=rank,
        delta_l2=delta_l2,
        residual_l2=residual_l2,
        relative_residual_l2=(residual_l2 / delta_l2) if delta_l2 else 0.0,
        captured_energy=max(0.0, min(1.0, captured)),
        rmse=rmse,
        lora_a=a_key,
        lora_b=b_key,
    )
    tensors = {
        a_key: lora_a.to(dtype=write_dtype),
        b_key: lora_b.to(dtype=write_dtype),
    }
    return report, tensors


def adapter_config(args: argparse.Namespace, selected: list[str]) -> dict[str, Any]:
    target_modules = sorted({leaf_module_name(name) for name in selected})
    alpha = args.alpha if args.alpha is not None else float(args.rank)
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": args.base_model_name_or_path or args.base,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": args.rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }


def report_json(args: argparse.Namespace, selected: list[str], reports: list[TensorReport]) -> dict[str, Any]:
    total_delta_sq = sum(r.delta_l2 * r.delta_l2 for r in reports)
    total_resid_sq = sum(r.residual_l2 * r.residual_l2 for r in reports)
    total_elements = sum(r.shape[0] * r.shape[1] for r in reports)
    return {
        "target": args.target,
        "base": args.base,
        "rank": args.rank,
        "alpha": args.alpha if args.alpha is not None else float(args.rank),
        "selected_tensors": len(selected),
        "processed_tensors": len(reports),
        "global": {
            "delta_l2": math.sqrt(total_delta_sq),
            "residual_l2": math.sqrt(total_resid_sq),
            "relative_residual_l2": math.sqrt(total_resid_sq / total_delta_sq) if total_delta_sq else 0.0,
            "captured_energy": 1.0 - (total_resid_sq / total_delta_sq) if total_delta_sq else 1.0,
            "rmse": math.sqrt(total_resid_sq / total_elements) if total_elements else 0.0,
            "elements": total_elements,
        },
        "tensors": [
            {
                "name": r.name,
                "shape": list(r.shape),
                "rank": r.rank,
                "delta_l2": r.delta_l2,
                "residual_l2": r.residual_l2,
                "relative_residual_l2": r.relative_residual_l2,
                "captured_energy": r.captured_energy,
                "rmse": r.rmse,
                "lora_A": r.lora_a,
                "lora_B": r.lora_b,
            }
            for r in reports
        ],
    }


def prepare_out_dir(path: Path, no_clobber: bool) -> None:
    if no_clobber:
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            if (path / name).exists():
                raise SystemExit(f"error: refusing to overwrite {path / name}")
    path.mkdir(parents=True, exist_ok=True)


def print_selection(selected: list[str], target_refs: dict[str, TensorRef]) -> None:
    print(f"Selected {len(selected)} rank-2 tensors:")
    for name in selected:
        print(f"  {name} {list(target_refs[name].shape)}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    target_refs = resolve_model(args.target)
    base_refs = resolve_model(args.base)
    selected = selected_tensor_names(args, target_refs, base_refs)
    if not selected:
        raise SystemExit("error: no rank-2 common tensors selected")
    if args.dry_run:
        print_selection(selected, target_refs)
        return 0

    out_dir = Path(args.out_dir).expanduser().resolve()
    prepare_out_dir(out_dir, args.no_clobber)
    compute_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
    write_dtype = output_dtype(args.dtype)

    adapter_tensors: dict[str, torch.Tensor] = {}
    reports: list[TensorReport] = []
    for index, name in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] extracting {name} {list(target_refs[name].shape)}", file=sys.stderr)
        report, tensors = compute_tensor_report(
            name, target_refs[name], base_refs[name], args, compute_dtype, write_dtype
        )
        reports.append(report)
        adapter_tensors.update(tensors)
        print(
            f"    captured={report.captured_energy:.6f} "
            f"rel_resid={report.relative_residual_l2:.6f} rmse={report.rmse:.8g}",
            file=sys.stderr,
        )

    save_file(adapter_tensors, out_dir / "adapter_model.safetensors")
    (out_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config(args, selected), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = report_json(args, selected, reports)
    (out_dir / "extraction_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    readme = out_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Extracted LoRA Adapter\n\n"
            "This adapter was generated by `scripts/extract_lora_from_safetensor_delta.py`.\n"
            "See `extraction_report.json` for reconstruction metrics.\n",
            encoding="utf-8",
        )

    global_metrics = payload["global"]
    print("")
    print(f"Wrote {out_dir}")
    print(
        f"Global reconstruction: captured_energy={global_metrics['captured_energy']:.6f} "
        f"relative_residual_l2={global_metrics['relative_residual_l2']:.6f} "
        f"rmse={global_metrics['rmse']:.8g}"
    )
    print(f"Adapter tensors: {len(adapter_tensors)}")
    print(
        f"Size: {shutil.disk_usage(out_dir).used if False else sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file()) / (1024 * 1024):.2f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
