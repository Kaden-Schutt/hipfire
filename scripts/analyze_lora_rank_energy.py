#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire - see LICENSE and NOTICE in the project root.

"""CPU-only rank-energy scan for safetensors model deltas.

This analyzes delta = target - base for selected rank-2 tensors and estimates
how much delta energy is captured at candidate LoRA ranks. It does not write an
adapter; use it to decide whether uniform-rank LoRA is sensible or whether a
mixed-rank / distillation route is needed.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
except ImportError as exc:
    raise SystemExit(
        "error: analyze_lora_rank_energy.py requires torch and safetensors "
        "(install them in the active Python environment)"
    ) from exc


INDEX_NAME = "model.safetensors.index.json"


@dataclass
class TensorRef:
    name: str
    path: Path
    shape: tuple[int, ...]
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze low-rank energy in safetensors weight deltas.")
    parser.add_argument("target", help="Target/fine-tuned model directory, index file, or safetensors file.")
    parser.add_argument("base", help="Base model directory, index file, or safetensors file.")
    parser.add_argument(
        "--ranks",
        default="16,32,64,128,256",
        help="Comma-separated ranks to report. One max-rank SVD is used per tensor (default: 16,32,64,128,256).",
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
        help="Randomized SVD power iterations (default: 1).",
    )
    parser.add_argument("--include", help="Regex; only matching tensor names are considered.")
    parser.add_argument("--exclude", help="Regex; matching tensor names are skipped after default exclusions.")
    parser.add_argument("--include-embeddings", action="store_true", help="Allow embed_tokens tensors.")
    parser.add_argument("--include-lm-head", action="store_true", help="Allow lm_head tensors.")
    parser.add_argument("--include-vision", action="store_true", help="Allow model.visual tensors.")
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="Debug aid: process at most this many selected tensors.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="CPU computation dtype (default: float32).",
    )
    parser.add_argument(
        "--thresholds",
        default="0.5,0.8,0.9",
        help="Comma-separated captured-energy thresholds for mixed-rank suggestions (default: 0.5,0.8,0.9).",
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    parser.add_argument("--csv-out", help="Optional CSV output path.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for randomized SVD.")
    parser.add_argument("--threads", type=int, default=0, help="Set torch CPU thread count.")
    parser.add_argument("--dry-run", action="store_true", help="Only list selected tensors.")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or values[0] <= 0:
        raise SystemExit("error: --ranks must contain positive integers")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or values[0] <= 0.0 or values[-1] >= 1.0:
        raise SystemExit("error: --thresholds must be between 0 and 1")
    return values


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


def selected_tensor_names(args: argparse.Namespace, target_refs: dict[str, TensorRef], base_refs: dict[str, TensorRef]) -> list[str]:
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


def tensor_group(name: str) -> str:
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name:
        return "attention"
    if ".linear_attn." in name:
        return "linear_attention"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "model.visual." in name:
        return "vision"
    return "other"


def randomized_singular_values(delta: torch.Tensor, max_rank: int, oversample: int, power_iters: int) -> torch.Tensor:
    m, n = delta.shape
    k = min(max_rank + oversample, m, n)
    omega = torch.randn(n, k, dtype=delta.dtype)
    q, _ = torch.linalg.qr(delta @ omega, mode="reduced")
    for _ in range(power_iters):
        z, _ = torch.linalg.qr(delta.T @ q, mode="reduced")
        q, _ = torch.linalg.qr(delta @ z, mode="reduced")
    small = q.T @ delta
    return torch.linalg.svdvals(small)[: min(max_rank, k)].contiguous()


def analyze_tensor(
    name: str,
    target_ref: TensorRef,
    base_ref: TensorRef,
    ranks: list[int],
    thresholds: list[float],
    compute_dtype: torch.dtype,
    oversample: int,
    power_iters: int,
) -> dict[str, Any]:
    target = read_tensor(target_ref, compute_dtype)
    base = read_tensor(base_ref, compute_dtype)
    delta = target - base
    del target
    del base

    max_rank = min(max(ranks), min(delta.shape))
    delta_l2_sq = float(torch.sum(delta * delta).item())
    delta_l2 = math.sqrt(delta_l2_sq)
    singular_values = randomized_singular_values(delta, max_rank, oversample, power_iters)
    energy = torch.cumsum(singular_values * singular_values, dim=0)
    rank_metrics = {}
    for rank in ranks:
        effective_rank = min(rank, len(singular_values))
        captured = float(energy[effective_rank - 1].item()) / delta_l2_sq if delta_l2_sq and effective_rank else 1.0
        captured = max(0.0, min(1.0, captured))
        rank_metrics[str(rank)] = {
            "effective_rank": effective_rank,
            "captured_energy": captured,
            "relative_residual_l2": math.sqrt(max(0.0, 1.0 - captured)),
        }
    threshold_ranks = {}
    for threshold in thresholds:
        found = None
        for idx, value in enumerate(energy.tolist(), start=1):
            if delta_l2_sq and value / delta_l2_sq >= threshold:
                found = idx
                break
        threshold_ranks[str(threshold)] = found
    return {
        "name": name,
        "group": tensor_group(name),
        "shape": list(delta.shape),
        "elements": delta.numel(),
        "delta_l2": delta_l2,
        "delta_l2_sq": delta_l2_sq,
        "max_analyzed_rank": len(singular_values),
        "rank_metrics": rank_metrics,
        "threshold_ranks": threshold_ranks,
    }


def aggregate(results: list[dict[str, Any]], ranks: list[int]) -> dict[str, Any]:
    groups = sorted({r["group"] for r in results})
    out: dict[str, Any] = {"global": aggregate_subset(results, ranks)}
    out["groups"] = {group: aggregate_subset([r for r in results if r["group"] == group], ranks) for group in groups}
    return out


def aggregate_subset(results: list[dict[str, Any]], ranks: list[int]) -> dict[str, Any]:
    total = sum(r["delta_l2_sq"] for r in results)
    rank_metrics = {}
    for rank in ranks:
        captured_sq = sum(r["delta_l2_sq"] * r["rank_metrics"][str(rank)]["captured_energy"] for r in results)
        captured = captured_sq / total if total else 1.0
        rank_metrics[str(rank)] = {
            "captured_energy": captured,
            "relative_residual_l2": math.sqrt(max(0.0, 1.0 - captured)),
        }
    return {
        "tensors": len(results),
        "elements": sum(r["elements"] for r in results),
        "delta_l2": math.sqrt(total),
        "rank_metrics": rank_metrics,
    }


def write_csv(path: str, results: list[dict[str, Any]], ranks: list[int], thresholds: list[float]) -> None:
    fieldnames = ["name", "group", "rows", "cols", "delta_l2", "max_analyzed_rank"]
    fieldnames += [f"captured_r{rank}" for rank in ranks]
    fieldnames += [f"rel_resid_r{rank}" for rank in ranks]
    fieldnames += [f"rank_for_{threshold:g}" for threshold in thresholds]
    with Path(path).expanduser().open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {
                "name": item["name"],
                "group": item["group"],
                "rows": item["shape"][0],
                "cols": item["shape"][1],
                "delta_l2": item["delta_l2"],
                "max_analyzed_rank": item["max_analyzed_rank"],
            }
            for rank in ranks:
                row[f"captured_r{rank}"] = item["rank_metrics"][str(rank)]["captured_energy"]
                row[f"rel_resid_r{rank}"] = item["rank_metrics"][str(rank)]["relative_residual_l2"]
            for threshold in thresholds:
                row[f"rank_for_{threshold:g}"] = item["threshold_ranks"][str(threshold)]
            writer.writerow(row)


def print_summary(results: list[dict[str, Any]], ranks: list[int], aggregates: dict[str, Any]) -> None:
    print(f"Analyzed {len(results)} rank-2 tensors.")
    print("")
    print("Global weighted capture:")
    for rank in ranks:
        metric = aggregates["global"]["rank_metrics"][str(rank)]
        print(
            f"  r{rank:<4d} captured={metric['captured_energy']:.6f} "
            f"rel_resid={metric['relative_residual_l2']:.6f}"
        )
    print("")
    print("By group:")
    for group, data in aggregates["groups"].items():
        parts = [
            f"r{rank}={data['rank_metrics'][str(rank)]['captured_energy']:.4f}"
            for rank in ranks
        ]
        print(f"  {group:16s} tensors={data['tensors']:3d} " + " ".join(parts))
    print("")
    worst = sorted(results, key=lambda r: r["rank_metrics"][str(max(ranks))]["captured_energy"])[:10]
    print(f"Worst tensors at r{max(ranks)}:")
    for item in worst:
        metric = item["rank_metrics"][str(max(ranks))]
        print(f"  captured={metric['captured_energy']:.6f} rel={metric['relative_residual_l2']:.6f} {item['name']}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    ranks = parse_int_list(args.ranks)
    thresholds = parse_float_list(args.thresholds)
    compute_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
    target_refs = resolve_model(args.target)
    base_refs = resolve_model(args.base)
    selected = selected_tensor_names(args, target_refs, base_refs)
    if not selected:
        raise SystemExit("error: no rank-2 common tensors selected")
    if args.dry_run:
        print(f"Selected {len(selected)} rank-2 tensors:")
        for name in selected:
            print(f"  {name} {list(target_refs[name].shape)}")
        return 0

    results = []
    for idx, name in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] analyzing {name} {list(target_refs[name].shape)}", file=sys.stderr)
        item = analyze_tensor(
            name,
            target_refs[name],
            base_refs[name],
            ranks,
            thresholds,
            compute_dtype,
            args.oversample,
            args.power_iters,
        )
        results.append(item)
        max_metric = item["rank_metrics"][str(max(ranks))]
        print(
            f"    r{max(ranks)} captured={max_metric['captured_energy']:.6f} "
            f"rel_resid={max_metric['relative_residual_l2']:.6f}",
            file=sys.stderr,
        )

    aggregates = aggregate(results, ranks)
    payload = {
        "target": args.target,
        "base": args.base,
        "ranks": ranks,
        "thresholds": thresholds,
        "selected_tensors": len(selected),
        "oversample": args.oversample,
        "power_iters": args.power_iters,
        "aggregate": aggregates,
        "tensors": results,
    }
    print_summary(results, ranks, aggregates)
    if args.json_out:
        Path(args.json_out).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")
    if args.csv_out:
        write_csv(args.csv_out, results, ranks, thresholds)
        print(f"Wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
