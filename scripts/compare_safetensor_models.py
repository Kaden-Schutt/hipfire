#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read
# hipfire - see LICENSE and NOTICE in the project root.

"""CPU-only weight-delta comparison for safetensors model directories.

This utility compares a target model against one or more candidate parent
models by tensor name. It streams tensors in row chunks on CPU and reports
global and grouped L2/RMSE/MAE/cosine metrics without loading a full model.
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
    from safetensors import safe_open
except ImportError as exc:
    raise SystemExit(
        "error: compare_safetensor_models.py requires torch and safetensors "
        "(install them in the active Python environment)"
    ) from exc


INDEX_NAME = "model.safetensors.index.json"


@dataclass
class TensorRef:
    name: str
    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass
class Accumulator:
    tensors: int = 0
    elements: int = 0
    target_l2_sq: float = 0.0
    candidate_l2_sq: float = 0.0
    delta_l2_sq: float = 0.0
    dot: float = 0.0
    abs_sum: float = 0.0
    max_abs: float = 0.0

    def add_tensor(self, other: "Accumulator") -> None:
        self.tensors += other.tensors
        self.elements += other.elements
        self.target_l2_sq += other.target_l2_sq
        self.candidate_l2_sq += other.candidate_l2_sq
        self.delta_l2_sq += other.delta_l2_sq
        self.dot += other.dot
        self.abs_sum += other.abs_sum
        self.max_abs = max(self.max_abs, other.max_abs)

    def metrics(self) -> dict[str, Any]:
        rmse = math.sqrt(self.delta_l2_sq / self.elements) if self.elements else 0.0
        target_l2 = math.sqrt(self.target_l2_sq)
        delta_l2 = math.sqrt(self.delta_l2_sq)
        candidate_l2 = math.sqrt(self.candidate_l2_sq)
        denom = target_l2 * candidate_l2
        return {
            "tensors": self.tensors,
            "elements": self.elements,
            "delta_l2": delta_l2,
            "target_l2": target_l2,
            "relative_l2": (delta_l2 / target_l2) if target_l2 else 0.0,
            "rmse": rmse,
            "mae": (self.abs_sum / self.elements) if self.elements else 0.0,
            "max_abs": self.max_abs,
            "cosine": (self.dot / denom) if denom else 0.0,
        }


@dataclass
class CandidateResult:
    label: str
    path: str
    totals: Accumulator = field(default_factory=Accumulator)
    groups: dict[str, Accumulator] = field(default_factory=dict)
    top_tensors: list[dict[str, Any]] = field(default_factory=list)
    skipped_missing: list[str] = field(default_factory=list)
    skipped_shape: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only tensor-wise difference metrics for safetensors models.")
    parser.add_argument("target", help="Target/fine-tuned model directory, index file, or safetensors file.")
    parser.add_argument("candidates", nargs="+", help="Candidate parent model directories/indexes/files.")
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for candidates, in the same order as candidate paths.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=512,
        help="Rows to read per chunk for tensors with rank >= 2 (default: 512).",
    )
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Only compare rank-2 tensors, useful for LoRA-relevant parent selection.",
    )
    parser.add_argument("--include", help="Regex; only tensor names matching it are compared.")
    parser.add_argument("--exclude", help="Regex; tensor names matching it are skipped.")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of largest per-tensor delta contributors to print and save (default: 20).",
    )
    parser.add_argument("--json-out", help="Optional path to write full JSON metrics.")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Set torch CPU thread count. 0 leaves torch default unchanged.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug aid: compare at most this many selected tensors.",
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
        shard = root / shard_name
        refs[name] = tensor_ref_from_file(name, shard)
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


def select_names(
    target_refs: dict[str, TensorRef],
    candidate_refs: list[dict[str, TensorRef]],
    linear_only: bool,
    include: str | None,
    exclude: str | None,
    limit: int,
) -> list[str]:
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None
    common = set(target_refs)
    for refs in candidate_refs:
        common &= set(refs)
    names = []
    for name in sorted(common):
        ref = target_refs[name]
        if linear_only and len(ref.shape) != 2:
            continue
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue
        names.append(name)
    return names[:limit] if limit else names


def tensor_group(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp"
    if "embed_tokens" in name:
        return "embedding"
    if name == "lm_head.weight" or ".lm_head." in name:
        return "lm_head"
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name:
        return "attention"
    if ".linear_attn." in name:
        return "linear_attention"
    if "norm" in name or "layernorm" in name:
        return "norm"
    return "other"


def read_chunk(ref: TensorRef, start: int, end: int) -> torch.Tensor:
    with safe_open(ref.path, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(ref.name)
        if len(ref.shape) == 0:
            tensor = tensor_slice[:]
        elif len(ref.shape) == 1:
            tensor = tensor_slice[start:end]
        else:
            tensor = tensor_slice[start:end]
    return tensor.to(dtype=torch.float32)


def compare_tensor(target: TensorRef, candidate: TensorRef, chunk_rows: int) -> Accumulator:
    rows = target.shape[0] if target.shape else 1
    step = max(1, chunk_rows) if len(target.shape) >= 2 else rows
    acc = Accumulator(tensors=1)
    for start in range(0, rows, step):
        end = min(rows, start + step)
        target_chunk = read_chunk(target, start, end)
        candidate_chunk = read_chunk(candidate, start, end)
        delta = target_chunk - candidate_chunk
        acc.elements += delta.numel()
        acc.target_l2_sq += float(torch.sum(target_chunk * target_chunk).item())
        acc.candidate_l2_sq += float(torch.sum(candidate_chunk * candidate_chunk).item())
        acc.delta_l2_sq += float(torch.sum(delta * delta).item())
        acc.dot += float(torch.sum(target_chunk * candidate_chunk).item())
        acc.abs_sum += float(torch.sum(torch.abs(delta)).item())
        acc.max_abs = max(acc.max_abs, float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0)
    return acc


def compare_candidate(
    label: str,
    path: str,
    target_refs: dict[str, TensorRef],
    candidate_refs: dict[str, TensorRef],
    names: list[str],
    chunk_rows: int,
    top: int,
) -> CandidateResult:
    result = CandidateResult(label=label, path=path)
    top_items: list[dict[str, Any]] = []
    for index, name in enumerate(names, start=1):
        target = target_refs[name]
        candidate = candidate_refs[name]
        if target.shape != candidate.shape:
            result.skipped_shape.append(name)
            continue
        acc = compare_tensor(target, candidate, chunk_rows)
        result.totals.add_tensor(acc)
        group = tensor_group(name)
        result.groups.setdefault(group, Accumulator()).add_tensor(acc)
        metrics = acc.metrics()
        top_items.append(
            {
                "name": name,
                "shape": list(target.shape),
                "group": group,
                "delta_l2": metrics["delta_l2"],
                "relative_l2": metrics["relative_l2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_abs": metrics["max_abs"],
                "delta_l2_sq": acc.delta_l2_sq,
            }
        )
        if index % 25 == 0:
            print(f"  {label}: compared {index}/{len(names)} tensors", file=sys.stderr)
    top_items.sort(key=lambda item: item["delta_l2_sq"], reverse=True)
    result.top_tensors = top_items[:top]
    return result


def result_to_json(result: CandidateResult) -> dict[str, Any]:
    return {
        "label": result.label,
        "path": result.path,
        "totals": result.totals.metrics(),
        "groups": {name: acc.metrics() for name, acc in sorted(result.groups.items())},
        "top_tensors": result.top_tensors,
        "skipped_missing": result.skipped_missing,
        "skipped_shape": result.skipped_shape,
    }


def print_summary(selected: list[str], results: list[CandidateResult], top: int) -> None:
    print(f"Compared {len(selected)} common tensors.")
    print("")
    print("Overall closeness to target:")
    for result in sorted(results, key=lambda r: r.totals.metrics()["relative_l2"]):
        metrics = result.totals.metrics()
        print(
            f"  {result.label}: rel_l2={metrics['relative_l2']:.8f} "
            f"rmse={metrics['rmse']:.8g} mae={metrics['mae']:.8g} "
            f"cosine={metrics['cosine']:.10f} tensors={metrics['tensors']} "
            f"elements={metrics['elements']}"
        )
    print("")
    for result in results:
        print(f"{result.label} by group:")
        for name, acc in sorted(result.groups.items()):
            metrics = acc.metrics()
            print(
                f"  {name:16s} rel_l2={metrics['relative_l2']:.8f} "
                f"rmse={metrics['rmse']:.8g} tensors={metrics['tensors']}"
            )
        if top > 0:
            print(f"{result.label} top delta tensors:")
            for item in result.top_tensors[:top]:
                print(
                    f"  {item['delta_l2']:12.5g} rel={item['relative_l2']:.8f} rmse={item['rmse']:.8g} {item['name']}"
                )
        print("")


def default_label(path: str) -> str:
    resolved = Path(path).expanduser()
    if resolved.name == INDEX_NAME:
        return resolved.parent.name
    return resolved.name


def main() -> int:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if len(args.labels or []) not in (0, len(args.candidates)):
        raise SystemExit("error: --labels must have the same count as candidates")

    target_refs = resolve_model(args.target)
    candidate_refs = [resolve_model(path) for path in args.candidates]
    selected = select_names(
        target_refs,
        candidate_refs,
        linear_only=args.linear_only,
        include=args.include,
        exclude=args.exclude,
        limit=args.limit,
    )
    if not selected:
        raise SystemExit("error: no common tensors selected for comparison")

    labels = args.labels or [default_label(path) for path in args.candidates]
    results = []
    for label, path, refs in zip(labels, args.candidates, candidate_refs):
        print(f"Comparing target to {label}...", file=sys.stderr)
        results.append(compare_candidate(label, path, target_refs, refs, selected, args.chunk_rows, args.top))

    print_summary(selected, results, args.top)
    if args.json_out:
        payload = {
            "target": str(Path(args.target).expanduser()),
            "selected_tensors": len(selected),
            "linear_only": args.linear_only,
            "include": args.include,
            "exclude": args.exclude,
            "chunk_rows": args.chunk_rows,
            "results": [result_to_json(result) for result in results],
        }
        Path(args.json_out).expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
