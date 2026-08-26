#!/usr/bin/env python3
"""Compare opt-in Gemma4 logit traces and optional full FP32 dumps."""

from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import sys


def load_trace(path: Path) -> tuple[dict[int, dict], int]:
    rows = {}
    versions = set()
    vocabs = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            version = row.get("schema_version", 0)
            if type(version) is not int:
                raise ValueError(f"invalid schema_version={version!r} in {path}")
            if version not in (0, 1):
                raise ValueError(f"unsupported schema_version={version} in {path}")
            versions.add(version)
            missing = {"step", "vocab", "top1_token", "top_k"} - row.keys()
            if missing:
                raise ValueError(f"missing fields {sorted(missing)} in {path}")
            step = row["step"]
            if type(step) is not int or step < 0:
                raise ValueError(f"invalid step={step!r} in {path}")
            vocab = row["vocab"]
            if type(vocab) is not int or vocab <= 0:
                raise ValueError(f"invalid vocab={vocab!r} in {path}")
            vocabs.add(vocab)
            if version == 1:
                missing_v1 = {
                    "request_id",
                    "sampled_token",
                    "non_finite_logits",
                    "routes",
                } - row.keys()
                if missing_v1:
                    raise ValueError(f"missing v1 fields {sorted(missing_v1)} in {path}")
                if not isinstance(row["request_id"], str) or not row["request_id"]:
                    raise ValueError(f"invalid request_id in {path}")
                expected_routes = {
                    "batched_embedding_prefill",
                    "ple_branch_batched_prefill",
                    "ple_activation_fused_prefill",
                }
                routes = row["routes"]
                if not isinstance(routes, dict) or not expected_routes <= routes.keys():
                    raise ValueError(f"invalid routes at step {step} in {path}")
                if any(type(routes[key]) is not bool for key in expected_routes):
                    raise ValueError(f"non-boolean route at step {step} in {path}")
            non_finite = row.get("non_finite_logits", 0)
            if type(non_finite) is not int or not 0 <= non_finite <= vocab:
                raise ValueError(f"invalid non_finite_logits={non_finite} in {path}")
            sampled_token = row.get("sampled_token")
            if "sampled_token" in row and (
                type(sampled_token) is not int or not 0 <= sampled_token < vocab
            ):
                raise ValueError(f"invalid sampled_token={sampled_token!r} in {path}")
            if not isinstance(row["top_k"], list):
                raise ValueError(f"invalid top_k at step {row['step']} in {path}")
            if not row["top_k"] and non_finite == 0:
                raise ValueError(f"empty top_k without numerical failure in {path}")
            seen_tokens = set()
            for entry in row["top_k"]:
                if not isinstance(entry, dict) or "token" not in entry:
                    raise ValueError(f"invalid top_k entry at step {row['step']} in {path}")
                token = entry["token"]
                if type(token) is not int or not 0 <= token < vocab:
                    raise ValueError(f"invalid top_k token={token!r} in {path}")
                if token in seen_tokens:
                    raise ValueError(f"duplicate top_k token={token} in {path}")
                seen_tokens.add(token)
                if ("logit" in entry) == ("logit_special" in entry):
                    raise ValueError(f"invalid top_k logit encoding in {path}")
                if "logit" in entry and (
                    type(entry["logit"]) not in (int, float)
                    or not math.isfinite(entry["logit"])
                ):
                    raise ValueError(f"invalid finite logit in {path}")
                if "logit_special" in entry and entry["logit_special"] not in {
                    "+inf",
                    "-inf",
                }:
                    raise ValueError(f"invalid special logit in {path}")
            expected_top1 = row["top_k"][0]["token"] if row["top_k"] else None
            if row["top1_token"] is not None and type(row["top1_token"]) is not int:
                raise ValueError(f"invalid top1_token at step {step} in {path}")
            if row["top1_token"] != expected_top1:
                raise ValueError(f"top1/top_k mismatch at step {row['step']} in {path}")
            if step in rows:
                raise ValueError(f"duplicate step {step} in {path}")
            rows[step] = row
    if len(versions) > 1:
        raise ValueError(f"mixed schema versions {sorted(versions)} in {path}")
    if len(vocabs) > 1:
        raise ValueError(f"mixed vocab sizes {sorted(vocabs)} in {path}")
    return rows, next(iter(versions), 0)


def load_f32(path: Path) -> array:
    if path.stat().st_size % 4:
        raise ValueError(f"partial FP32 dump: {path} has {path.stat().st_size} bytes")
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // values.itemsize)
    if values.itemsize != 4:
        raise RuntimeError(f"unexpected float size: {values.itemsize}")
    if sys.byteorder != "little":
        values.byteswap()
    return values


def rank(values: array, token: int) -> int:
    target = values[token]
    return 1 + sum(value > target for value in values)


def compare_full(baseline_path: Path, candidate_path: Path, top_k: int) -> dict:
    baseline = load_f32(baseline_path)
    candidate = load_f32(candidate_path)
    if len(baseline) != len(candidate):
        raise ValueError(f"vocab mismatch: {len(baseline)} != {len(candidate)}")
    for label, values in (("baseline", baseline), ("candidate", candidate)):
        non_finite = sum(not math.isfinite(value) for value in values)
        if non_finite:
            raise ValueError(f"{label} dump contains {non_finite} non-finite logits")

    sum_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    baseline_norm = 0.0
    candidate_norm = 0.0
    max_abs = -1.0
    max_token = 0
    exact = 0
    for token, (left, right) in enumerate(zip(baseline, candidate)):
        delta = abs(float(left) - float(right))
        sum_abs += delta
        sum_sq += delta * delta
        dot += float(left) * float(right)
        baseline_norm += float(left) * float(left)
        candidate_norm += float(right) * float(right)
        exact += left == right
        if delta > max_abs:
            max_abs = delta
            max_token = token

    baseline_top = sorted(range(len(baseline)), key=baseline.__getitem__, reverse=True)[:top_k]
    candidate_top = sorted(range(len(candidate)), key=candidate.__getitem__, reverse=True)[:top_k]
    baseline_top1 = baseline_top[0]
    candidate_top1 = candidate_top[0]
    return {
        "vocab": len(baseline),
        "exact_values": exact,
        "exact_fraction": exact / len(baseline),
        "max_abs": max_abs,
        "max_abs_token": max_token,
        "mean_abs": sum_abs / len(baseline),
        "rms": math.sqrt(sum_sq / len(baseline)),
        "cosine": dot / math.sqrt(baseline_norm * candidate_norm),
        "baseline_top1": baseline_top1,
        "candidate_top1": candidate_top1,
        "baseline_top1_logit": baseline[baseline_top1],
        "candidate_top1_logit": candidate[candidate_top1],
        "baseline_top1_rank_in_candidate": rank(candidate, baseline_top1),
        "candidate_top1_rank_in_baseline": rank(baseline, candidate_top1),
        "top_k": top_k,
        "top_k_overlap": len(set(baseline_top) & set(candidate_top)),
        "baseline_top": [
            {"token": token, "logit": baseline[token]} for token in baseline_top
        ],
        "candidate_top": [
            {"token": token, "logit": candidate[token]} for token in candidate_top
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--baseline-full", type=Path)
    parser.add_argument("--candidate-full", type=Path)
    parser.add_argument("--full-step", type=int)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    baseline, baseline_version = load_trace(args.baseline_trace)
    candidate, candidate_version = load_trace(args.candidate_trace)
    if baseline_version != candidate_version:
        raise ValueError(
            f"trace schema mismatch: {baseline_version} != {candidate_version}"
        )
    common = sorted(set(baseline) & set(candidate))
    if not common:
        raise ValueError("cannot compare empty logit traces")
    for step in common:
        if baseline[step]["vocab"] != candidate[step]["vocab"]:
            raise ValueError(
                f"vocab mismatch at step {step}: "
                f"{baseline[step]['vocab']} != {candidate[step]['vocab']}"
            )
    missing_from_baseline = sorted(set(candidate) - set(baseline))
    missing_from_candidate = sorted(set(baseline) - set(candidate))
    step_set_divergence = min(
        missing_from_baseline + missing_from_candidate,
        default=None,
    )
    def top_k_tokens(record):
        return [entry.get("token") for entry in record.get("top_k", [])]

    top_k_value_divergence = next(
        (
            step
            for step in common
            if baseline[step].get("top_k") != candidate[step].get("top_k")
        ),
        None,
    )
    top_k_order_divergence = next(
        (
            step
            for step in common
            if top_k_tokens(baseline[step]) != top_k_tokens(candidate[step])
        ),
        None,
    )
    top_k_membership_divergence = next(
        (
            step
            for step in common
            if set(top_k_tokens(baseline[step]))
            != set(top_k_tokens(candidate[step]))
        ),
        None,
    )
    divergence = next(
        (
            step
            for step in common
            if baseline[step].get("top1_token") != candidate[step].get("top1_token")
        ),
        None,
    )
    sampled_metric_available = all(
        "sampled_token" in baseline[step] and "sampled_token" in candidate[step]
        for step in common
    )
    sampled_divergence = (
        next(
            (
                step
                for step in common
                if baseline[step]["sampled_token"]
                != candidate[step]["sampled_token"]
            ),
            None,
        )
        if sampled_metric_available
        else None
    )
    report = {
        "baseline_trace": str(args.baseline_trace),
        "candidate_trace": str(args.candidate_trace),
        "trace_schema_version": baseline_version,
        "baseline_steps": len(baseline),
        "candidate_steps": len(candidate),
        "common_steps": len(common),
        "last_common_step": common[-1],
        "trace_step_sets_equal": not (
            missing_from_baseline or missing_from_candidate
        ),
        "first_step_set_divergence": step_set_divergence,
        "missing_from_baseline": missing_from_baseline,
        "missing_from_candidate": missing_from_candidate,
        "first_top_k_value_divergence": top_k_value_divergence,
        "first_top_k_token_order_divergence": top_k_order_divergence,
        "first_top_k_membership_divergence": top_k_membership_divergence,
        "first_top1_divergence": divergence,
        "sampled_token_metric_available": sampled_metric_available,
        "first_sampled_token_divergence": sampled_divergence,
        "baseline_numerical_failure_steps": [
            step
            for step in sorted(baseline)
            if baseline[step].get("non_finite_logits", 0) > 0
        ],
        "candidate_numerical_failure_steps": [
            step
            for step in sorted(candidate)
            if candidate[step].get("non_finite_logits", 0) > 0
        ],
        "at_first_top_k_token_order_divergence": {
            "baseline": baseline.get(top_k_order_divergence),
            "candidate": candidate.get(top_k_order_divergence),
        },
        "at_first_top1_divergence": {
            "baseline": baseline.get(divergence),
            "candidate": candidate.get(divergence),
        },
        "at_first_sampled_token_divergence": {
            "baseline": baseline.get(sampled_divergence),
            "candidate": candidate.get(sampled_divergence),
        },
    }
    if bool(args.baseline_full) != bool(args.candidate_full):
        parser.error("--baseline-full and --candidate-full must be provided together")
    if args.baseline_full:
        if args.full_step is None:
            parser.error("--full-step is required with full-logit dumps")
        if args.full_step not in baseline or args.full_step not in candidate:
            parser.error(f"--full-step {args.full_step} is absent from a trace")
        expected_suffix = f".step-{args.full_step}.f32le"
        if not str(args.baseline_full).endswith(expected_suffix) or not str(
            args.candidate_full
        ).endswith(expected_suffix):
            parser.error(f"full-logit filenames must end in {expected_suffix}")
        report["full_logits_step"] = args.full_step
        report["full_logits"] = compare_full(
            args.baseline_full, args.candidate_full, args.top_k
        )
        if report["full_logits"]["vocab"] != baseline[args.full_step]["vocab"]:
            raise ValueError(
                "full-logit vocab does not match the trace: "
                f"{report['full_logits']['vocab']} != "
                f"{baseline[args.full_step]['vocab']}"
            )

    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
