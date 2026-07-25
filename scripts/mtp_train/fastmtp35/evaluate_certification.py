#!/usr/bin/env python3
"""Evaluate the sampled FastMTP promotion run and fail closed."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


LABELS = ("ar", "stock-mtp", "candidate-mtp")
QUALITY_FLAGS = ("runaway", "empty", "attractor")
CACHE_MIN_FRACTION = {
    # AR may legitimately lose part of an old assistant span when its visible
    # transcript differs from the retained hidden-reasoning cache, but it must
    # still reuse a substantial prefix on every continuation.
    "ar": 0.50,
    # Native MTP's verbatim assistant-body splice should be a strict extension.
    "stock-mtp": 0.80,
    "candidate-mtp": 0.80,
}
EXPECTED_CONTRACT = {
    "turns": 8,
    "sampling": "registry",
    "thinking": "med",
    "max_tokens": 4096,
    "kv_mode": "q8",
    "mtp_k": 3,
    "redline_shadow_iterations": 15,
}


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = numeric(rows, "decode_tok_s")
    taus = numeric(rows, "tau")
    post_first = rows[1:]
    cache_fractions = [
        float(row["cached"]) / float(row["ctx"])
        for row in post_first
        if isinstance(row.get("cached"), (int, float))
        and isinstance(row.get("ctx"), (int, float))
        and float(row["ctx"]) > 0
    ]
    return {
        "turns": len(rows),
        "median_decode_tok_s": statistics.median(rates) if rates else None,
        "mean_decode_tok_s": statistics.mean(rates) if rates else None,
        "median_tau": statistics.median(taus) if taus else None,
        "mean_tau": statistics.mean(taus) if taus else None,
        "cached_turns_after_first": sum(
            isinstance(row.get("cached"), (int, float))
            and float(row["cached"]) > 0
            for row in post_first
        ),
        "min_cached_fraction_after_first": (
            min(cache_fractions) if cache_fractions else None
        ),
        "recall_hits": sum(
            int(row.get("recall_hits", 0))
            for row in rows
            if isinstance(row.get("recall_hits", 0), (int, float))
        ),
        "recall_total": sum(
            int(row.get("recall_total", 0))
            for row in rows
            if isinstance(row.get("recall_total", 0), (int, float))
        ),
        "native_decode_measurements": sum(
            not bool(row.get("decode_estimated"))
            and isinstance(row.get("decode_tok_s"), (int, float))
            for row in rows
        ),
        **{
            flag: sum(bool(row.get(flag)) for row in rows)
            for flag in QUALITY_FLAGS
        },
    }


def evaluate(root: Path, expected_turns: int = 8) -> dict[str, Any]:
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    failures: list[str] = []

    certification_manifest_path = root / "certification-manifest.json"
    if certification_manifest_path.is_file():
        certification_manifest = json.loads(certification_manifest_path.read_text())
        summary["certification_manifest"] = certification_manifest
        if certification_manifest.get("schema_version") != 1:
            failures.append("certification manifest has an unsupported schema")
        contract = certification_manifest.get("contract")
        if contract != EXPECTED_CONTRACT:
            failures.append(
                f"certification contract {contract!r} does not match "
                f"{EXPECTED_CONTRACT!r}"
            )
        for artifact in ("trunk", "stock_mtp", "candidate_mtp", "session"):
            described = certification_manifest.get(artifact)
            if (
                not isinstance(described, dict)
                or not isinstance(described.get("bytes"), int)
                or described["bytes"] <= 0
                or not isinstance(described.get("sha256"), str)
                or len(described["sha256"]) != 64
            ):
                failures.append(
                    f"certification manifest has no valid {artifact} description"
                )
        stock = certification_manifest.get("stock_mtp", {})
        candidate = certification_manifest.get("candidate_mtp", {})
        if stock.get("sha256") == candidate.get("sha256"):
            failures.append("candidate and stock MTP sidecars have identical hashes")
    else:
        failures.append("certification-manifest.json is missing")

    for label in LABELS:
        rows = json.loads((root / f"{label}.json").read_text())
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{label}.json is not a list of result objects")
        rows_by_label[label] = rows
        summary[label] = metrics(rows)
        if len(rows) != expected_turns:
            failures.append(
                f"{label}: expected {expected_turns} turns, found {len(rows)}"
            )
        if summary[label]["native_decode_measurements"] != expected_turns:
            failures.append(
                f"{label}: expected {expected_turns} native decode measurements, "
                f"found {summary[label]['native_decode_measurements']}"
            )
        expected_cached_turns = max(0, expected_turns - 1)
        if summary[label]["cached_turns_after_first"] != expected_cached_turns:
            failures.append(
                f"{label}: expected prefix reuse on {expected_cached_turns} turns "
                f"after the first, found {summary[label]['cached_turns_after_first']}"
            )
        min_cache = summary[label]["min_cached_fraction_after_first"]
        required_cache = CACHE_MIN_FRACTION[label]
        if expected_cached_turns and (
            min_cache is None or min_cache < required_cache
        ):
            failures.append(
                f"{label}: minimum post-first cached fraction {min_cache} is below "
                f"{required_cache}"
            )

    for label in ("stock-mtp", "candidate-mtp"):
        if len(numeric(rows_by_label[label], "tau")) != expected_turns:
            failures.append(f"{label}: missing finite tau measurements")

    candidate = summary["candidate-mtp"]
    for flag in QUALITY_FLAGS:
        if candidate[flag]:
            failures.append(f"candidate-mtp: {candidate[flag]} {flag} turn(s)")
    if candidate["recall_total"] <= 0:
        failures.append("candidate-mtp: session emitted no scored recall checks")
    elif candidate["recall_hits"] != candidate["recall_total"]:
        failures.append(
            f"candidate-mtp: recall {candidate['recall_hits']}/"
            f"{candidate['recall_total']} did not pass"
        )

    candidate_rate = candidate["mean_decode_tok_s"]
    candidate_median = candidate["median_decode_tok_s"]
    for baseline in ("ar", "stock-mtp"):
        baseline_rate = summary[baseline]["mean_decode_tok_s"]
        baseline_median = summary[baseline]["median_decode_tok_s"]
        if candidate_rate is None or baseline_rate is None or candidate_rate <= baseline_rate:
            failures.append(
                f"candidate mean decode {candidate_rate} does not beat "
                f"{baseline} {baseline_rate}"
            )
        if (
            candidate_median is None
            or baseline_median is None
            or candidate_median <= baseline_median
        ):
            failures.append(
                f"candidate median decode {candidate_median} does not beat "
                f"{baseline} {baseline_median}"
            )

    stock_tau = summary["stock-mtp"]["mean_tau"]
    stock_median_tau = summary["stock-mtp"]["median_tau"]
    if candidate["mean_tau"] is None or stock_tau is None or candidate["mean_tau"] <= stock_tau:
        failures.append(
            f"candidate mean tau {candidate['mean_tau']} does not beat stock {stock_tau}"
        )
    if (
        candidate["median_tau"] is None
        or stock_median_tau is None
        or candidate["median_tau"] <= stock_median_tau
    ):
        failures.append(
            f"candidate median tau {candidate['median_tau']} does not beat "
            f"stock {stock_median_tau}"
        )

    redline = json.loads((root / "redline-shadow.json").read_text())
    redline_pass = redline.get("pass") is True
    if not redline_pass:
        failures.append("Redline shadow/parity report did not pass")

    summary["redline_shadow_pass"] = redline_pass
    summary["promotion_pass"] = not failures
    summary["failures"] = failures
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certification_dir", type=Path)
    parser.add_argument("--expected-turns", type=int, default=8)
    args = parser.parse_args()

    summary = evaluate(args.certification_dir, args.expected_turns)
    output = args.certification_dir / "summary.json"
    partial = output.with_suffix(".json.partial")
    partial.write_text(json.dumps(summary, indent=2) + "\n")
    partial.replace(output)
    print(json.dumps(summary, indent=2))
    if not summary["promotion_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
