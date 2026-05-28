#!/usr/bin/env python3
"""Fit TurboQuant value codebooks from captured rotated V samples.

This is the offline half of KV table calibration. The intended input is the
post-normalization, post-FWHT/sign scalar distribution from the TQV value
write path. One captured distribution can emit candidate tables for multiple
bitwidths.

Example:
  scripts/tqv_fit_tables.py \
    --input 256=/tmp/qwen35-2b-hd256-v.f32 \
    --bits 2,3,4 \
    --out /tmp/qwen35-2b-tqv-tables.json

For a cheap sanity check without a capture file:
  scripts/tqv_fit_tables.py --synthetic 128,256 --out /tmp/tqv-synth.json
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SampleSet:
    head_dim: int
    source: str
    values: list[float]
    sha256: str


def parse_csv_ints(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def read_f32le(path: Path, max_samples: int | None, seed: int) -> tuple[list[float], str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) % 4 != 0:
        raise ValueError(f"{path}: f32le input size is not a multiple of 4 bytes")

    n = len(data) // 4
    vals = array.array("f")
    vals.frombytes(data)
    if sys.byteorder != "little":
        vals.byteswap()
    values = [float(v) for v in vals]
    if max_samples is not None and n > max_samples:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(n), max_samples))
        values = [values[i] for i in idx]
    return values, digest


def synthetic_values(head_dim: int, n: int, seed: int) -> list[float]:
    # Idealized FWHT output for unit-norm vectors. Useful only for a smoke test;
    # real tables must be fitted from captured model activations.
    rng = random.Random(seed ^ head_dim)
    sigma = 1.0 / math.sqrt(head_dim)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("cannot compute quantile of an empty sample set")
    q = max(0.0, min(1.0, q))
    x = q * (len(sorted_vals) - 1)
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return sorted_vals[lo]
    t = x - lo
    return sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t


def fit_positive_lloyd(abs_vals: list[float], n_pos: int, max_iter: int) -> list[float]:
    if n_pos < 1:
        raise ValueError("n_pos must be positive")
    if not abs_vals:
        raise ValueError("sample set has no finite nonzero values")

    vals = sorted(abs_vals)
    centroids = [quantile(vals, (i + 0.5) / n_pos) for i in range(n_pos)]
    eps = max(vals[-1] * 1.0e-7, 1.0e-12)

    for _ in range(max_iter):
        thresholds = [(centroids[i] + centroids[i + 1]) * 0.5 for i in range(n_pos - 1)]
        sums = [0.0] * n_pos
        counts = [0] * n_pos
        bucket = 0
        for v in vals:
            while bucket < len(thresholds) and v > thresholds[bucket]:
                bucket += 1
            sums[bucket] += v
            counts[bucket] += 1

        changed = 0.0
        next_centroids = centroids[:]
        for i in range(n_pos):
            if counts[i] > 0:
                next_centroids[i] = sums[i] / counts[i]
            changed = max(changed, abs(next_centroids[i] - centroids[i]))
        centroids = next_centroids
        if changed <= eps:
            break

    return centroids


def fit_symmetric_table(values: Iterable[float], bits: int, max_iter: int) -> dict:
    if bits < 1 or bits > 8:
        raise ValueError(f"unsupported TQV bitwidth: {bits}")
    levels = 1 << bits
    if levels % 2 != 0:
        raise ValueError("symmetric nonzero tables require an even number of levels")

    clean = [v for v in values if math.isfinite(v)]
    abs_vals = [abs(v) for v in clean if v != 0.0]
    n_pos = levels // 2
    pos = fit_positive_lloyd(abs_vals, n_pos, max_iter)
    centroids = [-v for v in reversed(pos)] + pos
    thresholds = [(centroids[i] + centroids[i + 1]) * 0.5 for i in range(levels - 1)]
    mse = table_mse(clean, centroids, thresholds)
    return {
        "bits": bits,
        "levels": levels,
        "centroids": centroids,
        "thresholds": thresholds,
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def fit_ternary_table(values: Iterable[float]) -> dict:
    # Symmetric ternary {-a, 0, +a}. For a fixed threshold t, the best a is
    # mean(|x| | |x| > t). Sweep quantiles to avoid assuming Gaussianity.
    clean = [v for v in values if math.isfinite(v)]
    abs_vals = sorted(abs(v) for v in clean)
    if not abs_vals:
        raise ValueError("sample set has no finite values")

    best = None
    for q in [i / 400.0 for i in range(1, 360)]:
        t = quantile(abs_vals, q)
        tail = [v for v in abs_vals if v > t]
        if not tail:
            continue
        a = sum(tail) / len(tail)
        centroids = [-a, 0.0, a]
        thresholds = [-t, t]
        mse = table_mse(clean, centroids, thresholds)
        cand = (mse, a, t)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        raise ValueError("failed to fit ternary table")
    mse, a, t = best
    return {
        "bits": "ternary",
        "effective_bits": math.log2(3.0),
        "levels": 3,
        "centroids": [-a, 0.0, a],
        "thresholds": [-t, t],
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def table_mse(values: list[float], centroids: list[float], thresholds: list[float]) -> float:
    if not values:
        return float("nan")
    total = 0.0
    for v in values:
        idx = 0
        while idx < len(thresholds) and v > thresholds[idx]:
            idx += 1
        d = v - centroids[idx]
        total += d * d
    return total / len(values)


def c_array(values: list[float]) -> str:
    return "{" + ", ".join(f"{v:.9g}f" for v in values) + "}"


def table_to_c(name: str, table: dict) -> dict:
    bits = table["bits"]
    if bits == "ternary":
        suffix = "T"
    else:
        suffix = str(bits)
    upper = f"{name}_TQV{suffix}"
    return {
        "centroids": f"__constant__ float {upper}_C[{len(table['centroids'])}] = {c_array(table['centroids'])};",
        "thresholds": f"// thresholds for {upper}: {c_array(table['thresholds'])}",
    }


def load_inputs(args: argparse.Namespace) -> list[SampleSet]:
    out: list[SampleSet] = []
    for spec in args.input or []:
        if "=" not in spec:
            raise ValueError("--input must be HEAD_DIM=PATH")
        head_raw, path_raw = spec.split("=", 1)
        head_dim = int(head_raw)
        path = Path(path_raw)
        values, digest = read_f32le(path, args.max_samples, args.seed)
        out.append(SampleSet(head_dim, str(path), values, digest))

    for head_dim in parse_csv_ints(args.synthetic or ""):
        values = synthetic_values(head_dim, args.synthetic_samples, args.seed)
        digest = hashlib.sha256(
            struct.pack("<" + "f" * len(values), *values)
        ).hexdigest()
        out.append(SampleSet(head_dim, f"synthetic-normal:{head_dim}", values, digest))

    if not out:
        raise ValueError("provide at least one --input HEAD_DIM=PATH or --synthetic HEADS")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", help="captured f32le samples as HEAD_DIM=PATH")
    ap.add_argument("--bits", default="2,3,4", help="comma-separated TQV bitwidths; 1 means ternary TQ1/TQV1.58")
    ap.add_argument("--ternary", action="store_true", help="also fit {-a,0,+a} ternary table")
    ap.add_argument("--synthetic", help="comma-separated head dims for synthetic smoke data")
    ap.add_argument("--synthetic-samples", type=int, default=1_000_000)
    ap.add_argument("--max-samples", type=int, help="deterministically subsample each input")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0x545156)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bits = parse_csv_ints(args.bits)
    samples = load_inputs(args)
    artifact = {
        "schema": "hipfire.tqv_tables.v1",
        "fit": {
            "method": "symmetric_lloyd_max_empirical",
            "ternary_method": "threshold_sweep" if args.ternary else None,
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
        "tables": {},
    }

    for sample in samples:
        clean = [v for v in sample.values if math.isfinite(v)]
        if not clean:
            raise ValueError(f"{sample.source}: no finite samples")
        key = f"head_dim_{sample.head_dim}"
        tables = {}
        for bit in bits:
            if bit == 1:
                tables["tqv1_58"] = fit_ternary_table(clean)
            else:
                tables[f"tqv{bit}"] = fit_symmetric_table(clean, bit, args.max_iter)
        if args.ternary and "tqv1_58" not in tables:
            tables["tqv1_58"] = fit_ternary_table(clean)
        artifact["tables"][key] = {
            "source": sample.source,
            "source_sha256": sample.sha256,
            "sample_count": len(clean),
            "mean": sum(clean) / len(clean),
            "mean_abs": sum(abs(v) for v in clean) / len(clean),
            "rms": math.sqrt(sum(v * v for v in clean) / len(clean)),
            "tables": tables,
            "c_snippets": {
                name: table_to_c(f"TURBO_HD{sample.head_dim}", table)
                for name, table in tables.items()
            },
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    for head_key, head in artifact["tables"].items():
        print(f"{head_key}: samples={head['sample_count']} rms={head['rms']:.8g}")
        for name, table in head["tables"].items():
            print(f"  {name}: rmse={table['rmse']:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
