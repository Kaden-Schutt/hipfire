#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Fast next-token top-1 accuracy by position bucket for HFPLOG01.

Reads rows in multi-row chunks and uses array('f') + pure Python max.
Default buckets are length-adaptive so longer contexts get higher buckets.
"""
from __future__ import annotations

import argparse
import struct
import sys
from array import array

HDR = 24
MAGIC = b"HFPLOG01"


def read_tokens(path: str) -> list[int]:
    raw = open(path, "rb").read()
    if len(raw) % 4:
        sys.exit(f"tokens {path}: bad size {len(raw)}")
    return list(struct.unpack("<%dI" % (len(raw) // 4), raw))


def open_plog(path: str):
    f = open(path, "rb")
    if f.read(8) != MAGIC:
        sys.exit(f"{path}: bad magic")
    n, v = struct.unpack("<II", f.read(8))
    f.read(8)
    return f, n, v


def argmax_row(row: array) -> int:
    best_i = 0
    best_v = row[0]
    for i in range(1, len(row)):
        v = row[i]
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def default_buckets(n: int) -> list[tuple[int, int]]:
    """Half-open [lo, hi) over scored rows 0..n-2 (we still key by row t)."""
    edges = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    edges = [e for e in edges if e < n - 1]
    if not edges or edges[-1] != n - 1:
        edges.append(n - 1)
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            out.append((lo, hi))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plog")
    ap.add_argument("tokens")
    ap.add_argument("--width", type=int, default=0, help="fixed-width buckets if >0")
    args = ap.parse_args()

    toks = read_tokens(args.tokens)
    f, n, v = open_plog(args.plog)
    if len(toks) != n:
        sys.exit(f"token len {len(toks)} != n_tokens {n}")
    scored = n - 1
    if scored <= 0:
        sys.exit("nothing to score")

    # hit[t] for row t predicting toks[t+1], t in 0..scored-1
    hits = [0] * scored
    # stream rows 0..scored-1
    f.seek(HDR)
    # process in chunks of 8 rows to keep memory modest (~4MB/row at vocab 129k)
    chunk = 4
    t = 0
    while t < scored:
        take = min(chunk, scored - t)
        raw = f.read(take * v * 4)
        if len(raw) != take * v * 4:
            sys.exit(f"short read at t={t}")
        for j in range(take):
            row = array("f")
            off = j * v * 4
            row.frombytes(raw[off : off + v * 4])
            pred = argmax_row(row)
            hits[t + j] = 1 if pred == toks[t + j + 1] else 0
        t += take
        if t % 256 == 0 or t == scored:
            print(f"  scanned {t}/{scored}", file=sys.stderr)

    if args.width > 0:
        buckets = [(lo, min(lo + args.width, scored)) for lo in range(0, scored, args.width)]
    else:
        buckets = default_buckets(n)

    print(f"tokens n={n} scored={scored} plog={args.plog}")
    print(f"{'bucket':<14}{'hits':>10}{'acc':>10}")
    total_h = 0
    for lo, hi in buckets:
        h = sum(hits[lo:hi])
        m = hi - lo
        total_h += h
        print(f"[{lo},{hi})".ljust(14) + f"{h}/{m}".rjust(10) + f"{(h/m if m else 0):10.3f}")
    print(f"{'ALL':<14}{total_h}/{scored}".rjust(10) + f"{total_h/scored:10.3f}")


if __name__ == "__main__":
    main()
