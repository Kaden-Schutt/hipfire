#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Next-token PPL from one HFPLOG01 .plog + tokens.bin (pure Python).

Row t predicts token_ids[t+1]; scores 0..n-2. Matches parent/plog.rs.
"""
from __future__ import annotations
import argparse, math, struct, sys
from array import array
HDR, MAGIC = 24, b"HFPLOG01"

def read_tokens(path):
    raw = open(path, "rb").read()
    if len(raw) % 4:
        sys.exit(f"tokens {path}: size {len(raw)} not multiple of 4")
    return list(struct.unpack("<%dI" % (len(raw)//4), raw))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plog"); ap.add_argument("tokens")
    ap.add_argument("--bucket", type=int, default=0)
    args = ap.parse_args()
    toks = read_tokens(args.tokens)
    f = open(args.plog, "rb")
    if f.read(8) != MAGIC: sys.exit("bad magic")
    n, v = struct.unpack("<II", f.read(8)); f.read(8)
    if len(toks) != n: sys.exit(f"token len {len(toks)} != n_tokens {n}")
    nlls = []
    for t in range(n - 1):
        tgt = toks[t + 1]
        if tgt >= v: sys.exit(f"tgt out of vocab at {t+1}")
        f.seek(HDR + t * v * 4)
        row = array("f"); row.frombytes(f.read(v * 4))
        m = max(row)
        s = 0.0
        for x in row:
            s += math.exp(x - m)
        log_sum = m + math.log(s)
        nlls.append(-(row[tgt] - log_sum))
        if (t + 1) % 128 == 0:
            print(f"  scored {t+1}/{n-1}", file=sys.stderr)
    mean = sum(nlls) / len(nlls)
    ppl = math.exp(mean)
    print(f"plog:       {args.plog}")
    print(f"tokens:     {args.tokens}")
    print(f"n_tokens:   {n}")
    print(f"vocab:      {v}")
    print(f"scored:     {len(nlls)}")
    print(f"NLL/tok:    {mean:.10f}")
    print(f"PPL:        {ppl:.6f}")
    if args.bucket > 0:
        w = args.bucket
        print()
        print(f"{'bucket':<14}{'scored':>8}{'NLL/tok':>14}{'PPL':>12}")
        for lo in range(0, len(nlls), w):
            hi = min(lo + w, len(nlls))
            chunk = nlls[lo:hi]
            m = sum(chunk)/len(chunk)
            print(f"[{lo},{hi})".ljust(14) + f"{len(chunk):8d}{m:14.6f}{math.exp(m):12.4f}")
if __name__ == "__main__":
    main()
