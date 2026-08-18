#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt <kaden@hipfire.dev>
"""Full-resolution next-token accuracy by position for one or more `.plog` files.

Scores EVERY row rather than sampling, and reports accuracy in fixed-width
position buckets. The point is to distinguish two failure shapes that a coarse
scan cannot tell apart:

  STEP  accuracy drops at a position and then plateaus  -> something switches
        on at a position (a budget, a capacity, a wrap) and latches.
  RAMP  accuracy degrades continuously with position    -> something accumulates
        with context length.

Those imply completely different searches, so the distinction is worth the
extra rows. The older `plog_pos_scan.py` sampled 24 rows per bucket, which put
the whole `[512,1022)` verdict on 24 of 510 positions -- enough to see that
something was wrong, not enough to see its shape.

Position `t` is scored against `token_ids[t+1]`; see `parent::plog::compare`,
which owns the same shift.

Usage:
    plog_fine_scan.py A.plog [B.plog ...] tokens.bin [--width N]

Example (the run that showed the DS4 parent steps down near 448-512 and then
holds flat, rather than decaying smoothly):

    plog_fine_scan.py parent_1024.plog mq2r_1024.plog tokens.bin
"""
import array
import struct
import sys

HDR = 24
MAGIC = b"HFPLOG01"
DEFAULT_WIDTH = 64


def open_plog(path):
    f = open(path, "rb")
    if f.read(8) != MAGIC:
        sys.exit(f"{path}: bad magic (expected {MAGIC!r})")
    n_tokens, vocab = struct.unpack("<II", f.read(8))
    f.read(8)
    return f, n_tokens, vocab


def argmax_row(f, vocab, t):
    """Argmax of row `t`. Two C-level passes beat one Python-level pass."""
    f.seek(HDR + t * vocab * 4)
    a = array.array("f")
    a.frombytes(f.read(vocab * 4))
    return a.index(max(a))


def main(argv):
    args = [a for a in argv if a != "--width"]
    width = DEFAULT_WIDTH
    if "--width" in argv:
        i = argv.index("--width")
        width = int(argv[i + 1])
        args = argv[:i] + argv[i + 2:]
    if len(args) < 2:
        sys.exit(__doc__)

    paths, tok_path = args[:-1], args[-1]
    with open(tok_path, "rb") as tf:
        toks = [t[0] for t in struct.iter_unpack("<I", tf.read())]

    names, correct = [], {}
    for p in paths:
        f, n, vocab = open_plog(p)
        name = p.rsplit("/", 1)[-1]
        names.append(name)
        last = min(n, len(toks) - 1)
        correct[name] = {
            t: argmax_row(f, vocab, t) == toks[t + 1] for t in range(1, last)
        }
        f.close()

    print(f"tokens n={len(toks)}  bucket_width={width}")
    print()
    print("bucket".ljust(14) + "".join(n[:22].ljust(24) for n in names))
    for lo in range(0, len(toks) - 1, width):
        hi = lo + width
        row = f"[{lo},{hi})".ljust(14)
        for name in names:
            ts = [t for t in range(max(lo, 1), hi) if t in correct[name]]
            if ts:
                hits = sum(correct[name][t] for t in ts)
                row += f"{hits}/{len(ts)} = {hits / len(ts):.3f}".ljust(24)
            else:
                row += "-".ljust(24)
        print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
