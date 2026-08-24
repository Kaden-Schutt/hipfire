#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Dump the per-tensor quant-type layout of an HFQ container.

Answers "which tier is at which precision" for a shipped artifact without
loading it onto a GPU — e.g. whether a `.mq4` pins lm_head/embeddings at Q8
while `.mq4r` leaves them at MQ4.

Container layout (crates/hipfire-runtime/src/hfq.rs:299-406):
  [0..32)            fixed header; metadata starts at 32
  [32..json_end)     metadata JSON (brace-matched, strings/escapes honoured)
  u32                n_tensors
  per tensor:        u16 name_len | name | u8 quant_type | u8 n_dims
                     | n_dims x u32 shape | u32 group_size | u64 data_size
"""

import sys
import os
import collections

QT = {
    0: "Q4F16G64", 1: "F16", 2: "F32", 3: "Q8F16", 4: "Q4K", 5: "Q8HFQ",
    6: "HFQ4G256", 7: "HFQ4G128", 8: "HFQ6G256", 9: "HFQ2G256", 10: "HFQ2G128",
    11: "HFQ3G256", 12: "HFQ3G128", 13: "MQ4G256", 14: "MQ8G256", 15: "MQ6G256",
    16: "BF16", 17: "MQ3G256", 18: "MQ2G256", 19: "MQ2G256Lloyd", 20: "MQ3G256Lloyd",
    21: "HFP4G32", 22: "TidI32", 24: "MFP4G32", 28: "PARO4G128", 29: "PARO4G128T",
    30: "MQ4G256Lloyd", 31: "MQ5G256", 32: "MFP4G32Lloyd", 33: "MFP4G32P",
    34: "MFP4G32E8", 35: "MFP4G32E8SOA", 36: "MFP3G32E8", 37: "MFP2G32E8",
    38: "MQ2G256GL", 39: "MQ3G256GL",
}


def json_end_of(buf, start):
    depth, in_str, esc = 0, False, False
    for i in range(start, len(buf)):
        b = buf[i]
        if esc:
            esc = False
            continue
        if b == 0x5C and in_str:      # backslash
            esc = True
            continue
        if b == 0x22:                 # quote
            in_str = not in_str
            continue
        if not in_str:
            if b == 0x7B:             # {
                depth += 1
            elif b == 0x7D:           # }
                depth -= 1
                if depth == 0:
                    return i + 1
    return 0


def dump(path, window=64 << 20):
    buf = open(path, "rb").read(window)
    je = json_end_of(buf, 32)
    if je == 0:
        print("  metadata JSON not brace-terminated (window too small?)")
        return 1
    pos = je
    n = int.from_bytes(buf[pos:pos + 4], "little")
    pos += 4

    rows = []
    for _ in range(n):
        nl = int.from_bytes(buf[pos:pos + 2], "little"); pos += 2
        name = buf[pos:pos + nl].decode("utf-8", "replace"); pos += nl
        qt = buf[pos]; pos += 1
        nd = buf[pos]; pos += 1
        shape = []
        for _ in range(nd):
            shape.append(int.from_bytes(buf[pos:pos + 4], "little")); pos += 4
        pos += 4                                  # group_size
        dsz = int.from_bytes(buf[pos:pos + 8], "little"); pos += 8
        rows.append((name, qt, shape, dsz))

    print("  file: {}  ({} bytes, {} tensors)".format(
        os.path.basename(path), os.path.getsize(path), n))

    hist = collections.Counter(QT.get(qt, "qt{}".format(qt)) for _, qt, _, _ in rows)
    byqt = collections.defaultdict(int)
    for _, qt, _, dsz in rows:
        byqt[QT.get(qt, "qt{}".format(qt))] += dsz
    print("  histogram (count / bytes):")
    for k, v in sorted(hist.items(), key=lambda kv: -byqt[kv[0]]):
        print("    {:14s} {:5d}   {:>15,} B".format(k, v, byqt[k]))

    # The tiers that distinguish a `.mq4` from a `.mq4r`.
    print("  fixed-tier members:")
    for name, qt, shape, dsz in rows:
        low = name.lower()
        if any(t in low for t in ("embed", "lm_head", "output.weight", "token_embd")):
            print("    {:14s} {:52s} {}".format(QT.get(qt, "qt%d" % qt), name, shape))

    # One representative of each per-layer projection class from layer 0.
    print("  layer-0 projections:")
    seen = set()
    for name, qt, shape, dsz in rows:
        if ".layers.0." not in name and "blk.0." not in name:
            continue
        kind = name.rsplit(".", 2)[-2] if name.count(".") >= 2 else name
        if kind in seen:
            continue
        seen.add(kind)
        print("    {:14s} {:52s} {}".format(QT.get(qt, "qt%d" % qt), name, shape))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: dump_hfq_dtypes.py <file.hfq|.mq4|.mq4r> [...]")
        sys.exit(2)
    rc = 0
    for p in sys.argv[1:]:
        print("=" * 78)
        rc |= dump(p)
    sys.exit(rc)
