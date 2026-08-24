#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Per-tensor-class dtype table for HFQ containers.

`dump_hfq_dtypes.py` shows a histogram plus layer-0 samples. This groups EVERY
tensor by its class (the name with layer indices stripped) so two containers can
be diffed tier by tier — which is what distinguishes a `.mq4` from a `.mq4r`,
since they can differ by a single tensor class while sharing every other byte.
"""

import sys
import os
import re
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_hfq_dtypes import QT, json_end_of  # noqa: E402


def read_index(path, window=64 << 20):
    buf = open(path, "rb").read(window)
    je = json_end_of(buf, 32)
    pos = je
    n = int.from_bytes(buf[pos:pos + 4], "little")
    pos += 4
    rows = []
    for _ in range(n):
        nl = int.from_bytes(buf[pos:pos + 2], "little"); pos += 2
        name = buf[pos:pos + nl].decode("utf-8", "replace"); pos += nl
        qt = buf[pos]; pos += 1
        nd = buf[pos]; pos += 1
        for _ in range(nd):
            pos += 4
        pos += 4
        dsz = int.from_bytes(buf[pos:pos + 8], "little"); pos += 8
        rows.append((name, qt, dsz))
    return rows


def classify(name):
    # Collapse layer indices so per-layer tensors group into one class.
    c = re.sub(r"\.(layers|blk)\.\d+\.", ".L.", name)
    c = re.sub(r"\.\d+\.", ".N.", c)
    return c.replace("model.language_model.", "").replace("model.", "")


def table(path):
    rows = read_index(path)
    agg = collections.OrderedDict()
    for name, qt, dsz in rows:
        key = (classify(name), QT.get(qt, "qt%d" % qt))
        if key not in agg:
            agg[key] = [0, 0]
        agg[key][0] += 1
        agg[key][1] += dsz
    print("=" * 84)
    print("{}   {:,} B   {} tensors".format(os.path.basename(path), os.path.getsize(path), len(rows)))
    print("  {:<46s} {:<13s} {:>5s} {:>16s}".format("class", "dtype", "n", "bytes"))
    for (cls, dt), (cnt, byt) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print("  {:<46s} {:<13s} {:>5d} {:>16,}".format(cls[:46], dt, cnt, byt))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        table(p)
