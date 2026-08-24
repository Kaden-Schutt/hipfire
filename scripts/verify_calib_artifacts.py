#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Verify .calib.hfq artifacts from their own bytes.

The harness prints `coverage N/N [PASS]` and `max_consistency ... [CONSISTENT]`,
and neither is admissible evidence:

  * `max_consistency` compares diag(H) against sum(x^2) where BOTH are
    accumulated from the same staged buffer by two kernels, so it measures
    kernel agreement, not data capture. A tensor that was never collected is
    absent from both sides and the check still reads 0.000e0.
  * A coverage gate compares collected names against the harness's own expected
    set. If the expected set is wrong (as it was when calibration emitted
    runtime names instead of safetensors names), it agrees with itself.

This reads the artifact header directly and reports what is actually inside:
per-tensor token counts, projection-kind histogram, and byte size. A tensor
whose token count differs from the rest was under-collected even if coverage
said PASS.
"""

import glob
import json
import os
import sys
import collections


def read_header(path, window=8_000_000):
    blob = open(path, "rb").read(window)
    if not blob.startswith(b"HFQM"):
        return None, "not an HFQM container"
    start = blob.find(b"{")
    if start < 0:
        return None, "no JSON header found"
    depth = 0
    end = -1
    for i in range(start, len(blob)):
        ch = blob[i : i + 1]
        if ch == b"{":
            depth += 1
        elif ch == b"}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None, "unterminated JSON header (window too small)"
    try:
        return json.loads(blob[start:end].decode("utf-8", "replace")), None
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        return None, "header parse failed: {}".format(exc)


def main(paths):
    rc = 0
    for path in paths:
        name = os.path.basename(path)
        hdr, err = read_header(path)
        if hdr is None:
            print("  {:34s} FAIL  {}".format(name, err))
            rc = 1
            continue

        ptt = hdr.get("per_tensor_tokens", {}) or {}
        n_h = hdr.get("n_hessian")
        n_i = hdr.get("n_imatrix")
        toks = sorted(set(ptt.values()))
        size_gb = os.path.getsize(path) / 1e9

        # Every tensor must have seen the same token count. A short tensor means
        # capture dropped rows for it while the coverage gate still counted it.
        uniform = "uniform" if len(toks) == 1 else "RAGGED {}".format(toks[:4])
        consistent = (n_h == n_i == len(ptt))

        kinds = collections.Counter(k.rsplit(".", 2)[-2] if k.count(".") >= 2 else k for k in ptt)
        top = ", ".join("{}={}".format(k, v) for k, v in sorted(kinds.items())[:6])

        status = "OK " if (consistent and len(toks) == 1) else "BAD"
        if status == "BAD":
            rc = 1
        print(
            "  {:34s} {}  n_h={:<5} n_i={:<5} entries={:<5} tokens={:<9} {:>6.1f} GB  {}".format(
                name, status, n_h, n_i, len(ptt), toks[0] if toks else "-", size_gb, uniform
            )
        )
        print("      kinds: {}".format(top))
    return rc


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(
            p
            for p in glob.glob("/scratch/work/*.calib.hfq")
            if not any(s in p for s in ("smoke", "probe", "namecheck"))
        )
    sys.exit(main(args))
