#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Recover the AWQ recipe (scope and alpha) from a quantized artifact's bytes.

Neither `--awq-alpha` nor the F1/F2 whitelist scope is written into HFQ
metadata, so two arms that differ substantially in quality are distinguishable
only by filename — which does not survive a copy, a rename, or a hand-off.
Both are recoverable:

SCOPE (F1 vs F2), structural
  `awq_eligible` (main.rs:6415) emits a `<name>.awq_scale.weight` sidecar per
  eligible tensor. F1 covers input-side projections only; F2 adds o_proj /
  out_proj / down_proj / w_down. Presence or absence of output-side sidecars
  identifies the scope exactly.

ALPHA, numerical
  Scales are `s_raw[j] = RMS_act[j] ** alpha`, geo-mean normalized
  (main.rs:6303 compute_awq_scales). Therefore

      log s[j] = alpha * (log RMS[j] - mean_j log RMS[j])

  so alpha is the slope of log(scale) regressed on centred log(RMS), with a
  zero intercept by construction. RMS comes from the calibration imatrix
  (`in_sum2`), so recovering alpha requires the same imatrix the arm was built
  with. R^2 near 1 confirms the model; a poor fit means the scales did not come
  from this imatrix, which is itself worth knowing.
"""

import json
import math
import os
import struct
import sys
import collections


def read_index(path, window=64 << 20):
    buf = open(path, "rb").read(window)
    if buf[:4] not in (b"HFQM", b"HFQ\x00", b"HFQF"):
        pass  # tolerate variants; offsets below are what matter
    n = struct.unpack_from("<I", buf, 12)[0]
    mo = struct.unpack_from("<Q", buf, 16)[0]

    depth, end = 0, -1
    for i in range(mo, len(buf)):
        c = buf[i:i + 1]
        if c == b"{":
            depth += 1
        elif c == b"}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    pos = end
    (idx_n,) = struct.unpack_from("<I", buf, pos)
    pos += 4

    rows = []
    for _ in range(idx_n):
        (nl,) = struct.unpack_from("<H", buf, pos); pos += 2
        name = buf[pos:pos + nl].decode("utf-8", "replace"); pos += nl
        qt = buf[pos]; pos += 1
        nd = buf[pos]; pos += 1
        shape = list(struct.unpack_from("<%dI" % nd, buf, pos)); pos += 4 * nd
        pos += 4
        (dsz,) = struct.unpack_from("<Q", buf, pos); pos += 8
        rows.append({"name": name, "qt": qt, "shape": shape, "size": dsz})
    off = pos
    for r in rows:
        r["off"] = off
        off += r["size"]
    return rows


def f16_bytes_to_floats(b):
    return list(struct.unpack("<%de" % (len(b) // 2), b[: (len(b) // 2) * 2]))


def f32_bytes_to_floats(b):
    return list(struct.unpack("<%df" % (len(b) // 4), b[: (len(b) // 4) * 4]))


OUTPUT_SIDE = ("o_proj", "out_proj", "down_proj", "w_down", "wo")


def main(model, imatrix=None):
    rows = read_index(model)
    scales = [r for r in rows if r["name"].endswith("awq_scale.weight")]
    print("{}".format(os.path.basename(model)))
    print("  tensors={}  awq_scale sidecars={}".format(len(rows), len(scales)))

    if not scales:
        print("  scope: NO AWQ (plain quant, no sidecars)")
        return 0

    kinds = collections.Counter()
    for r in scales:
        base = r["name"].replace(".awq_scale.weight", "")
        kinds[base.rsplit(".", 1)[-1]] += 1
    out_side = {k: v for k, v in kinds.items() if any(t in k for t in OUTPUT_SIDE)}
    print("  scaled classes: {}".format(dict(sorted(kinds.items()))))
    print("  scope: {}".format("F2 (input + output-side)" if out_side
                               else "F1 (input-side only)"))
    if out_side:
        print("    output-side scaled: {}".format(dict(sorted(out_side.items()))))

    if not imatrix:
        print("  alpha: pass the imatrix used at build time to recover it")
        return 0

    im_rows = {r["name"]: r for r in read_index(imatrix)}
    fm = open(model, "rb")
    fi = open(imatrix, "rb")

    fits = []
    for r in scales[:64]:                      # a sample is plenty; alpha is global
        base = r["name"].replace(".awq_scale.weight", "")
        im = im_rows.get(base + ".imatrix")
        if im is None:
            continue
        fm.seek(r["off"]);  s_vals = f16_bytes_to_floats(fm.read(r["size"]))
        fi.seek(im["off"]); q_vals = f32_bytes_to_floats(fi.read(im["size"]))
        n = min(len(s_vals), len(q_vals))
        xs, ys = [], []
        for j in range(n):
            s, q = s_vals[j], q_vals[j]
            if s > 0 and q > 0:
                xs.append(0.5 * math.log(q))   # log RMS = 0.5*log(in_sum2), scale-invariant
                ys.append(math.log(s))
        if len(xs) < 32:
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx <= 0:
            continue
        slope = sxy / sxx
        syy = sum((y - my) ** 2 for y in ys)
        r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else float("nan")
        fits.append((slope, r2))

    if not fits:
        print("  alpha: could not match sidecars to imatrix entries")
        return 1
    fits.sort()
    med = fits[len(fits) // 2]
    print("  alpha: {:.4f}  (median over {} tensors, R^2={:.5f})".format(med[0], len(fits), med[1]))
    lo, hi = fits[0][0], fits[-1][0]
    print("         spread {:.4f} .. {:.4f}".format(lo, hi))
    if med[1] < 0.99:
        print("         WARNING: poor fit — these scales may not derive from this imatrix")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: awq_recipe_fingerprint.py <model.mq4> [imatrix.hfq]")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
