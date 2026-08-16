#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Extract an imatrix-only HFQM package from a full `.calib.hfq`.

A calibration package carries two payloads per weight:
  <name>.hessian   qt=130, K x K packed  -- consumed ONLY by --ldlq (GPTQ)
  <name>.imatrix   qt=2,   K floats      -- consumed by --awq

AWQ therefore needs a few MB out of a package that is tens of GB, because the
Hessian is quadratic in K and the imatrix is linear. Shipping the whole package
to a machine that will only run AWQ arms wastes hours of transfer for data the
quantizer never reads.

Container layout (crates/hipfire-runtime/src/hfq.rs:299-406):
  [0..32)  header (metadata_offset = 32)
  [32..je) metadata JSON, brace-matched
  u32      n_tensors
  per tensor: u16 name_len | name | u8 quant_type | u8 n_dims
              | n_dims x u32 shape | u32 group_size | u64 data_size
  then tensor payloads back-to-back in index order, starting at data_offset.
"""

import json
import os
import struct
import sys


def json_end_of(buf, start):
    depth, in_str, esc = 0, False, False
    for i in range(start, len(buf)):
        b = buf[i]
        if esc:
            esc = False
            continue
        if b == 0x5C and in_str:
            esc = True
            continue
        if b == 0x22:
            in_str = not in_str
            continue
        if not in_str:
            if b == 0x7B:
                depth += 1
            elif b == 0x7D:
                depth -= 1
                if depth == 0:
                    return i + 1
    raise SystemExit("metadata JSON not brace-terminated")


def main(src, dst, keep_suffix=".imatrix"):
    fh = open(src, "rb")
    head = fh.read(64 << 20)
    magic = head[:4]
    if magic != b"HFQM":
        raise SystemExit("not an HFQM container: {!r}".format(magic))

    je = json_end_of(head, 32)
    meta = json.loads(head[32:je].decode("utf-8", "replace"))

    pos = je
    (n,) = struct.unpack_from("<I", head, pos)
    pos += 4

    entries = []          # (name, qt, shape, group_size, data_size, raw_index_bytes)
    for _ in range(n):
        start = pos
        (nl,) = struct.unpack_from("<H", head, pos); pos += 2
        name = head[pos:pos + nl].decode("utf-8", "replace"); pos += nl
        qt = head[pos]; pos += 1
        nd = head[pos]; pos += 1
        shape = list(struct.unpack_from("<%dI" % nd, head, pos)); pos += 4 * nd
        (gs,) = struct.unpack_from("<I", head, pos); pos += 4
        (dsz,) = struct.unpack_from("<Q", head, pos); pos += 8
        entries.append((name, qt, shape, gs, dsz, head[start:pos]))

    data_offset = pos
    # Absolute payload offsets, accumulated in index order.
    offs, cur = [], data_offset
    for e in entries:
        offs.append(cur)
        cur += e[4]

    keep = [(i, e) for i, e in enumerate(entries) if e[0].endswith(keep_suffix)]
    if not keep:
        raise SystemExit("no tensors ending in {}".format(keep_suffix))

    new_meta = dict(meta)
    new_meta["artifact_kind"] = "calibration-imatrix-only"
    new_meta["n_hessian"] = 0
    new_meta["n_imatrix"] = len(keep)
    new_meta["derived_from"] = os.path.basename(src)
    new_meta["note"] = (
        "imatrix-only extract; AWQ consumes <name>.imatrix. --ldlq/GPTQ needs "
        "the full package because <name>.hessian was dropped."
    )
    meta_bytes = json.dumps(new_meta, separators=(",", ":")).encode()

    index = b"".join(e[5] for _, e in keep)
    out = open(dst, "wb")
    out.write(head[:32])
    out.write(meta_bytes)
    out.write(struct.pack("<I", len(keep)))
    out.write(index)
    for i, e in keep:
        fh.seek(offs[i])
        remaining = e[4]
        while remaining:
            chunk = fh.read(min(1 << 22, remaining))
            if not chunk:
                raise SystemExit("short read on payload for {}".format(e[0]))
            out.write(chunk)
            remaining -= len(chunk)
    out.close()

    src_sz, dst_sz = os.path.getsize(src), os.path.getsize(dst)
    print("  kept {} imatrix tensors of {}".format(len(keep), n))
    print("  {:,} B -> {:,} B  ({:.0f}x smaller)".format(src_sz, dst_sz, src_sz / max(dst_sz, 1)))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: extract_imatrix_only.py <full.calib.hfq> <out.imatrix.hfq>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
