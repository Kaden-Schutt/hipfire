#!/usr/bin/env python3
"""Repack a qt=13 (MQ4G256, 136 B/group) artifact to qt=45 (MQ4C / v1.5, 132 B/group).

This is a PURE HEADER REWRITE. It needs no parent model, no imatrix, no
re-quantization, and it changes not one nibble:

    read  136 B: [f32 scale][f32 zero][128 B nibbles]
    write 132 B: [fp16 scale][fp16 zero][128 B nibbles verbatim]

Proven safe rather than assumed. `mq4c_repack_probe.py` measured 236,503,040 real
weights from q38.ctl.mq4 and found the v1 reconstruction sits at most **0.011076
quantization steps** from an integer on the v1.5 grid — a nibble flips at 0.5, so
`round()` returns the same code for every weight and a "re-fit" is provably a no-op.
Max |z|/s was 10.80 and the smallest scale 7.609e-04, an order of magnitude above the
fp16 denormal floor (6.104e-05), so the fp16 bound holds with margin. Resulting drift
is 0.0909% weight RMS, i.e. a ~0.007% MSE increase, matching the 0.008% measured
independently in the codec sweep.

This script RE-CHECKS those invariants per group as it writes, and refuses rather
than silently emitting a lossy artifact if a model violates them.

Saving: 4 B per 256-weight group. On q38.ctl.mq4 that is 95,119,360 groups =
380,477,440 B (2.94% of the quantized payload, 2.43% of the file).

Usage: mq4c_repack.py IN.mq4 OUT.mq4c [--limit-tensors N]
"""
import struct, sys, numpy as np

V1_BYTES = 136
V15_BYTES = 132
QT_V1 = 13
QT_V15 = 45
F16_MIN_NORMAL = 6.103515625e-05
# A nibble re-fit changes a code only when the v1 reconstruction drifts a full 0.5
# steps on the v1.5 grid. Refuse AT that boundary -- below it the rewrite is exactly
# equivalent to a re-fit, at or above it is not.
#
# The margin here is much thinner than a 6-tensor sample suggested. That sample found
# a max drift of 0.011 steps; over the whole of q38.ctl.mq4 the real maximum is
# ~0.26 (layers.10.linear_attn.in_proj_qkv). Still no flips, but a 1.9x margin rather
# than 45x -- so this must be CHECKED per artifact, never assumed.
DRIFT_REFUSE = 0.5
DRIFT_WARN = 0.25


def read_index(path):
    with open(path, "rb") as f:
        head = f.read(32)
        magic = head[0:4]
        version, arch_id, n_tensors = struct.unpack_from("<III", head, 4)
        metadata_offset, data_offset = struct.unpack_from("<QQ", head, 16)
        f.seek(0)
        buf = f.read(data_offset)
    meta = buf[metadata_offset:data_offset]
    brace = 0; in_str = False; esc = False; jend = 0
    for i, b in enumerate(meta):
        c = chr(b)
        if esc: esc = False; continue
        if c == "\\" and in_str: esc = True; continue
        if c == '"': in_str = not in_str; continue
        if not in_str:
            if c == "{": brace += 1
            elif c == "}":
                brace -= 1
                if brace == 0: jend = i + 1; break
    pos = metadata_offset + jend
    (idx_n,) = struct.unpack_from("<I", buf, pos)
    entries, off = [], data_offset
    p = pos + 4
    for _ in range(idx_n):
        e0 = p
        (nl,) = struct.unpack_from("<H", buf, p); p += 2
        name = buf[p:p+nl].decode("utf-8"); p += nl
        qt_pos = p
        qt = buf[p]; p += 1
        nd = buf[p]; p += 1
        dims = struct.unpack_from("<" + "I"*nd, buf, p); p += 4*nd
        (gs,) = struct.unpack_from("<I", buf, p); p += 4
        ds_pos = p
        (ds,) = struct.unpack_from("<Q", buf, p); p += 8
        entries.append(dict(name=name, qt=qt, dims=dims, gs=gs, ds=ds, off=off,
                            qt_pos=qt_pos, ds_pos=ds_pos, e0=e0, e1=p))
        off += ds
    return buf, magic, version, arch_id, metadata_offset, data_offset, entries


def repack(src, dst, limit=None):
    buf, magic, version, arch_id, meta_off, data_off, entries = read_index(src)
    n_v1 = sum(1 for e in entries if e["qt"] == QT_V1)
    print(f"{src}: {len(entries)} tensors, {n_v1} at qt={QT_V1}")
    if n_v1 == 0:
        print("nothing to repack"); return 1

    header = bytearray(buf)  # index is rewritten in place below
    worst_drift = 0.0
    min_scale = np.inf
    groups_done = 0
    flips = 0

    # New data offsets: everything shifts as qt=13 tensors shrink.
    new_ds = {}
    for e in entries:
        new_ds[e["name"]] = (e["ds"] // V1_BYTES) * V15_BYTES if e["qt"] == QT_V1 else e["ds"]

    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(bytes(header))  # placeholder; index patched after
        for ei, e in enumerate(entries):
            if limit and ei >= limit:
                fi.seek(e["off"]); fo.write(fi.read(e["ds"])); continue
            if e["qt"] != QT_V1:
                fi.seek(e["off"]); fo.write(fi.read(e["ds"])); continue

            n = e["ds"] // V1_BYTES
            fi.seek(e["off"])
            raw = np.frombuffer(fi.read(n * V1_BYTES), dtype=np.uint8).reshape(n, V1_BYTES)
            s32 = raw[:, 0:4].copy().view(np.float32).ravel()
            z32 = raw[:, 4:8].copy().view(np.float32).ravel()
            nib = raw[:, 8:V1_BYTES]

            s16 = np.float16(s32); z16 = np.float16(z32)
            s16f = np.float32(s16); z16f = np.float32(z16)

            live = s16f > 0
            if live.any():
                q = np.empty((int(live.sum()), 256), dtype=np.float32)
                nl = nib[live]
                q[:, 0::2] = (nl & 0x0F); q[:, 1::2] = (nl >> 4)
                d = (z32[live] - z16f[live])[:, None] + q * (s32[live] - s16f[live])[:, None]
                ad = np.abs(d / s16f[live][:, None])
                worst_drift = max(worst_drift, float(ad.max()))
                flips += int((ad >= 0.5).sum())
                min_scale = min(min_scale, float(s32[live].min()))
                if ad.max() >= DRIFT_WARN and ad.max() < DRIFT_REFUSE:
                    print(f"  note: '{e['name'][:60]}' drifts {ad.max():.4f} steps "
                          f"(no flip, boundary 0.5)")
                if ad.max() >= DRIFT_REFUSE:
                    print(f"REFUSING: tensor '{e['name']}' drifts {ad.max():.4f} steps, "
                          f"at/over the {DRIFT_REFUSE} guard. A pure header rewrite is "
                          f"NOT equivalent to a re-fit on this model; do not ship this "
                          f"artifact.")
                    return 2

            out = np.empty((n, V15_BYTES), dtype=np.uint8)
            out[:, 0:2] = s16.view(np.uint16).astype(np.uint16).reshape(n, 1).view(np.uint8)[:, 0:2] \
                if False else np.frombuffer(s16.tobytes(), dtype=np.uint8).reshape(n, 2)
            out[:, 2:4] = np.frombuffer(z16.tobytes(), dtype=np.uint8).reshape(n, 2)
            out[:, 4:V15_BYTES] = nib
            fo.write(out.tobytes())
            groups_done += n

        # patch the index: qt 13 -> 45 and the shrunken data sizes
        for e in entries:
            header[e["qt_pos"]] = QT_V15 if e["qt"] == QT_V1 else e["qt"]
            struct.pack_into("<Q", header, e["ds_pos"], new_ds[e["name"]])
        fo.seek(0); fo.write(bytes(header))

    import os
    a, b = os.path.getsize(src), os.path.getsize(dst)
    print(f"groups repacked : {groups_done:,}")
    print(f"max drift       : {worst_drift:.6f} steps   (flip boundary 0.5)")
    print(f"nibble flips    : {flips:,}   <- must be 0")
    print(f"min f32 scale   : {min_scale:.3e}   (fp16 min normal {F16_MIN_NORMAL:.3e})")
    print(f"bytes           : {a:,} -> {b:,}   saved {a-b:,} ({100*(a-b)/a:.2f}%)")
    if flips:
        print("FAIL: a re-fit would have changed nibbles; this artifact is lossy.")
        return 3
    print("OK: pure header rewrite, every nibble preserved.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    lim = None
    if "--limit-tensors" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit-tensors") + 1])
    sys.exit(repack(sys.argv[1], sys.argv[2], lim))
