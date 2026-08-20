#!/usr/bin/env python3
"""Repack a qt=13 (MQ4G256, 136 B/group) artifact to qt=45 (MQ4C pad, 136 B/group).

This is a PURE HEADER REWRITE. It needs no parent model, no imatrix, no
re-quantization, and it changes not one nibble:

    read  136 B interleaved: [f32 scale][f32 zero][128 B nibbles]  (per group, v1)
    write 136 B interleaved pad: [0..4) fp16 header, [4..8) zero padding, [8..136) 128 B nibbles
      where for linear group idx i = row*gpr+g (gpr=K/256):
        header dword at A + i*136         : packed low fp16 scale, high fp16 zero
        zero padding at A + i*136 + 4     : 4 zero bytes
        payload at    A + i*136 + 8       : 128 B nibbles verbatim (same offset as v1)
    total per tensor is n*136 = m*gpr*136, the SAME SIZE as v1 so MQ4C_GROUP_BYTES=136.
    The 4 padding bytes are the deliberate price of putting the payload at +8 where v1
    has it, because a 132 B stride left the payload 4-byte aligned half the time. The
    2.43% size win of the earlier planar layout is deliberately given up — this repack
    is a SAME-SIZE transform (136 -> 136) that trades size for alignment and speed.

Rounding the v1 f32 header to fp16 perturbs reconstructed weights relative to
v1. A nibble re-fit changes a code only when that drift reaches 0.5
quantization steps on the v1.5 grid. fp16 round-trip safety for scales is
checked per artifact: a positive finite f32 scale that becomes non-positive or
non-finite in fp16 is refused (subnormal-but-nonzero fp16 remains valid).

This script checks those invariants for every converted group and refuses
rather than silently emitting an invalid artifact.

This is now SAME-SIZE: 136 -> 136, so it saves 0 bytes. The earlier 95M-group saving
of 380,477,440 B (2.43% file) no longer applies — alignment is bought with those bytes.

Usage: mq4c_repack.py IN.mq4 OUT.mq4c [--limit-tensors N]
"""
import struct, sys, numpy as np

V1_BYTES = 136
V15_BYTES = 136
QT_V1 = 13
QT_V15 = 45
F16_MIN_NORMAL = 6.103515625e-05
F16_MIN_SUBNORMAL = 5.960464477539063e-08  # 2**-24; smallest positive fp16
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


def _json_end(buf, start, end):
    """Return exclusive end offset of a balanced top-level JSON object in buf[start:end]."""
    brace_depth = 0
    in_string = False
    escape = False
    for i in range(start, end):
        b = buf[i]
        if escape:
            escape = False
            continue
        if b == 0x5C and in_string:  # backslash
            escape = True
            continue
        if b == 0x22:  # quote
            in_string = not in_string
            continue
        if in_string:
            continue
        if b == 0x7B:  # {
            brace_depth += 1
        elif b == 0x7D:  # }
            brace_depth -= 1
            if brace_depth == 0:
                return i + 1
    raise AssertionError("metadata JSON not brace-terminated")


def read_index(path):
    """Parse HFQM (or legacy HFQ\\0) header+metadata+index, bounded to [0:data_offset].

    Returns buf = bytearray of exactly bytes [0:data_offset] so the caller can
    write it as the output placeholder and patch qt/ds in-place. Never loads
    tensor payloads.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        assert len(magic) == 4, "truncated header"
        if magic == b"HFQM":
            rest = f.read(28)
            assert len(rest) == 28, "truncated HFQM header"
            header = magic + rest
            version = struct.unpack_from("<I", header, 4)[0]
            arch_id = struct.unpack_from("<I", header, 8)[0]
            n_tensors = struct.unpack_from("<I", header, 12)[0]
            metadata_offset = struct.unpack_from("<Q", header, 16)[0]
            data_offset = struct.unpack_from("<Q", header, 24)[0]
            assert metadata_offset >= 32, f"metadata_offset {metadata_offset} < 32"
            assert metadata_offset <= data_offset, (
                f"metadata_offset {metadata_offset} > data_offset {data_offset}"
            )
            assert data_offset >= 32, f"data_offset {data_offset} too small"
            f.seek(0)
            buf = bytearray(f.read(data_offset))
            assert len(buf) == data_offset, (
                f"truncated HFQM prefix: need {data_offset}, got {len(buf)}"
            )

            json_end = _json_end(buf, metadata_offset, data_offset)
            off = json_end
            assert off + 4 <= data_offset, "index count crosses data_offset"
            idx_n = struct.unpack_from("<I", buf, off)[0]
            off += 4
            assert idx_n == n_tensors, (
                f"index count {idx_n} != header n_tensors {n_tensors}"
            )

            entries = []
            for _ in range(n_tensors):
                assert off + 2 <= data_offset, "name_len crosses data_offset"
                name_len = struct.unpack_from("<H", buf, off)[0]
                off += 2
                assert off + name_len <= data_offset, "name crosses data_offset"
                name = bytes(buf[off:off + name_len]).decode()
                off += name_len
                assert off + 2 <= data_offset, "qt/ndim crosses data_offset"
                qt_pos = off
                qt = buf[off]
                off += 1
                ndim = buf[off]
                off += 1
                assert off + ndim * 4 <= data_offset, "dims cross data_offset"
                shape = []
                for __ in range(ndim):
                    shape.append(struct.unpack_from("<I", buf, off)[0])
                    off += 4
                assert off + 4 + 8 <= data_offset, "group_size/ds crosses data_offset"
                off += 4  # group_size (unused by repack)
                ds_pos = off
                ds = struct.unpack_from("<Q", buf, off)[0]
                off += 8
                entries.append(dict(
                    name=name, qt=qt, ds=ds, qt_pos=qt_pos, ds_pos=ds_pos, shape=shape,
                ))
            assert off <= data_offset, "index overrun past data_offset"

            cur = data_offset
            for e in entries:
                e["off"] = cur
                cur += e["ds"]
            return buf, magic, version, arch_id, metadata_offset, data_offset, entries

        if magic == b"HFQ\x00":
            # Legacy layout: 4+4+4+8+8+8 = 36-byte fixed header, then dense index.
            rest = f.read(32)
            assert len(rest) == 32, "truncated legacy HFQ header"
            header = magic + rest
            version = struct.unpack_from("<I", header, 4)[0]
            arch_id = struct.unpack_from("<I", header, 8)[0]
            metadata_offset = struct.unpack_from("<Q", header, 12)[0]
            data_offset = struct.unpack_from("<Q", header, 20)[0]
            n_tensors = struct.unpack_from("<Q", header, 28)[0]
            assert data_offset >= 36, f"legacy data_offset {data_offset} too small"
            f.seek(0)
            buf = bytearray(f.read(data_offset))
            assert len(buf) == data_offset, (
                f"truncated legacy HFQ prefix: need {data_offset}, got {len(buf)}"
            )
            off = 36
            entries = []
            for _ in range(n_tensors):
                assert off + 8 <= data_offset, "legacy name_len crosses data_offset"
                name_len = struct.unpack_from("<Q", buf, off)[0]
                off += 8
                assert off + name_len <= data_offset, "legacy name crosses data_offset"
                name = bytes(buf[off:off + name_len]).decode()
                off += name_len
                assert off + 1 + 8 + 8 + 8 <= data_offset, "legacy entry crosses data_offset"
                qt_pos = off
                qt = buf[off]
                off += 1
                ds_pos = off
                ds = struct.unpack_from("<Q", buf, off)[0]
                off += 8
                off += 8  # stored file offset (unused; recompute from data_offset)
                ndim = struct.unpack_from("<Q", buf, off)[0]
                off += 8
                assert off + ndim * 8 <= data_offset, "legacy dims cross data_offset"
                shape = []
                for __ in range(ndim):
                    shape.append(struct.unpack_from("<Q", buf, off)[0])
                    off += 8
                entries.append(dict(
                    name=name, qt=qt, ds=ds, qt_pos=qt_pos, ds_pos=ds_pos, shape=shape,
                ))
            assert off <= data_offset, "legacy index overrun past data_offset"
            cur = data_offset
            for e in entries:
                e["off"] = cur
                cur += e["ds"]
            return buf, magic, version, arch_id, metadata_offset, data_offset, entries

        raise AssertionError(f"bad magic {magic!r}; expected HFQM or legacy HFQ\\0")



def repack(src, dst, limit=None):
    import os
    import tempfile

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
    # qt_pos of entries actually converted this run (limit / non-qt13 leave these alone)
    converted_qt_pos = []

    # Same-size 136→136: data sizes and offsets stay unchanged; do not rewrite ds.
    dst_dir = os.path.dirname(os.path.abspath(dst)) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".mq4c_repack_", suffix=".tmp", dir=dst_dir
    )
    try:
        with open(src, "rb") as fi, os.fdopen(tmp_fd, "wb") as fo:
            tmp_fd = -1  # ownership transferred to fo
            fo.write(bytes(header))  # placeholder; index patched after
            n_converted = 0
            for ei, e in enumerate(entries):
                if e["qt"] != QT_V1:
                    fi.seek(e["off"]); fo.write(fi.read(e["ds"])); continue
                # Distinguish limit is None (unlimited) from limit == 0 (convert none).
                if limit is not None and n_converted >= limit:
                    fi.seek(e["off"]); fo.write(fi.read(e["ds"])); continue

                n = e["ds"] // V1_BYTES
                fi.seek(e["off"])
                raw = np.frombuffer(fi.read(n * V1_BYTES), dtype=np.uint8).reshape(n, V1_BYTES)
                s32 = raw[:, 0:4].copy().view(np.float32).ravel()
                z32 = raw[:, 4:8].copy().view(np.float32).ravel()
                nib = raw[:, 8:V1_BYTES]

                s16 = np.float16(s32); z16 = np.float16(z32)
                s16f = np.float32(s16); z16f = np.float32(z16)

                # Fail closed: positive finite f32 scale must stay positive finite in fp16.
                # Subnormal-but-nonzero fp16 is valid; only underflow-to-zero / non-finite fails.
                src_live = np.isfinite(s32) & (s32 > 0)
                fp16_bad = src_live & (~np.isfinite(s16f) | (s16f <= 0))
                if fp16_bad.any():
                    n_bad = int(fp16_bad.sum())
                    min_bad = float(s32[fp16_bad].min())
                    print(f"REFUSING: tensor '{e['name']}' has {n_bad} positive finite "
                          f"scale(s) that round to non-positive/non-finite fp16 "
                          f"(min source scale {min_bad:.3e}).")
                    return 4

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

                # Pad layout: per group 136 B: [0..4) header dword, [4..8) zeros, [8..136) nibbles
                # payload at +8 is exactly where v1 puts it; stride 136 same as v1.
                assert n * V15_BYTES == n * 136, "pad size invariant"
                hdr = np.empty((n, 4), dtype=np.uint8)
                hdr[:, 0:2] = np.frombuffer(s16.tobytes(), dtype=np.uint8).reshape(n, 2)
                hdr[:, 2:4] = np.frombuffer(z16.tobytes(), dtype=np.uint8).reshape(n, 2)
                payload = nib  # (n, 128) verbatim
                assert payload.nbytes == n * 128
                assert hdr.nbytes == n * 4
                # Interleaved write: header + 4 zero pad + payload per group
                pad = np.zeros((n, 4), dtype=np.uint8)
                # Build interleaved groups: 4+4+128 =136 per group
                # Use numpy concatenation per group would be heavy; write in loop in chunks
                # For speed, interleave via concatenation of (hdr, pad, payload) along axis 1
                interleaved = np.concatenate([hdr, pad, payload], axis=1)  # (n, 136)
                assert interleaved.nbytes == n * V15_BYTES
                fo.write(interleaved.tobytes())
                groups_done += n
                n_converted += 1
                converted_qt_pos.append(e["qt_pos"])

            # Patch qt 13→45 only for tensors actually converted; leave skipped qt13 alone.
            # Same-size transform: no ds rewrite.
            for qt_pos in converted_qt_pos:
                header[qt_pos] = QT_V15
            fo.seek(0); fo.write(bytes(header))

        a = os.path.getsize(src)
        b = os.path.getsize(tmp_path)
        print(f"groups repacked : {groups_done:,}")
        print(f"max drift       : {worst_drift:.6f} steps   (flip boundary 0.5)")
        print(f"nibble flips    : {flips:,}   <- must be 0")
        print(f"min f32 scale   : {min_scale:.3e}   "
              f"(fp16 min normal {F16_MIN_NORMAL:.3e}, min subnormal {F16_MIN_SUBNORMAL:.3e})")
        if a == b:
            print(f"bytes           : {a:,} -> {b:,}   saved {a-b:,} ({100*(a-b)/a:.2f}%)  (pad is same-size, expected)")
        else:
            print(f"bytes           : {a:,} -> {b:,}   saved {a-b:,} ({100*(a-b)/a:.2f}%)")
            print(f"note: pad repack is SAME-SIZE (136->136), so 0 saved is expected; non-zero delta indicates input was not v1-stride")
        # Atomically install only after full write + index patch succeeded.
        os.replace(tmp_path, dst)
        tmp_path = None
        if flips:
            print("FAIL: a re-fit would have changed nibbles; this artifact is lossy.")
            return 3
        print("OK: pure header rewrite, every nibble preserved (pad layout).")
        return 0
    finally:
        if tmp_fd >= 0:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    lim = int(sys.argv[3].split("=")[1]) if len(sys.argv) > 3 and "--limit" in sys.argv[3] else None
    if len(sys.argv) > 3 and sys.argv[3].startswith("--limit"):
        lim = int(sys.argv[3].split("=")[1]) if "=" in sys.argv[3] else int(sys.argv[4])
    sys.exit(repack(sys.argv[1], sys.argv[2], lim))
