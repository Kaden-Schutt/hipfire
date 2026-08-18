#!/usr/bin/env python3
"""Does a 136 B -> 132 B (v1 -> v1.5 / mq4c) repack need a nibble re-fit, or is it a
pure header rewrite?

v1  (qt=13): [0..4) f32 scale, [4..8) f32 zero, [8..136) nibbles   -- 136 B/group
v1.5 (qt=45): [0..2) fp16 scale, [2..4) fp16 zero, [4..132) nibbles -- 132 B/group

Same per-256 affine grid; only header precision changes. The spec says round-tripping
the header through fp16 BEFORE quantizing is mandatory, so the naive repack (rewrite
header, keep nibbles) is formally a different operation from a proper re-fit
    q' = round((w_hat - z16) / s16),  w_hat = z32 + q*s32
This measures whether they differ AT ALL on real weights.

Claim under test: drift/step = dz/s16 + q*ds/s16 stays far below the 0.5-step rounding
boundary, so q' == q everywhere and the repack is a pure header rewrite. That holds
only while |z|/s is moderate and s avoids fp16 denormals (min normal 6.104e-5), which
real tensors might violate -- hence measuring rather than asserting.
"""
import struct, sys, numpy as np

GROUP_BYTES = 136
F16_MIN_NORMAL = 6.103515625e-05


def index(path):
    with open(path, "rb") as f:
        head = f.read(32)
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
    (idx_n,) = struct.unpack_from("<I", buf, pos); pos += 4
    out, off = [], data_offset
    for _ in range(idx_n):
        (nl,) = struct.unpack_from("<H", buf, pos); pos += 2
        name = buf[pos:pos+nl].decode("utf-8"); pos += nl
        qt = buf[pos]; pos += 1
        nd = buf[pos]; pos += 1
        dims = struct.unpack_from("<" + "I"*nd, buf, pos); pos += 4*nd
        (gs,) = struct.unpack_from("<I", buf, pos); pos += 4
        (ds,) = struct.unpack_from("<Q", buf, pos); pos += 8
        out.append((name, qt, dims, gs, ds, off))
        off += ds
    return out


def probe(path, want_qt=13, max_tensors=6, max_groups_per=200_000):
    tens = [t for t in index(path) if t[1] == want_qt]
    if not tens:
        print(f"no qt={want_qt} tensors in {path}")
        return
    print(f"{path}: {len(tens)} qt={want_qt} tensors")
    # Spread the sample across depth rather than taking the first N adjacent tensors.
    pick = [tens[i] for i in np.linspace(0, len(tens) - 1, min(max_tensors, len(tens))).astype(int)]

    tot_groups = tot_weights = 0
    flips = 0
    max_delta = 0.0
    min_s = np.inf
    max_zs = 0.0
    denorm_groups = 0
    sq_v1 = 0.0   # sum of squared v1 reconstruction magnitude
    sq_err = 0.0  # sum of squared (repacked - v1)

    with open(path, "rb") as f:
        for (name, qt, dims, gs, ds, off) in pick:
            ngroups = ds // GROUP_BYTES
            n = int(min(ngroups, max_groups_per))
            f.seek(off)
            raw = np.frombuffer(f.read(n * GROUP_BYTES), dtype=np.uint8).reshape(n, GROUP_BYTES)

            s32 = raw[:, 0:4].copy().view(np.float32).ravel()
            z32 = raw[:, 4:8].copy().view(np.float32).ravel()
            nib = raw[:, 8:136]

            # unpack nibbles: byte i holds weight 2i (low) and 2i+1 (high)
            q = np.empty((n, 256), dtype=np.float32)
            q[:, 0::2] = (nib & 0x0F)
            q[:, 1::2] = (nib >> 4)

            s16 = np.float32(np.float16(s32))
            z16 = np.float32(np.float16(z32))

            live = s16 > 0
            if not live.any():
                continue
            # delta in units of the NEW step: how far the v1 reconstruction sits from
            # an integer on the v1.5 grid. |delta| >= 0.5 would flip a nibble.
            ds_ = (s32 - s16)[live]
            dz_ = (z32 - z16)[live]
            ql = q[live]
            delta = (dz_[:, None] + ql * ds_[:, None]) / s16[live][:, None]
            ad = np.abs(delta)
            max_delta = max(max_delta, float(ad.max()))
            flips += int((ad >= 0.5).sum())

            w_v1 = z32[live][:, None] + ql * s32[live][:, None]
            w_15 = z16[live][:, None] + ql * s16[live][:, None]
            sq_v1 += float((w_v1.astype(np.float64) ** 2).sum())
            sq_err += float(((w_15 - w_v1).astype(np.float64) ** 2).sum())

            min_s = min(min_s, float(s32[live].min()))
            nz = s32[live] > 0
            max_zs = max(max_zs, float((np.abs(z32[live][nz]) / s32[live][nz]).max()))
            denorm_groups += int((s32[live] < F16_MIN_NORMAL).sum())

            tot_groups += int(live.sum())
            tot_weights += int(live.sum()) * 256
            print(f"  {name[:58]:58s} groups={int(live.sum()):>7d}")

    print()
    print(f"groups probed        : {tot_groups:,}")
    print(f"weights probed       : {tot_weights:,}")
    print(f"max |delta| (steps)  : {max_delta:.6f}   (nibble flips at 0.5)")
    print(f"nibble flips needed  : {flips:,}   <- 0 means the re-fit is a NO-OP")
    print(f"min f32 scale        : {min_s:.3e}   (fp16 min normal {F16_MIN_NORMAL:.3e})")
    print(f"groups w/ denormal s : {denorm_groups:,}")
    print(f"max |zero| / scale   : {max_zs:.2f}")
    rel = (sq_err / sq_v1) ** 0.5 if sq_v1 else 0.0
    print(f"repack rel RMS drift : {rel:.3e}  ({100*rel:.4f}% of the v1 reconstruction)")
    print()
    if flips == 0:
        print("VERDICT: pure header rewrite. Every nibble is unchanged, so a v1 -> v1.5")
        print("repack needs no decode/re-encode pass and no parent model -- rewrite 8 header")
        print("bytes as 4, shift the payload 8->4, restride 136->132. The fp16 drift never")
        print("reaches a rounding boundary, so 'naive truncation' and 'proper re-fit' are")
        print("the SAME operation on this artifact.")
    else:
        print(f"VERDICT: re-fit REQUIRED. {flips:,} weights would land on a different level,")
        print("so a naive header truncation is NOT equivalent to the measured 0.008% arm.")


if __name__ == "__main__":
    probe(sys.argv[1] if len(sys.argv) > 1 else "/home/kaden/qcal/q38.ctl.mq4")
