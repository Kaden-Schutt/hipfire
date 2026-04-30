#!/usr/bin/env python3
"""sim_mq3.py — UPPER-BOUND simulation of 3-bit quantization on an MQ4 file.

What this script actually does:
  Reads an existing .mq4 file, finds all MQ4-G256 (quant_type=13) tensors,
  and snaps each 4-bit nibble (q4 in [0..15]) to the nearest of the
  8-element subset {0, 2, 4, 6, 9, 11, 13, 15} via lookup table SNAP_4.
  The 8 chosen indices are the closest-integer approximations of a uniform
  8-level grid spanning [0..15]. Engine reads the file unchanged: GEMV
  reconstructs as min + q4_sim * scale_4. Storage layout, scale, and min
  are preserved.

LIMITATION — this is NOT a faithful simulation of real MQ3-G256:
  Real MQ3 from f32 would compute q3 = round((w-min)*7/range) directly.
  This simulator instead computes q3' = round(q4_orig * 7/15) — a
  rounding-of-rounding because q4_orig is itself round((w-min)*15/range).
  For ~12% of weights near grid boundaries the double-rounding produces a
  different q3' than the f32-direct q3, with worst-case extra error of
  ~14% of range vs real MQ3's ~7% intrinsic 3-bit error. The simulator
  therefore PESSIMISTICALLY UPPER-BOUNDS the quality cost of MQ3:
  observed degradation is always at least as bad as real MQ3 would be.

  A faithful simulator would re-quantize from the original f32 / bf16
  safetensors weights, not from already-MQ4-quantized values. That's a
  separate larger script working on the pre-quantize input, not this one.
  Use this harness for fast go/no-go signals only — if even the
  upper-bound simulation produces fluent output, real MQ3 might be
  viable; if it collapses (as it does on Qwen3.5 0.8B/9B per the
  2026-04-30 ablation), real MQ3 might still work better but the gap
  must be characterized by re-quantizing from f32, not by reading more
  into this script's output.

Usage: ./scripts/sim_mq3.py <input.mq4> <output.mq4>   # requires +x bit
       python3 scripts/sim_mq3.py <input.mq4> <output.mq4>
"""
import json
import struct
import sys
from pathlib import Path

# Snap table: q4 in [0..15] -> nearest 3-bit grid level.
# q3 = round(q4 * 7/15); back to q4_storage = round(q3 * 15/7).
# Result: [0, 0, 2, 2, 4, 4, 6, 6, 9, 9, 11, 11, 13, 13, 15, 15]
SNAP_4 = [0, 0, 2, 2, 4, 4, 6, 6, 9, 9, 11, 11, 13, 13, 15, 15]
# Byte-level LUT: each byte holds 2 nibbles; map both at once.
SNAP_BYTE = bytes((SNAP_4[b & 0x0F] | (SNAP_4[(b >> 4) & 0x0F] << 4)) for b in range(256))

QT_MQ4_G256 = 13
GROUP_SIZE = 256
BLOCK_BYTES = 136  # 4 (scale) + 4 (min) + 128 (nibbles)


def parse_header_and_index(buf):
    """Return (arch_id, n_tensors, metadata_offset, data_offset, json_end_offset, tensors).
    Each tensor is dict with keys: name, qt, shape, group_size, data_offset, data_size."""
    assert buf[0:4] == b"HFQM", "not an HFQ/MQ file"
    arch_id = struct.unpack_from("<I", buf, 8)[0]
    n_tensors = struct.unpack_from("<I", buf, 12)[0]
    metadata_offset = struct.unpack_from("<Q", buf, 16)[0]
    data_offset = struct.unpack_from("<Q", buf, 24)[0]

    # Find end of JSON metadata by brace-matching
    p = metadata_offset
    depth = 0
    in_str = False
    esc = False
    json_end = 0
    while p < data_offset:
        b = buf[p]
        if esc:
            esc = False
        elif in_str and b == 0x5C:  # backslash
            esc = True
        elif b == 0x22:  # quote
            in_str = not in_str
        elif not in_str:
            if b == 0x7B:
                depth += 1
            elif b == 0x7D:
                depth -= 1
                if depth == 0:
                    json_end = p + 1
                    break
        p += 1
    assert json_end > 0, "JSON not terminated"

    # Tensor index follows metadata
    pos = json_end
    idx_n = struct.unpack_from("<I", buf, pos)[0]
    assert idx_n == n_tensors, f"index count {idx_n} != header {n_tensors}"
    pos += 4

    tensors = []
    cum = data_offset
    for _ in range(n_tensors):
        name_len = struct.unpack_from("<H", buf, pos)[0]
        pos += 2
        name = buf[pos:pos + name_len].decode("utf-8")
        pos += name_len
        qt = buf[pos]
        pos += 1
        n_dims = buf[pos]
        pos += 1
        shape = list(struct.unpack_from(f"<{n_dims}I", buf, pos))
        pos += 4 * n_dims
        group_size = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        data_size = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8
        tensors.append({
            "name": name, "qt": qt, "shape": shape,
            "group_size": group_size,
            "data_offset": cum, "data_size": data_size,
        })
        cum += data_size

    return arch_id, n_tensors, metadata_offset, data_offset, json_end, tensors


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    src_path = Path(sys.argv[1])
    dst_path = Path(sys.argv[2])

    src = src_path.read_bytes()
    arch_id, n_tensors, metadata_offset, data_offset, json_end, tensors = parse_header_and_index(src)
    print(f"arch_id={arch_id}  n_tensors={n_tensors}  data_offset={data_offset:#x}  total_size={len(src):,}")

    # Mutate via bytes.translate(SNAP_BYTE) on each block's nibble region.
    # bytes.translate is a single C call that applies a 256-byte LUT to a byte
    # string — fast and standard-library only. Per block: bytes [0..8] are
    # scale+min (preserved), bytes [8..136] are 128 nibble pairs (translated).
    dst = bytearray(src)

    mq4_count = 0
    mutated_groups = 0
    for t in tensors:
        if t["qt"] != QT_MQ4_G256:
            continue
        mq4_count += 1
        ds = t["data_size"]
        do = t["data_offset"]
        n_blocks, rem = divmod(ds, BLOCK_BYTES)
        if rem != 0:
            print(f"  WARN: tensor {t['name']} has data_size {ds} not a multiple of {BLOCK_BYTES} — skipping", file=sys.stderr)
            continue
        for b in range(n_blocks):
            nibble_start = do + b * BLOCK_BYTES + 8
            nibble_end = nibble_start + 128
            # Slice → bytes (copy) → translate (C call, 8× LUT-speed) → write back
            translated = bytes(dst[nibble_start:nibble_end]).translate(SNAP_BYTE)
            dst[nibble_start:nibble_end] = translated
        mutated_groups += n_blocks

    print(f"mutated tensors: {mq4_count}/{n_tensors}  groups touched: {mutated_groups:,}")

    dst_path.write_bytes(bytes(dst))
    print(f"wrote {dst_path}  ({dst_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
