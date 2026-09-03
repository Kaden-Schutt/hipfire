#!/usr/bin/env python3
"""G1: every escha_code tensor in the .hfq must be byte-identical to source.

Verbatim repack is the whole basis for claiming no codec loss, so this is a
memcmp against the tensor at its indexed offset — not a substring search,
which would be quadratic over a 12 GB file.

HFQ layout (see hipfire-quantize/src/hfq.rs::write_hfq):
  header 32B : magic[4] "HFQM", version u32, arch u32, n_tensors u32,
               metadata_offset u64, data_offset u64
  metadata   : JSON at metadata_offset
  index      : n_tensors u32, then per tensor
               name_len u16, name, quant_type u8, ndim u8,
               dims u32*ndim, group_size u32, data_len u64
  data       : at data_offset (4096-aligned), tensors concatenated in order
"""
import json, mmap, struct, sys
from pathlib import Path

ESCHA_QT = {42: "ESCHA2T16", 43: "ESCHA3T16"}


def hfq_tensors(mm):
    assert mm[:4] == b"HFQM", "not an HFQ file"
    version, arch, n_tensors = struct.unpack_from("<III", mm, 4)
    metadata_offset, data_offset = struct.unpack_from("<QQ", mm, 16)
    del version, arch
    # The index begins immediately after the metadata JSON blob, which the
    # writer emits with no length prefix — so walk the JSON to find its end.
    meta_end = metadata_offset
    depth, in_str, esc = 0, False, False
    while meta_end < data_offset:
        c = mm[meta_end]
        meta_end += 1
        if esc:
            esc = False
        elif in_str:
            if c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
        elif c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                break
    pos = meta_end
    (count,) = struct.unpack_from("<I", mm, pos)
    pos += 4
    assert count == n_tensors, f"index count {count} != header {n_tensors}"
    out, running = {}, 0
    for _ in range(n_tensors):
        (name_len,) = struct.unpack_from("<H", mm, pos)
        pos += 2
        name = bytes(mm[pos:pos + name_len]).decode()
        pos += name_len
        qt, ndim = struct.unpack_from("<BB", mm, pos)
        pos += 2
        pos += 4 * ndim
        pos += 4  # group_size
        (data_len,) = struct.unpack_from("<Q", mm, pos)
        pos += 8
        out[name] = (qt, data_offset + running, data_len)
        running += data_len
    return out


def safetensors_tensors(d):
    out = {}
    for shard in sorted(Path(d).glob("*.safetensors")):
        raw = shard.read_bytes()
        (n,) = struct.unpack_from("<Q", raw, 0)
        hdr = json.loads(raw[8:8 + n])
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            out[name] = raw[8 + n + s:8 + n + e]
    return out


def main(src, hfq_path):
    st = safetensors_tensors(src)
    codes = {k: v for k, v in st.items() if k.endswith(".escha_code")}
    with open(hfq_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        idx = hfq_tensors(mm)
        missing = [n for n in codes if n not in idx]
        wrong_qt, mismatch = [], []
        for name, src_bytes in codes.items():
            if name in missing:
                continue
            qt, off, ln = idx[name]
            if qt not in ESCHA_QT:
                wrong_qt.append(f"{name}: quant_type {qt}")
            elif ln != len(src_bytes) or mm[off:off + ln] != src_bytes:
                mismatch.append(name)
        print(f"escha_code tensors in source : {len(codes)}")
        print(f"  absent from the .hfq index : {len(missing)}")
        print(f"  wrong quant_type           : {len(wrong_qt)}")
        print(f"  present but not byte-equal : {len(mismatch)}")
        for n in (missing + wrong_qt + mismatch)[:10]:
            print("   ", n)
        if missing or wrong_qt or mismatch:
            return 1
    print("G1 PASS: every code stream is byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))