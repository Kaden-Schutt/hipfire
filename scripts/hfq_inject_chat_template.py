#!/usr/bin/env python3
"""Inject a chat_template.jinja into an existing .hfq's metadata WITHOUT re-quantizing.

HFQ layout: [32B header] [metadata JSON] [tensor index (sizes, no offsets)] [tensor data].
The index stores per-tensor *sizes*; the loader computes offsets cumulatively from the
header's data_offset (hfq.rs:163 `cumulative_offset = data_offset`). So we only grow the
metadata, bump data_offset by the delta, and copy the index + tensor bytes through verbatim.
"""
import argparse
import struct, json, shutil

def find_json_end(buf: bytes) -> int:
    depth = 0; in_str = False; esc = False
    for i, b in enumerate(buf):
        if esc: esc = False; continue
        if b == 0x5c and in_str: esc = True; continue   # backslash inside string
        if b == 0x22: in_str = not in_str; continue       # quote
        if not in_str:
            if b == 0x7b: depth += 1                       # {
            elif b == 0x7d:                                # }
                depth -= 1
                if depth == 0: return i + 1
    raise ValueError("no matching close brace for metadata JSON")

def inject(src: str, dst: str, jinja_path: str) -> bool:
    with open(src, "rb") as f:
        prefix = bytearray(f.read(0))  # placeholder
        head = f.read(32)
        assert head[:4] == b"HFQM", f"not an HFQM container: {head[:4]!r}"
        metadata_offset = struct.unpack_from("<Q", head, 16)[0]
        data_offset = struct.unpack_from("<Q", head, 24)[0]
        f.seek(0)
        prefix = bytearray(f.read(metadata_offset))        # header + any pre-metadata gap
        meta_region = f.read(data_offset - metadata_offset)  # [metadata JSON][index][pad]

    je = find_json_end(meta_region)
    md = json.loads(meta_region[:je].decode("utf-8"))
    index_and_pad = meta_region[je:]

    tc = md.get("tokenizer_config")
    if not isinstance(tc, dict):
        tc = {}
    existing = tc.get("chat_template")
    if isinstance(existing, str) and existing.strip():
        print(f"  {src}: already has a chat_template ({len(existing)} chars); skipping")
        return False

    tpl = open(jinja_path, encoding="utf-8").read()
    tc["chat_template"] = tpl
    md["tokenizer_config"] = tc
    new_meta = json.dumps(md, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    new_data_offset = metadata_offset + len(new_meta) + len(index_and_pad)
    struct.pack_into("<Q", prefix, 24, new_data_offset)    # patch data_offset only

    with open(dst, "wb") as out:
        out.write(prefix)
        out.write(new_meta)
        out.write(index_and_pad)
        with open(src, "rb") as f:
            f.seek(data_offset)
            shutil.copyfileobj(f, out, 64 * 1024 * 1024)

    print(f"  {src} -> {dst}: data_offset {data_offset} -> {new_data_offset} "
          f"(+{new_data_offset - data_offset}), template {len(tpl)} chars")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inject chat_template.jinja into an HFQM container without re-quantizing.",
    )
    parser.add_argument("src", help="source .hfq/.hfqm file")
    parser.add_argument("dst", help="destination file to write")
    parser.add_argument("jinja", help="chat_template.jinja path")
    args = parser.parse_args()
    inject(args.src, args.dst, args.jinja)
