#!/usr/bin/env python3
"""Graft raw-dtype (F16) tensor layers into an existing HFQM trunk quant
*without* re-running the compute-heavy GPTQ/AWQ pass. The text tower is copied
verbatim from the base quant; the new layers are appended to the tensor index
and data section at F16. Stand-alone: pure stdlib + numpy, no GPU/engine deps.

SCOPE — this is the VISION tool. hipfire-quantize skips `model.visual.*` (unless
--include-vision) when quantizing the language tower; the runtime loads those
vision weights raw-F16 from the trunk and dequantizes at load. This tool grafts
them back on so you don't pay for a fresh GPTQ run just to add vision.

  *** NOT FOR MTP HEADS. *** An MTP head is NOT raw tensors appended to the
  trunk. It is a separate HFQM container (`*.mtp`, arch_id=21) that needs tensor
  renaming (mtp.fc.weight -> eh_proj, ...), FWHT rotation (seeds 42/1042) + MQ4G256
  /Q8F16 quantization of the 8 weight tensors, F32-RAW norms (no +1.0 offset), and
  specific metadata. Use hipfire's `mtp_extract` bin (on the feat/mtp branch):
      cargo run --release --bin mtp_extract -- --hf-dir <dir> --output X.mtp [--quant mq4|q8]
  and `mq4_merge_mtp` for the optional `.mq4-mtp` bundle. See the MTP file-format
  handover. This Python tool would produce a structurally invalid MTP head.

Validated 2026-05-25 against the canonical 9B -vl build: HF-sourced (bf16->f16)
output is tensor-data byte-identical to qwen3.5-9b.mq4-awq-gptq-f2-lmhead-a100-vl
(all 1009 tensors, 0 mismatches); spliced model loads and runs vision in hipfire.

VISION (add Qwen3.5-VL vision tower to the 9B language quant):
  python3 scripts/splice_layers.py splice \\
    --base   /data/hipfire/qwen3.5-9b.mq4 \\
    --from-hf <HF Qwen3.5-9B snapshot dir> \\
    --prefix model.visual. \\
    --out    /data/hipfire/qwen3.5-9b.mq4-vl.hfq

Sources: --from-hf <dir> (HF safetensors, bf16->f16) or --from-hfq <file>
(copy already-F16 layers from a prior splice). --align-4k re-pages the data
section (default drops padding, matching the original splice). --validate-against
<ref.hfq> byte/tensor-compares the result.

HFQM layout (little-endian):
  [0:4]   magic "HFQM"
  [4:8]   version (u32, =1)
  [8:12]  arch_id (u32)
  [12:16] n_tensors (u32)
  [16:24] metadata_offset (u64, =32)
  [24:32] data_offset (u64)        # the Rust writer 4KB-aligns this
  [32 .. 32+meta_size]  JSON metadata
  index: u32 count, then per tensor:
         name_len u16, name utf8, qt u8, n_dims u8, shape u32*n_dims,
         group_size u32, data_size u64
  [data_offset ..] concatenated tensor data, in index order

Subcommands:
  analyze   <a.hfq> [b.hfq]            structure dump (+ diff if b given)
  splice    --base B --from-hfq V --out O [--validate-against R]
  splice    --base B --from-hf  DIR --out O [--validate-against R]
"""
import argparse, json, struct, sys

HEADER_SIZE = 32
QT = {0: "F32", 1: "F16", 6: "HFQ4G256", 7: "HFQ4G128", 12: "MQ3G256", 13: "MQ4G256", 30: "BF16"}
QT_F16 = 1


def read_hfqm(path):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[0:4] == b"HFQM", f"{path}: not HFQM"
    version = struct.unpack_from("<I", buf, 4)[0]
    arch_id = struct.unpack_from("<I", buf, 8)[0]
    n_tensors = struct.unpack_from("<I", buf, 12)[0]
    metadata_offset = struct.unpack_from("<Q", buf, 16)[0]
    data_offset = struct.unpack_from("<Q", buf, 24)[0]
    # metadata is JSON ending where the tensor-index count makes sense; brace-match.
    p, depth, in_str, esc, json_end = metadata_offset, 0, False, False, 0
    while p < data_offset:
        b = buf[p]
        if esc: esc = False
        elif in_str and b == 0x5C: esc = True
        elif b == 0x22: in_str = not in_str
        elif not in_str:
            if b == 0x7B: depth += 1
            elif b == 0x7D:
                depth -= 1
                if depth == 0:
                    json_end = p + 1; break
        p += 1
    assert json_end > 0, "JSON not terminated"
    meta_bytes = buf[metadata_offset:json_end]
    cfg = json.loads(meta_bytes.decode("utf-8"))

    pos = json_end
    idx_count = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    assert idx_count == n_tensors, f"index {idx_count} != header {n_tensors}"
    index_start = json_end
    entries = []          # list of dicts; entry_bytes preserved for byte-exact copy
    cum = data_offset
    for _ in range(n_tensors):
        e0 = pos
        name_len = struct.unpack_from("<H", buf, pos)[0]; pos += 2
        name = buf[pos:pos + name_len].decode("utf-8"); pos += name_len
        qt = buf[pos]; pos += 1
        n_dims = buf[pos]; pos += 1
        shape = list(struct.unpack_from(f"<{n_dims}I", buf, pos)); pos += 4 * n_dims
        group_size = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        data_size = struct.unpack_from("<Q", buf, pos)[0]; pos += 8
        entries.append({"name": name, "qt": qt, "n_dims": n_dims, "shape": shape,
                        "group_size": group_size, "data_size": data_size,
                        "data_off": cum, "entry_bytes": buf[e0:pos]})
        cum += data_size
    index_end = pos
    padding = buf[index_end:data_offset]
    return {"buf": buf, "version": version, "arch_id": arch_id, "n_tensors": n_tensors,
            "metadata_offset": metadata_offset, "data_offset": data_offset,
            "meta_bytes": meta_bytes, "cfg": cfg, "entries": entries,
            "index_start": index_start, "index_end": index_end, "padding": padding}


def encode_entry(name, qt, shape, group_size, data_size):
    nb = name.encode("utf-8")
    out = struct.pack("<H", len(nb)) + nb + struct.pack("<BB", qt, len(shape))
    for d in shape:
        out += struct.pack("<I", d)
    out += struct.pack("<I", group_size) + struct.pack("<Q", data_size)
    return out


def cmd_analyze(args):
    a = read_hfqm(args.a)
    print(f"{args.a}")
    print(f"  version={a['version']} arch_id={a['arch_id']} n_tensors={a['n_tensors']}")
    print(f"  metadata_offset={a['metadata_offset']} meta_size={len(a['meta_bytes'])}")
    print(f"  index_start={a['index_start']} index_end={a['index_end']} index_size={a['index_end']-a['index_start']}")
    print(f"  data_offset={a['data_offset']} padding={len(a['padding'])} (4KB-aligned={a['data_offset']%4096==0})")
    print(f"  file_size={len(a['buf'])} data_bytes={len(a['buf'])-a['data_offset']}")
    if not args.b:
        return
    b = read_hfqm(args.b)
    an = {e["name"] for e in a["entries"]}
    bn = {e["name"] for e in b["entries"]}
    only_b = [e for e in b["entries"] if e["name"] not in an]
    print(f"\n{args.b}")
    print(f"  n_tensors={b['n_tensors']} data_offset={b['data_offset']} padding={len(b['padding'])}")
    print(f"  tensors only in b: {len(only_b)}  bytes={sum(e['data_size'] for e in only_b)/1e9:.3f}GB")
    print(f"  data_offset delta b-a: {b['data_offset']-a['data_offset']}  (a index_size diff vs b: {(b['index_end']-b['index_start'])-(a['index_end']-a['index_start'])})")
    # does b preserve a's padding? and is b = a's text data + only_b data?
    print(f"  a padding == b padding: {a['padding']==b['padding']}")


def collect_layers_from_hfq(src_hfq, prefixes):
    v = read_hfqm(src_hfq)
    out = []
    for e in v["entries"]:
        if any(e["name"].startswith(p) for p in prefixes):
            raw = v["buf"][e["data_off"]:e["data_off"] + e["data_size"]]
            out.append({"name": e["name"], "qt": e["qt"], "shape": e["shape"],
                        "group_size": e["group_size"], "data": raw})
    return out


def _bf16_bytes_to_f16_bytes(raw):
    import numpy as np
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.astype(np.float16).tobytes()


def _read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def collect_layers_from_hf(hf_dir, prefixes):
    """Read tensors whose name starts with any prefix from an HF safetensors
    checkpoint (sharded or single-file), cast to F16."""
    import glob, os
    import numpy as np
    idxs = glob.glob(os.path.join(hf_dir, "*.index.json"))
    if idxs:
        wm = json.load(open(idxs[0]))["weight_map"]
    else:
        st = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
        if not st:
            raise SystemExit(f"no safetensors in {hf_dir}")
        hdr, _ = _read_st_header(st[0])
        wm = {k: os.path.basename(st[0]) for k in hdr if k != "__metadata__"}
    names = [k for k in wm if any(k.startswith(p) for p in prefixes)]
    if not names:
        raise SystemExit(f"no tensors matching {prefixes} in {hf_dir}")
    cache = {}
    out = []
    for name in names:
        shard = os.path.join(hf_dir, wm[name])
        if shard not in cache:
            hdr, data_start = _read_st_header(shard)
            cache[shard] = (hdr, data_start, open(shard, "rb"))
        hdr, data_start, f = cache[shard]
        meta = hdr[name]
        s, e = meta["data_offsets"]
        f.seek(data_start + s)
        raw = f.read(e - s)
        dt = meta["dtype"]
        if dt == "BF16":
            data = _bf16_bytes_to_f16_bytes(raw)
        elif dt in ("F16", "FP16"):
            data = raw
        elif dt in ("F32", "FP32"):
            data = np.frombuffer(raw, dtype=np.float32).astype(np.float16).tobytes()
        else:
            raise SystemExit(f"unhandled dtype {dt} for {name}")
        out.append({"name": name, "qt": QT_F16, "shape": list(meta["shape"]),
                    "group_size": 0, "data": data})
    return out


def cmd_splice(args):
    base = read_hfqm(args.base)
    prefixes = args.prefix or ["model.visual.", "visual."]
    if args.from_hfq:
        extra = collect_layers_from_hfq(args.from_hfq, prefixes)
        src = f"{args.from_hfq} (prefixes={prefixes})"
    elif args.from_hf:
        extra = collect_layers_from_hf(args.from_hf, prefixes)
        src = f"{args.from_hf} (prefixes={prefixes})"
    else:
        raise SystemExit("need --from-hf <dir> or --from-hfq <file>")
    base_names = {e["name"] for e in base["entries"]}
    extra = [t for t in extra if t["name"] not in base_names]
    print(f"base: {base['n_tensors']} tensors, data_offset={base['data_offset']}")
    print(f"adding {len(extra)} tensors from {src} "
          f"({sum(len(t['data']) for t in extra)/1e9:.3f}GB, "
          f"qts={sorted(set(QT.get(t['qt'],t['qt']) for t in extra))})")

    new_n = base["n_tensors"] + len(extra)
    # new index = base entry bytes (verbatim) + encoded extra entries
    new_index_entries = b"".join(e["entry_bytes"] for e in base["entries"])
    for t in extra:
        new_index_entries += encode_entry(t["name"], t["qt"], t["shape"], t["group_size"], len(t["data"]))
    new_index = struct.pack("<I", new_n) + new_index_entries
    # The original splice dropped base's 4KB-alignment padding (the reference -vl
    # has padding=0); data follows the index directly. Optionally re-align with --align-4k.
    pad = b""
    if args.align_4k:
        unaligned = base["metadata_offset"] + len(base["meta_bytes"]) + len(new_index)
        pad = b"\x00" * ((-unaligned) % 4096)
    new_data_offset = base["metadata_offset"] + len(base["meta_bytes"]) + len(new_index) + len(pad)
    base_text_data = base["buf"][base["data_offset"]:]
    extra_data = b"".join(t["data"] for t in extra)

    header = (b"HFQM" + struct.pack("<I", base["version"]) + struct.pack("<I", base["arch_id"])
              + struct.pack("<I", new_n) + struct.pack("<Q", base["metadata_offset"])
              + struct.pack("<Q", new_data_offset))
    out_bytes = header + base["meta_bytes"] + new_index + pad + base_text_data + extra_data
    with open(args.out, "wb") as f:
        f.write(out_bytes)
    print(f"wrote {args.out}: {len(out_bytes)} bytes, n_tensors={new_n}, data_offset={new_data_offset}")

    if args.validate_against:
        ref = open(args.validate_against, "rb").read()
        if out_bytes == ref:
            print(f"VALIDATE: BYTE-IDENTICAL to {args.validate_against}")
        else:
            print(f"VALIDATE: differs (out={len(out_bytes)} ref={len(ref)})")
            # tensor-level comparison
            o = read_hfqm(args.out); r = read_hfqm(args.validate_against)
            od = {e["name"]: e for e in o["entries"]}; rd = {e["name"]: e for e in r["entries"]}
            print(f"  n_tensors out={o['n_tensors']} ref={r['n_tensors']}")
            mismatch = 0
            for name in rd:
                if name not in od:
                    print(f"  MISSING in out: {name}"); mismatch += 1; continue
                ob = o["buf"][od[name]["data_off"]:od[name]["data_off"]+od[name]["data_size"]]
                rb = r["buf"][rd[name]["data_off"]:rd[name]["data_off"]+rd[name]["data_size"]]
                if ob != rb:
                    print(f"  DATA differs: {name} (out {len(ob)}B ref {len(rb)}B)"); mismatch += 1
            print(f"  tensor-data mismatches: {mismatch}/{r['n_tensors']}")
            print(f"  first-diff byte offset: {next((i for i in range(min(len(out_bytes),len(ref))) if out_bytes[i]!=ref[i]), 'none')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze"); a.add_argument("a"); a.add_argument("b", nargs="?"); a.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("splice", help="Append layers (vision/MTP) from a source into a base quant.")
    s.add_argument("--base", required=True, help="base .hfq quant (text tower)")
    s.add_argument("--from-hf", help="HF safetensors checkpoint dir (bf16->f16)")
    s.add_argument("--from-hfq", help="existing .hfq to copy layers from (e.g. a prior -vl)")
    s.add_argument("--prefix", action="append",
                   help="tensor-name prefix to graft (repeatable). default: model.visual.,visual. "
                        "(NOT for MTP heads — those need the mtp_extract bin; see module docstring)")
    s.add_argument("--out", required=True)
    s.add_argument("--align-4k", action="store_true", help="4KB-align data section (default: drop padding to match reference splice)")
    s.add_argument("--validate-against")
    s.set_defaults(fn=cmd_splice)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
