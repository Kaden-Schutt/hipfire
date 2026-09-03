#!/usr/bin/env python3
"""G1: every escha_code tensor in the .hfq must be byte-identical to source.

Verbatim repack is the whole basis for claiming no codec loss, so this is a
memcmp against the tensor at its indexed offset — not a substring search,
which would be quadratic over a 12 GB file.

Because it is the SOLE evidence for that contract, the count is asserted, not
merely printed. An earlier version compared each discovered code tensor and
printed PASS when it discovered none — which a non-recursive `*.safetensors`
glob against an HF cache root (shards live under `snapshots/<sha>/`) produced
silently. The gate now fails on: no shards, no index, no code tensors, or a
count that disagrees with `model.safetensors.index.json`.

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


class GateFailure(Exception):
    """A G1 precondition that, unmet, would make the comparison vacuous."""


def expected_code_names(d):
    """The set of `.escha_code` tensor names the SOURCE claims to contain.

    Read from `model.safetensors.index.json`, not from the shards, so that
    "how many did we compare" is checked against an INDEPENDENT statement of
    how many there should be. Deriving the expectation from the same shard
    walk that produces the comparands would make the count assertion circular
    — a walk that finds nothing would also expect nothing and still pass.

    `rglob`, not `glob`: an HF cache root keeps the index under
    `snapshots/<sha>/`, and a non-recursive search there finds nothing.
    """
    root = Path(d)
    if not root.is_dir():
        raise GateFailure(f"source {d!r} is not a directory")
    idxs = sorted(root.rglob("model.safetensors.index.json"))
    if not idxs:
        raise GateFailure(
            f"no model.safetensors.index.json under {d!r} — cannot establish the "
            "expected escha_code count, so a zero-comparison PASS could not be "
            "distinguished from a real one"
        )
    if len(idxs) > 1:
        raise GateFailure(
            "more than one model.safetensors.index.json under "
            f"{d!r} ({', '.join(str(p) for p in idxs)}) — ambiguous source"
        )
    weight_map = json.loads(idxs[0].read_text())["weight_map"]
    names = {k for k in weight_map if k.endswith(".escha_code")}
    if not names:
        raise GateFailure(
            f"{idxs[0]} lists no .escha_code tensors — this is not an escha "
            "checkpoint, and G1 would otherwise compare nothing and pass"
        )
    return names


def safetensors_tensors(d):
    out = {}
    shards = sorted(Path(d).rglob("*.safetensors"))
    if not shards:
        raise GateFailure(
            f"no *.safetensors shards under {d!r} — nothing to compare. "
            "(This is the failure mode the count assertions exist for: the "
            "loop below would not execute and every counter would stay 0.)"
        )
    for shard in shards:
        raw = shard.read_bytes()
        (n,) = struct.unpack_from("<Q", raw, 0)
        hdr = json.loads(raw[8:8 + n])
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            if name in out:
                raise GateFailure(
                    f"tensor {name!r} appears in more than one shard under "
                    f"{d!r} — the source tree holds two copies of the model "
                    "and G1 would silently compare only one of them"
                )
            out[name] = raw[8 + n + s:8 + n + e]
    return out


def main(src, hfq_path):
    # Preconditions FIRST. Every one of these, unmet, produces an empty
    # `codes` dict, an unexecuted comparison loop, all-zero counters and — in
    # the pre-fix script — "G1 PASS". G1 is the sole evidence for the
    # verbatim-repack contract, so a vacuous pass is worse than no gate.
    try:
        expected = expected_code_names(src)
        st = safetensors_tensors(src)
    except GateFailure as e:
        print(f"G1 FAIL: {e}")
        return 1
    codes = {k: v for k, v in st.items() if k.endswith(".escha_code")}
    if not codes:
        print(
            "G1 FAIL: the shards under "
            f"{src!r} contain no .escha_code tensors, though the index lists "
            f"{len(expected)}"
        )
        return 1
    if set(codes) != expected:
        only_index = sorted(expected - set(codes))
        only_shards = sorted(set(codes) - expected)
        print(
            f"G1 FAIL: shard walk found {len(codes)} escha_code tensors, the "
            f"index lists {len(expected)}"
        )
        for n in only_index[:10]:
            print("    in index, not in shards:", n)
        for n in only_shards[:10]:
            print("    in shards, not in index:", n)
        return 1

    with open(hfq_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        idx = hfq_tensors(mm)
        missing = [n for n in codes if n not in idx]
        wrong_qt, mismatch, compared = [], [], 0
        for name, src_bytes in codes.items():
            if name in missing:
                continue
            qt, off, ln = idx[name]
            if qt not in ESCHA_QT:
                wrong_qt.append(f"{name}: quant_type {qt}")
            elif ln != len(src_bytes) or mm[off:off + ln] != src_bytes:
                mismatch.append(name)
            else:
                compared += 1
        print(f"escha_code tensors in the source index : {len(expected)}")
        print(f"escha_code tensors found in the shards : {len(codes)}")
        print(f"  absent from the .hfq index           : {len(missing)}")
        print(f"  wrong quant_type                     : {len(wrong_qt)}")
        print(f"  present but not byte-equal           : {len(mismatch)}")
        print(f"  byte-identical (memcmp'd)            : {compared}")
        for n in (missing + wrong_qt + mismatch)[:10]:
            print("   ", n)
        if missing or wrong_qt or mismatch:
            return 1
        # Belt and braces: the loop above cannot reach here with
        # `compared != len(expected)`, but this is the assertion whose absence
        # let the old script print PASS having compared nothing, so it is
        # stated rather than reasoned about.
        if compared != len(expected):
            print(
                f"G1 FAIL: compared {compared} code streams, expected "
                f"{len(expected)}"
            )
            return 1
    print(f"G1 PASS: {compared}/{len(expected)} code streams byte-identical")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <source-model-dir> <model.hfq>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))