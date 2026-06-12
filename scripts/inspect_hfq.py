#!/usr/bin/env python3
"""inspect_hfq.py — show quant-type distribution and tensor summary for an HFQ file.

Usage:
  python3 scripts/inspect_hfq.py <model.hfq> [<model2.hfq> ...]
  python3 scripts/inspect_hfq.py --dist-only <model.hfq>   # machine-readable key=value

Output format  (default):
  model.hfq  (1.41 GiB, 320 tensors)
    qt  name         tensors   bytes     %
    ──────────────────────────────────────
     1  F16              36    54.2 MiB   3.6%  (e.g. linear_attn.A_log)
    16  BF16            284  1356.9 MiB  96.4%  (e.g. embed_tokens.weight)

--dist-only emits a machine-readable key=value distribution that can be stored
alongside curated perf-baseline evidence when a model file changes.
"""

import struct, sys, os, json
from collections import defaultdict

QT_NAMES = {
    0:  "Q4F16G64",
    1:  "F16",
    2:  "F32",
    3:  "Q8F16",
    6:  "HFQ4G256",
    7:  "HFQ4G128",
    8:  "HFQ6G256",
    11: "HFQ3G256",
    12: "HFQ3G128",
    13: "MQ4G256",
    14: "MQ8G256",
    15: "MQ6G256",
    16: "BF16",
    17: "MQ3G256",
    18: "MQ2G256",
    19: "MQ2G256Lloyd",
    20: "MQ3G256Lloyd",
    21: "HFP4G32",
    24: "MFP4G32",
    30: "MQ4G256Lloyd",
}

def _find_json_end(buf: bytes) -> int:
    depth = 0; in_str = False; esc = False
    for i, b in enumerate(buf):
        if esc: esc = False; continue
        if b == 0x5c and in_str: esc = True; continue
        if b == 0x22: in_str = not in_str; continue
        if not in_str:
            if b == 0x7b: depth += 1
            elif b == 0x7d:
                depth -= 1
                if depth == 0: return i + 1
    raise ValueError("metadata JSON has no closing brace")

def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PiB"

def parse_hfq(path: str) -> list[dict]:
    """Return list of {name, qt, shape, data_size} for every tensor in the HFQ."""
    with open(path, "rb") as f:
        head = f.read(32)
    assert head[:4] == b"HFQM", f"not an HFQ file: {path}"
    metadata_offset = struct.unpack_from("<Q", head, 16)[0]
    data_offset     = struct.unpack_from("<Q", head, 24)[0]
    n_tensors       = struct.unpack_from("<I", head, 12)[0]

    with open(path, "rb") as f:
        f.seek(metadata_offset)
        meta = f.read(data_offset - metadata_offset)

    pos = _find_json_end(meta)
    n = struct.unpack_from("<I", meta, pos)[0]; pos += 4
    assert n == n_tensors, f"index count mismatch: {n} vs header {n_tensors}"

    tensors = []
    for _ in range(n):
        name_len = struct.unpack_from("<H", meta, pos)[0]; pos += 2
        name     = meta[pos:pos+name_len].decode();        pos += name_len
        qt       = meta[pos];                              pos += 1
        n_dims   = meta[pos];                              pos += 1
        shape    = list(struct.unpack_from(f"<{n_dims}I", meta, pos)); pos += 4*n_dims
        _gs      = struct.unpack_from("<I", meta, pos)[0]; pos += 4
        size     = struct.unpack_from("<Q", meta, pos)[0]; pos += 8
        tensors.append({"name": name, "qt": qt, "shape": shape, "data_size": size})
    return tensors

def distribution(tensors: list[dict]) -> dict:
    """Return {qt: {count, bytes, example_name}} sorted by descending bytes."""
    d = defaultdict(lambda: {"count": 0, "bytes": 0, "example": ""})
    for t in tensors:
        r = d[t["qt"]]
        r["count"] += 1
        r["bytes"] += t["data_size"]
        if not r["example"]:
            r["example"] = t["name"]
    return dict(sorted(d.items(), key=lambda kv: -kv[1]["bytes"]))

def print_report(path: str, tensors: list[dict]) -> None:
    total_bytes = sum(t["data_size"] for t in tensors)
    dist = distribution(tensors)
    short = os.path.basename(path)
    print(f"\n{short}  ({_human_bytes(total_bytes)}, {len(tensors)} tensors)")
    print(f"  {'qt':>3}  {'name':<14}  {'tensors':>7}  {'bytes':>12}  {'%':>5}  example")
    print("  " + "─" * 72)
    for qt, r in dist.items():
        name = QT_NAMES.get(qt, f"qt={qt}")
        pct  = 100.0 * r["bytes"] / total_bytes if total_bytes else 0
        ex   = r["example"]
        # Trim long tensor names: keep the last two dot-segments
        parts = ex.split(".")
        ex_short = ".".join(parts[-2:]) if len(parts) > 2 else ex
        print(f"  {qt:>3}  {name:<14}  {r['count']:>7}  {_human_bytes(r['bytes']):>12}  {pct:>4.1f}%  {ex_short}")
    print()

def print_dist_only(path: str, tensors: list[dict]) -> None:
    """Machine-readable key=value suitable for tests/weight-distributions/."""
    stem = os.path.splitext(os.path.basename(path))[0]
    dist = distribution(tensors)
    total = sum(t["data_size"] for t in tensors)
    print(f"# {stem}  total={_human_bytes(total)}  tensors={len(tensors)}")
    for qt, r in dist.items():
        name = QT_NAMES.get(qt, f"qt{qt}")
        print(f"{stem}_{name.lower()}_tensors={r['count']}")
        print(f"{stem}_{name.lower()}_bytes={r['bytes']}")
    print()

def main() -> None:
    args = sys.argv[1:]
    dist_only = False
    if args and args[0] == "--dist-only":
        dist_only = True
        args = args[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    for path in args:
        tensors = parse_hfq(path)
        if dist_only:
            print_dist_only(path, tensors)
        else:
            print_report(path, tensors)

if __name__ == "__main__":
    main()
