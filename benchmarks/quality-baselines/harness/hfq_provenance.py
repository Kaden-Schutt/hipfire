#!/usr/bin/env python3
"""Print the build provenance of a hipfire .hfq.

Exists because the 2026-07-16 SP-E canary scored a Bonsai ternary model built
before that day's norm-bias fix and reported KLD 6.15 for a model that actually
measures 0.61 — and nothing in the artifact or the result table could reveal
it. `hipfire-quantize` now stamps a `hipfire_provenance` object into the .hfq
metadata; this reads it back so every eval can say what it actually scored.

Files written before that stamp existed fall back to mtime/size, which is
still enough to spot "this predates the fix".

Usage: hfq_provenance.py <model.hfq> [...]
"""
import datetime as _dt
import json
import struct
import sys
from pathlib import Path


def read_metadata(path):
    with open(path, "rb") as f:
        hdr = f.read(32)
        if len(hdr) < 32 or not hdr[0:3] == b"HFQ":
            raise ValueError(f"not an .hfq (magic {hdr[0:4]!r})")
        meta_off, data_off = struct.unpack("<QQ", hdr[16:32])
        f.seek(meta_off)
        raw = f.read(data_off - meta_off)

    # Brace-scan the leading JSON object (string- and escape-aware); the tensor
    # index follows it in the same region.
    depth = 0
    in_str = False
    esc = False
    for i, b in enumerate(raw):
        c = chr(b)
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[: i + 1].decode("utf-8", "replace"))
    raise ValueError("no complete JSON object in metadata region")


def describe(path):
    p = Path(path)
    if not p.exists():
        return f"{p}: MISSING"
    st = p.stat()
    mtime = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    head = f"{p.name}  ({st.st_size / 1e9:.2f} GB, mtime {mtime})"
    try:
        meta = read_metadata(p)
    except Exception as e:  # noqa: BLE001 - report, never abort a sweep
        return f"{head}\n    metadata unreadable: {e}"

    prov = meta.get("hipfire_provenance")
    if not prov:
        return (
            f"{head}\n"
            "    UNSTAMPED — built before provenance stamping; trust the mtime above, "
            "and re-convert if it predates a relevant fix"
        )
    built = prov.get("built_unix")
    when = (
        _dt.datetime.fromtimestamp(built).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(built, (int, float)) and built
        else "unknown"
    )
    lines = [
        head,
        f"    built    {when}  by {prov.get('tool', '?')} "
        f"{prov.get('tool_version', '?')} @ {prov.get('git_commit', 'unknown')}",
        f"    source   {prov.get('source', '?')}",
        f"    format   {prov.get('format', '?')}",
    ]
    if prov.get("source_url"):
        lines.append(f"    upstream {prov['source_url']}")
    if prov.get("license"):
        lines.append(f"    license  {prov['license']}")
    mods = prov.get("modifications") or []
    if mods:
        lines.append(f"    changes  ({len(mods)}) — Apache-2.0 §4(b) notice:")
        lines.extend(f"      - {m}" for m in mods)
    knobs = []
    if prov.get("awq_imatrix_alpha") is not None:
        knobs.append(f"awq-imatrix alpha={prov['awq_imatrix_alpha']:.3g}")
    if prov.get("awq_folded"):
        knobs.append(f"awq-folded={prov['awq_folded']}")
    nz = prov.get("ternary_nonzero_fraction")
    if nz:
        knobs.append(f"nonzero={nz * 100:.1f}%")
    if knobs:
        lines.append("    knobs    " + ", ".join(knobs))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        sys.exit(2)
    for a in sys.argv[1:]:
        print(describe(a))
