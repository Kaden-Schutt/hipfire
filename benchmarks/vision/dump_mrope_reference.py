#!/usr/bin/env python3
"""Emit HF get_rope_index reference positions as a JSON fixture for the
Rust mrope parity test. Run once; the output is committed so the Rust test
needs no torch.

Usage: python3 benchmarks/vision/dump_mrope_reference.py > \
    crates/hipfire-arch-qwen35-vl/tests/fixtures/mrope_reference.json
"""
import json, sys

MERGE = 2

def hf_positions(segments):
    """segments: list of ("text", n) or ("image", grid_h, grid_w).
    Mirrors modeling_qwen3_5.get_rope_index / get_vision_position_ids."""
    pos, cursor = [], 0
    for seg in segments:
        if seg[0] == "text":
            n = seg[1]
            for i in range(n):
                pos.append([cursor + i, cursor + i, cursor + i])
            cursor += n
        else:
            _, gh, gw = seg
            lh, lw = gh // MERGE, gw // MERGE
            for hh in range(lh):
                for ww in range(lw):
                    pos.append([cursor, cursor + hh, cursor + ww])
            cursor += max(gh, gw) // MERGE
    delta = max(max(p) for p in pos) + 1 - len(pos)
    return pos, delta

CASES = {
    # The real smoke-page grid: 945 visual tokens, advance 35.
    "smoke_70x54":   [("text", 12), ("image", 70, 54), ("text", 5)],
    # Wide grid: catches a hardcoded "use height" advance.
    "wide_28x70":    [("text", 3),  ("image", 28, 70), ("text", 4)],
    # Pure text: must equal plain sequential on all three axes.
    "text_only":     [("text", 40)],
    # Trailing text must resume from the cursor, not its own index.
    "text_img_text": [("text", 7),  ("image", 8, 12),  ("text", 9)],
}

out = {}
for name, segs in CASES.items():
    p, d = hf_positions(segs)
    out[name] = {
        "segments": [list(s) for s in segs],
        "merge": MERGE,
        "positions": p,
        "rope_delta": d,
        "n_tokens": len(p),
    }
json.dump(out, sys.stdout, indent=1)
