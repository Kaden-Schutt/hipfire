#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read

"""Grade dots-ocr ocr_e2e output against the vLLM reference.

Strategy A, step 3 of the dots-ocr precision investigation.

Inputs:
  --our  /tmp/our_ocr_output.txt   raw stdout from `cargo run --example ocr_e2e`
  --ref  benchmarks/references/dots_ocr_smoke_001_vllm.json

Grading:
  1. Parse both as JSON. If our output fails to parse → FAIL.
  2. Greedy-match each ref region to the closest-IoU region in ours.
  3. Per pair: bbox IoU, bbox L1 px deviation, text edit distance (ratio).
  4. Aggregate:
     - region_f1: F1 at IoU > 0.5 threshold (how many regions both found)
     - mean_text_distance: mean Levenshtein ratio over matched regions
     - exact_text_count: regions where text matches byte-for-byte
  5. Verdict:
     - PASS:      region_f1 > 0.9 AND mean_text_distance < 0.10
     - SOFT-PASS: region_f1 > 0.7
     - FAIL:      anything else

Usage:
    python3 scripts/grade_dots_ocr_e2e.py \\
        --our /tmp/our_ocr_output.txt \\
        --ref benchmarks/references/dots_ocr_smoke_001_vllm.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_ocr_json(text: str) -> tuple[list[dict] | None, str]:
    """Returns (regions, status). status='ok' on success, else error str."""
    text = text.strip()
    if not text:
        return None, "empty output"
    # Model may emit chatter before/after the JSON; find the first [ and matching ].
    start = text.find("[")
    if start == -1:
        return None, "no opening [ found"
    # Find balanced bracket.
    depth = 0
    end = -1
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None, "unbalanced brackets (truncated output?)"
    blob = text[start:end]
    try:
        regions = json.loads(blob)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    if not isinstance(regions, list):
        return None, f"top-level is {type(regions).__name__}, expected list"
    return regions, "ok"


def bbox_iou(a: list[int], b: list[int]) -> float:
    """IoU over [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def bbox_l1(a: list[int], b: list[int]) -> int:
    return sum(abs(x - y) for x, y in zip(a, b))


def levenshtein(a: str, b: str) -> int:
    """Standard edit distance. O(len(a) * len(b)) — fine for OCR-region scale."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def text_distance_ratio(a: str, b: str) -> float:
    """Normalised Levenshtein in [0, 1]. 0 = identical, 1 = completely different."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    return levenshtein(a, b) / n


def greedy_match(ref: list[dict], ours: list[dict]) -> list[tuple[int, int, float]]:
    """Greedy bipartite match by max IoU. Returns [(ref_idx, our_idx, iou), ...]."""
    pairs = []
    used_ours = set()
    for ri, r in enumerate(ref):
        best_idx = -1
        best_iou = 0.0
        for oi, o in enumerate(ours):
            if oi in used_ours:
                continue
            iou = bbox_iou(r["bbox"], o["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = oi
        if best_idx != -1:
            pairs.append((ri, best_idx, best_iou))
            used_ours.add(best_idx)
        else:
            pairs.append((ri, -1, 0.0))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", required=True, type=Path,
                    help="path to ocr_e2e stdout (raw decoded text)")
    ap.add_argument("--ref", required=True, type=Path,
                    help="path to vLLM reference JSON (eg dots_ocr_smoke_001_vllm.json)")
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU threshold for region-detection F1 (default: 0.5)")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-region match details")
    args = ap.parse_args()

    if not args.our.exists():
        print(f"FAIL: our output file missing: {args.our}", file=sys.stderr)
        return 2
    if not args.ref.exists():
        print(f"FAIL: reference file missing: {args.ref}", file=sys.stderr)
        return 2

    # Reference JSON: parsed_json field holds the ground-truth region list.
    ref_doc = json.loads(args.ref.read_text())
    if "parsed_json" not in ref_doc or ref_doc.get("parse_status") != "ok":
        print(f"FAIL: reference has parse_status={ref_doc.get('parse_status')!r}", file=sys.stderr)
        return 2
    ref_regions = ref_doc["parsed_json"]
    print(f"reference: {len(ref_regions)} regions ({args.ref.name})")

    # Our output.
    our_text = args.our.read_text()
    print(f"our output: {len(our_text)} chars ({args.our.name})")
    our_regions, parse_status = parse_ocr_json(our_text)
    if our_regions is None:
        print(f"\nVERDICT: FAIL — our output not parseable as JSON: {parse_status}")
        print(f"first 200 chars: {our_text[:200]!r}")
        return 1
    print(f"our output: {len(our_regions)} regions parsed ({parse_status})")

    # Match.
    pairs = greedy_match(ref_regions, our_regions)
    matched_at_threshold = sum(1 for (_, oi, iou) in pairs
                                if oi != -1 and iou >= args.iou_threshold)

    # Reverse-direction: penalise our extras (false positives).
    ref_matched_ours = {oi for (_, oi, iou) in pairs if oi != -1 and iou >= args.iou_threshold}
    n_our_extras = sum(1 for oi in range(len(our_regions)) if oi not in ref_matched_ours)

    precision = matched_at_threshold / max(1, len(our_regions))
    recall    = matched_at_threshold / max(1, len(ref_regions))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Per-region text scores (only on matched pairs).
    text_dists: list[float] = []
    exact_matches = 0
    for (ri, oi, iou) in pairs:
        if oi == -1 or iou < args.iou_threshold:
            continue
        r_text = ref_regions[ri].get("text", "")
        o_text = our_regions[oi].get("text", "")
        d = text_distance_ratio(r_text, o_text)
        text_dists.append(d)
        if r_text == o_text:
            exact_matches += 1

    mean_text_dist = sum(text_dists) / len(text_dists) if text_dists else 1.0

    # Verdict.
    if f1 > 0.9 and mean_text_dist < 0.10:
        verdict = "PASS"
    elif f1 > 0.7:
        verdict = "SOFT-PASS"
    else:
        verdict = "FAIL"

    print()
    print(f"=== grading ===")
    print(f"regions: ref={len(ref_regions)}, ours={len(our_regions)} (extras={n_our_extras})")
    print(f"matched at IoU>={args.iou_threshold}: {matched_at_threshold}/{len(ref_regions)}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
    print(f"text distance (matched regions): mean={mean_text_dist:.3f}, exact={exact_matches}/{len(text_dists)}")
    print(f"\nVERDICT: {verdict}")

    if args.verbose:
        print("\nper-region matches:")
        for (ri, oi, iou) in pairs:
            r = ref_regions[ri]
            if oi == -1:
                print(f"  ref[{ri}] {r['category']:14s} bbox={r['bbox']}  UNMATCHED")
                continue
            o = our_regions[oi]
            r_text = r.get("text", "")
            o_text = o.get("text", "")
            d = text_distance_ratio(r_text, o_text)
            l1 = bbox_l1(r["bbox"], o["bbox"])
            same_cat = "✓" if r.get("category") == o.get("category") else "✗"
            print(f"  ref[{ri}] {r['category']:14s} -> ours[{oi}] iou={iou:.3f} bbox_l1={l1:4d}px cat={same_cat} text_dist={d:.3f}")
        if n_our_extras > 0:
            print("\nour false-positive regions (no matching ref):")
            for oi, o in enumerate(our_regions):
                if oi not in ref_matched_ours:
                    print(f"  ours[{oi}] {o.get('category','?'):14s} bbox={o['bbox']}")

    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
