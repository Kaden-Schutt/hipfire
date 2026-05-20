#!/usr/bin/env python3
"""Step 1.5 — tokenize the slice with both stacks, byte-compare token streams.

If both produce identical streams: the GGUF anchor track is viable; eval can
proceed with both hipfire-side and llama.cpp-side candidates measured against
the BF16 reference using the same token IDs.

If they diverge: bridge work or drop the GGUF anchor entirely (per plan
rev-3.2 §"Tokenizer alignment + bridge investigation").

Usage:
  python3 tokenizer_parity.py \\
    --hipfire-hfq    <path-to-qwen3.5-9b.mq4 or similar .hfq>  \\
    --llamacpp-gguf  <path-to-qwen3.5-9b-bf16.gguf>            \\
    --slice          <path-to-slice.txt>                       \\
    [--llama-tokenize-bin <path>=llama-tokenize]               \\
    [--cargo-target-dir  <path>=./target]

Exit codes:
  0  byte-identical → continue with GGUF anchor track
  1  mismatch        → bridge work or drop anchor track
  2  setup error     (slice missing, binary missing, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path


def hipfire_tokenize(model: Path, slice_path: Path, target_dir: Path) -> list[int]:
    """Tokenize via hipfire's tokenizer (cargo example tokenize_slice)."""
    # Find the built binary; build if missing.
    bin_path = target_dir / "release" / "examples" / "tokenize_slice"
    if not bin_path.exists():
        print(f"  building tokenize_slice (not at {bin_path})...", file=sys.stderr)
        subprocess.run(
            ["cargo", "build", "--release", "-p", "hipfire-runtime", "--example", "tokenize_slice"],
            check=True,
        )

    out_bin = Path("/tmp") / f"tokenizer_parity_hipfire_{os.getpid()}.bin"
    try:
        subprocess.run(
            [str(bin_path), "--model", str(model), "--slice", str(slice_path), "--output", str(out_bin)],
            check=True,
            capture_output=True,
            text=True,
        )
        data = out_bin.read_bytes()
        n = len(data) // 4
        return list(struct.unpack(f"<{n}I", data))
    finally:
        out_bin.unlink(missing_ok=True)


def llamacpp_tokenize(gguf: Path, slice_path: Path, llama_tokenize_bin: str) -> list[int]:
    """Tokenize via llama-tokenize (built from the pinned llama.cpp commit)."""
    proc = subprocess.run(
        [
            llama_tokenize_bin,
            "-m", str(gguf),
            "-f", str(slice_path),
            "--ids",
            "--log-disable",
            "--no-bos",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Output looks like: [1, 2, 3, ...] (one Python list literal)
    out = proc.stdout.strip()
    if not out.startswith("["):
        raise ValueError(f"unexpected llama-tokenize output: {out[:200]!r}")
    return json.loads(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hipfire-hfq",   required=True, type=Path, help="path to a hipfire .hfq model")
    ap.add_argument("--llamacpp-gguf", required=True, type=Path, help="path to a llama.cpp .gguf (e.g. BF16 ref)")
    ap.add_argument("--slice",         required=True, type=Path, help="path to slice text file")
    ap.add_argument("--llama-tokenize-bin", default="llama-tokenize",
                    help="path to llama-tokenize binary (default: search PATH)")
    ap.add_argument("--cargo-target-dir", type=Path, default=Path(__file__).resolve().parents[3] / "target",
                    help="cargo target dir (default: workspace root /target)")
    ap.add_argument("--report-only", action="store_true",
                    help="don't fail on mismatch, just report")
    args = ap.parse_args()

    # Sanity checks
    for label, path in (
        ("--hipfire-hfq", args.hipfire_hfq),
        ("--llamacpp-gguf", args.llamacpp_gguf),
        ("--slice", args.slice),
    ):
        if not path.exists():
            print(f"error: {label} {path} does not exist", file=sys.stderr)
            return 2

    print(f"tokenizer_parity: comparing tokenizations of {args.slice}", file=sys.stderr)
    print(f"  hipfire side:    {args.hipfire_hfq}", file=sys.stderr)
    print(f"  llama.cpp side:  {args.llamacpp_gguf}", file=sys.stderr)
    print(f"  llama-tokenize:  {args.llama_tokenize_bin}", file=sys.stderr)

    # 1. hipfire side
    print("[1/3] hipfire tokenization...", file=sys.stderr)
    hf_ids = hipfire_tokenize(args.hipfire_hfq, args.slice, args.cargo_target_dir)
    print(f"      → {len(hf_ids)} tokens", file=sys.stderr)

    # 2. llama.cpp side
    print("[2/3] llama.cpp tokenization...", file=sys.stderr)
    gguf_ids = llamacpp_tokenize(args.llamacpp_gguf, args.slice, args.llama_tokenize_bin)
    print(f"      → {len(gguf_ids)} tokens", file=sys.stderr)

    # 3. Compare
    print("[3/3] comparing...", file=sys.stderr)
    if len(hf_ids) != len(gguf_ids):
        print(
            f"  LENGTH MISMATCH: hipfire={len(hf_ids)} llama.cpp={len(gguf_ids)} "
            f"(Δ = {len(hf_ids) - len(gguf_ids)})",
            file=sys.stderr,
        )
        return 0 if args.report_only else 1

    diffs = []
    for i, (a, b) in enumerate(zip(hf_ids, gguf_ids)):
        if a != b:
            diffs.append((i, a, b))
            if len(diffs) <= 10:
                # Print first 10 mismatches with context
                ctx_start = max(0, i - 2)
                ctx_end = min(len(hf_ids), i + 3)
                hf_ctx = hf_ids[ctx_start:ctx_end]
                gguf_ctx = gguf_ids[ctx_start:ctx_end]
                print(
                    f"  [pos {i:>6}] hipfire={a}  llama.cpp={b}    (ctx hf={hf_ctx} gguf={gguf_ctx})",
                    file=sys.stderr,
                )

    if not diffs:
        print(f"  OK — byte-identical tokenization across {len(hf_ids):,} tokens", file=sys.stderr)
        print(f"  GGUF anchor track is viable.", file=sys.stderr)
        return 0

    pct = 100.0 * len(diffs) / len(hf_ids)
    print(
        f"  TOKEN-STREAM MISMATCH: {len(diffs):,}/{len(hf_ids):,} positions differ ({pct:.3f}%)",
        file=sys.stderr,
    )
    if len(diffs) > 10:
        print(f"  ({len(diffs) - 10} more mismatches not shown)", file=sys.stderr)
    print(
        "  GGUF anchor track requires the bridge OR drop-anchor fallback "
        "(see plan rev-3.2 §'Tokenizer alignment').",
        file=sys.stderr,
    )
    return 0 if args.report_only else 1


if __name__ == "__main__":
    sys.exit(main())
