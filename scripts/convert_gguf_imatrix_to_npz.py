#!/usr/bin/env python3
"""Convert a llama.cpp GGUF imatrix file to a per-channel sum² .npz keyed
by hipfire hfq tensor names.

Output schema (matches what `mq4_masked_calib.py iterate
--awq-raw-sumsq-npz` consumes):

    <safe_key(hfq_name)> -> float32[K]  raw per-input-channel sum(x²)

Where `safe_key` is `name.replace("/", "__slash__").replace(".", "__dot__")`
— the same encoding `mq4_masked_calib.py` uses for `stats-merged.npz`
entries.

The unsloth Qwen3.5 GGUF imatrix exposes per-tensor `*.in_sum2` and
`*.counts` records. We:

1. Iterate `*.in_sum2` tensors,
2. For each, map the GGUF logical slot (e.g. `blk.7.attn_q.weight`) to
   candidate hipfire hfq names via `astrea.gguf_to_hfq_candidates`
   (`model.layers.N.self_attn.q_proj.weight`,
   `model.language_model.layers.N.self_attn.q_proj.weight`), and
3. Write the raw sum² vector under each candidate's `safe_key`.

The AWQ scale derivation `s[j] = (sum(x_j²))^(α/2)` is scale-invariant
under the `log_s -= mean(log_s)` normalization downstream, so dividing
by counts is unnecessary — we keep the raw values.

Usage:

    python3 scripts/convert_gguf_imatrix_to_npz.py \\
        --in /home/kaden/.hipfire/imatrix/unsloth/Qwen3.5-9B-GGUF/imatrix_unsloth.gguf_file \\
        --out /home/kaden/.hipfire/imatrix/unsloth-9b-raw-sumsq.npz

Optional `--mask PATH` filters output to tensors that appear in the
given iterate mask (saves space when only F1 names are needed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from astrea import gguf_to_hfq_candidates  # noqa: E402


def safe_key(name: str) -> str:
    return name.replace("/", "__slash__").replace(".", "__dot__")


def load_in_sum2_records(gguf_path: Path) -> dict[str, np.ndarray]:
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    out: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if not tensor.name.endswith(".in_sum2"):
            continue
        logical = tensor.name[: -len(".in_sum2")]
        data = np.asarray(tensor.data, dtype=np.float32).reshape(-1).copy()
        out[logical] = data
    return out


def load_mask_hfq_names(mask_path: Path) -> set[str]:
    payload = json.loads(mask_path.read_text())
    names = set()
    for row in payload.get("tensors", []):
        n = row.get("hfq_name")
        if n:
            names.add(n)
    return names


def build_payload(
    records: dict[str, np.ndarray],
    *,
    restrict_to: set[str] | None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[str]]:
    payload: dict[str, np.ndarray] = {}
    mapped: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for logical, vec in records.items():
        candidates = gguf_to_hfq_candidates(logical)
        kept: list[str] = []
        for hfq in candidates:
            if restrict_to is not None and hfq not in restrict_to:
                continue
            payload[safe_key(hfq)] = vec
            kept.append(hfq)
        mapped[logical] = kept
        if not kept:
            unmapped.append(logical)
    return payload, mapped, unmapped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", required=True, type=Path,
                    help="input GGUF imatrix file")
    ap.add_argument("--out", dest="dst", required=True, type=Path,
                    help="output .npz path")
    ap.add_argument("--mask", type=Path, default=None,
                    help="optional iterate mask JSON; emit only tensors whose hfq_name appears there")
    ap.add_argument("--stats-json", type=Path, default=None,
                    help="optional output stats.json path (defaults to <out>.stats.json)")
    args = ap.parse_args()

    if not args.src.exists():
        ap.error(f"--in {args.src} does not exist")

    records = load_in_sum2_records(args.src)
    if not records:
        ap.error(f"no *.in_sum2 records found in {args.src}")

    restrict = load_mask_hfq_names(args.mask) if args.mask else None
    payload, mapped, unmapped = build_payload(records, restrict_to=restrict)

    if not payload:
        ap.error("no tensors survived gguf→hfq mapping (and optional mask filter)")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dst, **payload)

    stats_json_path = args.stats_json or args.dst.with_suffix(args.dst.suffix + ".stats.json")
    stats_json_path.write_text(
        json.dumps(
            {
                "schema": "hipfire.imatrix.raw_sumsq.v0",
                "source_gguf": str(args.src),
                "mask_path": str(args.mask) if args.mask else None,
                "tensor_count": len(payload),
                "logical_count": len(records),
                "mapped": mapped,
                "unmapped_logical": unmapped,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"[convert-imatrix] wrote {len(payload)} entries to {args.dst}", file=sys.stderr)
    print(f"[convert-imatrix] stats: {stats_json_path}", file=sys.stderr)
    if unmapped:
        print(f"[convert-imatrix] {len(unmapped)} GGUF tensors did not map to any hfq name; "
              f"first 5: {unmapped[:5]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
