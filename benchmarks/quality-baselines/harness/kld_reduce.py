#!/usr/bin/env python3
"""kld_reduce.py — canonical reducer for hipfire quant-quality eval.

Reads per-sequence-KLD files (one per (variant, arch)), aggregates,
computes 95% bootstrap CI on the slice-mean, emits the result table
as markdown + a JSON sidecar for plotting.

Inputs:
  --result-dir <dir>     directory of per-sequence-KLD files (HFKSEQ format)
                         filename convention:
                           <variant>__<arch>__<scoring_mode>.kldseq   (preferred)
                           <variant>__<arch>.kldseq                   (legacy: auto-tagged
                                                                       scoring_mode="gguf"
                                                                       if variant contains
                                                                       "gguf-", else
                                                                       "per-token")
                         e.g., qwen3.5-9b.mq3__gfx1100__prefill.kldseq
                               qwen3.5-9b.gguf-q4_k_m__gfx1151.kldseq

Output:
  result-table.md        markdown table with mean ± CI, p99, etc.
  result-data.json       same data as JSON for downstream plot scripts

Run:
  python3 kld_reduce.py --result-dir results/2026-05-XX/per-seq/

Spec: docs/plans/issue-113-quant-quality-eval.md §"Result table format".
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Local import — kldref_format.py lives next to this script.
sys.path.insert(0, str(Path(__file__).parent))
from kldref_format import read_per_seq_kld  # noqa: E402


@dataclass
class Row:
    variant: str
    arch: str
    scoring_mode: str       # "prefill" (canonical hipfire), "per-token" (legacy hipfire),
                            # "gguf" (eval_gguf anchors). See docs/plans/eval_hipfire_speedup.md
                            # §"V1 result" for why per-token and prefill are separate
                            # measurement classes for hipfire variants.
    n_chunks: int
    mean_kld: float
    mean_kld_ci_lo: float    # 95% bootstrap lower
    mean_kld_ci_hi: float
    p99_kld: float
    ppl: float | None = None  # exp(mean NLL) over all scored tokens; None for v1 inputs
    notes: str = ""


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) where CI is 2.5th/97.5th percentile of
    bootstrapped resample-means."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def parse_filename(name: str) -> tuple[str, str, str]:
    """Returns (variant, arch, scoring_mode).

    Preferred form (3 fields): <variant>__<arch>__<scoring_mode>.kldseq
      e.g. qwen3.5-9b.mq3__gfx1100__prefill.kldseq

    Legacy form (2 fields): <variant>__<arch>.kldseq
      Auto-tagged: scoring_mode="gguf" if "gguf" in variant else "per-token".
      (The legacy hipfire kldseqs in 2026-05-08 were all per-token.)
    """
    stem = Path(name).stem  # strip .kldseq
    parts = stem.split("__")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        variant, arch = parts
        scoring_mode = "gguf" if "gguf" in variant else "per-token"
        return variant, arch, scoring_mode
    raise ValueError(
        f"filename {name!r} doesn't match <variant>__<arch>[__<scoring_mode>].kldseq"
    )


def reduce_one(path: Path) -> Row:
    means, p99s, nlls = read_per_seq_kld(path)
    variant, arch, scoring_mode = parse_filename(path.name)
    means_arr = np.asarray(means, dtype=np.float64)
    p99s_arr = np.asarray(p99s, dtype=np.float64)
    nlls_arr = np.asarray(nlls, dtype=np.float64)
    mean, ci_lo, ci_hi = bootstrap_mean_ci(means_arr)
    p99 = float(np.percentile(p99s_arr, 99))
    finite_nll = nlls_arr[np.isfinite(nlls_arr)]
    ppl: float | None = float(np.exp(finite_nll.mean())) if finite_nll.size else None
    return Row(
        variant=variant, arch=arch, scoring_mode=scoring_mode, n_chunks=len(means),
        mean_kld=mean, mean_kld_ci_lo=ci_lo, mean_kld_ci_hi=ci_hi,
        p99_kld=p99, ppl=ppl,
    )


def render_markdown_table(rows: list[Row]) -> str:
    """Render rows grouped by scoring_mode so historical per-token rows are
    visually separated from canonical prefill rows. Mixed-mode tables emit a
    prominent warning at the top — see PRD §5.3 ("modes are separate
    measurement classes; do not cross-compare").
    """
    out = []

    # Detect mode-collision: same (variant, arch) with multiple scoring modes.
    by_key: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        by_key.setdefault((r.variant, r.arch), set()).add(r.scoring_mode)
    mixed = {k: v for k, v in by_key.items() if len(v) > 1}

    if mixed:
        out.append(
            "> ⚠️  **Mixed-mode rows detected.** The following (variant, arch) "
            "pairs have rows in more than one scoring mode. KLDs from "
            "different modes are NOT directly comparable (kernel-path "
            "numerical effect, ~7% mean shift on hipfire MQ; see PRD §5.3)."
        )
        out.append(">")
        for (v, a), modes in sorted(mixed.items()):
            out.append(f">  - `{v}` @ `{a}` → {sorted(modes)}")
        out.append("")

    out.append("| Variant | Arch | Mode | n_chunks | Mean KLD ± 95% CI | p99 KLD | PPL | Notes |")
    out.append("|---|---|---|---:|---|---:|---:|---|")

    # Group by mode: prefill (canonical) first, then per-token (historical),
    # then gguf (anchors). Within each group, sort by (variant, arch).
    mode_order = {"prefill": 0, "per-token": 1, "gguf": 2}
    for r in sorted(rows, key=lambda r: (mode_order.get(r.scoring_mode, 99), r.variant, r.arch)):
        ci = f"{r.mean_kld:.4f} (CI {r.mean_kld_ci_lo:.4f}–{r.mean_kld_ci_hi:.4f})"
        ppl = f"{r.ppl:.3f}" if r.ppl is not None else "—"
        out.append(f"| {r.variant} | {r.arch} | {r.scoring_mode} | {r.n_chunks} | {ci} | {r.p99_kld:.3f} | {ppl} | {r.notes} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result-dir", required=True, help="directory of *.kldseq files")
    ap.add_argument("--out-md", default="result-table.md")
    ap.add_argument("--out-json", default="result-data.json")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    files = sorted(result_dir.glob("*.kldseq"))
    if not files:
        print(f"no *.kldseq files in {result_dir}", file=sys.stderr)
        return 1

    rows = [reduce_one(f) for f in files]
    md = render_markdown_table(rows)
    Path(args.out_md).write_text(md + "\n")
    Path(args.out_json).write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(md)
    print()
    print(f"wrote {args.out_md}, {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
