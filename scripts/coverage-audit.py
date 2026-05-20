#!/usr/bin/env python3
"""
coverage-audit.py -- cross-check an internal HIPFIRE_PROFILE dump against
a rocprofv3 _kernel_stats.csv and report blindspot kernels.

CLI:
    coverage-audit.py --internal <atlas.jsonl> --rocprof <stats.csv> \\
        [--threshold-pct 5] [--out <report.md>]

Exit codes:
    0  coverage is above the threshold (or rocprof CSV is empty)
    1  blindspot_total exceeds threshold_pct of rocprof_total
    2  argument / file error

rocprofv3 stats CSV format (columns):
    Name,Calls,TotalDurationNs,AverageNs,Percentage,MinNs,MaxNs,StdDev

Atlas JSONL format: each line is a JSON object with at minimum:
    {"metrics": {"internal_kernel_total_ms": <f>, ...}, ...}
When emitted by bench_qwen35_mq4 with HIPFIRE_ROCPROF_CSV set, the atlas
row already contains rocprof_coverage_pct etc. This script is useful for
running the audit independently (e.g. when the atlas row was emitted
without HIPFIRE_ROCPROF_CSV, or when comparing a fresh rocprof run).
"""

import argparse
import datetime
import json
import sys
from pathlib import Path


# --- rocprof CSV parser ----------------------------------------------------

def parse_rocprof_stats_csv(path):
    """
    Parse a rocprofv3 _kernel_stats.csv file.

    Returns a list of dicts with keys:
        name (str), calls (int), duration_us (float), percent (float)

    The Name column may contain commas (C++ template args). The CSV has exactly
    7 numeric tail columns, so Name = everything before the last 7 columns.
    """
    kernels = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"ERROR: cannot read rocprof CSV {path}: {e}") from e

    lines = text.splitlines()
    if not lines:
        raise SystemExit(f"ERROR: rocprof CSV is empty: {path}")

    header = lines[0].strip().lower()
    if "name" not in header or "calls" not in header:
        raise SystemExit(
            f"ERROR: rocprof CSV header does not look like kernel stats: {lines[0]!r}"
        )

    for lineno, raw in enumerate(lines[1:], start=2):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            print(
                f"WARN: rocprof CSV line {lineno}: expected >=8 columns, got {len(parts)}; skipping",
                file=sys.stderr,
            )
            continue
        n = len(parts)
        name = ",".join(parts[: n - 7]).strip()
        if not name:
            continue  # skip blank-name domain aggregate rows
        try:
            calls = int(parts[n - 7].strip())
            total_ns = float(parts[n - 6].strip())
            _avg_ns = float(parts[n - 5].strip())  # unused
            percent = float(parts[n - 4].strip())
        except ValueError as e:
            print(
                f"WARN: rocprof CSV line {lineno}: parse error ({e}); skipping",
                file=sys.stderr,
            )
            continue
        kernels.append(
            {
                "name": name,
                "calls": calls,
                "duration_us": total_ns / 1_000.0,
                "percent": percent,
            }
        )
    return kernels


# --- Internal profile extractor --------------------------------------------

def extract_internal_kernels(atlas_path):
    """
    Extract per-kernel names and internal total from an Atlas JSONL file.

    Returns (internal_total_us, [kernel_alias, ...]).

    The atlas row records `metrics.internal_kernel_total_ms` when
    HIPFIRE_ROCPROF_CSV was set during the bench run. Falls back to
    `metrics.prefill_kernel_ms` for older rows. When neither is present,
    returns (0.0, []) with a warning.

    The per-kernel breakdown is taken from `artifacts.profile_kernels` if
    present (emitted by kernel_atlas.py collect-ar --profile-prefill), or
    from `artifacts.rocprof_blindspots` for negative confirmation.
    """
    try:
        text = atlas_path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"ERROR: cannot read atlas JSONL {atlas_path}: {e}") from e

    rows = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARN: atlas line {lineno}: JSON parse error ({e}); skipping", file=sys.stderr)

    if not rows:
        raise SystemExit(f"ERROR: atlas JSONL contains no parseable rows: {atlas_path}")

    # Use the last row (most recent measurement).
    row = rows[-1]
    metrics = row.get("metrics", {})

    internal_total_ms = metrics.get(
        "internal_kernel_total_ms",
        metrics.get("prefill_kernel_ms", 0.0),
    )
    internal_total_us = float(internal_total_ms) * 1_000.0

    # Extract kernel alias names from profile_kernels artifact if present.
    aliases = []
    artifacts = row.get("artifacts", {})
    profile_kernels = artifacts.get("profile_kernels", [])
    for entry in profile_kernels:
        if isinstance(entry, dict) and entry.get("name"):
            aliases.append(entry["name"])

    if not aliases and internal_total_us > 0:
        print(
            "INFO: no profile_kernels artifact in atlas row -- "
            "coverage match will use internal_kernel_total_ms only",
            file=sys.stderr,
        )

    return internal_total_us, aliases


# --- Coverage computation --------------------------------------------------

def compute_coverage(rocprof_kernels, internal_aliases):
    """
    Compute coverage: classify each rocprof kernel as covered or blindspot.

    A rocprof entry is "covered" if any internal alias appears as a
    case-insensitive substring of the rocprof Name.
    """
    aliases_lower = [a.lower() for a in internal_aliases]
    rocprof_total_us = sum(k["duration_us"] for k in rocprof_kernels)
    covered = []
    blindspots = []
    for k in rocprof_kernels:
        name_lower = k["name"].lower()
        is_covered = any(
            alias == name_lower or alias in name_lower
            for alias in aliases_lower
        )
        if is_covered:
            covered.append(k)
        else:
            blindspots.append(k)

    blindspot_total_us = sum(k["duration_us"] for k in blindspots)
    coverage_pct = (
        (rocprof_total_us - blindspot_total_us) / rocprof_total_us * 100.0
        if rocprof_total_us > 0
        else 100.0
    )
    return {
        "rocprof_total_us": rocprof_total_us,
        "blindspot_total_us": blindspot_total_us,
        "coverage_pct": coverage_pct,
        "covered": covered,
        "blindspots": blindspots,
    }


# --- Report rendering ------------------------------------------------------

def render_report(
    atlas_path,
    rocprof_path,
    internal_total_us,
    cov,
    threshold_pct,
):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        "# rocprofv3 Coverage Audit",
        "",
        f"Generated: {ts}",
        f"Atlas: `{atlas_path}`",
        f"rocprof CSV: `{rocprof_path}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| rocprof total | {cov['rocprof_total_us'] / 1000:.1f} ms |",
        f"| internal total | {internal_total_us / 1000:.1f} ms |",
        f"| coverage | **{cov['coverage_pct']:.1f}%** |",
        f"| blindspot total | {cov['blindspot_total_us'] / 1000:.1f} ms ({len(cov['blindspots'])} kernels) |",
        f"| threshold | {threshold_pct:.0f}% |",
        f"| status | {'**PASS**' if cov['coverage_pct'] >= (100 - threshold_pct) else '**FAIL -- blindspot > threshold**'} |",
        "",
    ]

    if cov["blindspots"]:
        lines += [
            "## Blindspot Kernels",
            "",
            "Kernels rocprofv3 saw but the internal `HIPFIRE_PROFILE` did not track.",
            "Add `profile::begin_timer` wrappers to these kernel launch sites.",
            "",
            "| Rank | Kernel Name | Calls | Duration (ms) | % of rocprof |",
            "|------|-------------|-------|---------------|--------------|",
        ]
        bs_sorted = sorted(cov["blindspots"], key=lambda k: k["duration_us"], reverse=True)
        for rank, k in enumerate(bs_sorted, 1):
            lines.append(
                f"| {rank} | `{k['name']}` | {k['calls']} | {k['duration_us'] / 1000:.1f} | {k['percent']:.1f}% |"
            )
        lines.append("")

    if cov["covered"]:
        lines += [
            "## Covered Kernels (top 10 by duration)",
            "",
            "| Rank | Kernel Name | Calls | Duration (ms) | % of rocprof |",
            "|------|-------------|-------|---------------|--------------|",
        ]
        covered_sorted = sorted(cov["covered"], key=lambda k: k["duration_us"], reverse=True)
        for rank, k in enumerate(covered_sorted[:10], 1):
            lines.append(
                f"| {rank} | `{k['name']}` | {k['calls']} | {k['duration_us'] / 1000:.1f} | {k['percent']:.1f}% |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- Coverage >= 90%: internal profile is reliable.",
        "- Coverage < 90%: at least one significant kernel is un-tracked. Investigate blindspots.",
        "- Coverage < 75%: almost certainly a new kernel was added without a timer. Urgent.",
        "",
        "Cross-reference: `crates/rdna-compute/src/profile_rocprof.rs` (Rust impl).",
    ]
    return "\n".join(lines) + "\n"


# --- Main ------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--internal",
        required=True,
        metavar="ATLAS_JSONL",
        help="Atlas JSONL file from bench_qwen35_mq4 --emit-atlas",
    )
    parser.add_argument(
        "--rocprof",
        required=True,
        metavar="STATS_CSV",
        help="rocprofv3 _kernel_stats.csv (from rocprof-wrap.sh)",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        metavar="PCT",
        help="fail if blindspot_total > threshold_pct%% of rocprof_total (default: 5)",
    )
    parser.add_argument(
        "--out",
        metavar="REPORT_MD",
        help="write markdown report here (default: /tmp/coverage-audit-<timestamp>.md)",
    )
    args = parser.parse_args(argv)

    atlas_path = Path(args.internal)
    rocprof_path = Path(args.rocprof)
    threshold_pct = args.threshold_pct

    # Parse both sources.
    internal_total_us, internal_aliases = extract_internal_kernels(atlas_path)
    rocprof_kernels = parse_rocprof_stats_csv(rocprof_path)

    if not rocprof_kernels:
        print("INFO: rocprof CSV contains no kernel rows -- nothing to audit.")
        return 0

    cov = compute_coverage(rocprof_kernels, internal_aliases)
    blindspot_pct_of_rocprof = (
        cov["blindspot_total_us"] / cov["rocprof_total_us"] * 100.0
        if cov["rocprof_total_us"] > 0
        else 0.0
    )

    # One-line summary to stdout.
    status = "PASS" if blindspot_pct_of_rocprof <= threshold_pct else "FAIL"
    print(
        f"{status}  coverage={cov['coverage_pct']:.1f}%  "
        f"blindspot={cov['blindspot_total_us'] / 1000:.1f}ms/{cov['rocprof_total_us'] / 1000:.1f}ms  "
        f"({blindspot_pct_of_rocprof:.1f}% of rocprof, threshold={threshold_pct:.0f}%)  "
        f"blindspot_count={len(cov['blindspots'])}"
    )

    # Write markdown report.
    report_md = render_report(
        atlas_path=atlas_path,
        rocprof_path=rocprof_path,
        internal_total_us=internal_total_us,
        cov=cov,
        threshold_pct=threshold_pct,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md, encoding="utf-8")
        print(f"Report written: {out_path}", file=sys.stderr)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = Path(f"/tmp/coverage-audit-{ts}.md")
        default_out.write_text(report_md, encoding="utf-8")
        print(f"Report written: {default_out}", file=sys.stderr)

    return 1 if blindspot_pct_of_rocprof > threshold_pct else 0


if __name__ == "__main__":
    raise SystemExit(main())
