# rocprofv3 Coverage Auditing

## The Hidden Lever Problem

In the 2026-05-19 A3B prefill investigation, rocprofv3 showed `gemm_q8_0_batched`
consuming 65% of prefill GPU time (~348ms, 640 calls) while the internal
`HIPFIRE_PROFILE` summed to only ~85ms. The kernel had no `profile::begin_timer`
wrapper. The internal profile was missing two-thirds of the actual work.

## How to Run a Coverage Audit

```bash
# 1. Wrap a bench run under rocprofv3 tracing:
mkdir -p /tmp/cov-run
scripts/rocprof-wrap.sh /tmp/cov-run -- \
    env HIPFIRE_PROFILE=1 \
    ./target/release/examples/bench_qwen35_mq4 model.hfq \
        --prefill 256 --gen 1 \
        --emit-atlas /tmp/cov-run/atlas.jsonl

# 2. Audit the coverage:
scripts/coverage-audit.py \
    --internal /tmp/cov-run/atlas.jsonl \
    --rocprof  /tmp/cov-run/trace_kernel_stats.csv
```

`rocprof-wrap.sh` sets `HIPFIRE_ROCPROF_CSV=/tmp/cov-run/trace_kernel_stats.csv`
in the child environment, which the bench binary reads to attach cross-check
metrics directly to the atlas row (Rust side). `coverage-audit.py` re-runs
the same logic independently as a redundant check (Python side).

## How to Read the Report

**Coverage %**: fraction of rocprofv3 GPU time that the internal profile
accounts for. A kernel is "covered" if its internal alias (e.g.
`"gemm_q8_0_batched"`) appears as a substring of the rocprof mangled symbol.

**Top kernels**: ranked by rocprof duration -- the ground-truth hot path.

**Blindspots**: kernels rocprof saw but the internal profile missed entirely.
Each blindspot needs a `profile::begin_timer` wrapper at its dispatch site.

## When to Escalate

This audit is for **profiling / attribution** claims that rely on internal
timers (`HIPFIRE_PROFILE`, Atlas profile sections). It is **not** a universal
validation gate and does not replace routes in
[`docs/VALIDATION.md`](../VALIDATION.md).

| Coverage | Action for attribution claims |
|----------|--------|
| >= 90% | High observed coverage for the **named** trace — internal timers account for most rocprof GPU time on that run. Not correctness proof, not route proof, and not unconditional reliability of every unmatched kernel. |
| 75-90% | At least one significant kernel is un-tracked on this trace. Investigate blindspots before trusting internal attribution. |
| < 75% | Large blindspot risk on this trace (often a new launch without a timer). Fix attribution before relying on internal profile numbers. |

When a change adds a kernel launch site, quant format, or architecture port
**and** you will cite internal profile timers for Amdahl or hot-path claims,
run a coverage audit on a representative trace. Defer acceptance/route
requirements to [`VALIDATION.md`](../VALIDATION.md).

## Implementation References

- `crates/rdna-compute/src/profile_rocprof.rs` -- Rust: CSV parser, coverage
  computation, `stop_with_rocprof`, inline `#[cfg(test)]` tests
- `crates/hipfire-atlas/src/profile_report.rs` -- Atlas: `AtlasProfileReport`
  type + `AtlasRow::set_profile_report` serializer
- `scripts/rocprof-wrap.sh` -- shell wrapper that invokes rocprofv3 and sets
  `HIPFIRE_ROCPROF_CSV` for the child bench process
- `scripts/coverage-audit.py` -- standalone Python audit (redundant with Rust,
  usable without rebuilding the bench binary)
- `scripts/kernel_atlas.py` -- `parse_rocprof_coverage_section` +
  `annotate_rocprof_coverage` helpers; collect-ar surfaces the coverage
  metrics in the atlas rows and tags BLINDSPOT rows for downstream filtering.
