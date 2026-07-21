---
name: hipfire-tester
description: Guide a tester through hipfire bring-up, serve smoke, claim-scoped harnesses, and benchmark reporting on AMD RDNA/CDNA GPUs. Use when the user wants a route-oriented test plan or an upstream-ready tester report. Retired batteries are not current evidence — use [`docs/VALIDATION.md`](../../../docs/VALIDATION.md).
---

# hipfire-tester

Route-oriented bring-up, serve, and reporting skill. Pick the **narrowest**
validation route for the claim under test; do not invent a universal matrix.

Load [`guide.md`](guide.md) when running a full session or drafting a report.

## Authority

| Concern | Owner |
|---|---|
| Claim → route map | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |
| Docs / lifecycle / ownership | [`docs/INDEX.md`](../../../docs/INDEX.md) |
| Model tags, VRAM, registry files | [`registry/models.json`](../../../registry/models.json), [`docs/MODELS.md`](../../../docs/MODELS.md) |
| CLI surface | [`docs/CLI.md`](../../../docs/CLI.md) |
| Serve HTTP | [`docs/SERVE.md`](../../../docs/SERVE.md) |
| Config / env | [`docs/CONFIG.md`](../../../docs/CONFIG.md), [`docs/env-vars.md`](../../../docs/env-vars.md) |
| Perf protocol | [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) |
| Redline certification | [`docs/REDLINE.md`](../../../docs/REDLINE.md) |
| Quant design | [`docs/QUANTIZATION.md`](../../../docs/QUANTIZATION.md) |

**Fail closed:** unknown claim class → no improvised gate. Green no-GPU CI is
not GPU correctness. Registry presence is not runtime admission
([`docs/admissions.yml`](../../../docs/admissions.yml) records stay empty until
earned).

## What this skill is not

- Not a universal GPU replacement gate.
- Not numerical/state parity (use an arch-owned oracle when one exists; else blocked).
- Not Redline product timed-arm proof from harness success alone.
- Not promotion authority for benches, Atlas rows, or historical checkpoints.

## Quick start (bring-up only)

```bash
hipfire diag
hipfire pull qwen3.5:4b          # tag from registry/models.json; other tags OK if VRAM fits
hipfire run qwen3.5:4b "Explain WMMA in one paragraph."
```

If `diag` or first run fails, chain to `hipfire-diag` / `hipfire-autoheal`.

## Claim → harness (summary)

Full map: [`docs/VALIDATION.md`](../../../docs/VALIDATION.md). **Branch by
claim class immediately** after identity/build — do not run a fixed linear
battery. Maintained manual harnesses that exist today:

| Claim class | Route |
|---|---|
| New/changed numeric `.hip` | `target/release/examples/test_kernels` **then** the applicable model/path-level manual route for that arch/surface ([`docs/VALIDATION.md`](../../../docs/VALIDATION.md)). Channel alone is **not** the complete route. |
| Forward / fusion / KV numerical or state | Path-specific parity/state oracle for that arch/surface; **BLOCKED** if none exists — never substitute serve harness or perf. |
| User-facing serve semantics | `python3 scripts/serve_harness.py --model <path> …` (after parity when numbers/state can change) |
| LFM2.5 chat / thinking frame | `python3 scripts/serve_harness.py --model <path> --tag lfm2.5:… --sampling registry` |
| Optional wrapper (Redline + serve; perf on by default) | `scripts/gates.sh --model <path> --no-perf` for harness-only; omit `--no-perf` only on a clean tree willing to let `probe_commits.sh` mutate checkout |
| Redline capture / shadow evidence | `python3 scripts/redline_daemon_harness.py --model <path>` + [`REDLINE.md`](../../../docs/REDLINE.md) ladder |
| Prefill/decode floor vs committed baselines | `scripts/speed-gate.sh` when that path’s policy applies |
| Fresh-process A/B | `scripts/probe_commits.sh` (default arm of `gates.sh`; disable with `--no-perf`. Mutates worktree — clean tree required) |
| Product-CLI synthetic daemon bench probe | `hipfire bench <tag>` / `hipfire bench <tag> --matrix …` — synthetic `bench_prefill`/`bench_decode` probes, **not** production-path authority |
| Production-path throughput report | Native `serve_harness.py` JSON / daemon `done` fields (`prefill_tok_s`, `decode_tok_s`, `ttft_ms`) per [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) and [`docs/methodology/bench-suite.md`](../../../docs/methodology/bench-suite.md) |

**Retired batteries** are not current evidence. Route claims through [`docs/VALIDATION.md`](../../../docs/VALIDATION.md); never require a retired battery for merge, promotion, or report pass/fail.

## Report contract

Include identity, not folklore:

- GPU marketing name, gfx arch, ROCm/HIP version (`hipfire diag`)
- hipfire version / binary path; daemon or bench binary md5 when claiming perf
- Exact model path or registry tag + file basename; draft path if DFlash-related
- Relevant config (`kv_cache` / `dflash_mode` / thinking) and full command lines
- Prompt identity (path or md5) for any timed or semantic fixture
- Which **VALIDATION route** was run and what evidence class it covers
- Pass/fail per route; separate “measured” numbers from “admitted” claims
- Notes: hangs, OOM, first-run JIT cost, dirty worktree

Template and phase checklist: [`guide.md`](guide.md).
