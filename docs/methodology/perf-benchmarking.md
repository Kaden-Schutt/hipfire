# Perf benchmarking — methodology

Read this before claiming any kernel-level decode or prefill win. Written
after a session where a −13% regression shipped as a "+2% improvement"
because adjacent within-session measurements lied.

## 1. Within-session A/B is noisy

On gfx1100 specifically, `tok/s` measured on successive `bench_qwen35_mq4`
invocations within the same shell session can drift ±10–15% from each
other due to:

- DPM / clock-state settling. After many benches the GPU may sit at a
  different sclk DPM level than on a fresh process.
- Thermal history. The `edge` temp may look fine (<50 °C) while the memory
  sensor is 65 °C and memory clock is throttled.
- JIT cache state. A warm `.hipfire_kernels/` dir serves .hsaco blobs that
  may not match the current source if a hash collision or stale sidecar
  slipped through.
- rocprof / rocprofv3 instrumentation sometimes leaves the GPU in a
  degraded clock mode that a normal process exit doesn't clear.

**Rule:** Any `tok/s` delta under ~5 % measured within one shell session
is noise. Do not claim a win on it.

**DPM pinning via `HIPFIRE_DPM_WARMUP_SECS=N`:** Both `bench_qwen35_mq4`
and `dflash_spec_demo` accept this env var. It runs a 256 MB memset loop
for `N` seconds before the decode timer starts, which pins the card at
high DPM (effective ~770 GiB/s during warmup, verified with
`rocm-smi --showclocks`). Use 10 s for most runs. If bench-to-bench
variance is still >5 % with warmup, the noise source is elsewhere
(thermal, kernel cache, etc.) — don't ship until you understand it.

## 2. The speed-gate is the source of truth

`./scripts/speed-gate.sh` runs `bench_qwen35_mq4` on the 4 MQ4 sizes with
reproducible config (`HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1`, specific
prefill lengths, best-of-2) and compares against committed baselines in
`tests/speed-baselines/gfx1100.txt`.

- **If speed-gate passes, the change is shippable as far as perf goes.**
- **If speed-gate fails, the change is not shippable until the regression
  is understood** — not "noise", not "thermal", not "it worked in my last
  bench". Understand it.

Re-baselining (`--update-baselines`) is a load-bearing operation. Only
do it if the delta is intentional and explained in the commit message.
Never to mask an unexplained regression.

## 3. Verify across a fresh process before claiming a win

Before committing a perf change:

```bash
# On HEAD~1 (the commit before your change)
./scripts/probe_commits.sh $(git rev-parse HEAD~1) $(git rev-parse HEAD)
```

This force-checks out each commit, does an incremental build, and runs
the bench in a fresh process each time. The delta across commits is the
delta that matters; the delta within one shell is not.

If HEAD doesn't beat HEAD~1 on this probe, your within-session A/B lied.

## 4. Purge the JIT cache before a perf baseline

```bash
rm -rf .hipfire_kernels/
cargo build --release --features deltanet -p engine --example bench_qwen35_mq4
```

`include_str!` embeds kernel source at compile time, but the JIT cache
blobs in `.hipfire_kernels/*.hsaco` are keyed by a content-hash sidecar.
When the hash matches, the cached blob is served without recompilation.
A stale hash + stale blob combination silently runs old code against a
new binary. Always purge when baselining.

## 5. Bisecting decode regressions

`scripts/bisect_9b_decode.sh` is a `git bisect run`-compatible driver:

```bash
git bisect start
git bisect bad $REGRESSED_COMMIT
git bisect good $KNOWN_GOOD_COMMIT
git bisect run ./scripts/bisect_9b_decode.sh
```

Defaults: 9B MQ4 at `~/.hipfire/models/qwen3.5-9b.mq4`, pp=16 / warmup=3 /
gen=30, 125 tok/s threshold. Override via `HIPFIRE_9B_MODEL` and
`BISECT_TOK_S`. Build failures return 125 (skip). Takes ~2–5 min per step.

For spot checks rather than full bisect, `scripts/probe_commits.sh <hashes>`
runs the same bench on each listed commit without the pass/fail threshold.

## 6. Negative-result log

Things that looked like wins on gfx1100 / 7900 XTX in within-session A/B
but measured as no-op or regression on fresh-process probe. Commit hash
is where it was tried and reverted, so you can `git show` the kernel code.

| attempt | expected | measured | commit | notes |
|---|---|---|---|---|
| `__builtin_nontemporal_load` on HFQ4-G256 weight reads | +2 % (session A/B) | **−13 %** (fresh probe) | `0532579` → reverted in `34eb024` | Defeats default load-path wave coalescing. Don't try this on weight-streaming kernels. |
| block=64 "wave64" variant for `gemv_hfq4g256_residual` | +6.5 % | **no-op** | (discarded; see wave64 investigation branch before the revert) | The +6.5 % was an artifact of the nontemporal regression on the wave32 baseline. Once baseline is clean, block=64 and wave32 tie. |
| true wave64 compile (`-mwavefrontsize64` via `HIPFIRE_COMPILER_FLAGS` marker) | +4 % | **slightly worse than block=64** | (discarded) | RDNA3 wave64 doubles per-wave VGPR budget and halves occupancy without compensating throughput. |
| LA-preamble fusion: `fused_qk_l2_norm_scale + repeat_interleave → fused_qk_l2_norm_scale_repeat` | +2–4 % (fewer launches) | **neutral / slightly worse** (−2 % tok/s avg, within noise) | (discarded 2026-04-22) | 27B DFlash Fibonacci-continuation @ B=16: baseline avg 127.0 tok/s τ=6.24 vs fused 124.4 tok/s τ=6.00 (3-run each). Save ≈1 launch/LA-layer but cycle is kernel-compute-bound (ssync ≈ 38.6/55 ms); trimming 9 µs × a few launches is noise. τ itself is non-deterministic run-to-run (range 5.85–6.73), which dominates signal on short benches. |
| mw16 32-row block variant (shared X tile across 2 weight rows, halves block count) | +3–5 % via X BW amortization | **null** tok/s; τ determinism improves (locks to 5.316) | (discarded 2026-04-22) | 27B DFlash: baseline 112.9 tok/s (σ 4.4, τ 5.0–5.83) vs 32r 111.3 tok/s (σ 0.3, τ locked). X BW is tiny at batch=16 (~0.3 % of GEMM BW) so amortization saves nothing. τ determinism is a real side-effect — useful for #91 investigation but not tok/s. |
| mw16 K4 unroll (4 WMMAs per K-iter, 8 loads in flight) | +3–5 % from better load-latency hiding | **null** | (discarded 2026-04-22) | 27B DFlash 6-run: 112.7 tok/s vs baseline 112.9 (-0.2 %). Compiler at -O3 already pipelines the K2 kernel's 4 loads / 2 WMMAs adequately; K4 adds VGPR pressure without freeing new ILP slots. |
| mw16 LDS staging (cooperative A/B half-warp split into LDS, stride-17-dword row padding) | +10–20 % via dedup of duplicate wave32 A/B fragment loads | **−30 %** (222 µs/call vs 170 µs baseline) | (discarded 2026-04-22) | LDS round-trip + `__syncthreads` overhead dominates the theoretical dedup savings. Bank conflicts may also still fire on 16-lane column reads despite padding. mw16 inputs are already well-L1-cached across lane-halves. |
| mw16 non-residual (`Y = acc` not `+=`, skip pre-memset) | +2–5 % (fewer Y reads + skip memset) | kernel −2.5 % (38.1 vs 39.1 ms); **wallclock null** | (discarded 2026-04-22) | 27B DFlash 6-run: 112.8 tok/s vs baseline 112.9. The kernel-time saving is real but doesn't reduce wallclock because memset is async-overlapped with other stream work. |
|  | | | | |
| **Insight (#90, 2026-04-22)**: per-cycle host-timing probe showed `ssync = 39.8 ms` + `d2h = 11.4 ms` = 92 % of the 55 ms cycle wallclock. mw16 kernel time is 7.8 ms/cycle (14 % of wallclock). Further mw16 tile-level optimization has essentially no wallclock impact because mw16 isn't on the critical path — the cycle is serialization-bound, not kernel-compute-bound. Future perf work on this path should target ssync (`hipStreamSynchronize` in `commit_staging_to_ring`) and d2h sync overhead instead of kernel tiling. | | | | |
| `commit_staging_to_ring` async scatter (drop `stream_synchronize`, queue D2Ds async on `active_stream`) | −40 ms/cycle from the ssync | **null tok/s** (+1.0 % within noise; tokens byte-match baseline) | (discarded 2026-04-22) | ssync dropped from 39.8 ms → 0, but d2h grew from 11.4 ms → 50.8 ms to compensate. Total wallclock 54998 vs 55252 µs ≈ unchanged. The sync wasn't pure overhead — it was genuine GPU work time (kernels + memsets + d2d queued on the stream) that had to complete somewhere before the null-stream `download_f32` could return. Moving the wait from ssync-site to d2h-site preserves total. To actually eliminate the wait, the next cycle's work must be pipelined with the current cycle's d2h (inter-iteration overlap), not just scheduled on the same stream. |

Before starting a new kernel-level perf experiment, check this list. If
it's been tried and failed, don't rediscover unless you have a specific
reason to believe conditions changed (new ROCm version, new hardware,
different kernel family).
