# gfx906 MMQ — default-on flip + re-profile

Date: 2026-05-04
Hardware: gfx906 (MI50). HIP 6.4.3.
Workload: Qwen 3.5 9B mq4, `bench_qwen35_mq4`.

## TL;DR

Flipped MMQ from opt-in (HIPFIRE_MMQ=1) to **default-on at
batch_size ≥ 16** on gfx906 only. Same change pairs the per-arch
screening defaults: `mmq_screen=true`, `mmq_screen_threshold=0.50`.

End-to-end pp128 with no env vars: **141 → 461 tok/s (3.27×)**.

The cutover at 16 is empirically the right break-even point: below
pp16 the Q8_1 quantize + per-output launch overhead dominates and
FP16 wave64 wins; pp16 is already 1.46× over baseline; pp128 is
3.27×; pp512 is 3.89× and 74 % of stock llama.cpp.

## Why min_batch = 16, not 1 or 256

Small-batch sanity sweep on Qwen 9B mq4 (5 runs/config, last-run):

| Prefill | Baseline (FP16 wave64) | MMQ on | Speedup |
|---|---|---|---|
| pp2  | 68.9 | 30.0 | **0.44×** ⛔ |
| pp8  | 120.4 | 113.4 | 0.94× |
| pp16 | 131.4 | 192.2 | **1.46×** ✓ |
| pp32 | 136.1 | 276.4 | 2.03× |
| pp128 | 141 | 462 | 3.27× |

At pp2 MMQ is **>2× slower** because:
- Q8_1 quantize is ~14 µs/call (one per layer per prefill)
- 4 separate MMQ launches per layer (vs 1 fused FP16 wave64)
- mmq_x=8 tile means 75 % of column work is wasted at pp2

At pp8 MMQ runs the right tile size (mmq_x=8 perfectly fills) but
launch overhead still eats the gain. At pp16 work doubles while
launch cost stays fixed, tipping the balance.

Default-on threshold for non-gfx906 archs unchanged at 256 because
RDNA3+ has WMMA (faster than MMQ at small batches) — flip there
needs separate validation.

## Default-on full sweep (no env vars set)

| Prefill | Path picked | tok/s |
|---|---|---|
| pp2 | FP16 wave64 (below cutover) | 69 |
| pp8 | FP16 wave64 (below cutover) | 120 |
| **pp16** | **MMQ** | **192** |
| pp32 | MMQ | 276 |
| pp64 | MMQ | 365 |
| **pp128** | **MMQ** | **461** |
| pp256 | MMQ | 561 |
| pp512 | MMQ | 554 |

pp256 and pp512 are essentially tied — see "remaining bottleneck"
section below.

## Coherence gate at default-on

`./scripts/coherence-gate.sh` with no env vars:

| Row | Status | Notes |
|---|---|---|
| qwen3.5-0.8b.mq4 / cap | OK | wall 3.3s (was 4.9s pre-default-on; faster from prefill MMQ) |
| qwen3.5-4b.mq4 / code | OK | wall 13.6s (was 18.3s) |
| qwen3.5-9b.mq4 / reason | OK | wall 21.4s (was 22.1s) |
| qwen3.5-9b.mq4 / tool-call | OK | clean tool_call, no `<\|im_start\|>` |
| qwen3.5-9b.mq3 / reason-mq3 | OK | mq3 doesn't dispatch through gfx906 MMQ |
| qwen3.5-27b.mq3 / cap-mq3-27b | OK | also unrelated |

Decode-side stays on FP16 wave64 (batch=1 < min_batch=16), so
decode tok/s is unchanged.

## rocprof attribution shift (vs pre-threshold-bump baseline)

Same workload (`--prefill 128 --prefill-runs 1 --gen 0`) under
`rocprofv3 --kernel-trace --stats`. Family aggregation:

| Family | Pre-threshold-bump | Post-default-on | Δ |
|---|---|---|---|
| MMQ residual + gate_up + qkv | 35.3 % | **66.4 %** | +31 pp |
| qkvza_fp16_wave64 (now beta+alpha tail) | 28.9 % | **0.3 %** | −28 pp |
| qkv_fp16_wave64 | 8.0 % | **0.0 %** | −8 pp |
| residual_fp16_wave64 (decode + screen-fallback) | 21.6 % | 25.6 % | +4 pp |
| Q8_1 quantize | 0.5 % | 1.4 % | +0.9 pp |

Top 5 individual kernels post-default-on:

| Kernel | Calls | Avg | Share |
|---|---|---|---|
| `_full_set_x64` (gate_up + qkv) | 136 | 1.23 ms | 31.76 % |
| `residual_fp16_wave64` (decode mostly) | 200 | 0.67 ms | 25.61 % |
| `_full_add_x64` (residual K=12288 mlp-down) | 64 | 1.57 ms | 19.03 % |
| `_full_add_x16` (residual K=4096 attn-out) | 200 | 0.41 ms | 15.58 % |
| (other) | — | — | <3 % |

Two MMQ kernels alone (`_full_set_x64` + `_full_add_x64`) carry
**50.8 %** of GEMM time. Per-call improvements there would be the
biggest lever.

## Stock llama.cpp comparison

From `2026-05-04-llamacpp-stock-comparison.md`: stock on gfx906
hits 750 tok/s pp512.

| Path | pp512 | Fraction of stock |
|---|---|---|
| Pre-redesign (FP16 wave64) | 142 | 19 % |
| Post-redesign, threshold=0.10, screening on env-only | ≈ 370 | ≈ 49 % |
| Post-threshold-bump (manual env) | 554 | 74 % |
| **Post-default-on (no env vars)** | **554** | **74 %** |

Default-on doesn't change the ceiling — it only moves the result
into the no-config path. The remaining 26 % gap is structural:

1. The two MMQ kernels themselves (50.8 % of GEMM) are 1.23–1.57
   ms/call vs stock's 0.85 ms/call extrapolated — sync frequency
   (8/group vs stock's 2/group) is the suspected cause.
2. Decode side is unchanged (separate optimization path).

## Remaining bottleneck — pp256 ≈ pp512 plateau

| pp | tok/s |
|---|---|
| 128 | 461 |
| 256 | 561 |
| 512 | 554 |

The plateau between pp256 and pp512 (1 % apart) suggests something
**other than per-call kernel work** is the bottleneck. Likely
candidates:

- **Launch overhead** — at large batches the per-call work goes
  up linearly but launch overhead is fixed. If we're CPU-bound
  on dispatch, kernel speedup doesn't translate to throughput.
- **Per-prefill setup cost** — embedding lookup, tokenizer,
  prefix calculations may scale poorly.
- **HBM bandwidth ceiling** — possible but unlikely on gfx906
  (768 GB/s) given the modest fetch sizes per call.

User flagged: daemon never takes more than ~200 % CPU during
prefill. Worth investigating whether we're CPU-starved on the
launch path (single thread issuing kernel commands while another
collects completions). Recorded as plan §P1.

## Code change

In `crates/rdna-compute/src/dispatch.rs`:

```rust
// Per-arch defaults for MMQ screening.
let mmq_screen_default: bool = arch == "gfx906";
let mmq_screen_threshold_default: f32 = if arch == "gfx906" { 0.50 } else { 0.10 };

// In should_use_mmq():
let arch_min_batch: usize = if arch == "gfx906" { 16 } else { 256 };
let min_batch = std::env::var("HIPFIRE_MMQ_MIN_BATCH")
    .ok().and_then(|s| s.parse::<usize>().ok())
    .unwrap_or(arch_min_batch);
batch_size >= min_batch
```

Removed: the explicit `if arch == "gfx906" { return false; }` in
`should_use_mmq` (which forced gfx906 to opt-in via HIPFIRE_MMQ=1).

`HIPFIRE_MMQ=0` still disables; `HIPFIRE_MMQ=1` still forces on
at all batch sizes; `HIPFIRE_MMQ_MIN_BATCH=N` overrides the cutover.

## Reproducing

```sh
# Default-on, no env vars
$HIPFIRE/target/release/examples/bench_qwen35_mq4 \
  $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
  --prefill 128 --prefill-runs 5 --gen 0 --warmup 0

# Confirm decode (batch=1) still uses FP16 wave64
HIPFIRE_MMQ_TRACE=1 $HIPFIRE/target/release/examples/daemon < input.jsonl \
  | grep mmq-trace

# Coherence
$HIPFIRE/scripts/coherence-gate.sh

# rocprof
rocprofv3 --kernel-trace --stats -d ./run -o trace --output-format csv -- \
  $HIPFIRE/target/release/examples/bench_qwen35_mq4 \
  $HIPFIRE_MODELS/qwen3.5-9b.mq4 \
  --prefill 128 --prefill-runs 1 --gen 0 --warmup 0
```
