# Lloyd-Max codebook perplexity findings — 2026-05-01

**Source**: PRD `docs/plans/mq-sub4bit-research-queue.md` §Q1 (extended to MQ3)
**Hardware**: gfx1100 (RX 7900 XTX), wave32, fp16 WMMA
**Corpus**: `dev/bench/data/wikitext2-test.txt`
**Window**: ctx=2048, warmup=8, scored=2039 tokens, offset=0

## Aggregated table

| size | MQ4 (4 bpw) | MQ3 (3.25 bpw) | MQ3-Lloyd (3.5 bpw) | MQ2-Lloyd (2.25 bpw) | MQ2 uniform (2.25 bpw) |
|------|---:|---:|---:|---:|---:|
| 0.8B | **25.65** | 301.06 | **155.22** | 19,650.7 | 803,851.6 |
| 4B   | **12.73** | 45.24  | **22.56**  | 604.4    | n/m              |
| 9B   | **10.34** | 42.03  | **18.52**  | 2,162.9  | 120,108.4        |

**Codebook win factors**:
- 9B MQ3-Lloyd vs uniform MQ3: **2.27×** ppl reduction (42 → 18.5)
- 9B MQ2-Lloyd vs uniform MQ2: **55.5×** ppl reduction (120K → 2.2K)
- 0.8B MQ3-Lloyd vs uniform MQ3: 1.94× (still in collapse zone but halved)

## Headline result: Lloyd-Max MQ3 is shippable

**9B Lloyd-MQ3 ppl=18.52 is the closest any sub-4-bit format has gotten to MQ4
(10.34) — within 1.79×. Uniform MQ3 was at 4.07×.** For +7.7% bandwidth (112 vs
104 B/group, 8 fp16 centroids replacing fp32 scale+zero), we get a 2.27× ppl
cut. This is a clean product win — Lloyd-MQ3 should be the default 3-bit format,
not uniform MQ3.

The same algorithm consistently delivers ~2× across all model sizes (0.8B
1.94×, 4B 2.01×, 9B 2.27×) — Lloyd's per-block centroid optimization is
fundamentally the right shape for sub-4-bit weight reconstruction.

## What this means for the roadmap

**Ship 9B Lloyd-MQ3 as default, deprecate 9B uniform MQ3.** Quality cost vs MQ4
shrinks from 4× → 1.79× for nearly the same bandwidth (112 vs 136 B/group on
MQ4 = 17.6% smaller, vs MQ3's 23.5% smaller).

**4B Lloyd-MQ3 is borderline** (22.56 vs MQ4 12.73 = 1.77×). Eyeball the
4-prompt coherence battery; if fluent, ship; if attractor-loops, hold for
GPTQ (queue Q2).

**0.8B Lloyd-MQ3 is still collapse** (155 vs MQ4 26 = 6×). Sub-2B at any
sub-4-bit will need either GPTQ activation-weighted quantization (Q2) or
mixed-precision MQ-hybrid (Q4) to ship.

**Lloyd-MQ2 is research-only** — gated behind `HIPFIRE_ALLOW_MQ2_LLOYD=1`.
Even at 9B, ppl=2163 is text-quality collapse. The 55× win over uniform MQ2 is
informational, not shippable.

## Implementation cost summary

| component | files | lines |
|---|---|---|
| Lloyd-Max algorithms | `crates/hipfire-quantize/src/main.rs` | +200 (MQ2 + MQ3 + parallel) |
| Storage formats | qt=19 (MQ2-Lloyd, 72 B/group), qt=20 (MQ3-Lloyd, 112 B/group) | — |
| GEMV kernels | `kernels/src/gemv_mq{2,3}g256_lloyd.hip` | +120 |
| DType + dispatch | `crates/rdna-compute/src/dispatch.rs` | +50 |
| Engine wiring | `crates/engine/src/{hfq,llama,qwen35}.rs` | +60 |
| Perplexity harness | `crates/engine/examples/perplexity.rs` | +120 |
| Research-only guards | `--allow-mq2-lloyd`, `--allow-mq3-lloyd` env+flag | +30 |

Total: ~580 LOC. Quantizer parallelized via rayon `par_chunks_mut` over output
blocks: 9B Lloyd-MQ3 quantize runs in ~85s wall on a 24-core box (vs 5+ min
serial-Lloyd that didn't finish in the first attempt).

## Decoder perf cost (preliminary, gfx1100)

The 8-way switch in `gemv_mq3g256_lloyd.hip` doesn't optimize as cleanly as
uniform MQ3's `scale*q + zero`:

| variant | 9B decode (tok/s, single-window ppl harness) |
|---|---:|
| MQ3 uniform | ~141 (per `tests/speed-baselines/gfx1100.txt`) |
| MQ3-Lloyd   | 44 (perplexity harness, 100% per-token) |

The 3× slowdown is real but expected — switch-based codebook lookup is harder
to optimize than affine reconstruction. Likely recoverable with the same K4
unrolling pattern that brought uniform MQ3 from 114 → 141 tok/s. Tracked as
follow-up; quality win justifies shipping the slower decoder first.

## Next moves

1. **Lloyd-MQ3 K4-unroll kernel** — bring decode tok/s back to ~140 (parallel
   with 4 K-tile accumulators, same pattern as `gemv_hfq3g256.gfx1100.hip`).
2. **Lloyd-MQ4 (qt=21)** — 16 fp16 centroids + 4-bit indices = 168 B/group
   (+23.5% bandwidth over uniform MQ4). Test whether MQ4 has remaining
   quant-loss room; if yes, this becomes the default.
3. **Eyeball-validate 4B/9B Lloyd-MQ3** through the 4-prompt coherence battery.
4. **WMMA prefill Lloyd-MQ3 family** — uniform MQ3's WMMA family doesn't
   transfer directly because the recon is a switch, not affine. Need a fresh
   kernel or compromise to per-row GEMV at prefill.

## Files / artifacts

- `crates/engine/examples/perplexity.rs` — single-window NLL harness
- `benchmarks/run_ppl_baseline.sh` — sweep driver (MQ4/MQ3 baseline)
- `benchmarks/run_lloyd_compare.sh` — comparison driver
- `benchmarks/results/ppl_baseline_20260501T061036Z.md` — raw baseline
- `benchmarks/results/lloyd_max_findings_20260501.md` — this file
- Kernels: `kernels/src/gemv_mq{2,3}g256_lloyd.hip`
- Quantizer: `quantize_mq{2,3}g256_lloyd` in `crates/hipfire-quantize/src/main.rs`
  (rayon parallel over per-block Lloyd's iterations)
- Engine wiring: `DType::MQ{2,3}G256Lloyd` in dispatch + load + GEMV arms

## Quantize artifacts (ephemeral, NOT committed, NOT shipped to HF)

- `~/.hipfire/models/qwen3.5-{0.8b,4b,9b}.mq3-lloyd` (~480 MB / 2.1 GB / 4.6 GB)
- `~/.hipfire/models/qwen3.5-{0.8b,4b,9b}.mq2-lloyd` (~424 MB / 1.7 GB / 3.3 GB)
- `~/.hipfire/models/qwen3.5-{0.8b,9b}.mq2` (uniform MQ2 baseline)
