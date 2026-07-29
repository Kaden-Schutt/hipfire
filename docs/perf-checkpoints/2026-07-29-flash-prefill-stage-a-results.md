# Stage A results — query-tiled Q8 flash prefill attention

Date: 2026-07-29
Branch: `perf/flash-prefill-attention` (from `e2169c2d`, PR #534 head)
Spec: [`2026-07-29-attention-prefill-flash-kernel-design.md`](2026-07-29-attention-prefill-flash-kernel-design.md)
Hardware: gfx1151 (Strix Halo), Qwen3.6-35B-A3B MQ4R, Q8 KV, MTP off

## Verdict

**Stage A gate: PASS on the HIP backend, FAIL on the Redline backend.**

The kernel is correct and materially faster where it is used, but enabling it
degrades output when Redline's retained PM4 route is active — and Redline is
default-on for this model on gfx1151 as of `e2169c2d`. It therefore ships
opt-in only (`HIPFIRE_FLASH_PREFILL=1`), default off, and must not be enabled
alongside Redline until that interaction is understood.

## Correctness

`test_q8_flash_prefill` compares against `attention_q8_0_kv_batched` under
`|ref − new| ≤ ATOL + RTOL·|ref|` (ATOL 1e-5, RTOL 1e-4) plus cosine ≥ 1 − 1e-6.

All shapes PASS, worst element consuming 6.3% of the tolerance budget:

- CTX 32 / 512 / 1024 / 4096 / 8192 / 12288 at N 16..256
- boundaries 33/16, 31/8, 65/17, 97/33, 129/16 (`seq_len % BC != 0`,
  `seq_len < BC`, `batch_size % BR != 0`)
- ragged non-monotonic `positions[]`
- GQA kv_group 1 / 2 / 4 / 8

## Tile-size sweep (CTX=8192, N=256)

| BR | BC | LDS (B) | ms |
|---:|---:|---:|---:|
| 8 | 16 | 17,504 | **23.65** |
| 16 | 16 | 26,304 | 27.32 |
| 8 | 8 | 12,896 | 28.11 |
| 8 | 32 | 26,720 | 30.01 |
| 4 | 8 | 8,624 | 34.70 |
| 32 | 8 | 38,528 | 46.67 |
| 32 | 64 | 76,160 | N/A (>64 KB) |

Not monotonic in LDS — BR=4/BC=8 has half the winner's footprint and is 47%
slower. The optimum balances reuse (BR), tile efficiency (BC) and occupancy
(3 WG/CU). **Chosen: BR=8, BC=16.**

## Kernel-level context sweep (BR=8, BC=16, N=256)

| CTX | tiled fallback | flash | legacy LDS | flash vs tiled | flash vs legacy |
|---:|---:|---:|---:|---:|---:|
| 2048 | 9.44 | 5.59 | 3.76 | 1.69× | 0.67× |
| 4096 | 21.20 | 11.75 | 8.72 | 1.80× | 0.74× |
| 8192 | 43.42 | 23.71 | 22.73 | 1.83× | 0.96× |
| 12288 | 64.06 | 35.98 | 42.78 | 1.78× | 1.19× |
| 16000 | 84.72 | 47.53 | N/A (>64 KB LDS) | 1.78× | — |

Flash beats the tiled fallback by ~1.8× at **every** context. It loses to the
legacy LDS kernel below the break-even, which is measured between CTX 10240
(0.95×) and 11264 (1.17×). The crossing happens because legacy's LDS grows with
context and sheds occupancy while flash's is constant — the mechanism the spec
predicted, now measured.

Consequence: the spec's hoped-for "one kernel for all lengths" does not hold on
this hardware. The right policy is hybrid — legacy below ~10.2K, flash above,
and the tiled fallback retired from this path entirely.

## End-to-end prefill (HIP backend, real prose, `MIN_CTX=10240`)

| ctx | baseline ms | flash ms | speedup | baseline tok/s | flash tok/s |
|---:|---:|---:|---:|---:|---:|
| 7714 | 13498.0 | 12966.6 | 1.04× | 571.5 | 594.9 |
| 7839 | 13133.1 | 12702.4 | 1.03× | 596.9 | 617.1 |
| 11494 | 29609.0 | 27121.8 | 1.09× | 388.2 | 423.8 |
| 11687 | 31147.6 | 27888.5 | 1.12× | 375.2 | 419.1 |
| 13939 | 44749.9 | 36517.9 | **1.23×** | 311.5 | 381.7 |
| 14737 | 49864.2 | 40476.9 | **1.23×** | 295.5 | 364.1 |

Below `MIN_CTX` the numbers are ~1.0× as designed (flash is not used there).
Output quality: 6/6 `finish=stop`, coherent 146–217 word answers.

## The Redline failure

Same binary, same real-prose prompts, three arms:

| arm | result |
|---|---|
| flash ON, `HIPFIRE_REPLAY_BACKEND=hip` | 6/6 `finish=stop`, 146–217 words — clean |
| flash OFF, Redline (default) | 6/6 `finish=stop`, 182–206 words — clean |
| **flash ON, Redline (default)** | **1/6 clean; 5/6 truncated at 2–62 words** |

The flag-off arm exonerates the precompile-spec-list change, so the trigger is
specifically *using the flash kernel while Redline's retained route is active*.
Redline does not fail closed when the prefill attention kernel changes underneath
it — it produces degraded output instead.

One data point is unexplained and should be treated as a lead, not a
conclusion: `real7000r1` (ctx 7839) is below `MIN_CTX` and therefore ran on the
legacy kernel, yet it degenerated in the flash-ON Redline arm while being clean
in both other arms. That is inconsistent with a purely per-request effect and
suggests the retained route is perturbed process-wide.

This reproduces, with realistic in-distribution prompts and a proper HIP control,
the hypothesis that was earlier raised and then retracted. The retraction was
correct on the evidence available at the time — that run used random-word filler,
which degenerates the model identically on both backends and could not
discriminate. This result is not confounded that way.

## Methodology note

`coherence_probe --temperature 0.0` is **not** reliably deterministic on this
model: across 5 runs of an unchanged binary, 4 matched the baseline exactly and
1 degenerated into repeated token id 0 (caught by the probe's own
`loop_guard_mirror`). Single token-stream diffs are therefore not sufficient
evidence; repeat them. Two matching runs do not establish determinism.

## Next

1. Root-cause the Redline retained-route interaction. Until then flash stays
   opt-in and must not be combined with Redline.
2. Stage C (WMMA inner math) is gated on this. The tiling foundation it needs
   is in place, but there is no point optimising the inner loop while the
   kernel cannot be enabled on the default backend.
3. Independent of all of the above, the cheap win still stands: raising
   `Q8_BATCHED_LDS_CROSSOVER` from 8192 toward the real ~15.8K LDS bound is
   validated bitwise-identical and worth 1.26–1.72× in the 8.2K–11.5K band,
   because it replaces the tiled fallback with the legacy kernel exactly where
   the legacy kernel is still the faster of the two.
