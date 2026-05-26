# gfx1151 decode-attention PMC: GQA traffic verdict (2026-05-26)

Branch `feat/dots-ocr-phase-3-daemon` @ d8bfe006. ROCm 7.13 / rocprofv3 1.3.0.
Bench: `bench_decode_attention --seq 5100 --iters 1` (n_heads=12, kv=2, hd=128).

## PMC works here (gfx1100 blocker resolved)

The handover (`/tmp/gfx1151-decode-pmc-handover.md`) couldn't get FETCH/GL2C/SQ
counters on the 7900 XTX — flat zero. On gfx1151 they read **nonzero**:
single-counter sweep, deterministic merge on Kernel_Name, second (non-warmup)
dispatch.

| kernel | grid wg | FETCH_SIZE | GL2C_HIT | GL2C_MISS | hit% | SQ_WAVES | VALU |
|---|---|---|---|---|---|---|---|
| attention_flash_partial | 480 | ~115.8k | 3.88M | 1.17M | 77% | 1920 | 2.42M |
| attention_flash_gqa_partial | 80 | ~61.2k | 25.3k | 492k | 5% | 320 | 2.34M |
| attention_f32 | 96wave | 10.2k | 163k | 81.7k | 67% | 96 | 2.35M |
| attention_q8_0_kv | 96wave | 2.7k | 62.5k | 21.8k | 74% | 96 | 2.86M |

Plain timing: gqa 256µs, flash 819µs, f32 273µs, q8 272µs. gqa output exact-equal
to flash (maxdiff 0.0).

## Verdict on the 6× hypothesis: partly killed

- **Memory-bound: CONFIRMED.** VALU is flat across flash/gqa (2.42M vs 2.34M).
  The +3.2× wall win is not compute.
- **6× fewer KV bytes: NO.** HBM fetch only halves (1.9×), not 6×. Wave count
  drops exactly 6× (1920→320) as predicted, but per-head flash's redundancy was
  already absorbed in L2 (77% hit). GQA misses L2 (5% hit) and streams once from
  HBM, so the *HBM* delta is ~2×, not 6×. The reuse wins in L2, not at the HBM
  ceiling. GQA's gain = killing 6× wave launches + L2 round-trips.

q8 KV: 2.7k fetch (40× less than flash) but same ~272µs wall — KV bytes aren't
the decode floor here, dispatch/wave overhead is. Rejected lever, confirmed.

Tool of record stands: analytical bandwidth model + kernel-trace timing. PMC on
gfx1151 now available as an own-hardware cross-check.

## Negative result: partial+reduce fusion (one block per kv_head)

Fused single-launch gqa (no partials, no reduce, grid = n_kv_heads = 2): correct
(maxdiff 4e-8) but **24657µs vs 257µs gqa — 96× slower**. Reduce launch was never
the cost (48 waves, 244 fetch); collapsing 80wg→2wg destroys occupancy.
**Confirms occupancy, not launch overhead, is the decode floor.** Kernel left in
tree (`attention_flash_gqa_fused`) off the dispatch path.

## Occupancy sweep: chunk_size (HIPFIRE_GQA_CHUNK), seq 5100

| chunk | wg | µs | correct |
|---|---|---|---|
| 128 | 80 | 257.6 | ✓ |
| 64 | 160 | 252.8 | ✓ (~2%, in-noise) |
| 32 | 320 | 291.3 | ✗ maxdiff 7.0 — block<reduce assumes 64 |
| 16 | 640 | 412.4 | ✗ maxdiff 1e2 |

80 wg already saturates the 40-CU APU at this size; doubling buys ~2% (noise),
below 64 breaks correctness + regresses. Both cheap decode levers spent on
gfx1151: fusion dead, occupancy flat. `HIPFIRE_GQA_CHUNK` left in as the tuning
knob for gfx1100.

## gfx1100: re-run candidate — outcome likely differs

Optimum is gfx1100-specific, not transferable from this APU. Re-run there if:
- 96 CUs vs ~40 → 80 wg may *underfill* gfx1100; `HIPFIRE_GQA_CHUNK=64`/96 may
  pay where it's noise here. Sweep + PMC wave count.
- 960 GB/s GDDR6 vs 115 GB/s LPDDR5X → decode may be bandwidth-bound there;
  f16 KV (halves FETCH) could win where q8/f16 tied here. PMC dead on gfx1100,
  so use kernel-trace wall ÷ KV byte volume. Bench: `bench_decode_attention
  --seq 5100 --iters 100`; long-seq (12000) stresses bandwidth more.
