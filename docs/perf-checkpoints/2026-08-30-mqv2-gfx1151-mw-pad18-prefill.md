# 2026-08-30 — MQ V2 gfx1151 MW-LDS + pad-18 prefill program

**Lifecycle:** historical
**Disposition:** measured; both policy stages shipped (commits `c939b483c`, `806a85c74`); gfx1100 QKV/QKVZA MW row measured flat and reverted in `806a85c74`.

## Fixture

- Host: hipx, GPU HIP device 1 = gfx1151 (Radeon 8060S), ROCm 10.0.0-4
  (`/opt/rocm/bin/hipcc`, banners read HIP 7.15.26333 / clang 23.0.0git — normal
  for this packaging era), health counters 0 throughout.
- Models: `~/qcal/ladder-v2/artifacts/qwen3.8-27b.mq3v2.xt.hfq`
  (md5 `80bb9198e6a565fc006b2ae1b7c89eca`) and `qwen3.8-27b.mq6v2.xt.hfq`
  (md5 `de6eb059a10577b80f0a162e5c89249e`).
- Method: `hipfire bench <model> --matrix --pp 96,128,256,384,512 --ctx 512
  --tg 16 --json`, `HIPFIRE_LOCAL=1`, fresh process per run, 3 interleaved
  runs per side, medians of per-run medians. Synthetic pp/tg matrix — no
  prompt-content dependence.
- Binaries: baseline `02cb9fb17` (md5 `1723dc7a…`), stage-1 `c939b483c`
  (md5 `90477c2f…`), stage-2 `806a85c74`+ (built in `hipx:~/mqv2-flip`).
- Raw JSON: `.codeinsight+research/ledger-exec/{e1/e1-ab,flip-ab}/` (local
  discovery pointers; durable numbers below).

## Measured (gfx1151 prefill tok/s, median-of-3)

Stage 1 = MW-LDS QKV/QKVZA + pad-18 + per-BITS window decode (`c939b483c`).
Stage 2 = gate_up/residual MW flip with measured per-bits wave regions (`806a85c74`).

| SKU | pp | `02cb9fb17` | `c939b483c` | stage-2 | cumulative |
|---|---|---|---|---|---|
| MQ3 | 96 | 214.3 | 263.8 (+23.1%) | 340.0 (+28.6%) | **+58.7%** |
| MQ3 | 512 | 211.4 | 250.6 (+18.5%) | 347.7 (+37.9%) | **+64.5%** |
| MQ6 | 96 | 195.0 | 250.9 (+28.7%) | 321.1 (+27.7%) | **+64.7%** |
| MQ6 | 512 | 187.0 | 239.2 (+27.9%) | 337.0 (+40.8%) | **+80.2%** |

Full pp96/128/256/384/512 rows in the raw JSON; every row cleared the ≥2%
per-SKU gate at both stages. Decode controls flat (tg16@512: MQ3 +0.2%,
MQ6 +2.7% — not claimed; MW policy cannot fire at N=1).

Correctness: all MW kernels raw `f32::to_bits`-equal to per-format base
oracles (`test_mqv2_mw_gfx11`, 56 arms gfx1151 / 42 gfx1100); the E2 screen
(`tools/kernels/gfx1151/mb_gateup_resid_mw_vs_bt.hip`, 126 runs) gated every
arm on bit-equality before any timing was read.

## Negative results retained by the same gates

- gfx1100 QKV/QKVZA MW: microbench 1.42× vs BT12 did **not** survive to
  whole-model (`qwen3.8-27b.mq6v2.xt` +0.17% overall) — policy row reverted.
- gfx12 HFQ4 lm_head overwrite dispatch: −4.3% at B=8 / +3.3% at B=16 —
  dispatch change abandoned; dirty-Y parity harness retained.

Not transferable across model, quant, GPU, prompt, route, or method without
a new record.
