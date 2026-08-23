# Qwen3.5-4B MQ4 campaign — amendment 2 (profiler blind spot, lever falsifications, gate fixes)

**Date:** 2026-08-23 · **Lifecycle:** historical
**Parent:** [2026-08-22-qwen35-4b-mq4-300tps-campaign.md](2026-08-22-qwen35-4b-mq4-300tps-campaign.md)
**Branch:** `tune/iter3-gate-up-bt2` · **Worktree:** `~/projects/hipfire-tune4b`
**Fixture:** `Qwen/Qwen3.5-4B-MQ4/qwen3.5-4b.mq4`, md5 `712b69f8cf1016081cfa507c4d50e33d`
**GPU:** gfx1100 RX 7900 XTX, exclusive via `/home/ghazni/gpu-coord`.

## Standing result (unchanged)

Product path, 5 runs, shipped env (`HIPFIRE_QKVZA_FUSEDNORM=1 HIPFIRE_FA_KVWRITE_FOLD=1`,
image rebuilt from this branch): **tg128@64 = 208.86 tok/s median**, tg128@2048 = 204.28.
Confirms the parent doc's engine-only 208.49 reading on a different day/build.

## Profiler blind spot fixed: fused_gate_up was invisible

The decode-path FFN kernel `fused_gate_up_hfq4g256` launched without profiler
timers, so every DECODE PROFILE in this campaign under-counted it. Instrumented:
**4096 launches / 128 tok = 32/token, 42 µs/call serialized, 555 GiB/s, 21% of
decode — the #2 kernel line**, previously absent from all profiles. The launch
arithmetic that "closed" at 274/token was circular; real launches/token are ~306.
Corrected per-token budget (serialized µs): gate_up 1346, residual GEMV 1323,
rmsnorm_mq_rotate_awq 793 (already the DIRECT variant by default at k=2560),
lm_head 776, qkvza 722, gdn 320, silu_mul 300, gated_norm 257, conv1d 219,
qkv 215, fa_prep 87.

## Levers falsified this session (all in-engine, fresh-process, interleaved)

1. **Persistent dual-row residual GEMV** (`gemv_hfq4g256_residual_persist_r2`,
   bit-exact vs shipping): standalone DRAM-cold probes showed +50% GB/s on the
   wo shape (400→601), but in-engine A/B showed ZERO delta (both 21 µs/call
   serialized). The probe environment (rotating freshly-DMA'd buffers,
   back-to-back independent launches) is NOT representative; trust only
   in-engine measurement. Env-gated `HIPFIRE_RESIDUAL_PERSIST_R2=1`, default OFF.
2. **gate_up PAIR2** on 9216/9216/2560: 192.75 vs BASE 199.25 avg (-3.2%),
   byte-identical output. Matches the 27B negative. Opt-in flag kept as oracle.
3. **gate_up STAGE_X32** on 4B shape: HANGS nondeterministically (GPU pegged
   100% on an unfinished kernel, twice). Explicit opt-in only + code warning.
4. **AWQ_NORM_WAVEGRID**: neutral (198.9/198.5 vs 199.5/198.5), parity OK.
   Note `HIPFIRE_GFX1100_AWQ_NORM_DIRECT` is already DEFAULT-ON at k=2560 —
   the rmsnorm line in profiles is already the optimized variant.

## Bugs fixed in the gates (shipped regardless of perf outcome)

A mechanical relaxation of the seven `GFX1100_DENSE_GATE_UP_*` shape gates
introduced operator-precedence errors (`is_gfx1100 && shape27B || shape4B && ENV`),
letting the 4B-shape clauses bypass the arch check entirely. All seven gates are
now correctly grouped under `is_gfx1100()`. Also added a missing profiler timer
to the fused_gate_up launch path so future profiles include it.

## Ceiling assessment after this amendment

250 tok/s requires 4.00 ms/token ⇒ 647 GB/s sustained over every model byte
(2.54 GB/token incl. the q8_0 tied lm_head). The fleet achieves ~525 GB/s
average today; the largest kernels sit at falsified-local-optima schedules, the
lm_head streams at 86% of the 960 GB/s ceiling, and the MQ4 format's 136-byte
group structure caps short-row GEMV efficiency at a level no schedule change
reached in 10+ falsified attempts across two campaigns. Infinity Cache offers
no exploitable reuse: weights are touched exactly once per token and evicted
between tokens (2.5 GB stream through 96 MB L2); KV + DeltaNet state (few MB)
are already L2-resident. Under the stated constraints (plain AR, q8 KV, shipped
precisions) the honest engine-only ceiling remains ~210-220 tok/s on gfx1100;
250 requires either speculative decoding or weight-format changes, both out of
scope by ruling.

## Amendment 2b (same day): rocprofv3 ground truth re-baselines the budget

Event-instrumented DECODE PROFILES inflate tiny kernels 2-6x (per-launch event
sync serialization): e.g. `fused_rmsnorm_mq_rotate_awq` reads 12-13 us
instrumented but **4.6 us** under rocprofv3; `gated_delta_net_q8_compact2_b2`
13-14 -> 5.8 us; silu_mul 10 -> 1.9 us. True steady-state decode per token
(rocprofv3 dispatch records, uninstrumented durations):

| kernel | /tok | true avg | true rate |
|---|---|---|---|
| fused_gate_up_hfq4g256 | 32 | 34.1 us | ~735 GiB/s |
| gemv_hfq4g256_residual | 64 | 14.1 us (bimodal 10.4/17.9) | ~520/700 GiB/s |
| gemv_q8_0 (lm_head) | 1 | 758.9 us | ~840 GiB/s |
| fused_qkvza_hfq4g256 | 24 | 22.5 us | ~747 GiB/s |
| fused_rmsnorm_mq_rotate_awq | 64 | 4.6 us | latency-bound |
| fused_qkv_hfq4g256 | 8 | 18.6 us | ~745 GiB/s |
| gdn compact2_b2 + all producers | ~90 | 2-13 us | latency-bound |

Kernel-busy total ~4.18 ms/token; wall ~4.85 ms -> **~0.7 ms/token of
graph-node overhead + inter-kernel gaps across ~306 replay nodes**
(~2-5 us/node). Most weight-streaming GEMVs already run at 700-800 GiB/s;
the only laggard is the wo shape (~576 GiB/s).

CONSEQUENCE: the campaign pivot is launch-count reduction, not kernel
bandwidth. Three sessions of BW grinding hit falsified local optima because
the kernels were already fast; the recoverable pool is node overhead
(~0.3-0.5 ms via merging producer-chain launches into neighbors where the
redundant-compute tradeoff is favorable) plus the wo shape (~90 us if its
deficit is ever explained). Realistic ceiling with both: ~225-240 tok/s;
250 additionally requires the whole 2.53 GB stream at >750 GiB/s average,
above every demonstrated rate except lm_head's 840.
