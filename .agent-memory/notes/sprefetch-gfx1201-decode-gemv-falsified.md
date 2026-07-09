---
title: S_PREFETCH_DATA on gfx1201 a3b decode GEMVs FALSIFIED (DO-NOT-RETRY) — qkvza/gate_up/down all DEAD, TLP already hides latency
date: 2026-07-09
tags: [sprefetch, s_prefetch_data, gfx1201, rdna4, decode, gemv, moe, qkvza, gate_up, moe_down, falsified, do-not-retry, phantom-win, occupancy, certify-v2p]
---

**Verdict: DEAD across the board. Scalar S_PREFETCH_DATA is NOT a decode lever for
the a3b mq4r batch-1 decode GEMVs on gfx1201/R9700.** Certified vs f33ae7e9 with the
canonical `autoresearch/harness/ab_certify_v2p.sh` (adaptive-sampled decode tok/s +
token-id parity gate + coherence + profile), qwen3.6-35b-a3b.mq4r, kv q8, AUTO clock,
clean card (dev2/card2). All three targets: token-id EXACT, var_coh OK, perf DEAD.

| kernel (KERNEL arg) | variant | verdict | delta_pct | f | rounds | occ (base→var) |
|---|---|---|---|---|---|---|
| fused_qkvza_hfq4g256 | next-quad gp0+544 len0 (the extracted ledger "win") | **DEAD** | −0.26% | 0.625 | 4 | 47.8% |
| gemv_hfq4g256_moe_gate_up_indexed | next-quad gate+up len0 | **DEAD** | +0.16% | 0.611 | 6 | 71.6→51.0% |
| gemv_hfq4g256_moe_down_k8_indexed_batched_expanded | next-quad ×4row len0 | **DEAD** | +0.46% | 0.635 | 10 | 65.5% |

**Mechanism (harness profile_feedback, profile_standard PMC — this is the DEP_WAIT
answer):** the a3b batch-1 decode GEMVs are NOT latency-exposed. They run at **48–72%
achieved occupancy**, so thread-level parallelism (many concurrent waves) ALREADY hides
the weight-load latency. The original hypothesis ("83–94% exposed DEP_WAIT / SQ_WAIT_ANY
= exposed weight-load latency for prefetch to overlap") does not hold *in-model* — that
was an isolated-not-in-model read. `s_prefetch_data` is a wave-uniform SCALAR hint whose
only in-model effect is +VGPR pressure (gate_up 72→96, qkvza 72→88), which **DROPS
occupancy** (gate_up 71.6%→51.0%) and thus HURTS the very TLP that was hiding the latency.
Weights are also 61–76% L2-resident, so little DRAM latency remains to overlap. Harness
strings: qkvza "mem_busy 57.8%+occ 47.8%=TLP already hides latency, NOT MLP-limited";
gate_up "occ 71.6->51.0% DROPPED-WAVES … NOT MLP-limited"; down "no clear lever signal".

**Phantom-win falsified:** the ledger row `c3_fused_qkvza_prefetch4_gfx1201`
(swarm_gfx1201_fused_qkvza_hfq4g256.jsonl) reported **+3.78%** and was cited as an
unbanked, bankable win. It does NOT replicate: re-certified against f33ae7e9 on a clean
card it is DEAD −0.26% (f=0.625). It was never folded because it never actually wins
(NOT because it failed coherence — var_coh=OK). Another "synth/cross-build → prod-falsify"
per the memory pattern (feedback_v2_sgpr_lut, dot2_trickle_down, fp8_wmma). The qkvza
prefetch idiom (`gp0 + 544`, len 0) that scored +3.78% in a noisier cross-build A/B is
the same one that measures DEAD here.

**S_PREFETCH_DATA encoding (verified clang -S, gfx1201):**
`s_prefetch_data s[base:base+1], <byte-offset>, null, <LEN>` — base is a wave-uniform SGPR
pair (gate_ptr/up_ptr are uniform → legal), LEN is an immediate cache-line count (0..31
valid, one RDNA4 line = 128 B). The builtin's 2nd arg = LEN. len0 issues the instruction
(not folded away). This is the mechanism the qkvza-win idiom used (`len 0`).

**Dead-file correction:** `kernels/src/gemv_hfq4g256.gfx1201.hip` (the file the task pointed
at for the "existing" prefetch idiom, line 43) is NOT wired into the runtime — it is not
`include_str!`'d anywhere in crates/, there is no `kernels/compiled/` blob, and
`gemv_hfq4g256_for_arch()` routes gfx1200/gfx1201 to the BASE `gemv_hfq4g256.hip` (its
gfx1201 arm is commented out). So the "existing" S_PREFETCH on gfx1201 never executed
in-model. Only the wired MoE decode GEMVs matter, and prefetch on those is DEAD.

**Reproduce:** drop-in variant .hip (function name unchanged) then
`BASELINE_REF=f33ae7e9 SCLK=/sys/class/drm/card<N>/device/pp_dpm_sclk bash
autoresearch/harness/ab_certify_v2p.sh gfx1201 <dev> <card> ~/.hipfire/models/qwen3.6-35b-a3b.mq4r <KERNEL> <label> <variant.hip>`.
A gated same-daemon lever also shipped: `HIPFIRE_GFX1201_MOE_PREFETCH∈{1,2,3,4}` selects
gate_up prefetch variants in gemv_hfq4g256_moe_gate_up_indexed_prefetch.hip (default OFF).

**Op note:** `pkill -9` of a running kernel WEDGES that card — subsequent daemon generate
page-faults in the GPU `sample_top_p` kernel (gfxhub VM fault, code 700) while the forward
still loads/runs fine. card0/card1 got wedged this way; dev2/dev3 stayed clean. Prefer
`pkill -9 -x daemon`/`bench_qwen35_mq4` at a quiescent point; if a card page-faults in
sample_top_p across models+builds, it's a wedge, not a real bug — use a fresh card.

**Endpoint:** every single-kernel scalar-prefetch lever on gfx1201 a3b decode is dead;
decode is TLP-latency-hidden (occupancy-work-limited), not MLP/latency-exposed — consistent
with project_dp4a_moe_decode_bw_bound_falsified and the "decode kernel TAPPED" conclusion.
Do not re-file scalar prefetch on these kernels. The remaining regime-changer stays
MTP/spec-decode batch-K, not a single-kernel decode-GEMV edit.
