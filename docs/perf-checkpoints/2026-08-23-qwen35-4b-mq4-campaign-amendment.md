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

## Amendment 2c (same day): serve-path quality battery — PASS

`scripts/serve_harness.py --mode battery` against the standing config
(q8 KV, speculation off, thinking off, seed 42, max_tokens 256, image built
from this branch): 5/5 genre turns (code/reason/factual/prose/instruct)
coherent and on-topic; runaway=2 (length-cap at temp=1.0, not degeneracy),
empty=0, attractor=0. Serve-path decode 204.4 tok/s average — consistent
with the product bench (208.86 tg128@64) within cross-path variance.
Daemon md5 11b76b1ea809233ff275a3af3e5ce72a.

Evidence chain for the standing config is now complete: bit-exact parity
certifications (parent doc) + product bench + serve battery + rocprofv3
ground-truth budget.

## Amendment 2d (same day): the certified qkvza fold was orphaned dead code — wired, then measured STRONGLY NEGATIVE

Pickaxe over all branches proves `fused_qkvza_hfq4g256_fusednorm` (commit
3abbc05e, checkpoint addendum 3's "+0.7%") was NEVER callable from any
dispatch path: no qwen35-forward or dispatch code referenced it — only the
probe example. `HIPFIRE_QKVZA_FUSEDNORM=1` has been a silent no-op in every
run since, including this campaign's. Addendum 3's e2e claim could not have
been produced by the product path.

This session wires the route (opt-in arm in the low-level PROJ_QKVZA match,
ahead of scalar_prep) and measures it properly:
- Numerics: probe BITEXACT on all four outputs; greedy temp-0 text
  byte-identical across three prompts.
- Performance: FN arms 68.0 / 149.9 tok/s vs OFF 197.1 / 199.3 — strongly
  negative (idle-gated arms, clean windows). Same prologue-redundancy
  failure mode as the gate_up fold: per-row-block norm+rotate recompute
  across the ~12k-row grid overwhelms the saved producer launch.

Status: wiring KEPT as a functional negative oracle behind explicit opt-in;
comment updated with measured numbers. Lesson recorded: certify levers
through the PRODUCT dispatch path, not through probe examples that call
kernels directly.

## Amendment 2e (same day): HIPFIRE_GEMV_ROWS falsified on qwen3.5-4b shapes

Idle-gated interleaved screen (R1/R2/R4 x2, example path, 384-token gens):
R1 = 199.85 tok/s avg, R2 = 193.85 (-2.9%), R4 = 193.90 (-2.9%). The
existing opt-in multirow flag is negative on the 4B residual/w_down shapes
just as it was on the sizes where it was originally rejected. No change
shipped; default remains rows=1.

## Amendment 2f: rocprof ground-truth closes the persist-residual question

rocprofv3 dispatch records for OFF vs HIPFIRE_RESIDUAL_PERSIST_R2=1 (same
run shape as 2b): residual kernel durations are IDENTICAL under both
schedules (lower-cluster mean 10.4 us / upper-cluster 17.5-17.9 us,
n=6336 each arm); gate_up unchanged (31.4 vs 31.6 median). The standalone
probe's "+50% GB/s" never existed as GPU-time on real workloads - it was
an artifact of the probe's rotating-buffer environment. Triple evidence
(probe, instrumented e2e, rocprof ground truth) now agrees: the single-row
STAGE_X32 schedule is time-optimal in-engine for these shapes, and the
residual/w_down duration split is intrinsic to their K lengths, not
wave-tail. No further scheduling experiments warranted on this family.

## Amendment 2g: conv1d->GDN prologue fold ruled out structurally

The last unexplored fusion candidate fails on data-flow grounds, not
performance grounds. gated_delta_net_q8_fast's grid is [n_heads, tiles,
lanes]; with QK_HEAD_DIV=2, two state-head blocks consume the same Q/K
channel slice, and the causal-conv rolling state must be updated exactly
once per channel. Folding conv into the GDN prologue therefore forces
either duplicate state writes (race) or a designated-writer scheme that
needs cross-block ordering (cooperative-launch redesign of a certified
kernel). The ~120 us pool is unreachable at acceptable risk.

With this, every launch-reduction candidate identified across the campaign
is closed: measured-negative (persist residual, PAIR2, GEMV_ROWS,
gate_up/qkv folds), structurally unsound (conv1d fold), hang-prone
(STAGE_X32), or already shipped (qkvza fusednorm route - itself later
measured negative when finally engaged). The campaign's honest conclusion
stands: standing 208.86 tok/s; realistic ceiling ~225-240; 250 requires
mechanisms outside the objective's constraints.

## Amendment 2h: gated_norm->GDN fold ruled out structurally

The final fusion candidate also fails on geometry. The GDN kernel tiles each
head's HD=128 output rows across TILE_ROWS=4-row blocks - grid [32 heads,
32 tiles, 2 lanes] = 2048 blocks - because the delta-rule state update is
itself tiled. The consumer gated_norm requires a FULL head row (128 values)
for its RMS reduction, plus pairs of heads (256 values) for the FWHT rotate.
No GDN block ever holds a complete head, so an epilogue fold would demand
whole-head blocks (grid collapsing to ~32 blocks on 96 CUs, starving the
delta-rule compute itself) or a global round trip + separate kernel - which
is precisely the shipped gated_norm_mq_rotate_gfx1100 fusion already in the
tree (3.4 us true, fully fused norm+silu-gate+rotate).

CAMPAIGN CLOSED. All launch-reduction candidates are now closed by
measurement or structure. Standing: tg128@64 = 208.65 tok/s (3-time
confirmed). Open items reduced to: the wo-shape rate anomaly (~90 us,
measured three ways, mechanism unexplained).

## Amendment 2i: per-group LDS dequant-LUT residual GEMV — bit-exact, slightly negative

The last untried algorithmic restructure: publish each group's 16 possible
dequantized values (sc*i+zp, i=0..15) to shared memory once per group, then
replace per-element extract+cvt+MAD chains with LDS lookups. Same expression,
same summation order -> BIT-EXACT (greedy text byte-identical x2 prompts).

Result: 148.4-148.9 vs OFF 150.0-150.2 tok/s (idle-gated interleaved pairs)
= ~-1%. The per-group __syncthreads pair plus LDS lookup latency exceeds the
ALU savings, independently confirming the ledger's finding that these
kernels are latency/occupancy-bound rather than VALU-throughput-bound.

Kept default-OFF (HIPFIRE_RESIDUAL_LUT=1) as a wired negative oracle. This
closes the last algorithmic restructure candidate on the residual family;
with amendments 2d-2i the campaign's falsification set is complete across
schedule, algorithmic, and dispatch dimensions.

## Amendment 2j: per-shape residual rates pinned (rocprof duration clustering)

Both shapes share one GPU symbol, so the event-instrumented aggregate never
separated them. rocprofv3 dispatch durations cluster bimodally:
- wo   (m2560,k4096): ~10.3 us -> ~540 GB/s
- w_down(m2560,k9216): ~17.7 us -> ~706 GB/s
The persist-r2 arm shows identical cluster means - confirming no schedule
recovery exists (wave-tail theory disproven; scheduler backfills blocks).
The residual deficit vs other MQ4 streams scales inversely with K length,
consistent with fixed per-launch ramp/drain amortized over fewer weight
bytes. At best this pool is ~90 us/token and no tested schedule reaches it.

## Amendment 2k: LDS-staged activation sharing — falsified, campaign search space closed

Tested the last untried restructure: stage the full activation vector in
shared memory once per block, share across ROWS_PER_BLOCK rows (r1/r2/r4/r8
variants, idle-gated interleaved arms on qwen3.5-4b shapes):

| variant | gen tok/s | delta vs OFF |
|---|---|---|
| OFF | 150.8 | — |
| r1 (x staged per row) | 126.2 | -16% |
| r2 | 119.8 | -20% |
| r4 | 111.1 | -26% |
| r8 | 94.8 | -37% |

Monotonically worse with rows-per-block: the cooperative x-stage pass plus
LDS round trip costs far more than the redundant global/L1 x re-reads it
replaces. This also retroactively explains the earlier multirow/pair2
negatives - activation "sharing" is counterproductive on this L1/L2 topology
for GEMV-shaped work at these sizes.

With this, EVERY launch-reduction and memory-schedule candidate identified
across three campaigns is closed by measurement or structure. Standing:
208.65 tok/s product bench (triple-confirmed). Realistic ceiling under the
objective's constraints: ~225-240. Target 250 requires speculative decoding
or weight-format change - both excluded.

## Amendment 3 (2026-08-23 night): CRITICAL — engaging the qkvza fold costs -26%, not +0.7%

After wiring the formerly-orphaned route (amendment 2d), controlled A/B on the
product path (hipfire bench tg128@64, runs=2 interleaved):
- HIPFIRE_QKVZA_FUSEDNORM=1 : 155.54 tok/s
- flag absent               : 209.70 tok/s
=> Engaging fused_qkvza_hfq4g256_fusednorm costs **-26% decode**, far worse
than the standalone-probe estimate. Addendum 3's "+0.7% certified" was
measured while the code was ORPHANED (flag inert) — i.e., the claim was
fictional; only the bit-exact numerics certification was real.

ACTIONS:
- Default remains OFF (explicit opt-in required). The flag MUST NOT be set on
  qwen3.5-4b or similar shapes.
- All local wrapper scripts cleaned of the stale flag.
- Lesson recorded: a "certified" e2e claim produced through an orphaned code
  path validates nothing about product behavior.

## Final clean standing (post-cleanup)

tg128@64 = **209.67 tok/s** median of 5 (pp64 3769.9 / pp2048 4392.7),
config: KV=q8, spec off, fa-kvwrite fold ON, NO residual/persist/LUT/XLDS/
QKVZA_FUSEDNORM/GATE_UP_FUSEDNORM flags. Best confirmed number of the
campaign (+40% over the 150-era baseline two days prior; +8% over the
branch's committed speed floor).

## Amendment 4 (2026-08-23 night): GDN block-shape screen — flat; systematic flag audit clean

Re-ran the GDN compact2 block-shape screen against the CLEAN baseline
(previous attempt was invalidated by an accidentally-engaged harmful flag).
Product path, idle-gated interleaved single arms:
- b2 (default): 208.37 | b4: 208.74 | b8: 209.40 | b12: 208.47 | b16: 208.99
All within ±0.3% - GDN block shape is not a lever on qwen3.5-4b decode.
Also verified HIPFIRE_AWQ_NORM_DIRECT is active-by-default at k=2560 and
HIPFIRE_DN_STATE_EF stays ON (required for deterministic state).

Systematic flag-audit status for the qwen3.5-4b decode path: every developer
flag affecting the 306-node decode graph has now been screened or verified:
- Beneficial & default-on: AWQ_NORM_DIRECT, graph replay, fa-kvwrite fold,
  qkvza fusednorm route (INERT without explicit opt-in), DPM warmup
- Neutral: wavegrid norm, kv-backend, GDN compact2 shapes
- Harmful if engaged: QKVZA fusednorm (-26%), residual persist/LUT/XLDS,
  gate_up PAIR2/STAGE_X32(-26%/-hangs), GEMV_ROWS multirow

Standing clean config confirmed: tg128@64 = 208.4-209.7 tok/s band.

## Amendment 5: remaining engineering items requiring dedicated effort

One hypothesis remains untested: MQ4 GEMVs may be bound by L2 bandwidth on
ACTIVATION re-reads rather than DRAM weight streams. Traffic analysis shows
activation reads (M x K x 4B per launch) exceed weight bytes on every shape
(wo: 42MB acts vs 5.6MB weights). If binding, halving activation bytes via
f16 staging recovers ~100-200 us/token (+4-8%).

Why not attempted: proper f16 staging requires (a) producers emitting f16
activations (numerics change - NOT bit-exact, full quality validation
needed), or (b) per-layer f32->f16 conversion launches (+64 launches/token,
self-defeating), plus (c) rocprofv3 PMC counters (L2/perfcounter metrics) to
confirm the L2-bound hypothesis before implementation. Estimated effort:
1-2 focused sessions; expected ceiling if successful: ~220-235 tok/s.

This documents the boundary of what turn-scale work can reach. All other
candidates are closed by measurement (amendments 2a-2k) or structure.
