export const meta = {
  name: 'kernel-decode-unblock-r3',
  description: 'Round 3: HARDEN the R1+R2 gfx11/gfx12 decode-lever pool into an implementation-ready, pre-checked, correctly-measured certify queue (red-team + impl-specs + gap sweep + measurement protocol)',
  phases: [
    { title: 'Harden', detail: '12 agents red-team on-device reality, write impl specs, final gap, measurement protocol' },
    { title: 'Synthesize', detail: 'merge into one definitive ordered queue' },
    { title: 'Verify', detail: 'final adversarial pass on each queue item' },
    { title: 'Queue', detail: 'implementation-ready, pre-checked, ordered certify backlog' },
  ],
}

const CONTEXT = `
MISSION: RDNA a3b AR-decode (batch=1) = gfx11(gfx1100) ~164 tok/s, gfx12(gfx1201) ~155, BOTH ~100% GPU-busy in the isolated decode window. Near-equality is PHYSICALLY WRONG -> GUARANTEED headroom: gfx11 has ~1.5x BW/CUs/cache of gfx12 so an efficient gfx11 should decode ~1.5x (~230+). Both converging = latency-STALLED kernels (resident waves stalled on memory latency, too few outstanding requests = low MLP). 100% busy but only ~26% (gfx11)/~36% (gfx12) of BW ceiling.
GOAL: pure .hip kernel-body levers that raise memory-level parallelism / hide latency / use each arch's specific hardware. gfx11 PRIMARY. Compute is NOT the bottleneck. Preserve EXACT output math (coherence-gated). mq4r 4-bit (nibble dequant on read path); GEMV/weight-streaming dominated.
gfx11=7900XTX RDNA3: 96CU/6SE/wave32/32-waves-CU/64KB-LDS/2526MHz/960GB-s. 6MB L2 + 96MB MALL. 128B cacheline. VOPD dual-issue. fp16/bf16/int8 WMMA present but uses scalar MMQ (int8-WMMA LOST+coherence-falsified; fp16-WMMA unexplored). Has global_load_lds (direct global->LDS).
gfx12=R9700 RDNA4: 64CU/4SE/wave32/32-waves-CU/64KB-LDS/~2350MHz/640GB-s. ~64MB MALL. 256B cacheline. Native WMMA. FP8/FP4.
CENSUS (wall%/occ%/mem_busy%/VGPR): gfx11 qkvza 11.7/32/44/72 | gate_up_k8 9.8/50/60/96 | attention 8.2/0.5/54/64 | down_k8 7.6/41/51/88 | rmsnorm 6.5/0.1/10/48 | residual 5.4/31/48/72 | gemm_mmq 5.0/47/82/128 | gemv 4.9/89/91(SAT). Most hot kernels LOW-occ + mem 44-60% = latency-bound; attention 0.5% occ catastrophic.
TRIED+FAILED (never repeat): row-reuse (won gfx12, REGRESSED gfx11 -2%/-1.5%, +VGPR -occ); __launch_bounds__=24 NO-OP (work-limited); V-unroll 8->6 LOST -2.77%; traffic-cut banked +15.9%; hipGraph LOST; int8-WMMA gfx11 LOST+falsified.
MEASUREMENT REALITY (this session's hard-won traps -- the certify protocol MUST avoid them): decode is GPU-bound (100% busy) so wall decode_tok_s is a valid reference IF you isolate the decode window (dwall=tokens/decode_tok_s, sample the LAST dwall seconds -- a whole-run busy median is 35% garbage from load/gaps). kernel_decode_tok_s is None on the AR path -> certify falls back to wall decode_tok_s (fine). gfx11 has a ~30% VOID rate from DPM clock bounce (0<->1295MHz) -> use adaptive resampling, byte-identical prompt (record md5), fresh process per measure. Coherence gate mandatory on any numeric change (softmax amplifies 5% score drift into an attractor in ~10 tokens).
`

const POOL = `ACCUMULATED VETTED POOL (R1+R2, 60+60 ideas -> harshly refuted -> this survived). HARDEN each into certify-ready form; do NOT invent from scratch unless FINAL-GAP.
gfx11 backlog with readiness:
1.[needs-coherence-proof] Attention Phase-A KV register-ring prefetch P=4 :: attention_flash_q8_0_tile.hip Phase A(93-115) :: strip-mine Q.K by P=4 tokens, preload P K-blocks to a reg ring, P independent dots then P deferred __shfl_xor, per-token order preserved; outstanding KV 1->4 :: +3-5% (BIGGEST; but attention is 8.2% wall so cap ~+4%)
2.[certify-now] MoE-down branch-free tail restructure :: gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip(106-132) :: a3b down K=512->quads=0 so kernel IS the guarded tail (load g0->compute->load g1->compute, MLP=1); stage ALL guarded loads then ALL computes, MLP 1->2/3; keep in-bounds address guards :: +1-2% byte-identical
3.[certify-now] rmsnorm Phase-1a gather deep prefetch :: fused_rmsnorm_mq_rotate.hip(65-72) :: decode=ONE 8-wave workgroup on 96 CUs (occ 0.1% STRUCTURAL); prefetch the ascending-i SoS gather 1->3 outstanding; NOT float4 (that reorders the reduction) :: +1.5-2.5% byte-identical
4.[needs-coherence-proof] Attention Phase-D V register-ring prefetch P=4 :: attention Phase D(160-180) :: preload P tokens' V to reg ring, accumulate out0..3 in token order; LOAD-timing not unroll-width (unlike the failed 8->6) :: +1.5-3% additive with #1
5.[needs-isa-check] qkvza quad-loop cross-iteration pk prefetch :: fused_qkvza_hfq4g256.hip(86-124) :: K=2048->quads=2; hoist q+1's 4 pk loads above q's accumulate, MLP 4->8, +4 VGPR (72->76 ok); RISK LLVM may already unroll the 2-trip loop -> no-op :: +2-3%
6.[needs-isa-check] gate_up quad-loop cross-iteration pk prefetch :: gemv_hfq4g256_moe_gate_up_indexed_batched.hip(50-88) :: VGPR=96 == exact 16-wave budget; pk-only prefetch may cross the wave32 VGPR granularity boundary and DROP a wave = the exact -2% row-reuse failure mode :: +1-2% or REGRESSION
7.[certify-now] NT (non-temporal) loads on stream-once expert weights :: gate_up + down pk/sc/zp :: __builtin_nontemporal_load keeps reused x/KV MALL-resident; sign uncertain (win if eviction real) :: +/-1-2%
8.[certify-now] residual-GEMV tail restructure / cross-iter prefetch :: residual kernels :: +0.5-1.5%
9.[needs-isa-check] global_load_lds direct expert-weight streaming (RDNA3) :: async global->LDS frees VGPR + prefetches; likely no-op/small loss unless VGPR headroom helps occ :: uncertain
PLUS R1's deep-prefetch survivor (double-buffer the quad loop) for kernels WITH quads>0.
gfx12 backlog: deep-prefetch; R=2 multirow (256B line half-filled -> fill it, gfx12-only); NT (sharper, 64MB MALL); attention ring.`

const LENSES = [
  {key:'redteam-isa', t:'RED-TEAM the needs-isa-check levers (qkvza #5, gate_up #6, global_load_lds #9). PREDICT the on-device ISA reality: will LLVM already unroll+pre-hoist the 2-trip quad loop (making #5/#6 no-ops)? Will gate_up VGPR 96->100 cross the wave32 granularity and DROP occupancy 16->15 (regression, the row-reuse failure mode)? For EACH, give the exact PRE-CHECK to run on the .hsaco BEFORE any GPU certify (gfx-kernel-metadata: s_waitcnt vmcnt depth, VGPR count, resident-wave count) and the GO/NO-GO threshold. The point: never spend a GPU certify on a lever the ISA already says is dead.'},
  {key:'redteam-coherence', t:'RED-TEAM the needs-coherence-proof levers (attention Phase-A ring #1, Phase-D ring #4). Does the P=4 strip-mine PROVABLY preserve exact FP order (per-token partials, deferred shuffles in identical 16/8/4/2/1 order)? Where could a reorder sneak in? Give the exact bit-parity test (FP32 + HIPFIRE_DETERMINISTIC=1 vs baseline) + the coherence-gate + attractor-probe protocol to certify order preservation. If any formulation cannot be made bit-exact, say so.'},
  {key:'redteam-certifynow', t:'STRESS-TEST the "certify-now byte-identical" claims (#2 moe_down tail, #3 rmsnorm gather, #7 NT, #8 residual). Is each TRULY byte-identical, or is there a hidden reorder / aliasing / no-op reason? NT loads change cache policy not values -> confirm truly value-identical. Confirm moe_down guards stay in-bounds. Flag any that are NOT actually safe.'},
  {key:'impl-moedown', t:'IMPLEMENTATION SPEC for #2 (moe_down branch-free tail restructure) -- the top certify-now. Write the concrete before/after kernel-body structure (the guarded load-then-compute split), the exact staging registers, the in-bounds address predicate, and confirm the accumulator mapping is unchanged. Make it directly buildable.'},
  {key:'impl-rmsnorm', t:'IMPLEMENTATION SPEC for #3 (rmsnorm Phase-1a gather prefetch). Concrete before/after: the ascending-i prefetch ring over the strided gather, preserving local_sum accumulation order, writing x_shared identically, FWHT untouched. Directly buildable.'},
  {key:'impl-attn-ring', t:'IMPLEMENTATION SPEC for #1 (attention Phase-A KV ring, the biggest lever). Concrete before/after: the P=4 strip-mine, the register ring layout, the per-token partial accumulation, the deferred per-token shuffle reduction preserving order, the window-mask skip interaction. The order-preserving formulation must be explicit.'},
  {key:'impl-residual-nt', t:'IMPLEMENTATION SPEC for #8 (residual tail restructure) and #7 (NT loads): concrete kernel-body changes + the opt-in macro for NT + the A/B that measures the NT reuse tradeoff (both directions).'},
  {key:'stack', t:'STACKING PLAN: the certify-now byte-identical levers (#2 moe_down, #3 rmsnorm, #8 residual) touch DISJOINT kernels -> do they compose additively into one combined patch worth ~+3-5% with zero coherence risk? Confirm disjointness, the combined expected delta, and the order to land them. Also: which levers CONFLICT (share a kernel/VGPR budget) and must be A/Bd separately.'},
  {key:'final-gap', t:'FINAL COMPLETENESS SWEEP. After R1+R2, what hot-kernel slice or MLP mechanism is STILL uncovered? Specifically: gemm_grouped_mmq (5.0% wall, occ47/mem82 -- closest to saturated, is it addressable?), KV-cache memory LAYOUT for coalescing, address-generation overlap, LDS as a software-managed multi-stage prefetch buffer, wave-specialization (producer/consumer waves in one block). Name only genuinely-new, kernel-reachable levers.'},
  {key:'gfx12-harden', t:'HARDEN the gfx12 backlog (deep-prefetch, R=2 multirow to fill the 256B line, NT, attention ring). For each: the concrete gfx12-build change, the pre-check, and the gfx12-specific expected delta. Note: R=2 multirow WON on gfx12 already (baseline) -- extending it to qkvza/gate_up is the sharpest gfx12 lever; detail it.'},
  {key:'measure-protocol', t:'CERTIFY PROTOCOL design (avoid THIS session measurement traps). For the pool, specify the exact certify recipe: isolate-decode-window measurement, adaptive resampling to beat the ~30% gfx11 VOID/DPM clock bounce, byte-identical prompt+md5, fresh process per measure, the ISA pre-check gate (skip GPU if .hsaco says no-op/regression), the coherence gate + attractor probe for numeric changes, and the FLOOR delta (what counts as a real win vs noise). This is how we avoid a bajillion loser runs.'},
  {key:'sanity-delta', t:'SANITY-CHECK every expected-delta claim against Amdahl + wall%. attention ring claims +3-5% but attention is only 8.2% of wall (a 1.5x kernel speedup caps at ~+4%). qkvza is 11.7% wall. Re-derive a REALISTIC per-lever and total-stacked delta ceiling for gfx11, and rank the pool by (realistic delta) x (probability it is not a no-op/regression) x (coherence safety). Flag any over-claimed levers.'},
]

const IDEA_SCHEMA = {type:'object', additionalProperties:false, required:['findings'], properties:{ findings:{type:'array', items:{type:'object', additionalProperties:false, required:['lever','verdict_or_spec','detail','impact'], properties:{ lever:{type:'string'}, verdict_or_spec:{type:'string'}, detail:{type:'string'}, impact:{type:'string'} }}}}}
const SYNTH_SCHEMA = {type:'object', additionalProperties:false, required:['queue'], properties:{ queue:{type:'array', items:{type:'object', additionalProperties:false, required:['name','kernel','arch','ready','realistic_delta','priority'], properties:{ name:{type:'string'}, kernel:{type:'string'}, arch:{type:'string'}, ready:{type:'string'}, realistic_delta:{type:'string'}, priority:{type:'number'} }}}}}
const VERDICT_SCHEMA = {type:'object', additionalProperties:false, required:['verdict','reasoning'], properties:{ verdict:{type:'string', enum:['SHIP','HOLD','CUT']}, reasoning:{type:'string'} }}
const Q_ITEM = {type:'object', additionalProperties:false, required:['order','name','kernel','arch','change','precheck','coherence_check','certify_protocol','realistic_delta','kill_condition','ready'], properties:{ order:{type:'number'}, name:{type:'string'}, kernel:{type:'string'}, arch:{type:'string'}, change:{type:'string'}, precheck:{type:'string'}, coherence_check:{type:'string'}, certify_protocol:{type:'string'}, realistic_delta:{type:'string'}, kill_condition:{type:'string'}, ready:{type:'string'} }}
const QUEUE_SCHEMA = {type:'object', additionalProperties:false, required:['queue','stacked_gfx11_ceiling','summary'], properties:{ queue:{type:'array', items:Q_ITEM}, stacked_gfx11_ceiling:{type:'string'}, summary:{type:'string'} }}

phase('Harden')
log(`round 3: ${LENSES.length} agents hardening the R1+R2 pool into a certify queue`)
const sets = await parallel(LENSES.map(L => () =>
  agent(`${CONTEXT}\n\n${POOL}\n\nYOUR ROUND-3 TASK: ${L.t}\n\nReturn concrete findings (verdicts / implementation specs / protocols / gap-levers as the task dictates). Be specific and buildable; gfx11 first.`,
    {schema: IDEA_SCHEMA, phase:'Harden', label:`r3:${L.key}`})))
const findings = sets.filter(Boolean).flatMap(r => r.findings || [])
log(`round 3 collected ${findings.length} hardening findings`)

phase('Synthesize')
const synth = await agent(`${CONTEXT}\n\n${POOL}\n\nROUND-3 hardening findings (${findings.length}):\n${JSON.stringify(findings)}\n\nMerge into ONE definitive ordered certify queue. Fold in the red-team verdicts (drop levers the ISA/coherence red-team predicts will no-op/regress/break), the implementation specs, the stacking plan, the final-gap additions, and the sanity-checked realistic deltas. Order: certify-now byte-identical stackable FIRST, then needs-isa-check WITH their pre-check, then needs-coherence-proof WITH their gate. priority = higher first. Return the ranked queue.`,
  {schema: SYNTH_SCHEMA, phase:'Synthesize', effort:'high', label:'r3-synth'})
const q = (synth.queue || []).sort((a,b)=>(b.priority||0)-(a.priority||0)).slice(0,14)
log(`final adversarial pass on ${q.length} queue items`)

phase('Verify')
const verified = await parallel(q.map(c => () =>
  agent(`${CONTEXT}\n\nFINAL SHIP/HOLD/CUT decision on this certify-queue item. SHIP = high-conviction, pre-check defined, worth a GPU certify. HOLD = worth it but needs the named pre-check to pass first. CUT = likely no-op/regression/coherence-break, do NOT waste a certify. Be strict -- the whole point is to avoid loser runs.\n\nITEM:\n${JSON.stringify(c)}`,
    {schema: VERDICT_SCHEMA, phase:'Verify', label:`ship:${(c.name||'').slice(0,18)}`}).then(v => ({item:c, v}))))
const kept = verified.filter(x => x && x.v && x.v.verdict !== 'CUT')
log(`${kept.length}/${q.length} kept (not CUT)`)

phase('Queue')
const finalq = await agent(`${CONTEXT}\n\n${POOL}\n\nKept certify-queue items with SHIP/HOLD verdicts:\n${JSON.stringify(kept)}\n\nProduce the FINAL implementation-ready certify backlog. For each item: order (1=first), name, kernel, concrete change, precheck (the ISA/.hsaco or coherence check to run BEFORE the GPU certify, or "none" for byte-identical), coherence_check, certify_protocol (isolate-decode-window + adaptive resample + prompt md5 + floor delta), realistic_delta (Amdahl/wall%-honest), kill_condition (what result means abandon it), ready. Order the certify-now byte-identical stackable levers first. Give stacked_gfx11_ceiling = the honest total achievable gfx11 delta if the top levers all land. summary = the 3 to fire first and why the 164:155 physics guarantees the headroom is real.`,
  {schema: QUEUE_SCHEMA, phase:'Queue', effort:'high', label:'r3-queue'})
return { queue: finalq, stats:{ r3_findings: findings.length, ranked: q.length, kept: kept.length } }
