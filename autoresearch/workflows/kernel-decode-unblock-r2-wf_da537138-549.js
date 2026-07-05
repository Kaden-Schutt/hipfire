export const meta = {
  name: 'kernel-decode-unblock-r2',
  description: 'Round 2: deepen/combine/rescue/gap-fill on round 1 kernel-decode levers for gfx11(primary)/gfx12, adversarially vetted, growing the certify backlog',
  phases: [
    { title: 'Expand', detail: '12 agents deepen R1 levers, combine them, fill gaps, add new mechanisms' },
    { title: 'Synthesize', detail: 'merge R1 survivors + R2 additions into one ranked pool' },
    { title: 'Verify', detail: 'adversarial refutation (physics/reachability/coherence)' },
    { title: 'Plan', detail: 'growing gfx11-first vetted experiment backlog' },
  ],
}

const CONTEXT = `
MISSION: RDNA a3b AR-decode (batch=1, single token) measures gfx11(gfx1100)=~164 tok/s, gfx12(gfx1201)=~155 tok/s, BOTH ~100% GPU-busy during the isolated decode window. This near-equality is PHYSICALLY WRONG and GUARANTEES recoverable headroom: gfx11 has ~1.5x the bandwidth, ~1.5x the CUs, ~1.5x the last-level cache of gfx12, so an efficient gfx11 kernel set should decode ~1.5x gfx12 (~230+ tok/s). Both converging = kernels latency-STALLED: resident waves stalled on memory latency with too few outstanding requests (low memory-level parallelism) to hide it. GPU 100% busy but only ~26% (gfx11) / ~36% (gfx12) of bandwidth ceiling.

GOAL: kernel-level (.hip source ONLY) levers that RAISE AR-decode throughput by increasing memory-level parallelism / hiding memory latency / exploiting each arch's SPECIFIC hardware. PRIMARY: gfx11 (most unused headroom). SECONDARY: gfx12 ceiling. Compute is NOT the bottleneck; do NOT propose compute reduction.

HARD CONSTRAINTS: pure .hip kernel-body edits (loop CANNOT change host dispatch/grid/launch config). Fair game inside the kernel: __launch_bounds__, LDS, VGPR footprint, load width, prefetch depth, software pipelining, reduction shape, register/LDS tiling, split-K, dequant path, intra-kernel fusion, non-temporal loads, global_load_lds. Preserve EXACT output math (coherence-gated; attractor/garbage disqualifies). batch=1; GEMV/weight-streaming dominated; mq4r 4-bit (nibble dequant on the read path).

gfx11=7900XTX RDNA3/Navi31: 96 CU, 6 SE, 2xSIMD32/CU, wave32, 32 waves/CU, 64KB LDS, 2526 MHz, 24GB, ~960 GB/s. 6MB L2 + 96MB Infinity Cache(MALL). 128B cacheline. VOPD dual-issue. HAS fp16/bf16/int8 WMMA but uses scalar MMQ (sdot4/dp4a); int8/iu8-WMMA LOST + coherence-falsified; fp16/bf16-WMMA UNEXPLORED. RDNA3 has global_load_lds (direct global->LDS, bypasses VGPR).
gfx12=R9700 RDNA4/Navi48: 64 CU, 4 SE, wave32, 32 waves/CU, 64KB LDS, ~2350+ MHz, 32GB, ~640 GB/s. L2 + ~64MB MALL. 256B cacheline. Native WMMA. FP8/FP4.

CENSUS (rocprofv3 = wall%/occ%/MemUnitBusy%/VGPR):
 gfx11: qkvza 11.7/32/44/72 | moe_gate_up_k8 9.8/50/60/96 | attention_flash_q8 8.2/0.5/54/64 | moe_down_k8 7.6/41/51/88 | rmsnorm 6.5/0.1/10/48 | residual 5.4/31/48/72 | gemm_grouped_mmq 5.0/47/82/128 | gemv 4.9/89/91/72(SATURATED-leave)
 gfx12: qkvza 14.9/35/41/72 | moe_gate_up_k8 12.8/52/54/96 | moe_down 8.0/42/44/88 | multirow_r2 7.3/89/93/96(SAT) | residual 6.8/23/46/72 | gemm_wmma 5.9/94/97/64(SAT) | attention 5.4/0.8/47/64
 READING: hot kernels are LOW occupancy (0.5-52%), memory only 44-60% busy = latency-bound / low MLP. attention 0.5-0.8% occ = catastrophic. Saturated kernels (occ~90/mem~92) show the target.

TRIED+FAILED (never repeat): 2/4-row x-reuse (won gfx12, REGRESSED gfx11 -2%/-1.5%, added VGPR+dropped occ); __launch_bounds__=24 NO-OP (GEMV work-limited not register-limited); attention V-unroll 8->6 LOST -2.77%; baseline traffic-cut banked (+15.9%); hipGraph decode LOST -5..-18%; int8/iu8-WMMA gfx11 LOST + coherence-falsified.
`

const R1SEED = `ROUND 1 RESULT (60 ideas->10 ranked->1 survived brutal refutation). Levers proposed (BUILD ON these: deepen into concrete variants, combine, rescue the weak, fill gaps - do NOT just restate):
- [gfx11] HFQ4-G256 GEMV group-loop deep prefetch (double-buffer the quad loop) :: the R1 SURVIVOR. Software-pipeline the 'for q<quads' loop: hoist next quad's 4 packed-nibble loads into a prefetch reg buffer BEFORE the current accumulate, rolling 2-stage window; raises outstanding vmem ~4->8; byte-identical math (only load TIMING moves). Covers ~34% of gfx11 wall.
- [gfx11] Non-temporal (cache-bypass) loads on stream-once MoE expert weights :: at batch-1 each routed expert row is read ONCE/token (no reuse) yet cache-allocating loads evict the reused x/KV; __builtin_nontemporal_load on the 4-bit payload keeps reusable data resident. Free (no VGPR). Composes with prefetch.
- [gfx11] Flash-attention Phase-A register-ring KV prefetch (deferred shuffle) :: attention 0.5% occ, load->use distance 1, in-loop __shfl_xor serializes positions. Strip-mine by P=4: preload P K-blocks to a register ring, P independent dots, THEN P deferred butterflies; outstanding KV 1->4. VGPR free at floor occ.
- [gfx11] rmsnorm/fused-rotate float4 streaming + prefetch :: rmsnorm 0.1% occ, 10% mem_busy, 6.5% wall = pure latency stall. float4 (128b) loads over K/4 + 2-4 deep prefetch; preserve SoS reduction order for bit-parity.
- [gfx12] deep prefetch (same MLP diagnosis) + [gfx12] extend proven R=2 multirow to qkvza+gate_up (256B line half-filled by one row's 128B nibble payload) + [gfx12] NT loads (sharper, 64MB MALL) + [gfx12] attention ring.
Refutation was BRUTAL (9/10 killed) mostly on: 'LLVM may already hoist the loads' (prefetch), reuse-tradeoff uncertainty (NT), and reduction-reorder coherence risk (rings/trees). RESCUE means: address those exact objections with a concrete defense or a bit-parity-preserving formulation.`

const LENSES = [
  {key:'deepen-prefetch', t:'DEEPEN the R1 survivor (deep prefetch). Produce 2-3 CONCRETE implementation variants: 2-stage vs 3-stage vs 4-stage rolling window; register-buffer vs RDNA3 global_load_lds (direct global->LDS, frees VGPR); exact application to each of qkvza/gate_up/down/residual. CRITICALLY address the refutation objection "LLVM already hoists": how to FORCE deeper hoisting (manual double-buffer, s_waitcnt vmcnt hints, restrict/__builtin_amdgcn) and how to VERIFY via ISA that vmcnt depth actually increased. Keep math byte-identical.'},
  {key:'deepen-nt', t:'DEEPEN non-temporal loads. Concrete variants (which exact loads get NT: weight payload yes, x/scale/KV no), opt-in macro, and DESIGN the A/B that measures the reuse tradeoff (does routing correlation give cross-token expert reuse?). Add the combine with prefetch. Rescue the "reuse uncertainty" objection.'},
  {key:'deepen-attn', t:'DEEPEN the attention register-ring. Concrete P-depth choices (P=2/4/8) vs VGPR budget at floor occupancy; the exact deferred-shuffle formulation that PROVES bit-parity (preserve each position 16/8/4/2/1 order); the __launch_bounds__(32,16)->(32,8) relax and spill check. Rescue the reduction-reorder coherence objection with a provably-identical formulation.'},
  {key:'deepen-tiny', t:'DEEPEN the rmsnorm float4+prefetch to ALL the tiny grid-floored latency-bound kernels: rmsnorm, residual, topk_renorm, any norm/rotate. float4 loads + prefetch + preserved reduction order. These are ~0.1-0.5% occ = pure exposed latency; enumerate every such kernel and the concrete vectorized+prefetched form.'},
  {key:'combine-gemv', t:'COMBINE deep-prefetch + non-temporal + float4-wide + dequant-fusion into ONE super-GEMV kernel body (gate_up/down/qkvza). Do they compose or conflict (VGPR pressure, vmcnt interplay)? Propose the single best-combined kernel and the order to stack the levers. This is where non-linear wins hide.'},
  {key:'combine-attn', t:'COMBINE attention ring-prefetch + float4 KV loads + the sliding-window -INF skip + deferred reduction into one attention super-kernel. Handle the interaction of prefetch depth with the window skip (do not prefetch skipped positions). Byte-identical.'},
  {key:'gap-critic', t:'COMPLETENESS CRITIC. R1 used 12 lenses (MLP, pipeline, loadwidth, MALL, split-K, ILP/VOPD, dequant, fusion, MoE, attention, WMMA, cross-arch) + the levers above. What MECHANISM is STILL missing? Name 3-5 genuinely uncovered kernel-level latency-hiding / MLP / bandwidth-realization levers nobody proposed yet. Think: address-generation overlap, LDS as a software-managed prefetch stage, wave specialization, double-issue of vmem+ds, KV-cache layout for coalescing.'},
  {key:'quant-unpack', t:'mq4 DEQUANT-PATH co-design (kernel-side). Faster/wider nibble extraction: v_perm_b32 / byte-permute LUT, dual-nibble unpack of a whole dwordx4 at once, hoist+broadcast group scales, minimize the dequant instruction chain so it does NOT serialize load->use. Is nibble-extract currently on the critical path between the vmem load and the FMA? Concrete reformulation.'},
  {key:'isa-mem', t:'RDNA3/4 MEMORY-ISA lever. Use global_load_lds (RDNA3 direct global->LDS, async, bypasses VGPR so it both prefetches AND frees registers), buffer_load with soffset for the weight stream, explicit s_waitcnt vmcnt scheduling to widen the outstanding-request window, ds_read for the LDS stage. Which hot kernel benefits most from global_load_lds as a free async-prefetch + register-relief lever?'},
  {key:'splitk-work', t:'CREATE PARALLEL WORK (the real answer to the launch_bounds=24 no-op: you cannot schedule waves that do not exist). Split the K-reduction of the batch-1 GEMV across MORE waves/blocks with a coherence-SAFE combine (LDS tree in fixed order, or a deterministic 2-pass). This RAISES resident waves -> more MLP. Give the exact split factor + the bit-parity-preserving combine. This is distinct from row-reuse (which shrank parallelism).'},
  {key:'rescue', t:'RESCUE the refuted. R1 killed 9/10 candidates. Take the strongest REFUTED mechanisms and either (a) reformulate them to survive the exact objection (reduction-reorder->fixed-order combine; prefetch-already-hoisted->forced manual buffer + ISA proof; NT-reuse->measured A/B), or (b) confirm each truly dead with a physical reason. Output only defensible rescues.'},
  {key:'gfx11-diverge', t:'gfx11 PRIMARY divergence. gfx11 is 26% BW-efficient vs gfx12 36% -> it wastes MORE of its silicon. Propose concrete gfx11.hip forks that each capture ONE wasted advantage: 96MB MALL residency (prefetch depth + NT tuned to 96MB working set), +320 GB/s (deeper MLP than gfx12 needs), VOPD dual-issue (independent accumulator pairs), 96 vs 64 CUs. Quantify which advantage each lever unlocks and name the single highest-conviction gfx11 win.'},
]

const IDEA_SCHEMA = {type:'object', additionalProperties:false, required:['ideas'], properties:{ ideas:{type:'array', items:{type:'object', additionalProperties:false, required:['name','kernel','arch','mechanism','why_helps','novelty','coherence_risk','confidence'], properties:{ name:{type:'string'}, kernel:{type:'string'}, arch:{type:'string', enum:['gfx11','gfx12','both']}, mechanism:{type:'string'}, why_helps:{type:'string'}, novelty:{type:'string'}, coherence_risk:{type:'string'}, confidence:{type:'number'} }}}}}
const SYNTH_SCHEMA = {type:'object', additionalProperties:false, required:['candidates'], properties:{ candidates:{type:'array', items:{type:'object', additionalProperties:false, required:['name','kernels','arch_focus','mechanism','rationale','priority'], properties:{ name:{type:'string'}, kernels:{type:'string'}, arch_focus:{type:'string'}, mechanism:{type:'string'}, rationale:{type:'string'}, priority:{type:'number'} }}}}}
const VERDICT_SCHEMA = {type:'object', additionalProperties:false, required:['verdict','reasoning'], properties:{ verdict:{type:'string', enum:['PROMISING','WEAK','REFUTED']}, physically_plausible:{type:'boolean'}, reachable:{type:'boolean'}, coherence_safe:{type:'boolean'}, reasoning:{type:'string'} }}
const PLAN_ITEM = {type:'object', additionalProperties:false, required:['name','kernel','change','mechanism','expected','coherence_check','ready','rank'], properties:{ name:{type:'string'}, kernel:{type:'string'}, change:{type:'string'}, mechanism:{type:'string'}, expected:{type:'string'}, coherence_check:{type:'string'}, ready:{type:'string'}, rank:{type:'number'} }}
const PLAN_SCHEMA = {type:'object', additionalProperties:false, required:['gfx11_experiments','gfx12_experiments','summary'], properties:{ gfx11_experiments:{type:'array', items:PLAN_ITEM}, gfx12_experiments:{type:'array', items:PLAN_ITEM}, summary:{type:'string'} }}

phase('Expand')
log(`round 2: ${LENSES.length} agents deepening/combining/rescuing R1`)
const ideaSets = await parallel(LENSES.map(L => () =>
  agent(`${CONTEXT}\n\n${R1SEED}\n\nYOUR ROUND-2 TASK: ${L.t}\n\nProduce 3-5 CONCRETE levers for this task. Each: exact kernel, exact code-level mechanism, why it raises MLP/hides latency/uses the wasted arch advantage, novelty (vs R1 + tried-failed), coherence risk. gfx11 first. Be implementation-specific.`,
    {schema: IDEA_SCHEMA, phase:'Expand', label:`r2:${L.key}`})))
const allIdeas = ideaSets.filter(Boolean).flatMap(r => r.ideas || [])
log(`round 2 collected ${allIdeas.length} levers`)

phase('Synthesize')
const synth = await agent(`${CONTEXT}\n\n${R1SEED}\n\nROUND-2 levers (${allIdeas.length}):\n${JSON.stringify(allIdeas)}\n\nMerge these with R1's defensible levers into ONE ranked pool. DEDUPE, CLUSTER by mechanism, RANK by (raises BW-eff/hides latency for a batch-1 latency-bound GEMV) x (.hip-reachable) x (novelty) x (coherence-safe), biased to gfx11. Prefer concrete/combined/rescued levers over vague ones. Return top ~14 candidates.`,
  {schema: SYNTH_SCHEMA, phase:'Synthesize', effort:'high', label:'r2-synth'})
const cands = (synth.candidates || []).sort((a,b)=>(b.priority||0)-(a.priority||0)).slice(0,12)
log(`verifying top ${cands.length}`)

phase('Verify')
const VLENS = [
  {k:'physics', p:'PHYSICS: given the roofline (100% busy, 26-36% BW-eff, low occ, mem 44-60%), does this ACTUALLY raise MLP / realize bandwidth / hide latency by >3%, or is it hand-wavy? Default REFUTED if the mechanism does not clearly attack the latency-stall.'},
  {k:'reachability', p:'REACHABILITY: pure .hip kernel-body change (no host/grid change)? Not a duplicate of a tried-failed lever (row-reuse, launch_bounds occ, unroll, iu8-WMMA, traffic-cut, hipGraph)? Default REFUTED otherwise.'},
  {k:'coherence', p:'COHERENCE: preserves EXACT output math? Split-K/atomics/reordered reductions/WMMA/NT can change numerics -> attractor risk. Default REFUTED if numerics likely change enough to risk garbage.'},
]
const verified = await parallel(cands.map(c => () =>
  parallel(VLENS.map(vl => () =>
    agent(`${CONTEXT}\n\nAdversarially REFUTE this candidate. ${vl.p}\n\nCANDIDATE:\n${JSON.stringify(c)}`,
      {schema: VERDICT_SCHEMA, phase:'Verify', label:`v:${vl.k}:${(c.name||'').slice(0,16)}`})))
    .then(vs => ({candidate:c, verdicts:vs.filter(Boolean)}))))
const survivors = verified.filter(v => v && v.verdicts.filter(x=>x.verdict==='PROMISING').length >= 2)
log(`${survivors.length}/${cands.length} survived`)

phase('Plan')
const plan = await agent(`${CONTEXT}\n\n${R1SEED}\n\nR2 candidates that survived adversarial refutation (>=2/3 PROMISING):\n${JSON.stringify(survivors)}\n\nProduce the GROWING vetted experiment backlog: merge R1's strongest levers (esp. the deep-prefetch survivor) with these R2 survivors, deduped, ranked, gfx11_experiments (PRIMARY) and gfx12_experiments. For each: exact kernel, concrete .hip change, mechanism (which stall/wasted-advantage), expected direction+magnitude, coherence_check, and ready = "certify-now" (byte-identical, high conviction) or "needs-coherence-proof" or "needs-isa-check". rank 1 = highest. summary: name the 3 certify-now gfx11 levers to fire FIRST and restate why the 164:155 ratio makes the headroom physically guaranteed.`,
  {schema: PLAN_SCHEMA, phase:'Plan', effort:'high', label:'r2-plan'})
return { plan, stats:{ r2_ideas: allIdeas.length, ranked: cands.length, survived: survivors.length } }
