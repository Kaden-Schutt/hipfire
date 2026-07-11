export const meta = {
  name: 'kernel-decode-unblock',
  description: 'Swarm-brainstorm + adversarially verify CREATIVE kernel-level levers to unblock gfx11 (primary) and raise gfx12 a3b AR-decode throughput, given the physically-impossible 164:155 ratio',
  phases: [
    { title: 'Brainstorm', detail: '12 lens-agents generate novel MLP/latency-hiding/cache levers' },
    { title: 'Synthesize', detail: 'dedupe + cluster + rank into top candidates (gfx11-first)' },
    { title: 'Verify', detail: 'adversarial refutation of each candidate (physics/reachability/coherence)' },
    { title: 'Plan', detail: 'concrete ranked experiment plan' },
  ],
}

const CONTEXT = `
MISSION: RDNA a3b AR-decode (autoregressive, batch=1, single token) measures gfx11(gfx1100)=~164 tok/s and gfx12(gfx1201)=~155 tok/s, BOTH ~100% GPU-busy during the isolated decode window. This near-equality is PHYSICALLY WRONG and signals a real, capturable inefficiency -- it is NOT a ceiling. gfx11 has ~1.5x the bandwidth, ~1.5x the CUs, and ~1.5x the last-level cache of gfx12, so an efficient gfx11 kernel set should decode a3b at ~1.5x gfx12 (~230+ tok/s). Both chips converging means the kernels are latency-STALLED: waves are resident but stalled on memory latency with too few outstanding requests (low memory-level parallelism) to hide it. GPU is 100% busy but only ~26% (gfx11) / ~36% (gfx12) of the bandwidth ceiling.

GOAL: brainstorm CREATIVE, NOVEL, kernel-level (.hip source ONLY) levers that RAISE AR-decode throughput by increasing memory-level parallelism / hiding memory latency / exploiting each arch's SPECIFIC hardware advantages. PRIMARY: unblock gfx11 (most unused headroom). SECONDARY: raise the gfx12 ceiling. Compute is NOT the bottleneck (batch-1 GEMV is tiny) -- do NOT propose compute reduction; propose latency-hiding / MLP / bandwidth-realization / cache exploitation.

HARD CONSTRAINTS:
- Pure .hip kernel-body edits. The loop CANNOT change host dispatch / grid dims / launch config (Rust-side). Fair game INSIDE the kernel: __launch_bounds__, LDS use, VGPR footprint, load width, prefetch depth, software pipelining, reduction shape, register/LDS tiling, split-K, dequant path, intra-kernel fusion.
- Preserve EXACT output math (identical results). Coherence is gated; an attractor/garbage output disqualifies a lever.
- batch=1 single-token decode; GEMV/weight-streaming dominated; mq4r 4-bit quant (nibble dequant on the read path).

gfx11 = Radeon 7900XTX, RDNA3/Navi31: 96 CU, 6 Shader Engines, 2x SIMD32/CU, wave32, max 32 waves/CU, 64KB LDS/workgroup, 2526 MHz, 24GB GDDR6 ~960 GB/s. Cache: 6MB L2 + 96MB Infinity Cache (MALL). 128-byte cacheline. VOPD dual-issue (2 independent FP32 ops/cycle). HAS WMMA (fp16/bf16/int8) but hipfire uses SCALAR MMQ (sdot4/dp4a) for the HFQ4 grouped GEMM -- an int8/iu8-WMMA attempt LOST 5-12% + was coherence-falsified; the fp16/bf16-WMMA path is UNEXPLORED.

gfx12 = Radeon AI PRO R9700, RDNA4/Navi48: 64 CU, 4 Shader Engines, 2x SIMD32/CU, wave32, max 32 waves/CU, 64KB LDS, ~2350+ MHz, 32GB GDDR6 ~640 GB/s. Cache: L2 + ~64MB MALL. 256-byte cacheline. Native WMMA. FP8/FP4-capable.

CENSUS (rocprofv3; per hot decode kernel = wall% / occupancy% / MemUnitBusy% / VGPR):
 gfx11: qkvza 11.7/32/44/72 | moe_gate_up_k8 9.8/50/60/96 | attention_flash_q8 8.2/0.5/54/64 | moe_down_k8 7.6/41/51/88 | rmsnorm 6.5/0.1/10/48 | residual 5.4/31/48/72 | gemm_grouped_mmq 5.0/47/82/128 | gemv 4.9/89/91/72(SATURATED-leave)
 gfx12: qkvza 14.9/35/41/72 | moe_gate_up_k8 12.8/52/54/96 | moe_down 8.0/42/44/88 | multirow_r2 7.3/89/93/96(SATURATED) | residual 6.8/23/46/72 | gemm_wmma 5.9/94/97/64(SATURATED) | attention 5.4/0.8/47/64
 READING: most hot kernels are LOW occupancy (0.5-52%) with memory only 44-60% busy = NOT bandwidth-saturated = latency-bound / insufficient MLP. attention at occ 0.5-0.8% is catastrophically under-occupied. The kernels that ARE saturated (occ ~90%, mem ~92%) show what "good" looks like -- match that.

ALREADY TRIED AND FAILED (do NOT repeat):
- 2-row/4-row weight-row reuse (x-reuse): won on gfx12 but on gfx11 REGRESSED (-2% qkvza, -1.5% gate_up) -- added VGPR, dropped occupancy.
- occupancy via __launch_bounds__=24: NO-OP (+0.07%) -- GEMV is WORK-limited not register-limited; nothing more to schedule at batch-1 unless you CREATE parallel work.
- attention V-unroll 8->6: LOST -2.77%.
- baseline memory-traffic reduction: already applied (+15.9%, banked).
- hipGraph decode capture: LOST -5..-18%.
- int8/iu8-WMMA on gfx11 grouped GEMM: LOST 5-12% + coherence-falsified.

Be creative and SPECIFIC: name the exact kernel, the exact mechanism, and WHY it hides latency / realizes bandwidth / uses the arch advantage the current impl wastes.
`

const LENSES = [
  {key:'mlp', name:'Memory-Level Parallelism', desc:'Issue many INDEPENDENT outstanding global loads before consuming any (deepen load->use distance) so more memory requests are in flight per wave to hide latency. Unroll K with independent load chains; prefetch several weight tiles ahead into registers; raise outstanding-request depth. Core lever for a low-occupancy latency-bound GEMV.'},
  {key:'pipeline', name:'Software Pipelining / Multi-buffering', desc:'Double/triple-buffer weight tiles (registers or LDS) so load(tile N+1) overlaps dequant+FMA(tile N). Explicit prefetch-ahead loop; break the load->dequant->accumulate dependency chain so memory and ALU overlap.'},
  {key:'loadwidth', name:'Load Width & Transaction Efficiency', desc:'Widest vector loads (global_load_dwordx4/x8, 128-bit+), align weight/x access to the cacheline (128B gfx11 / 256B gfx12), buffer_load vs global_load, full-wave coalescing so each transaction moves a full line. mq4r nibble layout may fragment loads.'},
  {key:'mall', name:'Infinity-Cache (MALL) Exploitation', desc:'gfx11 has 96MB MALL (vs gfx12 64MB) that gfx12-tuned kernels ignore. Structure weight streaming + x reuse so working sets stay MALL-resident; access patterns that engage the HW prefetcher; keep activation x L2/MALL-resident across the layer. gfx11-SPECIFIC lever to realize its cache advantage.'},
  {key:'splitk', name:'Split-K / Parallel Reduction for REAL occupancy', desc:'batch-1 GEMV is work-limited (occ capped by too few output rows). Split the K-reduction across MORE blocks/waves (split-K + LDS/atomic combine), or multiple partial-dot waves per output row, to RAISE resident-wave count -> more MLP. Trades a combine for occupancy. Directly attacks why __launch_bounds__ was a no-op (create work, do not just permit it).'},
  {key:'ilp', name:'ILP / VOPD Dual-Issue / Reduction Restructuring', desc:'gfx11 VOPD dual-issues independent FP32 pairs. Structure dot-product accumulation as MULTIPLE independent accumulator chains (break the serial reduction dependency) so FMAs dual-issue and each FMA latency overlaps. Tree reductions; independent partial sums.'},
  {key:'dequant', name:'Dequant-Path Fusion (mq4 nibble)', desc:'mq4r 4-bit weights dequantize on the read path. Is nibble-extract+scale on the critical path serializing load->use? Fuse dequant into the vectorized load; unpack a whole dwordx4 of nibbles at once; hoist/precompute scales; bit tricks / v_perm / LUT to cut dequant latency so it does not stall the memory pipeline.'},
  {key:'fusion', name:'Intra-kernel Kernel Fusion', desc:'Fuse adjacent per-token decode kernels within ONE .hip (norm+gemv, gemv+residual, gate+up+silu) to eliminate intermediate global-memory round-trips and cut the number of latency-serial kernel boundaries per token. Fewer kernels/token = fewer reload/sync stalls on the critical path.'},
  {key:'moe', name:'MoE Structure Exploitation (a3b)', desc:'a3b activates 8 experts/token via k8-indexed gather (moe_gate_up_k8, moe_down_k8 are top wall%). Overlap expert-weight loads across the 8 experts (prefetch expert N+1 while computing N); coalesce the gather/index; stream expert weights through MALL. MoE-specific latency hiding.'},
  {key:'attention', name:'Attention Under-occupancy (occ 0.5-0.8%!)', desc:'attention_flash_q8_tile runs at 0.5-0.8% occupancy -- the single biggest anomaly. Restructure the flash tile (more parallel work/CU, split head/kv across more waves, raise resident waves) to hide its memory latency WITHOUT changing output. Diagnose why it is so under-occupied.'},
  {key:'wmma', name:'WMMA-for-Reduction (gfx11 UNEXPLORED)', desc:'gfx11 HAS fp16/bf16 WMMA matrix cores, unused for decode (only the int8 path was tried + failed). A low-register fp16/bf16-accumulate WMMA formulation of the GEMV reduction (batch-1 GEMV as a skinny matmul, or pack the K-reduction into WMMA fragments) could raise throughput where scalar MMQ stalls. Coherence-checked; distinct from the falsified iu8 path.'},
  {key:'crossarch', name:'Cross-Arch Differential (why is gfx11 NOT 1.5x?)', desc:'Diagnose EXACTLY what gfx11 leaves unused vs gfx12: its extra 32 CUs (grid/occupancy), its +320 GB/s bandwidth (MLP to realize it), its +32MB MALL (residency), VOPD (ILP). Propose gfx11-SPECIFIC kernel divergences (a gfx11.hip fork) that each capture one unused advantage; state which advantage each lever unlocks.'},
]

const IDEA_SCHEMA = {type:'object', additionalProperties:false, required:['ideas'], properties:{ ideas:{type:'array', items:{type:'object', additionalProperties:false, required:['name','kernel','arch','mechanism','why_helps','novelty','coherence_risk','confidence'], properties:{ name:{type:'string'}, kernel:{type:'string'}, arch:{type:'string', enum:['gfx11','gfx12','both']}, mechanism:{type:'string'}, why_helps:{type:'string'}, novelty:{type:'string'}, coherence_risk:{type:'string'}, confidence:{type:'number'} }}}}}

const SYNTH_SCHEMA = {type:'object', additionalProperties:false, required:['candidates'], properties:{ candidates:{type:'array', items:{type:'object', additionalProperties:false, required:['name','kernels','arch_focus','mechanism','rationale','priority'], properties:{ name:{type:'string'}, kernels:{type:'string'}, arch_focus:{type:'string'}, mechanism:{type:'string'}, rationale:{type:'string'}, priority:{type:'number'} }}}}}

const VERDICT_SCHEMA = {type:'object', additionalProperties:false, required:['verdict','reasoning'], properties:{ verdict:{type:'string', enum:['PROMISING','WEAK','REFUTED']}, physically_plausible:{type:'boolean'}, reachable:{type:'boolean'}, coherence_safe:{type:'boolean'}, reasoning:{type:'string'} }}

const PLAN_ITEM = {type:'object', additionalProperties:false, required:['name','kernel','change','mechanism','expected','coherence_check','rank'], properties:{ name:{type:'string'}, kernel:{type:'string'}, change:{type:'string'}, mechanism:{type:'string'}, expected:{type:'string'}, coherence_check:{type:'string'}, rank:{type:'number'} }}
const PLAN_SCHEMA = {type:'object', additionalProperties:false, required:['gfx11_experiments','gfx12_experiments','summary'], properties:{ gfx11_experiments:{type:'array', items:PLAN_ITEM}, gfx12_experiments:{type:'array', items:PLAN_ITEM}, summary:{type:'string'} }}

phase('Brainstorm')
log(`brainstorming across ${LENSES.length} lenses`)
const ideaSets = await parallel(LENSES.map(lens => () =>
  agent(`${CONTEXT}\n\nYOUR ASSIGNED LENS: ${lens.name}\n${lens.desc}\n\nFrom THIS lens only, brainstorm 3-5 CONCRETE, NOVEL kernel-level levers to raise a3b AR-decode throughput. Prioritize gfx11 (the under-exploited chip). For each: name the exact kernel, the exact code-level mechanism, WHY it increases memory-level-parallelism / hides latency / realizes the arch's wasted advantage, why it is NOVEL vs the tried-and-failed list, and its coherence risk. Be specific and technical; avoid anything already tried.`,
    {schema: IDEA_SCHEMA, phase:'Brainstorm', label:`lens:${lens.key}`})))
const allIdeas = ideaSets.filter(Boolean).flatMap(r => r.ideas || [])
log(`collected ${allIdeas.length} raw levers`)

phase('Synthesize')
const synth = await agent(`${CONTEXT}\n\nHere are ${allIdeas.length} brainstormed levers from a 12-lens swarm:\n${JSON.stringify(allIdeas)}\n\nDEDUPE (merge near-identical), CLUSTER by mechanism, and RANK into the strongest ~12 candidate experiments. Bias the ranking toward UNBLOCKING gfx11 (largest unused headroom) while keeping the best gfx12-ceiling levers. Rank by: (plausibility it actually raises bandwidth-efficiency / hides latency for a batch-1 latency-bound GEMV) x (kernel-.hip-reachability) x (novelty vs tried-and-failed) x (coherence safety). priority = higher is better. Return the merged top candidates.`,
  {schema: SYNTH_SCHEMA, phase:'Synthesize', effort:'high', label:'synth'})
const cands = (synth.candidates || []).sort((a,b)=>(b.priority||0)-(a.priority||0)).slice(0,10)
log(`verifying top ${cands.length} candidates adversarially`)

phase('Verify')
const VLENS = [
  {k:'physics', p:'PHYSICS: Given the roofline (100% busy, ~26-36% BW-efficient, low occupancy, memory 44-60% busy), will this lever ACTUALLY increase memory-level parallelism / realize bandwidth / raise throughput -- or is the reasoning hand-wavy? Would it plausibly move the needle >3%? Default REFUTED if the mechanism does not clearly attack the latency-stall.'},
  {k:'reachability', p:'REACHABILITY: Is this a PURE .hip kernel-body change (no host/grid/dispatch change needed)? Has it effectively been tried already (row-reuse, launch_bounds occupancy, unroll, iu8-WMMA, traffic-cut, hipGraph)? Default REFUTED if it needs host changes or duplicates a failed lever.'},
  {k:'coherence', p:'COHERENCE: Can this preserve EXACT output math? Split-K/atomics/reordered reductions/WMMA can change rounding -> attractor risk. Default REFUTED if it likely changes numerics enough to risk garbage output.'},
]
const verified = await parallel(cands.map(c => () =>
  parallel(VLENS.map(vl => () =>
    agent(`${CONTEXT}\n\nAdversarially evaluate ONE candidate lever, trying to REFUTE it. ${vl.p}\n\nCANDIDATE:\n${JSON.stringify(c)}\n\nGive a skeptical verdict.`,
      {schema: VERDICT_SCHEMA, phase:'Verify', label:`verify:${vl.k}:${(c.name||'').slice(0,20)}`})))
    .then(vs => ({candidate:c, verdicts:vs.filter(Boolean)}))))
const survivors = verified.filter(v => v && v.verdicts.filter(x=>x.verdict==='PROMISING').length >= 2)
log(`${survivors.length}/${cands.length} candidates survived adversarial refutation`)

phase('Plan')
const plan = await agent(`${CONTEXT}\n\nThese candidate levers survived adversarial refutation (>=2 of 3 skeptics rated PROMISING):\n${JSON.stringify(survivors)}\n\nProduce the FINAL ranked EXPERIMENT PLAN. Separate gfx11_experiments (PRIMARY -- most headroom) and gfx12_experiments (ceiling). For each: exact kernel, the concrete .hip change, the mechanism (which latency-stall/wasted-advantage it attacks), expected direction+rough magnitude, and the coherence_check to run. rank = 1 is highest priority. In summary: state the single highest-conviction gfx11 lever and why the 164:155 ratio makes headroom physically guaranteed.`,
  {schema: PLAN_SCHEMA, phase:'Plan', effort:'high', label:'plan'})
return { plan, stats:{ raw_ideas: allIdeas.length, ranked: cands.length, survived: survivors.length } }
