> **CLEAN-ROOM BOUND:** mechanisms re-expressed on hipfire HIP+mqN. NO competitor code, NO ggml/Q4_K/Vulkan import — sources are file+function POINTERS only, ideas re-derived independently.
> **Competitive context (verified, hashed):** hipfire WINS prefill (a3b 2239 vs ZINC 759, 2.95x); the entire deficit is DECODE (158 vs 166). Decode is batch-1 latency/utilization-bound.
> **Sources (read-only code mining):** zinc@5ab624c, ROCmFPX@a6a9376, rocmfp4-llama@4795079. 132-agent workflow (extract -> cross-check vs our kernels -> adversarial+clean-room verify -> synthesize). Companion: docs/investigations/competitor-doc-mining-2026-07-09.md.

# Competitor CODE-mining — gfx1201 decode mechanisms (CLEAN-ROOM)

> **BOUND:** every mechanism below is re-expressed as an *idea* to re-derive on HIP + the mqN quant family (mq4r / mq4 / mq6 / HFQ4-G256). No competitor code, no ggml/Q4_K/Q5_K block layout, no Vulkan/Metal primitive is imported. Sources are cited as file+function POINTERS only. The decode gap is the whole gap (gfx1201 a3b mq4r: **158 vs ZINC 166**; prefill already won 2239 vs 759), and decode is **batch-1 latency/utilization-bound**, so the ranking weights *decode-fit* heavily and treats the MTP/DFlash batch-K regime as the prime regime-changer.

---

## 1. Top liftable MECHANISMS (ranked: value × novelty × decode-fit × clean-room-ease)

### Tier A — the batch-K regime (hipfire's own "only remaining regime-changer")

Decode is under-saturated at batch-1; every Tier-A lever converts *N separate batch-1 dispatches* into *one dispatch that pays the expensive per-weight cost once and fans out over K candidate columns*. This is exactly the MTP/DFlash candidate-verify shape. None of these help pure AR batch-1 (num_cols=1 degenerates), so **all Tier-A wins must be measured on the daemon spec-decode/MTP verify step, not a microbench, and pass the DFlash coherence gate before any claim.**

**#1 — Weight-stationary batched-column GEMV (amortized-dequant multi-token decode)**
- Source POINTER: `zinc/src/shaders/dmmv_q5k.comp` (acc_mode>1 branch, ~L88–172); mirror `zinc/src/compute/dmmv.zig`.
- IDEA: one workgroup owns one output row; for each weight K-block, decode the dequant factors ONCE, then loop over `num_cols` X-columns accumulating into `num_cols` register sums before advancing to the next K-block. Amortizes global weight fetch + nibble-unpack (the expensive part) over N dot-products.
- Why it wins on RDNA4 decode: decode GEMV is BW/latency-bound; the weight stream dominates. Reusing one decoded weight against several candidate columns cuts weight traffic per useful output by 1/N.
- How THEY do it (conceptual): repurpose the accumulate-mode scalar as a token-column count (≤40), inner column loop inside the K-block dequant loop.
- How WE do it on HIP+mqN: add an inner `num_cols` loop over X-rows inside the existing mq4 K-block dequant loop in `fused_qkvza_hfq4g256.hip` (dense QKV), `gemv_hfq4g256_moe_gate_up_k8_indexed.hip` / `gemv_hfq4g256_moe_down_k8_indexed_batched_expanded.hip` (MoE), and the lm_head GEMV. This is the axis ORTHOGONAL to the row-tile hipfire already ships (row-tile reuses X *across rows*; this reuses WEIGHT *across columns/tokens*). Today the `_indexed_batched` kernels fan tokens out on `blockIdx.z` and independently re-fetch+re-dequant the same expert weight per token — zero cross-token register reuse.
- Caveats: **cap num_cols at ~4–8, never 40** — hipfire GEMVs already sit at 88–96 VGPR (verified via disassembly), a 40-wide accumulator array spills catastrophically. Fully effective on the DENSE projections (qkv, lm_head) where all K candidates share one weight matrix; **partial for MoE** — only co-routed candidates share an expert, so MoE needs a sort/gather-by-expert (→ #2) to realize reuse.
- our_status: **novel-liftable** (orthogonal axis genuinely absent).

**#2 — Small-batch-K expert dedup via the existing sorted-slot scatter (8-way ragged token-route batching)**
- Source POINTER: `zinc/src/shaders/dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1.comp`.
- IDEA: decode an expert weight-block once, dot it against the several tokens in the batch that routed to *that same expert*, masked by a per-slot active flag. This is the MoE realization of #1's weight-reuse.
- Why it wins: makes #1 effective for MoE gate_up/down, where a random K-token batch otherwise scatters across experts and defeats weight reuse.
- How WE do it: hipfire ALREADY owns the machinery — `gemm_hfq4g256_moe_grouped_mmq*.hip` + `moe_scatter_fused_k8.hip` sort all slots of a tile to one expert (`expert_tile_ids[]` + `sorted_slot_index[]`), but the header scopes it to `batch ≥ 256` (prefill). **Adapt the sort/scatter down to batch-K (K=4–16 verified tokens per spec-decode step)** rather than green-fielding a NUM_COLS-mask kernel; the sort/scatter infra already exists, just gated above 256. The decode-time `_indexed_batched` GEMVs currently do a per-token independent expert lookup with no cross-token dedup.
- our_status: **novel-liftable** (mechanism exists but only wired for prefill).

**#3 — Register-tiled multi-column (batch-width) GEMV**
- Source POINTER: `rocmfp4-llama/ggml/src/ggml-cuda/mmvf.cu` (ncols_dst template).
- IDEA: template the kernel on batch-width; load the weight-row element ONCE per K-step, FMA it against `ncols_dst` RHS vectors held in an unrolled register array. Same win as #1 for the dense/F16 path, expressed as a compile-time template rather than a runtime count.
- How WE do it: this is the dense-projection twin of #1 — `fused_qkvza_hfq4g256`, `fused_gate_up_hfq4g256`, lm_head GEMV. No `ncols_dst`-style construct exists anywhere in `kernels/src`. Prefer a **distinct kernel name per width variant** (module-name-keyed JIT cache) over a bare template parameter.
- our_status: **novel-liftable**. (Rank slightly below #1 because #1 already frames the same idea with the concrete mqN adaptation; treat #3 as the compile-time-specialized implementation strategy for #1 on dense projections.)

**#4 — Windowed routing-token-base offset (batch-K MoE building block)**
- Source POINTER: `zinc/src/shaders/moe_route_pack.comp` (push-constant token-base).
- IDEA: a base-offset kernel arg lets one scatter/pack kernel operate on an arbitrary `[base, base+n)` sub-window of a longer-lived routing cache that spans multiple forward-pass calls — so an MTP window's routing decisions can be packed incrementally without re-basing from scratch.
- How WE do it: trivial add (one extra int param + one address term) to the ~20 indexed-GEMV kernels that already compute `routing_base = bid*K_TOP`. It's the plumbing that makes #1/#2 composable across a multi-token MTP window rather than per-forward.
- our_status: **novel-liftable** (per-token base exists; cross-dispatch windowed base does not).

### Tier B — lm_head (the one huge-M decode op)

**#5 — M-adaptive GEMV topology switch / wide R=8 row-tile for huge-M lm_head**
- Source POINTERS: `zinc/src/compute/dmmv.zig` (M-threshold K-parallel→row-batch switch, ~L3193–3230); `zinc/src/shaders/dmmv_q6k_wide.comp` (NUM_ROWS=8 for vocab≥100k).
- IDEA: above an output-row threshold (M ≈ 64K–100K), switch away from K-parallel row-pair reduction (many tiny workgroups, hidden vector reloaded per WG) to a topology with far fewer, higher-input-reuse workgroups (row-per-thread batch layout, or R=8 row-tile). At vocab scale, workgroup-launch count and per-WG hidden-vector L1 residency become the bottleneck, not FLOPs.
- Why it wins on decode: lm_head runs once per token at M = vocab (150K–248K). hipfire picks a FIXED per-arch R (gfx1201 → R=2) with `use_wide` gated only on `m≥64`, no scaling into the 100K+ regime; and `gemm_hfq4g256_batched_lmhead` only takes the WMMA path at `batch_size>1`, so batch-1 AR/DFlash decode of a 150K-row lm_head falls to the scalar per-row-block GEMV — the exact "hundreds of thousands of tiny workgroups" regime this avoids.
- How WE do it: hipfire already OWNS the building blocks (`gemv_hfq4g256_multirow_r8`, `gemv_hfq4g256_wide`). Add an **M-magnitude branch in `gemv.rs`** that forces R=8 (or the batch-layout topology) for M above ~64–100K on HFQ4-G256 lm_head, sweeping the wave32 crossover (ZINC's 65536/100000 are Vulkan/wave64 numbers — retune, don't port).
- Caveats: NARROW — does NOT touch the qkvza/gate_up/moe_down hotspots (their M is thousands, far below threshold). Expect low-single-digit tok/s, not a hotspot win. **VGPR risk:** R2→R8 quadruples accumulators on kernels already at 88–96 VGPR — check zero-spill + occupancy with the `gfx-kernel-metadata` skill BEFORE trusting any number. Coherence gate mandatory (reduction-order change perturbs argmax). The FFN "R=2/3/4 multirow DEAD" verdict (gfx1201.md:56) does NOT transfer — that was small-M cache-resident; lm_head is huge-M DRAM-streamed.
- our_status: **novel-liftable**.

### Tier C — attention preamble fusion (small per-hit, stacks across all full-attn layers)

**#6 — Three-way RMSNorm + RoPE + KV-cache scatter-write in one dispatch**
- Source POINTER: `zinc/src/shaders/qk_norm_rope_kv_write_batched.comp` (~L43–150).
- IDEA: one kernel per (head, token) computes the per-head RMS scale, re-loads the original element, applies norm·weight, rotates via RoPE, and scatter-writes straight into the KV slot — no intermediate buffer. Collapses ~3 launches + a store+reload round-trip of the normalized-but-unrotated Q/K vector into one launch.
- Why it wins: batch-1 decode is launch-floor/round-trip sensitive; this removes a full global store+load of the per-head vector per layer per step.
- How WE do it: hipfire runs this as 5 launches (`rmsnorm_batched(Q)`, `rmsnorm_batched(K)`, `rope_partial_interleaved_f32_batched`, K-write, V-write — `qwen35.rs:9552–9647`), with `rmsnorm_batched` writing normalized Q/K back in place (`norm.rs:128–176`) that RoPE then re-reads. Fuse **norm+rope only** as the cheap two-way (subgroupAdd→`__shfl` reduce, F32 activations, no format lock).
- Caveats: expected **~0.6%** (hipfire's own ZINC-decode-gap figure) — below the 5% investigation threshold; **rocprof-gate it, never wall-clock**. Do NOT fold the KV-write stage: hipfire's KV-write is quant-format-aware (q8/fwht) + KvTierPlan-scheduled, so folding it forces KV quantization into the norm+rope kernel (high cost). Applies only to full-attention layers (a minority of a3b), NOT DeltaNet qkvza or the MoE GEMVs. Low ROI — do not prioritize over Tier A.
- our_status: **novel-liftable** (norm+rope two-way).

**#7 — Norm-scale applied inline at the RoPE load site**
- Source POINTER: same file, ~L103–135.
- IDEA: compute rms_inv once, then in the rotate loop re-read the ORIGINAL unnormalized element and multiply by rms_inv·weight at the point of consumption — never materialize a normalized array. A register-held re-derivation avoids even the "one extra global load" the source pays.
- How WE do it: this is the specific register-fusion that makes #6's two-way fusion win; same target kernels (`norm.rs` in-place rmsnorm + `rope_partial_interleaved_f32_batched`). head_dim=128 over 32 lanes is register-holdable, so we can beat the source by holding the scaled value rather than re-loading.
- Caveats: same ~0.6% / rocprof-gate / full-attn-only as #6; **sequence with #6/#8 to avoid double-counting** (they overlap the same round-trip). F32 → byte-parity checkable under `HIPFIRE_DETERMINISTIC=1`.
- our_status: **novel-liftable** (subset of #6).

**#8 — Merge Q-head and K-head rmsnorm into one dispatch (workgroup-id branch)**
- Source POINTER: same file, L10,44–49.
- IDEA: dispatch `n_q_heads+n_k_heads` workgroups; branch on workgroup-id to select Q vs K buffers while sharing the reduction/rotate path — one launch instead of two.
- How WE do it: hipfire's RoPE kernel already merges Q&K (`rope_partial_halfsplit.hip:44–58`), but the *preceding* q_norm/k_norm is two separate `rmsnorm_batched` calls (`qwen35.rs:9552–9567`). Merge those two into one wg-id-branched dispatch.
- our_status: **novel-liftable** (norm half; RoPE half already-have). Stacks cleanly into #6.

### Tier D — attention split-K reduce kernel occupancy (long-context decode)

**#9 — head_dim=256 reduce: 2-wave concurrent instead of same-32-lanes-serial (cleanest measured 2× serial)**
- Source POINTER: `zinc/src/shaders/flash_attn_split_merge.comp`.
- IDEA: `attention_flash_q8_0_reduce.hip` (fixed 32-thread block) runs its Pass-2 tile-accumulation loop TWICE serially for head_dim=256 (Qwen3.5, n_halves=2). Widen to a 64-thread block (2 waves, one per half, on separate SIMDs) so both halves run concurrently — a literal 2× serial repeat removed (unlike the SIMD-lockstep-masked cases). `global_sum` is half/d_base-independent, so each wave recomputes it redundantly — NO cross-wave sync needed.
- Why it wins: real wall-clock 2× on the reduce kernel for head_dim=256 archs; grows with context length (more tiles).
- How WE do it: change the launch config of `attention_flash_q8_0_reduce.hip` for the head_dim=256 path; head_dim=128 (n_halves=1) unaffected.
- Caveats: reduce kernel is only ~0.40% of decode wall (`docs/gfx1201-native-surface.md`), so absolute win is small but it's the least-ambiguous, cheapest structural fix in the batch. rocprof-gate.
- our_status: **novel-liftable**.

**#10 — Wave-parallel online-softmax merge (shard the global-max scan across lanes)**
- Source POINTER: `zinc/src/shaders/flash_attn_split_merge.comp`.
- IDEA: `attention_flash_q8_0_reduce.hip` Pass-1 has all 32 lanes redundantly loop over ALL n_tiles computing the identical `global_max` (no tid indexing) — a true O(n_tiles) lockstep-serial cost. Shard n_tiles across the 32 lanes (each owns ~n_tiles/32), local-max per lane, then a 5-step `__shfl_xor` butterfly → O(n_tiles/32 + 5).
- How WE do it: hipfire's SIBLING kernel `attention_flash_q8_0_tile.hip` already uses this exact `__shfl_xor` idiom (L155–159) — the transferable pattern is proven in-repo, just unapplied to this Pass-1 scan.
- Caveats: payoff scales with context length; near-noise at short ctx. Long-context-leaning lever.
- our_status: **novel-liftable**.

### Tier E — ISA / occupancy micro-lever (generic to every decode kernel)

**#11 — Explicit early VGPR pool deallocation (`s_sendmsg(MSG_DEALLOC_VGPRS)` before `s_endpgm`)**
- Source POINTER: `zinc/src/zinc_rt/isa/gfx1201/dmmv_q4_0_resident_grid.s`; corroborated `dmmv_q8_0_row_range_parallel.s`.
- IDEA: signal the wave scheduler that VGPRs are free once the last store has issued, so a new wave can be admitted before this one fully retires — raises waves-in-flight for a grid of small, VGPR-light latency-bound decode waves.
- Why it might win on RDNA4 decode: disassembly of 7 real hot hipfire kernels (gfx1100, ROCm 7.2.0) shows **zero `sendmsg` instructions** and VGPR 52–96 (0 spills) — i.e. these kernels ARE in a VGPR-occupancy-sensitive tier, and LLVM does NOT auto-insert this.
- How WE do it: HIP inline asm (`asm volatile("s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)" ::: )`) immediately before the final store, on one VGPR-tier kernel first (e.g. `gemv_hfq4g256_moe_gate_up_indexed` at ~96 VGPR).
- Caveats: **plausible-but-unverified microarchitectural lever** — MUST be measured via fresh-probe/rocprof per the Δ≥5% rule, not assumed. First check whether any `-mllvm` flag on ROCm 7.2.x controls auto-insertion before hand-rolling asm.
- our_status: **novel-liftable**.

### Tier F — codegen / dispatch

**#12 — Compile-time K specialization (bake hidden-dim as a `#define`, full-unroll the group loop)**
- Source POINTER: `zinc/src/compute/dmmv.zig`.
- IDEA: for kernels whose K is a per-(arch,hidden_dim) constant, compile with `K_CONST` fixed so the group loop fully unrolls and per-block index arithmetic folds to immediates — matters more on RDNA's smaller register file than the FLOP count suggests.
- How WE do it: hipfire already bakes `NUM_ROWS` as a compile-time `#define` (`MOE_DOWN_FUSED_NUM_ROWS`) and JIT-caches per shape, so extending the same per-shape cache to also bake `K` (currently `groups_per_row = K/256` runtime) is a natural, low-risk extension. Distinct module name per K variant (JIT-cache-keyed-by-name gotcha).
- Caveats: only a win where K is truly fixed per arch (hidden/inter dims), not the growing lm_head/ctx dims. Stacks with Tier A (specialize the batch-K variants).
- our_status: **novel-liftable**.

### Tier G — MoE dispatch/routing

**#13 — Fold the always-active shared-expert FFN into the routed-expert fan-out**
- Source POINTER: `zinc/src/compute/forward_zinc_rt.zig`.
- IDEA: schedule the shared expert's gate/up/down as one more slot inside the SAME batched fan-out dispatch as the routed experts, so its weight stream co-issues instead of running as a fully serial tail.
- Why it wins on a3b: Qwen3.5-A3B has `shared_expert_intermediate == routed moe_intermediate` (structurally ideal). The shared-expert gate_up is already fused with the router, but the shared-expert DOWN is a SEPARATE kernel dispatched BEFORE the routed `_indexed_batched_expanded` down kernel (`pipeline/mod.rs:452–511`) — two serial launches.
- How WE do it: merge the shared-down into the routed-down fan-out dispatch.
- our_status: **novel-liftable**.

**#14 — Packed (value,index) monotonic key for single-shuffle argmax topk**
- Source POINTER: `zinc/src/shaders/softmax_topk_v2.comp`.
- IDEA: encode (float value, int index) as one order-preserving uint32 so ONE `__shfl` max-butterfly finds both winner value AND index, replacing the two parallel shuffle streams per top-k round.
- How WE do it: `moe_topk_renorm_k8.hip::r4_topk_pair_reduce` currently runs two `__shfl_down` streams (value + idx) × 8 rounds. Pack a radix-sortable-float key with a `255-index` tie-break byte.
- **Load-bearing precision constraint:** the packed key truncates the float to ~24 bits. hipfire's topk carries UNtruncated fp32 specifically because 1-ULP drift here is a proven attractor trigger (#164 postmortem). Use the packed key ONLY to find the winning lane/index in one pass, then **re-read the exact non-truncated value** for downstream weight/renorm math — never forward the decoded-packed value. That constraint is why liftability is 4 not 5.
- our_status: **novel-liftable**.

**Also-liftable (below the cut, decode-relevant, batch onto the certify queue):**
- **Delayed softmax** (argmax on raw logits, softmax only the K selected) — `softmax_topk_v2.comp`. Valid by monotonicity; saves ~248 `expf` per router/layer/token at n_exp=256. But the fused-order variant is `tried-dead` (#164 1-ULP attractor) — the *reorder* is untested but must use direct division + full-precision carry-forward and re-run the 1500-token greedy + coherence protocol. liftability 3.
- **Single-warp (32-thread block) topk for n_exp=256** — kill the LDS+barrier cross-warp merge that 7 of 8 warps currently idle through; n_exp=256 = 8 regs/lane in one warp. liftability 4. Coherence-gate (reduction-order).
- **GQA-ratio batch amplification for attention kernel selection** — `rocmfp4-llama/fattn.cu`. Group the kv_group query heads sharing one KV head into one CTA to amortize the K/V tile fetch instead of re-fetching per query head. liftability 4, GQA archs (qwen35/qwen2/cohere2moe).
- **Selective load-time coarser decode shadow copy** — `forward_zinc_rt.zig`. Build a coarser-mqN decode-only copy of decode-hot tensors at load (amortized once), keep the higher-fidelity original for prefill. liftability 4 — a bytes/token lever, but a genuine quality-risk redesign; coherence-gate hard.
- **Bounded MSE-optimal per-group scale search (encoder, CPU-side)** — `ROCmFPX/rocmfpx.c`. Zero-kernel-risk build-time quality lever: neighbor-code SSE search (imatrix-weighted, clip-error pruned) replaces naive `range/15` minmax in `hipfire-quantize`. Complementary to AWQ (channel pre-scale), not overlapping. liftability 4 — improves decode *quality* headroom, not tok/s.
- **Time-bounded LRU + per-node validity snapshot for the hipGraph cache** — `rocmfp4-llama/common.cuh`. Structural drift-detection vs today's manual dirty-flags (the class that caused the gfx12 99→50 replay clobber). liftability 4, robustness not raw tok/s.

---

## 2. RE-TEST queue — DO-NOT-RETRY verdicts that competitor code + our own roofline reopen

**#R1 (FIRST) — wave64 for the ATTENTION kernel, on the correct latency-bound shape.**
Our wave64 "neutral/DEAD" verdict (`project_wave64_neutral_mq4_gemv_gfx1201_2026_07_08`, −0.2…−0.8%) was measured on **MQ4 GEMV at deliberately DRAM-saturating shapes** and reasoned "DRAM-bound, so wave-width is moot." But this session's own roofline shows decode is **latency/utilization-bound** (mem_busy 40–55%; BW extraction climbs 43%→79%→92% as occupancy rises 8→16→32 wv/CU — still occupancy-sensitive). `attention_flash_q8_0_tile.hip` runs native wave32 (`block=[32,1,1]`), its reduction is a fixed head_dim `__shfl_xor` row-reduce (structurally different from a K-streaming GEMV load), and no RDNA force-wave64 attention path has ever been compiled. hipfire HAS working `-mwavefrontsize64` infra. **Run a narrow isolated A/B on `attention_flash_q8_0_tile` before extending the GEMV verdict to this kernel.**

**#R2 — wave64 for `gated_delta_net_q8_fast`, same corrected framing.** GDN's row-cooperative state-update shape was never retested under the latency-bound/occupancy-sensitive lens — only a GEMV was. Re-run wave64 vs wave32 on `gated_delta_net_q8_fast` under the occupancy-probe methodology before treating it as closed.

**#R3 — row-per-lane zero-reduction GEMV at true decode batch-1 shapes.** ZINC's `dmmv_q4_0_resident_grid.s` / `dmmv_q8_0_row_range_parallel.s` put ONE full weight row on ONE lane with zero cross-lane reduction (parallelism from many independent in-flight rows + early-VGPR-dealloc for wave turnover). Our falsification recompiled hipfire's OWN cooperative-shfl-reduction kernel at wave64 on 70MB+ DRAM-saturating shapes — it never tested ZINC's actual launch config (independent-row, no reduction) in the small-VGPR high-turnover latency-bound decode regime it targets. Low prior it beats the N=2 row-tile, but the premise the smoke test answered was a different question. Cheap isolated A/B.

**#R4 — lm_head R=8 sweep (from #5).** Sweep `HIPFIRE_GEMV_ROWS=8` on the lm_head/vocab-projection call path INDEPENDENTLY of the FFN sweep. The FFN "multirow DEAD" verdict's premise (small-M, cache-resident, VGPR-bound) does not transfer to lm_head's huge-M DRAM-streamed shape. Gate on `gfx-kernel-metadata` zero-spill check first.

---

## 3. Orthogonal-compound map (which mechanisms stack)

- **Batch-K supercluster** (highest combined value): **#1 weight-stationary column loop × #3 batch-width template × #2 MoE expert-dedup × #4 windowed routing-base × #12 compile-time K-specialize.** Together these turn the MTP/DFlash verify step from N batch-1 dispatches into one weight-stationary, expert-sorted, K-baked dispatch. #1 is the algorithm; #3/#12 are the codegen strategy; #2/#4 make it work for MoE. This is the regime-changer — build as one coordinated arm.
- **Attention preamble fusion**: **#6 (norm+rope+kv) ⊇ #7 (inline norm-scale) ⊇ #8 (Q/K rmsnorm merge)** — these are nested, not additive; ship as ONE fused norm+rope kernel and count the win once (~0.6%). Do NOT double-count against the existing fused_qkvza lever.
- **Attention reduce**: **#9 (head_dim=256 2-wave) × #10 (wave-parallel Pass-1 max)** — independent, both on `attention_flash_q8_0_reduce.hip`; stack for the long-context head_dim=256 (Qwen3.5) path.
- **Generic occupancy**: **#11 (VGPR early-dealloc)** composes with EVERY decode GEMV including the Tier-A batch-K kernels (raises wave turnover on exactly the small-VGPR grids batch-K produces). Apply after #1 lands.
- **Router**: **#14 (packed argmax) × single-warp topk × delayed softmax** all target `moe_topk_renorm_k8` — stack, but each independently needs the 1-ULP coherence re-validation.
- **Cross-cutting guardrail**: every stacked change is protected by the cross-arch isolation guard; feed each mechanism to the certify loop as its own fleet A/B so compound interactions are measured, not assumed (bundle-interaction is real — the grok curated-drop measured WORSE than the sum).

---

## 4. Dropped

**Format/backend-locked (could not re-derive on mqN/HIP — mechanism-only or DROP):**
- Q4_K / Q5_K / Q5_1 / Q4_0 / Q6_K bit-layouts (packed 6-bit sub-scales, hi-bit planes, 12-byte scale tables) — ggml block-format-locked; mqN's flat 136B (fp32 scale + fp32 zp + 128B nibbles) already supersedes them. `dmmv_q*.comp`, `convert.cu`, `vecdotq.cuh`.
- MXFP4/NVFP4 Blackwell tensor-core MMA fast path — NVIDIA Hopper/Blackwell ISA-locked, compiled out off-Blackwell. `mmq.cu`. (Note: hipfire's own RDNA4 fp8/fp4 WMMA prefill idea is independent/unbuilt, different instruction set — not this.)
- Vulkan push-descriptor / push-constant bit-punning / descriptor-binding dual-dtype aliasing — HIP passes raw pointers; no binding layer to work around. `dmmv.zig`, `qk_norm_rope_kv_write_batched.comp`, `ssm_conv1d_batched.comp`, `pipeline.zig`.
- Programmatic Dependent Launch (PDL) — CUDA Hopper-only, no HIP/RDNA equivalent. `gated_delta_net.cu`.
- Vulkan `vkCmdDispatchIndirect` GPU-sized dispatch — no HIP primitive; the fixed-over-provisioned-grid + sentinel-skip fallback is the HIP-native answer and hipfire already ships it (`moe_scatter_fused_k8` −1 sentinel).

**Already-have (verified in-repo; do not re-lift):**
- Split-K flash-decoding + 2-pass exact merge (`attention_flash_q8_0_tile.hip` + `_reduce.hip`) — stricter than ZINC's online-softmax.
- 16-way ILP independent accumulators → hipfire's P=4 register-ring, LANDED +6.95% gfx11 (`project_attention_ring_gfx11_win`).
- vec4/uint4/128-bit wide loads — arch-GATED (gfx1100 +4.8%, gfx1151 +12.8%, gfx1201 +2.7%; gfx1010/1030 −51/−56%), corrects ZINC's blanket "faster on AMD" framing.
- Fused MoE-down + K_TOP register-accumulate + NUM_ROWS=2 row-tile — LANDED +3.5% gfx1201, explicitly credits ZINC lineage (`gemv_hfq4g256_moe_down_k8_indexed_fused_acc.hip`).
- gate+up dual-accumulate shared-X (`gemv_hfq4g256_moe_gate_up_k8_indexed.hip`); per-lane strided K-sweep single end-reduction; single-warp shuffle reduction (no LDS); DOG_X8 8-wide unpack+FMA; unaligned dword loads; affine single-FMA dequant (`sc*nibble+zp`); Q8_1 (d,sum_x) row-sum-reuse zero-point fold (gfx906 dp4a GEMM); block_q8_1_mmq half2 co-storage.
- GDN register/LDS-resident state, load-once/store-once (`gated_delta_net_q8_fast`, `_f32_batch_seq`); GDN tree-tape state checkpoint (superset of ZINC's fixed-K ring); conv1d register-shift ring buffer; thread-per-channel barrier-free conv; fused conv+SiLU (+ LFM2 double-gate).
- attn_sink folded once into split-K merge (deepseek4 SWA); MoE SGLang-style single-launch scatter (histogram+scan+scatter, sentinel-pad, no D2H sync); device-indexed capturable MoE grid; hipGraph capture/replay dual-lifetime; per-arch capability-gated wave64 kernel selection + `%warpSize` launch asserts; per-kernel `__launch_bounds__` tuning; fused_qkvza 4-way projection fusion; fused sigmoid-gate-scale GEMV epilogues; `__ockl_fdot2`/v_dot2 (dot2 GEMM family); DeltaNet tail-only YaRN RoPE (rope/nope split); Hyper-Connections `hc_mix_4stream` fused combine; L2-norm decoupled pre-pass; two-level R3c2 rmsnorm warptail reduction.

**Tried-dead (cite, do not blind-retry):**
- gate+up+SwiGLU single-dispatch fusion — ZINC's OWN logs say noise-to-regression on RDNA decode, REVERTED; hipfire concurs (`project_zinc_decode_gap_workflow`). A bounded HIP retest is defensible (their revert rationale was Vulkan-barrier-specific) but DO-NOT-RETRY-BLIND.
- int8/dp4a MoE decode-GEMV — BW/latency-bound (MemUnitBusy>VALUBusy, DEP_WAIT 83–94%), falsified twice (`project_dp4a_moe_decode_bw_bound_falsified`, `project_int8_dp4a_verify_falsified`); int8-WMMA (wmmqa) −5..12% + coherence-fail. Any ALU-reduction lever on this path (e.g. activation-sum precompute) hits the same wall — rocprof-gate before believing.
- Fused softmax+topk single kernel — 1-ULP renorm divergence → structural attractor on MQ4 MoE (#164), reverted to split softmax_f32 + direct-division topk.
- Single-thread-serial Sinkhorn / router row-normalize — PMC showed ~0.5% occupancy, replaced by lane-per-cell wave design (`hc_sinkhorn_4x4.hip`).
- gate_up row-tile — −2.9% gfx1201 (x already L1-hot), default-OFF (`gemv_hfq4g256_moe_gate_up_indexed_rowtile.hip`).
- Raw PM4/KFD/HSA/UMQ direct dispatch — redline Phases 1–3: KFD_IOC_CREATE_QUEUE EINVAL on RDNA3, HSA-AQL LOSES to HIP in production burst regime (3.97 vs 3.22 µs/dispatch), 9–16µs floor is GPU-side not host — hipGraph is the real lever. MMQ-on-RDNA4 rocBLAS fallback also independently DO-NOT-RETRY (5.6× slower).

---

## 5. Provenance appendix (POINTERS only — no code)

**ZINC (`.competitors/zinc/`):**
- `src/shaders/dmmv_q5k.comp` — #1 weight-stationary column loop (acc_mode).
- `src/compute/dmmv.zig` — #5 M-adaptive topology switch; #12 compile-time K; #13 shared-expert fold; GPU-indirect-dispatch analog.
- `src/shaders/dmmv_q6k_wide.comp` — #5 NUM_ROWS=8 vocab-wide.
- `src/shaders/qk_norm_rope_kv_write_batched.comp` — #6 3-way fusion; #7 inline norm-scale; #8 Q/K merge.
- `src/shaders/flash_attn_split_merge.comp` — #9 n_halves 2-wave; #10 wave-parallel Pass-1 max.
- `src/shaders/dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1.comp` — #2 ragged token-route batching.
- `src/shaders/moe_route_pack.comp` — #4 windowed routing-base.
- `src/shaders/softmax_topk_v2.comp` — #14 packed-key argmax; delayed softmax.
- `src/zinc_rt/isa/gfx1201/dmmv_q4_0_resident_grid.s`, `dmmv_q8_0_row_range_parallel.s` — #11 VGPR early-dealloc; #R3 row-per-lane zero-reduction.
- `src/compute/forward_zinc_rt.zig` — load-time coarser shadow copy; quantize-upstream-of-norm heuristic.

**rocmfp4-llama (`.competitors/rocmfp4-llama/ggml/`):**
- `src/ggml-cuda/mmvf.cu` — #3 batch-width column tile; block-size autotune.
- `src/ggml-cuda/fattn.cu` — GQA-ratio batch amplification; capability×head_dim crossover.
- `src/ggml-cuda/common.cuh` — LRU+validity graph cache; multi-stream fork/join; v_dot2/sudot4 ISA split.
- `src/ggml-cuda/gated_delta_net.cu`, `gla.cu` — register-resident recurrence variants (mostly already-have / #R2).

**ROCmFPX (`.competitors/ROCmFPX/ggml/`):**
- `rocmfpx/rocmfpx.c` — bounded MSE-optimal scale search (encoder quality lever).
- `src/ggml-cuda/dsv4.cu` — power-of-2 KV scale + RoPE-tail-excluded quant (deepseek4 compressed-KV future).

*(Verdict metadata: 4 clean-room-verified survivors [#1,#5,#6,#7] carried liftability 4 with full real∧applicable∧cleanroom∧¬refuted gates; #2,#3,#4,#8–#14 drawn from the tagged-context novel-liftable set at liftability 3–5. All decode-fit, all HIP+mqN re-derivable, zero verbatim import.)*
