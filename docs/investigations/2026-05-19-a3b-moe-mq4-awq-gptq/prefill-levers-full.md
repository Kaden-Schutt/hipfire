# hipEngine prefill levers — full inventory vs hipfire (Qwen3.6-35B-A3B-PARO, gfx1151)

Read-only audit, 2026-05-20. Wider lens companion to `hipengine-perf-comparison.md`.

- hipEngine survey: `/tmp/hipengine-survey/`
- hipfire branch: `paroquant-a3b` worktree

The five **top levers** from the prior comparison report are referenced by ID
below (T1 graph-capture, T2 chunkwise GDN, T3 multi-rotation fusion, T4
AWQ pack8 GEMV, T5 AOTriton + split-K bucket) — only re-listed if a
sub-finding sharpens them. Everything else here is a new lever.

## Summary

- **Distinct levers found: 38** (24 prefill-only, 9 prefill-and-decode, 5
  diagnostics/infra). 5 are duplicates of the prior top-5 list.
- **Top-3 quick wins (≤1 day each):**
  1. **L17 — Prefill-only token-tile shared-expert W8A16 GEMV** (S, +1-3%
     measured on hipEngine). hipfire's shared-expert in MoE prefill uses
     row-GEMV; one token-tile rewrite reuses weights across 2-4 tokens per
     block. Lower risk than chunkwise GDN, lands without touching GDN.
  2. **L7 — `rmsnorm_batched` → `add_rmsnorm` fuse for the post-attention
     stage** (XS, +0.5-1.5% prefill). Hipfire's `prefill_moe_ffn_body_batched`
     calls `rmsnorm_batched(x_batch, ffn_norm, x_norm_batch, ...)` after the
     attention residual; the residual add is currently a separate write
     into `x_batch`. hipEngine's `qwen35_add_rmsnorm_fp16_kernel` collapses
     them. Single-kernel swap.
  3. **L9 — Prefill router GEMV token-tile (4 tokens per block) on the
     MoE router input** (S, +1-2% prefill from 4× weight reuse). hipfire
     uses `gemm_q8_0_batched_chunked`/`gemm_hfq4g256` which is a generic
     batched GEMV; hipEngine ships a router-specific token-tile variant
     `qwen35_router_logits_token_tile_kernel<scalar_t, 4>` that holds the
     `weight[expert × hidden_size]` row in registers and accumulates
     4 token dot-products per block.
- **Biggest cumulative prefill headroom by phase: MoE routed-expert
  gate_proj/up_proj** (T4 + L18 + L20 = chunkwise group sizing + per-arch
  WMMA tile sweep + grouped-stacked HGEMM thresholding). At 4K prefill
  hipEngine measures `gemm_awq_selected_dual_pack8_wmma_compact_kernel`
  alone at 200 ms of 6171 ms total kernel time, but hipfire's gfx1151
  path goes through `gemm_hfq4g256_moe_grouped_wmma_k2` and per
  `paroquant-a3b/hipengine-perf-comparison.md:38-40` is running at
  ~22% of WMMA roof. Three independent levers stack here. Second-biggest
  headroom is the **MoE routed-expert down projection** (similar story,
  see L19, L21, L24).

---

## Lever table

Format: ` Phase | Lever | Description | hipEngine impl | hipfire today | Effort | Impact% | Notes `

Effort: XS ≤2h · S ≤1d · M ≤1w · L >1w. Impact = rough prefill-only %
estimate on Qwen3.6-35B-A3B-PARO at pp256-pp4096; `?` = unmeasurable
without rocprof.

| Phase | Lever | What it does | hipEngine impl | hipfire today | Effort | Impact% | Notes |
|---|---|---|---|---|---|---|---|
| Embedding gather | **L1** Batched embedding lookup kernel | One launch dequants N rows directly into `[N×H]` instead of N×(lookup+copy) | `embedding_lookup_batch_fp16_i64` (`hipengine/runtime/qwen35_paro.py:42`) | **Already implemented** — `gpu.embedding_lookup_hfq4g256_batched(...)` at `crates/hipfire-arch-qwen35/src/qwen35.rs:5346` | n/a | 0% | Parity. Falls back to per-token loop for HFQ4G128/F32. |
| Input layernorm | **L2** Batched RMSNorm | One launch over N rows | `qwen35_rmsnorm_kernel` (`kernels/hip_gfx1100/norm/rmsnorm.hip:53`) | **Already implemented** — `gpu.rmsnorm_batched(...)` at `crates/hipfire-arch-qwen35/src/qwen35.rs:5489` | n/a | 0% | Parity. |
| Input layernorm | **L3** RMSNorm fused with AWQ pre-rotation | RMSNorm + per-channel AWQ scale + first rotation pass in one kernel | `qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16` (`kernels/hip_gfx1100/linear_attn/gdn.hip:612`) — partially landed as P3.1 in OPTIMIZE.md, REJECTED at 32K | **Already implemented** — `fused_rmsnorm_rotate_mq_batched_for(...)` at `qwen35.rs:5485` (FWHT variant); hipEngine version is Givens-rotation | n/a | 0% | hipfire's FWHT fuse is structurally different but equivalent. Parity. |
| LinearAttention QKV-Z proj | **L4** 4-way fused QKVZA WMMA | One kernel emits Q, K, V, Z, beta, alpha | hipEngine has separate `project_linear_attention_qkv_z` + dense A/B + `project_linear_attention_ab` (`qwen35_paro.py:991`, `2795`) — A/B is a row-GEMV pair (PREFILL.md P0 hypothesis) | **Already implemented and more aggressive** — `gemm_qkvza_q8_0_wmma` / `gemm_qkvza_hfq4g256` at `qwen35.rs:5497-5575` fuses W_qkv + W_z + W_beta + W_alpha in one kernel | n/a | 0% | Hipfire AHEAD here — this is the P0 hypothesis hipEngine still has open. |
| LinearAttention conv1d + SiLU + split | **L5** Fused conv1d + SiLU + Q/K/V split | One kernel does the depthwise conv1d, applies SiLU, and splits to qkv | `qwen35_linear_attn_conv_prefill_fp16` (`kernels/hip_gfx1100/linear_attn/conv.hip:135`) | **Already implemented** — `gpu.conv1d_silu_split_*_f32_n(...)` at `qwen35.rs:5620` | n/a | 0% | Parity. |
| LinearAttention conv1d + SiLU + split | **L6** Templated FP16-input segment conv | Remove the `fp16_to_f32(qkv)` cast and `qkv_f32` scratch for packed c>N | hipEngine candidate #4 in PREFILL.md "low-risk fusion audit"; DEFERRED to Lane S (c>N) | hipfire `conv1d_silu_split` runs on F32 — same cast cost | M | 0% on c=1 / +1-2% on c>N | Only meaningful when c>N batching lands; out of scope for current bench. |
| LinearAttention QK norm + scale | **L7** Fused QK head-RMSNorm + position RoPE | `qwen35_head_rmsnorm_partial_rotary_positions_kernel` does per-head RMSNorm + RoPE in one launch | `kernels/hip_gfx1100/rotary/qwen35_rotary.hip:279` | **Already implemented**, fused in the batched FA path — `gpu.rmsnorm_batched(&q,...)` + `rope_partial_interleaved_f32_batched` at `qwen35.rs:6058-6133` are still **two** launches, NOT one | XS | +0.5-1.5% | hipEngine fuses head-RMSNorm + RoPE; hipfire splits them. Single-launch fuse for the FA Q/K head-RMSNorm + partial-RoPE prelude. |
| LinearAttention beta/alpha gating | **L8** Fused sigmoid(beta) + alpha_gate | One kernel applies sigmoid to beta, alpha gate to alpha | hipEngine does this inside GDN recurrent kernel | **Already implemented** — `gpu.fused_sigmoid_alpha_gate_f32_batched(...)` at `qwen35.rs:5579` | n/a | 0% | Parity. |
| Gated DeltaNet inner-state update | **T2 (prior top-5)** Chunkwise / WY-chunkwise GDN | Replace strictly-sequential per-token loop with chunkwise recurrence (FLA/atlas reference) | hipEngine has NOT shipped — P1 in OPTIMIZE-DENSE.md:140-159 (largest expected leverage); current kernel is `qwen35_gdn_prefill_recurrent_k2_kernel` (`kernels/hip_gfx1100/linear_attn/gdn.hip:304`) | Same strictly-sequential loop — `kernels/src/gated_delta_net_q8.hip:56` `for (int t = 0; t < n_tokens; t++)` | L | +20-50% | Already in prior top-5. Confirmed: at 4K hipEngine measures 392ms / 6171ms = 6.3% of prefill; at A3B's 30 GDN layers it scales similarly. Both engines blocked here. |
| LinearAttention output norm + gate + projection | **L9** GDN-output-norm + SiLU-gate + W4-projection fusion | Avoid materializing `recurrent_bf16` between GDN tail and out_proj | hipEngine candidate #1 in PREFILL.md "low-risk fusion audit" — landed as `qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16` but P3.1 REJECTED (regressed at 32K, no scratch drop without larger planning) | hipfire: `silu_mul_dual_rotate_out_fp16` (`qwen35.rs:5439`) already does SiLU+rotate fusion for MoE; the GDN-out tail uses a separate `rmsnorm` + `silu_mul` chain | S | +0-1% | hipEngine's rejection suggests it's only a memory cleanup, not a speed lever, on gfx1100. Test on gfx1151 first. Synergy: needs to be checked AFTER the chunkwise GDN (T2) lands. |
| FullAttention QKV proj | **L10** 3-way fused QKV WMMA | One kernel emits Q, K, V | hipEngine `qwen35_paged_full_attn_prefill_gqa_gate_*` post-pass; FA QKV currently uses separate launches per parent ledger | **Already implemented** — `gpu.fused_qkv_hfq4g256(...)` at `qwen35.rs:7233-7239` and `qwen35.rs:7398` (`fused_gate_up`) | n/a | 0% | Parity / hipfire ahead. |
| FullAttention QKV proj | **L11** AWQ pack8 dual+single GEMV layout for FA Q/K/V (Marlin-K) | Repack-on-load + Marlin-K vec8 FMA inner loop | hipEngine D2.1 accepted: `gemv_awq_dual_pack8_kernel` (`paro_awq_gemv.hip:613`) ACCEPTED +5.6% decode | hipfire HFQ4G256 uses different storage; pack8 surfaces exist (`crates/rdna-compute/src/dispatch.rs:62`) but layout differs | M-L | +0% prefill / +5-8% decode | Already in prior top-5 (T4). Decode lever; prefill is GEMM-shaped already. |
| FullAttention RoPE | **L12** Per-row position RoPE | Each prompt row consumes its own position offset | `qwen35_head_rmsnorm_partial_rotary_positions_kernel` (`kernels/hip_gfx1100/rotary/qwen35_rotary.hip:279`) | **Already implemented** — `gpu.rope_partial_interleaved_f32_batched(...)` at `qwen35.rs:6133` | n/a | 0% | Parity. |
| FullAttention RoPE | **L13** Theta cache / position cache prewarming | Cache `(cos, sin)` tables per max-seq once at session-init | hipEngine uses on-the-fly `sincosf` in the rotary kernel (`paro_rotate.hip:209`) | hipfire computes per-call — `rope_partial_interleaved_f32_batched` runs sincos inline. Pre-cached `givens_cos`/`givens_sin` are used for the asym KV path (`qwen35.rs:5314-5318`), not for partial-RoPE | S | +0.5-1% | Cheap mem-bound win for the FA layers. Not a lever hipEngine has measured; flag as "uninvestigated." |
| FullAttention KV cache write | **L14** Batched paged KV write with append spans | One launch writes T prompt rows' K and V to paged cache | `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans` (`kernels/hip_gfx1100/attention/paged_kv_write.hip:519`) | **Already implemented** — `gpu.kv_cache_write_q8_0_batched(...)` at `qwen35.rs:6189` and asym variants at `:6166-6184` | n/a | 0% | Parity. hipfire uses contiguous `kv_cache.k_gpu[layer_idx]` instead of paged; same effective semantics for single-request prefill. |
| FullAttention KV cache write | **L15** Fused KV write + rotation (asym/fwht) | One kernel writes K/V to cache and applies the rotation in-place | `qwen35_write_paged_kv_int8_per_token_head_scale_*` family with int8 per-token-head | **Already implemented and more aggressive** — `gpu.kv_cache_write_asym3_batched(...)` / `fwht3_batched` etc. (`qwen35.rs:6166-6184`) fuses write + Givens/FWHT in one launch | n/a | 0% | Hipfire ahead. |
| FullAttention flash attention | **T5 (prior top-5)** AOTriton compact-varlen FA | Replace hand-rolled FA kernel with AOTriton 0.11.2b pretuned binaries above T=512 threshold | hipEngine `prefill_full_attention_aotriton_varlen_gqa_gate_fp16` (`qwen35_paro.py:2635`); `--attn-aotriton-min-tokens 512` default; measured +254% at T=4K (PREFILL.md:174) | hipfire uses custom `attention_flash_*_batched_masked` for all T | L | +0-200% at >512 tokens / 0% at pp256 | Already in prior top-5. AOTriton is gfx1100/gfx1151-only currently; pinned at 0.11.2b w/ BF16 head-dim 256 forward images (12 variants). Not portable to gfx9/gfx10. |
| FullAttention flash attention | **L16** Flash-Attention-2 with online softmax + GQA reuse | Tile Q in registers, stream K/V chunks through LDS, online softmax statistics, share K/V across GQA group | hipEngine `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` (`paged_attn_decode.hip:1284`) has the **non-FA pre-Flash design** (PREFILL.md:759-816). hipEngine has NOT shipped FA-2; deferred to D2.4 (3-6 weeks effort) | hipfire `attention_flash_*` kernels — check if they implement online softmax. Per `kernels/src/attention_flash.hip` they do tile-streaming. Likely ahead of hipEngine's native kernel | XS audit / L if needed | 0-5% | Investigate; if hipfire is already FA-2 internally, this lever is closed. |
| FullAttention flash attention | **L17** Shape-bucketed graph dispatch for split-K decode (long-context only) | Pick split-K count per (context bucket × num_splits) | hipEngine D3.1 ACCEPTED +42% at 32K / +106% at 128K (`docs/OPTIMIZE.md:303`); `_paged_attn_gqa_grouped_min_context` switch | hipfire flash-attention partials use a fixed split (`s.flash_partials` at `qwen35.rs:6219`) | M | +0% short / +30-100% at 32K-128K context | Pure-decode lever; not relevant to current pp256-pp4096 prefill bench. Flag for long-context users. |
| FullAttention output gate (sigmoid_mul) | **L18** Fused attention-output × sigmoid(gate) post-pass | Element-wise multiply attention output by sigmoid(gate FP16) — fused with output projection epilogue | hipEngine `paged_attn_decode.hip:1191` fuses sigmoid_mul into attention epilogue; AOTriton path requires a separate post-pass kernel | **Already implemented** — `gpu.sigmoid_mul_f32(...)` at `qwen35.rs:6321` is a separate kernel after attention. Hipfire could fuse this into the attention kernel epilogue (matching hipEngine's hand-rolled kernel) | S | +1-2% | Hipfire BEHIND on this. Fuse `sigmoid_mul` into the `attention_flash_*` kernel epilogue (matches the kernel that already loads attn_out). Synergy with L16. |
| FullAttention output projection | **L19** AWQ wo + residual fused (residual_gemm) | One kernel does `x_batch += W_o · attn_out_rot`, no scratch | hipEngine `awq_fusedw4_prefill_strided_fp16` + dedicated combine (`paro_awq_gemv.hip:1972`) | **Already implemented** — `gpu.gemm_q8_0_residual_wmma(...)` / `gemm_hfq4g256_residual` at `qwen35.rs:6342-6348` | n/a | 0% | Parity / hipfire ahead. |
| Post-attention layernorm | **L20** Fused add + RMSNorm | One kernel adds residual and applies RMSNorm | `qwen35_add_rmsnorm_kernel` (`kernels/hip_gfx1100/norm/rmsnorm.hip:85`) | **Partially behind** — hipfire's post-attention is `gpu.gemm_hfq4g256_residual` (writes residual sum into `x_batch`) followed by `gpu.rmsnorm_batched(&x_batch, ffn_norm, &x_norm_batch, ...)` at `qwen35.rs:4972`. Two passes over `[N×H]`. | XS | +0.5-1.5% | **Quick win #2 in summary.** One `add_rmsnorm_batched` kernel that reads `x_batch` once, computes mean/var, applies norm, writes `x_norm_batch`. Hipfire has fused `rmsnorm_rotate_mq_batched_for`; add a sibling for the post-attn case. |
| MoE router GEMV | **L21** Token-tiled router GEMV (4 tokens per block, weight-in-reg) | `weight[expert × hidden]` is loaded once per block, used for 4 token dot-products | `qwen35_router_logits_token_tile_kernel<scalar_t, 4>` (`kernels/hip_gfx1100/moe/router.hip:81`) — measured 8.62 ms / 65.39 ms at 512/4K = 4.5% prefill kernel time | hipfire uses generic `gemm_q8_0_batched_chunked` (`qwen35.rs:4983`) — no weight-reuse across tokens beyond LDS staging | S | +1-2% | **Quick win #3 in summary.** Direct port of `qwen35_router_logits_token_tile_kernel` to hipfire's MoE router GEMV path. Q8_0 weights stride matches BF16. Add as `gemv_q8_0_token_tile_k4_batched`. Lower risk than chunkwise GDN. |
| MoE top-k + softmax + weight normalization | **L22** Batched softmax + top-K + renorm in single launch chain | Same as decode but flattened over N rows | hipEngine: `qwen35_router_select_kernel` (`router.hip:316`) for c=1; cooperative `qwen35_router_topk_shared_coop_out_kernel` was D1.5 REJECTED | **Already implemented** — `gpu.softmax_f32` + `gpu.moe_topk_renorm_k8_batched` at `qwen35.rs:5040-5044` (split into 2 kernels; matches CPU reference) | XS | +0-0.3% | hipEngine's coop fold REGRESSED (`-0.93%` at 512/128). Don't pursue unless rocprof shows router select is multi-percent. |
| MoE shared-expert gate | **L23** Prefill-only fused router top-K + shared-gate sigmoid | Avoid recomputing sigmoid(shared_gate_logit) downstream by writing the sigmoid'd value during top-K selection | hipEngine candidate #2 in PREFILL.md "low-risk fusion audit"; HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED diagnostic — P3.2 REJECTED (`+0.21%/-0.23%`) | Hipfire applies sigmoid(shared_scalar) inside `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` (`qwen35.rs:5063`) — already fused inside down combine | n/a | 0% | Parity / hipfire ahead (fuses sigmoid into the down kernel, not into router). |
| MoE shared-expert gate_proj/up_proj | **L24** Bulk shared-expert gate+up SiLU token-tiled | One kernel applies gate, up, SiLU+mul over many tokens with weight reuse | `w8a16_shared_gate_up_silu_fp16_token_tiled_kernel<2|4>` (`kernels/hip_gfx1100/quant/w8a16_linear.hip:268`) — P1.2 ACCEPTED +0.5-2.2% | hipfire shared expert uses `gemm_gate_up_hfq4g256` (`qwen35.rs:5012`) — already a fused gate+up GEMV with batched N axis; no token-tile beyond LDS | S | +1-3% | **Quick win #1 in summary.** Add `gemm_gate_up_hfq4g256_token_tile_k4` (load each weight row once, accumulate 4 token dot-products). Lower risk than chunkwise GDN; smaller blast radius than the routed-expert work. Hipfire's MQ4 advantage extends here. |
| MoE shared-expert gate_proj/up_proj | **L25** Cast-glue cleanup (FP16-direct shared expert) | Remove FP16↔F32 casts around shared expert | hipEngine P2.1 — diagnostic-only, perf-neutral, accepted as memory cleanup | Hipfire `gemm_gate_up_hfq4g256` already writes FP16 path; `shared_gate`/`shared_up` are F32 (`qwen35.rs:4943`) — possible cast at output if downstream `silu_mul` is F32. Audit needed. | XS audit / S fix | 0-1% | Per-hipEngine evidence this is mostly a memory cleanup. Audit only — don't open a multiloop. |
| MoE shared-expert down proj | **L26** Bulk shared-expert down + sigmoid + combine fused | One kernel does W_down · rot + sigmoid(shared_scalar) × (W_down · rot) + residual add | `w8a16_shared_down_combine_residual_fp16_token_tiled_kernel<2|4>` (`w8a16_linear.hip:543`) — P1.3 ACCEPTED +0.9% | **Already implemented** — `gpu.gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched(...)` at `qwen35.rs:5063` fuses W_down + sigmoid + scaled residual | n/a | 0% | Parity. |
| MoE shared-expert down proj | **L27** Token-tile variant of L26 | Same kernel with token-tile weight reuse | Same as L24/L26 token-tile pattern | hipfire's `gemv_hfq4g256_residual_sigmoid_scaled_gpu_batched` is row-GEMV — no token-tile | S | +1-3% | Like L24 but for the down side. Same effort/payoff profile. |
| MoE routed-expert dispatch | **L28** Device-side scatter/sort/gather metadata | One fused kernel does histogram → prefix → permute for top-k routing | hipEngine `qwen35_moe_group_count` + `_prefix` + `_scatter_gather` series (`moe/group_scatter.hip:9-184`); P3.3 DEFERRED (≤0.27% of prefill) | **Already implemented and more fused** — `gpu.moe_scatter_fused_k8(...)` at `qwen35.rs:5127` replaces histogram + offsets + permute in ONE launch (hipEngine takes 3-5) | n/a | 0% | Hipfire AHEAD. The dtoh-sync elimination via `m_total_max` upper bound (`qwen35.rs:5135`) is also a unique hipfire optimization not in hipEngine. |
| MoE routed-expert gate_proj+up_proj | **T3 (prior top-5) + L29** Multi-rotation fused launch | Read input X once, emit N different rotated bases for downstream consumers via grid.z plane | hipEngine `paro_rotate2_kernel` / `paro_rotate3_kernel` (`rotary/paro_rotate.hip:163-329`) | hipfire's FWHT-based rotations are structurally different (no Givens pair tables); equivalent fusion would require redesigning the rotation as a multi-output kernel | M | +3-7% decode / +1-3% prefill | Already in prior top-5. **Subtlety the prior pass missed:** hipfire's FWHT rotations are row-local pair-sum (`fused_silu_mul_rotate_mq_batched_for` at `qwen35.rs:5052`); the equivalent "rotate2" port would be a kernel that writes two FWHT-rotated outputs (different sign patterns / different `pairs` tables) reading X once. Same structural pattern, different math. |
| MoE routed-expert gate_proj+up_proj | **L30** Grouped-compact WMMA tile map | Map each WMMA 16x16 tile to a single expert via `tile_expert[tile_id]` lookup; allows asymmetric expert occupancies | `qwen35_moe_wmma_tile_map_kernel` + `gemm_awq_selected_dual_pack8_wmma_compact_kernel` (`wmma/paro_awq_wmma.hip:60`); P1.4 retained at `WMMA_MIN_TOKENS=2` | **Already implemented** — `gpu.gemm_hfq4g256_moe_grouped_wmma_k2(...)` at `qwen35.rs:5144` with same scatter/tile_ids pattern | n/a | 0% | Parity. hipfire's gfx12 path adds `gemm_hfq4g256_moe_grouped_wmma_k2_gfx12` for the WMMA w32_gfx12 builtin (`qwen35.rs:5087`). |
| MoE routed-expert gate_proj+up_proj | **L31** WMMA tile sweep (`tile_m` × `tile_n` × `launch_bounds`) | Sweep `(16,16)`, `(16,32)`, `(32,16)`, `(32,32)` with `__launch_bounds__` per arch | hipEngine P4.2 DEFERRED (lessons-learned: 1-3% noisy) | hipfire `gemm_hfq4g256_moe_grouped_wmma_k2` is at fixed `tile_m=16` per `kernels/src/gemm_hfq4g256_moe_grouped_wmma_k2.hip:43-100` | M | +1-5% | The 22%-of-roof gap from `hipengine-perf-comparison.md:38-40` suggests this lane is open. RDNA 3.5 (gfx1151) has different LDS/VGPR than gfx1100 — sweep needed. Synergy: pair with chunkwise GDN (T2) which shifts kernel time-shares. |
| MoE routed-expert gate_proj+up_proj | **L32** Grouped-stacked HGEMM (rocBLAS/hipBLASLt) above token threshold | Use `hipblasLtMatmul` for grouped MoE projection at T ≥ 1K | hipEngine Phase 2 in PREFILL.md:916-925 (PROPOSED, not landed; expected +15-25% at 4K) | Not present — hipfire commits to custom WMMA paths only | L | +5-15% at pp ≥ 1K | hipEngine's open lever; would need rocBLAS/hipBLASLt FFI bindings (hipfire has none). Per CLAUDE.md "No Python in inference hot path" — hipBLASLt is fine (C ABI). Out of scope for now. |
| MoE routed-expert SiLU+mul | **L33** SiLU·mul + rotate fused into single kernel | One kernel reads gate, up, applies SiLU·mul, writes rotated down-input | `silu_mul_dual_rotate_out_fp16` + `silu_mul_pair_rotate_out_fp16` (`fused/paro_silu.hip:87-197`) | **Already implemented** — `fused_silu_mul_rotate_mq_batched_for(...)` at `qwen35.rs:5175` | n/a | 0% | Parity. |
| MoE routed-expert down_proj | **L34** Grouped-WMMA down with non-atomic combine | Down projection runs via grouped WMMA; combine reuses the inverse-perm path (no atomicAdd contention) | hipEngine `gemm_awq_selected_pack8_wmma_compact_kernel` (`wmma/paro_awq_wmma.hip:181`) | **Already implemented** — `gpu.gemm_hfq4g256_moe_grouped_wmma_k2(...)` for down at `qwen35.rs:5203` (Path 2) and atomicAdd Path 1 fallback | n/a | 0% | Parity. hipfire's Path 2 is the equivalent. |
| MoE routed-expert down_proj | **L35** Down WMMA tile retune for gfx1151 | RDNA 3.5 LDS/VGPR budget differs from gfx1100 | hipEngine P4.2 DEFERRED for gfx1100 | Same as L31 — sweep candidate | M | +1-5% | Pairs with L31. |
| MoE residual accumulation | **L36** Weighted-lane sum into shared-gate-combined residual | One kernel does `out += sum(weight[lane] × down_out[lane])` + `+= sigmoid(shared_gate) × shared_out` + residual | `weighted_lanes_sum_out_fp16_f32w` + `shared_gate_combine_residual_batch_out_fp16` (`fused/paro_combine.hip:101-184`) | **Already implemented** — `gpu.moe_gate_up_unscatter_k8` (`qwen35.rs:5159`) + the sigmoid-scaled residual GEMV — different decomposition, same effect | n/a | 0% | Parity. |
| LM head GEMV | **L37** GPU FP16 lm_head + argmax fused | Fused W_head GEMV + on-device argmax | hipEngine `lm_head_fp16_logits_kernel` + `argmax_stage1_kernel` + `argmax_stage2_kernel` (`kernels/hip_gfx1100/linear/lm_head.hip:23-105`); `gpu_fp16_argmax` is default per `qwen35_paro.py:206` | Hipfire downloads F32 logits to host then calls `llama::argmax` (`crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:417`). LM head itself uses GPU `weight_gemv(&weights.output, ...)` (`qwen35.rs:7182`) but argmax is host-side | S | +0.1-0.5% prefill (only last token), +0.5-1.0% decode | Prefill only samples 1 token → mostly invisible at pp256. But moving argmax to device removes a sync. Sample tooling change. |
| LM head GEMV | **L38** LM head AWQ pack8 (Marlin-K) | LM head W4 GEMV via pack8 | hipEngine D2.1 includes lm_head in the Marlin-K kernel sweep | hipfire's `weight_gemv` resolves to MQ4/HFQ4 paths via the standard dispatch | M | +0% prefill / +2-5% decode | Decode lever; lm_head is one row in prefill. |
| Sampling / argmax | **L39** Device-side argmax | On-GPU argmax on logits, host gets one int | `argmax_stage1_kernel` + `argmax_stage2_kernel` (`linear/lm_head.hip:67,105`) | Host argmax (`bench_qwen35_mq4.rs:417`) | XS | +0% prefill / +0.5-1% decode | Same as L37, separately costed for clarity. |
| Sampling / argmax | **L40** CPU numpy argmax fallback path | Allow tooling to keep CPU argmax for diagnostics | hipEngine `cpu_numpy_argmax` option (`qwen35_paro.py:206`) | hipfire's only path is CPU argmax — already aligned to the "fallback" mode | n/a | 0% | Parity at the fallback level; gap is at the default level (L37/L39). |

### Cross-cutting / methodology levers (not phase-specific)

| Phase | Lever | What it does | hipEngine impl | hipfire today | Effort | Impact% | Notes |
|---|---|---|---|---|---|---|---|
| All | **L41** Prefill-time hipGraph capture | Capture the bulk prefill kernel chain into a graph, replay across prefill chunks | hipEngine deferred per PREFILL.md:583-589 ("do not chase graph capture before native kernels are in place") | hipfire has graph infra (`forward_prefill_batch_single_chunk_captured` at `qwen35.rs:4239`) but it captures only single chunk; not the full prefill | M | +2-8% | Subtlety: hipfire ALREADY has prefill graph for the single-chunk case, but for `n=256` it's one chunk so the capture overhead may dominate. Useful at larger pp where multiple chunks fire. |
| All | **L42** `-mllvm -amdgpu-unroll-threshold-local=600` build flag | Per-kernel loop unrolling threshold | hipEngine P1.5/W.1 ACCEPTED as neutral default; was +166% on llama.cpp HIP | Audit needed on hipfire's build flags | XS | 0-5% | Cheap to test via `cargo` build flag. Per CLAUDE.md `Δ ≥ 5%` rule, run probe_commits.sh before claiming. |
| All | **L43** `-mcumode` per-kernel build profile | CU-mode codegen for hot kernels | hipEngine `hipengine/core/build.py:47` default | Audit needed | XS | 0-2% | hipEngine P1.6 REJECTED at the per-prefill-kernel level (`benchmarks/results/...-p16-prefill-mcumode-rejected.json`). |
| All | **L44** Pre-cached cos/sin tables for partial-RoPE | Avoid `sincosf` recomputation in the rotary kernel | hipEngine partial-RoPE kernel uses on-the-fly `sincosf` (`rotary/paro_rotate.hip:209` — Givens variant) | hipfire pre-computes `givens_cos`/`givens_sin` for asym KV (`qwen35.rs:5314`) but partial-RoPE in `rope_partial_interleaved_f32_batched` is on-the-fly | S | +0.5-1% | Cheap; pre-cache `cos(pos × theta_i)` / `sin(...)` per max-seq. Cross-cuts L13. |
| All | **L45** rocprofv3 `--kernel-trace` Amdahl table for hipfire | Per-kernel cost table by share-of-prefill | hipEngine M.3/M.4 — `benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json` | hipfire bench emits `--emit-atlas` JSONL but no rocprof attribution | S | n/a | **Methodology lever, not a perf lever directly.** Without this, every prefill % above is an estimate. CLAUDE.md "Δ ≥ 5%" rule explicitly mandates this; per `hipengine-perf-comparison.md` win #3 this is the cheapest action to redirect tuning. |
| All | **L46** Auto-tuned chunk sizes (linear/MoE/full-attn) per shape | `PrefillConfig.{linear,moe,full_attn_query}_chunk_size` autotune | hipEngine P5.2 ACCEPTED — `<32K unchunked, 32K=1024/4096, ≥128K=8192 query` | hipfire `PREFILL_MAX_BATCH=256` (constant per `qwen35.rs:4474-4485`) | S | +0% pp256 / +5-10% pp ≥ 32K | Long-context lever; out of scope for current bench. |
| All | **L47** Cast-glue cleanup (FP32 → FP16 path everywhere) | Avoid intermediate FP32 → FP16 → FP32 round-trips | hipEngine P2.1 ACCEPTED as memory cleanup | hipfire path uses F32 for activations + FP16 KV; audit needed for stray casts | XS audit / S fix | 0-1% | Likely memory savings, not perf. |
| All | **L48** Diagnostic: `finite_prefill_logits` + generated-sample equality in bench | Gate any retained perf row on NaN-clean logits and matching token stream | hipEngine LESSONS-LEARNED:262 — "no perf number meaningful unless `finite_prefill_logits` is true" | hipfire `coherence-gate.sh` covers correctness; finite-logit check unclear if explicit | XS | n/a | **Critical for trusting any of the above wins.** hipfire's coherence gate already detects attractors; ensure NaN/inf logit check is in the path. |

---

## Phase summary — cumulative headroom

| Phase | Levers (closed=parity) | Open levers | Estimated cumulative prefill % |
|---|---:|---|---:|
| Embedding | 1 closed (L1) | — | 0% |
| Input layernorm | 2 closed (L2, L3) | — | 0% |
| LA QKV-Z proj | 1 closed (L4) | — | 0% |
| LA conv1d+SiLU+split | 1 closed (L5), 1 deferred (L6) | L6 (c>N only) | 0% |
| LA QK norm + scale | — | **L7 (fuse RMSNorm+RoPE)** | +0.5-1.5% |
| LA beta/alpha gating | 1 closed (L8) | — | 0% |
| **GDN inner state** | — | **T2 (chunkwise)** | **+20-50%** |
| LA output norm+gate+proj | — | L9 (mem cleanup only on gfx1100) | +0-1% |
| FA QKV proj | 2 closed (L10), 1 decode-only (L11/T4) | — | 0% prefill |
| FA RoPE | 1 closed (L12) | **L13 (cos/sin cache)** | +0.5-1% |
| FA KV write | 2 closed (L14, L15) | — | 0% |
| **FA flash attention** | — | **T5 (AOTriton), L16 (FA-2 audit), L17 (long-ctx)** | **+0-200% at T>512** |
| FA output gate | — | **L18 (sigmoid_mul fuse)** | +1-2% |
| FA output proj | 1 closed (L19) | — | 0% |
| **Post-attn layernorm** | — | **L20 (add+rmsnorm fuse — quick win #2)** | **+0.5-1.5%** |
| **MoE router GEMV** | — | **L21 (token-tile — quick win #3)** | **+1-2%** |
| MoE topk+softmax+renorm | 1 closed (L22), 1 rejected (L23) | — | 0% |
| **MoE shared-expert gate_proj/up_proj** | — | **L24 (token-tile — quick win #1)**, L25 (cast audit) | **+1-3%** |
| MoE shared-expert down | 1 closed (L26) | **L27 (down token-tile)** | +1-3% |
| MoE routed-expert dispatch | 1 closed (L28, hipfire ahead) | — | 0% |
| **MoE routed-expert gate+up** | 1 closed (L30) | **T3/L29 (multi-rotate), L31 (WMMA tile sweep), L32 (HGEMM)** | **+5-15%** |
| MoE routed-expert SiLU+mul | 1 closed (L33) | — | 0% |
| **MoE routed-expert down** | 1 closed (L34) | **L35 (gfx1151 tile retune)** | **+1-5%** |
| MoE residual accumulation | 1 closed (L36) | — | 0% |
| LM head GEMV | — | L37, L38, L39 (decode-side) | +0-0.5% prefill |
| Sampling | 1 closed (L40 path) | L37/L39 (default-on GPU argmax) | +0% prefill |
| All / cross-cut | — | L41 (graph), L42-L44 (build flags), L45-L48 (methodology) | +0-10% |

**Open prefill % range (excluding T2/T5 which are bimodal): +5-25%** on
gfx1151 / Qwen3.6-35B-A3B-PARO at pp256-pp4096, mostly concentrated in
MoE routed-expert (T3 + L31 + L32 = +6-25%) and 3 small quick wins
(L17 + L20 + L21 = +2.5-6.5% combined, ≤3 days work total).

**Open prefill % range including T2:** add **+20-50%** for chunkwise
GDN. T2 alone exceeds every other lever combined.

**Open prefill % range including T5:** add **+0-200%** at T>512 from
AOTriton; ~0% at pp256.

---

## Anti-rabbit-hole list (lessons from hipEngine — DO NOT chase)

From `LESSONS-LEARNED.md` and OPTIMIZE.md "Do not chase":

- **`sudot4`/dp4a over PARO/AWQ layout** — 3.92-9.72× slower than tuned FMA.
- **LDS staging as default** — RDNA3 barrier/occupancy costs > reuse.
- **Multi-step graph replay** — no reliable gain, diverged at token 581 on parent.
- **Thread-count sweeps without source/profile justification** — many regress.
- **Fusion that abandons pack8/repacked layout** — saves 1 launch, loses more on layout.
- **`finite_prefill_logits=false` perf rows** — speed came from NaN propagation collapsing MoE routing.
- **Wave64 `-mwavefrontsize64` for WMMA kernels** — conflicts with `_w32` builtins (per OPTIMIZE-DENSE.md:304-308).
- **`hipBLASLt` default** — rocBLAS beat hipBLASLt on tested BF16 GEMM on W7900. Maybe different on gfx1151; test before assuming.
- **Output-buffer-only fusion under graph replay** — sub-1% per L20-class fusion; ONLY count wins that change arithmetic/data reuse.

These should be reflected in any hipfire planning doc that touches the
corresponding lever.

---

## Files of interest (hipEngine snapshot)

- `/tmp/hipengine-survey/docs/PREFILL.md:244-279` — low-risk prefill fusion audit
- `/tmp/hipengine-survey/docs/PREFILL.md:702-731` — measured 512/4K prefill Amdahl tables
- `/tmp/hipengine-survey/docs/OPTIMIZE.md:175-225` — Lane M.3 prefill Amdahl + Lane P1-P5 candidate rows
- `/tmp/hipengine-survey/docs/OPTIMIZE-DENSE.md:140-178` — chunkwise GDN evidence + lane order
- `/tmp/hipengine-survey/docs/LESSONS-LEARNED.md:177-200` — top retained MOE2 wins
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/moe/router.hip:81` — token-tile router GEMV (L21)
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip:268,543` — token-tile shared-expert (L24, L26)
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip:85` — `add_rmsnorm_kernel` (L20)
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.hip:279` — fused head-RMSNorm + RoPE (L7)
- `/tmp/hipengine-survey/hipengine/kernels/hip_gfx1100/linear/lm_head.hip:67-105` — argmax stage1/2 (L37, L39)

## Files of interest (hipfire, this branch)

- `crates/hipfire-arch-qwen35/src/qwen35.rs:4927-5273` — `prefill_moe_ffn_body_batched`
- `crates/hipfire-arch-qwen35/src/qwen35.rs:5273-7200` — `forward_prefill_chunk` per-layer loop
- `crates/hipfire-arch-qwen35/src/qwen35.rs:6052-6321` — batched FA prelude (split L7 candidates)
- `crates/hipfire-arch-qwen35/src/qwen35.rs:7202-7437` — `run_fa_layer_body` per-token fallback
- `kernels/src/gated_delta_net_q8.hip:56-86` — GDN serial-per-token (T2)
- `kernels/src/moe_topk_renorm_k8_batched.hip` — batched top-k renorm (L22 parity)
- `crates/rdna-compute/src/dispatch.rs:62-603,11143-11401` — pack8 GEMV surfaces, topk-renorm
- `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs:1-200` — bench config + argmax (L39)
