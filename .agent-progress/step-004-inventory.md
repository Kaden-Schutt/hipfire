# STEP-004: Architecture × Forward-Entry-Point Inventory

Parity baseline for Step/manifest adoption. Every architecture registered in
the runtime (`arch_id` → carrier mapping, see `crates/hipfire-runtime/src/safetensors_source.rs`
`derive_arch_id` and the `Architecture` trait in `crates/hipfire-runtime/src/arch.rs`)
with its forward entry points and Step coverage, as of STEP-004.

## Inventory

| arch_id | Family | Forward entry point | File | Step status | Exception |
|---------|--------|---------------------|------|-------------|-----------|
| 0 | LLaMA | `forward_scratch_layers_lowered` (decode) | `crates/hipfire-arch-llama/src/llama.rs` | **Full** — `dense_forward` → `execute_steps` | — |
| 0 | LLaMA | `forward_prefill_batch` (prefill) | `crates/hipfire-arch-llama/src/llama.rs` | **Partial** — `execute_steps` for projections; KV ladder bespoke in tree/capture paths | Prefill capture path: graph-capture exception |
| 0 | LLaMA | `forward_scratch_compute` (single-GPU whole-stack) | `crates/hipfire-arch-llama/src/llama.rs` | Routes through `forward_scratch_layers_lowered` when `llama_forward_lowered_enabled()` (default ON) | — |
| 0 | LLaMA | PP via `PpModel::forward_token` | `crates/hipfire-runtime/src/pp_serve.rs` | **Full** — `build_layer_steps` → `execute_steps_tp` | — |
| 0 | LLaMA | TP via `TpServed::forward_token` | `crates/hipfire-runtime/src/tp_serve.rs` | **Full** — `build_layer_steps` → `execute_steps_tp` | — |
| 1 | Plain Qwen3 | Same as LLaMA (carried via LLaMA carrier) | `crates/hipfire-arch-llama/src/llama.rs` | **Full** | — |
| 5 | Qwen35 dense | `forward_scratch_layers_lowered` (decode) | `crates/hipfire-arch-qwen35/src/qwen35.rs` | **Full** — `run_layer_program` (SuperOp substrate) | — |
| 5 | Qwen35 dense | `forward_scratch_layers_inner` (hand decode) | `crates/hipfire-arch-qwen35/src/qwen35.rs` | **Full** — individual `execute_steps_mesh` calls via helpers; qk-norm/RoPE stay direct gpu calls (no Step twin for partial-interleaved RoPE; qk-norm has `Step::QkNorm` but the hand path predates it) | partial-interleaved RoPE (see exception 10) |
| 5 | Qwen35 dense | `forward_prefill_chunk` (prefill) | `crates/hipfire-arch-qwen35/src/qwen35.rs` | **Partial** — DeltaNet/MoE/attention Steps; batched projections/norm bespoke in chunk body | Prefill chunking/abort/checkpoint control flow (exception 8) |
| 5 | Qwen35 dense | `forward_scratch_layers_multi` (PP decode) | `crates/hipfire-arch-qwen35/src/qwen35.rs` | **Migrated in STEP-004 Inc 2** — QKVZA/QKV/gate-up via `*_via_execute_steps`, attention via `kv_cache_attention_dispatch` (with per-device givens override); DeltaNet recurrent + MoE FFN already Step-based | — |
| 5 | Qwen35 dense | `forward_ep` (EP decode) | `crates/hipfire-arch-qwen35/src/qwen35.rs` | **Full** — `run_layer_program_ep` (lowered substrate) | — |
| 6 | Qwen35 MoE | Same paths as arch 5 | `crates/hipfire-arch-qwen35/src/qwen35.rs` | Same status | — |
| 7 | Qwen2/VibeThinker | `forward_step_after_x` / `forward_step_after_x_lowered` | `crates/hipfire-arch-qwen2/src/qwen2.rs` | **Full** — individual `execute_steps_mesh` calls; lowered path uses `dense_forward` | — |
| 8 | dots.ocr | `forward_step` (vision + Qwen2 decoder) | `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs` | **Bespoke** | Vision exception (exception 1) — AXIS-004/VL-002 scope |
| 9 | DeepSeek4 | `forward_ep` / `forward_tp` (EP/TP) | `crates/hipfire-arch-deepseek4/src/forward.rs` | **Partial** — MoE via `execute_lowered_moe`; MLA/compressor/indexer/RoPE-tail are `EscapeKind` | MLA exception (exception 3) |
| 9 | DeepSeek4 | single-GPU decode (various) | `crates/hipfire-arch-deepseek4/src/forward.rs` | **Partial** — same MoE + MLA pattern | MLA exception (exception 3) |
| 10 | MiniMax | `decode_step_body` (decode) | `crates/hipfire-arch-minimax/src/forward.rs` | **Mostly Step** — QKV/attention/o_proj via `execute_steps`; MoE via `lower_moe_steps` + `execute_lowered_moe`; QK-norm via `Step::QkNorm`, FFN-norm + head via `Step::RmsnormAutomatic`/`Step::Gemv` (Inc 5) | partial-interleaved RoPE (exception 10); embedding/head-prefix (exception 4); decode-path KV write + attention direct (capture-geometry; `Step::Attend` used on EP/TP paths via `minimax_attn_block`) |
| 10 | MiniMax | `forward_ep` / `forward_tp` (EP/TP) | `crates/hipfire-arch-minimax/src/forward.rs` | Same pattern as single-GPU | Same |
| 11 | LFM2-MoE | `decode_step_layers_and_head` (decode) | `crates/hipfire-arch-lfm2moe/src/forward.rs` | **Full** — norm/projections/qk-norm/RoPE/dense FFN/head via Steps (dtype-driven FWHT rotation); MoE routing via `Step::ScoreActivation` + `Step::MoeRoute`; routed expert phases via `lower_moe_steps` + `execute_lowered_moe` (manifest-born `ExpertGroupPlan`, `sigmoid_topk` + `indexed_quantized`, Single) | Conv mixer (exception 5); embedding (exception 4) |
| 12 | Cohere2-MoE | `decode_step_body` (decode) | `crates/hipfire-arch-cohere2moe/src/forward.rs` | **Full** — parallel-block norm + QKV + dense FFN + router + head via Steps; routed expert phases via `lower_moe_steps` + `execute_lowered_moe` (empty router phase — the sigmoid+topk ran bespoke before the program) | interleaved RoPE (exception 6); per-expert GEMV fallback (exception 7); sigmoid+topk routing (exception 9) |
| — | Qwen35-VL | VL-specific prefill + shared post-prefill | `crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs` | **Bespoke** | Vision exception (exception 2) — VL-001 scope |

## Justified non-decoder exceptions

1. **dots.ocr (arch 8)** — entire forward is vision-specific. "Vision+OCR pipeline — not a standard decoder path." AXIS-004/VL-002 scope.
2. **Qwen35-VL** — VL-specific prefill. "Vision-conditioned prefill — shares post-prefill lifecycle but image processing is bespoke." VL-001 scope.
3. **DeepSeek4 MLA/compressor/indexer** — `EscapeKind` ops in the SuperOp substrate. "MLA latent compression, indexer KV gather, and tail-YaRN RoPE are architecturally unique to DeepSeek V4; handled via `EscapeKind` in the SuperOp substrate, not standard Step variants."
4. **Embedding lookup** (all arches) — pre-decoder. "Embedding lookup is a pre-decoder token-to-activation map; not a decoder layer op."
5. **LFM2 Conv mixer** — `SuperOpKind::Conv` handles it in the lowered substrate. Per-call Step path exception: "Causal convolution mixer is stateful and architecturally unique; handled via `SuperOpKind::Conv` when lowered."
6. **Cohere2 interleaved RoPE** — no `Step::Rope` variant for interleaved rotation. "Interleaved RoPE (pairs 2i/2i+1) has no Step variant; `Step::Rope` dispatches rotate_half only. AXIS-003 can add `RopeInterleaved` if Cohere2 needs mesh dispatch."
7. **Cohere2/LFM2 per-expert GEMV fallback** — non-indexed dtypes. "Per-expert GEMV fallback for BF16/Q8/F16 experts uses a CPU-side loop over selected experts; no indexed-MoE Step for these dtypes."
8. **Qwen35 prefill batched path** — "Prefill chunking/abort/checkpoint control flow is request-lifecycle, not a decoder op." (`Step::Gemm` / `Step::GemmKeyedBatched` exist; the chunk body builds batched projections inline.)
9. **Cohere2 sigmoid+topk routing** — "Cohere2 sigmoid+topk routing has no matching Step variant; `MoeRoute` requires gate_bias, `MoeSoftmaxTopK` adds unwanted softmax."
10. **MiniMax/Qwen35 partial-interleaved RoPE** — "Partial-interleaved RoPE (partial_rotary_factor + interleaved pairs) has no Step variant." (`gpu.rope_partial_interleaved_f32` / `rope_partial_interleaved_f32_batched`.)
11. **LFM2/Cohere2 routed MoE expert phases — RESOLVED in follow-up.** Both carriers now ship the manifest machinery (`weight_manifest` with `ExpertSharded` packed-fused surrogates, policy-aware `expert_group_manifest` with `sigmoid_topk` + `indexed_quantized`, model-owned config-keyed group-plan caches) and the forward paths lower the expert phases through `lower_moe_steps` + `execute_lowered_moe` (Single). LFM2's router phase carries `ScoreActivation` + `MoeRoute`; Cohere2's router phase is empty (its norm_topk_prob=false top-k has no Step variant — the bespoke sigmoid+topk runs before the program, exception 9). Both verified byte-identical to the pre-machinery direct kernel sequence on GPU (LFM2 8B-a1b + 350m, Cohere2 North-Mini-Code-1.0). Remaining: Cohere2's per-expert fallback loop (BF16/Q8/F16) stays direct (exception 7).

## Status

- STEP-004 increments 2–5 migrate every row flagged **Migrated in STEP-004 Inc N** (Qwen35 PP decode, LFM2 decode, Cohere2 decode, MiniMax decode completion).
- All migrations verified parity-preserving: PP=2 (qwen35) byte-identical pre/post migration; LFM2 (350m.mq4 / 350m.q8 / 8B-a1b), Cohere2 (North-Mini-Code-1.0.mq4), MiniMax (M2.7.mq2) deterministic decode token sequences identical pre/post migration (baselines in `.agent-progress/step-004-*-baseline.txt`).
- Remaining bespoke rows are all covered by a justified exception (column 6) or an explicit later axis (AXIS-002 routed-MoE manifest projection, AXIS-003 lowering, AXIS-004 VL).
- Inc 7 deleted the bespoke code replaced by the Qwen35 PP migration (`run_fused_qkvza_scalar_key`, `scalar_qkvza_key` + its obsolete dispatch-pinning test). The LFM2 dead "mirror" block helpers (`conv_mixer_block` etc., zero callers crate-wide) are pre-existing dead code left in place — user decides on removal.
- Pre-existing finding (NOT introduced by STEP-004, RESOLVED in follow-up): emulated pp=2 vs pp=1 on qwen3.5-0.8b.mq4 diverged at decode step 1 — root cause: `DeltaNetState::new_with_quant_multi` did not wire the error-feedback residual (see Follow-ups #4). The `pp_parity` cargo test now PASSES under `HIPFIRE_EMULATE_GPUS=2`. Physical-GPU PP parity (HW-004, external token-58 divergence on gfx1201 R9700s) remains BLOCKED and is not claimed closed by this fix.

## Follow-ups (tracked todos)

1. ~~**Add manifest machinery to LFM2/Cohere2 carriers**~~ — DONE: `weight_manifest` (`ExpertSharded` packed-fused surrogates, projection under Single), policy-aware `expert_group_manifest` (`sigmoid_topk` + `indexed_quantized`), model-owned config-keyed `moe_group_plans` caches, `state_manifest` (Conv/Kv). CPU tests pin the MoE-span plan resolution per arch.
2. ~~**Migrate LFM2/Cohere2 MoE expert phases to `lower_moe_steps` + `execute_lowered_moe`**~~ — DONE: LFM2 (router phase `ScoreActivation`+`MoeRoute`) and Cohere2 (empty router phase; bespoke routing stays, exception 9) both lower through the sealed Single executor; GPU parity byte-identical on LFM2 8B-a1b/350m and Cohere2 North-Mini-Code-1.0. Cohere2's per-expert fallback (BF16/Q8/F16) remains direct (exception 7). TP/EP resolution of the same manifests is the AXIS-002 continuation (MiniMax sealed exact-policy cache when policy threading lands).
3. **Decide fate of the LFM2 dead "mirror" block helpers** (`conv_mixer_block`, `attn_mixer_block`, `dense_gate_up_block`, `dense_down_block`, `moe_ffn_block` — zero callers crate-wide, pre-existing dead code; now stale vs the migrated live path).
4. ~~**Root-cause the pre-existing emulated pp=2 vs pp=1 divergence**~~ — ROOT-CAUSED AND FIXED: `DeltaNetState::new_with_quant_multi` (qwen35.rs) did not wire the error-feedback residual (`s_ef_residual` empty → the DeltaNet recurrence kernel used the stochastic requantization path, while the single-GPU ctor wires EF by default). Per-layer hidden-state bisect (env-gated `dump_hidden_localize` added to the multi path): pos-0 outputs identical, pos-1 dev0 layers diverge by ~1.7e-3 — a state-write difference at the first token. Fix: allocate the EF residual per LA layer on its manifest-derived device (same compact-LA order as `s_matrices`, F16, `HIPFIRE_DN_STATE_EF` gate mirrored). **The `pp_parity` cargo test now PASSES (50/50 tokens identical) with default env under `HIPFIRE_EMULATE_GPUS=2`** — previously red at step 1 since before STEP-004. The probe DIAG (env-gated per-layer dump in `forward_scratch_layers_multi`) was kept as a permanent parity-localization tool.

Also fixed during close-out: `x_rot_covers_deltanet_value_width_for_moe_configs` was missing its `#[test]` attribute (silently never ran); now runs and passes.
