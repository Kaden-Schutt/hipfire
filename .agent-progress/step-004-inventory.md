# STEP-004: Architecture × Forward-Entry-Point Inventory

> **Historical snapshot with post-merge corrections.** STEP-004 closed against
> the pre-merge branch inventory. The 2026-08-26 mainline absorption added or
> re-enabled default production `SuperOp` paths and the bespoke Muse/Glimmer
> decoder. They are not evidence of full Step adoption: STEP-005 owns SuperOp
> retirement, STEP-006 owns newly absorbed bespoke decoder families, and
> GEN-003 owns duplicated prefill/continuous-batch orchestration. Current status
> remains authoritative in
> [device-mesh-refactor-tracker.md](device-mesh-refactor-tracker.md).

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
| 5 | Qwen35 dense | `forward_scratch_layers_lowered` (decode) | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | **Production SuperOp — STEP-005** — default-on `run_layer_program`; the hand Step route below is the migration source | SuperOp is not Step completion |
| 5 | Qwen35 dense | `forward_scratch_layers_inner` (hand decode) | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | **Partial — STEP-005** — individual `execute_steps_mesh` calls coexist with direct qk-norm and partial-interleaved RoPE GPU calls | Add the missing typed operations before this becomes the sole route |
| 5 | Qwen35 dense | `forward_prefill_chunk` (prefill) | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` | **Partial — GEN-003** — DeltaNet/attention Steps and sealed MoE lowering coexist with bespoke batched projections, norms, chunk transitions, and request control | Shared prefill driver with explicit-row Step execution |
| 5 | Qwen35 dense | `forward_scratch_layers_multi` (PP decode) | `crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs` | **Migrated in STEP-004 Inc 2** — QKVZA/QKV/gate-up via `*_via_execute_steps`, attention via `kv_cache_attention_dispatch` (with per-device givens override); DeltaNet recurrent + MoE FFN already Step-based | — |
| 5 | Qwen35 dense | `forward_ep` (EP decode) | `crates/hipfire-arch-qwen35/src/qwen35/ep_batch.rs` | **Production SuperOp — STEP-005** — `run_layer_program_ep` owns a second execution and collective loop | Must move collectives into Step execution |
| 6 | Qwen35 / Qwen36 / Qwen38 MoE | Same paths as arch 5 | split `qwen35/{forward,prefill,ep_batch}.rs` | **Mixed — STEP-005** — model-owned plans and sealed MoE lowering exist, but default Single and EP routes still enter SuperOp executors | CPU-top-k per-expert fallback and missing typed operations remain migration work |
| 7 | Qwen2/VibeThinker | `forward_step_after_x` / `forward_step_after_x_lowered` | `crates/hipfire-arch-qwen2/src/qwen2.rs` | **Full** — individual `execute_steps_mesh` calls; lowered path uses `dense_forward` | — |
| 8 | dots.ocr | `forward_step` (vision + Qwen2 decoder) | `crates/hipfire-arch-dots-ocr/src/dots_ocr.rs` | **Bespoke** | Vision exception (exception 1) — AXIS-004/VL-002 scope |
| 9 | DeepSeek4 | `forward_ep` / `forward_tp` / MTP mesh routes | `crates/hipfire-arch-deepseek4/src/{forward,ep,mtp}.rs` | **Mixed — STEP-005** — EP reaches `run_layer_program_ep`; Step-backed MoE coexists with MLA/compressor/indexer/RoPE-tail escapes and capture-specific routes | Replace the second EP executor and escapes with typed Steps or executor-owned fused patterns |
| 9 | DeepSeek4 | single-GPU decode / MTP | `crates/hipfire-arch-deepseek4/src/{forward,mtp}.rs` | **Production SuperOp — STEP-005** — default-on `run_layer_program` wraps model-owned manifest/MoE pieces | MLA/compressor/indexer/tail-RoPE operation gaps |
| 10 | MiniMax | `decode_step_body` (decode) | `crates/hipfire-arch-minimax/src/forward.rs` | **Mixed — STEP-005** — the hand route is mostly Step, but the default-on single-GPU route enters `run_layer_program` | partial-interleaved RoPE, embedding/head-prefix, and capture geometry must remain below the Step contract |
| 10 | MiniMax | `forward_ep` / `forward_tp` (EP/TP) | `crates/hipfire-arch-minimax/src/forward.rs` | **Partial — STEP-005** — Step-backed mesh execution with load-certified rank layouts and sealed collectives still shares the hand route's missing partial-interleaved RoPE/capture operations | Complete typed operation coverage without adding another executor |
| 11 | LFM2-MoE | `decode_step_layers_and_head` (decode) | `crates/hipfire-arch-lfm2moe/src/forward.rs` | **Mixed — STEP-005** — the hand route is Step-complete for standard ops and sealed MoE lowering, but default-on production dispatch uses SuperOp block handlers | Conv mixer needs a typed Step or executor-owned fused pattern |
| 12 | Cohere2-MoE | `decode_step_body` (decode) | `crates/hipfire-arch-cohere2moe/src/forward.rs` | **Partial — STEP-005** — most parallel-block and routed-expert work is Step-backed, but interleaved RoPE, non-indexed per-expert fallback, and sigmoid+top-k routing remain direct | Add typed variants or executor-owned fused patterns |
| — | Qwen35-VL | VL-specific prefill + shared post-prefill | `crates/hipfire-arch-qwen35-vl/src/qwen35_vl.rs` | **Bespoke** | Vision exception (exception 2) — VL-001 scope |
| 13 | Gemma 4 | `forward_scratch_inner_lowered` (default-ON decode) | `crates/hipfire-arch-gemma4/src/lowered.rs` | **Production SuperOp — STEP-005** — `run_layer_program`; the retained typed-Step hand path is the migration source | hybrid sliding/full attention geometry remains architecture-owned below Step execution |
| 14 | Muse Glimmer | decode/prefill/batch forward family | `crates/hipfire-arch-muse-glimmer/src/{forward,forward_batch,batch}.rs` | **Bespoke — STEP-006** — no Step/SuperOp route | NoPE/full-attention alternation, gated attention, and split normalization/scaling require typed or fused Step representation; CAP continues to refuse PP/TP/EP |

## Non-decoder boundaries and open Step gaps

1. **dots.ocr (arch 8)** — entire forward is vision-specific. "Vision+OCR pipeline — not a standard decoder path." AXIS-004/VL-002 scope.
2. **Qwen35-VL** — VL-specific prefill. "Vision-conditioned prefill — shares post-prefill lifecycle but image processing is bespoke." VL-001 scope.
3. **DeepSeek4 MLA/compressor/indexer** — currently represented by `EscapeKind` ops in the SuperOp substrate. STEP-005 must replace these with typed Steps or executor-owned fused Step patterns while preserving the architecture-specific kernels.
4. **Embedding lookup** (all arches) — pre-decoder. "Embedding lookup is a pre-decoder token-to-activation map; not a decoder layer op."
5. **LFM2 Conv mixer** — currently handled by `SuperOpKind::Conv`. STEP-005 must express the stateful mixer as a typed Step or executor-owned fused Step pattern.
6. **Cohere2 interleaved RoPE** — open STEP-005 gap. Add an interleaved typed Step or executor-owned fused Step pattern; it is not a permanent decoder exception.
7. **Cohere2/LFM2 per-expert GEMV fallback** — open STEP-005 gap for non-indexed BF16/Q8/F16 experts. The current CPU-side selected-expert loop is not universal Step completion.
8. **Qwen35 prefill batched path** — tracked by GEN-003. Chunking/abort/checkpoint policy belongs to the shared request lifecycle; batched projections and norms belong below it in explicit-row Step execution.
9. **Cohere2 sigmoid+topk routing** — open STEP-005 gap. Add a matching routing Step without unwanted softmax semantics.
10. **MiniMax/Qwen35 partial-interleaved RoPE** — open STEP-005 gap. Add a typed partial-interleaved variant covering `partial_rotary_factor`.
11. **LFM2/Cohere2 routed MoE expert phases — RESOLVED in follow-up.** Both carriers now ship the manifest machinery (`weight_manifest` with `ExpertSharded` packed-fused surrogates, policy-aware `expert_group_manifest` with `sigmoid_topk` + `indexed_quantized`, model-owned config-keyed group-plan caches) and the forward paths lower the expert phases through `lower_moe_steps` + `execute_lowered_moe` (Single). LFM2's router phase carries `ScoreActivation` + `MoeRoute`; Cohere2's router phase is empty (its norm_topk_prob=false top-k has no Step variant — the bespoke sigmoid+topk runs before the program, exception 9). Both verified byte-identical to the pre-machinery direct kernel sequence on GPU (LFM2 8B-a1b + 350m, Cohere2 North-Mini-Code-1.0). Remaining: Cohere2's per-expert fallback loop (BF16/Q8/F16) stays direct (exception 7).
12. **Muse Glimmer** — the 3:1 sliding-RoPE/full-NoPE schedule, scale-less QK norm, gated attention projection, sandwich normalization, output multiplier, and softcap form one architecture-specific forward. No Step/SuperOp route currently preserves that combined contract; CAP-001 explicitly refuses PP/TP/EP rather than implying mesh support.

## Status

- STEP-004 increments 2–5 remain valid parity evidence for the paths they
  actually migrated: Qwen35 PP decode, the LFM2/Cohere2 hand routes, and
  MiniMax hand-route completion.
- The former universal-completion wording was invalidated by the post-merge
  inventory. `run_layer_program` is default-on for Qwen35, DeepSeek4,
  MiniMax, LFM2, and Gemma4; Qwen35 and DeepSeek4 EP reach
  `run_layer_program_ep`; Muse/Glimmer has no Step representation.
- STEP-005 is the production execution-spine correction. STEP-006 covers
  newly absorbed bespoke decoder families. GEN-003 separately removes
  duplicate prefill and continuous-batch orchestration.
- Recorded STEP-004 parity baselines remain useful migration oracles; they do
  not satisfy STEP-005, STEP-006, or GEN-003.

## Follow-ups (tracked todos)

1. ~~**Add manifest machinery to LFM2/Cohere2 carriers**~~ — DONE: `weight_manifest` (`ExpertSharded` packed-fused surrogates, projection under Single), policy-aware `expert_group_manifest` (`sigmoid_topk` + `indexed_quantized`), model-owned config-keyed `moe_group_plans` caches, `state_manifest` (Conv/Kv). CPU tests pin the MoE-span plan resolution per arch.
2. ~~**Migrate LFM2/Cohere2 MoE expert phases to `lower_moe_steps` + `execute_lowered_moe`**~~ — DONE: LFM2 (router phase `ScoreActivation`+`MoeRoute`) and Cohere2 (empty router phase; bespoke routing stays, exception 9) both lower through the sealed Single executor; GPU parity byte-identical on LFM2 8B-a1b/350m and Cohere2 North-Mini-Code-1.0. Cohere2's per-expert fallback (BF16/Q8/F16) remains direct (exception 7). TP/EP resolution of the same manifests is the AXIS-002 continuation (MiniMax sealed exact-policy cache when policy threading lands).
3. ~~**Decide fate of the LFM2 "mirror" block helpers**~~ — REOPENED BY STEP-005: mainline made these helpers default-on SuperOp production handlers. Their hand-route Step implementations are migration oracles, not justification for retaining a second executor.
4. ~~**Root-cause the pre-existing emulated pp=2 vs pp=1 divergence**~~ — ROOT-CAUSED AND FIXED: `DeltaNetState::new_with_quant_multi` (qwen35.rs) did not wire the error-feedback residual (`s_ef_residual` empty → the DeltaNet recurrence kernel used the stochastic requantization path, while the single-GPU ctor wires EF by default). Per-layer hidden-state bisect (env-gated `dump_hidden_localize` added to the multi path): pos-0 outputs identical, pos-1 dev0 layers diverge by ~1.7e-3 — a state-write difference at the first token. Fix: allocate the EF residual per LA layer on its manifest-derived device (same compact-LA order as `s_matrices`, F16, `HIPFIRE_DN_STATE_EF` gate mirrored). **The `pp_parity` cargo test now PASSES (50/50 tokens identical) with default env under `HIPFIRE_EMULATE_GPUS=2`** — previously red at step 1 since before STEP-004. The probe DIAG (env-gated per-layer dump in `forward_scratch_layers_multi`) was kept as a permanent parity-localization tool.
5. **Retire production SuperOp execution** — tracked by STEP-005. Remove every production `run_layer_program*`/`ForwardBindings` caller after deterministic parity, then delete the obsolete substrate.
6. **Migrate Muse/Glimmer and other newly absorbed bespoke decoders** — tracked by STEP-006. Preserve architecture-specific fused semantics below the Step contract; do not imply mesh support before an AXIS task admits it.

Also fixed during close-out: `x_rot_covers_deltanet_value_width_for_moe_configs` was missing its `#[test]` attribute (silently never ran); now runs and passes.
