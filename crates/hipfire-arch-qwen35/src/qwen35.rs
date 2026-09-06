// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Qwen3.5 model: hybrid DeltaNet (linear attention) + standard attention.
//! Feature-gated behind `deltanet`.
// Split from the former 24,837-line qwen35.rs (pure code motion; see each
// module for its role). Every public path `crate::qwen35::<name>` is
// preserved by the re-exports below; the pub(crate) block keeps the
// forward_slots.rs imports working unchanged.

pub mod batch;
pub mod config;
pub mod ep_batch;
pub mod forward;
pub mod load;
pub mod prefill;
pub mod weights;

pub use batch::{
    forward_decode_batch, forward_decode_batch_prepared, prepare_decode_batch_inputs,
    PrefillBatchScratch, Qwen35DecodeBatchState,
};
pub use config::{
    apply_reap_plan, config_from_hfq, config_from_metadata_json, config_from_safetensors,
    dense_tp_rank_layouts, local_dense_tp_config, validate_dense_tp, DenseTpRankLayout,
    DflashFusionCtx, LayerType, MaskEmbedOverride, MropeCtx, Qwen35BatchCompatibility,
    Qwen35BatchLoadConfig, Qwen35BatchParallelism, Qwen35Config, Qwen35EpBatchReceipt,
    Qwen35EpReduce, Qwen35EpTopology, TreeVerifyCtx,
};
pub use ep_batch::{
    forward_ep, forward_prefill_batch_ep, forward_prefill_batch_multi, forward_scratch_multi,
    validate_ep_batch_compatibility, Qwen35DecodeBatchEpState,
};
pub use forward::{
    dump_expert_stats, forward, forward_gpu, forward_prefill_dense_tp, forward_scratch,
    forward_scratch_dense_tp, forward_scratch_embed, forward_scratch_embed_mrope,
    forward_scratch_mrope, forward_scratch_with_hidden, forward_with_embedding,
    prepare_scratch_inputs, shard_all_moe_layers, shard_moe_experts, Qwen35Scratch,
    Qwen35ScratchSet,
};
pub use load::{
    load_weights, load_weights_dense_tp_rank, load_weights_ep_rank, preflight_weights_dense_tp,
    set_ep_expert_shard, EpShardGuard, HfqSource, Layout, ParoSource,
};
pub use prefill::{
    forward_prefill_batch, forward_prefill_batch_capped,
    forward_prefill_batch_single_chunk_captured, forward_prefill_batch_single_chunk_captured_opts,
    forward_prefill_batch_with_pbs, forward_prefill_batch_with_pbs_opts,
    prefill_batch_pbs_eligible, prefill_max_batch, prefill_max_batch_tp,
    qwen35_layer_batch_admissible, upload_prefill_batch_inputs, PREFILL_MAX_BATCH,
};
pub(crate) use prefill::{
    moe_ffn_batched_admissible, mq6_batched_admit_enabled_from_env, prefill_moe_ffn_body_batched,
    q8_prefill_wmma_enabled, run_fused_gate_up_key, run_fused_qkv_key, run_fused_qkvza_key,
    run_plain_gemm_key, run_residual_gemm_key,
};
pub use weights::{
    mixed_expert_tag, DeltaNetLayerWeights, DeltaNetMoeLayerWeights, DeltaNetState, ExpertWeights,
    FullAttnLayerWeights, FullAttnMoeLayerWeights, GpuTensorDescriptor, LayerWeights,
    MoeFfnWeights, MoeParoSidecars, ParoDescriptor, Qwen35EpConfigFingerprint, Qwen35EpShardInfo,
    Qwen35HfqSourceIdentity, Qwen35LayerSeal, Qwen35LocalExpertDescriptor, Qwen35MoeFfnSeal,
    Qwen35RankSeal, Qwen35Weights, SharedExpertWeights, StateQuant, WeightTensorDescriptor,
};
