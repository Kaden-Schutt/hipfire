// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch scheduler contract tests.
//!
//! No GPU. Exercises lane assignment, refill, commit_ready reservation,
//! abort/commit lifecycle, and fail-closed stale handling.
//!
//! Lives here rather than in `hipfire-engine` because the fixtures need an
//! architecture config type and that crate is deliberately arch-free.

use hipfire_engine::emit::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

fn lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn begin() -> MutexGuard<'static, ()> {
    let g = lock();
    batch_clear_all_terminals();
    hipfire_engine::terminal::clear_terminal_control();
    hipfire_engine::terminal::set_active_attempt_id(0);
    g
}

fn sampling(temp: f32, repeat: f32) -> BatchSampling {
    BatchSampling {
        temp,
        top_p: 0.8,
        top_k: None,
        min_p: None,
        repeat_penalty: repeat,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_window: 128,
    }
}
fn sampling_with_window(temp: f32, window: usize) -> BatchSampling {
    BatchSampling {
        temp,
        top_p: 0.8,
        top_k: None,
        min_p: None,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repeat_window: window,
    }
}
fn req(key: AttemptKey, sampling: BatchSampling) -> BatchPendingRequest {
    BatchPendingRequest {
        original_msg: serde_json::json!({
            "type": "generate",
            "id": key.id,
            "attempt_id": key.attempt_id,
            "prompt": "hi"
        }),
        key,
        prompt: "hi".into(),
        prompt_tokens: vec![1, 2, 3],
        started_in_think: false,
        system: None,
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        max_think_tokens: 0,
        max_tokens: 10,
        sampling,
    }
}

fn caps_for(arch_id: u32) -> saddle_core::caps::ArchCaps {
    hipfire_loader::carrier_for(arch_id)
        .map(|c| c.caps())
        .unwrap_or_default()
}
fn elig(
    arch_id: u32,
    pp: usize,
    ep_is_some: bool,
    has_image: bool,
    has_tools: bool,
    has_stop: bool,
    has_speculator: bool,
    has_adaptive: bool,
    has_pflash: bool,
    has_messages_history: bool,
    think_mode_is_nonthink: bool,
    serve_continuous_batch: bool,
    continuous_batch_size: usize,
) -> bool {
    let caps = caps_for(arch_id);
    let req = saddle_core::caps::BatchEligibilityRequest {
        pp,
        ep_is_some,
        has_image,
        has_tools,
        has_stop,
        has_speculator,
        has_adaptive,
        has_pflash,
        has_messages_history,
        think_mode_is_nonthink,
        serve_continuous_batch,
        continuous_batch_size,
    };
    is_batch_eligible(&caps, &req)
}
#[test]
fn batch_eligible_only_qwen_text_single_gpu() {
    let _l = begin();
    assert!(elig(
        5, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(elig(
        6, 1, false, false, false, false, false, false, false, false, true, true, 2
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, false, false, true, false, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, false, false, true, true, 1
    ));
    assert!(!elig(
        9, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 2, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, true, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, true, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, true, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, true, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, true, false, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, true, false, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, true, false, true, true, 4
    ));
    assert!(!elig(
        5, 1, false, false, false, false, false, false, false, true, true, true, 4
    ));
}

#[test]
fn batch_eligible_allows_dense_lfm11_and_preserves_qwen() {
    let _l = begin();
    // LFM dense (arch 11) follows same pure exclusions as Qwen; MoE status is not checked here.
    assert!(elig(
        11, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(elig(
        11, 1, false, false, false, false, false, false, false, false, true, true, 2
    ));
    // Same pure exclusions as Qwen: B=1, pp!=1, ep, images, tools, stops, spec, adaptive, pflash, history, think.
    assert!(!elig(
        11, 1, false, false, false, false, false, false, false, false, true, false, 4
    ));
    assert!(!elig(
        11, 1, false, false, false, false, false, false, false, false, true, true, 1
    ));
    assert!(!elig(
        11, 2, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        11, 1, true, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        11, 1, false, true, false, false, false, false, false, false, true, true, 4
    ));
    // Unknown arch beside 5/6/11 stays ineligible.
    assert!(!elig(
        12, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(!elig(
        9, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    // Qwen still eligible (preserve existing behavior).
    assert!(elig(
        5, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
    assert!(elig(
        6, 1, false, false, false, false, false, false, false, false, true, true, 4
    ));
}

#[test]
fn lfm_dense_is_dense_and_route_is_lfm_ar() {
    let _l = begin();
    // Pure helper is state-independent: dense check is config-level.
    let dense = hipfire_arch_lfm2moe::config::Lfm2MoeConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        num_hidden_layers: 24,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 64,
        conv_kernel_size: 3,
        intermediate_size: 4096,
        moe_intermediate_size: 1792,
        num_experts: 0,
        num_experts_per_tok: 0,
        num_dense_layers: 24,
        rope_theta: 5_000_000.0,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 128000,
        norm_topk_prob: false,
        use_expert_bias: false,
        routed_scaling_factor: 1.0,
        tie_word_embeddings: true,
        layer_types: vec![hipfire_arch_lfm2moe::config::MixerKind::Attention; 24],
        reap_keep: None,
    };
    assert!(dense.is_dense());
    let moe = hipfire_arch_lfm2moe::config::Lfm2MoeConfig {
        num_experts: 32,
        num_experts_per_tok: 4,
        ..dense.clone()
    };
    assert!(!moe.is_dense());
    // Route selection is pure and state-independent.
    let base = hipfire_generate::ar::GenerationRouteInputs {
        arch_id: 11,
        ep: false,
        pp: 1,
        has_speculator: false,
        speculator_is_mtp: false,
        deepseek4_spec_requested: false,
        ngram_can_sample: false,
        temp: 0.1,
        user_explicit_sampling: false,
        min_p: None,
        nonneutral_penalties: false,
        force_ar_chat: false,
        temp_spec_env_off: false,
        fast_sample_on: true,
        supports_temp_swor: false,
        supports_chain_nucleus_verify: false,
        kv_adaptive: false,
        mesh_carrier: false,
    };
    assert_eq!(
        hipfire_generate::ar::select_generation_route(&base),
        hipfire_generate::ar::GenerationRoute::LfmAr
    );
    let spec = hipfire_generate::ar::GenerationRouteInputs {
        has_speculator: true,
        temp: 0.0,
        ..base
    };
    assert_eq!(
        hipfire_generate::ar::select_generation_route(&spec),
        hipfire_generate::ar::GenerationRoute::LfmSpec
    );
    let qwen = hipfire_generate::ar::GenerationRouteInputs { arch_id: 5, ..base };
    assert_eq!(
        hipfire_generate::ar::select_generation_route(&qwen),
        hipfire_generate::ar::GenerationRoute::QwenAr
    );
}
#[test]
fn lfm_bundle_batch_gate_admits_dense_staged_lfm() {
    let _l = begin();
    use hipfire_generate::ar::GenerationRoute;
    use hipfire_generate::batch::lfm_bundle_batch_gate;
    // A dense, staged LFM bundle that passes every gate IS batch-eligible
    // (B1: the pre-fix code ended this branch with an unconditional false).
    assert!(lfm_bundle_batch_gate(
        GenerationRoute::LfmAr,
        true,
        true,
        true
    ));
    // Each gate independently excludes.
    assert!(!lfm_bundle_batch_gate(
        GenerationRoute::QwenAr,
        true,
        true,
        true
    ));
    assert!(!lfm_bundle_batch_gate(
        GenerationRoute::LfmAr,
        false,
        true,
        true
    ));
    assert!(!lfm_bundle_batch_gate(
        GenerationRoute::LfmAr,
        true,
        false,
        true
    ));
    assert!(!lfm_bundle_batch_gate(
        GenerationRoute::LfmAr,
        true,
        true,
        false
    ));
}

#[test]
fn ep_evidence_fail_closed_without_receipt() {
    let _l = begin();
    // No runtime receipt: only generic continuous-batch route evidence is
    // attached — expert_parallel / rank_count / reduce claims must never
    // be fabricated (fail closed).
    let mut envelope = serde_json::json!({"type": "done", "id": "r1", "tokens": 5});
    hipfire_generate::batch::attach_ep_batch_evidence_fail_closed(
        &mut envelope,
        None,
        4,
        1,
        4096,
        2,
    );
    assert_eq!(
        envelope["execution_mode"],
        serde_json::json!("continuous_batch_independent")
    );
    assert_eq!(
        envelope["continuous_batch"]["executed"],
        serde_json::json!(true)
    );
    assert_eq!(envelope["continuous_batch"]["slots"], serde_json::json!(4));
    assert_eq!(envelope["continuous_batch"]["lane"], serde_json::json!(1));
    assert!(envelope["continuous_batch"].get("parallelism").is_none());
    assert!(envelope["continuous_batch"].get("rank_count").is_none());
    assert!(envelope["continuous_batch"].get("rank_mask").is_none());
    assert!(envelope["continuous_batch"].get("reduce").is_none());
    assert!(envelope["continuous_batch"].get("epoch").is_none());
}
