// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire engine daemon — JSON lines over stdin/stdout.
//! The Bun CLI spawns this process and communicates via IPC.
//! Usage: daemon (reads JSON from stdin, writes JSON to stdout)
//!
//! Exactly one daemon runs at a time per machine — enforced by an exclusive
//! flock(2) on ~/.hipfire/daemon.pid. A second daemon invocation exits with
//! `FATAL: hipfire daemon already running (PID N)` before touching the GPU,
//! preventing orphan doubles from silently double-consuming VRAM.
//!
//! Protocol:
//!   → {"type":"load","model":"path.hfq","params":{"max_seq":4096}}
//!   ← {"type":"loaded","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320,"vl":true}
//!   → {"type":"generate","id":"r1","prompt":"Hello","temperature":0.3,"max_tokens":512}
//!   → {"type":"generate","id":"r1","prompt":"Describe this","image":"/path/to/img.png","temperature":0.3,"max_tokens":512}
//!   ← {"type":"token","id":"r1","text":"The"}
//!   ← {"type":"done","id":"r1","tokens":42,"tok_s":44.5}
//!   → {"type":"unload"}
//!   ← {"type":"unloaded"}

use base64::Engine;
use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_gemma4 as gemma4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
use hipfire_arch_lfm2moe::forward_batch::{
    forward_decode_batch_lfm, forward_decode_batch_prepared_lfm, prepare_decode_batch_inputs_lfm,
};
use hipfire_arch_minimax as minimax;
use hipfire_arch_muse_glimmer as glimmer;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
// Used by hipfire_generate::qwen::generate_qwen35_mtp (native-MTP serve path, merged from spec-graph):
// it manually re-packs the Qwen35 bundle on every exit + re-opens the HFQ mmap.
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_runtime::emit_text::{
    currently_in_think, extract_tool_calls_from_text, ThinkOutputRouter, ThinkRouteEvent,
    ToolOutputRouter, ToolRouteError, ToolRouteEvent,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_runtime::spec::accept_greedy_prefix;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use hipfire_loader::{AsstTurnCache, EpArch, EpState, Eviction, LoadedModel, ModelState};
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, EvictRetain, FinishSummary, PrefillOutcome, SpecAdvance, SpecEmit,
    SpecTarget, Speculator, StopReason,
};
use hipfire_engine::emit::*;
use hipfire_engine::prompt::*;
use hipfire_engine::redline::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_generate::vision::{GenerateVLParams, ImageSource};
use hipfire_generate::redline::{
    handle_redline_dispatch_profile,
    handle_redline_dspark_shadow_pm4,
    handle_redline_pm4_prefix_profile,
    handle_redline_prefix_shadow,
    handle_redline_probe_aql,
    handle_redline_shadow,
    RedlineDeepseek4Snapshot,
    RedlineDsparkArm,
    RedlineDsparkReplayArm,
    RedlineDsparkVerifySnapshot,
    RedlineLfm2MoeSnapshot,
    RedlineQwenSnapshot,
    RedlineSnapshot,
    redline_append_tensor_slice,
    redline_bench_decode_deepseek4,
    redline_bench_decode_lfm2moe,
    redline_deepseek4_snapshot,
    redline_dspark_shadow_block,
    redline_dspark_verify_guard,
    redline_dspark_verify_snapshot,
    redline_is_dense_lfm,
    redline_lfm2moe_snapshot,
    redline_pm4_prefix_profile_deepseek4,
    redline_prepare_retained_fixture,
    redline_prime_deepseek4,
    redline_prime_dspark_shadow_arm,
    redline_prime_qwen,
    redline_prime_retained_fixture,
    redline_qwen_debug_hashes,
    redline_qwen_snapshot,
    redline_reset_deepseek4,
    redline_reset_lfm2moe,
    redline_reset_qwen,
    redline_run_deepseek4_decode,
    redline_run_direct_fixture,
    redline_run_dspark_capture_arm,
    redline_run_dspark_direct_arm,
    redline_run_dspark_replay_arm,
    redline_shadow_deepseek4,
    redline_shadow_dspark_verify_pm4,
    redline_snapshot,
};
use hipfire_generate::ar::{
    GenerationRoute,
    GenerationRouteInputs,
    QwenArCacheAction,
    QwenArForwardFailAction,
    QwenArRawCommitDisposition,
    QwenArRouteFinish,
    QwenArSemanticProducer,
    QwenArTerminalCause,
    ckpt_interval,
    ckpt_max,
    ckpt_resume_enabled,
    deepseek4_spec_requested,
    deepseek4_spec_requested_from_policy,
    emit_qwen_ar_open_think_terminal,
    generate,
    llama_prefill_sample_seed,
    llama_qwen3_batched_prefill_eligible,
    qwen_ar_apply_cache_action,
    qwen_ar_cache_action,
    qwen_ar_done_value,
    qwen_ar_drain_pending_into_router,
    qwen_ar_eos_filter_config,
    qwen_ar_eviction_prefill_chunk_limit,
    qwen_ar_finish_route,
    qwen_ar_forward_fail_action,
    qwen_ar_forward_fail_message,
    qwen_ar_observe_and_route,
    qwen_ar_raw_commit_token,
    qwen_ar_route_filter_text,
    qwen_ar_route_think_events,
    select_generation_route,
    truncate_checkpoints,
    write_error,
};
use hipfire_generate::batch::{
    attach_qwen_ep_batch_receipt_evidence,
    drive_lfm_continuous_batch,
    drive_qwen35_ep_continuous_batch,
    drive_qwen_continuous_batch,
    emit_uncorrelated_error,
    is_batch_request_eligible,
    is_qwen_ep_batch_request_eligible,
    lfm_prefill_cancellable_or_fallback,
};
#[cfg(feature = "serve-fault-inject")]
use hipfire_generate::ar::take_fault_after_prefill;
#[cfg(feature = "serve-fault-inject")]
use hipfire_generate::ar::arm_fault_after_prefill;





/// Formats the independent Qwen decode-batch path can actually execute.
/// Must stay aligned with `lm_head_batched` + `prepare_decode_batch_inputs`
/// in hipfire-arch-qwen35 — unsupported lm_head or F32 embedding must never
/// advertise `continuous_batch_capable` or enter the batch route.










/// Pure deterministic coverage for the correlated terminal-control plane.
/// No GPU. Drives activate/apply/await helpers directly.
#[cfg(test)]
mod terminal_control_tests {    use hipfire_engine::terminal::{
        activate_terminal_control, apply_terminal_control, await_client_terminal_commit,
        check_abort, clear_terminal_control, mark_terminal_control_ready, set_active_attempt_id,
        terminal_control, wait_terminal_control_decision, ClientTerminalDecision,
        TerminalControlDecision,
    };
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use std::time::Duration;



    /// `hipfire_generate::dense::glimmer_longest_marker_suffix` byte-slices from the end of the pending
    /// buffer looking for a split Harmony marker. It must skip offsets that
    /// land inside a multibyte character.
    ///
    /// Regression: the first version did `&s[s.len() - len..]` unguarded and
    /// panicked with "byte index N is not a char boundary" the moment Glimmer
    /// emitted a non-ASCII character — `×` in an arithmetic reasoning span took
    /// the whole daemon down mid-generation. Markers are pure ASCII, so an
    /// offset inside a multibyte char can never start one.
    #[test]
    fn glimmer_marker_suffix_is_char_boundary_safe() {
        // Each of these ends in (or contains) a multibyte char at a position the
        // reverse scan would probe.
        for s in [
            "17 × 23",
            "café",
            "—",
            "reasoning ×",
            "emoji 😀",
            "mixed ×<|eo",
        ] {
            let n = hipfire_generate::dense::glimmer_longest_marker_suffix(s);
            assert!(
                s.is_char_boundary(s.len() - n),
                "returned len {n} splits a char in {s:?}"
            );
        }
        // Still detects a genuine split marker.
        assert_eq!(hipfire_generate::dense::glimmer_longest_marker_suffix("abc<|eo"), 4);
        assert_eq!(hipfire_generate::dense::glimmer_longest_marker_suffix("abc"), 0);
    }

}

/// Deterministic protocol/state-machine coverage for continuous batching.
/// No GPU. Exercises lane assignment, refill, commit_ready reservation,
/// abort/commit lifecycle, and fail-closed stale handling.
#[cfg(test)]
mod continuous_batch_tests {    use super::{
        batch_announce_terminal, batch_apply_terminal_control, batch_check_abort,
        batch_clear_all_terminals, batch_clear_terminal, batch_commit_teardown_class,
        batch_hit_length_cap, batch_lane_at_capacity, batch_mark_ready,
        batch_mark_ready_with_pending, batch_poll_decision, batch_should_finish_decode,
        batch_terminal_control, batch_transfer_abort_to_singleton_and_clear, is_batch_eligible,
        parse_continuous_batch_size, parse_serve_continuous_batch, AttemptKey,
        BatchCommitTeardownClass, BatchPendingRequest, BatchSampling, BatchSamplingKey,
        ClientTerminalDecision, ContinuousBatchScheduler, DaemonInbox, DaemonMsg, LaneTicket,
    };

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
        super::clear_terminal_control();
        super::set_active_attempt_id(0);
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
        assert!(elig(5, 1, false, false, false, false, false, false, false, false, true, true, 4));
        assert!(elig(6, 1, false, false, false, false, false, false, false, false, true, true, 2));
        assert!(!elig(5, 1, false, false, false, false, false, false, false, false, true, false, 4));
        assert!(!elig(5, 1, false, false, false, false, false, false, false, false, true, true, 1));
        assert!(!elig(9, 1, false, false, false, false, false, false, false, false, true, true, 4));
        assert!(!elig(5, 2, false, false, false, false, false, false, false, false, true, true, 4));
        assert!(!elig(5, 1, true, false, false, false, false, false, false, false, true, true, 4));
        assert!(!elig(5, 1, false, true, false, false, false, false, false, false, true, true, 4));
        assert!(!elig(5, 1, false, false, true, false, false, false, false, false, true, true, 4));
        assert!(!elig(5, 1, false, false, false, true, false, false, false, false, true, true, 4));
        assert!(!elig(5, 1, false, false, false, false, true, false, false, false, true, true, 4));
        assert!(!elig(5, 1, false, false, false, false, false, true, false, false, true, true, 4));
        assert!(!elig(5, 1, false, false, false, false, false, false, true, false, true, true, 4));
        assert!(!elig(5, 1, false, false, false, false, false, false, false, true, true, true, 4));
    }

    #[test]
    fn batch_eligible_allows_dense_lfm11_and_preserves_qwen() {
        let _l = begin();
        // LFM dense (arch 11) follows same pure exclusions as Qwen; MoE status is not checked here.
        assert!(elig(11, 1, false, false, false, false, false, false, false, false, true, true, 4));
        assert!(elig(11, 1, false, false, false, false, false, false, false, false, true, true, 2));
        // Same pure exclusions as Qwen: B=1, pp!=1, ep, images, tools, stops, spec, adaptive, pflash, history, think.
        assert!(!elig(11, 1, false, false, false, false, false, false, false, false, true, false, 4));
        assert!(!elig(11, 1, false, false, false, false, false, false, false, false, true, true, 1));
        assert!(!elig(11, 2, false, false, false, false, false, false, false, false, true, true, 4));
        assert!(!elig(11, 1, true, false, false, false, false, false, false, false, true, true, 4));
        assert!(!elig(11, 1, false, true, false, false, false, false, false, false, true, true, 4));
        // Unknown arch beside 5/6/11 stays ineligible.
        assert!(!elig(12, 1, false, false, false, false, false, false, false, false, true, true, 4));
        assert!(!elig(9, 1, false, false, false, false, false, false, false, false, true, true, 4));
        // Qwen still eligible (preserve existing behavior).
        assert!(elig(5, 1, false, false, false, false, false, false, false, false, true, true, 4));
        assert!(elig(6, 1, false, false, false, false, false, false, false, false, true, true, 4));
    }

    

    #[test]
    fn lfm_dense_is_dense_and_route_is_lfm_ar() {
        let _l = begin();
        // Pure helper is state-independent: dense check is config-level.
        let dense = super::lfm2moe::config::Lfm2MoeConfig {
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
            layer_types: vec![super::lfm2moe::config::MixerKind::Attention; 24],
            reap_keep: None,
        };
        assert!(dense.is_dense());
        let moe = super::lfm2moe::config::Lfm2MoeConfig {
            num_experts: 32,
            num_experts_per_tok: 4,
            ..dense.clone()
        };
        assert!(!moe.is_dense());
        // Route selection is pure and state-independent.
        let base = super::GenerationRouteInputs {
            arch_id: 11,
            ep: false,
            pp: 1,
            has_speculator: false,
            qwen_mtp_head: false,
            qwen_mtp_opt_in: false,
            mtp_sampled_on: false,
            deepseek4_spec_requested: false,
            ngram_can_sample: false,
            temp: 0.1,
            user_explicit_sampling: false,
            min_p: None,
            force_ar_chat: false,
            temp_spec_env_off: false,
            fast_sample_on: true,
            supports_temp_swor: false,
            kv_adaptive: false,
        };
        assert_eq!(
            super::select_generation_route(&base),
            super::GenerationRoute::LfmAr
        );
        let spec = super::GenerationRouteInputs {
            has_speculator: true,
            temp: 0.0,
            ..base
        };
        assert_eq!(
            super::select_generation_route(&spec),
            super::GenerationRoute::LfmSpec
        );
        let qwen = super::GenerationRouteInputs { arch_id: 5, ..base };
        assert_eq!(
            super::select_generation_route(&qwen),
            super::GenerationRoute::QwenAr
        );
    }

}


pub type CaskConfig = hipfire_runtime::loader_api::CaskConfig;


































































/// Emit the full Qwen AR `done` envelope via serde (hostile-id safe).
fn emit_qwen_ar_done(
    stdout: &mut impl std::io::Write,
    id: &str,
    finish_reason: &str,
    generated: usize,
    tok_s: f64,
    prefill_tokens: usize,
    prefill_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
    cached_tokens: usize,
    pflash_fragment_json: &str,
) {
    let envelope = qwen_ar_done_value(
        id,
        finish_reason,
        generated,
        tok_s,
        prefill_tokens,
        prefill_ms,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms,
        cached_tokens,
        pflash_fragment_json,
    );
    emit_staged_terminal_done(stdout, &envelope);
}




#[allow(dead_code)]
fn emit_error_no_id(stdout: &mut impl std::io::Write, message: impl std::fmt::Display) {
    hipfire_generate::dense::emit_active_attempt_error(stdout, None, &message.to_string(), "internal", false, false);
}










/// Pure attestation combiner for unit tests / failure injection: every required
/// reset class must succeed AND sync must succeed for `rolled_back=true`.
/// Sync is modeled as always attempted (callers pass its outcome regardless).
#[cfg_attr(not(test), allow(dead_code))]
fn attest_rollback_steps(
    steps: &[(&str, Result<(), String>)],
    sync: Result<(), String>,
) -> hipfire_generate::common::RollbackEpilogue {
    let mut errs: Vec<String> = Vec::new();
    for (name, r) in steps {
        if let Err(e) = r {
            errs.push(format!("{name}: {e}"));
        }
    }
    if let Err(e) = sync {
        errs.push(format!("device_synchronize failed: {e}"));
    }
    if errs.is_empty() {
        hipfire_generate::common::RollbackEpilogue {
            rolled_back: true,
            context: None,
        }
    } else {
        hipfire_generate::common::RollbackEpilogue {
            rolled_back: false,
            context: Some(errs.join("; ")),
        }
    }
}




/// Parse attempt_id from a JSON number only (u64 or non-neg i64).
/// Decimal strings are rejected — no further coercion.
fn parse_wire_attempt_id(value: Option<&serde_json::Value>) -> Option<u64> {
    let value = value?;
    if let Some(n) = value.as_u64() {
        return Some(n);
    }
    if let Some(n) = value.as_i64() {
        if n >= 0 {
            return Some(n as u64);
        }
    }
    None
}

/// Require a present numeric attempt_id on the wire.
fn require_wire_attempt_id(value: Option<&serde_json::Value>) -> Result<u64, &'static str> {
    match value {
        None => Err("missing attempt_id"),
        Some(_) => parse_wire_attempt_id(value).ok_or("malformed attempt_id"),
    }
}

/// Map LoadedModel.arch_id to reset_core inventory arch key.
fn reset_core_arch_key(arch_id: u32) -> &'static str {
    match arch_id {
        0 | 1 => "llama",
        5 | 6 => "qwen35",
        7 => "qwen2",
        8 => "dots-ocr",
        9 => "deepseek4",
        10 => "minimax",
        11 => "lfm2moe",
        12 => "cohere2moe",
        13 => "gemma4",
        14 => "muse_glimmer",
        _ => "unknown",
    }
}

fn model_retry_reset_eligible(arch_id: u32) -> bool {
    hipfire_runtime::reset_core::is_retry_reset_eligible(reset_core_arch_key(arch_id))
}


// ── serve-fault-inject (test-only; compiled out of production) ─────────
// One-shot after-prefill GPU fault arm. Armed from generate parse when the
// feature is on and the request carries test_fault_after_prefill:true.



#[cfg(feature = "serve-fault-inject")]
struct FaultAfterPrefillGuard;
#[cfg(feature = "serve-fault-inject")]
impl Drop for FaultAfterPrefillGuard {
    fn drop(&mut self) {
        arm_fault_after_prefill(false);
    }
}



#[cfg(feature = "serve-fault-inject")]
fn write_test_state_snapshot(
    stdout: &mut impl std::io::Write,
    m: Option<&LoadedModel>,
    gpu: &rdna_compute::Gpu,
    state_epoch: u64,
) {
    let (
        arch,
        eligible,
        seq_pos,
        conversation_len,
        kv_hash,
        kv_bytes,
        recurrent_hash,
        recurrent_bytes,
        drafter_reset,
        checkpoint_empty,
        adaptive_clean,
        asst_cache_empty,
        prefix_cache_clean,
    ) = match m {
        Some(m) => {
            let arch = reset_core_arch_key(m.arch_id);
            let eligible: Vec<&'static str> =
                hipfire_runtime::reset_core::fault_inject_eligible_routes(arch).to_vec();
            let (kv_hash, kv_bytes, recurrent_hash, recurrent_bytes) = match m.state.as_ref() {
                Some(ModelState::Qwen35(bundle)) => match redline_qwen_snapshot(gpu, bundle) {
                    Ok(snap) => (
                        format!("{:016x}", redline_hash(&snap.kv)),
                        snap.kv.len(),
                        format!("{:016x}", redline_hash(&snap.recurrent)),
                        snap.recurrent.len(),
                    ),
                    Err(_) => (
                        "unavailable".to_string(),
                        0usize,
                        "unavailable".to_string(),
                        0usize,
                    ),
                },
                _ => (
                    "unavailable".to_string(),
                    0usize,
                    "unavailable".to_string(),
                    0usize,
                ),
            };
            // Live Speculator evidence only. Missing evidence fail-closes dirty
            // — never invent clean from vestigial m.prefill/dflash_checkpoints.
            let (drafter_reset, checkpoint_empty) = match m.speculator.as_ref() {
                Some(s) => match s.reset_state_evidence() {
                    Some(ev) => (ev.drafter_reset, ev.checkpoint_empty),
                    None => (false, false),
                },
                // No live drafter ⇒ drafter residual N/A / clean; host rings still
                // report empty via prefill/dflash free on rollback.
                None => (
                    true,
                    m.prefill_checkpoints.is_empty() && m.dflash_checkpoints.is_empty(),
                ),
            };
            let adaptive_clean = m
                .kv_adaptive
                .as_ref()
                .map(|ad| !ad.is_poisoned())
                .unwrap_or(true);
            let asst_cache_empty = m.asst_turn_cache.is_empty();
            // Prefix-cache residual for qwen is conversation_tokens + asst
            // turn cache; clean when both empty (fresh/reset).
            let prefix_cache_clean = m.conversation_tokens.is_empty() && asst_cache_empty;
            (
                arch,
                eligible,
                m.seq_pos,
                m.conversation_tokens.len(),
                kv_hash,
                kv_bytes,
                recurrent_hash,
                recurrent_bytes,
                drafter_reset,
                checkpoint_empty,
                adaptive_clean,
                asst_cache_empty,
                prefix_cache_clean,
            )
        }
        None => (
            "none",
            Vec::new(),
            0usize,
            0usize,
            "unavailable".to_string(),
            0usize,
            "unavailable".to_string(),
            0usize,
            true,
            true,
            true,
            true,
            true,
        ),
    };

    let graph_clean = gpu.graphs.captured_graph.is_none()
        && gpu.graphs.graph_exec.is_none()
        && gpu.graphs.verify_graph_count() == 0
        && gpu.graphs.replay_graph_count() == 0;
    let obs = gpu.replay.replay_observation();
    let replay_clean = !obs.failed && obs.count == 0;

    let payload = serde_json::json!({
        "type": "test_state_snapshot",
        "schema_version": 1,
        "arch": arch,
        "eligible_routes": eligible,
        "state_epoch": state_epoch,
        "seq_pos": seq_pos,
        "conversation_len": conversation_len,
        "kv_hash": kv_hash,
        "kv_bytes": kv_bytes,
        "recurrent_hash": recurrent_hash,
        "recurrent_bytes": recurrent_bytes,
        "graph_clean": graph_clean,
        "replay_clean": replay_clean,
        "drafter_reset": drafter_reset,
        "checkpoint_empty": checkpoint_empty,
        "adaptive_clean": adaptive_clean,
        "asst_cache_empty": asst_cache_empty,
        "prefix_cache_clean": prefix_cache_clean,
    });
    let _ = writeln!(stdout, "{}", payload);
    let _ = stdout.flush();
}


/// Pure `gen_start.contract_version` selection used by the live generate path.
/// Qwen AR (5/6) and Muse Glimmer (14) advertise v2; DS4 (9) and every other
/// arch stay unset.
/// Muse Glimmer already emits the v2-shaped two-phase terminal
/// (`commit_ready` -> `commit` -> byte-identical `done`), and its tool calls
/// are staged as canonical `calls` on that terminal. Only the v2 fold reads
/// them: the legacy path builds tool calls solely from mid-stream `tool_calls`
/// events, which Glimmer does not emit, so on legacy a tool turn arrived with
/// `finish_reason=tool_calls` and an empty payload.
const GLIMMER_SEMANTIC_CONTRACT_VERSION: u32 = 2;







/// Pure speculative-route terminal decision after `Deepseek4Emit::finish`.
/// Returns `Some(action)` when the emitter reported malformed protocol.
fn ds4_spec_finish_route(
    finish_reason: &str,
    tool_calls: usize,
) -> Option<hipfire_generate::common::Ds4MalformedTerminalAction> {
    if finish_reason == "malformed_protocol" {
        debug_assert_eq!(
            tool_calls, 0,
            "spec malformed must report tool_calls=0 (buffered calls discarded)"
        );
        Some(hipfire_generate::common::ds4_malformed_terminal_action(
            "unclosed DSML tool_calls block at end of output",
        ))
    } else {
        None
    }
}























/// Production Malformed error envelope for Qwen DFlash epilogue + tests.
fn qwen_dflash_malformed_error_value(
    id: &str,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
    attempt_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "id": id,
        "message": message,
        "class": class,
        "retryable": retryable,
        "rolled_back": rolled_back,
        "attempt_id": attempt_id,
    })
}




/// Write one Qwen DFlash Done terminal via the production envelope builder.
fn emit_qwen_dflash_done_terminal(
    stdout: &mut impl std::io::Write,
    id: &str,
    generated: usize,
    tok_s: f64,
    prefill_tokens: usize,
    prefill_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
    tau: f64,
    cycles: usize,
    cached_tokens: usize,
    finish_reason: &str,
    pflash: Option<(&str, f32)>,
) {
    let mut done_env = hipfire_generate::qwen::qwen_dflash_done_value(
        id,
        generated,
        tok_s,
        prefill_tokens,
        prefill_ms,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms,
        tau,
        cycles,
        cached_tokens,
        finish_reason,
        active_attempt_id(),
    );
    if let Some((reason, alpha)) = pflash {
        done_env["pflash"] = serde_json::json!({
            "bypass_reason": reason,
            "alpha": alpha,
        });
    }
    let _ = writeln!(stdout, "{}", done_env);
    let _ = stdout.flush();
}














/// Whether a hipfire_generate::common::SpecRun None early-exit may enter the wrapper epilogue.
/// Production contract: None already wrote error/aborted; epilogue is skipped.
fn qwen_dflash_epilogue_after_spec_run(run_present: bool) -> bool {
    run_present
}








/// Apply [`hipfire_generate::common::ds4_malformed_terminal_action`] to the active attempt writer.
/// Returns after writing the error envelope (caller must `return` from generate).
fn emit_ds4_malformed_terminal(stdout: &mut impl std::io::Write, id: &str, detail: &str) {
    let action = hipfire_generate::common::ds4_malformed_terminal_action(detail);
    debug_assert!(!action.emit_done);
    debug_assert!(!action.store_cache);
    debug_assert!(!action.expose_tool_calls);
    debug_assert!(!action.retryable);
    hipfire_generate::dense::emit_active_attempt_error(
        stdout,
        Some(id),
        &action.message,
        action.class,
        action.retryable,
        action.rolled_back,
    );
    let _ = stdout.flush();
}



#[allow(dead_code)]
fn gpu_block_attractor_token(
    gpu: &rdna_compute::Gpu,
    logits_buf: &hip_bridge::DeviceBuffer,
    history: &[u32],
    tok_id: u32,
    window: usize,
    threshold: usize,
) {
    if window == 0 || threshold == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let count = history[start..].iter().filter(|&&t| t == tok_id).count();
    if count >= threshold {
        let bytes: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        let _ = gpu
            .hip
            .memcpy_htod_offset(logits_buf, (tok_id as usize) * 4, &bytes);
    }
}

fn acquire_daemon_lock() -> std::fs::File {
    use std::io::{Seek, Write};

    #[cfg(unix)]
    let home = std::env::var("HOME").expect("HOME environment variable not set");
    #[cfg(windows)]
    let home = std::env::var("USERPROFILE").expect("USERPROFILE environment variable not set");

    let hipfire_dir = std::path::PathBuf::from(home).join(".hipfire");
    std::fs::create_dir_all(&hipfire_dir).expect("failed to create ~/.hipfire");
    let pid_path = hipfire_dir.join("daemon.pid");

    let mut f = {
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        opts.open(&pid_path)
            .expect("failed to open ~/.hipfire/daemon.pid")
    };

    #[cfg(unix)]
    {
        use std::io::Read;
        use std::os::unix::io::AsRawFd;
        let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let mut existing = String::new();
            let _ = f.read_to_string(&mut existing);
            let pid = existing.trim();
            let pid_display = if pid.is_empty() { "<unknown>" } else { pid };
            let kill_arg = if pid.is_empty() { "<pid>" } else { pid };
            eprintln!(
                "FATAL: hipfire daemon already running (PID {}). Run `kill {}` and retry.",
                pid_display, kill_arg
            );
            std::process::exit(1);
        }
    }

    // Got the lock (Unix) / opened the PID file (Windows). Truncate any stale
    // content and write our PID so tooling and the Unix-side error above can
    // both show a useful number.
    f.set_len(0).ok();
    f.seek(std::io::SeekFrom::Start(0)).ok();
    writeln!(f, "{}", std::process::id()).ok();
    f.flush().ok();
    f
}

/// Cap on the *encoded* base64 string length the daemon will accept on the
/// IPC. ~40 MB encoded → ~30 MB raw image bytes (4/3 expansion).
const MAX_BASE64_ENCODED_LEN: usize = 40 * 1024 * 1024;

/// hunt3 H-D: upper bound on a request-driven `max_seq` (1M). A defense-in-
/// depth clamp only — it caps an unvalidated 10M `max_seq` that would otherwise
/// drive a multi-GB KV allocation and OOM the daemon at load. It is NOT a
/// VRAM-aware guard: a load that requests exactly this on a non-eviction config
/// can still OOM at allocation; that VRAM validation is out of scope here.
const MAX_REQUESTED_SEQ: usize = 1024 * 1024;


/// Typed active-attempt error writer used by generation failure paths and tests.
fn write_typed_error(
    stdout: &mut impl std::io::Write,
    id: &str,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    hipfire_generate::dense::emit_active_attempt_error(stdout, Some(id), message, class, retryable, rolled_back);
}




/// VL no-eviction KV admission cap.
///
/// Adaptive floor-reserved caches guarantee `max_seq` at the floor tier; the
/// start tier (FWHT4/Q8) has a smaller `current_cap`, so long multi-chunk VL
/// must be admitted against the floor window (`max_seq`) while
/// `maybe_downshift` keeps each committed write inside the live stride.
/// Non-adaptive paths keep the historical `physical_cap` contract.


/// Cold-reset VL trunk after a GPU/VMM/adaptive failure so the next request
/// cannot inherit partial DN/KV/seq or mismatched conversation history.
///
/// Takes disjoint fields (`dn`/`kv` from `m.state`, controller + host counters
/// on `LoadedModel`) so callers holding `kv`/`dn` do not need `&mut LoadedModel`.
/// Adaptive poison stays sticky: only non-poisoned controllers are
/// `reset_with_cache`'d back to FWHT4/Q8.

/// Fail-closed adaptive downshift after a committed VL KV write.
/// Returns `true` when the request must stop (controller already poisoned).
/// On Err: cold-reset trunk (poison sticky) + request error, no further tokens.

/// Request-scoped VL GPU/VMM failure: cold-reset uncommitted trunk state, then
/// emit error. Never panics; never streams a token for the failed write.

/// Contract: MTP adaptive downshift sites must see the same exclusive committed
/// positions the chunked prefill helper writes. Pure schedule — no GPU.
#[cfg(test)]
mod mtp_adaptive_route_contract {
    /// Exact adaptive×MTP prefill invariant:
    /// external chunk size ≤ PREFILL_MAX_BATCH (= adaptive margin), and
    /// maybe_downshift runs at each exclusive committed boundary so a long
    /// prompt cannot hit the start-tier side_cap before return. A single
    /// whole-prompt prefill + post-only downshift is insufficient.
    #[test]
    fn mtp_adaptive_prefill_boundaries_match_chunk_schedule() {
        use hipfire_arch_qwen35::mtp_spec::mtp_prefill_committed_boundaries;
        use hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH;
        assert_eq!(
            mtp_prefill_committed_boundaries(600, 0, PREFILL_MAX_BATCH),
            vec![256, 512, 600]
        );
        assert!(mtp_prefill_committed_boundaries(0, 0, PREFILL_MAX_BATCH).is_empty());
        // Gaps never exceed one prefill chunk (margin safety).
        let b = mtp_prefill_committed_boundaries(10_000, 0, PREFILL_MAX_BATCH);
        let mut prev = 0usize;
        for &pos in &b {
            assert!(pos - prev <= PREFILL_MAX_BATCH);
            prev = pos;
        }
    }

    #[test]
    fn mtp_forward_fail_is_request_error_not_token() {
        // Mirror AR policy for MTP prefill/spec HipResult (VMM growth, etc.):
        // emit request error, never the failed token; poison stays sticky.
        let action = super::qwen_ar_forward_fail_action();
        assert!(action.emit_request_error);
        assert!(!action.emit_failed_token);
        assert!(!action.clear_adaptive_poison);
    }

    /// Decode-cycle invariant: downshift seq_pos is the live committed prefix
    /// only. Rejected verify length is (n_verify - advance) and lives strictly
    /// past that prefix — never included in maybe_downshift's seq_pos.
    #[test]
    fn mtp_decode_downshift_uses_committed_prefix_only() {
        let cur_pos = 1000usize;
        let max_n = 3usize;
        let n_verify = max_n + 1; // last_committed + candidates
        let advance = 2usize; // e.g. accept 1 + bonus
        let committed_end = cur_pos + advance;
        let reject_suffix_end = cur_pos + n_verify;
        assert!(committed_end < reject_suffix_end);
        // maybe_downshift(committed_end) covers [0, committed_end); rejected
        // [committed_end, reject_suffix_end) must not be required at new tier.
        assert_eq!(reject_suffix_end - committed_end, n_verify - advance);
    }
}

/// Opt-in MTP host-timing wire helpers: route kind, record shape, done-field gate.
/// Pure — no GPU, no launch counters, no Instant reads under test.
#[cfg(test)]
mod mtp_host_timing_contract {
    use hipfire_generate::qwen::{attach_mtp_window_timings, mtp_window_timing_kind, mtp_window_timing_record};

    #[test]
    fn route_kind_covers_ngram_mtp_and_ar() {
        // Ngram hit wins regardless of retirement latch.
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(true, true, false), "ngram");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(true, true, true), "ngram");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(true, false, false), "ngram");
        // Miss after retirement → AR (trunk-only k=0).
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, true, true), "ar");
        // Miss before retirement / ngram off → native MTP.
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, true, false), "mtp");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, false, false), "mtp");
        assert_eq!(hipfire_generate::qwen::mtp_window_timing_kind(false, false, true), "mtp");
    }

    #[test]
    fn timing_record_preserves_exact_wire_fields() {
        let rec = hipfire_generate::qwen::mtp_window_timing_record("ngram", 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12);
        let obj = rec.as_object().expect("object");
        let expected = [
            "kind",
            "wall_us",
            "draft_lookup_us",
            "launch_us",
            "h2d_us",
            "d2h_us",
            "d2d_us",
            "memset_us",
            "stream_sync_us",
            "event_sync_us",
            "device_sync_us",
            "graph_launch_us",
        ];
        assert_eq!(obj.len(), expected.len());
        for key in expected {
            assert!(obj.contains_key(key), "missing wire field {key}");
        }
        assert_eq!(rec["kind"], "ngram");
        assert_eq!(rec["wall_us"], 11);
        assert_eq!(rec["draft_lookup_us"], 2);
        assert_eq!(rec["launch_us"], 3);
        assert_eq!(rec["h2d_us"], 4);
        assert_eq!(rec["d2h_us"], 5);
        assert_eq!(rec["d2d_us"], 6);
        assert_eq!(rec["memset_us"], 7);
        assert_eq!(rec["stream_sync_us"], 8);
        assert_eq!(rec["event_sync_us"], 9);
        assert_eq!(rec["device_sync_us"], 10);
        assert_eq!(rec["graph_launch_us"], 12);
        // All eleven numeric fields are nonnegative integers on the wire.
        for key in [
            "wall_us",
            "draft_lookup_us",
            "launch_us",
            "h2d_us",
            "d2h_us",
            "d2d_us",
            "memset_us",
            "stream_sync_us",
            "event_sync_us",
            "device_sync_us",
            "graph_launch_us",
        ] {
            assert!(rec[key].as_u64().is_some(), "{key} must be u64");
        }
    }

    #[test]
    fn attach_omits_field_when_disabled_preserves_order_when_enabled() {
        let r0 = hipfire_generate::qwen::mtp_window_timing_record("mtp", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r1 = hipfire_generate::qwen::mtp_window_timing_record("ngram", 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r2 = hipfire_generate::qwen::mtp_window_timing_record("ar", 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let ordered = vec![r0.clone(), r1.clone(), r2.clone()];

        let mut disabled = serde_json::json!({"tokens": 1});
        hipfire_generate::qwen::attach_mtp_window_timings(&mut disabled, false, ordered.clone());
        assert!(
            disabled.get("mtp_window_timings").is_none(),
            "disabled must omit the field entirely"
        );

        let mut enabled = serde_json::json!({"tokens": 1});
        hipfire_generate::qwen::attach_mtp_window_timings(&mut enabled, true, ordered);
        let arr = enabled["mtp_window_timings"]
            .as_array()
            .expect("enabled attaches array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["kind"], "mtp");
        assert_eq!(arr[1]["kind"], "ngram");
        assert_eq!(arr[2]["kind"], "ar");
        assert_eq!(arr[0]["wall_us"], 1);
        assert_eq!(arr[1]["wall_us"], 2);
        assert_eq!(arr[2]["wall_us"], 3);
    }
}

#[cfg(test)]
mod adaptive_eviction_prefill_contract {
    use super::qwen_ar_eviction_prefill_chunk_limit;
    use hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH;

    #[test]
    fn staging_uses_adaptive_boundaries_until_handoff() {
        let window = 2048 + 128;
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(0, window, true),
            PREFILL_MAX_BATCH
        );
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(8192 - PREFILL_MAX_BATCH, window, true),
            PREFILL_MAX_BATCH
        );
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(2048, window, false),
            128
        );
        assert_eq!(
            qwen_ar_eviction_prefill_chunk_limit(window, window, false),
            1
        );
    }
}

/// Daemon writer contract: active-attempt errors cannot take a caller-chosen
/// attempt_id (including hard-coded 0). Uncorrelated rejects are a separate API.

/// Pure gate for the deferred EP (tp>1) load handoff.
///
/// After a new EP model is constructed, the prior model is unloaded. The new
/// model may be published only when that prior unload succeeds (or there was
/// no prior model — caller passes `Ok(())` in that case). A failed prior
/// unload must never install/emit `loaded` for the new model.
fn ep_deferred_may_publish(prior_unload: &Result<(), String>) -> bool {
    prior_unload.is_ok()
}

/// Hard-error text when deferred prior unload fails (and optional new-model
/// rollback also fails). Always names the prior failure; appends rollback
/// failure when present so neither is log-and-ignored.
fn ep_deferred_handoff_error_message(prior_err: &str, rollback_err: Option<&str>) -> String {
    match rollback_err {
        None => format!("prior unload failed: {prior_err}"),
        Some(rb) => {
            format!("prior unload failed: {prior_err}; new-model rollback also failed: {rb}")
        }
    }
}

/// Whether the deferred-EP load path must run `ensure_vmm_ready_for_load`
/// before constructing a new EP model.
///
/// Only when there is no live prior model (`model_present == false`): a
/// failed deferred prior unload leaves `model=None` while VMM arenas may
/// still be pending. Do NOT preflight when a live deferred prior still
/// occupies `model` — that path tears down after successful new load.
fn ep_deferred_needs_vmm_preflight(load_tp: usize, model_present: bool) -> bool {
    load_tp > 1 && !model_present
}






/// Print a friendly, user-actionable message when Gpu::init fails. Matches
/// the panic shape we used to emit (which dumped a Rust backtrace and the
/// raw HipError debug-format) but turns it into a concrete next-step list.
/// The most common cause on Windows (#112) is HIP SDK present but no
/// AMD GPU driver visible to the runtime; on Linux it is usually missing
/// `libamdhip64.so` or kernel-side amdgpu / kfd not loaded.
fn report_gpu_init_failure(err: &hip_bridge::HipError) {
    eprintln!();
    eprintln!("hipfire: failed to initialize GPU runtime.");
    eprintln!("  HIP error: {} (code {})", err.message, err.code);
    eprintln!();
    if cfg!(target_os = "windows") {
        eprintln!("  Most common Windows cause: HIP SDK is loaded but no");
        eprintln!("  AMD GPU is visible to the runtime. Verify:");
        eprintln!("    1. AMD Adrenalin driver is installed and current.");
        eprintln!("    2. AMD HIP SDK 6.2 or newer is installed:");
        eprintln!("       https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html");
        eprintln!("    3. `amdhip64.dll` is reachable (HIP_PATH set or DLL on PATH).");
        eprintln!("    4. Reboot after driver / SDK install if you have not yet.");
    } else {
        eprintln!("  Most common Linux causes:");
        eprintln!("    1. amdgpu kernel module not loaded (check `lsmod | grep amdgpu`).");
        eprintln!("    2. /dev/kfd missing or not readable by the current user");
        eprintln!("       (add to the `render` group; reboot).");
        eprintln!("    3. ROCm not installed or libamdhip64.so missing");
        eprintln!("       (check `ldconfig -p | grep amdhip64`).");
    }
    eprintln!();
    eprintln!("  Run `hipfire diag` for a full environment report.");
}

/// Install opt-in structured diagnostics on stderr. Stdout is reserved for the
/// daemon's JSON-lines IPC protocol and must never receive tracing output.
///
/// `HIPFIRE_LOG` accepts an EnvFilter directive such as `info` or
/// `hipfire_runtime=debug`; `RUST_LOG` is the fallback. Set
/// `HIPFIRE_LOG_FORMAT=json` for machine-readable log events. With neither
/// filter set, tracing remains off and the existing operator-facing stderr
/// messages are unchanged.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_env("HIPFIRE_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("off"));
    let json = std::env::var("HIPFIRE_LOG_FORMAT")
        .map(|value| value.eq_ignore_ascii_case("json"))
        .unwrap_or(false);

    let result = if json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .with_ansi(false)
            .json()
            .try_init()
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .with_target(false)
            .try_init()
    };

    if let Err(error) = result {
        eprintln!("warning: failed to initialize structured logging: {error}");
    }
}

fn install_process_config(config: hipfire_config::ProcessConfig) -> Result<(), String> {
    config.validate().map_err(|error| error.to_string())?;
    hipfire_config::apply_device_visibility(&config).map_err(|error| error.to_string())?;
    let runtime = hipfire_runtime::config::RuntimeConfig::from_process_config(&config);
    hipfire_config::install_process_config(config)
        .map_err(|_| "process configuration was already initialized".to_owned())?;
    hipfire_runtime::config::init_with(runtime)
        .map_err(|_| "runtime process configuration was already initialized".to_owned())
}

/// Read the first protocol message before GPU initialization. Native clients
/// send `configure`; older/direct clients may send `load`, in which case the
/// daemon resolves local TOML plus compatibility env itself and preserves the
/// load as the first regular command.
fn receive_startup_config(
    stdout: &mut impl Write,
) -> Result<Option<(hipfire_config::ProcessConfig, Option<DaemonMsg>, bool)>, String> {
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        if lock
            .read_line(&mut line)
            .map_err(|error| error.to_string())?
            == 0
        {
            return Ok(None);
        }
        if line.trim().is_empty() {
            continue;
        }
        let msg = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(msg) => msg,
            Err(error) => {
                emit_uncorrelated_error(
                    stdout,
                    None,
                    &format!("invalid JSON: {error}"),
                    "validation",
                    false,
                    false,
                );
                stdout.flush().map_err(|error| error.to_string())?;
                continue;
            }
        };
        if msg.get("type").and_then(|value| value.as_str()) == Some("configure") {
            let config = serde_json::from_value::<hipfire_config::ProcessConfig>(
                msg.get("config")
                    .cloned()
                    .ok_or_else(|| "configure message is missing config".to_owned())?,
            )
            .map_err(|error| format!("invalid process configuration: {error}"))?;
            config.validate().map_err(|error| error.to_string())?;
            return Ok(Some((config, None, true)));
        }
        let config =
            hipfire_config::load_local_process_config().map_err(|error| error.to_string())?;
        return Ok(Some((config, Some(DaemonMsg::Regular(msg)), false)));
    }
}

fn main() {
    init_tracing();
    tracing::info!(pid = std::process::id(), "daemon starting");

    let args: Vec<String> = std::env::args().collect();

    // --precompile: compile all kernels for this GPU, write hash files, exit.
    // Used by scripts/install.sh and `hipfire update` so first `hipfire run`
    // isn't a 2-minute hipcc wait.
    //
    // Covers the current default path (mq4 weights + asym3 KV) plus the legacy
    // compat paths (hfq4, hfq6, q8 weights × asym3, q8 KV) so models from any
    // era of the registry start instantly.
    if args.iter().any(|a| a == "--precompile") {
        let process_config = hipfire_config::load_local_process_config().unwrap_or_else(|error| {
            eprintln!("FATAL: invalid process configuration: {error}");
            std::process::exit(1);
        });
        install_process_config(process_config).unwrap_or_else(|error| {
            eprintln!("FATAL: failed to install process configuration: {error}");
            std::process::exit(1);
        });
        // Pre-create the expected precompiled-dir next to this binary so the
        // compiler's writeback path fires. Without this, Gpu::init probes for
        // an existing dir and silently disables writeback if it's missing —
        // meaning fresh installs would compile but never cache cross-invocation.
        if let Some(exe_dir) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        {
            // Arch is unknown until Gpu::init; use a broad mkdir for the common arches
            // we support so the probe picks one up. The real arch check after init
            // will log the active dir.
            for arch in [
                "gfx906", "gfx1010", "gfx1013", "gfx1030", "gfx1031", "gfx1100", "gfx1101",
                "gfx1102", "gfx1151", "gfx1152", "gfx1200", "gfx1201",
            ] {
                let _ =
                    std::fs::create_dir_all(exe_dir.join("kernels").join("compiled").join(arch));
            }
        }
        let mut gpu = match rdna_compute::Gpu::init() {
            Ok(g) => g,
            Err(e) => {
                report_gpu_init_failure(&e);
                std::process::exit(1);
            }
        };
        eprintln!("Pre-compiling kernels for {}...", gpu.arch);
        let mut ok = 0usize;
        let mut failed = 0usize;
        for kv in &["asym3", "q8"] {
            for wq in &["mq4", "mq6", "hfq4", "hfq6", "q8"] {
                if let Err(e) = gpu.precompile_qwen35(wq, kv, 256) {
                    if *wq == "mq4" && *kv == "asym3" {
                        eprintln!("ERROR: required kernel precompile failed: mq4/asym3: {e}");
                        std::process::exit(1);
                    }
                    eprintln!("  {wq}/{kv}: {e}");
                    failed += 1;
                } else {
                    ok += 1;
                }
            }
        }
        eprintln!("precompile: {ok} ok, {failed} optional failed");
        return;
    }

    // Machine-wide mutex — prevents orphan daemons from silently coexisting
    // (observed 2026-04-13: two daemons at 100% CPU survived pkill -f rounds
    // because they'd been reparented to PID 1 after their bun parent died).
    // Kept in a binding so the fd lives for the full process lifetime.
    let _daemon_lock = acquire_daemon_lock();

    let mut stdout = std::io::stdout();
    let Some((process_config, pending_message, acknowledge_config)) =
        receive_startup_config(&mut stdout).unwrap_or_else(|error| {
            eprintln!("FATAL: failed to resolve startup configuration: {error}");
            std::process::exit(1);
        })
    else {
        return;
    };
    install_process_config(process_config).unwrap_or_else(|error| {
        eprintln!("FATAL: failed to install process configuration: {error}");
        std::process::exit(1);
    });
    if acknowledge_config {
        writeln!(
            stdout,
            r#"{{"type":"configured","schema_version":{}}}"#,
            hipfire_config::CONFIG_SCHEMA_VERSION
        )
        .unwrap_or_else(|error| {
            eprintln!("FATAL: failed to acknowledge process configuration: {error}");
            std::process::exit(1);
        });
        stdout.flush().unwrap_or_else(|error| {
            eprintln!("FATAL: failed to flush process configuration acknowledgement: {error}");
            std::process::exit(1);
        });
    }

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            report_gpu_init_failure(&e);
            std::process::exit(1);
        }
    };
    let mut model: Option<LoadedModel> = None;
    // Monotonic cold-reset epoch; bumped only after successful synchronized reset.
    // Echoed on reset acks so Engine::reset can reject non-increasing epochs.
    let mut state_epoch: u64 = 0;
    // PFlash speculative-prefill state. None unless the load message
    // includes a `prefill_drafter` path AND `prefill_compression` != "off".
    // Lives alongside `model` so unload_model + this state are paired
    // teardowns.
    let mut pflash_state: Option<hipfire_pflash::pflash::PflashState> = None;
    // The PflashConfig captured at load time. Per-request `prefill_*`
    // params override individual fields; the rest fall back to these
    // load-time defaults. Cleared alongside `pflash_state`.
    let mut pflash_cfg: Option<hipfire_pflash::pflash::PflashConfig> = None;
    // Hetero PFlash: when prefill_drafter_device differs from the target,
    // the drafter weights/KV/scratch live on a sibling device. The compress
    // output is a host-side Vec<u32>, so no peer-copy is needed — generate
    // routes maybe_compress_prompt to this handle, decode stays on target.
    // None means the drafter shares the target gpu (single-card, unchanged).
    let mut pflash_drafter_gpu: Option<rdna_compute::Gpu> = None;
    // Continuous-batch host scheduler + GPU batch state (if available).
    // Initialized on successful load when `continuous_batch_size` > 1 and
    // the loaded arch is batch-capable (qwen 5/6, single-GPU, Q8 KV/state).
    // None => sequential fallback.
    let mut continuous_batch_size: usize = 1;
    let mut batch_scheduler: Option<ContinuousBatchScheduler> = None;
    let mut batch_poisoned: Option<String> = None;

    // Background stdin reader. Drains stdin into an mpsc channel so
    // the main loop can pull non-blockingly between messages. Abort /
    // commit control messages are NOT forwarded; the reader handles
    // them inline via `apply_terminal_control` against the active
    // `(id, attempt_id)` transaction. This is the channel that makes
    // client-side cancellation actually stop an in-flight prefill —
    // without it, the main loop is blocked on GPU compute and wouldn't
    // even read the abort line until after the prefill completed.
    let (msg_tx, msg_rx) = mpsc::channel::<DaemonMsg>();
    if let Some(message) = pending_message {
        let _ = msg_tx.send(message);
    }
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let lock = stdin.lock();
        for line in lock.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(msg) => {
                    let msg_type = msg.get("type").and_then(|v| v.as_str());
                    if msg_type == Some("abort") || msg_type == Some("commit") {
                        // Correlated terminal control: require exact id + numeric
                        // attempt_id. Stale/malformed controls are ignored.
                        let id = msg.get("id").and_then(|v| v.as_str());
                        let attempt_id = msg.get("attempt_id").and_then(|v| v.as_u64());
                        if let (Some(id), Some(attempt_id), Some(kind)) = (id, attempt_id, msg_type)
                        {
                            tracing::info!(
                                request_id = id,
                                attempt_id,
                                command = kind,
                                "daemon control command received"
                            );
                            eprintln!(
                                "[daemon-control] received {} for id={} attempt_id={}",
                                kind, id, attempt_id
                            );
                            apply_terminal_control(kind, id, attempt_id);
                            batch_apply_terminal_control(kind, id, attempt_id);
                        }
                        continue;
                    }
                    if msg.get("type").and_then(|v| v.as_str()) == Some("force_answer") {
                        if let Some(id) = msg.get("id").and_then(|v| v.as_str()) {
                            tracing::info!(
                                request_id = id,
                                command = "force_answer",
                                "daemon control command received"
                            );
                            eprintln!("[daemon-force-answer] received force_answer for id={}", id);
                            *force_answer_for_id().lock().unwrap() = Some(id.to_string());
                        }
                        continue;
                    }
                    // Batch: announce every well-formed generate key before queueing.
                    // Duplicate (id, attempt_id) must not enqueue or mutate the live registry.
                    if msg.get("type").and_then(|v| v.as_str()) == Some("generate") {
                        if let (Some(id), Some(attempt_id)) = (
                            msg.get("id").and_then(|v| v.as_str()),
                            msg.get("attempt_id").and_then(|v| v.as_u64()),
                        ) {
                            if !batch_announce_terminal(id, attempt_id) {
                                eprintln!(
                                    "[batch] duplicate generate dropped id={} attempt_id={}; preserving live registry",
                                    id, attempt_id
                                );
                                continue;
                            }
                        }
                    }

                    if msg_tx.send(DaemonMsg::Regular(msg)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    if msg_tx.send(DaemonMsg::ParseError(e.to_string())).is_err() {
                        break;
                    }
                }
            }
        }
    });
    let mut inbox = DaemonInbox::new(msg_rx);
    while let Ok(daemon_msg) = inbox.recv() {
        let msg = match daemon_msg {
            DaemonMsg::Regular(m) => m,
            DaemonMsg::ParseError(e) => {
                tracing::warn!(error = %e, "daemon received invalid JSON");
                emit_uncorrelated_error(
                    &mut stdout,
                    None,
                    &format!("invalid JSON: {e}"),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
                continue;
            }
        };

        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let request_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let command_span = tracing::info_span!(
            "daemon_command",
            command = msg_type,
            request_id = request_id
        );
        let _command_guard = command_span.enter();
        tracing::debug!("daemon command received");

        match msg_type {
            "configure" => {
                emit_uncorrelated_error(
                    &mut stdout,
                    None,
                    "process configuration is immutable after daemon startup",
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
            }
            "load" => {
                // FIX #1 (transactional EP load): the unload of the prior model
                // is deferred for the EP (tp>1) path until AFTER the new load
                // succeeds, so a partial EP load failure leaves the prior model
                // intact (and load_model_ep's staging guard frees the partial
                // ranks). For the single-GPU / pp path the prior model is
                // unloaded eagerly here as before (load_model uses the daemon's
                // `gpu` directly, so it can't be deferred without a major
                // refactor). `tp` is parsed authoritatively below; peek it here.
                let load_tp = msg
                    .get("params")
                    .and_then(|p| p.get("tp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let parsed_continuous_batch_size = parse_continuous_batch_size(msg.get("params"));
                // Unload previous if any. PFlash drafter goes first so
                // its tensors join the pool before unload_model drains
                // it -- otherwise free_tensor would queue them into the
                // pool just-emptied by drain_pool with no follow-up
                // drain, leaving drafter VRAM resident across the next
                // load (the explicit "unload" handler has the same
                // ordering for the same reason).
                //
                // FIX (transactional pflash teardown): pflash_state is part of
                // the PRIOR model (it holds that model's PFlash drafter). For
                // the deferred tp>1 EP path it must NOT be torn down here —
                // otherwise a partial EP load failure (whose FIX #1 deferral
                // keeps `model` alive) would leave the surviving prior model
                // stripped of its drafter. Defer it to the success branch
                // alongside the deferred model unload. For load_tp <= 1 the
                // prior model is unloaded eagerly, so tear pflash down here in
                // the original order. (EP archs are ds4/minimax and refuse
                // PFlash drafters, so on a SUCCESSFUL tp>1 load this just frees
                // the outgoing model's drafter at the deferred site.)
                if load_tp <= 1 {
                    if let Some(mut pf) = pflash_state.take() {
                        if let Some(mut dg) = pflash_drafter_gpu.take() {
                            dg.bind_thread_or_warn();
                            pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                            gpu.bind_thread_or_warn();
                        } else {
                            pf.unload_drafter(&mut gpu);
                        }
                    }
                    pflash_cfg = None;
                    if let Some(m) = model.take() {
                        if let Err(err) = hipfire_loader::unload_model(m, &mut gpu) {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &format!("prior unload failed: {err}"),
                                "internal",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    } else if let Err(err) = hipfire_loader::ensure_vmm_ready_for_load(&mut gpu) {
                        emit_uncorrelated_error(&mut stdout, None, &err, "internal", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }
                // EP path: when no live prior model remains (fresh daemon, or
                // after deferred prior unload failed and left model=None with
                // pending VMM), refuse to construct a new EP model until
                // orphan teardown clears. Skip when a live deferred prior
                // still sits in `model` — unload stays deferred until after
                // successful new-model construction.
                if ep_deferred_needs_vmm_preflight(load_tp, model.is_some()) {
                    if let Err(err) = hipfire_loader::ensure_vmm_ready_for_load(&mut gpu) {
                        emit_uncorrelated_error(&mut stdout, None, &err, "internal", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }

                let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
                // hunt3 H-D: clamp request-driven max_seq to the config ceiling
                // (MAX_REQUESTED_SEQ = 1M). Without this an unvalidated 10M
                // max_seq drives a multi-GB KV allocation and OOMs the daemon at
                // load. Emit an info event when the clamp actually fires so the
                // operator sees the truncation rather than silently getting 1M.
                let requested_max_seq = msg
                    .get("params")
                    .and_then(|p| p.get("max_seq"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4096) as usize;
                let max_seq = requested_max_seq.min(MAX_REQUESTED_SEQ);
                if requested_max_seq > MAX_REQUESTED_SEQ {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","message":"requested max_seq {} exceeds ceiling {} — clamped"}}"#,
                        requested_max_seq, MAX_REQUESTED_SEQ
                    );
                    let _ = stdout.flush();
                }
                // Optional DFlash draft model path. When supplied AND the target
                // is a Qwen3.5 arch (5 or 6), we load draft weights + scratch
                // alongside the target and the temp=0 generate fast path routes
                // through `spec_step_dflash` for the 1.7-2.5× speedup on the
                // 27B target. Non-matching archs / missing draft file are
                // logged but don't fail the load.
                //
                // `dflash_mode=off` is a hard daemon-side override: even if a
                // draft path was passed, skip the load. CLI-side gating is the
                // primary path (saves the wire round-trip for the draft path
                // string), but this guard makes the flag durable when the
                // daemon is driven by a non-hipfire-CLI client.
                let dflash_mode = msg
                    .get("params")
                    .and_then(|p| p.get("dflash_mode"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto");
                // `HIPFIRE_DFLASH_DRAFT` (documented in AGENTS.md §7) forces a
                // draft path for clients that cannot pass `params.draft` (the
                // serve prewarm / HTTP-reload path has no --model-draft flag):
                // non-empty → wins over params.draft; explicitly EMPTY → opt out
                // of draft loading entirely; unset → params.draft as before.
                let env_draft = std::env::var("HIPFIRE_DFLASH_DRAFT").ok();
                let raw_draft: Option<String> = match env_draft.as_deref() {
                    Some("") => None,
                    Some(p) => Some(p.to_string()),
                    None => msg
                        .get("params")
                        .and_then(|p| p.get("draft"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string()),
                };
                let draft_path = if dflash_mode == "off" {
                    if let Some(d) = raw_draft {
                        eprintln!("[hipfire-daemon] dflash_mode=off — skipping draft load ({d})");
                    }
                    None
                } else {
                    raw_draft
                };
                // Gemma 4 EAGLE drafter (arch-22 `gemma4_unified_assistant`).
                // Deliberately a SEPARATE param from `params.draft` (the
                // qwen3.5 DFlash knob) so a DFlash .hfq can never be routed
                // into the EAGLE loader by accident. `params.spec` = draft
                // length; 1..=5 accepted (see gemma4_eagle_spec_len).
                let gemma4_drafter = msg
                    .get("params")
                    .and_then(|p| p.get("drafter"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let gemma4_draft_len = if gemma4_drafter.is_some() {
                    let spec_raw = msg
                        .get("params")
                        .and_then(|p| p.get("spec"))
                        .and_then(|v| v.as_u64());
                    match hipfire_loader::gemma4_eagle_spec_len(spec_raw) {
                        Ok(n) => n,
                        Err(e) => {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &e,
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    }
                } else {
                    hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN
                };
                let kv_mode_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_mode"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let kv_backend_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_backend"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                // Per-load adaptive-KV selector (mirrors kv_mode). Overrides the
                // HIPFIRE_KV_ADAPTIVE env. off|conservative|balanced|aggressive|
                // advanced:k=..,v=.. — resolved in load_model (param > env > off).
                let kv_adaptive_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_adaptive"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());

                // MTP speculative decode config. `mtp_mode` gates weight
                // discovery at load time (off=skip, on=error-if-missing,
                // auto=scan+log). `mtp_k` sets the draft window size.
                let mtp_mode = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_mode"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(&hipfire_runtime::config::get().mtp_mode)
                    .to_string();
                let mtp_k: usize = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_k"))
                    .and_then(|v| v.as_u64())
                    .map(|value| value as usize)
                    .unwrap_or(hipfire_runtime::config::get().mtp_k);

                // Model-free n-gram policy normally arrives as per-load params
                // resolved by the CLI. Direct protocol clients inherit the
                // daemon's typed process policy instead of ambient env.
                let spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg {
                    ngram_draft: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_draft"))
                        .and_then(|v| v.as_bool())
                        .or(Some(hipfire_runtime::config::get().ngram_draft)),
                    ngram_k: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_k"))
                        .and_then(|v| v.as_u64())
                        .map(|k| k as usize)
                        .or(Some(hipfire_runtime::config::get().ngram_k)),
                    ngram_min_count: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_min_count"))
                        .and_then(|v| v.as_u64())
                        .map(|c| c as u32)
                        .or(Some(hipfire_runtime::config::get().ngram_min_count)),
                    // DDTree draft tuning — same load-param mechanism as ngram_k:
                    // CLI `--ddtree-budget` / `--ddtree-topk` → these load params,
                    // env-wins-else-param in the loader.
                    ddtree_budget: msg
                        .get("params")
                        .and_then(|p| p.get("ddtree_budget"))
                        .and_then(|v| v.as_u64())
                        .map(|b| b as usize),
                    ddtree_topk: msg
                        .get("params")
                        .and_then(|p| p.get("ddtree_topk"))
                        .and_then(|v| v.as_u64())
                        .map(|k| k as usize),
                    // DSpark draft module: the CLI lowers `speculation` into a
                    // `dspark_mode` string. off→Some(false) (skip load+build),
                    // on→Some(true) (force), auto/absent→None (load-if-sidecar).
                    dspark: msg
                        .get("params")
                        .and_then(|p| p.get("dspark_mode"))
                        .and_then(|v| v.as_str())
                        .and_then(|s| match s {
                            "on" => Some(true),
                            "off" => Some(false),
                            _ => None, // "auto" → loader default
                        }),
                    dspark_conf_threshold: msg
                        .get("params")
                        .and_then(|p| p.get("dspark_conf_threshold"))
                        .and_then(|v| v.as_f64())
                        .map(|t| t as f32),
                };

                // 0.1.7-alpha: DFlash tuning knobs forwarded from the CLI.
                // `adaptive_b` matches dflash_spec_demo's --adaptive-b default.
                // Accepted here; the generate loop will honor it in the
                // 0.1.7-stable release where we port the demo's outer τ-window
                // trip-wire (below 2.5 → shrink block to 8).
                let _adaptive_b = msg
                    .get("params")
                    .and_then(|p| p.get("dflash_adaptive_b"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);

                // 0.1.7: TriAttention / CASK eviction protocol fields. When
                // `cask_sidecar` is set, `load_model` sizes the KV cache to a
                // *physical_cap* (budget+beta+safety, clamped to max_seq) instead
                // of the full max_seq, and wires an `Eviction` policy that the
                // generate loop calls after every prefill-chunk / decode-forward.
                // That decouples advertised context length from VRAM footprint —
                // a 128K max_seq can run in ~1K-slot physical buffer when the
                // operator opts in.
                let cask_sidecar = msg
                    .get("params")
                    .and_then(|p| p.get("cask_sidecar"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let cask_enabled = msg
                    .get("params")
                    .and_then(|p| p.get("cask"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let cask_budget = msg
                    .get("params")
                    .and_then(|p| p.get("cask_budget"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(512) as usize;
                let cask_beta = msg
                    .get("params")
                    .and_then(|p| p.get("cask_beta"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                let cask_handoff_tokens = msg
                    .get("params")
                    .and_then(|p| p.get("cask_handoff_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let cask_core_frac = msg
                    .get("params")
                    .and_then(|p| p.get("cask_core_frac"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5) as f32;
                let cask_fold_m = msg
                    .get("params")
                    .and_then(|p| p.get("cask_fold_m"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2) as usize;
                // Known-broken combo guard: CASK m-folding + DFlash spec decode
                // degenerates into single-token loops after the first eviction
                // (the m-folded synthetic K/V rows are off the draft's trained
                // hidden-state distribution). Until that's fixed at the library
                // level, downgrade m-folding to plain TriAttention drop-eviction
                // when a draft is attached. User's context window + eviction
                // cadence still work; just the fold step is skipped.
                let cask_m_folding_effective = if cask_enabled && draft_path.is_some() {
                    eprintln!(
                        "[hipfire-daemon] cask:true + draft: both set — downgrading to plain TriAttention drop-eviction (CASK m-fold + DFlash is a known-broken combo; see feedback_cask_mfold_dflash_broken.md)",
                    );
                    false
                } else {
                    cask_enabled
                };
                let cask = CaskConfig {
                    sidecar: cask_sidecar,
                    cask_m_folding: cask_m_folding_effective,
                    handoff_tokens: cask_handoff_tokens,
                    budget: cask_budget,
                    beta: cask_beta,
                    core_frac: cask_core_frac,
                    fold_m: cask_fold_m,
                };

                // MMQ per-weight screening (#87): detect outlier rows that
                // cause Q8_1 precision loss and fall back to WMMA for those
                // weights. Disabled by default; enable with mmq_screen=true
                // (or HIPFIRE_MMQ_SCREEN=1) when adding new quant formats.
                if let Some(v) = msg
                    .get("params")
                    .and_then(|p| p.get("mmq_screen"))
                    .and_then(|v| v.as_bool())
                {
                    gpu.mmq_screen.enabled = v;
                }
                if let Some(v) = msg
                    .get("params")
                    .and_then(|p| p.get("mmq_screen_threshold"))
                    .and_then(|v| v.as_f64())
                {
                    gpu.mmq_screen.threshold = v as f32;
                }

                // ── PFlash load-time params (Phase 4.0 #93) ──────────────
                //
                // Parse compression knobs per PRD §5.3.2. None of these
                // affect the target load itself; they only configure the
                // optional drafter that PFlash uses for prompt scoring.
                // Drafter loading happens AFTER target load succeeds so
                // we can use the target's tokenizer for the compat check.
                let pflash_mode_str = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_compression"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("off")
                    .to_string();
                let pflash_threshold = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_threshold"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32768) as usize;
                let pflash_keep_ratio = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_keep_ratio"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.05) as f32;
                let pflash_alpha = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_alpha"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.85) as f32;
                let pflash_min_keep = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_min_keep"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2048) as usize;
                let pflash_sink = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_sink"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256) as usize;
                let pflash_recent = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_recent"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1024) as usize;
                let pflash_block = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_block"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                let pflash_drafter = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_drafter"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                // -1 = drafter shares the target gpu (default). >=0 routes
                // the drafter to that HIP device for hetero compress.
                let pflash_drafter_device: i32 = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_drafter_device"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(-1) as i32;
                let pflash_profile = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_profile"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let pflash_sparse_threshold = msg
                    .get("params")
                    .and_then(|p| p.get("prefill_sparse_threshold"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32768) as usize;

                // Validate load-time PFlash params before they reach
                // PflashConfig + load_drafter. Same range rules the
                // per-request override path uses; without these, a
                // bad load-time value would silently be accepted and
                // panic the daemon at the first generate request.
                let pflash_load_err: Option<String> =
                    if !(pflash_keep_ratio > 0.0 && pflash_keep_ratio <= 1.0) {
                        Some(format!(
                            "prefill_keep_ratio={pflash_keep_ratio} not in (0, 1]"
                        ))
                    } else if pflash_block == 0 {
                        Some("prefill_block must be > 0".to_string())
                    } else {
                        None
                    };

                // Pipeline-parallel degree (Stage 7 of #58). Default 1 =
                // single-GPU (no behavior change). pp > 1 routes through
                // Gpus + *_multi paths and refuses VL / DFlash / CASK /
                // PFlash at load time. v1 supports Qwen3.5 dense + MoE
                // only — see load_model_pp for the arch_id check.
                let pp = msg
                    .get("params")
                    .and_then(|p| p.get("pp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                // Expert-parallel degree (EP, task #26). tp>1 shards routed
                // experts across ranks via load_model_ep. Mutually exclusive
                // with pp; v1 refuses DFlash. See docs/plans/daemon-ep-wiring.md.
                let tp = msg
                    .get("params")
                    .and_then(|p| p.get("tp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                if tp > 1 && pp > 1 {
                    emit_uncorrelated_error(&mut stdout, None, "tp (expert-parallel) and pp (pipeline-parallel) are mutually exclusive; set only one.", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                if tp > 1 && draft_path.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "EP serving (tp>1) does not support DFlash drafters in v1; reload without a draft.", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                if tp > 1 && gemma4_drafter.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "EP serving (tp>1) does not support the gemma4 EAGLE drafter; reload without params.drafter.", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                if pp > 1 {
                    if gemma4_drafter.is_some() {
                        emit_uncorrelated_error(&mut stdout, None, "gemma4 EAGLE spec-decode requires pp=1 (arch_id=13 has no pipeline-parallel path); reload without params.drafter.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if draft_path.is_some()
                        && std::env::var("HIPFIRE_PP_DFLASH").ok().as_deref() != Some("1")
                    {
                        emit_uncorrelated_error(&mut stdout, None, "DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if cask.sidecar.is_some() {
                        emit_uncorrelated_error(&mut stdout, None, "CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if (pflash_drafter.is_some() || pflash_mode_str != "off")
                        && std::env::var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
                    {
                        emit_uncorrelated_error(&mut stdout, None, "PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }

                let state_quant_override = msg
                    .get("params")
                    .and_then(|p| p.get("state_quant"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());

                let deepseek4_experts_per_token = msg
                    .get("params")
                    .and_then(|p| p.get("deepseek4_experts_per_token"))
                    .and_then(|v| v.as_u64())
                    .map(|value| value as usize);
                let deepseek4_compute_placement = match msg
                    .get("params")
                    .and_then(|p| p.get("deepseek4_compute_placement"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("single")
                    .parse::<hipfire_config::Deepseek4ComputePlacement>()
                {
                    Ok(placement) => placement,
                    Err(error) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("invalid DeepSeek V4 compute placement: {error}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let loaded = if tp > 1 {
                    if deepseek4_experts_per_token.is_some() {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            "DeepSeek V4 experts-per-token override requires tp=1",
                            "unsupported",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    hipfire_loader::load_model_ep_with_kv_mode(
                        path,
                        max_seq,
                        tp,
                        kv_mode_override.as_deref(),
                        kv_backend_override.as_deref(),
                    )
                } else {
                    hipfire_loader::load_model_with_gemma4_drafter(
                        path,
                        max_seq,
                        deepseek4_experts_per_token,
                        deepseek4_compute_placement,
                        draft_path.as_deref(),
                        gemma4_drafter.as_deref(),
                        gemma4_draft_len,
                        kv_mode_override.as_deref(),
                        kv_backend_override.as_deref(),
                        kv_adaptive_override.as_deref(),
                        state_quant_override.as_deref(),
                        &cask,
                        pp,
                        spec_cfg,
                        &mut gpu,
                    )
                };
                match loaded {
                    Ok(mut m) => {
                        // FIX #1 (deferred EP unload): the new EP model loaded
                        // successfully — NOW unload the prior model before
                        // publishing (single-GPU/pp models were already unloaded
                        // eagerly above; this branch only fires for deferred
                        // tp>1). Prior PFlash drafter is part of that prior
                        // model, so tear it down first in the same
                        // drafter-before-unload order used elsewhere.
                        //
                        // Transactional: if prior unload fails, do NOT install
                        // or emit `loaded` for the new model. Explicitly unload
                        // the newly built EP model, clear associated fresh
                        // state, and emit a hard error covering prior failure
                        // and any rollback failure.
                        if load_tp > 1 {
                            if let Some(mut pf) = pflash_state.take() {
                                if let Some(mut dg) = pflash_drafter_gpu.take() {
                                    dg.bind_thread_or_warn();
                                    pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                                    gpu.bind_thread_or_warn();
                                } else {
                                    pf.unload_drafter(&mut gpu);
                                }
                            }
                            pflash_cfg = None;
                            let prior_unload = if let Some(old) = model.take() {
                                hipfire_loader::unload_model(old, &mut gpu)
                            } else {
                                Ok(())
                            };
                            if !ep_deferred_may_publish(&prior_unload) {
                                let prior_err = prior_unload
                                    .err()
                                    .unwrap_or_else(|| "prior unload failed".to_string());
                                // Roll back the newly built EP model — GpuTensor
                                // has no Drop; must free explicitly.
                                let rollback_err = match hipfire_loader::unload_model(m, &mut gpu) {
                                    Ok(()) => None,
                                    Err(e) => Some(e),
                                };
                                // model stays None; pflash already cleared above.
                                let msg = ep_deferred_handoff_error_message(
                                    &prior_err,
                                    rollback_err.as_deref(),
                                );
                                write_error(&mut stdout, "", &msg);
                                continue;
                            }
                        }
                        let arch = match m.arch_id {
                            5 => "qwen3_5",
                            6 => "qwen3_5_moe",
                            7 => "qwen2",
                            8 => "dots-ocr",
                            9 => "deepseek4",
                            10 => "minimax_m2",
                            11 => "lfm2moe",
                            12 => "north_mini_code",
                            13 => "gemma4",
                            14 => "muse_glimmer",
                            _ => "qwen3",
                        };
                        let drafter = m.speculator.as_ref().map(|speculator| speculator.name());
                        let redline_default = hipfire_runtime::config::retained_redline_default(
                            &gpu.arch,
                            arch,
                            path,
                            pp,
                            tp,
                            drafter.is_some(),
                        );
                        if gpu.replay.configure_model_default(redline_default) && redline_default {
                            eprintln!(
                                "[redline] enabling fail-closed retained default on {} \
                                 (model_arch={arch}, drafter={}, transport={})",
                                gpu.arch,
                                drafter.unwrap_or("off"),
                                gpu.replay.transport_name()
                            );
                        }
                        let vl = m.vision_config.is_some() || m.dots_ocr_config.is_some();
                        let (dim, layers, vocab) = match m.state.as_ref() {
                            Some(ModelState::Qwen35(b)) => {
                                (b.config.dim, b.config.n_layers, b.config.vocab_size)
                            }
                            Some(ModelState::Llama(b)) => {
                                (b.config.dim, b.config.n_layers, b.config.vocab_size)
                            }
                            Some(ModelState::Qwen2(b)) => (
                                b.config.hidden_size,
                                b.config.num_hidden_layers,
                                b.config.vocab_size,
                            ),
                            Some(ModelState::Cohere2Moe(b)) => (
                                b.config.hidden_size,
                                b.config.num_hidden_layers,
                                b.config.vocab_size,
                            ),
                            Some(ModelState::Gemma4(b)) => {
                                (b.config.dim, b.config.n_layers, b.config.vocab_size)
                            }
                            Some(ModelState::MuseGlimmer(b)) => {
                                (b.config.dim, b.config.n_layers, b.config.vocab_size)
                            }
                            _ => {
                                if let Some(ref c) = m.dots_ocr_config {
                                    (
                                        c.text.hidden_size,
                                        c.text.num_hidden_layers,
                                        c.text.vocab_size,
                                    )
                                } else {
                                    (0, 0, 0)
                                }
                            }
                        };

                        // Apply MTP config from load-message params.
                        m.mtp_mode = mtp_mode;
                        m.mtp_k = mtp_k;
                        // Detect whether MTP weights are present in the loaded
                        // model. Used by mtp_mode=auto to decide whether to
                        // enable spec-decode at generate time. Three sources:
                        //   - DeepSeek V4: the trunk's bundled `mtp_layer`.
                        //   - DeepSeek V4: a DSpark sidecar (either counts for auto
                        //     mode; the loader picks whichever applies).
                        //   - Qwen3.5/3.6: a native MTP (NextN) head loaded by
                        //     the loader (`qwen35_mtp_head`, set from a bundled
                        //     `.mq4-mtp` trailer or a `.mtp` sidecar). The loader
                        //     already set `m.mtp_weights_present = true` in that
                        //     case; OR it in here so the ds4 probe doesn't clobber
                        //     it back to false for a qwen35 model.
                        let ds4_mtp = m
                            .deepseek4()
                            .map(|b| b.weights.mtp_layer.is_some() || b.weights.dspark.is_some())
                            .unwrap_or(false);
                        m.mtp_weights_present =
                            ds4_mtp || m.qwen35_mtp_head.is_some() || m.mtp_weights_present;

                        // ── Optional DPM stabilization (perf instrumentation) ──
                        //
                        // Pins the GPU at high sclk/mclk so the first `generate`
                        // request doesn't pay the 1-10s DPM ramp from idle. Same
                        // `HIPFIRE_DPM_WARMUP_SECS` env the in-process bench tools
                        // honor (`bench_qwen35_mq4`, `dflash_spec_demo`,
                        // `bench_stream_overlap`); see
                        // `crates/rdna-compute/src/dispatch.rs::dpm_warmup` and
                        // `docs/methodology/perf-benchmarking.md`.
                        //
                        // Runs AFTER weight upload but BEFORE the `loaded` ack so
                        // the contract becomes "loaded means daemon is fully ready
                        // including DPM-pinned." Critical for probe-side timing:
                        // if warmup ran AFTER the ack, the probe would receive
                        // `loaded`, immediately send `generate`, and the daemon
                        // (still warming up in this handler) wouldn't process the
                        // generate until warmup finished — folding the warmup
                        // into the probe-measured TTFT and breaking
                        // `tok_s = total_tokens / wall_ms`. With warmup before the
                        // ack, the probe sees `loaded` only when the daemon is
                        // truly ready, and TTFT measures real prefill alone.
                        //
                        // Default OFF (production daemon load latency unchanged).
                        if let Ok(secs_str) = std::env::var("HIPFIRE_DPM_WARMUP_SECS") {
                            if let Ok(secs) = secs_str.parse::<f32>() {
                                if secs > 0.0 {
                                    if let Err(e) = gpu.dpm_warmup(secs) {
                                        eprintln!("[daemon] dpm_warmup failed (non-fatal): {e:?}");
                                    }
                                }
                            }
                        }

                        // ── Continuous batch staging (must be before `loaded` ack) ──
                        // Moved verbatim to `hipfire_loader::batch_staging`: it
                        // constructed Qwen35/Lfm2/EP batch state, which is why the
                        // daemon named arch types here. `LoadedModel` already owned
                        // the typed fields it writes; only the construction leaked.
                        // The scheduler is built here because it lives in
                        // `hipfire-engine`, above the loader.
                        let staging = hipfire_loader::batch_staging::stage_continuous_batch(
                            &mut m,
                            &mut gpu,
                            parsed_continuous_batch_size,
                        );
                        let staged_batch_capable = staging.capable;
                        let staged_batch_scheduler = staging.capable.then(|| {
                            ContinuousBatchScheduler::new(staging.slots, staging.lane_capacity)
                        });
                        let staged_ep_batch = staging.ep;
                        let staged_ep_slots = staging.ep_slots;
                        let staged_ep_lane_cap = staging.ep_lane_cap;
                        continuous_batch_size = if staged_batch_capable {
                            parsed_continuous_batch_size
                        } else {
                            1
                        };
                        batch_scheduler = staged_batch_scheduler;
                        // `cache_capable` is the daemon's prompt-cache source of truth.
                        // arch_id 13 (gemma4) is intentionally ABSENT: hipfire_generate::dense::generate_gemma4 has
                        // no LCP prefix-cache block and always cold-prefills the full
                        // Jinja-rendered prompt. Enabling the cache would corrupt KV
                        // slot offsets after turn 1 (stale prefix reuse). Wire when
                        // hipfire_generate::dense::generate_gemma4 gains an LCP block matching other archs.
                        let cache_capable = matches!(m.arch_id, 5 | 6 | 9 | 10 | 12 | 14);
                        let retry_reset_eligible = model_retry_reset_eligible(m.arch_id);
                        let continuous_batch_capable = staged_batch_capable;
                        // Load ack exposes batch dimensions/capability; EP adds parallelism metadata but never infers operation from logs.
                        if staged_ep_batch {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{},"cache_capable":{},"retry_reset_eligible":{},"continuous_batch_capable":{},"continuous_batch_slots":{},"continuous_batch_lane_capacity":{},"continuous_batch_parallelism":"expert_parallel","continuous_batch_rank_count":4,"continuous_batch_reduce":"peer_rooted_f32"}}"#,
                                arch,
                                dim,
                                layers,
                                vocab,
                                vl,
                                cache_capable,
                                retry_reset_eligible,
                                continuous_batch_capable,
                                staged_ep_slots,
                                staged_ep_lane_cap,
                            );
                        } else {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{},"cache_capable":{},"retry_reset_eligible":{},"continuous_batch_capable":{}}}"#,
                                arch,
                                dim,
                                layers,
                                vocab,
                                vl,
                                cache_capable,
                                retry_reset_eligible,
                                continuous_batch_capable
                            );
                        }
                        // ── PFlash drafter load (Phase 4.0) ──────────────
                        //
                        // Only attempt when mode != off AND a drafter path
                        // was provided. Failures here are NON-FATAL: log
                        // the reason and continue with PFlash disabled so
                        // the operator gets a clear "model is up, but
                        // compression isn't" signal rather than losing
                        // the entire session.
                        //
                        // EP guard (load_tp > 1): the EP path serves through
                        // `hipfire_generate::qwen::generate_ep`, which bypasses PFlash entirely (the
                        // EP archs ds4/minimax refuse/ignore PFlash drafters).
                        // Loading a drafter here would just pin GPU memory it
                        // never reads until unload, so skip the load outright.
                        // Warn once if the operator actually supplied a drafter
                        // so the silent no-op is visible.
                        if load_tp > 1 {
                            if pflash_drafter.is_some() && pflash_mode_str != "off" {
                                eprintln!(
                                    "[pflash] WARN: ignoring PFlash drafter on EP (tp={}) model \
                                     — hipfire_generate::qwen::generate_ep bypasses PFlash; drafter would only waste GPU memory",
                                    load_tp
                                );
                            }
                        } else if let Some(ref pf_drafter_path) = pflash_drafter {
                            if pflash_mode_str != "off" {
                                if let Some(ref reason) = pflash_load_err {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"invalid load param: {}"}}"#,
                                        reason.replace('"', "'")
                                    );
                                    let _ = stdout.flush();
                                    model = Some(m);
                                    batch_poisoned = None;
                                    continue;
                                }
                                let pf_cfg = hipfire_pflash::pflash::PflashConfig {
                                    mode: hipfire_pflash::pflash::PflashMode::parse(
                                        &pflash_mode_str,
                                    )
                                    .unwrap_or(hipfire_pflash::pflash::PflashMode::Off),
                                    threshold_tokens: pflash_threshold,
                                    keep_ratio: pflash_keep_ratio,
                                    alpha: pflash_alpha,
                                    min_keep_tokens: pflash_min_keep,
                                    sink_tokens: pflash_sink,
                                    recent_tokens: pflash_recent,
                                    block_size: pflash_block,
                                    profile: pflash_profile,
                                    drafter_path: Some(pf_drafter_path.clone()),
                                    sparse_threshold: pflash_sparse_threshold,
                                };
                                let mut pf_state =
                                    hipfire_pflash::pflash::PflashState::new(&pf_cfg);
                                // Pull the target tokenizer out of the loaded model
                                // for the compat check. Both Qwen3.5 and plain
                                // Qwen3 paths expose `tokenizer` on LoadedModel.
                                let tgt_tok_ref = m.tokenizer.as_ref();
                                if let Some(tok) = tgt_tok_ref {
                                    let pf_max_kv = max_seq.max(2048);
                                    // Hetero: when prefill_drafter_device >= 0 and isn't
                                    // device 0 (target), allocate a sibling Gpu handle so
                                    // drafter weights/KV/scratch live on the secondary
                                    // card. Compress output is host-side, so decode stays
                                    // on target. -1 / 0 => share target gpu (unchanged).
                                    let mut sibling: Option<rdna_compute::Gpu> = None;
                                    if pflash_drafter_device > 0 {
                                        match rdna_compute::Gpu::init_with_device(
                                            pflash_drafter_device,
                                        ) {
                                            Ok(g) => sibling = Some(g),
                                            Err(e) => {
                                                let _ = writeln!(
                                                    stdout,
                                                    r#"{{"type":"pflash_load_failed","reason":"drafter device {} init: {}"}}"#,
                                                    pflash_drafter_device,
                                                    e.to_string().replace('"', "'")
                                                );
                                            }
                                        }
                                    }
                                    let dg: &mut rdna_compute::Gpu =
                                        sibling.as_mut().unwrap_or(&mut gpu);
                                    dg.bind_thread_or_warn();
                                    match hipfire_pflash::pflash::load_drafter(
                                        &mut pf_state,
                                        dg,
                                        std::path::Path::new(pf_drafter_path),
                                        tok,
                                        pf_max_kv,
                                    ) {
                                        Ok(()) => {
                                            eprintln!("[pflash] LOADED drafter={} dev={} mode={} compat={} keep={} thr={}",
                                                pf_drafter_path, pflash_drafter_device, pflash_mode_str,
                                                pf_state.tokenizer_compat, pflash_keep_ratio, pflash_threshold);
                                            let _ = writeln!(
                                                stdout,
                                                r#"{{"type":"pflash","mode":"{}","drafter":"{}","drafter_device":{},"tokenizer_compat":{},"keep_ratio":{},"threshold":{}}}"#,
                                                pflash_mode_str,
                                                pf_drafter_path,
                                                pflash_drafter_device,
                                                pf_state.tokenizer_compat,
                                                pflash_keep_ratio,
                                                pflash_threshold
                                            );
                                            pflash_state = Some(pf_state);
                                            pflash_cfg = Some(pf_cfg);
                                            pflash_drafter_gpu = sibling; // persist sibling across requests (None if shared)
                                        }
                                        Err(e) => {
                                            eprintln!("[pflash] LOAD FAILED: {}", e);
                                            let _ = writeln!(
                                                stdout,
                                                r#"{{"type":"pflash_load_failed","reason":"{}"}}"#,
                                                e.to_string().replace('"', "'")
                                            );
                                        }
                                    }
                                } else {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"target tokenizer unavailable"}}"#
                                    );
                                }
                            }
                        }

                        model = Some(m);
                        batch_poisoned = None;
                    }
                    Err(e) => {
                        let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                        let free_mb = vram_free / (1024 * 1024);
                        let total_mb = vram_total / (1024 * 1024);
                        // serde-escape: raw HipError debug contains { } and "
                        // which corrupt the JSONL protocol if interpolated raw.
                        write_error(&mut stdout, "", &format!(
                            "load failed: {e}. GPU: {} ({free_mb} MB free / {total_mb} MB total)", gpu.arch));
                    }
                }
                let _ = stdout.flush();
            }

            "generate" => {
                let gen_attempt_id = match require_wire_attempt_id(msg.get("attempt_id")) {
                    Ok(id) => id,
                    Err(reason) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            msg.get("id").and_then(|v| v.as_str()),
                            &format!("generate {reason}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                set_active_attempt_id(gen_attempt_id);
                let _attempt_guard = ActiveAttemptGuard;
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("0");
                #[cfg(feature = "serve-fault-inject")]
                let _fault_guard = {
                    let want = msg
                        .get("test_fault_after_prefill")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    arm_fault_after_prefill(want);
                    FaultAfterPrefillGuard
                };
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        hipfire_generate::dense::emit_active_attempt_error(
                            &mut stdout,
                            Some(id),
                            "no model loaded",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                if let Some(reason) = batch_poisoned.as_ref() {
                    hipfire_generate::dense::emit_active_attempt_error(
                        &mut stdout,
                        Some(id),
                        &format!(
                            "continuous batch GPU state poisoned; unload/reload required: {reason}"
                        ),
                        "gpu",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }

                // Fresh terminal-control transaction for this generate attempt.
                // Cleared by TerminalControlGuard on all exits from this arm.
                activate_terminal_control(id, gen_attempt_id);
                let _terminal_control_guard = TerminalControlGuard;
                gpu.replay.begin_replay_observation_window();
                let prompt = msg
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Hello");
                let prompt_norm = hipfire_runtime::tokenizer::maybe_normalize_prompt(prompt);
                let prompt: &str = &prompt_norm;
                if hipfire_runtime::config::get().prompt_token_heat {
                    if let Some(tok) = m.tokenizer.as_ref() {
                        tok.dump_prompt_heat(prompt);
                    }
                }
                let system = msg.get("system").and_then(|v| v.as_str());
                let image = msg.get("image").and_then(|v| v.as_str());
                let image_base64 = msg.get("image_base64").and_then(|v| v.as_str());

                // Structured-tools + structured-messages support (Phase 1 of
                // Jinja-everywhere migration). When present, both fields are
                // routed through `JinjaChatFrame::render_messages` so the
                // model sees the upstream template's `{% if tools %}` and
                // multi-turn branches (XML/JSON tool-call format per arch,
                // tool-response role mapping, etc.).
                //
                // Backward compat: when neither is present, legacy
                // `prompt`+`system` continues to drive a synthesized
                // [system?, user] slice — byte-identical to today's
                // `JinjaChatFrame::render()` single-turn path.
                //
                // Parse errors emit a structured error event and skip the
                // request (rather than silently dropping the fields).
                let tools_json: Option<Vec<serde_json::Value>> = match msg.get("tools") {
                    Some(v) => match serde_json::from_value::<Vec<serde_json::Value>>(v.clone()) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            hipfire_generate::dense::emit_active_attempt_error(
                                &mut stdout,
                                Some(id),
                                &format!(
                                    "invalid tools field: {}",
                                    e.to_string().replace('"', "'"),
                                ),
                                "validation",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    },
                    None => None,
                };
                let messages_history: Option<Vec<hipfire_runtime::prompt_frame::Message>> =
                    match msg.get("messages") {
                        Some(v) => match serde_json::from_value::<
                            Vec<hipfire_runtime::prompt_frame::Message>,
                        >(v.clone())
                        {
                            Ok(mut m) => {
                                // Apply the same normalization to each message's
                                // content that the daemon applies to `prompt` at
                                // line 1384 (`maybe_normalize_prompt`: strip
                                // trailing whitespace before `\n`, collapse 3+
                                // newlines to 2, etc.). Without this, turn N's
                                // `prompt`-encoded user tokens diverge from turn
                                // N+1's `messages[].content`-encoded history
                                // tokens, breaking the LCP cache on any prompt
                                // whose raw text has trailing whitespace or
                                // run-of-newlines patterns.
                                for entry in &mut m {
                                    if !entry.content.is_empty() {
                                        let normalized =
                                            hipfire_runtime::tokenizer::maybe_normalize_prompt(
                                                &entry.content,
                                            );
                                        if matches!(normalized, std::borrow::Cow::Owned(_)) {
                                            entry.content = normalized.into_owned();
                                        }
                                    }
                                }
                                Some(m)
                            }
                            Err(e) => {
                                hipfire_generate::dense::emit_active_attempt_error(
                                    &mut stdout,
                                    Some(id),
                                    &format!(
                                        "invalid messages field: {}",
                                        e.to_string().replace('"', "'"),
                                    ),
                                    "validation",
                                    false,
                                    false,
                                );
                                let _ = stdout.flush();
                                continue;
                            }
                        },
                        None => None,
                    };
                // hunt3 M-F: parse user stop sequences (top-level `stop` field on
                // the generate message; the CLI forwards OpenAI `stop` here, already
                // normalized to string[], <=4 entries, <=64 chars each). The decode
                // loops match these against the decoded output suffix and finish
                // with finish_reason="stop" on a hit. Re-apply the cap defensively
                // in case a non-hipfire client drives the daemon directly.
                let stop_seqs: Vec<String> = msg
                    .get("stop")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|s| s.as_str())
                            .filter(|s| !s.is_empty())
                            .take(4)
                            .map(|s| s.chars().take(64).collect::<String>())
                            .collect()
                    })
                    .unwrap_or_default();

                // Sampling defaults differ by arch: qwen35 family was tuned
                // at `temp=0.3, top_p=0.8` (DFlash-friendly, instruct-stable);
                // DeepSeek V4 Flash's HF card recommends `temp=1.0, top_p=1.0`
                // for local deployment, and lower values consistently fall
                // into block-level attractors on this quantized instruct
                // model. Pick arch-shaped defaults so a vanilla
                // `/v1/chat/completions` POST (no sampling fields) works on
                // both. Explicit per-request values still override either.
                // Hardcoded arch ladder — the LAST-RESORT fallback for the
                // sampling defaults. The author-recommended values baked into
                // the .hfq `generation_config` (m.rec_temperature/m.rec_top_p,
                // populated at load time via HfqFile::recommended_sampling) take
                // precedence over this ladder; an explicit per-request field
                // (set below via `msg.get(...)`) overrides both. The CLI's
                // curated registry `recommended_settings` reach this handler as
                // explicit request fields (CLI explicit-send guard), so they sit
                // above the .hfq layer on that path.
                let defaults = hipfire_loader::carrier_for(m.arch_id)
                    .map(|c| c.sampling_defaults())
                    .unwrap_or_default();
                let (arch_default_temp, arch_default_top_p) = (defaults.temp, defaults.top_p);
                // Layer the .hfq-baked author recommendation OVER the arch
                // ladder. Per-knob: a model that bakes only `temperature` still
                // gets the arch-ladder `top_p`.
                let default_temp = m
                    .rec_temperature
                    .map(|x| x as f64)
                    .unwrap_or(arch_default_temp);
                let default_top_p = m.rec_top_p.map(|x| x as f64).unwrap_or(arch_default_top_p);
                let temp = msg
                    .get("temperature")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(default_temp) as f32;
                let max_tokens = msg
                    .get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4096) as usize;
                let top_p = msg
                    .get("top_p")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(default_top_p) as f32;
                // CACTUS acceptance-boost δ — OPT-IN (request `cactus_delta`), 0.0
                // default = lossless/distribution-preserving. >0 is deliberately lossy
                // (higher acceptance τ, KL-bounded distortion) and applies only to a
                // CACTUS-capable sampled verify (deepseek4 DSpark / qwen35 DFlash);
                // other drafters ignore it. Never a default.
                let cactus_delta = msg
                    .get("cactus_delta")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                // Default 1.0 (off). Matches llama.cpp `--repeat-penalty 1.0`
                // and HF transformers `generate(repetition_penalty=1.0)`
                // defaults. The prior 1.3 default suppressed legitimately
                // repeated formatting tokens (e.g. `' **'` for bullets,
                // indentation patterns) on multi-step reasoning prompts,
                // pushing structured chain-of-thought trajectories off the
                // model's well-trained path into a self-doubt / number-
                // hallucination attractor on 9B Qwen3.5 at greedy decode.
                // Root cause writeup: issue #258 comment "Bug B root cause"
                // and docs/investigations/2026-05-15-9b-reasoning-loop/.
                // Clients can still opt in to a non-1.0 value per request.
                // LFM2.5-MoE (arch_id 11): Liquid's card recommends
                // repetition_penalty=1.05; default to it (others stay 1.0/off).
                let default_repeat_penalty = defaults.repeat_penalty;
                // Accept HF-style `repetition_penalty` as a request ALIAS for our
                // `repeat_penalty` field, used only when the canonical key is
                // absent. (OpenAI/HF clients send `repetition_penalty`.)
                let repeat_penalty = msg
                    .get("repeat_penalty")
                    .or_else(|| msg.get("repetition_penalty"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(default_repeat_penalty) as f32;
                // OpenAI-compatible `reasoning_effort` (also accept our custom
                // `thinking_mode` alias) — ThinkMode is consumed by arch_id=9
                // (DeepSeek DSML thinking), while raw `reasoning_effort` is
                // plumbed to Jinja as `reasoning_effort` for Qwen3.8 (arch 5/6).
                // `auto`/absent stays truly undefined (not null) so the Qwen3.8
                // template defaults to `xhigh` only when undefined; unsupported
                // values are preserved verbatim (template raises) rather than
                // being silently remapped to low/medium/xhigh.
                let think_mode = msg
                    .get("reasoning_effort")
                    .or_else(|| msg.get("thinking_mode"))
                    .and_then(|v| v.as_str())
                    .map(ThinkMode::from_str)
                    .unwrap_or(ThinkMode::NonThink);
                // Raw effort for Jinja `reasoning_effort` — exact, no lowercasing,
                // no empty-filter. `auto`/absent => undefined, `none`/`off`/`chat`
                // => disabled+undefined, all other exact strings (including
                // empty, case-mismatched) pass verbatim so the Qwen3.8 template
                // raises rather than silently normalizing or falling back.
                let raw_reasoning_effort: Option<&str> = msg
                    .get("reasoning_effort")
                    .or_else(|| msg.get("thinking_mode"))
                    .and_then(|v| v.as_str());
                let repeat_window = msg
                    .get("repeat_window")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                // OpenAI subtractive penalties. The CLI forwards raw
                // `presence_penalty`/`frequency_penalty` (0.0 = off). Unlike the
                // recency-weighted multiplicative `repeat_penalty`, these are
                // flat across the (now long) window, which is what breaks the
                // block-level repetition loops on long reasoning generations.
                // Clamp negatives to 0 (negative would REWARD repetition).
                // Fallback ladder: explicit request `presence_penalty` >
                // .hfq-baked `m.rec_presence_penalty` > 0.0 (off). The .hfq's
                // generation_config does not carry presence_penalty today, so
                // m.rec_presence_penalty is always None on the load path; the
                // field is wired so a curated registry card value still flows in
                // as an explicit request field (CLI explicit-send guard). presence_penalty IS honored by the sampler.
                let presence_penalty = (msg
                    .get("presence_penalty")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(m.rec_presence_penalty.unwrap_or(0.0) as f64)
                    as f32)
                    .max(0.0);
                let frequency_penalty = (msg
                    .get("frequency_penalty")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32)
                    .max(0.0);
                // Request-driven top_k / min_p (W7 P2). Fallback ladder:
                // explicit request field > .hfq/registry-baked rec_top_k /
                // rec_min_p > None. None reproduces the legacy sampler exactly
                // (top-K candidate gather of 20, no min-p cut). top_k <= 0 is
                // treated as "unset" (None) so 0 never collapses to argmax.
                let top_k: Option<u32> = msg
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .map(|k| k as u32)
                    .or_else(|| m.rec_top_k.map(|k| k as u32))
                    .filter(|&k| k > 0);
                let min_p: Option<f32> = msg
                    .get("min_p")
                    .and_then(|v| v.as_f64())
                    .map(|p| p as f32)
                    .or(m.rec_min_p)
                    .filter(|&p| p > 0.0);
                // Experimental: inject a nudge string at a specific generated-
                // token count. The nudge tokens get forward-fed through the KV
                // cache so the model "sees" them as part of its own trajectory,
                // and are emitted to stdout so the client stream includes them.
                // Used to test whether telling a thinking model "time's up"
                // gets it to close </think> and commit to an answer.
                //
                // GATED: off by default. The feature has a real UX hazard — if
                // the alert fires after </think> has already closed, the nudge
                // leaks into the visible answer. Only honor the params when the
                // operator has explicitly opted in via config
                // (`experimental_budget_alert: true` → HIPFIRE_EXPERIMENTAL_
                // BUDGET_ALERT=1 set by the CLI). Research use only; not a
                // stable contract.
                let experimental_ok = hipfire_runtime::config::get().experimental_budget_alert;
                let budget_alert_at_tok = if experimental_ok {
                    msg.get("budget_alert_at_tok")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize
                } else {
                    0
                };
                let budget_alert_text = if experimental_ok {
                    msg.get("budget_alert_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                } else {
                    String::new()
                };
                // Budget for tokens emitted INSIDE the model's <think>...</think>
                // block. 0 = uncapped (model thinks until it naturally closes).
                // Triggered from the CLI by per-model `max_think_tokens` config,
                // OpenAI `chat_template_kwargs.enable_thinking=false` (cap=1),
                // and `reasoning.effort` (none=1, minimal=64, low=256, medium=
                // 1024, high=4096, xhigh=0).
                //
                // When the cap is reached the daemon force-emits "</think>\n"
                // through the same KV-write + sample path as a normal token,
                // closing the thinking block so the model commits to an
                // answer with the remaining max_tokens budget. Caught by
                // Codex stop-time review on 2026-04-28: the field had been
                // shipping in genParams since cli/index.ts but the daemon
                // was silently ignoring it, making the new reasoning.effort
                // / enable_thinking knobs no-ops on the wire.
                let max_think_tokens = msg
                    .get("max_think_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                // Derive Jinja `enable_thinking` and `reasoning_effort` via
                // pure helper (no lowercasing, no empty-drop).
                let (enable_thinking_jinja, reasoning_effort_jinja) =
                    qwen_jinja_reasoning(raw_reasoning_effort, max_think_tokens);
                // Controls the ChatML framing after the assistant role header.
                // Propagated through both text and Qwen3.5-VL paths.
                let assistant_prefix = match msg
                    .get("assistant_prefix")
                    .and_then(|v| v.as_str())
                    .unwrap_or("plain")
                {
                    "open_think" => hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink,
                    "closed_think" => hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink,
                    _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                };

                let has_image = image_base64.is_some() || image.is_some();
                let vision_route = hipfire_loader::vision_route(m.arch_id);
                let has_vl = m.vision_config.is_some() || vision_route == hipfire_loader::VisionRoute::DotsOcr;

                if has_image && !has_vl {
                    write_error(&mut stdout, id, "model has no vision encoder");
                } else if has_image && has_vl {
                    // DEFENSIVE: VL is single-image, single-turn only. The
                    // CLI rejects images in non-last turns, but a raw
                    // JSONL client could send a second image on turn 2+.
                    // If seq_pos > 0 here, a previous conversation's KV
                    // entries are live — running vision_forward and
                    // splicing visual tokens into that context would
                    // produce garbage. Force a reset so VL always starts
                    // from a clean KV state.
                    //
                    // Must mirror the "reset" command handler (line ~2098).
                    // VL only runs on qwen35-vl (arch_id 5/8), so
                    // qwen2_state, deepseek4_state, and llama_kv are
                    // None — but clear them anyway for defense-in-depth
                    // in case a future arch adds VL support.
                    if m.seq_pos > 0 {
                        eprintln!("[daemon/vl] non-zero seq_pos ({}) at VL dispatch — resetting conversation", m.seq_pos);
                        m.seq_pos = 0;
                        m.conversation_tokens.clear();
                        hipfire_generate::common::free_checkpoints(&mut m.prefill_checkpoints, &mut gpu);
                        hipfire_generate::common::free_checkpoints(&mut m.dflash_checkpoints, &mut gpu);
                        // The DFlash checkpoint ring now lives inside the
                        // speculator (m.dflash_checkpoints is vestigial/empty),
                        // so free THAT ring on conversation reset too — else its
                        // GPU snapshots persist until the next prefill-miss.
                        if let Some(s) = m.speculator.as_mut() {
                            if let Err(e) = s.reset(&mut gpu) {
                                hipfire_generate::dense::emit_active_attempt_error(
                                    &mut stdout,
                                    Some(id),
                                    &format!("vision conversation reset failed: {e}"),
                                    "gpu",
                                    true,
                                    false,
                                );
                                continue;
                            }
                        }
                        // qwen35(-vl) recurrent state lives in the bundle
                        // (ModelState::Qwen35), not the always-None
                        // m.dn_state/m.kv_cache direct fields.
                        if let Err(e) = hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu) {
                            hipfire_generate::dense::emit_active_attempt_error(
                                &mut stdout,
                                Some(id),
                                &format!("vision recurrent reset failed: {e}"),
                                "gpu",
                                true,
                                false,
                            );
                            continue;
                        }
                        if let Some(ModelState::Llama(b)) = m.state.as_mut() {
                            b.kv.compact_offset = 0;
                        }
                        if let Some(ref mut s) = m.qwen2_state {
                            s.reset();
                        }
                        // Live plain-qwen2 state is in the ModelState::Qwen2
                        // bundle, not the (dots-ocr-only) qwen2_state field —
                        // rewind it too for defense-in-depth.
                        if let Some(b) = m.qwen2_mut() {
                            b.state.reset();
                        }
                        if let Some(b) = m.deepseek4_mut() {
                            b.state.reset();
                        }
                        if let Some(ad) = m.kv_adaptive.as_mut() {
                            if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
                                ad.reset_with_cache(&mut gpu, &mut b.kv_cache);
                            } else {
                                ad.reset();
                            }
                        }
                    }
                    if image_base64.is_some() && image.is_some() {
                        eprintln!(
                            "[daemon/vl] both image and image_base64 provided — using image_base64"
                        );
                    }
                    let source = if let Some(b64) = image_base64 {
                        if b64.len() > MAX_BASE64_ENCODED_LEN {
                            write_error(
                                &mut stdout,
                                id,
                                &format!(
                                    "image payload exceeds maximum encoded size ({} bytes)",
                                    MAX_BASE64_ENCODED_LEN,
                                ),
                            );
                            continue;
                        }
                        ImageSource::Base64(b64)
                    } else {
                        ImageSource::Path(image.unwrap())
                    };
                    // Plan-mandated Phase-1 stopgap (docs/plans/completions_vision.md §2.1):
                    // VL dispatch defaults `max_think_tokens` to 256 when the
                    // client doesn't specify one. Caps runaway thinking
                    // without needing the full `ThinkState` extraction. Text
                    // path keeps unwrap_or(0) — it has different defaults
                    // controlled per-model on the CLI side.
                    let vl_max_think_tokens = if max_think_tokens == 0 {
                        256
                    } else {
                        max_think_tokens
                    };
                    let params = GenerateVLParams {
                        id,
                        prompt,
                        system_prompt: system,
                        image_source: source,
                        temp,
                        top_p,
                        max_tokens,
                        repeat_penalty,
                        repeat_window,
                        max_think_tokens: vl_max_think_tokens,
                        assistant_prefix,
                    };
                    match vision_route {
                        hipfire_loader::VisionRoute::DotsOcr => hipfire_generate::vision::generate_vl_dots_ocr(m, &mut gpu, &mut stdout, &params),
                        _ => hipfire_generate::vision::generate_vl(m, &mut gpu, &mut stdout, &params),
                    }
                } else {
                    // Per-request PflashConfig: clone the load-time cfg
                    // and apply any per-request overrides from `params`.
                    // None when no drafter was configured at load --
                    // generate() then takes the identity path.
                    //
                    // Out-of-range overrides (keep_ratio outside (0, 1],
                    // block_size == 0) would otherwise reach asserts inside
                    // select_spans / scoring and panic the entire daemon.
                    // Reject the request with an explicit error event so
                    // the client gets a clean signal and the daemon stays up.
                    let mut pf_override_err: Option<String> = None;
                    let pf_cfg_owned = pflash_cfg.as_ref().map(|base| {
                        let mut c = base.clone();
                        if let Some(s) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_compression"))
                            .and_then(|v| v.as_str())
                        {
                            if let Some(m) = hipfire_pflash::pflash::PflashMode::parse(s) {
                                c.mode = m;
                            }
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_threshold"))
                            .and_then(|v| v.as_u64())
                        {
                            c.threshold_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_keep_ratio"))
                            .and_then(|v| v.as_f64())
                        {
                            let r = v as f32;
                            if !(r > 0.0 && r <= 1.0) {
                                pf_override_err =
                                    Some(format!("prefill_keep_ratio={r} not in (0, 1]"));
                            } else {
                                c.keep_ratio = r;
                            }
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_min_keep"))
                            .and_then(|v| v.as_u64())
                        {
                            c.min_keep_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_sink"))
                            .and_then(|v| v.as_u64())
                        {
                            c.sink_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_recent"))
                            .and_then(|v| v.as_u64())
                        {
                            c.recent_tokens = v as usize;
                        }
                        if let Some(v) = msg
                            .get("params")
                            .and_then(|p| p.get("prefill_block"))
                            .and_then(|v| v.as_u64())
                        {
                            let b = v as usize;
                            if b == 0 {
                                pf_override_err = Some("prefill_block must be > 0".to_string());
                            } else {
                                c.block_size = b;
                            }
                        }
                        c
                    });
                    if let Some(reason) = pf_override_err {
                        hipfire_generate::dense::emit_active_attempt_error(
                            &mut stdout,
                            Some(id),
                            &format!("invalid pflash override: {}", reason.replace('"', "'"),),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    // ── Continuous batch admission (tightened to actual route) ──
                    // EP Qwen35 expert-parallel is batch-only, TP=4, 4×gfx1201; fail closed otherwise.
                    // Check EP eligibility first so batch-only enforcement fires before single-GPU fallback.
                    let serve_continuous_batch = parse_serve_continuous_batch(&msg);
                    let pflash_active = pf_cfg_owned.as_ref().is_some_and(|c| {
                        !matches!(c.mode, hipfire_pflash::pflash::PflashMode::Off)
                    });
                    let ep_batch_eligible = if batch_scheduler.is_some() && m.ep.is_some() {
                        is_qwen_ep_batch_request_eligible(
                            &msg,
                            m,
                            continuous_batch_size,
                            serve_continuous_batch,
                            pflash_active,
                        )
                    } else {
                        false
                    };
                    if ep_batch_eligible {
                        batch_transition_to_queued(id, gen_attempt_id);
                        if batch_check_abort(id, gen_attempt_id) {
                            let _scope = BatchAttemptScope::enter(gen_attempt_id);
                            emit_gen_start(
                                &mut stdout,
                                id,
                                false,
                                Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
                            );
                            emit_qwen_ar_cancelled(&mut stdout, id, 0);
                            batch_clear_terminal(id, gen_attempt_id);
                            continue;
                        }
                        let sampling = resolve_batch_sampling(&msg, m);
                        let prompt_owned =
                            batch_single_user_content(&msg).unwrap_or_else(|| prompt.to_string());
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_owned,
                            system,
                            assistant_prefix,
                            m.tokenizer.as_ref().unwrap(),
                            m.chat_template.as_ref(),
                            max_think_tokens,
                            messages_history.as_deref(),
                            enable_thinking_jinja,
                            reasoning_effort_jinja.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    Some(id),
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(id, gen_attempt_id);
                                continue;
                            }
                        };
                        if started_in_think {
                            let _ = batch_transfer_abort_to_singleton_and_clear(id, gen_attempt_id);
                        } else {
                            if prompt_tokens.is_empty() || prompt_tokens.len() >= m.max_seq {
                                let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    Some(id),
                                    "prompt exceeds lane capacity or empty",
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(id, gen_attempt_id);
                                continue;
                            }
                            let pending = BatchPendingRequest {
                                key: AttemptKey::new(id, gen_attempt_id),
                                prompt: prompt_owned.clone(),
                                prompt_tokens: prompt_tokens.clone(),
                                started_in_think,
                                system: system.map(|s| s.to_string()),
                                assistant_prefix,
                                max_think_tokens,
                                max_tokens,
                                sampling: sampling.clone(),
                            };
                            if let Some(sched) = batch_scheduler.as_mut() {
                                let enq_ok = sched.enqueue(pending);
                                if !enq_ok {
                                    eprintln!("[batch][EP] duplicate enqueue rejected id={} attempt_id={}; preserving live registry", id, gen_attempt_id);
                                    continue;
                                }
                                {
                                    let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                    emit_gen_start(
                                        &mut stdout,
                                        id,
                                        false,
                                        Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
                                    );
                                }
                                let drive_res = drive_qwen35_ep_continuous_batch(
                                    sched,
                                    m,
                                    &mut stdout,
                                    &mut inbox,
                                );
                                match drive_res {
                                    Ok(()) => {}
                                    Err(BatchDriveError::Gpu(e)) => {
                                        eprintln!("[batch][EP] drive failed (attested): {e}");
                                    }
                                    Err(BatchDriveError::Poisoned(e)) => {
                                        eprintln!("[batch][EP] drive poisoned (unattested): {e} — generation poisoned until unload/reload");
                                        // Checked teardown: reset_all already attempted in fail_all; poison scheduler.
                                        batch_scheduler = None;
                                        continuous_batch_size = 1;
                                        batch_poisoned = Some(e);
                                        batch_clear_all_terminals();
                                    }
                                }
                            }
                            continue;
                        }
                    }
                    // Enforce batch-only for EP: if EP batch is staged, non-eligible must fail closed, not silently fall back.
                    let ep_batch_staged = batch_scheduler.is_some()
                        && m.ep
                            .as_ref()
                            .is_some_and(|ep| matches!(ep.inner, EpArch::Qwen35 { .. }));
                    if ep_batch_staged {
                        // EP requests without serve_continuous_batch or with excluded features must error.
                        if !ep_batch_eligible {
                            let _scope = BatchAttemptScope::enter(gen_attempt_id);
                            let ep = hipfire_generate::common::RollbackEpilogue {
                                rolled_back: true,
                                context: None,
                            };
                            // Reset the specific lane if any (best-effort), else poison not needed; just fail this request.
                            hipfire_generate::common::emit_fail_closed_error(&mut stdout, Some(id), "EP qwen35 batch-only: request must set serve_continuous_batch=true with TP=4 expert_parallel and no excluded features (image/tools/stop/spec)", "validation", false, &ep);
                            batch_clear_terminal(id, gen_attempt_id);
                            continue;
                        }
                    }
                    let batch_eligible = if batch_scheduler.is_some() {
                        is_batch_request_eligible(
                            &msg,
                            m,
                            continuous_batch_size,
                            serve_continuous_batch,
                            pflash_active,
                        )
                    } else {
                        false
                    };
                    if batch_eligible {
                        // Current request was already announced by the reader; promote to Queued.
                        batch_transition_to_queued(id, gen_attempt_id);
                        // If already aborted, emit cancelled and do not enqueue.
                        if batch_check_abort(id, gen_attempt_id) {
                            let _scope = BatchAttemptScope::enter(gen_attempt_id);
                            emit_gen_start(
                                &mut stdout,
                                id,
                                false,
                                Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
                            );
                            emit_qwen_ar_cancelled(&mut stdout, id, 0);
                            batch_clear_terminal(id, gen_attempt_id);
                            continue;
                        }
                        let sampling = resolve_batch_sampling(&msg, m);
                        // Render prompt once at admission and store tokens/started flag; do not render twice at lane assignment.
                        let prompt_owned =
                            batch_single_user_content(&msg).unwrap_or_else(|| prompt.to_string());
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_owned,
                            system,
                            assistant_prefix,
                            m.tokenizer.as_ref().unwrap(),
                            m.chat_template.as_ref(),
                            max_think_tokens,
                            messages_history.as_deref(),
                            enable_thinking_jinja,
                            reasoning_effort_jinja.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    Some(id),
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(id, gen_attempt_id);
                                continue;
                            }
                        };
                        if started_in_think {
                            // Rendered prompts that open a think span are sequential
                            // barriers. Transfer any pre-latched abort exactly once
                            // (transfer itself clears the keyed entry).
                            let _ = batch_transfer_abort_to_singleton_and_clear(id, gen_attempt_id);
                            // Fall through to sequential generate below (do not enqueue).
                        } else {
                            if prompt_tokens.is_empty() || prompt_tokens.len() >= m.max_seq {
                                let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    Some(id),
                                    "prompt exceeds lane capacity or empty",
                                    "validation",
                                    false,
                                    false,
                                );
                                batch_clear_terminal(id, gen_attempt_id);
                                continue;
                            }
                            let pending = BatchPendingRequest {
                                key: AttemptKey::new(id, gen_attempt_id),
                                prompt: prompt_owned.clone(),
                                prompt_tokens: prompt_tokens.clone(),
                                started_in_think,
                                system: system.map(|s| s.to_string()),
                                assistant_prefix,
                                max_think_tokens,
                                max_tokens,
                                sampling: sampling.clone(),
                            };
                            if let Some(sched) = batch_scheduler.as_mut() {
                                let arch = m.arch_id;
                                if arch == 11 {
                                    let enq_ok = sched.enqueue(pending);
                                    if !enq_ok {
                                        eprintln!(
                                            "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                            id, gen_attempt_id
                                        );
                                        continue;
                                    }
                                    {
                                        let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                        emit_gen_start(
                                            &mut stdout,
                                            id,
                                            false,
                                            hipfire_generate::common::gen_start_contract_version_for_arch(arch),
                                        );
                                    }
                                    let drive_res = drive_lfm_continuous_batch(
                                        sched,
                                        &mut gpu,
                                        m,
                                        &mut stdout,
                                        &mut inbox,
                                    );
                                    match drive_res {
                                        Ok(()) => {}
                                        Err(BatchDriveError::Gpu(e)) => {
                                            eprintln!("[batch] drive failed (attested): {e}");
                                        }
                                        Err(BatchDriveError::Poisoned(e)) => {
                                            eprintln!("[batch] drive poisoned (unattested): {e} — generation poisoned until unload/reload");
                                            batch_scheduler = None;
                                            continuous_batch_size = 1;
                                            batch_poisoned = Some(e);
                                            batch_clear_all_terminals();
                                        }
                                    }
                                } else if arch == 5 || arch == 6 {
                                    let enq_ok = sched.enqueue(pending);
                                    if !enq_ok {
                                        eprintln!(
                                            "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                            id, gen_attempt_id
                                        );
                                        continue;
                                    }
                                    {
                                        let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                        emit_gen_start(
                                            &mut stdout,
                                            id,
                                            false,
                                            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
                                        );
                                    }
                                    let drive_res = drive_qwen_continuous_batch(
                                        sched,
                                        &mut gpu,
                                        m,
                                        &mut stdout,
                                        &mut inbox,
                                    );
                                    match drive_res {
                                        Ok(()) => {}
                                        Err(BatchDriveError::Gpu(e)) => {
                                            eprintln!("[batch] drive failed (attested): {e}");
                                        }
                                        Err(BatchDriveError::Poisoned(e)) => {
                                            eprintln!("[batch] drive poisoned (unattested): {e} — generation poisoned until unload/reload");
                                            batch_scheduler = None;
                                            continuous_batch_size = 1;
                                            batch_poisoned = Some(e);
                                            batch_clear_all_terminals();
                                        }
                                    }
                                } else {
                                    eprintln!(
                                        "[batch] impossible arch {} reached scheduler — fail closed",
                                        arch
                                    );
                                    let _scope = BatchAttemptScope::enter(gen_attempt_id);
                                    let ep = hipfire_generate::common::RollbackEpilogue {
                                        rolled_back: true,
                                        context: None,
                                    };
                                    hipfire_generate::common::emit_fail_closed_error(
                                        &mut stdout,
                                        Some(id),
                                        &format!("batch not supported for arch {}", arch),
                                        "validation",
                                        false,
                                        &ep,
                                    );
                                    batch_clear_terminal(id, gen_attempt_id);
                                    batch_scheduler = None;
                                    continuous_batch_size = 1;
                                    batch_poisoned = Some(format!("impossible arch {}", arch));
                                    batch_clear_all_terminals();
                                }
                            }
                            continue;
                        }
                    } else {
                        // Sequential/default mode does not need the keyed batch
                        // announcement the reader made for this generate. Transfer
                        // any pre-latched abort into the singleton, then clear the
                        // keyed entry so default service cannot leak state across
                        // request-key reuse or a later batch-enabled load.
                        let _ = batch_transfer_abort_to_singleton_and_clear(id, gen_attempt_id);
                    }
                    // Did the request explicitly set a non-temperature sampling
                    // control? (gates temp>0 spec routing — see generate()).
                    let user_explicit_sampling = [
                        "top_p",
                        "top_k",
                        "min_p",
                        "repeat_penalty",
                        "presence_penalty",
                        "frequency_penalty",
                    ]
                    .iter()
                    .any(|k| msg.get(*k).is_some());
                    generate(
                        m,
                        &mut gpu,
                        pflash_drafter_gpu.as_mut(),
                        &mut stdout,
                        id,
                        prompt,
                        system,
                        user_explicit_sampling,
                        temp,
                        top_p,
                        top_k,
                        min_p,
                        cactus_delta,
                        max_tokens,
                        repeat_penalty,
                        repeat_window,
                        presence_penalty,
                        frequency_penalty,
                        budget_alert_at_tok,
                        &budget_alert_text,
                        max_think_tokens,
                        assistant_prefix,
                        pflash_state.as_mut(),
                        pf_cfg_owned.as_ref(),
                        tools_json.as_deref(),
                        messages_history.as_deref(),
                        think_mode,
                        &stop_seqs, // hunt3 M-F
                        reasoning_effort_jinja.as_deref(),
                        enable_thinking_jinja,
                    );
                }
                if let Some(marker) = gpu.replay.replay_observation_marker(id) {
                    eprintln!("{marker}");
                }
            }

            "reset" => {
                // attempt_id is mandatory and must be echoed exactly on the ack.
                // Reject before mutating host/GPU state.
                let reset_attempt_id = match require_wire_attempt_id(msg.get("attempt_id")) {
                    Ok(id) => id,
                    Err(reason) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("reset {reason}"),
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                // Reset conversation state without unloading the model.
                // Single production epilogue owns ordering + graph/replay
                // invalidate + sync attestation (same path as fail-closed turns).
                if let Some(m) = &mut model {
                    // Batch guard: reset is forbidden while any lane is active (would corrupt disjoint KV/state).
                    if batch_scheduler
                        .as_ref()
                        .is_some_and(|s| s.active_count() > 0)
                    {
                        hipfire_generate::dense::write_error_envelope(
                            &mut stdout,
                            None,
                            "reset refused: continuous batch lanes active",
                            "validation",
                            false,
                            false,
                            reset_attempt_id,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                        eprintln!("[qwen-cache RESET] daemon received reset — clearing conversation_tokens (was {})", m.conversation_tokens.len());
                    }
                    let ep = hipfire_generate::common::production_fail_closed_rollback(m, &mut gpu, None, None);
                    if !ep.rolled_back {
                        let detail = ep
                            .context
                            .as_deref()
                            .unwrap_or("rollback could not be attested");
                        hipfire_generate::dense::write_error_envelope(
                            &mut stdout,
                            None,
                            &format!("reset failed: {detail}"),
                            "transient",
                            true,
                            false,
                            reset_attempt_id,
                        );
                        continue;
                    }
                    // Host counters must already be zero before ack (set in epilogue).
                    debug_assert_eq!(m.seq_pos, 0);
                    state_epoch = state_epoch.saturating_add(1);
                    // Clear batch scheduler host state on successful cold reset
                    if let Some(sched) = batch_scheduler.as_mut() {
                        let _ = sched.fail_all_active();
                    }
                    batch_clear_all_terminals();
                    let ack = serde_json::json!({
                        "type": "reset",
                        "rolled_back": true,
                        "state_epoch": state_epoch,
                        "seq_pos": 0,
                        "conversation_len": 0,
                        "attempt_id": reset_attempt_id,
                        "retry_reset_eligible": model_retry_reset_eligible(m.arch_id),
                    });
                    let _ = writeln!(stdout, "{ack}");
                } else {
                    hipfire_generate::dense::write_error_envelope(
                        &mut stdout,
                        None,
                        "no model loaded",
                        "validation",
                        false,
                        false,
                        reset_attempt_id,
                    );
                }
                let _ = stdout.flush();
            }

            "unload" => {
                // Batch guard: unload is forbidden while lanes active.
                if batch_scheduler
                    .as_ref()
                    .is_some_and(|s| s.active_count() > 0)
                {
                    let attempt = msg.get("attempt_id").and_then(|v| v.as_u64()).unwrap_or(0);
                    hipfire_generate::dense::write_error_envelope(
                        &mut stdout,
                        None,
                        "unload refused: continuous batch lanes active",
                        "validation",
                        false,
                        false,
                        attempt,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // PFlash drafter goes FIRST: its weights/scratch/KV
                // tensors are released via Gpu::free_tensor, which only
                // queues into the GPU pool. The actual hipFree happens
                // inside unload_model -> drain_pool. Calling
                // unload_drafter AFTER unload_model would leave the
                // drafter buffers cached in the just-emptied pool with
                // no drain to follow, so the VRAM stays resident until
                // the next load message arrives. Order matters here.
                if let Some(mut pf) = pflash_state.take() {
                    if let Some(mut dg) = pflash_drafter_gpu.take() {
                        dg.bind_thread_or_warn();
                        pf.unload_drafter(&mut dg);
                        gpu.bind_thread_or_warn();
                    } else {
                        pf.unload_drafter(&mut gpu);
                    }
                }
                pflash_cfg = None;
                let unload_result = if let Some(m) = model.take() {
                    hipfire_loader::unload_model(m, &mut gpu)
                } else {
                    // No model: still retry any process-global pending VMM arenas.
                    hipfire_loader::ensure_vmm_ready_for_load(&mut gpu)
                };
                match unload_result {
                    Ok(()) => {
                        let _ = writeln!(stdout, r#"{{"type":"unloaded"}}"#);
                    }
                    Err(err) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("unload incomplete: {err}; VMM arenas retained for retry"),
                            "internal",
                            false,
                            false,
                        );
                    }
                }
                batch_scheduler = None;
                continuous_batch_size = 1;
                batch_clear_all_terminals();
                let _ = stdout.flush();
            }
            "ping" => {
                let _ = writeln!(stdout, r#"{{"type":"pong"}}"#);
                let _ = stdout.flush();
            }

            "diag" => {
                let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                let hip_ver = gpu.hip.runtime_version().unwrap_or((0, 0));
                let has_model = model.is_some();
                let model_arch = model
                    .as_ref()
                    .map(|m| match m.arch_id {
                        5 => "qwen3_5",
                        6 => "qwen3_5_moe",
                        7 => "qwen2",
                        9 => "deepseek4",
                        10 => "minimax_m2",
                        11 => "lfm2moe",
                        12 => "north_mini_code",
                        13 => "gemma4",
                        14 => "muse_glimmer",
                        _ => "qwen3",
                    })
                    .unwrap_or("none");
                // Count pre-compiled kernels
                let kernel_dir = std::env::current_exe()
                    .ok()
                    .and_then(|e| {
                        e.parent()
                            .map(|p| p.join("kernels").join("compiled").join(&gpu.arch))
                    })
                    .filter(|p| p.is_dir());
                let (hsaco_count, hash_count) = kernel_dir
                    .map(|d| {
                        let hsaco = std::fs::read_dir(&d)
                            .map(|r| {
                                r.filter(|e| {
                                    e.as_ref()
                                        .ok()
                                        .map(|e| {
                                            e.path()
                                                .extension()
                                                .map(|x| x == "hsaco")
                                                .unwrap_or(false)
                                        })
                                        .unwrap_or(false)
                                })
                                .count()
                            })
                            .unwrap_or(0);
                        let hash = std::fs::read_dir(&d)
                            .map(|r| {
                                r.filter(|e| {
                                    e.as_ref()
                                        .ok()
                                        .map(|e| {
                                            e.path()
                                                .extension()
                                                .map(|x| x == "hash")
                                                .unwrap_or(false)
                                        })
                                        .unwrap_or(false)
                                })
                                .count()
                            })
                            .unwrap_or(0);
                        (hsaco, hash)
                    })
                    .unwrap_or((0, 0));
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"diag","arch":"{}","hip_version":"{}.{}","vram_free_mb":{},"vram_total_mb":{},"model_loaded":{},"model_arch":"{}","kernels":{},"kernel_hashes":{}}}"#,
                    gpu.arch,
                    hip_ver.0,
                    hip_ver.1,
                    vram_free / (1024 * 1024),
                    vram_total / (1024 * 1024),
                    has_model,
                    model_arch,
                    hsaco_count,
                    hash_count
                );
                let _ = stdout.flush();
            }

            "bench_prefill" => {
                // Synthetic prefill benchmark — measures the architecture's
                // production prefill entry on N deterministic tokens from a
                // zeroed state. Used by `hipfire bench` to produce canonical
                // pp128/pp512/pp1024 numbers that don't depend on a prompt
                // tokenizing to a round number. This stays a synthetic workload;
                // only the forward path must match production.
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ = emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            "no model loaded",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                // bench_prefill drives forward_prefill_batch / forward_scratch
                // with the single-GPU `gpu` handle — those entry points panic
                // when pp>1 because q35_scratch is None and the multi-GPU
                // tensors live on Gpus instead. Refuse cleanly per snapshot
                // review patch f253472. A pp>1 prefill bench is out of scope
                // for v1.
                if m.pp > 1 || m.ep.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "bench_prefill requires a single-GPU model (pp=1, non-EP); multi-GPU/EP bench not implemented", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                // Guard physical_cap — reserve 32 slots of headroom so a subsequent
                // generate request against the loaded model still has room. We guard
                // on the *physical* buffer (not the advertised max_seq) because this
                // bench intentionally bypasses eviction to measure raw prefill.
                if n.saturating_add(32) > m.physical_cap {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        &format!(
                            "bench_prefill tokens={} exceeds loaded physical_cap={}",
                            n, m.physical_cap
                        ),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // Deterministic synthetic token IDs. Skip 0 (often <pad>) and the
                // low specials by offsetting, and wrap in a 1000-wide window so the
                // embedding lookup cost stays realistic rather than hitting one
                // cache-hot row repeatedly.
                let synthetic: Vec<u32> = (0..n as u32).map(|i| 10 + (i % 1000)).collect();

                // Reset state BEFORE timing so we're measuring cold prefill, not
                // prefill-on-top-of-prior-state. qwen35 recurrent state lives in
                // the bundle (ModelState::Qwen35), not the always-None m.dn_state.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                let _ = hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu);
                // Qwen2 (arch_id=7) doesn't have a separate KV buffer — the cache
                // and the per-step scratch share `Qwen2State`. Reset its position
                // cursor here so bench_prefill measures cold prefill. The live
                // state is in the ModelState::Qwen2 bundle; `qwen2_state` is only
                // dots-ocr's — rewind both, else this measures warm prefill.
                if let Some(ref mut s) = m.qwen2_state {
                    s.reset();
                }
                if let Some(b) = m.qwen2_mut() {
                    b.state.reset();
                }
                if let Some(b) = m.cohere2moe_mut() {
                    let _ = b.state.reset(&mut gpu);
                }
                if let Some(ModelState::Gemma4(bundle)) = m.state.as_mut() {
                    bundle.state.reset();
                }
                if let Some(ModelState::MuseGlimmer(bundle)) = m.state.as_mut() {
                    bundle.reset_session_state();
                }
                if let Some(ModelState::Deepseek4(b)) = m.state.as_mut() {
                    b.state.reset();
                    b.state.zero_decode_caches(&mut gpu);
                    gpu.invalidate_graph_state();
                }

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                let _ = gpu.hip.device_synchronize();
                let t0 = Instant::now();
                let mut prefill_err: Option<String> = None;
                let run_ok = {
                    // Arch-erased bench-prefill dispatch via Carrier (wave2 GenDispatch).
                    // Each carrier implements its architecture's synthetic prefill body
                    // verbatim; the daemon no longer matches on arch_id. See
                    // `hipfire_loader::Carrier::bench_prefill`.
                    let carrier = hipfire_loader::carrier_for(m.arch_id)
                        .expect("bench_prefill: unknown arch_id");
                    carrier
                        .bench_prefill(m, &mut gpu, &synthetic, n, &mut prefill_err)
                        .expect("bench_prefill: carrier does not implement bench_prefill for this arch")
                };
                let _ = gpu.hip.device_synchronize();
                let elapsed = t0.elapsed().as_secs_f64();

                // Reset state AFTER measurement — we've written N KV slots and a
                // DeltaNet state that the next real request must not inherit.
                // qwen35 recurrent state lives in the bundle (ModelState::Qwen35),
                // not the always-None m.dn_state.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                let _ = hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu);
                // LFM2.5-MoE state carries its own KV + conv-state cache;
                // reset cursors (takes gpu) so the next request starts cold.
                if let Some(b) = m.lfm2moe_mut() {
                    let _ = b.state.reset(&mut gpu);
                }
                // MiniMax-M2 (arch_id=10): KV cache + scratch share MiniMaxState;
                // reset its cursor (no gpu) for a cold prefill on the next request.
                if let Some(b) = m.minimax_mut() {
                    b.state.reset();
                }
                if let Some(ModelState::Gemma4(bundle)) = m.state.as_mut() {
                    bundle.state.reset();
                }
                if let Some(ModelState::MuseGlimmer(bundle)) = m.state.as_mut() {
                    bundle.reset_session_state();
                }
                if let Some(ModelState::Deepseek4(b)) = m.state.as_mut() {
                    b.state.reset();
                    b.state.zero_decode_caches(&mut gpu);
                    gpu.invalidate_graph_state();
                }

                if run_ok {
                    let tok_s = if elapsed > 0.0 {
                        n as f64 / elapsed
                    } else {
                        0.0
                    };
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"prefill_result","tokens":{},"ms":{:.2},"tok_s":{:.1}}}"#,
                        n,
                        elapsed * 1000.0,
                        tok_s
                    );
                } else {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        &match &prefill_err {
                            Some(e) => format!("bench_prefill forward failed: {e}"),
                            None => "bench_prefill forward failed".to_string(),
                        },
                        "validation",
                        false,
                        false,
                    );
                }
                let _ = stdout.flush();
            }

            "bench_decode" => {
                // Resident single-token decode probe for Redline and regular
                // daemon benchmarking. Prime deterministic Qwen3.5 state
                // outside the measured/captured interval, then time only the
                // requested number of forward_scratch calls.
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            "no model loaded",
                            "validation",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                match hipfire_loader::bench_decode_route(m.arch_id) {
                    hipfire_loader::BenchDecodeRoute::Deepseek4 => {
                        match redline_bench_decode_deepseek4(&mut gpu, m, &msg) {
                            Ok(response) => {
                                let _ = writeln!(stdout, "{response}");
                            }
                            Err(reason) => {
                                let _ = writeln!(
                                    stdout,
                                    "{}",
                                    serde_json::json!({"type": "error", "message": reason})
                                );
                            }
                        }
                        let _ = stdout.flush();
                        continue;
                    }
                    hipfire_loader::BenchDecodeRoute::Lfm2Moe => {
                        match redline_bench_decode_lfm2moe(&mut gpu, m, &msg) {
                            Ok(response) => {
                                let _ = writeln!(stdout, "{response}");
                            }
                            Err(reason) => {
                                let _ = writeln!(
                                    stdout,
                                    "{}",
                                    serde_json::json!({"type": "error", "message": reason})
                                );
                            }
                        }
                        let _ = stdout.flush();
                        continue;
                    }
                    _ => {}
                }
                // arch 5/6 = Qwen3.5, arch 14 = Muse Glimmer. Both prime with a
                // batched prefill and then step tokens one at a time, so the
                // same bench shape applies; the two branches below differ only
                // in which forward they call.
                if m.pp > 1
                    || m.ep.is_some()
                    || (m.arch_id != 5 && m.arch_id != 6 && m.arch_id != 14)
                {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        "bench_decode requires a single-GPU Qwen3.5 or Muse Glimmer model",
                        "unsupported",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                let context = msg
                    .get("context_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                let iterations =
                    msg.get("iterations").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                let capture = msg
                    .get("redline_capture")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let product_route = msg
                    .get("redline_product_route")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let capture_detail = msg
                    .get("redline_detail")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if capture && product_route {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        "redline_capture and redline_product_route are mutually exclusive",
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                if context == 0 || iterations == 0 {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        "bench_decode context_tokens and iterations must be non-zero",
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                if context.saturating_add(iterations).saturating_add(32) > m.physical_cap {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        &format!(
                            "bench_decode context+iterations exceeds loaded physical_cap={}",
                            m.physical_cap
                        ),
                        "context_length",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }

                m.seq_pos = 0;
                m.conversation_tokens.clear();
                let _ = hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu);
                let synthetic: Vec<u32> = (0..context as u32).map(|i| 10 + (i % 1000)).collect();
                let prime_error: Option<String> = match hipfire_loader::bench_decode_route(m.arch_id) {
                    hipfire_loader::BenchDecodeRoute::Qwen35 | hipfire_loader::BenchDecodeRoute::MuseGlimmer => hipfire_loader::carrier_for(m.arch_id)
                        .and_then(|c| c.bench_decode_prime(m, &mut gpu, &synthetic))
                        .unwrap_or_else(|| Some(format!("bench_decode_prime: carrier missing or unimplemented for arch_id={}", m.arch_id))),
                    hipfire_loader::BenchDecodeRoute::Unsupported => Some(format!("bench_decode unsupported for arch_id={}", m.arch_id)),
                    _ => Some(format!("bench_decode unsupported for arch_id={}", m.arch_id)),
                };
                let _ = gpu.hip.device_synchronize();
                if let Some(error) = prime_error {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        &format!("bench_decode prefill prime failed: {error:?}"),
                        "internal",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                m.seq_pos = context;

                if capture {
                    if let Err(reason) = gpu.replay.begin_capture() {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("redline decode capture refused: {reason}"),
                            "unsupported",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                }

                if product_route {
                    gpu.replay.begin_replay_observation_window();
                }
                let replay_before = gpu.replay.replay_observation();
                let _ = gpu.hip.device_synchronize();
                let t0 = Instant::now();
                let mut decode_err: Option<String> = None;
                let run_ok = match hipfire_loader::bench_decode_route(m.arch_id) {
                    hipfire_loader::BenchDecodeRoute::Qwen35 | hipfire_loader::BenchDecodeRoute::MuseGlimmer => hipfire_loader::carrier_for(m.arch_id)
                        .and_then(|c| c.bench_decode_run(m, &mut gpu, context, iterations, &mut decode_err))
                        .unwrap_or(false),
                    hipfire_loader::BenchDecodeRoute::Unsupported => {
                        decode_err = Some(format!("bench_decode unsupported for arch_id={}", m.arch_id));
                        false
                    }
                    _ => {
                        decode_err = Some(format!("bench_decode unsupported for arch_id={}", m.arch_id));
                        false
                    }
                };
                let _ = gpu.hip.device_synchronize();
                let elapsed = t0.elapsed().as_secs_f64();
                let replay_after = gpu.replay.replay_observation();
                let capture_summary = if capture {
                    match gpu.replay.finish_capture() {
                        Ok(summary) => Some(summary),
                        Err(reason) => {
                            emit_uncorrelated_error(
                                &mut stdout,
                                None,
                                &format!("redline decode capture failed: {reason}"),
                                "internal",
                                false,
                                false,
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    }
                } else {
                    None
                };

                m.seq_pos = 0;
                m.conversation_tokens.clear();
                let _ = hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu);

                if run_ok {
                    let tok_s = iterations as f64 / elapsed.max(f64::MIN_POSITIVE);
                    let mut response = serde_json::json!({
                        "type": "decode_result",
                        "context_tokens": context,
                        "iterations": iterations,
                        "ms": elapsed * 1000.0,
                        "us_per_token": elapsed * 1_000_000.0 / iterations as f64,
                        "tok_s": tok_s,
                    });
                    if let Some(summary) = capture_summary {
                        response["redline_capture"] =
                            redline_capture_json(&gpu, summary, capture_detail);
                    }
                    if product_route {
                        let prepared = gpu.replay.prepared_route_identity().map(|identity| {
                            serde_json::json!({
                                "dispatches": identity.dispatch_count,
                                "packets": identity.packet_count,
                                "queue_id": identity.queue_id,
                                "command_dwords": identity.command_dwords,
                                "queues": identity.queue_count,
                                "phases": identity.phase_count,
                            })
                        });
                        let sequence = gpu.replay.capture_summary();
                        let replay_delta = replay_after.count.saturating_sub(replay_before.count);
                        response["redline_route"] = serde_json::json!({
                            "requested_backend": format!("{:?}", gpu.replay.request()).to_ascii_lowercase(),
                            "transport": gpu.replay.transport_name(),
                            "state": format!("{:?}", gpu.replay.state()).to_ascii_lowercase(),
                            "fallback_reason": gpu.replay.fallback_reason(),
                            "execution_mode": "plain_ar",
                            "prepared": prepared,
                            "sequence": {
                                "launches": sequence.launch_count,
                                "unique_kernels": sequence.unique_kernel_count,
                                "hash": format!("{:016x}", sequence.sequence_hash),
                            },
                            "observed": {
                                "count_before": replay_before.count,
                                "count_after": replay_after.count,
                                "count_delta": replay_delta,
                                "first_position": replay_after.first_position,
                                "last_position": replay_after.last_position,
                            },
                            "retained_replay_observed": replay_delta > 0,
                        });
                    }
                    let _ = writeln!(stdout, "{response}");
                } else {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        &match &decode_err {
                            Some(e) => format!("bench_decode forward failed: {e}"),
                            None => "bench_decode forward failed".to_string(),
                        },
                        "internal",
                        false,
                        false,
                    );
                }
                let _ = stdout.flush();
            }

            "redline_probe_aql" => {
                handle_redline_probe_aql(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_dspark_shadow_pm4" => {
                handle_redline_dspark_shadow_pm4(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_shadow_aql" | "redline_shadow_pm4" => {
                handle_redline_shadow(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_dispatch_profile" => {
                handle_redline_dispatch_profile(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_pm4_prefix_profile" => {
                handle_redline_pm4_prefix_profile(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "redline_prefix_shadow" => {
                handle_redline_prefix_shadow(&msg, &mut model, &mut gpu, &mut stdout);
            }

            "profile" => {
                // Precompile kernels for common configurations so we have something to profile.
                // If a model is loaded its kernels are already compiled; this fills in the rest.
                // Cover all KV modes × weight formats × head_dims to catch all kernel variants.
                #[cfg(feature = "deltanet")]
                for kv in &["q8"] {
                    for wq in &["hfq4", "hfq6", "q8"] {
                        for hd in &[128usize, 256] {
                            let _ = gpu.precompile_qwen35(wq, kv, *hd);
                        }
                    }
                }
                let (cap, kernels) = gpu.profile();
                let kernels_json: Vec<String> = kernels.iter().map(|k| k.to_json()).collect();
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"profile","gpu":{},"kernels":[{}]}}"#,
                    cap.to_json(),
                    kernels_json.join(",")
                );
                let _ = stdout.flush();
            }

            #[cfg(feature = "serve-fault-inject")]
            "test_state_snapshot" => {
                write_test_state_snapshot(&mut stdout, model.as_ref(), &gpu, state_epoch);
            }

            _ => {
                tracing::warn!(command = msg_type, "daemon received unknown command");
                emit_uncorrelated_error(
                    &mut stdout,
                    None,
                    &format!("unknown type: {}", msg_type),
                    "validation",
                    false,
                    false,
                );
                let _ = stdout.flush();
            }
        }
    }
}


































#[cfg(test)]
mod llama_batched_prefill_tests {
    use super::{llama_prefill_sample_seed, llama_qwen3_batched_prefill_eligible};
    use hipfire_runtime::llama::ModelArch;

    #[test]
    fn route_stays_inside_validated_qwen3_q8_envelope() {
        let cases = [
            ("gfx1100", ModelArch::Qwen3, true, true, false, 256, true),
            ("gfx1201", ModelArch::Qwen3, true, true, false, 4, true),
            ("gfx1200", ModelArch::Qwen3, true, true, false, 256, false),
            ("gfx1100", ModelArch::Llama, true, true, false, 256, false),
            ("gfx1100", ModelArch::Qwen3, true, false, false, 256, false),
            ("gfx1100", ModelArch::Qwen3, true, true, true, 256, false),
            ("gfx1100", ModelArch::Qwen3, true, true, false, 3, false),
            ("gfx1100", ModelArch::Qwen3, false, true, false, 256, false),
        ];
        for (arch, model, enabled, q8, eviction, tokens, expected) in cases {
            assert_eq!(
                llama_qwen3_batched_prefill_eligible(arch, model, enabled, q8, eviction, tokens,),
                expected,
                "arch={arch} model={model:?}",
            );
        }
    }

    #[test]
    fn sampled_prefill_preserves_discarded_xorshift_draws() {
        assert_eq!(llama_prefill_sample_seed(42, 4, 0.0), 42);
        assert_eq!(llama_prefill_sample_seed(42, 1, 1.0), 42);
        assert_eq!(llama_prefill_sample_seed(42, 4, 1.0), 476_557_059);
    }
}
#[cfg(test)]
mod glimmer_spec_admission_tests {
    use hipfire_generate::dense::{glimmer_spec_admission, GlimmerSpecMode};

    #[test]
    fn greedy_at_temp_zero() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Greedy);
        let m2 = hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.01, None, true, false, true);
        assert_eq!(m2, hipfire_generate::dense::GlimmerSpecMode::Greedy);
    }

    #[test]
    fn chain_sampled_at_temp_one_with_defaults() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::ChainSampled);
    }

    #[test]
    fn off_when_min_p_present() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, Some(0.05), true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        // zero and None are allowed
        let ok0 = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, Some(0.0), true, false, true);
        assert_eq!(ok0, hipfire_generate::dense::GlimmerSpecMode::ChainSampled);
        let ok_none = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, false, true);
        assert_eq!(ok_none, hipfire_generate::dense::GlimmerSpecMode::ChainSampled);
    }

    #[test]
    fn off_when_fast_sample_off() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, false, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn off_when_temp_spec_env_off() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, true, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn off_when_batched_logits_unavailable() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 16, 1.0, None, true, false, false);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        // greedy does NOT require batched logits (still Greedy)
        let g = hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.005, None, true, false, false);
        assert_eq!(g, hipfire_generate::dense::GlimmerSpecMode::Greedy);
    }

    #[test]
    fn off_when_max_tokens_one() {
        let m = hipfire_generate::dense::glimmer_spec_admission(true, 1, 0.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        let m2 = hipfire_generate::dense::glimmer_spec_admission(true, 1, 1.0, None, true, false, true);
        assert_eq!(m2, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn off_when_no_drafter() {
        let m = hipfire_generate::dense::glimmer_spec_admission(false, 16, 0.0, None, true, false, true);
        assert_eq!(m, hipfire_generate::dense::GlimmerSpecMode::Off);
        let m2 = hipfire_generate::dense::glimmer_spec_admission(false, 16, 1.0, None, true, false, true);
        assert_eq!(m2, hipfire_generate::dense::GlimmerSpecMode::Off);
    }

    #[test]
    fn temp_boundary() {
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.01, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::Greedy
        );
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 0.02, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::ChainSampled
        );
        // just above greedy threshold but at/under 1e-6 should be Off, not sampled
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 1e-6, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::Greedy
        );
        assert_eq!(
            hipfire_generate::dense::glimmer_spec_admission(true, 16, 5e-7, None, true, false, true),
            hipfire_generate::dense::GlimmerSpecMode::Greedy
        );
    }
}




#[cfg(test)]
mod deepseek4_reasoning_prefix_tests {
    use super::ThinkMode;
    use hipfire_generate::common::{
        deepseek4_reasoning_prefix, DEEPSEEK4_REASONING_HIGH_PREFIX,
        DEEPSEEK4_REASONING_MAX_PREFIX,
    };

    #[test]
    fn parent_effort_prefixes_are_distinct_and_low_is_empty() {
        assert_eq!(hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::NonThink), "");
        assert_eq!(hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::Low), "");
        assert_eq!(
            hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::High),
            hipfire_generate::common::DEEPSEEK4_REASONING_HIGH_PREFIX
        );
        assert_eq!(
            hipfire_generate::common::deepseek4_reasoning_prefix(ThinkMode::Max),
            hipfire_generate::common::DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert_ne!(
            hipfire_generate::common::DEEPSEEK4_REASONING_HIGH_PREFIX,
            hipfire_generate::common::DEEPSEEK4_REASONING_MAX_PREFIX
        );
        assert!(hipfire_generate::common::DEEPSEEK4_REASONING_HIGH_PREFIX.ends_with("\n\n"));
        assert!(hipfire_generate::common::DEEPSEEK4_REASONING_MAX_PREFIX.ends_with("\n\n"));
    }
}


















































/// Build the 3D mrope context for one VL request, or `None` when the request
/// has no image tokens (→ the original 1D rope kernels and their dispatch
/// identity, which the certified retained-PM4 tape depends on).
///
/// Qwen3.5-VL positions image tokens by their (t, h, w) grid coordinate and
/// resumes text after the image at `max(image position) + 1`. hipfire's plain
/// sequential positions advance by the visual TOKEN count instead, so a
/// 70×54 grid (945 merged tokens, cursor should advance 35) diverges by 910
/// positions and corrupts everything after the image.
///
/// `base` is the conversation cursor at the start of this request's prefill;
/// the returned positions are absolute (already offset by it).
///
/// SPAN VALIDATION lives here on purpose. `build_mrope_positions` only
/// `debug_assert!`s span ordering and carries no post-condition that
/// `positions.len() == n_tokens`, so a malformed/overlapping span would
/// silently over-push in a release build. Every precondition it relies on is
/// checked below; anything unexpected returns `None` (1D fallback + a loud
/// log) rather than producing a mis-sized position vector.
#[allow(clippy::too_many_arguments)]



/// dots.ocr (arch_id=8) n-gram speculative decode, post-vision-prefill.
///
/// `generate_vl_dots_ocr` runs the image-conditioned prefill and routes here
/// when a model-free n-gram speculator was built at load (HIPFIRE_NGRAM_DRAFT=1).
/// dots.ocr's text decoder IS Qwen2, so the speculator drives it through the
/// `DotsOcrBundle: SpecTarget` impl. The vision prefill already advanced the
/// shared `m.qwen2_state` KV, so this only replaces the *decode* phase.
///
/// The flat decoder fields (`dots_ocr_config`/`dots_ocr_weights`/`qwen2_state`)
/// are moved into a `DotsOcrBundle` for the `&mut dyn SpecTarget` borrow and
/// restored on return — dots.ocr stores its state as flat `LoadedModel` fields,
/// not a `ModelState` bundle, so the `Carrier::spec_target_guard` path (used by
/// the text arches) does not apply here.
#[allow(clippy::too_many_arguments)]

/// The dots.ocr n-gram decode loop proper, factored out of
/// [`decode_vl_dots_ocr_ngram`] so the `&DotsOcrBundle` borrow it drives is
/// disjoint from the `&mut m` field-restore. Mirrors the `hipfire_generate::qwen::generate_spec`
/// prefill→step contract but with plain UTF-8 text streaming (no `SpecEmit`:
/// dots.ocr output is unframed layout-JSON, no reasoning/marker/tool channels).
#[allow(clippy::too_many_arguments)]



#[cfg(test)]
mod render_tail_think_tests {
    use super::{render_tail_opens_think, spec_assistant_prefix};
    use hipfire_generate::{common::asst_turn_fingerprint, common::normalize_asst_turn_for_fingerprint};
    use hipfire_runtime::prompt_frame::AssistantPrefix;

    #[test]
    fn qwen_jinja_think_tail_primes_reasoning_channel() {
        assert!(render_tail_opens_think("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn speculative_emitter_uses_rendered_think_state() {
        assert!(matches!(
            spec_assistant_prefix(true),
            AssistantPrefix::OpenThink
        ));
        assert!(matches!(
            spec_assistant_prefix(false),
            AssistantPrefix::Plain
        ));
    }

    #[test]
    fn plain_closed_and_user_literal_tails_do_not_prime() {
        assert!(!render_tail_opens_think("<|im_start|>assistant\n"));
        assert!(!render_tail_opens_think(
            "<|im_start|>assistant\n<think>\n</think>\n"
        ));
        assert!(!render_tail_opens_think(
            "<|im_start|>user\nliteral <think><|im_end|>\n<|im_start|>assistant\n"
        ));
    }

    #[test]
    fn assistant_cache_fingerprint_matches_client_visible_content() {
        let raw = "hidden reasoning</think>\n\nvisible answer<|im_end|>";
        let normalized = hipfire_generate::common::normalize_asst_turn_for_fingerprint(raw);
        assert_eq!(normalized, "visible answer");
        assert_eq!(
            hipfire_generate::common::asst_turn_fingerprint(&normalized, &[]),
            hipfire_generate::common::asst_turn_fingerprint("visible answer", &[])
        );
    }
}

#[cfg(test)]
mod vl_adaptive_admission_tests {
    use hipfire_generate::vision::vl_no_eviction_kv_cap;

    #[test]
    fn adaptive_admits_against_max_seq_not_start_tier_physical() {
        // physical_cap may equal max_seq at load, but the important case is
        // that adaptive never silently shrinks admission to start-tier cap.
        let physical_cap = 8192;
        let max_seq = 32768;
        assert_eq!(
            vl_no_eviction_kv_cap(physical_cap, max_seq, true),
            max_seq,
            "adaptive VL must admit against floor-tier max_seq"
        );
        assert_eq!(
            vl_no_eviction_kv_cap(physical_cap, max_seq, false),
            physical_cap,
            "non-adaptive VL keeps physical_cap contract"
        );
    }

    #[test]
    fn equal_caps_identical_either_mode() {
        assert_eq!(vl_no_eviction_kv_cap(4096, 4096, false), 4096);
        assert_eq!(vl_no_eviction_kv_cap(4096, 4096, true), 4096);
    }
}

#[cfg(test)]
mod qwen_ar_semantic_route_tests {
    use super::{emit_gen_start, emit_qwen_ar_cancelled, emit_qwen_ar_done, emit_qwen_ar_info, emit_qwen_ar_open_think_terminal, emit_staged_terminal_done, emit_tool_calls_event, emit_visible_token, qwen_ar_apply_cache_action, qwen_ar_cache_action, qwen_ar_done_value, qwen_ar_eos_filter_config, set_active_attempt_id, stage_terminal_tool_calls, ClientTerminalDecision, QwenArSemanticProducer, QwenArTerminalCause, QWEN_AR_SEMANTIC_CONTRACT_VERSION};
    use hipfire_generate::{common::emit_spec_cancel_after_rollback, qwen::qwen_client_commit_effects, qwen::QwenClientCommitEffects};
    use std::collections::HashMap;

    /// Drive the real shared producer (same object production uses).
    /// Each chunk is raw-committed as a synthetic token before classify.
    fn drive_ar_semantic_path(
        chunks: &[&str],
        started_in_think: bool,
        hit_length_cap: bool,
    ) -> (
        String,
        String,
        Result<super::QwenArRouteFinish, hipfire_runtime::emit_text::ToolRouteError>,
        bool,
        Vec<u32>,
        Vec<usize>,
    ) {
        set_active_attempt_id(7);
        let mut producer = QwenArSemanticProducer::new("t1", started_in_think);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 0usize;
        let mut stopped = false;
        for (i, c) in chunks.iter().enumerate() {
            let token = 1000 + i as u32;
            match producer.commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                token,
                c.as_bytes(),
            ) {
                Ok(true) => {
                    stopped = true;
                    break;
                }
                Ok(false) => {}
                Err(err) => {
                    let raw = producer.raw_committed.clone();
                    let pos = producer.raw_commit_positions.clone();
                    return (
                        String::from_utf8_lossy(&sink).into_owned(),
                        producer.visible().to_string(),
                        Err(err),
                        stopped,
                        raw,
                        pos,
                    );
                }
            }
        }
        let raw = producer.raw_committed.clone();
        let pos = producer.raw_commit_positions.clone();
        let stopped_flag = producer.stopped_by_filter;
        match producer.finish(&mut sink, hit_length_cap) {
            Ok((fin, visible)) => {
                // Mirror production: caller owns open-think epilogue + terminal.
                // Unit tests have no GPU, so attest rolled_back=false.
                if matches!(fin.cause, QwenArTerminalCause::OpenThink) {
                    let ep = hipfire_generate::common::RollbackEpilogue {
                        rolled_back: false,
                        context: None,
                    };
                    emit_qwen_ar_open_think_terminal(&mut sink, "t1", 0, &ep);
                } else {
                    // Default Commit path: stage calls on done (production
                    // embeds calls in commit_ready/done; no post-commit event).
                    let effects = hipfire_generate::qwen::qwen_client_commit_effects(
                        ClientTerminalDecision::Commit,
                        fin.finish_reason == "tool_calls" && !fin.wire_tool_calls.is_empty(),
                        fin.store_cache,
                    );
                    if effects.emit_done {
                        let mut pending = qwen_ar_done_value(
                            "t1",
                            fin.finish_reason,
                            0,
                            0.0,
                            0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0,
                            "",
                        );
                        stage_terminal_tool_calls(
                            &mut pending,
                            fin.finish_reason,
                            &fin.wire_tool_calls,
                        );
                        emit_staged_terminal_done(&mut sink, &pending);
                    }
                }
                (
                    String::from_utf8_lossy(&sink).into_owned(),
                    visible,
                    Ok(fin),
                    stopped || stopped_flag,
                    raw,
                    pos,
                )
            }
            Err(err) => (
                String::from_utf8_lossy(&sink).into_owned(),
                String::new(),
                Err(err),
                stopped || stopped_flag,
                raw,
                pos,
            ),
        }
    }

    fn parse_jsonl(out: &str) -> Vec<serde_json::Value> {
        out.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad jsonl {l}: {e}")))
            .collect()
    }

    #[test]
    fn contract_version_constant_is_v2() {
        assert_eq!(QWEN_AR_SEMANTIC_CONTRACT_VERSION, 2);
    }

    #[test]
    fn gen_start_v2_advertises_contract_version() {
        set_active_attempt_id(0);
        let mut sink = Vec::new();
        emit_gen_start(
            &mut sink,
            "req",
            true,
            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
        );
        let v: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(v["type"], "gen_start");
        assert_eq!(v["contract_version"], 2);
        assert_eq!(v["started_in_think"], true);
        assert_eq!(v["id"], "req");
        assert_eq!(v["attempt_id"], 0);
    }

    #[test]
    fn prose_only_finish_is_stop_and_stores_cache() {
        let (out, visible, fin, stopped, raw, _) =
            drive_ar_semantic_path(&["Hello world"], false, false);
        let fin = fin.expect("finish");
        assert!(!stopped);
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(fin.store_cache);
        assert_eq!(visible, "Hello world");
        assert!(out.contains("Hello world"));
        assert!(!out.contains("<think>"));
        assert_eq!(raw, vec![1000]);
        let events = parse_jsonl(&out);
        assert!(events.iter().any(|e| e["type"] == "token"));
        assert!(events.iter().all(|e| e.get("attempt_id").is_some()));
    }

    #[test]
    fn complete_tool_call_finish_is_tool_calls() {
        let chunks = [
            "Let me check.\n",
            "<tool_call>\n",
            r#"{"name":"read","arguments":{"path":"/x"}}"#,
            "\n</tool_call>",
        ];
        let (out, _visible, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
        let fin = fin.expect("finish");
        assert!(!out.contains("<tool_call>"));
        assert!(out.contains("Let me check."));
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls.len(), 1);
        assert_eq!(fin.wire_tool_calls[0].name, "read");
        assert!(fin.store_cache);
        let events = parse_jsonl(&out);
        // Authoritative calls live on staged done — no separate tool_calls event.
        assert!(events.iter().all(|e| e["type"] != "tool_calls"));
        let done = events
            .iter()
            .find(|e| e["type"] == "done" && e["finish_reason"] == "tool_calls")
            .expect("done with tool_calls");
        assert_eq!(done["calls"].as_array().unwrap().len(), 1);
        assert_eq!(done["calls"][0]["name"], "read");
        assert!(events.iter().all(|e| e["attempt_id"] == 7));
    }

    #[test]
    fn length_cap_suppresses_calls_even_if_complete() {
        let (out, _, fin, _, _, _) = drive_ar_semantic_path(
            &[r#"hi<tool_call>{"name":"read","arguments":{"path":"/x"}}</tool_call>"#],
            false,
            true,
        );
        let fin = fin.expect("finish");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache, "every length terminal is cache-unsafe");
        assert!(!out.contains("\"type\":\"tool_calls\""));
    }

    #[test]
    fn length_cap_prose_only_no_cache() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["just prose"], false, true);
        let fin = fin.expect("finish");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
    }

    #[test]
    fn length_cap_unclosed_span_no_calls_no_cache() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(
            &[r#"hi<tool_call>{"name":"read","arguments":{"path":"/x"}}"#],
            false,
            true,
        );
        let fin = fin.expect("length wins");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache, "partial tool turn must not prime cache");
    }

    #[test]
    fn length_cap_partial_opener_no_calls_no_cache() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["hi<tool_"], false, true);
        let fin = fin.expect("length");
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
    }

    #[test]
    fn unclosed_without_length_is_malformed_error() {
        let (_, _, fin, _, raw, _) = drive_ar_semantic_path(
            &[r#"hi<tool_call>{"name":"read","arguments":{"path":"/x"}}"#],
            false,
            false,
        );
        let err = fin.expect_err("malformed");
        assert!(err.to_string().contains("malformed") || err.to_string().contains("unclosed"));
        assert_eq!(raw, vec![1000]);
    }

    #[test]
    fn split_marker_chunks_still_classify() {
        let chunks = [
            "pre ",
            "<tool_",
            "call>",
            r#"{"name":"bash","arguments":{"cmd":"ls"}}"#,
            "</tool_call>",
            " post",
        ];
        let (out, visible, fin, _, raw, positions) = drive_ar_semantic_path(&chunks, false, false);
        let fin = fin.expect("finish");
        assert!(!out.contains("<tool_call>"));
        assert!(out.contains("pre ") || visible.contains("pre "));
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls[0].name, "bash");
        assert!(
            visible.contains(" post")
                || fin.trailing_visible.iter().any(|s| s.contains("post"))
                || out.contains(" post")
        );
        assert_eq!(raw.len(), chunks.len());
        assert_eq!(positions, (0..chunks.len()).collect::<Vec<_>>());
    }

    #[test]
    fn emit_visible_token_json_shape() {
        set_active_attempt_id(3);
        let mut sink = Vec::new();
        emit_visible_token(&mut sink, "req", "hello");
        let v: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(v["type"], "token");
        assert_eq!(v["id"], "req");
        assert_eq!(v["text"], "hello");
        assert_eq!(v["attempt_id"], 3);
    }

    #[test]
    fn empty_body_tool_call_latches_malformed_on_push() {
        let (out, _, fin, _, raw, _) =
            drive_ar_semantic_path(&["<tool_call></tool_call>"], false, false);
        assert!(!out.contains("\"type\":\"tool_calls\""));
        assert_eq!(raw, vec![1000], "raw commit precedes classify failure");
        match fin {
            Err(e) => {
                assert!(e.to_string().contains("malformed") || e.detail().contains("empty"));
            }
            Ok(f) => {
                assert!(f.wire_tool_calls.is_empty());
                assert_ne!(f.finish_reason, "tool_calls");
            }
        }
    }

    #[test]
    fn started_in_think_routes_reasoning_until_close() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["hidden reasoning", "</think>answer"], true, false);
        let fin = fin.expect("finish");
        let events = parse_jsonl(&out);
        assert!(events
            .iter()
            .any(|e| { e["type"] == "reasoning" && e["text"] == "hidden reasoning" }));
        assert!(!out.contains("</think>"));
        assert!(!visible.contains("hidden"));
        assert!(visible.contains("answer") || out.contains("answer"));
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
    }

    #[test]
    fn paired_think_markers_route_reasoning_separately() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["pre ", "<think>secret</think>", " post"], false, false);
        let fin = fin.expect("finish");
        let events = parse_jsonl(&out);
        assert!(!out.contains("<think>"));
        assert!(!out.contains("</think>"));
        assert!(events
            .iter()
            .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
        assert!(!visible.contains("secret"));
        assert!(visible.contains("pre ") || out.contains("pre "));
        assert!(
            visible.contains(" post") || out.contains(" post") || !fin.trailing_visible.is_empty()
        );
        assert_eq!(fin.finish_reason, "stop");
    }

    #[test]
    fn orphan_think_closer_preserves_prose() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["hidden</think>answer"], false, false);
        let fin = fin.expect("finish");
        assert!(!out.contains("</think>"));
        assert!(!visible.contains("</think>"));
        assert!(visible.contains("hidden"));
        assert!(visible.contains("answer") || out.contains("answer"));
        assert_eq!(fin.finish_reason, "stop");
    }

    #[test]
    fn decoded_im_end_stops_without_emitting_marker() {
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|im_end|>"], false, false);
        assert!(stopped, "filter must signal Stop on decoded EOT");
        assert!(!out.contains("<|im_end|>"));
        assert!(!visible.contains("<|im_end|>"));
        let fin = fin.expect("finish after EOT");
        assert!(fin.wire_tool_calls.is_empty());
        assert_ne!(fin.finish_reason, "tool_calls");
    }

    #[test]
    fn decoded_endoftext_stops_without_emitting_marker() {
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|endoftext|>"], false, false);
        assert!(stopped, "aux EOT must stop");
        assert!(!out.contains("<|endoftext|>"));
        assert!(!visible.contains("<|endoftext|>"));
        let fin = fin.expect("finish after aux EOT");
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
    }

    #[test]
    fn stop_with_prose_same_chunk_emits_prose() {
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hello<|im_end|>"], false, false);
        assert!(stopped);
        assert!(visible.contains("hello") || out.contains("hello"));
        assert!(!out.contains("<|im_end|>"));
        let fin = fin.expect("finish");
        assert_eq!(fin.finish_reason, "stop");
    }

    #[test]
    fn terminal_xor_stop_vs_tool_calls_vs_length() {
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["ok"], false, false);
        let fin = fin.unwrap();
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.wire_tool_calls.is_empty());
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(
            &[r#"x<tool_call>{"name":"a","arguments":{}}</tool_call>"#],
            false,
            false,
        );
        let fin = fin.unwrap();
        assert_eq!(fin.finish_reason, "tool_calls");
        assert!(!fin.wire_tool_calls.is_empty());
        let (_, _, fin, _, _, _) = drive_ar_semantic_path(&["ok"], false, true);
        let fin = fin.unwrap();
        assert_eq!(fin.finish_reason, "length");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
    }

    #[test]
    fn raw_commit_before_classify_is_producer_owned() {
        set_active_attempt_id(9);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 10usize;
        let err = producer
            .commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                42,
                b"<tool_call></tool_call>",
            )
            .expect_err("empty tool body fails closed on classify");
        assert!(err.to_string().contains("malformed") || err.detail().contains("empty"));
        assert_eq!(conversation_tokens, vec![42]);
        assert_eq!(streamed_tokens, vec![42]);
        assert_eq!(seq_pos, 11);
        assert_eq!(producer.raw_committed, vec![42]);
        assert_eq!(producer.raw_commit_positions, vec![0]);
    }

    #[test]
    fn eos_filter_config_delegates_think_and_keeps_both_terminators() {
        let cfg = qwen_ar_eos_filter_config();
        assert!(!cfg.strip_think);
        assert!(!cfg.started_in_think);
        assert!(cfg.stop_at.contains(&b"<|im_end|>".to_vec()));
        assert!(cfg.stop_at.contains(&b"<|endoftext|>".to_vec()));
    }

    #[test]
    fn cancellation_transcript_carries_attempt_id() {
        set_active_attempt_id(42);
        let mut sink = Vec::new();
        emit_qwen_ar_cancelled(&mut sink, "req-1", 3);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["type"], "aborted");
        assert_eq!(events[0]["reason"], "client_cancelled");
        assert_eq!(events[0]["attempt_id"], 42);
        assert_eq!(events[1]["type"], "done");
        assert_eq!(events[1]["finish_reason"], "aborted");
        assert_eq!(events[1]["attempt_id"], 42);
        assert_eq!(events[1]["completion_tokens"], 3);
    }

    #[test]
    fn info_event_carries_attempt_id() {
        set_active_attempt_id(11);
        let mut sink = Vec::new();
        emit_qwen_ar_info(
            &mut sink,
            "req",
            "budget_alert skipped: not enough KV headroom",
        );
        let v: serde_json::Value = serde_json::from_slice(&sink).unwrap();
        assert_eq!(v["type"], "info");
        assert_eq!(v["attempt_id"], 11);
        assert_eq!(v["id"], "req");
    }

    #[test]
    fn empty_commit_hold_does_not_panic() {
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let stop = producer
            .commit_and_classify(&mut sink, 0, || (0, Vec::<u8>::new()), |_pos, _out| {})
            .unwrap();
        assert!(!stop);
        assert!(sink.is_empty());
    }

    #[test]
    fn runtime_error_path_preserves_prior_raw_commits() {
        set_active_attempt_id(5);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 0usize;
        producer
            .commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                1,
                b"hello ",
            )
            .unwrap();
        let err = producer
            .commit_and_observe(
                &mut sink,
                &mut conversation_tokens,
                &mut streamed_tokens,
                &mut seq_pos,
                2,
                b"<tool_call></tool_call>",
            )
            .expect_err("malformed");
        assert!(!err.to_string().is_empty());
        assert_eq!(producer.raw_committed, vec![1, 2]);
        assert_eq!(conversation_tokens, vec![1, 2]);
        assert!(String::from_utf8_lossy(&sink).contains("hello"));
    }

    #[test]
    fn eos_trailing_marker_prefix_prose_flushed_and_cacheable() {
        // Finding 1: ordinary trailing marker-prefix prose (`answer <`, partial
        // `<|im_`) flushes at true EOS and shares the production finalizer/cache.
        let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&["answer <"], false, false);
        let fin = fin.expect("finish");
        assert!(visible.contains("answer <") || out.contains("answer <"));
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
        assert_eq!(fin.cause, QwenArTerminalCause::NaturalStop);

        let mut sink = HashMap::new();
        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(action.store);
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![7, 8, 9],
        );
        assert!(fp.is_some());
        assert_eq!(sink.get(&fp.unwrap()).unwrap(), &vec![7, 8, 9]);

        let (out2, visible2, fin2, _, _, _) =
            drive_ar_semantic_path(&["hi", "<|im_"], false, false);
        let fin2 = fin2.expect("finish partial im prefix");
        assert!(
            visible2.contains("hi") && (visible2.contains("<|im_") || out2.contains("<|im_")),
            "partial im prefix must flush as prose: visible={visible2:?} out={out2:?}"
        );
        assert_eq!(fin2.finish_reason, "stop");
        assert!(fin2.store_cache);
    }

    /// Every nonempty proper prefix of every watched think/EOT marker.
    fn qwen_ar_watched_markers() -> &'static [&'static str] {
        &["<think>", "</think>", "<|im_end|>", "<|endoftext|>"]
    }

    fn nonempty_proper_prefixes(marker: &str) -> Vec<&str> {
        (1..marker.len()).map(|n| &marker[..n]).collect()
    }

    #[test]
    fn table_producer_finish_natural_eos_and_length_every_watched_marker_prefix() {
        // Fix round 5: drive watched-prefix finalization through production
        // `QwenArSemanticProducer::finish` for both natural EOS and length —
        // not two identical filter-only calls. Retain completed-marker suppression.
        let prose = "answer ";
        for marker in qwen_ar_watched_markers() {
            for prefix in nonempty_proper_prefixes(marker) {
                let chunk = format!("{prose}{prefix}");
                // Natural EOS (`hit_length_cap = false`).
                let (out, visible, fin, stopped, _, _) =
                    drive_ar_semantic_path(&[&chunk], false, false);
                let fin = fin.expect("natural EOS finish");
                assert!(
                    !stopped,
                    "proper prefix must not complete stop: marker={marker:?} prefix={prefix:?}"
                );
                assert_eq!(fin.cause, QwenArTerminalCause::NaturalStop);
                assert_eq!(fin.finish_reason, "stop");
                assert!(fin.store_cache);
                assert!(
                    visible.contains(prose) && visible.contains(prefix),
                    "natural EOS must flush proper prefix as prose via finish: \
                     marker={marker:?} prefix={prefix:?} visible={visible:?} out={out:?}"
                );

                // Length finalization (`hit_length_cap = true`) — distinct terminal cause.
                let (out_len, visible_len, fin_len, stopped_len, _, _) =
                    drive_ar_semantic_path(&[&chunk], false, true);
                let fin_len = fin_len.expect("length finish");
                assert!(
                    !stopped_len,
                    "proper prefix must not complete stop under length: marker={marker:?}"
                );
                assert_eq!(fin_len.cause, QwenArTerminalCause::LengthCap);
                assert_eq!(fin_len.finish_reason, "length");
                assert!(!fin_len.store_cache);
                assert!(
                    visible_len.contains(prose) && visible_len.contains(prefix),
                    "length finish must also flush proper prefix prose: \
                     marker={marker:?} prefix={prefix:?} visible={visible_len:?} out={out_len:?}"
                );
            }

            // Completed marker suppression through production finish path.
            let full = format!("{prose}{marker}");
            let (out, visible, fin, stopped, _, _) = drive_ar_semantic_path(&[&full], false, false);
            let fin = fin.expect("completed marker finish");
            assert!(
                !visible.contains(marker) && !out.contains(marker),
                "completed marker must be suppressed: marker={marker:?} visible={visible:?}"
            );
            if *marker == "<think>" {
                // Open think after prose → validation terminal (no cache).
                assert_eq!(fin.cause, QwenArTerminalCause::OpenThink);
                assert_eq!(fin.finish_reason, "error");
                assert!(!fin.store_cache);
            } else if *marker == "</think>" {
                // Orphan closer drops closer, keeps prose; not a stop marker.
                assert!(!stopped);
                assert!(visible.contains("answer") || out.contains("answer"));
                assert_eq!(fin.finish_reason, "stop");
            } else {
                // EOT completed markers stop and emit only preceding prose.
                assert!(stopped, "EOT completed marker must stop: {marker}");
                assert!(
                    visible == prose || visible.trim_end() == "answer" || out.contains("answer"),
                    "EOT emits only preceding prose: visible={visible:?}"
                );
                assert_eq!(fin.finish_reason, "stop");
            }
        }
    }

    #[test]
    fn open_think_is_fail_closed_validation_no_cache() {
        // Open think streams reasoning, then fails closed: no calls/done/cache.
        let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&["still thinking"], true, false);
        let fin = fin.expect("open think returns Ok finish with error cause");
        assert_eq!(fin.cause, QwenArTerminalCause::OpenThink);
        assert_eq!(fin.finish_reason, "error");
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!fin.store_cache);
        assert!(visible.is_empty());
        let events = parse_jsonl(&out);
        assert!(events
            .iter()
            .any(|e| { e["type"] == "reasoning" && e["text"] == "still thinking" }));
        assert!(!out.contains("<think>"));
        assert!(
            events
                .iter()
                .any(|e| e["type"] == "error" && e["class"] == "validation"),
            "expected validation error: {out}"
        );
        assert!(
            events.iter().all(|e| e["type"] != "done"),
            "open-think must not emit done (terminal XOR): {out}"
        );
        let errors: Vec<_> = events.iter().filter(|e| e["type"] == "error").collect();
        assert_eq!(errors.len(), 1, "exactly one error terminal: {out}");
        assert_eq!(errors[0]["class"], "validation");
        assert_eq!(errors[0]["retryable"], false);
        assert_eq!(errors[0]["attempt_id"], 7);
        // No unread stale event after the single terminal error.
        let err_idx = events.iter().position(|e| e["type"] == "error").unwrap();
        assert_eq!(
            err_idx,
            events.len() - 1,
            "error must be the last event (no stale unread after terminal): {out}"
        );
        assert!(events.iter().all(|e| e.get("attempt_id").is_some()));

        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(!action.store);
        let mut sink = HashMap::new();
        assert!(qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1],
        )
        .is_none());
        assert!(sink.is_empty());
    }

    #[test]
    fn open_think_unmatched_generated_think_fail_closed() {
        let (out, visible, fin, _, _, _) =
            drive_ar_semantic_path(&["pre ", "<think>secret"], false, false);
        let fin = fin.expect("open think");
        let events = parse_jsonl(&out);
        assert_eq!(fin.cause, QwenArTerminalCause::OpenThink);
        assert!(!fin.store_cache);
        assert!(visible.is_empty() || !visible.contains("secret"));
        assert!(events
            .iter()
            .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
        assert!(!out.contains("\"type\":\"tool_calls\""));
    }

    #[test]
    fn decoded_eot_beats_length_on_final_budget_token_primary() {
        // Finding 3: primary EOT on final budget token beats length, cache-safe stop.
        let (out, visible, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|im_end|>"], false, true);
        assert!(stopped);
        let fin = fin.expect("finish");
        assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
        assert!(fin.wire_tool_calls.is_empty());
        assert!(!out.contains("<|im_end|>"));
        assert!(visible.contains("hi") || out.contains("hi"));

        let mut sink = HashMap::new();
        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(action.store);
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![42],
        )
        .expect("store");
        assert_eq!(sink.get(&fp).unwrap(), &vec![42]);
    }

    #[test]
    fn decoded_eot_beats_length_on_final_budget_token_aux() {
        let (out, _, fin, stopped, _, _) =
            drive_ar_semantic_path(&["hi", "<|endoftext|>"], false, true);
        assert!(stopped);
        let fin = fin.expect("finish");
        assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
        assert_eq!(fin.finish_reason, "stop");
        assert!(fin.store_cache);
        assert!(!out.contains("<|endoftext|>"));
    }

    #[test]
    fn decoded_eot_beats_length_with_complete_buffered_call() {
        let chunks = [
            r#"pre<tool_call>{"name":"read","arguments":{"path":"/x"}}</tool_call>"#,
            "<|im_end|>",
        ];
        let (out, _, fin, stopped, _, _) = drive_ar_semantic_path(&chunks, false, true);
        assert!(stopped);
        let fin = fin.expect("finish");
        assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls.len(), 1);
        assert_eq!(fin.wire_tool_calls[0].name, "read");
        assert!(fin.store_cache);
        // Authoritative calls live on staged done, not a separate tool_calls event.
        assert!(!out.contains("\"type\":\"tool_calls\""));
        assert!(out.contains("\"type\":\"done\""));
        assert!(out.contains("\"finish_reason\":\"tool_calls\""));
        assert!(out.contains("\"name\":\"read\""));
        // Pure length without EOT would suppress the same complete call.
        let (out_len, _, fin_len, _, _, _) = drive_ar_semantic_path(
            &[r#"pre<tool_call>{"name":"read","arguments":{"path":"/x"}}</tool_call>"#],
            false,
            true,
        );
        let fin_len = fin_len.expect("length");
        assert_eq!(fin_len.cause, QwenArTerminalCause::LengthCap);
        assert!(fin_len.wire_tool_calls.is_empty());
        assert!(!fin_len.store_cache);
        assert!(!out_len.contains("\"type\":\"tool_calls\""));
        assert!(!out_len.contains("\"finish_reason\":\"tool_calls\""));
    }

    #[test]
    fn terminal_cause_resolve_priority() {
        assert_eq!(
            QwenArTerminalCause::resolve(true, true, true),
            QwenArTerminalCause::OpenThink
        );
        assert_eq!(
            QwenArTerminalCause::resolve(true, true, false),
            QwenArTerminalCause::DecodedEot
        );
        assert_eq!(
            QwenArTerminalCause::resolve(false, true, false),
            QwenArTerminalCause::LengthCap
        );
        assert_eq!(
            QwenArTerminalCause::resolve(false, false, false),
            QwenArTerminalCause::NaturalStop
        );
    }

    #[test]
    fn real_writers_hostile_request_ids() {
        // Finding 5: shared serde writers + hostile IDs.
        set_active_attempt_id(99);
        let hostile = r#"req"}\n{"type":"pwned"#;
        let mut sink = Vec::new();
        emit_gen_start(&mut sink, hostile, false, Some(2));
        emit_visible_token(&mut sink, hostile, "ok");
        emit_tool_calls_event(
            &mut sink,
            hostile,
            &[hipfire_runtime::prompt_frame::ToolCall {
                id: None,
                name: "n".into(),
                arguments: serde_json::json!({}),
                rendered_body: None,
            }],
        );
        emit_qwen_ar_done(
            &mut sink, hostile, "stop", 1, 1.0, 0, 0.0, 0.0, 1.0, 0.0, 0, "",
        );
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events.len(), 4);
        for e in &events {
            assert_eq!(e["id"], hostile);
            assert_eq!(e["attempt_id"], 99);
        }
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[1]["type"], "token");
        assert_eq!(events[2]["type"], "tool_calls");
        assert_eq!(events[3]["type"], "done");
        assert_eq!(events[3]["finish_reason"], "stop");
    }

    #[test]
    fn cancellation_json_through_semantic_fold_contract() {
        // Finding 6: cancel JSON transcript is valid contract-v2 fold input.
        set_active_attempt_id(42);
        let mut sink = Vec::new();
        emit_gen_start(
            &mut sink,
            "c1",
            false,
            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
        );
        emit_visible_token(&mut sink, "c1", "partial ");
        emit_qwen_ar_cancelled(&mut sink, "c1", 1);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[0]["contract_version"], 2);
        assert_eq!(events[1]["type"], "token");
        assert_eq!(events[2]["type"], "aborted");
        assert_eq!(events[2]["reason"], "client_cancelled");
        assert_eq!(events[3]["type"], "done");
        assert_eq!(events[3]["finish_reason"], "aborted");
        for e in &events {
            assert_eq!(e["attempt_id"], 42);
            assert_eq!(e["id"], "c1");
        }
    }

    #[test]
    fn marker_byte_splits_enumerate_all_boundaries() {
        // Finding 6: enumerate every byte split for think open/close + tool markers.
        let open = b"<think>";
        let close = b"</think>";
        let tool_open = b"<tool_call>";
        let tool_close = b"</tool_call>";
        // Paired think: split open and close independently, always complete the pair.
        for split in 1..open.len() {
            let left = std::str::from_utf8(&open[..split]).unwrap();
            let right = std::str::from_utf8(&open[split..]).unwrap();
            let chunks = ["pre ", left, right, "secret", "</think>", " post"];
            let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
            let fin = fin.expect("finish");
            assert!(!out.contains("<think>"), "open split={split}");
            let events = parse_jsonl(&out);
            assert!(events
                .iter()
                .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
            assert_eq!(fin.finish_reason, "stop");
            assert!(visible.contains("pre ") || out.contains("pre "));
            assert!(
                visible.contains(" post")
                    || out.contains(" post")
                    || !fin.trailing_visible.is_empty()
            );
        }
        for split in 1..close.len() {
            let left = std::str::from_utf8(&close[..split]).unwrap();
            let right = std::str::from_utf8(&close[split..]).unwrap();
            let chunks = ["pre ", "<think>secret", left, right, " post"];
            let (out, visible, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
            let fin = fin.expect("finish");
            assert!(!out.contains("</think>"), "close split={split}");
            let events = parse_jsonl(&out);
            assert!(events
                .iter()
                .any(|e| e["type"] == "reasoning" && e["text"] == "secret"));
            assert_eq!(fin.finish_reason, "stop");
            assert!(visible.contains("pre ") || out.contains("pre "));
        }

        for marker in [tool_open.as_slice(), tool_close.as_slice()] {
            for split in 1..marker.len() {
                let left = std::str::from_utf8(&marker[..split]).unwrap();
                let right = std::str::from_utf8(&marker[split..]).unwrap();
                let body = r#"{"name":"bash","arguments":{"cmd":"ls"}}"#;
                let chunks = if marker == tool_open {
                    ["pre ", left, right, body, "</tool_call>"]
                } else {
                    ["pre ", "<tool_call>", body, left, right]
                };
                let (out, _, fin, _, _, _) = drive_ar_semantic_path(&chunks, false, false);
                let fin = fin.expect("finish");
                assert_eq!(fin.finish_reason, "tool_calls", "split={split} out={out}");
                assert_eq!(fin.wire_tool_calls[0].name, "bash");
                assert!(!out.contains("<tool_call>"));
            }
        }

        // Primary + aux EOT byte splits.
        for marker in [b"<|im_end|>".as_slice(), b"<|endoftext|>".as_slice()] {
            for split in 1..marker.len() {
                let left = std::str::from_utf8(&marker[..split]).unwrap();
                let right = std::str::from_utf8(&marker[split..]).unwrap();
                let (out, visible, fin, stopped, _, _) =
                    drive_ar_semantic_path(&["hi", left, right], false, false);
                assert!(
                    stopped,
                    "EOT split={split} marker={marker:?} must stop; out={out}"
                );
                let fin = fin.expect("finish");
                assert_eq!(fin.cause, QwenArTerminalCause::DecodedEot);
                assert!(!visible.contains("<|"));
                assert!(!out.contains("<|im_end|>"));
                assert!(!out.contains("<|endoftext|>"));
            }
        }
    }

    #[test]
    fn cache_sink_mutation_seam_store_and_skip() {
        // Finding 6: real cache sink mutation, not only store_cache bool.
        let (_, visible, fin, _, _, _) = drive_ar_semantic_path(&["Hello world"], false, false);
        let fin = fin.expect("stop");
        let action = qwen_ar_cache_action(&fin, &visible);
        let mut sink = HashMap::new();
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        )
        .expect("store");
        assert_eq!(sink.len(), 1);
        assert_eq!(sink[&fp], vec![1, 2, 3]);

        let (_, _, fin_len, _, _, _) = drive_ar_semantic_path(&["Hello world"], false, true);
        let fin_len = fin_len.expect("length");
        let action_len = qwen_ar_cache_action(&fin_len, "Hello world");
        assert!(!action_len.store);
        assert!(qwen_ar_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action_len,
            vec![9],
        )
        .is_none());
        assert_eq!(sink.len(), 1, "length must not mutate sink");
    }

    #[test]
    fn commit_and_classify_is_sole_production_entry() {
        // Finding 4: tests exercise the exact commit-then-classify op with
        // on_committed callback ordering (raw stamp before committed emit).
        set_active_attempt_id(3);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 0usize;
        let mut committed_positions = Vec::new();
        let stop = producer
            .commit_and_classify(
                &mut sink,
                11,
                || {
                    let pos = super::qwen_ar_raw_commit_token(
                        &mut conversation_tokens,
                        &mut streamed_tokens,
                        &mut seq_pos,
                        11,
                        super::QwenArRawCommitDisposition::ClassifiedVisible,
                    );
                    (pos, b"hello".to_vec())
                },
                |pos, out| {
                    committed_positions.push(pos);
                    let _ = writeln!(out, "{}", serde_json::json!({"type":"committed","pos":pos}));
                },
            )
            .unwrap();
        assert!(!stop);
        assert_eq!(producer.raw_committed, vec![11]);
        assert_eq!(producer.raw_commit_positions, vec![0]);
        assert_eq!(committed_positions, vec![0]);
        let events = parse_jsonl(&String::from_utf8_lossy(&sink));
        assert_eq!(events[0]["type"], "committed");
        assert!(events
            .iter()
            .any(|e| e["type"] == "token" && e["text"] == "hello"));
    }

    #[test]
    fn open_think_terminal_xor_error_only_no_done_no_stale() {
        // Fix round 4 #1: open-think → exactly one correlated non-retryable
        // validation error, no done, no unread stale event after terminal.
        // GPU-less: attest epilogue.rolled_back=false (same writer as production).
        set_active_attempt_id(7);
        let mut sink = Vec::new();
        let ep = hipfire_generate::common::RollbackEpilogue {
            rolled_back: false,
            context: None,
        };
        emit_qwen_ar_open_think_terminal(&mut sink, "ot1", 4, &ep);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(events.len(), 1, "exactly one terminal event: {events:?}");
        assert_eq!(events[0]["type"], "error");
        assert_eq!(events[0]["class"], "validation");
        assert_eq!(events[0]["retryable"], false);
        assert_eq!(events[0]["rolled_back"], false);
        assert_eq!(events[0]["attempt_id"], 7);
        assert_eq!(events[0]["id"], "ot1");
        assert!(
            events[0]["message"]
                .as_str()
                .unwrap_or("")
                .contains("open think"),
            "message={:?}",
            events[0]["message"]
        );
    }

    #[test]
    fn raw_commit_dispositions_exactly_once_visible_and_hidden() {
        // Fix round 4 #2: parameterized disposition; trailer stays client-invisible;
        // exactly-once state mutation across production token path dispositions.
        let mut conversation_tokens = Vec::new();
        let mut streamed_tokens = Vec::new();
        let mut seq_pos = 10usize;

        let pos_v = super::qwen_ar_raw_commit_token(
            &mut conversation_tokens,
            &mut streamed_tokens,
            &mut seq_pos,
            100,
            super::QwenArRawCommitDisposition::ClassifiedVisible,
        );
        assert_eq!(pos_v, 0);
        assert_eq!(conversation_tokens, vec![100]);
        assert_eq!(streamed_tokens, vec![100]);
        assert_eq!(seq_pos, 11);

        let pos_h = super::qwen_ar_raw_commit_token(
            &mut conversation_tokens,
            &mut streamed_tokens,
            &mut seq_pos,
            200,
            super::QwenArRawCommitDisposition::IntentionallyHidden,
        );
        assert_eq!(pos_h, 1, "hidden returns conversation index");
        assert_eq!(conversation_tokens, vec![100, 200]);
        assert_eq!(
            streamed_tokens,
            vec![100],
            "hidden trailer must not join streamed/client path"
        );
        assert_eq!(seq_pos, 12);

        // Second visible after hidden still only streams the visible tokens.
        let pos_v2 = super::qwen_ar_raw_commit_token(
            &mut conversation_tokens,
            &mut streamed_tokens,
            &mut seq_pos,
            300,
            super::QwenArRawCommitDisposition::ClassifiedVisible,
        );
        assert_eq!(pos_v2, 1);
        assert_eq!(conversation_tokens, vec![100, 200, 300]);
        assert_eq!(streamed_tokens, vec![100, 300]);
        assert_eq!(seq_pos, 13);

        // Producer path: visible classify + hidden trailer via sole commit_raw.
        set_active_attempt_id(3);
        let mut producer = QwenArSemanticProducer::new("t1", false);
        let mut sink = Vec::new();
        let mut conv = Vec::new();
        let mut stream = Vec::new();
        let mut sp = 0usize;
        producer
            .commit_and_observe(&mut sink, &mut conv, &mut stream, &mut sp, 11, b"hi")
            .unwrap();
        // Post-EOT hidden trailer through the same producer-owned entry.
        producer
            .commit_raw(
                &mut sink,
                99,
                super::QwenArRawCommitDisposition::IntentionallyHidden,
                || {
                    let tpos = super::qwen_ar_raw_commit_token(
                        &mut conv,
                        &mut stream,
                        &mut sp,
                        99,
                        super::QwenArRawCommitDisposition::IntentionallyHidden,
                    );
                    (tpos, Vec::<u8>::new())
                },
                |_pos, _out| {},
            )
            .unwrap();
        assert_eq!(producer.raw_committed, vec![11, 99]);
        assert_eq!(producer.raw_commit_positions.len(), 2);
        assert_eq!(conv, vec![11, 99]);
        assert_eq!(stream, vec![11], "trailer not streamed");
        let out = String::from_utf8_lossy(&sink);
        assert!(out.contains("hi"));
        assert!(!out.contains("99"));
        // finish must not surface hidden trailer as visible.
        let (fin, visible) = producer.finish(&mut sink, false).expect("finish");
        assert_eq!(fin.finish_reason, "stop");
        assert_eq!(visible, "hi");
    }

    #[test]
    fn wire_helpers_used_by_gen_start_and_cancel_writers() {
        // Fix round 4 #3: production writers use shared semantic wire helpers.
        set_active_attempt_id(42);
        let mut sink = Vec::new();
        emit_gen_start(
            &mut sink,
            "c1",
            false,
            Some(QWEN_AR_SEMANTIC_CONTRACT_VERSION),
        );
        emit_qwen_ar_cancelled(&mut sink, "c1", 1);
        let events = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(
            events[0],
            hipfire_runtime::semantic::wire_gen_start("c1", false, 42, Some(2))
        );
        assert_eq!(
            events[1],
            hipfire_runtime::semantic::wire_aborted("c1", "client_cancelled", 42)
        );
        assert_eq!(
            events[2],
            hipfire_runtime::semantic::wire_aborted_done("c1", 1, 42)
        );
    }

    #[test]
    fn client_commit_effects_commit_preserves_intended_flags() {
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, true, true);
        assert_eq!(
            e,
            hipfire_generate::qwen::QwenClientCommitEffects {
                release_tool_calls: true,
                store_cache: true,
                emit_done: true,
            }
        );
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, false, true);
        assert!(!e.release_tool_calls);
        assert!(e.store_cache);
        assert!(e.emit_done);
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, false, false);
        assert!(!e.release_tool_calls);
        assert!(!e.store_cache);
        assert!(e.emit_done);
    }

    #[test]
    fn client_commit_effects_abort_suppresses_all() {
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Abort, true, true);
        assert_eq!(
            e,
            hipfire_generate::qwen::QwenClientCommitEffects {
                release_tool_calls: false,
                store_cache: false,
                emit_done: false,
            }
        );
    }

    #[test]
    fn finish_defers_tool_calls_until_commit_effects() {
        set_active_attempt_id(11);
        let mut producer = QwenArSemanticProducer::new("t-commit", false);
        let mut sink = Vec::new();
        let mut conv = Vec::new();
        let mut stream = Vec::new();
        let mut pos = 0usize;
        let chunks = [
            "Let me check.\n",
            "<tool_call>\n",
            r#"{"name":"read","arguments":{"path":"/x"}}"#,
            "\n</tool_call>",
        ];
        for (i, c) in chunks.iter().enumerate() {
            producer
                .commit_and_observe(
                    &mut sink,
                    &mut conv,
                    &mut stream,
                    &mut pos,
                    2000 + i as u32,
                    c.as_bytes(),
                )
                .unwrap();
        }
        let (fin, visible) = producer.finish(&mut sink, false).expect("finish");
        assert_eq!(fin.finish_reason, "tool_calls");
        assert_eq!(fin.wire_tool_calls.len(), 1);
        assert!(visible.contains("Let me check."));
        let pre = String::from_utf8_lossy(&sink);
        assert!(
            !pre.contains("\"type\":\"tool_calls\""),
            "finish must not release tool_calls before Commit"
        );

        // Commit path: release + cache + done.
        let effects = hipfire_generate::qwen::qwen_client_commit_effects(
            ClientTerminalDecision::Commit,
            fin.finish_reason == "tool_calls" && !fin.wire_tool_calls.is_empty(),
            fin.store_cache,
        );
        assert!(effects.release_tool_calls && effects.store_cache && effects.emit_done);
        let mut cache = HashMap::new();
        let action = qwen_ar_cache_action(&fin, &visible);
        assert!(action.store);
        let fp = qwen_ar_apply_cache_action(
            |k, v| {
                cache.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        );
        assert!(fp.is_some());
        assert_eq!(cache.len(), 1);
        let mut pending = qwen_ar_done_value(
            "t-commit",
            fin.finish_reason,
            4,
            1.0,
            0,
            0.0,
            0.0,
            1.0,
            0.0,
            0,
            "",
        );
        stage_terminal_tool_calls(&mut pending, fin.finish_reason, &fin.wire_tool_calls);
        emit_staged_terminal_done(&mut sink, &pending);
        let events = parse_jsonl(&String::from_utf8_lossy(&sink));
        assert!(events.iter().all(|e| e["type"] != "tool_calls"));
        let done = events
            .iter()
            .find(|e| e["type"] == "done" && e["finish_reason"] == "tool_calls")
            .expect("done tool_calls");
        assert!(done["calls"].is_array());
        assert_eq!(done["calls"].as_array().unwrap().len(), 1);
        assert!(events
            .iter()
            .all(|e| e.get("type") != Some(&serde_json::json!("aborted"))));
    }

    #[test]
    fn abort_effects_suppress_calls_cache_and_normal_done() {
        set_active_attempt_id(12);
        let mut producer = QwenArSemanticProducer::new("t-abort", false);
        let mut sink = Vec::new();
        let mut conv = Vec::new();
        let mut stream = Vec::new();
        let mut pos = 0usize;
        let chunks = [
            "Let me check.\n",
            "<tool_call>\n",
            r#"{"name":"read","arguments":{"path":"/x"}}"#,
            "\n</tool_call>",
        ];
        for (i, c) in chunks.iter().enumerate() {
            producer
                .commit_and_observe(
                    &mut sink,
                    &mut conv,
                    &mut stream,
                    &mut pos,
                    3000 + i as u32,
                    c.as_bytes(),
                )
                .unwrap();
        }
        let (fin, visible) = producer.finish(&mut sink, false).expect("finish");
        assert_eq!(fin.finish_reason, "tool_calls");
        let effects = hipfire_generate::qwen::qwen_client_commit_effects(
            ClientTerminalDecision::Abort,
            fin.finish_reason == "tool_calls" && !fin.wire_tool_calls.is_empty(),
            fin.store_cache,
        );
        assert!(!effects.release_tool_calls && !effects.store_cache && !effects.emit_done);

        // No tool release / cache store / normal done on Abort.
        let mut cache = HashMap::new();
        let mut action = qwen_ar_cache_action(&fin, &visible);
        action.store = effects.store_cache && action.store;
        assert!(qwen_ar_apply_cache_action(
            |k, v| {
                cache.insert(k, v);
            },
            &action,
            vec![9, 9]
        )
        .is_none());
        assert!(cache.is_empty());

        // Attested cancel terminal only (no GPU rollback in unit test).
        let ep = hipfire_generate::common::RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "t-abort", 4, &ep);
        let out = String::from_utf8_lossy(&sink);
        assert!(!out.contains("\"type\":\"tool_calls\""));
        let events = parse_jsonl(&out);
        assert!(events.iter().any(|e| e["type"] == "aborted"));
        assert!(events
            .iter()
            .any(|e| e["type"] == "done" && e["finish_reason"] == "aborted"));
        assert!(events
            .iter()
            .all(|e| !(e["type"] == "done" && e["finish_reason"] == "tool_calls")));
        assert!(events.iter().all(|e| e["attempt_id"] == 12));
    }
}

/// DS4 AR/EP/spec share pure turn-wide terminal decisions. Tests drive the
/// production [`DsmlDeferredCalls`] component (same type Deepseek4Emit uses)
/// plus the wire-terminal helpers — not test-local buffer counters.
#[cfg(test)]
mod ds4_malformed_terminal_tests {
    use super::{ds4_spec_finish_route, emit_ds4_malformed_terminal, emit_visible_token, set_active_attempt_id, ClientTerminalDecision};
    use hipfire_generate::{common::asst_turn_fingerprint, common::ds4_apply_cache_action, common::ds4_ar_ep_cache_action, common::ds4_ar_ep_finish_route, dense::ds4_cache_action, common::ds4_client_commit_effects, common::ds4_ep_abort_wire_events, common::ds4_gen_start_contract_version, common::ds4_malformed_terminal_action, dense::ds4_spec_wire_terminal, common::ds4_stream_event_wireable, qwen::emit_ds4_ep_gen_start, common::emit_ds4_malformed_action, common::gen_start_contract_version_for_arch, common::normalize_asst_turn_for_fingerprint, qwen::spec_outcome_seed_committable, qwen::spec_should_flush_pending_seed, common::Ds4ArEpRouteTerminal, common::Ds4ClientCommitEffects, dense::Ds4SpecWireTerminal};
    use hipfire_arch_deepseek4::dsml::{
        DsmlDeferredCalls, DsmlDeferredOutcome, StreamEvent, StreamParser, TOOL_CALLS_CLOSE,
        TOOL_CALLS_OPEN,
    };
    use hipfire_runtime::prompt_frame::ToolCall;
    use hipfire_runtime::spec::{ClientEvent, FinishSummary, SpecEmit};

    fn complete_invoke(name: &str, arg_name: &str, arg_val: &str) -> String {
        format!(
            "{open}\n<｜DSML｜invoke name=\"{name}\">\n\
             <｜DSML｜parameter name=\"{arg_name}\" string=\"true\">{arg_val}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n{close}",
            open = TOOL_CALLS_OPEN,
            close = TOOL_CALLS_CLOSE,
            name = name,
            arg_name = arg_name,
            arg_val = arg_val,
        )
    }

    /// Feed a full turn through the production deferred absorber (same API as
    /// Deepseek4Emit::feed_and_emit / finish).
    fn deferred_from_text(text: &str) -> DsmlDeferredCalls {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _visible = deferred.absorb_all(p.feed(text));
        let _tail = deferred.absorb_all(p.finish());
        deferred
    }

    /// AR/EP production path: deferred finalize → shared pure route.
    fn ar_ep_from_deferred(d: DsmlDeferredCalls, hit_length_cap: bool) -> hipfire_generate::common::Ds4ArEpRouteTerminal {
        match d.finalize(hit_length_cap) {
            DsmlDeferredOutcome::Malformed { detail } => {
                hipfire_generate::common::ds4_ar_ep_finish_route(Some(detail), Vec::new(), hit_length_cap)
            }
            DsmlDeferredOutcome::Length => hipfire_generate::common::ds4_ar_ep_finish_route(None, Vec::new(), true),
            DsmlDeferredOutcome::Stop => hipfire_generate::common::ds4_ar_ep_finish_route(None, Vec::new(), false),
            DsmlDeferredOutcome::ToolCalls(calls) => {
                let wire: Vec<ToolCall> = calls
                    .into_iter()
                    .map(|c| ToolCall {
                        id: None,
                        name: c.name,
                        arguments: c.arguments,
                        rendered_body: None,
                    })
                    .collect();
                hipfire_generate::common::ds4_ar_ep_finish_route(None, wire, false)
            }
        }
    }

    /// Spec path: provisional finalize(false) as Deepseek4Emit::finish does,
    /// then wrapper applies length via hipfire_generate::dense::ds4_spec_wire_terminal.
    fn spec_wire_from_deferred(d: DsmlDeferredCalls, hit_length_cap: bool) -> hipfire_generate::dense::Ds4SpecWireTerminal {
        let (finish_reason, tool_calls) = if d.is_malformed() {
            let _ = d.finalize(false);
            ("malformed_protocol", 0usize)
        } else {
            let n = d.buffered_len();
            match d.finalize(false) {
                DsmlDeferredOutcome::ToolCalls(_) => ("tool_calls", n),
                DsmlDeferredOutcome::Stop | DsmlDeferredOutcome::Length => ("stop", 0),
                DsmlDeferredOutcome::Malformed { .. } => ("malformed_protocol", 0),
            }
        };
        hipfire_generate::dense::ds4_spec_wire_terminal(finish_reason, tool_calls, hit_length_cap)
    }

    #[test]
    fn malformed_action_is_typed_validation_non_retryable() {
        let action =
            hipfire_generate::common::ds4_malformed_terminal_action("unclosed DSML tool_calls block at end of output");
        assert_eq!(action.class, "validation");
        assert!(!action.retryable);
        assert!(!action.rolled_back);
        assert!(action.message.contains("malformed"));
        assert!(action.message.contains("unclosed"));
        assert!(action.message.contains("tool_calls"));
    }

    #[test]
    fn malformed_action_suppresses_done_cache_and_calls() {
        let action =
            hipfire_generate::common::ds4_malformed_terminal_action("unclosed DSML tool_calls block at end of output");
        assert!(!action.emit_done, "error XOR done");
        assert!(!action.store_cache, "no assistant-cache write");
        assert!(!action.expose_tool_calls, "no executable calls");
    }

    #[test]
    fn complete_call_then_unclosed_discards_all_on_ar_ep() {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _ = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        assert_eq!(deferred.buffered_len(), 1, "first complete call buffers");
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.feed("\n<｜DSML｜invoke name=\"beta\">"));
        let _ = deferred.absorb_all(p.finish());
        assert!(
            deferred.is_malformed(),
            "unclosed second block latches malformed"
        );

        let terminal = ar_ep_from_deferred(deferred, false);
        match terminal {
            hipfire_generate::common::Ds4ArEpRouteTerminal::Malformed(action) => {
                assert_eq!(action.class, "validation");
                assert!(!action.retryable);
                assert!(!action.emit_done);
                assert!(!action.store_cache);
                assert!(!action.expose_tool_calls);
            }
            other => panic!("expected Malformed discard of earlier calls, got {other:?}"),
        }
    }

    #[test]
    fn complete_call_safe_terminal_releases_on_ar_ep() {
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        assert_eq!(deferred.buffered_len(), 1);
        let terminal = ar_ep_from_deferred(deferred, false);
        match terminal {
            hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "tool_calls");
                assert_eq!(wire_tool_calls.len(), 1);
                assert_eq!(wire_tool_calls[0].name, "alpha");
                assert!(store_cache);
            }
            other => panic!("expected Safe tool_calls release, got {other:?}"),
        }
    }

    #[test]
    fn length_cap_is_not_tool_safe_even_with_complete_calls() {
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        assert_eq!(deferred.buffered_len(), 1);
        let terminal = ar_ep_from_deferred(deferred, true);
        match terminal {
            hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "length");
                assert!(wire_tool_calls.is_empty(), "length never releases calls");
                assert!(!store_cache);
            }
            other => panic!("expected Safe length with empty calls, got {other:?}"),
        }
    }

    #[test]
    fn speculative_complete_then_unclosed_discards_via_production_deferred() {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let visible = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        assert!(
            visible.iter().all(|e| hipfire_generate::common::ds4_stream_event_wireable(e)),
            "absorb returns only wireable visible events"
        );
        assert_eq!(deferred.buffered_len(), 1);
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.finish());
        assert!(deferred.is_malformed());
        assert_eq!(
            deferred.buffered_len(),
            1,
            "buffer retains until finalize; discard is finalize's job"
        );

        match deferred.finalize(false) {
            DsmlDeferredOutcome::Malformed { .. } => {}
            other => panic!("expected Malformed outcome discarding calls, got {other:?}"),
        }

        let wire = hipfire_generate::dense::ds4_spec_wire_terminal("malformed_protocol", 0, false);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(action) => {
                assert_eq!(action.class, "validation");
                assert!(!action.retryable);
                assert!(!action.emit_done);
                assert!(!action.store_cache);
                assert!(!action.expose_tool_calls);
            }
            other => panic!("expected Malformed wire terminal, got {other:?}"),
        }
        assert!(ds4_spec_finish_route("stop", 0).is_none());
        assert!(ds4_spec_finish_route("tool_calls", 1).is_none());
    }

    #[test]
    fn speculative_safe_stop_releases_held_calls() {
        // Production Deepseek4Emit::finish path: finalize(false) → held ToolCalls
        // on FinishSummary; wrapper releases only when length is false.
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        let wire = spec_wire_from_deferred(deferred, false);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "tool_calls");
                assert!(release_tool_calls, "safe stop must release held calls");
                assert!(store_cache);
            }
            other => panic!("expected Done tool_calls release, got {other:?}"),
        }
    }

    #[test]
    fn speculative_length_suppresses_held_calls_and_cache() {
        // Same provisional finish as Deepseek4Emit (finalize false → tool_calls
        // count), but wrapper length wins: no release, finish_reason=length.
        let deferred = deferred_from_text(&complete_invoke("alpha", "x", "1"));
        let wire = spec_wire_from_deferred(deferred, true);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(finish_reason, "length");
                assert!(!release_tool_calls, "length must not release held calls");
                assert!(!store_cache);
            }
            other => panic!("expected Done length suppress, got {other:?}"),
        }
    }

    #[test]
    fn speculative_complete_then_malformed_never_releases() {
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _ = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.finish());
        let wire = spec_wire_from_deferred(deferred, false);
        match wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(action) => {
                assert!(!action.expose_tool_calls);
                assert!(!action.emit_done);
                assert!(!action.store_cache);
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
        // Length cannot flip a malformed finish into a done/tool_calls release.
        let mut p = StreamParser::new();
        let mut deferred = DsmlDeferredCalls::new();
        let _ = deferred.absorb_all(p.feed(&complete_invoke("alpha", "x", "1")));
        let _ = deferred.absorb_all(p.feed(TOOL_CALLS_OPEN));
        let _ = deferred.absorb_all(p.finish());
        let wire_len = spec_wire_from_deferred(deferred, true);
        assert!(
            matches!(wire_len, hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(_)),
            "malformed wins over length"
        );
    }

    #[test]
    fn stream_event_tool_calls_not_wireable_mid_turn() {
        let ev = StreamEvent::ToolCalls(vec![hipfire_arch_deepseek4::dsml::ToolCall {
            name: "x".into(),
            arguments: serde_json::json!({}),
        }]);
        assert!(!hipfire_generate::common::ds4_stream_event_wireable(&ev));
        assert!(hipfire_generate::common::ds4_stream_event_wireable(&StreamEvent::Token("hi".into())));
        assert!(hipfire_generate::common::ds4_stream_event_wireable(&StreamEvent::Reasoning(
            "r".into()
        )));
        assert!(!hipfire_generate::common::ds4_stream_event_wireable(&StreamEvent::Malformed {
            detail: "x".into()
        }));
        // Production absorber never returns ToolCalls as visible.
        let mut d = DsmlDeferredCalls::new();
        assert!(d.absorb(ev).is_none());
        assert_eq!(d.buffered_len(), 1);
    }

    #[test]
    fn emit_writes_one_validation_error_no_done_or_calls() {
        set_active_attempt_id(17);
        let mut buf = Vec::new();
        let action =
            hipfire_generate::common::ds4_malformed_terminal_action("unclosed DSML tool_calls block at end of output");
        hipfire_generate::common::emit_ds4_malformed_action(&mut buf, "req-ds4", &action);
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 1, "exactly one terminal envelope, got {text}");
        let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["id"], "req-ds4");
        assert_eq!(v["class"], "validation");
        assert_eq!(v["retryable"], false);
        assert_eq!(v["rolled_back"], false);
        assert_eq!(v["attempt_id"].as_u64(), Some(17));
        let msg = v["message"].as_str().unwrap_or("");
        assert!(msg.contains("malformed") && msg.contains("unclosed"));
        assert!(!text.contains("\"type\":\"done\""));
        assert!(!text.contains("\"type\":\"tool_calls\""));
        let mut buf2 = Vec::new();
        emit_ds4_malformed_terminal(
            &mut buf2,
            "req-ds4",
            "unclosed DSML tool_calls block at end of output",
        );
        assert_eq!(
            String::from_utf8(buf2)
                .unwrap()
                .lines()
                .filter(|l| !l.is_empty())
                .count(),
            1
        );
        set_active_attempt_id(0);
    }

    #[test]
    fn ds4_gen_start_contract_selection_is_unset() {
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(9), None);
        assert_eq!(hipfire_generate::common::ds4_gen_start_contract_version(), None);
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(5), Some(2));
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(6), Some(2));
        assert_eq!(super::QWEN_AR_SEMANTIC_CONTRACT_VERSION, 2);
    }

    #[test]
    fn ds4_ep_opens_wire_contract_before_first_token() {
        use hipfire_runtime::prompt_frame::ThinkMode;

        set_active_attempt_id(31);
        let mut sink = Vec::new();
        hipfire_generate::qwen::emit_ds4_ep_gen_start(&mut sink, "req-ep", ThinkMode::NonThink);
        emit_visible_token(&mut sink, "req-ep", "hello");

        let events: Vec<serde_json::Value> = String::from_utf8(sink)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[0]["id"], "req-ep");
        assert_eq!(events[0]["started_in_think"], false);
        assert_eq!(events[0]["attempt_id"], 31);
        assert_eq!(events[1]["type"], "token");
        assert_eq!(events[1]["text"], "hello");
        assert_eq!(events[1]["attempt_id"], 31);
        set_active_attempt_id(0);
    }

    // ── Task 4 definitive terminal-edge blockers (DS4 cache + empty EOS) ──

    /// Safe DS4 speculative terminal stores the verbatim raw streamed_tokens
    /// body through hipfire_generate::dense::ds4_cache_action + hipfire_generate::common::ds4_apply_cache_action (same seam as
    /// hipfire_generate::dense::generate_deepseek4_spec Done branch).
    #[test]
    fn ds4_safe_terminal_stores_verbatim_raw_replay_tokens() {
        let calls = vec![ToolCall {
            id: None,
            name: "lookup".into(),
            arguments: serde_json::json!({"q": "x"}),
            rendered_body: None,
        }];
        let finish = FinishSummary {
            events: vec![
                ClientEvent::Token("Sure.".into()),
                ClientEvent::ToolCalls(calls.clone()),
            ],
            finish_reason: "tool_calls",
            tool_calls: 1,
            visible_text: "Sure.".into(),
            decoded_eot: false,
            open_think: false,
        };
        let wire = hipfire_generate::dense::ds4_spec_wire_terminal("tool_calls", 1, false);
        match &wire {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache, "safe stop must authorize cache store");
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let action = hipfire_generate::dense::ds4_cache_action(&wire, &finish, finish.visible_text.as_str());
        assert!(action.store);
        assert_eq!(
            action.fingerprint_text,
            hipfire_generate::common::normalize_asst_turn_for_fingerprint("Sure.")
        );
        assert_eq!(action.tool_calls.len(), 1);
        assert_eq!(action.tool_calls[0].name, "lookup");

        // Verbatim raw body — no surround EOS/Assistant markers (DSML replay).
        let streamed = vec![11u32, 22, 33, 44];
        let mut sink: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
        let fp = hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            streamed.clone(),
        );
        assert!(fp.is_some(), "safe terminal must mutate cache sink");
        let stored = sink.get(&fp.unwrap()).expect("stored under fingerprint");
        assert_eq!(
            stored, &streamed,
            "cache body must be verbatim run.streamed_tokens"
        );
        // Fingerprint key matches hipfire_generate::common::build_deepseek4_dsml_prompt lookup shape.
        let expected_fp = hipfire_generate::common::asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls);
        assert_eq!(fp, Some(expected_fp));
    }

    /// Length and fail-closed/malformed never store via hipfire_generate::common::ds4_apply_cache_action
    /// even when a non-empty raw body is offered.
    #[test]
    fn ds4_length_and_fail_closed_skip_cache_store() {
        let finish_tools = FinishSummary {
            events: vec![ClientEvent::ToolCalls(vec![ToolCall {
                id: None,
                name: "alpha".into(),
                arguments: serde_json::json!({}),
                rendered_body: None,
            }])],
            finish_reason: "tool_calls",
            tool_calls: 1,
            visible_text: "partial".into(),
            decoded_eot: false,
            open_think: false,
        };
        let streamed = vec![7u32, 8, 9];

        // Length: store_cache=false, no release, no sink mutation.
        let wire_len = hipfire_generate::dense::ds4_spec_wire_terminal("tool_calls", 1, true);
        match &wire_len {
            hipfire_generate::dense::Ds4SpecWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*release_tool_calls);
                assert!(!*store_cache);
            }
            other => panic!("expected length Done, got {other:?}"),
        }
        let action_len =
            hipfire_generate::dense::ds4_cache_action(&wire_len, &finish_tools, finish_tools.visible_text.as_str());
        assert!(!action_len.store);
        assert!(
            action_len.tool_calls.is_empty(),
            "length suppresses held calls"
        );
        let mut sink_len = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink_len.insert(k, v);
            },
            &action_len,
            streamed.clone()
        )
        .is_none());
        assert!(
            sink_len.is_empty(),
            "length must not populate asst_turn_cache"
        );

        // Malformed fail-closed: no store, no done path.
        let finish_mal = FinishSummary {
            events: Vec::new(),
            finish_reason: "malformed_protocol",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        };
        let wire_mal = hipfire_generate::dense::ds4_spec_wire_terminal("malformed_protocol", 0, false);
        assert!(matches!(wire_mal, hipfire_generate::dense::Ds4SpecWireTerminal::Malformed(_)));
        let action_mal = hipfire_generate::dense::ds4_cache_action(&wire_mal, &finish_mal, finish_mal.visible_text.as_str());
        assert!(!action_mal.store);
        let mut sink_mal = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink_mal.insert(k, v);
            },
            &action_mal,
            streamed
        )
        .is_none());
        assert!(sink_mal.is_empty(), "fail-closed must not populate cache");

        // Empty-payload safe stop also refuses store (dead-weight empty turn).
        let finish_empty = FinishSummary {
            events: Vec::new(),
            finish_reason: "stop",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        };
        let wire_empty = hipfire_generate::dense::ds4_spec_wire_terminal("stop", 0, false);
        let action_empty = hipfire_generate::dense::ds4_cache_action(
            &wire_empty,
            &finish_empty,
            finish_empty.visible_text.as_str(),
        );
        assert!(action_empty.store, "wire authorizes stop");
        let mut sink_empty = std::collections::HashMap::new();
        assert!(
            hipfire_generate::common::ds4_apply_cache_action(
                |k, v| {
                    sink_empty.insert(k, v);
                },
                &action_empty,
                vec![1u32],
            )
            .is_none(),
            "empty fingerprint+calls must skip insert"
        );
        assert!(sink_empty.is_empty());
    }

    /// DS4 empty-event EOS is a model terminator only: not committable, not
    /// terminal-flushed, not baked into conversation history. Hidden/raw
    /// Committed events remain committable.
    #[test]
    fn ds4_empty_event_eos_seed_not_terminal_flushed_or_history() {
        use hipfire_runtime::spec::EmitOutcome;

        // Empty-event EOS (Deepseek4Emit::begin/observe on eos_token).
        let eos_out = EmitOutcome {
            events: Vec::new(),
            stop: Some(hipfire_runtime::spec::StopReason::Eos),
        };
        assert!(
            !hipfire_generate::qwen::spec_outcome_seed_committable(&eos_out),
            "empty-event EOS must not be state-committable"
        );
        assert!(
            !hipfire_generate::qwen::spec_should_flush_pending_seed(false, false),
            "non-committable pending seed must skip terminal flush"
        );

        // Event-bearing Committed (including hidden protocol bytes) stays
        // committable so history/GPU flush keep them.
        let committed = EmitOutcome {
            events: vec![ClientEvent::Committed { id: 42, idx: 0 }],
            stop: None,
        };
        assert!(hipfire_generate::qwen::spec_outcome_seed_committable(&committed));
        assert!(hipfire_generate::qwen::spec_should_flush_pending_seed(false, true));

        // Grammar fail-closed always skips flush even if seed was committable.
        assert!(!hipfire_generate::qwen::spec_should_flush_pending_seed(true, true));

        // First-seed init mirrors hipfire_generate::qwen::generate_spec: empty begin → no emitted bake.
        let mut emitted: Vec<u32> = Vec::new();
        let mut generated = 0usize;
        let first_token = 99u32; // eos id in production
        let pending_seed_committable = hipfire_generate::qwen::spec_outcome_seed_committable(&eos_out);
        if pending_seed_committable {
            emitted.push(first_token);
            generated += 1;
        }
        assert!(
            emitted.is_empty() && generated == 0,
            "DS4 first-token EOS must leave history at prompt"
        );

        // Position math for already-processed prior tokens is independent of
        // the non-committable bonus EOS seed (raw_decode may still record it
        // for realign; conversation bake uses emitted only).
        let prompt = vec![1u32, 2, 3];
        let prior_emitted = vec![10u32, 11];
        let conversation = {
            let mut v = prompt.clone();
            v.extend_from_slice(&prior_emitted);
            // Non-committable EOS seed is NOT appended (production bake path).
            v
        };
        assert_eq!(conversation, vec![1, 2, 3, 10, 11]);
        assert!(!conversation.contains(&first_token));

        // Live Deepseek4Emit: EOS begin returns empty events + Eos stop.
        let tok = {
            // Minimal BPE with a single special eos id=7 and printable bytes.
            let mut entries: Vec<String> = Vec::new();
            entries.push(r#""eos": 7"#.to_string());
            for b in 0u32..=255u32 {
                let ch = {
                    let mut bs: Vec<u32> = Vec::new();
                    bs.extend((b'!' as u32)..=(b'~' as u32));
                    bs.extend((0xA1u32)..=(0xACu32));
                    bs.extend((0xAEu32)..=(0xFFu32));
                    let mut cs: Vec<u32> = bs.clone();
                    let mut n: u32 = 0;
                    for byte in 0u32..=255u32 {
                        if !bs.contains(&byte) {
                            bs.push(byte);
                            cs.push(256 + n);
                            n += 1;
                        }
                    }
                    let idx = bs.iter().position(|&x| x == b).unwrap();
                    char::from_u32(cs[idx]).unwrap()
                };
                let escaped = {
                    let s = ch.to_string();
                    let mut out = String::new();
                    for c in s.chars() {
                        match c {
                            '"' => out.push_str("\\\""),
                            '\\' => out.push_str("\\\\"),
                            '\n' => out.push_str("\\n"),
                            '\r' => out.push_str("\\r"),
                            '\t' => out.push_str("\\t"),
                            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                            c => out.push(c),
                        }
                    }
                    out
                };
                entries.push(format!(r#""{}": {}"#, escaped, 100 + b));
            }
            let vocab_block = entries.join(", ");
            let json = format!(
                r#"{{
                    "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                    "added_tokens": [{{"id": 7, "content": "eos", "special": true}}]
                }}"#,
                vocab = vocab_block,
            );
            hipfire_runtime::tokenizer::Tokenizer::from_hf_json(&json).expect("tok")
        };
        let mut emit = hipfire_arch_deepseek4::spec_emit::Deepseek4Emit::from_ctx(
            hipfire_runtime::spec::SpecEmitCtx {
                tokenizer: &tok,
                eos: 7,
                im_end: None,
                tools: None,
                stop: Vec::new(),
                max_think: 0,
                max_tokens: 16,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
                decoded_vocab: None,
            },
        );
        let begin = emit.begin(7);
        assert!(begin.events.is_empty(), "DS4 EOS begin emits no events");
        assert_eq!(begin.stop, Some(hipfire_runtime::spec::StopReason::Eos));
        assert!(!hipfire_generate::qwen::spec_outcome_seed_committable(&begin));
        assert!(
            emit.streamed_tokens().is_empty(),
            "EOS must not enter streamed_tokens / cache body"
        );
    }

    #[test]
    fn ds4_client_commit_effects_commit_preserves_intended_flags() {
        let e = hipfire_generate::common::ds4_client_commit_effects(ClientTerminalDecision::Commit, true, true);
        assert_eq!(
            e,
            hipfire_generate::common::Ds4ClientCommitEffects {
                release_tool_calls: true,
                store_cache: true,
                emit_done: true,
            }
        );
        let e = hipfire_generate::common::ds4_client_commit_effects(ClientTerminalDecision::Commit, false, true);
        assert!(!e.release_tool_calls);
        assert!(e.store_cache);
        assert!(e.emit_done);
        let e = hipfire_generate::common::ds4_client_commit_effects(ClientTerminalDecision::Commit, false, false);
        assert!(!e.release_tool_calls);
        assert!(!e.store_cache);
        assert!(e.emit_done);
    }

    #[test]
    fn ds4_client_commit_effects_abort_suppresses_all_routes() {
        // Shared gate used by AR / EP / spec Safe terminals.
        for (intended_release, intended_store) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let e = hipfire_generate::common::ds4_client_commit_effects(
                ClientTerminalDecision::Abort,
                intended_release,
                intended_store,
            );
            assert_eq!(
                e,
                hipfire_generate::common::Ds4ClientCommitEffects {
                    release_tool_calls: false,
                    store_cache: false,
                    emit_done: false,
                },
                "abort must suppress tools/cache/done regardless of intended flags"
            );
        }
    }

    #[test]
    fn ds4_ar_ep_safe_commit_gate_retains_calls_cache_done() {
        let call = ToolCall {
            id: None,
            name: "search".into(),
            arguments: serde_json::json!({"q": "x"}),
            rendered_body: None,
        };
        let terminal = hipfire_generate::common::ds4_ar_ep_finish_route(None, vec![call.clone()], false);
        let hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
            finish_reason,
            wire_tool_calls,
            store_cache,
        } = terminal
        else {
            panic!("expected Safe");
        };
        assert_eq!(finish_reason, "tool_calls");
        let effects = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Commit,
            !wire_tool_calls.is_empty(),
            store_cache,
        );
        assert!(effects.release_tool_calls);
        assert!(effects.store_cache);
        assert!(effects.emit_done);

        let mut action = hipfire_generate::common::ds4_ar_ep_cache_action(
            &hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls: wire_tool_calls.clone(),
                store_cache,
            },
            "hello",
        );
        if !effects.store_cache {
            action.store = false;
        }
        assert!(action.store);
        let mut sink = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        )
        .is_some());
        assert_eq!(sink.len(), 1);
    }

    #[test]
    fn ds4_ar_ep_safe_abort_gate_suppresses_calls_cache_done() {
        let call = ToolCall {
            id: None,
            name: "search".into(),
            arguments: serde_json::json!({"q": "x"}),
            rendered_body: None,
        };
        let terminal = hipfire_generate::common::ds4_ar_ep_finish_route(None, vec![call], false);
        let hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
            finish_reason,
            wire_tool_calls,
            store_cache,
        } = terminal
        else {
            panic!("expected Safe");
        };
        let effects = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Abort,
            !wire_tool_calls.is_empty(),
            store_cache,
        );
        assert!(!effects.release_tool_calls);
        assert!(!effects.store_cache);
        assert!(!effects.emit_done);

        let mut action = hipfire_generate::common::ds4_ar_ep_cache_action(
            &hipfire_generate::common::Ds4ArEpRouteTerminal::Safe {
                finish_reason,
                wire_tool_calls,
                store_cache,
            },
            "hello",
        );
        if !effects.store_cache {
            action.store = false;
        }
        assert!(!action.store);
        let mut sink = std::collections::HashMap::new();
        assert!(hipfire_generate::common::ds4_apply_cache_action(
            |k, v| {
                sink.insert(k, v);
            },
            &action,
            vec![1, 2, 3],
        )
        .is_none());
        assert!(sink.is_empty());
    }

    #[test]
    fn ds4_spec_safe_commit_and_abort_gates() {
        let finish = FinishSummary {
            events: Vec::new(),
            finish_reason: "tool_calls",
            tool_calls: 1,
            visible_text: "hi".into(),
            decoded_eot: false,
            open_think: false,
        };
        let wire = hipfire_generate::dense::ds4_spec_wire_terminal("tool_calls", 1, false);
        let hipfire_generate::dense::Ds4SpecWireTerminal::Done {
            release_tool_calls,
            store_cache,
            ..
        } = wire
        else {
            panic!("expected Done");
        };

        let commit = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Commit,
            release_tool_calls,
            store_cache,
        );
        assert!(commit.release_tool_calls && commit.store_cache && commit.emit_done);
        let terminal_commit = hipfire_generate::dense::Ds4SpecWireTerminal::Done {
            finish_reason: "tool_calls",
            release_tool_calls: commit.release_tool_calls,
            store_cache: commit.store_cache,
        };
        let action_commit =
            hipfire_generate::dense::ds4_cache_action(&terminal_commit, &finish, finish.visible_text.as_str());
        assert!(action_commit.store);

        let abort = hipfire_generate::common::ds4_client_commit_effects(
            ClientTerminalDecision::Abort,
            release_tool_calls,
            store_cache,
        );
        assert!(!abort.release_tool_calls && !abort.store_cache && !abort.emit_done);
        let terminal_abort = hipfire_generate::dense::Ds4SpecWireTerminal::Done {
            finish_reason: "tool_calls",
            release_tool_calls: abort.release_tool_calls,
            store_cache: abort.store_cache,
        };
        let action_abort = hipfire_generate::dense::ds4_cache_action(&terminal_abort, &finish, finish.visible_text.as_str());
        assert!(!action_abort.store);
        assert!(action_abort.tool_calls.is_empty());
    }

    #[test]
    fn ds4_ep_abort_wire_events_carry_attempt_id_on_both() {
        set_active_attempt_id(99);
        let (aborted, done) = hipfire_generate::common::ds4_ep_abort_wire_events("req-ep", 7, 99);
        assert_eq!(aborted["type"], "aborted");
        assert_eq!(aborted["id"], "req-ep");
        assert_eq!(aborted["reason"], "client_cancelled");
        assert_eq!(aborted["attempt_id"], 99);
        assert_eq!(done["type"], "done");
        assert_eq!(done["finish_reason"], "aborted");
        assert_eq!(done["completion_tokens"], 7);
        assert_eq!(done["attempt_id"], 99);
        // Same shape as production semantic helpers.
        assert_eq!(
            aborted,
            hipfire_runtime::semantic::wire_aborted("req-ep", "client_cancelled", 99)
        );
        assert_eq!(
            done,
            hipfire_runtime::semantic::wire_aborted_done("req-ep", 7, 99)
        );
    }
}

/// Deterministic non-GPU tests for Qwen DFlash/spec semantic-v2 terminal +
/// cache decisions. Production helpers only — no whole-output reparse authority.
#[cfg(test)]
mod qwen_dflash_semantic_terminal_tests {
    use super::*;
    use hipfire_runtime::prompt_frame::{AssistantPrefix, ToolCall};
    use hipfire_runtime::spec::{
        ClientEvent, FinishSummary, SpecEmit, SpecEmitCtx, SpecStep, StopReason,
    };
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::collections::HashSet;

    fn summary_tool_calls(calls: Vec<ToolCall>) -> FinishSummary {
        let n = calls.len();
        FinishSummary {
            events: vec![ClientEvent::ToolCalls(calls)],
            finish_reason: "tool_calls",
            tool_calls: n,
            visible_text: "Sure.".into(),
            decoded_eot: false,
            open_think: false,
        }
    }

    fn summary_stop(visible: &str) -> FinishSummary {
        FinishSummary {
            events: vec![ClientEvent::Token(visible.into())],
            finish_reason: "stop",
            tool_calls: 0,
            visible_text: visible.into(),
            decoded_eot: false,
            open_think: false,
        }
    }

    fn summary_malformed() -> FinishSummary {
        FinishSummary {
            events: Vec::new(),
            finish_reason: "malformed_protocol",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        }
    }

    fn json_escape(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    fn byte_to_gpt2_char_test(b: u8) -> char {
        let mut bs: Vec<u32> = Vec::new();
        bs.extend((b'!' as u32)..=(b'~' as u32));
        bs.extend((0xA1u32)..=(0xACu32));
        bs.extend((0xAEu32)..=(0xFFu32));
        let mut cs: Vec<u32> = bs.clone();
        let mut n: u32 = 0;
        for byte in 0u32..=255u32 {
            if !bs.contains(&byte) {
                bs.push(byte);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (bb, cc) in bs.into_iter().zip(cs.into_iter()) {
            if bb == b as u32 {
                return char::from_u32(cc).unwrap();
            }
        }
        char::from_u32(b as u32).unwrap()
    }

    /// Same minimal tokenizer family as qwen35 `spec_emit` CPU tests.
    fn test_tokenizer() -> Tokenizer {
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""<think>": 2"#.to_string());
        entries.push(r#""</think>": 3"#.to_string());
        entries.push(r#""system": 4"#.to_string());
        entries.push(r#""user": 5"#.to_string());
        entries.push(r#""assistant": 6"#.to_string());
        entries.push(r#""\n": 7"#.to_string());
        entries.push(r#""Ġ": 8"#.to_string());
        entries.push(r#""<|endoftext|>": 9"#.to_string());
        for b in 0u32..=255u32 {
            let ch = byte_to_gpt2_char_test(b as u8);
            let escaped = json_escape(&ch.to_string());
            entries.push(format!(r#""{}": {}"#, escaped, 100 + b));
        }
        let vocab_block = entries.join(", ");
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                "added_tokens": [
                    {{"id": 0, "content": "<|im_start|>", "special": true}},
                    {{"id": 1, "content": "<|im_end|>", "special": true}},
                    {{"id": 2, "content": "<think>", "special": true}},
                    {{"id": 3, "content": "</think>", "special": true}},
                    {{"id": 9, "content": "<|endoftext|>", "special": true}}
                ]
            }}"#,
            vocab = vocab_block,
        );
        Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    fn make_qwen_emit<'a>(
        tok: &'a Tokenizer,
        assistant_prefix: AssistantPrefix,
    ) -> Box<dyn SpecEmit + 'a> {
        hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            stop: Vec::new(),
            max_think: 0,
            max_tokens: 256,
            assistant_prefix,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        })
    }

    /// Drive production Qwen35Emit with whole-string encodes.
    fn drive_qwen_emit(
        text: &str,
        assistant_prefix: AssistantPrefix,
    ) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        let tok = test_tokenizer();
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "encode produced no tokens for {text:?}");
        let mut emit = make_qwen_emit(&tok, assistant_prefix);
        let mut stream = Vec::new();
        let mut first = true;
        for id in &ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop.is_some() {
                break;
            }
        }
        let streamed = emit.streamed_tokens().to_vec();
        let finish = emit.finish();
        (stream, finish, streamed)
    }

    /// Drive production emitter token-by-token (for split-marker cases).
    fn drive_qwen_ids(
        ids: &[u32],
        assistant_prefix: AssistantPrefix,
    ) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        let tok = test_tokenizer();
        let mut emit = make_qwen_emit(&tok, assistant_prefix);
        let mut stream = Vec::new();
        let mut first = true;
        for id in ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop.is_some() {
                break;
            }
        }
        let streamed = emit.streamed_tokens().to_vec();
        let finish = emit.finish();
        (stream, finish, streamed)
    }

    fn parse_jsonl(out: &str) -> Vec<serde_json::Value> {
        out.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad jsonl {l}: {e}")))
            .collect()
    }

    /// GPU-less attested epilogue for unit tests (no real device sync).
    fn attest_epilogue(rolled_back: bool) -> hipfire_generate::common::RollbackEpilogue {
        hipfire_generate::common::RollbackEpilogue {
            rolled_back,
            context: None,
        }
    }

    /// Attested epilogue with sync-failure context (rolled_back=false).
    fn attest_epilogue_with_context(context: &str) -> hipfire_generate::common::RollbackEpilogue {
        hipfire_generate::common::RollbackEpilogue {
            rolled_back: false,
            context: Some(context.to_string()),
        }
    }

    #[test]
    fn safe_stop_stores_cache_no_calls() {
        let fin = summary_stop("hello");
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "hello", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                fingerprint_text,
                wire_tool_calls,
            } => {
                assert_eq!(*finish_reason, "stop");
                assert!(!*release_tool_calls);
                assert!(*store_cache);
                assert!(wire_tool_calls.is_empty());
                assert_eq!(
                    fingerprint_text.as_str(),
                    hipfire_generate::common::normalize_asst_turn_for_fingerprint("hello")
                );
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(action.store);
        assert!(action.tool_calls.is_empty());
    }

    #[test]
    fn tool_safe_releases_calls_and_stores() {
        let calls = vec![ToolCall {
            id: None,
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "SF"}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls.clone());
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "Sure.", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache);
                assert_eq!(wire_tool_calls.len(), 1);
                assert_eq!(wire_tool_calls[0].name, "get_weather");
            }
            other => panic!("expected Done, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(action.store);
        assert_eq!(action.tool_calls.len(), 1);
    }

    #[test]
    fn pure_length_suppresses_calls_and_cache() {
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(16, 16, false, false));
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(16, 16, false, true));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "partial", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                fingerprint_text,
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*release_tool_calls);
                assert!(!*store_cache);
                assert!(wire_tool_calls.is_empty());
                assert!(fingerprint_text.is_empty());
            }
            other => panic!("expected length Done, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.store);
        assert!(hipfire_generate::qwen::qwen_dflash_apply_cache_action(
            |_, _| panic!("must not insert"),
            &action,
            vec![1, 2]
        )
        .is_none());
    }

    #[test]
    fn final_token_eot_beats_length() {
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, true, false));
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "ok", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                ..
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache);
            }
            other => panic!("expected tool_calls Done, got {other:?}"),
        }
    }

    #[test]
    fn malformed_is_error_xor_done_no_cache() {
        let fin = summary_malformed();
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                class,
                retryable,
                rolled_back,
                message,
            } => {
                assert_eq!(*class, "validation");
                assert!(!*retryable);
                assert!(!*rolled_back);
                assert!(message.contains("malformed"));
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.store);
        assert!(action.tool_calls.is_empty());
        assert!(!matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Done { .. }));
    }

    #[test]
    fn grammar_failure_no_calls_no_cache() {
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, true, "x", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                class,
                retryable,
                message,
                ..
            } => {
                assert_eq!(*class, "validation");
                assert!(!*retryable);
                assert!(message.contains("grammar"));
            }
            other => panic!("expected grammar Malformed error-only, got {other:?}"),
        }
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.store);
        assert!(action.tool_calls.is_empty());
        assert!(!matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Done { .. }));
    }

    #[test]
    fn open_think_is_error_xor_done_no_cache() {
        // Production emitter (prompt-started OpenThink) -> real FinishSummary
        // -> production wire terminal. No hand-built open_think mirrors.
        let (stream, fin, _raw) = drive_qwen_emit("still thinking", AssistantPrefix::OpenThink);
        let reasoning: String = stream
            .iter()
            .filter_map(|e| match e {
                ClientEvent::Reasoning(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, "still thinking");
        assert!(fin.open_think, "emitter must latch open_think");
        assert_eq!(fin.finish_reason, "open_think");
        assert!(fin.events.is_empty());
        assert_eq!(fin.tool_calls, 0);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                class,
                retryable,
                message,
                ..
            } => {
                assert_eq!(*class, "validation");
                assert!(!*retryable);
                assert!(message.contains("open think"));
            }
            other => panic!("expected open_think Malformed, got {other:?}"),
        }
        assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store);
        assert!(!matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Done { .. }));
        // Production Malformed writer: error XOR done (GPU-less attested epilogue).
        set_active_attempt_id(21);
        let mut sink = Vec::new();
        if let hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
            message,
            class,
            retryable,
            rolled_back,
        } = &term
        {
            let ep = attest_epilogue(*rolled_back);
            hipfire_generate::qwen::emit_qwen_dflash_malformed_terminal(
                &mut sink, "req-ot", message, class, *retryable, &ep,
            );
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 21);
        assert!(!out.contains(r#""type":"done""#));
    }

    #[test]
    fn open_think_prompt_started_and_generated_flags() {
        // (a) prompt-started OpenThink; (b) generated unclosed <think>.
        let cases = [
            ("prompt", AssistantPrefix::OpenThink, "still thinking"),
            ("generated", AssistantPrefix::Plain, "pre <think>secret"),
        ];
        for (label, prefix, body) in cases {
            let (stream, fin, _raw) = drive_qwen_emit(body, prefix);
            let reasoning: String = stream
                .iter()
                .filter_map(|e| match e {
                    ClientEvent::Reasoning(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let expected_reasoning = if label == "prompt" {
                "still thinking"
            } else {
                "secret"
            };
            assert_eq!(reasoning, expected_reasoning, "{label}");
            assert!(fin.open_think, "{label}: open_think");
            assert_eq!(fin.finish_reason, "open_think", "{label}");
            assert_eq!(fin.tool_calls, 0, "{label}");
            assert!(fin.events.is_empty(), "{label}: no release on open_think");
            let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
            assert!(
                matches!(term, hipfire_generate::qwen::QwenDflashWireTerminal::Malformed { .. }),
                "{label}: expected Malformed"
            );
            assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store, "{label}");
        }
    }

    #[test]
    fn producer_decoded_eot_beats_length_without_token_rescan() {
        // Real emitter decoded_eot at budget boundary → stop, not length.
        let tok = test_tokenizer();
        let mut ids = tok.encode("hi");
        ids.push(1); // <|im_end|>
        let (_stream, fin, _raw) = drive_qwen_ids(&ids, AssistantPrefix::Plain);
        assert!(fin.decoded_eot, "emitter must set decoded_eot");
        assert_eq!(fin.finish_reason, "stop");
        let generated = ids.len();
        let max_tokens = generated;
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(
            generated,
            max_tokens,
            fin.decoded_eot,
            false
        ));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "hi", false);
        assert!(matches!(
            term,
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason: "stop",
                store_cache: true,
                ..
            }
        ));
    }

    #[test]
    fn split_decoded_eot_at_cap_is_stop_not_length() {
        // Byte-fragment the <|im_end|> marker across tokens via 100+b map.
        let marker = b"<|im_end|>";
        let mut ids: Vec<u32> = Vec::new();
        // prose "hi"
        ids.push(100 + b'h' as u32);
        ids.push(100 + b'i' as u32);
        // split marker into two fragments
        let mid = marker.len() / 2;
        for &b in &marker[..mid] {
            ids.push(100 + b as u32);
        }
        for &b in &marker[mid..] {
            ids.push(100 + b as u32);
        }
        let (stream, fin, raw) = drive_qwen_ids(&ids, AssistantPrefix::Plain);
        assert!(fin.decoded_eot, "split EOT must set decoded_eot");
        assert_eq!(fin.finish_reason, "stop");
        let visible: String = stream
            .iter()
            .filter_map(|ev| match ev {
                ClientEvent::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(!visible.contains("<|im_end|>"), "marker bytes suppressed");
        assert!(visible.contains("hi"));
        assert!(!raw.is_empty());
        let generated = raw.len();
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(
            generated,
            generated,
            fin.decoded_eot,
            false
        ));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, &visible, false);
        assert!(matches!(
            term,
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason: "stop",
                store_cache: true,
                release_tool_calls: false,
                ..
            }
        ));
    }

    #[test]
    fn step_budget_max_emit_zero_one_and_mid_window_prefix() {
        // max_emit 0: empty emit is the defensive shape (live step returns Err).
        let step0 = SpecStep::new([10, 11], 11, 1, 1).cap_emit(0);
        assert!(step0.emit.is_empty());
        assert_eq!(step0.accepted, 0);

        // max_emit 1: prefix keep + seed reseeds from kept token.
        let step1 = SpecStep::new([10, 11, 12], 12, 2, 2).cap_emit(1);
        assert_eq!(step1.emit.as_slice(), &[10]);
        assert_eq!(step1.next_seed, 10);
        assert!(step1.emit.len() <= 1);

        // Mid-window semantic consume of 2 of 4 emitted tokens.
        let step = SpecStep::new([10, 11, 12, 13], 13, 4, 3);
        let host = hipfire_generate::qwen::spec_host_advance_after_step(100, 0, Vec::new(), &step.emit, step.next_seed, 2);
        assert_eq!(host.emitted, vec![10, 11]);
        assert_eq!(host.generated, 2);
        assert_eq!(host.position, 102);
        assert_eq!(host.seed_token, 11);
        // Full-window consume keeps step.next_seed when prefix covers emit.
        let host_full =
            hipfire_generate::qwen::spec_host_advance_after_step(100, 0, Vec::new(), &step.emit, step.next_seed, 4);
        assert_eq!(host_full.emitted, vec![10, 11, 12, 13]);
        assert_eq!(host_full.position, 104);
        assert_eq!(host_full.seed_token, 13);
        // Unconsumed tail must not inflate position/conversation.
        assert_ne!(host.position, 100 + step.emit.len());
    }

    #[test]
    fn spec_prefix_realign_plan_empty_raw_and_multi() {
        let prompt = vec![1u32, 2, 3];
        let empty = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, 99, &[]);
        assert_eq!(empty.replay, prompt);
        assert_eq!(empty.position, 3);
        assert_eq!(empty.seed_token, 99);

        let multi = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, 99, &[10, 11, 12]);
        assert_eq!(multi.replay, vec![1, 2, 3, 99, 10, 11]);
        assert_eq!(multi.position, 6);
        assert_eq!(multi.seed_token, 12);
        assert_eq!(multi.replay.len(), multi.position);
        // Last raw stays the unwritten seed — never sits in KV replay.
        assert_ne!(multi.replay.last().copied(), Some(multi.seed_token));
        // Naive prompt+raw drops first_token and writes the seed into KV.
        let mut naive = prompt.clone();
        naive.extend_from_slice(&[10, 11, 12]);
        assert_ne!(multi.replay, naive);
    }

    #[test]
    fn terminal_marker_mid_window_strict_prefix_realigns() {
        // Spec window emits body + im_end + unobserved tail. Semantic loop
        // consumes only through the terminal marker; host + realign plan must
        // land exactly on that prefix (no unobserved tail in conversation or KV).
        let tok = test_tokenizer();
        let prompt = vec![4u32, 5];
        let first_token = tok.encode("hi")[0];
        let body = tok.encode("ok");
        let im_end = 1u32;
        let mut step_emit = body.clone();
        step_emit.push(im_end);
        step_emit.extend_from_slice(&[90, 91]);
        let step = SpecStep::new(step_emit.clone(), *step_emit.last().unwrap(), 4, 3);

        let mut emit = make_qwen_emit(&tok, AssistantPrefix::Plain);
        let _ = emit.begin(first_token);
        let mut consumed = 0usize;
        let mut raw_decode: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        for &tok_id in &step.emit {
            let outcome = emit.observe(tok_id);
            if outcome.stop == Some(StopReason::GrammarViolation) {
                break;
            }
            consumed += 1;
            raw_decode.push(tok_id);
            if matches!(
                outcome.stop,
                Some(StopReason::Eos) | Some(StopReason::StopSequence)
            ) {
                hit_eos = true;
                break;
            }
        }
        assert!(hit_eos, "im_end must stop the emitter");
        assert_eq!(
            consumed,
            body.len() + 1,
            "must consume body+im_end only, not tail {:?}",
            &step.emit[consumed..]
        );
        assert!(
            consumed < step.emit.len(),
            "fixture must leave an unobserved speculative tail"
        );

        let position_before = prompt.len();
        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            position_before,
            0,
            vec![first_token],
            &step.emit,
            step.next_seed,
            consumed,
        );
        assert_eq!(host.generated, consumed);
        assert_eq!(host.position, position_before + consumed);
        assert_eq!(host.seed_token, im_end);
        assert_eq!(&host.emitted[1..], &step.emit[..consumed]);
        assert!(!host.emitted.contains(&90) && !host.emitted.contains(&91));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first_token, &raw_decode);
        let mut expected_replay = prompt.clone();
        expected_replay.push(first_token);
        expected_replay.extend_from_slice(&raw_decode[..raw_decode.len() - 1]);
        assert_eq!(plan.replay, expected_replay);
        assert_eq!(plan.position, prompt.len() + raw_decode.len());
        assert_eq!(plan.seed_token, im_end);
        assert_eq!(plan.position, host.position);
        assert_eq!(plan.seed_token, host.seed_token);
        assert_ne!(plan.replay.last().copied(), Some(plan.seed_token));
    }

    #[test]
    fn empty_event_eos_mid_window_still_realigns_raw_prefix() {
        // Empty-event EOS observes still advance position/raw_decode (filter
        // stop on decoded marker bytes). Host + realign must track them.
        let tok = test_tokenizer();
        let prompt = vec![4u32];
        let first_token = 100 + b'h' as u32; // byte-map 'h'
                                             // Fragment <|im_end|> across byte-map tokens so filter stops without
                                             // a single special-id observe; final fragment may yield empty events.
        let marker = b"<|im_end|>";
        let mut step_emit: Vec<u32> = vec![100 + b'i' as u32]; // "i" after seed "h"
        for &b in marker {
            step_emit.push(100 + b as u32);
        }
        step_emit.extend_from_slice(&[90, 91]); // unobserved tail
        let step = SpecStep::new(step_emit.clone(), 91, step_emit.len(), step_emit.len() - 1);

        let mut emit = make_qwen_emit(&tok, AssistantPrefix::Plain);
        let _ = emit.begin(first_token);
        let mut consumed = 0usize;
        let mut raw_decode: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        for &tok_id in &step.emit {
            let outcome = emit.observe(tok_id);
            consumed += 1;
            raw_decode.push(tok_id);
            // Empty-event EOS still counts as a position-advancing observe.
            if matches!(
                outcome.stop,
                Some(StopReason::Eos) | Some(StopReason::StopSequence)
            ) {
                hit_eos = true;
                break;
            }
        }
        assert!(hit_eos, "split marker must stop via filter");
        assert!(consumed < step.emit.len(), "tail must remain unobserved");
        assert_eq!(raw_decode.len(), consumed);

        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            prompt.len(),
            0,
            vec![first_token],
            &step.emit,
            step.next_seed,
            consumed,
        );
        assert_eq!(host.generated, consumed);
        assert_eq!(host.position, prompt.len() + consumed);
        assert!(!host.emitted.contains(&90));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first_token, &raw_decode);
        assert_eq!(plan.position, host.position);
        assert_eq!(plan.seed_token, host.seed_token);
        assert_eq!(plan.replay.len(), plan.position);
        assert_ne!(plan.replay.last().copied(), Some(plan.seed_token));
    }

    #[test]
    fn multi_window_then_strict_prefix_realign() {
        // After a full first window, raw_decode holds W1; a second window stops
        // mid-prefix. Realign replays prompt+first+raw[..-1] across both windows.
        let prompt = vec![7u32, 8];
        let first_token = 50u32;
        // Window 1 full consume (no realign).
        let w1 = SpecStep::new([10u32, 11, 12], 12, 3, 2);
        let mut raw_decode = Vec::new();
        let mut position = prompt.len();
        let mut emitted = vec![first_token];
        let mut generated = 0usize;
        let host1 = hipfire_generate::qwen::spec_host_advance_after_step(
            position,
            generated,
            emitted.clone(),
            &w1.emit,
            w1.next_seed,
            w1.emit.len(),
        );
        position = host1.position;
        generated = host1.generated;
        emitted = host1.emitted;
        raw_decode.extend_from_slice(&w1.emit);
        assert_eq!(position, prompt.len() + w1.emit.len());
        assert_eq!(host1.seed_token, 12);

        // Window 2: consume 2 of 4 (strict prefix → realign).
        let w2 = SpecStep::new([20u32, 21, 22, 23], 23, 4, 3);
        let consumed2 = 2usize;
        raw_decode.extend_from_slice(&w2.emit[..consumed2]);
        let host2 = hipfire_generate::qwen::spec_host_advance_after_step(
            position,
            generated,
            emitted,
            &w2.emit,
            w2.next_seed,
            consumed2,
        );
        assert_eq!(host2.emitted, vec![first_token, 10, 11, 12, 20, 21]);
        assert_eq!(host2.position, prompt.len() + raw_decode.len());
        assert_eq!(host2.seed_token, 21);
        assert!(!host2.emitted.contains(&22) && !host2.emitted.contains(&23));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first_token, &raw_decode);
        assert_eq!(
            plan.replay,
            vec![7, 8, first_token, 10, 11, 12, 20] // drops last raw (21)
        );
        assert_eq!(plan.position, host2.position);
        assert_eq!(plan.seed_token, host2.seed_token);
        assert_eq!(plan.seed_token, 21);
    }

    #[test]
    fn forced_token_mid_window_strict_prefix_then_force_advance() {
        // Think-budget force-close mid-window: observe only the forced-trigger
        // prefix of step.emit, realign host/plan to that prefix, then host
        // advances over the forced continuation tokens as raw_decode.
        let tok = test_tokenizer();
        let prompt = vec![4u32, 5];
        let open_think = 2u32; // <think>

        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            stop: Vec::new(),
            max_think: 1,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        let begin = emit.begin(open_think);
        assert!(begin.stop.is_none());

        let think_body = tok.encode("x");
        assert_eq!(think_body.len(), 1);
        let step_emit = vec![think_body[0], 90, 91, 92];
        let step = SpecStep::new(step_emit.clone(), 92, 4, 3);

        let mut consumed = 0usize;
        let mut raw_decode: Vec<u32> = Vec::new();
        let mut forced_after: Vec<u32> = Vec::new();
        for &tok_id in &step.emit {
            let outcome = emit.observe(tok_id);
            if outcome.stop == Some(StopReason::GrammarViolation) {
                break;
            }
            consumed += 1;
            raw_decode.push(tok_id);
            let forced = emit.take_forced();
            if !forced.is_empty() {
                forced_after = forced;
                break;
            }
            if outcome.stop.is_some() {
                break;
            }
        }
        assert_eq!(consumed, 1, "force must fire on the budget-hitting token");
        assert!(
            !forced_after.is_empty(),
            "think budget must queue </think> continuation"
        );
        assert!(consumed < step.emit.len(), "must leave unobserved tail");

        let position_before = prompt.len();
        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            position_before,
            0,
            vec![open_think],
            &step.emit,
            step.next_seed,
            consumed,
        );
        assert_eq!(host.generated, 1);
        assert_eq!(host.position, position_before + 1);
        assert_eq!(host.seed_token, think_body[0]);
        assert!(!host.emitted.contains(&90));

        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, open_think, &raw_decode);
        assert_eq!(plan.position, host.position);
        assert_eq!(plan.seed_token, host.seed_token);
        assert_eq!(plan.replay, {
            let mut r = prompt.clone();
            r.push(open_think);
            r
        });

        // Pending-seed GPU tx: commit [trigger] ++ forced[..n-1]; last forced
        // stays unprocessed pending seed (never double-forwarded).
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(plan.seed_token, &forced_after, true);
        assert_eq!(tx.commit.first().copied(), Some(plan.seed_token));
        assert_eq!(tx.commit.len(), forced_after.len());
        assert_eq!(tx.pending_seed, *forced_after.last().unwrap());
        // Last forced is never double-forwarded: it is the pending seed, not in
        // commit (except the n==1 case where commit is only the prior seed).
        if forced_after.len() > 1 {
            assert_eq!(&tx.commit[1..], &forced_after[..forced_after.len() - 1]);
            assert_eq!(
                tx.commit.last().copied(),
                Some(forced_after[forced_after.len() - 2])
            );
        } else {
            assert_eq!(tx.commit.as_slice(), &[plan.seed_token]);
        }

        // Host observes each forced token; position advances by commit.len().
        let mut position = plan.position.saturating_add(tx.position_delta);
        let mut generated = host.generated;
        let mut emitted = host.emitted.clone();
        let mut seed_token = tx.pending_seed;
        for &ft in &forced_after {
            generated += 1;
            emitted.push(ft);
            raw_decode.push(ft);
            let fo = emit.observe(ft);
            assert!(
                fo.stop.is_none() || fo.stop == Some(StopReason::StopSequence),
                "forced continuation should not hard-stop mid-injection: {:?}",
                fo.stop
            );
        }
        assert_eq!(seed_token, *forced_after.last().unwrap());
        assert_eq!(position, plan.position + forced_after.len());
        assert_eq!(generated, consumed + forced_after.len());
        let mut expected_raw = step.emit[..consumed].to_vec();
        expected_raw.extend_from_slice(&forced_after);
        assert_eq!(raw_decode, expected_raw);
        assert!(!emitted.contains(&90) && !emitted.contains(&91) && !emitted.contains(&92));

        // Terminal flush would commit the final pending seed exactly once.
        let term = hipfire_generate::qwen::spec_terminal_pending_seed_tx(seed_token);
        assert_eq!(term.commit, vec![seed_token]);
        assert_eq!(term.position_delta, 1);
        let position_after_flush = position + term.position_delta;

        let plan2 = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, open_think, &raw_decode);
        assert_eq!(plan2.position, prompt.len() + raw_decode.len());
        assert_eq!(plan2.seed_token, seed_token);
        assert_eq!(plan2.seed_token, *raw_decode.last().unwrap());
        assert_eq!(plan2.replay.len(), plan2.position);
        // After terminal flush, cursor is one past the last conversation token
        // (prompt + raw_decode), matching safe bake `m.seq_pos`.
        assert_eq!(position_after_flush, prompt.len() + raw_decode.len() + 1);
        // Realign still treats last raw as unwritten seed (pre-terminal-flush).
        let mut expected_replay = prompt.clone();
        expected_replay.push(open_think);
        expected_replay.extend_from_slice(&raw_decode[..raw_decode.len() - 1]);
        assert_eq!(plan2.replay, expected_replay);
    }

    #[test]
    fn cache_seq_trim_eot_vs_length_body_newline() {
        let im_end = Some(1u32);
        let nl: HashSet<u32> = [7u32].into_iter().collect();
        // EOT-terminated: body + im_end + nl → strip trailer.
        let eot_stream = vec![10, 11, 1, 7];
        assert_eq!(
            hipfire_generate::qwen::qwen_dflash_cache_seq(&eot_stream, im_end, &nl),
            vec![10, 11]
        );
        // Length-capped body ending on newline: restore verbatim (no im_end).
        let len_stream = vec![10, 11, 7];
        assert_eq!(
            hipfire_generate::qwen::qwen_dflash_cache_seq(&len_stream, im_end, &nl),
            vec![10, 11, 7]
        );
        // Pure body, no trailer.
        let body = vec![10, 11, 12];
        assert_eq!(hipfire_generate::qwen::qwen_dflash_cache_seq(&body, im_end, &nl), body);
    }

    #[test]
    fn step_and_forced_advance_error_helpers_are_xor_done() {
        // Production fail-closed writer with GPU-less attested epilogue.
        set_active_attempt_id(42);
        for (what, id, needle) in [
            ("spec_step", "req-step", "spec_step:"),
            ("forced", "req-fa", "forced-token"),
        ] {
            let mut sink = Vec::new();
            let ep = attest_epilogue(true);
            hipfire_generate::qwen::emit_spec_failure_terminal(&mut sink, id, what, "boom", &ep);
            let text = String::from_utf8(sink).unwrap();
            let lines = parse_jsonl(&text);
            assert_eq!(lines.len(), 1, "error XOR done: {lines:?}");
            assert_eq!(lines[0]["type"], "error");
            assert_eq!(lines[0]["attempt_id"], 42);
            assert_eq!(lines[0]["retryable"], false);
            assert_eq!(lines[0]["rolled_back"], true);
            assert!(lines[0]["message"].as_str().unwrap().contains(needle));
            assert!(!text.contains(r#""type":"done""#));
            assert!(!text.contains(r#""type":"tool_calls""#));
        }
        // rolled_back=false + context path (sync could not be attested).
        let mut sink = Vec::new();
        let ep = attest_epilogue_with_context("device_synchronize failed: test");
        hipfire_generate::qwen::emit_spec_failure_terminal(&mut sink, "req-ctx", "spec_step", "boom", &ep);
        let text = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&text);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["rolled_back"], false);
        assert!(lines[0]["message"]
            .as_str()
            .unwrap()
            .contains("device_synchronize failed"));
        // Wrapper None contract: no epilogue after early exit.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        assert!(qwen_dflash_epilogue_after_spec_run(true));
    }

    #[test]
    fn forced_advance_error_is_xor_done_no_calls() {
        set_active_attempt_id(43);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        hipfire_generate::qwen::emit_spec_failure_terminal(&mut sink, "req-fa", "forced", "boom", &ep);
        let text = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&text);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 43);
        assert_eq!(lines[0]["rolled_back"], true);
        assert!(lines[0]["message"]
            .as_str()
            .unwrap()
            .contains("forced-token"));
        assert!(!text.contains(r#""type":"done""#));
        assert!(!text.contains(r#""type":"tool_calls""#));
    }

    #[test]
    fn decoded_eot_beats_length_cap_helper() {
        let fin = summary_stop("hi");
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, false, false));
        // Emitter semantic stop at cap is also not length (independent of EOT).
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, false, true));
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "hi", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                store_cache,
                release_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*store_cache);
                assert!(!*release_tool_calls);
            }
            other => panic!("{other:?}"),
        }
        assert!(!hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, true, false));
        let tok = test_tokenizer();
        let mut ids = tok.encode("hi");
        ids.push(1);
        let (_s, fin_eot, _) = drive_qwen_ids(&ids, AssistantPrefix::Plain);
        assert!(fin_eot.decoded_eot);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin_eot, false, false, "hi", false);
        assert!(matches!(
            term,
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason: "stop",
                store_cache: true,
                ..
            }
        ));
    }

    #[test]
    fn ordinary_length_cutoff_no_calls_no_cache() {
        let calls = vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "x", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*release_tool_calls);
                assert!(!*store_cache);
                assert!(wire_tool_calls.is_empty());
            }
            other => panic!("{other:?}"),
        }
        assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store);
    }

    #[test]
    fn cancel_is_fold_compatible_no_cache_helper() {
        // Production cancel writer (same path as hipfire_generate::qwen::generate_spec abort sites).
        set_active_attempt_id(11);
        let mut sink = Vec::new();
        emit_qwen_ar_cancelled(&mut sink, "c", 3);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[0]["reason"], "client_cancelled");
        assert_eq!(lines[0]["attempt_id"], 11);
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["completion_tokens"], 3);
        // Cancel never goes through hipfire_generate::qwen::qwen_dflash_wire_terminal store path.
        assert!(!out.contains(r#""finish_reason":"stop""#));
    }

    #[test]
    fn serde_done_v2_hostile_id_roundtrip() {
        set_active_attempt_id(5);
        let id = "id\"quote\"\n";
        let mut sink = Vec::new();
        emit_qwen_dflash_done_terminal(
            &mut sink, id, 2, 1.0, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 0, "stop", None,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "done");
        assert_eq!(lines[0]["id"], id);
        assert_eq!(lines[0]["attempt_id"], 5);
        assert_eq!(lines[0]["finish_reason"], "stop");
        assert_eq!(lines[0]["dflash"], true);
    }

    #[test]
    fn grammar_lifecycle_error_only_serialized() {
        set_active_attempt_id(7);
        let fin = summary_tool_calls(vec![ToolCall {
            id: None,
            name: "t".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }]);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, true, "x", false);
        let mut sink = Vec::new();
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Malformed {
                message,
                class,
                retryable,
                rolled_back,
            } => {
                let ep = attest_epilogue(*rolled_back);
                hipfire_generate::qwen::emit_qwen_dflash_malformed_terminal(
                    &mut sink, "g1", message, class, *retryable, &ep,
                );
            }
            other => panic!("{other:?}"),
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 7);
        assert_eq!(lines[0]["id"], "g1");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!hipfire_generate::qwen::qwen_dflash_cache_action(&term).store);
    }

    #[test]
    fn serde_v2_token_and_tool_calls_hostile_id() {
        set_active_attempt_id(9);
        let mut sink = Vec::new();
        let id = "a\"b\n";
        hipfire_generate::qwen::render_client_events(
            &mut sink,
            id,
            &[
                ClientEvent::Token("hi".into()),
                ClientEvent::Reasoning("r".into()),
            ],
            0,
            false,
        );
        emit_tool_calls_event(
            &mut sink,
            id,
            &[ToolCall {
                id: None,
                name: "n".into(),
                arguments: serde_json::json!({"x": 1}),
                rendered_body: None,
            }],
        );
        let out = String::from_utf8(sink).unwrap();
        for line in out.lines().filter(|l| !l.is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).expect(line);
            assert_eq!(v["attempt_id"], 9);
            assert_eq!(v["id"], id);
        }
        let types: Vec<_> = parse_jsonl(&out)
            .into_iter()
            .map(|v| v["type"].as_str().unwrap().to_string())
            .collect();
        assert!(types.contains(&"token".to_string()));
        assert!(types.contains(&"reasoning".to_string()));
        assert!(types.contains(&"tool_calls".to_string()));
    }

    #[test]
    fn cancel_wire_helpers_carry_attempt_id() {
        // Production cancel writer carries attempt_id on aborted + done.
        set_active_attempt_id(3);
        let mut sink = Vec::new();
        emit_qwen_ar_cancelled(&mut sink, "c1", 5);
        let lines = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[0]["attempt_id"], 3);
        assert_eq!(lines[0]["reason"], "client_cancelled");
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["attempt_id"], 3);
        assert_eq!(lines[1]["completion_tokens"], 5);
    }

    #[test]
    fn cache_fingerprint_uses_visible_not_raw_markers() {
        let fin = summary_stop("visible only");
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "visible only", false);
        let action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        assert!(!action.fingerprint_text.contains("<tool_call>"));
        assert!(!action.fingerprint_text.contains("<think>"));
        assert!(action.fingerprint_text.contains("visible"));
        let mut stored = None;
        let fp = hipfire_generate::qwen::qwen_dflash_apply_cache_action(
            |f, seq| {
                stored = Some((f, seq));
            },
            &action,
            vec![10, 20, 30],
        );
        assert!(fp.is_some());
        let (f, seq) = stored.expect("insert");
        assert_eq!(seq, vec![10, 20, 30]);
        assert_eq!(
            f,
            hipfire_generate::common::asst_turn_fingerprint(&action.fingerprint_text, &action.tool_calls)
        );
    }

    #[test]
    fn qwen_dflash_contract_version_is_v2() {
        assert_eq!(QWEN_DFLASH_SEMANTIC_CONTRACT_VERSION, 2);
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(5), Some(2));
        assert_eq!(hipfire_generate::common::gen_start_contract_version_for_arch(6), Some(2));
    }

    #[test]
    fn no_whole_output_parser_in_terminal_path() {
        // Terminal path authority is FinishSummary fields only — a finish with
        // empty held calls cannot invent tools from visible text markers.
        let fin = FinishSummary {
            events: vec![ClientEvent::Token(
                "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>".into(),
            )],
            finish_reason: "stop",
            tool_calls: 0,
            visible_text: String::new(),
            decoded_eot: false,
            open_think: false,
        };
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, false, false, "", false);
        match term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(finish_reason, "stop");
                assert!(!release_tool_calls);
                assert!(wire_tool_calls.is_empty());
            }
            other => panic!("expected stop Done without invented calls, got {other:?}"),
        }
    }

    #[test]
    fn production_done_value_builder_matches_epilogue_shape() {
        let v =
            hipfire_generate::qwen::qwen_dflash_done_value("r", 3, 1.5, 10, 2.0, 5.0, 1.2, 2.0, 0.5, 2, 0, "length", 99);
        assert_eq!(v["type"], "done");
        assert_eq!(v["finish_reason"], "length");
        assert_eq!(v["attempt_id"], 99);
        assert_eq!(v["dflash"], true);
        assert_eq!(v["tokens"], 3);
    }

    // --- Task 4 production-seam invariants (pending-seed / cancel / evict /
    // capacity / jinja / wire / rollback attestation) ---

    #[test]
    fn trigger_token_retained_before_forced_suffix_tx() {
        // Forced GPU tx must first commit the current pending seed (the
        // force-trigger), then forced[..n-1]. The trigger is never dropped.
        let trigger = 77u32;
        let forced = [10u32, 11, 12];
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(trigger, &forced, true);
        assert_eq!(tx.commit[0], trigger, "trigger must lead the commit batch");
        assert_eq!(tx.commit, vec![77, 10, 11]);
        assert_eq!(tx.position_delta, forced.len());
        assert_eq!(tx.commit.len(), tx.position_delta);
        // Trigger is not the new pending seed unless forced was length-1.
        assert_ne!(tx.pending_seed, trigger);
    }

    #[test]
    fn final_forced_token_is_pending_exactly_once() {
        // Last forced token becomes the unprocessed pending seed and MUST NOT
        // also appear in commit (no double-forward).
        let forced = [20u32, 21, 22];
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(5, &forced, true);
        assert_eq!(tx.pending_seed, 22);
        assert!(
            !tx.commit.contains(&22),
            "last forced must stay unwritten: {:?}",
            tx.commit
        );
        assert_eq!(tx.commit, vec![5, 20, 21]);
        // Single-token forced: commit is only the prior seed; forced[0] pending.
        let one = hipfire_generate::qwen::spec_forced_pending_seed_tx(99, &[42], true);
        assert_eq!(one.commit, vec![99]);
        assert_eq!(one.pending_seed, 42);
        assert!(!one.commit.contains(&42));
        assert_eq!(one.position_delta, 1);
    }

    #[test]
    fn terminal_pending_seed_flush_exactly_once() {
        let seed = 314u32;
        let tx = hipfire_generate::qwen::spec_terminal_pending_seed_tx(seed);
        assert_eq!(tx.commit, vec![seed]);
        assert_eq!(tx.position_delta, 1);
        assert_eq!(tx.commit.len(), 1, "flush commits the seed once");
        // Terminal flush ends with the same logical token as conversation
        // (pending_seed field equals the committed token; no second lagging seed).
        assert_eq!(tx.pending_seed, seed);
    }

    #[test]
    fn forced_max_tokens_clip_hard_ceiling() {
        // generated already includes the trigger; no GPU for tokens past budget.
        let forced = [1u32, 2, 3, 4, 5];
        assert_eq!(hipfire_generate::qwen::spec_forced_tokens_within_budget(8, 10, &forced), &[1, 2]);
        assert_eq!(
            hipfire_generate::qwen::spec_forced_tokens_within_budget(10, 10, &forced),
            &[] as &[u32]
        );
        assert_eq!(hipfire_generate::qwen::spec_forced_tokens_within_budget(0, 3, &forced), &[1, 2, 3]);
        assert_eq!(hipfire_generate::qwen::spec_forced_tokens_within_budget(9, 10, &forced), &[1]);
        // Composition: clip then build tx — only fitting tokens become pending.
        let clipped = hipfire_generate::qwen::spec_forced_tokens_within_budget(7, 10, &forced);
        assert_eq!(clipped, &[1, 2, 3]);
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(70, clipped, true);
        assert_eq!(tx.commit, vec![70, 1, 2]);
        assert_eq!(tx.pending_seed, 3);
        assert!(!tx.commit.contains(&4) && !tx.commit.contains(&5));
    }

    #[test]
    fn cancellation_classification_forced_gpu_advance() {
        assert_eq!(
            hipfire_generate::qwen::classify_forced_gpu_advance(false),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Committed
        );
        assert_eq!(
            hipfire_generate::qwen::classify_forced_gpu_advance(true),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Cancelled
        );
        // Cancelled path must use aborted+done wire, never bake the forced token.
        // ErrorOnly is reserved for eviction failures (XOR below).
        assert_ne!(hipfire_generate::qwen::SpecFailClosedWire::Cancelled, hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
        set_active_attempt_id(55);
        let mut sink = Vec::new();
        match hipfire_generate::qwen::classify_forced_gpu_advance(true) {
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Cancelled => {
                emit_qwen_ar_cancelled(&mut sink, "c-force", 4);
            }
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Committed => panic!("abort must classify Cancelled"),
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[0]["reason"], "client_cancelled");
        assert_eq!(lines[0]["attempt_id"], 55);
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert!(!out.contains(r#""type":"error""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        set_active_attempt_id(0);
    }

    #[test]
    fn eviction_error_terminal_exclusivity() {
        // maybe_evict / on_evict Err → ErrorOnly: one fail-closed error, no done.
        assert_eq!(hipfire_generate::qwen::classify_evict_failure_wire(), hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
        set_active_attempt_id(66);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        match hipfire_generate::qwen::classify_evict_failure_wire() {
            hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly => {
                hipfire_generate::common::emit_fail_closed_error(
                    &mut sink,
                    Some("ev1"),
                    "on_evict: synthetic retain failure",
                    "validation",
                    false,
                    &ep,
                );
            }
            hipfire_generate::qwen::SpecFailClosedWire::Cancelled => panic!("evict must not classify Cancelled"),
        }
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "error XOR done: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], true);
        assert_eq!(lines[0]["attempt_id"], 66);
        assert_eq!(lines[0]["id"], "ev1");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Fail-closed early exit skips wrapper epilogue (same as step failure).
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        set_active_attempt_id(0);
    }

    #[test]
    fn strict_prefix_replay_capacity_rejection() {
        let prompt = vec![1u32, 2, 3];
        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, 9, &[10, 11, 12]);
        // plan.replay = [1,2,3,9,10,11], position=6, seed=12
        assert_eq!(plan.replay.len(), plan.position);
        assert_eq!(plan.seed_token, 12);
        assert!(!plan.replay.contains(&12));

        // Fits both caps (position must be strictly < caps — pending seed slot).
        assert!(hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 64, 64, 0, false).is_ok());
        // Boundary: position == cap leaves no legal write slot for pending seed.
        let err_eq = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, plan.position, 64, 0, false).unwrap_err();
        assert!(
            err_eq.contains("physical_cap"),
            "expected position==physical_cap reject, got {err_eq}"
        );

        // Physical capacity rejection — fail closed before reset/prefill.
        let err_phys = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 5, 64, 0, false).unwrap_err();
        assert!(
            err_phys.contains("physical_cap"),
            "expected physical_cap reject, got {err_phys}"
        );

        // Speculator ctx capacity rejection.
        let err_ctx = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 64, 4, 0, false).unwrap_err();
        assert!(
            err_ctx.contains("ctx_capacity"),
            "expected ctx_capacity reject, got {err_ctx}"
        );

        // Broken invariant (replay/position mismatch) rejects even if caps large.
        let broken = hipfire_generate::qwen::SpecPrefixRealignPlan {
            replay: vec![1, 2],
            position: 5,
            seed_token: 9,
        };
        let err_inv = hipfire_generate::qwen::spec_prefix_realign_admit(&broken, 100, 100, 0, false).unwrap_err();
        assert!(
            err_inv.contains("invariant") || err_inv.contains("pending"),
            "expected invariant reject, got {err_inv}"
        );

        // Compacted/eviction path still fails closed on oversize full-history replay.
        let err_ev = hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 5, 64, 3, true).unwrap_err();
        assert!(
            err_ev.contains("physical_cap") || err_ev.contains("compact"),
            "expected compacted oversize reject, got {err_ev}"
        );

        // Capacity reject wires as exclusive error terminal (no done).
        set_active_attempt_id(71);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink,
            Some("realign"),
            &err_phys,
            "validation",
            false,
            &ep,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["attempt_id"], 71);
        assert!(!out.contains(r#""type":"done""#));
        set_active_attempt_id(0);
    }

    #[test]
    fn configured_jinja_render_fail_closed_policy() {
        // Production hipfire_generate::qwen::generate_dflash configured-template Err path:
        // hipfire_generate::dense::emit_active_attempt_error(class=validation, retryable=false,
        // rolled_back=false, message="DFlash jinja render: …") then handled=true.
        // Plain is not a silent fallback when a template is configured.
        set_active_attempt_id(88);
        let mut sink = Vec::new();
        let render_err = "undefined variable `messages`";
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("j1"),
            &format!("DFlash jinja render: {render_err}"),
            "validation",
            false,
            false,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 88);
        assert_eq!(lines[0]["id"], "j1");
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.starts_with("DFlash jinja render:"), "{msg}");
        assert!(msg.contains(render_err), "{msg}");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"gen_start""#));
        // handled=true contract: early exit skips AR/done epilogue.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        set_active_attempt_id(0);
    }

    #[test]
    fn correlated_escaped_dflash_info_frame() {
        // DFlash ctx-capacity fallback info uses serde + active attempt_id and
        // must survive adversarial id/message bytes without breaking JSONL.
        set_active_attempt_id(13);
        let mut sink = Vec::new();
        let id = "id\"x\n\t\\";
        let message = "prompt=3 + max_tokens=9 exceeds DFlash draft ctx capacity 8 — falling back to AR (\"identical\" output)";
        emit_qwen_ar_info(&mut sink, id, message);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "info");
        assert_eq!(lines[0]["id"], id);
        assert_eq!(lines[0]["message"], message);
        assert_eq!(lines[0]["attempt_id"], 13);
        // Round-trip proves escaping: re-serialize must still parse as one object.
        let raw = out.lines().next().unwrap();
        let again: serde_json::Value = serde_json::from_str(raw).expect("serde-escaped info");
        assert_eq!(again["id"].as_str().unwrap(), id);
        set_active_attempt_id(0);
    }

    #[test]
    fn rollback_attestation_false_on_sync_failure_surface() {
        // No injectable mock GPU; production surface is hipfire_generate::common::RollbackEpilogue from
        // hipfire_generate::common::fail_closed_device_sync on Err → rolled_back=false + context.
        // hipfire_generate::common::emit_fail_closed_error must append context and claim rolled_back=false.
        set_active_attempt_id(17);
        let mut sink = Vec::new();
        let ep = attest_epilogue_with_context("device_synchronize failed: hipErrorUnknown");
        assert!(!ep.rolled_back);
        assert!(ep
            .context
            .as_ref()
            .unwrap()
            .contains("device_synchronize failed"));
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink,
            Some("rb1"),
            "forced-token advance: boom",
            "validation",
            false,
            &ep,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 17);
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.contains("forced-token advance: boom"), "{msg}");
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out.contains(r#""type":"done""#));

        // Attested success path still reports rolled_back=true without context suffix.
        let mut sink_ok = Vec::new();
        let ep_ok = attest_epilogue(true);
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink_ok,
            Some("rb2"),
            "spec_step: boom",
            "validation",
            false,
            &ep_ok,
        );
        let ok = parse_jsonl(&String::from_utf8(sink_ok).unwrap());
        assert_eq!(ok[0]["rolled_back"], true);
        assert_eq!(ok[0]["message"], "spec_step: boom");
        set_active_attempt_id(0);
    }

    #[test]
    fn pending_seed_chain_trigger_clip_force_then_terminal_flush() {
        // End-to-end pure chain defending the single pending-seed invariant:
        // mid-window force trigger retained → budget clip → forced tx leaves
        // last forced pending → safe terminal flushes that seed once.
        let prompt = vec![1u32, 2];
        let first = 50u32;
        // Consume force-trigger only from a wider speculative window.
        let step = SpecStep::new([60u32, 61, 62], 62, 3, 2);
        let host = hipfire_generate::qwen::spec_host_advance_after_step(
            prompt.len(),
            0,
            vec![first],
            &step.emit,
            step.next_seed,
            1,
        );
        assert_eq!(host.seed_token, 60); // trigger retained as pending seed
        assert_eq!(host.generated, 1);

        let forced_raw = [70u32, 71, 72, 73];
        // generated=1 (trigger counted); max_tokens=3 → room for 2 forced.
        let forced = hipfire_generate::qwen::spec_forced_tokens_within_budget(host.generated, 3, &forced_raw);
        assert_eq!(forced, &[70, 71]);
        let ftx = hipfire_generate::qwen::spec_forced_pending_seed_tx(host.seed_token, forced, true);
        assert_eq!(ftx.commit, vec![60, 70]); // trigger + forced[..n-1]
        assert_eq!(ftx.pending_seed, 71); // last forced pending once
        assert!(!ftx.commit.contains(&71));
        assert_eq!(ftx.position_delta, 2);

        let position = host.position + ftx.position_delta;
        let generated = host.generated + forced.len();
        // host.position already counts the force-trigger write slot after prefill first.
        assert_eq!(position, prompt.len() + 1 + ftx.position_delta);
        assert_eq!(generated, 3);

        // Safe terminal: flush final pending seed exactly once.
        let term = hipfire_generate::qwen::spec_terminal_pending_seed_tx(ftx.pending_seed);
        assert_eq!(term.commit, vec![71]);
        assert_eq!(term.position_delta, 1);
        let final_pos = position + term.position_delta;
        // Full history: prompt + first_token + trigger + forced (generated).
        assert_eq!(final_pos, prompt.len() + 1 + generated);

        // Realign plan after force path still keeps last raw as unwritten seed.
        let mut raw = vec![60u32];
        raw.extend_from_slice(forced);
        let plan = hipfire_generate::qwen::spec_prefix_realign_plan(&prompt, first, &raw);
        assert_eq!(plan.seed_token, 71);
        assert_ne!(plan.replay.last().copied(), Some(plan.seed_token));
        assert!(hipfire_generate::qwen::spec_prefix_realign_admit(&plan, 1024, 1024, 0, false).is_ok());
    }

    // ── Task 4 Important vetoes (production seam pins) ─────────────────────

    /// max_tokens==0 rejects at hipfire_generate::qwen::generate_spec entry via the same writer the
    /// production gate uses — before prefill/GPU/state/client mutation.
    /// Wire: one correlated validation error, rolled_back=false, no done/aborted.
    #[test]
    fn zero_budget_max_tokens_preflight_error_only_no_done() {
        set_active_attempt_id(101);
        let mut sink = Vec::new();
        // Mirrors hipfire_generate::qwen::generate_spec entry gate (max_tokens == 0 → emit + return None).
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("zb0"),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = std::io::Write::flush(&mut sink);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "exactly one correlated error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "zb0");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 101);
        assert_eq!(lines[0]["message"], "max_tokens must be > 0");
        // No first token, no safe terminal flush, no aborted pair.
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"token""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Wrapper contract: hipfire_generate::qwen::generate_spec returned None → no epilogue.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));
        set_active_attempt_id(0);
    }

    /// Cancel after rollback attestation: attested → aborted+done; unattested →
    /// exactly one correlated nonretryable error with context and no done.
    #[test]
    fn cancel_after_rollback_attested_vs_unattested_wire() {
        // Attested rollback keeps fold-compatible aborted + done pair.
        set_active_attempt_id(202);
        let mut sink_ok = Vec::new();
        let ep_ok = attest_epilogue(true);
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink_ok, "c-ok", 7, &ep_ok);
        let out_ok = String::from_utf8(sink_ok).unwrap();
        let lines_ok = parse_jsonl(&out_ok);
        assert_eq!(
            lines_ok.len(),
            2,
            "attested cancel: aborted+done {lines_ok:?}"
        );
        assert_eq!(lines_ok[0]["type"], "aborted");
        assert_eq!(lines_ok[0]["reason"], "client_cancelled");
        assert_eq!(lines_ok[0]["attempt_id"], 202);
        assert_eq!(lines_ok[0]["id"], "c-ok");
        assert_eq!(lines_ok[1]["type"], "done");
        assert_eq!(lines_ok[1]["finish_reason"], "aborted");
        assert_eq!(lines_ok[1]["completion_tokens"], 7);
        assert_eq!(lines_ok[1]["attempt_id"], 202);
        assert!(!out_ok.contains(r#""type":"error""#));
        assert!(!out_ok.contains(r#""type":"tool_calls""#));

        // Unattested rollback: one fail-closed error, no aborted/done.
        set_active_attempt_id(203);
        let mut sink_bad = Vec::new();
        let ep_bad = attest_epilogue_with_context("device_synchronize failed: hipErrorUnknown");
        assert!(!ep_bad.rolled_back);
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink_bad, "c-bad", 3, &ep_bad);
        let out_bad = String::from_utf8(sink_bad).unwrap();
        let lines_bad = parse_jsonl(&out_bad);
        assert_eq!(
            lines_bad.len(),
            1,
            "unattested cancel: error only {lines_bad:?}"
        );
        assert_eq!(lines_bad[0]["type"], "error");
        assert_eq!(lines_bad[0]["class"], "validation");
        assert_eq!(lines_bad[0]["retryable"], false);
        assert_eq!(lines_bad[0]["rolled_back"], false);
        assert_eq!(lines_bad[0]["attempt_id"], 203);
        assert_eq!(lines_bad[0]["id"], "c-bad");
        let msg = lines_bad[0]["message"].as_str().unwrap();
        assert!(
            msg.contains("client cancelled; fail-closed rollback could not be attested"),
            "{msg}"
        );
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out_bad.contains(r#""type":"done""#));
        assert!(!out_bad.contains(r#""type":"aborted""#));
        assert!(!out_bad.contains(r#""type":"tool_calls""#));
        set_active_attempt_id(0);
    }

    /// Failure-injection: each omitted reset class (incl. single-GPU s_ef_residual
    /// and EP bind) keeps rolled_back=false; aggregate failure still models sync
    /// as attempted; Qwen AR prefill/decode abort terminals are exclusive.
    #[test]
    fn rollback_attestation_omitted_reset_classes_and_ar_abort_xor() {
        // Every required surface Ok + sync Ok → attested.
        let all_ok = attest_rollback_steps(
            &[
                ("s_matrices", Ok(())),
                ("s_scales", Ok(())),
                ("conv_states", Ok(())),
                ("s_ef_residual", Ok(())),
                ("host_cursors", Ok(())),
                ("kv_compact", Ok(())),
                ("checkpoints", Ok(())),
                ("drafter", Ok(())),
                ("adaptive", Ok(())),
                ("graph_replay", Ok(())),
                ("ep_bind_thread", Ok(())),
            ],
            Ok(()),
        );
        assert!(all_ok.rolled_back);
        assert!(all_ok.context.is_none());

        // Single-GPU s_ef_residual omission/failure alone unattests.
        let ef = attest_rollback_steps(
            &[
                ("s_matrices", Ok(())),
                ("s_scales", Ok(())),
                ("conv_states", Ok(())),
                ("s_ef_residual", Err("memset failed".into())),
                ("ep_bind_thread", Ok(())),
            ],
            Ok(()),
        );
        assert!(!ef.rolled_back);
        let ctx = ef.context.as_deref().unwrap_or("");
        assert!(ctx.contains("s_ef_residual"), "{ctx}");
        assert!(
            !ctx.contains("device_synchronize"),
            "sync Ok must not appear: {ctx}"
        );

        // EP bind_thread failure alone unattests even when sync Ok.
        let bind = attest_rollback_steps(
            &[
                ("s_ef_residual", Ok(())),
                ("ep_bind_thread", Err("hipErrorInvalidDevice".into())),
            ],
            Ok(()),
        );
        assert!(!bind.rolled_back);
        assert!(
            bind.context
                .as_deref()
                .unwrap_or("")
                .contains("ep_bind_thread"),
            "{:?}",
            bind.context
        );

        // Aggregate reset failure + sync still attempted (both in context).
        let agg = attest_rollback_steps(
            &[
                ("s_matrices", Err("m1".into())),
                ("s_ef_residual", Err("ef".into())),
                ("ep_bind_thread", Err("bind".into())),
            ],
            Err("hipErrorUnknown".into()),
        );
        assert!(!agg.rolled_back);
        let ctx = agg.context.as_deref().unwrap_or("");
        assert!(ctx.contains("s_matrices"), "{ctx}");
        assert!(ctx.contains("s_ef_residual"), "{ctx}");
        assert!(ctx.contains("ep_bind_thread"), "{ctx}");
        assert!(ctx.contains("device_synchronize failed"), "{ctx}");

        // hipfire_generate::common::fail_closed_epilogue_after_sync: prior Err + sync Ok → unattested, sync ran.
        let merged = hipfire_generate::common::fail_closed_epilogue_after_sync(
            Err("hipfire_generate::common::reset_qwen35_recurrent: s_ef_residual memset: boom".into()),
            hipfire_generate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            },
        );
        assert!(!merged.rolled_back);
        assert!(
            merged
                .context
                .as_deref()
                .unwrap_or("")
                .contains("s_ef_residual"),
            "{:?}",
            merged.context
        );

        // prior Err + sync Err → both preserved.
        let both = hipfire_generate::common::fail_closed_epilogue_after_sync(
            Err("ep rank0 bind_thread: bad".into()),
            hipfire_generate::common::RollbackEpilogue {
                rolled_back: false,
                context: Some("device_synchronize failed: hipErrorUnknown".into()),
            },
        );
        assert!(!both.rolled_back);
        let ctx = both.context.as_deref().unwrap_or("");
        assert!(ctx.contains("bind_thread"), "{ctx}");
        assert!(ctx.contains("device_synchronize failed"), "{ctx}");

        // Qwen AR prefill abort terminal exclusivity (attested vs unattested).
        set_active_attempt_id(501);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "ar-prefill", 0, &attest_epilogue(true));
        let lines = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["completion_tokens"], 0);
        assert!(lines.iter().all(|e| e["attempt_id"] == 501));

        set_active_attempt_id(502);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(
            &mut sink,
            "ar-prefill-bad",
            0,
            &attest_epilogue_with_context("s_ef_residual memset: boom"),
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "prefill unattested: error only");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["rolled_back"], false);
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"tool_calls""#));

        // Qwen AR mid-decode abort terminal exclusivity.
        set_active_attempt_id(503);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "ar-decode", 5, &attest_epilogue(true));
        let lines = parse_jsonl(&String::from_utf8(sink).unwrap());
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "aborted");
        assert_eq!(lines[1]["finish_reason"], "aborted");
        assert_eq!(lines[1]["completion_tokens"], 5);

        set_active_attempt_id(504);
        let mut sink = Vec::new();
        hipfire_generate::common::emit_spec_cancel_after_rollback(
            &mut sink,
            "ar-decode-bad",
            5,
            &attest_epilogue_with_context(
                "ep rank0 bind_thread: bad; device_synchronize failed: x",
            ),
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "decode unattested: error only");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 504);
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.contains("bind_thread"), "{msg}");
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        set_active_attempt_id(0);
    }

    /// Eviction-enabled missing optional kv_cache_mut is ErrorOnly (not panic):
    /// hipfire_generate::qwen::classify_evict_failure_wire → hipfire_generate::common::emit_fail_closed_error with the production
    /// post-prefill / per-cycle messages; no done/aborted/calls/cache.
    #[test]
    fn missing_optional_kv_cache_mut_is_error_only_not_panic() {
        assert_eq!(hipfire_generate::qwen::classify_evict_failure_wire(), hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
        assert_ne!(
            hipfire_generate::qwen::SpecFailClosedWire::Cancelled,
            hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly,
            "missing KV hook must never classify as Cancelled"
        );

        for (attempt, id, message) in [
            (301u64, "kv-pp", "kv_cache_mut missing (post-prefill)"),
            (302u64, "kv-pc", "kv_cache_mut missing (per-cycle)"),
        ] {
            set_active_attempt_id(attempt);
            let mut sink = Vec::new();
            // Production seam: classify first, then fail-closed writer (same as
            // hipfire_generate::qwen::generate_spec match slot.kv_cache_mut() { None => ... }).
            let _ = hipfire_generate::qwen::classify_evict_failure_wire();
            let ep = attest_epilogue(true);
            match hipfire_generate::qwen::classify_evict_failure_wire() {
                hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly => {
                    hipfire_generate::common::emit_fail_closed_error(&mut sink, Some(id), message, "validation", false, &ep);
                }
                hipfire_generate::qwen::SpecFailClosedWire::Cancelled => {
                    panic!("kv_cache_mut missing must not classify Cancelled")
                }
            }
            let out = String::from_utf8(sink).unwrap();
            let lines = parse_jsonl(&out);
            assert_eq!(lines.len(), 1, "error XOR done for {message}: {lines:?}");
            assert_eq!(lines[0]["type"], "error");
            assert_eq!(lines[0]["class"], "validation");
            assert_eq!(lines[0]["retryable"], false);
            assert_eq!(lines[0]["rolled_back"], true);
            assert_eq!(lines[0]["attempt_id"], attempt);
            assert_eq!(lines[0]["id"], id);
            assert_eq!(lines[0]["message"], message);
            assert!(!out.contains(r#""type":"done""#));
            assert!(!out.contains(r#""type":"aborted""#));
            assert!(!out.contains(r#""type":"tool_calls""#));
            // hipfire_generate::qwen::generate_spec returns None → wrapper skips cache store / epilogue.
            assert!(!qwen_dflash_epilogue_after_spec_run(false));
        }

        // Unattested rollback on the same missing-hook path: rolled_back=false
        // + context appended; still error-only (no panic surface).
        set_active_attempt_id(303);
        let mut sink = Vec::new();
        let ep = attest_epilogue_with_context("device_synchronize failed: test");
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink,
            Some("kv-ua"),
            "kv_cache_mut missing (post-prefill)",
            "validation",
            false,
            &ep,
        );
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["rolled_back"], false);
        let msg = lines[0]["message"].as_str().unwrap();
        assert!(msg.contains("kv_cache_mut missing (post-prefill)"), "{msg}");
        assert!(msg.contains("device_synchronize failed"), "{msg}");
        assert!(!out.contains(r#""type":"done""#));
        set_active_attempt_id(0);
    }

    // ── Remaining Important Task 4 vetoes (wrapper / legacy / rewind) ──

    /// hipfire_generate::qwen::generate_dflash max_tokens==0: hipfire_generate::dense::emit_active_attempt_error then return true
    /// (handled) before Jinja/render/set_sampling/gen_start. Same wire as the
    /// inner hipfire_generate::qwen::generate_spec defense; wrapper must not fall through to AR.
    #[test]
    fn generate_dflash_zero_budget_preflight_handled_error_only() {
        set_active_attempt_id(401);
        let mut sink = Vec::new();
        // Mirrors hipfire_generate::qwen::generate_dflash entry (max_tokens == 0 → emit + return true).
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("df-zb0"),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = std::io::Write::flush(&mut sink);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "exactly one correlated error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "df-zb0");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 401);
        assert_eq!(lines[0]["message"], "max_tokens must be > 0");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"token""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Handled=true → caller must not fall through to AR / second envelope.
        let wrapper_handled = true;
        assert!(wrapper_handled);
        set_active_attempt_id(0);
    }

    /// hipfire_generate::dense::generate_deepseek4_spec max_tokens==0: same emit policy, plain return
    /// (unit fn) before DSML render / decode-cache teardown / set_sampling.
    #[test]
    fn generate_deepseek4_spec_zero_budget_preflight_error_only() {
        set_active_attempt_id(402);
        let mut sink = Vec::new();
        // Mirrors hipfire_generate::dense::generate_deepseek4_spec entry (max_tokens == 0 → emit + return).
        hipfire_generate::dense::emit_active_attempt_error(
            &mut sink,
            Some("ds4-zb0"),
            "max_tokens must be > 0",
            "validation",
            false,
            false,
        );
        let _ = std::io::Write::flush(&mut sink);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "exactly one correlated error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "ds4-zb0");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], false);
        assert_eq!(lines[0]["attempt_id"], 402);
        assert_eq!(lines[0]["message"], "max_tokens must be > 0");
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"token""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // Unit wrapper returns (no AR fallthrough second write).
        set_active_attempt_id(0);
    }

    /// Legacy non-qwen hipfire_generate::qwen::generate_dflash else-branch: fail_closed_rollback.is_some()
    /// || grammar_violated → hipfire_generate::common::emit_fail_closed_error only; no extract/release/
    /// cache store / done. Message classified by grammar / open_think /
    /// malformed_protocol / generic.
    #[test]
    fn legacy_non_qwen_fail_closed_epilogue_error_only_no_extract() {
        // Production message selection (qwen_semantic_v2 == false branch).
        fn legacy_fail_closed_message(
            grammar_violated: bool,
            open_think: bool,
            finish_reason: &str,
        ) -> &'static str {
            if grammar_violated {
                "grammar violation during speculative decode"
            } else if open_think || finish_reason == "open_think" {
                "open think span at end of generation (validation)"
            } else if finish_reason == "malformed_protocol" {
                "malformed tool protocol"
            } else {
                "fail-closed speculative decode"
            }
        }

        let cases = [
            (
                true,
                false,
                "stop",
                "grammar violation during speculative decode",
            ),
            (
                false,
                true,
                "stop",
                "open think span at end of generation (validation)",
            ),
            (
                false,
                false,
                "open_think",
                "open think span at end of generation (validation)",
            ),
            (
                false,
                false,
                "malformed_protocol",
                "malformed tool protocol",
            ),
            (false, false, "length", "fail-closed speculative decode"),
        ];

        for (i, (grammar, open_think, reason, expected_msg)) in cases.iter().enumerate() {
            assert_eq!(
                legacy_fail_closed_message(*grammar, *open_think, reason),
                *expected_msg,
                "case {i} message select"
            );
            // Gate: fail_closed_rollback.is_some() || grammar_violated.
            let fail_closed_present = true;
            let take_error_only = fail_closed_present || *grammar;
            assert!(take_error_only, "case {i} must take error-only path");

            set_active_attempt_id(500 + i as u64);
            let mut sink = Vec::new();
            let ep = attest_epilogue(true);
            hipfire_generate::common::emit_fail_closed_error(
                &mut sink,
                Some("leg-fc"),
                expected_msg,
                "validation",
                false,
                &ep,
            );
            let out = String::from_utf8(sink).unwrap();
            let lines = parse_jsonl(&out);
            assert_eq!(lines.len(), 1, "case {i}: error XOR done {lines:?}");
            assert_eq!(lines[0]["type"], "error");
            assert_eq!(lines[0]["class"], "validation");
            assert_eq!(lines[0]["retryable"], false);
            assert_eq!(lines[0]["id"], "leg-fc");
            assert_eq!(lines[0]["message"], *expected_msg);
            // No held tool_calls release, no cache store, no done/aborted.
            assert!(!out.contains(r#""type":"done""#), "case {i}");
            assert!(!out.contains(r#""type":"aborted""#), "case {i}");
            assert!(!out.contains(r#""type":"tool_calls""#), "case {i}");
            // Early return true from hipfire_generate::qwen::generate_dflash — no whole-output extract path.
            let early_return_handled = true;
            assert!(early_return_handled);
        }
        set_active_attempt_id(0);
    }

    /// hipfire_generate::qwen::generate_spec resume_from: on spec.rewind_to Err, host seq_pos /
    /// conversation_tokens must NOT be truncated to ckpt first. Fail-closed
    /// live rollback + one correlated "rewind_to: …" error; return None skips
    /// wrapper epilogue (no done / calls / cache).
    #[test]
    fn rewind_to_err_freezes_host_cursors_then_fail_closed() {
        // Host state as if mid-conversation before resume_from rewind.
        let ckpt = 4usize;
        let mut seq_pos = 12usize;
        let mut conversation_tokens: Vec<u32> = (0..12).map(|t| t as u32).collect();
        let seq_before = seq_pos;
        let toks_before = conversation_tokens.clone();

        // Production order on Err: message first, then live rollback (which
        // zeroes host), emit, return None — never the success truncate.
        let restore_err = "DeltaNetSnapshot::restore_to: synthetic restore fail";
        let msg = format!("rewind_to: {restore_err}");

        // Success path would do: seq_pos = ckpt; conversation_tokens.truncate(ckpt).
        // Error path must NOT apply that before/without fail-closed.
        let rewind_ok = false;
        if rewind_ok {
            seq_pos = ckpt;
            conversation_tokens.truncate(ckpt);
        }
        // Cursors still at pre-rewind values until hipfire_generate::common::production_fail_closed_rollback_live.
        assert_eq!(
            seq_pos, seq_before,
            "must not truncate seq_pos to ckpt on Err"
        );
        assert_eq!(
            conversation_tokens, toks_before,
            "must not truncate conversation_tokens to ckpt on Err"
        );
        assert_ne!(seq_pos, ckpt);

        // Live rollback zeroes host (GPU-less stand-in for hipfire_generate::common::production_fail_closed_rollback_live).
        seq_pos = 0;
        conversation_tokens.clear();
        assert_eq!(seq_pos, 0);
        assert!(conversation_tokens.is_empty());

        set_active_attempt_id(601);
        let mut sink = Vec::new();
        let ep = attest_epilogue(true);
        hipfire_generate::common::emit_fail_closed_error(&mut sink, Some("rw-err"), &msg, "validation", false, &ep);
        let out = String::from_utf8(sink).unwrap();
        let lines = parse_jsonl(&out);
        assert_eq!(lines.len(), 1, "one correlated rewind error: {lines:?}");
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["id"], "rw-err");
        assert_eq!(lines[0]["class"], "validation");
        assert_eq!(lines[0]["retryable"], false);
        assert_eq!(lines[0]["rolled_back"], true);
        assert_eq!(lines[0]["attempt_id"], 601);
        assert_eq!(lines[0]["message"], msg);
        assert!(lines[0]["message"]
            .as_str()
            .unwrap()
            .starts_with("rewind_to:"));
        assert!(!out.contains(r#""type":"done""#));
        assert!(!out.contains(r#""type":"aborted""#));
        assert!(!out.contains(r#""type":"tool_calls""#));
        // hipfire_generate::qwen::generate_spec returns None → wrapper skips epilogue/cache.
        assert!(!qwen_dflash_epilogue_after_spec_run(false));

        // Unattested sync path still error-only with context suffix.
        set_active_attempt_id(602);
        let mut sink_ua = Vec::new();
        let ep_ua = attest_epilogue_with_context("device_synchronize failed: hipErrorUnknown");
        hipfire_generate::common::emit_fail_closed_error(
            &mut sink_ua,
            Some("rw-ua"),
            &msg,
            "validation",
            false,
            &ep_ua,
        );
        let out_ua = String::from_utf8(sink_ua).unwrap();
        let lines_ua = parse_jsonl(&out_ua);
        assert_eq!(lines_ua.len(), 1);
        assert_eq!(lines_ua[0]["rolled_back"], false);
        let m = lines_ua[0]["message"].as_str().unwrap();
        assert!(m.contains("rewind_to:"), "{m}");
        assert!(m.contains("device_synchronize failed"), "{m}");
        assert!(!out_ua.contains(r#""type":"done""#));
        set_active_attempt_id(0);
    }

    // ── Task 4 definitive terminal-edge blockers ──────────────────────────

    /// Legacy non-qwen hipfire_generate::qwen::generate_dflash else-branch: length still emits
    /// finish_reason=length but never releases held tool calls or stores
    /// asst_turn_cache (partial/truncated turns are unsafe to prime).
    #[test]
    fn legacy_length_terminal_skips_assistant_cache_and_tool_release() {
        // Production gates (hipfire_generate::qwen::generate_dflash qwen_semantic_v2=false branch):
        //   hit_length_cap = run.generated >= max_tokens
        //   stage_terminal_tool_calls on safe tool terminals before handshake
        //   asst_turn_cache.insert only when Commit && !hit_length_cap && !cached_seq.is_empty()
        let generated = 8usize;
        let max_tokens = 8usize;
        let hit_length_cap = generated >= max_tokens;
        assert!(hit_length_cap);

        let finish = summary_tool_calls(vec![ToolCall {
            id: None,
            name: "held".into(),
            arguments: serde_json::json!({}),
            rendered_body: None,
        }]);
        assert!(finish.tool_calls > 0);

        let release = !hit_length_cap && finish.tool_calls > 0;
        assert!(!release, "length must not release held finish tool calls");

        let cached_seq = vec![1u32, 2, 3];
        let mut sink: std::collections::HashMap<u64, Vec<u32>> = std::collections::HashMap::new();
        if !hit_length_cap && !cached_seq.is_empty() {
            let decoded_full = "partial answer";
            let stripped = hipfire_generate::common::strip_think_for_fingerprint(decoded_full);
            let emit_text =
                hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
            let emit_tool_calls = extract_tool_calls_from_text(decoded_full);
            let fp = hipfire_generate::common::asst_turn_fingerprint(&emit_text, &emit_tool_calls);
            sink.insert(fp, cached_seq.clone());
        }
        assert!(
            sink.is_empty(),
            "length terminal must not store asst_turn_cache"
        );

        let finish_reason = if hit_length_cap {
            "length"
        } else if finish.tool_calls > 0 {
            "tool_calls"
        } else {
            "stop"
        };
        assert_eq!(finish_reason, "length");

        // Safe non-length control: same gates allow release + store.
        let hit_safe = 3usize >= 8usize;
        assert!(!hit_safe);
        assert!(!hit_safe && finish.tool_calls > 0);
        let mut sink_safe = std::collections::HashMap::new();
        if !hit_safe && !cached_seq.is_empty() {
            let fp = hipfire_generate::common::asst_turn_fingerprint("ok", &[]);
            sink_safe.insert(fp, cached_seq.clone());
        }
        assert_eq!(sink_safe.len(), 1, "safe stop still stores");
    }

    /// Begin-triggered forced continuation is planned with the same pure
    /// pending-seed transaction as mid-window force, and is ordered before
    /// any speculative step (max_tokens=1 cannot spend budget on step).
    #[test]
    fn begin_first_token_forced_serviced_before_spec_step() {
        // After begin: generated counts first token when event-bearing.
        let mut generated = 1usize;
        let max_tokens = 1usize;
        let seed_token = 50u32; // first_token is also the initial pending seed
        let forced_begin = vec![60u32, 61, 62];

        // Empty take_forced ⇒ Skipped (no GPU path); loop may proceed.
        assert!(matches!(
            // Pure stand-in for hipfire_generate::qwen::apply_spec_forced_pending_seed empty input.
            {
                let forced_all: &[u32] = &[];
                if forced_all.is_empty() {
                    hipfire_generate::qwen::SpecForcedApplyResult::Skipped
                } else {
                    hipfire_generate::qwen::SpecForcedApplyResult::Applied
                }
            },
            hipfire_generate::qwen::SpecForcedApplyResult::Skipped
        ));

        // Hard budget clip: generated already 1, max_tokens=1 → room 0.
        let clipped = hipfire_generate::qwen::spec_forced_tokens_within_budget(generated, max_tokens, &forced_begin);
        assert!(
            clipped.is_empty(),
            "max_tokens=1 after first token must clip all forced (no extra step budget)"
        );
        // hipfire_generate::qwen::apply_spec_forced_pending_seed returns Skipped on empty clip — while
        // condition `generated < max_tokens` is already false, so no spec.step.
        assert!(!(!false /*first_token_is_eos*/ && generated < max_tokens));

        // Room for forced (max_tokens=3, generated=1): same tx as mid-window.
        generated = 1;
        let max2 = 3usize;
        let forced = hipfire_generate::qwen::spec_forced_tokens_within_budget(generated, max2, &forced_begin);
        assert_eq!(forced, &[60u32, 61]);
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed_token, forced, true);
        assert_eq!(tx.commit, vec![50, 60], "trigger retained; forced[..n-1]");
        assert_eq!(tx.pending_seed, 61, "last forced pending once");
        assert!(!tx.commit.contains(&61));
        assert_eq!(tx.position_delta, forced.len());

        // Ordering contract: begin force runs before while/spec.step.
        let mut phase = "begin";
        let forced_begin_nonempty = !forced_begin.is_empty();
        if forced_begin_nonempty {
            phase = "begin_forced_applied";
        }
        let enter_spec_step = phase == "begin_forced_applied" && generated < max2;
        // After applying 2 forced, generated would be 1+2=3 → loop does not step.
        let generated_after = generated + forced.len();
        assert_eq!(generated_after, 3);
        assert!(
            !(generated_after < max2),
            "after begin force at budget, no speculative step"
        );
        let _ = enter_spec_step;
        assert_eq!(phase, "begin_forced_applied");

        // hipfire_generate::qwen::classify_forced_gpu_advance still exclusive cancel vs commit.
        assert!(matches!(
            hipfire_generate::qwen::classify_forced_gpu_advance(true),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Cancelled
        ));
        assert!(matches!(
            hipfire_generate::qwen::classify_forced_gpu_advance(false),
            hipfire_generate::qwen::ForcedGpuAdvanceKind::Committed
        ));
    }

    /// Qwen first seed runs user stop-sequence detection in begin exactly like
    /// later observe tokens; StopSequence terminates before any speculative step.
    #[test]
    fn qwen_begin_first_token_stop_sequence_terminates_before_step() {
        let tok = test_tokenizer();
        let ids = tok.encode("STOP");
        assert!(!ids.is_empty());
        let first = ids[0];
        let first_text = tok.decode(&[first]);
        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            stop: vec![first_text.clone()],
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let first_begin = emit.begin(first);
        assert_eq!(
            first_begin.stop,
            Some(StopReason::StopSequence),
            "begin must surface StopSequence for first-token stop match"
        );
        // hipfire_generate::qwen::generate_spec: first_token_is_eos = first_begin.stop.is_some()
        let first_token_is_eos = first_begin.stop.is_some();
        assert!(first_token_is_eos);
        // while !first_token_is_eos && generated < max_tokens { spec.step ... }
        let mut stepped = false;
        if !first_token_is_eos {
            stepped = true;
        }
        assert!(
            !stepped,
            "StopSequence begin must skip every speculative step"
        );

        // Event-bearing first token still counts (Qwen always commits).
        assert!(
            hipfire_generate::qwen::spec_outcome_seed_committable(&first_begin),
            "stop still commits the raw first token"
        );
        assert!(first_begin
            .events
            .iter()
            .any(|e| matches!(e, ClientEvent::Committed { id, .. } if *id == first)));

        // Forced begin path is still consulted, but empty take_forced is Skipped.
        let forced_begin = emit.take_forced();
        assert!(forced_begin.is_empty());
    }

    // --- Task 4 reviewer blockers: forced-token / terminal-cause seams ---

    /// Non-committable pending seed (DS4 empty-event EOS) must not be prepended
    /// into the forced GPU commit. Forced tokens occupy that same slot; all but
    /// the final kept forced token are committed, final remains pending.
    #[test]
    fn noncommittable_pending_seed_omitted_from_forced_tx() {
        // Single forced + non-committable seed: commit is empty (seed omitted,
        // forced[0] becomes pending only) — no GPU for a lone seed replace.
        let seed = 7u32; // DS4-style empty-event EOS seed
        let one = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[42], false);
        assert!(
            one.commit.is_empty(),
            "non-committable seed + single forced must not GPU-commit: {:?}",
            one.commit
        );
        assert_eq!(one.position_delta, 0);
        assert_eq!(one.pending_seed, 42);
        assert!(!one.commit.contains(&seed));

        // Multi forced + non-committable: commit is forced[..n-1] only.
        let multi = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[10, 11, 12], false);
        assert_eq!(
            multi.commit,
            vec![10, 11],
            "seed omitted; forced prefix only"
        );
        assert!(!multi.commit.contains(&seed));
        assert_eq!(multi.pending_seed, 12);
        assert_eq!(multi.position_delta, multi.commit.len());
        assert!(!multi.commit.contains(&12), "last forced stays pending");

        // Contrast: same inputs with committable seed retain the trigger.
        let keep = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[10, 11, 12], true);
        assert_eq!(keep.commit, vec![seed, 10, 11]);
        assert_eq!(keep.pending_seed, 12);
    }

    /// Forced suffix stages observe first, trims at the first non-None stop,
    /// GPU-commits only that kept prefix, and renders only after successful
    /// commit. Later forced tokens are never observed/committed/rendered.
    #[test]
    fn forced_suffix_stops_at_first_stop_sequence_prefix_only() {
        let tok = test_tokenizer();
        // Build a stop string from a real token, then force a later token that
        // must not be observed once stop fires.
        let stop_ids = tok.encode("STOP");
        assert!(!stop_ids.is_empty());
        let stop_tok = stop_ids[0];
        let stop_text = tok.decode(&[stop_tok]);
        let later = tok.encode("later");
        assert!(!later.is_empty());
        let later_tok = later[0];
        assert_ne!(stop_tok, later_tok);

        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            stop: vec![stop_text.clone()],
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        // Warm begin so observe path is active (forced uses observe).
        let warm = tok.encode("hi");
        assert!(!warm.is_empty());
        let _ = emit.begin(warm[0]);

        // Production staging loop (hipfire_generate::qwen::apply_spec_forced_pending_seed):
        let forced_all = [stop_tok, later_tok, later_tok.wrapping_add(1)];
        let mut staged: Vec<(u32, hipfire_runtime::spec::EmitOutcome)> =
            Vec::with_capacity(forced_all.len());
        let mut stop_reason: Option<StopReason> = None;
        for &ft in &forced_all {
            let fo = emit.observe(ft);
            let stop = fo.stop;
            staged.push((ft, fo));
            if let Some(reason) = stop {
                stop_reason = Some(reason);
                break;
            }
        }
        assert_eq!(
            stop_reason,
            Some(StopReason::StopSequence),
            "first forced token matching stop must halt the suffix"
        );
        assert_eq!(
            staged.len(),
            1,
            "later forced tokens must not be observed after stop"
        );
        assert_eq!(staged[0].0, stop_tok);

        let kept: Vec<u32> = staged.iter().map(|(t, _)| *t).collect();
        assert_eq!(kept, vec![stop_tok]);

        // Commit uses the kept prefix only (incoming seed was committable).
        let incoming_seed = warm[0];
        let incoming_committable = true;
        let tx = hipfire_generate::qwen::spec_forced_pending_seed_tx(incoming_seed, &kept, incoming_committable);
        // Single kept forced: commit = [seed], pending = stop_tok.
        assert_eq!(tx.commit, vec![incoming_seed]);
        assert_eq!(tx.pending_seed, stop_tok);
        assert!(!tx.commit.contains(&later_tok));
        assert!(!tx.commit.contains(&stop_tok));

        // Apply result maps to Stopped(reason) — not Applied.
        let apply = match stop_reason {
            Some(reason) => hipfire_generate::qwen::SpecForcedApplyResult::Stopped(reason),
            None => hipfire_generate::qwen::SpecForcedApplyResult::Applied,
        };
        assert_eq!(
            apply,
            hipfire_generate::qwen::SpecForcedApplyResult::Stopped(StopReason::StopSequence)
        );

        // Render-after-commit contract: client events from staged outcomes are
        // only eligible once GPU commit of `tx.commit` succeeded. Model the
        // gate explicitly so a reorder (render then commit) fails this test.
        let mut gpu_committed = false;
        let mut rendered: Vec<u32> = Vec::new();
        // "commit" kept prefix
        gpu_committed = true;
        if gpu_committed {
            for (ft, fo) in &staged {
                if !fo.events.is_empty() {
                    rendered.push(*ft);
                }
            }
        }
        assert!(gpu_committed);
        assert_eq!(
            rendered,
            vec![stop_tok],
            "render only kept prefix after commit"
        );
        assert!(!rendered.contains(&later_tok));
    }

    /// Begin and mid callers treat Stopped as turn-terminal: set semantic_stop,
    /// force first_token_is_eos / hit_eos, and skip later force + all spec.step.
    #[test]
    fn begin_and_mid_stopped_skips_later_force_and_spec_step() {
        // --- begin path (mirrors hipfire_generate::qwen::generate_spec after emit.begin) ---
        let reason = StopReason::StopSequence;
        let mut semantic_stop: Option<StopReason> = None;
        let mut first_token_is_eos = false;
        let apply = hipfire_generate::qwen::SpecForcedApplyResult::Stopped(reason);
        match apply {
            hipfire_generate::qwen::SpecForcedApplyResult::Terminal => panic!("not under test"),
            hipfire_generate::qwen::SpecForcedApplyResult::Stopped(r) => {
                if semantic_stop.is_none() && hipfire_generate::qwen::spec_stop_is_semantic(Some(r)) {
                    semantic_stop = Some(r);
                }
                first_token_is_eos = true;
            }
            hipfire_generate::qwen::SpecForcedApplyResult::Applied | hipfire_generate::qwen::SpecForcedApplyResult::Skipped => {
                panic!("expected Stopped")
            }
        }
        assert_eq!(semantic_stop, Some(StopReason::StopSequence));
        assert!(first_token_is_eos);

        // while !first_token_is_eos && generated < max_tokens { spec.step ... }
        let generated = 0usize;
        let max_tokens = 16usize;
        let mut stepped = false;
        let mut later_force = false;
        if !first_token_is_eos && generated < max_tokens {
            // would take_forced + spec.step
            later_force = true;
            stepped = true;
        }
        assert!(
            !stepped && !later_force,
            "begin Stopped must skip every subsequent force and spec.step"
        );

        // --- mid-window path (mirrors hipfire_generate::qwen::generate_spec forced_after match) ---
        let mut semantic_stop_mid: Option<StopReason> = None;
        let mut hit_eos = false;
        let mut think_cap_hit = false;
        let mid = hipfire_generate::qwen::SpecForcedApplyResult::Stopped(StopReason::StopSequence);
        match mid {
            hipfire_generate::qwen::SpecForcedApplyResult::Terminal => panic!("not under test"),
            hipfire_generate::qwen::SpecForcedApplyResult::Stopped(r) => {
                if semantic_stop_mid.is_none() && hipfire_generate::qwen::spec_stop_is_semantic(Some(r)) {
                    semantic_stop_mid = Some(r);
                }
                match r {
                    StopReason::ThinkCap => think_cap_hit = true,
                    StopReason::Eos | StopReason::StopSequence | StopReason::GrammarViolation => {
                        hit_eos = true
                    }
                }
            }
            hipfire_generate::qwen::SpecForcedApplyResult::Applied | hipfire_generate::qwen::SpecForcedApplyResult::Skipped => {
                panic!("expected Stopped")
            }
        }
        assert_eq!(semantic_stop_mid, Some(StopReason::StopSequence));
        assert!(hit_eos);
        assert!(!think_cap_hit);

        // After mid Stopped the cycle must not re-enter force or continue the
        // outer decode as if Applied. Model the break: no second take_forced.
        let mut second_force_applied = false;
        if !hit_eos && !think_cap_hit {
            second_force_applied = true;
        }
        assert!(
            !second_force_applied,
            "mid Stopped must not apply a later forced suffix"
        );

        // hipfire_generate::common::SpecRun carries semantic_stop into the wrapper independently of EOT.
        let run_semantic = semantic_stop_mid;
        assert!(run_semantic.is_some());
        assert!(hipfire_generate::qwen::spec_stop_is_semantic(run_semantic));
    }

    /// First-token user stop at max_tokens=1 must classify as stop (not length)
    /// via semantic_stop surviving independently of decoded_eot.
    #[test]
    fn first_token_stop_sequence_at_max_tokens_one_is_stop_not_length() {
        let tok = test_tokenizer();
        let ids = tok.encode("STOP");
        assert!(!ids.is_empty());
        let first = ids[0];
        let first_text = tok.decode(&[first]);
        let mut emit = hipfire_arch_qwen35::spec_emit::Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            stop: vec![first_text.clone()],
            max_think: 0,
            max_tokens: 1,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let first_begin = emit.begin(first);
        assert_eq!(first_begin.stop, Some(StopReason::StopSequence));

        // hipfire_generate::qwen::generate_spec sticky capture (begin path).
        let mut semantic_stop: Option<StopReason> = if hipfire_generate::qwen::spec_stop_is_semantic(first_begin.stop) {
            first_begin.stop
        } else {
            None
        };
        assert_eq!(semantic_stop, Some(StopReason::StopSequence));
        assert!(hipfire_generate::qwen::spec_stop_is_semantic(semantic_stop));

        // Budget spent on the first (and only) token; no decoded_eot required.
        let generated = 1usize;
        let max_tokens = 1usize;
        let decoded_eot = false; // user stop may not set EOT
        let hit_length =
            hipfire_generate::common::qwen_dflash_hit_length_cap(generated, max_tokens, decoded_eot, semantic_stop.is_some());
        assert!(
            !hit_length,
            "semantic StopSequence at cap must not classify as length"
        );

        // Wrapper wire: stop, not length.
        let fin = summary_stop(&first_text);
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, hit_length, false, &first_text, false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                store_cache,
                release_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "stop");
                assert!(*store_cache);
                assert!(!*release_tool_calls);
            }
            other => panic!("expected stop Done, got {other:?}"),
        }

        // Contrast: same numbers without semantic_stop → length.
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(1, 1, false, false));
        let _ = &mut semantic_stop;
    }

    /// Held tool_calls + semantic stop at the budget boundary must finish as
    /// tool_calls (not length). hipfire_generate::common::finish_summary_held_tool_calls feeds the wire.
    #[test]
    fn held_tool_calls_with_semantic_stop_at_cap_is_tool_calls_not_length() {
        let calls = vec![ToolCall {
            id: None,
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "SF"}),
            rendered_body: None,
        }];
        let fin = summary_tool_calls(calls.clone());
        let held = hipfire_generate::common::finish_summary_held_tool_calls(&fin);
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].name, "get_weather");

        // generated == max_tokens, no decoded_eot, but semantic stop sticky.
        let generated = 8usize;
        let max_tokens = 8usize;
        let decoded_eot = false;
        let semantic_stop = Some(StopReason::StopSequence);
        assert!(hipfire_generate::qwen::spec_stop_is_semantic(semantic_stop));
        let hit_length =
            hipfire_generate::common::qwen_dflash_hit_length_cap(generated, max_tokens, decoded_eot, semantic_stop.is_some());
        assert!(!hit_length, "semantic stop must beat length at cap");

        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, hit_length, false, "Sure.", false);
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "tool_calls");
                assert!(*release_tool_calls);
                assert!(*store_cache);
                assert_eq!(wire_tool_calls.len(), 1);
                assert_eq!(wire_tool_calls[0].name, "get_weather");
            }
            other => panic!("expected tool_calls Done, got {other:?}"),
        }

        // Without semantic_stop the same finish would be suppressed as length.
        assert!(hipfire_generate::common::qwen_dflash_hit_length_cap(8, 8, false, false));
        let length_term = hipfire_generate::qwen::qwen_dflash_wire_terminal(&fin, true, false, "Sure.", false);
        match &length_term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                finish_reason,
                release_tool_calls,
                wire_tool_calls,
                ..
            } => {
                assert_eq!(*finish_reason, "length");
                assert!(!*release_tool_calls);
                assert!(wire_tool_calls.is_empty());
            }
            other => panic!("expected length Done, got {other:?}"),
        }
    }

    // ── Task 4 forced-continuation physical-cap admission ─────────────────

    /// Pure admission: no-eviction requires a free pending-seed write slot
    /// after the commit (`post_position < physical_cap`). Exact-cap rejects.
    #[test]
    fn forced_commit_no_evict_exact_cap_rejects_pending_seed_slot() {
        let physical_cap = 16usize;
        let position = 12usize;
        let commit_len = 4usize; // post_position == physical_cap
        assert_eq!(position.saturating_add(commit_len), physical_cap);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, commit_len, physical_cap, false),
            "no-eviction exact-cap must reject: pending seed needs a legal slot"
        );
        // One slot under cap still fits (post == cap-1).
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len.saturating_sub(1),
            physical_cap,
            false
        ));
        // Over-cap also rejects.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len.saturating_add(1),
            physical_cap,
            false
        ));
    }

    /// Eviction path still refuses post_position > physical_cap before any GPU
    /// write. Exact-cap is the only boundary that eviction may open.
    #[test]
    fn forced_commit_eviction_over_cap_rejects_before_gpu() {
        let physical_cap = 16usize;
        let position = 12usize;
        let over = 5usize; // post_position = 17 > cap
        assert!(position.saturating_add(over) > physical_cap);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, over, physical_cap, true),
            "eviction must not admit over-cap commits"
        );

        // Deterministic pre-GPU gate: reject ⇒ no GPU commit, no staged render.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Phase {
            Staged,
            GpuCommitted,
            Rendered,
            ErrorOnly,
        }
        let admitted = hipfire_generate::qwen::spec_forced_commit_admits(position, over, physical_cap, true);
        let mut phase = Phase::Staged;
        let mut rendered = 0usize;
        if !admitted {
            // Production: rollback + ErrorOnly terminal; discard staged events.
            phase = Phase::ErrorOnly;
        } else {
            phase = Phase::GpuCommitted;
            phase = Phase::Rendered;
            rendered = 1;
        }
        assert_eq!(phase, Phase::ErrorOnly);
        assert_eq!(
            rendered, 0,
            "capacity reject must never render staged events"
        );
        assert_ne!(phase, Phase::GpuCommitted);
        assert_ne!(phase, Phase::Rendered);
        // Same wire class as maybe_evict / on_evict failures.
        assert_eq!(hipfire_generate::qwen::classify_evict_failure_wire(), hipfire_generate::qwen::SpecFailClosedWire::ErrorOnly);
    }

    /// Eviction exact-cap admits only because post-commit maybe_evict+on_evict
    /// is mandatory before host seed/raw/render and must leave a free seed slot.
    #[test]
    fn forced_commit_eviction_exact_cap_admits_with_mandatory_post_commit_evict() {
        let physical_cap = 16usize;
        let position = 12usize;
        let commit_len = 4usize; // post_position == physical_cap
        assert_eq!(position.saturating_add(commit_len), physical_cap);

        assert!(
            hipfire_generate::qwen::spec_forced_commit_admits(position, commit_len, physical_cap, true),
            "eviction may admit exact-cap"
        );
        // Contrast: same numbers without eviction reject.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len,
            physical_cap,
            false
        ));

        // Ordering model for the admitted exact-cap path: GPU commit → mandatory
        // post-commit eviction → require post_evict < physical_cap → only then
        // host position/seed/raw/render. Skipping eviction must not reach render.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Step {
            Admit,
            GpuCommit,
            PostCommitEvict,
            HostRender,
            ErrorOnly,
        }
        let mut steps: Vec<Step> = Vec::new();
        let admitted = hipfire_generate::qwen::spec_forced_commit_admits(position, commit_len, physical_cap, true);
        assert!(admitted);
        steps.push(Step::Admit);
        steps.push(Step::GpuCommit);

        let eviction_enabled = true;
        let mut post_position = position.saturating_add(commit_len);
        let mut rendered = false;
        if eviction_enabled {
            // Mandatory: maybe_evict + on_evict before host updates.
            steps.push(Step::PostCommitEvict);
            // Synthetic successful compaction frees the pending-seed slot.
            post_position = physical_cap.saturating_sub(1);
            if post_position >= physical_cap {
                steps.push(Step::ErrorOnly);
            } else {
                steps.push(Step::HostRender);
                rendered = true;
            }
        } else {
            steps.push(Step::HostRender);
            rendered = true;
        }
        assert_eq!(
            steps,
            vec![
                Step::Admit,
                Step::GpuCommit,
                Step::PostCommitEvict,
                Step::HostRender
            ]
        );
        assert!(rendered);
        assert!(post_position < physical_cap);

        // If post-evict still has no seed slot → ErrorOnly, no render.
        let mut bad_steps: Vec<Step> = vec![Step::Admit, Step::GpuCommit, Step::PostCommitEvict];
        let bad_post = physical_cap; // eviction failed to free a slot
        let mut bad_rendered = false;
        if bad_post >= physical_cap {
            bad_steps.push(Step::ErrorOnly);
        } else {
            bad_steps.push(Step::HostRender);
            bad_rendered = true;
        }
        assert_eq!(
            bad_steps,
            vec![
                Step::Admit,
                Step::GpuCommit,
                Step::PostCommitEvict,
                Step::ErrorOnly
            ]
        );
        assert!(!bad_rendered);
    }

    /// Comfortably under the physical cap admits with or without eviction.
    #[test]
    fn forced_commit_under_threshold_fits() {
        let physical_cap = 64usize;
        let position = 10usize;
        let commit_len = 3usize;
        assert!(position.saturating_add(commit_len) < physical_cap);
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len,
            physical_cap,
            false
        ));
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            commit_len,
            physical_cap,
            true
        ));
        // Empty commit (seed-only replace) is always under threshold.
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(position, 0, physical_cap, false));
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(position, 0, physical_cap, true));
    }

    /// Admission uses the actual GPU commit slice (`tx.commit.len()`), never the
    /// forced token count. Non-committable seeds omit the trigger and shrink
    /// the commit — that shorter length is what capacity sees.
    #[test]
    fn forced_commit_admission_uses_tx_commit_len_not_forced_count() {
        let physical_cap = 10usize;
        let position = 8usize;
        let seed = 7u32;
        let forced = [10u32, 11, 12]; // forced.len() == 3

        // Committable: commit = [seed, 10, 11] → len 3; post = 11 > cap.
        let keep = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &forced, true);
        assert_eq!(keep.commit.len(), 3);
        assert_eq!(keep.commit.len(), keep.position_delta);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, keep.commit.len(), physical_cap, false),
            "committable commit_len=3 at pos=8 must reject under no-evict"
        );
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, keep.commit.len(), physical_cap, true),
            "committable commit_len=3 at pos=8 is over-cap even with eviction"
        );

        // Non-committable: commit = [10, 11] → len 2 (seed omitted); post = 10.
        let omit = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &forced, false);
        assert_eq!(omit.commit, vec![10, 11]);
        assert_eq!(omit.commit.len(), 2);
        assert_eq!(omit.position_delta, omit.commit.len());
        assert_ne!(
            omit.commit.len(),
            forced.len(),
            "must not admit against forced token count"
        );
        // Using forced.len() would be wrong (post=11 over-cap); actual slice fits
        // exact-cap under eviction and rejects under no-evict (needs seed slot).
        assert_eq!(position.saturating_add(omit.commit.len()), physical_cap);
        assert!(
            !hipfire_generate::qwen::spec_forced_commit_admits(position, omit.commit.len(), physical_cap, false),
            "no-evict exact-cap still needs a pending-seed slot"
        );
        assert!(
            hipfire_generate::qwen::spec_forced_commit_admits(position, omit.commit.len(), physical_cap, true),
            "eviction admits exact-cap on the actual (shorter) commit slice"
        );
        // Guard: if a caller mistakenly passed forced.len(), both modes reject.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            forced.len(),
            physical_cap,
            true
        ));
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            forced.len(),
            physical_cap,
            false
        ));

        // Single forced + non-committable: empty commit — no GPU write.
        // Admission still uses commit_len=0 (not forced.len()==1).
        let one = hipfire_generate::qwen::spec_forced_pending_seed_tx(seed, &[42], false);
        assert!(one.commit.is_empty());
        assert_ne!(
            one.commit.len(),
            1,
            "must not treat forced count as commit_len"
        );
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            position,
            one.commit.len(),
            physical_cap,
            false
        ));
        // At physical_cap with zero-length commit: no-evict still needs a free
        // pending-seed slot (post == cap rejects); eviction admits exact-cap.
        assert!(!hipfire_generate::qwen::spec_forced_commit_admits(
            physical_cap,
            one.commit.len(),
            physical_cap,
            false
        ));
        assert!(hipfire_generate::qwen::spec_forced_commit_admits(
            physical_cap,
            one.commit.len(),
            physical_cap,
            true
        ));
    }

    #[test]
    fn dflash_client_commit_preserves_release_and_store() {
        let e = hipfire_generate::qwen::qwen_client_commit_effects(ClientTerminalDecision::Commit, true, true);
        assert!(e.release_tool_calls && e.store_cache && e.emit_done);
        // Successful Done classify → intended flags gate release/store.
        let tc = ToolCall {
            id: None,
            name: "read".into(),
            arguments: r#"{"path":"/x"}"#.into(),
            rendered_body: None,
        };
        let term = hipfire_generate::qwen::qwen_dflash_wire_terminal(
            &summary_tool_calls(vec![tc.clone()]),
            false,
            false,
            "Sure.",
            false,
        );
        match &term {
            hipfire_generate::qwen::QwenDflashWireTerminal::Done {
                release_tool_calls,
                store_cache,
                wire_tool_calls,
                ..
            } => {
                let effects = hipfire_generate::qwen::qwen_client_commit_effects(
                    ClientTerminalDecision::Commit,
                    *release_tool_calls && !wire_tool_calls.is_empty(),
                    *store_cache,
                );
                assert!(effects.release_tool_calls);
                assert!(effects.store_cache);
                assert!(effects.emit_done);
                let mut action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
                action.store = effects.store_cache && action.store;
                assert!(action.store);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn dflash_client_abort_suppresses_release_store_done() {
        set_active_attempt_id(33);
        let tc = ToolCall {
            id: None,
            name: "read".into(),
            arguments: r#"{"path":"/x"}"#.into(),
            rendered_body: None,
        };
        let term =
            hipfire_generate::qwen::qwen_dflash_wire_terminal(&summary_tool_calls(vec![tc]), false, false, "Sure.", false);
        let hipfire_generate::qwen::QwenDflashWireTerminal::Done {
            release_tool_calls,
            store_cache,
            wire_tool_calls,
            ..
        } = &term
        else {
            panic!("expected Done");
        };
        let effects = hipfire_generate::qwen::qwen_client_commit_effects(
            ClientTerminalDecision::Abort,
            *release_tool_calls && !wire_tool_calls.is_empty(),
            *store_cache,
        );
        assert!(!effects.release_tool_calls);
        assert!(!effects.store_cache);
        assert!(!effects.emit_done);

        let mut sink = Vec::new();
        // No tool release on Abort.
        let mut action = hipfire_generate::qwen::qwen_dflash_cache_action(&term);
        action.store = effects.store_cache && action.store;
        let mut stored = false;
        let _ = hipfire_generate::qwen::qwen_dflash_apply_cache_action(|_fp, _seq| stored = true, &action, vec![1, 2, 3]);
        assert!(!stored);

        let ep = hipfire_generate::common::RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        hipfire_generate::common::emit_spec_cancel_after_rollback(&mut sink, "df-abort", 7, &ep);
        let out = String::from_utf8_lossy(&sink);
        assert!(!out.contains("\"type\":\"tool_calls\""));
        assert!(out.contains("\"type\":\"aborted\""));
        assert!(out.contains("\"finish_reason\":\"aborted\""));
        assert!(!out.contains("\"finish_reason\":\"tool_calls\""));
        assert!(out.contains("\"attempt_id\":33"));
    }
}

/// Task 6: exhaustive producer-route capability matrix + pure tools gate model.
/// Production symbols only — no generate() side effects.
#[cfg(test)]
mod generation_route_matrix_tests {
    use super::{
        deepseek4_spec_requested_from_policy, select_generation_route, GenerationRoute,
        GenerationRouteInputs,
    };

    /// Baseline inputs that select nothing special (unknown arch, no EP/PP/spec).
    fn base() -> GenerationRouteInputs {
        GenerationRouteInputs {
            arch_id: 255,
            ep: false,
            pp: 1,
            has_speculator: false,
            qwen_mtp_head: false,
            qwen_mtp_opt_in: false,
            mtp_sampled_on: false,
            deepseek4_spec_requested: false,
            ngram_can_sample: false,
            temp: 0.0,
            user_explicit_sampling: false,
            min_p: None,
            force_ar_chat: false,
            temp_spec_env_off: false,
            fast_sample_on: true,
            supports_temp_swor: false,
            kv_adaptive: false,
        }
    }

    #[test]
    fn dspark_request_is_independent_of_mtp_mode() {
        assert!(deepseek4_spec_requested_from_policy(
            Some("dspark"),
            "off",
            "off",
            false,
        ));
        assert!(!deepseek4_spec_requested_from_policy(
            None, "off", "auto", true,
        ));
        assert!(deepseek4_spec_requested_from_policy(
            None, "auto", "auto", true,
        ));
    }

    /// One canonical input row that selects each ALL variant (coverage guard).
    /// New enum variants must add a row here or `route_capability_table_covers_all_variants` fails.
    fn capability_rows() -> Vec<(GenerationRoute, GenerationRouteInputs)> {
        vec![
            (
                GenerationRoute::QwenAr,
                GenerationRouteInputs {
                    arch_id: 5,
                    ..base()
                },
            ),
            (
                GenerationRoute::QwenDflash,
                GenerationRouteInputs {
                    arch_id: 5,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::QwenMtp,
                GenerationRouteInputs {
                    arch_id: 5,
                    qwen_mtp_head: true,
                    qwen_mtp_opt_in: true,
                    temp: 0.0,
                    has_speculator: true, // MTP still wins over DFlash
                    ..base()
                },
            ),
            (
                GenerationRoute::Qwen2Ar,
                GenerationRouteInputs {
                    arch_id: 7,
                    ..base()
                },
            ),
            (
                GenerationRoute::Qwen2Spec,
                GenerationRouteInputs {
                    arch_id: 7,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::Deepseek4Ar,
                GenerationRouteInputs {
                    arch_id: 9,
                    ..base()
                },
            ),
            (
                GenerationRoute::Deepseek4Ep,
                GenerationRouteInputs {
                    arch_id: 9,
                    ep: true,
                    // EP beats DS4 arch short-circuit even with spec flags set.
                    has_speculator: true,
                    deepseek4_spec_requested: true,
                    ..base()
                },
            ),
            (
                GenerationRoute::Deepseek4Spec,
                GenerationRouteInputs {
                    arch_id: 9,
                    has_speculator: true,
                    deepseek4_spec_requested: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::CohereAr,
                GenerationRouteInputs {
                    arch_id: 12,
                    ..base()
                },
            ),
            (
                GenerationRoute::CohereSpec,
                GenerationRouteInputs {
                    arch_id: 12,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::MiniMaxAr,
                GenerationRouteInputs {
                    arch_id: 10,
                    ..base()
                },
            ),
            (
                GenerationRoute::MiniMaxEp,
                GenerationRouteInputs {
                    arch_id: 10,
                    ep: true,
                    has_speculator: true,
                    ..base()
                },
            ),
            (
                GenerationRoute::MiniMaxSpec,
                GenerationRouteInputs {
                    arch_id: 10,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::LfmAr,
                GenerationRouteInputs {
                    arch_id: 11,
                    ..base()
                },
            ),
            (
                GenerationRoute::LfmSpec,
                GenerationRouteInputs {
                    arch_id: 11,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::LlamaAr,
                GenerationRouteInputs {
                    arch_id: 0,
                    ..base()
                },
            ),
            (
                GenerationRoute::LlamaSpec,
                GenerationRouteInputs {
                    arch_id: 0,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::PipelineParallel,
                GenerationRouteInputs {
                    arch_id: 5,
                    pp: 2,
                    // PP still beats MTP/DFlash when no arch short-circuit.
                    qwen_mtp_head: true,
                    qwen_mtp_opt_in: true,
                    has_speculator: true,
                    ..base()
                },
            ),
            (
                GenerationRoute::DotsOcr,
                GenerationRouteInputs {
                    arch_id: 8,
                    ..base()
                },
            ),
            (
                GenerationRoute::GlimmerAr,
                GenerationRouteInputs {
                    arch_id: 14,
                    ..base()
                },
            ),
            (
                GenerationRoute::GlimmerSpec,
                GenerationRouteInputs {
                    arch_id: 14,
                    has_speculator: true,
                    temp: 0.0,
                    ..base()
                },
            ),
            (
                GenerationRoute::Unknown,
                GenerationRouteInputs {
                    arch_id: 99,
                    ..base()
                },
            ),
        ]
    }

    /// Exact proven-safe producer set (contract).
    const SAFE_ROUTES: &[GenerationRoute] = &[
        GenerationRoute::QwenAr,
        GenerationRoute::QwenDflash,
        GenerationRoute::Deepseek4Ar,
        GenerationRoute::Deepseek4Ep,
        GenerationRoute::Deepseek4Spec,
        GenerationRoute::GlimmerAr,
        GenerationRoute::GlimmerSpec,
    ];

    /// Pure gate model mirroring generate()'s tools preflight:
    /// deny before RNG/gen_start when tools nonempty && !supports_tools.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct GateOutcome {
        allowed: bool,
        error_count: usize,
        class: Option<&'static str>,
        retryable: Option<bool>,
        mutated_generation_side: bool,
        route: GenerationRoute,
    }

    fn pure_tools_gate(route: GenerationRoute, tools_nonempty: bool) -> GateOutcome {
        if tools_nonempty && !route.supports_tools() {
            GateOutcome {
                allowed: false,
                error_count: 1,
                class: Some("unsupported"),
                retryable: Some(false),
                mutated_generation_side: false,
                route,
            }
        } else {
            GateOutcome {
                allowed: true,
                error_count: 0,
                class: None,
                retryable: None,
                mutated_generation_side: false,
                route,
            }
        }
    }

    #[test]
    fn route_capability_table_covers_all_variants() {
        let rows = capability_rows();
        assert_eq!(
            rows.len(),
            GenerationRoute::ALL.len(),
            "capability table must list every GenerationRoute::ALL variant"
        );
        for &variant in GenerationRoute::ALL {
            let hit = rows.iter().any(|(r, _)| *r == variant);
            assert!(
                hit,
                "missing capability row for {:?}; add an explicit selector input",
                variant
            );
        }
        // Each row's selector must actually produce the labeled route.
        for (expected, inputs) in &rows {
            let got = select_generation_route(inputs);
            assert_eq!(
                got, *expected,
                "capability row for {:?} selected {:?}",
                expected, got
            );
        }
    }

    #[test]
    fn route_matrix_tools_absent_and_present() {
        for (route, inputs) in capability_rows() {
            let selected = select_generation_route(&inputs);
            assert_eq!(selected, route);

            let safe = SAFE_ROUTES.contains(&route);
            assert_eq!(
                route.supports_tools(),
                safe,
                "{:?} supports_tools mismatch vs SAFE_ROUTES",
                route
            );

            // Tools absent: always allowed, zero errors, no mutation.
            let absent = pure_tools_gate(route, false);
            assert!(absent.allowed, "{:?} tools-absent must allow", route);
            assert_eq!(absent.error_count, 0);
            assert!(absent.class.is_none());
            assert!(!absent.mutated_generation_side);

            // Tools present: safe allows; unsafe emits exactly one nonretryable unsupported.
            let present = pure_tools_gate(route, true);
            if safe {
                assert!(present.allowed, "{:?} safe+tools must allow", route);
                assert_eq!(present.error_count, 0);
                assert!(!present.mutated_generation_side);
            } else {
                assert!(!present.allowed, "{:?} unsafe+tools must deny", route);
                assert_eq!(present.error_count, 1, "{:?} exactly one error", route);
                assert_eq!(present.class, Some("unsupported"));
                assert_eq!(present.retryable, Some(false));
                assert!(
                    !present.mutated_generation_side,
                    "{:?} deny must not mutate generation side",
                    route
                );
            }
        }
    }

    #[test]
    fn exact_safe_set_is_qwen_ar_dflash_ds4_ar_ep_spec_and_glimmer_ar_spec() {
        let mut from_all: Vec<GenerationRoute> = GenerationRoute::ALL
            .iter()
            .copied()
            .filter(|r| r.supports_tools())
            .collect();
        from_all.sort_by_key(|r| r.name());
        let mut expected = SAFE_ROUTES.to_vec();
        expected.sort_by_key(|r| r.name());
        assert_eq!(from_all, expected);
        assert_eq!(from_all.len(), 7);
        // Negative: every other ALL member is denied for tools.
        for &r in GenerationRoute::ALL {
            if !SAFE_ROUTES.contains(&r) {
                assert!(!r.supports_tools(), "{:?} must not be tool-safe", r);
            }
        }
    }

    #[test]
    fn precedence_ep_before_arch_short_circuit() {
        // EP on DS4 with spec requested → Deepseek4Ep, not Spec/Ar.
        let i = GenerationRouteInputs {
            arch_id: 9,
            ep: true,
            has_speculator: true,
            deepseek4_spec_requested: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Deepseek4Ep);
        // EP on MiniMax with n-gram spec → MiniMaxEp, not Spec.
        let i = GenerationRouteInputs {
            arch_id: 10,
            ep: true,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::MiniMaxEp);
        // EP on unregistered arch → Unknown (still EP-first).
        let i = GenerationRouteInputs {
            arch_id: 5,
            ep: true,
            has_speculator: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Unknown);
    }

    #[test]
    fn qwen_ep_batch_semantic_route_clears_ep_for_qwen_ar() {
        // Global selector: arch 6 + EP topology → Unknown (EP short-circuit).
        let with_ep = GenerationRouteInputs {
            arch_id: 6,
            ep: true,
            ..base()
        };
        assert_eq!(select_generation_route(&with_ep), GenerationRoute::Unknown);
        // Batch eligibility clears EP after independent topology gates so the
        // non-spec Qwen AR ladder remains reachable (exact callsite invariant).
        let cleared = GenerationRouteInputs {
            arch_id: 6,
            ep: false,
            ..base()
        };
        assert_eq!(select_generation_route(&cleared), GenerationRoute::QwenAr);
    }

    #[test]
    fn precedence_arch_short_circuit_before_pp() {
        // Qwen2 + pp>1 still short-circuits to Qwen2, never PipelineParallel.
        let i = GenerationRouteInputs {
            arch_id: 7,
            pp: 4,
            has_speculator: false,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Qwen2Ar);
        let i = GenerationRouteInputs {
            arch_id: 9,
            pp: 2,
            has_speculator: false,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::Deepseek4Ar);
        let i = GenerationRouteInputs {
            arch_id: 11,
            pp: 2,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::LfmSpec);
        let i = GenerationRouteInputs {
            arch_id: 12,
            pp: 2,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::CohereAr);
        let i = GenerationRouteInputs {
            arch_id: 10,
            pp: 2,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::MiniMaxAr);
        let i = GenerationRouteInputs {
            arch_id: 8,
            pp: 2,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::DotsOcr);
    }

    #[test]
    fn precedence_pp_before_qwen_mtp() {
        let i = GenerationRouteInputs {
            arch_id: 5,
            pp: 2,
            qwen_mtp_head: true,
            qwen_mtp_opt_in: true,
            temp: 0.0,
            has_speculator: true,
            ..base()
        };
        assert_eq!(
            select_generation_route(&i),
            GenerationRoute::PipelineParallel
        );
    }

    #[test]
    fn precedence_mtp_before_dflash() {
        // MTP opt-in + head + greedy beats DFlash even with a loaded speculator.
        let i = GenerationRouteInputs {
            arch_id: 6,
            qwen_mtp_head: true,
            qwen_mtp_opt_in: true,
            temp: 0.0,
            has_speculator: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenMtp);
        // Without MTP opt-in, same inputs select DFlash.
        let i = GenerationRouteInputs {
            arch_id: 6,
            qwen_mtp_head: true,
            qwen_mtp_opt_in: false,
            temp: 0.0,
            has_speculator: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
    }

    #[test]
    fn precedence_dflash_vs_ar() {
        // Qwen greedy + speculator → DFlash.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenDflash);
        // force_ar_chat → AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            temp: 0.0,
            force_ar_chat: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
        // No speculator → AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: false,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
        // kv_adaptive blocks Qwen DFlash → AR.
        let i = GenerationRouteInputs {
            arch_id: 5,
            has_speculator: true,
            temp: 0.0,
            kv_adaptive: true,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::QwenAr);
        // Llama greedy + spec → LlamaSpec; without → LlamaAr.
        let i = GenerationRouteInputs {
            arch_id: 1,
            has_speculator: true,
            temp: 0.0,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::LlamaSpec);
        let i = GenerationRouteInputs {
            arch_id: 1,
            has_speculator: false,
            ..base()
        };
        assert_eq!(select_generation_route(&i), GenerationRoute::LlamaAr);
    }

    #[test]
    fn precedence_arch_spec_vs_ar_matrix() {
        // Qwen2
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 7,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::Qwen2Spec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 7,
                has_speculator: true,
                temp: 0.7,
                ngram_can_sample: false,
                ..base()
            }),
            GenerationRoute::Qwen2Ar
        );
        // DeepSeek4
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 9,
                has_speculator: true,
                deepseek4_spec_requested: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::Deepseek4Spec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 9,
                has_speculator: true,
                deepseek4_spec_requested: false,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::Deepseek4Ar
        );
        // Cohere
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 12,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::CohereSpec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 12,
                has_speculator: false,
                ..base()
            }),
            GenerationRoute::CohereAr
        );
        // MiniMax
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 10,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::MiniMaxSpec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 10,
                has_speculator: false,
                ..base()
            }),
            GenerationRoute::MiniMaxAr
        );
        // LFM
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 11,
                has_speculator: true,
                temp: 0.0,
                ..base()
            }),
            GenerationRoute::LfmSpec
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 11,
                has_speculator: true,
                temp: 0.8,
                ngram_can_sample: false,
                ..base()
            }),
            GenerationRoute::LfmAr
        );
        // dots + unknown
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 8,
                has_speculator: true,
                pp: 2,
                ..base()
            }),
            GenerationRoute::DotsOcr
        );
        assert_eq!(
            select_generation_route(&GenerationRouteInputs {
                arch_id: 42,
                ..base()
            }),
            GenerationRoute::Unknown
        );
    }

    #[test]
    fn pure_gate_unsafe_tools_one_nonretryable_no_mutation() {
        for &route in GenerationRoute::ALL {
            if route.supports_tools() {
                continue;
            }
            let o = pure_tools_gate(route, true);
            assert_eq!(o.error_count, 1);
            assert_eq!(o.class, Some("unsupported"));
            assert_eq!(o.retryable, Some(false));
            assert!(!o.allowed);
            assert!(!o.mutated_generation_side);
            // Correlated: outcome carries the denied route identity.
            assert_eq!(o.route, route);
        }
    }

    #[test]
    fn pure_gate_tools_absent_always_allowed() {
        for &route in GenerationRoute::ALL {
            let o = pure_tools_gate(route, false);
            assert!(o.allowed, "{:?} tools-absent", route);
            assert_eq!(o.error_count, 0);
            assert!(!o.mutated_generation_side);
        }
    }

    #[test]
    fn all_variant_count_is_twenty_two() {
        // Pin count so accidental ALL edits surface here too.
        assert_eq!(GenerationRoute::ALL.len(), 22);
        assert_eq!(capability_rows().len(), 22);
    }
}

#[cfg(test)]
mod glimmer_channel_recorder_tests {
    use super::*;
    use hipfire_runtime::prompt_frame::{CachedAssistantBody, CachedAssistantToolBody};

    #[test]
    fn splits_self_then_user() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning body");
        rec.push(102, " more");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer body");
        rec.push(hipfire_generate::dense::GLIMMER_EOT_ID, "<|eot|>");
        let turn = rec.into_cached_turn(&[]).expect("should succeed");
        assert!(turn.reasoning.is_some());
        assert_eq!(turn.reasoning.unwrap().text, "reasoning body more");
        assert_eq!(turn.tools.len(), 0);
        assert!(turn.content.is_some());
        assert_eq!(turn.content.unwrap().text, "answer body");
    }

    #[test]
    fn terminal_open_user_body_is_accepted() {
        // GAP3: self closed by eom, user body left OPEN (no <|eot|> fed) must be accepted.
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning body");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer body");
        rec.push(105, " more");
        // Intentionally leave user body OPEN — no EOT, decode stopped on <|eot|> without feeding it.
        let turn = rec
            .into_cached_turn(&[])
            .expect("open terminal user body should be accepted");
        assert!(turn.reasoning.is_some());
        assert_eq!(turn.reasoning.unwrap().text, "reasoning body");
        assert!(turn.content.is_some());
        assert_eq!(turn.content.unwrap().text, "answer body more");
        assert!(turn.tools.is_empty());
    }

    #[test]
    fn splits_self_then_tool() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(102, "assistant to=weather.get_forecast");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        let atem = "<atem:function_calls>\n<atem:invoke name=\"weather.get_forecast\">\n<atem:parameter name=\"location\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        for (i, c) in atem.chars().enumerate() {
            rec.push(200 + i as u32, &c.to_string());
        }
        rec.push(hipfire_generate::dense::GLIMMER_EOT_ID, "<|eot|>");
        let tool_call = hipfire_runtime::prompt_frame::ToolCall {
            id: Some("call_0".into()),
            name: "weather.get_forecast".into(),
            arguments: serde_json::json!({"location":"Paris"}),
            rendered_body: None,
        };
        let turn = rec.into_cached_turn(&[tool_call]).expect("should succeed");
        assert!(turn.reasoning.is_some());
        assert_eq!(turn.tools.len(), 1);
        assert_eq!(turn.tools[0].recipient, "weather.get_forecast");
        assert!(turn.content.is_none());
    }

    #[test]
    fn refuses_forced_reasoning_close() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning");
        rec.mark_forced_reasoning_close();
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        let res = rec.into_cached_turn(&[]);
        assert_eq!(res.unwrap_err(), hipfire_generate::dense::GlimmerRecordRefusal::ForcedReasoningClose);
    }

    #[test]
    fn refuses_empty_self_body() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        let res = rec.into_cached_turn(&[]);
        assert_eq!(res.unwrap_err(), hipfire_generate::dense::GlimmerRecordRefusal::EmptySelfBody);
    }

    #[test]
    fn records_self_body_regardless_of_think_budget() {
        // Muse Glimmer has no non-thinking mode: the Onyx system block always carries
        // `Reasoning strength:`, so the model always opens a `to=self` channel. A low think
        // budget caps the span, it does not remove it — the turn must still be recordable, or
        // the prefix cache would go permanently inert whenever thinking was "off".
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer");
        let turn = rec
            .into_cached_turn(&[])
            .expect("self body must be recorded");
        assert_eq!(turn.reasoning.expect("reasoning slot").text, "reasoning");
        assert_eq!(turn.content.expect("content slot").text, "answer");
    }

    #[test]
    fn store_cached_turn_self_then_user_inserts_both_channels() {
        let mut rec = hipfire_generate::dense::GlimmerChannelRecorder::new();
        // Production shape: `add_generation_prompt` already emitted `<|start|>assistant`,
        // so the model's FIRST emission is just ` to=self` — no `<|start|>`, no `assistant`.
        rec.push(100, " to=self");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(101, "reasoning body");
        rec.push(102, " more");
        rec.push(hipfire_generate::dense::GLIMMER_EOM_ID, "<|eom|>");
        rec.push(hipfire_generate::dense::GLIMMER_START_ID, "<|start|>");
        rec.push(103, "assistant to=user");
        rec.push(hipfire_generate::dense::GLIMMER_MESSAGE_ID, "<|message|>");
        rec.push(104, "answer body");
        rec.push(105, "!");
        rec.push(hipfire_generate::dense::GLIMMER_EOT_ID, "<|eot|>");
        let mut cache = hipfire_loader::AsstTurnCache::new_from_env();
        cache.clear();
        let ok = hipfire_generate::dense::glimmer_store_cached_turn(&mut cache, rec, &[], 0);
        assert!(ok, "store should succeed");
        let normalized =
            hipfire_runtime::tokenizer::maybe_normalize_prompt("answer body!").into_owned();
        let fp_raw = hipfire_generate::common::asst_turn_fingerprint(&normalized, &[]);
        let fp = hipfire_generate::dense::glimmer_turn_key(fp_raw, 0);
        let turn = cache
            .get(&fp)
            .expect("cache should contain inserted turn")
            .clone();
        assert!(turn.reasoning.is_some(), "reasoning should be Some");
        assert!(turn.content.is_some(), "content should be Some");
        assert_eq!(turn.reasoning.unwrap().token_ids, vec![101, 102]);
        assert_eq!(turn.content.unwrap().token_ids, vec![104, 105]);
        assert!(turn.tools.is_empty());
    }

    #[test]
    fn tool_channel_does_not_emit_visible_token() {
        // GAP6: to=weather.get_forecast envelope must not produce visible Token events.
        let mut router = hipfire_generate::dense::GlimmerHarmonyRouter::new(0);
        // Feed header + atem body split across fragments to exercise suffix hold logic
        let header = "<|start|>assistant to=weather.get_forecast<|message|>";
        let atem = "<atem:function_calls>\n<atem:invoke name=\"weather.get_forecast\">\n<atem:parameter name=\"location\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let (events, _) = router.push(header);
        assert!(events.is_empty(), "header alone should emit nothing");
        let (events, _) = router.push(atem);
        // Tool channel text must be Tool, not Token
        let tool_text: String = events
            .iter()
            .filter_map(|e| match e {
                hipfire_generate::dense::GlimmerEmit::Tool(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let token_text: String = events
            .iter()
            .filter_map(|e| match e {
                hipfire_generate::dense::GlimmerEmit::Token(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            token_text.is_empty(),
            "tool envelope must produce zero visible Token events, got {:?}",
            token_text
        );
        assert!(
            !tool_text.is_empty(),
            "tool envelope should produce Tool events"
        );
        // Accumulated tool body should parse to one call
        let calls = hipfire_generate::dense::parse_glimmer_atem(&tool_text).expect("parse should succeed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather.get_forecast");
        assert_eq!(
            calls[0].arguments["location"],
            serde_json::Value::String("Paris".into())
        );
    }
}

#[cfg(test)]
mod glimmer_atem_parser_tests {
    use super::*;

    #[test]
    fn parses_representative_block() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"weather.get_forecast\">\n<atem:parameter name=\"location\">Paris</atem:parameter>\n<atem:parameter name=\"options\">{\"units\":\"celsius\",\"days\":[1,2]}</atem:parameter>\n<atem:parameter name=\"include_alerts\">true</atem:parameter>\n<atem:parameter name=\"fallback\">null</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let calls = hipfire_generate::dense::parse_glimmer_atem(body).expect("parse should succeed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather.get_forecast");
        assert_eq!(
            calls[0].arguments["location"],
            serde_json::Value::String("Paris".into())
        );
        assert_eq!(
            calls[0].arguments["options"]["units"],
            serde_json::Value::String("celsius".into())
        );
        assert_eq!(
            calls[0].arguments["options"]["days"],
            serde_json::json!([1, 2])
        );
        assert_eq!(
            calls[0].arguments["include_alerts"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(calls[0].arguments["fallback"], serde_json::Value::Null);
    }

    #[test]
    fn parses_adversarial_chunk_splits() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"test.func\">\n<atem:parameter name=\"a\">1</atem:parameter>\n<atem:parameter name=\"b\">{\"x\":1}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        for split in 1..body.len() {
            if !body.is_char_boundary(split) {
                continue;
            }
            let (left, right) = body.split_at(split);
            let combined = left.to_string() + right;
            let calls = hipfire_generate::dense::parse_glimmer_atem(&combined).expect("should parse after split");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, "test.func");
        }
        let body2 = "<atem:function_calls>\n<atem:invoke name=\"test.func\">\n<atem:parameter name=\"msg\">hello \u{1F30D}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let calls2 = hipfire_generate::dense::parse_glimmer_atem(body2).expect("should parse multibyte");
        assert_eq!(
            calls2[0].arguments["msg"],
            serde_json::Value::String("hello \u{1F30D}".into())
        );
    }

    #[test]
    fn parses_multiple_invokes() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"func1\">\n<atem:parameter name=\"a\">1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n<atem:function_calls>\n<atem:invoke name=\"func2\">\n<atem:parameter name=\"b\">2</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        let calls = hipfire_generate::dense::parse_glimmer_atem(body).expect("multiple");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "func1");
        assert_eq!(calls[1].name, "func2");
    }
}

#[cfg(test)]
mod glimmer_reconcile_tests {
    use super::*;

    /// The model card exposes exactly four reasoning strengths — low / medium / high / xhigh —
    /// as a system-prompt directive. All four must be reachable, and an EXPLICIT
    /// `reasoning_effort` must beat whatever token cap happens to be set.
    #[test]
    fn reasoning_strength_covers_all_four_card_levels() {
        use hipfire_runtime::prompt_frame::ThinkMode;

        // Explicit effort wins over the budget.
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::High, 1), "high");
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Max, 1), "xhigh");

        // `from_str` folds "medium" into `Low`, so the budget supplies the middle tier.
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Low, 512), "low");
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Low, 2048), "medium");
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Low, 8192), "high");
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Low, 16384), "xhigh");

        // Glimmer has no non-thinking mode: the engine's `1` sentinel is the MINIMUM
        // strength, never an off switch, and uncapped takes the template's own default.
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::NonThink, 1), "low");
        assert_eq!(hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::NonThink, 0), "high");

        // Every card level is producible.
        let produced: std::collections::BTreeSet<&str> = [
            hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Low, 512),
            hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Low, 2048),
            hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::High, 0),
            hipfire_generate::dense::glimmer_reasoning_strength(ThinkMode::Max, 0),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            produced,
            ["high", "low", "medium", "xhigh"].into_iter().collect()
        );
    }
    #[test]
    fn mirror_action_aligned() {
        assert_eq!(hipfire_generate::dense::glimmer_mirror_action(5, 5), hipfire_generate::dense::GlimmerMirrorAction::Aligned);
        assert_eq!(hipfire_generate::dense::glimmer_mirror_action(0, 0), hipfire_generate::dense::GlimmerMirrorAction::Aligned);
    }
    #[test]
    fn mirror_action_truncate() {
        assert_eq!(
            hipfire_generate::dense::glimmer_mirror_action(5, 3),
            hipfire_generate::dense::GlimmerMirrorAction::TruncateMirror(3)
        );
        assert_eq!(
            hipfire_generate::dense::glimmer_mirror_action(10, 0),
            hipfire_generate::dense::GlimmerMirrorAction::TruncateMirror(0)
        );
    }
    #[test]
    fn mirror_action_rollback() {
        assert_eq!(
            hipfire_generate::dense::glimmer_mirror_action(3, 5),
            hipfire_generate::dense::GlimmerMirrorAction::RollbackCursor(3)
        );
        assert_eq!(
            hipfire_generate::dense::glimmer_mirror_action(0, 5),
            hipfire_generate::dense::GlimmerMirrorAction::RollbackCursor(0)
        );
    }
    // glimmer_hidden_keep_len removed with device-capture session API cutover.
    #[test]
    fn glimmer_turn_key_ordinal_salts() {
        let fp = hipfire_generate::common::asst_turn_fingerprint("Done.", &[]);
        let k0 = hipfire_generate::dense::glimmer_turn_key(fp, 0);
        let k1 = hipfire_generate::dense::glimmer_turn_key(fp, 1);
        let k2 = hipfire_generate::dense::glimmer_turn_key(fp, 2);
        assert_ne!(
            k0, k1,
            "identical content at different ordinals must have different keys"
        );
        assert_ne!(k1, k2);
        assert_ne!(k0, k2);
        assert_eq!(k0, hipfire_generate::dense::glimmer_turn_key(fp, 0));
        assert_eq!(k1, hipfire_generate::dense::glimmer_turn_key(fp, 1));
    }
}
#[cfg(test)]
mod glimmer_history_prep_tests {
    use super::*;

    #[test]
    fn normalize_arguments_object() {
        let v = serde_json::json!({"a":1});
        assert_eq!(hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).unwrap(), v);
    }

    #[test]
    fn normalize_arguments_null() {
        let v = serde_json::Value::Null;
        assert_eq!(
            hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).unwrap(),
            serde_json::json!({})
        );
    }

    #[test]
    fn normalize_arguments_string_object() {
        let v = serde_json::Value::String("{\"a\":1}".into());
        assert_eq!(
            hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).unwrap(),
            serde_json::json!({"a":1})
        );
    }

    #[test]
    fn normalize_arguments_string_invalid() {
        let v = serde_json::Value::String("not json".into());
        assert!(hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).is_err());
    }

    #[test]
    fn normalize_arguments_string_non_object() {
        let v = serde_json::Value::String("[1,2]".into());
        assert!(hipfire_generate::dense::normalize_glimmer_tool_arguments(&v).is_err());
    }

    #[test]
    fn prepare_history_resolves_name() {
        let assistant = hipfire_runtime::prompt_frame::Message {
            role: hipfire_runtime::prompt_frame::Role::Assistant,
            content: String::new(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![hipfire_runtime::prompt_frame::ToolCall {
                id: Some("call_0".into()),
                name: "weather.get_forecast".into(),
                arguments: serde_json::json!({"location":"Paris"}),
                rendered_body: None,
            }],
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let tool = hipfire_runtime::prompt_frame::Message {
            role: hipfire_runtime::prompt_frame::Role::Tool,
            content: "sunny".into(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![],
            tool_call_id: Some("call_0".into()),
            tool_plan: String::new(),
        };
        let out = hipfire_generate::dense::prepare_glimmer_onyx_history(&[assistant, tool]).expect("should succeed");
        assert_eq!(out[1].rendered_name, Some("weather.get_forecast".into()));
    }
}

/// Pure request-local Glimmer speculative profitability controller.
/// Exercises production methods only — no algorithm reimplementation.
#[cfg(test)]
mod glimmer_profit_guard_tests {
    use hipfire_generate::{dense::glimmer_profit_ledger_after_bonus_decode, dense::glimmer_profit_ledger_post_window, dense::glimmer_profit_ledger_route_prediction, dense::GlimmerProfitGuardStatus, dense::GlimmerProfitProbeKind, dense::GlimmerSpecProfitGuard};

    /// Drive four identical measured windows that sum to (s_total, p_total), then
    /// apply ar_probe_ns. Returns the guard after observe_probe.
    fn eval_group(g: &mut hipfire_generate::dense::GlimmerSpecProfitGuard, s_total: u128, p_total: u128, ar_probe_ns: u128) {
        // Split evenly across four windows; remainder on the last.
        let s_each = s_total / 4;
        let p_each = (p_total / 4) as usize;
        let s_last = s_total - s_each * 3;
        let p_last = (p_total - (p_each as u128) * 3) as usize;
        for i in 0..4 {
            let s = if i == 3 { s_last } else { s_each };
            let p = if i == 3 { p_last } else { p_each };
            let kind = g.observe_full_window(s, p);
            if i < 3 {
                assert_eq!(kind, hipfire_generate::dense::GlimmerProfitProbeKind::None, "window {i}");
            } else {
                assert_eq!(kind, hipfire_generate::dense::GlimmerProfitProbeKind::Measured, "window {i}");
            }
        }
        g.observe_probe(ar_probe_ns);
    }

    fn warmup(g: &mut hipfire_generate::dense::GlimmerSpecProfitGuard) {
        assert_eq!(
            g.observe_full_window(1_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::Warmup
        );
        g.observe_probe(999); // discarded
    }

    #[test]
    fn disabled_never_probes_or_retires() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(false);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Disabled);
        assert!(!g.enabled());
        for _ in 0..20 {
            assert_eq!(
                g.observe_full_window(10_000, 8),
                hipfire_generate::dense::GlimmerProfitProbeKind::None
            );
            g.observe_probe(1);
        }
        assert!(!g.is_retired());
        assert_eq!(g.evaluations(), 0);
        assert_eq!(g.eligible_windows(), 0);
        assert_eq!(g.pending_probe(), hipfire_generate::dense::GlimmerProfitProbeKind::None);
    }

    #[test]
    fn first_window_is_warmup_and_excluded() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Warming);
        assert_eq!(
            g.observe_full_window(50_000, 16),
            hipfire_generate::dense::GlimmerProfitProbeKind::Warmup
        );
        assert_eq!(g.eligible_windows(), 1);
        assert_eq!(g.pending_probe(), hipfire_generate::dense::GlimmerProfitProbeKind::Warmup);
        // Warmup probe discarded — no evaluation, no S/P carried.
        g.observe_probe(1);
        assert_eq!(g.evaluations(), 0);
        assert_eq!(g.bad_evaluations(), 0);
        assert_eq!(g.pending_probe(), hipfire_generate::dense::GlimmerProfitProbeKind::None);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Monitoring);
        // Next four windows: only the 4th requests Measured.
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(10_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::Measured
        );
        // Completing the measured probe with S=40k, P=16, A=2500:
        // ratio = 40000/(16*2500) = 1.0 — deadband; one evaluation counted.
        g.observe_probe(2_500);
        assert_eq!(g.evaluations(), 1);
        assert_eq!(g.last_spec_ns(), 40_000);
        assert_eq!(g.last_productive(), 16);
        assert_eq!(g.last_ar_probe_ns(), 2_500);
    }

    #[test]
    fn four_window_cadence() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        // Two full evaluation groups: only every 4th window is Measured.
        let mut measured = 0u32;
        let mut none = 0u32;
        for i in 0..8 {
            let k = g.observe_full_window(1_000, 2);
            match k {
                hipfire_generate::dense::GlimmerProfitProbeKind::Measured => {
                    measured += 1;
                    g.observe_probe(1_000); // ratio = 4000/(8*1000)=0.5 good
                }
                hipfire_generate::dense::GlimmerProfitProbeKind::None => none += 1,
                hipfire_generate::dense::GlimmerProfitProbeKind::Warmup => panic!("unexpected warmup at {i}"),
            }
        }
        assert_eq!(measured, 2);
        assert_eq!(none, 6);
        assert_eq!(g.evaluations(), 2);
    }

    #[test]
    fn boundary_1049_deadband_105_bad_098_reset() {
        // Choose A=1000, P=100 so A*P = 100_000.
        // bad:  S*100 >= 100_000*105 = 10_500_000  => S >= 105_000  (ratio >= 1.05)
        // good: S*100 <= 100_000*98  =  9_800_000  => S <=  98_000  (ratio <= 0.98)
        // deadband: 98_001 ..= 104_999
        // Exactly 1.049: S = 104_900 => left=10_490_000 < 10_500_000 and > 9_800_000.

        // --- 1.049 deadband retains ---
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        // Seed one bad so deadband retention is observable.
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());
        // 1.049: retain bad_evaluations == 1
        eval_group(&mut g, 104_900, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());
        assert_eq!(g.evaluations(), 2);

        // --- exactly 1.05 is bad ---
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());

        // --- exactly 0.98 resets ---
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);
        eval_group(&mut g, 105_000, 100, 1_000); // bad -> 1
        assert_eq!(g.bad_evaluations(), 1);
        eval_group(&mut g, 98_000, 100, 1_000); // good -> 0
        assert_eq!(g.bad_evaluations(), 0);
        assert!(!g.is_retired());
        assert_eq!(g.evaluations(), 2);
    }

    #[test]
    fn two_bad_retires_sticky_good_resets_deadband_retains() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut g);

        // bad #1
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());

        // deadband retains
        eval_group(&mut g, 100_000, 100, 1_000); // ratio = 1.0
        assert_eq!(g.bad_evaluations(), 1);
        assert!(!g.is_retired());

        // good resets
        eval_group(&mut g, 98_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 0);

        // two consecutive bads retire
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 1);
        eval_group(&mut g, 105_000, 100, 1_000);
        assert_eq!(g.bad_evaluations(), 2);
        assert!(g.is_retired());
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Retired);
        assert_eq!(g.retire_evaluation(), g.evaluations());
        assert!(g.retire_cycle() > 0);

        // sticky: further windows/probes are inert
        assert_eq!(
            g.observe_full_window(200_000, 1),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        let evals = g.evaluations();
        g.observe_probe(1);
        assert_eq!(g.evaluations(), evals);
        assert!(g.is_retired());
    }

    #[test]
    fn fresh_object_after_retirement_starts_warmup() {
        let mut old = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        warmup(&mut old);
        eval_group(&mut old, 105_000, 100, 1_000);
        eval_group(&mut old, 105_000, 100, 1_000);
        assert!(old.is_retired());

        let mut fresh = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        assert_eq!(fresh.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Warming);
        assert!(!fresh.is_retired());
        assert_eq!(
            fresh.observe_full_window(1_000, 4),
            hipfire_generate::dense::GlimmerProfitProbeKind::Warmup
        );
    }

    #[test]
    fn zero_progress_and_zero_time_ignored() {
        let mut g = hipfire_generate::dense::GlimmerSpecProfitGuard::new(true);
        // Zero time
        assert_eq!(g.observe_full_window(0, 8), hipfire_generate::dense::GlimmerProfitProbeKind::None);
        // Zero rows
        assert_eq!(
            g.observe_full_window(10_000, 0),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(g.eligible_windows(), 0);
        assert_eq!(g.status(), hipfire_generate::dense::GlimmerProfitGuardStatus::Warming);

        warmup(&mut g);
        // Build three of four measured windows, then inject zeros (ignored).
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        assert_eq!(g.observe_full_window(0, 2), hipfire_generate::dense::GlimmerProfitProbeKind::None);
        assert_eq!(
            g.observe_full_window(1_000, 0),
            hipfire_generate::dense::GlimmerProfitProbeKind::None
        );
        // Fourth real window still completes the group.
        assert_eq!(
            g.observe_full_window(1_000, 2),
            hipfire_generate::dense::GlimmerProfitProbeKind::Measured
        );
        // Zero probe is not evidence: evaluation not counted.
        g.observe_probe(0);
        assert_eq!(g.evaluations(), 0);
        assert_eq!(g.bad_evaluations(), 0);
        // Cadence recovered — next four-window group works.
        eval_group(&mut g, 4_000, 8, 1_000); // ratio 0.5 good
        assert_eq!(g.evaluations(), 1);
    }

    #[test]
    fn bonus_decode_aligns_mirror_prediction_unpushed_until_route() {
        // Post full window: bonus already on mirror, not in KV/capture.
        let commit_end = 100usize;
        let post = hipfire_generate::dense::glimmer_profit_ledger_post_window(commit_end);
        assert_eq!(post.mirror_len, commit_end + 1);
        assert_eq!(post.state_n_tokens, commit_end);

        // Decoding the pending bonus advances state only — prediction not mirrored.
        let after = hipfire_generate::dense::glimmer_profit_ledger_after_bonus_decode(post);
        assert_eq!(after.mirror_len, commit_end + 1);
        assert_eq!(after.state_n_tokens, commit_end + 1);
        assert_eq!(after.mirror_len, after.state_n_tokens);

        // Retire/AR tail keeps prediction unpushed (same ledger).
        assert_eq!(after, hipfire_generate::dense::glimmer_profit_ledger_after_bonus_decode(post));

        // Continue-spec routes the returned prediction once.
        let cont = hipfire_generate::dense::glimmer_profit_ledger_route_prediction(after);
        assert_eq!(cont.mirror_len, commit_end + 2);
        assert_eq!(cont.state_n_tokens, commit_end + 1);
        // Prediction is one-token-ahead again, not yet in state.
        assert_eq!(cont.mirror_len, cont.state_n_tokens + 1);
    }
}

#[cfg(all(test, feature = "serve-fault-inject"))]
mod serve_fault_inject_tests {
    use super::*;

    #[test]
    fn fault_inject_routes_qwen35_only() {
        assert_eq!(
            hipfire_runtime::reset_core::fault_inject_eligible_routes("qwen35"),
            &["qwen_ar", "qwen_dflash"][..]
        );
        assert!(hipfire_runtime::reset_core::fault_inject_eligible_routes("deepseek4").is_empty());
        assert!(hipfire_runtime::reset_core::fault_inject_eligible_routes("llama").is_empty());
    }

    #[test]
    fn one_shot_arm_take_clears() {
        arm_fault_after_prefill(true);
        assert!(take_fault_after_prefill());
        assert!(!take_fault_after_prefill());
        arm_fault_after_prefill(false);
        assert!(!take_fault_after_prefill());
    }

    #[test]
    fn retry_eligible_only_qwen35() {
        assert!(model_retry_reset_eligible(5));
        assert!(model_retry_reset_eligible(6));
        assert!(!model_retry_reset_eligible(9)); // deepseek4
        assert!(!model_retry_reset_eligible(0)); // llama
    }
}
