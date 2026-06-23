// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 decode: the per-token / batched decode-step kernels and the
//! batch-decode driver, plus the decode-capability/signature validators the
//! request loop runs before admitting a batched-decode request.
//!
//! Covers the fused-dense and grouped-MoE native/layer-chunked decode kernels,
//! the serial reference, chunk-range planning, decode scratch allocation, and
//! the runtime-surface / session-signature / model-capability validators.
//! Extracted verbatim from the former `main.rs` monolith (no behavior change);
//! items called from `main.rs` are `pub`.

use std::io::Write;
use std::time::Instant;

use hipfire_arch_qwen35::qwen35;
use hipfire_generate::{
    qwen35_decode_batch_requested_auto, qwen35_decode_batch_scheduler_metadata,
    qwen35_generate_batch_decode_step_done_json,
    qwen35_grouped_moe_decode_auto_latency_gate_passed, select_qwen35_decode_batch_backend,
    GenerateBatchDecodeEnvelope, GenerateBatchDecodeSession, GenerateBatchPrefillEnvelope,
    Qwen35DecodeBatchBackend, Qwen35DecodeBatchStepResult, Qwen35DecodeTokenOutcome,
};
use hipfire_model::{is_qwen35_dense_arch_id, is_qwen35_family_arch_id, is_qwen35_moe_arch_id};
use hipfire_state::model_worker_runtime_view_json;

use crate::model::LoadedModel;
use crate::session::{
    loaded_model_state_arena_backend, loaded_model_worker_runtime_view, qwen35_activate_session,
    qwen35_restore_or_error, qwen35_save_active_session,
    sequence_state_arena_resident_session_count, Qwen35RequestSessionState,
};

// ── Batch-decode admission validators ──────────────────────────────────────
// Run before a `generate_batch_decode_step` request is admitted: confirm the
// loaded model's arch/pp/KV/state and the per-session state signatures match
// what the fused-dense / grouped-MoE decode kernels require, else fall back to
// the serial path or reject with a clear error.

/// Gate: batched decode is single-GPU qwen35/qwen35-moe only, and incompatible
/// with DFlash or active eviction.
pub fn validate_qwen35_decode_batch_runtime_surface(
    arch_id: u32,
    pp: usize,
    dflash_loaded: bool,
    eviction_active: bool,
) -> Result<(), String> {
    if !is_qwen35_family_arch_id(arch_id) || pp != 1 {
        return Err(format!(
            "generate_batch_decode_step currently supports single-GPU qwen35/qwen35-moe only (arch_id={arch_id} pp={pp})"
        ));
    }
    if dflash_loaded {
        return Err(
            "generate_batch_decode_step is not supported on DFlash-loaded models".to_string(),
        );
    }
    if eviction_active {
        return Err(
            "generate_batch_decode_step is not supported with active eviction state".to_string(),
        );
    }
    Ok(())
}

/// Capture a session's KV/DeltaNet quant state signature, compared across
/// sessions to confirm a batch can share the fused kernel.
pub fn qwen35_fused_dense_decode_signature(
    state: &Qwen35RequestSessionState,
) -> qwen35::DensePrefillSessionBatchStateSignature {
    qwen35::DensePrefillSessionBatchStateSignature {
        kv_physical_cap: state.kv_cache().physical_cap,
        kv_compact_offset: state.kv_cache().compact_offset,
        kv_quantized: state.kv_cache().quantized,
        kv_quant_q8: state.kv_cache().quant_q8,
        kv_quant_asym2: state.kv_cache().quant_asym2,
        kv_quant_asym3: state.kv_cache().quant_asym3,
        kv_quant_asym4: state.kv_cache().quant_asym4,
        kv_quant_fwht: state.kv_cache().quant_fwht,
        dn_quant: state.dn_state().quant,
    }
}

/// Validate that a batch of dense-decode session signatures satisfies the
/// fused-prefix full-precision contract.
pub fn validate_qwen35_fused_dense_decode_session_signatures(
    config: &qwen35::Qwen35Config,
    signatures: &[qwen35::DensePrefillSessionBatchStateSignature],
    session_count: usize,
) -> Result<(), String> {
    let execution_plan = qwen35::DensePrefillSessionBatchExecutionPlan {
        rounds: Vec::new(),
        state_routes: Vec::new(),
        total_rows: session_count,
        max_rows_per_round: session_count,
        multi_state_rounds: 1,
        multi_state_prefix_rounds: 1,
        multi_state_prefix_rows: session_count,
        singleton_tail: None,
    };
    qwen35::validate_dense_prefill_session_batch_fused_prefix_full_precision_contract(
        config,
        signatures,
        &execution_plan,
    )
}

/// Validate a batch of grouped-MoE decode session signatures against the q8
/// state contract.
pub fn validate_qwen35_grouped_moe_decode_session_signatures(
    config: &qwen35::Qwen35Config,
    signatures: &[qwen35::DensePrefillSessionBatchStateSignature],
    session_count: usize,
    arch: &str,
) -> Result<(), String> {
    let execution_plan = qwen35::DensePrefillSessionBatchExecutionPlan {
        rounds: Vec::new(),
        state_routes: Vec::new(),
        total_rows: session_count,
        max_rows_per_round: session_count,
        multi_state_rounds: 1,
        multi_state_prefix_rounds: 1,
        multi_state_prefix_rows: session_count,
        singleton_tail: None,
    };
    qwen35::validate_grouped_moe_prefill_session_batch_q8_state_contract(
        config,
        signatures,
        &execution_plan,
        arch,
    )
}

/// Confirm the loaded model is dense Qwen35 (arch_id 5) with FP32 KV + DeltaNet
/// state and fused-kernel-compatible weights — the model-level gate for the
/// fused-dense decode path.
pub fn validate_qwen35_fused_dense_decode_model_capability(
    m: &LoadedModel,
    session_count: usize,
) -> Result<(), String> {
    if !is_qwen35_dense_arch_id(m.arch_id) {
        return Err(format!(
            "qwen35 fused dense decode requires dense Qwen35 arch_id=5; loaded arch_id={}",
            m.arch_id
        ));
    }
    let _ = session_count;
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 fused dense decode requires qwen35 config".to_string())?;
    if config.num_experts != 0 || config.has_shared_expert {
        return Err(
            "qwen35 fused dense decode supports dense Qwen35 only; grouped-MoE stays serial_reference"
                .to_string(),
        );
    }
    let kv_mode = m
        .q35_kv_mode
        .as_deref()
        .ok_or_else(|| "qwen35 fused dense decode requires known KV mode".to_string())?;
    if !matches!(kv_mode, "fp32" | "f32") {
        return Err(format!(
            "qwen35 fused dense decode requires FP32 KV state; loaded kv_mode={kv_mode}; use HIPFIRE_QWEN35_DECODE_BATCH=serial"
        ));
    }
    let state_quant = m.q35_state_quant.ok_or_else(|| {
        "qwen35 fused dense decode requires known DeltaNet state quant".to_string()
    })?;
    if state_quant != qwen35::StateQuant::FP32 {
        return Err(format!(
            "qwen35 fused dense decode requires FP32 DeltaNet state; loaded state={state_quant:?}; use HIPFIRE_QWEN35_DECODE_BATCH=serial"
        ));
    }
    let weights = m
        .q35_weights
        .as_ref()
        .ok_or_else(|| "qwen35 fused dense decode requires qwen35 weights".to_string())?;
    qwen35::validate_dense_prefill_session_batch_fused_prefix_full_precision_weights(weights)
        .map_err(|e| format!("qwen35 fused dense decode unsupported weights: {e}"))?;
    if m.q35_scratch.is_none() {
        return Err("qwen35 fused dense decode requires single-GPU qwen35 scratch".to_string());
    }
    Ok(())
}

/// Model-level gate for the grouped-MoE decode path: Qwen35 MoE (arch_id 6),
/// ≥2 sessions, grouped-MoE config + scratch present.
pub fn validate_qwen35_grouped_moe_decode_model_capability(
    m: &LoadedModel,
    session_count: usize,
    arch: &str,
) -> Result<(), String> {
    if !is_qwen35_moe_arch_id(m.arch_id) {
        return Err(format!(
            "qwen35 grouped-MoE decode requires Qwen35 MoE arch_id=6; loaded arch_id={}",
            m.arch_id
        ));
    }
    if session_count < 2 {
        return Err("qwen35 grouped-MoE decode requires at least two sessions".to_string());
    }
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 grouped-MoE decode requires qwen35 config".to_string())?;
    if config.num_experts == 0 || !config.has_shared_expert {
        return Err("qwen35 grouped-MoE decode requires a grouped-MoE Qwen35 config".to_string());
    }
    if m.q35_scratch.is_none() {
        return Err("qwen35 grouped-MoE decode requires single-GPU qwen35 scratch".to_string());
    }
    let signatures = vec![
        qwen35::DensePrefillSessionBatchStateSignature {
            kv_physical_cap: 128,
            kv_compact_offset: 0,
            kv_quantized: true,
            kv_quant_q8: true,
            kv_quant_asym2: false,
            kv_quant_asym3: false,
            kv_quant_asym4: false,
            kv_quant_fwht: false,
            dn_quant: qwen35::StateQuant::Q8,
        };
        session_count
    ];
    validate_qwen35_grouped_moe_decode_session_signatures(config, &signatures, session_count, arch)
        .map_err(|e| format!("qwen35 grouped-MoE decode unsupported model contract: {e}"))?;
    Ok(())
}

/// Confirm every session named in the decode request is actually resident
/// before stepping the batch.
pub fn validate_qwen35_decode_resident_sessions(
    m: &LoadedModel,
    envelope: &GenerateBatchDecodeEnvelope,
    backend_label: &str,
) -> Result<(), String> {
    for session in &envelope.sessions {
        let state = m.q35_sessions.get(&session.session_id).ok_or_else(|| {
            format!(
                "decode session {} is not resident for {backend_label} decode",
                session.session_id
            )
        })?;
        let logical_position = state.seq_pos + state.kv_cache().compact_offset;
        if logical_position != session.logical_position {
            return Err(format!(
                "decode session {} logical_position mismatch: expected={} resident={}",
                session.session_id, session.logical_position, logical_position
            ));
        }
    }
    Ok(())
}

/// Resident-session check specialized for the fused-dense path (also collects
/// the per-session signatures for the contract validators).
pub fn validate_qwen35_fused_dense_decode_resident_sessions(
    m: &LoadedModel,
    envelope: &GenerateBatchDecodeEnvelope,
) -> Result<(), String> {
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 fused dense decode requires qwen35 config".to_string())?;
    let mut signatures = Vec::with_capacity(envelope.sessions.len());
    for session in &envelope.sessions {
        let state = m.q35_sessions.get(&session.session_id).ok_or_else(|| {
            format!(
                "decode session {} is not resident for fused dense decode",
                session.session_id
            )
        })?;
        let logical_position = state.seq_pos + state.kv_cache().compact_offset;
        if logical_position != session.logical_position {
            return Err(format!(
                "decode session {} logical_position mismatch: expected={} resident={}",
                session.session_id, session.logical_position, logical_position
            ));
        }
        signatures.push(qwen35_fused_dense_decode_signature(state));
    }
    if signatures.len() == 1 {
        signatures.push(signatures[0]);
    }
    validate_qwen35_fused_dense_decode_session_signatures(config, &signatures, signatures.len())
        .map_err(|e| format!("qwen35 fused dense decode unsupported resident state: {e}"))
}

/// Emit the `generate_batch_prefill_ready` capability event for the real
/// (non-dummy) qwen35 backend.
pub fn emit_generate_batch_prefill_ready(
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
) {
    let line = serde_json::json!({
        "type": "generate_batch_prefill_ready",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "supported": true,
        "mode": "qwen35_serial_exact_token_prefill",
        "reason": "qwen35_serial_exact_token_prefill_available",
    });
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

/// Emit a `generate_batch_prefill_ready` event reporting the request is
/// unsupported, with the reason.
pub fn emit_generate_batch_prefill_unsupported(
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
    reason: &str,
) {
    let line = serde_json::json!({
        "type": "generate_batch_prefill_unsupported",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "supported": false,
        "reason": reason,
    });
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}
/// Drive one `generate_batch_decode_step`: validate + admit the batch, select
/// the decode backend (fused-dense / grouped-MoE native or layer-chunked, or
/// serial reference), step every resident session one token, and emit the
/// per-session outcomes + the batch done event. The multi-session decode entry
/// point the request loop calls.
pub fn run_generate_batch_decode_step_qwen35(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchDecodeEnvelope,
) -> Result<(), String> {
    validate_qwen35_decode_batch_runtime_surface(
        m.arch_id,
        m.pp,
        m.dflash.is_some(),
        m.eviction.is_some(),
    )?;
    let requested_backend =
        std::env::var("HIPFIRE_QWEN35_DECODE_BATCH").unwrap_or_else(|_| "auto".to_string());
    let mut backend = select_qwen35_decode_batch_backend(
        requested_backend.as_str(),
        m.arch_id,
        envelope.session_count,
    )?;
    if qwen35_decode_batch_requested_auto(requested_backend.as_str())
        && m.arch_id == 5
        && envelope.session_count >= 2
    {
        qwen35_save_active_session(m, gpu)?;
    }
    if qwen35_decode_batch_requested_auto(requested_backend.as_str())
        && m.arch_id == 6
        && qwen35_grouped_moe_decode_auto_latency_gate_passed(envelope.session_count)
    {
        qwen35_save_active_session(m, gpu)?;
    }
    if qwen35_decode_batch_requested_auto(requested_backend.as_str())
        && m.arch_id == 5
        && envelope.session_count >= 2
        && validate_qwen35_fused_dense_decode_model_capability(m, envelope.session_count).is_ok()
        && validate_qwen35_fused_dense_decode_resident_sessions(m, envelope).is_ok()
    {
        backend = Qwen35DecodeBatchBackend::FusedDenseLayerChunked;
    }
    if qwen35_decode_batch_requested_auto(requested_backend.as_str())
        && m.arch_id == 6
        && qwen35_grouped_moe_decode_auto_latency_gate_passed(envelope.session_count)
        && validate_qwen35_grouped_moe_decode_model_capability(
            m,
            envelope.session_count,
            gpu.arch.as_str(),
        )
        .is_ok()
        && validate_qwen35_decode_resident_sessions(m, envelope, "grouped-MoE auto").is_ok()
    {
        backend = Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked;
    }
    // Hierarchical KV is a per-token-attention feature (hot-ring read lives in
    // kv_cache_attention_dispatch). The fused layer-chunked decode backends run
    // their own batched attention and bypass it → the two-tier cache is never read
    // and decode degenerates. Force the SerialReference decode (per-token
    // forward_scratch per session), which honours the dispatch. Per-token decode is
    // slower but correct; hier is a KV-memory feature, not a throughput one. (Mirrors
    // the prefill-side override in qwen35_prefill_suffix_batch.)
    if std::env::var("HIPFIRE_KV_HIERARCHICAL").ok().as_deref() == Some("1") {
        backend = Qwen35DecodeBatchBackend::SerialReference;
    }
    if backend == Qwen35DecodeBatchBackend::FusedDenseLayerChunked {
        validate_qwen35_fused_dense_decode_model_capability(m, envelope.session_count)?;
    } else if backend == Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked {
        validate_qwen35_grouped_moe_decode_model_capability(
            m,
            envelope.session_count,
            gpu.arch.as_str(),
        )?;
    }
    let im_end = {
        let tokenizer = m
            .tokenizer
            .as_ref()
            .ok_or_else(|| "generate_batch_decode_step requires a tokenizer".to_string())?;
        tokenizer.encode("<|im_end|>")
    };
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    let t0 = Instant::now();
    let step_result = match backend {
        Qwen35DecodeBatchBackend::SerialReference => Qwen35DecodeBatchStepResult {
            session_lines: qwen35_decode_step_serial_reference(
                m,
                gpu,
                stdout,
                envelope,
                im_end_token,
            )?,
            chunk_count: 1,
            chunk_size: envelope.session_count,
        },
        Qwen35DecodeBatchBackend::FusedDenseLayerChunked => {
            qwen35_decode_step_fused_dense_layer_chunked(m, gpu, stdout, envelope, im_end_token)?
        }
        Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked => {
            qwen35_decode_step_fused_grouped_moe_layer_chunked(
                m,
                gpu,
                stdout,
                envelope,
                im_end_token,
            )?
        }
    };
    for line in &step_result.session_lines {
        let _ = writeln!(stdout, "{line}");
    }
    let worker = loaded_model_worker_runtime_view(m);
    let scheduler_metadata = qwen35_decode_batch_scheduler_metadata(
        requested_backend.as_str(),
        m.arch_id,
        backend,
        envelope.session_count,
        envelope.cached_prefix_tokens,
    );
    let done = qwen35_generate_batch_decode_step_done_json(
        envelope,
        &step_result,
        backend,
        &scheduler_metadata,
        t0.elapsed().as_secs_f64() * 1000.0,
        sequence_state_arena_resident_session_count(loaded_model_state_arena_backend(m), m),
        model_worker_runtime_view_json(&worker),
    );
    let _ = writeln!(stdout, "{done}");
    let _ = stdout.flush();
    Ok(())
}

/// Max sessions per native-decode chunk for a given batch size (the kernels
/// process the batch in chunks of at most this many rows).
pub fn qwen35_decode_batch_max_chunk_size(session_count: usize) -> usize {
    std::env::var("HIPFIRE_QWEN35_DECODE_BATCH_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(session_count)
        .max(1)
}

/// Whether the dense native-decode multi-row kernel is enabled (env override).
pub fn qwen35_decode_dense_native_multirow_enabled() -> bool {
    matches!(
        std::env::var("HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

/// Whether internal parity-checking (native vs serial reference) is enabled
/// (debug env override).
pub fn qwen35_decode_internal_parity_enabled() -> bool {
    matches!(
        std::env::var("HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

/// Compact debug summary of a logits vector (argmax + a few stats) for the
/// internal-parity diagnostics.
pub fn qwen35_logits_debug_summary(
    gpu: &rdna_compute::Gpu,
    logits: &rdna_compute::GpuTensor,
    token_a: u32,
    token_b: u32,
) -> String {
    let Ok(values) = gpu.download_f32(logits) else {
        return "logits_download=failed".to_string();
    };
    let token_a_idx = token_a as usize;
    let token_b_idx = token_b as usize;
    let token_a_value = values.get(token_a_idx).copied().unwrap_or(f32::NAN);
    let token_b_value = values.get(token_b_idx).copied().unwrap_or(f32::NAN);
    let mut top: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top = top
        .into_iter()
        .take(4)
        .map(|(idx, value)| format!("{idx}:{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("token_{token_a}={token_a_value:.6} token_{token_b}={token_b_value:.6} top=[{top}]")
}

/// Sample/select the next token for one session from its decode logits and
/// package the per-session [`Qwen35DecodeTokenOutcome`] (token + stop state).
pub fn qwen35_decode_token_outcome(
    m: &LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    logits: &rdna_compute::GpuTensor,
    max_tokens_remaining: usize,
    im_end_token: Option<u32>,
) -> Result<Qwen35DecodeTokenOutcome, String> {
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| "generate_batch_decode_step requires a tokenizer".to_string())?;
    let token = gpu
        .argmax_f32(logits, config.vocab_size)
        .map_err(|e| format!("qwen35 decode argmax: {e:?}"))?;
    let is_terminator =
        token == config.eos_token || im_end_token == Some(token) || tokenizer.is_terminator(token);
    let stop = is_terminator || max_tokens_remaining <= 1;
    let text = if is_terminator {
        String::new()
    } else {
        tokenizer.decode(&[token])
    };
    Ok(Qwen35DecodeTokenOutcome { token, text, stop })
}

// ── Decode-step kernels ─────────────────────────────────────────────────────
// One decode step (one token per session) under different execution strategies:
// the serial reference (per-session forward, the correctness baseline), the
// fused-dense and grouped-MoE variants in layer-chunked and native-chunk forms,
// and the single-row native fast path. The driver picks one per the validated
// backend; chunk-range planning and scratch allocation support them.

/// Serial reference decode step: advance each session one token via the
/// per-session forward — the correctness baseline and the universal fallback.
pub fn qwen35_decode_step_serial_reference(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchDecodeEnvelope,
    im_end_token: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    qwen35_save_active_session(m, gpu)?;
    let mut session_lines = Vec::with_capacity(envelope.sessions.len());
    for session in &envelope.sessions {
        qwen35_activate_session(m, gpu, &session.session_id)?;
        let mut state = Qwen35RequestSessionState::take_from_loaded(m, gpu)?;
        let logical_position = state.seq_pos + state.kv_cache().compact_offset;
        if logical_position != session.logical_position {
            qwen35_restore_or_error(stdout, &session.id, m, gpu, state);
            return Err(format!(
                "decode session {} logical_position mismatch: expected={} resident={}",
                session.session_id, session.logical_position, logical_position
            ));
        }
        let scratch = m
            .q35_scratch
            .as_ref()
            .ok_or_else(|| "qwen35 scratch missing".to_string())?;
        let outcome = qwen35_decode_token_outcome(
            m,
            gpu,
            &scratch.logits,
            session.max_tokens_remaining,
            im_end_token,
        )?;
        state.conversation_tokens.push(outcome.token);
        {
            let config = m
                .q35_config
                .as_ref()
                .ok_or_else(|| "qwen35 config missing".to_string())?;
            let weights = m
                .q35_weights
                .as_ref()
                .ok_or_else(|| "qwen35 weights missing".to_string())?;
            let scratch = m
                .q35_scratch
                .as_ref()
                .ok_or_else(|| "qwen35 scratch missing".to_string())?;
            qwen35::forward_scratch(
                gpu,
                weights,
                config,
                outcome.token,
                state.seq_pos,
                state.sequence_state.kv.as_mut().expect("qwen35 session KV"),
                state
                    .sequence_state
                    .recurrent
                    .as_mut()
                    .expect("qwen35 session dn")
                    .as_any_mut()
                    .downcast_mut::<qwen35::DeltaNetState>()
                    .expect("qwen35 session dn"),
                scratch,
            )
            .map_err(|e| format!("qwen35 decode forward_scratch: {e:?}"))?;
            gpu.memcpy_dtod_auto(
                &state.logits.buf,
                &scratch.logits.buf,
                scratch.logits.buf.size(),
            )
            .map_err(|e| format!("save qwen35 decode logits snapshot: {e:?}"))?;
        }
        state.seq_pos += 1;
        let new_logical_position = state.seq_pos + state.kv_cache().compact_offset;
        qwen35_restore_or_error(stdout, &session.id, m, gpu, state);
        session_lines.push(serde_json::json!({
            "type": "generate_batch_decode_step_session_done",
            "id": envelope.id,
            "batch_id": envelope.batch_id,
            "session_id": session.id,
            "runtime_state_handle": session.session_id,
            "token": outcome.token,
            "text": outcome.text,
            "stop": outcome.stop,
            "logical_position": new_logical_position,
        }));
    }
    Ok(session_lines)
}

pub fn qwen35_decode_step_fused_dense_layer_chunked(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    _stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchDecodeEnvelope,
    im_end_token: Option<u32>,
) -> Result<Qwen35DecodeBatchStepResult, String> {
    qwen35_save_active_session(m, gpu)?;
    validate_qwen35_fused_dense_decode_resident_sessions(m, envelope)?;

    let chunk_size = qwen35_decode_batch_max_chunk_size(envelope.session_count);
    qwen35_decode_step_fused_dense_native_chunks(m, gpu, envelope, im_end_token, chunk_size)
}

pub fn qwen35_decode_step_fused_grouped_moe_layer_chunked(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    _stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchDecodeEnvelope,
    im_end_token: Option<u32>,
) -> Result<Qwen35DecodeBatchStepResult, String> {
    qwen35_save_active_session(m, gpu)?;
    validate_qwen35_decode_resident_sessions(m, envelope, "grouped-MoE chunked")?;

    let chunk_size = qwen35_decode_batch_max_chunk_size(envelope.session_count);
    qwen35_decode_step_fused_grouped_moe_native_chunks(m, gpu, envelope, im_end_token, chunk_size)
}

pub fn qwen35_decode_step_fused_dense_native_chunks(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    im_end_token: Option<u32>,
    chunk_size: usize,
) -> Result<Qwen35DecodeBatchStepResult, String> {
    let effective_chunk_size = if qwen35_decode_dense_native_multirow_enabled() {
        chunk_size
    } else {
        1
    };
    let chunks = qwen35_decode_native_chunk_ranges(envelope.session_count, effective_chunk_size)?;
    let mut session_lines = Vec::with_capacity(envelope.sessions.len());

    for (start, end) in &chunks {
        let mut chunk_lines = qwen35_decode_step_fused_dense_native_chunk(
            m,
            gpu,
            envelope,
            &envelope.sessions[*start..*end],
            im_end_token,
        )?;
        session_lines.append(&mut chunk_lines);
    }

    Ok(Qwen35DecodeBatchStepResult {
        session_lines,
        chunk_count: chunks.len(),
        chunk_size: effective_chunk_size.min(envelope.session_count),
    })
}

pub fn qwen35_decode_step_fused_grouped_moe_native_chunks(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    im_end_token: Option<u32>,
    chunk_size: usize,
) -> Result<Qwen35DecodeBatchStepResult, String> {
    let chunks = qwen35_decode_native_chunk_ranges(envelope.session_count, chunk_size)?;
    let mut session_lines = Vec::with_capacity(envelope.sessions.len());

    for (start, end) in &chunks {
        let chunk = &envelope.sessions[*start..*end];
        let mut chunk_lines = if chunk.len() == 1 {
            qwen35_decode_step_fused_dense_native_singleton(
                m,
                gpu,
                envelope,
                &chunk[0],
                im_end_token,
            )?
        } else {
            qwen35_decode_step_fused_grouped_moe_native_chunk(
                m,
                gpu,
                envelope,
                chunk,
                im_end_token,
            )?
        };
        session_lines.append(&mut chunk_lines);
    }

    Ok(Qwen35DecodeBatchStepResult {
        session_lines,
        chunk_count: chunks.len(),
        chunk_size: chunk_size.min(envelope.session_count),
    })
}

pub fn qwen35_decode_step_fused_grouped_moe_native_chunk(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    chunk: &[GenerateBatchDecodeSession],
    im_end_token: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    qwen35_ensure_decode_prefill_batch_scratch(m, gpu, chunk.len())?;

    let mut states: Vec<(GenerateBatchDecodeSession, Qwen35RequestSessionState)> =
        Vec::with_capacity(chunk.len());
    for session in chunk {
        let state = m.q35_sessions.remove(&session.session_id).ok_or_else(|| {
            format!(
                "decode session {} is not resident for fused grouped-MoE native decode",
                session.session_id
            )
        })?;
        states.push((session.clone(), state));
    }

    let result = (|| -> Result<Vec<serde_json::Value>, String> {
        let mut outcomes = Vec::with_capacity(states.len());
        for (session, state) in &states {
            let logical_position = state.seq_pos + state.kv_cache().compact_offset;
            if logical_position != session.logical_position {
                return Err(format!(
                    "decode session {} logical_position mismatch: expected={} resident={}",
                    session.session_id, session.logical_position, logical_position
                ));
            }
            outcomes.push(qwen35_decode_token_outcome(
                m,
                gpu,
                &state.logits,
                session.max_tokens_remaining,
                im_end_token,
            )?);
        }
        let mut oracle_states = if qwen35_decode_internal_parity_enabled() {
            let mut cloned = Vec::with_capacity(states.len());
            for (session, state) in &states {
                cloned.push((
                    session.clone(),
                    Qwen35RequestSessionState::fork_from(gpu, state)?,
                ));
            }
            Some(cloned)
        } else {
            None
        };

        for ((_, state), outcome) in states.iter_mut().zip(outcomes.iter()) {
            state.conversation_tokens.push(outcome.token);
        }

        let token_rows: Vec<[u32; 1]> = outcomes.iter().map(|outcome| [outcome.token]).collect();
        let weights = m
            .q35_weights
            .as_ref()
            .ok_or_else(|| "qwen35 weights missing".to_string())?;
        let config = m
            .q35_config
            .as_ref()
            .ok_or_else(|| "qwen35 config missing".to_string())?;
        let scratch = m
            .q35_scratch
            .as_ref()
            .ok_or_else(|| "qwen35 scratch missing".to_string())?;
        let pbs = scratch
            .prefill_batch
            .as_ref()
            .ok_or_else(|| "qwen35 grouped-MoE decode native batch scratch missing".to_string())?;
        let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = states
            .iter_mut()
            .zip(token_rows.iter())
            .map(|((_, state), tokens)| qwen35::DensePrefillSessionBatchRow {
                tokens,
                start_pos: state.seq_pos,
                kv_cache: state.sequence_state.kv.as_mut().expect("qwen35 session KV"),
                dn_state: state
                    .sequence_state
                    .recurrent
                    .as_mut()
                    .expect("qwen35 session dn")
                    .as_any_mut()
                    .downcast_mut::<qwen35::DeltaNetState>()
                    .expect("qwen35 session dn"),
                logits: &state.logits,
            })
            .collect();
        qwen35::forward_prefill_grouped_moe_session_batch(
            gpu, weights, config, &mut rows, scratch, pbs,
        )
        .map_err(|e| format!("qwen35 fused grouped-MoE native decode advance: {e:?}"))?;
        drop(rows);

        let mut lines = Vec::with_capacity(states.len());
        for ((session, state), outcome) in states.iter_mut().zip(outcomes.iter()) {
            state.seq_pos += 1;
            let new_logical_position = state.seq_pos + state.kv_cache().compact_offset;
            lines.push(serde_json::json!({
                "type": "generate_batch_decode_step_session_done",
                "id": envelope.id,
                "batch_id": envelope.batch_id,
                "session_id": session.id,
                "runtime_state_handle": session.session_id,
                "token": outcome.token,
                "text": outcome.text,
                "stop": outcome.stop,
                "logical_position": new_logical_position,
            }));
        }
        if let Some(oracle_states) = oracle_states.as_mut() {
            let config = m
                .q35_config
                .as_ref()
                .ok_or_else(|| "qwen35 config missing".to_string())?;
            let weights = m
                .q35_weights
                .as_ref()
                .ok_or_else(|| "qwen35 weights missing".to_string())?;
            let scratch = m
                .q35_scratch
                .as_ref()
                .ok_or_else(|| "qwen35 scratch missing".to_string())?;
            for (((session, fused_state), outcome), (_, oracle_state)) in states
                .iter()
                .zip(outcomes.iter())
                .zip(oracle_states.iter_mut())
            {
                let oracle_outcome = qwen35_decode_token_outcome(
                    m,
                    gpu,
                    &oracle_state.logits,
                    session.max_tokens_remaining,
                    im_end_token,
                )?;
                if oracle_outcome.token != outcome.token {
                    return Err(format!(
                        "qwen35 fused grouped-MoE native decode parity mismatch before advance for {}: fused_token={} serial_token={}",
                        session.session_id, outcome.token, oracle_outcome.token
                    ));
                }
                oracle_state.conversation_tokens.push(oracle_outcome.token);
                qwen35::forward_scratch(
                    gpu,
                    weights,
                    config,
                    oracle_outcome.token,
                    oracle_state.seq_pos,
                    oracle_state
                        .sequence_state
                        .kv
                        .as_mut()
                        .expect("qwen35 session KV"),
                    oracle_state
                        .sequence_state
                        .recurrent
                        .as_mut()
                        .expect("qwen35 session dn")
                        .as_any_mut()
                        .downcast_mut::<qwen35::DeltaNetState>()
                        .expect("qwen35 session dn"),
                    scratch,
                )
                .map_err(|e| {
                    format!("qwen35 grouped-MoE decode internal serial parity advance: {e:?}")
                })?;
                gpu.memcpy_dtod_auto(
                    &oracle_state.logits.buf,
                    &scratch.logits.buf,
                    scratch.logits.buf.size(),
                )
                .map_err(|e| {
                    format!("save qwen35 grouped-MoE decode internal parity logits: {e:?}")
                })?;
                oracle_state.seq_pos += 1;
                let fused_next = gpu
                    .argmax_f32(&fused_state.logits, config.vocab_size)
                    .map_err(|e| format!("qwen35 grouped-MoE fused parity fused argmax: {e:?}"))?;
                let serial_next = gpu
                    .argmax_f32(&oracle_state.logits, config.vocab_size)
                    .map_err(|e| format!("qwen35 grouped-MoE fused parity serial argmax: {e:?}"))?;
                if fused_next != serial_next {
                    let fused_summary = qwen35_logits_debug_summary(
                        gpu,
                        &fused_state.logits,
                        fused_next,
                        serial_next,
                    );
                    let serial_summary = qwen35_logits_debug_summary(
                        gpu,
                        &oracle_state.logits,
                        fused_next,
                        serial_next,
                    );
                    return Err(format!(
                        "qwen35 fused grouped-MoE native decode parity mismatch after advance for {}: fused_next={} serial_next={} fused_logits=({}) serial_logits=({})",
                        session.session_id, fused_next, serial_next, fused_summary, serial_summary
                    ));
                }
            }
        }
        Ok(lines)
    })();

    for (session, state) in states {
        m.q35_sessions.insert(session.session_id, state);
    }

    result
}

pub fn qwen35_decode_native_chunk_ranges(
    session_count: usize,
    chunk_size: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if session_count <= chunk_size {
        return Ok(vec![(0, session_count)]);
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < session_count {
        let end = (start + chunk_size).min(session_count);
        ranges.push((start, end));
        start = end;
    }
    Ok(ranges)
}

pub fn qwen35_ensure_decode_prefill_batch_scratch(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    min_rows: usize,
) -> Result<(), String> {
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    let scratch = m
        .q35_scratch
        .as_mut()
        .ok_or_else(|| "qwen35 scratch missing".to_string())?;
    let needs_alloc = scratch
        .prefill_batch
        .as_ref()
        .map(|pbs| pbs.max_batch < min_rows)
        .unwrap_or(true);
    if needs_alloc {
        if let Some(existing) = scratch.prefill_batch.take() {
            existing.free_gpu(gpu);
        }
        let configured_max = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 2)
            .unwrap_or(qwen35::PREFILL_MAX_BATCH);
        let max_batch = configured_max.max(min_rows);
        scratch.prefill_batch = Some(
            qwen35::PrefillBatchScratch::new(gpu, config, max_batch)
                .map_err(|e| format!("alloc qwen35 decode native batch scratch: {e:?}"))?,
        );
    }
    Ok(())
}

pub fn qwen35_decode_step_fused_dense_native_chunk(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    chunk: &[GenerateBatchDecodeSession],
    im_end_token: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    if chunk.len() == 1 {
        return qwen35_decode_step_fused_dense_native_singleton(
            m,
            gpu,
            envelope,
            &chunk[0],
            im_end_token,
        );
    }
    qwen35_ensure_decode_prefill_batch_scratch(m, gpu, chunk.len())?;

    let mut states: Vec<(GenerateBatchDecodeSession, Qwen35RequestSessionState)> =
        Vec::with_capacity(chunk.len());
    for session in chunk {
        let state = m.q35_sessions.remove(&session.session_id).ok_or_else(|| {
            format!(
                "decode session {} is not resident for fused dense native decode",
                session.session_id
            )
        })?;
        states.push((session.clone(), state));
    }

    let result = (|| -> Result<Vec<serde_json::Value>, String> {
        let mut outcomes = Vec::with_capacity(states.len());
        for (session, state) in &states {
            let logical_position = state.seq_pos + state.kv_cache().compact_offset;
            if logical_position != session.logical_position {
                return Err(format!(
                    "decode session {} logical_position mismatch: expected={} resident={}",
                    session.session_id, session.logical_position, logical_position
                ));
            }
            outcomes.push(qwen35_decode_token_outcome(
                m,
                gpu,
                &state.logits,
                session.max_tokens_remaining,
                im_end_token,
            )?);
        }
        let mut oracle_states = if qwen35_decode_internal_parity_enabled() {
            let mut cloned = Vec::with_capacity(states.len());
            for (session, state) in &states {
                cloned.push((
                    session.clone(),
                    Qwen35RequestSessionState::fork_from(gpu, state)?,
                ));
            }
            Some(cloned)
        } else {
            None
        };

        for ((_, state), outcome) in states.iter_mut().zip(outcomes.iter()) {
            state.conversation_tokens.push(outcome.token);
        }

        let token_rows: Vec<[u32; 1]> = outcomes.iter().map(|outcome| [outcome.token]).collect();
        let weights = m
            .q35_weights
            .as_ref()
            .ok_or_else(|| "qwen35 weights missing".to_string())?;
        let config = m
            .q35_config
            .as_ref()
            .ok_or_else(|| "qwen35 config missing".to_string())?;
        let scratch = m
            .q35_scratch
            .as_ref()
            .ok_or_else(|| "qwen35 scratch missing".to_string())?;
        let pbs = scratch
            .prefill_batch
            .as_ref()
            .ok_or_else(|| "qwen35 decode native batch scratch missing".to_string())?;
        let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = states
            .iter_mut()
            .zip(token_rows.iter())
            .map(|((_, state), tokens)| qwen35::DensePrefillSessionBatchRow {
                tokens,
                start_pos: state.seq_pos,
                kv_cache: state.sequence_state.kv.as_mut().expect("qwen35 session KV"),
                dn_state: state
                    .sequence_state
                    .recurrent
                    .as_mut()
                    .expect("qwen35 session dn")
                    .as_any_mut()
                    .downcast_mut::<qwen35::DeltaNetState>()
                    .expect("qwen35 session dn"),
                logits: &state.logits,
            })
            .collect();
        qwen35::forward_prefill_dense_session_batch(gpu, weights, config, &mut rows, scratch, pbs)
            .map_err(|e| format!("qwen35 fused dense native decode advance: {e:?}"))?;
        drop(rows);

        let mut lines = Vec::with_capacity(states.len());
        for ((session, state), outcome) in states.iter_mut().zip(outcomes.iter()) {
            state.seq_pos += 1;
            let new_logical_position = state.seq_pos + state.kv_cache().compact_offset;
            lines.push(serde_json::json!({
                "type": "generate_batch_decode_step_session_done",
                "id": envelope.id,
                "batch_id": envelope.batch_id,
                "session_id": session.id,
                "runtime_state_handle": session.session_id,
                "token": outcome.token,
                "text": outcome.text,
                "stop": outcome.stop,
                "logical_position": new_logical_position,
            }));
        }
        if let Some(oracle_states) = oracle_states.as_mut() {
            let config = m
                .q35_config
                .as_ref()
                .ok_or_else(|| "qwen35 config missing".to_string())?;
            let weights = m
                .q35_weights
                .as_ref()
                .ok_or_else(|| "qwen35 weights missing".to_string())?;
            let scratch = m
                .q35_scratch
                .as_ref()
                .ok_or_else(|| "qwen35 scratch missing".to_string())?;
            for (((session, fused_state), outcome), (_, oracle_state)) in states
                .iter()
                .zip(outcomes.iter())
                .zip(oracle_states.iter_mut())
            {
                let oracle_outcome = qwen35_decode_token_outcome(
                    m,
                    gpu,
                    &oracle_state.logits,
                    session.max_tokens_remaining,
                    im_end_token,
                )?;
                if oracle_outcome.token != outcome.token {
                    return Err(format!(
                        "qwen35 fused dense native decode parity mismatch before advance for {}: fused_token={} serial_token={}",
                        session.session_id, outcome.token, oracle_outcome.token
                    ));
                }
                oracle_state.conversation_tokens.push(oracle_outcome.token);
                qwen35::forward_scratch(
                    gpu,
                    weights,
                    config,
                    oracle_outcome.token,
                    oracle_state.seq_pos,
                    oracle_state
                        .sequence_state
                        .kv
                        .as_mut()
                        .expect("qwen35 session KV"),
                    oracle_state
                        .sequence_state
                        .recurrent
                        .as_mut()
                        .expect("qwen35 session dn")
                        .as_any_mut()
                        .downcast_mut::<qwen35::DeltaNetState>()
                        .expect("qwen35 session dn"),
                    scratch,
                )
                .map_err(|e| format!("qwen35 decode internal serial parity advance: {e:?}"))?;
                gpu.memcpy_dtod_auto(
                    &oracle_state.logits.buf,
                    &scratch.logits.buf,
                    scratch.logits.buf.size(),
                )
                .map_err(|e| format!("save qwen35 decode internal parity logits: {e:?}"))?;
                oracle_state.seq_pos += 1;
                let fused_next = gpu
                    .argmax_f32(&fused_state.logits, config.vocab_size)
                    .map_err(|e| format!("qwen35 fused parity fused argmax: {e:?}"))?;
                let serial_next = gpu
                    .argmax_f32(&oracle_state.logits, config.vocab_size)
                    .map_err(|e| format!("qwen35 fused parity serial argmax: {e:?}"))?;
                if fused_next != serial_next {
                    let fused_summary = qwen35_logits_debug_summary(
                        gpu,
                        &fused_state.logits,
                        fused_next,
                        serial_next,
                    );
                    let serial_summary = qwen35_logits_debug_summary(
                        gpu,
                        &oracle_state.logits,
                        fused_next,
                        serial_next,
                    );
                    return Err(format!(
                        "qwen35 fused dense native decode parity mismatch after advance for {}: fused_next={} serial_next={} fused_logits=({}) serial_logits=({})",
                        session.session_id, fused_next, serial_next, fused_summary, serial_summary
                    ));
                }
            }
        }
        Ok(lines)
    })();

    for (session, state) in states {
        m.q35_sessions.insert(session.session_id, state);
    }

    result
}

pub fn qwen35_decode_step_fused_dense_native_singleton(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    session: &GenerateBatchDecodeSession,
    im_end_token: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    qwen35_activate_session(m, gpu, &session.session_id)?;
    let mut state = Qwen35RequestSessionState::take_from_loaded(m, gpu)?;

    let result = (|| -> Result<Vec<serde_json::Value>, String> {
        let logical_position = state.seq_pos + state.kv_cache().compact_offset;
        if logical_position != session.logical_position {
            return Err(format!(
                "decode session {} logical_position mismatch: expected={} resident={}",
                session.session_id, session.logical_position, logical_position
            ));
        }
        let outcome = qwen35_decode_token_outcome(
            m,
            gpu,
            &state.logits,
            session.max_tokens_remaining,
            im_end_token,
        )?;
        state.conversation_tokens.push(outcome.token);
        {
            let config = m
                .q35_config
                .as_ref()
                .ok_or_else(|| "qwen35 config missing".to_string())?;
            let weights = m
                .q35_weights
                .as_ref()
                .ok_or_else(|| "qwen35 weights missing".to_string())?;
            let scratch = m
                .q35_scratch
                .as_ref()
                .ok_or_else(|| "qwen35 scratch missing".to_string())?;
            qwen35::forward_scratch(
                gpu,
                weights,
                config,
                outcome.token,
                state.seq_pos,
                state.sequence_state.kv.as_mut().expect("qwen35 session KV"),
                state
                    .sequence_state
                    .recurrent
                    .as_mut()
                    .expect("qwen35 session dn")
                    .as_any_mut()
                    .downcast_mut::<qwen35::DeltaNetState>()
                    .expect("qwen35 session dn"),
                scratch,
            )
            .map_err(|e| format!("qwen35 fused dense native singleton decode advance: {e:?}"))?;
            gpu.memcpy_dtod_auto(
                &state.logits.buf,
                &scratch.logits.buf,
                scratch.logits.buf.size(),
            )
            .map_err(|e| format!("save qwen35 native singleton logits snapshot: {e:?}"))?;
        }
        state.seq_pos += 1;
        let new_logical_position = state.seq_pos + state.kv_cache().compact_offset;
        Ok(vec![serde_json::json!({
            "type": "generate_batch_decode_step_session_done",
            "id": envelope.id,
            "batch_id": envelope.batch_id,
            "session_id": session.id,
            "runtime_state_handle": session.session_id,
            "token": outcome.token,
            "text": outcome.text,
            "stop": outcome.stop,
            "logical_position": new_logical_position,
        })])
    })();

    let restore_result = state.restore_into_loaded(m, gpu);
    let save_result = restore_result.and_then(|()| qwen35_save_active_session(m, gpu));
    if let Err(err) = save_result {
        return Err(err);
    }
    result
}
