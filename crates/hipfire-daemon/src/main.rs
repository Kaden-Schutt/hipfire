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
//! preventing orphan doubles from silently double-consuming VRAM. Startup also
//! takes per-resource leases in /tmp before HIP init so multi-worktree agents
//! contend on GPU/NPU/CPU resources through the runtime, not a shell helper.
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

use hipfire_arch_deepseek4 as deepseek4;
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative::{self};
use hipfire_evidence::{RouterHistogramEvidence, RouterHistogramLayer, RuntimeOneshotEvidence};
use hipfire_generate::eos_filter::EosFilter;
use hipfire_generate::loop_guard::StopReason;
use hipfire_generate::sampler::{collect_unclosed_attractor_blocks, SamplerConfig};
#[cfg(test)]
use hipfire_generate::validate_qwen35_fused_dense_prefill_batch_preflight;
use hipfire_generate::{
    build_qwen35_fused_dense_prefill_batch_contract, compute_qwen35_prefix_hash,
    plan_generate_batch_prefill_qwen35, prefix_hash_preflight_done_json,
    qwen35_decode_batch_requested_auto, qwen35_decode_batch_scheduler_metadata,
    qwen35_fused_prefill_boundary_cuts, qwen35_generate_batch_decode_step_done_json,
    qwen35_generate_batch_prefill_done_json, qwen35_generate_batch_prefill_session_done_json,
    qwen35_grouped_moe_decode_auto_latency_gate_passed, qwen35_prefill_checkpoint_boundary_kind,
    qwen35_prefill_checkpoint_session_id, qwen35_prefill_scratch_target_batch,
    select_qwen35_decode_batch_backend, select_qwen35_prefill_batch_backend,
    validate_generate_batch_decode, validate_generate_batch_prefill,
    validate_prefix_hash_preflight, validate_qwen35_fused_grouped_moe_prefill_batch_preflight,
    GenerateBatchDecodeEnvelope, GenerateBatchDecodeSession, GenerateBatchPrefillEnvelope,
    GenerateBatchPrefillPlan, GenerateBatchPrefillSession, GenerateVLParams, ImageSource,
    PrefixHashPreflightCandidate, PrefixHashPreflightEnvelope, Qwen35DecodeBatchBackend,
    Qwen35DecodeBatchStepResult, Qwen35DecodeTokenOutcome, Qwen35FusedDensePrefillInputKind,
    Qwen35PrefillBatchBackend, Qwen35PrefillBatchResult, Qwen35PrefillCheckpointHook,
    Qwen35PrefillCheckpointKind, Qwen35PrefillSessionResult, Qwen35PreparedPrefillSession,
    Qwen35SemanticBoundaryCheckpoint,
};
use hipfire_model::{is_qwen35_dense_arch_id, is_qwen35_family_arch_id, is_qwen35_moe_arch_id};
use hipfire_prompt as prompt_frame;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::sampler;
use hipfire_state::{
    described_sequence_state_json, generate_state_kinds_include_required,
    model_worker_runtime_view_json, parse_describe_sequence_state_request,
    parse_release_sequence_state_request, parse_release_sessions_request,
    parse_reserve_session_state_request, parse_unload_worker_request,
    parsed_handle_may_target_generic, release_sessions_done_json, release_state_done_json,
    reserve_session_state_done_json, reserve_session_state_rejected_json,
    sequence_state_reservation_plan, sequence_state_reservation_plan_for_reserved_bytes,
    session_state_reservation_describe_json, unload_worker_done_json, GenericSequenceStateArena,
    SequenceStateArenaBackend, SequenceStateCheckpointRequest, SequenceStateForkRequest,
    SequenceStatePageKind,
};
#[cfg(test)]
use hipfire_state::{
    generic_state_reservation_descriptors, parse_reserve_session_state_kinds,
    parse_sequence_state_handle, sequence_state_handle_id, sequence_state_handle_parts,
    sequence_state_page_descriptor_json, SequenceStateHandle,
};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

mod events;
use events::{
    emit_committed_event, emit_error_with_id, emit_filter_action, emit_stream_event, write_error,
    MAX_BASE64_ENCODED_LEN,
};
mod output_filter;
use output_filter::{
    chat_output_filter_from_profile, loop_guard_from_runtime_config, normalize_daemon_prompt,
    normalize_request_stop_sequences,
};
mod model;
use model::{effective_raw, LoadedModel, RAW_OVERRIDE};
mod dummy;
mod memory;
use dummy::{
    emit_dummy_generate_batch_prefill_ready, run_generate_batch_prefill_dummy, DummyModelState,
};
mod session;
use session::*;
mod load;
use load::*;
mod generate_vl;
use generate_vl::{generate_vl, generate_vl_dots_ocr};

/// CASK/TriAttention params forwarded by the CLI at load time. Zero-initialized
/// CaskConfig{sidecar: None, ..} means no eviction — matches 0.1.7-alpha behavior.
#[derive(Default)]
pub(crate) struct CaskConfig {
    pub(crate) sidecar: Option<String>,
    /// true = CASK m-folding; false = plain TriAttention drop-eviction.
    pub(crate) cask_m_folding: bool,
    pub(crate) budget: usize,
    pub(crate) beta: usize,
    pub(crate) core_frac: f32,
    pub(crate) fold_m: usize,
}

fn daemon_runtime_context(model: &LoadedModel) -> serde_json::Value {
    serde_json::json!({
        "schema": 1,
        "runner": "hipfire-daemon",
        "hipfire_version": env!("CARGO_PKG_VERSION"),
        "model_path": &model.model_path,
        "arch_id": model.arch_id,
        "pipeline_parallel_degree": model.pp,
        "max_seq": model.max_seq,
        "physical_cap": model.physical_cap,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_daemon_runtime_oneshot_evidence(
    dir: &str,
    model: &LoadedModel,
    gpu: &rdna_compute::Gpu,
    id: &str,
    prompt_tokens: usize,
    emitted_tokens: usize,
    prefill_secs: f64,
    decode_secs: f64,
    ttft_ms: f64,
) {
    let (vram_free_bytes, vram_total_bytes) = gpu.hip.get_vram_info().unwrap_or((0, 0));
    let vram_used_mb =
        ((vram_total_bytes.saturating_sub(vram_free_bytes)) as f64 / (1024.0 * 1024.0)) as u64;
    let vram_total_mb = (vram_total_bytes as f64 / (1024.0 * 1024.0)) as u64;
    let runtime_context = daemon_runtime_context(model);
    let evidence = RuntimeOneshotEvidence {
        case_id: id,
        prompt_path: "daemon://generate",
        prompt_tokens,
        emitted_tokens,
        prefill_forward_calls: if prompt_tokens == 0 { 0 } else { 1 },
        decode_forward_calls: emitted_tokens,
        prefill_secs,
        decode_secs,
        ttft_ms,
        vram_used_mb,
        vram_total_mb,
    };
    if let Err(err) =
        hipfire_evidence::write_runtime_oneshot_evidence(Path::new(dir), &runtime_context, evidence)
    {
        eprintln!("[daemon/evidence] failed to write runtime oneshot evidence: {err}");
    }
}

struct DaemonMoeRouterHistogramGuard {
    active: bool,
}

impl DaemonMoeRouterHistogramGuard {
    fn start(evidence_dir: Option<&str>, config: &qwen35::Qwen35Config) -> Self {
        let active = evidence_dir.is_some() && config.num_experts > 0;
        if active {
            qwen35::reset_moe_router_histogram(config.num_experts, config.num_experts_per_tok);
        }
        Self { active }
    }

    fn take(mut self) -> Option<qwen35::MoeRouterHistogram> {
        self.active = false;
        qwen35::take_moe_router_histogram()
    }
}

impl Drop for DaemonMoeRouterHistogramGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = qwen35::take_moe_router_histogram();
        }
    }
}

fn write_daemon_moe_router_evidence(
    dir: &str,
    model: &LoadedModel,
    id: &str,
    hist: qwen35::MoeRouterHistogram,
) {
    if hist.routed_slots == 0 {
        return;
    }
    let runtime_context = daemon_runtime_context(model);
    let evidence = RouterHistogramEvidence {
        case_id: id,
        prompt_path: "daemon://generate",
        collection_scope: "qwen35_moe_daemon_ar_forward_calls",
        num_experts: hist.num_experts,
        k_top: hist.k_top,
        routed_tokens: hist.routed_tokens,
        routed_slots: hist.routed_slots,
        top1_histogram: hist.top1_histogram,
        topk_histogram: hist.topk_histogram,
        weight_sums: hist.weight_sums,
        dropped_indices: hist.dropped_indices,
        per_layer: hist
            .per_layer
            .into_iter()
            .map(|layer| RouterHistogramLayer {
                layer_idx: layer.layer_idx,
                top1_histogram: layer.top1_histogram,
                topk_histogram: layer.topk_histogram,
                weight_sums: layer.weight_sums,
                dropped_indices: layer.dropped_indices,
                routed_tokens: layer.routed_tokens,
                routed_slots: layer.routed_slots,
                cooccurrence: layer.cooccurrence,
            })
            .collect(),
    };
    if let Err(err) = hipfire_evidence::write_router_histogram_evidence(
        Path::new(dir),
        &runtime_context,
        evidence,
    ) {
        eprintln!("[daemon/evidence] failed to write MoE router evidence: {err}");
    }
}

/// Acquire a machine-wide exclusive lock on ~/.hipfire/daemon.pid.
///
/// On Unix: flock(2) is the kernel-level lock. The kernel releases it
/// automatically on process death (including SIGKILL), so no manual
/// cleanup is required — stale PID file contents are fine, the fd is
/// what holds the lock.
///
/// On Windows: no kernel-level lock; we write the PID file but don't
/// guarantee single-instance semantics. A second daemon launch may
/// silently overwrite the PID. This matches the v0.1.0-alpha Windows
/// behavior; tightening it is tracked in a follow-up.
///
/// Returns the File handle; caller MUST keep it alive for the process
/// lifetime (on Unix, dropping it closes the fd and releases the lock).
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

fn chat_output_filter(m: &LoadedModel, request_stop_sequences: &[String]) -> EosFilter {
    chat_output_filter_from_profile(m.chat_template_profile.as_ref(), request_stop_sequences)
}

fn validate_qwen35_decode_batch_runtime_surface(
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

fn qwen35_fused_dense_decode_signature(
    state: &Qwen35RequestSessionState,
) -> qwen35::DensePrefillSessionBatchStateSignature {
    qwen35::DensePrefillSessionBatchStateSignature {
        kv_physical_cap: state.kv_cache.physical_cap,
        kv_compact_offset: state.kv_cache.compact_offset,
        kv_quantized: state.kv_cache.quantized,
        kv_quant_q8: state.kv_cache.quant_q8,
        kv_quant_asym2: state.kv_cache.quant_asym2,
        kv_quant_asym3: state.kv_cache.quant_asym3,
        kv_quant_asym4: state.kv_cache.quant_asym4,
        kv_quant_fwht: state.kv_cache.quant_fwht,
        dn_quant: state.dn_state.quant,
    }
}

fn validate_qwen35_fused_dense_decode_session_signatures(
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

fn validate_qwen35_grouped_moe_decode_session_signatures(
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

fn validate_qwen35_fused_dense_decode_model_capability(
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

fn validate_qwen35_grouped_moe_decode_model_capability(
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

fn validate_qwen35_decode_resident_sessions(
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
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
        if logical_position != session.logical_position {
            return Err(format!(
                "decode session {} logical_position mismatch: expected={} resident={}",
                session.session_id, session.logical_position, logical_position
            ));
        }
    }
    Ok(())
}

fn validate_qwen35_fused_dense_decode_resident_sessions(
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
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
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

fn emit_generate_batch_prefill_ready(
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

fn emit_generate_batch_prefill_unsupported(
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

#[cfg(test)]
mod generate_batch_prefill_tests {
    use super::*;
    use hipfire_arch_qwen35::qwen35::LayerType;
    use hipfire_model::ModelWorkerId;
    use hipfire_state::{
        describe_sequence_state_descriptors, parsed_handle_may_target_loaded_state,
        qwen35_sequence_state_handle, ModelWorkerMemoryView, ModelWorkerRuntimeView,
        ParsedSequenceStateHandle, SequenceStatePageDescriptor, SequenceStatePrefixHash,
    };
    fn test_dense_qwen35_config() -> qwen35::Qwen35Config {
        qwen35::Qwen35Config {
            dim: 16,
            n_layers: 2,
            vocab_size: 32,
            norm_eps: 1e-6,
            eos_token: 0,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            rope_theta: 1_000_000.0,
            partial_rotary_factor: 0.25,
            attn_output_gate: true,
            is_vl_text: false,
            mrope_interleaved: false,
            mrope_section: [0, 0, 0],
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            conv_kernel_dim: 4,
            hidden_dim: 32,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            has_shared_expert: false,
            norm_topk_prob: false,
            layer_types: vec![LayerType::FullAttention, LayerType::LinearAttention],
            paged_experts: false,
            vram_budget_bytes: u64::MAX,
        }
    }

    fn test_grouped_moe_qwen35_config() -> qwen35::Qwen35Config {
        qwen35::Qwen35Config {
            num_experts: 4,
            num_experts_per_tok: 8,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            has_shared_expert: true,
            norm_topk_prob: true,
            ..test_dense_qwen35_config()
        }
    }

    fn fp32_decode_state_signature() -> qwen35::DensePrefillSessionBatchStateSignature {
        qwen35::DensePrefillSessionBatchStateSignature {
            kv_physical_cap: 128,
            kv_compact_offset: 0,
            kv_quantized: false,
            kv_quant_q8: false,
            kv_quant_asym2: false,
            kv_quant_asym3: false,
            kv_quant_asym4: false,
            kv_quant_fwht: false,
            dn_quant: qwen35::StateQuant::FP32,
        }
    }

    fn q8_decode_state_signature() -> qwen35::DensePrefillSessionBatchStateSignature {
        qwen35::DensePrefillSessionBatchStateSignature {
            kv_quantized: true,
            kv_quant_q8: true,
            dn_quant: qwen35::StateQuant::Q8,
            ..fp32_decode_state_signature()
        }
    }

    #[test]
    fn validates_minimal_prompt_envelope() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-1",
            "batch_id": "batch-1",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                },
                "params": {
                    "max_tokens": 8,
                    "temperature": 0.0
                }
            }]
        });

        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        assert_eq!(envelope.id, "probe-1");
        assert_eq!(envelope.batch_id, "batch-1");
        assert_eq!(envelope.session_count, 1);
        assert!(!envelope.sessions[0].semantic_boundary_checkpoints);
    }

    #[test]
    fn validates_suffix_token_envelope_with_model_identity() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "batch_id": "batch-2",
            "model": "qwen3.5:9b",
            "sessions": [{
                "id": "req-1",
                "suffix_tokens": [1, 2, 3],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            }]
        });

        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        assert_eq!(envelope.id, "0");
        assert_eq!(envelope.batch_id, "batch-2");
        assert_eq!(envelope.session_count, 1);
    }

    #[test]
    fn validates_runtime_state_handle_for_attachable_prefix() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "batch_id": "batch-attach",
            "model": "qwen3.5:9b",
            "sessions": [{
                "id": "req-1",
                "suffix_tokens": [4, 5],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 12,
                    "cached_prefix_tokens": 12,
                    "runtime_state_handle": "qwen35-checkpoint:req-0"
                }
            }]
        });

        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        assert_eq!(
            envelope.sessions[0]
                .state_handle
                .runtime_state_handle
                .as_deref(),
            Some("qwen35-checkpoint:req-0")
        );
    }

    #[test]
    fn validates_runtime_prefix_hash_for_attachable_prefix() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "batch_id": "batch-1",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "suffix_tokens": [4, 5],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 3,
                    "cached_prefix_tokens": 3,
                    "runtime_state_handle": "qwen35-checkpoint:req-0",
                    "prefix_hash": {
                        "algorithm": "xxh128",
                        "value": "0123456789abcdef0123456789abcdef",
                        "prefix_len": 3
                    }
                }
            }]
        });

        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        let hash = envelope.sessions[0]
            .state_handle
            .prefix_hash
            .as_ref()
            .expect("prefix hash parsed");
        assert_eq!(hash.algorithm, "xxh128");
        assert_eq!(hash.prefix_len, 3);
    }

    #[test]
    fn rejects_malformed_runtime_prefix_hash() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "batch_id": "batch-1",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "suffix_tokens": [4, 5],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 3,
                    "cached_prefix_tokens": 3,
                    "runtime_state_handle": "qwen35-checkpoint:req-0",
                    "prefix_hash": {
                        "algorithm": "xxh128",
                        "value": "ABC",
                        "prefix_len": 3
                    }
                }
            }]
        });

        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("32 lowercase hex"));
    }

    #[test]
    fn validates_prefix_hash_preflight_envelope() {
        let msg = serde_json::json!({
            "type": "prefix_hash_preflight",
            "id": "prefix-1",
            "worker_key_id": "worker-a",
            "boundary_policy": "semantic_chat_template",
            "session": {
                "id": "req-1",
                "prompt": "hello",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hello"}
                ],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                },
                "params": {
                    "assistant_prefix": "open_think",
                    "max_think_tokens": 16
                }
            }
        });

        let envelope = validate_prefix_hash_preflight(&msg).expect("valid preflight");
        assert_eq!(envelope.id, "prefix-1");
        assert_eq!(envelope.boundary_policy, "semantic_chat_template");
        assert_eq!(envelope.session.id, "req-1");
        assert_eq!(envelope.session.messages_history.as_ref().unwrap().len(), 2);
        assert_eq!(envelope.session.assistant_prefix, "open_think");
        assert_eq!(envelope.session.max_think_tokens, 16);
    }

    #[test]
    fn validates_generate_batch_decode_step_envelope() {
        let msg = serde_json::json!({
            "type": "generate_batch_decode_step",
            "id": "decode-1",
            "batch_id": "decode-batch-1",
            "worker_key_id": "worker-a",
            "cached_prefix_tokens": 12,
            "sessions": [
                {
                    "id": "req-1",
                    "session_id": "qwen35-checkpoint:batch:req-1:8",
                    "logical_position": 8,
                    "max_tokens_remaining": 4
                },
                {
                    "id": "req-2",
                    "session_id": "qwen35-checkpoint:batch:req-2:8",
                    "logical_position": 8,
                    "max_tokens_remaining": 3
                }
            ]
        });

        let envelope = validate_generate_batch_decode(&msg).expect("valid decode envelope");
        assert_eq!(envelope.id, "decode-1");
        assert_eq!(envelope.batch_id, "decode-batch-1");
        assert_eq!(envelope.session_count, 2);
        assert_eq!(envelope.cached_prefix_tokens, 12);
        assert_eq!(
            envelope.sessions[0].session_id,
            "qwen35-checkpoint:batch:req-1:8"
        );
        assert_eq!(envelope.sessions[1].max_tokens_remaining, 3);
    }

    #[test]
    fn rejects_invalid_generate_batch_decode_step_envelope() {
        let missing_worker = serde_json::json!({
            "type": "generate_batch_decode_step",
            "batch_id": "decode-batch-1",
            "sessions": [{
                "id": "req-1",
                "session_id": "runtime-1",
                "logical_position": 8,
                "max_tokens_remaining": 1
            }]
        });
        let err = validate_generate_batch_decode(&missing_worker).unwrap_err();
        assert!(err.contains("worker_key_id or model"));

        let zero_remaining = serde_json::json!({
            "type": "generate_batch_decode_step",
            "batch_id": "decode-batch-1",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "session_id": "runtime-1",
                "logical_position": 8,
                "max_tokens_remaining": 0
            }]
        });
        let err = validate_generate_batch_decode(&zero_remaining).unwrap_err();
        assert!(err.contains("max_tokens_remaining"));

        let invalid_cached_prefix = serde_json::json!({
            "type": "generate_batch_decode_step",
            "batch_id": "decode-batch-1",
            "worker_key_id": "worker-a",
            "cached_prefix_tokens": -1,
            "sessions": [{
                "id": "req-1",
                "session_id": "runtime-1",
                "logical_position": 8,
                "max_tokens_remaining": 1
            }]
        });
        let err = validate_generate_batch_decode(&invalid_cached_prefix).unwrap_err();
        assert!(err.contains("cached_prefix_tokens"));
    }

    #[test]
    fn rejects_prefix_hash_preflight_suffix_or_unknown_policy() {
        let suffix = serde_json::json!({
            "type": "prefix_hash_preflight",
            "id": "prefix-1",
            "worker_key_id": "worker-a",
            "session": {
                "id": "req-1",
                "suffix_tokens": [1, 2, 3],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            }
        });
        let err = match validate_prefix_hash_preflight(&suffix) {
            Ok(_) => panic!("suffix preflight unexpectedly accepted"),
            Err(err) => err,
        };
        assert!(err.contains("must include prompt"));

        let bad_policy = serde_json::json!({
            "type": "prefix_hash_preflight",
            "id": "prefix-1",
            "worker_key_id": "worker-a",
            "boundary_policy": "every_token",
            "session": {
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            }
        });
        let err = match validate_prefix_hash_preflight(&bad_policy) {
            Ok(_) => panic!("bad boundary policy unexpectedly accepted"),
            Err(err) => err,
        };
        assert!(err.contains("semantic_chat_template"));
    }

    #[test]
    fn qwen35_prefix_hash_is_domain_separated() {
        let kinds = vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()];
        let reordered = vec!["deltanet_recurrent".to_string(), "attention_kv".to_string()];
        let base = compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "plain", 0, &[1, 2, 3]);
        let same = compute_qwen35_prefix_hash(5, Some("q8"), &reordered, "plain", 0, &[1, 2, 3]);
        let different_tokens =
            compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "plain", 0, &[1, 2, 4]);
        let different_think =
            compute_qwen35_prefix_hash(5, Some("q8"), &kinds, "open_think", 0, &[1, 2, 3]);

        assert_eq!(base, same);
        assert_eq!(base.algorithm, "xxh128");
        assert_eq!(base.value.len(), 32);
        assert_eq!(base.prefix_len, 3);
        assert_ne!(base, different_tokens);
        assert_ne!(base, different_think);
    }

    fn test_prefill_boundary_session(
        id: &str,
        tokens: &[u32],
        boundaries: &[usize],
    ) -> Qwen35PreparedPrefillSession {
        Qwen35PreparedPrefillSession {
            id: id.to_string(),
            tokens: tokens.to_vec(),
            cached_prefix_tokens: 0,
            replay_as_generated_suffix: false,
            state_kinds: vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()],
            assistant_prefix: "plain".to_string(),
            max_think_tokens: 0,
            boundary_checkpoints: boundaries
                .iter()
                .enumerate()
                .map(
                    |(boundary_index, &prefix_len)| Qwen35SemanticBoundaryCheckpoint {
                        checkpoint_id: None,
                        prefix_len,
                        hash: SequenceStatePrefixHash {
                            algorithm: "xxh128".to_string(),
                            value: format!("{prefix_len:032x}"),
                            prefix_len,
                        },
                        boundary: "message_end".to_string(),
                        boundary_index,
                    },
                )
                .collect(),
        }
    }

    #[test]
    fn fused_prefill_boundary_cuts_allow_single_session_serial_segments() {
        let synchronized = vec![
            test_prefill_boundary_session("req-a", &[1, 2, 3, 4], &[2]),
            test_prefill_boundary_session("req-b", &[5, 6, 7, 8], &[2]),
        ];
        assert_eq!(
            qwen35_fused_prefill_boundary_cuts(&synchronized).expect("valid cuts"),
            Some(vec![2, 4])
        );

        let uneven_tail = vec![
            test_prefill_boundary_session("req-a", &[1], &[]),
            test_prefill_boundary_session("req-b", &[5, 6, 7, 8], &[1]),
        ];
        assert_eq!(
            qwen35_fused_prefill_boundary_cuts(&uneven_tail).expect("valid mixed cuts"),
            Some(vec![1, 4])
        );
    }

    #[test]
    fn fused_prefill_boundary_cuts_cover_multiple_boundaries_and_suffix_replay_fallback() {
        let multi_cut = vec![
            test_prefill_boundary_session("req-a", &[1, 2, 3, 4], &[1, 3]),
            test_prefill_boundary_session("req-b", &[5, 6, 7, 8], &[1, 3]),
        ];
        assert_eq!(
            qwen35_fused_prefill_boundary_cuts(&multi_cut).expect("valid multi-cut layout"),
            Some(vec![1, 3, 4])
        );

        let mut replay = test_prefill_boundary_session("req-a", &[1, 2, 3, 4], &[2]);
        replay.replay_as_generated_suffix = true;
        let err = qwen35_fused_prefill_boundary_cuts(&[replay]).unwrap_err();
        assert!(err.contains("only supported for full-prompt prefill"));
    }

    #[test]
    fn model_worker_runtime_view_json_reports_state_page_descriptors() {
        let worker = ModelWorkerRuntimeView {
            worker_id: ModelWorkerId {
                value: "qwen35:worker-a".to_string(),
            },
            max_seq: 32768,
            physical_cap: 2048,
            max_resident_workers: 1,
            resident_workers: 1,
            state_arena_backend: SequenceStateArenaBackend::Qwen35Wrapped,
            resident_sessions: 1,
            state_page_descriptors: vec![
                SequenceStatePageDescriptor {
                    session_id: "checkpoint-a".to_string(),
                    handle: SequenceStateHandle {
                        id: "checkpoint-a".to_string(),
                        kind: "qwen35_session".to_string(),
                        generation: 0,
                    },
                    kind: SequenceStatePageKind::Kv,
                    label: "qwen35.kv_cache".to_string(),
                    logical_position: 16,
                    resident_bytes: 128,
                    allocation_epoch: 0,
                    owns_pages: false,
                    shape: vec![2, 16, 4, 32],
                    placement: "hip:arch5:device0".to_string(),
                    role: "resident".to_string(),
                },
                SequenceStatePageDescriptor {
                    session_id: "checkpoint-a".to_string(),
                    handle: SequenceStateHandle {
                        id: "checkpoint-a".to_string(),
                        kind: "qwen35_session".to_string(),
                        generation: 0,
                    },
                    kind: SequenceStatePageKind::DeltaNet,
                    label: "qwen35.deltanet_state".to_string(),
                    logical_position: 16,
                    resident_bytes: 96,
                    allocation_epoch: 0,
                    owns_pages: false,
                    shape: vec![48, 48, 48],
                    placement: "hip:arch5:device0".to_string(),
                    role: "resident".to_string(),
                },
                SequenceStatePageDescriptor {
                    session_id: "checkpoint-a".to_string(),
                    handle: SequenceStateHandle {
                        id: "checkpoint-a".to_string(),
                        kind: "qwen35_session".to_string(),
                        generation: 0,
                    },
                    kind: SequenceStatePageKind::Logits,
                    label: "qwen35.logits_snapshot".to_string(),
                    logical_position: 16,
                    resident_bytes: 64,
                    allocation_epoch: 0,
                    owns_pages: false,
                    shape: vec![16],
                    placement: "hip:arch5:device0".to_string(),
                    role: "resident".to_string(),
                },
                SequenceStatePageDescriptor {
                    session_id: "checkpoint-a".to_string(),
                    handle: SequenceStateHandle {
                        id: "checkpoint-a".to_string(),
                        kind: "qwen35_session".to_string(),
                        generation: 0,
                    },
                    kind: SequenceStatePageKind::BackendPrivate,
                    label: "qwen35.prefix_metadata".to_string(),
                    logical_position: 16,
                    resident_bytes: 32,
                    allocation_epoch: 0,
                    owns_pages: false,
                    shape: vec![1],
                    placement: "host".to_string(),
                    role: "resident".to_string(),
                },
            ],
            memory: ModelWorkerMemoryView {
                model_file_bytes: 256,
                model_weight_bytes: 224,
                runtime_base_bytes: 64,
                runtime_session_bytes: 320,
                runtime_state_bytes: 384,
                total_resident_bytes: 608,
                evictable_state_bytes: 320,
            },
        };

        let json = model_worker_runtime_view_json(&worker);
        assert_eq!(json["state_arena_backend"], "qwen35_wrapped");
        assert_eq!(json["state_arena_owns_pages"], true);
        assert_eq!(json["state_allocator"]["page_ownership"], "backend_wrapped");
        assert_eq!(
            json["state_allocator"]["eviction_policy"],
            "manual_release_only"
        );
        assert_eq!(json["state_allocator"]["spill_target"], "disabled");
        assert_eq!(json["state_allocator"]["copy_on_write_attach"], false);
        assert_eq!(
            json["state_arena_operations"],
            serde_json::json!([
                "reserve_session_state",
                "attach_checkpoint",
                "fork_checkpoint",
                "release_state",
                "describe_state"
            ])
        );
        assert_eq!(json["max_seq"], 32768);
        assert_eq!(json["physical_cap"], 2048);
        assert_eq!(json["state_page_descriptor_entries"], 4);
        assert_eq!(json["state_page_descriptor_bytes"], 320);
        assert_eq!(json["model_file_bytes"], 256);
        assert_eq!(json["model_weight_bytes"], 224);
        assert_eq!(json["runtime_base_bytes"], 64);
        assert_eq!(json["runtime_session_bytes"], 320);
        assert_eq!(json["runtime_state_bytes"], 384);
        assert_eq!(json["total_resident_bytes"], 608);
        assert_eq!(json["evictable_state_bytes"], 320);
        assert_eq!(
            json["state_page_descriptors"][0]["state_kind"],
            "attention_kv"
        );
        assert_eq!(
            json["state_page_descriptors"][0]["page_kind"],
            "attention_kv"
        );
        assert_eq!(
            json["state_page_descriptors"][0]["handle"]["id"],
            "checkpoint-a"
        );
        assert_eq!(json["state_page_descriptors"][0]["allocation_epoch"], 0);
        assert_eq!(json["state_page_descriptors"][0]["owns_pages"], false);
        assert_eq!(
            json["state_page_descriptors"][0]["shape"],
            serde_json::json!([2, 16, 4, 32])
        );
        assert_eq!(
            json["state_page_descriptors"][1]["state_kind"],
            "deltanet_recurrent"
        );
        assert_eq!(
            json["state_page_descriptors"][1]["label"],
            "qwen35.deltanet_state"
        );
        assert_eq!(
            json["state_page_descriptors"][1]["shape"],
            serde_json::json!([48, 48, 48])
        );
        assert_eq!(json["state_page_descriptors"][2]["state_kind"], "logits");
        assert_eq!(
            json["state_page_descriptors"][2]["label"],
            "qwen35.logits_snapshot"
        );
        assert_eq!(
            json["state_page_descriptors"][2]["shape"],
            serde_json::json!([16])
        );
        assert_eq!(
            json["state_page_descriptors"][3]["state_kind"],
            "backend_private"
        );
        assert_eq!(json["state_page_descriptors"][3]["placement"], "host");
    }

    #[test]
    fn model_worker_runtime_view_json_reports_unsupported_state_arena_without_pages() {
        let worker = ModelWorkerRuntimeView {
            worker_id: ModelWorkerId {
                value: "worker:arch9:pp1:q8".to_string(),
            },
            max_seq: 4096,
            physical_cap: 4096,
            max_resident_workers: 1,
            resident_workers: 1,
            state_arena_backend: SequenceStateArenaBackend::Unsupported,
            resident_sessions: 0,
            state_page_descriptors: Vec::new(),
            memory: ModelWorkerMemoryView {
                model_file_bytes: 512,
                model_weight_bytes: 384,
                runtime_base_bytes: 0,
                runtime_session_bytes: 0,
                runtime_state_bytes: 0,
                total_resident_bytes: 384,
                evictable_state_bytes: 0,
            },
        };

        let json = model_worker_runtime_view_json(&worker);
        assert_eq!(json["state_arena_backend"], "unsupported");
        assert_eq!(json["state_arena_owns_pages"], false);
        assert_eq!(json["state_arena_operations"], serde_json::json!([]));
        assert_eq!(json["resident_sessions"], 0);
        assert_eq!(json["state_page_descriptor_entries"], 0);
        assert_eq!(json["state_page_descriptor_bytes"], 0);
        assert_eq!(json["state_page_descriptors"], serde_json::json!([]));
        assert_eq!(json["runtime_session_bytes"], 0);
        assert_eq!(json["evictable_state_bytes"], 0);
    }

    #[test]
    fn reserve_session_state_kinds_default_deduplicate_and_alias() {
        let defaults = parse_reserve_session_state_kinds(&serde_json::json!({})).unwrap();
        assert_eq!(
            defaults,
            vec![SequenceStatePageKind::Kv, SequenceStatePageKind::DeltaNet]
        );

        let parsed = parse_reserve_session_state_kinds(&serde_json::json!({
            "state_kinds": [
                "attention_kv",
                "deltanet_recurrent",
                "attention_kv",
                "architecture_specific"
            ]
        }))
        .unwrap();
        assert_eq!(
            parsed,
            vec![
                SequenceStatePageKind::Kv,
                SequenceStatePageKind::DeltaNet,
                SequenceStatePageKind::BackendPrivate
            ]
        );

        let err = parse_reserve_session_state_kinds(&serde_json::json!({
            "state_kinds": ["unknown_state"]
        }))
        .unwrap_err();
        assert!(err.contains("unsupported kind unknown_state"));
    }

    #[test]
    fn generic_state_reservation_descriptors_are_owned_handles() {
        let handle = SequenceStateHandle {
            id: "reserve-a".to_string(),
            kind: "generic_reserved_state".to_string(),
            generation: 7,
        };
        let descriptors = generic_state_reservation_descriptors(
            "worker-a",
            &handle,
            &[
                SequenceStatePageKind::Kv,
                SequenceStatePageKind::DeltaNet,
                SequenceStatePageKind::Logits,
            ],
            2048,
            10,
        );

        assert_eq!(descriptors.len(), 3);
        assert_eq!(descriptors[0].handle.id, "reserve-a");
        assert_eq!(descriptors[0].handle.kind, "generic_reserved_state");
        assert_eq!(descriptors[0].handle.generation, 7);
        assert_eq!(descriptors[0].session_id, "reserve-a");
        assert_eq!(descriptors[0].resident_bytes, 4);
        assert_eq!(descriptors[0].allocation_epoch, 7);
        assert!(descriptors[0].owns_pages);
        assert_eq!(descriptors[1].resident_bytes, 3);
        assert_eq!(descriptors[2].resident_bytes, 3);
        assert_eq!(descriptors[0].shape, vec![2048]);
        assert_eq!(descriptors[2].shape, vec![1]);
        assert_eq!(descriptors[0].placement, "host:reserved:worker-a");
        assert_eq!(descriptors[0].role, "reserved");

        let json = sequence_state_page_descriptor_json(&descriptors[0]);
        assert_eq!(json["handle"]["kind"], "generic_reserved_state");
        assert_eq!(json["handle"]["generation"], 7);
        assert_eq!(json["state_kind"], "attention_kv");
        assert_eq!(json["resident_bytes"], 4);
        assert_eq!(json["allocation_epoch"], 7);
        assert_eq!(json["owns_pages"], true);
    }

    fn test_state_descriptor(
        id: &str,
        kind: &str,
        generation: u64,
        page_kind: SequenceStatePageKind,
        bytes: usize,
    ) -> SequenceStatePageDescriptor {
        SequenceStatePageDescriptor {
            session_id: id.to_string(),
            handle: SequenceStateHandle {
                id: id.to_string(),
                kind: kind.to_string(),
                generation,
            },
            kind: page_kind,
            label: format!("{kind}.{}", page_kind.as_str()),
            logical_position: 16,
            resident_bytes: bytes,
            allocation_epoch: generation,
            owns_pages: generation != 0,
            shape: vec![1],
            placement: "hip:arch5:device0".to_string(),
            role: "resident".to_string(),
        }
    }

    #[test]
    fn sequence_state_descriptor_lookup_binds_qwen35_epoch_handles() {
        let descriptors = vec![
            test_state_descriptor(
                "qwen35-checkpoint:batch:req:16",
                "qwen35_checkpoint",
                41,
                SequenceStatePageKind::Kv,
                128,
            ),
            test_state_descriptor(
                "qwen35-checkpoint:batch:req:16",
                "qwen35_checkpoint",
                41,
                SequenceStatePageKind::DeltaNet,
                96,
            ),
            test_state_descriptor(
                "qwen35-checkpoint:batch:req:16",
                "qwen35_checkpoint",
                42,
                SequenceStatePageKind::Logits,
                64,
            ),
        ];
        let handle = ParsedSequenceStateHandle {
            id: "qwen35-checkpoint:batch:req:16".to_string(),
            kind: Some("qwen35_checkpoint".to_string()),
            generation: Some(41),
        };
        let matched =
            describe_sequence_state_descriptors(descriptors.clone(), &handle).expect("match");
        assert_eq!(matched.len(), 2);
        assert!(matched.iter().all(|descriptor| {
            descriptor.handle.kind == "qwen35_checkpoint" && descriptor.allocation_epoch == 41
        }));

        let stale = ParsedSequenceStateHandle {
            generation: Some(40),
            ..handle
        };
        assert!(describe_sequence_state_descriptors(descriptors, &stale).is_none());
    }

    #[test]
    fn parsed_state_handle_kind_routes_generic_and_qwen35_surfaces() {
        let generic = parse_sequence_state_handle(&serde_json::json!({
            "id": "reserve-a",
            "kind": "generic_reserved_state",
            "generation": 7
        }))
        .unwrap();
        assert!(parsed_handle_may_target_generic(&generic));
        assert!(!parsed_handle_may_target_loaded_state(&generic));

        let qwen35 = parse_sequence_state_handle(&serde_json::json!({
            "id": "qwen35-checkpoint:batch:req:16",
            "kind": "qwen35_checkpoint",
            "allocation_epoch": 41
        }))
        .unwrap();
        assert!(!parsed_handle_may_target_generic(&qwen35));
        assert!(parsed_handle_may_target_loaded_state(&qwen35));
        assert_eq!(qwen35.generation, Some(41));

        let legacy = parse_sequence_state_handle(&serde_json::json!("session-a")).unwrap();
        assert!(parsed_handle_may_target_generic(&legacy));
        assert!(parsed_handle_may_target_loaded_state(&legacy));
    }

    #[test]
    fn qwen35_checkpoint_handles_report_owned_epoch_identity() {
        let session_handle = qwen35_sequence_state_handle("session-a", 11);
        assert_eq!(session_handle.id, "session-a");
        assert_eq!(session_handle.kind, "qwen35_session");
        assert_eq!(session_handle.generation, 11);

        let checkpoint_handle = qwen35_sequence_state_handle("qwen35-checkpoint:batch:req:16", 12);
        assert_eq!(checkpoint_handle.kind, "qwen35_checkpoint");
        assert_eq!(checkpoint_handle.generation, 12);
    }

    #[test]
    fn sequence_state_handle_id_accepts_string_or_handle_object() {
        assert_eq!(
            sequence_state_handle_id(&serde_json::json!("reserve-a")),
            Some("reserve-a")
        );
        assert_eq!(
            sequence_state_handle_id(&serde_json::json!({
                "id": "reserve-b",
                "kind": "generic_reserved_state",
                "generation": 9
            })),
            Some("reserve-b")
        );
        assert_eq!(
            sequence_state_handle_parts(&serde_json::json!({
                "id": "reserve-b",
                "kind": "generic_reserved_state",
                "generation": 9
            })),
            Some(("reserve-b", Some(9)))
        );
        assert_eq!(sequence_state_handle_id(&serde_json::json!("")), None);
        assert_eq!(
            sequence_state_handle_id(&serde_json::json!({"kind": "missing_id"})),
            None
        );
    }

    #[test]
    fn generic_state_arena_rejects_stale_generation_handles() {
        let mut arena = GenericSequenceStateArena::new();
        let first = arena.reserve(
            "worker-a",
            "reserve-a".to_string(),
            &[SequenceStatePageKind::Kv],
            128,
            64,
            0,
        );
        assert_eq!(first.handle.generation, 1);
        assert!(arena.describe("reserve-a", Some(1)).is_some());
        assert!(arena.describe("reserve-a", Some(2)).is_none());
        assert_eq!(
            arena.release(vec![("reserve-a".to_string(), Some(2))]),
            (0, 0)
        );
        assert!(arena.describe("reserve-a", Some(1)).is_some());
        assert_eq!(
            arena.release(vec![("reserve-a".to_string(), Some(1))]),
            (1, 64)
        );

        let second = arena.reserve(
            "worker-a",
            "reserve-a".to_string(),
            &[SequenceStatePageKind::DeltaNet],
            128,
            32,
            0,
        );
        assert_eq!(second.handle.generation, 2);
        assert!(arena.describe("reserve-a", Some(1)).is_none());
        assert!(arena.describe("reserve-a", Some(2)).is_some());
    }

    #[test]
    fn generic_state_arena_purges_ttl_and_releases_by_worker() {
        let mut arena = GenericSequenceStateArena::new();
        let expiring = arena.reserve(
            "worker-a",
            "reserve-expiring".to_string(),
            &[SequenceStatePageKind::Kv],
            128,
            16,
            1,
        );
        let persistent = arena.reserve(
            "worker-a",
            "reserve-persistent".to_string(),
            &[SequenceStatePageKind::DeltaNet],
            128,
            32,
            0,
        );
        let other_worker = arena.reserve(
            "worker-b",
            "reserve-other".to_string(),
            &[SequenceStatePageKind::Logits],
            128,
            64,
            0,
        );
        assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 48);
        assert_eq!(arena.outstanding_bytes_for_worker("worker-b"), 64);

        std::thread::sleep(std::time::Duration::from_millis(5));
        arena.purge_expired();
        assert!(arena
            .describe("reserve-expiring", Some(expiring.handle.generation))
            .is_none());
        assert!(arena
            .describe("reserve-persistent", Some(persistent.handle.generation))
            .is_some());
        assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 32);

        arena.release_worker("worker-a");
        assert!(arena
            .describe("reserve-persistent", Some(persistent.handle.generation))
            .is_none());
        assert!(arena
            .describe("reserve-other", Some(other_worker.handle.generation))
            .is_some());
        assert_eq!(arena.outstanding_bytes_for_worker("worker-a"), 0);
        assert_eq!(arena.outstanding_bytes_for_worker("worker-b"), 64);
    }

    #[test]
    fn rejects_missing_worker_identity() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "batch_id": "batch-1",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                }
            }]
        });

        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("worker_key_id or model"));
    }

    #[test]
    fn accepts_empty_suffix_for_attached_checkpoint_reuse() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "batch_id": "batch-1",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "suffix_tokens": [],
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 4,
                    "cached_prefix_tokens": 4,
                    "runtime_state_handle": "qwen35-checkpoint:batch-0:req-0:4"
                }
            }]
        });

        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        assert_eq!(
            envelope.sessions[0].suffix_tokens.as_ref().unwrap().len(),
            0
        );
        assert_eq!(envelope.sessions[0].state_handle.cached_prefix_tokens, 4);
    }

    #[test]
    fn rejects_duplicate_session_ids() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-1",
            "batch_id": "batch-dup",
            "worker_key_id": "worker-a",
            "sessions": [
                {
                    "id": "req-1",
                    "prompt": "first",
                    "state_handle": {
                        "state_kinds": ["attention_kv"],
                        "logical_position": 0
                    }
                },
                {
                    "id": "req-1",
                    "prompt": "second",
                    "state_handle": {
                        "state_kinds": ["attention_kv"],
                        "logical_position": 0
                    },
                    "params": { "max_tokens": 8 }
                }
            ]
        });

        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("duplicate session id"));
    }

    #[test]
    fn rejects_missing_state_handle_fields() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-2",
            "batch_id": "batch-missing-state",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello"
            }]
        });

        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains(".state_handle must be an object"));
    }

    #[test]
    fn rejects_invalid_state_handle_field_values() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-3",
            "batch_id": "batch-invalid-state",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv", "bad_kind"],
                    "logical_position": 0
                }
            }]
        });
        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("unsupported kind"));
    }

    #[test]
    fn rejects_state_handle_missing_state_kinds() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-kinds",
            "batch_id": "batch-missing-state-kinds",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "logical_position": 0
                }
            }]
        });
        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("state_kinds"));
    }

    #[test]
    fn rejects_state_handle_negative_cached_prefix_tokens() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-cached",
            "batch_id": "batch-cached-prefix-negative",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0,
                    "cached_prefix_tokens": -1
                }
            }]
        });
        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("cached_prefix_tokens"));
    }

    #[test]
    fn rejects_empty_runtime_state_handle() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-runtime-handle",
            "batch_id": "batch-runtime-handle",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "suffix_tokens": [1],
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 1,
                    "cached_prefix_tokens": 1,
                    "runtime_state_handle": ""
                }
            }]
        });
        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("runtime_state_handle"));
    }

    #[test]
    fn rejects_state_handle_missing_logical_position() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-logic",
            "batch_id": "batch-missing-logical-position",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"]
                }
            }]
        });
        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains("logical_position"));
    }

    #[test]
    fn rejects_unsupported_params_shape() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-4",
            "batch_id": "batch-params",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                },
                "params": ["max_tokens", 32]
            }]
        });
        let err = validate_generate_batch_prefill(&msg).unwrap_err();
        assert!(err.contains(".params must be an object"));
    }

    #[test]
    fn validates_semantic_boundary_checkpoint_param() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-boundary",
            "batch_id": "batch-boundary",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                },
                "params": {
                    "semantic_boundary_checkpoints": true
                }
            }]
        });
        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        assert!(envelope.sessions[0].semantic_boundary_checkpoints);

        let bad = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-boundary-bad",
            "batch_id": "batch-boundary-bad",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                },
                "params": {
                    "semantic_boundary_checkpoints": "yes"
                }
            }]
        });
        let err = validate_generate_batch_prefill(&bad).unwrap_err();
        assert!(err.contains(".params.semantic_boundary_checkpoints must be a boolean"));
    }

    #[test]
    fn rejects_prompt_suffix_contract_violations() {
        let msg_both = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-5",
            "batch_id": "batch-contract",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "suffix_tokens": [1],
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                }
            }]
        });
        let err_both = validate_generate_batch_prefill(&msg_both).unwrap_err();
        assert!(err_both.contains("exactly one"));

        let msg_none = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "probe-6",
            "batch_id": "batch-contract-2",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                }
            }]
        });
        let err_none = validate_generate_batch_prefill(&msg_none).unwrap_err();
        assert!(err_none.contains("exactly one"));
    }

    #[test]
    fn plans_dense_qwen35_multi_session_as_fused_candidate() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "prefill",
            "batch_id": "batch",
            "worker_key_id": "worker-a",
            "sessions": [
                {
                    "id": "req-1",
                    "prompt": "hello",
                    "state_handle": {
                        "state_kinds": ["attention_kv", "deltanet_recurrent"],
                        "logical_position": 0
                    }
                },
                {
                    "id": "req-2",
                    "prompt": "world",
                    "state_handle": {
                        "state_kinds": ["attention_kv", "deltanet_recurrent"],
                        "logical_position": 0
                    }
                }
            ]
        });

        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        assert_eq!(
            plan_generate_batch_prefill_qwen35(5, envelope.session_count),
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate
        );
    }

    #[test]
    fn keeps_singletons_serial_and_marks_moe_as_grouped_candidate() {
        let singleton = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "prefill",
            "batch_id": "batch",
            "worker_key_id": "worker-a",
            "sessions": [{
                "id": "req-1",
                "prompt": "hello",
                "state_handle": {
                    "state_kinds": ["attention_kv", "deltanet_recurrent"],
                    "logical_position": 0
                }
            }]
        });
        let singleton = validate_generate_batch_prefill(&singleton).expect("valid envelope");
        assert_eq!(
            plan_generate_batch_prefill_qwen35(5, singleton.session_count),
            GenerateBatchPrefillPlan::SerialExact
        );

        let moe = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "prefill",
            "batch_id": "batch",
            "worker_key_id": "worker-a",
            "sessions": [
                {
                    "id": "req-1",
                    "prompt": "hello",
                    "state_handle": {
                        "state_kinds": ["attention_kv", "deltanet_recurrent"],
                        "logical_position": 0
                    }
                },
                {
                    "id": "req-2",
                    "prompt": "world",
                    "state_handle": {
                        "state_kinds": ["attention_kv", "deltanet_recurrent"],
                        "logical_position": 0
                    }
                }
            ]
        });
        let moe = validate_generate_batch_prefill(&moe).expect("valid envelope");
        assert_eq!(
            plan_generate_batch_prefill_qwen35(6, moe.session_count),
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate
        );
    }

    #[test]
    fn dummy_state_counter_increments_across_prefill_and_generate() {
        let msg = serde_json::json!({
            "type": "generate_batch_prefill",
            "id": "prefill",
            "batch_id": "batch",
            "worker_key_id": "dummy-worker",
            "sessions": [{
                "id": "req-1",
                "prompt": "one two three",
                "state_handle": {
                    "state_kinds": ["attention_kv"],
                    "logical_position": 0
                }
            }]
        });
        let envelope = validate_generate_batch_prefill(&msg).expect("valid envelope");
        let mut dummy = DummyModelState::default();
        let consumed = dummy.consume_prefill_session(&envelope.sessions[0]);
        assert_eq!(consumed, 3);
        assert_eq!(dummy.sessions.get("req-1").copied(), Some(3));

        let counter = dummy.sessions.entry("req-1".to_string()).or_insert(0);
        let emitted = *counter;
        *counter += 1;
        assert_eq!(emitted, 3);
        assert_eq!(dummy.sessions.get("req-1").copied(), Some(4));
    }

    #[test]
    fn dummy_release_sessions_removes_state() {
        let mut dummy = DummyModelState::default();
        dummy.sessions.insert("keep".to_string(), 1);
        dummy.sessions.insert("drop".to_string(), 2);

        let released = dummy.release_sessions(&["drop".to_string(), "missing".to_string()]);
        assert_eq!(released, 1);
        assert_eq!(dummy.session_count(), 1);
        assert!(dummy.sessions.contains_key("keep"));
        assert!(!dummy.sessions.contains_key("drop"));
    }

    #[test]
    fn selects_fused_backend_by_default_for_fused_candidate_plans() {
        let fused_grouped_ok = || Ok::<(), String>(());
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
                None,
                fused_grouped_ok(),
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::FusedDense
        );
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
                None,
                fused_grouped_ok(),
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::FusedGroupedMoe
        );
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::SerialExact,
                Some("auto"),
                fused_grouped_ok(),
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::SerialReference
        );
    }

    #[test]
    fn selects_dense_layer_chunked_decode_backend_only_for_dense_batches() {
        assert_eq!(
            select_qwen35_decode_batch_backend("auto", 5, 2).unwrap(),
            Qwen35DecodeBatchBackend::SerialReference
        );
        assert_eq!(
            select_qwen35_decode_batch_backend("auto", 6, 8).unwrap(),
            Qwen35DecodeBatchBackend::SerialReference
        );
        assert_eq!(
            select_qwen35_decode_batch_backend("fused", 5, 2).unwrap(),
            Qwen35DecodeBatchBackend::FusedDenseLayerChunked
        );
        assert_eq!(
            select_qwen35_decode_batch_backend("fused", 5, 1).unwrap(),
            Qwen35DecodeBatchBackend::FusedDenseLayerChunked
        );
        let moe_err = select_qwen35_decode_batch_backend("fused", 6, 2).unwrap_err();
        assert!(moe_err.contains("not dense Qwen35"));
        assert_eq!(
            select_qwen35_decode_batch_backend("fused_grouped_moe", 6, 2).unwrap(),
            Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked
        );
        for batch_size in [2, 4, 8] {
            assert_eq!(
                select_qwen35_decode_batch_backend("fused_grouped_moe", 6, batch_size).unwrap(),
                Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked,
                "explicit grouped-MoE decode should admit B={batch_size}"
            );
        }
        let singleton_err =
            select_qwen35_decode_batch_backend("fused_grouped_moe", 6, 1).unwrap_err();
        assert!(singleton_err.contains("at least two sessions"));
        let grouped_dense_err =
            select_qwen35_decode_batch_backend("fused_grouped_moe", 5, 2).unwrap_err();
        assert!(grouped_dense_err.contains("not Qwen35 grouped-MoE"));
    }

    #[test]
    fn decode_batch_runtime_surface_rejects_spec_decode_and_eviction_state() {
        validate_qwen35_decode_batch_runtime_surface(5, 1, false, false).unwrap();
        validate_qwen35_decode_batch_runtime_surface(6, 1, false, false).unwrap();

        let pp_err = validate_qwen35_decode_batch_runtime_surface(5, 2, false, false).unwrap_err();
        assert!(pp_err.contains("single-GPU qwen35/qwen35-moe"));

        let arch_err =
            validate_qwen35_decode_batch_runtime_surface(9, 1, false, false).unwrap_err();
        assert!(arch_err.contains("single-GPU qwen35/qwen35-moe"));

        let dflash_err =
            validate_qwen35_decode_batch_runtime_surface(5, 1, true, false).unwrap_err();
        assert_eq!(
            dflash_err,
            "generate_batch_decode_step is not supported on DFlash-loaded models"
        );

        let eviction_err =
            validate_qwen35_decode_batch_runtime_surface(5, 1, false, true).unwrap_err();
        assert_eq!(
            eviction_err,
            "generate_batch_decode_step is not supported with active eviction state"
        );
    }

    #[test]
    fn decode_batch_scheduler_metadata_reports_backend_state_and_fallback() {
        let small_auto_grouped = qwen35_decode_batch_scheduler_metadata(
            "auto",
            6,
            Qwen35DecodeBatchBackend::SerialReference,
            2,
            3,
        );
        assert_eq!(small_auto_grouped.selected_backend, "serial_reference");
        assert_eq!(
            small_auto_grouped.fallback_reason,
            "auto_grouped_moe_serial_small_batch_latency_gate"
        );
        assert_eq!(small_auto_grouped.cached_prefix_tokens, 3);

        let auto_grouped = qwen35_decode_batch_scheduler_metadata(
            "auto",
            6,
            Qwen35DecodeBatchBackend::SerialReference,
            8,
            7,
        );
        assert_eq!(auto_grouped.selected_backend, "serial_reference");
        assert_eq!(auto_grouped.batch_size, 8);
        assert_eq!(
            auto_grouped.compatible_state_kinds,
            vec!["attention_kv", "deltanet_recurrent"]
        );
        assert_eq!(auto_grouped.cached_prefix_tokens, 7);
        assert_eq!(
            auto_grouped.fallback_reason,
            "auto_grouped_moe_serial_pending_latency_gate"
        );

        let explicit_grouped = qwen35_decode_batch_scheduler_metadata(
            "fused_grouped_moe",
            6,
            Qwen35DecodeBatchBackend::FusedGroupedMoeLayerChunked,
            4,
            0,
        );
        assert_eq!(
            explicit_grouped.selected_backend,
            "fused_grouped_moe_layer_chunked"
        );
        assert_eq!(explicit_grouped.fallback_reason, "none");

        let explicit_serial = qwen35_decode_batch_scheduler_metadata(
            "serial",
            5,
            Qwen35DecodeBatchBackend::SerialReference,
            2,
            0,
        );
        assert_eq!(
            explicit_serial.fallback_reason,
            "requested_serial_reference"
        );
    }

    #[test]
    fn grouped_moe_decode_auto_latency_gate_requires_b4_or_larger() {
        assert!(!qwen35_grouped_moe_decode_auto_latency_gate_passed(2));
        assert!(qwen35_grouped_moe_decode_auto_latency_gate_passed(4));
        assert!(qwen35_grouped_moe_decode_auto_latency_gate_passed(8));
    }

    #[test]
    fn fused_dense_decode_accepts_only_fp32_uncompacted_state_signatures() {
        let config = test_dense_qwen35_config();
        let fp32 = fp32_decode_state_signature();
        validate_qwen35_fused_dense_decode_session_signatures(&config, &[fp32, fp32], 2)
            .expect("fp32 dense decode state should be admitted");

        let mut compacted = fp32;
        compacted.kv_compact_offset = 8;
        let err = validate_qwen35_fused_dense_decode_session_signatures(
            &config,
            &[compacted, compacted],
            2,
        )
        .unwrap_err();
        assert!(err.contains("compacted KV offset"));

        let mut quantized_kv = fp32;
        quantized_kv.kv_quantized = true;
        quantized_kv.kv_quant_q8 = true;
        let err = validate_qwen35_fused_dense_decode_session_signatures(
            &config,
            &[quantized_kv, quantized_kv],
            2,
        )
        .unwrap_err();
        assert!(err.contains("quantized KV state"));

        let mut q8_dn = fp32;
        q8_dn.dn_quant = qwen35::StateQuant::Q8;
        let err =
            validate_qwen35_fused_dense_decode_session_signatures(&config, &[q8_dn, q8_dn], 2)
                .unwrap_err();
        assert!(err.contains("Q8 DeltaNet state"));
    }

    #[test]
    fn fused_dense_decode_batch_max_can_force_smaller_chunks() {
        unsafe {
            std::env::set_var("HIPFIRE_QWEN35_DECODE_BATCH_MAX", "1");
        }
        assert_eq!(qwen35_decode_batch_max_chunk_size(4), 1);
        unsafe {
            std::env::set_var("HIPFIRE_QWEN35_DECODE_BATCH_MAX", "3");
        }
        assert_eq!(qwen35_decode_batch_max_chunk_size(8), 3);
        unsafe {
            std::env::remove_var("HIPFIRE_QWEN35_DECODE_BATCH_MAX");
        }
        assert_eq!(qwen35_decode_batch_max_chunk_size(4), 4);
    }

    #[test]
    fn native_decode_chunk_ranges_support_bounded_chunks() {
        assert_eq!(
            qwen35_decode_native_chunk_ranges(4, 1).unwrap(),
            vec![(0, 1), (1, 2), (2, 3), (3, 4)]
        );
        assert_eq!(
            qwen35_decode_native_chunk_ranges(1, 1).unwrap(),
            vec![(0, 1)]
        );
        assert_eq!(
            qwen35_decode_native_chunk_ranges(8, 4).unwrap(),
            vec![(0, 4), (4, 8)]
        );
        assert_eq!(
            qwen35_decode_native_chunk_ranges(5, 4).unwrap(),
            vec![(0, 4), (4, 5)]
        );
        assert_eq!(
            qwen35_decode_native_chunk_ranges(4, 2).unwrap(),
            vec![(0, 2), (2, 4)]
        );
        assert_eq!(
            qwen35_decode_native_chunk_ranges(8, 2).unwrap(),
            vec![(0, 2), (2, 4), (4, 6), (6, 8)]
        );
    }

    #[test]
    fn grouped_moe_decode_contract_admits_q8_state_batches_and_rejects_fallback_cases() {
        let config = test_grouped_moe_qwen35_config();
        for batch_size in [2, 4, 8] {
            let signatures = vec![q8_decode_state_signature(); batch_size];
            validate_qwen35_grouped_moe_decode_session_signatures(
                &config,
                &signatures,
                batch_size,
                "gfx1151",
            )
            .unwrap_or_else(|err| panic!("B={batch_size} should admit grouped-MoE decode: {err}"));
        }

        let fp32 = vec![fp32_decode_state_signature(); 2];
        let err =
            validate_qwen35_grouped_moe_decode_session_signatures(&config, &fp32, 2, "gfx1151")
                .unwrap_err();
        assert!(err.contains("must use Q8 KV state"));

        let dense = test_dense_qwen35_config();
        let q8 = vec![q8_decode_state_signature(); 2];
        let err = validate_qwen35_grouped_moe_decode_session_signatures(&dense, &q8, 2, "gfx1151")
            .unwrap_err();
        assert!(err.contains("requires Qwen35 MoE/A3B weights"));

        let err = validate_qwen35_grouped_moe_decode_session_signatures(&config, &q8, 2, "gfx906")
            .unwrap_err();
        assert!(err.contains("requires an RDNA grouped-MoE target"));
    }

    #[test]
    fn auto_grouped_moe_decode_stays_serial_when_native_route_is_unsupported() {
        let config = test_grouped_moe_qwen35_config();
        let signatures = vec![q8_decode_state_signature(); 8];
        let unsupported = validate_qwen35_grouped_moe_decode_session_signatures(
            &config,
            &signatures,
            8,
            "gfx906",
        )
        .unwrap_err();
        assert!(unsupported.contains("requires an RDNA grouped-MoE target"));

        let backend = select_qwen35_decode_batch_backend("auto", 6, 8).unwrap();
        assert_eq!(backend, Qwen35DecodeBatchBackend::SerialReference);
        let metadata = qwen35_decode_batch_scheduler_metadata("auto", 6, backend, 8, 11);
        assert_eq!(metadata.selected_backend, "serial_reference");
        assert_eq!(
            metadata.fallback_reason,
            "auto_grouped_moe_serial_pending_latency_gate"
        );
        assert_eq!(metadata.cached_prefix_tokens, 11);
    }

    #[test]
    fn admits_fused_backend_only_for_dense_fused_candidate_plan() {
        let fused_grouped_ok = || Ok::<(), String>(());
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
                Some("fused"),
                fused_grouped_ok(),
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::FusedDense
        );
        let err = select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::SerialExact,
            Some("fused"),
            fused_grouped_ok(),
        )
        .unwrap_err();
        assert!(err.contains("not fused-eligible"));
        let moe_err = select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
            Some("fused"),
            fused_grouped_ok(),
        )
        .unwrap();
        assert_eq!(moe_err, Qwen35PrefillBatchBackend::FusedGroupedMoe);
    }

    #[test]
    fn admits_explicit_grouped_moe_backend_only_for_grouped_candidate_plan() {
        let fused_grouped_ok = || Ok::<(), String>(());
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
                Some("fused_moe"),
                fused_grouped_ok(),
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::FusedGroupedMoe
        );
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
                Some("grouped_moe"),
                fused_grouped_ok(),
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::FusedGroupedMoe
        );
        let dense_err = select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
            Some("fused_moe"),
            fused_grouped_ok(),
        )
        .unwrap_err();
        assert!(dense_err.contains("not grouped-MoE eligible"));
        let serial_err = select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::SerialExact,
            Some("grouped_moe"),
            fused_grouped_ok(),
        )
        .unwrap_err();
        assert!(serial_err.contains("not grouped-MoE eligible"));
    }

    #[test]
    fn auto_grouped_moe_falls_back_to_serial_when_fused_capability_fails() {
        let unsupported = Err::<(), String>(
            "grouped MoE session fused prefix currently requires K_TOP=8, got 10".to_string(),
        );
        assert_eq!(
            select_qwen35_prefill_batch_backend(
                GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
                None,
                unsupported.clone(),
            )
            .unwrap(),
            Qwen35PrefillBatchBackend::SerialReference
        );
        let err = select_qwen35_prefill_batch_backend(
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
            Some("fused_moe"),
            unsupported,
        )
        .unwrap_err();
        assert!(err.contains("requires K_TOP=8"));
    }

    #[test]
    fn paged_prefill_scratch_defaults_to_live_rows() {
        assert_eq!(
            qwen35_prefill_scratch_target_batch(true, 16, None, qwen35::PREFILL_MAX_BATCH),
            16
        );
        assert_eq!(
            qwen35_prefill_scratch_target_batch(true, 1, None, qwen35::PREFILL_MAX_BATCH),
            2
        );
        assert_eq!(
            qwen35_prefill_scratch_target_batch(true, 16, Some("64"), qwen35::PREFILL_MAX_BATCH),
            64
        );
        assert_eq!(
            qwen35_prefill_scratch_target_batch(false, 16, None, qwen35::PREFILL_MAX_BATCH),
            qwen35::PREFILL_MAX_BATCH
        );
    }

    fn test_prepared_prefill_session(
        id: &str,
        tokens: Vec<u32>,
        cached_prefix_tokens: usize,
        replay_as_generated_suffix: bool,
    ) -> Qwen35PreparedPrefillSession {
        Qwen35PreparedPrefillSession {
            id: id.to_string(),
            tokens,
            cached_prefix_tokens,
            replay_as_generated_suffix,
            state_kinds: vec!["attention_kv".to_string(), "deltanet_recurrent".to_string()],
            assistant_prefix: "plain".to_string(),
            max_think_tokens: 0,
            boundary_checkpoints: Vec::new(),
        }
    }

    #[test]
    fn fused_dense_preflight_rejects_non_fused_candidate_plan() {
        let prepared = vec![test_prepared_prefill_session(
            "req-1",
            vec![1, 2, 3],
            0,
            false,
        )];

        let err = validate_qwen35_fused_dense_prefill_batch_preflight(
            &prepared,
            GenerateBatchPrefillPlan::SerialExact,
        )
        .unwrap_err();
        assert!(err.contains("requires plan=fused_dense_qwen35_candidate"));
    }

    #[test]
    fn fused_dense_preflight_rejects_empty_session_token_slices() {
        let prepared = vec![
            test_prepared_prefill_session("req-1", Vec::new(), 0, true),
            test_prepared_prefill_session("req-2", vec![4], 0, true),
        ];

        let err = validate_qwen35_fused_dense_prefill_batch_preflight(
            &prepared,
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        )
        .unwrap_err();
        assert!(err.contains("requires non-empty session token slices"));
    }

    #[test]
    fn fused_dense_preflight_rejects_single_session_batches() {
        let prepared = vec![test_prepared_prefill_session(
            "req-1",
            vec![1, 2, 3],
            0,
            false,
        )];

        let err = validate_qwen35_fused_dense_prefill_batch_preflight(
            &prepared,
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        )
        .unwrap_err();
        assert!(err.contains("requires at least two sessions"));
    }

    #[test]
    fn fused_dense_preflight_rejects_mixed_prompt_and_suffix_batches() {
        let prepared = vec![
            test_prepared_prefill_session("req-1", vec![1, 2, 3], 0, false),
            test_prepared_prefill_session("req-2", vec![4], 16, true),
        ];

        let err = validate_qwen35_fused_dense_prefill_batch_preflight(
            &prepared,
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        )
        .unwrap_err();
        assert!(err.contains("cannot mix full-prompt prefill and generated-suffix replay"));
    }

    #[test]
    fn fused_grouped_moe_preflight_uses_grouped_candidate_plan() {
        let prepared = vec![
            test_prepared_prefill_session("req-1", vec![1, 2, 3], 0, false),
            test_prepared_prefill_session("req-2", vec![4], 0, false),
        ];

        validate_qwen35_fused_grouped_moe_prefill_batch_preflight(
            &prepared,
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate,
        )
        .expect("valid grouped-MoE preflight");
        let err = validate_qwen35_fused_grouped_moe_prefill_batch_preflight(
            &prepared,
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        )
        .unwrap_err();
        assert!(err.contains("requires plan=grouped_moe_qwen35_candidate"));
    }

    #[test]
    fn builds_dense_fused_worker_contract_for_prompt_batch() {
        let prepared = vec![
            test_prepared_prefill_session("req-1", vec![1, 2, 3], 0, false),
            test_prepared_prefill_session("req-2", vec![4, 5], 0, false),
        ];

        let contract = build_qwen35_fused_dense_prefill_batch_contract(
            &prepared,
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        )
        .expect("valid fused dense contract");
        assert_eq!(contract.total_tokens, 5);
        assert_eq!(
            contract.input_kind,
            Qwen35FusedDensePrefillInputKind::FullPrompt
        );
        assert_eq!(contract.sessions[0].id, "req-1");
        assert_eq!(contract.sessions[1].tokens, &[4, 5]);
    }

    #[test]
    fn builds_dense_fused_worker_contract_for_suffix_batch() {
        let prepared = vec![
            test_prepared_prefill_session("req-1", vec![11], 8, true),
            test_prepared_prefill_session("req-2", vec![12, 13], 8, true),
        ];

        let contract = build_qwen35_fused_dense_prefill_batch_contract(
            &prepared,
            GenerateBatchPrefillPlan::FusedDenseQwen35Candidate,
        )
        .expect("valid fused dense suffix contract");
        assert_eq!(contract.total_tokens, 3);
        assert_eq!(
            contract.input_kind,
            Qwen35FusedDensePrefillInputKind::GeneratedSuffixReplay
        );
        assert_eq!(contract.sessions[0].cached_prefix_tokens, 8);
        assert!(contract.sessions[1].replay_as_generated_suffix);
    }
}

fn emit_qwen35_prefill_checkpoint(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    arena_backend: SequenceStateArenaBackend,
    hook: Qwen35PrefillCheckpointHook<'_>,
) -> Result<String, String> {
    if qwen35_prefill_checkpoint_boundary_kind(hook).is_empty() {
        return Err("qwen35 prefill checkpoint boundary kind is empty".to_string());
    }
    let checkpoint_id = qwen35_prefill_checkpoint_session_id(hook);
    sequence_state_arena_checkpoint_session_state(
        arena_backend,
        m,
        gpu,
        SequenceStateCheckpointRequest {
            source_session_id: hook.source_state_handle,
            dest_session_id: &checkpoint_id,
            expected_logical_position: hook.logical_position,
            requested_prefix_hash: None,
            checkpoint_prefix_hash: Some(hook.prefix_hash),
        },
    )?;
    Ok(checkpoint_id)
}

fn qwen35_prefill_active_session(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    tokens: &[u32],
    replay_as_generated_suffix: bool,
) -> Result<usize, String> {
    if tokens.is_empty() {
        return Ok(0);
    }
    if m.seq_pos + tokens.len() > m.physical_cap {
        return Err(format!(
            "generate_batch_prefill exceeds loaded KV budget: seq_pos={} + prefill={} > physical_cap={}",
            m.seq_pos,
            tokens.len(),
            m.physical_cap
        ));
    }
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
        .ok_or_else(|| "qwen35 scratch missing; PP batch-prefill is not supported".to_string())?;
    let kv = m
        .kv_cache
        .as_mut()
        .ok_or_else(|| "qwen35 active session missing KV cache".to_string())?;
    let dn = m
        .dn_state
        .as_mut()
        .ok_or_else(|| "qwen35 active session missing DeltaNet state".to_string())?;
    if replay_as_generated_suffix {
        for &token in tokens {
            m.conversation_tokens.push(token);
            qwen35::forward_scratch(gpu, weights, config, token, m.seq_pos, kv, dn, scratch)
                .map_err(|e| format!("qwen35 forward_scratch suffix replay failed: {e:?}"))?;
            m.seq_pos += 1;
        }
    } else {
        let pos = m.seq_pos;
        qwen35::forward_prefill_batch(
            gpu, weights, config, tokens, pos, kv, dn, scratch, None, None, None, None,
        )
        .map_err(|e| format!("qwen35 forward_prefill_batch failed: {e:?}"))?;
        m.seq_pos += tokens.len();
        m.conversation_tokens.extend_from_slice(tokens);
    }
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("qwen35 batch-prefill session sync failed: {e:?}"))?;
    m.q35_active_prefilled_generated_suffix_len = if replay_as_generated_suffix {
        tokens.len()
    } else {
        0
    };
    Ok(tokens.len())
}

fn qwen35_prefill_owned_session_serial_segment(
    gpu: &mut rdna_compute::Gpu,
    weights: &qwen35::Qwen35Weights,
    config: &qwen35::Qwen35Config,
    scratch: &qwen35::Qwen35Scratch,
    state: &mut Qwen35RequestSessionState,
    tokens: &[u32],
) -> Result<usize, String> {
    for &token in tokens {
        qwen35::forward_scratch(
            gpu,
            weights,
            config,
            token,
            state.seq_pos,
            &mut state.kv_cache,
            &mut state.dn_state,
            scratch,
        )
        .map_err(|e| format!("qwen35 serial boundary prefill segment failed: {e:?}"))?;
        state.seq_pos += 1;
        state.conversation_tokens.push(token);
    }
    Ok(tokens.len())
}

fn qwen35_materialize_batch_prefill_prompt(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
) -> Result<Vec<u32>, String> {
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| "tokenizer not loaded".to_string())?;
    let prompt = session.prompt.as_deref().unwrap_or("");
    let prompt_norm = normalize_daemon_prompt(prompt);
    let prompt = prompt_norm.as_ref();
    let raw_q_tokens = tokenizer.encode(prompt);
    // Prompt-hash/preload sessions that declare a zero logical position need
    // to materialize the full prompt from position zero even if another active
    // resident session has advanced m.seq_pos. Attached prompt sessions also
    // render from zero so the daemon can slice off the cached prefix that was
    // fingerprinted by prefix_hash_preflight.
    let seq_pos_for_prompt = if session.state_handle.runtime_state_handle.is_some()
        || (session.state_handle.logical_position == 0
            && session.state_handle.cached_prefix_tokens == 0)
    {
        0
    } else {
        m.seq_pos
    };
    let assistant_prefix =
        prompt_frame::AssistantPrefix::from_label(Some(&session.assistant_prefix));
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && seq_pos_for_prompt == 0 && m.chat_template.is_some();
    let system_prompt = session.system_prompt.as_deref();
    let tools = session.tools.as_deref();
    let messages_history = session.messages_history.as_deref();

    if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: session.max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => Ok(tokenizer.encode(&rendered)),
            Err(e) => {
                eprintln!(
                    "[daemon] batch-prefill jinja render failed ({e}) -- falling back to Plain"
                );
                Ok(prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build_with_user_tokens(&raw_q_tokens))
            }
        }
    } else {
        Ok(prompt_frame::ChatFrame {
            tokenizer,
            system: if seq_pos_for_prompt == 0 {
                system_prompt
            } else {
                None
            },
            user: "",
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build_with_user_tokens(&raw_q_tokens))
    }
}

fn qwen35_prefix_hash_candidates(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
) -> Result<Vec<PrefixHashPreflightCandidate>, String> {
    let full_tokens = qwen35_materialize_batch_prefill_prompt(m, session)?;
    qwen35_prefix_hash_candidates_for_tokens(m, session, &full_tokens)
}

fn qwen35_prefix_hash_candidates_for_tokens(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
    full_tokens: &[u32],
) -> Result<Vec<PrefixHashPreflightCandidate>, String> {
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or_else(|| "tokenizer not loaded".to_string())?;
    let full_hash = compute_qwen35_prefix_hash(
        m.arch_id,
        m.q35_kv_mode.as_deref(),
        &session.state_handle.state_kinds,
        &session.assistant_prefix,
        session.max_think_tokens,
        full_tokens,
    );
    let mut candidates = Vec::new();
    let boundary_tokens: Vec<(&str, Vec<u32>)> = [
        ("message_end", "<|im_end|>"),
        ("vision_end", "<|vision_end|>"),
        ("tool_end", "<|tool_call_end|>"),
        ("tool_response_end", "<|tool_response_end|>"),
    ]
    .into_iter()
    .filter_map(|(boundary, marker)| {
        let marker_tokens = tokenizer
            .special_token_id(marker)
            .map(|id| vec![id])
            .unwrap_or_else(|| tokenizer.encode(marker));
        if marker_tokens.is_empty() {
            None
        } else {
            Some((boundary, marker_tokens))
        }
    })
    .collect();
    let mut boundary_index = 0usize;
    let mut push_boundary_candidate = |candidates: &mut Vec<PrefixHashPreflightCandidate>,
                                       prefix_len: usize,
                                       boundary: &str| {
        if prefix_len == 0 || prefix_len >= full_tokens.len() {
            return;
        }
        let hash = compute_qwen35_prefix_hash(
            m.arch_id,
            m.q35_kv_mode.as_deref(),
            &session.state_handle.state_kinds,
            &session.assistant_prefix,
            session.max_think_tokens,
            &full_tokens[..prefix_len],
        );
        if !candidates
            .iter()
            .any(|candidate: &PrefixHashPreflightCandidate| {
                candidate.hash.prefix_len == hash.prefix_len && candidate.hash.value == hash.value
            })
        {
            candidates.push(PrefixHashPreflightCandidate {
                hash,
                boundary: boundary.to_string(),
                boundary_index,
                checkpoint_id: None,
            });
            boundary_index += 1;
        }
    };
    for (idx, _) in full_tokens.iter().enumerate() {
        let prefix_len = idx + 1;
        if prefix_len >= full_tokens.len() {
            continue;
        }
        let Some((boundary, _)) = boundary_tokens.iter().find(|(_, marker_tokens)| {
            prefix_len >= marker_tokens.len()
                && full_tokens[prefix_len - marker_tokens.len()..prefix_len] == marker_tokens[..]
        }) else {
            continue;
        };
        push_boundary_candidate(&mut candidates, prefix_len, boundary);
    }

    let assistant_start: Vec<u32> = [
        tokenizer.encode("<|im_start|>"),
        tokenizer.encode("assistant"),
        tokenizer.encode("\n"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !assistant_start.is_empty() && full_tokens.len() > assistant_start.len() {
        for idx in 0..=full_tokens.len() - assistant_start.len() {
            if full_tokens[idx..idx + assistant_start.len()] == assistant_start[..] {
                push_boundary_candidate(&mut candidates, idx, "assistant_turn_start");
            }
        }
    }
    candidates.push(PrefixHashPreflightCandidate {
        hash: full_hash,
        boundary: "full".to_string(),
        boundary_index: candidates.len(),
        checkpoint_id: None,
    });
    candidates.sort_by_key(|candidate| candidate.hash.prefix_len);
    Ok(candidates)
}

fn qwen35_semantic_boundary_checkpoints(
    m: &LoadedModel,
    session: &GenerateBatchPrefillSession,
    full_tokens: &[u32],
) -> Result<Vec<Qwen35SemanticBoundaryCheckpoint>, String> {
    if !session.semantic_boundary_checkpoints {
        return Ok(Vec::new());
    }
    if matches!(
        std::env::var("HIPFIRE_PREFIX_BOUNDARY_CHECKPOINTS")
            .ok()
            .as_deref(),
        Some("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ) {
        return Ok(Vec::new());
    }
    let candidates = qwen35_prefix_hash_candidates_for_tokens(m, session, full_tokens)?;
    if std::env::var_os("HIPFIRE_DEBUG_PREFIX_BOUNDARIES").is_some() {
        eprintln!(
            "[daemon] prefix boundary candidates session={} tokens={} candidates={}",
            session.id,
            full_tokens.len(),
            candidates.len()
        );
        for candidate in &candidates {
            eprintln!(
                "[daemon] prefix boundary candidate session={} boundary={} index={} len={} hash={}",
                session.id,
                candidate.boundary,
                candidate.boundary_index,
                candidate.hash.prefix_len,
                candidate.hash.value
            );
        }
    }
    Ok(candidates
        .into_iter()
        .filter(|candidate| candidate.boundary != "full")
        .filter(|candidate| candidate.hash.prefix_len > 0)
        .filter(|candidate| candidate.hash.prefix_len < full_tokens.len())
        .map(|candidate| Qwen35SemanticBoundaryCheckpoint {
            checkpoint_id: None,
            prefix_len: candidate.hash.prefix_len,
            hash: candidate.hash,
            boundary: candidate.boundary,
            boundary_index: candidate.boundary_index,
        })
        .collect())
}

fn run_prefix_hash_preflight_qwen35(
    m: &LoadedModel,
    stdout: &mut std::io::Stdout,
    envelope: &PrefixHashPreflightEnvelope,
) -> Result<(), String> {
    if !is_qwen35_family_arch_id(m.arch_id) {
        return Err(format!(
            "prefix_hash_preflight currently supports qwen35/qwen35-moe only (arch_id={})",
            m.arch_id
        ));
    }
    if envelope.boundary_policy != "semantic_chat_template" {
        return Err(
            "prefix_hash_preflight.boundary_policy must be semantic_chat_template".to_string(),
        );
    }
    let candidates = qwen35_prefix_hash_candidates(m, &envelope.session)?;
    let line =
        prefix_hash_preflight_done_json(&envelope.id, &envelope.boundary_policy, &candidates)?;
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
    Ok(())
}

fn qwen35_prefill_suffix_batch(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    let (attach_only, non_empty): (Vec<_>, Vec<_>) = prepared
        .iter()
        .partition(|session| session.tokens.is_empty());
    if !attach_only.is_empty() {
        let effective_backend = if non_empty.len() < 2 {
            Qwen35PrefillBatchBackend::SerialReference
        } else {
            backend
        };
        let mut sessions_by_id: HashMap<String, Qwen35PrefillSessionResult> = HashMap::new();
        let mut total_prefill_tokens = 0usize;
        let mut mode = match effective_backend {
            Qwen35PrefillBatchBackend::SerialReference => "serial_prefill",
            Qwen35PrefillBatchBackend::FusedDense => "qwen35_fused_dense_prefill",
            Qwen35PrefillBatchBackend::FusedGroupedMoe => "qwen35_fused_grouped_moe_prefill",
        };
        if !non_empty.is_empty() {
            let non_empty_prepared: Vec<Qwen35PreparedPrefillSession> =
                non_empty.into_iter().cloned().collect();
            let result = qwen35_prefill_suffix_batch(
                m,
                gpu,
                batch_id,
                &non_empty_prepared,
                plan,
                effective_backend,
            )?;
            total_prefill_tokens += result.total_prefill_tokens;
            mode = result.mode;
            for session in result.sessions {
                sessions_by_id.insert(session.id.clone(), session);
            }
        }
        for session in attach_only {
            qwen35_activate_session(m, gpu, &session.id)?;
            qwen35_save_active_session(m, gpu)?;
            let saved = m.q35_sessions.get(&session.id).ok_or_else(|| {
                format!(
                    "qwen35 attach-only session {} missing after activation",
                    session.id
                )
            })?;
            let logical_position = saved.seq_pos + saved.kv_cache.compact_offset;
            let prefix_hash = compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                &session.state_kinds,
                &session.assistant_prefix,
                session.max_think_tokens,
                &saved.conversation_tokens,
            );
            if let Some(saved) = m.q35_sessions.get_mut(&session.id) {
                saved.prefix_hash = Some(prefix_hash.clone());
            }
            sessions_by_id.insert(
                session.id.clone(),
                Qwen35PrefillSessionResult {
                    id: session.id.clone(),
                    prefill_tokens: 0,
                    logical_position,
                    cached_prefix_tokens: session.cached_prefix_tokens,
                    prefix_hash,
                    debug_sample_token: None,
                    boundary_checkpoints: Vec::new(),
                },
            );
        }
        let sessions = prepared
            .iter()
            .map(|session| {
                sessions_by_id
                    .remove(&session.id)
                    .ok_or_else(|| format!("qwen35 prefill result missing session {}", session.id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Qwen35PrefillBatchResult {
            mode,
            plan,
            backend: effective_backend,
            total_prefill_tokens,
            sessions,
        });
    }

    if matches!(
        backend,
        Qwen35PrefillBatchBackend::FusedDense | Qwen35PrefillBatchBackend::FusedGroupedMoe
    ) {
        if let Err(err) = qwen35_fused_prefill_boundary_cuts(prepared) {
            if std::env::var_os("HIPFIRE_DEBUG_PREFIX_BOUNDARIES").is_some() {
                eprintln!("[daemon] fused prefill boundary checkpoint fallback: {err}");
            }
            return qwen35_prefill_suffix_batch_serial_reference(
                m,
                gpu,
                batch_id,
                prepared,
                plan,
                Qwen35PrefillBatchBackend::SerialReference,
            );
        }
    }

    match backend {
        Qwen35PrefillBatchBackend::SerialReference => {
            qwen35_prefill_suffix_batch_serial_reference(m, gpu, batch_id, prepared, plan, backend)
        }
        Qwen35PrefillBatchBackend::FusedDense => {
            qwen35_prefill_suffix_batch_fused_dense(m, gpu, batch_id, prepared, plan, backend)
        }
        Qwen35PrefillBatchBackend::FusedGroupedMoe => {
            qwen35_prefill_suffix_batch_fused_grouped_moe(m, gpu, batch_id, prepared, plan, backend)
        }
    }
}

fn emit_qwen35_owned_prefill_checkpoint(
    sessions: &mut HashMap<String, Qwen35RequestSessionState>,
    gpu: &mut rdna_compute::Gpu,
    hook: Qwen35PrefillCheckpointHook<'_>,
    source: &mut Qwen35RequestSessionState,
) -> Result<String, String> {
    if qwen35_prefill_checkpoint_boundary_kind(hook).is_empty() {
        return Err("qwen35 prefill checkpoint boundary kind is empty".to_string());
    }
    let logical_position = source.seq_pos + source.kv_cache.compact_offset;
    if logical_position != hook.logical_position {
        return Err(format!(
            "qwen35 owned prefill checkpoint source {} logical_position mismatch: expected={} resident={}",
            hook.source_state_handle, hook.logical_position, logical_position
        ));
    }
    source.prefix_hash = Some(hook.prefix_hash.clone());
    let checkpoint_id = qwen35_prefill_checkpoint_session_id(hook);
    let checkpoint = Qwen35RequestSessionState::fork_from(gpu, source)?;
    sessions.insert(checkpoint_id.clone(), checkpoint);
    Ok(checkpoint_id)
}

fn qwen35_prefill_suffix_batch_fused_grouped_moe(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    if plan != GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate {
        return Err(format!(
            "qwen35 grouped-MoE fused prefill-session batch worker requires plan={}, got {}",
            GenerateBatchPrefillPlan::GroupedMoeQwen35Candidate.as_str(),
            plan.as_str()
        ));
    }
    validate_qwen35_fused_grouped_moe_prefill_batch_preflight(prepared, plan)?;
    qwen35_save_active_session(m, gpu)?;
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    if config.num_experts == 0 || !config.has_shared_expert {
        return Err(
            "qwen35 grouped-MoE fused prefill-session batch requires MoE/A3B weights".to_string(),
        );
    }
    let weights = m
        .q35_weights
        .as_ref()
        .ok_or_else(|| "qwen35 weights missing".to_string())?;
    let boundary_cuts = qwen35_fused_prefill_boundary_cuts(prepared)?;
    let mut owned_sessions: Vec<(String, Qwen35RequestSessionState)> =
        Vec::with_capacity(prepared.len());
    for spec in prepared {
        let state = match m.q35_sessions.remove(&spec.id) {
            Some(state) => state,
            None => match qwen35_allocate_session_state(m, gpu) {
                Ok(state) => state,
                Err(e) => {
                    for (restore_id, restore_state) in owned_sessions {
                        m.q35_sessions.insert(restore_id, restore_state);
                    }
                    return Err(e);
                }
            },
        };
        if state.seq_pos + spec.tokens.len() > m.physical_cap {
            let id = spec.id.to_string();
            let seq_pos = state.seq_pos;
            m.q35_sessions.insert(id.clone(), state);
            for (restore_id, restore_state) in owned_sessions {
                m.q35_sessions.insert(restore_id, restore_state);
            }
            return Err(format!(
                "generate_batch_prefill exceeds loaded KV budget for session {}: seq_pos={} + prefill={} > physical_cap={}",
                id,
                seq_pos,
                spec.tokens.len(),
                m.physical_cap
            ));
        }
        owned_sessions.push((spec.id.to_string(), state));
    }

    if let Some(boundary_cuts) = boundary_cuts {
        let total_tokens = prepared.iter().map(|spec| spec.tokens.len()).sum::<usize>();
        let mut progress = vec![0usize; prepared.len()];
        let mut boundary_checkpoints_by_session = vec![Vec::new(); prepared.len()];
        let mut shape_total_tokens = 0usize;
        for &cut in &boundary_cuts {
            let active_indices: Vec<usize> = prepared
                .iter()
                .enumerate()
                .filter_map(|(idx, spec)| {
                    let end = spec.tokens.len().min(cut);
                    (progress[idx] < end).then_some(idx)
                })
                .collect();
            if active_indices.len() < 2 {
                let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                    "qwen35 scratch missing; grouped-MoE serial boundary segment is pp=1 only"
                        .to_string()
                })?;
                for &idx in &active_indices {
                    let start = progress[idx];
                    let end = prepared[idx].tokens.len().min(cut);
                    let state = &mut owned_sessions[idx].1;
                    let segment_tokens = match qwen35_prefill_owned_session_serial_segment(
                        gpu,
                        weights,
                        config,
                        scratch,
                        state,
                        &prepared[idx].tokens[start..end],
                    ) {
                        Ok(tokens) => tokens,
                        Err(err) => {
                            for (id, state) in owned_sessions {
                                m.q35_sessions.insert(id, state);
                            }
                            return Err(err);
                        }
                    };
                    shape_total_tokens += segment_tokens;
                    progress[idx] = end;
                    for mut boundary in prepared[idx]
                        .boundary_checkpoints
                        .iter()
                        .filter(|boundary| boundary.prefix_len == end)
                        .cloned()
                    {
                        let hook = Qwen35PrefillCheckpointHook {
                            batch_id,
                            session_id: &prepared[idx].id,
                            source_state_handle: &prepared[idx].id,
                            logical_position: end,
                            kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                                boundary: &boundary.boundary,
                                boundary_index: boundary.boundary_index,
                            },
                            prefix_hash: &boundary.hash,
                        };
                        let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                        let checkpoint_id = emit_qwen35_owned_prefill_checkpoint(
                            &mut m.q35_sessions,
                            gpu,
                            hook,
                            state,
                        )
                        .map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                prepared[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                        boundary.checkpoint_id = Some(checkpoint_id);
                        boundary_checkpoints_by_session[idx].push(boundary);
                    }
                }
                continue;
            }
            let worker_result = {
                let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
                    "qwen35 scratch missing; grouped-MoE fused prefill is pp=1 only".to_string()
                })?;
                let scratch_target_batch = qwen35_prefill_scratch_target_batch(
                    config.paged_experts,
                    total_tokens,
                    std::env::var("HIPFIRE_PREFILL_MAX_BATCH").ok().as_deref(),
                    qwen35::PREFILL_MAX_BATCH,
                );
                let needs_scratch = scratch
                    .prefill_batch
                    .as_ref()
                    .map(|pbs| pbs.max_batch < scratch_target_batch)
                    .unwrap_or(true);
                if needs_scratch {
                    if let Some(existing) = scratch.prefill_batch.take() {
                        existing.free_gpu(gpu);
                    }
                    scratch.prefill_batch = Some(
                        qwen35::PrefillBatchScratch::new(gpu, config, scratch_target_batch)
                            .map_err(|e| {
                                format!("allocate qwen35 grouped-MoE fused prefill scratch: {e:?}")
                            })?,
                    );
                }
                let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
                if pbs_max_batch < total_tokens {
                    return Err(format!(
                        "qwen35 grouped-MoE fused prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                        total_tokens,
                    ));
                }
                let pbs = scratch.prefill_batch.as_ref().unwrap();
                let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(idx, (_, state))| {
                        let end = prepared[idx].tokens.len().min(cut);
                        (progress[idx] < end).then(|| qwen35::DensePrefillSessionBatchRow {
                            tokens: &prepared[idx].tokens[progress[idx]..end],
                            start_pos: state.seq_pos,
                            kv_cache: &mut state.kv_cache,
                            dn_state: &mut state.dn_state,
                            logits: &state.logits,
                        })
                    })
                    .collect();
                qwen35::forward_prefill_grouped_moe_session_batch(
                    gpu, weights, config, &mut rows, scratch, pbs,
                )
            };
            let shape = match worker_result {
                Ok(shape) => shape,
                Err(e) => {
                    for (id, state) in owned_sessions {
                        m.q35_sessions.insert(id, state);
                    }
                    return Err(format!(
                        "qwen35 grouped-MoE fused boundary prefill-session batch backend failed: {e:?}; \
                         use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
                    ));
                }
            };
            shape_total_tokens += shape.total_tokens;
            for idx in active_indices {
                let start = progress[idx];
                let end = prepared[idx].tokens.len().min(cut);
                let state = &mut owned_sessions[idx].1;
                state.seq_pos += end - start;
                state
                    .conversation_tokens
                    .extend_from_slice(&prepared[idx].tokens[start..end]);
                progress[idx] = end;
                for mut boundary in prepared[idx]
                    .boundary_checkpoints
                    .iter()
                    .filter(|boundary| boundary.prefix_len == end)
                    .cloned()
                {
                    let hook = Qwen35PrefillCheckpointHook {
                        batch_id,
                        session_id: &prepared[idx].id,
                        source_state_handle: &prepared[idx].id,
                        logical_position: end,
                        kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                            boundary: &boundary.boundary,
                            boundary_index: boundary.boundary_index,
                        },
                        prefix_hash: &boundary.hash,
                    };
                    let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                    let checkpoint_id =
                        emit_qwen35_owned_prefill_checkpoint(&mut m.q35_sessions, gpu, hook, state).map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                prepared[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                    boundary.checkpoint_id = Some(checkpoint_id);
                    boundary_checkpoints_by_session[idx].push(boundary);
                }
            }
        }
        let mut sessions = Vec::with_capacity(owned_sessions.len());
        for (idx, (id, mut state)) in owned_sessions.into_iter().enumerate() {
            state.prefilled_generated_suffix_len = 0;
            let logical_position = state.seq_pos + state.kv_cache.compact_offset;
            let prefix_hash = compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                &prepared[idx].state_kinds,
                &prepared[idx].assistant_prefix,
                prepared[idx].max_think_tokens,
                &state.conversation_tokens,
            );
            state.prefix_hash = Some(prefix_hash.clone());
            sessions.push(Qwen35PrefillSessionResult {
                id: id.clone(),
                prefill_tokens: prepared[idx].tokens.len(),
                logical_position,
                cached_prefix_tokens: prepared[idx].cached_prefix_tokens,
                prefix_hash,
                debug_sample_token: None,
                boundary_checkpoints: std::mem::take(&mut boundary_checkpoints_by_session[idx]),
            });
            m.q35_sessions.insert(id, state);
        }
        return Ok(Qwen35PrefillBatchResult {
            mode: "qwen35_fused_grouped_moe_prefill_boundary_chunked",
            plan,
            backend,
            total_prefill_tokens: shape_total_tokens,
            sessions,
        });
    }

    let worker_result = {
        let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
            "qwen35 scratch missing; grouped-MoE fused prefill is pp=1 only".to_string()
        })?;
        let total_tokens = prepared.iter().map(|spec| spec.tokens.len()).sum::<usize>();
        let scratch_target_batch = qwen35_prefill_scratch_target_batch(
            config.paged_experts,
            total_tokens,
            std::env::var("HIPFIRE_PREFILL_MAX_BATCH").ok().as_deref(),
            qwen35::PREFILL_MAX_BATCH,
        );
        let needs_scratch = scratch
            .prefill_batch
            .as_ref()
            .map(|pbs| pbs.max_batch < scratch_target_batch)
            .unwrap_or(true);
        if needs_scratch {
            if let Some(existing) = scratch.prefill_batch.take() {
                existing.free_gpu(gpu);
            }
            scratch.prefill_batch = Some(
                qwen35::PrefillBatchScratch::new(gpu, config, scratch_target_batch).map_err(
                    |e| format!("allocate qwen35 grouped-MoE fused prefill scratch: {e:?}"),
                )?,
            );
        }
        let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
        if pbs_max_batch < total_tokens {
            return Err(format!(
                "qwen35 grouped-MoE fused prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                total_tokens,
            ));
        }
        let pbs = scratch.prefill_batch.as_ref().unwrap();
        let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
            .iter_mut()
            .zip(prepared.iter())
            .map(|((_, state), spec)| qwen35::DensePrefillSessionBatchRow {
                tokens: &spec.tokens,
                start_pos: state.seq_pos,
                kv_cache: &mut state.kv_cache,
                dn_state: &mut state.dn_state,
                logits: &state.logits,
            })
            .collect();
        qwen35::forward_prefill_grouped_moe_session_batch(
            gpu, weights, config, &mut rows, scratch, pbs,
        )
    };

    let shape = match worker_result {
        Ok(shape) => shape,
        Err(e) => {
            for (id, state) in owned_sessions {
                m.q35_sessions.insert(id, state);
            }
            return Err(format!(
                "qwen35 grouped-MoE fused prefill-session batch backend failed: {e:?}; \
                 use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
            ));
        }
    };

    let mut sessions = Vec::with_capacity(owned_sessions.len());
    for ((id, mut state), spec) in owned_sessions.into_iter().zip(prepared.iter()) {
        state.seq_pos += spec.tokens.len();
        state.conversation_tokens.extend_from_slice(&spec.tokens);
        state.prefilled_generated_suffix_len = if spec.replay_as_generated_suffix {
            spec.tokens.len()
        } else {
            0
        };
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
        let debug_sample_token = if spec.replay_as_generated_suffix
            && std::env::var_os("HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE").is_some()
        {
            let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                "qwen35 scratch missing; fused grouped-MoE debug sampling unavailable".to_string()
            })?;
            let mut rng_state = 0x13579BDFu32;
            let cfg = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                blocked_tokens: Vec::new(),
            };
            Some(sampler::sample(
                gpu,
                &state.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                &spec.tokens,
                &cfg,
                &mut rng_state,
            ))
        } else {
            None
        };
        let prefix_hash = compute_qwen35_prefix_hash(
            m.arch_id,
            m.q35_kv_mode.as_deref(),
            &spec.state_kinds,
            &spec.assistant_prefix,
            spec.max_think_tokens,
            &state.conversation_tokens,
        );
        state.prefix_hash = Some(prefix_hash.clone());
        sessions.push(Qwen35PrefillSessionResult {
            id: id.clone(),
            prefill_tokens: spec.tokens.len(),
            logical_position,
            cached_prefix_tokens: spec.cached_prefix_tokens,
            prefix_hash,
            debug_sample_token,
            boundary_checkpoints: Vec::new(),
        });
        m.q35_sessions.insert(id, state);
    }

    Ok(Qwen35PrefillBatchResult {
        mode: if prepared[0].replay_as_generated_suffix {
            "qwen35_fused_grouped_moe_suffix_replay"
        } else {
            "qwen35_fused_grouped_moe_prefill"
        },
        plan,
        backend,
        total_prefill_tokens: shape.total_tokens,
        sessions,
    })
}

fn qwen35_prefill_suffix_batch_fused_dense(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    let contract = build_qwen35_fused_dense_prefill_batch_contract(prepared, plan)?;

    // Worker API seam for the real dense implementation:
    //
    //   prefill_suffix_batch(&mut [&mut RequestSession])
    //
    // The serial-reference worker below owns the correctness oracle: every
    // session has isolated KV, DeltaNet recurrent state, conversation tokens,
    // and logits. The fused worker must preserve the same ownership contract.
    //
    // Do not call qwen35::forward_prefill_batch over concatenated session
    // tokens here. That function batches rows inside ONE causal sequence and
    // ONE DeltaNetState; using it across independent request sessions would
    // leak KV/DN state and produce numerically plausible but wrong continuations.
    //
    // The next implementation step is an arch-level dense-Qwen35 session batch
    // worker that accepts per-session KV/DN/logits handles and writes one
    // independent result row per session.
    qwen35_save_active_session(m, gpu)?;
    let config = m
        .q35_config
        .as_ref()
        .ok_or_else(|| "qwen35 config missing".to_string())?;
    let weights = m
        .q35_weights
        .as_ref()
        .ok_or_else(|| "qwen35 weights missing".to_string())?;
    let boundary_cuts = qwen35_fused_prefill_boundary_cuts(prepared)?;
    let mut owned_sessions: Vec<(String, Qwen35RequestSessionState)> =
        Vec::with_capacity(contract.sessions.len());
    for spec in &contract.sessions {
        let state = match m.q35_sessions.remove(spec.id) {
            Some(state) => state,
            None => match qwen35_allocate_session_state(m, gpu) {
                Ok(state) => state,
                Err(e) => {
                    for (restore_id, restore_state) in owned_sessions {
                        m.q35_sessions.insert(restore_id, restore_state);
                    }
                    return Err(e);
                }
            },
        };
        if state.seq_pos + spec.tokens.len() > m.physical_cap {
            let id = spec.id.to_string();
            let seq_pos = state.seq_pos;
            m.q35_sessions.insert(id.clone(), state);
            for (restore_id, restore_state) in owned_sessions {
                m.q35_sessions.insert(restore_id, restore_state);
            }
            return Err(format!(
                "generate_batch_prefill exceeds loaded KV budget for session {}: seq_pos={} + prefill={} > physical_cap={}",
                id,
                seq_pos,
                spec.tokens.len(),
                m.physical_cap
            ));
        }
        owned_sessions.push((spec.id.to_string(), state));
    }

    if let Some(boundary_cuts) = boundary_cuts {
        let mut progress = vec![0usize; contract.sessions.len()];
        let mut boundary_checkpoints_by_session = vec![Vec::new(); contract.sessions.len()];
        let mut shape_total_tokens = 0usize;
        for &cut in &boundary_cuts {
            let active_indices: Vec<usize> = contract
                .sessions
                .iter()
                .enumerate()
                .filter_map(|(idx, spec)| {
                    let end = spec.tokens.len().min(cut);
                    (progress[idx] < end).then_some(idx)
                })
                .collect();
            if active_indices.len() < 2 {
                let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                    "qwen35 scratch missing; fused dense serial boundary segment is pp=1 only"
                        .to_string()
                })?;
                for &idx in &active_indices {
                    let start = progress[idx];
                    let end = contract.sessions[idx].tokens.len().min(cut);
                    let state = &mut owned_sessions[idx].1;
                    let segment_tokens = match qwen35_prefill_owned_session_serial_segment(
                        gpu,
                        weights,
                        config,
                        scratch,
                        state,
                        &contract.sessions[idx].tokens[start..end],
                    ) {
                        Ok(tokens) => tokens,
                        Err(err) => {
                            for (id, state) in owned_sessions {
                                m.q35_sessions.insert(id, state);
                            }
                            return Err(err);
                        }
                    };
                    shape_total_tokens += segment_tokens;
                    progress[idx] = end;
                    for mut boundary in prepared[idx]
                        .boundary_checkpoints
                        .iter()
                        .filter(|boundary| boundary.prefix_len == end)
                        .cloned()
                    {
                        let hook = Qwen35PrefillCheckpointHook {
                            batch_id,
                            session_id: contract.sessions[idx].id,
                            source_state_handle: contract.sessions[idx].id,
                            logical_position: end,
                            kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                                boundary: &boundary.boundary,
                                boundary_index: boundary.boundary_index,
                            },
                            prefix_hash: &boundary.hash,
                        };
                        let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                        let checkpoint_id = emit_qwen35_owned_prefill_checkpoint(
                            &mut m.q35_sessions,
                            gpu,
                            hook,
                            state,
                        )
                        .map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                contract.sessions[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                        boundary.checkpoint_id = Some(checkpoint_id);
                        boundary_checkpoints_by_session[idx].push(boundary);
                    }
                }
                continue;
            }
            let worker_result = {
                let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
                    "qwen35 scratch missing; fused dense prefill is pp=1 only".to_string()
                })?;
                let needs_scratch = scratch
                    .prefill_batch
                    .as_ref()
                    .map(|pbs| pbs.max_batch < contract.total_tokens)
                    .unwrap_or(true);
                if needs_scratch {
                    if let Some(existing) = scratch.prefill_batch.take() {
                        existing.free_gpu(gpu);
                    }
                    let max_batch = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| v >= 2)
                        .unwrap_or(qwen35::PREFILL_MAX_BATCH)
                        .max(contract.total_tokens);
                    scratch.prefill_batch = Some(
                        qwen35::PrefillBatchScratch::new(gpu, config, max_batch).map_err(|e| {
                            format!("allocate qwen35 fused dense prefill scratch: {e:?}")
                        })?,
                    );
                }
                let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
                if pbs_max_batch < contract.total_tokens {
                    return Err(format!(
                        "qwen35 fused dense prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                        contract.total_tokens,
                    ));
                }
                let pbs = scratch.prefill_batch.as_ref().unwrap();
                let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(idx, (_, state))| {
                        let end = contract.sessions[idx].tokens.len().min(cut);
                        (progress[idx] < end).then(|| qwen35::DensePrefillSessionBatchRow {
                            tokens: &contract.sessions[idx].tokens[progress[idx]..end],
                            start_pos: state.seq_pos,
                            kv_cache: &mut state.kv_cache,
                            dn_state: &mut state.dn_state,
                            logits: &state.logits,
                        })
                    })
                    .collect();
                qwen35::forward_prefill_dense_session_batch(
                    gpu, weights, config, &mut rows, scratch, pbs,
                )
            };
            let shape = match worker_result {
                Ok(shape) => shape,
                Err(e) => {
                    for (id, state) in owned_sessions {
                        m.q35_sessions.insert(id, state);
                    }
                    return Err(format!(
                        "qwen35 fused dense boundary prefill-session batch backend failed: {e:?}; \
                         use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
                    ));
                }
            };
            shape_total_tokens += shape.total_tokens;
            for idx in active_indices {
                let start = progress[idx];
                let end = contract.sessions[idx].tokens.len().min(cut);
                let state = &mut owned_sessions[idx].1;
                state.seq_pos += end - start;
                state
                    .conversation_tokens
                    .extend_from_slice(&contract.sessions[idx].tokens[start..end]);
                progress[idx] = end;
                for mut boundary in prepared[idx]
                    .boundary_checkpoints
                    .iter()
                    .filter(|boundary| boundary.prefix_len == end)
                    .cloned()
                {
                    let hook = Qwen35PrefillCheckpointHook {
                        batch_id,
                        session_id: contract.sessions[idx].id,
                        source_state_handle: contract.sessions[idx].id,
                        logical_position: end,
                        kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                            boundary: &boundary.boundary,
                            boundary_index: boundary.boundary_index,
                        },
                        prefix_hash: &boundary.hash,
                    };
                    let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                    let checkpoint_id =
                        emit_qwen35_owned_prefill_checkpoint(&mut m.q35_sessions, gpu, hook, state).map_err(|e| {
                            format!(
                                "qwen35 session {} failed to create fused semantic boundary checkpoint {}: {}",
                                contract.sessions[idx].id, checkpoint_id_for_error, e
                            )
                        })?;
                    boundary.checkpoint_id = Some(checkpoint_id);
                    boundary_checkpoints_by_session[idx].push(boundary);
                }
            }
        }
        let mut sessions = Vec::with_capacity(owned_sessions.len());
        for (idx, (id, mut state)) in owned_sessions.into_iter().enumerate() {
            state.prefilled_generated_suffix_len = 0;
            let logical_position = state.seq_pos + state.kv_cache.compact_offset;
            let prefix_hash = compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                contract.sessions[idx].state_kinds,
                contract.sessions[idx].assistant_prefix,
                contract.sessions[idx].max_think_tokens,
                &state.conversation_tokens,
            );
            state.prefix_hash = Some(prefix_hash.clone());
            sessions.push(Qwen35PrefillSessionResult {
                id: id.clone(),
                prefill_tokens: contract.sessions[idx].tokens.len(),
                logical_position,
                cached_prefix_tokens: contract.sessions[idx].cached_prefix_tokens,
                prefix_hash,
                debug_sample_token: None,
                boundary_checkpoints: std::mem::take(&mut boundary_checkpoints_by_session[idx]),
            });
            m.q35_sessions.insert(id, state);
        }
        return Ok(Qwen35PrefillBatchResult {
            mode: "qwen35_fused_dense_prefill_boundary_chunked",
            plan,
            backend,
            total_prefill_tokens: shape_total_tokens,
            sessions,
        });
    }

    let worker_result = {
        let scratch = m.q35_scratch.as_mut().ok_or_else(|| {
            "qwen35 scratch missing; fused dense prefill is pp=1 only".to_string()
        })?;
        let needs_scratch = scratch
            .prefill_batch
            .as_ref()
            .map(|pbs| pbs.max_batch < contract.total_tokens)
            .unwrap_or(true);
        if needs_scratch {
            if let Some(existing) = scratch.prefill_batch.take() {
                existing.free_gpu(gpu);
            }
            let max_batch = std::env::var("HIPFIRE_PREFILL_MAX_BATCH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v >= 2)
                .unwrap_or(qwen35::PREFILL_MAX_BATCH)
                .max(contract.total_tokens);
            scratch.prefill_batch = Some(
                qwen35::PrefillBatchScratch::new(gpu, config, max_batch)
                    .map_err(|e| format!("allocate qwen35 fused dense prefill scratch: {e:?}"))?,
            );
        }
        let pbs_max_batch = scratch.prefill_batch.as_ref().unwrap().max_batch;
        if pbs_max_batch < contract.total_tokens {
            return Err(format!(
                "qwen35 fused dense prefill scratch max_batch={pbs_max_batch} is smaller than required fused rows {}; increase HIPFIRE_PREFILL_MAX_BATCH or restart the daemon",
                contract.total_tokens,
            ));
        }
        let pbs = scratch.prefill_batch.as_ref().unwrap();
        let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = owned_sessions
            .iter_mut()
            .zip(contract.sessions.iter())
            .map(|((_, state), spec)| qwen35::DensePrefillSessionBatchRow {
                tokens: spec.tokens,
                start_pos: state.seq_pos,
                kv_cache: &mut state.kv_cache,
                dn_state: &mut state.dn_state,
                logits: &state.logits,
            })
            .collect();
        qwen35::forward_prefill_dense_session_batch(gpu, weights, config, &mut rows, scratch, pbs)
    };

    let shape = match worker_result {
        Ok(shape) => shape,
        Err(e) => {
            for (id, state) in owned_sessions {
                m.q35_sessions.insert(id, state);
            }
            return Err(format!(
                "qwen35 fused dense prefill-session batch backend failed: {e:?}; \
                 use HIPFIRE_QWEN35_PREFILL_SESSION_BATCH=auto or serial"
            ));
        }
    };

    let mut sessions = Vec::with_capacity(owned_sessions.len());
    for ((id, mut state), spec) in owned_sessions.into_iter().zip(contract.sessions.iter()) {
        state.seq_pos += spec.tokens.len();
        state.conversation_tokens.extend_from_slice(spec.tokens);
        state.prefilled_generated_suffix_len = if spec.replay_as_generated_suffix {
            spec.tokens.len()
        } else {
            0
        };
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
        let debug_sample_token = if spec.replay_as_generated_suffix
            && std::env::var_os("HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE").is_some()
        {
            let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                "qwen35 scratch missing; fused dense debug sampling unavailable".to_string()
            })?;
            let mut rng_state = 0x13579BDFu32;
            let cfg = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                blocked_tokens: Vec::new(),
            };
            Some(sampler::sample(
                gpu,
                &state.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                spec.tokens,
                &cfg,
                &mut rng_state,
            ))
        } else {
            None
        };
        let prefix_hash = compute_qwen35_prefix_hash(
            m.arch_id,
            m.q35_kv_mode.as_deref(),
            spec.state_kinds,
            spec.assistant_prefix,
            spec.max_think_tokens,
            &state.conversation_tokens,
        );
        state.prefix_hash = Some(prefix_hash.clone());
        sessions.push(Qwen35PrefillSessionResult {
            id: id.clone(),
            prefill_tokens: spec.tokens.len(),
            logical_position,
            cached_prefix_tokens: spec.cached_prefix_tokens,
            prefix_hash,
            debug_sample_token,
            boundary_checkpoints: Vec::new(),
        });
        m.q35_sessions.insert(id, state);
    }

    Ok(Qwen35PrefillBatchResult {
        mode: match contract.input_kind {
            Qwen35FusedDensePrefillInputKind::FullPrompt => "qwen35_fused_dense_prefill",
            Qwen35FusedDensePrefillInputKind::GeneratedSuffixReplay => {
                "qwen35_fused_dense_suffix_replay"
            }
        },
        plan,
        backend,
        total_prefill_tokens: shape.total_tokens,
        sessions,
    })
}

fn qwen35_prefill_suffix_batch_serial_reference(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    batch_id: &str,
    prepared: &[Qwen35PreparedPrefillSession],
    plan: GenerateBatchPrefillPlan,
    backend: Qwen35PrefillBatchBackend,
) -> Result<Qwen35PrefillBatchResult, String> {
    // Reference implementation: exact serial activation/prefill over isolated
    // per-session KV + DeltaNet state. This is the correctness oracle for the
    // future fused dense-Qwen35 path. Do not replace this with concatenating
    // sessions into one `forward_prefill_batch` call: that would share one
    // DeltaNet recurrent state and one causal sequence across independent
    // requests.
    let mut total_prefill_tokens = 0usize;
    let mut results = Vec::with_capacity(prepared.len());
    for session in prepared {
        qwen35_activate_session(m, gpu, &session.id)?;
        let mut boundary_checkpoints = Vec::new();
        let prefilled = if session.boundary_checkpoints.is_empty()
            || session.replay_as_generated_suffix
        {
            qwen35_prefill_active_session(
                m,
                gpu,
                &session.tokens,
                session.replay_as_generated_suffix,
            )?
        } else {
            let mut prefilled = 0usize;
            let mut boundaries = session.boundary_checkpoints.clone();
            boundaries.sort_by_key(|boundary| boundary.prefix_len);
            for mut boundary in boundaries {
                if boundary.prefix_len <= prefilled || boundary.prefix_len > session.tokens.len() {
                    continue;
                }
                let segment = &session.tokens[prefilled..boundary.prefix_len];
                prefilled += qwen35_prefill_active_session(m, gpu, segment, false)?;
                let logical_position = qwen35_active_logical_position(m)?;
                if logical_position != boundary.prefix_len {
                    return Err(format!(
                        "qwen35 semantic boundary checkpoint position mismatch for session {}: boundary_len={} logical_position={}",
                        session.id, boundary.prefix_len, logical_position
                    ));
                }
                let hook = Qwen35PrefillCheckpointHook {
                    batch_id,
                    session_id: &session.id,
                    source_state_handle: &session.id,
                    logical_position,
                    kind: Qwen35PrefillCheckpointKind::SemanticBoundary {
                        boundary: &boundary.boundary,
                        boundary_index: boundary.boundary_index,
                    },
                    prefix_hash: &boundary.hash,
                };
                let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
                let checkpoint_id = emit_qwen35_prefill_checkpoint(
                    m,
                    gpu,
                    loaded_model_state_arena_backend(m),
                    hook,
                )
                .map_err(|e| {
                    format!(
                        "qwen35 session {} failed to create semantic boundary checkpoint {}: {}",
                        session.id, checkpoint_id_for_error, e
                    )
                })?;
                qwen35_activate_session(m, gpu, &session.id)?;
                boundary.checkpoint_id = Some(checkpoint_id);
                boundary_checkpoints.push(boundary);
            }
            if prefilled < session.tokens.len() {
                prefilled +=
                    qwen35_prefill_active_session(m, gpu, &session.tokens[prefilled..], false)?;
            }
            prefilled
        };
        let logical_position = qwen35_active_logical_position(m)?;
        let debug_sample_token = if session.replay_as_generated_suffix
            && std::env::var_os("HIPFIRE_GENERATE_BATCH_PREFILL_DEBUG_SAMPLE").is_some()
        {
            let config = m
                .q35_config
                .as_ref()
                .ok_or_else(|| "qwen35 config missing".to_string())?;
            let scratch = m.q35_scratch.as_ref().ok_or_else(|| {
                "qwen35 scratch missing; PP batch-prefill is not supported".to_string()
            })?;
            let mut rng_state = 0x13579BDFu32;
            let cfg = SamplerConfig {
                temperature: 0.0,
                top_p: 1.0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                blocked_tokens: Vec::new(),
            };
            Some(sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                config.vocab_size,
                &session.tokens,
                &cfg,
                &mut rng_state,
            ))
        } else {
            None
        };
        qwen35_save_active_session(m, gpu)?;
        let prefix_hash = {
            let saved = m.q35_sessions.get(&session.id).ok_or_else(|| {
                format!("qwen35 session {} missing after prefill save", session.id)
            })?;
            compute_qwen35_prefix_hash(
                m.arch_id,
                m.q35_kv_mode.as_deref(),
                &session.state_kinds,
                &session.assistant_prefix,
                session.max_think_tokens,
                &saved.conversation_tokens,
            )
        };
        if let Some(saved) = m.q35_sessions.get_mut(&session.id) {
            saved.prefix_hash = Some(prefix_hash.clone());
        }
        total_prefill_tokens += prefilled;
        results.push(Qwen35PrefillSessionResult {
            id: session.id.clone(),
            prefill_tokens: prefilled,
            logical_position,
            cached_prefix_tokens: session.cached_prefix_tokens,
            prefix_hash,
            debug_sample_token,
            boundary_checkpoints,
        });
    }

    Ok(Qwen35PrefillBatchResult {
        mode: "serial_prefill",
        plan,
        backend,
        total_prefill_tokens,
        sessions: results,
    })
}

fn run_generate_batch_prefill_serial_qwen35(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    envelope: &GenerateBatchPrefillEnvelope,
    pflash_active: bool,
) -> Result<(), String> {
    if !is_qwen35_family_arch_id(m.arch_id) {
        return Err(format!(
            "generate_batch_prefill currently supports qwen35/qwen35-moe only (arch_id={})",
            m.arch_id
        ));
    }
    if m.pp > 1 {
        return Err(
            "generate_batch_prefill does not support pipeline-parallel models yet".to_string(),
        );
    }
    if m.dflash.is_some() {
        return Err("generate_batch_prefill does not support DFlash-loaded models yet".to_string());
    }
    if m.eviction.is_some() {
        return Err(
            "generate_batch_prefill does not support CASK/TriAttention eviction yet".to_string(),
        );
    }
    if pflash_active {
        return Err("generate_batch_prefill does not support PFlash compression yet".to_string());
    }
    let arena_backend = loaded_model_state_arena_backend(m);

    let plan = plan_generate_batch_prefill_qwen35(m.arch_id, envelope.session_count);
    let requested_backend = std::env::var("HIPFIRE_QWEN35_PREFILL_SESSION_BATCH").ok();
    let fused_grouped_moe_supported =
        validate_qwen35_fused_grouped_moe_prefill_model_capability(m, envelope.session_count);
    let backend = select_qwen35_prefill_batch_backend(
        plan,
        requested_backend.as_deref(),
        fused_grouped_moe_supported,
    )?;
    let started = serde_json::json!({
        "type": "generate_batch_prefill_started",
        "id": envelope.id,
        "batch_id": envelope.batch_id,
        "sessions": envelope.session_count,
        "mode": "serial_prefill",
        "plan": plan.as_str(),
        "backend": backend.as_str(),
    });
    let _ = writeln!(stdout, "{started}");
    let _ = stdout.flush();

    let t0 = Instant::now();
    let mut prepared = Vec::with_capacity(envelope.sessions.len());
    for session in &envelope.sessions {
        if !generate_state_kinds_include_required(
            &session.state_handle.state_kinds,
            SequenceStatePageKind::Kv,
        ) {
            return Err(format!(
                "generate_batch_prefill session {} missing attention_kv state kind",
                session.id
            ));
        }
        if !generate_state_kinds_include_required(
            &session.state_handle.state_kinds,
            SequenceStatePageKind::DeltaNet,
        ) {
            return Err(format!(
                "generate_batch_prefill session {} missing deltanet_recurrent state kind",
                session.id
            ));
        }

        if let Some(runtime_state_handle) = session.state_handle.runtime_state_handle.as_deref() {
            sequence_state_arena_fork_session_state(
                arena_backend,
                m,
                gpu,
                SequenceStateForkRequest {
                    source_session_id: runtime_state_handle,
                    dest_session_id: &session.id,
                    requested_prefix_hash: session.state_handle.prefix_hash.as_ref(),
                },
            )
            .map_err(|e| {
                format!(
                    "generate_batch_prefill session {} failed to attach checkpoint {}: {}",
                    session.id, runtime_state_handle, e
                )
            })?;
        }

        let resident = sequence_state_arena_is_session_resident(arena_backend, m, &session.id);
        if !resident
            && (session.state_handle.logical_position > 0
                || session.state_handle.cached_prefix_tokens > 0)
        {
            return Err(format!(
                "generate_batch_prefill session {} references cached state at logical_position={} cached_prefix_tokens={} but no resident session exists",
                session.id,
                session.state_handle.logical_position,
                session.state_handle.cached_prefix_tokens
            ));
        }

        let created = sequence_state_arena_activate_session(arena_backend, m, gpu, &session.id)?;
        let mut boundary_checkpoints = Vec::new();
        let tokens: Vec<u32> = if session.prompt.is_some() {
            let full_tokens = qwen35_materialize_batch_prefill_prompt(m, session)?;
            if session.state_handle.runtime_state_handle.is_some() {
                let prefix_len = session
                    .state_handle
                    .prefix_hash
                    .as_ref()
                    .map(|hash| hash.prefix_len)
                    .unwrap_or(session.state_handle.cached_prefix_tokens);
                if prefix_len > full_tokens.len() {
                    return Err(format!(
                        "generate_batch_prefill prompt session {} cached prefix length {} exceeds rendered token length {}",
                        session.id,
                        prefix_len,
                        full_tokens.len()
                    ));
                }
                full_tokens[prefix_len..].to_vec()
            } else if session.state_handle.logical_position != 0
                || session.state_handle.cached_prefix_tokens != 0
            {
                return Err(format!(
                    "generate_batch_prefill prompt session {} must start at logical_position=0 cached_prefix_tokens=0 in the first slice",
                    session.id
                ));
            } else {
                let _ = created;
                sequence_state_arena_reset_active_session(arena_backend, m, gpu)?;
                boundary_checkpoints =
                    qwen35_semantic_boundary_checkpoints(m, session, &full_tokens)?;
                full_tokens
            }
        } else {
            let current_position = sequence_state_arena_active_logical_position(arena_backend, m)?;
            if created && session.state_handle.logical_position != 0 {
                return Err(format!(
                    "generate_batch_prefill suffix session {} is new but logical_position={} (expected 0)",
                    session.id, session.state_handle.logical_position
                ));
            }
            if !created && current_position != session.state_handle.logical_position {
                return Err(format!(
                    "generate_batch_prefill session {} logical_position mismatch: request={} resident={}",
                    session.id, session.state_handle.logical_position, current_position
                ));
            }
            session.suffix_tokens.clone().unwrap_or_default()
        };

        prepared.push(Qwen35PreparedPrefillSession {
            id: session.id.clone(),
            tokens,
            cached_prefix_tokens: session.state_handle.cached_prefix_tokens,
            replay_as_generated_suffix: session.suffix_tokens.is_some(),
            state_kinds: session.state_handle.state_kinds.clone(),
            assistant_prefix: session.assistant_prefix.clone(),
            max_think_tokens: session.max_think_tokens,
            boundary_checkpoints,
        });
    }

    let result = qwen35_prefill_suffix_batch(m, gpu, &envelope.batch_id, &prepared, plan, backend)?;
    for session in &result.sessions {
        let hook = Qwen35PrefillCheckpointHook {
            batch_id: &envelope.batch_id,
            session_id: &session.id,
            source_state_handle: &session.id,
            logical_position: session.logical_position,
            kind: Qwen35PrefillCheckpointKind::Final,
            prefix_hash: &session.prefix_hash,
        };
        let checkpoint_id_for_error = qwen35_prefill_checkpoint_session_id(hook);
        let checkpoint_id =
            emit_qwen35_prefill_checkpoint(m, gpu, arena_backend, hook).map_err(|e| {
                format!(
                    "generate_batch_prefill session {} failed to create checkpoint {}: {}",
                    session.id, checkpoint_id_for_error, e
                )
            })?;
        let line = qwen35_generate_batch_prefill_session_done_json(
            envelope,
            session,
            &checkpoint_id,
            &result,
        );
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let worker = loaded_model_worker_runtime_view(m);
    let done = qwen35_generate_batch_prefill_done_json(
        envelope,
        &result,
        elapsed_ms,
        sequence_state_arena_resident_session_count(arena_backend, m),
        model_worker_runtime_view_json(&worker),
    );
    let _ = writeln!(stdout, "{done}");
    let _ = stdout.flush();
    Ok(())
}

fn run_generate_batch_decode_step_qwen35(
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

fn qwen35_decode_batch_max_chunk_size(session_count: usize) -> usize {
    std::env::var("HIPFIRE_QWEN35_DECODE_BATCH_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(session_count)
        .max(1)
}

fn qwen35_decode_dense_native_multirow_enabled() -> bool {
    matches!(
        std::env::var("HIPFIRE_QWEN35_DECODE_NATIVE_MULTIROW")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

fn qwen35_decode_internal_parity_enabled() -> bool {
    matches!(
        std::env::var("HIPFIRE_QWEN35_DECODE_INTERNAL_PARITY")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

fn qwen35_logits_debug_summary(
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

fn qwen35_decode_token_outcome(
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

fn qwen35_decode_step_serial_reference(
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
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
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
                &mut state.kv_cache,
                &mut state.dn_state,
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
        let new_logical_position = state.seq_pos + state.kv_cache.compact_offset;
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

fn qwen35_decode_step_fused_dense_layer_chunked(
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

fn qwen35_decode_step_fused_grouped_moe_layer_chunked(
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

fn qwen35_decode_step_fused_dense_native_chunks(
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

fn qwen35_decode_step_fused_grouped_moe_native_chunks(
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

fn qwen35_decode_step_fused_grouped_moe_native_chunk(
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
            let logical_position = state.seq_pos + state.kv_cache.compact_offset;
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
                kv_cache: &mut state.kv_cache,
                dn_state: &mut state.dn_state,
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
            let new_logical_position = state.seq_pos + state.kv_cache.compact_offset;
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
                    &mut oracle_state.kv_cache,
                    &mut oracle_state.dn_state,
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

fn qwen35_decode_native_chunk_ranges(
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

fn qwen35_ensure_decode_prefill_batch_scratch(
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

fn qwen35_decode_step_fused_dense_native_chunk(
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
            let logical_position = state.seq_pos + state.kv_cache.compact_offset;
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
                kv_cache: &mut state.kv_cache,
                dn_state: &mut state.dn_state,
                logits: &state.logits,
            })
            .collect();
        qwen35::forward_prefill_dense_session_batch(gpu, weights, config, &mut rows, scratch, pbs)
            .map_err(|e| format!("qwen35 fused dense native decode advance: {e:?}"))?;
        drop(rows);

        let mut lines = Vec::with_capacity(states.len());
        for ((session, state), outcome) in states.iter_mut().zip(outcomes.iter()) {
            state.seq_pos += 1;
            let new_logical_position = state.seq_pos + state.kv_cache.compact_offset;
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
                    &mut oracle_state.kv_cache,
                    &mut oracle_state.dn_state,
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

fn qwen35_decode_step_fused_dense_native_singleton(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    session: &GenerateBatchDecodeSession,
    im_end_token: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    qwen35_activate_session(m, gpu, &session.session_id)?;
    let mut state = Qwen35RequestSessionState::take_from_loaded(m, gpu)?;

    let result = (|| -> Result<Vec<serde_json::Value>, String> {
        let logical_position = state.seq_pos + state.kv_cache.compact_offset;
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
                &mut state.kv_cache,
                &mut state.dn_state,
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
        let new_logical_position = state.seq_pos + state.kv_cache.compact_offset;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: daemon [options]\n\
             \n\
             Reads JSON requests from stdin and writes JSON events to stdout.\n\
             \n\
             Options:\n\
               --precompile        compile/cache kernels for the current GPU and exit\n\
               --help, -h          print this help"
        );
        return;
    }

    // --precompile: compile all kernels for this GPU, write hash files, exit.
    // Used by scripts/install.sh and `hipfire update` so first `hipfire run`
    // isn't a 2-minute hipcc wait.
    //
    // Covers the current default path (mq4 weights + asym3 KV) plus the legacy
    // compat paths (hfq4, hfq6, q8 weights × asym3, q8 KV) so models from any
    // era of the registry start instantly.
    if args.iter().any(|a| a == "--precompile") {
        let _resource_lease = hipfire_daemon_adapter::acquire_resource_lease_or_exit();
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
        let mut errors = 0usize;
        for kv in &["asym3", "q8"] {
            for wq in &["mq4", "mq6", "hfq4", "hfq6", "q8"] {
                if let Err(e) = gpu.precompile_qwen35(wq, kv, 256) {
                    eprintln!("  {wq}/{kv}: {e}");
                    errors += 1;
                }
            }
        }
        if errors > 0 {
            eprintln!("Kernel precompilation finished with {errors} failure(s) — the missing kernels will JIT on first use.");
        } else {
            eprintln!("Kernel precompilation done.");
        }
        return;
    }

    // Machine-wide mutex — prevents orphan daemons from silently coexisting
    // (observed 2026-04-13: two daemons at 100% CPU survived pkill -f rounds
    // because they'd been reparented to PID 1 after their bun parent died).
    // Kept in a binding so the fd lives for the full process lifetime.
    let _daemon_lock = acquire_daemon_lock();
    let _resource_lease = hipfire_daemon_adapter::acquire_resource_lease_or_exit();
    hipfire_runtime::logging::init_stderr_logging("daemon");

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            report_gpu_init_failure(&e);
            std::process::exit(1);
        }
    };
    let mut model: Option<LoadedModel> = None;
    let mut active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
    let mut resident_models: std::collections::HashMap<String, LoadedModel> =
        std::collections::HashMap::new();
    let mut generic_state_arena = GenericSequenceStateArena::new();
    // PFlash speculative-prefill state. None unless the load message
    // includes a `prefill_drafter` path AND `prefill_compression` != "off".
    // Lives alongside `model` so unload_model + this state are paired
    // teardowns.
    let mut pflash_state: Option<hipfire_arch_qwen35::pflash::PflashState> = None;
    // The PflashConfig captured at load time. Per-request `prefill_*`
    // params override individual fields; the rest fall back to these
    // load-time defaults. Cleared alongside `pflash_state`.
    let mut pflash_cfg: Option<hipfire_arch_qwen35::pflash::PflashConfig> = None;
    // Hetero PFlash: when prefill_drafter_device differs from the target,
    // the drafter weights/KV/scratch live on a sibling device. The compress
    // output is a host-side Vec<u32>, so no peer-copy is needed — generate
    // routes maybe_compress_prompt to this handle, decode stays on target.
    // None means the drafter shares the target gpu (single-card, unchanged).
    let mut pflash_drafter_gpu: Option<rdna_compute::Gpu> = None;
    let mut dummy_model: Option<DummyModelState> = None;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","message":"invalid JSON: {}"}}"#,
                    e
                );
                let _ = stdout.flush();
                continue;
            }
        };

        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let protocol_load = if msg_type == "load" {
            serde_json::from_value::<hipfire_model::ModelLoadRequest>(msg.clone()).ok()
        } else {
            None
        };

        match msg_type {
            "load" => {
                let requested_worker_id = message_worker_id(&msg);
                // Unload previous if any. PFlash drafter goes first so
                // its tensors join the pool before unload_model drains
                // it -- otherwise free_tensor would queue them into the
                // pool just-emptied by drain_pool with no follow-up
                // drain, leaving drafter VRAM resident across the next
                // load (the explicit "unload" handler has the same
                // ordering for the same reason).
                if requested_worker_id == active_worker_id {
                    generic_state_arena.release_worker(&requested_worker_id);
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
                        unload_model(m, &mut gpu);
                    }
                } else {
                    if let Err(e) = park_active_model(
                        &mut model,
                        &mut gpu,
                        &active_worker_id,
                        &mut resident_models,
                    ) {
                        write_error(&mut stdout, "", &format!("worker switch failed: {e}"));
                        let _ = stdout.flush();
                        continue;
                    }
                    active_worker_id = requested_worker_id.clone();
                }
                if let Some(m) = resident_models.remove(&requested_worker_id) {
                    generic_state_arena.release_worker(&requested_worker_id);
                    unload_model(m, &mut gpu);
                }
                dummy_model = None;

                let path = protocol_load
                    .as_ref()
                    .map(|req| req.model.as_str())
                    .or_else(|| msg.get("model").and_then(|v| v.as_str()))
                    .unwrap_or("");
                let dummy_requested = msg
                    .get("params")
                    .and_then(|p| p.get("dummy_model"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if dummy_requested {
                    dummy_model = Some(DummyModelState::default());
                    tracing::info!(
                        model = "hipfire:dummy",
                        arch = "qwen35_dummy",
                        "dummy model loaded"
                    );
                    let line = serde_json::json!({
                        "type": "loaded",
                        "worker_key_id": requested_worker_id,
                        "arch": "qwen35_dummy",
                        "cache_capable": false,
                        "dim": 16,
                        "layers": 1,
                        "vocab": 1024,
                        "vl": false,
                    });
                    let _ = writeln!(stdout, "{line}");
                    let _ = stdout.flush();
                    continue;
                }

                let max_seq = protocol_load
                    .as_ref()
                    .map(|req| req.params.max_seq as usize)
                    .or_else(|| {
                        msg.get("params")
                            .and_then(|p| p.get("max_seq"))
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                    })
                    .unwrap_or(4096);
                let requested_physical_cap = protocol_load
                    .as_ref()
                    .and_then(|req| req.params.physical_cap.map(|v| v as usize))
                    .or_else(|| {
                        msg.get("params")
                            .and_then(|p| p.get("physical_cap"))
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                    })
                    .filter(|v| *v > 0);
                let raw_dflash_mode = msg
                    .get("params")
                    .and_then(|p| p.get("dflash_mode"))
                    .and_then(|v| v.as_str());
                let raw_draft_param = msg
                    .get("params")
                    .and_then(|p| p.get("draft"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty());
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
                let dflash_mode = protocol_load
                    .as_ref()
                    .and_then(|req| req.params.dflash_mode.as_deref())
                    .or(raw_dflash_mode)
                    .unwrap_or("auto");
                let raw_draft = protocol_load
                    .as_ref()
                    .and_then(|req| req.params.draft.as_deref())
                    .or(raw_draft_param)
                    .filter(|s| !s.is_empty());
                let draft_path = if dflash_mode == "off" {
                    if raw_draft.is_some() {
                        eprintln!(
                            "[hipfire-daemon] dflash_mode=off — skipping draft load ({})",
                            raw_draft.unwrap()
                        );
                    }
                    None
                } else {
                    raw_draft.map(|s| s.to_string())
                };
                let kv_mode_override = protocol_load
                    .as_ref()
                    .and_then(|req| req.params.kv_cache.as_deref())
                    .or_else(|| {
                        msg.get("params")
                            .and_then(|p| p.get("kv_mode"))
                            .and_then(|v| v.as_str())
                    })
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());

                // MTP speculative decode config. `mtp_mode` gates weight
                // discovery at load time (off=skip, on=error-if-missing,
                // auto=scan+log). `mtp_k` sets the draft window size.
                let mtp_mode = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_mode"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto")
                    .to_string();
                // Default K=2: empirically the sweet spot for Qwen3.5 MTP
                // (0.8B: τ=1.66 @ K=2 vs 1.62 @ K=3/4, and best tok/s — higher
                // K just wastes draft forwards that acceptance tapering rejects;
                // see NEXT-STEPS Phase B4). Overridable per-load via mtp_k.
                let mtp_k: usize = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_k"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2) as usize;

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
                let cask_sidecar = protocol_load
                    .as_ref()
                    .and_then(|req| req.params.cask_sidecar.as_deref())
                    .or_else(|| {
                        msg.get("params")
                            .and_then(|p| p.get("cask_sidecar"))
                            .and_then(|v| v.as_str())
                    })
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
                    gpu.mmq_screen = v;
                }
                if let Some(v) = msg
                    .get("params")
                    .and_then(|p| p.get("mmq_screen_threshold"))
                    .and_then(|v| v.as_f64())
                {
                    gpu.mmq_screen_threshold = v as f32;
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
                if pp > 1 {
                    if draft_path.is_some()
                        && std::env::var("HIPFIRE_PP_DFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap."}}"#
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if cask.sidecar.is_some() {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap"}}"#
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if (pflash_drafter.is_some() || pflash_mode_str != "off")
                        && std::env::var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap"}}"#
                        );
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

                match load_model(
                    path,
                    max_seq,
                    requested_physical_cap,
                    draft_path.as_deref(),
                    kv_mode_override.as_deref(),
                    state_quant_override.as_deref(),
                    &cask,
                    pp,
                    &mut gpu,
                ) {
                    Ok(mut m) => {
                        let arch = match m.arch_id {
                            5 => "qwen3_5",
                            6 => "qwen3_5_moe",
                            7 => "qwen2",
                            8 => "dots-ocr",
                            9 => "deepseek4",
                            10 => "minimax_m2",
                            11 => "lfm2moe",
                            _ => "qwen3",
                        };
                        let vl = m.vision_config.is_some() || m.dots_ocr_config.is_some();
                        let (dim, layers, vocab) = if let Some(ref c) = m.q35_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else if let Some(ref c) = m.llama_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else if let Some(ref c) = m.qwen2_config {
                            (c.hidden_size, c.num_hidden_layers, c.vocab_size)
                        } else if let Some(ref c) = m.dots_ocr_config {
                            (
                                c.text.hidden_size,
                                c.text.num_hidden_layers,
                                c.text.vocab_size,
                            )
                        } else if let Some(ref c) = m.minimax_config {
                            (c.hidden_size, c.num_hidden_layers, c.vocab_size)
                        } else if let Some((d, l, v)) = {
                            #[cfg(feature = "arch-lfm2moe")]
                            {
                                m.lfm2moe_config
                                    .as_ref()
                                    .map(|c| (c.hidden_size, c.num_hidden_layers, c.vocab_size))
                            }
                            #[cfg(not(feature = "arch-lfm2moe"))]
                            {
                                None::<(usize, usize, usize)>
                            }
                        } {
                            (d, l, v)
                        } else {
                            (0, 0, 0)
                        };

                        // Apply MTP config from load-message params.
                        m.mtp_mode = mtp_mode;
                        m.mtp_k = mtp_k;
                        // Detect whether MTP weights are present in the loaded
                        // model. DeepSeek V4: mtp_layer in weights. Qwen3.5/3.6
                        // (arch 5/6): a bundled `-mq4+mtp.hfq` trailer or a
                        // sibling `.mtp.hfq` sidecar. Used by mtp_mode to decide
                        // whether to drive the MTP spec-decode path at generate.
                        let qwen35_mtp_present = (m.arch_id == 5 || m.arch_id == 6) && {
                            let bundled = hipfire_arch_qwen35::mtp_head::detect_bundled_mtp_offset(
                                std::path::Path::new(&m.model_path),
                            )
                            .ok()
                            .flatten()
                            .is_some();
                            let sidecar =
                                std::path::Path::new(&m.model_path.replace(".hfq", ".mtp.hfq"))
                                    .exists();
                            bundled || sidecar
                        };
                        m.mtp_weights_present = qwen35_mtp_present
                            || m.deepseek4_weights
                                .as_ref()
                                .and_then(|w| w.mtp_layer.as_ref())
                                .is_some();

                        // ── Optional DPM stabilization (perf instrumentation) ──
                        //
                        // Pins the GPU at high sclk/mclk so the first `generate`
                        // request doesn't pay the 1-10s DPM ramp from idle. Same
                        // `HIPFIRE_DPM_WARMUP_SECS` env the in-process bench tools
                        // honor (`bench_qwen35_speed`, `dflash_spec_demo`,
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

                        let model_worker =
                            model_worker_runtime_view_json(&loaded_model_worker_runtime_view(&m));
                        let cache_capable = m.arch_id == 9 || is_qwen35_family_arch_id(m.arch_id);
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({
                                "type": "loaded",
                                "worker_key_id": requested_worker_id,
                                "arch": arch,
                                "cache_capable": cache_capable,
                                "dim": dim,
                                "layers": layers,
                                "vocab": vocab,
                                "vl": vl,
                                "model_worker": model_worker,
                            })
                        );

                        // ── PFlash drafter load (Phase 4.0) ──────────────
                        //
                        // Only attempt when mode != off AND a drafter path
                        // was provided. Failures here are NON-FATAL: log
                        // the reason and continue with PFlash disabled so
                        // the operator gets a clear "model is up, but
                        // compression isn't" signal rather than losing
                        // the entire session.
                        if let Some(ref pf_drafter_path) = pflash_drafter {
                            if pflash_mode_str != "off" {
                                if let Some(ref reason) = pflash_load_err {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"invalid load param: {}"}}"#,
                                        reason.replace('"', "'")
                                    );
                                    let _ = stdout.flush();
                                    model = Some(m);
                                    continue;
                                }
                                let pf_cfg = hipfire_arch_qwen35::pflash::PflashConfig {
                                    mode: hipfire_arch_qwen35::pflash::PflashMode::parse(
                                        &pflash_mode_str,
                                    )
                                    .unwrap_or(hipfire_arch_qwen35::pflash::PflashMode::Off),
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
                                    hipfire_arch_qwen35::pflash::PflashState::new(&pf_cfg);
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
                                    match hipfire_arch_qwen35::pflash::load_drafter(
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
                // Explicit per-request raw-prompt override (optional `"raw"`
                // bool). Absent → None → auto default (raw iff no chat_template).
                // Always set, so it resets every request (no cross-request leak).
                RAW_OVERRIDE.with(|c| c.set(msg.get("raw").and_then(|v| v.as_bool())));
                let protocol_generate =
                    serde_json::from_value::<hipfire_generate::GenerateTextRequest>(msg.clone())
                        .ok();
                let id = protocol_generate
                    .as_ref()
                    .map(|req| req.id.as_str())
                    .or_else(|| msg.get("id").and_then(|v| v.as_str()))
                    .unwrap_or("0");
                let target_worker_id = message_worker_id(&msg);
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                let session_id = msg
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(id);
                let prefill_already_done = msg
                    .get("prefill_already_done")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if let Some(dummy) = dummy_model.as_mut() {
                    let prompt = protocol_generate
                        .as_ref()
                        .map(|req| req.prompt.as_str())
                        .or_else(|| msg.get("prompt").and_then(|v| v.as_str()))
                        .unwrap_or("Hello");
                    let max_tokens = protocol_generate
                        .as_ref()
                        .map(|req| req.sampling.max_tokens as usize)
                        .or_else(|| {
                            msg.get("max_tokens")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize)
                        })
                        .unwrap_or(512);
                    tracing::debug!(
                        request_id = id,
                        session_id,
                        max_tokens,
                        prefill_already_done,
                        "dummy generate"
                    );
                    dummy.generate(
                        &mut stdout,
                        id,
                        session_id,
                        prompt,
                        prefill_already_done,
                        max_tokens,
                    );
                    continue;
                }
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ =
                            writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let session_id = msg
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty());
                let prefill_already_done = msg
                    .get("prefill_already_done")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let prefilled_prompt_tokens = msg
                    .get("prefilled_prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                if is_qwen35_family_arch_id(m.arch_id) && m.pp == 1 {
                    let target_session_id = session_id.unwrap_or(QWEN35_LEGACY_SESSION_ID);
                    if let Err(e) = qwen35_activate_session(m, &mut gpu, target_session_id) {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                } else if session_id.is_some() || prefill_already_done {
                    emit_error_with_id(
                        &mut stdout,
                        id,
                        "session_id/prefill_already_done are only supported for single-GPU qwen35/qwen35-moe",
                    );
                    continue;
                }
                let prompt = protocol_generate
                    .as_ref()
                    .map(|req| req.prompt.as_str())
                    .or_else(|| msg.get("prompt").and_then(|v| v.as_str()))
                    .unwrap_or("Hello");
                let prompt_norm = normalize_daemon_prompt(prompt);
                let prompt: &str = &prompt_norm;
                if std::env::var("HIPFIRE_PROMPT_TOKEN_HEAT").ok().as_deref() == Some("1") {
                    if let Some(tok) = m.tokenizer.as_ref() {
                        tok.dump_prompt_heat(prompt);
                    }
                }
                let system = protocol_generate
                    .as_ref()
                    .and_then(|req| req.system.as_deref())
                    .or_else(|| msg.get("system").and_then(|v| v.as_str()));
                let image = msg.get("image").and_then(|v| v.as_str());
                let image_base64 = protocol_generate
                    .as_ref()
                    .and_then(|req| req.image_base64.as_deref())
                    .or_else(|| msg.get("image_base64").and_then(|v| v.as_str()));

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
                let tools_json: Option<Vec<serde_json::Value>> = if let Some(tools) =
                    protocol_generate.as_ref().and_then(|req| req.tools.clone())
                {
                    match serde_json::from_value::<Vec<serde_json::Value>>(tools) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                id,
                                e.to_string().replace('"', "'"),
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    }
                } else {
                    match msg.get("tools") {
                        Some(v) => {
                            match serde_json::from_value::<Vec<serde_json::Value>>(v.clone()) {
                                Ok(t) => Some(t),
                                Err(e) => {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                        id,
                                        e.to_string().replace('"', "'"),
                                    );
                                    let _ = stdout.flush();
                                    continue;
                                }
                            }
                        }
                        None => None,
                    }
                };
                let messages_history: Option<Vec<prompt_frame::Message>> = if let Some(messages) =
                    protocol_generate
                        .as_ref()
                        .and_then(|req| req.messages.clone())
                {
                    Some(messages)
                } else {
                    match msg.get("messages") {
                        Some(v) => {
                            match serde_json::from_value::<Vec<prompt_frame::Message>>(v.clone()) {
                                Ok(m) => Some(m),
                                Err(e) => {
                                    let _ = writeln!(
                                        stdout,
                                        r#"{{"type":"error","id":"{}","message":"invalid messages field: {}"}}"#,
                                        id,
                                        e.to_string().replace('"', "'"),
                                    );
                                    let _ = stdout.flush();
                                    continue;
                                }
                            }
                        }
                        None => None,
                    }
                };
                let request_stop_sequences = protocol_generate
                    .as_ref()
                    .and_then(|req| req.stop.clone())
                    .unwrap_or_else(|| normalize_request_stop_sequences(msg.get("stop")));
                // Sampling defaults differ by arch: qwen35 family was tuned
                // at `temp=0.3, top_p=0.8` (DFlash-friendly, instruct-stable);
                // DeepSeek V4 Flash's HF card recommends `temp=1.0, top_p=1.0`
                // for local deployment, and lower values consistently fall
                // into block-level attractors on this quantized instruct
                // model. Pick arch-shaped defaults so a vanilla
                // `/v1/chat/completions` POST (no sampling fields) works on
                // both. Explicit per-request values still override either.
                let (default_temp, default_top_p) = if m.arch_id == 11 {
                    // LFM2.5-MoE (11): Liquid's model card recommends specific
                    // sampling — temperature=0.2, top_p=0.80 (+ repetition_penalty
                    // 1.05, set below). Use those exact values, not the generic
                    // MoE-instruct (temp=1.0) default — they're tuned for this
                    // model and keep it on-distribution.
                    (0.2_f64, 0.80_f64)
                } else if m.arch_id == 9 || m.arch_id == 10 {
                    // DeepSeek V4 (9) + MiniMax-M2 (10): quantized instruct
                    // MoE models that fall into block-level attractors under
                    // pure greedy. Default to the HF-recommended sampling
                    // (temp=1.0, top_p=1.0); explicit per-request values
                    // still override.
                    (1.0_f64, 1.0_f64)
                } else {
                    (0.3_f64, 0.8_f64)
                };
                let temp = protocol_generate
                    .as_ref()
                    .map(|req| req.sampling.temperature)
                    .or_else(|| msg.get("temperature").and_then(|v| v.as_f64()))
                    .unwrap_or(default_temp) as f32;
                let max_tokens = protocol_generate
                    .as_ref()
                    .map(|req| req.sampling.max_tokens as usize)
                    .or_else(|| {
                        msg.get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                    })
                    .unwrap_or(512);
                let top_p = protocol_generate
                    .as_ref()
                    .and_then(|req| req.sampling.top_p)
                    .or_else(|| msg.get("top_p").and_then(|v| v.as_f64()))
                    .unwrap_or(default_top_p) as f32;
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
                let default_repeat_penalty = if m.arch_id == 11 { 1.05_f64 } else { 1.0_f64 };
                let repeat_penalty = protocol_generate
                    .as_ref()
                    .and_then(|req| req.sampling.repeat_penalty)
                    .or_else(|| msg.get("repeat_penalty").and_then(|v| v.as_f64()))
                    .unwrap_or(default_repeat_penalty) as f32;
                // OpenAI-compatible `reasoning_effort` (also accept our custom
                // `thinking_mode` alias) — only consumed by arch_id=9 today.
                // Default = NonThink, matching the safe HF chat frame.
                let think_mode = protocol_generate
                    .as_ref()
                    .and_then(|req| {
                        req.reasoning_effort
                            .as_deref()
                            .or(req.thinking_mode.as_deref())
                            .or(req.thinking.as_deref())
                    })
                    .or_else(|| {
                        msg.get("reasoning_effort")
                            .or_else(|| msg.get("thinking_mode"))
                            .and_then(|v| v.as_str())
                    })
                    .map(ThinkMode::from_str)
                    .unwrap_or(ThinkMode::NonThink);
                let repeat_window = msg
                    .get("repeat_window")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                let presence_penalty = protocol_generate
                    .as_ref()
                    .and_then(|req| req.presence_penalty)
                    .or_else(|| msg.get("presence_penalty").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0)
                    .max(0.0) as f32;
                let frequency_penalty = protocol_generate
                    .as_ref()
                    .and_then(|req| req.frequency_penalty)
                    .or_else(|| msg.get("frequency_penalty").and_then(|v| v.as_f64()))
                    .unwrap_or(0.0)
                    .max(0.0) as f32;
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
                let experimental_ok = std::env::var("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT")
                    .ok()
                    .as_deref()
                    == Some("1");
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
                let max_think_tokens = protocol_generate
                    .as_ref()
                    .and_then(|req| req.max_think_tokens.map(u64::from))
                    .or_else(|| msg.get("max_think_tokens").and_then(|v| v.as_u64()))
                    .unwrap_or(0) as usize;

                // assistant_prefix: "plain", "open_think", or "closed_think"
                // Controls the ChatML framing after the assistant role header.
                // Consumed by the text path; VL path does not yet propagate
                // it (tracked as a follow-up to the post-#169 rebase).
                let assistant_prefix = prompt_frame::AssistantPrefix::from_label(
                    protocol_generate
                        .as_ref()
                        .and_then(|req| req.assistant_prefix.as_deref())
                        .or_else(|| msg.get("assistant_prefix").and_then(|v| v.as_str())),
                );

                let has_image = image_base64.is_some() || image.is_some();
                let is_dots_ocr = m.arch_id == 8;
                let has_vl = m.vision_config.is_some() || is_dots_ocr;

                if has_image && !has_vl {
                    write_error(&mut stdout, id, "model has no vision encoder");
                } else if has_image && has_vl {
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
                    };
                    if is_dots_ocr {
                        generate_vl_dots_ocr(m, &mut gpu, &mut stdout, &params);
                    } else {
                        generate_vl(m, &mut gpu, &mut stdout, &params);
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
                            if let Some(m) = hipfire_arch_qwen35::pflash::PflashMode::parse(s) {
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
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","id":"{}","message":"invalid pflash override: {}"}}"#,
                            id,
                            reason.replace('"', "'"),
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    generate(
                        m,
                        &mut gpu,
                        pflash_drafter_gpu.as_mut(),
                        &mut stdout,
                        id,
                        prompt,
                        system,
                        temp,
                        top_p,
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
                        prefill_already_done,
                        prefilled_prompt_tokens,
                        &request_stop_sequences,
                        protocol_generate
                            .as_ref()
                            .and_then(|req| req.evidence_dir.as_deref())
                            .or_else(|| msg.get("evidence_dir").and_then(|v| v.as_str())),
                    );
                }
            }

            "generate_batch_prefill" => match validate_generate_batch_prefill(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
                    if dummy_model.is_none() {
                        match activate_model_worker(
                            &target_worker_id,
                            &mut active_worker_id,
                            &mut model,
                            &mut gpu,
                            &mut resident_models,
                        ) {
                            Ok(true) => {}
                            Ok(false) => {
                                emit_error_with_id(
                                    &mut stdout,
                                    &envelope.id,
                                    format!("unknown model worker {target_worker_id}"),
                                );
                                continue;
                            }
                            Err(e) => {
                                emit_error_with_id(
                                    &mut stdout,
                                    &envelope.id,
                                    format!("worker switch failed: {e}"),
                                );
                                continue;
                            }
                        }
                    }
                    if envelope.is_probe() {
                        if dummy_model.is_some() {
                            emit_dummy_generate_batch_prefill_ready(&mut stdout, &envelope);
                            continue;
                        }
                        match model.as_ref() {
                            Some(m) if is_qwen35_family_arch_id(m.arch_id) && m.pp == 1 => {
                                emit_generate_batch_prefill_ready(&mut stdout, &envelope);
                            }
                            Some(m) => {
                                let reason = format!(
                                    "generate_batch_prefill currently supports qwen35/qwen35-moe only (arch_id={})",
                                    m.arch_id
                                );
                                emit_generate_batch_prefill_unsupported(
                                    &mut stdout,
                                    &envelope,
                                    &reason,
                                );
                            }
                            None => {
                                emit_generate_batch_prefill_unsupported(
                                    &mut stdout,
                                    &envelope,
                                    "no model loaded",
                                );
                            }
                        }
                        continue;
                    }
                    if let Some(dummy) = dummy_model.as_mut() {
                        tracing::info!(
                            request_id = envelope.id,
                            batch_id = envelope.batch_id,
                            sessions = envelope.session_count,
                            "dummy generate_batch_prefill"
                        );
                        if let Err(e) =
                            run_generate_batch_prefill_dummy(dummy, &mut stdout, &envelope)
                        {
                            emit_error_with_id(&mut stdout, &envelope.id, e);
                        }
                        continue;
                    }
                    let m = match model.as_mut() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(&mut stdout, &envelope.id, "no model loaded");
                            continue;
                        }
                    };
                    if let Err(e) = run_generate_batch_prefill_serial_qwen35(
                        m,
                        &mut gpu,
                        &mut stdout,
                        &envelope,
                        pflash_state.is_some(),
                    ) {
                        emit_error_with_id(&mut stdout, &envelope.id, e);
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut stdout, id, e);
                }
            },

            "prefix_hash_preflight" => match validate_prefix_hash_preflight(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                    let m = match model.as_ref() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(&mut stdout, &envelope.id, "no model loaded");
                            continue;
                        }
                    };
                    if let Err(e) = run_prefix_hash_preflight_qwen35(m, &mut stdout, &envelope) {
                        emit_error_with_id(&mut stdout, &envelope.id, e);
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut stdout, id, e);
                }
            },

            "generate_batch_decode_step" => match validate_generate_batch_decode(&msg) {
                Ok(envelope) => {
                    let target_worker_id = message_worker_id(&msg);
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                &envelope.id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                    let m = match model.as_mut() {
                        Some(m) => m,
                        None => {
                            emit_error_with_id(&mut stdout, &envelope.id, "no model loaded");
                            continue;
                        }
                    };
                    if let Err(e) =
                        run_generate_batch_decode_step_qwen35(m, &mut gpu, &mut stdout, &envelope)
                    {
                        emit_error_with_id(&mut stdout, &envelope.id, e);
                    }
                }
                Err(e) => {
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_error_with_id(&mut stdout, id, e);
                }
            },

            "release_sessions" => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("release");
                let target_worker_id = message_worker_id(&msg);
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                let request = match parse_release_sessions_request(&msg, &target_worker_id) {
                    Ok(request) => request,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                };
                if let Some(dummy) = dummy_model.as_mut() {
                    let released = dummy.release_sessions(&request.sessions);
                    let done = release_sessions_done_json(
                        id,
                        request.sessions.len(),
                        released,
                        dummy.session_count(),
                        None,
                    );
                    let _ = writeln!(stdout, "{done}");
                    let _ = stdout.flush();
                    continue;
                }
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        emit_error_with_id(&mut stdout, id, "no model loaded");
                        continue;
                    }
                };
                let arena_backend = loaded_model_state_arena_backend(m);
                match sequence_state_arena_release_sessions(
                    arena_backend,
                    m,
                    &mut gpu,
                    &request.sessions,
                ) {
                    Ok(released) => {
                        let worker = loaded_model_worker_runtime_view(m);
                        let done = release_sessions_done_json(
                            id,
                            request.sessions.len(),
                            released,
                            sequence_state_arena_resident_session_count(arena_backend, m),
                            Some(&worker),
                        );
                        let _ = writeln!(stdout, "{done}");
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, id, e),
                }
            }

            "reserve_session_state" => {
                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("reserve");
                let target_worker_id = message_worker_id(&msg);
                generic_state_arena.purge_expired();
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                id,
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                let request = match parse_reserve_session_state_request(&msg, &target_worker_id) {
                    Ok(request) => request,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                };
                let reservation_id = request.reservation_id.clone().unwrap_or_else(|| {
                    format!(
                        "reserve:{}:{}",
                        request.worker_id,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0)
                    )
                });
                let reservation_plan = if let Some(m) = model.as_ref() {
                    let budget = request
                        .budget_bytes
                        .unwrap_or_else(resident_state_reservation_budget_bytes);
                    let arena_backend = loaded_model_state_arena_backend(m);
                    let descriptors = sequence_state_arena_page_descriptors(arena_backend, m);
                    sequence_state_reservation_plan(
                        &descriptors,
                        request.physical_cap,
                        generic_state_arena.outstanding_bytes_for_worker(&request.worker_id),
                        budget,
                    )
                } else if dummy_model.is_some() {
                    let budget = request
                        .budget_bytes
                        .unwrap_or_else(resident_state_reservation_budget_bytes);
                    sequence_state_reservation_plan_for_reserved_bytes(1024, 0, 0, budget)
                } else {
                    emit_error_with_id(&mut stdout, id, "no model loaded");
                    continue;
                };
                if reservation_plan.rejected_for_memory_pressure {
                    let rejected = reserve_session_state_rejected_json(
                        id,
                        &request.worker_id,
                        reservation_plan.reserved_bytes,
                        reservation_plan.current_session_bytes,
                        reservation_plan.outstanding_reserved_bytes,
                        reservation_plan.projected_reserved_bytes,
                        reservation_plan.budget_bytes,
                    );
                    let _ = writeln!(stdout, "{rejected}");
                    let _ = stdout.flush();
                    continue;
                }
                let reservation = generic_state_arena.reserve(
                    &request.worker_id,
                    reservation_id.clone(),
                    &request.state_kinds,
                    request.physical_cap,
                    reservation_plan.reserved_bytes,
                    request.ttl_ms,
                );
                let done = reserve_session_state_done_json(
                    id,
                    &reservation,
                    reservation_plan.current_session_bytes,
                    reservation_plan.outstanding_reserved_bytes,
                    reservation_plan.projected_reserved_bytes,
                    reservation_plan.budget_bytes,
                );
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            "describe_state" => {
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("describe-state");
                generic_state_arena.purge_expired();
                let request = match parse_describe_sequence_state_request(&msg) {
                    Ok(request) => request,
                    Err(e) => {
                        emit_error_with_id(&mut stdout, id, e);
                        continue;
                    }
                };
                if parsed_handle_may_target_generic(&request.handle) {
                    if let Some(reservation) =
                        generic_state_arena.describe(&request.handle.id, request.handle.generation)
                    {
                        let done = session_state_reservation_describe_json(id, reservation);
                        let _ = writeln!(stdout, "{done}");
                        let _ = stdout.flush();
                        continue;
                    }
                }
                let Some(described) = describe_loaded_sequence_state(
                    &active_worker_id,
                    model.as_ref(),
                    &resident_models,
                    &request.handle,
                ) else {
                    emit_error_with_id(
                        &mut stdout,
                        id,
                        format!(
                            "describe_state unknown runtime_state_handle {}",
                            request.handle.id
                        ),
                    );
                    continue;
                };
                let done = described_sequence_state_json(id, &described);
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            "release_session_state_reservation" | "release_state" => {
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("release-reservation");
                let request = parse_release_sequence_state_request(&msg);
                let generic_handles = request
                    .handles
                    .iter()
                    .filter(|handle| parsed_handle_may_target_generic(handle))
                    .map(|handle| (handle.id.clone(), handle.generation))
                    .collect::<Vec<_>>();
                let (generic_released, generic_released_bytes) =
                    generic_state_arena.release(generic_handles);
                let (loaded_released, loaded_released_bytes) =
                    match release_loaded_sequence_state_handles(
                        &mut model,
                        &mut resident_models,
                        &mut gpu,
                        &request.handles,
                    ) {
                        Ok(released) => released,
                        Err(e) => {
                            emit_error_with_id(&mut stdout, id, e);
                            continue;
                        }
                    };
                let done = release_state_done_json(
                    request.response_kind,
                    id,
                    generic_released,
                    generic_released_bytes,
                    loaded_released,
                    loaded_released_bytes,
                );
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            "worker_status" | "list_workers" => {
                let status = resident_worker_status_json(
                    &active_worker_id,
                    model.as_ref(),
                    &resident_models,
                );
                let _ = writeln!(stdout, "{status}");
                let _ = stdout.flush();
            }

            "inventory" => {
                let inventory = daemon_accelerator_inventory(&mut gpu);
                let mut payload = serde_json::to_value(inventory)
                    .unwrap_or_else(|_| serde_json::json!({"source": "daemon", "devices": []}));
                payload["type"] = serde_json::json!("inventory");
                let _ = writeln!(stdout, "{payload}");
                let _ = stdout.flush();
            }

            "reset" => {
                let target_worker_id = message_worker_id(&msg);
                if dummy_model.is_none() {
                    match activate_model_worker(
                        &target_worker_id,
                        &mut active_worker_id,
                        &mut model,
                        &mut gpu,
                        &mut resident_models,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("unknown model worker {target_worker_id}"),
                            );
                            continue;
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("worker switch failed: {e}"),
                            );
                            continue;
                        }
                    }
                }
                // Reset conversation state without unloading the model.
                if let Some(dummy) = dummy_model.as_mut() {
                    generic_state_arena.release_worker(&target_worker_id);
                    dummy.reset();
                    let _ = writeln!(stdout, r#"{{"type":"reset"}}"#);
                    let _ = stdout.flush();
                    continue;
                }
                // Under eviction, also zero the compact_offset so absolute
                // RoPE phase restarts from zero for the fresh conversation.
                if let Some(ref mut m) = model {
                    generic_state_arena.release_worker(&target_worker_id);
                    m.seq_pos = 0;
                    m.conversation_tokens.clear();
                    m.q35_sessions.clear();
                    m.q35_active_session_id = if is_qwen35_family_arch_id(m.arch_id)
                        && m.pp == 1
                        && m.kv_cache.is_some()
                        && m.dn_state.is_some()
                    {
                        m.q35_active_state_allocation_epoch = next_qwen35_state_allocation_epoch();
                        Some(QWEN35_LEGACY_SESSION_ID.to_string())
                    } else {
                        m.q35_active_state_allocation_epoch = 0;
                        None
                    };
                    // Multi-GPU branch: route per-LA-layer memsets through
                    // pp_dn_la_to_device so each buffer is zeroed on its
                    // owning device. The single-GPU `gpu` parameter is left
                    // alone — its scratch state isn't aliased to per-device
                    // tensors when pp > 1.
                    if m.pp > 1 {
                        if let (Some(ref dn), Some(ref mut gpus), Some(ref la)) = (
                            m.dn_state.as_ref(),
                            m.pp_gpus.as_mut(),
                            m.pp_dn_la_to_device.as_ref(),
                        ) {
                            for (i, s) in dn.s_matrices.iter().enumerate() {
                                let g = &mut gpus.devices[la[i] as usize];
                                let _ = g.bind_thread();
                                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                            }
                            for (i, s) in dn.s_scales.iter().enumerate() {
                                let g = &mut gpus.devices[la[i] as usize];
                                let _ = g.bind_thread();
                                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                            }
                            for (i, s) in dn.conv_states.iter().enumerate() {
                                let g = &mut gpus.devices[la[i] as usize];
                                let _ = g.bind_thread();
                                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
                            }
                        }
                    } else if let Some(ref dn) = m.dn_state {
                        // Zero DeltaNet recurrent state (Qwen3.5)
                        for s in &dn.s_matrices {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.s_scales {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                        for s in &dn.conv_states {
                            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                        }
                    }
                    if let Some(kv) = m.kv_cache.as_mut() {
                        kv.compact_offset = 0;
                    }
                    if let Some(kv) = m.llama_kv.as_mut() {
                        kv.compact_offset = 0;
                    }
                    // arch_id=7: rewind the Qwen2State position cursor so
                    // the next prefill writes from KV[0]. Without this, a
                    // reset between turns would leak the prior turn's KV
                    // entries into attention for the new turn — fluent
                    // garbage, no panic. See `Qwen2State::reset` doc.
                    if let Some(ref mut s) = m.qwen2_state {
                        s.reset();
                    }
                    // arch_id=9: same rationale for DeepSeek V4. Prior to
                    // 2026-05-24 the V4F state was NEVER reset, so
                    // `state.n_tokens` accumulated across requests and
                    // every new prefill wrote AFTER the previous turn's
                    // KV residue — fitting symptom for the multi-turn
                    // pi-coding-agent corruption (`CLion` for
                    // `CLionProjects`, `/home/n/` for `/home/nick/`).
                    // See `DeepseekV4State::reset` doc.
                    if let Some(ref mut s) = m.deepseek4_state {
                        s.reset();
                        // Drop the captured V4F decode hipGraph alongside
                        // the state. The captured kernarg blobs hold
                        // session-1's device-buffer pointers; a fresh
                        // capture on session-2 binds against session-2's
                        // pointers and host scalars. Without this the
                        // replay path crashes with "illegal memory access"
                        // on the post-launch logits D2H — the captured
                        // graph dispatched against a stale slot/n_valid
                        // computation that mis-ordered against this
                        // session's prefill state. The matching
                        // `ar_forward_warmed_up = false` in `reset()`
                        // ensures we retrace warmup → capture → replay
                        // rather than jumping straight back to replay.
                        gpu.invalidate_graph_state();
                    }
                    // arch_id=10 (MiniMax-M2): clear KV cursor between turns.
                    // No captured hipGraph on this path, so no graph
                    // invalidation needed.
                    if let Some(ref mut s) = m.minimax_state {
                        s.reset();
                    }
                    // arch_id=11 (LFM2.5-MoE): clear KV + conv-state cursors
                    // between turns. reset() also zeroes the rolling conv
                    // states on-GPU, so it takes `gpu` and returns Result.
                    #[cfg(feature = "arch-lfm2moe")]
                    if let Some(ref mut s) = m.lfm2moe_state {
                        let _ = s.reset(&mut gpu);
                    }
                    let _ = writeln!(stdout, r#"{{"type":"reset","seq_pos":0}}"#);
                } else {
                    let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                }
                let _ = stdout.flush();
            }

            "unload" => {
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
                        pf.unload_drafter(&mut dg); // sibling-device drafter: free on its own handle, then drop
                        gpu.bind_thread_or_warn();
                    } else {
                        pf.unload_drafter(&mut gpu);
                    }
                }
                pflash_cfg = None;
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }
                for (_, m) in resident_models.drain() {
                    unload_model(m, &mut gpu);
                }
                generic_state_arena.clear();
                dummy_model = None;
                active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
                let _ = writeln!(stdout, r#"{{"type":"unloaded"}}"#);
                let _ = stdout.flush();
            }

            "unload_worker" => {
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unload_worker");
                let request = parse_unload_worker_request(&msg, DEFAULT_MODEL_WORKER_ID);
                let worker_id = request.worker_id;
                let mut unloaded = false;
                generic_state_arena.release_worker(&worker_id);
                if worker_id == active_worker_id {
                    if let Some(m) = model.take() {
                        unload_model(m, &mut gpu);
                        unloaded = true;
                    }
                    active_worker_id = DEFAULT_MODEL_WORKER_ID.to_string();
                    if let Some((next_worker_id, next_model)) = resident_models
                        .iter()
                        .next()
                        .map(|(k, _)| k.clone())
                        .and_then(|k| resident_models.remove(&k).map(|m| (k, m)))
                    {
                        active_worker_id = next_worker_id;
                        model = Some(next_model);
                    }
                } else if let Some(m) = resident_models.remove(&worker_id) {
                    unload_model(m, &mut gpu);
                    unloaded = true;
                }
                let done = unload_worker_done_json(
                    id,
                    &worker_id,
                    unloaded,
                    resident_models.len() + usize::from(model.is_some()),
                );
                let _ = writeln!(stdout, "{done}");
                let _ = stdout.flush();
            }

            "ping" => {
                let _ = writeln!(stdout, r#"{{"type":"pong"}}"#);
                let _ = stdout.flush();
            }

            // Calibrate the resident model in place (no reload): run the Tier-1
            // collector over a corpus and write a .calib.hfq. The data plane stays
            // daemon-internal — only the request + the resulting path/summary cross
            // JSONL. Single-GPU qwen3.5-family bf16 only (capture fires at the
            // bf16 chokepoints); additive and gated, never on the decode hot path.
            "collect" => {
                // Parse fields directly from the JSON message (the daemon is the
                // server side; the typed CollectRequest contract lives in
                // hipfire-daemon-protocol for clients). Field names must match.
                let Some(corpus) = msg.get("corpus").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(&mut stdout, "", "collect: missing 'corpus'".to_string());
                    continue;
                };
                let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(&mut stdout, "", "collect: missing 'output'".to_string());
                    continue;
                };
                let max_tokens = msg
                    .get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(512);
                let kldref = msg.get("kldref").and_then(|v| v.as_bool()).unwrap_or(false);
                let Some(m) = model.as_ref() else {
                    emit_error_with_id(&mut stdout, "", "collect: no model loaded".to_string());
                    continue;
                };
                if m.pp != 1 {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "collect: requires a single-GPU resident model (pp == 1)".to_string(),
                    );
                    continue;
                }
                let (Some(weights), Some(config), Some(tokenizer)) = (
                    m.q35_weights.as_ref(),
                    m.q35_config.as_ref(),
                    m.tokenizer.as_ref(),
                ) else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "collect: resident model is not a qwen3.5-family model with a tokenizer"
                            .to_string(),
                    );
                    continue;
                };
                let text = match std::fs::read_to_string(&corpus) {
                    Ok(t) => t,
                    Err(e) => {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("collect: read corpus {corpus}: {e}"),
                        );
                        continue;
                    }
                };
                let all = tokenizer.encode(&text);
                let n_tok = all.len().min(max_tokens);
                let tokens = &all[..n_tok];
                let opts = qwen35::CalibOpts {
                    kldref,
                    kldref_topk: 64,
                };
                let provenance = [
                    ("source_model", serde_json::json!(m.model_path)),
                    ("corpus", serde_json::json!(corpus)),
                    ("n_calib_tokens", serde_json::json!(n_tok)),
                ];
                // Streams the .calib.hfq directly to `output` one tensor at a
                // time (no full-RAM materialization), returning a summary.
                match qwen35::collect_calibration_artifacts(
                    &mut gpu,
                    weights,
                    config,
                    tokens,
                    &opts,
                    std::path::Path::new(&output),
                    &provenance,
                ) {
                    Ok(summary) => {
                        let resp = serde_json::json!({
                            "type": "collected",
                            "output": output,
                            "n_hessian": summary.n_hessian,
                            "n_calib_tokens": n_tok,
                            "max_consistency": summary.max_consistency,
                        });
                        let _ = writeln!(stdout, "{resp}");
                        let _ = stdout.flush();
                    }
                    Err(e) => emit_error_with_id(&mut stdout, "", format!("collect: {e}")),
                }
            }

            // PFlash drafter TEACHER: forward the resident qwen3.5 target over a
            // corpus and emit per-chunk per-block cosine-K scores at the shallow +
            // mid FullAttention layers — the labels `pflash_drafter_train` distils
            // (teacher/student split, docs/plans/2026-06-19-training-via-daemon-forward.md).
            // Output is JSONL, one line per chunk; the trainer's daemon-label
            // loader converts it to the v2 label cache.
            "pflash_labels" => {
                let Some(corpus) = msg.get("corpus").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: missing 'corpus'".to_string(),
                    );
                    continue;
                };
                let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from)
                else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: missing 'output'".to_string(),
                    );
                    continue;
                };
                let seq = msg
                    .get("seq")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(512);
                let block = msg
                    .get("block")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(64);
                let n_chunks = msg
                    .get("n_chunks")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(40);
                let Some(m) = model.as_ref() else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: no model loaded".to_string(),
                    );
                    continue;
                };
                let (Some(weights), Some(config), Some(tokenizer)) = (
                    m.q35_weights.as_ref(),
                    m.q35_config.as_ref(),
                    m.tokenizer.as_ref(),
                ) else {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: resident model is not a qwen3.5-family model".to_string(),
                    );
                    continue;
                };
                let fa = qwen35::full_attention_layers(config);
                if fa.is_empty() {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "pflash_labels: no FullAttention layers".to_string(),
                    );
                    continue;
                }
                let shallow = fa[0];
                let mid = fa[fa.len() / 2];
                let text = match std::fs::read_to_string(&corpus) {
                    Ok(t) => t,
                    Err(e) => {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("pflash_labels: read {corpus}: {e}"),
                        );
                        continue;
                    }
                };
                let all = tokenizer.encode(&text);
                if all.len() < n_chunks * seq {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        format!(
                            "pflash_labels: corpus too small: {} toks < {}",
                            all.len(),
                            n_chunks * seq
                        ),
                    );
                    continue;
                }
                let mut out_file = match std::fs::File::create(&output) {
                    Ok(f) => std::io::BufWriter::new(f),
                    Err(e) => {
                        emit_error_with_id(
                            &mut stdout,
                            "",
                            format!("pflash_labels: create {output}: {e}"),
                        );
                        continue;
                    }
                };
                let mut failed = false;
                for ci in 0..n_chunks {
                    let toks = all[ci * seq..(ci + 1) * seq].to_vec();
                    match qwen35::capture_pflash_block_scores(
                        &mut gpu,
                        weights,
                        config,
                        &toks,
                        block,
                        &[shallow, mid],
                    ) {
                        Ok(scores) => {
                            let line = serde_json::json!({
                                "chunk": ci,
                                "tokens": toks,
                                "shallow_scores": scores[0],
                                "mid_scores": scores[1],
                            });
                            if writeln!(out_file, "{line}").is_err() {
                                failed = true;
                                break;
                            }
                        }
                        Err(e) => {
                            emit_error_with_id(
                                &mut stdout,
                                "",
                                format!("pflash_labels: chunk {ci}: {e}"),
                            );
                            failed = true;
                            break;
                        }
                    }
                }
                use std::io::Write as _;
                let _ = out_file.flush();
                if failed {
                    continue;
                }
                // Dump the shared fp32 embedding once (the drafter shares it RO).
                let embed_path = format!("{output}.embed.bin");
                let embed_dims = match qwen35::dump_embed_fp32(
                    &mut gpu,
                    weights,
                    config,
                    std::path::Path::new(&embed_path),
                ) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        emit_error_with_id(&mut stdout, "", format!("pflash_labels: embed: {e}"));
                        None
                    }
                };
                let resp = serde_json::json!({
                    "type": "pflash_labels",
                    "output": output,
                    "embed": embed_dims.map(|_| embed_path.clone()),
                    "embed_vocab": embed_dims.map(|(v, _)| v),
                    "embed_dim": embed_dims.map(|(_, d)| d),
                    "n_chunks": n_chunks,
                    "seq": seq,
                    "block": block,
                    "shallow_layer": shallow,
                    "mid_layer": mid,
                });
                let _ = writeln!(stdout, "{resp}");
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
                // Synthetic prefill benchmark — measures forward_prefill_batch on N
                // deterministic tokens from a zeroed state. Used by `hipfire bench`
                // to produce canonical pp128/pp512/pp1024 numbers that don't depend
                // on the user's prompt tokenizing to a round number.
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ =
                            writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
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
                if m.pp > 1 {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill requires pp=1 (multi-GPU bench not implemented)"}}"#
                    );
                    let _ = stdout.flush();
                    continue;
                }
                let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                // Guard physical_cap — reserve 32 slots of headroom so a subsequent
                // generate request against the loaded model still has room. We guard
                // on the *physical* buffer (not the advertised max_seq) because this
                // bench intentionally bypasses eviction to measure raw prefill.
                if n + 32 > m.physical_cap {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill tokens={} exceeds loaded physical_cap={}"}}"#,
                        n, m.physical_cap
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
                // prefill-on-top-of-prior-state.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                if let Some(ref dn) = m.dn_state {
                    for s in &dn.s_matrices {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.s_scales {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.conv_states {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                }
                // Qwen2 (arch_id=7) doesn't have a separate KV buffer — the cache
                // and the per-step scratch share `Qwen2State`. Reset its position
                // cursor here so bench_prefill measures cold prefill.
                if let Some(ref mut s) = m.qwen2_state {
                    s.reset();
                }
                // MiniMax-M2 (arch_id=10): same — KV cache + scratch share
                // MiniMaxState; reset its cursor for a cold prefill bench.
                if let Some(ref mut s) = m.minimax_state {
                    s.reset();
                }
                // LFM2.5-MoE (arch_id=11): same — KV + conv-state cache share
                // Lfm2MoeState; reset cursors (takes gpu) for a cold bench.
                #[cfg(feature = "arch-lfm2moe")]
                if let Some(ref mut s) = m.lfm2moe_state {
                    let _ = s.reset(&mut gpu);
                }

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                let _ = gpu.hip.device_synchronize();
                let t0 = Instant::now();
                let run_ok = if is_qwen35_family_arch_id(m.arch_id) {
                    let config = m.q35_config.as_ref().unwrap();
                    let weights = m.q35_weights.as_ref().unwrap();
                    let scratch = m.q35_scratch.as_ref().unwrap();
                    let kv = m.kv_cache.as_mut().unwrap();
                    let dn = m.dn_state.as_mut().unwrap();
                    qwen35::forward_prefill_batch(
                        &mut gpu, weights, config, &synthetic, 0, kv, dn, scratch, None, None,
                        None, None,
                    )
                    .is_ok()
                } else if m.arch_id == 7 {
                    // Qwen2 has no batched prefill kernel yet — per-token loop
                    // mirroring the LLaMA fallback path. The loop seeds
                    // position via `state.next_pos` (already reset above to 0).
                    let config = m.qwen2_config.as_ref().unwrap();
                    let weights = m.qwen2_weights.as_ref().unwrap();
                    let state = m.qwen2_state.as_mut().unwrap();
                    let mut ok = true;
                    for &tok in &synthetic {
                        if qwen2::forward_step(&mut gpu, weights, config, state, tok).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    ok
                } else if m.arch_id == 9 {
                    // DeepSeek V4 warm-pass: per-token decode_step. Saturates
                    // the kernel cache (HC, indexer, compressor,
                    // attention, MoE) on a short synthetic prompt
                    // before any user-facing generate. Not the
                    // production prefill path (that's
                    // forward_prefill_batch_chunked in `generate`).
                    let config = m.deepseek4_config.as_ref().unwrap();
                    let weights = m.deepseek4_weights.as_ref().unwrap();
                    let state = m.deepseek4_state.as_mut().unwrap();
                    let mut ok = true;
                    for (i, &tok) in synthetic.iter().enumerate() {
                        if deepseek4::forward::decode_step(
                            config, weights, state, &mut gpu, tok, i as u32,
                        )
                        .is_err()
                        {
                            ok = false;
                            break;
                        }
                    }
                    ok
                } else if m.arch_id == 10 {
                    // MiniMax-M2 warm-pass: per-token decode_step over the
                    // synthetic prompt. Saturates the GQA + QK-norm + RoPE +
                    // MoE kernel set before any user-facing generate. This
                    // IS the production prefill shape (no batched kernel).
                    let config = m.minimax_config.as_ref().unwrap();
                    let weights = m.minimax_weights.as_ref().unwrap();
                    let state = m.minimax_state.as_mut().unwrap();
                    let mut ok = true;
                    for (i, &tok) in synthetic.iter().enumerate() {
                        if minimax::forward::decode_step(
                            config, weights, state, &mut gpu, tok, i as u32,
                        )
                        .is_err()
                        {
                            ok = false;
                            break;
                        }
                    }
                    ok
                } else if cfg!(feature = "arch-lfm2moe") && m.arch_id == 11 {
                    // LFM2.5-MoE warm-pass: per-token decode_step over the
                    // synthetic prompt. Saturates the conv + GQA + QK-norm +
                    // RoPE + top-4 MoE kernel set before any user-facing
                    // generate. This IS the production prefill shape (no
                    // batched kernel).
                    #[cfg(feature = "arch-lfm2moe")]
                    {
                        let config = m.lfm2moe_config.as_ref().unwrap();
                        let weights = m.lfm2moe_weights.as_ref().unwrap();
                        let state = m.lfm2moe_state.as_mut().unwrap();
                        let mut ok = true;
                        for (i, &tok) in synthetic.iter().enumerate() {
                            if lfm2moe::forward::decode_step(
                                config, weights, state, &mut gpu, tok, i as u32,
                            )
                            .is_err()
                            {
                                ok = false;
                                break;
                            }
                        }
                        ok
                    }
                    #[cfg(not(feature = "arch-lfm2moe"))]
                    {
                        false
                    }
                } else {
                    let config = m.llama_config.as_ref().unwrap();
                    let weights = m.llama_weights.as_ref().unwrap();
                    let scratch = m.llama_scratch.as_ref().unwrap();
                    let kv = m.llama_kv.as_mut().unwrap();
                    let mut ok = true;
                    for (i, &tok) in synthetic.iter().enumerate() {
                        if llama::forward_scratch(
                            &mut gpu, weights, config, tok, i, kv, scratch, 0.0, 1.0, 42, 0, 1.0,
                        )
                        .is_err()
                        {
                            ok = false;
                            break;
                        }
                    }
                    ok
                };
                let _ = gpu.hip.device_synchronize();
                let elapsed = t0.elapsed().as_secs_f64();

                // Reset state AFTER measurement — we've written N KV slots and a
                // DeltaNet state that the next real request must not inherit.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                if let Some(ref dn) = m.dn_state {
                    for s in &dn.s_matrices {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.s_scales {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
                    for s in &dn.conv_states {
                        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
                    }
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
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill forward failed"}}"#
                    );
                }
                let _ = stdout.flush();
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

            _ => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","message":"unknown type: {}"}}"#,
                    msg_type
                );
                let _ = stdout.flush();
            }
        }
    }
}

/// DFlash-powered greedy decode. Mirrors `generate`'s ChatML shape and
/// token-streaming output but replaces the AR sample loop with
/// `spec_step_dflash` cycles — each cycle drafts B tokens via the diffusion
/// model and verifies them in one target forward, committing accept_len+1
/// at a time.
///
/// Single-turn: this path always resets target state at entry, matching the
/// stateless OpenAI chat-completions contract. Multi-turn callers that
/// persist KV across HTTP requests are out of scope for this integration —
/// they can keep using the AR path.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// MTP (Multi-Token Prediction) spec-decode generate path for Qwen3.5/3.6.
///
/// Selected when a qwen35 model carries a co-trained MTP head (bundled
/// `-mq4+mtp.hfq` or sibling `.mtp.hfq`), `mtp_mode != "off"`, and no DFlash
/// drafter is loaded. Uses the **non-tree** `mtp_spec::spec_step_mtp` (linear
/// autoregressive K-token draft) because the FP32 DeltaNet state the small
/// models run on does not support tree-mode replay (see TODO.md /
/// `qwen35::default_state_quant`). The MTP head is lazy-loaded from
/// `m.model_path` per request and freed at function end (cache-on-slot is a
/// future optimization).
fn generate_mtp(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
) {
    use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
    use hipfire_arch_qwen35::mtp_spec::{self, MtpSpecState};
    use hipfire_arch_qwen35::qwen35;
    use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};

    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Prompt build mirrors generate_dflash: Jinja chat template when enabled,
    // else the hand-rolled ChatFrame::Plain scaffold.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let prompt_tokens: Vec<u32> = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(mh) => mh,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                eprintln!("[daemon] jinja render failed in mtp path ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build()
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: system_prompt,
            user: prompt,
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build()
    };

    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // Fresh target state — MTP runs its own prefill from position 0.
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    {
        let dn = m.dn_state.as_ref().unwrap();
        for s in &dn.s_matrices {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.s_scales {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.conv_states {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
    }

    // Assemble a transient ModelSlot, mirroring generate_dflash's take/putback.
    let target_config = m.q35_config.as_ref().unwrap().clone();
    let weights = m.q35_weights.take().expect("q35 weights");
    let kv_cache = m.kv_cache.take().expect("kv cache");
    let dn_state = m.dn_state.take().expect("dn state");
    let scratch = m.q35_scratch.take().expect("q35 scratch");
    macro_rules! putback {
        ($t:expr) => {{
            m.q35_weights = Some($t.weights);
            m.kv_cache = Some($t.kv_cache);
            m.dn_state = Some($t.dn_state);
            m.q35_scratch = Some($t.scratch);
        }};
    }
    let hfq = match HfqFile::open(Path::new(&m.model_path)) {
        Ok(h) => h,
        Err(e) => {
            write_error(stdout, id, &format!("reopen model: {e}"));
            m.q35_weights = Some(weights);
            m.kv_cache = Some(kv_cache);
            m.dn_state = Some(dn_state);
            m.q35_scratch = Some(scratch);
            return;
        }
    };
    let mut target = ModelSlot {
        name: String::from("target"),
        hfq,
        config: target_config,
        weights,
        kv_cache,
        dn_state,
        scratch,
        slot_config: ModelSlotConfig::default(),
    };

    let max_seq_total = m.physical_cap.max(m.max_seq);
    let max_n = m.mtp_k.clamp(1, 8);
    let eos_token = target.config.eos_token;
    let dim = target.config.dim;

    if prompt_tokens.len() + max_tokens * (max_n + 1) + 16 > max_seq_total {
        write_error(
            stdout,
            id,
            &format!(
                "prompt ({}) + max_tokens ({}) × (max_n+1) won't fit in max_seq {}",
                prompt_tokens.len(),
                max_tokens,
                max_seq_total
            ),
        );
        putback!(target);
        return;
    }

    // Load the MTP head from the bundled trailer (or a sibling .mtp.hfq).
    let head = match mtp_head::load_mtp_head_bundled(Path::new(&m.model_path), gpu, max_seq_total) {
        Ok(Some(h)) => h,
        Ok(None) => {
            // No bundled trailer — try a sibling "<stem>.mtp.hfq".
            let sidecar = m.model_path.replace(".hfq", ".mtp.hfq");
            match mtp_head::load_mtp_head(Path::new(&sidecar), gpu, max_seq_total) {
                Ok(h) => h,
                Err(e) => {
                    write_error(stdout, id, &format!("load mtp head: {e}"));
                    putback!(target);
                    return;
                }
            }
        }
        Err(e) => {
            write_error(stdout, id, &format!("load bundled mtp head: {e}"));
            putback!(target);
            return;
        }
    };

    let kv_mode = m
        .q35_kv_mode
        .as_deref()
        .and_then(|s| MtpKvMode::parse(s).ok())
        .unwrap_or(MtpKvMode::Q8);
    let mut state =
        match MtpSpecState::new_for_slot_with_kv_mode(gpu, &target, &head, max_n, kv_mode) {
            Ok(s) => s,
            Err(e) => {
                write_error(stdout, id, &format!("alloc MtpSpecState: {e}"));
                head.free_gpu(gpu);
                putback!(target);
                return;
            }
        };

    let t0 = Instant::now();

    // Run prefill + spec-decode loop; centralize cleanup after.
    // Returns (hit_eos, generated, cycles, accepted_total, decode_secs).
    let run: Result<(bool, usize, usize, usize, f64), String> = (|| {
        // Prefill the prompt through the batched WMMA path.
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            &prompt_tokens,
            0,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| format!("prefill forward_prefill_batch: {e:?}"))?;

        // Snapshot trunk's post-output-norm hidden at the last prefill position.
        state
            .capture_prev_hidden_from_scratch_tmp(gpu, &target.scratch.tmp, dim)
            .map_err(|e| format!("capture prev_hidden: {e:?}"))?;

        // Seed token = argmax of trunk logits at the last prefill position.
        let logits0 = gpu
            .download_f32(&target.scratch.logits)
            .map_err(|e| format!("download seed logits: {e:?}"))?;
        let mut seed_token = 0u32;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits0.iter().enumerate() {
            if v > best {
                best = v;
                seed_token = i as u32;
            }
        }

        // Streaming state (mirrors generate_dflash).
        let mut streamed_tokens: Vec<u32> = Vec::new();
        let mut bytes_fed_to_filter = 0usize;
        let mut filter = chat_output_filter(m, request_stop_sequences);
        let mut generated = 0usize;
        let mut think_count: usize = 0;
        let mut prev_in_think: bool = false;

        // Helper closure semantics inlined: stream one committed token, return
        // (hit_eos, think_cap_hit).
        let emit_token = |stdout: &mut std::io::Stdout,
                          tok: u32,
                          streamed_tokens: &mut Vec<u32>,
                          bytes_fed_to_filter: &mut usize,
                          filter: &mut EosFilter,
                          think_count: &mut usize,
                          prev_in_think: &mut bool|
         -> (bool, bool) {
            streamed_tokens.push(tok);
            emit_committed_event(
                stdout,
                id,
                tok,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(streamed_tokens);
            let new_bytes = &all_bytes[*bytes_fed_to_filter..];
            *bytes_fed_to_filter = all_bytes.len();
            if emit_filter_action(stdout, id, filter.observe(new_bytes)) {
                return (true, false);
            }
            let hit_eos =
                tok == eos_token || im_end_token == Some(tok) || tokenizer.is_terminator(tok);
            let mut think_cap_hit = false;
            if !hit_eos && max_think_tokens > 0 {
                let raw = tokenizer.decode_bytes(streamed_tokens);
                let raw_str = std::str::from_utf8(&raw).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think && !*prev_in_think {
                    *think_count = 0;
                }
                if in_think {
                    *think_count += 1;
                }
                *prev_in_think = in_think;
                if in_think && *think_count >= max_think_tokens {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"token","id":"{}","text":"</think>\n"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    think_cap_hit = true;
                }
            }
            (hit_eos, think_cap_hit)
        };

        // Emit the seed token first.
        let (mut hit_eos, mut think_cap_hit) = emit_token(
            stdout,
            seed_token,
            &mut streamed_tokens,
            &mut bytes_fed_to_filter,
            &mut filter,
            &mut think_count,
            &mut prev_in_think,
        );
        generated += 1;

        let mut last_committed = seed_token;
        let mut cur_pos = prompt_tokens.len();
        let mut cycles = 0usize;
        let mut accepted_total = 0usize;
        let t_decode = Instant::now();

        while !hit_eos && !think_cap_hit && generated < max_tokens {
            if cur_pos + max_n + 1 >= max_seq_total {
                break;
            }
            let result = mtp_spec::spec_step_mtp(
                gpu,
                &mut target,
                &head,
                &mut state,
                cur_pos,
                last_committed,
                eos_token,
            )
            .map_err(|e| format!("spec_step_mtp: {e:?}"))?;
            cycles += 1;
            accepted_total += result.accept_count;

            // result.committed already EXCLUDES the seed (unlike DFlash).
            for &tok in &result.committed {
                if generated >= max_tokens {
                    break;
                }
                let (eos, cap) = emit_token(
                    stdout,
                    tok,
                    &mut streamed_tokens,
                    &mut bytes_fed_to_filter,
                    &mut filter,
                    &mut think_count,
                    &mut prev_in_think,
                );
                generated += 1;
                if eos {
                    hit_eos = true;
                    break;
                }
                if cap {
                    think_cap_hit = true;
                    break;
                }
            }
            if let Some(&last) = result.committed.last() {
                last_committed = last;
            }
            cur_pos += result.advance;
            if result.hit_eos {
                hit_eos = true;
            }
        }
        Ok((
            hit_eos,
            generated,
            cycles,
            accepted_total,
            t_decode.elapsed().as_secs_f64(),
        ))
    })();

    // Cleanup: free MTP state + head, put trunk pieces back on the model.
    state.free_gpu(gpu);
    head.free_gpu(gpu);
    putback!(target);

    match run {
        Ok((_hit_eos, generated, cycles, accepted_total, decode_secs)) => {
            let tok_s = generated as f64 / decode_secs.max(1e-9);
            // τ = committed tokens per spec cycle (excludes the prefill seed).
            let tau = if cycles > 0 {
                generated.saturating_sub(1) as f64 / cycles as f64
            } else {
                0.0
            };
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"decode_tok_s":{:.1},"mtp":true,"tau":{:.2},"cycles":{},"accepted":{},"max_n":{}}}"#,
                id, generated, tok_s, tok_s, tau, cycles, accepted_total, max_n,
            );
            let _ = stdout.flush();
        }
        Err(e) => {
            write_error(stdout, id, &e);
        }
    }
}

fn generate_dflash(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    pflash_bypass_reason: Option<&str>,
    pflash_alpha: Option<f32>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
) {
    use hipfire_arch_qwen35::speculative::{
        spec_step_ddtree_batched, spec_step_ddtree_path_c, spec_step_dflash, ModelSlot,
        ModelSlotConfig, Phase2Snapshots, SpecStats,
    };

    // Prompt build: same two-path branch as the AR-path generate() — when
    // `HIPFIRE_JINJA_CHAT=1` AND the model carries a chat_template, render
    // via `JinjaChatFrame` so structured `tools` / `messages` can reach
    // the upstream template's `{% if tools %}` / multi-turn branches.
    // Otherwise fall back to the hand-rolled `ChatFrame::Plain` scaffold
    // (byte-identical to the prior DFlash-path build).
    //
    // DFlash is single-turn by construction — `seq_pos` is reset to 0
    // below before seed_target_hidden_from_prompt runs — so we never
    // need to guard on `seq_pos == 0` here.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let prompt_tokens: Vec<u32> = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                eprintln!(
                    "[daemon] jinja render failed in dflash path ({e}) — falling back to Plain"
                );
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build()
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: system_prompt,
            user: prompt,
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build()
    };

    // `im_end_token` is still needed downstream for the EOS check.
    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // Fresh target state — DFlash seed_target_hidden_from_prompt does its own
    // full prefill, so we reset first to avoid double-accounting.
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    {
        let dn = m.dn_state.as_ref().unwrap();
        for s in &dn.s_matrices {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.s_scales {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
        for s in &dn.conv_states {
            let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
        }
    }
    let chat_template_profile = m.chat_template_profile.clone();
    let df = m.dflash.as_mut().unwrap();
    df.target_hidden_host.clear();
    df.draft_scratch.reset_upload_tracking();

    // Assemble a transient ModelSlot for the spec helpers — they both take
    // `&mut ModelSlot`. We own the pieces on LoadedModel individually, so
    // take them, build the ModelSlot, run, then put them back.
    //
    // ModelSlot needs its own HfqFile field but spec_step_dflash doesn't
    // actually touch it. Reopening via mmap is essentially free (few µs).
    let target_config = m.q35_config.as_ref().unwrap().clone();
    let weights = m.q35_weights.take().expect("q35 weights");
    let kv_cache = m.kv_cache.take().expect("kv cache");
    let dn_state = m.dn_state.take().expect("dn state");
    let scratch = m.q35_scratch.take().expect("q35 scratch");
    let hfq = match HfqFile::open(Path::new(&m.model_path)) {
        Ok(h) => h,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"reopen model: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            m.q35_weights = Some(weights);
            m.kv_cache = Some(kv_cache);
            m.dn_state = Some(dn_state);
            m.q35_scratch = Some(scratch);
            return;
        }
    };
    let slot_config = ModelSlotConfig::default();
    let mut target = ModelSlot {
        name: String::from("target"),
        hfq,
        config: target_config,
        weights,
        kv_cache,
        dn_state,
        scratch,
        slot_config,
    };

    let t0 = Instant::now();
    let ctx_capacity = df.ctx_capacity;
    // Capacity checks. With eviction enabled the advertised context window is
    // effectively unbounded (eviction fires between spec cycles), but the
    // *prompt* must still fit in one physical_cap span because
    // seed_target_hidden_from_prompt writes it per-token without chunking.
    let eff_prompt_cap = if m.eviction.is_some() {
        m.physical_cap
    } else {
        ctx_capacity
    };
    if prompt_tokens.len() + df.block_size > eff_prompt_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt+block_size exceeds {} {} (eviction {})"}}"#,
            id,
            if m.eviction.is_some() {
                "physical_cap"
            } else {
                "ctx_capacity"
            },
            eff_prompt_cap,
            if m.eviction.is_some() { "on" } else { "off" },
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        m.kv_cache = Some(target.kv_cache);
        m.dn_state = Some(target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }
    if m.eviction.is_none() && prompt_tokens.len() + max_tokens + df.block_size > ctx_capacity {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt+max_tokens exceeds ctx_capacity {} (enable cask_sidecar for long decode)"}}"#,
            id, ctx_capacity,
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        m.kv_cache = Some(target.kv_cache);
        m.dn_state = Some(target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }

    // Seed target_hidden via the demo's helper — runs a per-token prefill
    // with hidden extraction into hidden_rb, then downloads prompt-length
    // worth of rows into target_hidden_host. The draft's first forward
    // uses these as context.
    if let Err(e) = speculative::seed_target_hidden_from_prompt(
        gpu,
        &mut target,
        &mut df.hidden_rb,
        &mut df.target_hidden_host,
        &prompt_tokens,
    ) {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prefill: {}"}}"#,
            id, e
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        m.kv_cache = Some(target.kv_cache);
        m.dn_state = Some(target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }
    // Prime the draft's GPU target_hidden buffer from the prompt rows so the
    // first spec step can skip the CPU→GPU upload of the whole context.
    if let Err(e) = speculative::scatter_hidden_block_to_interleaved(
        gpu,
        &df.hidden_rb,
        &df.draft_scratch.target_hidden,
        0,
        prompt_tokens.len(),
        prompt_tokens.len(),
    ) {
        eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
    }
    df.draft_scratch.uploaded_target_hidden_rows = prompt_tokens.len();
    df.draft_scratch.target_hidden_abs_positions = (0..prompt_tokens.len() as i32).collect();

    // First emit = target's argmax at the final prompt position. seed_target_hidden
    // already ran the per-token forward for every prompt token; its scratch.logits
    // holds the post-prompt logits.
    let first_logits = match gpu.download_f32(&target.scratch.logits) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"download logits: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            m.q35_weights = Some(target.weights);
            m.kv_cache = Some(target.kv_cache);
            m.dn_state = Some(target.dn_state);
            m.q35_scratch = Some(target.scratch);
            return;
        }
    };
    let first_token = first_logits
        .iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv {
                (i as u32, v)
            } else {
                (best, bv)
            }
        })
        .0;

    let t_prefill = Instant::now();

    // Decode loop — spec_step_dflash returns a committed batch per cycle.
    let mut emitted: Vec<u32> = vec![first_token];
    let mut streamed_tokens: Vec<u32> = Vec::new();
    // `bytes_fed_to_filter` is the index into the freshly-decoded byte
    // stream past which we have not yet handed bytes to the filter.
    // The filter owns UTF-8 boundary buffering and any future arch
    // quirks (Gemma 4 marker holdback, strip-think, byte-level stop_at);
    // see crates/engine/src/eos_filter.rs.
    let mut bytes_fed_to_filter = 0usize;
    let mut filter =
        chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);
    let mut position = prompt_tokens.len();
    let mut seed_token = first_token;
    let mut stats = SpecStats::new(df.block_size);
    // max_think_tokens enforcement state (mirrors the AR path).
    let mut think_count: usize = 0;
    let mut prev_in_think = false;
    let mut generated = 0usize;

    // Post-prefill compaction (FlashCASK pattern from dflash_spec_demo).
    // If the prompt already filled past budget+beta, compact once before
    // entering the spec loop so the first spec_step writes at physical slot
    // `budget`. compact_offset is maintained on target.kv_cache; subsequent
    // forwards inside spec_step_dflash read it for RoPE phase automatically.
    if let Some(ref ev) = m.eviction {
        if let Some(res) = ev.maybe_evict(gpu, &mut target.kv_cache, position).unwrap() {
            let pre_phys = position;
            eprintln!(
                "[dflash] post-prefill evict: {} -> {} (compact_offset={})",
                pre_phys, res.new_physical, target.kv_cache.compact_offset,
            );
            position = res.new_physical;
            if !res.retain_mask.is_empty() {
                let _ = speculative::apply_eviction_retain_to_draft(
                    gpu,
                    &mut df.draft_scratch,
                    &res.retain_mask,
                    df.draft_config.num_extract(),
                    df.draft_config.hidden,
                    pre_phys,
                );
                speculative::compact_target_hidden_host(
                    &mut df.target_hidden_host,
                    &res.retain_mask,
                    df.draft_config.num_extract(),
                    df.draft_config.hidden,
                );
            }
        }
    }

    // Emit the first token immediately so TTFT is the prefill time.
    streamed_tokens.push(first_token);
    emit_committed_event(
        stdout,
        id,
        first_token,
        streamed_tokens.len() - 1,
        t0.elapsed().as_millis() as u64,
    );
    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
    let new_bytes = &all_bytes[bytes_fed_to_filter..];
    bytes_fed_to_filter = all_bytes.len();
    let first_filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));
    generated += 1;

    // First-token EOS guard. The first token is already emitted above; if
    // it is itself a terminator, do not seed another drafted/verified block.
    // The committed-tail check inside the loop applies the same terminator
    // test to every subsequent token.
    let first_token_is_eos = first_token == target.config.eos_token
        || im_end_token == Some(first_token)
        || tokenizer.is_terminator(first_token);

    let mut rng_state: u64 = 0x13579BDFu64;

    // Resolve `HIPFIRE_DDTREE_PATH_C` ONCE before the decode loop. The
    // previous version re-read the env-var on every spec cycle which
    // is microseconds of waste on a hot path. Validate eagerly: invalid
    // values fall back to spec_step_ddtree_batched (the documented
    // behavior) but warn so misconfigurations don't fail silently.
    //
    // Only meaningful when DDTree itself is enabled (HIPFIRE_DDTREE_BUDGET).
    // `phase1` runs Step 1 only (linear main-path verify); `phase2` adds
    // the lazy branch FA-only re-verify (Steps 2+3). See
    // `docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd`.
    let path_c_mode_owned: Option<&'static str> = match std::env::var("HIPFIRE_DDTREE_PATH_C").ok()
    {
        None => None,
        Some(s) if s.is_empty() => None,
        Some(s) if s == "phase1" => Some("phase1"),
        Some(s) if s == "phase2" => Some("phase2"),
        Some(s) => {
            if df.ddtree.is_some() {
                eprintln!(
                    "[hipfire-daemon] HIPFIRE_DDTREE_PATH_C={:?} is not 'phase1' or 'phase2'. \
                     Falling back to spec_step_ddtree_batched.",
                    s
                );
            }
            None
        }
    };

    // Fast path exit conditions (mirrors the dflash_spec_demo outer loop).
    while !first_filter_stop && !first_token_is_eos && generated < max_tokens {
        if position + df.block_size >= ctx_capacity {
            break;
        }

        // Dispatch: when DDTree is configured (HIPFIRE_DDTREE_BUDGET set
        // at startup), route through `spec_step_ddtree_batched`. Otherwise
        // keep the existing chain-mode `spec_step_dflash` path. The two
        // produce the same `SpecStepResult` shape so the rest of the loop
        // is unchanged. Note: `spec_step_ddtree_batched` is greedy-only
        // (temp=0); the daemon currently runs at 0.0_f32 so this matches.
        let path_c_mode = path_c_mode_owned;
        let step_result = if let Some(dd) = df.ddtree.as_mut() {
            if path_c_mode == Some("phase1") || path_c_mode == Some("phase2") {
                let phase2_snaps = if path_c_mode == Some("phase2") {
                    Some(Phase2Snapshots {
                        parent_pre_snap: &mut dd.path_c_parent_pre_snap,
                        main_end_snap: &mut dd.path_c_main_end_snap,
                    })
                } else {
                    None
                };
                spec_step_ddtree_path_c(
                    gpu,
                    &mut target,
                    &df.draft_weights,
                    &df.draft_config,
                    &mut df.draft_scratch,
                    &mut df.hidden_rb,
                    &mut df.target_hidden_host,
                    &mut df.target_snap,
                    &mut df.gdn_tape,
                    &df.verify_scratch,
                    position,
                    seed_token,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                    phase2_snaps,
                )
            } else {
                spec_step_ddtree_batched(
                    gpu,
                    &mut target,
                    &df.draft_weights,
                    &df.draft_config,
                    &mut df.draft_scratch,
                    &mut df.hidden_rb,
                    &mut df.target_hidden_host,
                    &mut df.target_snap,
                    &mut dd.post_seed_snap,
                    &mut df.gdn_tape,
                    &dd.scratch,
                    &df.verify_scratch,
                    position,
                    seed_token,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                )
            }
        } else {
            spec_step_dflash(
                gpu,
                &mut target,
                &df.draft_weights,
                &df.draft_config,
                &mut df.draft_scratch,
                &mut df.hidden_rb,
                &mut df.target_hidden_host,
                &mut df.target_snap,
                &df.verify_scratch,
                position,
                seed_token,
                None, // ctx_slice = full history
                Some(&mut df.gdn_tape),
                0.0_f32, // temperature
                &mut rng_state,
                None, // block_size override
                None, // ngram_cache
                &emitted,
                0.0_f32, // cactus_delta
                None,    // pld_spine
                1.0_f32, // repeat_penalty (off)
                0,       // repeat_window
            )
        };
        let step = match step_result {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"spec_step: {}"}}"#,
                    id, e
                );
                let _ = stdout.flush();
                break;
            }
        };
        stats.record(&step);
        let committed_tail: Vec<u32> = step.committed.iter().skip(1).copied().collect();

        let mut hit_eos = false;
        let mut think_cap_hit = false;
        for &tok in &committed_tail {
            if generated >= max_tokens {
                break;
            }
            emitted.push(tok);
            streamed_tokens.push(tok);
            emit_committed_event(
                stdout,
                id,
                tok,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if emit_filter_action(stdout, id, filter.observe(new_bytes)) {
                hit_eos = true;
                break;
            }
            generated += 1;
            if tok == target.config.eos_token
                || im_end_token == Some(tok)
                || tokenizer.is_terminator(tok)
            {
                hit_eos = true;
                break;
            }

            // max_think_tokens enforcement (mirrors the AR path). Track
            // <think>/<⁄think> in decoded text and count tokens inside.
            if max_think_tokens > 0 {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think && !prev_in_think {
                    think_count = 0;
                }
                if in_think {
                    think_count += 1;
                }
                prev_in_think = in_think;

                if in_think && think_count >= max_think_tokens {
                    // Force-close: emit </think>\n and break out of this batch.
                    // Unlike the AR path we can't splice into the KV cache mid-
                    // spec-cycle, so we just stream the close text and break.
                    // The next request will start fresh.
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"token","id":"{}","text":"</think>\n"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    think_cap_hit = true;
                    break;
                }
            }
        }
        let rollback = speculative::spec_rollback_parity_decision_for_step(position, &step);
        if !rollback.allow_single_session {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"DFlash rollback parity guard failed: {}"}}"#,
                id, rollback.reason
            );
            let _ = stdout.flush();
            break;
        }
        debug_assert!(!rollback.allow_multi_request_verify_batch);
        position = rollback.next_position;
        seed_token = step.bonus_token;
        // Per-cycle eviction (FlashCASK). Fires whenever current physical
        // has grown to budget+β since the last compaction. No-op when
        // physical < budget+β, so non-firing cycles pay only the check cost.
        if let Some(ref ev) = m.eviction {
            if let Some(res) = ev.maybe_evict(gpu, &mut target.kv_cache, position).unwrap() {
                let pre_phys = position;
                position = res.new_physical;
                if !res.retain_mask.is_empty() {
                    let _ = speculative::apply_eviction_retain_to_draft(
                        gpu,
                        &mut df.draft_scratch,
                        &res.retain_mask,
                        df.draft_config.num_extract(),
                        df.draft_config.hidden,
                        pre_phys,
                    );
                    speculative::compact_target_hidden_host(
                        &mut df.target_hidden_host,
                        &res.retain_mask,
                        df.draft_config.num_extract(),
                        df.draft_config.hidden,
                    );
                }
            }
        }
        if hit_eos || think_cap_hit {
            break;
        }
    }

    // Put target state back on LoadedModel so the next request sees fresh
    // (reset) state. We zero DN/kv on entry anyway, but we still need the
    // ownership back.
    m.q35_weights = Some(target.weights);
    m.kv_cache = Some(target.kv_cache);
    m.dn_state = Some(target.dn_state);
    m.q35_scratch = Some(target.scratch);
    m.seq_pos = position;
    m.conversation_tokens = emitted.clone();

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prompt_tokens.len() as f64 / prefill_s
    } else {
        0.0
    };
    let tau = if stats.cycles > 0 {
        stats.accepted_tokens as f64 / stats.cycles as f64
    } else {
        0.0
    };
    // Per PRD §3.1, when PFlash bypassed (e.g. dflash_decode_active for
    // this branch) the `done` object must surface the bypass reason and
    // alpha alongside the dflash perf metrics. Build a small fragment
    // when both are available; otherwise empty for back-compat.
    let pflash_done_field = match (pflash_bypass_reason, pflash_alpha) {
        (Some(r), Some(a)) => format!(
            r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
            r.replace('"', "'"),
            a,
        ),
        _ => String::new(),
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1},"dflash":true,"tau":{:.2},"cycles":{}{}}}"#,
        id,
        generated,
        tok_s,
        prompt_tokens.len(),
        prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        prefill_s * 1000.0,
        tau,
        stats.cycles,
        pflash_done_field,
    );
    let _ = stdout.flush();
}

/// Multi-GPU pipeline-parallel AR decode (Stage 7 of #58). Mirrors the pp=1
/// `generate` Qwen3.5 branch feature-for-feature: ChatFrame ChatML wrap,
/// EosFilter UTF-8 streaming + strip-think + stop_at, LoopGuard n-gram
/// detection, repeat penalty, attractor block on unclosed tool/think
/// openers, max_think_tokens force-close, budget-alert nudge, ChatML \n
/// trailer. Forward calls fan out to per-device tensors via
/// `gpus.devices[dev]` and `scratch_set.per_device[dev]`; the final
/// sample lives on `gpus.output_device`. DFlash, CASK, PFlash, VL and
/// arch_id < 5 are refused upstream at load.
#[allow(clippy::too_many_arguments)]
fn generate_multi(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
) {
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        if let (Some(ref dn), Some(ref mut gpus), Some(ref la)) = (
            m.dn_state.as_ref(),
            m.pp_gpus.as_mut(),
            m.pp_dn_la_to_device.as_ref(),
        ) {
            for (i, s) in dn.s_matrices.iter().enumerate() {
                let g = &mut gpus.devices[la[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            for (i, s) in dn.s_scales.iter().enumerate() {
                let g = &mut gpus.devices[la[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            for (i, s) in dn.conv_states.iter().enumerate() {
                let g = &mut gpus.devices[la[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
        }
        if let Some(kv) = m.kv_cache.as_mut() {
            kv.compact_offset = 0;
        }
    }

    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let raw_q_tokens = tokenizer.encode(prompt);

    // PFlash compression on first turn (seq_pos == 0). Drafter runs on the
    // daemon's single-GPU `gpu` handle, which binds to the same physical
    // device as `pp_gpus.devices[0]` (HIP enumerates within ROCR_VISIBLE).
    // VRAM is shared between the two Gpu handles via the HIP heap, so
    // drafter weights coexist with the target's dev 0 portion. Output is
    // a Vec<u32> of kept token IDs which feeds forward_prefill_batch_multi
    // unchanged. Mode=Off / drafter unloaded falls through to raw tokens.
    let request_kind = match tokenizer.special_token_id("<tool_call>") {
        Some(tid) => {
            let in_user = raw_q_tokens.iter().any(|&t| t == tid);
            let in_system = system_prompt
                .map(|s| tokenizer.encode(s).iter().any(|&t| t == tid))
                .unwrap_or(false);
            if in_user || in_system {
                hipfire_arch_qwen35::pflash::RequestKind::ToolCall
            } else {
                hipfire_arch_qwen35::pflash::RequestKind::Text
            }
        }
        None => hipfire_arch_qwen35::pflash::RequestKind::Text,
    };
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.seq_pos == 0 {
            match hipfire_arch_qwen35::pflash::maybe_compress_prompt(
                gpu,
                state,
                cfg,
                &raw_q_tokens,
                request_kind,
                &[],
            ) {
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Compressed(cp)) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"total_ms":{}}}"#,
                        id,
                        cp.source_tokens,
                        cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5,
                        cp.compressed_md5,
                        cp.timings.score_ms,
                        cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    cp.token_ids
                }
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Bypass { reason }) => {
                    if !matches!(reason, hipfire_arch_qwen35::pflash::BypassReason::ModeOff) {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id,
                            reason.as_str().replace('"', "'"),
                        );
                        let _ = stdout.flush();
                    }
                    raw_q_tokens
                }
                Err(e) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = stdout.flush();
                    raw_q_tokens
                }
            }
        } else {
            raw_q_tokens
        }
    } else {
        raw_q_tokens
    };

    // ChatML framing — two paths, same shape as the single-GPU AR
    // generate() (line 3147+):
    //
    //   1) HIPFIRE_JINJA_CHAT=1 + model has chat_template + seq_pos==0
    //      → render via JinjaChatFrame so structured tools/messages
    //      reach the upstream template. PFlash compression is bypassed
    //      under Jinja (q_tokens is unused; the rendered prompt string
    //      is re-tokenized straight through).
    //
    //   2) Default: hand-rolled ChatFrame::Plain scaffold, byte-
    //      identical to the pp=1 default path so multi-turn behavior
    //      matches between pp=1 and pp>1 when both run the same prompt.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.seq_pos == 0 && m.chat_template.is_some();
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                eprintln!("[daemon] jinja render failed in pp path ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: if m.seq_pos == 0 { system_prompt } else { None },
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: if m.seq_pos == 0 { system_prompt } else { None },
            user: "",
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build_with_user_tokens(&q_tokens)
    };

    let trailer = nl.len();
    if m.seq_pos + new_tokens.len() + max_tokens + trailer > m.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
            id,
            m.seq_pos,
            new_tokens.len(),
            max_tokens,
            trailer,
            m.physical_cap
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    let tool_call_pair = match (
        tokenizer.special_token_id("<tool_call>"),
        tokenizer.special_token_id("</tool_call>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };

    let prefill_tokens = new_tokens.len();
    let t0 = Instant::now();
    let chat_template_profile = m.chat_template_profile.clone();

    let config = m.q35_config.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch_set = m.pp_scratch_set.as_ref().unwrap();
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();
    let gpus = m.pp_gpus.as_mut().unwrap();

    let dev_last = gpus.output_device;
    let vocab_size = config.vocab_size;
    let repeat_buf_cap =
        (scratch_set.per_device[dev_last].repeat_buf.buf.size() / 4).min(repeat_window);

    if let Err(e) = qwen35::forward_prefill_batch_multi(
        gpus,
        weights,
        config,
        &new_tokens,
        m.seq_pos,
        kv,
        dn,
        scratch_set,
    ) {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"forward_prefill_batch_multi: {}"}}"#,
            id, e
        );
        let _ = stdout.flush();
        return;
    }
    m.seq_pos += new_tokens.len();
    m.conversation_tokens.extend_from_slice(&new_tokens);

    // ngram scope: generated tokens only (matches pp=1).
    let ngram_scope_start = m.conversation_tokens.len();

    let mut rng_state: u32 = 0x13579BDFu32;

    let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
        .into_iter()
        .chain(think_pair.into_iter())
        .collect();

    // First sample on the output device.
    let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
    let mut blocked0: Vec<u32> = Vec::new();
    collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
    let cfg0 = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window: repeat_buf_cap,
        presence_penalty,
        frequency_penalty,
        blocked_tokens: blocked0,
    };
    let tok0 = {
        let s_last = &scratch_set.per_device[dev_last];
        let g_last = &mut gpus.devices[dev_last];
        sampler::sample(
            g_last,
            &s_last.logits,
            &s_last.sample_buf,
            &s_last.repeat_buf,
            vocab_size,
            ngram_scope,
            &cfg0,
            &mut rng_state,
        )
    };
    let t_prefill = Instant::now();
    let mut next_token = tok0;

    let mut generated = 0usize;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter =
        chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);
    let mut alert_fired = false;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;
    let loop_guard = loop_guard_from_runtime_config();

    while generated < max_tokens {
        generated += 1;
        m.conversation_tokens.push(next_token);
        streamed_tokens.push(next_token);
        emit_committed_event(
            stdout,
            id,
            next_token,
            streamed_tokens.len() - 1,
            t0.elapsed().as_millis() as u64,
        );
        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

        if let Err(e) = qwen35::forward_scratch_multi(
            gpus,
            weights,
            config,
            next_token,
            m.seq_pos,
            kv,
            dn,
            scratch_set,
        ) {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"forward_scratch_multi decode: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            return;
        }
        m.seq_pos += 1;

        if next_token == config.eos_token {
            break;
        }
        if filter_stop {
            break;
        }
        if im_end_token == Some(next_token) {
            break;
        }
        if tokenizer.is_terminator(next_token) {
            break;
        }

        // max_think_tokens enforcement: same decoded-text scan as pp=1.
        if max_think_tokens > 0 {
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let open_idx = raw_str.rfind("<think>");
            let close_idx = raw_str.rfind("</think>");
            let in_think = match (open_idx, close_idx) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                _ => false,
            };
            if in_think {
                if !prev_in_think {
                    think_count = 1;
                } else {
                    think_count += 1;
                }
            } else {
                think_count = 0;
            }
            prev_in_think = in_think;

            if in_think && think_count >= max_think_tokens {
                let close_tokens = tokenizer.encode("</think>\n");
                let budget_left = max_tokens.saturating_sub(generated);
                let take = close_tokens.len().min(budget_left);
                for &t in &close_tokens[..take] {
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        t,
                        m.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        eprintln!("[daemon] max_think close forward_scratch_multi: {}", e);
                        break;
                    }
                    m.seq_pos += 1;
                    m.conversation_tokens.push(t);
                    streamed_tokens.push(t);
                    emit_committed_event(
                        stdout,
                        id,
                        t,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes = &all_bytes[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes.len();
                    let _ = emit_filter_action(stdout, id, filter.observe(new_bytes));
                    generated += 1;
                }
                think_count = 0;
                prev_in_think = false;
                if generated >= max_tokens {
                    break;
                }
            }
        }

        // N-gram loop detector (token-side, no GPU work).
        if let Some(StopReason::NgramRepeat { count, .. }) = loop_guard.check(&streamed_tokens) {
            let window_len = loop_guard.window_len(streamed_tokens.len());
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
                id, count, window_len
            );
            let _ = stdout.flush();
            break;
        }

        // Budget-alert injection: gated to inside an open <think> block.
        if !alert_fired
            && budget_alert_at_tok > 0
            && generated >= budget_alert_at_tok
            && !budget_alert_text.is_empty()
        {
            alert_fired = true;
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = match (raw_str.rfind("<think>"), raw_str.rfind("</think>")) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                _ => false,
            };
            if !in_think {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                    id
                );
                let _ = stdout.flush();
                let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
                let mut blocked: Vec<u32> = Vec::new();
                collect_unclosed_attractor_blocks(
                    ngram_scope,
                    &attractor_pairs,
                    20,
                    2,
                    &mut blocked,
                );
                let cfg = SamplerConfig {
                    temperature: temp,
                    top_p,
                    repeat_penalty,
                    repeat_window: repeat_buf_cap,
                    presence_penalty,
                    frequency_penalty,
                    blocked_tokens: blocked,
                };
                next_token = {
                    let s_last = &scratch_set.per_device[dev_last];
                    let g_last = &mut gpus.devices[dev_last];
                    sampler::sample(
                        g_last,
                        &s_last.logits,
                        &s_last.sample_buf,
                        &s_last.repeat_buf,
                        vocab_size,
                        ngram_scope,
                        &cfg,
                        &mut rng_state,
                    )
                };
                continue;
            }
            let nudge_tokens = tokenizer.encode(budget_alert_text);
            let budget_left = max_tokens.saturating_sub(generated);
            let nudge_len = nudge_tokens.len().min(budget_left);
            let need_kv = m.seq_pos + nudge_len + (max_tokens - generated - nudge_len) + nl.len();
            if nudge_len > 0 && need_kv <= m.physical_cap {
                for &tok in &nudge_tokens[..nudge_len] {
                    m.conversation_tokens.push(tok);
                    streamed_tokens.push(tok);
                    emit_committed_event(
                        stdout,
                        id,
                        tok,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes2.len();
                    let _ = emit_filter_action(stdout, id, filter.observe(new_bytes2));
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        tok,
                        m.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        eprintln!("[daemon] budget_alert forward_scratch_multi: {}", e);
                        break;
                    }
                    m.seq_pos += 1;
                    generated += 1;
                }
            } else if nudge_len < nudge_tokens.len() {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#,
                    id, nudge_len, budget_left
                );
                let _ = stdout.flush();
            } else {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#,
                    id
                );
                let _ = stdout.flush();
            }
            if generated >= max_tokens {
                break;
            }
        }

        // Steady-state sample.
        let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
        let mut blocked: Vec<u32> = Vec::new();
        collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked);
        let cfg = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked,
        };
        next_token = {
            let s_last = &scratch_set.per_device[dev_last];
            let g_last = &mut gpus.devices[dev_last];
            sampler::sample(
                g_last,
                &s_last.logits,
                &s_last.sample_buf,
                &s_last.repeat_buf,
                vocab_size,
                ngram_scope,
                &cfg,
                &mut rng_state,
            )
        };
    }

    // ChatML \n trailer so the next turn opens cleanly.
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            if let Err(e) = qwen35::forward_scratch_multi(
                gpus,
                weights,
                config,
                t,
                m.seq_pos,
                kv,
                dn,
                scratch_set,
            ) {
                eprintln!("[daemon] trailer forward_scratch_multi: {}", e);
                break;
            }
            m.seq_pos += 1;
            m.conversation_tokens.push(t);
        }
    }

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 {
        generated as f64 / total_s
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_s > 0.0 {
        prefill_tokens as f64 / prefill_s
    } else {
        0.0
    };
    let decode_tok_s = if decode_s > 0.0 {
        generated as f64 / decode_s
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
        id,
        generated,
        tok_s,
        prefill_tokens,
        prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        prefill_s * 1000.0
    );
    let _ = stdout.flush();
}

#[allow(clippy::too_many_arguments)]
fn generate(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    drafter_gpu: Option<&mut rdna_compute::Gpu>,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    think_mode: ThinkMode,
    prefill_already_done: bool,
    prefilled_prompt_tokens: Option<usize>,
    request_stop_sequences: &[String],
    evidence_dir: Option<&str>,
) {
    // Seed the process-global CPU sampler RNG for this request. CPU fallback and
    // grammar/VL-style sampling should not inherit RNG state from prior requests.
    hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);

    // Compress runs on the PFlash drafter handle when one is set (hetero
    // sibling device), else on the target gpu. The handle is consumed at
    // the seq_pos==0 compress site; decode always uses `gpu`.
    let mut drafter_gpu = drafter_gpu;
    // arch_id=7 (hipfire-arch-qwen2) short-circuit. The standard
    // generate() body is qwen35/llama-shaped and would panic on
    // None unwraps for q35_*/llama_* fields when applied to a
    // Qwen2 model. Route here BEFORE PFlash / DFlash / multi-GPU
    // / ChatML scaffolding since none of those are wired for
    // arch_id=7 yet (R3 bring-up scope).
    if m.arch_id == 7 {
        // Silence the qwen35/llama-only params we deliberately don't
        // honor on this path. See generate_qwen2 doc for the deferral
        // list.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            tools,
            messages_history,
            prefill_already_done,
        );
        generate_qwen2(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
        );
        return;
    }
    if m.arch_id == 9 {
        // arch_id=9 (DeepSeek V4 Flash). Standalone bring-up — same
        // shape as the qwen2 short-circuit above. PFlash / DFlash / VL
        // / multi-GPU / sampler-budget / ChatML scaffolding all bypass.
        // We honour `system_prompt`, `temp`, `top_p`, `tools`, and
        // `messages_history` per HF V4 chat template + sampling
        // recommendations; everything else routes through future
        // follow-ups.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
        );
        let _ = (repeat_penalty, repeat_window);
        generate_deepseek4(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            think_mode,
            tools,
            messages_history,
        );
        return;
    }
    if m.arch_id == 10 {
        // arch_id=10 (MiniMax-M2). Minimal AR bring-up — same shape as the
        // qwen2 / deepseek4 short-circuits above. PFlash / DFlash / VL /
        // multi-GPU / sampler-budget / grammar / tools-execution all bypass.
        // We honour `system_prompt`, `temp`, `top_p`, and (via JinjaChatFrame)
        // `messages_history` + `tools` rendering; spec-decode / MTP / grammar
        // are out of scope for the scaffold.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            repeat_penalty,
            repeat_window,
            think_mode,
        );
        generate_minimax(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            max_think_tokens,
            tools,
            messages_history,
        );
        return;
    }
    #[cfg(feature = "arch-lfm2moe")]
    if m.arch_id == 11 {
        // arch_id=11 (LFM2.5-MoE). Minimal AR bring-up — same shape as the
        // qwen2 / deepseek4 / minimax short-circuits above. PFlash / DFlash /
        // VL / multi-GPU / sampler-budget / grammar / tools-execution all
        // bypass. We honour `system_prompt`, `temp`, `top_p`, and (via
        // JinjaChatFrame) `messages_history` + `tools` rendering; spec-decode
        // / MTP / grammar are out of scope for the scaffold.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            repeat_penalty,
            repeat_window,
            think_mode,
        );
        generate_lfm2moe(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            max_think_tokens,
            tools,
            messages_history,
        );
        return;
    }
    // Multi-GPU pipeline-parallel dispatch (Stage 7 of #58). pp>1 is refused
    // at load when DFlash / CASK / PFlash / VL is requested, so this branch
    // doesn't need to thread any of those args through.
    if m.pp > 1 {
        generate_multi(
            m,
            gpu,
            pflash_state,
            pflash_cfg,
            stdout,
            id,
            prompt,
            system_prompt,
            temp,
            top_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            request_stop_sequences,
        );
        return;
    }
    // DFlash fast path -- only when a draft model is loaded AND temperature is
    // effectively 0 (DFlash is greedy-only in this integration). Skip the
    // normal AR sampling setup entirely.
    //
    // Exception: thinking-on + max_think_tokens currently needs the AR path.
    // DFlash's budget cap can close/strip the think span but does not yet
    // continue into visible answer text after the forced close. AR already
    // splices </think> through KV and continues generation, so route budgeted
    // thinking requests there until DFlash continuation is implemented.
    let budgeted_thinking_needs_ar = max_think_tokens > 0
        && !matches!(assistant_prefix, prompt_frame::AssistantPrefix::ClosedThink);
    if prefill_already_done && m.dflash.is_some() {
        write_error(
            stdout,
            id,
            "prefill_already_done is not supported on DFlash-loaded models",
        );
        return;
    }
    if prefill_already_done && pflash_state.is_some() {
        write_error(
            stdout,
            id,
            "prefill_already_done is not supported when PFlash state is loaded",
        );
        return;
    }

    if m.dflash.is_some()
        && temp <= 1e-6
        && is_qwen35_family_arch_id(m.arch_id)
        && !budgeted_thinking_needs_ar
    {
        // PFlash + DFlash decode path is not yet wired -- the DFlash spec
        // loop builds its own prompt token stream internally, so the
        // generate() PFlash block below never runs. Surface this loud so
        // an operator who set prefill_compression != off sees a clear
        // bypass event instead of silently getting full-prefill behavior
        // they didn't ask for. Compression-on-DFlash lands in a future
        // phase that threads PflashState through generate_dflash().
        let mut dflash_bypass_reason: Option<&'static str> = None;
        let dflash_alpha = pflash_cfg.as_ref().map(|c| c.alpha);
        if let Some(cfg) = pflash_cfg.as_ref() {
            if cfg.mode != hipfire_arch_qwen35::pflash::PflashMode::Off {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"pflash_bypass","id":"{}","reason":"dflash_decode_active (pflash compression on the DFlash path is a follow-up; set dflash_mode=off to compress with AR decode)"}}"#,
                    id,
                );
                let _ = stdout.flush();
                dflash_bypass_reason = Some("dflash_decode_active");
            }
        }
        // max_think_tokens is now enforced inside generate_dflash (it
        // mirrors the AR path's <think>/</think> counter). The "ignored
        // on DFlash" warning that used to live here is gone -- the cap
        // is real on both paths now.
        generate_dflash(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            dflash_bypass_reason,
            dflash_alpha,
            tools,
            messages_history,
            request_stop_sequences,
        );
        // Silence unused-variable warnings for the params DFlash doesn't
        // consume (top_p / repeat penalties are AR-only sampling knobs;
        // pflash_state is bypassed on the DFlash decode path).
        let _ = (
            top_p,
            repeat_penalty,
            repeat_window,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
        );
        return;
    }

    // MTP spec-decode: qwen35 model with a co-trained MTP head, no DFlash
    // drafter, greedy, mtp_mode enabled. Uses the non-tree spec_step_mtp
    // (FP32 DeltaNet state is tree-incompatible — see generate_mtp / TODO.md).
    if m.dflash.is_none()
        && m.mtp_weights_present
        && m.mtp_mode != "off"
        && temp <= 1e-6
        && is_qwen35_family_arch_id(m.arch_id)
        && !prefill_already_done
        && !budgeted_thinking_needs_ar
    {
        generate_mtp(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            tools,
            messages_history,
            request_stop_sequences,
        );
        let _ = (
            top_p,
            repeat_penalty,
            repeat_window,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
        );
        return;
    }

    let is_qwen35_ar = is_qwen35_family_arch_id(m.arch_id);
    let mut q35_session = if is_qwen35_ar {
        match Qwen35RequestSessionState::take_from_loaded(m, gpu) {
            Ok(session) => Some(session),
            Err(e) => {
                write_error(stdout, id, &format!("qwen35 request session state: {e}"));
                return;
            }
        }
    } else {
        None
    };

    // Auto-reset on multi-turn rollover. When eviction is active (operator
    // enabled cask_sidecar at load), the physical buffer is bounded by
    // budget+beta+safety regardless of conversation length, so reset never
    // needs to fire — eviction reclaims slots after each token. When eviction
    // is OFF, physical grows unbounded up to max_seq; reset when we'd overrun.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    let current_seq_pos = q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos);
    if !prefill_already_done
        && m.eviction.is_none()
        && current_seq_pos + prompt_est + max_tokens > m.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            current_seq_pos, m.max_seq
        );
        if let Some(session) = q35_session.as_mut() {
            session.reset(gpu);
        } else {
            m.seq_pos = 0;
            m.conversation_tokens.clear();
            if let Some(kv) = m.llama_kv.as_mut() {
                kv.compact_offset = 0;
            }
        }
    }

    // `nl` is needed for the trailer write after natural <|im_end|>
    // termination; `im_end` derives the EOS-check token id. Other
    // ChatML scaffolding tokens are now built inside hipfire-prompt.
    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let raw_q_tokens = tokenizer.encode(prompt);

    // ── PFlash compression (Phase 4.1 #93) ──────────────────────────────
    //
    // Only on first turn (seq_pos == 0). Multi-turn compression of newly-
    // added user content has knock-on effects on prior KV state that we
    // haven't validated yet, so subsequent turns always bypass.
    //
    // Compression operates on the user's actual content tokens
    // (`raw_q_tokens`); chat-template scaffolding (im_start / role / nl /
    // im_end) wraps the result AFTER and is never compressed away.
    // Empty must_keep_spans is correct: there are no chat boundaries
    // INSIDE q_tokens (they live in the scaffolding the daemon adds).
    //
    // Bypass / compressed status is reported as a `pflash_compressed` or
    // `pflash_bypass` event so operators can see what the request actually
    // ran through.
    //
    // Tool-call detection: the prompt may contain a `<tool_call>` token
    // that the parser uses for structure. Compressing those tokens away
    // would corrupt the response shape, so we surface a ToolCall request
    // kind to the gate and let `decide_bypass` reject the request loudly.
    //
    // Two scan locations:
    //   1. raw_q_tokens (the user message itself).
    //   2. system_prompt -- the OpenAI serve path puts tool definitions
    //      and the `<tool_call>` format example in the system prompt
    //      when `body.tools` is present (cli/index.ts buildSystem). A
    //      first-turn user message with tools therefore needs a system-
    //      prompt scan or it would slip through as Text and get its
    //      schema text mangled by compression.
    //
    // Detection is best-effort -- the special-token id is missing on
    // older vocabs, in which case the gate just routes through Text.
    let request_kind = match tokenizer.special_token_id("<tool_call>") {
        Some(tid) => {
            let in_user = raw_q_tokens.iter().any(|&t| t == tid);
            let in_system = system_prompt
                .map(|s| tokenizer.encode(s).iter().any(|&t| t == tid))
                .unwrap_or(false);
            if in_user || in_system {
                hipfire_arch_qwen35::pflash::RequestKind::ToolCall
            } else {
                hipfire_arch_qwen35::pflash::RequestKind::Text
            }
        }
        None => hipfire_arch_qwen35::pflash::RequestKind::Text,
    };

    // Stashed CompressedPrompt summary (when compression actually fired);
    // appended to the `done` event later so a streaming client gets one
    // consolidated line. None means no compression happened on this request.
    let mut pflash_summary: Option<hipfire_arch_qwen35::pflash::CompressedPrompt> = None;
    // Bypass reason when compression was attempted but skipped (mode != Off
    // and a drafter was loaded). PRD §3.1 requires "bypass reason if
    // skipped" in the done object.
    let mut pflash_bypass_reason: Option<String> = None;
    // Effective alpha for this request (from cfg if pflash_state is loaded).
    // PRD §3.1 lists alpha as a required done-object field.
    let pflash_alpha: Option<f32> = pflash_cfg.map(|c| c.alpha);
    // Helper: render the JSON field fragment for `done` per PRD §3.1.
    // Three states:
    //   - compressed: full metadata + alpha
    //   - bypass (non-Off, drafter loaded): alpha + bypass_reason
    //   - nothing: empty string so backwards-compatible clients see the
    //     original done shape
    fn pflash_done_fragment(
        s: &Option<hipfire_arch_qwen35::pflash::CompressedPrompt>,
        bypass_reason: &Option<String>,
        alpha: Option<f32>,
    ) -> String {
        match (s, bypass_reason) {
            (Some(cp), _) => format!(
                r#","pflash":{{"source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"alpha":{:.6},"score_ms":{},"total_ms":{},"source_md5":"{}","compressed_md5":"{}"}}"#,
                cp.source_tokens,
                cp.kept_tokens,
                cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                alpha.unwrap_or(0.0),
                cp.timings.score_ms,
                cp.timings.total_ms,
                cp.source_md5,
                cp.compressed_md5,
            ),
            (None, Some(reason)) => format!(
                r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
                reason.replace('"', "'"),
                alpha.unwrap_or(0.0),
            ),
            (None, None) => String::new(),
        }
    }
    if std::env::var("HIPFIRE_PFLASH_DEBUG").is_ok() {
        eprintln!(
            "[pflash] gen: state={} cfg-present seq_pos={} q={} drafter_gpu={}",
            pflash_state.is_some(),
            q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos),
            raw_q_tokens.len(),
            drafter_gpu.is_some()
        );
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        let seq_pos = q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos);
        if seq_pos == 0 {
            let compress_gpu: &mut rdna_compute::Gpu = drafter_gpu.as_deref_mut().unwrap_or(gpu);
            // Sibling-device drafter: bind its device before compress, then
            // restore the target binding for decode. No-op when shared.
            compress_gpu.bind_thread_or_warn();
            let decision = hipfire_arch_qwen35::pflash::maybe_compress_prompt(
                compress_gpu,
                state,
                cfg,
                &raw_q_tokens,
                request_kind,
                &[],
            );
            gpu.bind_thread_or_warn();
            match decision {
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Compressed(cp)) => {
                    eprintln!(
                        "[pflash] COMPRESSED {} -> {} tok dev1 ({}ms)",
                        cp.source_tokens, cp.kept_tokens, cp.timings.total_ms
                    );
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"select_ms":{},"gather_ms":{},"total_ms":{}}}"#,
                        id,
                        cp.source_tokens,
                        cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5,
                        cp.compressed_md5,
                        cp.timings.score_ms,
                        cp.timings.select_ms,
                        cp.timings.gather_ms,
                        cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    let token_ids = cp.token_ids.clone();
                    pflash_summary = Some(cp);
                    token_ids
                }
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Bypass { reason }) => {
                    eprintln!(
                        "[pflash] BYPASS reason={} q={}",
                        reason.as_str(),
                        raw_q_tokens.len()
                    );
                    // Only emit bypass events for non-trivial reasons.
                    // ModeOff is the silent default; nothing to report.
                    if !matches!(reason, hipfire_arch_qwen35::pflash::BypassReason::ModeOff) {
                        let r = reason.as_str();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id,
                            r.replace('"', "'"),
                        );
                        let _ = stdout.flush();
                        // Stash for the `done` object too so a single-line
                        // log scrape sees both the bypass reason and the
                        // request's prefill timings.
                        pflash_bypass_reason = Some(r);
                    }
                    raw_q_tokens
                }
                Err(e) => {
                    eprintln!("[pflash] ERROR compress: {e}");
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id,
                        e.to_string().replace('"', "'"),
                    );
                    let _ = stdout.flush();
                    raw_q_tokens
                }
            }
        } else {
            raw_q_tokens
        }
    } else {
        raw_q_tokens
    };

    // ChatML framing — two paths:
    //
    //   1) `HIPFIRE_JINJA_CHAT=1` AND model carries an embedded chat_template
    //      AND first turn (seq_pos == 0): render through `JinjaChatFrame`
    //      against the upstream HF Jinja template, producing the byte
    //      sequence the model was actually trained on (fixes the "hand-roll
    //      drifted from upstream template" class — XML tool calls on
    //      Qwen3.5/3.6 instead of JSON, `<|im_start|>user` for tool
    //      responses instead of `<|im_start|>tool`, etc.). PFlash
    //      compression is bypassed under Jinja for now (q_tokens not
    //      reusable when the template renders to a String).
    //
    //   2) Default: hand-rolled `prompt_frame::ChatFrame::Plain`
    //      scaffold, byte-identical to today's behavior.
    //
    // Multi-turn (seq_pos > 0) currently always uses path 2 — Jinja
    // single-turn parity is Stage 2; multi-turn message-history state on
    // the daemon side is Stage 2 follow-up.
    //
    // Thinking-off interop with `assistant_prefix`: the CLI sets BOTH
    // `max_think_tokens = 1` AND `assistant_prefix = ClosedThink` when
    // the request asks for non-thinking. The Jinja path keys off
    // `max_think_tokens != 1` for `enable_thinking`; the Plain path
    // honors `assistant_prefix` directly (ClosedThink emits a closed
    // `<think></think>` block after the assistant prefix). Each path
    // picks up the signal it needs.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let seq_pos_for_prompt = if prefill_already_done {
        0
    } else {
        q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos)
    };
    let try_jinja = jinja_enabled && seq_pos_for_prompt == 0 && m.chat_template.is_some();
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        // Phase 1 of Jinja-everywhere migration: when the caller supplies
        // either a `tools` array or a `messages` history (or both), route
        // through `render_messages` so the upstream template's
        // `{% if tools %}` / multi-turn branches fire. With neither
        // supplied, fall through to the single-turn `render()` convenience,
        // which is byte-identical to the synthesized [system?, user]
        // path that shipped under HIPFIRE_JINJA_CHAT=1 before this change.
        let render_result = if tools.is_some() || messages_history.is_some() {
            // Synthesize [system?, user] when no explicit history was
            // provided. Tools-with-legacy-prompt is the natural OpenAI
            // function-calling shape (one turn + tool definitions).
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(prompt_frame::Message {
                        role: prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                eprintln!("[daemon] jinja render failed ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: if seq_pos_for_prompt == 0 {
                system_prompt
            } else {
                None
            },
            user: "", // unused: we pass tokens directly via build_with_user_tokens
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build_with_user_tokens(&q_tokens)
    };

    // KV-budget guard. Without eviction the physical buffer is the hard cap;
    // we must fit prefill + generation + trailer in one allocation. With
    // eviction, physical is bounded by physical_cap regardless of total tokens
    // — the chunked prefill below calls maybe_evict between chunks, and the
    // decode loop evicts after every token. The only ceiling under eviction is
    // the advertised context window (max_seq) — refuse requests that would
    // overflow it in absolute position terms (current absolute + new).
    let trailer = nl.len();
    let current_seq_pos = q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos);
    let budget_prefill_tokens = if prefill_already_done {
        0
    } else {
        new_tokens.len()
    };
    let absolute_pos = if let Some(session) = q35_session.as_ref() {
        session.seq_pos + session.kv_cache.compact_offset
    } else {
        m.seq_pos + m.llama_kv.as_ref().map(|kv| kv.compact_offset).unwrap_or(0)
    };
    if m.eviction.is_none() {
        if current_seq_pos + budget_prefill_tokens + max_tokens + trailer > m.physical_cap {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
                id, current_seq_pos, budget_prefill_tokens, max_tokens, trailer, m.physical_cap
            );
            let _ = stdout.flush();
            if let Some(session) = q35_session.take() {
                qwen35_restore_or_error(stdout, id, m, gpu, session);
            }
            return;
        }
    } else if absolute_pos + budget_prefill_tokens + max_tokens + trailer > m.max_seq {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds advertised context window: absolute={} + prefill={} + max_tokens={} + trailer={} > max_seq={}"}}"#,
            id, absolute_pos, budget_prefill_tokens, max_tokens, trailer, m.max_seq
        );
        let _ = stdout.flush();
        if let Some(session) = q35_session.take() {
            qwen35_restore_or_error(stdout, id, m, gpu, session);
        }
        return;
    }

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
    // Special-token attractor blocking (#111). Resolve the token IDs once;
    // each pair is `Some` only when the tokenizer registers both opener
    // and closer as single special tokens (Qwen3+ vocabs). Older vocabs
    // return `None` and the block is silently skipped — no behavior
    // change.
    let tool_call_pair = match (
        tokenizer.special_token_id("<tool_call>"),
        tokenizer.special_token_id("</tool_call>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };
    let prefill_tokens = new_tokens.len();
    let t0 = Instant::now();

    if is_qwen35_family_arch_id(m.arch_id) {
        // Qwen3.5 / Qwen3.5-MoE — multi-turn: prefill only the NEW turn tokens,
        // continuing from session.seq_pos (KV cache + DeltaNet state are cumulative)
        let mut session = q35_session.take().expect("qwen35 request session state");
        if prefill_already_done {
            let current_position = session.seq_pos + session.kv_cache.compact_offset;
            let expected_position = prefilled_prompt_tokens.unwrap_or(new_tokens.len());
            if current_position != expected_position {
                write_error(
                    stdout,
                    id,
                    &format!(
                        "prefill_already_done requested but active session position {} does not match expected prefilled prompt token count {}",
                        current_position,
                        expected_position
                    ),
                );
                qwen35_restore_or_error(stdout, id, m, gpu, session);
                return;
            }
        }
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        let kv = &mut session.kv_cache;
        let dn = &mut session.dn_state;
        let moe_router_histogram = DaemonMoeRouterHistogramGuard::start(evidence_dir, config);

        // Prefill this turn's tokens via the batched prefill entry point.
        // On gfx11+ for MQ4/HFQ4/MQ6/HFQ6 weights this hits the WMMA GEMM
        // fast path; other archs fall back to dp2 / FP16-packed / scalar
        // variants. The one sequential hotspot inside is the gated_delta_net
        // Q8 state update (N sequential per-token calls per LA layer, byte-
        // exact with decode to keep the quality gate green).
        //
        // Note: forward_prefill_batch launches HIP kernels asynchronously.
        // The t_prefill mark below lives AFTER the first sample_top_p, whose
        // D2H readback of tok0 forces a device sync — that's the point at
        // which the first token is actually ready to stream. Placing the
        // mark earlier captures CPU-dispatch time, which under-reports
        // prefill by a large factor (prefill_tok_s ~5–10× too optimistic).
        //
        // Under eviction: chunk prefill to the (budget+beta) eviction window
        // and call `maybe_evict` between chunks so physical never exceeds
        // physical_cap. Chunk size caps out at physical capacity available —
        // when physical is at post-evict `budget`, a full `beta`-sized chunk
        // can run before the next eviction fires.
        if !prefill_already_done {
            if let Some(ref ev) = m.eviction {
                let window = ev.budget() + ev.beta();
                let mut remaining: &[u32] = &new_tokens;
                while !remaining.is_empty() {
                    let space = window.saturating_sub(session.seq_pos).max(1);
                    let chunk_len = remaining.len().min(space);
                    let (chunk, rest) = remaining.split_at(chunk_len);
                    qwen35::forward_prefill_batch(
                        gpu,
                        weights,
                        config,
                        chunk,
                        session.seq_pos,
                        kv,
                        dn,
                        scratch,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap();
                    session.seq_pos += chunk_len;
                    if let Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                    {
                        session.seq_pos = new_phys;
                    }
                    remaining = rest;
                }
            } else {
                qwen35::forward_prefill_batch(
                    gpu,
                    weights,
                    config,
                    &new_tokens,
                    session.seq_pos,
                    kv,
                    dn,
                    scratch,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
                session.seq_pos += new_tokens.len();
            }
            session.conversation_tokens.extend_from_slice(&new_tokens);
        }

        // ngram scope for the repeat penalty: ONLY generated tokens (never the
        // prompt). Prior design included the user's prompt as an anti-loop
        // anchor, but that penalizes the very tokens we're asked to recall
        // (names, numbers, facts) under MQ4/MQ6 quantizations that are more
        // RP-sensitive than llama.cpp's Q4_K. First sample: empty scope (no
        // generated tokens yet); subsequent samples: generated-so-far only.
        let ngram_scope_start = if prefill_already_done {
            session
                .conversation_tokens
                .len()
                .saturating_sub(session.prefilled_generated_suffix_len)
        } else {
            session.conversation_tokens.len()
        };
        session.prefilled_generated_suffix_len = 0;

        // Generate. GPU-side sampling eliminates per-token logits download +
        // CPU softmax + CPU repeat penalty. Closes the 2× gap between raw
        // bench throughput and daemon throughput.
        //
        // Kernel signature reads `repeat_tokens[0..repeat_window]`, so we
        // only need to upload the tokens that will actually be read — no
        // need to clear the buffer between calls. The upload is on the same
        // stream as the sample kernel launch, so the copy and compute pipeline
        // naturally.
        let vocab_size = config.vocab_size;
        let mut rng_state: u32 = 0x13579BDFu32;
        let repeat_buf_cap = (scratch.repeat_buf.buf.size() / 4).min(repeat_window);

        // Build the list of paired (open, close) attractor pairs once;
        // collect_unclosed_attractor_blocks decides per-call
        // which openers (if any) trip the depth threshold.
        let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
            .into_iter()
            .chain(think_pair.into_iter())
            .collect();

        // First sample: use conversation so far as scope.
        let ngram_scope = &session.conversation_tokens[ngram_scope_start..];
        // #111 attractor block: empty `ngram_scope` on first sample (no
        // generated tokens yet), so the unclosed-depth is always 0 and
        // `blocked` is empty. Still call collect_* for symmetry with
        // the loop body, in case a future change moves this block into
        // a multi-step warmup.
        let mut blocked0: Vec<u32> = Vec::new();
        collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
        let cfg0 = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            // Window is bounded by the GPU repeat_buf capacity. Pre-PR3 code did this
            // bound by setting `scope_start = len - repeat_buf_cap`
            // and passing `scope.len()` to the kernel; we let
            // sampler::sample do the same `min(window, buf_cap)`
            // internally.
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked0,
        };
        let tok0 = sampler::sample(
            gpu,
            &scratch.logits,
            &scratch.sample_buf,
            &scratch.repeat_buf,
            vocab_size,
            ngram_scope,
            &cfg0,
            &mut rng_state,
        );
        // First token is ready (sample_top_p's D2H forces GPU sync). This is
        // the user-observable "time to first token" boundary — prefill above,
        // decode loop below.
        let t_prefill = Instant::now();
        let mut next_token = tok0;

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        // `bytes_fed_to_filter` is the index into the freshly-decoded
        // byte stream past which we have not yet handed bytes to the
        // filter. The filter owns UTF-8 boundary buffering and any
        // future arch quirks (Gemma 4 marker holdback, strip-think,
        // byte-level stop_at); see crates/engine/src/eos_filter.rs.
        let mut bytes_fed_to_filter = 0usize;
        let mut filter = chat_output_filter(m, request_stop_sequences);
        let mut alert_fired = false;
        // max_think_tokens enforcement state. think_count increments only
        // while we observe ourselves to be inside a `<think>...</think>`
        // block via the same decoded-text scan budget_alert uses. When the
        // cap is hit we splice "</think>\n" into the stream (KV write +
        // stdout emit + advance generated) so the model finishes thinking
        // and commits to an answer with the remaining max_tokens budget.
        // Re-armable: if the model later opens another <think> in the same
        // turn (rare) the counter resets and the cap re-fires.
        let mut think_count: usize = 0;
        let mut prev_in_think: bool = false;

        // N-gram loop detector: track 4-gram token sequences. When any
        // 4-gram repeats more than `ngram_loop_threshold` times in the
        // last `ngram_window` tokens, force EOS. This catches answer-phase
        // repetition loops that the think cap and repeat penalty miss.
        // Operates on token IDs (no decode overhead).
        // Implementation lives in `hipfire-generate` loop_guard; defaults read from
        // HIPFIRE_NGRAM_LOOP_THRESHOLD (default 8, 0 = disabled) and
        // HIPFIRE_NGRAM_WINDOW (default 256). See loop_guard.rs.
        let loop_guard = loop_guard_from_runtime_config();

        // `while` instead of `for 0..max_tokens` so budget-alert injection
        // (which increments `generated` beyond the iteration count) can't
        // push generated past max_tokens: each loop start rechecks the cap.
        while generated < max_tokens {
            generated += 1;
            session.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            emit_committed_event(
                stdout,
                id,
                next_token,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            // Incremental UTF-8 + filter routing: feed only the new
            // bytes since last call, let the filter buffer any partial
            // codepoint or marker prefix until disambiguated.
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

            // Write this token's K/V to the cache FIRST so the next turn
            // always starts from a fully-written context. Breaking before
            // forward_scratch used to leave a hole at the im_end/eos
            // position — the next turn then attended over zero-init K/V
            // at that slot.
            //
            // Under eviction, session.seq_pos is the *physical* write slot; we
            // advance and call maybe_evict immediately so the next write
            // never overruns physical_cap. compact_offset bookkeeping on
            // the cache itself keeps RoPE phase correct across evictions.
            if let Err(e) = qwen35::forward_scratch(
                gpu,
                weights,
                config,
                next_token,
                session.seq_pos,
                kv,
                dn,
                scratch,
            ) {
                write_error(
                    stdout,
                    id,
                    &format!("qwen35 decode forward_scratch failed: {e:?}"),
                );
                qwen35_restore_or_error(stdout, id, m, gpu, session);
                return;
            }
            session.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                {
                    session.seq_pos = new_phys;
                }
            }
            if filter_stop {
                break;
            }

            if next_token == config.eos_token {
                break;
            }
            if im_end_token == Some(next_token) {
                break;
            }
            if tokenizer.is_terminator(next_token) {
                break;
            }

            // max_think_tokens enforcement. Track whether we're inside an
            // open <think>...</think> block and how many tokens we've
            // emitted there. When the cap is hit, splice "</think>\n" into
            // the stream (KV write + stdout emit + advance generated) so
            // the model commits to an answer with the remaining budget.
            // Same decoded-text scan budget_alert uses; counter is
            // incremented per-iteration only when we're still inside.
            if max_think_tokens > 0 {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think {
                    if !prev_in_think {
                        think_count = 1;
                    } else {
                        think_count += 1;
                    }
                } else {
                    think_count = 0;
                }
                prev_in_think = in_think;

                if in_think && think_count >= max_think_tokens {
                    // Force-close. Encode the close sequence and run each
                    // token through the KV write + emit path the same way
                    // a normally-sampled token does. This ensures the
                    // model's next sample is conditioned on having "said"
                    // </think>\n itself, instead of seeing a hidden-state
                    // discontinuity. Respect max_tokens — clip the close
                    // sequence if not enough room remains and bail.
                    let close_tokens = tokenizer.encode("</think>\n");
                    let budget_left = max_tokens.saturating_sub(generated);
                    let take = close_tokens.len().min(budget_left);
                    for &t in &close_tokens[..take] {
                        qwen35::forward_scratch(
                            gpu,
                            weights,
                            config,
                            t,
                            session.seq_pos,
                            kv,
                            dn,
                            scratch,
                        )
                        .unwrap();
                        session.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult {
                                new_physical: new_phys,
                                ..
                            }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                            {
                                session.seq_pos = new_phys;
                            }
                        }
                        session.conversation_tokens.push(t);
                        streamed_tokens.push(t);
                        emit_committed_event(
                            stdout,
                            id,
                            t,
                            streamed_tokens.len() - 1,
                            t0.elapsed().as_millis() as u64,
                        );
                        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes = &all_bytes[bytes_fed_to_filter..];
                        bytes_fed_to_filter = all_bytes.len();
                        let _ = emit_filter_action(stdout, id, filter.observe(new_bytes));
                        generated += 1;
                    }
                    think_count = 0;
                    prev_in_think = false;
                    if generated >= max_tokens {
                        break;
                    }
                }
            }

            // N-gram loop detector: check if any 4-gram in the recent window
            // repeats excessively. When detected, emit an info message and
            // force EOS to prevent wasting the remaining token budget on
            // repetitive output. Logic lives in `hipfire-generate` loop_guard.
            if let Some(StopReason::NgramRepeat { count, .. }) = loop_guard.check(&streamed_tokens)
            {
                let window_len = loop_guard.window_len(streamed_tokens.len());
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
                    id, count, window_len
                );
                let _ = stdout.flush();
                break;
            }

            // Budget-alert injection: once we hit the configured token count,
            // splice the nudge text into the stream. Tokens are emitted to
            // stdout (so the client sees them) AND forward-fed through the KV
            // cache (so the model's next sample is conditioned on having
            // "said" them itself). Injected tokens count against `max_tokens`
            // — we never exceed the caller's requested budget — so we clip
            // the nudge if not enough room remains, and break out of the
            // outer loop if the budget is fully spent after injection.
            if !alert_fired
                && budget_alert_at_tok > 0
                && generated >= budget_alert_at_tok
                && !budget_alert_text.is_empty()
            {
                alert_fired = true;
                // Only inject while the model is inside an open <think> block.
                // The whole point of the feature is to nudge the model's
                // reasoning; firing past </think> just graffities the visible
                // answer with a system-alert string. Check the raw decoded
                // text rather than token IDs since <think> tokenizes as a
                // multi-token sequence in Qwen3.5's vocab.
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let think_open_idx = raw_str.rfind("<think>");
                let think_close_idx = raw_str.rfind("</think>");
                let in_think = match (think_open_idx, think_close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if !in_think {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    // Fall through — resample next token as normal
                    let ngram_scope = &session.conversation_tokens[ngram_scope_start..];
                    let mut blocked: Vec<u32> = Vec::new();
                    collect_unclosed_attractor_blocks(
                        ngram_scope,
                        &attractor_pairs,
                        20,
                        2,
                        &mut blocked,
                    );
                    let cfg = SamplerConfig {
                        temperature: temp,
                        top_p,
                        repeat_penalty,
                        repeat_window: repeat_buf_cap,
                        presence_penalty,
                        frequency_penalty,
                        blocked_tokens: blocked,
                    };
                    next_token = sampler::sample(
                        gpu,
                        &scratch.logits,
                        &scratch.sample_buf,
                        &scratch.repeat_buf,
                        vocab_size,
                        ngram_scope,
                        &cfg,
                        &mut rng_state,
                    );
                    continue;
                }
                let nudge_tokens = tokenizer.encode(budget_alert_text);
                let budget_left = max_tokens.saturating_sub(generated);
                let nudge_len = nudge_tokens.len().min(budget_left);
                // KV headroom check — don't run past physical_cap. If we don't
                // have room for the clipped nudge, skip entirely rather than
                // emit a partial nudge that poisons the trajectory. Under
                // eviction the physical check is trivially satisfied (budget
                // always holds post-evict), but we still respect the check for
                // the non-eviction path.
                let need_kv =
                    session.seq_pos + nudge_len + (max_tokens - generated - nudge_len) + nl.len();
                if nudge_len > 0 && (m.eviction.is_some() || need_kv <= m.physical_cap) {
                    for &tok in &nudge_tokens[..nudge_len] {
                        session.conversation_tokens.push(tok);
                        streamed_tokens.push(tok);
                        emit_committed_event(
                            stdout,
                            id,
                            tok,
                            streamed_tokens.len() - 1,
                            t0.elapsed().as_millis() as u64,
                        );
                        // Emit the injected token's text to stdout so the client
                        // sees it as part of the stream (will be inside <think>
                        // if that's the current state, and get stripped client-
                        // side just like any other think token).
                        let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                        bytes_fed_to_filter = all_bytes2.len();
                        let _ = emit_filter_action(stdout, id, filter.observe(new_bytes2));
                        if let Err(e) = qwen35::forward_scratch(
                            gpu,
                            weights,
                            config,
                            tok,
                            session.seq_pos,
                            kv,
                            dn,
                            scratch,
                        ) {
                            write_error(
                                stdout,
                                id,
                                &format!("qwen35 budget-alert forward_scratch failed: {e:?}"),
                            );
                            qwen35_restore_or_error(stdout, id, m, gpu, session);
                            return;
                        }
                        session.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult {
                                new_physical: new_phys,
                                ..
                            }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                            {
                                session.seq_pos = new_phys;
                            }
                        }
                        generated += 1;
                    }
                } else if nudge_len < nudge_tokens.len() {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#,
                        id, nudge_len, budget_left
                    );
                    let _ = stdout.flush();
                } else {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                }
                // Respect max_tokens: if injection used the remainder, bail
                // before sampling another model token.
                if generated >= max_tokens {
                    break;
                }
            }

            // Decide which paired-opener tokens (if any) trip the depth
            // threshold over a 20-token window. #111 attractor block —
            // cheap when not tripped, ~5 µs per blocked token when
            // tripped (single 4-byte H2D into the logits buffer
            // performed inside sampler::sample).
            let ngram_scope = &session.conversation_tokens[ngram_scope_start..];
            let mut blocked: Vec<u32> = Vec::new();
            collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked);
            let cfg = SamplerConfig {
                temperature: temp,
                top_p,
                repeat_penalty,
                repeat_window: repeat_buf_cap,
                presence_penalty,
                frequency_penalty,
                blocked_tokens: blocked,
            };
            // GPU sample: reads scratch.logits (already on GPU), writes
            // token+rng to scratch.sample_buf. Blocks only on the 8-byte
            // D2H readback inside sampler::sample.
            next_token = sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                vocab_size,
                ngram_scope,
                &cfg,
                &mut rng_state,
            );
        }
        // session.seq_pos is already the "next physical write slot" — advanced
        // per-token in the decode loop above, and evicted back down to
        // `budget` whenever maybe_evict fired. No post-loop fix-up needed.

        // ChatML requires \n after <|im_end|>. Run it through forward so KV cache
        // and DeltaNet state stay in sync with seq_pos.
        if im_end_token == Some(*session.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty()
        {
            for &t in &nl {
                if let Err(e) = qwen35::forward_scratch(
                    gpu,
                    weights,
                    config,
                    t,
                    session.seq_pos,
                    kv,
                    dn,
                    scratch,
                ) {
                    write_error(
                        stdout,
                        id,
                        &format!("qwen35 ChatML newline forward_scratch failed: {e:?}"),
                    );
                    qwen35_restore_or_error(stdout, id, m, gpu, session);
                    return;
                }
                session.seq_pos += 1;
                if let Some(ref ev) = m.eviction {
                    if let Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                    {
                        session.seq_pos = new_phys;
                    }
                }
                session.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 {
            generated as f64 / total_s
        } else {
            0.0
        };
        let prefill_tok_s = if prefill_s > 0.0 {
            prefill_tokens as f64 / prefill_s
        } else {
            0.0
        };
        let decode_tok_s = if decode_s > 0.0 {
            generated as f64 / decode_s
        } else {
            0.0
        };
        if let Some(dir) = evidence_dir {
            write_daemon_runtime_oneshot_evidence(
                dir,
                m,
                gpu,
                id,
                prefill_tokens,
                generated,
                prefill_s,
                decode_s,
                prefill_s * 1000.0,
            );
            if let Some(hist) = moe_router_histogram.take() {
                write_daemon_moe_router_evidence(dir, m, id, hist);
            }
        }
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}{}}}"#,
            id,
            generated,
            tok_s,
            prefill_tokens,
            prefill_s * 1000.0,
            prefill_tok_s,
            decode_tok_s,
            prefill_s * 1000.0,
            pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha),
        );
        let _ = stdout.flush();
        qwen35_restore_or_error(stdout, id, m, gpu, session);
    } else {
        // Qwen3 / LLaMA path -- multi-turn aware
        let chat_template_profile = m.chat_template_profile.clone();
        let config = m.llama_config.as_ref().unwrap();
        let weights = m.llama_weights.as_ref().unwrap();
        let scratch = m.llama_scratch.as_ref().unwrap();
        let kv = m.llama_kv.as_mut().unwrap();

        let mut rng_state = 42u32;
        for (i, &tok) in new_tokens.iter().enumerate() {
            let pos = m.seq_pos + i;
            let (_, rng) = llama::forward_scratch(
                gpu, weights, config, tok, pos, kv, scratch, temp, top_p, rng_state, 0, 1.0,
            )
            .unwrap();
            rng_state = rng;
        }
        let this_turn_prompt_len_llama = new_tokens.len();
        m.seq_pos += new_tokens.len();
        m.conversation_tokens.extend_from_slice(&new_tokens);
        let ngram_scope_start_llama = m.conversation_tokens.len() - this_turn_prompt_len_llama;

        let mut out_bytes = [0u8; 8];
        gpu.hip
            .memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf)
            .unwrap();
        let mut next_token =
            u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
        rng_state = u32::from_ne_bytes([out_bytes[4], out_bytes[5], out_bytes[6], out_bytes[7]]);
        // Prefill ends here: prompt is processed AND first token is ready (D2H
        // sync is the user-observable "time to first token" boundary). Decode
        // below measures the pure forward+sample steady-state.
        let t_prefill = Instant::now();

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        // `bytes_fed_to_filter` is the index into the freshly-decoded
        // byte stream past which we have not yet handed bytes to the
        // filter. The filter owns UTF-8 boundary buffering and any
        // future arch quirks (Gemma 4 marker holdback, strip-think,
        // byte-level stop_at); see crates/engine/src/eos_filter.rs.
        let mut bytes_fed_to_filter = 0usize;
        let mut filter =
            chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);

        for _ in 0..max_tokens {
            generated += 1;
            m.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            emit_committed_event(
                stdout,
                id,
                next_token,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

            // Scope repeat_buf to this turn's prompt + generated tokens
            // (same logic as the Qwen3.5 path: prompt anchor + current turn).
            let rw = repeat_window.min(64);
            let scope_start =
                ngram_scope_start_llama.max(m.conversation_tokens.len().saturating_sub(rw));
            let hist_slice = &m.conversation_tokens[scope_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip
                .memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes)
                .unwrap();

            // Write K/V for this token FIRST so the next turn's context is
            // always fully populated. The sampled next_token from this call
            // is discarded when we break on im_end/eos — wasteful by one
            // launch but avoids a KV cache gap at the terminator.
            let pos = m.seq_pos + generated - 1;
            let (tok, rng) = llama::forward_scratch(
                gpu,
                weights,
                config,
                next_token,
                pos,
                kv,
                scratch,
                temp,
                top_p,
                rng_state,
                hist_slice.len(),
                repeat_penalty,
            )
            .unwrap();

            if next_token == config.eos_token {
                break;
            }
            if filter_stop {
                break;
            }
            if im_end_token == Some(next_token) {
                break;
            }
            if tokenizer.is_terminator(next_token) {
                break;
            }

            next_token = tok;
            rng_state = rng;
        }
        m.seq_pos += generated;

        // ChatML \n boundary — run through forward to keep KV cache in sync
        if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
            for &t in &nl {
                let (_, rng2) = llama::forward_scratch(
                    gpu, weights, config, t, m.seq_pos, kv, scratch, temp, top_p, rng_state, 0, 1.0,
                )
                .unwrap();
                rng_state = rng2;
                m.seq_pos += 1;
                m.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 {
            generated as f64 / total_s
        } else {
            0.0
        };
        let prefill_tok_s = if prefill_s > 0.0 {
            prefill_tokens as f64 / prefill_s
        } else {
            0.0
        };
        let decode_tok_s = if decode_s > 0.0 {
            generated as f64 / decode_s
        } else {
            0.0
        };
        if let Some(dir) = evidence_dir {
            write_daemon_runtime_oneshot_evidence(
                dir,
                m,
                gpu,
                id,
                prefill_tokens,
                generated,
                prefill_s,
                decode_s,
                prefill_s * 1000.0,
            );
        }
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}{}}}"#,
            id,
            generated,
            tok_s,
            prefill_tokens,
            prefill_s * 1000.0,
            prefill_tok_s,
            decode_tok_s,
            prefill_s * 1000.0,
            pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha),
        );
        let _ = stdout.flush();
    }
}

/// DeepSeek V4 Flash generate path (arch_id=9, hipfire-arch-deepseek4).
///
/// Parity with `deepseek4_chat`: batched chunked prefill +
/// optional MTP spec-decode + greedy argmax sampler. PBS is pre-allocated
/// once at load time (`m.deepseek4_pbs`), reused across every turn.
///
/// Env knobs (read fresh per generate call so they can be toggled
/// without daemon restart):
///   HIPFIRE_DEEPSEEK4_SPEC_DECODE=1     opt-in MTP speculative decode
///   HIPFIRE_DEEPSEEK4_SPEC_K=N          drafts per spec-decode window (default 3)
///   HIPFIRE_DEEPSEEK4_TOP_K=N           top-k filter (default 0 = off; HF rec)
///   HIPFIRE_DEEPSEEK4_SEED=N            PRNG seed (default: time-based)
///
/// Sampling defaults follow the HF model card for `deepseek-ai/DeepSeek-V4-Flash`:
/// `temperature = 1.0, top_p = 1.0`. Pure greedy (`temp ≤ 1e-6`) is
/// supported but actively dangerous on this quantized instruct model —
/// once a code fence opens, `import X\n` self-reinforces into a block-
/// level token loop. Use `temp = 1.0` (HF default) to avoid the attractor.
///
/// Chat template (per HF `encoding/README.md` for V4): non-thinking-mode
/// frame `<｜begin▁of▁sentence｜>{system?}<｜User｜>{msg}<｜Assistant｜></think>`.
/// The model expects the `</think>` immediately after `<｜Assistant｜>` in
/// non-thinking mode, even though no thinking block was generated — this
/// signals "skip reasoning, go straight to response." Omitting it leaves
/// the model in undefined-behavior territory.
///
/// Deliberately bypasses qwen35/llama machinery — no PFlash, no DFlash,
/// no CASK eviction, no ChatML scaffolding, no tool-use, no `<think>` /
/// `max_think_tokens`, no repeat penalty, no VL, no multi-GPU
/// pipeline-parallel.
///
/// On context overflow the DeepSeek V4 state is hard-reset — DeepSeek V4 has no
/// eviction path of its own and the SWA cache wraps automatically below
/// the sliding-window bound.
/// HuggingFace DeepSeek V4 thinking modes (per `encoding/README.md`).
///
/// The chat template choice changes the open-token after `<｜Assistant｜>`
/// and (for `Max`) prepends an extended reasoning instruction.
#[derive(Copy, Clone, Debug)]
pub enum ThinkMode {
    /// Non-thinking. Frame: `<｜Assistant｜></think>{response}`.
    /// Model skips reasoning, replies directly. HF default for chat.
    NonThink,
    /// Thinking-high. Frame: `<｜Assistant｜><think>{reasoning}</think>{response}`.
    /// Model produces a `<think>` block before responding.
    High,
    /// Thinking-max. Same frame as `High`, plus prepended
    /// "Reasoning Effort: Absolute maximum..." system instruction.
    /// HF recommends context ≥ 384K for this mode.
    Max,
}

impl ThinkMode {
    /// Map a JSONL field value (OpenAI-compatible `reasoning_effort` or
    /// project-custom `thinking_mode`) to a mode.
    /// Accepted: "none|off|chat|minimal" → NonThink;
    ///           "low|medium|high|thinking" → High;
    ///           "max" → Max. Anything else → NonThink (safe default).
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "max" => Self::Max,
            "high" | "thinking" | "low" | "medium" => Self::High,
            _ => Self::NonThink,
        }
    }
}

fn generate_deepseek4(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
) {
    let tokenizer = match m.tokenizer.as_ref() {
        Some(t) => t,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        }
    };
    let cfg = match m.deepseek4_config.as_ref() {
        Some(c) => c,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"deepseek4_config missing on arch_id=9 generate"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        }
    };
    let weights = m
        .deepseek4_weights
        .as_ref()
        .expect("deepseek4_weights missing on arch_id=9 generate");
    let pbs = m
        .deepseek4_pbs
        .as_ref()
        .expect("deepseek4_pbs missing on arch_id=9 generate");
    let state = m
        .deepseek4_state
        .as_mut()
        .expect("deepseek4_state missing on arch_id=9 generate");
    let eos_tok = m.deepseek4_eos_tok;

    // DeepSeek V4 non-thinking chat template (per HF encoding/README.md):
    //   <｜begin▁of▁sentence｜>{system?}<｜User｜>{msg}<｜Assistant｜></think>
    //
    // The `</think>` immediately after `<｜Assistant｜>` is REQUIRED in
    // non-thinking mode — it tells the model "skip the reasoning block,
    // go straight to the response." Without it the model is in
    // undefined-behavior territory. Raw prompts (no chat-template wrap)
    // also collapse to attractor garbage on this quantized instruct
    // model. Multi-turn / thinking-mode plumbing is a follow-up; this
    // emits a single non-thinking turn per /generate call.
    let lookup = |s: &str| -> Option<u32> {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            Some(ids[0])
        } else {
            None
        }
    };
    let bos_tok = lookup("<｜begin▁of▁sentence｜>");
    let user_tok = lookup("<｜User｜>");
    let asst_tok = lookup("<｜Assistant｜>");

    // HF "Reasoning Effort: Absolute maximum..." preamble for `Max` mode.
    // Quoted from the model card's encoding/README.md.
    const MAX_THINK_PREAMBLE: &str =
        "Reasoning Effort: Absolute maximum with no shortcuts permitted. \
You MUST be very thorough in your thinking and comprehensively decompose the problem.";

    // Build the effective system message: optional user-supplied system
    // text + (if request has tools) the DSML "## Tools" preamble.
    //
    // HF reference render: the system role is rendered as `{content}`
    // (raw, no role prefix), then appended with `"\n\n" + render_tools`
    // when tools are present. For an empty system + tools this becomes
    // `"" + "\n\n" + tools_block` = `"\n\n" + tools_block` — the model
    // was trained to see two newlines BEFORE `## Tools` even with no
    // system content. Omitting them puts the model in off-distribution
    // territory; observed 2026-05-23 to drive the V4F MQ2-Lloyd
    // checkpoint into `<｜DSML｜tool_cin> / <｜DSML｜-cin>` attractor
    // loops on no-system + 4-tools requests. The leading `\n\n` is
    // load-bearing — do not drop.
    let tools_block: Option<String> = tools
        .filter(|t| !t.is_empty())
        .map(|t| deepseek4::dsml::tools_prompt_block(t));
    let effective_system: Option<String> = match (
        system_prompt.filter(|s| !s.is_empty()),
        tools_block.as_deref(),
    ) {
        (Some(sys), Some(tb)) => Some(format!("{sys}\n\n{tb}")),
        (Some(sys), None) => Some(sys.to_string()),
        (None, Some(tb)) => Some(format!("\n\n{tb}")),
        (None, None) => None,
    };

    let mut prompt_ids: Vec<u32> = Vec::new();
    if let Some(b) = bos_tok {
        prompt_ids.push(b);
    }
    if matches!(think_mode, ThinkMode::Max) {
        prompt_ids.extend(tokenizer.encode(MAX_THINK_PREAMBLE));
    }
    if let Some(ref sys) = effective_system {
        prompt_ids.extend(tokenizer.encode(sys));
    }

    // Multi-turn history. Each prior message gets rendered as a turn:
    //   user → `<｜User｜>{content}{tool_results?}`
    //   assistant → `<｜Assistant｜>{content_or_dsml}<｜end▁of▁sentence｜>`
    // Tool result messages (role=tool) attach to the previous user turn
    // wrapped in `<tool_result>…</tool_result>` per HF encoding/README.md.
    // The CURRENT user prompt is appended last (outside this loop).
    if let Some(history) = messages_history {
        // Skip the leading system message (if any) — already handled.
        // Skip the trailing user prompt — we add it explicitly after.
        // Heuristic: if last message is role=user, treat its content as
        // the live prompt and drop it here.
        use prompt_frame::Role;
        let trim_end = if matches!(history.last().map(|m| m.role), Some(Role::User)) {
            1
        } else {
            0
        };
        let end = history.len().saturating_sub(trim_end);
        // Track whether the previous emission was already a tool_result
        // wrapped in a user turn — when YES, the next consecutive tool
        // message MUST NOT open a new `<｜User｜>` marker; instead it
        // stacks its `<tool_result>` body into the existing user turn.
        // Matches the reference imatrix dataset renderer in
        // `gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py:196-201`
        // — OpenAI's parallel-tool-call flow produces consecutive tool
        // messages (one per parallel call), and a fresh `<｜User｜>`
        // between them isn't what V4F was trained on.
        let mut pending_tool_result = false;
        for msg in &history[..end] {
            match msg.role {
                Role::System => {
                    // Already handled via effective_system; skip.
                }
                Role::User => {
                    if let Some(u) = user_tok {
                        prompt_ids.push(u);
                    }
                    prompt_ids.extend(tokenizer.encode(&msg.content));
                    pending_tool_result = false;
                }
                Role::Tool => {
                    // Wrap as `<tool_result>{escaped}</tool_result>`. Open
                    // a new user turn ONLY if the prior message wasn't
                    // already a tool_result.
                    if !pending_tool_result {
                        if let Some(u) = user_tok {
                            prompt_ids.push(u);
                        }
                    }
                    prompt_ids.extend(
                        tokenizer.encode(&deepseek4::dsml::render_tool_result(&msg.content)),
                    );
                    pending_tool_result = true;
                }
                Role::Assistant => {
                    // Daemon-emitted surround tokens that bracket every
                    // assistant turn in V4F format:
                    //   <｜Assistant｜>{</think> when not in think-replay}
                    //     {turn body — content + tool_calls}
                    //   <｜end▁of▁sentence｜>
                    //
                    // The cache stores ONLY the inner turn body (the
                    // tokens the model itself emitted during decode).
                    // The surround tokens are deterministic functions
                    // of `msg.content` and `think_mode` and must be
                    // emitted IDENTICALLY on both hit and miss paths so
                    // the prompt-cache LCP can extend through every
                    // prior assistant turn.
                    if let Some(a) = asst_tok {
                        prompt_ids.push(a);
                    }
                    let starts_with_think_tag =
                        msg.content.starts_with("<think>") || msg.content.starts_with("</think>");
                    if !starts_with_think_tag {
                        prompt_ids.extend(tokenizer.encode("</think>"));
                    }

                    // Prefix-cache fast path: if we previously emitted
                    // this exact assistant turn, replay the model's
                    // verbatim token sequence instead of re-rendering
                    // via DSML + BPE encode (which is not bijective —
                    // multi-char DSML special tokens picked greedily
                    // during decode can come back out of
                    // `tokenizer.encode(render(...))` as a longer
                    // sequence with different boundaries, capping the
                    // LCP at the assistant-turn boundary).
                    let fp =
                        prompt_frame::assistant_turn_fingerprint(&msg.content, &msg.tool_calls);
                    if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        eprintln!(
                            "[asst-cache lookup] fp={:#018x} content.len={} tool_calls={} hit={}",
                            fp,
                            msg.content.len(),
                            msg.tool_calls.len(),
                            m.asst_turn_cache.contains_key(&fp),
                        );
                    }
                    if let Some(cached) = m.asst_turn_cache.get(&fp) {
                        prompt_ids.extend_from_slice(cached);
                    } else {
                        // Cache miss — render the turn the long way.
                        if !msg.content.is_empty() && msg.content != "null" {
                            prompt_ids.extend(tokenizer.encode(&msg.content));
                        }
                        if !msg.tool_calls.is_empty() {
                            let dsml_calls: Vec<hipfire_arch_deepseek4::dsml::ToolCall> = msg
                                .tool_calls
                                .iter()
                                .map(|c| hipfire_arch_deepseek4::dsml::ToolCall {
                                    name: c.name.clone(),
                                    arguments: c.arguments.clone(),
                                })
                                .collect();
                            let dsml = hipfire_arch_deepseek4::dsml::render_assistant_tool_calls(
                                &dsml_calls,
                            );
                            prompt_ids.extend(tokenizer.encode(&dsml));
                        }
                    }

                    // Close the assistant turn with the EOS marker so
                    // the next turn starts cleanly.
                    prompt_ids.push(m.deepseek4_eos_tok);
                    pending_tool_result = false;
                }
            }
        }
    }

    // Append the live user turn ONLY when `prompt` carries one. When the
    // serve has handed us a structured `messages` history that already
    // ends in a tool result (mid-conversation, model is meant to continue
    // generating the next assistant turn) it sends `prompt=""` — in that
    // case we MUST NOT emit an empty `<｜User｜><｜Assistant｜>` wrapper,
    // because the empty-user turn is off-distribution and the V4F MQ2-
    // Lloyd checkpoint drifts into invented paths / repeated wrong tool
    // calls when fed one.
    if !prompt.is_empty() {
        if let Some(u) = user_tok {
            prompt_ids.push(u);
        }
        prompt_ids.extend(tokenizer.encode(prompt));
    }
    if let Some(a) = asst_tok {
        prompt_ids.push(a);
    }
    // Thinking-mode signal token immediately after `<｜Assistant｜>`:
    //   NonThink → `</think>`   (skip reasoning, respond directly)
    //   High|Max → `<think>`    (open a reasoning block)
    match think_mode {
        ThinkMode::NonThink => prompt_ids.extend(tokenizer.encode("</think>")),
        ThinkMode::High | ThinkMode::Max => prompt_ids.extend(tokenizer.encode("<think>")),
    }

    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    if std::env::var("HIPFIRE_DEEPSEEK4_DUMP_PROMPT")
        .ok()
        .as_deref()
        == Some("1")
    {
        let rendered = tokenizer.decode(&prompt_ids);
        let path = format!(
            "/tmp/hipfire-prompt-{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );
        let _ = std::fs::write(
            &path,
            format!("# tokens: {}\n{}\n", prompt_ids.len(), rendered),
        );
        eprintln!("[v4f prompt dump] tokens={} → {}", prompt_ids.len(), path);
    }

    // Triaged config resolution for MTP speculative decode.
    // Priority: 1. legacy env var → 2. generic env var → 3. stored config → default.
    let spec_mode = std::env::var("HIPFIRE_DEEPSEEK4_SPEC_DECODE")
        .ok()
        .map(|v| v == "1")
        .unwrap_or_else(|| match std::env::var("HIPFIRE_MTP_MODE").ok().as_deref() {
            Some("on") => true,
            Some("off") => false,
            _ => m.mtp_mode == "on" || (m.mtp_mode == "auto" && m.mtp_weights_present),
        });
    let spec_k: usize = std::env::var("HIPFIRE_DEEPSEEK4_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            std::env::var("HIPFIRE_MTP_K")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(m.mtp_k);

    let t0 = Instant::now();

    // ── Prefix-cache LCP detection ──────────────────────────────────
    //
    // Reasonix's prompt-caching model (`tmp/reasonix_arch.md` Pillar 1):
    // construct prompts as `immutable_prefix + append_only_log` so the
    // backend's prefix cache hits on every turn. Reasonix is a CLIENT
    // that targets DeepSeek's server-side cache; for LOCAL inference we
    // implement the server side here.
    //
    // Compare the freshly-tokenized prompt against the tokens we know
    // are already resident in the V4F KV / SWA / compressed-KV rings
    // from the prior request (`m.conversation_tokens`). If the new
    // prompt FULLY EXTENDS the prior conversation — i.e., starts with
    // the entire `conversation_tokens` — we can skip prefill for those
    // tokens and only prefill the suffix.
    //
    // SWA-safety analysis for partial LCP (lcp < prior.len()):
    //
    // Suppose prior wrote positions [0..prior_max_pos], turn 2's suffix
    // writes [lcp..prompt_ids.len()-1]. After turn 2's prefill the new
    // max position is `prompt_ids.len() - 1`. The model's first decode
    // attends to a window of `min(prompt_ids.len(), 128)` positions
    // ending at `prompt_ids.len() - 1`. Each window position maps to a
    // unique ring slot via `pos % 128`. For correctness, every slot in
    // that window must currently hold K_rotated for the matching
    // position:
    //
    //   * For positions in `[0..lcp-1]` — turn 1 wrote them, content
    //     matches by LCP definition. Untouched since.
    //   * For positions in `[lcp..prompt_ids.len()-1]` — turn 2's suffix
    //     prefill just wrote them. Content matches the new prompt.
    //
    // Stale-slot risk: if turn 1 had written a slot at some position
    // `P_late ∈ [lcp..prior_max_pos]` AND turn 2 doesn't overwrite that
    // slot, the slot holds K_rotated for P_late, not the new prompt's
    // token at that position. The window read returns wrong content.
    //
    // Turn 2's suffix prefill covers positions [lcp..prompt_ids.len()-1].
    // To overwrite every slot turn 1 wrote in `[lcp..prior_max_pos]`,
    // we need `prompt_ids.len() - 1 ≥ prior_max_pos`, i.e.
    // `prompt_ids.len() ≥ prior.len()`. Equivalently: the new prompt
    // must be at least as long as the cached conversation.
    //
    // We additionally guard `lcp == prior.len() && prompt_ids.len() ==
    // prior.len()` (full match, nothing to do) with a noop check
    // downstream (suffix_tokens is empty).
    //
    // After the daemon's `reset` handler clears `m.conversation_tokens`
    // (legacy stateless path), `prior` is empty and `lcp = 0` → full
    // prefill. For prefix-cache mode the serve stops calling reset for
    // V4F and lets this LCP detection drive cache-hit accounting.
    let lcp: usize = {
        let prior = &m.conversation_tokens;
        if prior.is_empty() || prompt_ids.len() < prior.len() {
            0
        } else {
            let mut n = 0usize;
            while n < prior.len() && prior[n] == prompt_ids[n] {
                n += 1;
            }
            // Edge case: new prompt is byte-identical to the cached
            // conversation. Suffix would be empty and
            // `forward_prefill_batch_chunked` errors on that. Step the
            // LCP back one so we always prefill ≥ 1 token (and the
            // post-prefill logits are well-defined for the first
            // decode step). Costs us one token of cache credit on
            // exact-repeat prompts — rare in practice.
            if n == prompt_ids.len() && n > 0 {
                n - 1
            } else {
                n
            }
        }
    };

    if lcp == 0 {
        // Cache miss — start a fresh conversation in V4F's state.
        state.reset();
        m.conversation_tokens.clear();
        // Tear down the captured V4F decode hipGraph alongside the
        // state, same rationale as the daemon's `"reset"` handler:
        // a fresh-context turn invalidates every device-buffer pointer
        // and host scalar the captured graph baked in at capture time
        // (state.attn_state_buf slot/n_valid/k_active values derived
        // from the prior n_tokens, compressor ring/commit slots, etc.).
        // Without this, the warmup-then-replay state machine fires
        // warmup on the first decode (because `state.reset()` clears
        // `ar_forward_warmed_up`), then immediately replays the STALE
        // graph on the second decode and crashes with the same
        // "download logits (graph path): illegal memory access" we
        // saw on multi-turn pi sessions before the explicit-reset fix.
        gpu.invalidate_graph_state();
    }
    let start_pos: u32 = lcp as u32;

    // Slice off the suffix — the only tokens we actually need to prefill.
    // For lcp=0 this is the full prompt; for a full cache hit on a turn
    // that adds N new tokens this is just those N.
    let suffix_tokens: &[u32] = &prompt_ids[lcp..];

    // Prefill: batched chunked through PBS. If spec_mode, also fill the
    // MTP layer's SWA cache (prefill_with_mtp_fill) so the first
    // draft step sees a populated MTP history.
    let prefill_result = if spec_mode {
        deepseek4::forward::prefill_with_mtp_fill(
            cfg,
            weights,
            state,
            gpu,
            pbs,
            suffix_tokens,
            start_pos,
        )
    } else {
        deepseek4::forward::forward_prefill_batch_chunked(
            cfg,
            weights,
            state,
            gpu,
            suffix_tokens,
            start_pos,
            pbs,
        )
    };
    let last_logits = match prefill_result {
        Ok(l) => l,
        Err(e) => {
            emit_error_with_id(stdout, id, format!("deepseek4prefill failed: {e:?}"));
            return;
        }
    };
    // `forward_prefill_batch_chunked` does NOT advance `state.n_tokens`.
    // Callers are responsible for it (mirrors deepseek4_chat's explicit
    // `state.n_tokens = pos as u64;` at deepseek4_chat.rs:324). Without this,
    // the next decode_step queries the SWA cache at the BOS position
    // instead of the next-prediction position and the model emits
    // attractor garbage at greedy temp=0. The MTP-fill prefill DOES
    // advance internally (forward.rs:7453), so we only need to update
    // for the plain-prefill branch.
    if !spec_mode {
        state.n_tokens = (start_pos as usize + suffix_tokens.len()) as u64;
    }
    // Keep `m.conversation_tokens` in lockstep with what's actually
    // resident in the KV/SWA/compressed-KV rings:
    //   - On a CACHE MISS (lcp==0): replace with prompt_ids (we just
    //     full-prefilled the whole prompt).
    //   - On a CACHE HIT (lcp>0): truncate the prior tracker down to
    //     `lcp` before appending the suffix. For partial LCP this
    //     matters — tokens in the prior tracker beyond `lcp` came
    //     from a previous turn's decode but the slots they lived in
    //     have just been overwritten by the suffix prefill. Leaving
    //     them in the tracker would let the NEXT request's LCP
    //     comparison run off the end of what's actually cached and
    //     make divergent assumptions about ring contents.
    if lcp == 0 {
        m.conversation_tokens.clear();
        m.conversation_tokens.extend_from_slice(&prompt_ids);
    } else {
        m.conversation_tokens.truncate(lcp);
        m.conversation_tokens.extend_from_slice(suffix_tokens);
    }
    let cached_tokens: usize = lcp;

    // Sync to ensure all prefill kernels have completed before stopping
    // the timer (head's download_f32 already syncs but defensive).
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = t0.elapsed().as_millis();

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    let pos_after_prefill = state.n_tokens as u32;
    let mut spec_windows: u64 = 0;
    let mut spec_drafts_offered: u64 = 0;
    let mut spec_drafts_accepted: u64 = 0;

    // Sampler. HF DeepSeek-V4-Flash card recommends temp=1.0, top_p=1.0
    // for local deployment; we honor that as the default. Pure greedy
    // (temp <= 1e-6) is supported but enters block-level attractors on
    // structured prompts.
    let top_k: usize = std::env::var("HIPFIRE_DEEPSEEK4_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let seed: u64 = std::env::var("HIPFIRE_DEEPSEEK4_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E3779B97F4A7C15)
        });
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    // Track whether the decode loop saw a complete
    // `<｜DSML｜tool_calls>` block close. Drives `finish_reason` in the
    // `done` envelope below.
    let mut tool_calls_parsed_count: usize = 0;
    if spec_mode {
        // Spec-decode loop. The verifier picks argmax (greedy) so accept
        // semantics stay deterministic. When tools are present, thread
        // the same DSML grammar matcher through the MTP draft and main
        // verifier logits, then parse the emitted stream into tool_calls
        // events just like the plain decode loop.
        let tool_schemas: Vec<deepseek4::grammar::ToolSchema> = tools
            .map(|arr| {
                arr.iter()
                    .map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let parameters = func.get("parameters");
                        let params: Vec<String> = parameters
                            .and_then(|p| p.get("properties"))
                            .and_then(|p| p.as_object())
                            .map(|m| m.keys().cloned().collect())
                            .unwrap_or_default();
                        let required: Vec<String> = parameters
                            .and_then(|p| p.get("required"))
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        deepseek4::grammar::ToolSchema {
                            name,
                            params,
                            required,
                        }
                    })
                    .filter(|s: &deepseek4::grammar::ToolSchema| !s.name.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let grammar_active = !tool_schemas.is_empty();
        let mut parser = match think_mode {
            ThinkMode::High | ThinkMode::Max => deepseek4::dsml::StreamParser::new_in_think(),
            ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
        };
        let mut matcher = deepseek4::grammar::Matcher::new(tool_schemas);
        let decoded_vocab_arc: Option<std::sync::Arc<Vec<String>>> = if grammar_active {
            if m.decoded_vocab.is_none() {
                let n = tokenizer.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tokenizer.decode(&[id as u32])).collect();
                m.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.decoded_vocab.clone()
        } else {
            None
        };
        let empty_vocab: Vec<String> = Vec::new();
        let decoded_vocab: &[String] = decoded_vocab_arc
            .as_deref()
            .map(|v| v.as_slice())
            .unwrap_or(&empty_vocab);
        let mut grammar_mask: Vec<bool> = vec![true; decoded_vocab.len()];
        let mut emit_tool_calls_buf: Vec<prompt_frame::ToolCall> = Vec::new();
        use hipfire_arch_deepseek4::dsml::StreamEvent;
        let mut absorb_event = |ev: &StreamEvent| {
            if let StreamEvent::ToolCalls(calls) = ev {
                for c in calls {
                    emit_tool_calls_buf.push(prompt_frame::ToolCall {
                        name: c.name.clone(),
                        arguments: c.arguments.clone(),
                    });
                }
            }
        };

        let mut spec_last_token = deepseek4::spec_decode::logits_argmax(&last_logits) as u32;
        let mut spec_last_position = pos_after_prefill;
        let mut last_hidden_ref = state.mtp_last_hidden.as_ref().map(|t| t as *const _);
        'outer: while generated_count < max_tokens {
            let lh: Option<&rdna_compute::GpuTensor> = unsafe {
                last_hidden_ref.and_then(|p| (p as *const rdna_compute::GpuTensor).as_ref())
            };
            let r = match if grammar_active {
                deepseek4::spec_decode::speculative_decode_step_with_pbs_grammar(
                    cfg,
                    weights,
                    state,
                    gpu,
                    pbs,
                    spec_last_token,
                    spec_last_position,
                    lh,
                    spec_k,
                    &mut matcher,
                    decoded_vocab,
                    &mut grammar_mask,
                )
            } else {
                deepseek4::spec_decode::speculative_decode_step_with_pbs(
                    cfg,
                    weights,
                    state,
                    gpu,
                    pbs,
                    spec_last_token,
                    spec_last_position,
                    lh,
                    spec_k,
                )
            } {
                Ok(r) => r,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("deepseek4spec-decode failed: {e:?}"));
                    let _ = stdout.flush();
                    return;
                }
            };
            spec_windows += 1;
            spec_drafts_offered += spec_k as u64;
            spec_drafts_accepted += r.n_accepted as u64;

            for &t in &r.accepted_tokens {
                if generated_count >= max_tokens || t == eos_tok {
                    break 'outer;
                }
                let frag = tokenizer.decode(&[t]);
                if grammar_active {
                    for ev in parser.feed(&frag) {
                        absorb_event(&ev);
                        emit_stream_event(stdout, id, ev);
                    }
                } else {
                    // Build through serde_json so `id` (user-supplied) and
                    // `frag` (model-generated UTF-8 with possible `"`/`\`)
                    // can't corrupt the JSONL line.
                    let envelope = serde_json::json!({
                        "type": "token",
                        "id": id,
                        "text": frag,
                    });
                    let _ = writeln!(stdout, "{}", envelope);
                }
                emit_committed_event(
                    stdout,
                    id,
                    t,
                    generated_count,
                    decode_t0.elapsed().as_millis() as u64,
                );
                let _ = stdout.flush();
                m.conversation_tokens.push(t);
                generated_count += 1;
            }
            if let Some(&t) = r.accepted_tokens.last() {
                spec_last_position += r.accepted_tokens.len() as u32;
                spec_last_token = t;
            }
            last_hidden_ref = state.mtp_last_hidden.as_ref().map(|t| t as *const _);
        }
        if grammar_active {
            for ev in parser.finish() {
                absorb_event(&ev);
                emit_stream_event(stdout, id, ev);
            }
            let _ = stdout.flush();
            drop(absorb_event);
            tool_calls_parsed_count = emit_tool_calls_buf.len();
        }
    } else {
        // Plain decode loop. Sampler honours `temp` + `top_p` from the
        // request; HF default is temp=1.0, top_p=1.0 (multinomial across
        // the full vocab, no nucleus cut). Greedy (temp <= 1e-6) is
        // dangerous — see fn doc.
        //
        // Tokens are fed through a DSML stream parser that recognises
        // `<think>…</think>` reasoning blocks and
        // `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` tool-call blocks. The
        // parser emits:
        //   - StreamEvent::Token(text)       → JSONL `{type:"token"}`
        //   - StreamEvent::Reasoning(text)   → JSONL `{type:"reasoning"}`
        //   - StreamEvent::ToolCalls(calls)  → JSONL `{type:"tool_calls"}`
        // Markers split across token boundaries are buffered until they
        // resolve. The CLI / HTTP layer maps these to OpenAI SSE chunks.
        // Prime the parser's initial state to match the bootstrap tag
        // we appended to `prompt_ids`. In High/Max think modes the
        // prompt ends with `<think>` and the model's first generated
        // token is the body of that thinking block — without
        // `new_in_think()` the parser would sit in `Normal` and
        // misclassify every reasoning token as plain content,
        // including the trailing `</think>` which then leaks into
        // `message.content`. NonThink mode appends `</think>` (closing
        // a zero-length think block) so the response starts in Normal.
        let mut parser = match think_mode {
            ThinkMode::High | ThinkMode::Max => deepseek4::dsml::StreamParser::new_in_think(),
            ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
        };

        // Grammar-guided decoding setup. When tools are present, we mask
        // the logits against a small state machine that mirrors the DSML
        // format — inside a `<｜DSML｜tool_calls>` block the model can
        // only emit token IDs whose decoded text is a prefix of a legal
        // continuation (e.g. `<｜DSML｜invoke name="` or a schema-defined
        // tool name). In free-emission states (`Out`, `InParamBody`,
        // and any time tools is None / empty) the mask is all-true and
        // the mask compute is skipped.
        //
        // Why this exists: V4F MQ2-Lloyd has damaged logit precision on
        // format-structural tokens — even with the byte-identical HF
        // system prompt at temp=1.0 it deterministically emits invented
        // variants like `<｜DSML｜tool_cbl>`, `<｜DSML｜calling>`,
        // `</｜DSML｜paper>` that no parser can recover. The mask makes
        // those tokens unreachable at the sampler level.
        let tool_schemas: Vec<deepseek4::grammar::ToolSchema> = tools
            .map(|arr| {
                arr.iter()
                    .map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let parameters = func.get("parameters");
                        let params: Vec<String> = parameters
                            .and_then(|p| p.get("properties"))
                            .and_then(|p| p.as_object())
                            .map(|m| m.keys().cloned().collect())
                            .unwrap_or_default();
                        let required: Vec<String> = parameters
                            .and_then(|p| p.get("required"))
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        deepseek4::grammar::ToolSchema {
                            name,
                            params,
                            required,
                        }
                    })
                    .filter(|s: &deepseek4::grammar::ToolSchema| !s.name.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let grammar_active = !tool_schemas.is_empty();
        let mut matcher = deepseek4::grammar::Matcher::new(tool_schemas);
        // Precompute (or fetch the cached) decoded vocab. `tokenizer.decode`
        // per id over ~129k ids is allocator-heavy enough that doing it
        // per-request adds tens of ms of pure overhead to every tool-
        // using V4F turn. The cache lives on `LoadedModel.decoded_vocab`
        // as an `Arc<Vec<String>>` and is cleared on model unload.
        //
        // Borrow note: `m.decoded_vocab` is a disjoint field from
        // `m.deepseek4_state` (which `state` holds `&mut` to) and from
        // `m.tokenizer` (which `tokenizer` holds `&` to), so the
        // assignment compiles under Rust's split-borrows.
        let decoded_vocab_arc: Option<std::sync::Arc<Vec<String>>> = if grammar_active {
            if m.decoded_vocab.is_none() {
                let n = tokenizer.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tokenizer.decode(&[id as u32])).collect();
                m.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.decoded_vocab.clone()
        } else {
            None
        };
        let empty_vocab: Vec<String> = Vec::new();
        let decoded_vocab: &[String] = decoded_vocab_arc
            .as_deref()
            .map(|v| v.as_slice())
            .unwrap_or(&empty_vocab);
        let mut grammar_mask: Vec<bool> = vec![true; decoded_vocab.len()];

        // Apply mask to the prefill-returned logits before the first
        // sample (matcher is in `Out` here so this is a no-op, but the
        // codepath stays uniform).
        let mut first_logits = last_logits;
        if grammar_active && !matcher.is_free() {
            matcher.token_mask(&decoded_vocab, &mut grammar_mask);
            deepseek4::grammar::Matcher::apply_mask_to_logits(&grammar_mask, &mut first_logits);
        }
        let mut next_tok: u32 =
            deepseek4::sampling::sample_token(&first_logits, temp, top_k, top_p, &mut rng);
        let mut pos = pos_after_prefill;
        // Token-cache capture for the prefix-cache replay path. We
        // mirror the parser events into local accumulators so that —
        // after decode completes — we can fingerprint the just-emitted
        // assistant turn by (content_text, tool_calls) and store the
        // exact token IDs that the model emitted at
        // `conversation_tokens[decode_start..decode_end]`.
        //
        // Why mirror rather than re-parse: the streamed events from
        // `parser.feed` carry the parser's reconstructed structure
        // (Reasoning fragments split off from Token, ToolCalls
        // assembled from `<｜DSML｜tool_calls>` blocks). Replaying that
        // here once captures all the logical structure without a
        // second tokenizer pass.
        let decode_start_tokens_idx = m.conversation_tokens.len();
        let mut emit_text_buf = String::new();
        let mut emit_tool_calls_buf: Vec<prompt_frame::ToolCall> = Vec::new();
        use hipfire_arch_deepseek4::dsml::StreamEvent;
        let mut absorb_event = |ev: &StreamEvent| {
            match ev {
                StreamEvent::Token(t) => emit_text_buf.push_str(t),
                // Reasoning fragments are NOT replayed in the next
                // turn (the daemon's history loop emits a fresh
                // `</think>` after `<｜Assistant｜>` based on the
                // current `think_mode`; the prior `<think>…</think>`
                // body is dropped). So we don't include reasoning in
                // the fingerprint either — two turns that produced
                // the same content + tool_calls but different
                // reasoning hash to the same key and reuse the same
                // cached tokens, which is correct because what we
                // CACHE excludes the reasoning span (it lives BEFORE
                // the daemon-emitted `</think>` in the cached tokens
                // — see below).
                StreamEvent::Reasoning(_) => {}
                StreamEvent::ToolCalls(calls) => {
                    for c in calls {
                        emit_tool_calls_buf.push(prompt_frame::ToolCall {
                            name: c.name.clone(),
                            arguments: c.arguments.clone(),
                        });
                    }
                }
            }
        };

        while generated_count < max_tokens && next_tok != eos_tok {
            let frag = tokenizer.decode(&[next_tok]);
            for ev in parser.feed(&frag) {
                absorb_event(&ev);
                emit_stream_event(stdout, id, ev);
            }
            emit_committed_event(
                stdout,
                id,
                next_tok,
                generated_count,
                decode_t0.elapsed().as_millis() as u64,
            );
            let _ = stdout.flush();
            m.conversation_tokens.push(next_tok);
            if grammar_active {
                matcher.advance(&frag);
            }
            generated_count += 1;
            match deepseek4::forward::decode_step_with_graph(
                cfg, weights, state, gpu, next_tok, pos,
            ) {
                Ok(mut logits) => {
                    if grammar_active && !matcher.is_free() {
                        matcher.token_mask(&decoded_vocab, &mut grammar_mask);
                        deepseek4::grammar::Matcher::apply_mask_to_logits(
                            &grammar_mask,
                            &mut logits,
                        );
                    }
                    next_tok =
                        deepseek4::sampling::sample_token(&logits, temp, top_k, top_p, &mut rng);
                    pos += 1;
                }
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("deepseek4decode failed: {e:?}"));
                    let _ = stdout.flush();
                    return;
                }
            }
        }
        // Flush any buffered partial markers / content.
        for ev in parser.finish() {
            absorb_event(&ev);
            emit_stream_event(stdout, id, ev);
        }
        let _ = stdout.flush();

        // Cache the just-emitted token sequence under its (content,
        // tool_calls) fingerprint so the next request's V4F history
        // render can replay verbatim and avoid BPE re-encode drift.
        // Trim leading EOS/zero residue defensively (the loop never
        // pushes EOS, but a future model that emits EOS mid-stream
        // shouldn't end up with EOS landing in the cached tokens).
        drop(absorb_event); // release the &mut emit_*_buf borrow
                            // Now that the closure is dropped, we can read the buffers
                            // immutably. Snapshot the tool_calls count so the `done`
                            // envelope below can carry `finish_reason: "tool_calls"`.
        tool_calls_parsed_count = emit_tool_calls_buf.len();
        // Skip caching when the turn produced no replay-able payload —
        // empty trimmed content AND no tool_calls. The fingerprint for
        // such turns collides on the hash of `("assistant", "")` so
        // any subsequent empty-emission turn (the model giving up with
        // a trailing whitespace fragment) overwrites the prior entry.
        // Pi typically doesn't replay empty assistant turns at all, so
        // the cache entry is dead weight at best and a subtle
        // mis-replay risk at worst (Pi sends content="" + tool_calls=[]
        // for a different reason and our cache hands back the wrong
        // tokens). Two write conditions to satisfy: at least one
        // visible event (text OR tool_calls) AND at least one raw
        // token actually emitted.
        let have_replayable_payload =
            !emit_text_buf.trim().is_empty() || !emit_tool_calls_buf.is_empty();
        if have_replayable_payload
            && generated_count > 0
            && m.conversation_tokens.len() > decode_start_tokens_idx
        {
            let cached_seq: Vec<u32> = m.conversation_tokens[decode_start_tokens_idx..].to_vec();
            let fp = prompt_frame::assistant_turn_fingerprint(&emit_text_buf, &emit_tool_calls_buf);
            if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[asst-cache store] fp={:#018x} content.len={} tool_calls={} tokens={}",
                    fp,
                    emit_text_buf.len(),
                    emit_tool_calls_buf.len(),
                    cached_seq.len(),
                );
            }
            m.asst_turn_cache.insert(fp, cached_seq);
        }
    }

    m.seq_pos = state.n_tokens as usize;

    let _ = gpu.hip.device_synchronize();
    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 && decode_ms > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };

    // Build the done envelope through serde_json so the new
    // `cached_tokens` field (V4F prefix-cache LCP hit count) interleaves
    // cleanly with the legacy `prefill_tokens` / `prefill_ms` / spec
    // counters. The TTL of stale {} interpolation here is exactly the
    // surface area we just fixed in `emit_error_with_id` — same risk
    // class.
    //
    // `prefill_tokens` semantics: number of tokens actually FED to the
    // forward path this turn (i.e., suffix_tokens.len(), == total
    // prompt minus cached prefix). Cache-hit accounting:
    //   prompt_tokens (sent by client)       = prompt_ids.len()
    //   cached_tokens (prefix-cache hit)     = cached_tokens (= lcp)
    //   prefill_tokens (actually prefilled)  = suffix_tokens.len()
    // Sum: cached + prefill == prompt_tokens. The CLI's OpenAI-compat
    // layer maps `cached_tokens` → `usage.prompt_tokens_details.cached_tokens`.
    let prompt_tokens_total = prompt_ids.len();
    let prefill_tokens_actual = suffix_tokens.len();
    // Tell the OpenAI-compat layer how the decode loop exited. Without
    // this the CLI fell back to "stop" for every non-tool-call turn,
    // hiding `max_tokens` truncation behind a natural-completion signal
    // — strict clients use `finish_reason: "length"` to decide whether
    // to retry with a longer budget.
    //
    //   tool_calls — at least one complete `<｜DSML｜tool_calls>` block
    //                was parsed (`tool_calls_parsed_count > 0`). Wins
    //                over "length" even when max_tokens hit after the
    //                block closed.
    //   length     — generated_count reached max_tokens with no
    //                completed tool_calls block.
    //   stop       — model emitted EOS, or generated_count is < max
    //                because the spec-decode loop accepted EOS in the
    //                middle of an accepted-tokens chunk.
    //
    // `tool_calls_parsed_count` is set inside the non-spec branch
    // immediately after parser.finish(); spec_mode leaves it at 0.
    let finish_reason: &'static str = if tool_calls_parsed_count > 0 {
        "tool_calls"
    } else if generated_count >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let done_envelope = if spec_mode {
        let accept_pct = if spec_drafts_offered > 0 {
            spec_drafts_accepted as f64 / spec_drafts_offered as f64 * 100.0
        } else {
            0.0
        };
        serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": generated_count,
            "tok_s": tok_s,
            "prompt_tokens": prompt_tokens_total,
            "prefill_tokens": prefill_tokens_actual,
            "cached_tokens": cached_tokens,
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "finish_reason": finish_reason,
            "spec_k": spec_k,
            "spec_windows": spec_windows,
            "spec_accept_pct": accept_pct,
        })
    } else {
        serde_json::json!({
            "type": "done",
            "id": id,
            "tokens": generated_count,
            "tok_s": tok_s,
            "prompt_tokens": prompt_tokens_total,
            "prefill_tokens": prefill_tokens_actual,
            "cached_tokens": cached_tokens,
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "finish_reason": finish_reason,
        })
    };
    let _ = writeln!(stdout, "{}", done_envelope);
    let _ = stdout.flush();
}

/// Qwen2 generate path (arch_id=7, hipfire-arch-qwen2).
///
/// Phase-1 bring-up scope: encode prompt → prefill → greedy decode loop
/// → stream `{"type":"token",...}` events → `{"type":"done",...}`.
///
/// Deliberately bypasses qwen35/llama machinery — no PFlash, no DFlash,
/// no eviction, no ChatML scaffolding, no tool-use, no `<think>` /
/// `max_think_tokens`, no repeat penalty, no top-p sampling. These
/// land as the surrounding daemon features mature for the Qwen2 path.
/// `temp` is currently honored only as a "≤ 1e-6 means greedy"
/// signal; anything else falls back to greedy too (no sampler wired).
///
/// Conversation state on the daemon side advances via
/// `m.seq_pos` (mirrors the qwen35/llama bookkeeping) plus
/// `state.next_pos` inside `Qwen2State`. On context overflow we hard
/// reset (no CASK eviction on arch_id=7) — same fallback the
/// llama path uses.
fn generate_qwen2(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    _system_prompt: Option<&str>,
    _temp: f32,
    _top_p: f32,
    max_tokens: usize,
    _repeat_penalty: f32,
    _repeat_window: usize,
) {
    let tokenizer = match m.tokenizer.as_ref() {
        Some(t) => t,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        }
    };
    let cfg = match m.qwen2_config.as_ref() {
        Some(c) => c,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"qwen2_config missing on arch_id=7 generate"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        }
    };
    let weights = m
        .qwen2_weights
        .as_ref()
        .expect("qwen2_weights missing on arch_id=7 generate");
    let state = m
        .qwen2_state
        .as_mut()
        .expect("qwen2_state missing on arch_id=7 generate");

    let prompt_ids = tokenizer.encode(prompt);
    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    // Capacity guard. No eviction on arch_id=7 yet — reset state when
    // the requested run would overflow the KV budget.
    if state.next_pos + prompt_ids.len() + max_tokens > state.max_seq {
        eprintln!(
            "[daemon] arch_id=7 context full ({}/{}) — resetting Qwen2State.next_pos",
            state.next_pos, state.max_seq,
        );
        state.reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }

    let t0 = Instant::now();

    // Prefill: forward_step per prompt token. The last call leaves
    // logits in state.logits — these are the predictions for the
    // first generated token.
    for &tok in &prompt_ids {
        if let Err(e) = qwen2::forward_step(gpu, weights, cfg, state, tok) {
            emit_error_with_id(stdout, id, format!("qwen2 prefill failed: {e:?}"));
            let _ = stdout.flush();
            return;
        }
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // Decode loop. Greedy argmax for now (see fn doc for sampling
    // scope). The first generated token is argmax of the prefill's
    // final logits; each subsequent token requires another
    // forward_step.
    let mut generated_count: usize = 0;
    let eos_set: &[u32] = &cfg.eos_token_ids;
    let decode_t0 = Instant::now();
    let mut next_tok = match gpu.argmax_f32(&state.logits, cfg.vocab_size) {
        Ok(t) => t,
        Err(e) => {
            emit_error_with_id(stdout, id, format!("argmax failed: {e:?}"));
            let _ = stdout.flush();
            return;
        }
    };

    loop {
        if generated_count >= max_tokens {
            break;
        }
        if eos_set.contains(&next_tok) {
            break;
        }
        // Emit text fragment for this token. Tokenizer.decode handles
        // BPE byte-fragment reassembly; for special tokens that decode
        // to an empty string we still advance the loop. Build through
        // serde_json so `id` (user-supplied) and `frag` (arbitrary
        // UTF-8 with possible `"` / `\` / control chars) can't corrupt
        // the JSONL line.
        let frag = tokenizer.decode(&[next_tok]);
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        match qwen2::forward_step_greedy(gpu, weights, cfg, state, next_tok) {
            Ok(t) => next_tok = t,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("forward_step_greedy failed: {e:?}"));
                let _ = stdout.flush();
                return;
            }
        }
    }

    // Daemon bookkeeping: seq_pos matches Qwen2State's internal cursor.
    m.seq_pos = state.next_pos;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 && decode_ms > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.2},"prefill_ms":{},"total_ms":{}}}"#,
        id, generated_count, tok_s, prefill_ms, total_ms,
    );
    let _ = stdout.flush();
}

/// MiniMax-M2 (arch_id=10) generate path — minimal AR bring-up.
///
/// Mirrors `generate_qwen2`'s shape (prefill = per-token loop, decode =
/// per-token loop, JSONL `token` / `done` events) with two differences:
///
///   1. Prompt build goes through `JinjaChatFrame` when `HIPFIRE_JINJA_CHAT=1`
///      and the model carries a chat_template (so MiniMax-M2's own ChatML-ish
///      template + `tools` / `messages` reach the upstream Jinja branches),
///      falling back to the hand-rolled `ChatFrame::Plain` scaffold otherwise.
///   2. `minimax::forward::decode_step` returns the full logits `Vec<f32>`
///      (the state does NOT stash a greedy next-token), so sampling runs
///      host-side via `deepseek4::sampling::sample_token` on that vector.
///
/// Out of scope for the scaffold (and intentionally NOT wired): spec-decode,
/// MTP, grammar-constrained decoding, tool-call parsing/execution, repeat
/// penalty, multi-GPU, eviction/prefix-cache. Correctness first.
#[allow(clippy::too_many_arguments)]
fn generate_minimax(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    max_think_tokens: usize,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
) {
    if m.tokenizer.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }
    if m.minimax_config.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"minimax_config missing on arch_id=10 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    // ── Prompt build (same two-path branch as the qwen35 AR path) ──
    // `primed_think` records whether the rendered prompt actually ended with
    // the MiniMax `<think>` generation-primer, so we only re-emit the opener
    // (below) when the model truly begins inside the reasoning block. A jinja
    // render failure that falls back to the Plain frame leaves it false.
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(h) => h,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
                            content: prompt.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                        synthesized = v;
                        &synthesized
                    }
                };
                frame.render_messages(messages_slice, tools, None)
            } else {
                frame.render()
            };
            match render_result {
                Ok(rendered) => {
                    primed_think = rendered.trim_end().ends_with("<think>");
                    tokenizer.encode(&rendered)
                }
                Err(e) => {
                    eprintln!("[daemon] jinja render failed in minimax path ({e}) — falling back to Plain");
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                        raw: effective_raw(m),
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                raw: effective_raw(m),
            }
            .build()
        }
    };

    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.minimax_eos_tok;

    // Capacity guard. No eviction on arch_id=10 — reset the KV cursor when
    // the requested run would overflow the budget. (max_seq + n_tokens live
    // on the state.)
    let overflow = {
        let state = m.minimax_state.as_ref().unwrap();
        state.n_tokens + prompt_ids.len() + max_tokens > state.max_seq
    };
    if overflow {
        let (n, cap) = {
            let state = m.minimax_state.as_ref().unwrap();
            (state.n_tokens, state.max_seq)
        };
        eprintln!("[daemon] arch_id=10 context full ({n}/{cap}) — resetting MiniMaxState",);
        m.minimax_state.as_mut().unwrap().reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token. Disjoint field borrows of
    // `m` (config / weights / state) let us also push to
    // `m.conversation_tokens` in the same scope (same pattern as
    // generate_qwen2). The LAST decode_step's logits are the predictions
    // for the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let cfg = m.minimax_config.as_ref().unwrap();
        let weights = m.minimax_weights.as_ref().unwrap();
        let state = m.minimax_state.as_mut().unwrap();
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_ids {
            match minimax::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("minimax prefill failed: {e:?}"));
                    return;
                }
            }
            position += 1;
        }
    }
    for &tok in &prompt_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // MiniMax-M2's chat template unconditionally primes the assistant turn
    // with `<think>\n` (chat_template.jinja generation-prompt block), so the
    // model's GENERATED tokens begin *inside* the reasoning block and it only
    // ever emits the closing `</think>`. Every downstream `<think>` consumer —
    // the serve reasoning_content/content split, the run/chat-path stripper,
    // and the history `stripThinkingInline` — keys on a LEADING `<think>` and
    // so never engages, leaking the chain-of-thought into `message.content`.
    // The primer is already in the KV from prefill; re-emit it into the token
    // stream (display-only, not pushed to state) so the assistant message is a
    // well-formed `<think>...</think>...` block for every consumer.
    if primed_think {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type": "token", "id": id, "text": "<think>\n"}),
        );
        let _ = stdout.flush();
    }

    // ── Decode loop. Sample host-side from the running logits vector.
    // `temp <= 0` makes sample_token greedy; otherwise top_p nucleus.
    // Seed the PRNG from wall-clock nanos so successive same-prompt runs
    // don't lock-step (greedy is still deterministic). ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        // Sample next token from the most recent logits.
        let next_tok = deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if next_tok == eos_tok {
            break;
        }

        // Emit the text fragment. Build through serde_json so a user-supplied
        // `id` or arbitrary-UTF-8 fragment can't corrupt the JSONL line.
        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        // Advance one step on the freshly sampled token.
        let step = {
            let cfg = m.minimax_config.as_ref().unwrap();
            let weights = m.minimax_weights.as_ref().unwrap();
            let state = m.minimax_state.as_mut().unwrap();
            let position = state.n_tokens as u32;
            minimax::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("minimax decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.minimax_state.as_ref().unwrap().n_tokens;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.2},"prefill_ms":{},"total_ms":{}}}"#,
        id, generated_count, tok_s, prefill_ms, total_ms,
    );
    let _ = stdout.flush();
}

/// LFM2.5-MoE (arch_id=11) generate path — minimal AR bring-up.
///
/// Structurally identical to `generate_minimax` (prefill = per-token loop,
/// decode = per-token loop, JSONL `token` / `done` events). Only the arch
/// types and `forward::decode_step` path differ. Out of scope (and not
/// wired): spec-decode, MTP, grammar, tool-call parsing/execution, repeat
/// penalty, multi-GPU, eviction/prefix-cache. Correctness first.
#[cfg(feature = "arch-lfm2moe")]
#[allow(clippy::too_many_arguments)]
fn generate_lfm2moe(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    max_think_tokens: usize,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
) {
    if m.tokenizer.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }
    if m.lfm2moe_config.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"lfm2moe_config missing on arch_id=11 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    // ── Prompt build (same two-path branch as the minimax AR path) ──
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(h) => h,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
                            content: prompt.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                        synthesized = v;
                        &synthesized
                    }
                };
                frame.render_messages(messages_slice, tools, None)
            } else {
                frame.render()
            };
            match render_result {
                Ok(rendered) => tokenizer.encode(&rendered),
                Err(e) => {
                    eprintln!("[daemon] jinja render failed in lfm2moe path ({e}) — falling back to Plain");
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                        raw: effective_raw(m),
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                raw: effective_raw(m),
            }
            .build()
        }
    };

    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.lfm2moe_eos_tok;

    // Capacity guard. No eviction on arch_id=11 — reset the KV + conv-state
    // cursors when the requested run would overflow the budget.
    let overflow = {
        let state = m.lfm2moe_state.as_ref().unwrap();
        state.n_tokens + prompt_ids.len() + max_tokens > state.max_seq
    };
    if overflow {
        let (n, cap) = {
            let state = m.lfm2moe_state.as_ref().unwrap();
            (state.n_tokens, state.max_seq)
        };
        eprintln!("[daemon] arch_id=11 context full ({n}/{cap}) — resetting Lfm2MoeState",);
        let _ = m.lfm2moe_state.as_mut().unwrap().reset(gpu);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token. The LAST decode_step's logits
    // are the predictions for the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let cfg = m.lfm2moe_config.as_ref().unwrap();
        let weights = m.lfm2moe_weights.as_ref().unwrap();
        let state = m.lfm2moe_state.as_mut().unwrap();
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_ids {
            match lfm2moe::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("lfm2moe prefill failed: {e:?}"));
                    return;
                }
            }
            position += 1;
        }
    }
    for &tok in &prompt_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // ── Decode loop. Sample host-side from the running logits vector. ──
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    let mut rng = deepseek4::sampling::Xorshift::new(seed);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        let next_tok = deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if next_tok == eos_tok {
            break;
        }

        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        let step = {
            let cfg = m.lfm2moe_config.as_ref().unwrap();
            let weights = m.lfm2moe_weights.as_ref().unwrap();
            let state = m.lfm2moe_state.as_mut().unwrap();
            let position = state.n_tokens as u32;
            lfm2moe::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm2moe decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.lfm2moe_state.as_ref().unwrap().n_tokens;

    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.2},"prefill_ms":{},"total_ms":{}}}"#,
        id, generated_count, tok_s, prefill_ms, total_ms,
    );
    let _ = stdout.flush();
}
