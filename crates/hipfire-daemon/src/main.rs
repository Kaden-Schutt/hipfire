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
#[cfg(test)]
use hipfire_generate::validate_qwen35_fused_dense_prefill_batch_preflight;
use hipfire_generate::{
    validate_generate_batch_decode, validate_generate_batch_prefill,
    validate_prefix_hash_preflight, GenerateVLParams, ImageSource,
};
use hipfire_model::is_qwen35_family_arch_id;
use hipfire_prompt as prompt_frame;
use hipfire_runtime::llama;
use hipfire_state::{
    described_sequence_state_json, model_worker_runtime_view_json,
    parse_describe_sequence_state_request, parse_release_sequence_state_request,
    parse_release_sessions_request, parse_reserve_session_state_request,
    parse_unload_worker_request, parsed_handle_may_target_generic, release_sessions_done_json,
    release_state_done_json, reserve_session_state_done_json, reserve_session_state_rejected_json,
    sequence_state_reservation_plan, sequence_state_reservation_plan_for_reserved_bytes,
    session_state_reservation_describe_json, unload_worker_done_json, GenericSequenceStateArena,
};
#[cfg(test)]
use hipfire_state::{
    generic_state_reservation_descriptors, parse_reserve_session_state_kinds,
    parse_sequence_state_handle, sequence_state_handle_id, sequence_state_handle_parts,
    sequence_state_page_descriptor_json, SequenceStateHandle,
};
use std::io::{BufRead, Write};
use std::time::Instant;

// These modules now live in `hipfire-serving-core` (workstream A0). Re-import
// them at the crate root so existing `crate::events` / `crate::model` /
// `crate::session` / … paths in the daemon's other modules keep resolving
// unchanged.
use dummy::{
    emit_dummy_generate_batch_prefill_ready, run_generate_batch_prefill_dummy, DummyModelState,
};
use events::{emit_error_with_id, write_error, MAX_BASE64_ENCODED_LEN};
use evidence::{
    write_daemon_moe_router_evidence, write_daemon_runtime_oneshot_evidence,
    DaemonMoeRouterHistogramGuard,
};
use generate_vl::{generate_vl, generate_vl_dots_ocr};
use hipfire_serving_core::{
    dummy, events, evidence, generate_vl, load, model, output_filter, qwen35_decode,
    qwen35_prefill, request, session,
};
use load::*;
use model::{CaskConfig, LoadedModel, RAW_OVERRIDE};
use output_filter::{
    chat_output_filter, normalize_daemon_prompt, normalize_request_stop_sequences,
};
use qwen35_decode::*;
use qwen35_prefill::*;
use request::ThinkMode;
use session::*;
mod generate;
mod generate_arch;
use generate::*;

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
/// Returns the [`FlockGuard`]; caller MUST keep it alive for the process
/// lifetime (dropping it closes the fd and releases the lock).
fn acquire_daemon_lock() -> hipfire_lock::FlockGuard {
    #[cfg(unix)]
    let home = std::env::var("HOME").expect("HOME environment variable not set");
    #[cfg(windows)]
    let home = std::env::var("USERPROFILE").expect("USERPROFILE environment variable not set");

    let pid_path = std::path::PathBuf::from(home)
        .join(".hipfire")
        .join("daemon.pid");

    let mut guard =
        hipfire_lock::FlockGuard::open(&pid_path).expect("failed to open ~/.hipfire/daemon.pid");
    match guard.try_lock() {
        Ok(true) => {}
        Ok(false) => {
            // Already held: surface the holder PID (written below by the live
            // daemon) in the fatal message.
            let holder = guard.holder().unwrap_or_default();
            let pid = holder.trim();
            let pid_display = if pid.is_empty() { "<unknown>" } else { pid };
            let kill_arg = if pid.is_empty() { "<pid>" } else { pid };
            eprintln!(
                "FATAL: hipfire daemon already running (PID {}). Run `kill {}` and retry.",
                pid_display, kill_arg
            );
            std::process::exit(1);
        }
        Err(e) => panic!("failed to flock ~/.hipfire/daemon.pid: {e}"),
    }

    // Got the lock. Record our PID so the fatal message above (in a second
    // daemon) and external tooling can show a useful number. `flock` is on the
    // open fd, so rewriting the contents doesn't drop the lock.
    let _ = guard.write_holder(&std::process::id().to_string());
    guard
}

#[cfg(test)]
mod generate_batch_prefill_tests {
    use super::*;
    use hipfire_arch_qwen35::qwen35::LayerType;
    use hipfire_generate::{
        build_qwen35_fused_dense_prefill_batch_contract, compute_qwen35_prefix_hash,
        plan_generate_batch_prefill_qwen35, qwen35_decode_batch_scheduler_metadata,
        qwen35_fused_prefill_boundary_cuts, qwen35_grouped_moe_decode_auto_latency_gate_passed,
        qwen35_prefill_scratch_target_batch, select_qwen35_decode_batch_backend,
        select_qwen35_prefill_batch_backend,
        validate_qwen35_fused_grouped_moe_prefill_batch_preflight, GenerateBatchPrefillPlan,
        Qwen35DecodeBatchBackend, Qwen35FusedDensePrefillInputKind, Qwen35PrefillBatchBackend,
        Qwen35PreparedPrefillSession, Qwen35SemanticBoundaryCheckpoint,
    };
    use hipfire_model::ModelWorkerId;
    use hipfire_state::{
        describe_sequence_state_descriptors, parsed_handle_may_target_loaded_state,
        qwen35_sequence_state_handle, ModelWorkerMemoryView, ModelWorkerRuntimeView,
        ParsedSequenceStateHandle, SequenceStateArenaBackend, SequenceStatePageDescriptor,
        SequenceStatePageKind, SequenceStatePrefixHash,
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

            // Train a PFlash importance-scorer drafter in-process against the
            // resident target (teacher/student split). STEP 1: plumbing only —
            // validates args + the hipfire-train link; the loop wiring lands in
            // step 3. See docs/plans/2026-06-19-train-as-daemon-op.md.
            "train_drafter" => {
                let arch = msg
                    .get("arch")
                    .and_then(|v| v.as_str())
                    .unwrap_or("ssm")
                    .to_string();
                if arch != "ssm" && arch != "attention" {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        format!("train_drafter: unknown arch '{arch}' (expected ssm|attention)"),
                    );
                    continue;
                }
                if model.as_ref().and_then(|m| m.q35_config.as_ref()).is_none() {
                    emit_error_with_id(
                        &mut stdout,
                        "",
                        "train_drafter: no qwen3.5-family model loaded".to_string(),
                    );
                    continue;
                }
                // Parse the train block into the SHARED TrainCfg (proves the
                // hipfire-train dep links and the op speaks the lib's types).
                let t = msg
                    .get("train")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let labels = msg
                    .get("labels")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let getu = |o: &serde_json::Value, k: &str, d: usize| -> usize {
                    o.get(k)
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(d)
                };
                let getf = |o: &serde_json::Value, k: &str, d: f32| -> f32 {
                    o.get(k)
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .unwrap_or(d)
                };
                let cfg = hipfire_train::train_loop::TrainCfg {
                    seq: getu(&labels, "seq", 512),
                    block: getu(&labels, "block", 64),
                    n_eval: getu(&labels, "n_eval", 20),
                    epochs: getu(&t, "epochs", 300),
                    lr: getf(&t, "lr", 1e-3),
                    wd: getf(&t, "wd", 0.0),
                    tau: getf(&t, "tau", 0.1),
                    eval_every: getu(&t, "eval_every", 15),
                    report_train: t
                        .get("report_train")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                };
                let resp = serde_json::json!({
                    "type": "train_drafter",
                    "status": "not_implemented",
                    "detail": "step 1 plumbing only — loop wiring lands in step 3",
                    "arch": arch,
                    "label_source": labels.get("source").and_then(|v| v.as_str()).unwrap_or("file"),
                    "cfg": {
                        "seq": cfg.seq, "block": cfg.block, "n_eval": cfg.n_eval,
                        "epochs": cfg.epochs, "lr": cfg.lr, "wd": cfg.wd,
                        "tau": cfg.tau, "eval_every": cfg.eval_every,
                    },
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
