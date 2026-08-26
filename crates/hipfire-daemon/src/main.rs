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
use hipfire_runtime::emit_text::{
    currently_in_think, extract_tool_calls_from_text, ThinkOutputRouter, ThinkRouteEvent,
    ToolOutputRouter, ToolRouteError, ToolRouteEvent,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_runtime::spec::accept_greedy_prefix;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use hipfire_engine::emit::*;
use hipfire_engine::prompt::*;
use hipfire_engine::redline::*;
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
#[cfg(feature = "serve-fault-inject")]
use hipfire_generate::ar::arm_fault_after_prefill;
#[cfg(feature = "serve-fault-inject")]
use hipfire_generate::ar::take_fault_after_prefill;
use hipfire_generate::ar::{
    ckpt_interval, ckpt_max, ckpt_resume_enabled, deepseek4_spec_requested,
    deepseek4_spec_requested_from_policy, emit_qwen_ar_done, emit_qwen_ar_open_think_terminal,
    generate, llama_prefill_sample_seed, llama_qwen3_batched_prefill_eligible,
    model_retry_reset_eligible, qwen_ar_apply_cache_action, qwen_ar_cache_action,
    qwen_ar_done_value, qwen_ar_drain_pending_into_router, qwen_ar_eos_filter_config,
    qwen_ar_eviction_prefill_chunk_limit, qwen_ar_finish_route, qwen_ar_forward_fail_action,
    qwen_ar_forward_fail_message, qwen_ar_observe_and_route, qwen_ar_raw_commit_token,
    qwen_ar_route_filter_text, qwen_ar_route_think_events, reset_core_arch_key,
    select_generation_route, truncate_checkpoints, write_error, GenerationRoute,
    GenerationRouteInputs, QwenArCacheAction, QwenArForwardFailAction, QwenArRawCommitDisposition,
    QwenArRouteFinish, QwenArSemanticProducer, QwenArTerminalCause,
};
use hipfire_generate::batch::{
    attach_qwen_ep_batch_receipt_evidence, drive_lfm_continuous_batch,
    drive_qwen35_ep_continuous_batch, drive_qwen_continuous_batch, emit_uncorrelated_error,
    is_batch_request_eligible, is_qwen_ep_batch_request_eligible,
    lfm_prefill_cancellable_or_fallback,
};
use hipfire_generate::redline::{
    handle_redline_dflash_verify_shadow_pm4, handle_redline_dispatch_profile,
    handle_redline_dspark_shadow_pm4, handle_redline_pm4_prefix_profile,
    handle_redline_prefix_shadow, handle_redline_probe_aql, handle_redline_shadow,
    redline_append_tensor_slice, redline_bench_decode_deepseek4, redline_bench_decode_lfm2moe,
    redline_deepseek4_snapshot, redline_dspark_shadow_block, redline_dspark_verify_guard,
    redline_dspark_verify_snapshot, redline_is_dense_lfm, redline_lfm2moe_snapshot,
    redline_pm4_prefix_profile_deepseek4, redline_prepare_retained_fixture,
    redline_prime_deepseek4, redline_prime_dspark_shadow_arm, redline_prime_qwen,
    redline_prime_retained_fixture, redline_qwen_debug_hashes, redline_qwen_snapshot,
    redline_reset_deepseek4, redline_reset_lfm2moe, redline_reset_qwen,
    redline_run_deepseek4_decode, redline_run_direct_fixture, redline_run_dspark_capture_arm,
    redline_run_dspark_direct_arm, redline_run_dspark_replay_arm, redline_shadow_deepseek4,
    redline_shadow_dspark_verify_pm4, redline_snapshot, RedlineDeepseek4Snapshot, RedlineDsparkArm,
    RedlineDsparkReplayArm, RedlineDsparkVerifySnapshot, RedlineLfm2MoeSnapshot,
    RedlineQwenSnapshot, RedlineSnapshot,
};
use hipfire_generate::vision::{GenerateVLParams, ImageSource};
use hipfire_loader::parallel_capability::{ModelVariant, ParallelAxis, RawParallelRequest};
use hipfire_loader::{
    admit_path, classify_source, load_admitted, AdmittedLoad, AsstTurnCache, EpArch, EpState,
    Eviction, LoadedModel, ModelLoadOptions,
};
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::multi_gpu::{DeviceMesh, DimKind};
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, EvictRetain, FinishSummary, PrefillOutcome, SpecAdvance, SpecEmit,
    SpecTarget, Speculator, StopReason,
};

/// Formats the independent Qwen decode-batch path can actually execute.
/// Must stay aligned with `lm_head_batched` + `prepare_decode_batch_inputs`
/// in hipfire-arch-qwen35 — unsupported lm_head or F32 embedding must never
/// advertise `continuous_batch_capable` or enter the batch route.

pub type CaskConfig = hipfire_runtime::loader_api::CaskConfig;

#[allow(dead_code)]
fn emit_error_no_id(stdout: &mut impl std::io::Write, message: impl std::fmt::Display) {
    hipfire_generate::dense::emit_active_attempt_error(
        stdout,
        None,
        &message.to_string(),
        "internal",
        false,
        false,
    );
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
            let (kv_hash, kv_bytes, recurrent_hash, recurrent_bytes) = match m.qwen35() {
                Some(bundle) => match redline_qwen_snapshot(gpu, bundle) {
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
    hipfire_generate::dense::emit_active_attempt_error(
        stdout,
        Some(id),
        message,
        class,
        retryable,
        rolled_back,
    );
}

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

/// Whether the deferred multi-device load path must run
/// `ensure_vmm_ready_for_load` before constructing a new model.
///
/// Only when there is no live prior model (`model_present == false`): a
/// failed deferred prior unload leaves `model=None` while VMM arenas may
/// still be pending. Do NOT preflight when a live deferred prior still
/// occupies `model` — that path tears down after successful new load.
fn ep_deferred_needs_vmm_preflight(deferred_unload: bool, model_present: bool) -> bool {
    deferred_unload && !model_present
}

/// Resolved device-mesh load plan (CAP-001). Produced BEFORE any prior-model
/// unload, GPU mutation, or load side-effect; decides the lifecycle (eager vs
/// deferred prior-model unload) and the dispatch route (admission-routed
/// `load_admitted` vs the beta legacy qwen35 PP wrapper).
enum DaemonLoadPlan {
    /// Beta legacy qwen35 hand-coded pipeline-parallel
    /// (`load_model_with_gemma4_drafter` with pp>1 → carrier registry →
    /// `load_qwen35_pp`/`pp_gpus`, served by `generate_multi`). Eager
    /// prior-model unload (the legacy wrapper uses the daemon's `gpu`
    /// directly); PFlash not suppressed (`HIPFIRE_PP_PFLASH=1` opts in).
    LegacyQwen35Pp { pp: usize },
    /// Admission-routed load. `deferred_unload`: multi-device (effective TP
    /// or EP) defers prior-model unload so a partial multi-GPU load failure
    /// doesn't orphan VRAM; Single/PP unload eagerly (as before).
    Admitted {
        admitted: hipfire_loader::AdmittedLoad,
        effective: RawParallelRequest,
        deferred_unload: bool,
    },
}

impl DaemonLoadPlan {
    fn deferred_unload(&self) -> bool {
        match self {
            DaemonLoadPlan::LegacyQwen35Pp { .. } => false,
            DaemonLoadPlan::Admitted {
                deferred_unload, ..
            } => *deferred_unload,
        }
    }

    fn is_single(&self) -> bool {
        matches!(
            self,
            DaemonLoadPlan::Admitted { effective, .. }
                if effective.axis() == ParallelAxis::Single
        )
    }

    fn is_ep(&self) -> bool {
        matches!(
            self,
            DaemonLoadPlan::Admitted { effective, .. } if effective.axis() == ParallelAxis::Ep
        )
    }

    fn is_mesh(&self) -> bool {
        matches!(
            self,
            DaemonLoadPlan::Admitted { effective, .. }
                if matches!(effective.axis(), ParallelAxis::Tp | ParallelAxis::Pp)
        )
    }
}

/// Resolve a load request into a [`DaemonLoadPlan`].
///
/// Beta-compat PP classification happens BEFORE admission: qwen35 dense/MoE
/// (the only carriers with a hand-coded pp>1 path) keeps the legacy PP load;
/// the policy table marks qwen35 Pp "Planned" (GEN-001), so admission must
/// not see it. qwen35-VL stays on the admission path (AXIS-004 Planned).
fn daemon_load_plan(path: &str, pp: usize, tp: usize, ep: usize) -> Result<DaemonLoadPlan, String> {
    if pp > 1 && tp <= 1 && ep <= 1 {
        let src = ModelSource::from_path(path)?;
        if let Ok((carrier, variant)) = hipfire_loader::classify_source(&src) {
            if carrier.name() == "qwen35" && !matches!(variant, ModelVariant::Qwen35Vl) {
                // The legacy PP plan unloads eagerly; the carrier's own
                // safetensors + pp>1 refusal would fire only AFTER the live
                // prior model is torn down. Refuse the source shape here,
                // preflight, canonically (same rule as the admitted mesh
                // axes — legacy qwen35 PP is also HFQ-only).
                if !matches!(src, ModelSource::Hfq(_)) {
                    return Err(format!(
                        "[CAP-001] PP load requires an HFQ model file (safetensors directory sources are single-GPU only): {}",
                        path
                    ));
                }
                return Ok(DaemonLoadPlan::LegacyQwen35Pp { pp });
            }
        }
    }
    let raw = RawParallelRequest::new(pp, tp, ep);
    let admitted = hipfire_loader::admit_path(path, raw)?;
    let effective = admitted.admission().effective();
    // Dense mesh carriers (effective Tp / PpDense) load through
    // `load_admitted`, whose TP/PP helpers are HFQ-only — they read the
    // carrier's typed tensors from the HFQ container. A safetensors
    // directory would only be refused deterministically INSIDE the load,
    // which for eager-PP teardown happens AFTER the live prior model is
    // destroyed. Refuse the source shape here, preflight, canonically.
    if matches!(effective.axis(), ParallelAxis::Tp | ParallelAxis::Pp) && !admitted.is_hfq_source()
    {
        return Err(format!(
            "[CAP-001] {} load requires an HFQ model file (safetensors directory sources are single-GPU only): {}",
            effective.axis().name(),
            path
        ));
    }
    // Multi-device (effective TP or EP) defers prior-model unload so a
    // partial multi-GPU load failure cannot orphan the live prior model;
    // Single and PP unload eagerly.
    let deferred_unload = effective.tp > 1 || effective.ep > 1;
    Ok(DaemonLoadPlan::Admitted {
        admitted,
        effective,
        deferred_unload,
    })
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
                // unloaded eagerly (load_model uses the daemon's `gpu`
                // directly, so it can't be deferred without a major refactor).
                // The eager unload itself runs AFTER admission below — a load
                // refusal (mistyped path, Planned/Unsupported axis, param
                // combo) must never destroy the live prior model.
                //
                // ── Device-mesh load planning (CAP-001) ─────────────
                // Parse path + raw axis degrees FIRST, before any prior-model
                // unload, GPU mutation, or load side-effect. Admission
                // precedes all load-triggered work: mesh construction,
                // secondary-device init, unload, and per-axis dispatch.
                let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let raw_pp = msg
                    .get("params")
                    .and_then(|p| p.get("pp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let raw_tp = msg
                    .get("params")
                    .and_then(|p| p.get("tp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let raw_ep = msg
                    .get("params")
                    .and_then(|p| p.get("ep"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let parsed_continuous_batch_size = parse_continuous_batch_size(msg.get("params"));
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
                    mtp_mode: match mtp_mode.as_str() {
                        "on" => Some(true),
                        "off" => Some(false),
                        _ => None, // "auto" → loader default
                    },
                    mtp: match mtp_mode.as_str() {
                        "on" => Some(true),
                        "off" => Some(false),
                        _ => None, // "auto" → loader default
                    },
                    mtp_k: Some(mtp_k),
                };

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

                // Pipeline-parallel / expert-parallel degrees. Raw values were
                // parsed at the top of the arm (CAP-001 preflight); these are
                // the local spellings the rest of the handler uses.
                let pp = raw_pp;
                let tp = raw_tp;
                // ── Admission preflight (CAP-001) ─────────────────────
                // Runs BEFORE any prior-model unload or GPU mutation: a
                // refusal (mistyped path, Planned/Unsupported axis — e.g.
                // qwen35/DS4/LFM TP, qwen2 PP, qwen35 MoE EP — or a
                // composition error) emits [CAP-001]/[COMP-001] and leaves
                // the live prior model fully intact. The plan decides the
                // lifecycle (eager vs deferred prior-model unload) and the
                // dispatch route (admission-routed `load_admitted` vs the
                // beta legacy qwen35 PP wrapper).
                let state_quant_override = msg
                    .get("params")
                    .and_then(|p| p.get("state_quant"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let load_plan = match daemon_load_plan(path, pp, tp, raw_ep) {
                    Ok(plan) => plan,
                    Err(e) => {
                        emit_uncorrelated_error(&mut stdout, None, &e, "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let deferred_unload = load_plan.deferred_unload();
                let is_ep_route = load_plan.is_ep();
                let is_mesh_route = load_plan.is_mesh();
                if raw_ep > 1 && load_plan.is_single() {
                    eprintln!(
                        "[hipfire-daemon] requested ep={raw_ep} resolved to Single (dense-EP normalization, CAP-001)"
                    );
                }
                // Route guards — all BEFORE the eager prior-model unload.
                if is_ep_route && draft_path.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "EP serving does not support DFlash drafters in v1; reload without a draft.", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                if is_ep_route && gemma4_drafter.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "EP serving does not support the gemma4 EAGLE drafter; reload without params.drafter.", "unsupported", false, false);
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
                // Device-mesh dense carriers (admitted TP / dense PP) keep
                // weights + KV inside the carrier and serve through the
                // MeshCarrier generation route. Options the TP/PP load
                // helpers do not consume must be refused loudly here, never
                // silently ignored.
                if is_mesh_route {
                    if draft_path.is_some() {
                        emit_uncorrelated_error(&mut stdout, None, "DFlash speculative decode requires a single-GPU model; mesh (TP/PP) loads do not support drafters.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if gemma4_drafter.is_some() {
                        emit_uncorrelated_error(&mut stdout, None, "gemma4 EAGLE spec-decode requires a single-GPU model; mesh (TP/PP) loads do not support drafters.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if kv_mode_override.is_some()
                        || kv_backend_override.is_some()
                        || kv_adaptive_override.is_some()
                        || state_quant_override.is_some()
                    {
                        emit_uncorrelated_error(&mut stdout, None, "KV mode/backend/adaptive/state-quant overrides require a single-GPU model; mesh (TP/PP) loads use the carrier's own KV allocation.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if cask.sidecar.is_some() || cask_enabled {
                        emit_uncorrelated_error(&mut stdout, None, "CASK / TriAttention eviction requires a single-GPU model; mesh (TP/PP) loads do not support eviction.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                    if pflash_drafter.is_some() || pflash_mode_str != "off" {
                        emit_uncorrelated_error(&mut stdout, None, "PFlash prefill compression requires a single-GPU model; mesh (TP/PP) loads do not support it.", "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }
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
                // Route-option guard for the eager legacy qwen35 PP path: the
                // ds4-specific experts-per-token override is meaningless
                // there. Refuse BEFORE the prior-model teardown — the late
                // match-arm guard would otherwise fire only after the live
                // prior model was unloaded.
                if matches!(load_plan, DaemonLoadPlan::LegacyQwen35Pp { .. })
                    && deepseek4_experts_per_token.is_some()
                {
                    emit_uncorrelated_error(
                        &mut stdout,
                        None,
                        "DeepSeek V4 experts-per-token override requires a single-GPU model",
                        "unsupported",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // ── Deterministic eager-route preflight (loader-owned) ──
                // Runs every option/source validation the load path performs —
                // KV-backend parse + VMM allowlist + pp compat, qwen35
                // state-quant syntax, MTP-K window, DFlash target quant, and
                // the DeepSeek carrier's experts-per-token / heterogeneous
                // preconditions — BEFORE any prior-model teardown. The load
                // wrappers invoke the same validators defensively.
                let preflight_src = match ModelSource::from_path(path) {
                    Ok(src) => src,
                    Err(e) => {
                        emit_uncorrelated_error(&mut stdout, None, &e, "unsupported", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let preflight_carrier = match &load_plan {
                    DaemonLoadPlan::LegacyQwen35Pp { .. } => "qwen35",
                    DaemonLoadPlan::Admitted { .. } => {
                        match hipfire_loader::classify_source(&preflight_src) {
                            Ok((carrier, _)) => carrier.name(),
                            Err(e) => {
                                emit_uncorrelated_error(
                                    &mut stdout,
                                    None,
                                    &e,
                                    "unsupported",
                                    false,
                                    false,
                                );
                                let _ = stdout.flush();
                                continue;
                            }
                        }
                    }
                };
                let preflight_pp = match &load_plan {
                    DaemonLoadPlan::LegacyQwen35Pp { pp } => *pp,
                    DaemonLoadPlan::Admitted { effective, .. } => effective.pp,
                };
                // normalize_mtp_k (1..=8) runs only on the admitted MESH
                // routes (Tp / PpDense via `load_admitted`). Every other
                // daemon route dispatches to `load_model_with_gemma4_drafter`
                // (which never calls normalize_mtp_k — spec_build clamps qwen35
                // MTP K to 1..=10), so mtp_k 9/10 stays valid there.
                let mtp_k_admitted_route = is_mesh_route;
                let gpu_arch = gpu.arch.clone();
                let preflight = hipfire_loader::EagerLoadPreflight {
                    kv_mode: kv_mode_override.as_deref(),
                    kv_backend: kv_backend_override.as_deref(),
                    state_quant: state_quant_override.as_deref(),
                    mtp_k: Some(mtp_k),
                    mtp_k_admitted_route,
                    draft_requested: draft_path.is_some(),
                    gpu_arch: &gpu_arch,
                    deepseek4_experts_per_token,
                    deepseek4_compute_placement: Some(deepseek4_compute_placement.clone()),
                    deepseek4_dspark: spec_cfg.dspark,
                    path,
                    max_seq,
                    cask: &cask,
                    gemma4_drafter: gemma4_drafter.as_deref(),
                    kv_adaptive: kv_adaptive_override.as_deref(),
                    draft_path: draft_path.as_deref(),
                    tp: raw_tp,
                };
                if let Err(e) = hipfire_loader::preflight_eager_load(
                    &preflight_src,
                    preflight_carrier,
                    preflight_pp,
                    &preflight,
                    &mut gpu,
                ) {
                    emit_uncorrelated_error(&mut stdout, None, &e, "validation", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                // ── Prior-model teardown (AFTER admission) ───────────
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
                // the deferred multi-device (TP/EP) path it must NOT be torn
                // down here — otherwise a partial load failure (whose
                // deferral keeps `model` alive) would leave the surviving
                // prior model stripped of its drafter. Defer it to the
                // success branch alongside the deferred model unload. For the
                // eager paths the prior model is unloaded eagerly, so tear
                // pflash down here in the original order. (EP archs are
                // ds4/minimax and refuse PFlash drafters, so on a SUCCESSFUL
                // multi-device load this just frees the outgoing model's
                // drafter at the deferred site.)
                if !deferred_unload {
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
                // Deferred path: when no live prior model remains (fresh
                // daemon, or after deferred prior unload failed and left
                // model=None with pending VMM), refuse to construct a new
                // multi-device model until orphan teardown clears. Skip when
                // a live deferred prior still sits in `model` — unload stays
                // deferred until after successful new-model construction.
                if ep_deferred_needs_vmm_preflight(deferred_unload, model.is_some()) {
                    if let Err(err) = hipfire_loader::ensure_vmm_ready_for_load(&mut gpu) {
                        emit_uncorrelated_error(&mut stdout, None, &err, "internal", false, false);
                        let _ = stdout.flush();
                        continue;
                    }
                }
                let loaded = match load_plan {
                    DaemonLoadPlan::LegacyQwen35Pp { pp } => {
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
                    }
                    DaemonLoadPlan::Admitted { effective, .. }
                        if effective.axis() == ParallelAxis::Single =>
                    {
                        // Single-GPU: the DeepSeek V4 experts-per-token
                        // override is a SUPPORTED knob here (threaded into
                        // `load_model_with_gemma4_drafter` → the ds4 carrier).
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
                            1,
                            spec_cfg,
                            &mut gpu,
                        )
                    }
                    DaemonLoadPlan::Admitted { admitted, .. } if is_mesh_route => {
                        // Device-mesh dense carriers (Tp / PpDense axis):
                        // the admitted token routes through `load_admitted`,
                        // which owns the mesh (`Gpus::from_mesh`) — the
                        // daemon's single `gpu` handle is unused for these.
                        let mesh = DeviceMesh::rect(&[
                            (DimKind::Pp, pp),
                            (DimKind::Tp, tp),
                            (DimKind::Ep, raw_ep),
                        ]);
                        let opts = ModelLoadOptions::new(max_seq).with_spec(spec_cfg);
                        hipfire_loader::load_admitted(admitted, &mesh, opts, None)
                    }
                    DaemonLoadPlan::Admitted { effective, .. } => {
                        // Effective Ep axis (ds4/minimax; qwen35 EP is
                        // Planned and was refused at preflight). The EP
                        // loader re-admits internally with the effective
                        // degree — its [CAP-001] diagnostics are identical.
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
                            effective.ep,
                            kv_mode_override.as_deref(),
                            kv_backend_override.as_deref(),
                        )
                    }
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
                        // the newly built model, clear associated fresh state,
                        // and emit a hard error covering prior failure and any
                        // rollback failure.
                        if deferred_unload {
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
                        let vl = m.vision_config().is_some() || m.dots_ocr().is_some();
                        let (dim, layers, vocab) = match m.state.as_ref() {
                            Some(st) => {
                                let arch =
                                    st.as_ref() as &dyn hipfire_runtime::arch_model::ArchModel;
                                (arch.dim(), arch.n_layers(), arch.vocab_size())
                            }
                            None => (0, 0, 0),
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
                            // `mtp_weights_present` is set by the loader to exactly
                            // `qwen35_mtp_head.is_some()`, so the old middle term was
                            // redundant AND required the daemon to name an arch crate it
                            // does not depend on.
                            ds4_mtp || m.mtp_weights_present;

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
                        let reasoning_contract = hipfire_loader::carrier_for(m.arch_id)
                            .map(|c| c.caps().reasoning_contract.wire_name())
                            .unwrap_or("unsupported");
                        // Probe reasoning effort capability only for QwenJinja;
                        // all other contracts emit safe false/[] without probing.
                        let (reasoning_effort_native, reasoning_efforts): (bool, Vec<&str>) = {
                            let is_qwen_jinja = reasoning_contract == "qwen_jinja";
                            if is_qwen_jinja {
                                if let (Some(tok), Some(tmpl)) =
                                    (m.tokenizer.as_ref(), m.chat_template.as_ref())
                                {
                                    let cap =
                                        hipfire_runtime::prompt_frame::probe_effort_capability(
                                            tok, tmpl,
                                        );
                                    (cap.native, cap.supported)
                                } else {
                                    (false, Vec::new())
                                }
                            } else {
                                (false, Vec::new())
                            }
                        };
                        let reasoning_efforts_json = serde_json::to_string(&reasoning_efforts)
                            .unwrap_or_else(|_| "[]".to_string());
                        // Load ack exposes batch dimensions/capability; EP adds parallelism metadata but never infers operation from logs.
                        if staged_ep_batch {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{},"reasoning_contract":"{}","reasoning_effort_native":{},"reasoning_efforts":{},"cache_capable":{},"retry_reset_eligible":{},"continuous_batch_capable":{},"continuous_batch_slots":{},"continuous_batch_lane_capacity":{},"continuous_batch_parallelism":"expert_parallel","continuous_batch_rank_count":4,"continuous_batch_reduce":"peer_rooted_f32"}}"#,
                                arch,
                                dim,
                                layers,
                                vocab,
                                vl,
                                reasoning_contract,
                                reasoning_effort_native,
                                reasoning_efforts_json,
                                cache_capable,
                                retry_reset_eligible,
                                continuous_batch_capable,
                                staged_ep_slots,
                                staged_ep_lane_cap,
                            );
                        } else {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{},"reasoning_contract":"{}","reasoning_effort_native":{},"reasoning_efforts":{},"cache_capable":{},"retry_reset_eligible":{},"continuous_batch_capable":{}}}"#,
                                arch,
                                dim,
                                layers,
                                vocab,
                                vl,
                                reasoning_contract,
                                reasoning_effort_native,
                                reasoning_efforts_json,
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
                        // Multi-device guard (deferred TP/EP + mesh TP/PP):
                        // those paths serve through carriers that bypass
                        // PFlash entirely; loading a drafter here would just
                        // pin GPU memory it never reads until unload, so skip
                        // the load outright. Warn once if the operator
                        // actually supplied a drafter so the silent no-op is
                        // visible.
                        if deferred_unload || is_mesh_route {
                            if pflash_drafter.is_some() && pflash_mode_str != "off" {
                                eprintln!(
                                    "[pflash] WARN: ignoring PFlash drafter on multi-device load \
                                     (deferred={deferred_unload} mesh={is_mesh_route}) — the serving \
                                     path bypasses PFlash; drafter would only waste GPU memory",
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
                        Ok(t) => (!t.is_empty()).then_some(t),
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
                // OpenAI logprobs. The gateway has already validated the range
                // (0..=20, requires logprobs=true) and rejects rather than clamps,
                // so a value arriving here is well-formed. None leaves every token
                // envelope byte-identical to before.
                let logprobs_top_k: Option<usize> = msg
                    .get("logprobs")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                    .then(|| {
                        msg.get("top_logprobs")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0) as usize
                    })
                    .filter(|k| *k > 0);
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
                // Typed thinking flag: when present it is authoritative for Jinja
                // enablement. HTTP normalization always sends it; direct JSONL
                // clients may omit it and fall back to legacy effort/cap inference.
                let thinking_enabled: Option<bool> =
                    msg.get("thinking_enabled").and_then(|v| v.as_bool());
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
                // This is an independent explicit cap (never derived from
                // effort) — 0 means uncapped, 1 means immediately closed.
                // Legacy direct JSONL without `thinking_enabled` still infers
                // disable from `max_think==1` via the helper's fallback, but
                // new clients send `thinking_enabled` as authority and keep
                // `max_think_tokens` independent.
                //
                // When the cap is reached the daemon force-emits "</think>\n"
                // through the same KV-write + sample path as a normal token,
                // closing the thinking block so the model commits to an
                // answer with the remaining max_tokens budget. Caught by
                // Codex stop-time review on 2026-04-28: the field had been
                // shipping in the CLI's generate params but the daemon
                // was silently ignoring it, making the new reasoning.effort
                // / enable_thinking knobs no-ops on the wire.
                let max_think_tokens = msg
                    .get("max_think_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                // Derive Jinja `enable_thinking` and `reasoning_effort` via
                // pure helper. `thinking_enabled` is authoritative when
                // present; legacy effort/max_think inference is preserved
                // only for direct old JSONL clients that omit it.
                let (enable_thinking_jinja, reasoning_effort_jinja) =
                    qwen_jinja_reasoning(thinking_enabled, raw_reasoning_effort, max_think_tokens);
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
                let has_vl = m.vision_config().is_some() || m.dots_ocr().is_some();

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
                    // Same lifecycle authority as the "reset" command handler
                    // (LoadedModel::reset_context): host session/cache + arch
                    // state + speculator reset + graph/replay invalidate +
                    // sync attestation. VL only runs on qwen35-vl (arch_id
                    // 5|6) and dots-ocr (arch_id 8), but the authority resets
                    // every arch for defense-in-depth in case a future arch
                    // adds VL support.
                    if m.seq_pos > 0 {
                        eprintln!("[daemon/vl] non-zero seq_pos ({}) at VL dispatch — resetting conversation", m.seq_pos);
                        let reset = m.reset_context(&mut gpu, None, None);
                        if !reset.rolled_back {
                            let detail = reset
                                .context
                                .as_deref()
                                .unwrap_or("rollback could not be attested");
                            hipfire_generate::dense::emit_active_attempt_error(
                                &mut stdout,
                                Some(id),
                                &format!("vision conversation reset failed: {detail}"),
                                "gpu",
                                true,
                                false,
                            );
                            continue;
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
                        hipfire_loader::VisionRoute::DotsOcr => {
                            hipfire_generate::vision::generate_vl_dots_ocr(
                                m,
                                &mut gpu,
                                &mut stdout,
                                &params,
                            )
                        }
                        _ => {
                            hipfire_generate::vision::generate_vl(m, &mut gpu, &mut stdout, &params)
                        }
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
                            // Admission must match the DRIVER's capacity: the
                            // staged lane_capacity (attention-LDS-clamped) can
                            // be below m.max_seq, and the Qwen/EP assignment
                            // loops do not re-check length — a prompt in
                            // [lane_capacity, max_seq) would poison the batch.
                            let lane_capacity = batch_scheduler
                                .as_ref()
                                .map(|s| s.lane_capacity)
                                .unwrap_or(m.max_seq);
                            if prompt_tokens.is_empty() || prompt_tokens.len() >= lane_capacity {
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
                                original_msg: msg.clone(),
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
                            // Admission must match the DRIVER's capacity: the
                            // staged lane_capacity (attention-LDS-clamped) can
                            // be below m.max_seq, and the Qwen assignment loop
                            // does not re-check length — a prompt in
                            // [lane_capacity, max_seq) would poison the batch.
                            let lane_capacity = batch_scheduler
                                .as_ref()
                                .map(|s| s.lane_capacity)
                                .unwrap_or(m.max_seq);
                            // LFM lanes budget prompt + max_tokens (the refill
                            // gate `batch_lfm_exceeds_capacity` enforces the
                            // span); a prompt-only check would admit a request
                            // that then fails at refill AFTER gen_start/GPU
                            // work. Qwen/EP lanes are prompt-only — mirror the
                            // driver's respective gates.
                            let exceeds_capacity = if m.arch_id == 11 {
                                !hipfire_engine::terminal::batch_lfm_admission_ok(
                                    prompt_tokens.len(),
                                    max_tokens,
                                    lane_capacity,
                                )
                            } else {
                                prompt_tokens.is_empty() || prompt_tokens.len() >= lane_capacity
                            };
                            if exceeds_capacity {
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
                                original_msg: msg.clone(),
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
                        logprobs_top_k,
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
                    // Single lifecycle authority: host session/cache + arch
                    // state + speculator reset + graph/replay invalidate +
                    // sync attestation (same ordering as every fail-closed
                    // rollback, via LoadedModel::reset_context).
                    let reset = m.reset_context(&mut gpu, None, None);
                    if !reset.rolled_back {
                        let detail = reset
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
                if m.pp > 1 || m.ep.is_some() || m.tp_model.is_some() || m.pp_model.is_some() {
                    emit_uncorrelated_error(&mut stdout, None, "bench_prefill requires a single-GPU model (pp=1, non-EP, non-mesh); multi-GPU/EP/mesh bench not implemented", "unsupported", false, false);
                    let _ = stdout.flush();
                    continue;
                }
                let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                let capture = msg
                    .get("redline_capture")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let capture_detail = msg
                    .get("redline_detail")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
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
                // the bundle (qwen35 arch bundle, accessed via ArchModel);
                // `LoadedModel.dn_state` was removed (was always None) — the live
                // DeltaNet state is in the bundle.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                let _ = hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu);
                if let Some(st) = m.state.as_mut() {
                    let key = st.as_ref().arch_key();
                    let _ = st.as_mut().reset_session_state(&mut gpu);
                    if key == "deepseek4" {
                        gpu.invalidate_graph_state();
                    }
                }

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                if capture {
                    if let Err(reason) = gpu.replay.begin_capture() {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("redline prefill capture refused: {reason}"),
                            "unsupported",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                }
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
                        .expect(
                            "bench_prefill: carrier does not implement bench_prefill for this arch",
                        )
                };
                let _ = gpu.hip.device_synchronize();
                let elapsed = t0.elapsed().as_secs_f64();
                let capture_summary = if capture {
                    gpu.replay.finish_capture().map(Some)
                } else {
                    Ok(None)
                };

                // Reset state AFTER measurement — we've written N KV slots and a
                // DeltaNet state that the next real request must not inherit.
                // qwen35 recurrent state lives in the bundle (qwen35 arch bundle,
                // accessed via ArchModel); `LoadedModel.dn_state` was removed
                // (was always None).
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                let _ = hipfire_generate::common::reset_qwen35_recurrent(m, &mut gpu);
                if let Some(st) = m.state.as_mut() {
                    let key = st.as_ref().arch_key();
                    let _ = st.as_mut().reset_session_state(&mut gpu);
                    if key == "deepseek4" {
                        gpu.invalidate_graph_state();
                    }
                }
                let capture_summary = match capture_summary {
                    Ok(summary) => summary,
                    Err(reason) => {
                        emit_uncorrelated_error(
                            &mut stdout,
                            None,
                            &format!("redline prefill capture failed: {reason}"),
                            "internal",
                            false,
                            false,
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };

                if run_ok {
                    let tok_s = if elapsed > 0.0 {
                        n as f64 / elapsed
                    } else {
                        0.0
                    };
                    if let Some(summary) = capture_summary {
                        let response = serde_json::json!({
                            "type": "prefill_result",
                            "tokens": n,
                            "ms": elapsed * 1000.0,
                            "tok_s": tok_s,
                            "redline_capture": redline_capture_json(
                                &gpu,
                                summary,
                                capture_detail,
                            ),
                        });
                        let _ = writeln!(stdout, "{response}");
                    } else {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"prefill_result","tokens":{},"ms":{:.2},"tok_s":{:.1}}}"#,
                            n,
                            elapsed * 1000.0,
                            tok_s
                        );
                    }
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
                    || m.tp_model.is_some()
                    || m.pp_model.is_some()
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
                let prime_error: Option<String> =
                    match hipfire_loader::bench_decode_route(m.arch_id) {
                        hipfire_loader::BenchDecodeRoute::Qwen35
                        | hipfire_loader::BenchDecodeRoute::MuseGlimmer => {
                            hipfire_loader::carrier_for(m.arch_id)
                                .and_then(|c| c.bench_decode_prime(m, &mut gpu, &synthetic))
                                .unwrap_or_else(|| {
                                    Some(format!(
                            "bench_decode_prime: carrier missing or unimplemented for arch_id={}",
                            m.arch_id
                        ))
                                })
                        }
                        hipfire_loader::BenchDecodeRoute::Unsupported => Some(format!(
                            "bench_decode unsupported for arch_id={}",
                            m.arch_id
                        )),
                        _ => Some(format!(
                            "bench_decode unsupported for arch_id={}",
                            m.arch_id
                        )),
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
                    hipfire_loader::BenchDecodeRoute::Qwen35
                    | hipfire_loader::BenchDecodeRoute::MuseGlimmer => {
                        hipfire_loader::carrier_for(m.arch_id)
                            .and_then(|c| {
                                c.bench_decode_run(
                                    m,
                                    &mut gpu,
                                    context,
                                    iterations,
                                    &mut decode_err,
                                )
                            })
                            .unwrap_or(false)
                    }
                    hipfire_loader::BenchDecodeRoute::Unsupported => {
                        decode_err = Some(format!(
                            "bench_decode unsupported for arch_id={}",
                            m.arch_id
                        ));
                        false
                    }
                    _ => {
                        decode_err = Some(format!(
                            "bench_decode unsupported for arch_id={}",
                            m.arch_id
                        ));
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

            "redline_dflash_verify_shadow_pm4" => {
                handle_redline_dflash_verify_shadow_pm4(&msg, &mut model, &mut gpu, &mut stdout);
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
mod tests {
    use super::*;

    /// Minimal PlainQwen3 HFQ (arch 1, qwen3 config) — admission-classifiable
    /// without tensor payloads.
    fn hfq_fixture(dir: &std::path::Path) -> String {
        let path = dir.join("model.hfq");
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "rope_theta": 1_000_000.0,
        });
        let metadata = serde_json::json!({ "config": config }).to_string();
        hipfire_runtime::hfq::write_hfqm_package_mem(&path, 1, &metadata, &[])
            .expect("write hfq fixture");
        path.to_str().unwrap().to_string()
    }

    /// Minimal safetensors directory (config.json only) that classifies as
    /// PlainQwen3 — SafetensorsSource::open needs no tensor files.
    fn safetensors_dir_fixture(dir: &std::path::Path) -> String {
        let config = serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "rope_theta": 1_000_000.0,
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&config).expect("serialize config"),
        )
        .expect("write config.json");
        // SafetensorsSource::open requires at least one .safetensors file
        // with a parseable header; an empty-tensor file satisfies the probe
        // (payloads are never read).
        let mut st = vec![2u8, 0, 0, 0, 0, 0, 0, 0]; // header length = 2
        st.extend_from_slice(b"{}");
        std::fs::write(dir.join("model.safetensors"), st).expect("write safetensors file");
        dir.to_str().unwrap().to_string()
    }

    #[test]
    fn daemon_load_plan_admits_mesh_hfq_and_refuses_directory_sources() {
        let base = std::env::temp_dir().join(format!("hipfire-daemon-plan-{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("fixture dir");

        // HFQ sources admit for both mesh axes (the MeshCarrier route).
        let hfq = hfq_fixture(&base);
        let plan = daemon_load_plan(&hfq, 1, 2, 1).expect("HFQ TP admitted");
        assert!(matches!(
            &plan,
            DaemonLoadPlan::Admitted { effective, .. }
                if effective.axis() == ParallelAxis::Tp
        ));
        let plan = daemon_load_plan(&hfq, 2, 1, 1).expect("HFQ PP admitted");
        assert!(matches!(
            &plan,
            DaemonLoadPlan::Admitted { effective, .. }
                if effective.axis() == ParallelAxis::Pp
        ));

        // Directory sources stay legal for Single…
        let dir = safetensors_dir_fixture(&base);
        let plan = daemon_load_plan(&dir, 1, 1, 1).expect("dir Single admitted");
        assert!(matches!(
            &plan,
            DaemonLoadPlan::Admitted { effective, .. }
                if effective.axis() == ParallelAxis::Single
        ));
        // …but refuse for mesh axes HERE (preflight), canonically — the
        // loader's own HFQ-only refusal would fire only after the eager PP
        // prior-model unload.
        let err = match daemon_load_plan(&dir, 1, 2, 1) {
            Err(e) => e,
            Ok(_) => panic!("dir TP refused preflight"),
        };
        assert!(err.starts_with("[CAP-001]"), "{err}");
        assert!(err.contains("requires an HFQ model file"), "{err}");
        let err = match daemon_load_plan(&dir, 2, 1, 1) {
            Err(e) => e,
            Ok(_) => panic!("dir PP refused preflight"),
        };
        assert!(err.starts_with("[CAP-001]"), "{err}");
        assert!(err.contains("requires an HFQ model file"), "{err}");

        // The legacy qwen35 PP classification must refuse directory sources
        // too (that plan unloads eagerly — the carrier's own safetensors +
        // pp>1 refusal would fire only after the live model is gone).
        let dir = std::path::PathBuf::from(dir).join("qwen35");
        std::fs::create_dir_all(&dir).expect("qwen35 dir");
        let config = serde_json::json!({
            "architectures": ["Qwen3.5ForCausalLM"],
            "model_type": "qwen3_5",
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&config).expect("serialize qwen35 config"),
        )
        .expect("write qwen35 config.json");
        let mut st = vec![2u8, 0, 0, 0, 0, 0, 0, 0];
        st.extend_from_slice(b"{}");
        std::fs::write(dir.join("model.safetensors"), st).expect("write qwen35 safetensors");
        let err = match daemon_load_plan(dir.to_str().unwrap(), 2, 1, 1) {
            Err(e) => e,
            Ok(_) => panic!("qwen35 dir PP refused preflight"),
        };
        assert!(err.starts_with("[CAP-001]"), "{err}");
        assert!(err.contains("requires an HFQ model file"), "{err}");
    }
}
