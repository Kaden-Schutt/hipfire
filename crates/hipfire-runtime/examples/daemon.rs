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

use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_arch_llama::Llama;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, LayerType, Qwen35ScratchSet};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_arch_qwen35::speculative::{
    self, DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use hip_bridge::HipResult;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

/// Eviction policy wrapper — dispatches to plain TriAttention or CASK m-folding.
enum Eviction {
    Plain(EvictionCtx),
    Cask(CaskCtx),
}

impl Eviction {
    fn maybe_evict(
        &self,
        gpu: &mut rdna_compute::Gpu,
        kv: &mut llama::KvCache,
        physical: usize,
    ) -> HipResult<Option<hipfire_runtime::triattn::EvictionResult>> {
        match self {
            Eviction::Plain(c) => c.maybe_evict(gpu, kv, physical),
            Eviction::Cask(c) => c.maybe_evict(gpu, kv, physical),
        }
    }
    fn budget(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.budget,
            Eviction::Cask(c) => c.base.budget,
        }
    }
    fn beta(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.beta,
            Eviction::Cask(c) => c.base.beta,
        }
    }
    fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        match self {
            Eviction::Plain(c) => c.free_gpu(gpu),
            Eviction::Cask(c) => c.free_gpu(gpu),
        }
    }
}

/// CASK/TriAttention params forwarded by the CLI at load time. Zero-initialized
/// CaskConfig{sidecar: None, ..} means no eviction — matches 0.1.7-alpha behavior.
#[derive(Default)]
struct CaskConfig {
    sidecar: Option<String>,
    /// true = CASK m-folding; false = plain TriAttention drop-eviction.
    cask_m_folding: bool,
    budget: usize,
    beta: usize,
    core_frac: f32,
    fold_m: usize,
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
/// GPU-side attractor blockers for the AR generate path (#111).
///
/// MQ4 quant pressure makes structured-output special tokens (`<tool_call>`,
/// `<think>`) into self-reinforcing attractors: the model emits the same
/// special token hundreds of times in a row, never reaching the JSON body
/// (or in stacked-opener shapes that downstream regex parsers cannot
/// recover). The CPU-side `apply_ngram_block` is not in this path (its
/// per-token D2H + H2D would tank decode tok/s) and the GPU sampler's
/// repeat-penalty alone doesn't break a strong single-token loop fast
/// enough at the user-validated `RP=1.05` floor.
///
/// The unclosed-opener depth counter has moved to
/// `hipfire_runtime::sampler::collect_unclosed_attractor_blocks` (PR 3 of the
/// engine-modularization plan); the resulting blocked-token list is
/// applied to the GPU logits buffer by `hipfire_runtime::sampler::sample`
/// before the sampling kernel launches. The `gpu_block_attractor_token`
/// helper below is the simpler fallback for unpaired tokens — trips on
/// `count >= threshold` regardless of structure — kept here as
/// reference for a future per-token attractor block.
//
// ─── Probe-mode `committed` event emitter ────────────────────────────────
//
// When `HIPFIRE_EMIT_TOKEN_IDS=1` is set, the daemon emits a
// `{"type":"committed",...}` event for every token it commits (i.e. every
// time a sampled token is appended to `streamed_tokens` /
// `conversation_tokens`). This is a parallel stream alongside the
// existing `{"type":"token","text":"..."}` events; it carries the raw
// token ID, the per-request position, and ms-since-request-start.
//
// Why a parallel stream and not a `tok_id` field on the existing token
// event: `EosFilter` can hold/merge/strip/stop bytes across multiple
// committed tokens (many-to-one and zero-to-one relationships); a
// `tok_id` field on a text event would lie about which token produced
// the visible chunk. The runtime-protective synthetic emit at the
// `</think>` force-close site is intentionally NOT paired with a
// `committed` event, because no token was actually committed there.
//
// Off by default — env var read once on first call. The probe binary
// (`examples/coherence_probe.rs`) sets the env on the daemon child it
// spawns. Existing JSONL clients see no change.
fn emit_committed_event(
    stdout: &mut std::io::Stdout,
    id: &str,
    tok_id: u32,
    pos: usize,
    t_ms: u64,
) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let on = *ENABLED.get_or_init(|| {
        std::env::var("HIPFIRE_EMIT_TOKEN_IDS").ok().as_deref() == Some("1")
    });
    if !on {
        return;
    }
    let _ = writeln!(
        stdout,
        r#"{{"type":"committed","id":"{}","tok_id":{},"pos":{},"t_ms":{}}}"#,
        id, tok_id, pos, t_ms
    );
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
    if window == 0 || threshold == 0 { return; }
    let start = history.len().saturating_sub(window);
    let count = history[start..].iter().filter(|&&t| t == tok_id).count();
    if count >= threshold {
        let bytes: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        let _ = gpu.hip.memcpy_htod_offset(logits_buf, (tok_id as usize) * 4, &bytes);
    }
}

fn acquire_daemon_lock() -> std::fs::File {
    use std::io::{Seek, Write};

    #[cfg(unix)]
    let home = std::env::var("HOME").expect("HOME environment variable not set");
    #[cfg(windows)]
    let home = std::env::var("USERPROFILE")
        .expect("USERPROFILE environment variable not set");

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

const IMAGE_SIZE: usize = 448;
const IMAGE_PAD_ID: u32 = 248056;
const VISION_START_ID: u32 = 248053;
const VISION_END_ID: u32 = 248054;

/// Optional DFlash speculative-decoding state. Populated when `load` supplies
/// a matching draft (.hfq arch=20) via `params.draft`. Used by the daemon's
/// `generate` fast path when temperature == 0 — falls back to AR sampling
/// otherwise (DFlash is greedy-only in this integration).
struct DflashState {
    draft_config: DflashConfig,
    draft_weights: DflashWeights,
    draft_scratch: DflashScratch,
    hidden_rb: HiddenStateRingBuffer,
    verify_scratch: VerifyScratch,
    target_snap: DeltaNetSnapshot,
    gdn_tape: GdnTape,
    /// CPU-side ring of target hidden states (num_extract × hidden per pos)
    /// — seeded from the prompt, extended by each verify's accepted rows.
    /// Drives the draft's diffusion forward.
    target_hidden_host: Vec<f32>,
    /// Max ctx the draft was initialized for (ring buffer cap).
    ctx_capacity: usize,
    /// Block size the draft was trained at.
    block_size: usize,
    /// Optional DDTree state. Populated only when `HIPFIRE_DDTREE_BUDGET` is
    /// set to a positive integer at daemon startup. None = DDTree disabled,
    /// the decode loop falls through to `spec_step_dflash` (chain mode).
    /// See `spec_step_ddtree_batched` for the tree-verify path.
    ddtree: Option<DdtreeState>,
}

/// Side state for DDTree-mode speculative decoding. Allocated alongside
/// the rest of `DflashState` at model-load time when DDTree is enabled,
/// reused across all decode cycles.
struct DdtreeState {
    /// Second DeltaNetSnapshot used by `spec_step_ddtree_batched`: snap0 =
    /// pre-seed (lives in `DflashState::target_snap`), snap1 = post-seed.
    /// The batched verify forward uses both to bracket the tree-verify pass.
    post_seed_snap: DeltaNetSnapshot,
    /// Persistent tree-verify scratch (attn_bias, parent_indices, kv-gather
    /// staging, pre-RoPE K capture). Sized for `budget` non-root nodes.
    scratch: DdtreeScratch,
    /// Maximum non-root tree nodes per cycle. Read once at daemon startup
    /// from `HIPFIRE_DDTREE_BUDGET` (positive integer required to enable).
    budget: usize,
    /// Per-position top-K width fed into the DDTree builder. Read from
    /// `HIPFIRE_DDTREE_TOPK` (default 4 — matches paper Algorithm 1's
    /// typical setting on dense Qwen targets).
    topk: usize,
    /// Path C Phase 2 auxiliary snapshots. Used only when
    /// `HIPFIRE_DDTREE_PATH_C=phase2`. Allocated unconditionally when DDTree
    /// is enabled — DN state buffers are small (a few KB each on 27B) and
    /// avoiding the gate keeps allocation deterministic at session start.
    /// See `speculative::Phase2Snapshots` for what each snapshot holds.
    path_c_parent_pre_snap: DeltaNetSnapshot,
    path_c_main_end_snap: DeltaNetSnapshot,
}

struct LoadedModel {
    arch_id: u32,
    /// Pipeline-parallel degree. 1 = single-GPU (all existing fields below in
    /// use, q35_scratch populated). >1 = multi-GPU (pp_gpus + pp_scratch_set
    /// populated; q35_scratch stays None; kv_cache + dn_state still hold the
    /// per-layer-routed tensors since the struct types are the same as
    /// single-GPU). Refusal contracts in load_model_pp keep DFlash, CASK,
    /// PFlash, VL and arch_id < 5 out of this branch.
    pp: usize,
    /// Owned multi-GPU orchestrator when `pp > 1`. The single-GPU path
    /// continues to use the daemon's main `Gpu` directly.
    pp_gpus: Option<Gpus>,
    /// Per-device scratch when `pp > 1`. Replaces `q35_scratch`.
    pp_scratch_set: Option<Qwen35ScratchSet>,
    /// LA-layer → device map returned by `DeltaNetState::new_with_quant_multi`,
    /// kept so `unload_model` and the reset handler can route per-layer
    /// memsets to the correct device.
    pp_dn_la_to_device: Option<Vec<u8>>,
    // Qwen3.5 state
    q35_config: Option<qwen35::Qwen35Config>,
    q35_weights: Option<qwen35::Qwen35Weights>,
    q35_scratch: Option<qwen35::Qwen35Scratch>,
    kv_cache: Option<llama::KvCache>,
    dn_state: Option<DeltaNetState>,
    // Qwen3 state
    llama_config: Option<llama::LlamaConfig>,
    llama_weights: Option<llama::LlamaWeights>,
    llama_scratch: Option<llama::ForwardScratch>,
    llama_kv: Option<llama::KvCache>,
    // Vision state (VL models only)
    vision_config: Option<qwen35_vl::VisionConfig>,
    vision_weights: Option<qwen35_vl::VisionWeights>,
    // Shared
    tokenizer: Option<hipfire_runtime::tokenizer::Tokenizer>,
    // Multi-turn conversation state
    //
    // `seq_pos` is the *physical* write position in the KV cache (the value
    // passed to `forward_scratch(..., pos, ...)`). With no eviction, physical
    // == absolute, so seq_pos simply grows. Under eviction, seq_pos is bounded
    // to `physical_cap`; absolute position = seq_pos + kv.compact_offset.
    seq_pos: usize,
    /// Advertised context window — client-facing capacity, the upper bound on
    /// absolute conversation length. Without eviction this equals
    /// `physical_cap` (the buffer size); under eviction it can be much larger.
    max_seq: usize,
    /// Physical KV buffer capacity, in slots. Allocators size per-layer K/V
    /// for this many tokens. Under eviction, budget+beta <= physical_cap;
    /// without eviction, physical_cap == max_seq.
    physical_cap: usize,
    /// When Some(_), the daemon calls `maybe_evict` after every prefill-chunk
    /// and every decode-forward so the physical cache stays bounded by
    /// `physical_cap` even when `max_seq` advertises a much larger window.
    eviction: Option<Eviction>,
    conversation_tokens: Vec<u32>, // full token history for repeat penalty
    // Target model file path — cached so the DFlash fast path can reopen the
    // HfqFile mmap to construct a transient ModelSlot without reloading
    // weights. `HfqFile::open` is a cheap mmap operation.
    model_path: String,
    // DFlash speculative decoding state (populated when load supplied a draft).
    dflash: Option<DflashState>,
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

    // --precompile: compile all kernels for this GPU, write hash files, exit.
    // Used by scripts/install.sh and `hipfire update` so first `hipfire run`
    // isn't a 2-minute hipcc wait.
    //
    // Covers the current default path (mq4 weights + asym3 KV) plus the legacy
    // compat paths (hfq4, hfq6, q8 weights × asym3, q8 KV) so models from any
    // era of the registry start instantly.
    if args.iter().any(|a| a == "--precompile") {
        // Pre-create the expected precompiled-dir next to this binary so the
        // compiler's writeback path fires. Without this, Gpu::init probes for
        // an existing dir and silently disables writeback if it's missing —
        // meaning fresh installs would compile but never cache cross-invocation.
        if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
            // Arch is unknown until Gpu::init; use a broad mkdir for the common arches
            // we support so the probe picks one up. The real arch check after init
            // will log the active dir.
            for arch in ["gfx906", "gfx1010", "gfx1013", "gfx1030", "gfx1031", "gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1152", "gfx1200", "gfx1201"] {
                let _ = std::fs::create_dir_all(exe_dir.join("kernels").join("compiled").join(arch));
            }
        }
        let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => { report_gpu_init_failure(&e); std::process::exit(1); }
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

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => { report_gpu_init_failure(&e); std::process::exit(1); }
    };
    let mut model: Option<LoadedModel> = None;
    // PFlash speculative-prefill state. None unless the load message
    // includes a `prefill_drafter` path AND `prefill_compression` != "off".
    // Lives alongside `model` so unload_model + this state are paired
    // teardowns.
    let mut pflash_state: Option<hipfire_arch_qwen35::pflash::PflashState> = None;
    // The PflashConfig captured at load time. Per-request `prefill_*`
    // params override individual fields; the rest fall back to these
    // load-time defaults. Cleared alongside `pflash_state`.
    let mut pflash_cfg: Option<hipfire_arch_qwen35::pflash::PflashConfig> = None;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() { continue; }

        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(stdout, r#"{{"type":"error","message":"invalid JSON: {}"}}"#, e);
                let _ = stdout.flush();
                continue;
            }
        };

        let msg_type = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match msg_type {
            "load" => {
                // Unload previous if any. PFlash drafter goes first so
                // its tensors join the pool before unload_model drains
                // it -- otherwise free_tensor would queue them into the
                // pool just-emptied by drain_pool with no follow-up
                // drain, leaving drafter VRAM resident across the next
                // load (the explicit "unload" handler has the same
                // ordering for the same reason).
                if let Some(mut pf) = pflash_state.take() {
                    pf.unload_drafter(&mut gpu);
                }
                pflash_cfg = None;
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }

                let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let max_seq = msg.get("params").and_then(|p| p.get("max_seq")).and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
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
                let dflash_mode = msg.get("params").and_then(|p| p.get("dflash_mode"))
                    .and_then(|v| v.as_str()).unwrap_or("auto");
                let raw_draft = msg.get("params").and_then(|p| p.get("draft")).and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty());
                let draft_path = if dflash_mode == "off" {
                    if raw_draft.is_some() {
                        eprintln!("[hipfire-daemon] dflash_mode=off — skipping draft load ({})", raw_draft.unwrap());
                    }
                    None
                } else {
                    raw_draft.map(|s| s.to_string())
                };
                let kv_mode_override = msg.get("params").and_then(|p| p.get("kv_mode")).and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty()).map(|s| s.to_string());

                // 0.1.7-alpha: DFlash tuning knobs forwarded from the CLI.
                // `adaptive_b` matches dflash_spec_demo's --adaptive-b default.
                // Accepted here; the generate loop will honor it in the
                // 0.1.7-stable release where we port the demo's outer τ-window
                // trip-wire (below 2.5 → shrink block to 8).
                let _adaptive_b = msg.get("params").and_then(|p| p.get("dflash_adaptive_b"))
                    .and_then(|v| v.as_bool()).unwrap_or(true);

                // 0.1.7: TriAttention / CASK eviction protocol fields. When
                // `cask_sidecar` is set, `load_model` sizes the KV cache to a
                // *physical_cap* (budget+beta+safety, clamped to max_seq) instead
                // of the full max_seq, and wires an `Eviction` policy that the
                // generate loop calls after every prefill-chunk / decode-forward.
                // That decouples advertised context length from VRAM footprint —
                // a 128K max_seq can run in ~1K-slot physical buffer when the
                // operator opts in.
                let cask_sidecar = msg.get("params").and_then(|p| p.get("cask_sidecar"))
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
                let cask_enabled = msg.get("params").and_then(|p| p.get("cask"))
                    .and_then(|v| v.as_bool()).unwrap_or(false);
                let cask_budget = msg.get("params").and_then(|p| p.get("cask_budget"))
                    .and_then(|v| v.as_u64()).unwrap_or(512) as usize;
                let cask_beta = msg.get("params").and_then(|p| p.get("cask_beta"))
                    .and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                let cask_core_frac = msg.get("params").and_then(|p| p.get("cask_core_frac"))
                    .and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                let cask_fold_m = msg.get("params").and_then(|p| p.get("cask_fold_m"))
                    .and_then(|v| v.as_u64()).unwrap_or(2) as usize;
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
                // weights. Enabled by default; disable with mmq_screen=false.
                if let Some(v) = msg.get("params").and_then(|p| p.get("mmq_screen")).and_then(|v| v.as_bool()) {
                    gpu.mmq_screen = v;
                }
                if let Some(v) = msg.get("params").and_then(|p| p.get("mmq_screen_threshold")).and_then(|v| v.as_f64()) {
                    gpu.mmq_screen_threshold = v as f32;
                }

                // ── PFlash load-time params (Phase 4.0 #93) ──────────────
                //
                // Parse compression knobs per PRD §5.3.2. None of these
                // affect the target load itself; they only configure the
                // optional drafter that PFlash uses for prompt scoring.
                // Drafter loading happens AFTER target load succeeds so
                // we can use the target's tokenizer for the compat check.
                let pflash_mode_str = msg.get("params").and_then(|p| p.get("prefill_compression"))
                    .and_then(|v| v.as_str()).unwrap_or("off").to_string();
                let pflash_threshold = msg.get("params").and_then(|p| p.get("prefill_threshold"))
                    .and_then(|v| v.as_u64()).unwrap_or(32768) as usize;
                let pflash_keep_ratio = msg.get("params").and_then(|p| p.get("prefill_keep_ratio"))
                    .and_then(|v| v.as_f64()).unwrap_or(0.05) as f32;
                let pflash_alpha = msg.get("params").and_then(|p| p.get("prefill_alpha"))
                    .and_then(|v| v.as_f64()).unwrap_or(0.85) as f32;
                let pflash_min_keep = msg.get("params").and_then(|p| p.get("prefill_min_keep"))
                    .and_then(|v| v.as_u64()).unwrap_or(2048) as usize;
                let pflash_sink = msg.get("params").and_then(|p| p.get("prefill_sink"))
                    .and_then(|v| v.as_u64()).unwrap_or(256) as usize;
                let pflash_recent = msg.get("params").and_then(|p| p.get("prefill_recent"))
                    .and_then(|v| v.as_u64()).unwrap_or(1024) as usize;
                let pflash_block = msg.get("params").and_then(|p| p.get("prefill_block"))
                    .and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                let pflash_drafter = msg.get("params").and_then(|p| p.get("prefill_drafter"))
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string());
                let pflash_profile = msg.get("params").and_then(|p| p.get("prefill_profile"))
                    .and_then(|v| v.as_bool()).unwrap_or(false);
                let pflash_sparse_threshold = msg.get("params").and_then(|p| p.get("prefill_sparse_threshold"))
                    .and_then(|v| v.as_u64()).unwrap_or(32768) as usize;

                // Validate load-time PFlash params before they reach
                // PflashConfig + load_drafter. Same range rules the
                // per-request override path uses; without these, a
                // bad load-time value would silently be accepted and
                // panic the daemon at the first generate request.
                let pflash_load_err: Option<String> =
                    if !(pflash_keep_ratio > 0.0 && pflash_keep_ratio <= 1.0) {
                        Some(format!("prefill_keep_ratio={pflash_keep_ratio} not in (0, 1]"))
                    } else if pflash_block == 0 {
                        Some("prefill_block must be > 0".to_string())
                    } else { None };

                // Pipeline-parallel degree (Stage 7 of #58). Default 1 =
                // single-GPU (no behavior change). pp > 1 routes through
                // Gpus + *_multi paths and refuses VL / DFlash / CASK /
                // PFlash at load time. v1 supports Qwen3.5 dense + MoE
                // only — see load_model_pp for the arch_id check.
                let pp = msg.get("params").and_then(|p| p.get("pp"))
                    .and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                if pp > 1 {
                    if draft_path.is_some()
                        && std::env::var("HIPFIRE_PP_DFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap."}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                    if cask.sidecar.is_some() {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                    if (pflash_drafter.is_some() || pflash_mode_str != "off")
                        && std::env::var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                }

                match load_model(path, max_seq, draft_path.as_deref(), kv_mode_override.as_deref(), &cask, pp, &mut gpu) {
                    Ok(m) => {
                        let arch = match m.arch_id {
                            5 => "qwen3_5",
                            6 => "qwen3_5_moe",
                            _ => "qwen3",
                        };
                        let vl = m.vision_config.is_some();
                        let (dim, layers, vocab) = if let Some(ref c) = m.q35_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else if let Some(ref c) = m.llama_config {
                            (c.dim, c.n_layers, c.vocab_size)
                        } else { (0, 0, 0) };
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

                        let _ = writeln!(stdout, r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{}}}"#, arch, dim, layers, vocab, vl);

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
                                    let _ = writeln!(stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"invalid load param: {}"}}"#,
                                        reason.replace('"', "'"));
                                    let _ = stdout.flush();
                                    model = Some(m);
                                    continue;
                                }
                                let pf_cfg = hipfire_arch_qwen35::pflash::PflashConfig {
                                    mode: hipfire_arch_qwen35::pflash::PflashMode::parse(&pflash_mode_str)
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
                                let mut pf_state = hipfire_arch_qwen35::pflash::PflashState::new(&pf_cfg);
                                // Pull the target tokenizer out of the loaded model
                                // for the compat check. Both Qwen3.5 and plain
                                // Qwen3 paths expose `tokenizer` on LoadedModel.
                                let tgt_tok_ref = m.tokenizer.as_ref();
                                if let Some(tok) = tgt_tok_ref {
                                    let pf_max_kv = max_seq.max(2048);
                                    match hipfire_arch_qwen35::pflash::load_drafter(
                                        &mut pf_state, &mut gpu,
                                        std::path::Path::new(pf_drafter_path),
                                        tok, pf_max_kv,
                                    ) {
                                        Ok(()) => {
                                            let _ = writeln!(stdout,
                                                r#"{{"type":"pflash","mode":"{}","drafter":"{}","tokenizer_compat":{},"keep_ratio":{},"threshold":{}}}"#,
                                                pflash_mode_str, pf_drafter_path,
                                                pf_state.tokenizer_compat,
                                                pflash_keep_ratio, pflash_threshold);
                                            pflash_state = Some(pf_state);
                                            pflash_cfg = Some(pf_cfg);
                                        }
                                        Err(e) => {
                                            let _ = writeln!(stdout,
                                                r#"{{"type":"pflash_load_failed","reason":"{}"}}"#,
                                                e.to_string().replace('"', "'"));
                                        }
                                    }
                                } else {
                                    let _ = writeln!(stdout,
                                        r#"{{"type":"pflash_load_failed","reason":"target tokenizer unavailable"}}"#);
                                }
                            }
                        }

                        model = Some(m);
                    }
                    Err(e) => {
                        let (vram_free, vram_total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
                        let free_mb = vram_free / (1024 * 1024);
                        let total_mb = vram_total / (1024 * 1024);
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"load failed: {}. GPU: {} ({} MB free / {} MB total)"}}"#, e, gpu.arch, free_mb, total_mb);
                    }
                }
                let _ = stdout.flush();
            }

            "generate" => {
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                };

                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("0");
                let prompt = msg.get("prompt").and_then(|v| v.as_str()).unwrap_or("Hello");
                let prompt_norm = hipfire_runtime::tokenizer::maybe_normalize_prompt(prompt);
                let prompt: &str = &prompt_norm;
                if std::env::var("HIPFIRE_PROMPT_TOKEN_HEAT").ok().as_deref() == Some("1") {
                    if let Some(tok) = m.tokenizer.as_ref() { tok.dump_prompt_heat(prompt); }
                }
                let system = msg.get("system").and_then(|v| v.as_str());
                let image = msg.get("image").and_then(|v| v.as_str());
                let temp = msg.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.3) as f32;
                let max_tokens = msg.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
                let top_p = msg.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
                let repeat_penalty = msg.get("repeat_penalty").and_then(|v| v.as_f64()).unwrap_or(1.3) as f32;
                let repeat_window = msg.get("repeat_window").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
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
                let experimental_ok = std::env::var("HIPFIRE_EXPERIMENTAL_BUDGET_ALERT").ok().as_deref() == Some("1");
                let budget_alert_at_tok = if experimental_ok {
                    msg.get("budget_alert_at_tok").and_then(|v| v.as_u64()).unwrap_or(0) as usize
                } else { 0 };
                let budget_alert_text = if experimental_ok {
                    msg.get("budget_alert_text").and_then(|v| v.as_str()).unwrap_or("").to_string()
                } else { String::new() };
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
                let max_think_tokens = msg.get("max_think_tokens")
                    .and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                if image.is_some() && m.vision_config.is_some() {
                    generate_vl(m, &mut gpu, &mut stdout, id, prompt, system, image.unwrap(), temp, top_p, max_tokens, repeat_penalty, repeat_window);
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
                        if let Some(s) = msg.get("params").and_then(|p| p.get("prefill_compression")).and_then(|v| v.as_str()) {
                            if let Some(m) = hipfire_arch_qwen35::pflash::PflashMode::parse(s) { c.mode = m; }
                        }
                        if let Some(v) = msg.get("params").and_then(|p| p.get("prefill_threshold")).and_then(|v| v.as_u64()) {
                            c.threshold_tokens = v as usize;
                        }
                        if let Some(v) = msg.get("params").and_then(|p| p.get("prefill_keep_ratio")).and_then(|v| v.as_f64()) {
                            let r = v as f32;
                            if !(r > 0.0 && r <= 1.0) {
                                pf_override_err = Some(format!(
                                    "prefill_keep_ratio={r} not in (0, 1]"));
                            } else {
                                c.keep_ratio = r;
                            }
                        }
                        if let Some(v) = msg.get("params").and_then(|p| p.get("prefill_min_keep")).and_then(|v| v.as_u64()) {
                            c.min_keep_tokens = v as usize;
                        }
                        if let Some(v) = msg.get("params").and_then(|p| p.get("prefill_sink")).and_then(|v| v.as_u64()) {
                            c.sink_tokens = v as usize;
                        }
                        if let Some(v) = msg.get("params").and_then(|p| p.get("prefill_recent")).and_then(|v| v.as_u64()) {
                            c.recent_tokens = v as usize;
                        }
                        if let Some(v) = msg.get("params").and_then(|p| p.get("prefill_block")).and_then(|v| v.as_u64()) {
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
                            id, reason.replace('"', "'"),
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    generate(
                        m, &mut gpu, &mut stdout, id, prompt, system,
                        temp, top_p, max_tokens, repeat_penalty, repeat_window,
                        budget_alert_at_tok, &budget_alert_text, max_think_tokens,
                        pflash_state.as_mut(),
                        pf_cfg_owned.as_ref(),
                    );
                }
            }

            "reset" => {
                // Reset conversation state without unloading the model.
                // Under eviction, also zero the compact_offset so absolute
                // RoPE phase restarts from zero for the fresh conversation.
                if let Some(ref mut m) = model {
                    m.seq_pos = 0;
                    m.conversation_tokens.clear();
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
                    if let Some(kv) = m.kv_cache.as_mut() { kv.compact_offset = 0; }
                    if let Some(kv) = m.llama_kv.as_mut() { kv.compact_offset = 0; }
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
                    pf.unload_drafter(&mut gpu);
                }
                pflash_cfg = None;
                if let Some(m) = model.take() {
                    unload_model(m, &mut gpu);
                }
                let _ = writeln!(stdout, r#"{{"type":"unloaded"}}"#);
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
                let model_arch = model.as_ref().map(|m| match m.arch_id {
                    5 => "qwen3_5",
                    6 => "qwen3_5_moe",
                    _ => "qwen3",
                }).unwrap_or("none");
                // Count pre-compiled kernels
                let kernel_dir = std::env::current_exe().ok()
                    .and_then(|e| e.parent().map(|p| p.join("kernels").join("compiled").join(&gpu.arch)))
                    .filter(|p| p.is_dir());
                let (hsaco_count, hash_count) = kernel_dir.map(|d| {
                    let hsaco = std::fs::read_dir(&d).map(|r| r.filter(|e| e.as_ref().ok().map(|e| e.path().extension().map(|x| x == "hsaco").unwrap_or(false)).unwrap_or(false)).count()).unwrap_or(0);
                    let hash = std::fs::read_dir(&d).map(|r| r.filter(|e| e.as_ref().ok().map(|e| e.path().extension().map(|x| x == "hash").unwrap_or(false)).unwrap_or(false)).count()).unwrap_or(0);
                    (hsaco, hash)
                }).unwrap_or((0, 0));
                let _ = writeln!(stdout,
                    r#"{{"type":"diag","arch":"{}","hip_version":"{}.{}","vram_free_mb":{},"vram_total_mb":{},"model_loaded":{},"model_arch":"{}","kernels":{},"kernel_hashes":{}}}"#,
                    gpu.arch, hip_ver.0, hip_ver.1,
                    vram_free / (1024 * 1024), vram_total / (1024 * 1024),
                    has_model, model_arch, hsaco_count, hash_count
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
                        let _ = writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
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
                    let _ = writeln!(stdout,
                        r#"{{"type":"error","message":"bench_prefill requires pp=1 (multi-GPU bench not implemented)"}}"#);
                    let _ = stdout.flush();
                    continue;
                }
                let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                // Guard physical_cap — reserve 32 slots of headroom so a subsequent
                // generate request against the loaded model still has room. We guard
                // on the *physical* buffer (not the advertised max_seq) because this
                // bench intentionally bypasses eviction to measure raw prefill.
                if n + 32 > m.physical_cap {
                    let _ = writeln!(stdout,
                        r#"{{"type":"error","message":"bench_prefill tokens={} exceeds loaded physical_cap={}"}}"#,
                        n, m.physical_cap);
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
                    for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                }

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                let _ = gpu.hip.device_synchronize();
                let t0 = Instant::now();
                let run_ok = if m.arch_id == 5 || m.arch_id == 6 {
                    let config = m.q35_config.as_ref().unwrap();
                    let weights = m.q35_weights.as_ref().unwrap();
                    let scratch = m.q35_scratch.as_ref().unwrap();
                    let kv = m.kv_cache.as_mut().unwrap();
                    let dn = m.dn_state.as_mut().unwrap();
                    qwen35::forward_prefill_batch(&mut gpu, weights, config, &synthetic, 0, kv, dn, scratch, None, None, None, None).is_ok()
                } else {
                    let config = m.llama_config.as_ref().unwrap();
                    let weights = m.llama_weights.as_ref().unwrap();
                    let scratch = m.llama_scratch.as_ref().unwrap();
                    let kv = m.llama_kv.as_mut().unwrap();
                    let mut ok = true;
                    for (i, &tok) in synthetic.iter().enumerate() {
                        if llama::forward_scratch(&mut gpu, weights, config, tok, i, kv, scratch, 0.0, 1.0, 42, 0, 1.0).is_err() {
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
                    for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                    for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
                }

                if run_ok {
                    let tok_s = if elapsed > 0.0 { n as f64 / elapsed } else { 0.0 };
                    let _ = writeln!(stdout,
                        r#"{{"type":"prefill_result","tokens":{},"ms":{:.2},"tok_s":{:.1}}}"#,
                        n, elapsed * 1000.0, tok_s);
                } else {
                    let _ = writeln!(stdout, r#"{{"type":"error","message":"bench_prefill forward failed"}}"#);
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
                let _ = writeln!(stdout,
                    r#"{{"type":"profile","gpu":{},"kernels":[{}]}}"#,
                    cap.to_json(), kernels_json.join(",")
                );
                let _ = stdout.flush();
            }

            _ => {
                let _ = writeln!(stdout, r#"{{"type":"error","message":"unknown type: {}"}}"#, msg_type);
                let _ = stdout.flush();
            }
        }
    }
}

fn load_model(path: &str, max_seq: usize, draft_path: Option<&str>, kv_mode_override: Option<&str>, cask: &CaskConfig, pp: usize, gpu: &mut rdna_compute::Gpu) -> Result<LoadedModel, String> {
    if pp > 1 {
        // Refusal contracts (DFlash, CASK sidecar) are enforced upstream in
        // the "load" event handler so the operator gets a structured error
        // before any HFQ open / weight allocation. By the time we get here
        // with pp>1, draft_path is None and cask.sidecar is None.
        let _ = (draft_path, cask);
        return load_model_pp(path, max_seq, kv_mode_override, pp, gpu);
    }
    // Per-load kv_mode (sent in load message params) overrides the env var.
    // Lets the CLI set size-aware defaults — e.g. Qwen3.5-27B prefers asym4
    // since layer-count compounding of asym3 noise flips argmax at decision
    // boundaries on deep stacks.
    let kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("tokenizer not found")?;

    // DFlash speculative-decode requires the target's lm_head to have a
    // batched-GEMM kernel (used for verify and DDTree top-K). Only
    // Q8_0 (qt=3) / HFQ4G256 (qt=6) / MQ4G256 (qt=13) are wired into
    // speculative.rs's `try_batched` predicate (lines 2083-2087,
    // 2606-2609); every other dtype falls through to a per-row sequential
    // GEMV path that hangs spec verify (observed: 1 token in 240 s on
    // 27B MQ3 + dflash-mq4 draft).
    //
    // Refuse fast at the HFQ-index level — BEFORE any weight upload, KV
    // alloc, or scratch alloc — so we don't strand ~12 GB of VRAM in the
    // pool when the operator passed a draft against an unsupported target.
    // Read the lm_head tensor's `quant_type` byte directly from the index
    // (no GPU work). lm_head can be a separate tensor or tied to
    // embed_tokens, and the tensor names differ by arch:
    //   - Qwen3.5/3.6 separate: "lm_head.weight" or "model.language_model.lm_head.weight"
    //   - Qwen3.5/3.6 tied:     "model.language_model.embed_tokens.weight"
    //   - LLaMA separate:       "lm_head.weight"
    //   - LLaMA tied:           "model.embed_tokens.weight"
    // Cover all four; the order mirrors what qwen35::load_weights /
    // hfq::load_weights_hfq do at runtime, so the qt we read here is the
    // qt that will end up driving `weights.output.gpu_dtype`.
    if draft_path.is_some() {
        let lm_qt = hfq.tensor_data("lm_head.weight")
            .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
            .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
            .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
            .map(|(info, _)| info.quant_type);
        // MQ3 (qt=17) batched lm_head + WMMA prefill kernels exist on gfx11
        // only (`gemm_hfq3g256_batched_lmhead` + `is_batchable_la` admits MQ3
        // for gfx1100/1101/1102/1150/1151). On other archs, MQ3 lm_head still
        // falls through to per-row GEMV that hangs verify. Whitelist:
        //   - Always: Q8_0=3, HFQ4G256=6, MQ4G256=13
        //   - gfx11 only: MQ3G256=17
        // MQ2 (qt=18) is not yet wired into speculative.rs match arms.
        // MQ3 WMMA family is ported to gfx11 (RDNA3) and gfx12 (RDNA4).
        // Keep them grouped under the same flag — the builtin name differs
        // (_w32 vs _w32_gfx12) but the dispatch wrappers route per-arch.
        let arch_is_gfx11 = matches!(
            gpu.arch.as_str(),
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
            | "gfx1200" | "gfx1201"
        );
        let supported = match lm_qt {
            Some(3 | 6 | 13) => true,
            Some(17) => arch_is_gfx11,
            _ => false,
        };
        if !supported {
            let qt_desc = match lm_qt {
                Some(qt) => format!("quant_type={qt}"),
                None => "no lm_head/embed_tokens tensor found at any known name".to_string(),
            };
            return Err(format!(
                "DFlash draft requested but target lm_head {} is not \
                 supported by speculative.rs's batched GEMM paths on this arch \
                 ({}). Supported: Q8_0 (qt=3), HFQ4G256 (qt=6), MQ4G256 (qt=13) \
                 always; MQ3G256 (qt=17) on gfx11 only. Other dtypes \
                 (MQ2 qt=18, MQ6/MQ8, HFQ3/HFQ2, HFQ4G128, HFQ6, F16, …) fall \
                 through to a per-row GEMV that hangs verify. Reload without a \
                 draft, or use an MQ4 / HFQ4 / Q8 target. (PRD Phase 2: extend \
                 speculative.rs match arms + add gemm_*_batched_lmhead kernels \
                 for the remaining dtypes.)",
                qt_desc, gpu.arch
            ));
        }

        // Defense-in-depth: refuse if any body weight is MQ2 (qt=18). MQ3
        // is now allowed on gfx11 dense (arch_id=5) because the WMMA prefill
        // family (qkvza/qkv/gate_up/residual hfq3) and
        // `gemm_hfq3g256_batched_lmhead` are wired. MQ3 is REFUSED on:
        //   - non-gfx11 archs (no batched WMMA prefill kernels)
        //   - MoE/A3B targets (arch_id=6) — the MoE LA/FA prefill branches
        //     and `moe_ffn_all_mq4` predicate are MQ4-only; MQ3 weights
        //     would silently fall through to HFQ4 kernels with the wrong
        //     104-vs-136 byte stride. (Future: wire MQ3 into the MoE
        //     batched branches and the MoE FFN expert kernels.)
        // MQ2 body still has no batched WMMA kernels anywhere.
        let arch_is_dense_qwen35 = hfq.arch_id == 5;
        let mq3_supported = arch_is_gfx11 && arch_is_dense_qwen35;
        let mq_unsupported = hfq.first_tensor_with_quant_type(18).map(|n| ("MQ2 (qt=18)", n));
        let mq_unsupported = mq_unsupported.or_else(|| {
            if !mq3_supported {
                hfq.first_tensor_with_quant_type(17).map(|n| ("MQ3 (qt=17)", n))
            } else {
                None
            }
        });
        if let Some((qt_label, name)) = mq_unsupported {
            let arch_reason = if !arch_is_dense_qwen35 && qt_label.starts_with("MQ3") {
                format!("arch_id={} (MoE/A3B-class) has no MQ3 MoE kernels", hfq.arch_id)
            } else {
                format!("arch={} lacks the corresponding batched WMMA prefill family", gpu.arch)
            };
            return Err(format!(
                "DFlash draft requested but model contains {qt_label} weight \
                 `{name}` and {arch_reason}. The prefill fast-path falls back \
                 to per-token `forward_scratch` for every spec verify cycle \
                 (or worse, a kernel-stride mismatch on MoE) — defeating \
                 DFlash's speedup. Reload without a draft, or use an MQ4 / \
                 HFQ4 / Q8 target. (Future: port MQ3/MQ2 to MoE branches and \
                 additional archs.)"
            ));
        }
    }

    // Derive physical_cap. With eviction (cask.sidecar set), the physical
    // buffer only needs to hold budget+beta+safety slots; max_seq is the
    // advertised window the client targets. Without eviction, the two are
    // identical (prior behavior).
    //
    // The `HIPFIRE_KV_PHYSICAL_CAP` env var is an explicit operator override —
    // useful for ablations or reproducing dflash_spec_demo settings.
    let physical_cap = if cask.sidecar.is_some() {
        let env_override = std::env::var("HIPFIRE_KV_PHYSICAL_CAP").ok()
            .and_then(|s| s.parse::<usize>().ok());
        let safety = 256usize;
        let floor = cask.budget + cask.beta + 4;
        let derived = cask.budget + cask.beta + safety;
        env_override.unwrap_or(derived).clamp(floor, max_seq)
    } else {
        max_seq
    };

    if hfq.arch_id == 5 || hfq.arch_id == 6 {
        // Qwen3.5 DeltaNet (arch=5 dense, arch=6 MoE/A3B). PR 8: dispatch
        // through the `Architecture` trait for the bring-up triple
        // (config → load → state). Forward passes below still call
        // `qwen35::*` directly — see crates/hipfire-arch-qwen35/src/arch.rs
        // for why static dispatch wins for the hot path.
        use hipfire_runtime::arch::Architecture;
        use hipfire_arch_qwen35::Qwen35;
        use hipfire_arch_qwen35_vl::Qwen35Vl;
        let config = <Qwen35 as Architecture>::config_from_hfq(&hfq)
            .map_err(|e| e.to_string())?;

        // Detect VL model: vision_config presence (from HFQ metadata) AND
        // actual vision tensors are required. Text-only Qwen3.5 models can
        // have vision_config in metadata without the patch_embed weights.
        // PR 9: bring-up triple now goes through the Qwen35Vl trait impl;
        // forward (`qwen35_vl::vision_forward`) stays a direct static call.
        let has_vision_tensors = hfq.tensor_data("model.visual.patch_embed.proj.weight").is_some();
        let vision_config = <Qwen35Vl as Architecture>::config_from_hfq(&hfq).ok();
        let (vision_config, vision_weights) = if let Some(vc) = vision_config {
            if has_vision_tensors {
                let vw = <Qwen35Vl as Architecture>::load_weights(&mut hfq, &vc, gpu)
                    .map_err(|e| format!("{e}"))?;
                eprintln!("  VL model: vision encoder (hidden={}, layers={})", vc.hidden_size, vc.num_layers);
                (Some(vc), Some(vw))
            } else {
                (None, None) // text-only model, no vision tensors
            }
        } else {
            (None, None)
        };

        let weights = <Qwen35 as Architecture>::load_weights(&mut hfq, &config, gpu)?;

        // MMQ per-weight screening (#87): pre-screen all weight matrices at
        // load time so the first prefill doesn't pay the screening overhead.
        // Results are cached by device pointer in gpu.mmq_screen_cache.
        if gpu.mmq_screen && matches!(gpu.arch.as_str(), "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151" | "gfx1152") {
            let t0 = std::time::Instant::now();
            let (n_safe, n_unsafe) = screen_weights_qwen35(&weights, gpu);
            let elapsed = t0.elapsed();
            eprintln!(
                "  MMQ screening: {n_safe} safe, {n_unsafe} unsafe (threshold={:.2}, {:.1}ms)",
                gpu.mmq_screen_threshold, elapsed.as_secs_f64() * 1000.0,
            );
        }

        // KV cache modes (RotorQuant-style asymmetric: K rotated + V Q8):
        //   asym3 (default) — K at 3-bit rotated, V at Q8_0. 5.5× vs fp32.
        //                     Best quality/compression tradeoff — RotorQuant "planar3".
        //   asym4 — K at 4-bit rotated, V at Q8_0. 5.1× (slightly safer).
        //   asym2 — K at 2-bit rotated, V at Q8_0. 6.0× (loses rare-token tail).
        //   q8    — K+V both Q8_0. 3.76× (reference quality).
        //
        // Legacy "turbo{2,3,4}" aliases map to asym{2,3,4} for backward compat.
        //
        // All allocators go through the `_capped` entry points with
        // physical_cap derived above. Without eviction, physical_cap==max_seq
        // and these match the back-compat wrappers byte-for-byte.
        let kv = match kv_mode.as_str() {
            "q8" => {
                eprintln!("  KV cache: Q8");
                llama::KvCache::new_gpu_q8_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, physical_cap).map_err(|e| format!("{e}"))?
            }
            "asym4" | "turbo4" => {
                llama::KvCache::new_gpu_asym4_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, physical_cap).map_err(|e| format!("{e}"))?
            }
            "asym2" | "turbo2" => {
                llama::KvCache::new_gpu_asym2_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, physical_cap).map_err(|e| format!("{e}"))?
            }
            "asym3" | "turbo3" | "turbo" | "auto" | "" => {
                llama::KvCache::new_gpu_asym3_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, physical_cap).map_err(|e| format!("{e}"))?
            }
            other => {
                eprintln!("  KV cache: unrecognized '{other}', defaulting to asym3");
                llama::KvCache::new_gpu_asym3_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, physical_cap).map_err(|e| format!("{e}"))?
            }
        };
        // MoE models (num_experts > 0) have ~10x smaller hidden-state
        // magnitudes than dense models, making Q8 DeltaNet state quantization
        // error proportionally larger. Use FP32 state to avoid cumulative
        // drift that degenerates output after ~200 tokens.
        let dn_quant = if config.num_experts > 0 {
            eprintln!("  DeltaNet state: FP32 (MoE model — Q8 drift mitigation)");
            hipfire_arch_qwen35::qwen35::StateQuant::FP32
        } else {
            hipfire_arch_qwen35::qwen35::StateQuant::Q8
        };
        let dn = DeltaNetState::new_with_quant(gpu, &config, dn_quant).map_err(|e| format!("{e}"))?;
        // Flash partials size with physical_cap (bounds the max_tiles the
        // flash kernel must address). When physical_cap == max_seq this is
        // identical to sizing-by-max_seq; under eviction it's much smaller.
        let scratch = qwen35::Qwen35Scratch::new_with_kv_max(gpu, &config, 128, physical_cap).map_err(|e| format!("{e}"))?;

        // Build eviction policy if the operator supplied a sidecar. Qwen3 (arch_id < 5)
        // lacks the FA/LA hybrid wiring TriAttention needs, so sidecars only take
        // effect on arch_id 5/6 — see the cask.rs docs for why CASK targets full-
        // attention layers only.
        let eviction = if let Some(ref sidecar_path) = cask.sidecar {
            let centers = TriAttnCenters::load(Path::new(sidecar_path))
                .map_err(|e| format!("load cask sidecar {}: {e}", sidecar_path))?;
            let fa_layer_ids: Vec<usize> = config.layer_types.iter().enumerate()
                .filter_map(|(i, t)| if *t == LayerType::FullAttention { Some(i) } else { None })
                .collect();
            if fa_layer_ids.is_empty() {
                eprintln!("  cask_sidecar set but model has no FullAttention layers — ignoring");
                None
            } else {
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                let base = EvictionCtx::new(
                    gpu, &centers, fa_layer_ids, cask.budget, cask.beta,
                    config.n_heads, config.n_kv_heads, config.head_dim,
                    n_rot, config.rope_theta, physical_cap,
                ).map_err(|e| format!("build EvictionCtx: {e}"))?;
                if cask.cask_m_folding {
                    eprintln!(
                        "  eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                        cask.core_frac, cask.fold_m, cask.budget, cask.beta, physical_cap,
                    );
                    Some(Eviction::Cask(CaskCtx::new(base, cask.core_frac, cask.fold_m)))
                } else {
                    eprintln!(
                        "  eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                        cask.budget, cask.beta, physical_cap,
                    );
                    Some(Eviction::Plain(base))
                }
            }
        } else { None };
        // Optional DFlash draft: load the draft model's weights + a fresh set
        // of per-cycle scratch buffers (hidden ring, verify scratch, GdnTape,
        // DeltaNetSnapshot) sized for the target's max_seq. If the draft file
        // is missing or arch-mismatched, we log and continue without DFlash
        // (temp==0 requests will fall back to AR sampling).
        let dflash = if let Some(dp) = draft_path {
            // DFlash state (hidden_rb + target_hidden_host) sizes linearly with
            // the ctx_capacity argument. Pass `physical_cap` instead of
            // `max_seq` so eviction's smaller buffer caps VRAM: a 128K-advertised
            // model with physical_cap=896 allocates an 896-slot ring, not 128K.
            // Without eviction, physical_cap == max_seq so the behavior matches.
            match load_dflash_state(dp, physical_cap, &config, &dn, gpu) {
                Ok(state) => {
                    eprintln!(
                        "  DFlash draft loaded: {} (layers={}, hidden={}, block={})",
                        dp, state.draft_config.n_layers, state.draft_config.hidden,
                        state.draft_config.block_size,
                    );
                    Some(state)
                }
                Err(e) => {
                    eprintln!("  DFlash draft load failed ({}): {} — falling back to AR only", dp, e);
                    None
                }
            }
        } else { None };

        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1, pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            q35_config: Some(config), q35_weights: Some(weights), q35_scratch: Some(scratch),
            kv_cache: Some(kv), dn_state: Some(dn),
            llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
            vision_config, vision_weights,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap, eviction,
            conversation_tokens: Vec::new(),
            model_path: path.to_string(),
            dflash,
        })
    } else {
        // Qwen3 / LLaMA — no eviction supported on this path (TriAttention needs
        // the FA/LA hybrid wiring from arch_id 5/6). physical_cap == max_seq.
        // PR 11: dispatch through the `Architecture` trait for the bring-up
        // triple (config → load → scratch). Forward passes below still call
        // `llama::*` directly — see crates/hipfire-arch-llama/src/arch.rs
        // for why static dispatch wins for the hot path.
        use hipfire_runtime::arch::Architecture;
        let config = <Llama as Architecture>::config_from_hfq(&hfq)
            .map_err(|e| e.to_string())?;
        let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        eprintln!("  KV cache: Q8");
        let kv = llama::KvCache::new_gpu_q8(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).map_err(|e| format!("{e}"))?;
        let scratch = <Llama as Architecture>::new_state(gpu, &config)?;
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1, pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            q35_config: None, q35_weights: None, q35_scratch: None,
            kv_cache: None, dn_state: None,
            llama_config: Some(config), llama_weights: Some(weights), llama_scratch: Some(scratch), llama_kv: Some(kv),
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap: max_seq, eviction: None,
            conversation_tokens: Vec::new(),
            model_path: path.to_string(),
            dflash: None,
        })
    }
}

/// Multi-GPU pipeline-parallel load path (Stage 7 of #58). Refuses VL,
/// non-Qwen3.5 architectures and (transitively, via the upstream "load"
/// handler) DFlash, CASK and PFlash. Returns a `LoadedModel` with `pp_gpus`,
/// `pp_scratch_set` and `pp_dn_la_to_device` populated; the daemon's primary
/// `gpu` parameter is unused on this path. Eviction is refused at this layer
/// because TriAttention/CASK/PFlash live on a single device and are not v1
/// targets for pp>1 — physical_cap == max_seq accordingly.
fn load_model_pp(
    path: &str,
    max_seq: usize,
    kv_mode_override: Option<&str>,
    pp: usize,
    _gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    let kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .ok_or("tokenizer not found")?;

    if hfq.arch_id != 5 && hfq.arch_id != 6 {
        return Err(format!(
            "pp>1 supports Qwen3.5 dense (arch_id=5) and Qwen3.5-MoE / \
             Qwen3.6-A3B (arch_id=6) only; got arch_id={}. LLaMA / Qwen3 \
             dense (arch_id<5) is pp=1 only.",
            hfq.arch_id
        ));
    }
    if qwen35_vl::vision_config_from_hfq(&hfq).is_some()
        && hfq.tensor_data("model.visual.patch_embed.proj.weight").is_some()
    {
        return Err("pp>1 does not support VL models in v1; see issue #58 v1.1 roadmap".into());
    }

    let config = qwen35::config_from_hfq(&hfq).ok_or("failed to read Qwen3.5 config")?;

    // HIPFIRE_PP_LAYERS="a,b,..." overrides uniform split. Length must equal
    // pp; sum must equal n_layers; each entry >= 1. Used to shift layers off
    // dev 0 when token_embd asymmetry caps max_seq under uniform split.
    let mut gpus = match std::env::var("HIPFIRE_PP_LAYERS").ok().filter(|s| !s.is_empty()) {
        Some(spec) => {
            let counts: Result<Vec<usize>, _> = spec
                .split(',')
                .map(|s| s.trim().parse::<usize>())
                .collect();
            let counts = counts.map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
            if counts.len() != pp {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS has {} entries, expected pp={}",
                    counts.len(), pp
                ));
            }
            let sum: usize = counts.iter().sum();
            if sum != config.n_layers {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS sum={} != n_layers={}",
                    sum, config.n_layers
                ));
            }
            eprintln!("  HIPFIRE_PP_LAYERS override: {:?}", counts);
            Gpus::init_layers(&counts).map_err(|e| format!("{e}"))?
        }
        None => Gpus::init_uniform(pp, config.n_layers).map_err(|e| format!("{e}"))?,
    };

    let weights = qwen35::load_weights_multi(&hfq, &config, &mut gpus).map_err(|e| format!("{e}"))?;

    // KV cache (asym3 default, q8/asym4/asym2 selectable). physical_cap ==
    // max_seq on this path — eviction is refused at load.
    let kv = match kv_mode.as_str() {
        "q8" => llama::KvCache::new_gpu_q8_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?,
        "asym4" | "turbo4" => llama::KvCache::new_gpu_asym4_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?,
        "asym2" | "turbo2" => llama::KvCache::new_gpu_asym2_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?,
        "asym3" | "turbo3" | "turbo" | "auto" | "" => llama::KvCache::new_gpu_asym3_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?,
        other => {
            eprintln!("  KV cache: unrecognized '{other}', defaulting to asym3");
            llama::KvCache::new_gpu_asym3_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        }
    };

    // MoE state-quant rule mirrors the pp=1 path (Q8 drift on small hidden
    // states); apply at the multi entry point so bit-equivalence with pp=1
    // forward output holds when both run on the same model.
    let dn_quant = if config.num_experts > 0 {
        eprintln!("  DeltaNet state: FP32 (MoE model — Q8 drift mitigation)");
        qwen35::StateQuant::FP32
    } else {
        qwen35::StateQuant::Q8
    };
    let (dn, la_to_device) =
        DeltaNetState::new_with_quant_multi(&mut gpus, &config, dn_quant).map_err(|e| format!("{e}"))?;

    let scratch_set = Qwen35ScratchSet::new_with_kv_max_multi(&mut gpus, &config, 128, max_seq).map_err(|e| format!("{e}"))?;

    // ROCm 6.4.3 gotcha: enable_peer_access AFTER all allocations are live.
    // See multi_gpu.rs::enable_peer_all docstring for the silent-success bug
    // when the call precedes hipMalloc.
    let _peer = gpus.enable_peer_all().map_err(|e| format!("enable_peer_all: {e}"))?;

    eprintln!(
        "  pp={pp} loaded: layer_to_device={:?}, output_device={}, peer_access={}",
        gpus.layer_to_device, gpus.output_device, gpus.peer_access_enabled,
    );

    Ok(LoadedModel {
        arch_id: hfq.arch_id,
        pp,
        pp_gpus: Some(gpus),
        pp_scratch_set: Some(scratch_set),
        pp_dn_la_to_device: Some(la_to_device),
        q35_config: Some(config),
        q35_weights: Some(weights),
        q35_scratch: None,
        kv_cache: Some(kv),
        dn_state: Some(dn),
        llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
        vision_config: None, vision_weights: None,
        tokenizer: Some(tokenizer),
        seq_pos: 0, max_seq, physical_cap: max_seq, eviction: None,
        conversation_tokens: Vec::new(),
        model_path: path.to_string(),
        dflash: None,
    })
}

/// Pre-screen all Qwen3.5/3.6 weight matrices for MMQ safety (#87).
/// Returns (n_safe, n_unsafe). Results are cached in gpu.mmq_screen_cache.
fn screen_weights_qwen35(weights: &qwen35::Qwen35Weights, gpu: &mut rdna_compute::Gpu) -> (usize, usize) {
    use hipfire_arch_qwen35::qwen35::LayerWeights;
    let mut n_safe = 0usize;
    let mut n_unsafe = 0usize;

    for layer in &weights.layers {
        // Collect all weight tensors for this layer that could use MMQ
        let wts: Vec<(&hipfire_runtime::llama::WeightTensor, &str)> = match layer {
            LayerWeights::DeltaNet(l) => vec![
                (&l.wqkv, "qkvza.qkv"), (&l.wz, "qkvza.z"),
                (&l.w_beta, "qkvza.beta"), (&l.w_alpha, "qkvza.alpha"),
                (&l.w_gate, "gate_up.gate"), (&l.w_up, "gate_up.up"),
                (&l.wo, "residual"),
            ],
            LayerWeights::FullAttn(l) => vec![
                (&l.wq, "qkv.q"), (&l.wk, "qkv.k"), (&l.wv, "qkv.v"),
                (&l.w_gate, "gate_up.gate"), (&l.w_up, "gate_up.up"),
                (&l.wo, "residual"),
            ],
            LayerWeights::DeltaNetMoe(l) => vec![
                (&l.wqkv, "qkvza.qkv"), (&l.wz, "qkvza.z"),
                (&l.w_beta, "qkvza.beta"), (&l.w_alpha, "qkvza.alpha"),
                (&l.wo, "residual"),
            ],
            LayerWeights::FullAttnMoe(l) => vec![
                (&l.wq, "qkv.q"), (&l.wk, "qkv.k"), (&l.wv, "qkv.v"),
                (&l.wo, "residual"),
            ],
        };

        for (wt, _name) in wts {
            // MMQ kernels only operate on HFQ4G256 weights. Other formats
            // (MQ3, MQ2, HFQ6, etc.) use different dispatch paths and must
            // not be fed to the HFQ4-specific screening kernels — buffer
            // layout mismatch would read past the end. See PR #106.
            if !matches!(wt.gpu_dtype, rdna_compute::DType::HFQ4G256 | rdna_compute::DType::MQ4G256) {
                continue;
            }
            if gpu.mmq_screen_weight(&wt.buf, wt.m, wt.k) {
                n_safe += 1;
            } else {
                n_unsafe += 1;
            }
        }
    }

    (n_safe, n_unsafe)
}

fn unload_model(m: LoadedModel, gpu: &mut rdna_compute::Gpu) {
    // Multi-GPU branch (Stage 7 of #58). Frees per-device tensors through the
    // Gpus orchestrator, then invalidates per-device caches so the next load
    // can't inherit stale verdicts at recycled device addresses. Order
    // matches the alloc order in load_model_pp reversed: scratch → kv → dn →
    // weights, so each free targets a still-live owner.
    if m.pp > 1 {
        let mut gpus = m.pp_gpus.expect("pp>1 must carry pp_gpus");
        if let Some(scratch_set) = m.pp_scratch_set { scratch_set.free_gpu_multi(&mut gpus); }
        if let Some(kv) = m.kv_cache { kv.free_gpu_multi(&mut gpus); }
        if let Some(dn) = m.dn_state {
            let la_to_device = m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
            dn.free_gpu_multi(&mut gpus, &la_to_device);
        }
        if let Some(w) = m.q35_weights { w.free_gpu_multi(&mut gpus); }
        for g in gpus.devices.iter_mut() {
            g.invalidate_weight_caches();
            g.invalidate_graph_state();
            g.drain_pool();
        }
        let _ = gpu;
        return;
    }
    // DFlash state: draft weights have free_gpu; ring / snapshot / tape /
    // verify_scratch don't expose one — their GpuTensors / DeviceBuffers will
    // leak until daemon exit if the caller cycles load/unload mid-session.
    // Acceptable for the daemon since unload is rare and the weights are the
    // bulk of the VRAM anyway.
    if let Some(df) = m.dflash {
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
    }
    // Free eviction context (centers + scratch tensors) if active.
    if let Some(ev) = m.eviction { ev.free_gpu(gpu); }
    // Free KV cache + DeltaNet state + scratch first (small fraction of VRAM).
    if let Some(kv) = m.kv_cache { kv.free_gpu(gpu); }
    if let Some(dn) = m.dn_state { dn.free_gpu(gpu); }
    if let Some(s) = m.q35_scratch { s.free_gpu(gpu); }
    if let Some(kv) = m.llama_kv { kv.free_gpu(gpu); }
    if let Some(s) = m.llama_scratch { s.free_gpu(gpu); }
    // Weights are the bulk of VRAM (~80%). Free them too so idle eviction
    // actually returns VRAM to the system, not just the cache.
    if let Some(w) = m.q35_weights { w.free_gpu(gpu); }
    if let Some(w) = m.llama_weights { w.free_gpu(gpu); }
    if let Some(w) = m.vision_weights { w.free_gpu(gpu); }
    // Drop pointer-keyed caches whose keys point at weight buffers that are
    // about to be returned to the pool. Without this, the next model loaded
    // can land at the same device address and silently inherit stale
    // verdicts (mmq_screen_cache) or leaked FP16 shadows (fp16_shadow_cache).
    gpu.invalidate_weight_caches();
    // Tear down any captured hipGraphs (single-slot AR forward graph plus
    // DFlash verify and replay graph caches). These bake KV-cache, scratch,
    // and draft-weight pointers into kernarg memory at capture time; the
    // tensors backing those pointers are freed above, so replaying after
    // a model swap would dispatch against dangling or wrong-content
    // memory.
    gpu.invalidate_graph_state();
    gpu.drain_pool();
}

fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &qwen35::Qwen35Config,
    target_dn: &DeltaNetState,
    gpu: &mut rdna_compute::Gpu,
) -> Result<DflashState, String> {
    let hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("open draft: {e}"))?;
    let draft_config = DflashConfig::from_hfq(&hfq).ok_or("parse DflashConfig")?;
    let draft_weights = DflashWeights::load(gpu, &hfq, &draft_config).map_err(|e| format!("load weights: {e}"))?;
    let draft_scratch = DflashScratch::new_with_mq(
        gpu, &draft_config, draft_config.block_size, ctx_capacity, draft_weights.has_mq,
    ).map_err(|e| format!("draft scratch: {e}"))?;

    // Hidden ring: one row per target-layer selected by the draft config,
    // captured during each target forward. Sized so the whole context plus
    // one block fits without aliasing. Cheap (< 100 MB) next to the draft
    // weights themselves.
    let hidden_rb = HiddenStateRingBuffer::new(
        gpu,
        target_config.n_layers,
        draft_config.num_extract(),
        draft_config.hidden,
        ctx_capacity + draft_config.block_size,
        hipfire_arch_qwen35::qwen35::PREFILL_MAX_BATCH.max(draft_config.block_size),
    ).map_err(|e| format!("hidden_rb: {e}"))?;

    let target_snap = DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("target_snap: {e}"))?;

    // Read DDTree budget env-var BEFORE sizing GdnTape / VerifyScratch.
    // When DDTree is enabled, both must be sized for `1 + budget` nodes
    // per cycle (the linearized tree includes one root slot plus all
    // tree nodes), not just `block_size`. Reading the env-var here keeps
    // a single source of truth and avoids re-allocating these scratches
    // after the model is on GPU.
    //
    // DdtreeScratch::attn_bias is sized `max_n²` (max_n = 1 + budget),
    // so the allocation is quadratic in budget. The paper's Algorithm 1
    // typically uses budget ≤ 22; we cap at 256 to leave huge headroom
    // while killing the OOM cliff from a typo'd budget value (`=10000`
    // would request 400 MB just for attn_bias; `=100000` would OOM most
    // GPUs). Invalid / out-of-range values warn loudly and disable
    // DDTree rather than silently falling through.
    const DDTREE_BUDGET_MAX: usize = 256;
    let ddtree_budget_env: usize = match std::env::var("HIPFIRE_DDTREE_BUDGET").ok() {
        None => 0,
        Some(s) if s.is_empty() => 0,
        Some(s) => match s.parse::<usize>() {
            Ok(0) => 0,
            Ok(n) if n <= DDTREE_BUDGET_MAX => n,
            Ok(n) => {
                eprintln!(
                    "[hipfire-daemon] HIPFIRE_DDTREE_BUDGET={} exceeds cap {DDTREE_BUDGET_MAX} \
                     (attn_bias is O(budget²); typical values are 12-22). Disabling DDTree.",
                    n
                );
                0
            }
            Err(_) => {
                eprintln!(
                    "[hipfire-daemon] HIPFIRE_DDTREE_BUDGET={:?} is not a non-negative integer. \
                     Disabling DDTree.",
                    s
                );
                0
            }
        },
    };
    let scratch_max_n = if ddtree_budget_env > 0 {
        std::cmp::max(draft_config.block_size, 1 + ddtree_budget_env)
    } else {
        draft_config.block_size
    };

    let gdn_tape = GdnTape::new_for_config(gpu, target_config, scratch_max_n)
        .map_err(|e| format!("gdn_tape: {e}"))?;
    let verify_scratch = VerifyScratch::with_prefill(
        gpu,
        scratch_max_n,
        target_config.dim,
        target_config.vocab_size,
        target_config.dim,
        target_config,
    ).map_err(|e| format!("verify_scratch: {e}"))?;

    let target_hidden_host: Vec<f32> = Vec::with_capacity(
        ctx_capacity * draft_config.num_extract() * draft_config.hidden,
    );
    let block_size = draft_config.block_size;

    // Optional DDTree allocation. `HIPFIRE_DDTREE_BUDGET=<n>` (positive
    // integer) wires the decode loop to `spec_step_ddtree_batched` instead
    // of `spec_step_dflash`. `HIPFIRE_DDTREE_TOPK=<k>` controls the
    // per-position top-K (default 4). Anything else, or budget=0, leaves
    // the existing DFlash chain-mode path untouched.
    let ddtree = match Some(ddtree_budget_env).filter(|&n| n > 0) {
        Some(budget) => {
            // topk caps the per-position branching factor in the tree
            // builder. Algorithm 1's typical setting is 4; the active
            // kernel `run_dflash_draft_for_topk_gpu` (called by both
            // `spec_step_ddtree_batched` and `spec_step_ddtree_path_c`)
            // asserts `k >= 1 && k <= 8` at speculative.rs:3302 and panics
            // outside that range. Take the kernel's bound as authoritative;
            // anything looser would let env-var values pass daemon
            // validation but blow up at the first decode cycle.
            //
            // Two upper bounds:
            //   - DDTREE_TOPK_KERNEL_MAX = 8 — kernel's hardcoded assert.
            //   - vocab_size — extra correctness cap for tiny-vocab /
            //     character-level targets where vocab can be < 8.
            //
            // Effective cap = min(8, vocab_size). Default = min(4, vocab_size).
            const DDTREE_TOPK_KERNEL_MAX: usize = 8;
            let vocab = target_config.vocab_size;
            let effective_topk_max = std::cmp::min(DDTREE_TOPK_KERNEL_MAX, vocab);
            let default_topk = std::cmp::min(4usize, vocab.max(1));
            let topk = match std::env::var("HIPFIRE_DDTREE_TOPK").ok() {
                None => default_topk,
                Some(s) if s.is_empty() => default_topk,
                Some(s) => match s.parse::<usize>() {
                    Ok(k) if k >= 1 && k <= effective_topk_max => k,
                    Ok(k) => {
                        eprintln!(
                            "[hipfire-daemon] HIPFIRE_DDTREE_TOPK={k} out of range [1, {effective_topk_max}] \
                             (vocab_size={vocab}). Falling back to default topk={default_topk}."
                        );
                        default_topk
                    }
                    Err(_) => {
                        eprintln!(
                            "[hipfire-daemon] HIPFIRE_DDTREE_TOPK={:?} is not a positive integer. \
                             Falling back to default topk={default_topk}.",
                            s
                        );
                        default_topk
                    }
                },
            };
            let post_seed_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree post_seed_snap: {e}"))?;
            let path_c_parent_pre_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree path_c_parent_pre_snap: {e}"))?;
            let path_c_main_end_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
                .map_err(|e| format!("ddtree path_c_main_end_snap: {e}"))?;
            let n_fa_layers = target_config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count();
            // qkv_dim mirrors GdnTape::new_for_config: linear-attention
            // qkv row width (k_dim × 2 + v_dim).
            let k_dim = target_config.linear_num_key_heads
                * target_config.linear_key_head_dim;
            let v_dim = target_config.linear_num_value_heads
                * target_config.linear_value_head_dim;
            let qkv_dim = k_dim * 2 + v_dim;
            let scratch = DdtreeScratch::new(
                gpu,
                budget,
                target_config.n_kv_heads,
                target_config.head_dim,
                qkv_dim,
                n_fa_layers,
            )
            .map_err(|e| format!("ddtree scratch: {e}"))?;
            eprintln!(
                "[hipfire-daemon] DDTree enabled: budget={budget}, topk={topk}, n_fa_layers={n_fa_layers}"
            );
            Some(DdtreeState {
                post_seed_snap,
                scratch,
                budget,
                topk,
                path_c_parent_pre_snap,
                path_c_main_end_snap,
            })
        }
        None => None,
    };

    Ok(DflashState {
        draft_config,
        draft_weights,
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        target_hidden_host,
        ctx_capacity,
        block_size,
        ddtree,
    })
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
fn generate_dflash(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    pflash_bypass_reason: Option<&str>,
    pflash_alpha: Option<f32>,
) {
    use hipfire_arch_qwen35::speculative::{
        spec_step_ddtree_batched, spec_step_ddtree_path_c, spec_step_dflash, ModelSlot,
        ModelSlotConfig, Phase2Snapshots, SpecStats,
    };

    // Tokenize with ChatML wrapping (identical to the AR path). System prompt
    // is always prepended because this fast path is single-turn.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_tokens = hipfire_runtime::prompt_frame::ChatFrame {
        tokenizer,
        system: system_prompt,
        user: prompt,
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        raw: false,
    }
    .build();

    // `im_end_token` is still needed downstream for the EOS check.
    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };

    // Fresh target state — DFlash seed_target_hidden_from_prompt does its own
    // full prefill, so we reset first to avoid double-accounting.
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    {
        let dn = m.dn_state.as_ref().unwrap();
        for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
    }
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
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"reopen model: {}"}}"#, id, e);
            let _ = stdout.flush();
            m.q35_weights = Some(weights); m.kv_cache = Some(kv_cache);
            m.dn_state = Some(dn_state); m.q35_scratch = Some(scratch);
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
    let eff_prompt_cap = if m.eviction.is_some() { m.physical_cap } else { ctx_capacity };
    if prompt_tokens.len() + df.block_size > eff_prompt_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt+block_size exceeds {} {} (eviction {})"}}"#,
            id,
            if m.eviction.is_some() { "physical_cap" } else { "ctx_capacity" },
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
        gpu, &mut target, &mut df.hidden_rb, &mut df.target_hidden_host, &prompt_tokens,
    ) {
        let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"prefill: {}"}}"#, id, e);
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
        gpu, &df.hidden_rb, &df.draft_scratch.target_hidden,
        0, prompt_tokens.len(), prompt_tokens.len(),
    ) {
        eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
    }
    df.draft_scratch.uploaded_target_hidden_rows = prompt_tokens.len();
    df.draft_scratch.target_hidden_abs_positions =
        (0..prompt_tokens.len() as i32).collect();

    // First emit = target's argmax at the final prompt position. seed_target_hidden
    // already ran the per-token forward for every prompt token; its scratch.logits
    // holds the post-prompt logits.
    let first_logits = match gpu.download_f32(&target.scratch.logits) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"download logits: {}"}}"#, id, e);
            let _ = stdout.flush();
            m.q35_weights = Some(target.weights);
            m.kv_cache = Some(target.kv_cache);
            m.dn_state = Some(target.dn_state);
            m.q35_scratch = Some(target.scratch);
            return;
        }
    };
    let first_token = first_logits.iter().enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv { (i as u32, v) } else { (best, bv) }
        }).0;

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
    let mut filter = EosFilter::new(EosFilterConfig::default());
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
                    gpu, &mut df.draft_scratch, &res.retain_mask,
                    df.draft_config.num_extract(), df.draft_config.hidden, pre_phys,
                );
            }
        }
    }

    // Emit the first token immediately so TTFT is the prefill time.
    streamed_tokens.push(first_token);
    emit_committed_event(stdout, id, first_token, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
    let new_bytes = &all_bytes[bytes_fed_to_filter..];
    bytes_fed_to_filter = all_bytes.len();
    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
        let text = std::str::from_utf8(&text_bytes).unwrap();
        let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
        let _ = stdout.flush();
    }
    generated += 1;

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
    let path_c_mode_owned: Option<&'static str> = match std::env::var("HIPFIRE_DDTREE_PATH_C").ok() {
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
    while generated < max_tokens {
        if position + df.block_size >= ctx_capacity { break; }

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
                    gpu, &mut target, &df.draft_weights, &df.draft_config,
                    &mut df.draft_scratch, &mut df.hidden_rb, &mut df.target_hidden_host,
                    &mut df.target_snap, &mut df.gdn_tape, &df.verify_scratch,
                    position, seed_token,
                    None,                      // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                    phase2_snaps,
                )
            } else {
                spec_step_ddtree_batched(
                    gpu, &mut target, &df.draft_weights, &df.draft_config,
                    &mut df.draft_scratch, &mut df.hidden_rb, &mut df.target_hidden_host,
                    &mut df.target_snap, &mut dd.post_seed_snap, &mut df.gdn_tape,
                    &dd.scratch, &df.verify_scratch,
                    position, seed_token,
                    None,                      // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                )
            }
        } else {
            spec_step_dflash(
                gpu, &mut target, &df.draft_weights, &df.draft_config,
                &mut df.draft_scratch, &mut df.hidden_rb, &mut df.target_hidden_host,
                &mut df.target_snap, &df.verify_scratch,
                position, seed_token,
                None,                      // ctx_slice = full history
                Some(&mut df.gdn_tape),
                0.0_f32,                   // temperature
                &mut rng_state,
                None,                      // block_size override
                None,                      // ngram_cache
                &emitted,
                0.0_f32,                   // cactus_delta
                None,                      // pld_spine
                1.0_f32,                   // repeat_penalty (off)
                0,                         // repeat_window
            )
        };
        let step = match step_result {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"spec_step: {}"}}"#, id, e);
                let _ = stdout.flush();
                break;
            }
        };
        stats.record(&step);
        let committed_tail: Vec<u32> = step.committed.iter().skip(1).copied().collect();

        let mut hit_eos = false;
        let mut think_cap_hit = false;
        for &tok in &committed_tail {
            if generated >= max_tokens { break; }
            emitted.push(tok);
            streamed_tokens.push(tok);
            emit_committed_event(stdout, id, tok, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                let text = std::str::from_utf8(&text_bytes).unwrap();
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                let _ = stdout.flush();
            }
            generated += 1;
            if tok == target.config.eos_token || im_end_token == Some(tok) || tokenizer.is_terminator(tok) { hit_eos = true; break; }

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
                if in_think && !prev_in_think { think_count = 0; }
                if in_think { think_count += 1; }
                prev_in_think = in_think;

                if in_think && think_count >= max_think_tokens {
                    // Force-close: emit </think>\n and break out of this batch.
                    // Unlike the AR path we can't splice into the KV cache mid-
                    // spec-cycle, so we just stream the close text and break.
                    // The next request will start fresh.
                    let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":"</think>\n"}}"#, id);
                    let _ = stdout.flush();
                    think_cap_hit = true;
                    break;
                }
            }
        }
        position += step.accepted + 1;
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
                        gpu, &mut df.draft_scratch, &res.retain_mask,
                        df.draft_config.num_extract(), df.draft_config.hidden, pre_phys,
                    );
                }
            }
        }
        if hit_eos || think_cap_hit { break; }
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
    let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
    let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
    let prefill_tok_s = if prefill_s > 0.0 { prompt_tokens.len() as f64 / prefill_s } else { 0.0 };
    let tau = if stats.cycles > 0 { stats.accepted_tokens as f64 / stats.cycles as f64 } else { 0.0 };
    // Per PRD §3.1, when PFlash bypassed (e.g. dflash_decode_active for
    // this branch) the `done` object must surface the bypass reason and
    // alpha alongside the dflash perf metrics. Build a small fragment
    // when both are available; otherwise empty for back-compat.
    let pflash_done_field = match (pflash_bypass_reason, pflash_alpha) {
        (Some(r), Some(a)) => format!(
            r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
            r.replace('"', "'"), a,
        ),
        _ => String::new(),
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1},"dflash":true,"tau":{:.2},"cycles":{}{}}}"#,
        id, generated, tok_s, prompt_tokens.len(),
        prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0,
        tau, stats.cycles, pflash_done_field,
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
    _repeat_window: usize,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
) {
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[daemon] context full ({}/{}) — resetting conversation", m.seq_pos, m.max_seq);
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
        if let Some(kv) = m.kv_cache.as_mut() { kv.compact_offset = 0; }
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
                gpu, state, cfg, &raw_q_tokens, request_kind, &[],
            ) {
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Compressed(cp)) => {
                    let _ = writeln!(stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"total_ms":{}}}"#,
                        id, cp.source_tokens, cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5, cp.compressed_md5,
                        cp.timings.score_ms, cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    cp.token_ids
                }
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Bypass { reason }) => {
                    if !matches!(reason, hipfire_arch_qwen35::pflash::BypassReason::ModeOff) {
                        let _ = writeln!(stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id, reason.as_str().replace('"', "'"),
                        );
                        let _ = stdout.flush();
                    }
                    raw_q_tokens
                }
                Err(e) => {
                    let _ = writeln!(stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id, e.to_string().replace('"', "'"),
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

    // ChatML framing via the canonical hipfire_runtime::prompt_frame module.
    // Identical to the pp=1 path so multi-turn behavior matches byte-for-byte
    // when both paths run the same model on the same prompt history.
    let new_tokens = hipfire_runtime::prompt_frame::ChatFrame {
        tokenizer,
        system: if m.seq_pos == 0 { system_prompt } else { None },
        user: "",
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        raw: false,
    }
    .build_with_user_tokens(&q_tokens);

    let trailer = nl.len();
    if m.seq_pos + new_tokens.len() + max_tokens + trailer > m.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
            id, m.seq_pos, new_tokens.len(), max_tokens, trailer, m.physical_cap
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
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

    let config = m.q35_config.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch_set = m.pp_scratch_set.as_ref().unwrap();
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();
    let gpus = m.pp_gpus.as_mut().unwrap();

    let dev_last = gpus.output_device;
    let vocab_size = config.vocab_size;
    let repeat_buf_cap = scratch_set.per_device[dev_last].repeat_buf.buf.size() / 4;

    if let Err(e) = qwen35::forward_prefill_batch_multi(
        gpus, weights, config, &new_tokens, m.seq_pos, kv, dn, scratch_set,
    ) {
        let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"forward_prefill_batch_multi: {}"}}"#, id, e);
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
    sampler::collect_unclosed_attractor_blocks(
        ngram_scope, &attractor_pairs, 20, 2, &mut blocked0,
    );
    let cfg0 = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window: repeat_buf_cap,
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
    let mut filter = EosFilter::new(EosFilterConfig::default());
    let mut alert_fired = false;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;
    let loop_guard = hipfire_runtime::loop_guard::LoopGuard::from_env();

    while generated < max_tokens {
        generated += 1;
        m.conversation_tokens.push(next_token);
        streamed_tokens.push(next_token);
        emit_committed_event(stdout, id, next_token, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
            let text = std::str::from_utf8(&text_bytes).unwrap();
            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
            let _ = stdout.flush();
        }

        if let Err(e) = qwen35::forward_scratch_multi(gpus, weights, config, next_token, m.seq_pos, kv, dn, scratch_set) {
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"forward_scratch_multi decode: {}"}}"#, id, e);
            let _ = stdout.flush();
            return;
        }
        m.seq_pos += 1;

        if next_token == config.eos_token { break; }
        if im_end_token == Some(next_token) { break; }
        if tokenizer.is_terminator(next_token) { break; }

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
                if !prev_in_think { think_count = 1; } else { think_count += 1; }
            } else {
                think_count = 0;
            }
            prev_in_think = in_think;

            if in_think && think_count >= max_think_tokens {
                let close_tokens = tokenizer.encode("</think>\n");
                let budget_left = max_tokens.saturating_sub(generated);
                let take = close_tokens.len().min(budget_left);
                for &t in &close_tokens[..take] {
                    if let Err(e) = qwen35::forward_scratch_multi(gpus, weights, config, t, m.seq_pos, kv, dn, scratch_set) {
                        eprintln!("[daemon] max_think close forward_scratch_multi: {}", e);
                        break;
                    }
                    m.seq_pos += 1;
                    m.conversation_tokens.push(t);
                    streamed_tokens.push(t);
                    emit_committed_event(stdout, id, t, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
                    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes = &all_bytes[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes.len();
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                        let text = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                        let _ = stdout.flush();
                    }
                    generated += 1;
                }
                think_count = 0;
                prev_in_think = false;
                if generated >= max_tokens { break; }
            }
        }

        // N-gram loop detector (token-side, no GPU work).
        if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
            loop_guard.check(&streamed_tokens)
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

        // Budget-alert injection: gated to inside an open <think> block.
        if !alert_fired && budget_alert_at_tok > 0 && generated >= budget_alert_at_tok && !budget_alert_text.is_empty() {
            alert_fired = true;
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = match (raw_str.rfind("<think>"), raw_str.rfind("</think>")) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                _ => false,
            };
            if !in_think {
                let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#, id);
                let _ = stdout.flush();
                let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
                let mut blocked: Vec<u32> = Vec::new();
                sampler::collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked);
                let cfg = SamplerConfig {
                    temperature: temp, top_p, repeat_penalty,
                    repeat_window: repeat_buf_cap,
                    blocked_tokens: blocked,
                };
                next_token = {
                    let s_last = &scratch_set.per_device[dev_last];
                    let g_last = &mut gpus.devices[dev_last];
                    sampler::sample(g_last, &s_last.logits, &s_last.sample_buf, &s_last.repeat_buf, vocab_size, ngram_scope, &cfg, &mut rng_state)
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
                    emit_committed_event(stdout, id, tok, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
                    let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes2.len();
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes2) {
                        let t = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&t).unwrap_or_default());
                        let _ = stdout.flush();
                    }
                    if let Err(e) = qwen35::forward_scratch_multi(gpus, weights, config, tok, m.seq_pos, kv, dn, scratch_set) {
                        eprintln!("[daemon] budget_alert forward_scratch_multi: {}", e);
                        break;
                    }
                    m.seq_pos += 1;
                    generated += 1;
                }
            } else if nudge_len < nudge_tokens.len() {
                let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#, id, nudge_len, budget_left);
                let _ = stdout.flush();
            } else {
                let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#, id);
                let _ = stdout.flush();
            }
            if generated >= max_tokens { break; }
        }

        // Steady-state sample.
        let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
        let mut blocked: Vec<u32> = Vec::new();
        sampler::collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked);
        let cfg = SamplerConfig {
            temperature: temp, top_p, repeat_penalty,
            repeat_window: repeat_buf_cap,
            blocked_tokens: blocked,
        };
        next_token = {
            let s_last = &scratch_set.per_device[dev_last];
            let g_last = &mut gpus.devices[dev_last];
            sampler::sample(g_last, &s_last.logits, &s_last.sample_buf, &s_last.repeat_buf, vocab_size, ngram_scope, &cfg, &mut rng_state)
        };
    }

    // ChatML \n trailer so the next turn opens cleanly.
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            if let Err(e) = qwen35::forward_scratch_multi(gpus, weights, config, t, m.seq_pos, kv, dn, scratch_set) {
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
    let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
    let prefill_tok_s = if prefill_s > 0.0 { prefill_tokens as f64 / prefill_s } else { 0.0 };
    let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
        id, generated, tok_s, prefill_tokens,
        prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0
    );
    let _ = stdout.flush();
}

#[allow(clippy::too_many_arguments)]
fn generate(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, id: &str, prompt: &str, system_prompt: Option<&str>, temp: f32, top_p: f32, max_tokens: usize, repeat_penalty: f32, repeat_window: usize, budget_alert_at_tok: usize, budget_alert_text: &str, max_think_tokens: usize, pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>, pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>) {
    // Multi-GPU pipeline-parallel dispatch (Stage 7 of #58). pp>1 is refused
    // at load when DFlash / CASK / PFlash / VL is requested, so this branch
    // doesn't need to thread any of those args through.
    if m.pp > 1 {
        generate_multi(
            m, gpu, pflash_state, pflash_cfg, stdout, id, prompt, system_prompt,
            temp, top_p, max_tokens, repeat_penalty, repeat_window,
            budget_alert_at_tok, budget_alert_text, max_think_tokens,
        );
        return;
    }
    // DFlash fast path -- only when a draft model is loaded AND temperature is
    // effectively 0 (DFlash is greedy-only in this integration). Skip the
    // normal AR sampling setup entirely.
    if m.dflash.is_some() && temp <= 1e-6 && (m.arch_id == 5 || m.arch_id == 6) {
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
        generate_dflash(m, gpu, stdout, id, prompt, system_prompt, max_tokens, max_think_tokens, dflash_bypass_reason, dflash_alpha);
        // Silence unused-variable warnings for the params we didn't need.
        let _ = (top_p, repeat_penalty, repeat_window, budget_alert_at_tok, budget_alert_text, pflash_state);
        return;
    }

    // Auto-reset on multi-turn rollover. When eviction is active (operator
    // enabled cask_sidecar at load), the physical buffer is bounded by
    // budget+beta+safety regardless of conversation length, so reset never
    // needs to fire — eviction reclaims slots after each token. When eviction
    // is OFF, physical grows unbounded up to max_seq; reset when we'd overrun.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.eviction.is_none() && m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[daemon] context full ({}/{}) — resetting conversation", m.seq_pos, m.max_seq);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        // Zero DeltaNet state on reset
        if let Some(ref dn) = m.dn_state {
            for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        }
        if let Some(kv) = m.kv_cache.as_mut() { kv.compact_offset = 0; }
        if let Some(kv) = m.llama_kv.as_mut() { kv.compact_offset = 0; }
    }

    // `nl` is needed for the trailer write after natural <|im_end|>
    // termination; `im_end` derives the EOS-check token id. Other
    // ChatML scaffolding tokens are now built inside hipfire_runtime::prompt_frame.
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
                cp.source_tokens, cp.kept_tokens,
                cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                alpha.unwrap_or(0.0),
                cp.timings.score_ms, cp.timings.total_ms,
                cp.source_md5, cp.compressed_md5,
            ),
            (None, Some(reason)) => format!(
                r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
                reason.replace('"', "'"),
                alpha.unwrap_or(0.0),
            ),
            (None, None) => String::new(),
        }
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.seq_pos == 0 {
            let decision = hipfire_arch_qwen35::pflash::maybe_compress_prompt(
                gpu, state, cfg, &raw_q_tokens, request_kind, &[],
            );
            match decision {
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Compressed(cp)) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_compressed","id":"{}","source_tokens":{},"kept_tokens":{},"keep_ratio":{:.6},"source_md5":"{}","compressed_md5":"{}","score_ms":{},"select_ms":{},"gather_ms":{},"total_ms":{}}}"#,
                        id, cp.source_tokens, cp.kept_tokens,
                        cp.kept_tokens as f32 / cp.source_tokens.max(1) as f32,
                        cp.source_md5, cp.compressed_md5,
                        cp.timings.score_ms, cp.timings.select_ms,
                        cp.timings.gather_ms, cp.timings.total_ms,
                    );
                    let _ = stdout.flush();
                    let token_ids = cp.token_ids.clone();
                    pflash_summary = Some(cp);
                    token_ids
                }
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Bypass { reason }) => {
                    // Only emit bypass events for non-trivial reasons.
                    // ModeOff is the silent default; nothing to report.
                    if !matches!(reason, hipfire_arch_qwen35::pflash::BypassReason::ModeOff) {
                        let r = reason.as_str();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"pflash_bypass","id":"{}","reason":"{}"}}"#,
                            id, r.replace('"', "'"),
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
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"pflash_error","id":"{}","reason":"{}"}}"#,
                        id, e.to_string().replace('"', "'"),
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

    // ChatML framing via the canonical hipfire_runtime::prompt_frame module.
    // System prompt is prepended only on the first turn (seq_pos == 0)
    // — subsequent turns continue the conversation in-place. The user
    // body comes in pre-tokenized as `q_tokens` because PFlash may
    // have compressed it upstream.
    let new_tokens = hipfire_runtime::prompt_frame::ChatFrame {
        tokenizer,
        system: if m.seq_pos == 0 { system_prompt } else { None },
        user: "", // unused: we pass tokens directly via build_with_user_tokens
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        raw: false,
    }
    .build_with_user_tokens(&q_tokens);

    // KV-budget guard. Without eviction the physical buffer is the hard cap;
    // we must fit prefill + generation + trailer in one allocation. With
    // eviction, physical is bounded by physical_cap regardless of total tokens
    // — the chunked prefill below calls maybe_evict between chunks, and the
    // decode loop evicts after every token. The only ceiling under eviction is
    // the advertised context window (max_seq) — refuse requests that would
    // overflow it in absolute position terms (current absolute + new).
    let trailer = nl.len();
    let absolute_pos = m.seq_pos
        + m.kv_cache.as_ref().map(|kv| kv.compact_offset).unwrap_or(0)
        + m.llama_kv.as_ref().map(|kv| kv.compact_offset).unwrap_or(0);
    if m.eviction.is_none() {
        if m.seq_pos + new_tokens.len() + max_tokens + trailer > m.physical_cap {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
                id, m.seq_pos, new_tokens.len(), max_tokens, trailer, m.physical_cap
            );
            let _ = stdout.flush();
            return;
        }
    } else if absolute_pos + new_tokens.len() + max_tokens + trailer > m.max_seq {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds advertised context window: absolute={} + prefill={} + max_tokens={} + trailer={} > max_seq={}"}}"#,
            id, absolute_pos, new_tokens.len(), max_tokens, trailer, m.max_seq
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
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

    if m.arch_id == 5 || m.arch_id == 6 {
        // Qwen3.5 / Qwen3.5-MoE — multi-turn: prefill only the NEW turn tokens,
        // continuing from m.seq_pos (KV cache + DeltaNet state are cumulative)
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        let kv = m.kv_cache.as_mut().unwrap();
        let dn = m.dn_state.as_mut().unwrap();

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
        if let Some(ref ev) = m.eviction {
            let window = ev.budget() + ev.beta();
            let mut remaining: &[u32] = &new_tokens;
            while !remaining.is_empty() {
                let space = window.saturating_sub(m.seq_pos).max(1);
                let chunk_len = remaining.len().min(space);
                let (chunk, rest) = remaining.split_at(chunk_len);
                qwen35::forward_prefill_batch(
                    gpu, weights, config, chunk, m.seq_pos, kv, dn, scratch,
                    None, None, None, None,
                ).unwrap();
                m.seq_pos += chunk_len;
                if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                    m.seq_pos = new_phys;
                }
                remaining = rest;
            }
        } else {
            qwen35::forward_prefill_batch(
                gpu, weights, config, &new_tokens, m.seq_pos, kv, dn, scratch,
                None, None, None, None,
            ).unwrap();
            m.seq_pos += new_tokens.len();
        }
        m.conversation_tokens.extend_from_slice(&new_tokens);

        // ngram scope for the repeat penalty: ONLY generated tokens (never the
        // prompt). Prior design included the user's prompt as an anti-loop
        // anchor, but that penalizes the very tokens we're asked to recall
        // (names, numbers, facts) under MQ4/MQ6 quantizations that are more
        // RP-sensitive than llama.cpp's Q4_K. First sample: empty scope (no
        // generated tokens yet); subsequent samples: generated-so-far only.
        let ngram_scope_start = m.conversation_tokens.len();

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
        let repeat_buf_cap = scratch.repeat_buf.buf.size() / 4;

        // Build the list of paired (open, close) attractor pairs once;
        // sampler::collect_unclosed_attractor_blocks decides per-call
        // which openers (if any) trip the depth threshold.
        let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
            .into_iter()
            .chain(think_pair.into_iter())
            .collect();

        // First sample: use conversation so far as scope.
        let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
        // #111 attractor block: empty `ngram_scope` on first sample (no
        // generated tokens yet), so the unclosed-depth is always 0 and
        // `blocked` is empty. Still call collect_* for symmetry with
        // the loop body, in case a future change moves this block into
        // a multi-step warmup.
        let mut blocked0: Vec<u32> = Vec::new();
        sampler::collect_unclosed_attractor_blocks(
            ngram_scope,
            &attractor_pairs,
            20,
            2,
            &mut blocked0,
        );
        let cfg0 = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            // Window is bounded by the GPU repeat_buf capacity (sized
            // at 64 in ForwardScratch::new). Pre-PR3 code did this
            // bound by setting `scope_start = len - repeat_buf_cap`
            // and passing `scope.len()` to the kernel; we let
            // sampler::sample do the same `min(window, buf_cap)`
            // internally.
            repeat_window: repeat_buf_cap,
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
        let mut filter = EosFilter::new(EosFilterConfig::default());
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
        // Implementation lives in `hipfire_runtime::loop_guard`; defaults read from
        // HIPFIRE_NGRAM_LOOP_THRESHOLD (default 8, 0 = disabled) and
        // HIPFIRE_NGRAM_WINDOW (default 256). See loop_guard.rs.
        let loop_guard = hipfire_runtime::loop_guard::LoopGuard::from_env();

        // `while` instead of `for 0..max_tokens` so budget-alert injection
        // (which increments `generated` beyond the iteration count) can't
        // push generated past max_tokens: each loop start rechecks the cap.
        while generated < max_tokens {
            generated += 1;
            m.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            emit_committed_event(stdout, id, next_token, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
            // Incremental UTF-8 + filter routing: feed only the new
            // bytes since last call, let the filter buffer any partial
            // codepoint or marker prefix until disambiguated.
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                let text = std::str::from_utf8(&text_bytes).unwrap();
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                let _ = stdout.flush();
            }

            // Write this token's K/V to the cache FIRST so the next turn
            // always starts from a fully-written context. Breaking before
            // forward_scratch used to leave a hole at the im_end/eos
            // position — the next turn then attended over zero-init K/V
            // at that slot.
            //
            // Under eviction, m.seq_pos is the *physical* write slot; we
            // advance and call maybe_evict immediately so the next write
            // never overruns physical_cap. compact_offset bookkeeping on
            // the cache itself keeps RoPE phase correct across evictions.
            qwen35::forward_scratch(gpu, weights, config, next_token, m.seq_pos, kv, dn, scratch).unwrap();
            m.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                    m.seq_pos = new_phys;
                }
            }

            if next_token == config.eos_token { break; }
            if im_end_token == Some(next_token) { break; }
            if tokenizer.is_terminator(next_token) { break; }

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
                    if !prev_in_think { think_count = 1; } else { think_count += 1; }
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
                        qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch).unwrap();
                        m.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                                m.seq_pos = new_phys;
                            }
                        }
                        m.conversation_tokens.push(t);
                        streamed_tokens.push(t);
                        emit_committed_event(stdout, id, t, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
                        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes = &all_bytes[bytes_fed_to_filter..];
                        bytes_fed_to_filter = all_bytes.len();
                        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                            let text = std::str::from_utf8(&text_bytes).unwrap();
                            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                            let _ = stdout.flush();
                        }
                        generated += 1;
                    }
                    think_count = 0;
                    prev_in_think = false;
                    if generated >= max_tokens { break; }
                }
            }

            // N-gram loop detector: check if any 4-gram in the recent window
            // repeats excessively. When detected, emit an info message and
            // force EOS to prevent wasting the remaining token budget on
            // repetitive output. Logic lives in `hipfire_runtime::loop_guard`.
            if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
                loop_guard.check(&streamed_tokens)
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
            if !alert_fired && budget_alert_at_tok > 0 && generated >= budget_alert_at_tok && !budget_alert_text.is_empty() {
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
                    let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#, id);
                    let _ = stdout.flush();
                    // Fall through — resample next token as normal
                    let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
                    let mut blocked: Vec<u32> = Vec::new();
                    sampler::collect_unclosed_attractor_blocks(
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
                let need_kv = m.seq_pos + nudge_len + (max_tokens - generated - nudge_len) + nl.len();
                if nudge_len > 0 && (m.eviction.is_some() || need_kv <= m.physical_cap) {
                    for &tok in &nudge_tokens[..nudge_len] {
                        m.conversation_tokens.push(tok);
                        streamed_tokens.push(tok);
                        emit_committed_event(stdout, id, tok, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
                        // Emit the injected token's text to stdout so the client
                        // sees it as part of the stream (will be inside <think>
                        // if that's the current state, and get stripped client-
                        // side just like any other think token).
                        let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                        bytes_fed_to_filter = all_bytes2.len();
                        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes2) {
                            let t = std::str::from_utf8(&text_bytes).unwrap();
                            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&t).unwrap_or_default());
                            let _ = stdout.flush();
                        }
                        qwen35::forward_scratch(gpu, weights, config, tok, m.seq_pos, kv, dn, scratch).unwrap();
                        m.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                                m.seq_pos = new_phys;
                            }
                        }
                        generated += 1;
                    }
                } else if nudge_len < nudge_tokens.len() {
                    let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert clipped or skipped: nudge_len={} budget_left={}"}}"#, id, nudge_len, budget_left);
                    let _ = stdout.flush();
                } else {
                    let _ = writeln!(stdout, r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not enough KV headroom"}}"#, id);
                    let _ = stdout.flush();
                }
                // Respect max_tokens: if injection used the remainder, bail
                // before sampling another model token.
                if generated >= max_tokens { break; }
            }

            // Decide which paired-opener tokens (if any) trip the depth
            // threshold over a 20-token window. #111 attractor block —
            // cheap when not tripped, ~5 µs per blocked token when
            // tripped (single 4-byte H2D into the logits buffer
            // performed inside sampler::sample).
            let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
            let mut blocked: Vec<u32> = Vec::new();
            sampler::collect_unclosed_attractor_blocks(
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
        // m.seq_pos is already the "next physical write slot" — advanced
        // per-token in the decode loop above, and evicted back down to
        // `budget` whenever maybe_evict fired. No post-loop fix-up needed.

        // ChatML requires \n after <|im_end|>. Run it through forward so KV cache
        // and DeltaNet state stay in sync with seq_pos.
        if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
            for &t in &nl {
                qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch).unwrap();
                m.seq_pos += 1;
                if let Some(ref ev) = m.eviction {
                    if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                        m.seq_pos = new_phys;
                    }
                }
                m.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
        let prefill_tok_s = if prefill_s > 0.0 { prefill_tokens as f64 / prefill_s } else { 0.0 };
        let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}{}}}"#,
            id, generated, tok_s, prefill_tokens,
            prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0,
            pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha),
        );
        let _ = stdout.flush();
    } else {
        // Qwen3 / LLaMA path -- multi-turn aware
        let config = m.llama_config.as_ref().unwrap();
        let weights = m.llama_weights.as_ref().unwrap();
        let scratch = m.llama_scratch.as_ref().unwrap();
        let kv = m.llama_kv.as_mut().unwrap();

        let mut rng_state = 42u32;
        for (i, &tok) in new_tokens.iter().enumerate() {
            let pos = m.seq_pos + i;
            let (_, rng) = llama::forward_scratch(gpu, weights, config, tok, pos, kv, scratch, temp, top_p, rng_state, 0, 1.0).unwrap();
            rng_state = rng;
        }
        let this_turn_prompt_len_llama = new_tokens.len();
        m.seq_pos += new_tokens.len();
        m.conversation_tokens.extend_from_slice(&new_tokens);
        let ngram_scope_start_llama = m.conversation_tokens.len() - this_turn_prompt_len_llama;

        let mut out_bytes = [0u8; 8];
        gpu.hip.memcpy_dtoh(&mut out_bytes, &scratch.sample_buf.buf).unwrap();
        let mut next_token = u32::from_ne_bytes([out_bytes[0], out_bytes[1], out_bytes[2], out_bytes[3]]);
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
        let mut filter = EosFilter::new(EosFilterConfig::default());

        for _ in 0..max_tokens {
            generated += 1;
            m.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            emit_committed_event(stdout, id, next_token, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                let text = std::str::from_utf8(&text_bytes).unwrap();
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                let _ = stdout.flush();
            }

            // Scope repeat_buf to this turn's prompt + generated tokens
            // (same logic as the Qwen3.5 path: prompt anchor + current turn).
            let rw = repeat_window.min(64);
            let scope_start = ngram_scope_start_llama.max(m.conversation_tokens.len().saturating_sub(rw));
            let hist_slice = &m.conversation_tokens[scope_start..];
            let hist_bytes: Vec<u8> = hist_slice.iter().flat_map(|t| t.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&scratch.repeat_buf.buf, &hist_bytes).unwrap();

            // Write K/V for this token FIRST so the next turn's context is
            // always fully populated. The sampled next_token from this call
            // is discarded when we break on im_end/eos — wasteful by one
            // launch but avoids a KV cache gap at the terminator.
            let pos = m.seq_pos + generated - 1;
            let (tok, rng) = llama::forward_scratch(gpu, weights, config, next_token, pos, kv, scratch, temp, top_p, rng_state, hist_slice.len(), repeat_penalty).unwrap();

            if next_token == config.eos_token { break; }
            if im_end_token == Some(next_token) { break; }
            if tokenizer.is_terminator(next_token) { break; }

            next_token = tok;
            rng_state = rng;
        }
        m.seq_pos += generated;

        // ChatML \n boundary — run through forward to keep KV cache in sync
        if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
            for &t in &nl {
                let (_, rng2) = llama::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, scratch, temp, top_p, rng_state, 0, 1.0).unwrap();
                rng_state = rng2;
                m.seq_pos += 1;
                m.conversation_tokens.push(t);
            }
        }

        let t_end = Instant::now();
        let total_s = t_end.duration_since(t0).as_secs_f64();
        let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
        let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
        let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
        let prefill_tok_s = if prefill_s > 0.0 { prefill_tokens as f64 / prefill_s } else { 0.0 };
        let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}{}}}"#,
            id, generated, tok_s, prefill_tokens,
            prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0,
            pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha),
        );
        let _ = stdout.flush();
    }
}

fn generate_vl(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, id: &str, prompt: &str, system_prompt: Option<&str>, image_path: &str, temp: f32, top_p: f32, max_tokens: usize, repeat_penalty: f32, repeat_window: usize) {
    // Capacity guard — VL prompts include vision tokens + text + ChatML framing
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let vision_config = m.vision_config.as_ref().unwrap();
    let n_patches = (IMAGE_SIZE / vision_config.patch_size) * (IMAGE_SIZE / vision_config.patch_size);
    let n_visual_tokens = n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);
    let prompt_est = tokenizer.encode(prompt).len() + n_visual_tokens + 20; // text + vision + ChatML overhead
    if m.eviction.is_none() && m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[daemon/vl] context full ({}/{}) — resetting conversation", m.seq_pos, m.max_seq);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        // Zero DeltaNet state on reset
        if let Some(ref dn) = m.dn_state {
            for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        }
        if let Some(kv) = m.kv_cache.as_mut() { kv.compact_offset = 0; }
    }
    let config = m.q35_config.as_ref().unwrap();
    let vision_config = m.vision_config.as_ref().unwrap();
    let vision_weights = m.vision_weights.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch = m.q35_scratch.as_ref().unwrap();
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();

    // Load and preprocess image (smart resize matching HuggingFace)
    eprintln!("[VL-DEBUG] preprocessing image: {}", image_path);
    let (pixels, img_h, img_w) = hipfire_arch_qwen35_vl::image::load_and_preprocess(
        Path::new(image_path),
        vision_config.patch_size,
        vision_config.spatial_merge_size,
    );
    eprintln!("[VL-DEBUG] preprocessed: {}x{}", img_w, img_h);
    let grid_h = img_h / vision_config.patch_size;
    let grid_w = img_w / vision_config.patch_size;
    let n_patches = grid_h * grid_w;
    let n_visual_tokens = n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);

    // Extract patches and run vision encoder
    let patches = hipfire_arch_qwen35_vl::image::extract_patches(
        &pixels, 3, img_h, img_w,
        vision_config.patch_size, vision_config.temporal_patch_size,
    );
    let visual_tokens = qwen35_vl::vision_forward(gpu, vision_weights, vision_config, &patches, grid_h, grid_w)
        .expect("vision forward failed");

    // Build VL prompt via hipfire_runtime::prompt_frame. The VL user body splices
    // vision tokens (`<|vision_start|>` + N × `<|image_pad|>` +
    // `<|vision_end|>`) BEFORE the textual prompt, separated by a newline.
    // We pre-assemble that as the user-body token sequence and pass it
    // through `build_with_user_tokens` so the role/newline/im_end
    // scaffolding stays canonical.
    let nl = tokenizer.encode("\n");
    let im_end = tokenizer.encode("<|im_end|>");
    let q_tokens = tokenizer.encode(prompt);

    let mut user_body: Vec<u32> = Vec::with_capacity(n_visual_tokens + q_tokens.len() + 4);
    user_body.push(VISION_START_ID);
    for _ in 0..n_visual_tokens {
        user_body.push(IMAGE_PAD_ID);
    }
    user_body.push(VISION_END_ID);
    user_body.extend_from_slice(&nl);
    user_body.extend_from_slice(&q_tokens);

    let prompt_tokens = hipfire_runtime::prompt_frame::ChatFrame {
        tokenizer,
        system: if m.seq_pos == 0 { system_prompt } else { None },
        user: "", // unused: we pass tokens directly via build_with_user_tokens
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        raw: false,
    }
    .build_with_user_tokens(&user_body);

    // KV-budget guard — physical_cap without eviction, absolute window with.
    // Mirrors the textual generate() contract; reserves trailer slots so
    // natural im_end termination can still write the ChatML \n.
    let trailer = nl.len();
    let absolute_pos_vl = m.seq_pos + kv.compact_offset;
    let over_budget = if m.eviction.is_none() {
        m.seq_pos + prompt_tokens.len() + max_tokens + trailer > m.physical_cap
    } else {
        absolute_pos_vl + prompt_tokens.len() + max_tokens + trailer > m.max_seq
    };
    if over_budget {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > cap={} — reload model with a larger max_seq"}}"#,
            id, m.seq_pos, prompt_tokens.len(), max_tokens, trailer,
            if m.eviction.is_none() { m.physical_cap } else { m.max_seq },
        );
        let _ = stdout.flush();
        return;
    }

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let prefill_tokens = prompt_tokens.len();
    let t0 = Instant::now();

    // Prefill with vision token embedding for IMAGE_PAD positions.
    // VL prefill is already per-token (forward_scratch_embed isn't batched),
    // so we advance m.seq_pos in-loop and call maybe_evict after every write.
    let mut visual_idx = 0usize;
    for &token in prompt_tokens.iter() {
        if token == IMAGE_PAD_ID && visual_idx < n_visual_tokens {
            let emb = &visual_tokens[visual_idx * config.dim..(visual_idx + 1) * config.dim];
            qwen35::forward_scratch_embed(gpu, weights, config, emb, m.seq_pos, kv, dn, scratch)
                .expect("forward_scratch_embed failed");
            visual_idx += 1;
        } else {
            qwen35::forward_scratch(gpu, weights, config, token, m.seq_pos, kv, dn, scratch)
                .expect("forward_scratch failed");
        }
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                m.seq_pos = new_phys;
            }
        }
    }
    m.conversation_tokens.extend_from_slice(&prompt_tokens);

    // Generate. CPU-side sampling — VL path predates the GPU sampler
    // and downloads logits each step. The order of ops is preserved
    // from pre-PR3:
    //   - first sample: top-p only (no penalty, no ngram block);
    //   - subsequent samples: positional ngram-block, then
    //     repeat_penalty, then top-p sample.
    //
    // The positional ngram block writes -INF to the
    // *next-token-after-an-earlier-ngram-match* position — a
    // per-history-pattern decision rather than the identity-only
    // contract of SamplerConfig::blocked_tokens — so it stays inline
    // rather than going through the SamplerConfig path.
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    let vl_cfg_first = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty: 1.0,
        repeat_window: 0,
        blocked_tokens: Vec::new(),
    };
    let vl_cfg = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window,
        blocked_tokens: Vec::new(),
    };
    let mut next_token = sampler::sample_cpu(&mut logits, &[], &vl_cfg_first);
    let t_prefill = Instant::now();
    let mut generated = 0;

    for _ in 0..max_tokens {
        generated += 1;
        m.conversation_tokens.push(next_token);
        emit_committed_event(stdout, id, next_token, generated - 1, t0.elapsed().as_millis() as u64);
        let text = tokenizer.decode(&[next_token]);
        let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
        let _ = stdout.flush();

        if next_token == config.eos_token { break; }
        if im_end_token == Some(next_token) { break; }
        if tokenizer.is_terminator(next_token) { break; }

        qwen35::forward_scratch(gpu, weights, config, next_token, m.seq_pos, kv, dn, scratch).unwrap();
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                m.seq_pos = new_phys;
            }
        }
        logits = gpu.download_f32(&scratch.logits).unwrap();
        llama::apply_ngram_block(&mut logits, &m.conversation_tokens);
        next_token = sampler::sample_cpu(&mut logits, &m.conversation_tokens, &vl_cfg);
    }

    // ChatML \n boundary — run through forward to keep KV cache + DeltaNet in sync
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch).unwrap();
            m.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                    m.seq_pos = new_phys;
                }
            }
            m.conversation_tokens.push(t);
        }
    }

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
    let prefill_tok_s = if prefill_s > 0.0 { prefill_tokens as f64 / prefill_s } else { 0.0 };
    let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
        id, generated, tok_s, prefill_tokens,
        prefill_s * 1000.0, prefill_tok_s, decode_tok_s, prefill_s * 1000.0
    );
    let _ = stdout.flush();
}
