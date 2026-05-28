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

use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_arch_llama::Llama;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, LayerType, Qwen35ScratchSet};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_arch_qwen35::mtp_head::{self, Qwen35MtpHead, MtpKvMode};
use hipfire_arch_qwen35::mtp_spec::{self, MtpSpecState, MtpSamplingConfig};
use hipfire_arch_qwen35::speculative::{
    self, DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use hipfire_arch_qwen35_vl::image;
use base64::Engine;
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
/// CPU-side counterpart that applies the same depth-tracking attractor
/// block directly to a freshly-downloaded logits vector. Avoids the
/// htod-memcpy + redownload roundtrip the GPU variant required per token.
fn block_attractor_unclosed_cpu(
    logits: &mut [f32],
    history: &[u32],
    open_id: u32,
    close_id: u32,
    window: usize,
    threshold: usize,
) {
    if window == 0 || threshold == 0 || open_id == close_id { return; }
    let start = history.len().saturating_sub(window);
    let mut depth: i32 = 0;
    for &t in &history[start..] {
        if t == open_id { depth += 1; }
        else if t == close_id && depth > 0 { depth -= 1; }
    }
    if depth >= threshold as i32 {
        if let Some(slot) = logits.get_mut(open_id as usize) {
            *slot = f32::NEG_INFINITY;
        }
    }
}

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

/// Base stats for the `{"type":"done"}` event. All four generate
/// functions emit a done event with the same seven base fields; the
/// only divergence is path-specific extras (e.g. DFlash adds
/// `"dflash":true,"tau":..,"cycles":..`; MTP adds
/// `"spec_path":"mtp","mtp_k":..,"tau":..,"accept_rate":..,"cycles":..,"mtp_sampling":..`;
/// AR adds `pflash` info via a separate helper).
///
/// Step 2 of docs/plans/mtp_multi_refactor.md v2.1: extract the
/// shared base-emission so each call site shrinks from ~6 lines of
/// `writeln!` plumbing to one helper call + the path extras string.
#[derive(Clone, Copy)]
struct DoneStats {
    tokens: usize,
    tok_s: f64,
    prefill_tokens: usize,
    prefill_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ms: f64,
}

/// Emit a `{"type":"done", ...}` event with the seven base fields plus
/// optional path-specific extras. `path_extras` MUST start with a
/// leading comma (e.g. `,"dflash":true,"tau":3.40,"cycles":12`) when
/// non-empty, OR be the empty string. Mirrors the existing
/// `pflash_done_fragment` convention so the AR call sites can keep
/// using that helper unchanged.
///
/// Flushes stdout after writing — matches the previous inline
/// `writeln! + stdout.flush()` pattern at every call site.
fn emit_done_event(
    stdout: &mut std::io::Stdout,
    id: &str,
    stats: &DoneStats,
    path_extras: &str,
) {
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}{}}}"#,
        id, stats.tokens, stats.tok_s, stats.prefill_tokens,
        stats.prefill_ms, stats.prefill_tok_s, stats.decode_tok_s, stats.ttft_ms,
        path_extras,
    );
    let _ = stdout.flush();
}

/// Inputs to [`build_prompt_frame`], the shared chat-framing helper for
/// generate / generate_multi / generate_mtp. Bundled as a struct rather
/// than 11 positional args so the call sites stay legible.
///
/// Why this exists: each of the three functions has a ~70-line
/// Jinja-or-Plain branch that does the same thing — render via
/// JinjaChatFrame when `HIPFIRE_JINJA_CHAT=1` + first turn + template
/// present, else hand-roll a ChatFrame with `build_with_user_tokens`.
/// Path-specific differences (which q_tokens to feed, how the system
/// prompt is gated on `seq_pos`) are captured in this struct.
///
/// NOT used by generate_dflash — that path is single-turn (no
/// `seq_pos == 0` gate on Jinja) AND uses `ChatFrame.build()` with the
/// raw prompt string instead of pre-tokenized user tokens. The
/// resulting divergence isn't worth a switchable parameter; dflash
/// keeps its inline framing per the v2.1 plan.
struct FrameInputs<'a> {
    tokenizer: &'a hipfire_runtime::tokenizer::Tokenizer,
    /// `m.chat_template.as_deref()`; None disables the Jinja path.
    chat_template: Option<&'a str>,
    system_prompt: Option<&'a str>,
    /// User prompt text — passed to JinjaChatFrame.user. The Plain
    /// fallback path uses `user_tokens` instead and ignores this.
    prompt_text: &'a str,
    /// Pre-tokenized user content. For PFlash-enabled paths
    /// (generate, generate_multi) this is the post-compression
    /// `q_tokens`. For generate_mtp this is `raw_q_tokens`. Either
    /// way the Plain fallback wraps it with chat-template scaffolding
    /// via `build_with_user_tokens`.
    user_tokens: &'a [u32],
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    max_think_tokens: usize,
    tools: Option<&'a [serde_json::Value]>,
    messages_history: Option<&'a [hipfire_runtime::prompt_frame::Message]>,
    /// Current `m.seq_pos`. Used to gate the Jinja path on first turn,
    /// and to gate the Plain system-prompt scaffolding on multi-turn.
    seq_pos: usize,
    /// Label spliced into the "[daemon] jinja render failed in $X path"
    /// warning. Use `"pp"`, `"mtp"`, or `""` (empty = AR).
    err_path_label: &'static str,
}

/// Build the prompt-frame token vector that goes into prefill. Step 2.2
/// of docs/plans/mtp_multi_refactor.md v2.1; shared by 3 of the 4 in-
/// scope generate functions.
fn build_prompt_frame(inputs: &FrameInputs<'_>) -> Vec<u32> {
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && inputs.seq_pos == 0 && inputs.chat_template.is_some();
    if try_jinja {
        let template = inputs.chat_template.unwrap();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer: inputs.tokenizer,
            template,
            system: inputs.system_prompt,
            user: inputs.prompt_text,
            enable_thinking: inputs.max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if inputs.tools.is_some() || inputs.messages_history.is_some() {
            let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
            let messages_slice: &[hipfire_runtime::prompt_frame::Message] = match inputs.messages_history {
                Some(msgs) => msgs,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = inputs.system_prompt {
                        v.push(hipfire_runtime::prompt_frame::Message {
                            role: hipfire_runtime::prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
                        content: inputs.prompt_text.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    synthesized = v;
                    &synthesized
                }
            };
            frame.render_messages(messages_slice, inputs.tools, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => inputs.tokenizer.encode(&rendered),
            Err(e) => {
                if inputs.err_path_label.is_empty() {
                    eprintln!("[daemon] jinja render failed ({e}) — falling back to Plain");
                } else {
                    eprintln!("[daemon] jinja render failed in {} path ({e}) — falling back to Plain",
                              inputs.err_path_label);
                }
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer: inputs.tokenizer,
                    system: inputs.system_prompt,
                    user: "",
                    assistant_prefix: inputs.assistant_prefix,
                    raw: false,
                }
                .build_with_user_tokens(inputs.user_tokens)
            }
        }
    } else {
        // Plain path. Multi-turn aware: system scaffolding only on the
        // first turn so continuations don't re-emit the system message.
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer: inputs.tokenizer,
            system: if inputs.seq_pos == 0 { inputs.system_prompt } else { None },
            user: "",
            assistant_prefix: inputs.assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(inputs.user_tokens)
    }
}

/// Bundled inputs to the generate family. Step 3 of
/// docs/plans/mtp_multi_refactor.md v2.1: the four generate
/// functions (generate / generate_multi / generate_mtp /
/// generate_dflash) collectively take 21 distinct params. Threading
/// them positionally is unreadable. `GenerateCtx<'a>` collects them
/// in one struct so each signature becomes
/// `fn generate_x(m, gpu_or_gpus, ctx)`.
///
/// Field-by-field divergence handling:
/// - Sampling fields (`temp`, `top_p`, `repeat_penalty`,
///   `repeat_window`) are always passed; dflash ignores them since
///   it's greedy-only by construction.
/// - `budget_alert_at_tok` / `budget_alert_text` only fire in the AR
///   path today. MTP/DFlash ignore them.
/// - `drafter_gpu` is only consumed by AR's PFlash compress site;
///   stored as `&mut` so the caller can borrow it back after the
///   ctx scope ends.
/// - `pflash_state` / `pflash_cfg` consumed by AR's compress site;
///   resolved at dispatcher level into `pflash_bypass_reason` /
///   `pflash_alpha` for the dflash leaf (dflash doesn't itself run
///   compress, so it only needs the resolved indicators).
struct GenerateCtx<'a> {
    // I/O
    stdout: &'a mut std::io::Stdout,
    id: &'a str,
    // Prompt
    prompt: &'a str,
    system_prompt: Option<&'a str>,
    tools: Option<&'a [serde_json::Value]>,
    messages_history: Option<&'a [hipfire_runtime::prompt_frame::Message]>,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    // Sampling
    temp: f32,
    top_p: f32,
    repeat_penalty: f32,
    repeat_window: usize,
    // Decode budget
    max_tokens: usize,
    max_think_tokens: usize,
    budget_alert_at_tok: usize,
    budget_alert_text: &'a str,
    // Co-GPU resources (AR PFlash path only)
    drafter_gpu: Option<&'a mut rdna_compute::Gpu>,
    pflash_state: Option<&'a mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&'a hipfire_arch_qwen35::pflash::PflashConfig>,
    /// Resolved by the dispatcher when generate routes to
    /// generate_dflash: when PFlash mode is non-Off AND a drafter is
    /// loaded, dflash gets a documented `pflash_bypass_reason` event
    /// so operators can see why compression didn't fire on the
    /// DFlash path. `None` when PFlash wasn't requested or the
    /// dispatcher didn't compute it.
    pflash_bypass_reason: Option<&'a str>,
    /// `pflash_cfg.alpha` echoed by the dflash done event for PRD
    /// §3.1 compliance. `None` when PFlash wasn't requested.
    pflash_alpha: Option<f32>,
}

/// Which generate path the dispatcher routes a request to. Step 4 of
/// docs/plans/mtp_multi_refactor.md v2.1: making the path decision
/// explicit catches load-handler refusal-matrix bugs (a combo that
/// shouldn't reach a function actually CAN) before they cascade into
/// runtime confusion.
///
/// Note: `SpecPath` depends on BOTH `LoadedModel` state AND request-
/// scope args (temp, repeat_penalty, max_think_tokens,
/// assistant_prefix) because today's dispatch in `generate` mixes
/// them. See [`pick_path`].
///
/// Exhaustive over POST-LOAD-VALIDATION combos only. The load
/// handler refuses several combos at request time (MTP+DFlash,
/// CASK+pp>1, etc.), so `pick_path` doesn't need to handle them.
///
/// Named `SpecPath` (not `Path`) to avoid colliding with `std::path::Path`
/// which is used elsewhere in this file for model file paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecPath {
    /// Qwen2 arch (arch_id 7). Routed to generate_qwen2 unchanged
    /// in v1; not folded into the unification.
    Qwen2,
    /// pp=1, no spec, AR sampling. The "generate inline AR body".
    Ar,
    /// pp>1 trunk, AR sampling. generate_multi today.
    PpAr,
    /// pp=1, MTP spec-decode (single-gpu or hetero via drafter_state).
    /// generate_mtp today.
    Mtp,
    /// pp>1, MTP spec-decode (Stage 2b's payoff path). Will exist
    /// after step 5d lands; today falls through to PpAr.
    PpMtp,
    /// pp=1, DFlash spec-decode (greedy-only). generate_dflash today.
    Dflash,
}

/// Decide which path a request would take given the loaded model
/// state and the request's runtime args. Used to assert the
/// dispatch decision is consistent with the path-arm convention,
/// AND will become the central match in step 5's unified
/// generate_qwen35.
fn pick_path(m: &LoadedModel, ctx: &GenerateCtx<'_>) -> SpecPath {
    // Mirror the dispatch order in `generate`: arch_id check first,
    // then PP, then DFlash (with all its runtime gates), then MTP
    // (with its own runtime gates), else AR.
    if m.arch_id == 7 {
        return SpecPath::Qwen2;
    }
    if m.pp > 1 {
        // PP+MTP routes to Mtp ONLY when step 5d's SpecPath::PpMtp arm
        // exists. Today (pre-step-5d) the dispatcher falls through to
        // generate_multi for the PP+MTP case — record that as PpAr so
        // the assertion matches today's behavior. Step 5d flips this
        // to PpMtp.
        if m.mtp.is_some() {
            return SpecPath::PpMtp;
        }
        return SpecPath::PpAr;
    }
    let budgeted_thinking_needs_ar = ctx.max_think_tokens > 0
        && !matches!(ctx.assistant_prefix,
                     hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink);
    if m.dflash.is_some() && ctx.temp <= 1e-6 && (m.arch_id == 5 || m.arch_id == 6)
        && !budgeted_thinking_needs_ar {
        return SpecPath::Dflash;
    }
    let mtp_blocked_by_penalty = ctx.repeat_penalty > 1.0 + 1e-3;
    if m.mtp.is_some() && (m.arch_id == 5 || m.arch_id == 6)
        && !budgeted_thinking_needs_ar && !mtp_blocked_by_penalty {
        return SpecPath::Mtp;
    }
    SpecPath::Ar
}

/// Per-token streaming step. The pattern that appears in every
/// generate function's decode loop (and in the max_think force-close
/// inner loop, and in the im_end trailer write): push to
/// `streamed_tokens`, fire the committed-event, decode bytes
/// incrementally, feed the new slice to the EOS filter, and emit a
/// `{"type":"token","text":...}` event when the filter releases bytes.
///
/// Callers separately push to `m.conversation_tokens` if they want
/// the token tracked for cumulative state. Seed-emit sites don't —
/// the seed is included in the prefill prompt and KV state already
/// reflects it. Decode-step sites do.
///
/// Step 2.5 of docs/plans/mtp_multi_refactor.md v2.1.
fn stream_token_step(
    stdout: &mut std::io::Stdout,
    id: &str,
    tok: u32,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    streamed_tokens: &mut Vec<u32>,
    bytes_fed_to_filter: &mut usize,
    filter: &mut EosFilter,
    t_start: Instant,
) {
    streamed_tokens.push(tok);
    emit_committed_event(stdout, id, tok, streamed_tokens.len() - 1, t_start.elapsed().as_millis() as u64);
    let all_bytes = tokenizer.decode_bytes(streamed_tokens);
    let new_bytes = &all_bytes[*bytes_fed_to_filter..];
    *bytes_fed_to_filter = all_bytes.len();
    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
        if let Ok(text) = std::str::from_utf8(&text_bytes) {
            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
            let _ = stdout.flush();
        }
    }
}

/// Update the `<think>...</think>` state from the decoded-so-far bytes.
/// Returns `(in_think_now, new_think_count)`. Caller stores
/// `prev_in_think = in_think_now`, `think_count = new_count`, then
/// decides whether to force-close (or break, etc.) when
/// `new_count >= max_think_tokens`.
///
/// Shared between generate / generate_multi / generate_dflash. NOT
/// used by generate_mtp because that path's detector uses
/// unterminated `"<think"` / `"</think"` patterns instead of the full
/// `"<think>"` / `"</think>"` (deliberate; preserves MTP's existing
/// cross-token-fragment matching). Step 2.4 of
/// docs/plans/mtp_multi_refactor.md v2.1.
fn detect_think_state(
    decoded_bytes: &[u8],
    prev_in_think: bool,
    prev_think_count: usize,
) -> (bool, usize) {
    let raw_str = std::str::from_utf8(decoded_bytes).unwrap_or("");
    let open_idx = raw_str.rfind("<think>");
    let close_idx = raw_str.rfind("</think>");
    let in_think = match (open_idx, close_idx) {
        (Some(o), Some(c)) => o > c,
        (Some(_), None) => true,
        _ => false,
    };
    let new_count = if in_think {
        if !prev_in_think { 1 } else { prev_think_count + 1 }
    } else {
        0
    };
    (in_think, new_count)
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

/// Cap on the *encoded* base64 string length the daemon will accept on the
/// IPC. ~40 MB encoded → ~30 MB raw image bytes (4/3 expansion).
const MAX_BASE64_ENCODED_LEN: usize = 40 * 1024 * 1024;

/// Emit a single-line `{"type":"error","id":"...","message":"..."}` JSON
/// line on the IPC stream. Uses `serde_json` so user-controlled error
/// strings (image decoder messages, base64 errors) can't desync the
/// protocol by injecting embedded `"`, `\`, or newline bytes.
fn write_error(stdout: &mut std::io::Stdout, id: &str, message: &str) {
    let line = serde_json::json!({
        "type": "error",
        "id": id,
        "message": message,
    });
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

enum ImageSource<'a> {
    Path(&'a str),
    Base64(&'a str),
}

struct GenerateVLParams<'a> {
    id: &'a str,
    prompt: &'a str,
    system_prompt: Option<&'a str>,
    image_source: ImageSource<'a>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    max_think_tokens: usize,
}

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

/// Optional MTP (multi-token prediction) speculative-decoding state.
/// Populated when `load` supplies an MTP head (sidecar `.mtp` or bundled
/// `.mq4-mtp` trailer). Used by `generate_mtp` for both greedy and sampling
/// decode — unlike DFlash, MTP supports temperature > 0 via residual
/// acceptance.
struct MtpState {
    head: Qwen35MtpHead,
    spec_state: MtpSpecState,
    max_n: usize,
    p_min: f32,
    compressed: bool,
    /// Stage 2 (PP+MTP combo): drafter-side state when running hetero
    /// MTP. `None` for the single-gpu MTP path. When `Some`, the MTP
    /// head + scratch + KV + token_embd mirror live on the drafter gpu
    /// indicated by the spec function call (which is `output_device` in
    /// the PP=2 layout); the cycle dispatcher selects
    /// `spec_step_mtp_compressed_serial_hetero` instead of
    /// `spec_step_mtp_compressed_serial`.
    drafter_state: Option<hipfire_arch_qwen35::mtp_spec::MtpHeteroDrafterState>,
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
    // Qwen2 state (arch_id=7 — hipfire-arch-qwen2 standalone). The
    // KV cache lives inside Qwen2State, so there's no separate
    // qwen2_kv field. None on every other arch path.
    qwen2_config: Option<qwen2::Qwen2Config>,
    qwen2_weights: Option<qwen2::Qwen2Weights>,
    qwen2_state: Option<qwen2::Qwen2State>,
    // dots.ocr state (arch_id=8 — Qwen2-VL family). The text decoder is
    // Qwen2: `dots_ocr_config.text` / `dots_ocr_weights.text` feed
    // `qwen2::forward_step*`, and the per-step decode state reuses the
    // `qwen2_state` field above. `dots_ocr_weights.vision` holds the
    // resident vision-tower weights for `dots_ocr::vision_forward`.
    dots_ocr_config: Option<dots_ocr::DotsOcrConfig>,
    dots_ocr_weights: Option<dots_ocr::DotsOcrWeights>,
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
    // MTP speculative decoding state (populated when load supplied an MTP head
    // sidecar or bundled trailer). Mutually exclusive with DFlash — the load
    // handler errors if both are requested.
    mtp: Option<MtpState>,
    // Upstream HF Jinja chat_template, extracted from the HFQ
    // `tokenizer_config.chat_template` at load time. `None` when the source
    // model didn't ship one (rare for instruct models). Only consumed when
    // `HIPFIRE_JINJA_CHAT=1` is set; otherwise the daemon's hand-rolled
    // `prompt_frame::ChatFrame::Plain` scaffolding is used as today.
    //
    // Stage 2 partial: AR generate() path only. DFlash, multi-GPU PP>1, and
    // VL paths still hit the Plain scaffold.
    chat_template: Option<String>,
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
    // Hetero PFlash: when prefill_drafter_device differs from the target,
    // the drafter weights/KV/scratch live on a sibling device. The compress
    // output is a host-side Vec<u32>, so no peer-copy is needed — generate
    // routes maybe_compress_prompt to this handle, decode stays on target.
    // None means the drafter shares the target gpu (single-card, unchanged).
    let mut pflash_drafter_gpu: Option<rdna_compute::Gpu> = None;

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
                // weights. Disabled by default; enable with mmq_screen=true
                // (or HIPFIRE_MMQ_SCREEN=1) when adding new quant formats.
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
                // -1 = drafter shares the target gpu (default). >=0 routes
                // the drafter to that HIP device for hetero compress.
                let pflash_drafter_device: i32 = msg.get("params").and_then(|p| p.get("prefill_drafter_device"))
                    .and_then(|v| v.as_i64()).unwrap_or(-1) as i32;
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

                let state_quant_override = msg.get("params").and_then(|p| p.get("state_quant")).and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty()).map(|s| s.to_string());

                // MTP speculative decode params. Mutually exclusive with DFlash
                // draft — the load handler errors if both are requested.
                let mtp_mode = msg.get("params").and_then(|p| p.get("mtp_mode"))
                    .and_then(|v| v.as_str()).unwrap_or("auto");
                let raw_mtp_head = msg.get("params").and_then(|p| p.get("mtp_head")).and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty());
                let mtp_head_path = if mtp_mode == "off" {
                    if raw_mtp_head.is_some() {
                        eprintln!("[hipfire-daemon] mtp_mode=off — skipping MTP head load ({})", raw_mtp_head.unwrap());
                    }
                    None
                } else {
                    raw_mtp_head.map(|s| s.to_string())
                };
                let mtp_k = msg.get("params").and_then(|p| p.get("mtp_k"))
                    .and_then(|v| v.as_u64()).unwrap_or(3) as usize;
                let mtp_p_min = msg.get("params").and_then(|p| p.get("mtp_p_min"))
                    .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let mtp_kv_mode_str = msg.get("params").and_then(|p| p.get("mtp_kv_mode"))
                    .and_then(|v| v.as_str()).unwrap_or("q8");

                // Mutual exclusion: MTP + DFlash are both spec paths.
                if mtp_head_path.is_some() && draft_path.is_some() {
                    let _ = writeln!(stdout, r#"{{"type":"error","message":"MTP head and DFlash draft are mutually exclusive speculative decode paths. Remove one (set mtp_mode=off or dflash_mode=off) and retry."}}"#);
                    let _ = stdout.flush();
                    continue;
                }
                // MTP + cask/eviction: the MTP verify-then-rollback KV write
                // pattern isn't reconciled with the eviction position
                // accounting (the generate_mtp eviction branch is defensive
                // only). Refuse the combination rather than silently corrupt
                // long-context output.
                if mtp_head_path.is_some() && cask_enabled {
                    let _ = writeln!(stdout, r#"{{"type":"error","message":"MTP head and cask/eviction are not yet compatible (KV rollback vs eviction repositioning). Disable one (set mtp_mode=off or cask=false) and retry."}}"#);
                    let _ = stdout.flush();
                    continue;
                }

                match load_model(path, max_seq, draft_path.as_deref(), kv_mode_override.as_deref(), state_quant_override.as_deref(), &cask, pp, mtp_head_path.as_deref(), mtp_k, mtp_p_min, mtp_kv_mode_str, &mut gpu) {
                    Ok(mut m) => {
                        // glm5 D2: the pre-load mtp+cask guard only catches an
                        // EXPLICIT mtp_head param. A bundled .mq4-mtp (trailer
                        // auto-detected in load_model, mtp_head_path=None) slips
                        // through. Degrade MTP off (free its GPU state) when cask
                        // is active, rather than run the unreconciled combo.
                        if cask_enabled && m.mtp.is_some() {
                            eprintln!("[hipfire-daemon] bundled MTP head detected but cask/eviction is active — disabling MTP (incompatible); using AR/cask");
                            if let Some(mtp) = m.mtp.take() {
                                mtp.spec_state.free_gpu(&mut gpu);
                                mtp.head.free_gpu(&mut gpu);
                            }
                        }
                        let arch = match m.arch_id {
                            5 => "qwen3_5",
                            6 => "qwen3_5_moe",
                            7 => "qwen2",
                            8 => "dots-ocr",
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
                            (c.text.hidden_size, c.text.num_hidden_layers, c.text.vocab_size)
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
                                    // Hetero: when prefill_drafter_device >= 0 and isn't
                                    // device 0 (target), allocate a sibling Gpu handle so
                                    // drafter weights/KV/scratch live on the secondary
                                    // card. Compress output is host-side, so decode stays
                                    // on target. -1 / 0 => share target gpu (unchanged).
                                    let mut sibling: Option<rdna_compute::Gpu> = None;
                                    if pflash_drafter_device > 0 {
                                        match rdna_compute::Gpu::init_with_device(pflash_drafter_device) {
                                            Ok(g) => sibling = Some(g),
                                            Err(e) => {
                                                let _ = writeln!(stdout,
                                                    r#"{{"type":"pflash_load_failed","reason":"drafter device {} init: {}"}}"#,
                                                    pflash_drafter_device, e.to_string().replace('"', "'"));
                                            }
                                        }
                                    }
                                    let dg: &mut rdna_compute::Gpu = sibling.as_mut().unwrap_or(&mut gpu);
                                    dg.bind_thread_or_warn();
                                    match hipfire_arch_qwen35::pflash::load_drafter(
                                        &mut pf_state, dg,
                                        std::path::Path::new(pf_drafter_path),
                                        tok, pf_max_kv,
                                    ) {
                                        Ok(()) => {
                                            eprintln!("[pflash] LOADED drafter={} dev={} mode={} compat={} keep={} thr={}",
                                                pf_drafter_path, pflash_drafter_device, pflash_mode_str,
                                                pf_state.tokenizer_compat, pflash_keep_ratio, pflash_threshold);
                                            let _ = writeln!(stdout,
                                                r#"{{"type":"pflash","mode":"{}","drafter":"{}","drafter_device":{},"tokenizer_compat":{},"keep_ratio":{},"threshold":{}}}"#,
                                                pflash_mode_str, pf_drafter_path, pflash_drafter_device,
                                                pf_state.tokenizer_compat,
                                                pflash_keep_ratio, pflash_threshold);
                                            pflash_state = Some(pf_state);
                                            pflash_cfg = Some(pf_cfg);
                                            pflash_drafter_gpu = sibling; // persist sibling across requests (None if shared)
                                        }
                                        Err(e) => {
                                            eprintln!("[pflash] LOAD FAILED: {}", e);
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
                        // serde-escape: raw HipError debug contains { } and "
                        // which corrupt the JSONL protocol if interpolated raw.
                        write_error(&mut stdout, "", &format!(
                            "load failed: {e}. GPU: {} ({free_mb} MB free / {total_mb} MB total)", gpu.arch));
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
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                id, e.to_string().replace('"', "'"),
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    },
                    None => None,
                };
                let messages_history: Option<Vec<hipfire_runtime::prompt_frame::Message>> = match msg.get("messages") {
                    Some(v) => match serde_json::from_value::<Vec<hipfire_runtime::prompt_frame::Message>>(v.clone()) {
                        Ok(m) => Some(m),
                        Err(e) => {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"error","id":"{}","message":"invalid messages field: {}"}}"#,
                                id, e.to_string().replace('"', "'"),
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    },
                    None => None,
                };
                let temp = msg.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.3) as f32;
                let max_tokens = msg.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
                let top_p = msg.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
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
                let repeat_penalty = msg.get("repeat_penalty").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
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

                // assistant_prefix: "plain", "open_think", or "closed_think"
                // Controls the ChatML framing after the assistant role header.
                // Consumed by the text path; VL path does not yet propagate
                // it (tracked as a follow-up to the post-#169 rebase).
                let assistant_prefix = match msg.get("assistant_prefix")
                    .and_then(|v| v.as_str()).unwrap_or("plain")
                {
                    "open_think" => hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink,
                    "closed_think" => hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink,
                    _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                };

                let has_image = image_base64.is_some() || image.is_some();
                let is_dots_ocr = m.arch_id == 8;
                let has_vl = m.vision_config.is_some() || is_dots_ocr;

                if has_image && !has_vl {
                    write_error(&mut stdout, id, "model has no vision encoder");
                } else if has_image && has_vl {
                    if image_base64.is_some() && image.is_some() {
                        eprintln!("[daemon/vl] both image and image_base64 provided — using image_base64");
                    }
                    let source = if let Some(b64) = image_base64 {
                        if b64.len() > MAX_BASE64_ENCODED_LEN {
                            write_error(&mut stdout, id, &format!(
                                "image payload exceeds maximum encoded size ({} bytes)",
                                MAX_BASE64_ENCODED_LEN,
                            ));
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
                    let vl_max_think_tokens = if max_think_tokens == 0 { 256 } else { max_think_tokens };
                    let params = GenerateVLParams {
                        id, prompt, system_prompt: system, image_source: source,
                        temp, top_p, max_tokens, repeat_penalty, repeat_window,
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
                    let mut gen_ctx = GenerateCtx {
                        stdout: &mut stdout,
                        id,
                        prompt,
                        system_prompt: system,
                        tools: tools_json.as_deref(),
                        messages_history: messages_history.as_deref(),
                        assistant_prefix,
                        temp, top_p, repeat_penalty, repeat_window,
                        max_tokens, max_think_tokens,
                        budget_alert_at_tok,
                        budget_alert_text: &budget_alert_text,
                        drafter_gpu: pflash_drafter_gpu.as_mut(),
                        pflash_state: pflash_state.as_mut(),
                        pflash_cfg: pf_cfg_owned.as_ref(),
                        pflash_bypass_reason: None,
                        pflash_alpha: None,
                    };
                    generate_qwen35(m, &mut gpu, &mut gen_ctx);
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
                    // arch_id=7: rewind the Qwen2State position cursor so
                    // the next prefill writes from KV[0]. Without this, a
                    // reset between turns would leak the prior turn's KV
                    // entries into attention for the new turn — fluent
                    // garbage, no panic. See `Qwen2State::reset` doc.
                    if let Some(ref mut s) = m.qwen2_state { s.reset(); }
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
                    7 => "qwen2",
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
                // Qwen2 (arch_id=7) doesn't have a separate KV buffer — the cache
                // and the per-step scratch share `Qwen2State`. Reset its position
                // cursor here so bench_prefill measures cold prefill.
                if let Some(ref mut s) = m.qwen2_state { s.reset(); }

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

/// Resolve the chat_template to use for a loaded model, walking the
/// override-precedence chain:
///
///   1. `HIPFIRE_CHAT_TEMPLATE_FILE` env var → if set and file readable,
///      that template wins. Operator escape hatch / debugging knob.
///   2. Per-model file at `~/.hipfire/templates/<sanitized-tag>.j2`
///      where the tag is derived from the model file basename
///      (`qwen3.5-9b.mq4` → `qwen3.5-9b.mq4.j2`). User-controllable
///      override for a specific model without env-var globalness.
///   3. HFQ-embedded `tokenizer_config.chat_template`. Default — what
///      the model was trained with.
///   4. None of the above → `None`. The render path falls back to the
///      hand-rolled `ChatFrame::Plain` scaffold (current default
///      behavior under Stage 2's `HIPFIRE_JINJA_CHAT=1` gate). Stage 5+
///      will tighten this to a hard error once Plain is removed.
///
/// Per-request inline override (`chat_template_kwargs.chat_template`,
/// vLLM-style) is the responsibility of the per-request render call,
/// not load-time resolution. It belongs in Stage 5 alongside the CLI
/// passthrough refactor; this resolver only handles the load-time
/// sources.
fn resolve_chat_template(
    hfq: &hipfire_runtime::hfq::HfqFile,
    model_path: &str,
) -> Option<String> {
    // 1. Env-var override.
    if let Ok(env_path) = std::env::var("HIPFIRE_CHAT_TEMPLATE_FILE") {
        if !env_path.is_empty() {
            match std::fs::read_to_string(&env_path) {
                Ok(s) => {
                    eprintln!("[chat_template] using HIPFIRE_CHAT_TEMPLATE_FILE={}", env_path);
                    return Some(s);
                }
                Err(e) => eprintln!(
                    "[chat_template] HIPFIRE_CHAT_TEMPLATE_FILE={env_path} failed to read ({e}); falling through"
                ),
            }
        }
    }

    // 2. Per-model file at ~/.hipfire/templates/<basename>.j2.
    if let Some(home) = std::env::var_os("HOME") {
        let basename = std::path::Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !basename.is_empty() {
            let per_model = std::path::Path::new(&home)
                .join(".hipfire")
                .join("templates")
                .join(format!("{basename}.j2"));
            if per_model.is_file() {
                match std::fs::read_to_string(&per_model) {
                    Ok(s) => {
                        eprintln!("[chat_template] using per-model override {}", per_model.display());
                        return Some(s);
                    }
                    Err(e) => eprintln!(
                        "[chat_template] per-model file {} failed to read ({e}); falling through",
                        per_model.display()
                    ),
                }
            }
        }
    }

    // 3. HFQ-embedded.
    hfq.chat_template()
}

fn parse_state_quant(mode: Option<&str>) -> Result<hipfire_arch_qwen35::qwen35::StateQuant, String> {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!("unsupported DeltaNet state_quant '{other}' (expected q8|fp32|q4)")),
    }
}

fn state_quant_label(q: hipfire_arch_qwen35::qwen35::StateQuant) -> &'static str {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match q {
        StateQuant::FP32 => "FP32",
        StateQuant::Q8 => "Q8",
        StateQuant::Q4 => "Q4",
    }
}

fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
    hfq.tensors()
        .iter()
        .map(|t| {
            t.shape
                .iter()
                .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
        })
        .sum()
}

fn warn_tiny_model_state(hfq: &HfqFile, q: hipfire_arch_qwen35::qwen35::StateQuant) {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    const TINY_MODEL_PARAMS: u128 = 2_000_000_000;
    let params = hfq_parameter_count(hfq);
    if params < TINY_MODEL_PARAMS && q != StateQuant::FP32 {
        eprintln!(
            "  warning: model has ~{:.2}B params; FP32 DeltaNet state is recommended below 2B for long-generation coherence (current: {})",
            params as f64 / 1.0e9,
            state_quant_label(q)
        );
    }
}

fn load_model(path: &str, max_seq: usize, draft_path: Option<&str>, kv_mode_override: Option<&str>, state_quant_override: Option<&str>, cask: &CaskConfig, pp: usize, mtp_head_path: Option<&str>, mtp_k: usize, mtp_p_min: f32, mtp_kv_mode_str: &str, gpu: &mut rdna_compute::Gpu) -> Result<LoadedModel, String> {
    if pp > 1 {
        // Refusal contracts (DFlash, CASK sidecar) are enforced upstream in
        // the "load" event handler so the operator gets a structured error
        // before any HFQ open / weight allocation. By the time we get here
        // with pp>1, draft_path is None and cask.sidecar is None.
        let _ = (draft_path, cask);
        return load_model_pp(
            path, max_seq, kv_mode_override, state_quant_override, pp,
            mtp_head_path, mtp_k, mtp_p_min, mtp_kv_mode_str, gpu,
        );
    }
    // Per-load kv_mode (sent in load message params) overrides the env var.
    // Lets the CLI set size-aware defaults — e.g. Qwen3.5-27B prefers asym4
    // since layer-count compounding of asym3 noise flips argmax at decision
    // boundaries on deep stacks.
    let kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    // ─── ParoQuant / safetensors directory path ────────────────────────────
    // If the path is a directory with config.json, try loading as a
    // SafetensorsSource (ParoQuant, AWQ, etc.) instead of HFQ.
    if Path::new(path).is_dir() {
        return load_model_safetensors(path, max_seq, &kv_mode, gpu);
    }

    let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

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

    if hfq.arch_id == 7 {
        // Qwen2 dense (hipfire-arch-qwen2). Standalone bring-up — no
        // eviction, no DFlash, no PFlash, no VL. The Architecture
        // trait surface gives us config + weights + state in three
        // calls; forward is direct `qwen2::forward_step` below.
        if draft_path.is_some() {
            return Err("DFlash not supported on arch_id=7 (hipfire-arch-qwen2 bring-up). \
                       Reload without a draft.".to_string());
        }
        if cask.sidecar.is_some() {
            return Err("CASK eviction not supported on arch_id=7 (hipfire-arch-qwen2 bring-up). \
                       Reload without --cask-sidecar.".to_string());
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_runtime::arch::Architecture;
        use hipfire_arch_qwen2::Qwen2;
        let config = <Qwen2 as Architecture>::config_from_hfq(&hfq)?;
        let weights = <Qwen2 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config, max_seq)
            .map_err(|e| format!("qwen2: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1, pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            q35_config: None, q35_weights: None, q35_scratch: None,
            kv_cache: None, dn_state: None,
            llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
            qwen2_config: Some(config), qwen2_weights: Some(weights), qwen2_state: Some(state),
            dots_ocr_config: None, dots_ocr_weights: None,
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap: max_seq, eviction: None,
            conversation_tokens: Vec::new(),
            model_path: path.to_string(),
            dflash: None,
            mtp: None,
            chat_template,
        });
    }

    if hfq.arch_id == 8 {
        // dots.ocr (Qwen2-VL family). Text decoder is Qwen2; vision tower
        // is the 42-block DotsVisionTransformer. Both load side-by-side in
        // DotsOcrWeights and stay resident. Single-image, greedy decode at
        // bring-up — no eviction, DFlash, CASK, or PP.
        if draft_path.is_some() {
            return Err("DFlash not supported on arch_id=8 (dots.ocr). Reload without a draft.".to_string());
        }
        if cask.sidecar.is_some() {
            return Err("CASK eviction not supported on arch_id=8 (dots.ocr). Reload without --cask-sidecar.".to_string());
        }
        if pp > 1 {
            return Err("pipeline-parallel (pp>1) not supported on arch_id=8 (dots.ocr).".to_string());
        }
        let _ = kv_mode;
        let _ = state_quant_override;
        use hipfire_runtime::arch::Architecture;
        use hipfire_arch_dots_ocr::DotsOcr;
        let config = <DotsOcr as Architecture>::config_from_hfq(&hfq)?;
        let weights = <DotsOcr as Architecture>::load_weights(&mut hfq, &config, gpu)?;
        // Size the decode KV cache to the requested window (the trait's
        // new_state uses a default max_seq; OCR prompts are long).
        let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config.text, max_seq)
            .map_err(|e| format!("dots-ocr: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        let chat_template = resolve_chat_template(&hfq, path);
        return Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1, pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            q35_config: None, q35_weights: None, q35_scratch: None,
            kv_cache: None, dn_state: None,
            llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
            qwen2_config: None, qwen2_weights: None, qwen2_state: Some(state),
            dots_ocr_config: Some(config), dots_ocr_weights: Some(weights),
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap: max_seq, eviction: None,
            conversation_tokens: Vec::new(),
            model_path: path.to_string(),
            dflash: None,
            mtp: None,
            chat_template,
        });
    }

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
        // Disabled by default on all arches; opt-in via mmq_screen=true or
        // HIPFIRE_MMQ_SCREEN=1. gfx906 is included for the opt-in case so
        // its ~700 µs/weight screening-reference dispatch doesn't surprise
        // first prefill if a user enables it.
        if gpu.mmq_screen && matches!(gpu.arch.as_str(), "gfx906" | "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151" | "gfx1152") {
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
        // Q8 DeltaNet state can accumulate quality drift on long generation.
        // The load-time override exists for coherence A/B probes.
        let dn_quant = parse_state_quant(state_quant_override)?;
        eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
        warn_tiny_model_state(&hfq, dn_quant);
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
            let centers = TriAttnCenters::load(Path::new(sidecar_path)).map_err(|e| {
                use std::io::ErrorKind;
                let p = Path::new(sidecar_path);
                let why = match e.kind() {
                    // os error 2: open failed. Disambiguate missing vs dangling symlink.
                    ErrorKind::NotFound if p.symlink_metadata().is_ok() =>
                        format!("dangling symlink (target absent): {sidecar_path}"),
                    ErrorKind::NotFound => format!("file not found: {sidecar_path}"),
                    ErrorKind::InvalidData => format!("bad format ({e}): {sidecar_path}"),
                    ErrorKind::UnexpectedEof => format!("truncated/corrupt sidecar: {sidecar_path}"),
                    _ => format!("read error ({e}): {sidecar_path}"),
                };
                format!("cask sidecar load failed — {why} (regen: hipfire sidecar-gen, or HIPFIRE_CASK_OFF=1)")
            })?;
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

        let chat_template = resolve_chat_template(&hfq, path);

        // Build LoadedModel without MTP first, then optionally load the MTP
        // head by temporarily constructing a ModelSlot from the model's fields.
        let mut loaded = LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1, pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            q35_config: Some(config), q35_weights: Some(weights), q35_scratch: Some(scratch),
            kv_cache: Some(kv), dn_state: Some(dn),
            llama_config: None, llama_weights: None, llama_scratch: None, llama_kv: None,
            qwen2_config: None, qwen2_weights: None, qwen2_state: None,
            dots_ocr_config: None, dots_ocr_weights: None,
            vision_config, vision_weights,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap, eviction,
            conversation_tokens: Vec::new(),
            model_path: path.to_string(),
            dflash,
            mtp: None,
            chat_template,
        };

        // Optional MTP head: load from sidecar or bundled trailer.
        // Only for Qwen3.5/3.6 (arch_id 5/6). Compat guard checks
        // n_embd/vocab_size/rope_theta against the trunk config.
        // On mismatch or load failure, degrade to AR with a warning.
        if (loaded.arch_id == 5 || loaded.arch_id == 6) && loaded.dflash.is_none() {
            let mtp_result: Result<Option<MtpState>, String> = (|| {
                let head: Qwen35MtpHead = if let Some(mp) = mtp_head_path {
                    mtp_head::load_mtp_head(Path::new(mp), gpu, physical_cap)
                        .map_err(|e| format!("load mtp sidecar '{}': {e}", mp))?
                } else {
                    match mtp_head::load_mtp_head_bundled(Path::new(path), gpu, physical_cap) {
                        Ok(Some(h)) => h,
                        Ok(None) => return Ok(None),
                        Err(e) => return Err(format!("load bundled mtp: {e}")),
                    }
                };

                let trunk_config = loaded.q35_config.as_ref().unwrap();

                if head.config.n_embd != trunk_config.dim {
                    return Err(format!(
                        "MTP head n_embd={} != trunk dim={}", head.config.n_embd, trunk_config.dim,
                    ));
                }
                if head.config.vocab_size != trunk_config.vocab_size {
                    return Err(format!(
                        "MTP head vocab_size={} != trunk vocab={}", head.config.vocab_size, trunk_config.vocab_size,
                    ));
                }
                let trunk_rope_theta = trunk_config.rope_theta as f32;
                // Relative tolerance: rope_theta can be 1e7+ where f32 ulp
                // alone exceeds an absolute 1.0, falsely rejecting a matching
                // head.
                if (head.config.rope_theta - trunk_rope_theta).abs()
                    > trunk_rope_theta.abs() * 1e-3 + 1.0 {
                    return Err(format!(
                        "MTP head rope_theta={} != trunk rope_theta={}", head.config.rope_theta, trunk_rope_theta,
                    ));
                }

                let compressed = head.weights.lm_head_draft.is_some();
                let kv_mode = MtpKvMode::parse(mtp_kv_mode_str)
                    .map_err(|e| format!("{e}"))?;

                // Temporarily build a ModelSlot to satisfy
                // new_for_slot_with_kv_mode's signature. It only reads
                // config/dn_state for sizing — the slot is discarded after
                // allocation.
                let slot_config = speculative::ModelSlotConfig::default();
                let tmp_hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
                let tmp_weights = loaded.q35_weights.take().expect("q35 weights");
                let tmp_kv = loaded.kv_cache.take().expect("kv cache");
                let tmp_dn = loaded.dn_state.take().expect("dn state");
                let tmp_scratch = loaded.q35_scratch.take().expect("q35 scratch");
                let mut target = speculative::ModelSlot {
                    name: String::from("target"),
                    hfq: tmp_hfq,
                    config: trunk_config.clone(),
                    weights: tmp_weights,
                    kv_cache: tmp_kv,
                    dn_state: tmp_dn,
                    scratch: tmp_scratch,
                    slot_config,
                };

                let spec_state = MtpSpecState::new_for_slot_with_kv_mode(
                    gpu, &target, &head, mtp_k, kv_mode,
                ).map_err(|e| format!("alloc MtpSpecState: {e}"))?;

                // Put model fields back.
                loaded.q35_weights = Some(target.weights);
                loaded.kv_cache = Some(target.kv_cache);
                loaded.dn_state = Some(target.dn_state);
                loaded.q35_scratch = Some(target.scratch);

                let mut spec_state = spec_state;
                if mtp_p_min > 0.0 {
                    spec_state.set_p_min(mtp_p_min);
                }

                // Compressed (cvs sidecar) heads need the compressed-logits
                // scratch allocated before the first spec_step, or it panics
                // ("logits_compressed not allocated"). Mirrors mtp_only_demo.
                // Bundled full-vocab heads (compressed_vocab_size=None) skip
                // this — spec_step_mtp_compressed_serial uses the trunk lm_head.
                if let Some(cvs) = head.weights.compressed_vocab_size {
                    spec_state.mtp_scratch.ensure_compressed_logits(gpu, cvs)
                        .map_err(|e| format!("alloc logits_compressed: {e}"))?;
                    spec_state.ensure_compressed_lm_logits(gpu, cvs)
                        .map_err(|e| format!("alloc mtp_lm_logits_compressed: {e}"))?;
                }

                eprintln!(
                    "  MTP head loaded: {} (n_embd={}, vocab={}, compressed={}, k={}, p_min={:.2})",
                    mtp_head_path.unwrap_or("(bundled)"),
                    head.config.n_embd, head.config.vocab_size,
                    compressed, mtp_k, mtp_p_min,
                );

                Ok(Some(MtpState {
                    head,
                    spec_state,
                    max_n: mtp_k,
                    p_min: mtp_p_min,
                    compressed,
                    drafter_state: None,
                }))
            })();

            match mtp_result {
                Ok(Some(state)) => loaded.mtp = Some(state),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("  MTP head load failed ({}) — falling back to AR/DFlash only", e);
                }
            }
        }

        Ok(loaded)
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
        let chat_template = resolve_chat_template(&hfq, path);
        Ok(LoadedModel {
            arch_id: hfq.arch_id,
            pp: 1, pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            q35_config: None, q35_weights: None, q35_scratch: None,
            kv_cache: None, dn_state: None,
            llama_config: Some(config), llama_weights: Some(weights), llama_scratch: Some(scratch), llama_kv: Some(kv),
            qwen2_config: None, qwen2_weights: None, qwen2_state: None,
            dots_ocr_config: None, dots_ocr_weights: None,
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap: max_seq, eviction: None,
            conversation_tokens: Vec::new(),
            model_path: path.to_string(),
            dflash: None,
            mtp: None,
            chat_template,
        })
    }
}

/// Load a model from a HuggingFace safetensors directory (ParoQuant, AWQ, etc.).
fn load_model_safetensors(
    path: &str, max_seq: usize, kv_mode: &str, gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::safetensors_source::SafetensorsSource;
    use hipfire_runtime::model_source::ModelSource;

    eprintln!("  opening safetensors directory: {path}");
    let source = SafetensorsSource::open(Path::new(path))
        .map_err(|e| format!("safetensors open: {e}"))?;

    let arch_id = source.arch_id();
    let qm = source.quant_config().map(|q| q.method.as_str()).unwrap_or("none");
    eprintln!("  arch_id={arch_id}, quant_method={qm}");

    // Tokenizer from tokenizer.json
    let tokenizer = if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
            .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))?
    } else {
        return Err("no tokenizer.json found in model directory".into());
    };

    // HF safetensors use half-split RoPE convention (rotate_half)
    // — upstream now defaults to halfsplit, no flag needed
    let chat_template = source.chat_template();

    if arch_id == 0 || arch_id == 1 {
        // LLaMA / Qwen3 — standard attention, no DeltaNet
        let config = hipfire_runtime::hfq::config_from_safetensors_llama(&source)
            .ok_or("failed to parse LLaMA/Qwen3 config from config.json")?;

        eprintln!("  LLaMA/Qwen3: dim={}, layers={}, heads={}, kv_heads={}, head_dim={}, qk_norm={}",
            config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.head_dim, config.has_qk_norm);

        let weights = hipfire_runtime::hfq::load_weights_paroquant_llama(&source, &config, gpu)
            .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;

        // asym3 K-cache asserts head_dim==256 (Qwen 3.5/3.6 family). Qwen3
        // dense checkpoints (e.g. shisa-Qwen3-0.6B-PARO, head_dim=128) need
        // q8 for auto/default selection; explicit "asym3" still routes to
        // the panicking constructor so caller-misconfigured runs surface.
        let asym3_auto = matches!(kv_mode, "turbo3" | "turbo" | "auto" | "");
        let kv = match kv_mode {
            "q8" => llama::KvCache::new_gpu_q8_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
            "asym4" | "turbo4" => llama::KvCache::new_gpu_asym4_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
            "asym3" => llama::KvCache::new_gpu_asym3_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
            _ if asym3_auto && config.head_dim == 256 => llama::KvCache::new_gpu_asym3_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
            _ => llama::KvCache::new_gpu_q8_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
        }.map_err(|e| format!("KvCache: {e}"))?;

        let scratch = llama::ForwardScratch::new(gpu, &config)
            .map_err(|e| format!("ForwardScratch::new: {e:?}"))?;

        return Ok(LoadedModel {
            arch_id,
            pp: 1,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            q35_config: None,
            q35_weights: None,
            q35_scratch: None,
            qwen2_config: None,
            qwen2_weights: None,
            qwen2_state: None,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            kv_cache: None,
            dn_state: None,
            llama_config: Some(config),
            llama_weights: Some(weights),
            llama_scratch: Some(scratch),
            llama_kv: Some(kv),
            vision_config: None,
            vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0,
            max_seq,
            physical_cap: max_seq,
            eviction: None,
            conversation_tokens: Vec::new(),
            model_path: path.to_string(),
            dflash: None,
            mtp: None,
            chat_template,
        });
    }

    if arch_id != 5 && arch_id != 6 {
        return Err(format!("safetensors loading only supports LLaMA/Qwen3 (arch_id 0/1) and Qwen3.5/3.6 (arch_id 5/6), got {arch_id}"));
    }

    // Parse config (reuse Qwen35's config parser via metadata_json)
    let config = qwen35::config_from_safetensors(&source)
        .ok_or("failed to parse Qwen3.5 config from config.json")?;

    eprintln!("  Qwen3.5/3.6: dim={}, layers={}, heads={}", config.dim, config.n_layers, config.n_heads);

    // Load weights via ParoQuant path
    let weights = qwen35::load_weights_paroquant(&source, &config, gpu)
        .map_err(|e| format!("load_weights_paroquant: {e:?}"))?;

    // KV cache: default to asym3 (matches the main Qwen35 path)
    let effective_max_seq = max_seq;
    let kv_cache = match kv_mode {
        "q8" => llama::KvCache::new_gpu_q8_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
        "asym4" | "turbo4" => llama::KvCache::new_gpu_asym4_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
        _ => llama::KvCache::new_gpu_asym3_capped(gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq),
    }.map_err(|e| format!("KvCache: {e}"))?;
    let dn_state = DeltaNetState::new(gpu, &config)
        .map_err(|e| format!("DeltaNetState::new: {e:?}"))?;
    let scratch = qwen35::Qwen35Scratch::new(gpu, &config, 256)
        .map_err(|e| format!("Qwen35Scratch::new: {e:?}"))?;

    Ok(LoadedModel {
        arch_id,
        pp: 1,
        pp_gpus: None,
        pp_scratch_set: None,
        pp_dn_la_to_device: None,
        q35_config: Some(config),
        q35_weights: Some(weights),
        q35_scratch: Some(scratch),
        qwen2_config: None,
        qwen2_weights: None,
        qwen2_state: None,
        dots_ocr_config: None,
        dots_ocr_weights: None,
        kv_cache: Some(kv_cache),
        dn_state: Some(dn_state),
        llama_config: None,
        llama_weights: None,
        llama_scratch: None,
        llama_kv: None,
        vision_config: None,
        vision_weights: None,
        tokenizer: Some(tokenizer),
        seq_pos: 0,
        max_seq: effective_max_seq,
        physical_cap: effective_max_seq,
        eviction: None,
        conversation_tokens: Vec::new(),
        model_path: path.to_string(),
        dflash: None,
        mtp: None,
        chat_template,
    })
}

/// Multi-GPU pipeline-parallel load path (Stage 7 of #58). Refuses VL,
/// non-Qwen3.5 architectures and (transitively, via the upstream "load"
/// handler) DFlash, CASK and PFlash. Returns a `LoadedModel` with `pp_gpus`,
/// `pp_scratch_set` and `pp_dn_la_to_device` populated; the daemon's primary
/// `gpu` parameter is unused on this path. Eviction is refused at this layer
/// because TriAttention/CASK/PFlash live on a single device and are not v1
/// targets for pp>1 — physical_cap == max_seq accordingly.
#[allow(clippy::too_many_arguments)]
fn load_model_pp(
    path: &str,
    max_seq: usize,
    kv_mode_override: Option<&str>,
    state_quant_override: Option<&str>,
    pp: usize,
    mtp_head_path: Option<&str>,
    mtp_k: usize,
    mtp_p_min: f32,
    mtp_kv_mode_str: &str,
    _gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    let kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

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

    // HIPFIRE_KV_FILTER=1 enables the *_filtered_multi KV constructors
    // (Stage 1 of the pp+mtp plan). For Qwen 3.5/3.6 hybrid models with
    // 48 LinearAttention + 16 FullAttention layers, this allocates a
    // 1-element placeholder for each LA layer's K/V slot instead of a
    // full slab, saving ~3× KV VRAM. Default off until validated;
    // turn on for the longer-ctx PP runs that motivate Stage 2.
    let kv_filter_on = std::env::var("HIPFIRE_KV_FILTER").ok().as_deref() == Some("1");
    let is_kv_layer: Vec<bool> = config.layer_types.iter()
        .map(|t| matches!(t, hipfire_arch_qwen35::qwen35::LayerType::FullAttention))
        .collect();
    if kv_filter_on {
        let n_kv = is_kv_layer.iter().filter(|b| **b).count();
        eprintln!("  HIPFIRE_KV_FILTER=1 → using filtered_multi KV ({n_kv}/{} layers carry KV)", is_kv_layer.len());
    }

    // KV cache (asym3 default, q8/asym4/asym2/fwht{4,3,2} selectable).
    // physical_cap == max_seq on this path — eviction is refused at load.
    let kv = match kv_mode.as_str() {
        "q8" => if kv_filter_on {
            llama::KvCache::new_gpu_q8_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        } else {
            llama::KvCache::new_gpu_q8_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        },
        "asym4" | "turbo4" => if kv_filter_on {
            llama::KvCache::new_gpu_asym4_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        } else {
            llama::KvCache::new_gpu_asym4_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        },
        "asym2" | "turbo2" => if kv_filter_on {
            llama::KvCache::new_gpu_asym2_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        } else {
            llama::KvCache::new_gpu_asym2_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        },
        "asym3" | "turbo3" | "turbo" | "auto" | "" => if kv_filter_on {
            llama::KvCache::new_gpu_asym3_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        } else {
            llama::KvCache::new_gpu_asym3_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        },
        "fwht4" => if kv_filter_on {
            llama::KvCache::new_gpu_fwht4_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        } else {
            llama::KvCache::new_gpu_fwht4_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        },
        "fwht3" => if kv_filter_on {
            llama::KvCache::new_gpu_fwht3_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        } else {
            llama::KvCache::new_gpu_fwht3_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        },
        "fwht2" => if kv_filter_on {
            llama::KvCache::new_gpu_fwht2_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        } else {
            llama::KvCache::new_gpu_fwht2_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
        },
        other => {
            eprintln!("  KV cache: unrecognized '{other}', defaulting to asym3");
            if kv_filter_on {
                llama::KvCache::new_gpu_asym3_capped_filtered_multi(&mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
            } else {
                llama::KvCache::new_gpu_asym3_capped_multi(&mut gpus, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, max_seq).map_err(|e| format!("{e}"))?
            }
        }
    };

    // Mirror the pp=1 state-mode parser so pp parity probes can force the
    // same DeltaNet state representation.
    let dn_quant = parse_state_quant(state_quant_override)?;
    eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
    warn_tiny_model_state(&hfq, dn_quant);
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

    // Per-device VRAM-after-load report. Useful for validating Stage 1
    // (HIPFIRE_KV_FILTER) savings + Stage 2 (PP+MTP layout) budget planning.
    for (dev_idx, g) in gpus.devices.iter().enumerate() {
        if g.bind_thread().is_ok() {
            if let Ok((free, total)) = g.hip.get_vram_info() {
                let used = total.saturating_sub(free);
                eprintln!(
                    "  dev {dev_idx} ({}): {:.2} GiB used / {:.2} GiB total ({:.2} GiB free)",
                    g.arch,
                    used as f64 / (1u64 << 30) as f64,
                    total as f64 / (1u64 << 30) as f64,
                    free as f64 / (1u64 << 30) as f64,
                );
            }
        }
    }

    let mut loaded = LoadedModel {
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
        qwen2_config: None, qwen2_weights: None, qwen2_state: None,
        dots_ocr_config: None, dots_ocr_weights: None,
        vision_config: None, vision_weights: None,
        tokenizer: Some(tokenizer),
        seq_pos: 0, max_seq, physical_cap: max_seq, eviction: None,
        conversation_tokens: Vec::new(),
        model_path: path.to_string(),
        dflash: None,
        mtp: None,
        chat_template: resolve_chat_template(&hfq, path),
    };

    // ─── Stage 2: PP + MTP-on-output_device ─────────────────────────────
    //
    // If the load supplied an MTP head (--mtp-head / bundled trailer), load
    // it onto gpus.devices[gpus.output_device]. The output_device convention
    // (verify lm_head + output_norm live there) makes that gpu the natural
    // home for the MTP head — verify_hidden is allocated there, so the
    // per-cycle prev_hidden handoff collapses from a cross-device peer copy
    // to a same-device memcpy_dtod_at.
    //
    // Also mirrors trunk.weights.token_embd from dev 0 onto output_device
    // (peer_clone_tensor with same-device shortcut when output_device == 0).
    //
    // Only compressed-sidecar heads are supported in v1 hetero — the
    // bundled .mq4-mtp path needs the trunk.output mirror too which we
    // haven't built (see docs/plans/mtp_multi_gpu_split_audit.md).
    let mut mtp_load_block = || -> Result<Option<MtpState>, String> {
        // Bail early when no head was requested AND no bundled trailer
        // exists on the trunk file.
        let want_bundled = mtp_head_path.is_none()
            && hipfire_arch_qwen35::mtp_head::detect_bundled_mtp_offset(Path::new(path))
                .ok().flatten().is_some();
        if mtp_head_path.is_none() && !want_bundled {
            return Ok(None);
        }

        // Borrow gpus mutably; we'll restore loaded.pp_gpus after the load.
        let mut gpus = loaded.pp_gpus.take().expect("pp_gpus populated above");
        let output_device = gpus.output_device;
        let trunk_weights = loaded.q35_weights.take().expect("q35 weights populated");
        let trunk_config = loaded.q35_config.as_ref().expect("q35 config populated").clone();

        // Load MTP head onto output_device. The head's KV cache size is
        // physical_cap (= max_seq on this path); fits comfortably given
        // the audit's per-device budget.
        let head_res = {
            let dev = &mut gpus.devices[output_device];
            if let Some(mp) = mtp_head_path {
                hipfire_arch_qwen35::mtp_head::load_mtp_head(Path::new(mp), dev, max_seq)
                    .map_err(|e| format!("load mtp sidecar '{}': {e}", mp))
            } else {
                hipfire_arch_qwen35::mtp_head::load_mtp_head_bundled(Path::new(path), dev, max_seq)
                    .map_err(|e| format!("load bundled mtp: {e}"))?
                    .ok_or_else(|| "no bundled mtp trailer".to_string())
            }
        };
        let head: Qwen35MtpHead = match head_res {
            Ok(h) => h,
            Err(e) => {
                // Restore state so the caller still gets a working pp model.
                loaded.pp_gpus = Some(gpus);
                loaded.q35_weights = Some(trunk_weights);
                return Err(e);
            }
        };

        // Config-compat checks (mirror pp=1 path).
        if head.config.n_embd != trunk_config.dim {
            let e = format!(
                "MTP head n_embd={} != trunk dim={}",
                head.config.n_embd, trunk_config.dim,
            );
            head.free_gpu(&mut gpus.devices[output_device]);
            loaded.pp_gpus = Some(gpus);
            loaded.q35_weights = Some(trunk_weights);
            return Err(e);
        }
        if head.config.vocab_size != trunk_config.vocab_size {
            let e = format!(
                "MTP head vocab_size={} != trunk vocab={}",
                head.config.vocab_size, trunk_config.vocab_size,
            );
            head.free_gpu(&mut gpus.devices[output_device]);
            loaded.pp_gpus = Some(gpus);
            loaded.q35_weights = Some(trunk_weights);
            return Err(e);
        }
        let trunk_rope_theta = trunk_config.rope_theta as f32;
        if (head.config.rope_theta - trunk_rope_theta).abs()
            > trunk_rope_theta.abs() * 1e-3 + 1.0 {
            let e = format!(
                "MTP head rope_theta={} != trunk rope_theta={}",
                head.config.rope_theta, trunk_rope_theta,
            );
            head.free_gpu(&mut gpus.devices[output_device]);
            loaded.pp_gpus = Some(gpus);
            loaded.q35_weights = Some(trunk_weights);
            return Err(e);
        }

        // Stage 2 v1 requires the compressed-sidecar path (cvs head). The
        // bundled full-vocab path would need a trunk.output mirror too,
        // which we haven't built. Refuse early with a clear error.
        let cvs = match head.weights.compressed_vocab_size {
            Some(c) => c,
            None => {
                let e = "Stage 2 PP+MTP requires a compressed-sidecar .mtp \
                    head (e.g. qwen3.6-27b-cvs16384.mtp). Bundled full-vocab \
                    heads need the trunk.output mirror, not implemented in v1.".to_string();
                head.free_gpu(&mut gpus.devices[output_device]);
                loaded.pp_gpus = Some(gpus);
                loaded.q35_weights = Some(trunk_weights);
                return Err(e);
            }
        };
        let compressed = true;

        // Mirror trunk.token_embd from dev 0 (per load_weights_multi's
        // Variant 2 convention) onto output_device. peer_clone_tensor has
        // a same-device shortcut, so if output_device == 0 this is a D2D
        // memcpy. The token_embd ownership stays on dev 0 — the mirror
        // is read by the chain's embed_lookup on output_device.
        //
        // Borrow gymnastics: trunk_weights is taken; we put it back AFTER
        // the mirror's source borrow ends.
        // token_embd lives on dev 0 (Variant 2 load convention); mirror
        // it onto output_device. When output_device == 0 (single-band
        // PP or pp=1 hetero), src and dst are the same gpu — use
        // clone_tensor_same with a single &mut. When output_device != 0,
        // use split_pair_mut to get the (&Gpu, &mut Gpu) pair without
        // aliasing. This replaces the Stage 2a `unsafe { &mut *dev0_ptr }`
        // workaround per Step −1 of docs/plans/mtp_multi_refactor.md.
        let mirror_result = if output_device == 0 {
            let g = gpus.single_mut(0);
            hipfire_runtime::mtp_mirror::clone_tensor_same(g, &trunk_weights.token_embd)
                .map_err(|e| format!("mirror token_embd: {e}"))
        } else {
            let (src_gpu, dst_gpu) = gpus.split_pair_mut(0, output_device);
            hipfire_runtime::mtp_mirror::clone_tensor_peer(src_gpu, dst_gpu, &trunk_weights.token_embd)
                .map_err(|e| format!("mirror token_embd: {e}"))
        };

        let mirrored_token_embd = match mirror_result {
            Ok(t) => t,
            Err(e) => {
                head.free_gpu(&mut gpus.devices[output_device]);
                loaded.pp_gpus = Some(gpus);
                loaded.q35_weights = Some(trunk_weights);
                return Err(e);
            }
        };
        let embd_format = trunk_weights.embd_format;

        // Allocate trunk-side MtpSpecState on output_device (where verify_hidden
        // gets written by the trunk verify) — that's also where prev_hidden
        // should logically live for the spec function's bookkeeping, though
        // the actual prev_hidden the chain reads is drafter_state.prev_hidden
        // (which is the same device in this layout — so the spec function's
        // same-device shortcut kicks in).
        //
        // Need a transient ModelSlot to satisfy MtpSpecState::new_for_slot_with_kv_mode.
        let kv_mode_mtp = match MtpKvMode::parse(mtp_kv_mode_str) {
            Ok(m) => m,
            Err(e) => {
                head.free_gpu(&mut gpus.devices[output_device]);
                gpus.devices[output_device].free_tensor(mirrored_token_embd).ok();
                loaded.pp_gpus = Some(gpus);
                loaded.q35_weights = Some(trunk_weights);
                return Err(format!("mtp kv mode parse: {e}"));
            }
        };

        // The trunk-side MtpSpecState needs a ModelSlot — but our trunk
        // weights live multi-gpu. Build the spec state directly with the
        // output_device gpu and the trunk_config (the constructor reads
        // config + dn_state for sizing, not weights). We DO need dn_state;
        // take it from loaded.
        let tmp_dn = loaded.dn_state.take().expect("dn_state populated");
        let tmp_kv = loaded.kv_cache.take().expect("kv_cache populated");
        let tmp_scratch = loaded.q35_scratch.take(); // None on pp path
        let _ = tmp_scratch;

        // ModelSlot wants its OWN single-gpu scratch + weights. Since on the
        // pp path the trunk is already multi-gpu, we build a TRANSIENT
        // ModelSlot only for the MtpSpecState constructor's signature, then
        // immediately destructure it to reclaim the components. The actual
        // forward path uses gpus directly, not the slot.
        let slot_config = speculative::ModelSlotConfig::default();
        let tmp_hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
        // Build a transient single-device Qwen35Scratch on output_device.
        // The MtpSpecState constructor only needs it for shape/size
        // inspection; not used at runtime in the hetero spec function.
        let tmp_q35_scratch_for_slot = qwen35::Qwen35Scratch::new_with_kv_max(
            &mut gpus.devices[output_device], &trunk_config, 128, max_seq,
        ).map_err(|e| format!("alloc transient q35 scratch: {e}"))?;

        let mut target = speculative::ModelSlot {
            name: String::from("target"),
            hfq: tmp_hfq,
            config: trunk_config.clone(),
            weights: trunk_weights,
            kv_cache: tmp_kv,
            dn_state: tmp_dn,
            scratch: tmp_q35_scratch_for_slot,
            slot_config,
        };

        let spec_state_res = MtpSpecState::new_for_slot_with_kv_mode(
            &mut gpus.devices[output_device], &target, &head, mtp_k, kv_mode_mtp,
        );

        let mut spec_state = match spec_state_res {
            Ok(s) => s,
            Err(e) => {
                head.free_gpu(&mut gpus.devices[output_device]);
                gpus.devices[output_device].free_tensor(mirrored_token_embd).ok();
                // Restore loaded state (and free transient scratch).
                target.scratch.free_gpu(&mut gpus.devices[output_device]);
                loaded.q35_weights = Some(target.weights);
                loaded.kv_cache = Some(target.kv_cache);
                loaded.dn_state = Some(target.dn_state);
                loaded.pp_gpus = Some(gpus);
                return Err(format!("alloc MtpSpecState: {e}"));
            }
        };
        if mtp_p_min > 0.0 {
            spec_state.set_p_min(mtp_p_min);
        }
        // Compressed scratch on the trunk-side MtpSpecState — needed by the
        // single-gpu spec path, but the hetero path uses drafter_state's
        // own mtp_scratch.logits_compressed instead. Alloc both for safety
        // (cheap; ~64 KB).
        spec_state.mtp_scratch.ensure_compressed_logits(&mut gpus.devices[output_device], cvs)
            .map_err(|e| format!("alloc trunk-side logits_compressed: {e}"))?;
        spec_state.ensure_compressed_lm_logits(&mut gpus.devices[output_device], cvs)
            .map_err(|e| format!("alloc trunk-side mtp_lm_logits_compressed: {e}"))?;

        // Drafter-side state on output_device (= same device as spec state).
        // mtp_scratch.ensure_compressed_logits is called by new_for_slot's
        // caller convention; do it explicitly.
        let drafter_state_res = hipfire_arch_qwen35::mtp_spec::MtpHeteroDrafterState::new_for_slot(
            &mut gpus.devices[output_device], &head, mtp_k, kv_mode_mtp,
            mirrored_token_embd, embd_format,
        );
        let mut drafter_state = match drafter_state_res {
            Ok(d) => d,
            Err(e) => {
                spec_state.free_gpu(&mut gpus.devices[output_device]);
                head.free_gpu(&mut gpus.devices[output_device]);
                target.scratch.free_gpu(&mut gpus.devices[output_device]);
                loaded.q35_weights = Some(target.weights);
                loaded.kv_cache = Some(target.kv_cache);
                loaded.dn_state = Some(target.dn_state);
                loaded.pp_gpus = Some(gpus);
                return Err(format!("alloc MtpHeteroDrafterState: {e}"));
            }
        };
        drafter_state.mtp_scratch.ensure_compressed_logits(
            &mut gpus.devices[output_device], cvs,
        ).map_err(|e| format!("alloc drafter logits_compressed: {e}"))?;

        // Restore borrowed pieces.
        target.scratch.free_gpu(&mut gpus.devices[output_device]);
        loaded.q35_weights = Some(target.weights);
        loaded.kv_cache = Some(target.kv_cache);
        loaded.dn_state = Some(target.dn_state);
        loaded.pp_gpus = Some(gpus);

        eprintln!(
            "  MTP head loaded on output_device={} (n_embd={}, vocab={}, cvs={}, k={}, p_min={:.2})",
            output_device,
            head.config.n_embd, head.config.vocab_size,
            cvs, mtp_k, mtp_p_min,
        );

        Ok(Some(MtpState {
            head,
            spec_state,
            max_n: mtp_k,
            p_min: mtp_p_min,
            compressed,
            drafter_state: Some(drafter_state),
        }))
    };

    match mtp_load_block() {
        Ok(Some(state)) => loaded.mtp = Some(state),
        Ok(None) => {}
        Err(e) => {
            eprintln!("  MTP head load failed ({}) — falling back to AR-only on PP", e);
        }
    }

    Ok(loaded)
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
    // MTP state: head weights + spec state (verify hidden/logits, KV cache,
    // DN snapshot, scratch buffers).
    if let Some(mtp) = m.mtp {
        mtp.spec_state.free_gpu(gpu);
        mtp.head.free_gpu(gpu);
    }
    // Free eviction context (centers + scratch tensors) if active.
    if let Some(ev) = m.eviction { ev.free_gpu(gpu); }
    // Free KV cache + DeltaNet state + scratch first (small fraction of VRAM).
    if let Some(kv) = m.kv_cache { kv.free_gpu(gpu); }
    if let Some(dn) = m.dn_state { dn.free_gpu(gpu); }
    if let Some(s) = m.q35_scratch { s.free_gpu(gpu); }
    if let Some(kv) = m.llama_kv { kv.free_gpu(gpu); }
    if let Some(s) = m.llama_scratch { s.free_gpu(gpu); }
    // Qwen2 state holds both the per-step scratch AND the KV cache — one
    // free_gpu call handles both. (Compare LLaMA where ForwardScratch and
    // KvCache are separate fields.)
    if let Some(s) = m.qwen2_state { s.free_gpu(gpu); }
    // Weights are the bulk of VRAM (~80%). Free them too so idle eviction
    // actually returns VRAM to the system, not just the cache.
    if let Some(w) = m.q35_weights { w.free_gpu(gpu); }
    if let Some(w) = m.llama_weights { w.free_gpu(gpu); }
    if let Some(w) = m.qwen2_weights { w.free_gpu(gpu); }
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
    ctx: &mut GenerateCtx<'_>,
) {
    debug_assert_eq!(pick_path(m, ctx), SpecPath::Dflash,
        "generate_dflash called with path={:?}", pick_path(m, ctx));
    let stdout: &mut std::io::Stdout = &mut *ctx.stdout;
    let id: &str = ctx.id;
    let prompt: &str = ctx.prompt;
    let system_prompt: Option<&str> = ctx.system_prompt;
    let max_tokens: usize = ctx.max_tokens;
    let max_think_tokens: usize = ctx.max_think_tokens;
    let assistant_prefix = ctx.assistant_prefix;
    let pflash_bypass_reason: Option<&str> = ctx.pflash_bypass_reason;
    let pflash_alpha: Option<f32> = ctx.pflash_alpha;
    let tools: Option<&[serde_json::Value]> = ctx.tools;
    let messages_history: Option<&[hipfire_runtime::prompt_frame::Message]> = ctx.messages_history;

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
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking: max_think_tokens != 1,
            bos_token: None,
        };
        let render_result = if tools.is_some() || messages_history.is_some() {
            let synthesized: Vec<hipfire_runtime::prompt_frame::Message>;
            let messages_slice: &[hipfire_runtime::prompt_frame::Message] = match messages_history {
                Some(m) => m,
                None => {
                    let mut v = Vec::new();
                    if let Some(sys) = system_prompt {
                        v.push(hipfire_runtime::prompt_frame::Message {
                            role: hipfire_runtime::prompt_frame::Role::System,
                            content: sys.to_string(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
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
                eprintln!("[daemon] jinja render failed in dflash path ({e}) — falling back to Plain");
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
                    raw: false,
                }
                .build()
            }
        }
    } else {
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer,
            system: system_prompt,
            user: prompt,
            assistant_prefix,
            raw: false,
        }
        .build()
    };

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
                let (in_think, new_count) = detect_think_state(&raw_so_far, prev_in_think, think_count);
                think_count = new_count;
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
    let path_extras = format!(
        r#","dflash":true,"tau":{:.2},"cycles":{}{}"#,
        tau, stats.cycles, pflash_done_field,
    );
    emit_done_event(stdout, id, &DoneStats {
        tokens: generated,
        tok_s,
        prefill_tokens: prompt_tokens.len(),
        prefill_ms: prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms: prefill_s * 1000.0,
    }, &path_extras);
}

#[allow(clippy::too_many_arguments)]
fn generate_mtp(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    ctx: &mut GenerateCtx<'_>,
) {
    debug_assert_eq!(pick_path(m, ctx), SpecPath::Mtp,
        "generate_mtp called with path={:?}", pick_path(m, ctx));
    use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};

    // Local aliases so the body stays readable without `ctx.` prefix
    // everywhere. Sampling fields are Copy; refs are reborrowed.
    let stdout: &mut std::io::Stdout = &mut *ctx.stdout;
    let id: &str = ctx.id;
    let prompt: &str = ctx.prompt;
    let system_prompt: Option<&str> = ctx.system_prompt;
    let max_tokens: usize = ctx.max_tokens;
    let max_think_tokens: usize = ctx.max_think_tokens;
    let assistant_prefix = ctx.assistant_prefix;
    let temp: f32 = ctx.temp;
    let top_p: f32 = ctx.top_p;
    let repeat_penalty: f32 = ctx.repeat_penalty;
    let _repeat_window: usize = ctx.repeat_window; // unused in mtp path; MTP residual-sample uses top_k=20 hardcoded
    let tools: Option<&[serde_json::Value]> = ctx.tools;
    let messages_history: Option<&[hipfire_runtime::prompt_frame::Message]> = ctx.messages_history;

    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Multi-turn: prefill only the NEW turn's tokens on top of the cumulative
    // KV + DeltaNet state (mirrors the AR `generate` path). The conversation is
    // reset (seq_pos=0, clear tokens, zero DN below) ONLY when it would overflow
    // the KV budget — never unconditionally — so KV/DN are reused across turns.
    let raw_q_tokens = tokenizer.encode(prompt);
    let prompt_est = raw_q_tokens.len() + 8; // small chatml-framing margin
    if m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[hipfire-daemon] mtp: context full ({}/{}) — resetting conversation", m.seq_pos, m.max_seq);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }

    // Jinja full-render fires only on the first turn (seq_pos==0); continuations
    // use the hand-rolled incremental frame (system only on turn 0) — identical
    // to the AR path's multi-turn handling.
    let new_tokens: Vec<u32> = build_prompt_frame(&FrameInputs {
        tokenizer,
        chat_template: m.chat_template.as_deref(),
        system_prompt,
        prompt_text: prompt,
        user_tokens: &raw_q_tokens,
        assistant_prefix,
        max_think_tokens,
        tools,
        messages_history,
        seq_pos: m.seq_pos,
        err_path_label: "mtp",
    });

    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };

    // Fresh start (turn 0 or post-reset): zero the recurrent DeltaNet state.
    // Continuations (seq_pos>0) keep it — DN is cumulative across turns.
    if m.seq_pos == 0 {
        {
            let dn = m.dn_state.as_ref().unwrap();
            for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        }
        // Also reset the MTP head's private KV cache (glm5 C1): it holds stale
        // K/V rows from the prior conversation otherwise, contaminating the
        // head's attention on the first post-reset spec cycle.
        if let Some(mtp) = m.mtp.as_mut() {
            let _ = mtp.spec_state.mtp_kv.reset(gpu);
        }
    }
    let start_pos = m.seq_pos;

    // Assemble transient ModelSlot — same pattern as generate_dflash.
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

    let mtp_state = m.mtp.as_mut().unwrap();

    let t0 = Instant::now();

    // Capacity check (seq_pos-aware: prefill continues from start_pos).
    if start_pos + new_tokens.len() + mtp_state.max_n + 1 > m.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"seq_pos+prompt+max_n+1 exceeds physical_cap {}"}}"#,
            id, m.physical_cap,
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        m.kv_cache = Some(target.kv_cache);
        m.dn_state = Some(target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }

    // Configure sampling vs greedy per request. Sampling (residual accept) and
    // p_min early-exit do NOT compose — spec_step panics if both are active
    // (mtp_spec.rs ~1374) — so clear p_min when sampling and restore the
    // load-configured p_min when greedy. Sampling is only wired on the
    // compressed-serial path; full heads stay greedy. Seed per-request so
    // temp>0 actually varies output across requests (was hardcoded 42).
    // NOTE: repeat_penalty != 1.0 is gated to the AR path in `generate` (the
    // lossless MTP verify can't honor a penalty), so it never reaches here.
    // TODO: plumb request top_k / min_p (currently hardcoded 20 / 0.0).
    let use_sampling = temp > 1e-6 && mtp_state.compressed;
    if use_sampling {
        mtp_state.spec_state.set_p_min(0.0);
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        mtp_state.spec_state.set_sampling(
            MtpSamplingConfig { temp, top_k: 20, top_p, min_p: 0.0 },
            seed,
        );
    } else {
        // Greedy: ensure the load-configured p_min is the active value (a prior
        // sampling request on this slot may have zeroed it).
        mtp_state.spec_state.set_p_min(mtp_state.p_min);
    }

    // Prefill: batched forward over the prompt. After return,
    // target.scratch.tmp holds the post-output-norm hidden at the
    // last prompt position.
    // Graceful failure: on any prefill/seed error we must restore the taken
    // model fields into `m` (they were `.take()`n into `target`) and return,
    // not panic — a panic here crashes the whole serial daemon and leaves the
    // model half-disassembled.
    macro_rules! mtp_bail {
        ($msg:expr) => {{
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":{}}}"#, id, serde_json::to_string(&$msg).unwrap_or_default());
            let _ = stdout.flush();
            m.q35_weights = Some(target.weights);
            m.kv_cache = Some(target.kv_cache);
            m.dn_state = Some(target.dn_state);
            m.q35_scratch = Some(target.scratch);
            return;
        }};
    }

    if let Err(e) = qwen35::forward_prefill_batch(
        gpu,
        &target.weights,
        &target.config,
        &new_tokens,
        start_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None,   // hidden_rb
        None,   // per_token_hidden_out
        None,   // gdn_tape
        None,   // tree_verify
    ) {
        mtp_bail!(format!("mtp prefill: {e}"));
    }
    // Record this turn's prefill frame into the cumulative conversation (the
    // generated tokens are appended during decode below). seq_pos advances to
    // the end of the loop; we set m.seq_pos = position at the end.
    m.conversation_tokens.extend_from_slice(&new_tokens);

    // Seed prev_hidden from scratch.tmp (one d2d copy).
    if let Err(e) = mtp_state.spec_state.capture_prev_hidden_from_scratch_tmp(
        gpu, &target.scratch.tmp, target.config.dim,
    ) {
        mtp_bail!(format!("mtp capture prev_hidden: {e}"));
    }

    // First token via argmax of the trunk's post-prefill logits.
    let first_logits = match gpu.download_f32(&target.scratch.logits) {
        Ok(l) => l,
        Err(e) => mtp_bail!(format!("mtp download seed logits: {e}")),
    };
    let first_token = first_logits.iter().enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
            if v > bv { (i as u32, v) } else { (best, bv) }
        }).0;

    let t_prefill = Instant::now();

    // Decode loop.
    let mut emitted: Vec<u32> = vec![first_token];
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter = EosFilter::new(EosFilterConfig::default());
    let mut position = start_pos + new_tokens.len();
    let mut seed_token = first_token;
    let mut generated = 0usize;
    let mut total_cycles = 0usize;
    let mut total_accepted = 0usize;
    // Tokens committed by verify cycles (excludes the seed). Reported τ =
    // committed/cycle (matches mtp_only_demo's committed_per_cycle_avg), not
    // the draft-accept rate.
    let mut committed_from_cycles = 0usize;
    let eos_token = target.config.eos_token;

    let mut think_count: usize = 0;
    let mut prev_in_think = false;
    // N-gram attractor guard (same detector as the AR path). MTP has a
    // documented attractor history (mtp_bug.md); without this a degenerate
    // loop in production would run to max_tokens uncaught.
    let loop_guard = hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    // Stream the seed/first token. It is the genuine first output token (the
    // input seed to the first spec_step, which only commits *subsequent*
    // tokens). It was previously placed in `emitted` but never streamed, so
    // every MTP response was missing its first token.
    {
        streamed_tokens.push(first_token);
        emit_committed_event(stdout, id, first_token, streamed_tokens.len() - 1, t0.elapsed().as_millis() as u64);
        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
            if let Ok(text) = std::str::from_utf8(&text_bytes) {
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(text).unwrap_or_default());
                let _ = stdout.flush();
            }
        }
        generated = 1;
    }
    let first_token_terminal = first_token == eos_token
        || im_end_token == Some(first_token)
        || tokenizer.is_terminator(first_token);

    while generated < max_tokens && !first_token_terminal {
        if position + mtp_state.max_n + 1 >= m.physical_cap { break; }

        // Always use the compressed-serial path: it handles BOTH cvs-sidecar
        // heads (lm_head_draft present) AND full-vocab heads (branches
        // internally on lm_head_draft.is_none() to use the trunk's lm_head for
        // the discrete-token draft chain). The plain spec_step_mtp K-step chain
        // is the older lossy path with worse τ, so we never use it here.
        let step_result = mtp_spec::spec_step_mtp_compressed_serial(
            gpu,
            &mut target,
            &mtp_state.head,
            &mut mtp_state.spec_state,
            position,
            seed_token,
            eos_token,
        );

        let step = match step_result {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"mtp spec_step: {}"}}"#, id, e);
                let _ = stdout.flush();
                break;
            }
        };

        total_cycles += 1;
        total_accepted += step.accept_count;

        let mut hit_eos = false;
        let mut think_cap_hit = false;
        for &tok in &step.committed {
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
            committed_from_cycles += 1;
            if tok == eos_token || im_end_token == Some(tok) || tokenizer.is_terminator(tok) { hit_eos = true; break; }

            if max_think_tokens > 0 {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let open_idx = raw_str.rfind("<think");
                let close_idx = raw_str.rfind("</think");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think && !prev_in_think { think_count = 0; }
                if in_think { think_count += 1; }
                prev_in_think = in_think;

                if in_think && think_count >= max_think_tokens {
                    let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":"</think>\n"}}"#, id);
                    let _ = stdout.flush();
                    think_cap_hit = true;
                    break;
                }
            }
        }
        position += step.advance;
        seed_token = *step.committed.last().expect("non-empty commit");

        // Per-cycle eviction. MTP + cask/eviction is refused at load (the
        // verify-then-rollback KV write pattern isn't reconciled with the
        // eviction position accounting), so this is defensive: never `.unwrap()`
        // (a panic kills the serial daemon) and never silently ignore a
        // repositioning that the MTP path can't yet honor.
        if let Some(ref ev) = m.eviction {
            match ev.maybe_evict(gpu, &mut target.kv_cache, position) {
                Ok(None) => {}
                Ok(Some(_res)) => {
                    eprintln!("[hipfire-daemon] WARNING: eviction fired under MTP — position accounting not reconciled; output may degrade (MTP+cask should be refused at load)");
                }
                Err(e) => eprintln!("[hipfire-daemon] mtp eviction error (ignored): {e}"),
            }
        }

        // N-gram attractor guard (checked once per committed batch).
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

        if hit_eos || think_cap_hit { break; }
    }

    // F1: write the final committed token's KV + DN. spec_step leaves the last
    // committed token (typically the bonus) at slot `position` UNWRITTEN — it
    // would have been written as the next cycle's seed, but there is no next
    // cycle at loop exit. Without this, turn N+1 attends over a stale/hole slot
    // there and loses turn N's last token (often the <|im_end|> separator).
    // Mirrors the AR path's "write this token's K/V FIRST" guard. This also
    // reconciles seq_pos with conversation_tokens.len() (the +1 seed offset).
    if generated > 0 {
        match qwen35::forward_scratch(
            gpu, &target.weights, &target.config, seed_token, position,
            &mut target.kv_cache, &mut target.dn_state, &target.scratch,
        ) {
            Ok(()) => position += 1,
            Err(e) => eprintln!("[hipfire-daemon] mtp final-token KV write failed (ignored): {e}"),
        }
    }

    // Put target state back.
    m.q35_weights = Some(target.weights);
    m.kv_cache = Some(target.kv_cache);
    m.dn_state = Some(target.dn_state);
    m.q35_scratch = Some(target.scratch);
    // Append this turn's generated tokens (seed + committed) to the cumulative
    // conversation; the prefill frame (new_tokens) was already appended above.
    // seq_pos is the KV-authoritative next position the next turn continues from.
    m.conversation_tokens.extend_from_slice(&emitted);
    m.seq_pos = position;

    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
    let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
    let prefill_tok_s = if prefill_s > 0.0 { new_tokens.len() as f64 / prefill_s } else { 0.0 };
    // τ = committed tokens per verify cycle (matches mtp_only_demo's
    // committed_per_cycle_avg), not the draft-accept rate. total_accepted is
    // still surfaced separately for diagnostics.
    let tau = if total_cycles > 0 { committed_from_cycles as f64 / total_cycles as f64 } else { 0.0 };
    let accept_rate = if total_cycles > 0 { total_accepted as f64 / total_cycles as f64 } else { 0.0 };
    let mtp_sampling = temp > 1e-6;
    let path_extras = format!(
        r#","spec_path":"mtp","mtp_k":{},"tau":{:.2},"accept_rate":{:.2},"cycles":{},"mtp_sampling":{}"#,
        mtp_state.max_n, tau, accept_rate, total_cycles, mtp_sampling,
    );
    emit_done_event(stdout, id, &DoneStats {
        tokens: generated,
        tok_s,
        prefill_tokens: new_tokens.len(),
        prefill_ms: prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms: prefill_s * 1000.0,
    }, &path_extras);
}

/// Stage 2b payoff: PP+MTP. Trunk decoded across `m.pp_gpus.devices[..]`
/// with the MTP head + drafter co-located on `output_device` (set up at
/// load by `load_model_pp_with_mtp` in this file). Each cycle's K-step
/// draft chain runs on output_device; verify dispatches across all PP
/// devices via `forward_prefill_batch_multi_with_caps`; sample / accept
/// stays on output_device.
///
/// V1 scope (matches `spec_step_mtp_compressed_serial_multi` limits):
/// - compressed-sidecar MTP head only (`m.mtp.compressed == true`)
/// - greedy-only (temp = 0); `pick_path` already routed sampling to AR
/// - skip LoopGuard / attractor blocking / repeat_penalty / budget alerts
///   for v1. The pp2-mtp coherence-gate-pp case exercises greedy short
///   prompts where these aren't load-bearing.
fn generate_pp_mtp(m: &mut LoadedModel, ctx: &mut GenerateCtx<'_>) {
    debug_assert_eq!(pick_path(m, ctx), SpecPath::PpMtp,
        "generate_pp_mtp called with path={:?}", pick_path(m, ctx));

    let stdout: &mut std::io::Stdout = &mut *ctx.stdout;
    let id: &str = ctx.id;
    let prompt: &str = ctx.prompt;
    let system_prompt: Option<&str> = ctx.system_prompt;
    let max_tokens: usize = ctx.max_tokens;
    let max_think_tokens: usize = ctx.max_think_tokens;
    let assistant_prefix = ctx.assistant_prefix;
    let tools: Option<&[serde_json::Value]> = ctx.tools;
    let messages_history: Option<&[hipfire_runtime::prompt_frame::Message]> = ctx.messages_history;
    // V1 silences: pp+mtp is greedy-only (temp gated by pick_path), the
    // attractor / budget / pflash fields are intentionally skipped.
    let _ = (ctx.temp, ctx.top_p, ctx.repeat_penalty, ctx.repeat_window,
             ctx.budget_alert_at_tok, ctx.budget_alert_text,
             &mut ctx.drafter_gpu, &mut ctx.pflash_state, ctx.pflash_cfg,
             ctx.pflash_bypass_reason, ctx.pflash_alpha);

    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Multi-turn: prefill only the NEW turn's tokens on top of cumulative
    // KV + DN. Reset when overflow would push past max_seq (mirrors AR /
    // generate_mtp behavior).
    let raw_q_tokens = tokenizer.encode(prompt);
    let prompt_est = raw_q_tokens.len() + 8;
    if m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[hipfire-daemon] pp-mtp: context full ({}/{}) — resetting conversation",
                  m.seq_pos, m.max_seq);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }

    let new_tokens: Vec<u32> = build_prompt_frame(&FrameInputs {
        tokenizer,
        chat_template: m.chat_template.as_deref(),
        system_prompt,
        prompt_text: prompt,
        user_tokens: &raw_q_tokens,
        assistant_prefix,
        max_think_tokens,
        tools,
        messages_history,
        seq_pos: m.seq_pos,
        err_path_label: "pp-mtp",
    });

    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };

    // Fresh-start (turn 0 or post-reset): zero recurrent state on
    // output_device — the dn_state lives there per load_model_pp_with_mtp
    // convention; the MTP head's KV cache also resets.
    let output_device = m.pp_gpus.as_ref().expect("pp_gpus populated").output_device;
    if m.seq_pos == 0 {
        {
            let g = &mut m.pp_gpus.as_mut().unwrap().devices[output_device];
            let dn = m.dn_state.as_ref().unwrap();
            for s in &dn.s_matrices { let _ = g.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.s_scales { let _ = g.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.conv_states { let _ = g.hip.memset(&s.buf, 0, s.buf.size()); }
        }
        if let Some(mtp) = m.mtp.as_mut() {
            let g = &mut m.pp_gpus.as_mut().unwrap().devices[output_device];
            let _ = mtp.spec_state.mtp_kv.reset(g);
        }
    }
    let start_pos = m.seq_pos;

    let t0 = Instant::now();
    let dim = m.q35_config.as_ref().unwrap().dim;

    // Take the bookkeeping fields out of `m` so we can pass &mut refs
    // alongside &mut m.pp_gpus. Put them back unconditionally before
    // return (matches generate_mtp's "mtp_bail!" pattern).
    let weights = m.q35_weights.take().expect("q35 weights");
    let mut kv_cache = m.kv_cache.take().expect("kv cache");
    let mut dn_state = m.dn_state.take().expect("dn state");
    let pp_scratch_set = m.pp_scratch_set.take().expect("pp_scratch_set populated for PP path");
    let mut mtp_state = m.mtp.take().expect("mtp populated on PpMtp path");
    let drafter_state_opt = mtp_state.drafter_state.take();
    let mut drafter_state = match drafter_state_opt {
        Some(ds) => ds,
        None => {
            let _ = writeln!(stdout,
                r#"{{"type":"error","id":"{}","message":"pp-mtp: m.mtp.drafter_state is None — PP+MTP load must populate it"}}"#, id);
            let _ = stdout.flush();
            m.q35_weights = Some(weights);
            m.kv_cache = Some(kv_cache);
            m.dn_state = Some(dn_state);
            m.pp_scratch_set = Some(pp_scratch_set);
            mtp_state.drafter_state = None;
            m.mtp = Some(mtp_state);
            return;
        }
    };
    let target_config = m.q35_config.as_ref().unwrap().clone();

    // Restore everything before any early return / panic. Wrap the rest
    // of the function in a closure so we can put state back even on early
    // return / error.
    macro_rules! ppmtp_bail {
        ($msg:expr) => {{
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":{}}}"#,
                id, serde_json::to_string(&$msg).unwrap_or_default());
            let _ = stdout.flush();
            m.q35_weights = Some(weights);
            m.kv_cache = Some(kv_cache);
            m.dn_state = Some(dn_state);
            m.pp_scratch_set = Some(pp_scratch_set);
            mtp_state.drafter_state = Some(drafter_state);
            m.mtp = Some(mtp_state);
            return;
        }};
    }

    // Allocate the per_token_hidden_out capture buffer on output_device.
    // Sized for the WHOLE prefill (chunk loop offsets into it).
    let per_tok_hidden_out = {
        let g = &mut m.pp_gpus.as_mut().unwrap().devices[output_device];
        match g.alloc_tensor(&[new_tokens.len() * dim], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => ppmtp_bail!(format!("pp-mtp alloc per_tok_hidden_out: {e}")),
        }
    };

    // ── Prefill ──
    {
        let gpus = m.pp_gpus.as_mut().unwrap();
        if let Err(e) = qwen35::forward_prefill_batch_multi_with_caps(
            gpus, &weights, &target_config, &new_tokens, start_pos,
            &mut kv_cache, &mut dn_state, &pp_scratch_set,
            Some(&per_tok_hidden_out),
            None, // gdn_tape — v1 disabled (matches spec_step_mtp_compressed_serial_multi)
            None, // tree_verify — MTP verify is linear
        ) {
            // Free the per-tok buffer before bail; bail returns.
            let g = &mut m.pp_gpus.as_mut().unwrap().devices[output_device];
            let _ = g.free_tensor(per_tok_hidden_out);
            ppmtp_bail!(format!("pp-mtp prefill: {e}"));
        }
    }
    m.conversation_tokens.extend_from_slice(&new_tokens);

    // Seed MTP head's prev_hidden from the LAST row of per_tok_hidden_out.
    {
        let g = &mut m.pp_gpus.as_mut().unwrap().devices[output_device];
        let last_row_off = (new_tokens.len() - 1) * dim * 4;
        if let Err(e) = g.hip.memcpy_dtod_at(
            &mtp_state.spec_state.prev_hidden.buf, 0,
            &per_tok_hidden_out.buf, last_row_off,
            dim * 4,
        ) {
            let _ = g.free_tensor(per_tok_hidden_out);
            ppmtp_bail!(format!("pp-mtp seed prev_hidden: {e}"));
        }
        // Also seed the drafter_state's prev_hidden (the chain reads this
        // on cycle 0). Same source row.
        if let Err(e) = g.hip.memcpy_dtod_at(
            &drafter_state.prev_hidden.buf, 0,
            &per_tok_hidden_out.buf, last_row_off,
            dim * 4,
        ) {
            let _ = g.free_tensor(per_tok_hidden_out);
            ppmtp_bail!(format!("pp-mtp seed drafter_state.prev_hidden: {e}"));
        }
    }

    // First token via argmax of the trunk's post-prefill logits — but the
    // multi prefill doesn't write logits to a single-device buffer the way
    // single-gpu does. Run one extra single-token forward on output_device
    // using the LAST per_tok_hidden row → trunk.output GEMV → argmax.
    let first_token: u32 = {
        let g = &mut m.pp_gpus.as_mut().unwrap().devices[output_device];
        // Last row of per_tok_hidden is the post-output-norm hidden of the
        // last prompt token. trunk.output lives on output_device per PP
        // load layout, so GEMV is safe here.
        let w_out = &weights.output;
        // Allocate scratch logits + argmax (small, freed immediately).
        let logits = match g.alloc_tensor(&[target_config.vocab_size], rdna_compute::DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = g.free_tensor(per_tok_hidden_out);
                ppmtp_bail!(format!("pp-mtp alloc first-token logits: {e}"));
            }
        };
        let last_row_view = per_tok_hidden_out.sub_offset(
            (new_tokens.len() - 1) * dim, dim,
        );
        if let Err(e) = hipfire_runtime::llama::weight_gemv(g, w_out, &last_row_view, &logits) {
            let _ = g.free_tensor(logits);
            let _ = g.free_tensor(per_tok_hidden_out);
            ppmtp_bail!(format!("pp-mtp first-token GEMV: {e}"));
        }
        let logits_host = match g.download_f32(&logits) {
            Ok(v) => v,
            Err(e) => {
                let _ = g.free_tensor(logits);
                let _ = g.free_tensor(per_tok_hidden_out);
                ppmtp_bail!(format!("pp-mtp first-token download: {e}"));
            }
        };
        let _ = g.free_tensor(logits);
        logits_host.iter().enumerate()
            .fold((0u32, f32::NEG_INFINITY), |(best, bv), (i, &v)| {
                if v > bv { (i as u32, v) } else { (best, bv) }
            }).0
    };

    // Free the prefill capture buffer now — no longer needed.
    {
        let g = &mut m.pp_gpus.as_mut().unwrap().devices[output_device];
        let _ = g.free_tensor(per_tok_hidden_out);
    }

    let t_prefill = Instant::now();

    // ── Decode loop ──
    let mut emitted: Vec<u32> = vec![first_token];
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter = hipfire_runtime::eos_filter::EosFilter::new(
        hipfire_runtime::eos_filter::EosFilterConfig::default());
    let mut position = start_pos + new_tokens.len();
    let mut seed_token = first_token;
    let mut generated = 0usize;
    let mut total_cycles = 0usize;
    let mut total_accepted = 0usize;
    let mut committed_from_cycles = 0usize;
    let eos_token = target_config.eos_token;

    // Stream the seed/first token (same as generate_mtp).
    {
        streamed_tokens.push(first_token);
        emit_committed_event(stdout, id, first_token, streamed_tokens.len() - 1,
                             t0.elapsed().as_millis() as u64);
        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        if let hipfire_runtime::eos_filter::FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
            if let Ok(text) = std::str::from_utf8(&text_bytes) {
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#,
                    id, serde_json::to_string(text).unwrap_or_default());
                let _ = stdout.flush();
            }
        }
        generated = 1;
    }
    let first_token_terminal = first_token == eos_token
        || im_end_token == Some(first_token)
        || tokenizer.is_terminator(first_token);

    while generated < max_tokens && !first_token_terminal {
        if position + mtp_state.max_n + 1 >= m.physical_cap { break; }

        let step_result = {
            let gpus = m.pp_gpus.as_mut().unwrap();
            hipfire_arch_qwen35::mtp_spec::spec_step_mtp_compressed_serial_multi(
                gpus,
                output_device,
                &target_config,
                &weights,
                &mut kv_cache,
                &mut dn_state,
                &pp_scratch_set,
                &mtp_state.head,
                &mut mtp_state.spec_state,
                &mut drafter_state,
                position,
                seed_token,
                eos_token,
            )
        };
        let step = match step_result {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(stdout,
                    r#"{{"type":"error","id":"{}","message":"pp-mtp spec_step: {}"}}"#, id, e);
                let _ = stdout.flush();
                break;
            }
        };

        total_cycles += 1;
        total_accepted += step.accept_count;

        let mut hit_eos = false;
        for &tok in &step.committed {
            if generated >= max_tokens { break; }
            emitted.push(tok);
            streamed_tokens.push(tok);
            emit_committed_event(stdout, id, tok, streamed_tokens.len() - 1,
                                 t0.elapsed().as_millis() as u64);
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            if let hipfire_runtime::eos_filter::FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                let text = std::str::from_utf8(&text_bytes).unwrap();
                let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#,
                    id, serde_json::to_string(&text).unwrap_or_default());
                let _ = stdout.flush();
            }
            generated += 1;
            committed_from_cycles += 1;
            if tok == eos_token || im_end_token == Some(tok) || tokenizer.is_terminator(tok) {
                hit_eos = true; break;
            }
        }
        position += step.advance;
        seed_token = *step.committed.last().expect("non-empty commit");

        if hit_eos { break; }
    }

    // Final-token KV write (same pattern as generate_mtp:F1 fix).
    if generated > 0 {
        let gpus = m.pp_gpus.as_mut().unwrap();
        let scratch_set_ref = &pp_scratch_set;
        if let Err(e) = qwen35::forward_scratch_multi(
            gpus, &weights, &target_config, seed_token, position,
            &mut kv_cache, &mut dn_state, scratch_set_ref,
        ) {
            eprintln!("[hipfire-daemon] pp-mtp final-token KV write failed (ignored): {e}");
        } else {
            position += 1;
        }
    }

    // Put state back.
    m.q35_weights = Some(weights);
    m.kv_cache = Some(kv_cache);
    m.dn_state = Some(dn_state);
    m.pp_scratch_set = Some(pp_scratch_set);
    mtp_state.drafter_state = Some(drafter_state);
    m.mtp = Some(mtp_state);
    m.conversation_tokens.extend_from_slice(&emitted);
    m.seq_pos = position;

    let mtp_state_ref = m.mtp.as_ref().unwrap();
    let t_end = Instant::now();
    let total_s = t_end.duration_since(t0).as_secs_f64();
    let prefill_s = t_prefill.duration_since(t0).as_secs_f64();
    let decode_s = t_end.duration_since(t_prefill).as_secs_f64();
    let tok_s = if total_s > 0.0 { generated as f64 / total_s } else { 0.0 };
    let decode_tok_s = if decode_s > 0.0 { generated as f64 / decode_s } else { 0.0 };
    let prefill_tok_s = if prefill_s > 0.0 { new_tokens.len() as f64 / prefill_s } else { 0.0 };
    let tau = if total_cycles > 0 { committed_from_cycles as f64 / total_cycles as f64 } else { 0.0 };
    let accept_rate = if total_cycles > 0 { total_accepted as f64 / total_cycles as f64 } else { 0.0 };
    let path_extras = format!(
        r#","spec_path":"pp-mtp","mtp_k":{},"tau":{:.2},"accept_rate":{:.2},"cycles":{},"pp":{}"#,
        mtp_state_ref.max_n, tau, accept_rate, total_cycles, m.pp,
    );
    emit_done_event(stdout, id, &DoneStats {
        tokens: generated,
        tok_s,
        prefill_tokens: new_tokens.len(),
        prefill_ms: prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms: prefill_s * 1000.0,
    }, &path_extras);
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
fn generate_multi(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    ctx: &mut GenerateCtx<'_>,
) {
    // Allow both PpAr (true PP-AR) and PpMtp (pre-step-5d: PP+MTP
    // falls back to AR through this function). After step 5d's
    // Path::PpMtp arm lands and the dispatcher routes PP+MTP
    // elsewhere, the second arm should be removed.
    debug_assert!(
        matches!(pick_path(m, ctx), SpecPath::PpAr | SpecPath::PpMtp),
        "generate_multi called with path={:?}", pick_path(m, ctx),
    );
    // Local aliases preserve the existing body verbatim. Sampling /
    // budget fields are Copy; ref fields are reborrowed.
    let pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState> =
        ctx.pflash_state.as_deref_mut();
    let pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig> = ctx.pflash_cfg;
    let stdout: &mut std::io::Stdout = &mut *ctx.stdout;
    let id: &str = ctx.id;
    let prompt: &str = ctx.prompt;
    let system_prompt: Option<&str> = ctx.system_prompt;
    let temp: f32 = ctx.temp;
    let top_p: f32 = ctx.top_p;
    let max_tokens: usize = ctx.max_tokens;
    let repeat_penalty: f32 = ctx.repeat_penalty;
    let _repeat_window: usize = ctx.repeat_window;
    let budget_alert_at_tok: usize = ctx.budget_alert_at_tok;
    let budget_alert_text: &str = ctx.budget_alert_text;
    let max_think_tokens: usize = ctx.max_think_tokens;
    let assistant_prefix = ctx.assistant_prefix;
    let tools: Option<&[serde_json::Value]> = ctx.tools;
    let messages_history: Option<&[hipfire_runtime::prompt_frame::Message]> = ctx.messages_history;
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
    let new_tokens = build_prompt_frame(&FrameInputs {
        tokenizer,
        chat_template: m.chat_template.as_deref(),
        system_prompt,
        prompt_text: prompt,
        user_tokens: &q_tokens,
        assistant_prefix,
        max_think_tokens,
        tools,
        messages_history,
        seq_pos: m.seq_pos,
        err_path_label: "pp",
    });

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
    let loop_guard = hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    while generated < max_tokens {
        generated += 1;
        m.conversation_tokens.push(next_token);
        stream_token_step(stdout, id, next_token, tokenizer, &mut streamed_tokens, &mut bytes_fed_to_filter, &mut filter, t0);

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
            let (in_think, new_count) = detect_think_state(&raw_so_far, prev_in_think, think_count);
            think_count = new_count;
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
    emit_done_event(stdout, id, &DoneStats {
        tokens: generated,
        tok_s,
        prefill_tokens,
        prefill_ms: prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        ttft_ms: prefill_s * 1000.0,
    }, "");
}

/// Top-level generate dispatcher for the qwen35-family of paths
/// (arch_id 5/6 + Qwen2 routing for arch_id 7). Step 5a of
/// docs/plans/mtp_multi_refactor.md v2.1 renamed the prior
/// `generate` to `generate_qwen35` and switched its signature to the
/// `GenerateCtx` shape that the three leaf functions already use.
///
/// The body still contains the inline AR implementation AND the
/// short-circuit dispatches to generate_qwen2 / generate_multi /
/// generate_dflash / generate_mtp at the top. Subsequent step 5b/5c/5d
/// commits will collapse those dispatches into a single match on
/// pick_path(), folding each leaf's body in turn.
fn generate_qwen35(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    ctx: &mut GenerateCtx<'_>,
) {
    // Compute the dispatch decision BEFORE the alias-binding block —
    // pick_path takes `&GenerateCtx`, which conflicts with the `&mut
    // *ctx.stdout` reborrow below.
    let path = pick_path(m, ctx);
    // Local aliases preserve the existing body verbatim. Sampling /
    // budget fields are Copy; ref fields are reborrowed. The
    // drafter_gpu field is `take()`n into a mutable local so the
    // PFlash compress site can `as_deref_mut` it (matches the prior
    // `let mut drafter_gpu = drafter_gpu;` pattern).
    let stdout: &mut std::io::Stdout = &mut *ctx.stdout;
    let id: &str = ctx.id;
    let prompt: &str = ctx.prompt;
    let system_prompt: Option<&str> = ctx.system_prompt;
    let temp: f32 = ctx.temp;
    let top_p: f32 = ctx.top_p;
    let max_tokens: usize = ctx.max_tokens;
    let repeat_penalty: f32 = ctx.repeat_penalty;
    let repeat_window: usize = ctx.repeat_window;
    let budget_alert_at_tok: usize = ctx.budget_alert_at_tok;
    let budget_alert_text: &str = ctx.budget_alert_text;
    let max_think_tokens: usize = ctx.max_think_tokens;
    let assistant_prefix = ctx.assistant_prefix;
    let pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState> =
        ctx.pflash_state.as_deref_mut();
    let pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig> = ctx.pflash_cfg;
    let tools: Option<&[serde_json::Value]> = ctx.tools;
    let messages_history: Option<&[hipfire_runtime::prompt_frame::Message]> = ctx.messages_history;
    let mut drafter_gpu = ctx.drafter_gpu.take();
    // ── Step 5b: unified dispatch via pick_path ─────────────────────────
    //
    // Replaces the chain of four `if … return` short-circuits with one
    // explicit match on the SpecPath enum. Each arm rebuilds the per-leaf
    // GenerateCtx (separate borrows from the moved-out scalars/refs above)
    // and delegates to the leaf function exactly as the prior dispatches did.
    //
    // - PpAr and PpMtp BOTH delegate to generate_multi today. There is no
    //   generate_mtp_multi yet; the PpMtp branch lands in step 5d.
    // - The MTP-blocked-by-penalty warning still fires on the Ar arm when
    //   the model has MTP loaded but repeat_penalty > 1.0 (lossless verify
    //   can't honor a sampling penalty).
    // - Ar is the only arm that falls through to the inline body below.
    //
    // The body of generate_multi / generate_mtp / generate_dflash assertion-
    // checks pick_path on entry, so a divergence between this match and
    // pick_path() is caught immediately under debug.
    match path {
        SpecPath::Qwen2 => {
            // arch_id=7 (hipfire-arch-qwen2). The standard qwen35/llama
            // body would panic on None unwraps for q35_*/llama_* fields
            // when applied to a Qwen2 model. None of PFlash / DFlash /
            // multi-GPU / ChatML scaffolding is wired for arch_id=7 yet
            // (R3 bring-up scope).
            let _ = (budget_alert_at_tok, budget_alert_text, max_think_tokens,
                     assistant_prefix, pflash_state, pflash_cfg, tools, messages_history);
            generate_qwen2(
                m, gpu, stdout, id, prompt, system_prompt,
                temp, top_p, max_tokens, repeat_penalty, repeat_window,
            );
            return;
        }
        SpecPath::PpAr => {
            // Multi-GPU pipeline-parallel AR decode (Stage 7 of #58). pp>1
            // is refused at load when DFlash / CASK / PFlash / VL is
            // requested, so this branch doesn't need to thread any of
            // those args through.
            let mut multi_ctx = GenerateCtx {
                stdout, id, prompt, system_prompt, tools, messages_history,
                assistant_prefix,
                temp, top_p, repeat_penalty, repeat_window,
                max_tokens, max_think_tokens,
                budget_alert_at_tok, budget_alert_text,
                drafter_gpu: None,
                pflash_state,
                pflash_cfg,
                pflash_bypass_reason: None,
                pflash_alpha: None,
            };
            generate_multi(m, gpu, &mut multi_ctx);
            return;
        }
        SpecPath::PpMtp => {
            // Stage 2b payoff: pp>1 trunk + MTP head co-located on
            // output_device. Caller's gpu arg is ignored — generate_pp_mtp
            // pulls m.pp_gpus directly. Per-cycle MTP spec_step runs the
            // K-step draft chain on output_device and dispatches verify
            // across all PP devices via
            // forward_prefill_batch_multi_with_caps.
            let _ = gpu; // Unused: PP path uses m.pp_gpus directly.
            let mut pp_mtp_ctx = GenerateCtx {
                stdout, id, prompt, system_prompt, tools, messages_history,
                assistant_prefix,
                temp, top_p, repeat_penalty, repeat_window,
                max_tokens, max_think_tokens,
                budget_alert_at_tok, budget_alert_text,
                drafter_gpu: None,
                pflash_state,
                pflash_cfg,
                pflash_bypass_reason: None,
                pflash_alpha: None,
            };
            generate_pp_mtp(m, &mut pp_mtp_ctx);
            return;
        }
        SpecPath::Dflash => {
            // DFlash spec-decode (greedy-only). PFlash compression on the
            // DFlash path is not yet wired — emit a loud bypass event if
            // an operator set prefill_compression != off, so they see the
            // bypass instead of silent full-prefill behavior.
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
            let dflash_bypass_reason_owned = dflash_bypass_reason.map(|s| s.to_string());
            let mut dflash_ctx = GenerateCtx {
                stdout, id, prompt, system_prompt, tools, messages_history,
                assistant_prefix,
                temp, top_p, repeat_penalty, repeat_window,
                max_tokens, max_think_tokens,
                budget_alert_at_tok: 0,
                budget_alert_text: "",
                drafter_gpu: None,
                pflash_state: None,
                pflash_cfg: None,
                pflash_bypass_reason: dflash_bypass_reason_owned.as_deref(),
                pflash_alpha: dflash_alpha,
            };
            generate_dflash(m, gpu, &mut dflash_ctx);
            // top_p / repeat penalties are AR-only sampling knobs;
            // pflash_state is bypassed on the DFlash decode path.
            let _ = (budget_alert_at_tok, budget_alert_text, pflash_state);
            return;
        }
        SpecPath::Mtp => {
            // MTP spec-decode (greedy + temp>0 via compressed-serial
            // residual-accept). MTP doesn't use drafter_gpu / pflash_state /
            // pflash_cfg / budget_alert_* / pflash_bypass_reason /
            // pflash_alpha — pass None / empty defaults.
            let mut mtp_ctx = GenerateCtx {
                stdout, id, prompt, system_prompt, tools, messages_history,
                assistant_prefix,
                temp, top_p, repeat_penalty, repeat_window,
                max_tokens, max_think_tokens,
                budget_alert_at_tok: 0,
                budget_alert_text: "",
                drafter_gpu: None,
                pflash_state: None,
                pflash_cfg: None,
                pflash_bypass_reason: None,
                pflash_alpha: None,
            };
            generate_mtp(m, gpu, &mut mtp_ctx);
            let _ = (budget_alert_at_tok, budget_alert_text, pflash_state, pflash_cfg);
            return;
        }
        SpecPath::Ar => {
            // Falls through to the inline AR body below. If MTP is loaded
            // but blocked by a sampling penalty, surface that loud so the
            // operator sees the AR fallback instead of silently losing
            // their MTP speedup.
            if m.mtp.is_some() && repeat_penalty > 1.0 + 1e-3 {
                eprintln!("[hipfire-daemon] MTP bypassed (AR fallback): repeat_penalty={repeat_penalty:.3} != 1.0 is not supported by the lossless MTP verify");
            }
        }
    }
    // budgeted_thinking_needs_ar is consumed by pick_path. The inline AR
    // body below doesn't gate on it — think-budget bookkeeping happens in
    // the decode loop's <think>/</think> counter directly.

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
    if std::env::var("HIPFIRE_PFLASH_DEBUG").is_ok() {
        eprintln!("[pflash] gen: state={} cfg-present seq_pos={} q={} drafter_gpu={}",
            pflash_state.is_some(), m.seq_pos, raw_q_tokens.len(), drafter_gpu.is_some());
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.seq_pos == 0 {
            let compress_gpu: &mut rdna_compute::Gpu = drafter_gpu.as_deref_mut().unwrap_or(gpu);
            // Sibling-device drafter: bind its device before compress, then
            // restore the target binding for decode. No-op when shared.
            compress_gpu.bind_thread_or_warn();
            let decision = hipfire_arch_qwen35::pflash::maybe_compress_prompt(
                compress_gpu, state, cfg, &raw_q_tokens, request_kind, &[],
            );
            gpu.bind_thread_or_warn();
            match decision {
                Ok(hipfire_arch_qwen35::pflash::PflashDecision::Compressed(cp)) => {
                    eprintln!("[pflash] COMPRESSED {} -> {} tok dev1 ({}ms)", cp.source_tokens, cp.kept_tokens, cp.timings.total_ms);
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
                    eprintln!("[pflash] BYPASS reason={} q={}", reason.as_str(), raw_q_tokens.len());
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
                    eprintln!("[pflash] ERROR compress: {e}");
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
    let new_tokens = build_prompt_frame(&FrameInputs {
        tokenizer,
        chat_template: m.chat_template.as_deref(),
        system_prompt,
        prompt_text: prompt,
        user_tokens: &q_tokens,
        assistant_prefix,
        max_think_tokens,
        tools,
        messages_history,
        seq_pos: m.seq_pos,
        err_path_label: "",
    });

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
        let loop_guard = hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

        // `while` instead of `for 0..max_tokens` so budget-alert injection
        // (which increments `generated` beyond the iteration count) can't
        // push generated past max_tokens: each loop start rechecks the cap.
        while generated < max_tokens {
            generated += 1;
            m.conversation_tokens.push(next_token);
            // Incremental UTF-8 + filter routing: feed only the new
            // bytes since last call, let the filter buffer any partial
            // codepoint or marker prefix until disambiguated.
            stream_token_step(stdout, id, next_token, tokenizer, &mut streamed_tokens, &mut bytes_fed_to_filter, &mut filter, t0);

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
                let (in_think, new_count) = detect_think_state(&raw_so_far, prev_in_think, think_count);
                think_count = new_count;
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
        emit_done_event(stdout, id, &DoneStats {
            tokens: generated,
            tok_s,
            prefill_tokens,
            prefill_ms: prefill_s * 1000.0,
            prefill_tok_s,
            decode_tok_s,
            ttft_ms: prefill_s * 1000.0,
        }, &pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha));
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
            stream_token_step(stdout, id, next_token, tokenizer, &mut streamed_tokens, &mut bytes_fed_to_filter, &mut filter, t0);

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
        emit_done_event(stdout, id, &DoneStats {
            tokens: generated,
            tok_s,
            prefill_tokens,
            prefill_ms: prefill_s * 1000.0,
            prefill_tok_s,
            decode_tok_s,
            ttft_ms: prefill_s * 1000.0,
        }, &pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha));
    }
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
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#, id);
            let _ = stdout.flush();
            return;
        }
    };
    let cfg = match m.qwen2_config.as_ref() {
        Some(c) => c,
        None => {
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"qwen2_config missing on arch_id=7 generate"}}"#, id);
            let _ = stdout.flush();
            return;
        }
    };
    let weights = m.qwen2_weights.as_ref().expect("qwen2_weights missing on arch_id=7 generate");
    let state = m.qwen2_state.as_mut().expect("qwen2_state missing on arch_id=7 generate");

    let prompt_ids = tokenizer.encode(prompt);
    if prompt_ids.is_empty() {
        let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#, id);
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
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"qwen2 prefill failed: {:?}"}}"#, id, e);
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
            let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"argmax failed: {:?}"}}"#, id, e);
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
        // to an empty string we still advance the loop. JSON-escape the
        // chunk so embedded quotes / control chars don't break the
        // wire format.
        let frag = tokenizer.decode(&[next_tok]);
        let frag_escaped = json_escape(&frag);
        let _ = writeln!(
            stdout,
            r#"{{"type":"token","id":"{}","text":"{}"}}"#,
            id, frag_escaped,
        );
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        match qwen2::forward_step_greedy(gpu, weights, cfg, state, next_tok) {
            Ok(t) => next_tok = t,
            Err(e) => {
                let _ = writeln!(stdout, r#"{{"type":"error","id":"{}","message":"forward_step_greedy failed: {:?}"}}"#, id, e);
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

/// Minimal JSON string escaper for the daemon's token-stream emit
/// path. We have full control of the alphabet (BPE-decoded UTF-8)
/// and only need to escape: backslash, double-quote, and control
/// chars < 0x20. Non-ASCII bytes pass through as-is (the daemon
/// client parses with serde_json which accepts UTF-8 verbatim).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn generate_vl(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, params: &GenerateVLParams) {
    let GenerateVLParams { id, prompt, system_prompt, ref image_source, temp, top_p, max_tokens, repeat_penalty, repeat_window, max_think_tokens } = *params;
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let vision_config = m.vision_config.as_ref().unwrap();

    // Vision special-token IDs resolved from the tokenizer rather than
    // hardcoded constants. Different VL-capable Qwen variants ship with
    // different IDs for these tokens; a hardcoded mismatch silently
    // splices the wrong tokens into the prompt. Required at load time —
    // panic loudly here so the failure is at first-VL-request, not after
    // a successful but wrong forward pass.
    let image_pad_id = tokenizer.special_token_id("<|image_pad|>")
        .unwrap_or_else(|| panic!("VL tokenizer missing <|image_pad|> special token"));
    let vision_start_id = tokenizer.special_token_id("<|vision_start|>")
        .unwrap_or_else(|| panic!("VL tokenizer missing <|vision_start|> special token"));
    let vision_end_id = tokenizer.special_token_id("<|vision_end|>")
        .unwrap_or_else(|| panic!("VL tokenizer missing <|vision_end|> special token"));

    // Image preprocessing (CPU decode + smart resize). Cheap relative to
    // the GPU vision encoder, so we run it before the capacity check —
    // we need img_h/img_w to estimate visual tokens, and rejecting an
    // over-budget request before vision_forward saves expensive GPU work.
    let (pixels, img_h, img_w) = match image_source {
        ImageSource::Path(path) => {
            eprintln!("[VL-DEBUG] preprocessing image: path: {}", path);
            match image::load_and_preprocess(
                Path::new(path),
                vision_config.patch_size,
                vision_config.spatial_merge_size,
            ) {
                Ok(result) => result,
                Err(e) => {
                    write_error(stdout, id, &e);
                    return;
                }
            }
        }
        ImageSource::Base64(b64) => {
            // Strip optional `data:...;base64,` prefix. A `data:` URL
            // missing the comma separator is malformed — surface that
            // explicitly rather than letting it fall through to a
            // misleading "invalid byte 'd' at index 0" base64 error.
            let raw_b64 = if let Some(rest) = b64.strip_prefix("data:") {
                match rest.split_once(',') {
                    Some((_, after)) => after,
                    None => {
                        write_error(stdout, id, "malformed data URL: missing ',' separator");
                        return;
                    }
                }
            } else {
                b64
            };
            eprintln!("[VL-DEBUG] preprocessing image: <{}-byte buffer>", raw_b64.len());
            let bytes = match Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64) {
                Ok(b) => b,
                Err(e) => {
                    write_error(stdout, id, &format!("failed to decode base64 image data: {e}"));
                    return;
                }
            };
            match image::load_and_preprocess_from_bytes(
                &bytes,
                vision_config.patch_size,
                vision_config.spatial_merge_size,
            ) {
                Ok(result) => result,
                Err(e) => {
                    write_error(stdout, id, &e);
                    return;
                }
            }
        }
    };
    eprintln!("[VL-DEBUG] preprocessed: {}x{}", img_w, img_h);

    let grid_h = img_h / vision_config.patch_size;
    let grid_w = img_w / vision_config.patch_size;
    let n_patches = grid_h * grid_w;
    let n_visual_tokens = n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);

    // Capacity estimate including system prompt — a long system prompt
    // on first turn would otherwise let an over-budget request through
    // the soft check, only to fail the hard check after the expensive
    // vision encoder runs.
    let system_est = system_prompt.map(|s| tokenizer.encode(s).len()).unwrap_or(0);
    let prompt_est = tokenizer.encode(prompt).len() + system_est + n_visual_tokens + 20;

    if m.eviction.is_none() && m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!("[daemon/vl] context full ({}/{}) — resetting conversation", m.seq_pos, m.max_seq);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        if let Some(ref dn) = m.dn_state {
            for s in &dn.s_matrices { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.s_scales { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
            for s in &dn.conv_states { let _ = gpu.hip.memset(&s.buf, 0, s.buf.size()); }
        }
        if let Some(kv) = m.kv_cache.as_mut() { kv.compact_offset = 0; }
    }

    if m.eviction.is_none() && prompt_est + max_tokens > m.max_seq {
        write_error(stdout, id, &format!(
            "request size ({} tokens) exceeds loaded KV budget ({})",
            prompt_est + max_tokens, m.max_seq,
        ));
        return;
    }

    let config = m.q35_config.as_ref().unwrap();
    let vision_weights = m.vision_weights.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch = m.q35_scratch.as_ref().unwrap();
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();

    // Build the actual prompt token sequence BEFORE running the GPU vision
    // encoder so the hard capacity check uses the real prefill length, not
    // the estimate. The vision tower is the most expensive part of a VL
    // prefill — failing earlier saves the round-trip on over-budget requests.
    let nl = tokenizer.encode("\n");
    let im_end = tokenizer.encode("<|im_end|>");
    let q_tokens = tokenizer.encode(prompt);

    let mut user_body: Vec<u32> = Vec::with_capacity(n_visual_tokens + q_tokens.len() + 4);
    user_body.push(vision_start_id);
    for _ in 0..n_visual_tokens {
        user_body.push(image_pad_id);
    }
    user_body.push(vision_end_id);
    user_body.extend_from_slice(&nl);
    user_body.extend_from_slice(&q_tokens);

    let prompt_tokens = hipfire_runtime::prompt_frame::ChatFrame {
        tokenizer,
        system: if m.seq_pos == 0 { system_prompt } else { None },
        user: "", // unused: we pass tokens directly via build_with_user_tokens
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain, // VL always uses Plain
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
        write_error(stdout, id, &format!(
            "request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > cap={} — reload model with a larger max_seq",
            m.seq_pos, prompt_tokens.len(), max_tokens, trailer,
            if m.eviction.is_none() { m.physical_cap } else { m.max_seq },
        ));
        return;
    }

    // Now safe to run the expensive GPU vision encoder.
    let patches = hipfire_arch_qwen35_vl::image::extract_patches(
        &pixels, 3, img_h, img_w,
        vision_config.patch_size, vision_config.temporal_patch_size,
        vision_config.spatial_merge_size,
    );
    let visual_tokens = qwen35_vl::vision_forward(gpu, vision_weights, vision_config, &patches, grid_h, grid_w)
        .expect("vision forward failed");

    let im_end_token = if im_end.len() == 1 { Some(im_end[0]) } else { None };
    let prefill_tokens = prompt_tokens.len();
    let t0 = Instant::now();

    // Mirror the text path: <think>/</think> as paired open/close. The
    // previous implementation queried "💭" twice (open == close) which
    // collapsed depth tracking and made `in_think` always-false; the
    // force-close splice also encoded the open emoji, doubling the
    // unclosed depth instead of closing it.
    let think_pair = match (
        tokenizer.special_token_id("<think>"),
        tokenizer.special_token_id("</think>"),
    ) {
        (Some(o), Some(c)) => Some((o, c)),
        _ => None,
    };

    // Prefill with vision token embedding for image_pad positions. VL
    // prefill is per-token (forward_scratch_embed isn't batched), so we
    // advance m.seq_pos in-loop and call maybe_evict after every write.
    let mut visual_idx = 0usize;
    for &token in prompt_tokens.iter() {
        if token == image_pad_id && visual_idx < n_visual_tokens {
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
    // Attractor-block uses CPU-side mutation of the downloaded logits
    // vector (`block_attractor_unclosed_cpu`) instead of the previous
    // GPU memcpy + redownload — saves a full vocab-sized DMA per token.
    let mut logits = gpu.download_f32(&scratch.logits).unwrap();
    if let Some((open, close)) = think_pair {
        block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
    }
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
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;

    // N-gram loop detector — mirrors the text path. Catches answer-phase
    // attractor loops that the think cap and repeat penalty miss.
    let loop_guard = hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    while generated < max_tokens {
        generated += 1;
        m.conversation_tokens.push(next_token);
        emit_committed_event(stdout, id, next_token, generated - 1, t0.elapsed().as_millis() as u64);
        streamed_tokens.push(next_token);

        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[emitted_bytes..];
        let valid_len = match std::str::from_utf8(new_bytes) {
            Ok(_) => new_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_len > 0 {
            let text = std::str::from_utf8(&new_bytes[..valid_len]).unwrap();
            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
            let _ = stdout.flush();
            emitted_bytes += valid_len;
        }

        if next_token == config.eos_token { break; }
        if im_end_token == Some(next_token) { break; }
        if tokenizer.is_terminator(next_token) { break; }

        if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
            loop_guard.check(&streamed_tokens)
        {
            let window_len = loop_guard.window_len(streamed_tokens.len());
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
                id, count, window_len,
            );
            let _ = stdout.flush();
            break;
        }

        qwen35::forward_scratch(gpu, weights, config, next_token, m.seq_pos, kv, dn, scratch).unwrap();
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                m.seq_pos = new_phys;
            }
        }
        logits = gpu.download_f32(&scratch.logits).unwrap();
        llama::apply_ngram_block(&mut logits, &m.conversation_tokens);
        if let Some((open, close)) = think_pair {
            block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
        }

        next_token = sampler::sample_cpu(&mut logits, &m.conversation_tokens, &vl_cfg);

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
                    qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch).unwrap();
                    m.seq_pos += 1;
                    if let Some(ref ev) = m.eviction {
                        if let Some(hipfire_runtime::triattn::EvictionResult { new_physical: new_phys, .. }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap() {
                            m.seq_pos = new_phys;
                        }
                    }
                    m.conversation_tokens.push(t);
                    streamed_tokens.push(t);

                    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes = &all_bytes[emitted_bytes..];
                    let vl = match std::str::from_utf8(new_bytes) {
                        Ok(_) => new_bytes.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    if vl > 0 {
                        let text = std::str::from_utf8(&new_bytes[..vl]).unwrap();
                        let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
                        let _ = stdout.flush();
                        emitted_bytes += vl;
                    }
                    generated += 1;
                }
                think_count = 0;
                prev_in_think = false;
                if generated >= max_tokens { break; }
                logits = gpu.download_f32(&scratch.logits).unwrap();
                if let Some((open, close)) = think_pair {
                    block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
                }
                next_token = sampler::sample_cpu(&mut logits, &m.conversation_tokens, &vl_cfg);
            }
        }
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

/// dots.ocr (arch_id=8) VL generation. Single-image, greedy decode —
/// the phase-3 bring-up serving path that promotes the standalone
/// `ocr_e2e` example into the daemon.
///
/// Flow: preprocess image → `build_prompt_ids` (HF-exact framing) →
/// `vision_forward` → per-token prefill splicing merged visual
/// embeddings at `<|imgpad|>` slots → greedy decode to EOS, streaming
/// tokens in the daemon's JSONL protocol.
///
/// MVP scope: greedy only (sampling params ignored), single image,
/// per-token prefill, `--image <path>` only (base64 deferred). The text
/// side is Qwen2; the decode state reuses `m.qwen2_state`.
fn generate_vl_dots_ocr(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu, stdout: &mut std::io::Stdout, params: &GenerateVLParams) {
    use hipfire_arch_dots_ocr::image as dots_image;
    let t0 = Instant::now();
    let GenerateVLParams { id, prompt, ref image_source, max_tokens, .. } = *params;

    // 1. Preprocess image (CPU; no model borrow yet so error returns are clean).
    let img = match image_source {
        ImageSource::Path(path) => {
            eprintln!("[dots-ocr] preprocessing image: {path}");
            dots_image::preprocess_image(Path::new(path))
        }
        ImageSource::Base64(b64) => {
            // Strip an optional `data:<mime>;base64,` URL prefix.
            let raw_b64 = match b64.strip_prefix("data:") {
                Some(rest) => match rest.split_once(',') {
                    Some((_, after)) => after,
                    None => { write_error(stdout, id, "malformed data URL: missing ',' separator"); return; }
                },
                None => &b64[..],
            };
            eprintln!("[dots-ocr] preprocessing base64 image (<{}-byte payload>)", raw_b64.len());
            match Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64) {
                Ok(bytes) => dots_image::preprocess_image_bytes(&bytes),
                Err(e) => { write_error(stdout, id, &format!("dots.ocr: base64 decode failed: {e}")); return; }
            }
        }
    };
    let img = match img {
        Ok(i) => i,
        Err(e) => { write_error(stdout, id, &format!("dots.ocr image preprocess failed: {e}")); return; }
    };
    let n_visual = img.n_visual_tokens();
    let n_patches = img.n_patches();
    eprintln!("[dots-ocr] grid {}x{}, {} patches → {} visual tokens",
        img.grid_h, img.grid_w, n_patches, n_visual);

    let max_seq = m.max_seq;

    // 2. Model state (disjoint field borrows of `m`).
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = m.dots_ocr_config.as_ref().unwrap();
    let weights = m.dots_ocr_weights.as_ref().unwrap();
    let state = m.qwen2_state.as_mut().unwrap();
    let text_cfg = &config.text;
    let dim = text_cfg.hidden_size;

    // 3. Build the prompt (HF-exact framing; imgpad count == n_visual by construction).
    let prompt_ids = dots_ocr::build_prompt_ids(tokenizer, prompt, n_visual);
    if prompt_ids.len() + max_tokens > max_seq {
        write_error(stdout, id, &format!(
            "dots.ocr request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
            prompt_ids.len(), max_tokens, max_seq));
        return;
    }

    // 4. Vision encoder → merged visual tokens.
    let patch_cols = img.patches.len() / n_patches;
    let patches_gpu = match gpu.upload_f32(&img.patches, &[n_patches, patch_cols]) {
        Ok(t) => t,
        Err(e) => { write_error(stdout, id, &format!("dots.ocr patch upload failed: {e:?}")); return; }
    };
    let merged_gpu = match dots_ocr::vision_forward(gpu, &weights.vision, &config.vision, &patches_gpu, img.grid_h, img.grid_w) {
        Ok(t) => t,
        Err(e) => { let _ = gpu.free_tensor(patches_gpu); write_error(stdout, id, &format!("dots.ocr vision_forward failed: {e:?}")); return; }
    };
    let _ = gpu.free_tensor(patches_gpu);
    let merged = match gpu.download_f32(&merged_gpu) {
        Ok(v) => v,
        Err(e) => { let _ = gpu.free_tensor(merged_gpu); write_error(stdout, id, &format!("dots.ocr merger download failed: {e:?}")); return; }
    };
    let _ = gpu.free_tensor(merged_gpu);
    // Hard guard: merger output count MUST equal the imgpad-slot count, or
    // the splice silently corrupts the text context (PRD §"Vision token splicing").
    if merged.len() != n_visual * dim {
        write_error(stdout, id, &format!(
            "dots.ocr: merger produced {} values but prompt has {} <|imgpad|> slots × {} dims = {}",
            merged.len(), n_visual, dim, n_visual * dim));
        return;
    }

    // 5. Prefill: build the [seq × dim] embedding matrix (token-embedding
    // rows for text positions, spliced vision-merger rows at IMGPAD slots)
    // and run it through the batched prefill in one pass. Only the ~215
    // text positions need a GPU embedding lookup; the 4880 visual rows are
    // already host-resident in `merged`.
    state.reset();
    let t_prefill = Instant::now();
    let mut embeds = vec![0f32; prompt_ids.len() * dim];
    let emb_scratch = match gpu.alloc_tensor(&[dim], rdna_compute::DType::F32) {
        Ok(t) => t,
        Err(e) => { write_error(stdout, id, &format!("dots.ocr embed scratch alloc failed: {e:?}")); return; }
    };
    let mut visual_idx = 0usize;
    let mut embed_err: Option<String> = None;
    for (pos, &token) in prompt_ids.iter().enumerate() {
        if token == dots_ocr::IMGPAD_ID {
            embeds[pos * dim..(pos + 1) * dim]
                .copy_from_slice(&merged[visual_idx * dim..(visual_idx + 1) * dim]);
            visual_idx += 1;
        } else {
            // dots.ocr text weights are Q8_0 (q8.hfq).
            if let Err(e) = gpu.embedding_lookup_q8(&weights.text.token_embd, &emb_scratch, token, dim) {
                embed_err = Some(format!("embedding lookup: {e:?}")); break;
            }
            match gpu.download_f32(&emb_scratch) {
                Ok(row) => embeds[pos * dim..(pos + 1) * dim].copy_from_slice(&row),
                Err(e) => { embed_err = Some(format!("embedding download: {e:?}")); break; }
            }
        }
    }
    let _ = gpu.free_tensor(emb_scratch);
    if let Some(e) = embed_err {
        write_error(stdout, id, &format!("dots.ocr prefill embed build failed: {e}")); return;
    }
    if let Err(e) = qwen2::forward_prefill_batch_embeds(gpu, &weights.text, text_cfg, state, &embeds) {
        write_error(stdout, id, &format!("dots.ocr batched prefill failed: {e:?}")); return;
    }
    let prefill_tokens = prompt_ids.len();
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // 6. Greedy decode, streaming in the daemon JSONL protocol.
    let eos_set: Vec<u32> = if text_cfg.eos_token_ids.is_empty() {
        vec![text_cfg.eos_token_id]
    } else {
        text_cfg.eos_token_ids.clone()
    };
    let mut next = match gpu.argmax_f32(&state.logits, text_cfg.vocab_size) {
        Ok(t) => t,
        Err(e) => { write_error(stdout, id, &format!("dots.ocr argmax failed: {e:?}")); return; }
    };
    let t_gen = Instant::now();
    let mut streamed: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut generated = 0usize;
    // No ngram loop-guard here: dots.ocr layout-JSON legitimately repeats
    // short structures (`<td>…</td>`, `"category":`, bracket patterns), and
    // the default guard force-stops mid-table (observed: truncation at 391
    // tokens on a table-heavy page). The proven ocr_e2e path decodes
    // straight to EOS without a guard; see DotsOcr::loop_guard_overrides.

    while generated < max_tokens {
        if eos_set.contains(&next) { break; }
        emit_committed_event(stdout, id, next, generated, t0.elapsed().as_millis() as u64);
        generated += 1;
        streamed.push(next);

        // Incremental UTF-8 streaming — only emit complete code points.
        let all_bytes = tokenizer.decode_bytes(&streamed);
        let new_bytes = &all_bytes[emitted_bytes..];
        let valid_len = match std::str::from_utf8(new_bytes) {
            Ok(_) => new_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_len > 0 {
            let text = std::str::from_utf8(&new_bytes[..valid_len]).unwrap();
            let _ = writeln!(stdout, r#"{{"type":"token","id":"{}","text":{}}}"#, id, serde_json::to_string(&text).unwrap_or_default());
            let _ = stdout.flush();
            emitted_bytes += valid_len;
        }

        match qwen2::forward_step_greedy(gpu, &weights.text, text_cfg, state, next) {
            Ok(t) => next = t,
            Err(e) => { write_error(stdout, id, &format!("dots.ocr decode failed: {e:?}")); return; }
        }
    }

    let decode_s = t_gen.elapsed().as_secs_f64();
    let total_s = t0.elapsed().as_secs_f64();
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
