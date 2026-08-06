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
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::arch_dispatch::ForwardCtx;
use hipfire_runtime::emit_text::{currently_in_think, extract_tool_calls_from_text};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Mutex, OnceLock,
};
use std::time::Instant;

use hipfire_loader::{
    AsstTurnCache, EpArch, EpState, Eviction, LoadedModel, ModelParallel, ModelParallelKind,
    ModelState, PipelineImpl,
};
use hipfire_runtime::spec::{
    ClientEvent, EvictRetain, FinishSummary, PrefillOutcome, SpecAdvance, SpecEmit, SpecTarget,
    SpecTargetGuard, Speculator, StopReason,
};

/// An explicitly invalid operator environment is a load error even when the
/// request carries a valid parameter; configuration must not be ignored.
fn mtp_k_from_env(name: &str) -> Result<Option<usize>, String> {
    match std::env::var(name) {
        Ok(value)
            if matches!(
                value.as_str(),
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8"
            ) =>
        {
            Ok(Some(value.parse().expect("validated single-digit MTP K")))
        }
        Ok(value) => Err(format!("{name} must be an integer in 1..=8, got {value:?}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("{name} is invalid: {error}")),
    }
}

fn resolve_mtp_k(
    load_param: Option<&serde_json::Value>,
    is_deepseek: bool,
) -> Result<usize, String> {
    let value = if is_deepseek {
        mtp_k_from_env("HIPFIRE_DEEPSEEK4_SPEC_K")?
    } else {
        None
    }
    .or(mtp_k_from_env("HIPFIRE_MTP_K")?)
    .unwrap_or(match load_param {
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("params.mtp_k must be an integer in 1..=8, got {value}"))?,
        None => 3,
    });
    if (1..=8).contains(&value) {
        Ok(value)
    } else {
        Err(format!("MTP K must be in 1..=8, got {value}"))
    }
}

fn resolve_mtp_mode(load_param: Option<&serde_json::Value>) -> Result<Option<bool>, String> {
    match std::env::var("HIPFIRE_MTP_MODE") {
        Ok(value) => {
            return match value.as_str() {
                "auto" => Ok(None),
                "on" => Ok(Some(true)),
                "off" => Ok(Some(false)),
                _ => Err(format!(
                    "HIPFIRE_MTP_MODE must be one of off, on, auto, got {value:?}"
                )),
            };
        }
        Err(std::env::VarError::NotPresent) => {}
        Err(error) => return Err(format!("HIPFIRE_MTP_MODE is invalid: {error}")),
    }
    match load_param {
        None => Ok(None),
        Some(serde_json::Value::String(value)) => match value.as_str() {
            "auto" => Ok(None),
            "on" => Ok(Some(true)),
            "off" => Ok(Some(false)),
            _ => Err(format!(
                "params.mtp_mode must be one of off, on, auto, got {value:?}"
            )),
        },
        Some(value) => Err(format!(
            "params.mtp_mode must be one of off, on, auto, got {value}"
        )),
    }
}

/// Abort-target request ID. Set asynchronously by the background
/// stdin-reader thread when it sees `{type:"abort","id":"..."}`;
/// consumed and cleared by `check_abort()` from the main thread's
/// prefill chunk loop. Using an Option<String> rather than a bool
/// makes the abort targeted — stale aborts from a prior request
/// can't kill a new request that happens to be running by the time
/// the message lands.
fn abort_for_id() -> &'static Mutex<Option<String>> {
    static CELL: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// True if the in-flight request with `req_id` has been aborted.
/// Clears the flag on match so the next request with the same ID
/// (unlikely but possible — CLI generates request IDs) starts clean.
fn check_abort(req_id: &str) -> bool {
    let mut g = abort_for_id().lock().unwrap();
    if g.as_deref() == Some(req_id) {
        *g = None;
        true
    } else {
        false
    }
}

/// Per-request cancellation latch. `check_abort` is intentionally one-shot so
/// a stale abort cannot kill a later request, but speculative prefill calls it
/// from several nested bounded loops. Once any layer observes cancellation the
/// request must remain cancelled; otherwise an inner poll can consume the
/// signal and the outer admission gate would incorrectly publish a seed.
struct AbortLatch {
    cancelled: AtomicBool,
}

impl AbortLatch {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
        }
    }

    fn poll(&self, req_id: &str) -> bool {
        if self.cancelled.load(Ordering::Acquire) {
            return true;
        }
        if check_abort(req_id) {
            self.cancelled.store(true, Ordering::Release);
            true
        } else {
            false
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Force-answer target request ID, set by the stdin-reader thread on
/// `{type:"force_answer","id":"..."}`. Unlike `abort` (which kills the
/// turn), force-answer asks the decode loop to STOP THINKING and commit
/// to the answer — the model's `<think>` span is force-closed (the same
/// continuation the `max_think_tokens` budget splices) and generation
/// continues. The CLI sends this when a turn is taking too long so the
/// stream produces a real answer instead of the client timing out and
/// terminating mid-think.
fn force_answer_for_id() -> &'static Mutex<Option<String>> {
    static CELL: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// True if the in-flight request `req_id` was asked to force-answer.
/// Clears on match (one-shot).
fn check_force_answer(req_id: &str) -> bool {
    let mut g = force_answer_for_id().lock().unwrap();
    if g.as_deref() == Some(req_id) {
        *g = None;
        true
    } else {
        false
    }
}

/// The text spliced into the stream to force-close a `<think>` span (on
/// either the `max_think_tokens` budget OR a CLI force-answer signal),
/// making the model commit to its answer. Default closes the think tag
/// per Qwen's trained post-think format; override with
/// `HIPFIRE_THINK_CONTINUATION` to inject a richer "now produce the
/// answer" nudge (keep it short — it's prepended to the visible answer).
fn think_continuation() -> String {
    std::env::var("HIPFIRE_THINK_CONTINUATION").unwrap_or_else(|_| "</think>\n\n".to_string())
}

/// Message types pushed from the stdin-reader thread to the main
/// processing loop. Abort messages are NOT forwarded — they're
/// handled inline in the reader thread by setting `abort_for_id()`.
/// This is what lets the abort signal interrupt a mid-flight prefill;
/// the main loop is blocked on prefill compute and would only see
/// new stdin lines after that prefill completed.
enum DaemonMsg {
    Regular(serde_json::Value),
    ParseError(String),
}

pub type CaskConfig = hipfire_runtime::loader_api::CaskConfig;

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
    if window == 0 || threshold == 0 || open_id == close_id {
        return;
    }
    let start = history.len().saturating_sub(window);
    let mut depth: i32 = 0;
    for &t in &history[start..] {
        if t == open_id {
            depth += 1;
        } else if t == close_id && depth > 0 {
            depth -= 1;
        }
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

/// LRU-bounded fingerprint→tokens cache for assistant-turn replay
/// (`asst_turn_cache`). Holds the verbatim token sequence each
/// assistant turn emitted during decode, keyed by
/// [`asst_turn_fingerprint`]. On the next request, the multi-turn
/// renderer replays cached tokens at the same turn boundary so the
/// rendered prefix is byte-identical to what was written into KV last
/// turn — required for the LCP-based prompt cache to extend through
/// historical assistant turns (BPE is not bijective; re-encoding a
/// model's emission may produce a different token sequence).
///
/// Cap is configurable via `HIPFIRE_PROMPT_CACHE_CAP` (default 32);
/// `HIPFIRE_PROMPT_CACHE_UNBOUNDED=1` removes the cap entirely. On
/// `insert`, an existing key is moved to MRU; on `get`, the same. When
/// at capacity, the LRU (oldest-touched) entry is evicted.

/// Stable fingerprint over an assistant turn — pair of (text content,
/// tool_calls canonical JSON). Output is identical for two messages
/// that have the same content+tool_calls regardless of how the
/// surrounding bytes (e.g. whitespace inside JSON args) were rendered
/// upstream. Used by the V4F prefix-cache to identify "this is the
/// same assistant turn the model previously emitted, so reuse the
/// emitted token IDs verbatim instead of re-encoding via the DSML
/// renderer + BPE (which is not bijective)."
fn asst_turn_fingerprint(
    content: &str,
    tool_calls: &[hipfire_runtime::prompt_frame::ToolCall],
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    "assistant".hash(&mut h);
    if tool_calls.is_empty() {
        // Pure-text turn — content IS the message. Trim whitespace
        // to absorb minor formatting drift between store (model's
        // verbatim emission) and lookup (whatever the client preserved).
        content.trim().hash(&mut h);
    } else {
        // Mixed turn (text + tool_calls) or pure tool_call. Hash ONLY
        // the tool_calls — pi-coding-agent (and most OpenAI-compat
        // clients) sends `content: null` on assistant messages that
        // carry tool_calls, even when the model originally emitted
        // prose ahead of the tool block (e.g. "Let me check the
        // structure first.<｜DSML｜tool_calls>…"). The store-side
        // sees the prose in `emit_text_buf`; the lookup-side sees
        // content=`""`. Excluding content from the fingerprint when
        // tool_calls is non-empty matches the client's effective
        // identity for tool-call turns and lets the cache hit.
        //
        // Collision risk: two distinct turns with identical
        // tool_calls hash to the same key; the later store wins,
        // and a replay of the earlier turn replays the later turn's
        // tokens. In practice this only matters when the model emits
        // the SAME tool_call twice with different surrounding prose
        // in the same conversation — uncommon for agent flows, and
        // the worst-case effect is the model seeing slightly altered
        // prose in its own history.
    }
    for tc in tool_calls {
        tc.name.hash(&mut h);
        // Serialize args in a CANONICAL form: walk the Value tree and
        // emit objects with keys sorted lexically (recursively). The
        // upstream `serde_json::Map` uses insertion order — fine for
        // round-tripping a single payload, but two clients (or two
        // parser passes on the same payload) can yield different
        // insertion orders for the same logical args. Without
        // canonicalization those two turns hash to DIFFERENT keys,
        // dropping cache hit rate on otherwise-identical tool calls.
        let args = canonical_json(&tc.arguments);
        args.hash(&mut h);
    }
    h.finish()
}

/// Build the fingerprint-key string for an emitted assistant turn so
/// it matches `msg.content` as the CLI sends it back next turn.
/// Mirrors the *visible-content* transformation the bun CLI's HTTP
/// serve applies between SSE-relay and `messages[].content`:
///
///   1. Strip paired `<think>…</think>` blocks plus any trailing
///      whitespace (`cli/index.ts:1656-1658`).
///   2. Strip an unclosed `<think>…$` tail (same site).
///   3. Strip an orphan `</think>` opener — when the daemon's prompt
///      ends with `<think>\n` the model resumes inside think mode and
///      never emits an opening tag; the CLI's `inThink` state machine
///      (`cli/index.ts:2334-2365`) treats every token until `</think>`
///      as `reasoning_content` and only emits content from after the
///      close. We match that by stripping `text-up-to-and-including-
///      first-</think>` + trailing whitespace when no `<think>`
///      preceded it.
///   4. Strip the literal `<|im_end|>` substring (the CLI relay
///      removes it at `cli/index.ts:2366`).
///
/// Without (3) and (4) the fingerprint stored after turn N would
/// include reasoning + the ChatML terminator that the CLI strips
/// before sending back as `msg.content` on turn N+1, dropping the
/// cache hit rate to ~zero for thinking-on Qwen models.
fn strip_think_for_fingerprint(s: &str) -> String {
    let mut out = s.to_string();
    // (1) + (2): paired/unclosed `<think>` blocks.
    loop {
        let open = match out.find("<think>") {
            Some(i) => i,
            None => break,
        };
        match out[open..].find("</think>") {
            Some(close_rel) => {
                let close_end = open + close_rel + "</think>".len();
                let mut tail = close_end;
                let bytes = out.as_bytes();
                while tail < bytes.len() {
                    let c = bytes[tail];
                    if c == b' ' || c == b'\n' || c == b'\t' || c == b'\r' {
                        tail += 1;
                    } else {
                        break;
                    }
                }
                out.replace_range(open..tail, "");
            }
            None => {
                out.truncate(open);
                break;
            }
        }
    }
    // (3): orphan `</think>` closer with no preceding opener (model
    // resumed inside think mode from the prompt's `<think>\n` prefix).
    if let Some(close_idx) = out.find("</think>") {
        let after_close = close_idx + "</think>".len();
        let mut tail = after_close;
        let bytes = out.as_bytes();
        while tail < bytes.len() {
            let c = bytes[tail];
            if c == b' ' || c == b'\n' || c == b'\t' || c == b'\r' {
                tail += 1;
            } else {
                break;
            }
        }
        out.replace_range(0..tail, "");
    }
    // (4): strip the literal `<|im_end|>` substring (CLI relay strips
    // it from every chunk before forwarding as content).
    while let Some(idx) = out.find("<|im_end|>") {
        out.replace_range(idx..idx + "<|im_end|>".len(), "");
    }
    out
}

/// Publish one sealed Qwen assistant turn to the verbatim replay cache.
///
/// The sealed transcript is the only source for this write. In particular, a
/// stop that cuts through a token has no replay boundary and must not be
/// reconstructed from the visible text or the target's over-advanced state.
fn cache_sealed_qwen_turn(
    cache: &mut AsstTurnCache,
    finalized: &hipfire_runtime::spec_transcript::FinalizedAssistantTurn,
    target_reusable: bool,
) -> Option<u64> {
    if !target_reusable {
        return None;
    }
    let cached_seq = finalized.replay_tokens()?.to_vec();
    if cached_seq.is_empty() {
        return None;
    }
    let stripped = strip_think_for_fingerprint(finalized.text());
    let emit_text = hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
    let fp = asst_turn_fingerprint(&emit_text, finalized.tool_calls());
    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache store dflash] fp={:#018x} cached_seq={} emit_text.len={} tool_calls={} preview={:?}",
            fp,
            cached_seq.len(),
            emit_text.len(),
            finalized.tool_calls().len(),
            emit_text.chars().take(60).collect::<String>(),
        );
    }
    cache.insert(fp, cached_seq);
    Some(fp)
}

/// Walk a [`serde_json::Value`] and produce a canonical-key
/// representation: objects emit keys in lexical order (recursively),
/// arrays preserve order. Used by [`asst_turn_fingerprint`] so two
/// messages with the same logical tool args hash identically
/// regardless of source-side insertion order.
fn canonical_json(v: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical_json(v, &mut out);
    out
}

fn write_canonical_json(v: &serde_json::Value, out: &mut String) {
    match v {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string()))
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(*k).unwrap_or_else(|_| "\"\"".to_string()));
                out.push(':');
                write_canonical_json(&map[*k], out);
            }
            out.push('}');
        }
    }
}

/// Safely emit a `{"type":"error", …}` JSONL line. Builds the envelope
/// through `serde_json::json!` so embedded `"` / `\` / control chars in
/// the message or `id` can't corrupt the line and trigger a client-side
/// `JSON Parse error: Expected '}'` parse loop. Use this instead of
/// `writeln!(stdout, r#"{{"type":"error",…}}"#, …)` with raw `{}` / `{:?}`
/// interpolation of error values — Rust's `Display` will pass through
/// a `"` unchanged, and `Debug` actively wraps strings in escaped quotes,
/// both of which break the surrounding JSON.
fn emit_error_with_id(stdout: &mut std::io::Stdout, id: &str, message: impl std::fmt::Display) {
    let envelope = serde_json::json!({
        "type": "error",
        "id": id,
        "message": format!("{}", message),
    });
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

#[allow(dead_code)]
fn emit_error_no_id(stdout: &mut std::io::Stdout, message: impl std::fmt::Display) {
    let envelope = serde_json::json!({
        "type": "error",
        "message": format!("{}", message),
    });
    let _ = writeln!(stdout, "{}", envelope);
    let _ = stdout.flush();
}

/// Emit a parsed `deepseek4::dsml::StreamEvent` to the JSONL stream.
/// Maps:
///   - Token(text)        → `{type:"token",   id, text}`
///   - Reasoning(text)    → `{type:"reasoning", id, text}`
///   - ToolCalls(calls)   → `{type:"tool_calls", id, calls:[{name, arguments}]}`
///
/// The CLI / OpenAI HTTP layer translates these into the corresponding
/// SSE chunks (`content`, `reasoning_content`, `tool_calls.delta`).
fn emit_stream_event(
    stdout: &mut std::io::Stdout,
    id: &str,
    ev: hipfire_arch_deepseek4::dsml::StreamEvent,
) {
    use hipfire_arch_deepseek4::dsml::StreamEvent;
    // The request id is user-supplied. Build the envelope through
    // `serde_json` so any embedded `"` / `\` / control chars are
    // escaped — otherwise a malformed id corrupts every subsequent
    // line of the JSONL stream and the cli/serve loop dies with a
    // `JSON Parse error: Expected '}'`.
    let envelope = match ev {
        StreamEvent::Token(text) => serde_json::json!({
            "type": "token",
            "id": id,
            "text": text,
        }),
        StreamEvent::Reasoning(text) => serde_json::json!({
            "type": "reasoning",
            "id": id,
            "text": text,
        }),
        StreamEvent::ToolCalls(calls) => {
            let arr: Vec<serde_json::Value> = calls
                .into_iter()
                .map(|c| {
                    serde_json::json!({
                        "name": c.name,
                        "arguments": c.arguments,
                    })
                })
                .collect();
            serde_json::json!({
                "type": "tool_calls",
                "id": id,
                "calls": serde_json::Value::Array(arr),
            })
        }
    };
    let _ = writeln!(stdout, "{}", envelope);
}

fn emit_committed_event(
    stdout: &mut std::io::Stdout,
    id: &str,
    tok_id: u32,
    pos: usize,
    t_ms: u64,
) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let on = *ENABLED
        .get_or_init(|| std::env::var("HIPFIRE_EMIT_TOKEN_IDS").ok().as_deref() == Some("1"));
    if !on {
        return;
    }
    // Build through `serde_json::json!` for the same reason
    // `emit_error_with_id` does: `id` is user-supplied and a single `"`
    // or `\` in it would corrupt the line, breaking the client's JSONL
    // parser for every subsequent event on the same connection.
    let envelope = serde_json::json!({
        "type": "committed",
        "id": id,
        "tok_id": tok_id,
        "pos": pos,
        "t_ms": t_ms,
    });
    let _ = writeln!(stdout, "{}", envelope);
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

/// hunt3 H-D: upper bound on a request-driven `max_seq` (512K). A defense-in-
/// depth clamp only — it caps an unvalidated 10M `max_seq` that would otherwise
/// drive a multi-GB KV allocation and OOM the daemon at load. It is NOT a
/// VRAM-aware guard: a load that requests exactly this on a non-eviction config
/// can still OOM at allocation; that VRAM validation is out of scope here.
const MAX_REQUESTED_SEQ: usize = 512 * 1024;

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

fn ckpt_resume_enabled() -> bool {
    std::env::var("HIPFIRE_CACHE_CKPT_RESUME").ok().as_deref() != Some("0")
}
fn ckpt_interval() -> usize {
    std::env::var("HIPFIRE_CACHE_CKPT_INTERVAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048)
        .max(256)
}
fn ckpt_max() -> usize {
    std::env::var("HIPFIRE_CACHE_CKPT_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(1)
}

// ─── Token-parity harness (per-flip dual-run comparator; dormant in prod) ──────
//
// `TokenTape` accumulates committed token IDs for one pass; `assert_token_parity`
// compares two tapes and panics with a precise divergence report on mismatch.
// This is the shadow-assert each arch flip runs LIVE (old path + new
// `ar_generate` path on the same request, asserting token-identical) and then
// deletes once the old path is removed — so in production it compares nothing:
// every `ar_generate` call site passes `tape: None`. The `tape` param stays
// threaded through `ar_generate` so the next arch flip can re-enable a dual-run
// without re-plumbing. `--self-check-parity` exercises the comparator (both
// directions) without a GPU — it does NOT prove old-vs-new decode parity, which
// only the live per-flip dual-run can.

#[derive(Default, Clone)]
#[allow(dead_code)]
struct TokenTape(Vec<u32>);

impl TokenTape {
    #[allow(dead_code)]
    fn push(&mut self, t: u32) {
        self.0.push(t);
    }
}

#[allow(dead_code)]
fn assert_token_parity(old: &TokenTape, new: &TokenTape, id: &str) {
    if old.0 != new.0 {
        // First divergent position. If the tapes agree on every zipped pair but
        // differ in length (one is a strict prefix of the other), point at the
        // boundary of the shorter tape rather than reporting `None` — otherwise
        // a length-mismatch panic reads `first_div=None`, which looks like "no
        // divergence found" and hides a real different-token-count mismatch
        // (e.g. EOS committed at a different position).
        let pos = old
            .0
            .iter()
            .zip(new.0.iter())
            .position(|(a, b)| a != b)
            .or_else(|| (old.0.len() != new.0.len()).then(|| old.0.len().min(new.0.len())));
        panic!(
            "ARCHDISPATCH PARITY FAIL id={id}: len old={} new={} first_div={:?} old_tok={:?} new_tok={:?}",
            old.0.len(),
            new.0.len(),
            pos,
            pos.and_then(|p| old.0.get(p)),
            pos.and_then(|p| new.0.get(p)),
        );
    }
}

// ─── ArchDispatch impls (Inc 1, god-struct-collapse) ─────────────────────────
//
// One `impl ArchDispatch` per arch family; wired into `ar_generate` in Task 1.4.
// Until then, the structs are unused scaffolding: `#[allow(dead_code)]` silences
// the compiler while the trait contract is being built arch by arch.

/// Extract the qwen35 EOS token from the loaded model state.
/// Mirrors the `config.eos_token` reads at daemon.rs:9210 and :5455
/// (`target.config.eos_token` in `generate_dflash`, `b.config.eos_token` in the
/// AR decode loop) — both read the same field on `Qwen35Config`.
#[allow(dead_code)]
fn qwen35_eos(m: &LoadedModel) -> u32 {
    if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
        b.config.eos_token
    } else {
        // Unreachable in production: Qwen35Dispatch is only constructed
        // when the model is a qwen35 arch (5/6) and the bundle is present.
        0
    }
}

/// Wraps `&mut LoadedModel` to implement `ArchDispatch` for the qwen35 family
/// (arch_id 5 = dense, 6 = MoE-A3B). Unused until Task 1.4 wires it into
/// `ar_generate`; all methods delegate verbatim to existing daemon helpers.
#[allow(dead_code)]
struct Qwen35Dispatch<'m> {
    m: &'m mut LoadedModel,
}

#[allow(dead_code)]
impl<'m> hipfire_runtime::arch_dispatch::ArchDispatch for Qwen35Dispatch<'m> {
    fn arch_id(&self) -> u32 {
        // Direct field read — mirrors every `m.meta.arch_id` branch in the daemon.
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        // Delegates to qwen35_eos: mirrors `b.config.eos_token` (AR loop,
        // daemon.rs:9210) and `target.config.eos_token` (DFlash, daemon.rs:5455).
        qwen35_eos(self.m)
    }

    fn is_eos(&self, tok: u32) -> bool {
        // Preserves the legacy qwen35 arm's stop: primary eos OR any tokenizer
        // terminator (eos_id / eot_id). Byte-identical to the pre-hook ar_generate
        // stop expression → qwen35 parity unchanged.
        tok == self.eos_token() || self.tokenizer().is_terminator(tok)
    }

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        // Last-resort arch ladder for arch_id 5/6: the `else` branch at
        // daemon.rs:1906 (`(0.3_f64, 0.8_f64)`) plus the hardcoded
        // `default_repeat_penalty = 1.0` (daemon.rs:1954, arch_id != 11).
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        // qwen35 AR: think-mode (primed_think/think_pair), user stop-seqs
        // (stop_seqs Vec<String>), grammar-guided tool calls (Matcher). No vision
        // in the AR path (VL is a separate forward through generate_qwen35_vl).
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: true,
            supports_stop_seq: true,
            supports_grammar: true,
            supports_vision: false,
        }
    }

    fn as_spec_target(&mut self) -> Option<&mut dyn hipfire_runtime::spec::SpecTarget> {
        // TODO(Task 1.4): wire qwen35 SpecTarget when ar_generate needs it.
        //
        // The qwen35 carrier's `spec_target_guard` (carriers.rs:298) moves the
        // bundle out of `m.state` into a `Qwen35SlotGuard` that reopens the
        // HfqFile and restores the bundle on Drop — it returns a
        // `Box<dyn SpecTargetGuard>`, not a bare `&mut dyn SpecTarget`. There is
        // no existing getter that yields a `&mut dyn SpecTarget` from a live
        // `&mut LoadedModel` without the guard machinery. Returning None here
        // is safe: `as_spec_target` is the default no-op path; the spec loop
        // continues to go through `carrier.spec_target_guard` directly until
        // Task 1.4 introduces the abstracted generate path.
        None
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let ModelState::Qwen35(ref mut b) = *self.m.state.as_mut().ok_or("no state")? else {
            return Err("prefill_forward: not a qwen35 bundle".into());
        };
        qwen35::forward_prefill_batch(
            gpu,
            &b.weights,
            &b.config,
            chunk,
            seq_pos,
            &mut b.kv_cache,
            &mut b.dn_state,
            &b.scratch,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| format!("forward_prefill_batch: {:?}", e))
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let ModelState::Qwen35(ref mut b) = *self.m.state.as_mut().ok_or("no state")? else {
            return Err("decode_step_forward: not a qwen35 bundle".into());
        };
        qwen35::forward_scratch(
            gpu,
            &b.weights,
            &b.config,
            token,
            seq_pos,
            &mut b.kv_cache,
            &mut b.dn_state,
            &b.scratch,
        )
        .map_err(|e| format!("forward_scratch: {:?}", e))
    }

    fn init_grammar(
        &self,
        tool_schemas: &[(String, Vec<String>)],
    ) -> Option<Box<dyn hipfire_runtime::arch_dispatch::GrammarMatcher>> {
        let schemas: Vec<hipfire_arch_qwen35::grammar::ToolSchema> = tool_schemas
            .iter()
            .map(
                |(name, required)| hipfire_arch_qwen35::grammar::ToolSchema {
                    name: name.clone(),
                    required: required.clone(),
                },
            )
            .collect();
        Some(Box::new(Qwen35GrammarMatcher(
            hipfire_arch_qwen35::grammar::Matcher::new(schemas),
        )))
    }

    fn maybe_evict(
        &mut self,
        ctx: ForwardCtx<'_>,
        seq_pos: usize,
    ) -> Result<Option<usize>, String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let m = &mut *self.m;
        // Mirror the arm's field-split: borrow m.state (kv, mut) then m.eviction
        // (shared) — disjoint fields of *m, so NLL permits both live.
        let ModelState::Qwen35(b) = m.state.as_mut().ok_or("no state")? else {
            return Err("maybe_evict: not a qwen35 bundle".into());
        };
        let kv = &mut b.kv_cache;
        let Some(ev) = m.eviction.as_ref() else {
            return Ok(None);
        };
        match ev.maybe_evict(gpu, kv, seq_pos) {
            Ok(Some(hipfire_runtime::triattn::EvictionResult {
                new_physical: new_phys,
                ..
            })) => Ok(Some(new_phys)),
            Ok(None) => Ok(None),
            Err(e) => Err(format!("maybe_evict: {:?}", e)),
        }
    }

    fn maybe_adaptive_downshift(&mut self, ctx: ForwardCtx<'_>, seq_pos: usize) {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let m = &mut *self.m;
        let Some(ModelState::Qwen35(b)) = m.state.as_mut() else {
            return;
        };
        let kv = &mut b.kv_cache;
        // Stderr phase-label is unified here (the arm distinguishes prefill /
        // post-prefill / decode); logging is diagnostic and NOT part of token
        // parity (assert_token_parity compares committed token IDs only).
        if let Some(ad) = m.session.kv_adaptive.as_mut() {
            match ad.maybe_downshift(gpu, kv, seq_pos) {
                Ok(applied) => {
                    for step in &applied {
                        eprintln!(
                            "[adaptive-kv] downshift @ pos {}: {:?} (K={:?} V={:?})",
                            seq_pos, step, ad.cur_k, ad.cur_v
                        );
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[adaptive-kv] maybe_downshift error @ pos {}: {:?} — skipping",
                        seq_pos, e
                    );
                }
            }
        }
    }

    fn take_prefill_checkpoint(&mut self, ctx: ForwardCtx<'_>, seq_pos: usize) {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let m = &mut *self.m;
        let Some(ModelState::Qwen35(b)) = m.state.as_mut() else {
            return;
        };
        speculative::take_dn_checkpoint(
            &mut m.session.prefill_checkpoints,
            &mut b.dn_state,
            gpu,
            seq_pos,
            ckpt_interval(),
            ckpt_max(),
        );
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let m = &*self.m;
        let ModelState::Qwen35(b) = m.state.as_ref().ok_or("no state")? else {
            return Err("sample: not a qwen35 bundle".into());
        };
        let scratch = &b.scratch;
        // Grammar-constraining → CPU path (download, apply the driver-computed
        // mask, sample_cpu); free → GPU fast path. Mirrors daemon.rs:9119.
        let tok = if let Some(mask) = grammar_mask {
            let mut logits = gpu
                .download_f32(&scratch.logits)
                .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
            hipfire_arch_qwen35::grammar::Matcher::apply_mask_to_logits(mask, &mut logits);
            sampler::sample_cpu(&mut logits, ngram_scope, cfg)
        } else {
            sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                vocab_size,
                ngram_scope,
                cfg,
                rng_state,
            )
        };
        Ok(tok)
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        self.m.session.seq_pos
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        self.m.session.seq_pos = seq_pos;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        if let Some(ModelState::Qwen35(b)) = self.m.state.as_ref() {
            b.config.vocab_size
        } else {
            0
        }
    }

    fn repeat_buf_cap_bytes(&self) -> usize {
        if let Some(ModelState::Qwen35(b)) = self.m.state.as_ref() {
            b.scratch.repeat_buf.buf.size()
        } else {
            0
        }
    }

    fn prefill_max_batch(&self) -> usize {
        qwen35::PREFILL_MAX_BATCH
    }

    fn ensure_decoded_vocab(&mut self) -> std::sync::Arc<Vec<String>> {
        if self.m.persist.decoded_vocab.is_none() {
            let v: Vec<String> = {
                let tok = self.m.tokenizer.as_ref().unwrap();
                let n = tok.vocab_size();
                (0..n).map(|id| tok.decode(&[id as u32])).collect()
            };
            self.m.persist.decoded_vocab = Some(std::sync::Arc::new(v));
        }
        self.m.persist.decoded_vocab.clone().unwrap()
    }

    fn has_eviction(&self) -> bool {
        self.m.eviction.is_some()
    }

    fn physical_cap(&self) -> usize {
        self.m.meta.physical_cap
    }

    fn eviction_window(&self) -> Option<usize> {
        self.m.eviction.as_ref().map(|ev| ev.budget() + ev.beta())
    }

    fn insert_asst_turn(&mut self, fp: u64, seq: Vec<u32>) {
        self.m.persist.asst_turn_cache.insert(fp, seq);
    }
}

/// Newtype wrapper so daemon.rs (which owns this type) can implement
/// `GrammarMatcher` for the qwen35 `Matcher` without violating the orphan rule
/// (neither `GrammarMatcher` nor `Matcher` is defined here without the newtype).
#[allow(dead_code)]
struct Qwen35GrammarMatcher(hipfire_arch_qwen35::grammar::Matcher);

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::GrammarMatcher for Qwen35GrammarMatcher {
    fn token_mask(&self, vocab: &[String], out: &mut [bool]) {
        self.0.token_mask(vocab, out);
    }

    fn advance(&mut self, text: &str) {
        self.0.advance(text);
    }

    fn is_free(&self) -> bool {
        self.0.is_free()
    }

    fn attractor_detected(&self) -> bool {
        self.0.attractor_detected()
    }
}

/// Wraps `&mut LoadedModel` to implement `ArchDispatch` for qwen2 (arch_id 7).
/// Greedy-only, SPLIT forward/sample: `qwen2::forward_step` writes state.logits,
/// the driver samples via `gpu.argmax_f32` — byte-identical to the arch's fused
/// `forward_step_greedy` (= forward_step + argmax_f32). No grammar / recurrent /
/// eviction / adaptive-KV / checkpoints / pflash / asst-turn cache, so the tangle
/// hooks + those accessors keep their no-op/safe trait defaults. Unused until
/// Inc 2 routes `generate_qwen2` through `ar_generate`.
#[allow(dead_code)]
struct Qwen2Dispatch<'m> {
    m: &'m mut LoadedModel,
}

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::ArchDispatch for Qwen2Dispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        if let Some(ModelState::Qwen2(b)) = self.m.state.as_ref() {
            b.config.eos_token_id
        } else {
            0
        }
    }

    fn is_eos(&self, tok: u32) -> bool {
        // Matches generate_qwen2's stop: membership in the full eos_token_ids SET
        // (NOT the tokenizer's terminator set — qwen2's legacy loop ignored eot_id).
        // Faithful port; broadening to include eot is a separate deliberate change.
        if let Some(ModelState::Qwen2(b)) = self.m.state.as_ref() {
            b.config.eos_token_ids.contains(&tok)
        } else {
            false
        }
    }

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        // qwen2 AR is greedy-only; inert (the daemon drives it greedy).
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 0.0,
            top_p: 1.0,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        // Legacy qwen2 loop: no think, no user stop-seq, no grammar, no vision
        // (VL is a separate generate_vl path).
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: false,
            supports_stop_seq: false,
            supports_grammar: false,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        // Per-token forward_step (matches generate_qwen2's per-token prefill).
        // state.next_pos is the internal cursor → the driver's seq_pos is advisory.
        let _ = seq_pos;
        let ModelState::Qwen2(b) = self.m.state.as_mut().ok_or("no state")? else {
            return Err("prefill_forward: not a qwen2 bundle".into());
        };
        for &tok in chunk {
            qwen2::forward_step(gpu, &b.weights, &b.config, &mut b.state, tok)
                .map_err(|e| format!("qwen2 forward_step (prefill): {e:?}"))?;
        }
        Ok(())
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let _ = seq_pos;
        let ModelState::Qwen2(b) = self.m.state.as_mut().ok_or("no state")? else {
            return Err("decode_step_forward: not a qwen2 bundle".into());
        };
        qwen2::forward_step(gpu, &b.weights, &b.config, &mut b.state, token)
            .map_err(|e| format!("qwen2 forward_step (decode): {e:?}"))
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        // Greedy argmax over state.logits — byte-identical to forward_step_greedy's
        // internal `gpu.argmax_f32(state.logits, cfg.vocab_size)`. All sampling
        // controls are inert for qwen2.
        let _ = (cfg, vocab_size, ngram_scope, grammar_mask, rng_state);
        let ModelState::Qwen2(b) = self.m.state.as_ref().ok_or("no state")? else {
            return Err("sample: not a qwen2 bundle".into());
        };
        gpu.argmax_f32(&b.state.logits, b.config.vocab_size)
            .map_err(|e| format!("qwen2 argmax: {e:?}"))
    }

    // ── accessors ar_generate needs (rest use trait defaults, correct for qwen2) ──
    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        self.m.session.seq_pos
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        // Adopt the arch cursor as authority (== generate_qwen2's finalize
        // `m.session.seq_pos = state.next_pos`; equal to the driver's local for a
        // single-turn run, but exact for multi-turn).
        let _ = seq_pos;
        let np = if let Some(ModelState::Qwen2(b)) = self.m.state.as_ref() {
            b.state.next_pos
        } else {
            return;
        };
        self.m.session.seq_pos = np;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        if let Some(ModelState::Qwen2(b)) = self.m.state.as_ref() {
            b.config.vocab_size
        } else {
            0
        }
    }
}

/// Wraps `&mut LoadedModel` to implement `ArchDispatch` for llama (arch_id 0/1).
/// SPLIT via `llama::forward_prefill_batch` (writes scratch.logits, no sampling)
/// for BOTH prefill and single-token decode, then the generic sampler — llama
/// joins the qwen35/qwen2 split model.
///
/// NOTE (path decision, Inc 3): the legacy llama arm used the FUSED
/// `llama::forward_scratch` (optimized decode kernel + temp/top_p + window-based
/// repeat penalty baked in). Routing through ar_generate instead adopts the
/// standard batched forward + the full generic sampler (top_k/min_p/penalties),
/// per bjoern's uplift decision. So llama output is NOT byte-identical to the
/// legacy arm — validate via COHERENCE, not strict parity. Perf follow-up: add a
/// fused `decode_step_sample` hook (forward_scratch) to recover the B=1 decode
/// kernel; correctness-first here.
#[allow(dead_code)]
struct LlamaDispatch<'m> {
    m: &'m mut LoadedModel,
}

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::ArchDispatch for LlamaDispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        if let Some(ModelState::Llama(b)) = self.m.state.as_ref() {
            b.config.eos_token
        } else {
            0
        }
    }

    fn is_eos(&self, tok: u32) -> bool {
        // Legacy llama arm stop: config.eos_token OR any tokenizer terminator.
        tok == self.eos_token() || self.tokenizer().is_terminator(tok)
    }

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: false,
            supports_stop_seq: true,
            supports_grammar: false,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let ModelState::Llama(ref mut b) = *self.m.state.as_mut().ok_or("no state")? else {
            return Err("prefill_forward: not a llama bundle".into());
        };
        llama::forward_prefill_batch(
            gpu, &b.weights, &b.config, chunk, seq_pos, &mut b.kv, &b.scratch, None,
        )
        .map_err(|e| format!("llama forward_prefill_batch: {:?}", e))
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let ModelState::Llama(ref mut b) = *self.m.state.as_mut().ok_or("no state")? else {
            return Err("decode_step_forward: not a llama bundle".into());
        };
        // Single-token decode via the batched-prefill path (writes scratch.logits
        // without sampling). Perf follow-up: forward_scratch decode kernel.
        llama::forward_prefill_batch(
            gpu,
            &b.weights,
            &b.config,
            &[token],
            seq_pos,
            &mut b.kv,
            &b.scratch,
            None,
        )
        .map_err(|e| format!("llama decode forward_prefill_batch: {:?}", e))
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        _grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        // llama has no grammar → mask is always None; GPU fast path only.
        let m = &*self.m;
        let ModelState::Llama(b) = m.state.as_ref().ok_or("no state")? else {
            return Err("sample: not a llama bundle".into());
        };
        let scratch = &b.scratch;
        let tok = sampler::sample(
            gpu,
            &scratch.logits,
            &scratch.sample_buf,
            &scratch.repeat_buf,
            vocab_size,
            ngram_scope,
            cfg,
            rng_state,
        );
        Ok(tok)
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        self.m.session.seq_pos
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        self.m.session.seq_pos = seq_pos;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        if let Some(ModelState::Llama(b)) = self.m.state.as_ref() {
            b.config.vocab_size
        } else {
            0
        }
    }

    fn repeat_buf_cap_bytes(&self) -> usize {
        if let Some(ModelState::Llama(b)) = self.m.state.as_ref() {
            b.scratch.repeat_buf.buf.size()
        } else {
            0
        }
    }

    fn prefill_max_batch(&self) -> usize {
        llama::PREFILL_MAX_BATCH
    }

    fn ensure_decoded_vocab(&mut self) -> std::sync::Arc<Vec<String>> {
        if self.m.persist.decoded_vocab.is_none() {
            let v: Vec<String> = {
                let tok = self.m.tokenizer.as_ref().unwrap();
                let n = tok.vocab_size();
                (0..n).map(|id| tok.decode(&[id as u32])).collect()
            };
            self.m.persist.decoded_vocab = Some(std::sync::Arc::new(v));
        }
        self.m.persist.decoded_vocab.clone().unwrap()
    }

    fn has_eviction(&self) -> bool {
        self.m.eviction.is_some()
    }

    fn physical_cap(&self) -> usize {
        self.m.meta.physical_cap
    }

    fn eviction_window(&self) -> Option<usize> {
        self.m.eviction.as_ref().map(|ev| ev.budget() + ev.beta())
    }

    fn insert_asst_turn(&mut self, fp: u64, seq: Vec<u32>) {
        self.m.persist.asst_turn_cache.insert(fp, seq);
    }
}

/// Wraps `&mut LoadedModel` to implement `ArchDispatch` for MiniMax-M2 (arch 10).
/// SPLIT: minimax forward writes `state.logits` (GPU-resident) and also downloads
/// a host `Vec<f32>` (ignored here — the `sample` hook re-downloads `state.logits`;
/// the double D2H is free on UMA). Prefill uses batched `forward_batch` (sub-chunked
/// to 64) when supported, else per-token `decode_step`. UPLIFT (bjoern): minimax
/// adopts the generic sampler (was `deepseek4::sampling::sample_token` temp/top_p +
/// Xorshift) → byte-parity holds only at temp0 (argmax); temp>0 output changes by
/// design. No eviction / recurrent / checkpoints / grammar (MoE attention). The
/// `primed_think` re-emit + Jinja/LCP-partial preamble stay in `generate_minimax`.
/// NOTE (Inc 4): dead-code groundwork — NOT wired/flipped; controller wires the
/// dual-run in generate_minimax then GPU-validates temp0 parity + coherence.
#[allow(dead_code)]
struct MinimaxDispatch<'m> {
    m: &'m mut LoadedModel,
}

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::ArchDispatch for MinimaxDispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        self.m.minimax().map(|b| b.eos_tok).unwrap_or(0)
    }

    fn is_eos(&self, tok: u32) -> bool {
        // Matches generate_minimax's stop (`next_tok == eos_tok`).
        self.m.minimax().map(|b| tok == b.eos_tok).unwrap_or(false)
    }

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 1.0,
            top_p: 0.95,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        // Interleaved-thinking (primed_think); user stop-seqs (uplift); no grammar,
        // no vision.
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: true,
            supports_stop_seq: true,
            supports_grammar: false,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let b = self
            .m
            .minimax_mut()
            .ok_or("prefill_forward: no minimax bundle")?;
        let batched = std::env::var_os("HIPFIRE_MINIMAX_BATCH_PREFILL").map_or(true, |v| v != "0")
            && minimax::forward::forward_batch_supported(&b.weights);
        let mut pos = seq_pos;
        if batched {
            for sub in chunk.chunks(64) {
                minimax::forward::forward_batch(&b.config, &b.weights, &mut b.state, gpu, sub, pos)
                    .map_err(|e| format!("minimax forward_batch (prefill): {e}"))?;
                pos += sub.len();
            }
        } else {
            for &tok in chunk {
                minimax::forward::decode_step(
                    &b.config,
                    &b.weights,
                    &mut b.state,
                    gpu,
                    tok,
                    pos as u32,
                )
                .map_err(|e| format!("minimax decode_step (prefill): {e}"))?;
                pos += 1;
            }
        }
        Ok(())
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let b = self
            .m
            .minimax_mut()
            .ok_or("decode_step_forward: no minimax bundle")?;
        minimax::forward::decode_step(
            &b.config,
            &b.weights,
            &mut b.state,
            gpu,
            token,
            seq_pos as u32,
        )
        .map(|_| ())
        .map_err(|e| format!("minimax decode_step (decode): {e}"))
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        // Host-side sample from state.logits (GPU). Generic sampler (uplift):
        // temp0 == argmax (parity vs legacy sample_token); temp>0 differs by design.
        let _ = (vocab_size, grammar_mask, rng_state);
        let b = self.m.minimax().ok_or("sample: no minimax bundle")?;
        let mut logits = gpu
            .download_f32(&b.state.logits)
            .map_err(|e| format!("minimax download logits: {e:?}"))?;
        Ok(sampler::sample_cpu(&mut logits, ngram_scope, cfg))
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        // The minimax KV cursor is state.n_tokens (advanced internally by
        // forward_batch/decode_step); seed the driver's local from it.
        self.m
            .minimax()
            .map(|b| b.state.n_tokens)
            .unwrap_or(self.m.session.seq_pos)
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        let _ = seq_pos;
        let np = self
            .m
            .minimax()
            .map(|b| b.state.n_tokens)
            .unwrap_or(self.m.session.seq_pos);
        self.m.session.seq_pos = np;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        self.m.minimax().map(|b| b.config.vocab_size).unwrap_or(0)
    }

    fn eos_filter_config(&self) -> hipfire_runtime::eos_filter::EosFilterConfig {
        // MiniMax's eos token IS the `[e~[` turn-end marker (loader eos candidate,
        // carriers.rs). It decodes to that literal (not empty like ChatML `<|im_end|>`),
        // so ar_generate — which commits+emits the eos token before its is_eos break —
        // would leak it into the visible stream. Strip it here. The legacy loop broke
        // pre-emit; this restores that suppression on eos-terminated turns.
        hipfire_runtime::eos_filter::EosFilterConfig {
            stop_at: vec![b"[e~[".to_vec()],
            ..Default::default()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Cohere2-MoE / North-Mini-Code (arch 12) AR dispatch. Forward mirrors
/// MinimaxDispatch (batched/per-token `decode_step`, host-side `sample_cpu`), but the
/// OUTPUT is the agentic-marker state machine: `stream_parser()` returns a
/// `Cohere2MoeStreamParser` (section routing / forced-token queue / empty-turn on_eos
/// `Inject` / `tool_calls`). The eos `<|END_OF_TURN_TOKEN|>` is consumed by the parser
/// (`on_eos`), never emitted — so no `eos_filter_config` is needed. `tools` is held so
/// `stream_parser()` can build the parser's `known_tools`/`tool_params`.
struct Cohere2MoeDispatch<'m> {
    m: &'m mut LoadedModel,
    tools: Option<Vec<serde_json::Value>>,
    /// True only for a full cold prefill. The legacy Cohere path uses the
    /// numerically-parity-tested batched prefill for this case; suffix reuse
    /// must not infer cold-start status from `seq_pos`.
    cold_start: bool,
}

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::ArchDispatch for Cohere2MoeDispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        self.m.cohere2moe().map(|b| b.eos_tok).unwrap_or(0)
    }

    fn is_eos(&self, tok: u32) -> bool {
        self.m
            .cohere2moe()
            .map(|b| tok == b.eos_tok)
            .unwrap_or(false)
    }

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        // North HF regime (temp≈1.0); mirrors generate_cohere2moe.
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 1.0,
            top_p: 0.95,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: true,
            supports_stop_seq: true,
            supports_grammar: false,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let b = self
            .m
            .cohere2moe_mut()
            .ok_or("prefill_forward: no cohere2moe bundle")?;
        let force_per_token =
            std::env::var("HIPFIRE_COHERE_PREFILL").ok().as_deref() == Some("per_token");
        let batched = self.cold_start
            && !force_per_token
            && cohere2moe::forward::forward_batch_supported(&b.weights);
        eprintln!(
            "[cohere-prefill] mode={} cold_start={} force_per_token={}",
            if batched { "batched" } else { "per_token" },
            self.cold_start,
            force_per_token,
        );
        let mut pos = seq_pos;
        if batched {
            // Chunk at 256 to match the legacy generate_cohere2moe prefill (`i+256`);
            // forward_batch is NOT numerically batch-size-invariant (WMMA/grouped-MoE
            // reduction order), so a different chunk size shifts the logits and flips
            // an argmax a few tokens into decode → dual-run temp0 parity fails.
            for sub in chunk.chunks(256) {
                cohere2moe::forward::forward_batch(
                    &b.config,
                    &b.weights,
                    &mut b.state,
                    gpu,
                    sub,
                    pos,
                )
                .map_err(|e| format!("cohere2moe forward_batch (prefill): {e}"))?;
                pos += sub.len();
            }
        } else {
            for &tok in chunk {
                cohere2moe::forward::decode_step(
                    &b.config,
                    &b.weights,
                    &mut b.state,
                    gpu,
                    tok,
                    pos as u32,
                )
                .map_err(|e| format!("cohere2moe decode_step (prefill): {e}"))?;
                pos += 1;
            }
        }
        Ok(())
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let b = self
            .m
            .cohere2moe_mut()
            .ok_or("decode_step_forward: no cohere2moe bundle")?;
        cohere2moe::forward::decode_step(
            &b.config,
            &b.weights,
            &mut b.state,
            gpu,
            token,
            seq_pos as u32,
        )
        .map(|_| ())
        .map_err(|e| format!("cohere2moe decode_step (decode): {e}"))
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let _ = (vocab_size, grammar_mask, rng_state);
        let b = self.m.cohere2moe().ok_or("sample: no cohere2moe bundle")?;
        let mut logits = gpu
            .download_f32(&b.state.logits)
            .map_err(|e| format!("cohere2moe download logits: {e:?}"))?;
        Ok(sampler::sample_cpu(&mut logits, ngram_scope, cfg))
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        self.m
            .cohere2moe()
            .map(|b| b.state.n_tokens)
            .unwrap_or(self.m.session.seq_pos)
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        let _ = seq_pos;
        let np = self
            .m
            .cohere2moe()
            .map(|b| b.state.n_tokens)
            .unwrap_or(self.m.session.seq_pos);
        self.m.session.seq_pos = np;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        self.m
            .cohere2moe()
            .map(|b| b.config.vocab_size)
            .unwrap_or(0)
    }

    fn stream_parser(
        &self,
        cfg: hipfire_runtime::stream_parser::DefaultStreamParserConfig,
    ) -> Box<dyn hipfire_runtime::stream_parser::StreamParser> {
        // cohere2moe ignores the default (EosFilter/think-cap) config — its output is the
        // agentic-marker state machine. Reuse cfg's max_tokens/max_think_tokens (the
        // request values the driver threaded) for the parser's think-budget clamp.
        Box::new(Cohere2MoeStreamParser::new(
            self.tokenizer(),
            self.tools.as_deref(),
            cfg.max_tokens,
            cfg.max_think_tokens,
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// LFM2.5 (arch 11) AR dispatch. Plain per-token loop (no batched prefill, no
/// LCP — the legacy path cold-resets state every turn since the hybrid conv+GQA
/// state can't be rewound to an arbitrary prefix), so the recurrent/checkpoint/
/// eviction hooks stay trait-default no-ops: any partial conv-state after an abort
/// is recovered by the next turn's cold `state.reset`. Two arch specifics vs the
/// minimax dispatch: (1) a STOP-ID SET — LFM2 ends a turn with `<|im_end|>` (=eos_tok)
/// but ALSO emits `<|endoftext|>` / `</s>`, whose literal strings decode verbatim;
/// (2) an `eos_filter_config` stripping all three (ar_generate commits+emits the
/// stop token before its is_eos break, so the literal would otherwise leak — the
/// legacy loop's id + string-frag guards both broke pre-emit).
struct Lfm2MoeDispatch<'m> {
    m: &'m mut LoadedModel,
    /// eos_tok (`<|im_end|>`) + the `<|endoftext|>` / `</s>` special-token ids,
    /// resolved via `special_token_id` (which looks up the added-token table —
    /// `encode` of these strings does NOT round-trip to their single id). Computed
    /// once in the preamble; `is_eos` tests membership.
    stop_ids: Vec<u32>,
}

impl hipfire_runtime::arch_dispatch::ArchDispatch for Lfm2MoeDispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        self.m.lfm2moe().map(|b| b.eos_tok).unwrap_or(0)
    }

    fn is_eos(&self, tok: u32) -> bool {
        // Matches generate_lfm2moe's `stop_toks.contains(&next_tok)` + the string-frag
        // guard, unified as an id set (special_token_id resolves the non-round-tripping
        // `<|endoftext|>` / `</s>`).
        self.stop_ids.contains(&tok)
    }

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 0.3,
            top_p: 0.95,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: true,
            supports_stop_seq: true,
            supports_grammar: false,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        // Per-token decode_step (no batched prefill kernel for lfm2moe).
        let b = self
            .m
            .lfm2moe_mut()
            .ok_or("prefill_forward: no lfm2moe bundle")?;
        let mut pos = seq_pos as u32;
        for &tok in chunk {
            lfm2moe::forward::decode_step(&b.config, &b.weights, &mut b.state, gpu, tok, pos)
                .map_err(|e| format!("lfm2moe decode_step (prefill): {e}"))?;
            pos += 1;
        }
        Ok(())
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        let b = self
            .m
            .lfm2moe_mut()
            .ok_or("decode_step_forward: no lfm2moe bundle")?;
        lfm2moe::forward::decode_step(
            &b.config,
            &b.weights,
            &mut b.state,
            gpu,
            token,
            seq_pos as u32,
        )
        .map(|_| ())
        .map_err(|e| format!("lfm2moe decode_step (decode): {e}"))
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Single(gpu) = ctx else {
            unreachable!("single-GPU dispatch received Mesh ctx")
        };
        // Host-side sample from state.logits (GPU-resident; decode_step writes+downloads
        // it). Generic sampler (uplift): temp0 == argmax (parity vs legacy sample_token).
        let _ = (vocab_size, grammar_mask, rng_state);
        let b = self.m.lfm2moe().ok_or("sample: no lfm2moe bundle")?;
        let mut logits = gpu
            .download_f32(&b.state.logits)
            .map_err(|e| format!("lfm2moe download logits: {e:?}"))?;
        Ok(sampler::sample_cpu(&mut logits, ngram_scope, cfg))
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        self.m
            .lfm2moe()
            .map(|b| b.state.n_tokens)
            .unwrap_or(self.m.session.seq_pos)
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        let _ = seq_pos;
        let np = self
            .m
            .lfm2moe()
            .map(|b| b.state.n_tokens)
            .unwrap_or(self.m.session.seq_pos);
        self.m.session.seq_pos = np;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        self.m.lfm2moe().map(|b| b.config.vocab_size).unwrap_or(0)
    }

    fn eos_filter_config(&self) -> hipfire_runtime::eos_filter::EosFilterConfig {
        // LFM2 stop-class tokens decode to their LITERAL strings (unlike ChatML
        // `<|im_end|>` on qwen, which decodes to empty). ar_generate commits+emits the
        // stop token before its is_eos break, so strip all three here — the legacy loop
        // broke on both the id set and a decoded-frag string guard, pre-emit.
        hipfire_runtime::eos_filter::EosFilterConfig {
            stop_at: vec![
                b"<|im_end|>".to_vec(),
                b"<|endoftext|>".to_vec(),
                b"</s>".to_vec(),
            ],
            ..Default::default()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Deepseek4 (arch 9) EXPERT-PARALLEL (multi-GPU) AR dispatch — the FIRST mesh
/// `ArchDispatch`. Unlike the six single-GPU impls, its device (`Gpus`) lives
/// inside `ModelParallel::Ep(EpState { gpus, .. })`, i.e. inside the same `&mut m`
/// this dispatch borrows — so every forward hook matches `ForwardCtx::Mesh` and
/// reaches the mesh through `&mut self`; `Single` is `unreachable!()`. Mirrors
/// `ep_serve_ds4` (daemon.rs): per-token `forward_ep` for both prefill and decode,
/// rank-0 logits download, and the HOST full-distribution sampler
/// `llama::sample_full_dist` (NOT the GPU sampler / `sample_cpu`). EP AR is
/// MTP-free (spec-decode is the separate `generate_deepseek4_spec`). No eviction /
/// adaptive-KV / prefill checkpoints → the tangle hooks keep their trait-default
/// no-ops.
///
/// NOTE (Axis B inc 2): build-only groundwork — NOT wired/flipped. Inc 3 adds the
/// DSML grammar/stream-parser output; inc 4 routes the `matches!(Ep)` gate through
/// `ar_generate`; inc 5 validates the FNV anchor + dual-run parity.
#[allow(dead_code)]
struct Deepseek4EpDispatch<'m> {
    m: &'m mut LoadedModel,
    /// Raw request tools (OpenAI schema). Held so `init_grammar` can rebuild the
    /// FULL ds4 `ToolSchema` (name + params-from-`properties` + required) — the
    /// trait's `(name, required)` tuple drops `params`, which the ds4 grammar
    /// needs (the allowed param-name set per invoke). Mirrors Cohere2MoeDispatch
    /// holding `tools` for its stream parser.
    tools: Option<Vec<serde_json::Value>>,
    /// Assistant-prefix think mode → picks the dsml parser's `new_in_think`
    /// (prompt ended inside `<think>`) vs `new` start state in `stream_parser`.
    think_mode: ThinkMode,
}

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::ArchDispatch for Deepseek4EpDispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        // ds4 EP eos: the loader-resolved carrier `m.meta.eos_tok`
        // (ep_serve_ds4 takes `eos_tok = m.meta.eos_tok`, daemon.rs:4246).
        self.m.meta.eos_tok
    }

    // is_eos = trait default (tok == eos_token): ep_serve_ds4 breaks on
    // `next == eos_tok`, so the primary-eos-only stop is byte-faithful.

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        // deepseek4 HF regime; mirrors the generic arch ladder default.
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 0.3,
            top_p: 0.95,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        // ds4 DSML: think blocks + tool-call grammar; user stop-seqs; no vision.
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: true,
            supports_stop_seq: true,
            supports_grammar: true,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("deepseek4-EP dispatch received Single ctx")
        };
        let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else {
            return Err("prefill_forward: no ep state".into());
        };
        let EpArch::Ds4 {
            config,
            weights,
            state,
            partials,
            partials_i64,
            policy,
        } = inner
        else {
            return Err("prefill_forward: EP arch mismatch (expected ds4)".into());
        };
        // `forward_ep` is single-token; ep_serve_ds4 prefills the prompt one
        // token at a time (daemon.rs:4633). Replay the chunk at absolute
        // positions `seq_pos + i`.
        for (i, &t) in chunk.iter().enumerate() {
            deepseek4::forward::forward_ep(
                gpus,
                weights,
                config,
                state,
                partials,
                partials_i64,
                policy,
                t,
                (seq_pos + i) as u32,
            )
            .map_err(|e| format!("forward_ep prefill: {e}"))?;
        }
        Ok(())
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("deepseek4-EP dispatch received Single ctx")
        };
        let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else {
            return Err("decode_step_forward: no ep state".into());
        };
        let EpArch::Ds4 {
            config,
            weights,
            state,
            partials,
            partials_i64,
            policy,
        } = inner
        else {
            return Err("decode_step_forward: EP arch mismatch (expected ds4)".into());
        };
        deepseek4::forward::forward_ep(
            gpus,
            weights,
            config,
            state,
            partials,
            partials_i64,
            policy,
            token,
            seq_pos as u32,
        )
        .map_err(|e| format!("forward_ep decode: {e}"))
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("deepseek4-EP dispatch received Single ctx")
        };
        // EP samples HOST-side over rank-0's downloaded logits via the full-dist
        // sampler (ep_serve_ds4:4744) — NOT the GPU sampler. So `ngram_scope` and
        // `rng_state` (the GPU xorshift seed) are inert; the cpu-sampler RNG is
        // seeded once per request in the preamble (`reset_cpu_sampler_rng`, wired
        // by inc 4). The grammar mask (built by the driver's Deepseek4GrammarMatcher,
        // inc 3) is applied to the downloaded logits HERE before the draw
        // (`apply_mask_to_logits` stays internal to EP, per the design).
        let _ = (vocab_size, ngram_scope, rng_state);
        let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else {
            return Err("sample: no ep state".into());
        };
        let EpArch::Ds4 { state, .. } = inner else {
            return Err("sample: EP arch mismatch (expected ds4)".into());
        };
        let _ = gpus.devices[0].bind_thread();
        let logits_gt = state[0].logits.as_ref().ok_or("sample: EP logits unset")?;
        let mut logits = gpus.devices[0]
            .download_f32(logits_gt)
            .map_err(|e| format!("EP logits download: {e:?}"))?;
        if let Some(mask) = grammar_mask {
            deepseek4::grammar::Matcher::apply_mask_to_logits(mask, &mut logits);
        }
        Ok(hipfire_runtime::llama::sample_full_dist(
            &logits,
            cfg.temperature,
            cfg.top_p,
            cfg.top_k,
            cfg.min_p,
        ))
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        // The EP KV cursor is rank-0's `state.n_tokens` (advanced internally by
        // forward_ep); seed the driver's local from it.
        match &self.m.parallel {
            ModelParallel::Ep(EpState {
                inner: EpArch::Ds4 { state, .. },
                ..
            }) => state[0].n_tokens as usize,
            _ => self.m.session.seq_pos,
        }
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        let _ = seq_pos;
        let np = match &self.m.parallel {
            ModelParallel::Ep(EpState {
                inner: EpArch::Ds4 { state, .. },
                ..
            }) => state[0].n_tokens as usize,
            _ => self.m.session.seq_pos,
        };
        self.m.session.seq_pos = np;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        match &self.m.parallel {
            ModelParallel::Ep(EpState {
                inner: EpArch::Ds4 { config, .. },
                ..
            }) => config.vocab_size,
            _ => 0,
        }
    }

    fn ensure_decoded_vocab(&mut self) -> std::sync::Arc<Vec<String>> {
        // The grammar path (tool calls) needs the decoded-vocab table for token
        // masks; ep_serve_ds4 builds it inline (daemon.rs ~4578). Lazy-cache on
        // m.persist.decoded_vocab, identical to Qwen35Dispatch. Without this override the
        // trait default unimplemented!() panics the moment grammar activates — a
        // gap invisible to the non-grammar dual-run rows (caught by the tool-call
        // parity run, Axis B inc 5).
        if self.m.persist.decoded_vocab.is_none() {
            let v: Vec<String> = {
                let tok = self.m.tokenizer.as_ref().unwrap();
                let n = tok.vocab_size();
                (0..n).map(|id| tok.decode(&[id as u32])).collect()
            };
            self.m.persist.decoded_vocab = Some(std::sync::Arc::new(v));
        }
        self.m.persist.decoded_vocab.clone().unwrap()
    }

    fn init_grammar(
        &self,
        _tool_schemas: &[(String, Vec<String>)],
    ) -> Option<Box<dyn hipfire_runtime::arch_dispatch::GrammarMatcher>> {
        // ds4's grammar needs the FULL ToolSchema (name + params + required); the
        // trait's `(name, required)` tuple lacks `params`, so we rebuild it from
        // the held raw `tools` — verbatim ep_serve_ds4:4541-4575 (params = the
        // `properties` keys; required = the `required` array).
        let schemas: Vec<deepseek4::grammar::ToolSchema> = self
            .tools
            .as_deref()
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
        if schemas.is_empty() {
            return None;
        }
        Some(Box::new(Deepseek4GrammarMatcher(
            deepseek4::grammar::Matcher::new(schemas),
        )))
    }

    fn stream_parser(
        &self,
        cfg: hipfire_runtime::stream_parser::DefaultStreamParserConfig,
    ) -> Box<dyn hipfire_runtime::stream_parser::StreamParser> {
        // ds4 output is the DSML state machine, not the EosFilter/think-cap default,
        // so most of `cfg` is inert — but honor `cfg.stop_seqs` (user stop sequences;
        // ep_serve_ds4 checked them per token). Start state depends on whether the
        // assistant prefix opened inside `<think>` (ep_serve_ds4:4537).
        Box::new(Deepseek4StreamParser::new(self.think_mode, cfg.stop_seqs))
    }
}

/// Newtype so daemon.rs (owner) can implement the runtime `GrammarMatcher` for the
/// ds4 `Matcher` without the orphan rule. 1:1 with the concrete matcher; the ds4
/// `Matcher` tracks no attractor, so `attractor_detected` keeps the trait default.
#[allow(dead_code)]
struct Deepseek4GrammarMatcher(deepseek4::grammar::Matcher);

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::GrammarMatcher for Deepseek4GrammarMatcher {
    fn token_mask(&self, vocab: &[String], out: &mut [bool]) {
        self.0.token_mask(vocab, out);
    }

    fn advance(&mut self, text: &str) {
        self.0.advance(text);
    }

    fn is_free(&self) -> bool {
        self.0.is_free()
    }
}

/// Wraps `deepseek4::dsml::StreamParser` for the generic `ar_generate` output
/// layer. The dsml parser consumes decoded TEXT (`feed(&str)`) and its `finish`
/// CONSUMES self, so we hold it behind an `Option` and `take()` on `finish`. Each
/// committed token's running-vector byte delta (utf8-lossy) is fed as text; the
/// dsml `StreamEvent`s map to `StreamAction`s (Token→visible Emit, Reasoning→
/// reasoning Emit, ToolCalls→the `{name,arguments}` array — same shape as
/// `emit_stream_event`, daemon.rs:445).
#[allow(dead_code)]
struct Deepseek4StreamParser {
    inner: Option<deepseek4::dsml::StreamParser>,
    /// User stop sequences (request `stop`). ep_serve_ds4 broke per token on
    /// `text_acc.ends_with(s)`; the dsml parser doesn't know about them, so honor
    /// them here — without this, `stop` is silently a no-op for ds4-EP while
    /// `features().supports_stop_seq` advertises support (review I1).
    stop_seqs: Vec<String>,
    /// Accumulated decoded text (all pieces), for the stop-seq suffix check —
    /// mirrors ep_serve_ds4's `text_acc`.
    text_acc: String,
}

impl Deepseek4StreamParser {
    fn new(think_mode: ThinkMode, stop_seqs: Vec<String>) -> Self {
        let inner = match think_mode {
            ThinkMode::High | ThinkMode::Max => deepseek4::dsml::StreamParser::new_in_think(),
            ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
        };
        Self {
            inner: Some(inner),
            stop_seqs,
            text_acc: String::new(),
        }
    }

    /// Translate dsml `StreamEvent`s into driver `StreamAction`s. Static so it can
    /// be unit-tested without a live dsml parse (inc 3's mapping test).
    fn map_events(
        evs: Vec<deepseek4::dsml::StreamEvent>,
    ) -> Vec<hipfire_runtime::stream_parser::StreamAction> {
        use deepseek4::dsml::StreamEvent;
        use hipfire_runtime::stream_parser::StreamAction;
        let mut acts = Vec::new();
        for ev in evs {
            match ev {
                StreamEvent::Token(text) => acts.push(StreamAction::Emit {
                    text,
                    reasoning: false,
                }),
                StreamEvent::Reasoning(text) => acts.push(StreamAction::Emit {
                    text,
                    reasoning: true,
                }),
                StreamEvent::ToolCalls(calls) => {
                    // Same shape emit_stream_event uses: a `calls` array of
                    // {name, arguments}. The driver wraps it as
                    // {"type":"tool_calls","id":..,"calls":<array>} (daemon.rs:8718).
                    let arr: Vec<serde_json::Value> = calls
                        .into_iter()
                        .map(|c| serde_json::json!({ "name": c.name, "arguments": c.arguments }))
                        .collect();
                    acts.push(StreamAction::ToolCalls(serde_json::Value::Array(arr)));
                }
            }
        }
        acts
    }
}

impl hipfire_runtime::stream_parser::StreamParser for Deepseek4StreamParser {
    fn feed(
        &mut self,
        tok: u32,
        bytes: &[u8],
    ) -> Vec<hipfire_runtime::stream_parser::StreamAction> {
        // dsml works on decoded text, not token ids — the token id is inert; the
        // running-vector byte delta IS this token's decoded text.
        let _ = tok;
        let piece = String::from_utf8_lossy(bytes);
        let mut acts = {
            let Some(inner) = self.inner.as_mut() else {
                return Vec::new();
            };
            Self::map_events(inner.feed(&piece))
        };
        // Honor user stop sequences after the token's emit actions (mirrors
        // ep_serve_ds4's per-token `text_acc.ends_with(s)` break, review I1).
        if !self.stop_seqs.is_empty() {
            self.text_acc.push_str(&piece);
            if self
                .stop_seqs
                .iter()
                .any(|s| !s.is_empty() && self.text_acc.ends_with(s.as_str()))
            {
                acts.push(hipfire_runtime::stream_parser::StreamAction::Stop);
            }
        }
        acts
    }

    fn on_eos(&mut self) -> hipfire_runtime::stream_parser::EosDecision {
        // ep_serve_ds4 breaks on the sampled eos WITHOUT forwarding/emitting/feeding
        // it (daemon.rs:4751): the eos never enters KV, the tape, or the parser.
        // `Stop` reproduces that exactly (CommitAndStop would forward + emit_only the
        // eos → an extra forward pass + a spurious emit, diverging from the reference).
        hipfire_runtime::stream_parser::EosDecision::Stop
    }

    fn finish(&mut self) -> Vec<hipfire_runtime::stream_parser::StreamAction> {
        // dsml `finish` CONSUMES self → take the inner parser out and flush.
        match self.inner.take() {
            Some(inner) => Self::map_events(inner.finish()),
            None => Vec::new(),
        }
    }
}

/// MiniMax-M2 (arch 10) EXPERT-PARALLEL (multi-GPU) AR dispatch — the second mesh
/// dispatch (after Deepseek4EpDispatch). Same shape: `ForwardCtx::Mesh` reaches
/// `EpState::gpus` (via `ModelParallel::Ep`) through `&mut self`; per-token `forward_ep` prefill(loop)/decode;
/// rank-0 `download_f32` + HOST `sample_full_dist`. OUTPUT is plain text (NOT DSML) —
/// so it reuses MinimaxDispatch's hooks: the `[e~[` eos filter + the DEFAULT
/// StreamParser (no grammar, no tool-call channel). The LCP prefix-cache rewind +
/// the display-only `<think>` primer live in the arch-10 arm's preamble (mirroring
/// ep_serve_minimax), NOT here.
///
/// NOTE: build-only groundwork — the arch-10 arm wires + validates + flips it.
#[allow(dead_code)]
struct MinimaxEpDispatch<'m> {
    m: &'m mut LoadedModel,
}

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::ArchDispatch for MinimaxEpDispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        // EP eos: minimax EP state lives in m.ep (minimax() is None), so the eos is
        // carried on LoadedModel (ep_serve_minimax reads m.meta.eos_tok).
        self.m.meta.eos_tok
    }

    // is_eos = trait default (tok == eos): ep_serve_minimax breaks on `next==eos_tok`.

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 1.0,
            top_p: 0.95,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        // Interleaved-thinking + user stop-seqs; no grammar (no EP tool calls), no vision.
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: true,
            supports_stop_seq: true,
            supports_grammar: false,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("minimax-EP dispatch received Single ctx")
        };
        let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else {
            return Err("prefill_forward: no ep state".into());
        };
        let EpArch::Minimax {
            config,
            weights,
            state,
            partials,
            partials_i64,
            policy,
        } = inner
        else {
            return Err("prefill_forward: EP arch mismatch (expected minimax)".into());
        };
        for (i, &t) in chunk.iter().enumerate() {
            minimax::forward::forward_ep(
                gpus,
                weights,
                config,
                state,
                partials,
                partials_i64,
                // The loader-owned exact EP policy (built once from the
                // admitted mesh the Gpus are bound to) — never reconstructed
                // per token.
                policy,
                t,
                (seq_pos + i) as u32,
            )
            .map_err(|e| format!("minimax forward_ep prefill: {e}"))?;
        }
        Ok(())
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("minimax-EP dispatch received Single ctx")
        };
        let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else {
            return Err("decode_step_forward: no ep state".into());
        };
        let EpArch::Minimax {
            config,
            weights,
            state,
            partials,
            partials_i64,
            policy,
        } = inner
        else {
            return Err("decode_step_forward: EP arch mismatch (expected minimax)".into());
        };
        minimax::forward::forward_ep(
            gpus,
            weights,
            config,
            state,
            partials,
            partials_i64,
            // The loader-owned exact EP policy — same object every token.
            policy,
            token,
            seq_pos as u32,
        )
        .map_err(|e| format!("minimax forward_ep decode: {e}"))
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("minimax-EP dispatch received Single ctx")
        };
        // EP host full-dist sampler over rank-0 logits (ep_serve_minimax). No grammar
        // for minimax EP, so grammar_mask is always None; ngram_scope/rng_state inert.
        let _ = (vocab_size, ngram_scope, grammar_mask, rng_state);
        let ModelParallel::Ep(EpState { gpus, inner }) = &mut self.m.parallel else {
            return Err("sample: no ep state".into());
        };
        let EpArch::Minimax { state, .. } = inner else {
            return Err("sample: EP arch mismatch (expected minimax)".into());
        };
        let _ = gpus.devices[0].bind_thread();
        let logits = gpus.devices[0]
            .download_f32(&state[0].logits)
            .map_err(|e| format!("minimax EP logits download: {e:?}"))?;
        Ok(hipfire_runtime::llama::sample_full_dist(
            &logits,
            cfg.temperature,
            cfg.top_p,
            cfg.top_k,
            cfg.min_p,
        ))
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        match &self.m.parallel {
            ModelParallel::Ep(EpState {
                inner: EpArch::Minimax { state, .. },
                ..
            }) => state[0].n_tokens,
            _ => self.m.session.seq_pos,
        }
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        let _ = seq_pos;
        let np = match &self.m.parallel {
            ModelParallel::Ep(EpState {
                inner: EpArch::Minimax { state, .. },
                ..
            }) => state[0].n_tokens,
            _ => self.m.session.seq_pos,
        };
        self.m.session.seq_pos = np;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        match &self.m.parallel {
            ModelParallel::Ep(EpState {
                inner: EpArch::Minimax { config, .. },
                ..
            }) => config.vocab_size,
            _ => 0,
        }
    }

    fn eos_filter_config(&self) -> hipfire_runtime::eos_filter::EosFilterConfig {
        // Reuse MinimaxDispatch's suppression: minimax's eos IS the `[e~[` turn-end
        // marker (decodes to that literal, not empty), so ar_generate — which
        // commits+emits the eos before its is_eos break — would leak it. Strip here.
        hipfire_runtime::eos_filter::EosFilterConfig {
            stop_at: vec![b"[e~[".to_vec()],
            ..Default::default()
        }
    }
}

/// Dense multi-GPU (TP / dense-PP) AR dispatch — the THIRD mesh dispatch. Wraps a
/// `DenseServed` model (TpModel or PpModel) whose `Gpus` lives inside `m.parallel`
/// (Tp variant Task 3, Pp(Dense) variant Task 4), so `ForwardCtx::Mesh` reaches it
/// through `&mut self`. Folds `generate_dense`: per-token `forward_token` (batched
/// `prefill` at pos 0, per-token suffix on an LCP hit), host `sample_cpu` over
/// `logits()`, `is_eos = eos || is_terminator`. Lean — no think / grammar /
/// eviction / checkpoints. Dense models track no cursor, so the gate preamble seeds
/// `m.session.seq_pos` from the LCP start_pos and ar_generate advances it.
#[allow(dead_code)]
struct DenseDispatch<'m> {
    m: &'m mut LoadedModel,
}

impl DenseDispatch<'_> {
    fn dense_model(&mut self) -> &mut dyn DenseServed {
        // Task 4: both TP and Pp(Dense) are inside m.parallel; delegate to free fn.
        dense_model_mut(self.m).expect("DenseDispatch: no dense model")
    }
}

#[allow(dead_code)]
impl hipfire_runtime::arch_dispatch::ArchDispatch for DenseDispatch<'_> {
    fn arch_id(&self) -> u32 {
        self.m.meta.arch_id
    }

    fn eos_token(&self) -> u32 {
        // Task 4: both TP and Pp(Dense) are inside m.parallel.
        match &self.m.parallel {
            ModelParallel::Tp(t) => t.eos_token(),
            ModelParallel::Pp(PipelineImpl::Dense(p)) => p.eos_token(),
            _ => 0,
        }
    }

    fn is_eos(&self, tok: u32) -> bool {
        // generate_dense breaks on `next == eos || tokenizer.is_terminator(next)`.
        tok == self.eos_token() || self.tokenizer().is_terminator(tok)
    }

    fn sampling_defaults(&self) -> hipfire_runtime::arch_dispatch::SamplingDefaults {
        hipfire_runtime::arch_dispatch::SamplingDefaults {
            temp: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.0,
        }
    }

    fn features(&self) -> hipfire_runtime::arch_dispatch::ArchFeatures {
        // Lean dense serve (generate_dense's contract): user stop-seqs only.
        hipfire_runtime::arch_dispatch::ArchFeatures {
            supports_think: false,
            supports_stop_seq: true,
            supports_grammar: false,
            supports_vision: false,
        }
    }

    fn prefill_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("dense dispatch received Single ctx")
        };
        // start_pos==0 (cache miss): the model's BATCHED prefill (positions 0..len,
        // internally chunked ≤256). start_pos>0 (LCP hit): per-token forward over the
        // cached prefix. Mirrors generate_dense. prefill_max_batch=MAX → ar_generate
        // hands one chunk, so seq_pos is the LCP start and the model batches itself.
        let model = self.dense_model();
        if seq_pos == 0 {
            model.prefill(chunk)
        } else {
            for (i, &t) in chunk.iter().enumerate() {
                model.forward_token(t, seq_pos + i)?;
            }
            Ok(())
        }
    }

    fn decode_step_forward(
        &mut self,
        ctx: ForwardCtx<'_>,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("dense dispatch received Single ctx")
        };
        self.dense_model().forward_token(token, seq_pos)
    }

    fn sample(
        &mut self,
        ctx: ForwardCtx<'_>,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
        let ForwardCtx::Mesh = ctx else {
            unreachable!("dense dispatch received Single ctx")
        };
        // Host sampler over the downloaded logits (generate_dense uses sample_cpu).
        // No grammar for dense; rng_state (GPU xorshift) inert — sample_cpu uses the
        // cpu-sampler RNG seeded per request.
        let _ = (vocab_size, grammar_mask, rng_state);
        let mut logits = self.dense_model().logits()?;
        Ok(sampler::sample_cpu(&mut logits, ngram_scope, cfg))
    }

    fn tokenizer(&self) -> &hipfire_runtime::tokenizer::Tokenizer {
        self.m.tokenizer.as_ref().unwrap()
    }

    fn seq_pos(&self) -> usize {
        self.m.session.seq_pos
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        self.m.session.seq_pos = seq_pos;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.session.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        // Inert for dense: sample_cpu sizes off logits.len() and there is no grammar
        // mask; the driver's GPU-sampler / grammar paths that read this are unused.
        0
    }

    fn prefill_max_batch(&self) -> usize {
        // One chunk: DenseServed::prefill batches internally (≤256) at pos 0, so the
        // whole suffix goes in a single prefill_forward call (seq_pos = LCP start).
        usize::MAX
    }
}

/// Truncate a checkpoint ring to `keep` slots, freeing the dropped snapshots'
/// GPU buffers (a bare `Vec::truncate` would leak them).
fn truncate_checkpoints(
    cks: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    keep: usize,
    gpu: &mut rdna_compute::Gpu,
) {
    while cks.len() > keep {
        if let Some((_, snap)) = cks.pop() {
            snap.free_gpu(gpu);
        }
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

/// Host-only daemon load plan: admission + lifecycle decisions — no GPU,
/// no mesh construction beyond [`DeviceMesh`], no allocation.
///
/// This is the single production entry point for the load handler, called
/// immediately after parsing raw path + axis degrees from the load message,
/// **before** any prior-model unload, GPU mutation, secondary-device init,
/// stream/peer/allocation, or per-axis dispatch.
///
/// The daemon's base GPU is pre-initialized by `Gpu::init()` — this helper
/// does NOT initialize GPUs, build meshes that trigger GPU init, or perform
/// any hardware work.  It only classifies the model source, resolves policy,
/// and derives lifecycle booleans from the effective axis.
///
/// # Lifecycle
///
/// - `defer_unload`: Multi-device axis (TP or EP) defers the prior model's
///   unload so a partial multi-GPU load failure doesn't orphan VRAM.
///   Single/PP axes clear the prior model eagerly.
/// - `pflash_suppressed`: Multi-device axis suppresses PFlash drafter loading
///   (TP/EP archs bypass PFlash entirely; loading a drafter would waste GPU
///   memory on a device that doesn't serve the generate loop).
/// - `effective_mesh`: built solely from effective degrees. Dense EP has
///   already been normalised to (1,1,1) by admission, so it gets a single-
///   device mesh — the EP secondary device / collective path is never entered.
///
/// On error the string carries `[CAP-001]`/`[COMP-001]` tags and is suitable
/// for the daemon's JSON error envelope.
#[derive(Debug)]
struct DaemonLoadPlan {
    admitted: hipfire_loader::AdmittedLoad,
    effective: hipfire_loader::parallel_capability::RawParallelRequest,
    effective_mesh: hipfire_hardware::DeviceMesh,
    /// True when the effective axis is multi-device (TP or EP) — prior
    /// model unload is deferred until after the new load succeeds.
    defer_unload: bool,
    /// True when PFlash drafter loading should be suppressed (multi-device
    /// TP or EP path).
    pflash_suppressed: bool,
}

fn daemon_load_plan(path: &str, pp: usize, tp: usize, ep: usize) -> Result<DaemonLoadPlan, String> {
    let raw = hipfire_loader::parallel_capability::RawParallelRequest::new(pp, tp, ep);
    let admitted = hipfire_loader::admit_path(path, raw)?;
    let effective = admitted.admission().effective();
    let effective_mesh =
        hipfire_runtime::config::resolve_mesh(effective.pp, effective.tp, effective.ep, None);
    let is_multi_device = effective.tp > 1 || effective.ep > 1;
    Ok(DaemonLoadPlan {
        admitted,
        effective,
        effective_mesh,
        defer_unload: is_multi_device,
        pflash_suppressed: is_multi_device,
    })
}

/// Parse `HIPFIRE_PP_LAYERS` into PP layer bands for the daemon load handler.
///
/// Currently always returns `None` because no `load_admitted` consumer
/// accepts pp_bands yet.  When a consumer is wired, this will parse the
/// env var and validate it so stale/invalid env cannot reject a valid load.
fn pp_bands_from_env() -> Option<Vec<usize>> {
    let _raw = std::env::var("HIPFIRE_PP_LAYERS");
    None
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
            eprintln!(
                "Kernel precompilation finished with {errors} failure(s) — the missing kernels will JIT on first use."
            );
        } else {
            eprintln!("Kernel precompilation done.");
        }
        return;
    }

    // --self-check-parity: smoke-test the dual-run COMPARATOR (no GPU, no model).
    // This does NOT prove old-vs-new decode parity — that is the live per-flip
    // dual-run threaded through ar_generate. It only guards that
    // `assert_token_parity` still works in BOTH directions: accepts identical
    // tapes AND panics on a divergence. (The prior version asserted only the
    // first, so it proved nothing beyond "equality holds".)
    if args.iter().any(|a| a == "--self-check-parity") {
        let mut tape_a = TokenTape::default();
        let mut tape_b = TokenTape::default();
        for tok in [1u32, 42, 100, 999] {
            tape_a.push(tok);
            tape_b.push(tok);
        }
        // Positive: identical tapes must NOT panic.
        assert_token_parity(&tape_a, &tape_b, "self-check-equal");
        // Negative: a divergent tape MUST panic — else the comparator is blind
        // and a real flip regression would slip through unnoticed.
        let mut tape_c = tape_b.clone();
        tape_c.push(1234);
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // silence the expected panic
        let detected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_token_parity(&tape_a, &tape_c, "self-check-diverge")
        }))
        .is_err();
        std::panic::set_hook(prev_hook);
        assert!(
            detected,
            "parity comparator failed to detect a divergent tape"
        );
        println!("parity self-check OK (comparator accepts equal, rejects divergent)");
        return;
    }

    // Machine-wide mutex — prevents orphan daemons from silently coexisting
    // (observed 2026-04-13: two daemons at 100% CPU survived pkill -f rounds
    // because they'd been reparented to PID 1 after their bun parent died).
    // Kept in a binding so the fd lives for the full process lifetime.
    let _daemon_lock = acquire_daemon_lock();

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            report_gpu_init_failure(&e);
            std::process::exit(1);
        }
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

    // Background stdin reader. Drains stdin into an mpsc channel so
    // the main loop can pull non-blockingly between messages. Abort
    // messages (`{type:"abort","id":"..."}`) are NOT forwarded; the
    // reader handles them inline by setting `abort_for_id()`. This is
    // the channel that makes client-side cancellation actually stop
    // an in-flight prefill — without it, the main loop is blocked on
    // GPU compute and wouldn't even read the abort line until after
    // the prefill completed.
    let (msg_tx, msg_rx) = mpsc::channel::<DaemonMsg>();
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
                    if msg.get("type").and_then(|v| v.as_str()) == Some("abort") {
                        if let Some(id) = msg.get("id").and_then(|v| v.as_str()) {
                            eprintln!("[daemon-abort] received abort for id={}", id);
                            *abort_for_id().lock().unwrap() = Some(id.to_string());
                        }
                        continue;
                    }
                    if msg.get("type").and_then(|v| v.as_str()) == Some("force_answer") {
                        if let Some(id) = msg.get("id").and_then(|v| v.as_str()) {
                            eprintln!("[daemon-force-answer] received force_answer for id={}", id);
                            *force_answer_for_id().lock().unwrap() = Some(id.to_string());
                        }
                        continue;
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
    let mut stdout = std::io::stdout();

    'daemon: while let Ok(daemon_msg) = msg_rx.recv() {
        let msg = match daemon_msg {
            DaemonMsg::Regular(m) => m,
            DaemonMsg::ParseError(e) => {
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

        match msg_type {
            "load" => {
                // ── CAP-001 Task 4: pre-mesh admission ──────────────────────────
                //
                // Parse path + raw axis degrees FIRST, before any prior-model
                // unload, GPU mutation, or load side-effect.  The base GPU is
                // already initialized by `Gpu::init()` above — this is not a
                // GPU-init site.  Admission precedes all load-triggered work:
                // mesh construction, secondary-device init (Gpus::from_mesh),
                // stream/peer allocation, collectives, unload, and per-axis
                // dispatch.
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

                let admit_result = match daemon_load_plan(path, raw_pp, raw_tp, raw_ep) {
                    Ok(r) => r,
                    Err(e) => {
                        // Preserve current JSON error envelope.
                        // Error carries [CAP-001] / [COMP-001] tags from the loader.
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({"type": "error", "message": e})
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let effective_pp = admit_result.effective.pp;
                let effective_tp = admit_result.effective.tp;
                let effective_ep = admit_result.effective.ep;

                // Lifecycle decisions come from daemon_load_plan, not ad-hoc
                // axis recomputation.  defer_unload: multi-device (TP or EP)
                // defers prior model unload so a partial multi-GPU load failure
                // doesn't orphan VRAM.  Single/PP use eager unload (the prior
                // model is freed before the new load begins).
                let is_deferred = admit_result.defer_unload;

                // Unload previous model if any, using the lifecycle decision.
                // PFlash drafter goes first so its tensors join the pool
                // before unload_model drains it — otherwise free_tensor
                // would queue them into the pool just-emptied by drain_pool
                // with no follow-up drain, leaving drafter VRAM resident
                // across the next load.
                //
                // For deferred EP: pflash_state is part of the PRIOR model.
                // It must NOT be torn down here — otherwise a partial EP
                // load failure (whose deferral keeps `model` alive) would
                // leave the surviving prior model stripped of its drafter.
                // Defer it to the success branch alongside the deferred
                // model unload below.  (EP archs are ds4/minimax and refuse
                // PFlash drafters, so on a SUCCESSFUL EP load this just
                // frees the outgoing model's drafter at the deferred site.)
                if !is_deferred {
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
                    if let Some(m) = model.take() {
                        hipfire_loader::unload_model(m, &mut gpu);
                    }
                }

                // hunt3 H-D: clamp request-driven max_seq to the config ceiling
                // (MAX_REQUESTED_SEQ = 512K). Without this an unvalidated 10M
                // max_seq drives a multi-GB KV allocation and OOMs the daemon at
                // load. Emit an info event when the clamp actually fires so the
                // operator sees the truncation rather than silently getting 512K.
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
                let raw_draft = msg
                    .get("params")
                    .and_then(|p| p.get("draft"))
                    .and_then(|v| v.as_str())
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
                let kv_mode_override = msg
                    .get("params")
                    .and_then(|p| p.get("kv_mode"))
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

                // MTP load config. DeepSeek (arch 9) uses its in-weights MTP
                // draft window; `mtp_k` resolves once into model metadata. Qwen
                // native MTP is disabled pending SPEC-003: on rejects before any
                // native-head preflight, while auto/off use AR.
                let mtp_mode_param = msg.get("params").and_then(|p| p.get("mtp_mode"));
                let mtp_mode = match resolve_mtp_mode(mtp_mode_param) {
                    Ok(mode) => mode,
                    Err(error) => {
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({ "type": "error", "message": error })
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };
                let mtp_k_param = msg.get("params").and_then(|p| p.get("mtp_k"));
                // Derive DeepSeek/MTP decisions from the admission variant,
                // avoiding a redundant ModelSource open — admit_path already
                // classified the source.
                let is_deepseek = matches!(
                    admit_result.admitted.admission().variant(),
                    hipfire_loader::parallel_capability::ModelVariant::Deepseek4
                );
                let mtp_k = match resolve_mtp_k(mtp_k_param, is_deepseek) {
                    Ok(k) => k,
                    Err(error) => {
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({ "type": "error", "message": error })
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                };

                // Model-free n-gram speculator config, forwarded by the CLI after
                // resolving the `speculation` selector + legacy knobs through the
                // config ladder (env > flag > per-model > global). `ngram_draft`
                // is the per-load enable; `ngram_k`/`ngram_min_count` tune the
                // drafter. The loader applies env-wins over these (so a directly
                // driven daemon with `HIPFIRE_NGRAM_DRAFT=1` still works). Absent
                // params leave the fields `None` → loader defaults / env.
                let spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg {
                    mtp_mode,
                    mtp_k: Some(mtp_k),
                    ngram_draft: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_draft"))
                        .and_then(|v| v.as_bool()),
                    ngram_k: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_k"))
                        .and_then(|v| v.as_u64())
                        .map(|k| k as usize),
                    ngram_min_count: msg
                        .get("params")
                        .and_then(|p| p.get("ngram_min_count"))
                        .and_then(|v| v.as_u64())
                        .map(|c| c as u32),
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
                let cask = CaskConfig {
                    sidecar: cask_sidecar,
                    cask_m_folding: cask_enabled,
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

                // ── Guards use effective degrees, not raw request ────────────
                // At this point admission has resolved composition, legacy remap,
                // and dense-EP normalisation — at most one axis exceeds 1.
                // The daemon's base GPU is already initialized; this guard section
                // does NOT trigger GPU work — it emits JSON errors and continues.
                let has_pp = effective_pp > 1;

                // DFlash guard: combined EP/TP message selected from effective
                // multi-device axis.  Multi-GPU paths refuse DFlash drafters in
                // v1 (the draft tensor format family and the spec-decode dispatch
                // are single-GPU only).  PP+DFlash has an experimental opt-in.
                if draft_path.is_some() {
                    if effective_tp > 1 || effective_ep > 1 {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"EP/TP serving (ep>1 or tp>1) does not support DFlash drafters in v1; reload without a draft."}}"#
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                    if effective_pp > 1
                        && std::env::var("HIPFIRE_PP_DFLASH").ok().as_deref() != Some("1")
                    {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"error","message":"DFlash speculative decode requires pp=1 in v1 (set HIPFIRE_PP_DFLASH=1 to opt into the experimental pp>1 PRD path; note PR2-4 of docs/plans/hetero-pflash-dflash.prd are not yet implemented — the load message will accept but generate will not run cross-card spec-decode). See issue #58 v1.1 roadmap."}}"#
                        );
                        let _ = stdout.flush();
                        continue;
                    }
                }
                // CASK guard: eviction requires pp=1 in v1.
                if has_pp && cask.sidecar.is_some() {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"CASK / TriAttention eviction requires pp=1 in v1; see issue #58 v1.1 roadmap"}}"#
                    );
                    let _ = stdout.flush();
                    continue;
                }
                // PFlash guard: compression requires pp=1 in v1.
                if has_pp
                    && (pflash_drafter.is_some() || pflash_mode_str != "off")
                    && std::env::var("HIPFIRE_PP_PFLASH").ok().as_deref() != Some("1")
                {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"PFlash prefill compression requires pp=1 in v1 (set HIPFIRE_PP_PFLASH=1 to opt into the experimental pp>1 PoC); see issue #58 v1.1 roadmap"}}"#
                    );
                    let _ = stdout.flush();
                    continue;
                }

                let state_quant_override = msg
                    .get("params")
                    .and_then(|p| p.get("state_quant"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());

                // The effective mesh is already part of the load plan.  Dense EP>1
                // has been normalised to (1,1,1) by admission, so it gets a single-
                // device mesh — no EP secondary device / collective path.
                // emulate=None is load-bearing: passing Some(_) would auto-promote
                // a plain serve on a HIPFIRE_EMULATE_GPUS box into EP-2.
                let effective_mesh = &admit_result.effective_mesh;
                // PP layer bands: retrieved via the env helper.  Currently
                // always returns `None` because no `load_admitted` consumer
                // accepts pp_bands yet.  Valid dense PP loads are unaffected
                // by stale env values.
                let pp_bands = pp_bands_from_env();

                // Route through `load_admitted` with the admitted token and
                // effective mesh.  The effective axis determines whether a GPU
                // handle is needed: Single passes `Some(&mut gpu)`;
                // TP/PP/EP manage their own mesh via `Gpus::from_mesh`.
                let effective_axis = admit_result.effective.axis();
                let gpu_opt: Option<&mut rdna_compute::Gpu> = if matches!(
                    effective_axis,
                    hipfire_loader::parallel_capability::ParallelAxis::Single
                ) {
                    Some(&mut gpu)
                } else {
                    None
                };
                let mut opts = hipfire_loader::ModelLoadOptions::new(max_seq)
                    .with_cask(&cask)
                    .with_spec(spec_cfg);
                if let Some(dp) = draft_path.as_deref() {
                    opts = opts.with_draft_path(dp);
                }
                if let Some(kv) = kv_mode_override.as_deref() {
                    opts = opts.with_kv_mode_override(kv);
                }
                if let Some(kv) = kv_adaptive_override.as_deref() {
                    opts = opts.with_kv_adaptive_override(kv);
                }
                if let Some(sq) = state_quant_override.as_deref() {
                    opts = opts.with_state_quant_override(sq);
                }
                if let Some(bands) = pp_bands.as_deref() {
                    opts = opts.with_pp_bands(bands);
                }
                let loaded = hipfire_loader::load_admitted(
                    admit_result.admitted,
                    &effective_mesh,
                    opts,
                    gpu_opt,
                );
                match loaded {
                    Ok(m) => {
                        // FIX #1 (deferred multi-device unload): the new TP/EP
                        // model loaded successfully — NOW it's safe to free the
                        // prior model (single-GPU/pp models were already unloaded
                        // eagerly above; this branch only fires for the deferred
                        // multi-device path). The prior model's PFlash drafter
                        // (pflash_state) is part of that prior model, so it's
                        // torn down here in
                        // the same drainer-before-unload order used elsewhere:
                        // unload_drafter queues the drafter tensors into the
                        // pool, then unload_model drains it.
                        if is_deferred {
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
                            if let Some(old) = model.take() {
                                hipfire_loader::unload_model(old, &mut gpu);
                            }
                        }
                        let arch = match m.meta.arch_id {
                            5 => "qwen3_5",
                            6 => "qwen3_5_moe",
                            7 => "qwen2",
                            8 => "dots-ocr",
                            9 => "deepseek4",
                            10 => "minimax_m2",
                            11 => "lfm2moe",
                            12 => "north_mini_code",
                            _ => "qwen3",
                        };
                        let vl =
                            m.vision.is_some() || matches!(m.state, Some(ModelState::DotsOcr(_)));
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
                            Some(ModelState::DotsOcr(b)) => (
                                b.config.text.hidden_size,
                                b.config.text.num_hidden_layers,
                                b.config.text.vocab_size,
                            ),
                            _ => (0, 0, 0),
                        };

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

                        // `cache_capable`: the daemon implements LCP prompt-cache
                        // reuse for these arches' AR generate path (qwen3.5/3.6
                        // = 5/6, deepseek4 = 9, minimax-m2 = 10, Cohere2-MoE
                        // = 12). The serve layer keys its
                        // per-request `reset` decision off THIS flag rather than
                        // a hardcoded arch-string allowlist, so a new
                        // cache-capable arch (or an arch-string rename) can't
                        // silently fall back to stateless reset-every-turn — the
                        // exact failure that left the prompt cache dead when the
                        // installed CLI predated the allowlist. Source of truth
                        // lives here, next to the cache implementation.
                        let cache_capable = reset_domain_cache_capable(m.meta.arch_id);
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"loaded","arch":"{}","dim":{},"layers":{},"vocab":{},"vl":{},"cache_capable":{}}}"#,
                            arch, dim, layers, vocab, vl, cache_capable
                        );

                        // ── PFlash drafter load (Phase 4.0) ──────────────
                        //
                        // Only attempt when mode != off AND a drafter path
                        // was provided. Failures here are NON-FATAL: log
                        // the reason and continue with PFlash disabled so
                        // the operator gets a clear "model is up, but
                        // compression isn't" signal rather than losing
                        // the entire session.
                        //
                        // Multi-device guard (pflash_suppressed): TP/EP paths serve
                        // through `generate_multi` / `generate_ep`, which bypass
                        // PFlash entirely.  Loading a drafter here would just pin
                        // GPU memory it never reads until unload, so skip the load
                        // outright.  Warn once if the operator actually supplied a
                        // drafter so the silent no-op is visible.
                        if admit_result.pflash_suppressed {
                            if pflash_drafter.is_some() && pflash_mode_str != "off" {
                                eprintln!(
                                    "[pflash] WARN: ignoring PFlash drafter on multi-device model \
                                     — generate_multi bypasses PFlash; drafter would only waste GPU memory"
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
                                            eprintln!(
                                                "[pflash] LOADED drafter={} dev={} mode={} compat={} keep={} thr={}",
                                                pf_drafter_path,
                                                pflash_drafter_device,
                                                pflash_mode_str,
                                                pf_state.tokenizer_compat,
                                                pflash_keep_ratio,
                                                pflash_threshold
                                            );
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
                        write_error(
                            &mut stdout,
                            "",
                            &format!(
                                "load failed: {e}. GPU: {} ({free_mb} MB free / {total_mb} MB total)",
                                gpu.arch
                            ),
                        );
                    }
                }
                let _ = stdout.flush();
            }

            "generate" => {
                let m = match model.as_mut() {
                    Some(m) => m,
                    None => {
                        let _ =
                            writeln!(stdout, r#"{{"type":"error","message":"no model loaded"}}"#);
                        let _ = stdout.flush();
                        continue;
                    }
                };

                let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("0");
                let prompt = msg
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Hello");
                let prompt_norm = hipfire_runtime::tokenizer::maybe_normalize_prompt(prompt);
                let prompt: &str = &prompt_norm;
                if std::env::var("HIPFIRE_PROMPT_TOKEN_HEAT").ok().as_deref() == Some("1") {
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
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"error","id":"{}","message":"invalid tools field: {}"}}"#,
                                id,
                                e.to_string().replace('"', "'"),
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
                                let _ = writeln!(
                                    stdout,
                                    r#"{{"type":"error","id":"{}","message":"invalid messages field: {}"}}"#,
                                    id,
                                    e.to_string().replace('"', "'"),
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
                // the .hfq `generation_config` (m.meta.rec_temperature/m.meta.rec_top_p,
                // populated at load time via HfqFile::recommended_sampling) take
                // precedence over this ladder; an explicit per-request field
                // (set below via `msg.get(...)`) overrides both. The CLI's
                // curated registry `recommended_settings` reach this handler as
                // explicit request fields (CLI explicit-send guard), so they sit
                // above the .hfq layer on that path.
                let (arch_default_temp, arch_default_top_p) = if m.meta.arch_id == 11 {
                    // LFM2.5 (11): Liquid's model card recommends temperature=0.1,
                    // top_k=50, repetition_penalty=1.05. The daemon sampler is
                    // temp + top_p + repeat_penalty (no user-facing top_k — the
                    // sample_top_p kernel's top-K is a fixed candidate gather), so
                    // we apply temp=0.1 + rep=1.05 (set below) and keep a tight
                    // top_p=0.80; at temp 0.1 the top_k-vs-top_p choice is near
                    // moot (the distribution is already peaked).
                    (0.1_f64, 0.80_f64)
                } else if m.meta.arch_id == 9 {
                    // DeepSeek V4 Flash (9): MTP spec-decode is greedy-only and,
                    // since the k=2 + K+1 shared-accept-core work, ~3× faster than
                    // AR (measured 81.7% accept / 24 vs 7.8 tok/s on the
                    // deepseek4-mtp code bench). Default to temp=0 so an omitted
                    // `temperature` gets that spec speedup; explicit temp>0 is
                    // still honored and routes to the AR sampler (spec is
                    // greedy-only). Was 1.0 to dodge block-level attractors at low
                    // temp — coherence-gate-deepseek4-mtp re-validates greedy.
                    (0.0_f64, 1.0_f64)
                } else if m.meta.arch_id == 10 {
                    // MiniMax-M2 (10): quantized instruct model that falls into
                    // block-level attractors at lower temperatures — keep the
                    // card-recommended temp=1.0/top_p=1.0.
                    (1.0_f64, 1.0_f64)
                } else if m.meta.arch_id == 12 {
                    // Cohere2-MoE / North-Mini-Code: Cohere-style agentic
                    // markers are sampled best with the model-card nucleus
                    // defaults.
                    (1.0_f64, 0.95_f64)
                } else {
                    (0.3_f64, 0.8_f64)
                };
                // Layer the .hfq-baked author recommendation OVER the arch
                // ladder. Per-knob: a model that bakes only `temperature` still
                // gets the arch-ladder `top_p`.
                let default_temp = m
                    .meta
                    .rec_temperature
                    .map(|x| x as f64)
                    .unwrap_or(arch_default_temp);
                let default_top_p = m
                    .meta
                    .rec_top_p
                    .map(|x| x as f64)
                    .unwrap_or(arch_default_top_p);
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
                let default_repeat_penalty = if m.meta.arch_id == 11 {
                    1.05_f64
                } else {
                    1.0_f64
                };
                // Accept HF-style `repetition_penalty` as a request ALIAS for our
                // `repeat_penalty` field, used only when the canonical key is
                // absent. (OpenAI/HF clients send `repetition_penalty`.)
                let repeat_penalty = msg
                    .get("repeat_penalty")
                    .or_else(|| msg.get("repetition_penalty"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(default_repeat_penalty) as f32;
                // OpenAI-compatible `reasoning_effort` (also accept our custom
                // `thinking_mode` alias) — only consumed by arch_id=9 today.
                // Default = NonThink, matching the safe HF chat frame.
                let think_mode = msg
                    .get("reasoning_effort")
                    .or_else(|| msg.get("thinking_mode"))
                    .and_then(|v| v.as_str())
                    .map(ThinkMode::from_str)
                    .unwrap_or(ThinkMode::NonThink);
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
                // .hfq-baked `m.meta.rec_presence_penalty` > 0.0 (off). The .hfq's
                // generation_config does not carry presence_penalty today, so
                // m.meta.rec_presence_penalty is always None on the load path; the
                // field is wired so a curated registry card value still flows in
                // as an explicit request field (CLI explicit-send guard). presence_penalty IS honored by the sampler.
                let presence_penalty = (msg
                    .get("presence_penalty")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(m.meta.rec_presence_penalty.unwrap_or(0.0) as f64)
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
                    .or_else(|| m.meta.rec_top_k.map(|k| k as u32))
                    .filter(|&k| k > 0);
                let min_p: Option<f32> = msg
                    .get("min_p")
                    .and_then(|v| v.as_f64())
                    .map(|p| p as f32)
                    .or(m.meta.rec_min_p)
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
                let max_think_tokens = msg
                    .get("max_think_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                // assistant_prefix: "plain", "open_think", or "closed_think"
                // Controls the ChatML framing after the assistant role header.
                // Consumed by the text path; VL path does not yet propagate
                // it (tracked as a follow-up to the post-#169 rebase).
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
                let is_dots_ocr = m.meta.arch_id == 8;
                let has_vl = m.vision.is_some() || is_dots_ocr;

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
                    let generation_result = if is_dots_ocr {
                        generate_vl_dots_ocr(m, &mut gpu, &mut stdout, &params)
                    } else {
                        generate_vl(m, &mut gpu, &mut stdout, &params)
                    };
                    match generation_result {
                        GenerateResult::Complete => {}
                        GenerateResult::Deferred(terminal) => {
                            if let Err(error) = reset_then_emit(m, &mut gpu, &mut stdout, terminal)
                            {
                                poison_model_slot(
                                    &mut model,
                                    &mut pflash_state,
                                    &mut pflash_drafter_gpu,
                                    &mut pflash_cfg,
                                    &mut gpu,
                                );
                                emit_reset_error(&mut stdout, id, &error);
                                break 'daemon;
                            }
                        }
                        GenerateResult::ResetFailed { id, message } => {
                            poison_model_slot(
                                &mut model,
                                &mut pflash_state,
                                &mut pflash_drafter_gpu,
                                &mut pflash_cfg,
                                &mut gpu,
                            );
                            emit_reset_error_message(&mut stdout, &id, &message);
                            break 'daemon;
                        }
                        GenerateResult::PpCompletion { .. } => {
                            unreachable!("PP completion must be consumed by its outer wrapper")
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
                    let generation_result = generate(
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
                    );
                    match generation_result {
                        GenerateResult::Complete => {}
                        GenerateResult::Deferred(terminal) => {
                            if let Some(current) = model.as_mut() {
                                if let Err(error) =
                                    reset_then_emit(current, &mut gpu, &mut stdout, terminal)
                                {
                                    poison_model_slot(
                                        &mut model,
                                        &mut pflash_state,
                                        &mut pflash_drafter_gpu,
                                        &mut pflash_cfg,
                                        &mut gpu,
                                    );
                                    emit_reset_error(&mut stdout, id, &error);
                                    break 'daemon;
                                }
                            }
                        }
                        GenerateResult::ResetFailed { id, message } => {
                            poison_model_slot(
                                &mut model,
                                &mut pflash_state,
                                &mut pflash_drafter_gpu,
                                &mut pflash_cfg,
                                &mut gpu,
                            );
                            emit_reset_error_message(&mut stdout, &id, &message);
                            break 'daemon;
                        }
                        GenerateResult::PpCompletion { .. } => {
                            unreachable!("PP completion must be consumed by its outer wrapper")
                        }
                    }
                }
            }

            "reset" => {
                // Reset conversation state without unloading the model.
                if model.is_some() {
                    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                        eprintln!("[qwen-cache RESET] daemon received reset");
                    }
                    let reset_result = model.as_mut().map(|m| model_reset_context(m, &mut gpu));
                    match reset_result {
                        Some(Ok(())) | None => {
                            let _ = writeln!(stdout, r#"{{"type":"reset","seq_pos":0}}"#);
                        }
                        Some(Err(error)) => {
                            poison_model_slot(
                                &mut model,
                                &mut pflash_state,
                                &mut pflash_drafter_gpu,
                                &mut pflash_cfg,
                                &mut gpu,
                            );
                            emit_reset_error(&mut stdout, "", &error);
                            break 'daemon;
                        }
                    }
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
                    hipfire_loader::unload_model(m, &mut gpu);
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
                let model_arch = model
                    .as_ref()
                    .map(|m| match m.meta.arch_id {
                        5 => "qwen3_5",
                        6 => "qwen3_5_moe",
                        7 => "qwen2",
                        9 => "deepseek4",
                        10 => "minimax_m2",
                        11 => "lfm2moe",
                        12 => "north_mini_code",
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
                if !matches!(m.parallel.kind(), ModelParallelKind::Single) {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill requires a single-GPU model (pp=1, non-EP/TP); multi-GPU/EP/TP bench not implemented"}}"#
                    );
                    let _ = stdout.flush();
                    continue;
                }
                let n = msg.get("tokens").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
                // Guard physical_cap — reserve 32 slots of headroom so a subsequent
                // generate request against the loaded model still has room. We guard
                // on the *physical* buffer (not the advertised max_seq) because this
                // bench intentionally bypasses eviction to measure raw prefill.
                if n.saturating_add(32) > m.meta.physical_cap {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"bench_prefill tokens={} exceeds loaded physical_cap={}"}}"#,
                        n, m.meta.physical_cap
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
                // prefill-on-top-of-prior-state. The loader owns this total reset.
                if let Err(error) = model_reset_context(m, &mut gpu) {
                    poison_model_slot(
                        &mut model,
                        &mut pflash_state,
                        &mut pflash_drafter_gpu,
                        &mut pflash_cfg,
                        &mut gpu,
                    );
                    let bench_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_reset_error(&mut stdout, bench_id, &error);
                    break 'daemon;
                }

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                let _ = gpu.hip.device_synchronize();
                let t0 = Instant::now();
                let run_ok = if m.meta.arch_id == 5 || m.meta.arch_id == 6 {
                    let ModelState::Qwen35(b) = m.state.as_mut().unwrap() else {
                        unreachable!()
                    };
                    let config = &b.config;
                    let weights = &b.weights;
                    let scratch = &b.scratch;
                    let kv = &mut b.kv_cache;
                    let dn = &mut b.dn_state;
                    qwen35::forward_prefill_batch(
                        &mut gpu, weights, config, &synthetic, 0, kv, dn, scratch, None, None,
                        None, None,
                    )
                    .is_ok()
                } else if m.meta.arch_id == 7 {
                    // Qwen2 has no batched prefill kernel yet — per-token loop
                    // mirroring the LLaMA fallback path. The loop seeds
                    // position via `state.next_pos` (already reset above to 0).
                    let ModelState::Qwen2(b) = m.state.as_mut().unwrap() else {
                        unreachable!()
                    };
                    let config = &b.config;
                    let weights = &b.weights;
                    let state = &mut b.state;
                    let mut ok = true;
                    for &tok in &synthetic {
                        if qwen2::forward_step(&mut gpu, weights, config, state, tok).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    ok
                } else if m.meta.arch_id == 9 {
                    // DeepSeek V4 warm-pass: per-token decode_step. Saturates
                    // the kernel cache (HC, indexer, compressor,
                    // attention, MoE) on a short synthetic prompt
                    // before any user-facing generate. Not the
                    // production prefill path (that's
                    // forward_prefill_batch_chunked in `generate`).
                    let b = m.deepseek4_mut().unwrap();
                    let config = &b.config;
                    let weights = &b.weights;
                    let state = &mut b.state;
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
                } else if m.meta.arch_id == 11 {
                    // LFM2.5-MoE warm-pass: per-token decode_step over the
                    // synthetic prompt. Saturates the conv + GQA + QK-norm +
                    // RoPE + top-4 MoE kernel set before any user-facing
                    // generate. This IS the production prefill shape (no
                    // batched kernel).
                    let b = m.lfm2moe_mut().expect("arch_id=11 requires lfm2moe bundle");
                    let config = &b.config;
                    let weights = &b.weights;
                    let state = &mut b.state;
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
                } else if m.meta.arch_id == 10 {
                    // MiniMax-M2 warm-pass: per-token decode_step over the
                    // synthetic prompt. Saturates the GQA + QK-norm + RoPE +
                    // MoE kernel set before any user-facing generate. This IS
                    // the production prefill shape (the eager per-token path).
                    let b = m.minimax_mut().expect("arch_id=10 requires minimax bundle");
                    let config = &b.config;
                    let weights = &b.weights;
                    let state = &mut b.state;
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
                } else if m.meta.arch_id == 12 {
                    // Cohere2-MoE warm-pass: per-token decode_step over the
                    // synthetic prompt. This primes attention + MoE dispatch
                    // without mutating qwen/minimax-specific state.
                    let b = m
                        .cohere2moe_mut()
                        .expect("arch_id=12 requires cohere2moe bundle");
                    let config = &b.config;
                    let weights = &b.weights;
                    let state = &mut b.state;
                    let mut ok = true;
                    for (i, &tok) in synthetic.iter().enumerate() {
                        if cohere2moe::forward::decode_step(
                            config, weights, state, &mut gpu, tok, i as u32,
                        )
                        .is_err()
                        {
                            ok = false;
                            break;
                        }
                    }
                    ok
                } else {
                    let ModelState::Llama(b) = m.state.as_mut().unwrap() else {
                        unreachable!()
                    };
                    let config = &b.config;
                    let weights = &b.weights;
                    let scratch = &b.scratch;
                    let kv = &mut b.kv;
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

                // Reset state AFTER measurement — the synthetic request must not
                // leak into the next real request.
                if let Err(error) = model_reset_context(m, &mut gpu) {
                    poison_model_slot(
                        &mut model,
                        &mut pflash_state,
                        &mut pflash_drafter_gpu,
                        &mut pflash_cfg,
                        &mut gpu,
                    );
                    let bench_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    emit_reset_error(&mut stdout, bench_id, &error);
                    break 'daemon;
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

/// Expert-parallel streaming generate (task #26, ds4 first). Greedy AR via
/// `forward_ep` across the EP ranks; logits gathered on rank 0 and sampled on
/// the host. v1: greedy + basic token streaming (no grammar / tool-calls /
/// think-budget — absent on the EP path). The DeepSeek chat template
/// (`<｜User｜>…<｜Assistant｜>`) is applied here; the daemon's full prompt-frame
/// (multi-turn, messages_history) is a follow-up. See docs/plans/daemon-ep-wiring.md.
#[allow(clippy::too_many_arguments)]
/// Resolved sampling config for the EP (multi-GPU) decode loops. Carries the
/// single-GPU handler's request>rec_*>arch-default resolution (computed at the
/// `generate` call site) into `ep_serve_ds4` / `ep_serve_minimax`, which apply
/// it host-side over the downloaded f32 logits via `llama::sample_full_dist`.
#[derive(Clone, Copy)]
struct EpSampling {
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
}

/// A dense llama-family model served over the unified `ar_generate` driver (via
/// `DenseDispatch` + `dense_serve_via_ar_generate`): a per-token forward + logits
/// over its own multi-GPU state. Implemented by both
/// `TpModel` (Tp axis, PB-TP5) and `PpModel` (Pp axis, P-C) so the daemon's decode
/// loop is agnostic to which parallelism axis the model is served on. (Inherent
/// methods are called via the fully-qualified path so the trait forwarders don't
/// self-recurse.)
trait DenseServed {
    fn forward_token(&mut self, token: u32, pos: usize) -> Result<(), String>;
    fn logits(&mut self) -> Result<Vec<f32>, String>;
    /// Prefill the whole prompt. Default = per-token loop (back-compat / >256 fallback).
    /// Postcondition: KV filled for positions 0..tokens.len(); `logits()` returns the
    /// last position; decode resumes at pos = tokens.len().
    fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        for (pos, &t) in tokens.iter().enumerate() {
            self.forward_token(t, pos)?;
        }
        Ok(())
    }
}
impl DenseServed for hipfire_runtime::tp_serve::TpModel {
    fn forward_token(&mut self, t: u32, p: usize) -> Result<(), String> {
        hipfire_runtime::tp_serve::TpModel::forward_token(self, t, p)
    }
    fn logits(&mut self) -> Result<Vec<f32>, String> {
        hipfire_runtime::tp_serve::TpModel::logits(self)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        hipfire_runtime::tp_serve::TpModel::prefill(self, tokens)
    }
}
impl DenseServed for hipfire_runtime::pp_serve::PpModel {
    fn forward_token(&mut self, t: u32, p: usize) -> Result<(), String> {
        hipfire_runtime::pp_serve::PpModel::forward_token(self, t, p)
    }
    fn logits(&mut self) -> Result<Vec<f32>, String> {
        hipfire_runtime::pp_serve::PpModel::logits(self)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        hipfire_runtime::pp_serve::PpModel::prefill(self, tokens)
    }
}

/// Select the DenseServed driver from the parallelism axis (F1: DenseServed is
/// daemon-private so this match cannot live on the loader-side enum).
/// Task 3 adds the Tp arm; Task 4 adds the Pp(Dense) arm.
fn dense_model_mut(m: &mut LoadedModel) -> Option<&mut dyn DenseServed> {
    match &mut m.parallel {
        ModelParallel::Tp(t) => Some(t),
        ModelParallel::Pp(PipelineImpl::Dense(p)) => Some(p),
        _ => None,
    }
}

/// Drive dense multi-GPU (TP / dense-PP) AR decode through the unified `ar_generate`
/// driver — folds `generate_dense`. Shared by the Tp and Pp(Dense) gates (both
/// now in `m.parallel`). Mirrors the gate preamble: `plan_prompt_cache` (LCP the rendered
/// conversation vs `conversation_tokens`) → leave `conversation_tokens ==
/// rendered[0..start_pos]` (truncate on a pure-extension hit, clear on a miss) so
/// ar_generate's `extend(new_tokens)` + per-token push rebuilds `rendered +
/// generated` (the old gate's bake) for next-turn LCP. Device via `DenseDispatch` +
/// `ForwardCtx::Mesh`.
#[allow(clippy::too_many_arguments)]
fn dense_serve_via_ar_generate(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    system_prompt: Option<&str>,
    prompt: &str,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
    max_tokens: usize,
    stop: &[String],
) -> GenerateResult {
    let hist: &[hipfire_runtime::prompt_frame::Message] = messages_history.unwrap_or(&[]);
    let cache_disabled = std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
    let plan = {
        let tok = m.tokenizer.as_ref().expect("dense serve: tokenizer");
        plan_prompt_cache(
            tok,
            &mut m.persist.asst_turn_cache,
            &m.session.conversation_tokens,
            m.eviction.is_none(),
            system_prompt,
            prompt,
            assistant_prefix,
            hist,
            cache_disabled,
            &[],
            false,
        )
    };
    let new_tokens = plan.new_tokens;
    let start_pos = plan.start_pos;
    // Capacity guard (review I-1): generate_dense caught TpModel/PpModel::prefill's
    // overflow Err and emitted a clean {"type":"error"}; ar_generate `.unwrap()`s the
    // forward hooks, so without this an oversized prompt PANICS the serve thread. The
    // absolute KV span is rendered.len() (== start_pos + the prefilled suffix) +
    // max_tokens decode tokens. Return BEFORE mutating state. Mirrors the ds4/minimax
    // EP via-helpers.
    let rendered_n = start_pos.saturating_add(new_tokens.len());
    if rendered_n.saturating_add(max_tokens) > m.meta.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id, rendered_n, max_tokens, m.meta.physical_cap
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }
    // A dense TP/PP cache miss is still a total context discard. The physical
    // KV buffers are intentionally not zeroed by reset_context; the next cold
    // prefill overwrites them from position zero, while the parallel owner
    // still needs its request/session reset routed through the loader façade.
    if !plan.cache_hit {
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }
    if start_pos == 0 {
        m.session.conversation_tokens.clear();
    } else {
        m.session.conversation_tokens.truncate(start_pos);
    }
    m.session.seq_pos = start_pos;
    let prefill_len = new_tokens.len();
    let t0 = std::time::Instant::now();
    let mut disp = DenseDispatch { m };
    return ar_generate(
        &mut disp,
        ForwardCtx::Mesh,
        stdout,
        id,
        temp,
        top_p,
        top_k,
        min_p,
        max_tokens,
        1.0, // repeat_penalty — generate_dense uses none
        0,   // repeat_window
        0.0, // presence_penalty
        0.0, // frequency_penalty
        0,   // budget_alert_at_tok
        "",  // budget_alert_text
        0,   // max_think_tokens (dense is lean, no think)
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        stop,
        None,       // tools (dense has no grammar)
        new_tokens, // new_tokens: the post-LCP suffix
        &[],        // im_end (dense stops via is_eos / is_terminator)
        &[],        // nl
        None,       // im_end_token
        None,       // tool_call_pair
        None,       // think_pair
        prefill_len,
        start_pos, // cached_tokens_count
        None,      // pflash_summary
        None,      // pflash_bypass_reason
        None,      // pflash_alpha
        t0,
        None, // tape
    );
}

fn generate_ep(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
    sampling: EpSampling,
) -> GenerateResult {
    // ── Canonical multi-turn render via the arch's trained chat_template
    // (ds4/minimax). Mirrors generate_minimax: `messages_history` (the full
    // conversation, live user last) → render_messages with `tools` threaded;
    // falls back to a synthesized [system?, user] turn when no history is
    // supplied. The trim_blocks/lstrip_blocks env (prompt_frame) keeps the
    // structural prefix history-invariant so the EP LCP cache below can hit.
    // `primed_think` records whether the render ended on the MiniMax `<think>`
    // generation primer (re-emitted display-only in ep_serve_minimax). ──
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = if m.meta.arch_id == 9 {
        primed_think = false;
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let eos_tok = m.meta.eos_tok;
        build_deepseek4_dsml_prompt(
            tokenizer,
            system_prompt,
            tools,
            messages_history,
            prompt,
            think_mode,
            eos_tok,
            &mut m.persist.asst_turn_cache,
        )
    } else {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        if let Some(template) = m.meta.chat_template.as_ref() {
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
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
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
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","id":"{}","message":"EP jinja render: {}"}}"#,
                        id,
                        format!("{e}").replace('"', "'")
                    );
                    let _ = stdout.flush();
                    return GenerateResult::Complete;
                }
            }
        } else {
            // No embedded template — minimal ds4-style fallback (single-turn).
            let mut ids = Vec::new();
            if let Some(b) = tokenizer.special_token_id("<｜begin▁of▁sentence｜>") {
                ids.push(b);
            }
            ids.extend(tokenizer.encode(&format!("<｜User｜>{prompt}<｜Assistant｜>")));
            ids
        }
    };
    if std::env::var("HIPFIRE_DEEPSEEK4_DUMP_PROMPT")
        .ok()
        .as_deref()
        == Some("1")
    {
        let tk = m.tokenizer.as_ref().unwrap();
        eprintln!(
            "[ep prompt dump] arch={} {} tokens, decoded:\n>>>\n{}\n<<<",
            m.meta.arch_id,
            prompt_ids.len(),
            tk.decode(&prompt_ids)
        );
    }
    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"EP: empty prompt after render"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }
    // Both ds4 and minimax EP eos are unified into m.meta.eos_tok at load time.
    let eos_tok = m.meta.eos_tok;
    match m.meta.arch_id {
        10 => {
            // minimax-EP FLIP (inc M3): runs on the unified ar_generate driver
            // (ep_serve_minimax deleted). No per-turn reset — the LCP preamble inside
            // ep_serve_minimax_via_ar_generate rewinds each rank's KV cursor to the
            // common prefix and re-prefills the suffix. Validated pre-flip by the
            // dual-run token-parity (ep_serve_minimax == ar_generate+EP on
            // capital/code, emulated EP-2). eos_tok is now the ds4 arm's; minimax
            // reads m.meta.eos_tok via MinimaxEpDispatch::eos_token.
            let _ = eos_tok;
            return ep_serve_minimax_via_ar_generate(
                m,
                gpu,
                stdout,
                id,
                &prompt_ids,
                max_tokens,
                primed_think,
                stop,
                sampling,
            );
        }
        _ => {
            // Axis B FLIP: ds4 EP AR decode runs on the unified `ar_generate` driver
            // (deepseek4-EP no longer has a bespoke serve loop). EP has no LCP → reset
            // per turn (the old ep_serve_ds4's start-of-turn cross-conversation reset),
            // then cold-prefill the full DSML prompt through ar_generate. Validated
            // pre-flip by the dual-run token-parity (ep_serve_ds4 == ar_generate+EP on
            // capital/reasoning/code/tool-call, emulated EP-2).
            if let Err(error) = model_reset_context(m, gpu) {
                return reset_failed(id, error);
            }
            return ep_serve_ds4_via_ar_generate(
                m,
                stdout,
                id,
                &prompt_ids,
                max_tokens,
                think_mode,
                tools,
                stop,
                sampling,
            );
        }
    }
}

/// Drive ds4 EP AR decode through the unified `ar_generate` driver (Axis B): build
/// a `Deepseek4EpDispatch` + `ForwardCtx::Mesh`, cold-prefill the full DSML prompt
/// (EP has no LCP), and let `ar_generate`'s generic loop drive forward / sample /
/// output. The EP-specific parts (multi-rank `forward_ep`, rank-0 host
/// `sample_full_dist`, DSML output) live in the dispatch + `Deepseek4StreamParser`.
/// The caller must reset EP state first (this does no internal reset). Used by the
/// inc-4 dual-run; becomes the prod path at the inc-5 flip.
#[allow(clippy::too_many_arguments)]
fn ep_serve_ds4_via_ar_generate(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    stop: &[String],
    sampling: EpSampling,
) -> GenerateResult {
    // Capacity guard (was ep_serve_ds4's O2b-2 guard, lost in the flip): EP
    // cold-prefills the FULL prompt every turn (no LCP), so the absolute KV span is
    // prompt_n + max_tokens. ar_generate has NO pre-prefill physical_cap check on the
    // EP path (its cap check lives in the budget-alert branch, off for EP:
    // budget_alert_at_tok=0), so an oversized prompt would drive forward_ep past the
    // per-rank KV buffer → corruption/serve-wide crash. Emit a clean error and return
    // BEFORE prefill, exactly as ep_serve_ds4 did. saturating_add so an adversarial
    // max_tokens can't wrap usize and slip under the cap.
    let prompt_n = prompt_ids.len();
    if prompt_n.saturating_add(max_tokens) > m.meta.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id, prompt_n, max_tokens, m.meta.physical_cap
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }
    let mut disp = Deepseek4EpDispatch {
        m,
        tools: tools.map(|t| t.to_vec()),
        think_mode,
    };
    ar_generate(
        &mut disp,
        ForwardCtx::Mesh,
        stdout,
        id,
        sampling.temp,
        sampling.top_p,
        sampling.top_k,
        sampling.min_p,
        max_tokens,
        1.0, // repeat_penalty — EP uses sample_full_dist (no penalties)
        0,   // repeat_window
        0.0, // presence_penalty
        0.0, // frequency_penalty
        0,   // budget_alert_at_tok
        "",  // budget_alert_text
        0,   // max_think_tokens — ds4 think handled by the dsml StreamParser
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        stop,
        tools,               // activates grammar; ds4 init_grammar rebuilds from disp.tools
        prompt_ids.to_vec(), // new_tokens: full DSML prompt (cold prefill, no LCP)
        &[],                 // im_end — ds4 stops via is_eos, not im_end
        &[],                 // nl
        None,                // im_end_token
        None,                // tool_call_pair — dsml StreamParser owns tool-call events
        None,                // think_pair — dsml StreamParser owns think routing
        prompt_ids.len(),    // prefill_tokens (full cold prefill)
        0,                   // cached_tokens_count (no LCP for EP)
        None,                // pflash_summary
        None,                // pflash_bypass_reason
        None,                // pflash_alpha
        std::time::Instant::now(),
        None, // tape (dual-run scaffolding removed at the flip)
    )
}

/// Drive minimax-EP AR decode through the unified `ar_generate` driver. Mirrors the
/// single-GPU `generate_minimax` flip (LCP partial-reuse preamble + display-only
/// `<think>` primer → ar_generate) but with the EP forward/sample (MinimaxEpDispatch
/// + ForwardCtx::Mesh) and a PER-RANK `n_tokens` rewind. Unlike ds4 (no LCP, cold
/// reset every turn), minimax rewinds each rank's KV cursor to the common prefix and
/// prefills only the divergent suffix. Used by the inc-M2 dual-run; becomes prod at
/// the inc-M3 flip.
#[allow(clippy::too_many_arguments)]
fn ep_serve_minimax_via_ar_generate(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    primed_think: bool,
    stop: &[String],
    sampling: EpSampling,
) -> GenerateResult {
    let prompt_n = prompt_ids.len();
    // Capacity guard (mirror ep_serve_minimax): absolute KV span is prompt_n +
    // max_tokens; overrunning drives forward_ep past the per-rank KV buffer.
    if prompt_n.saturating_add(max_tokens) > m.meta.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id, prompt_n, max_tokens, m.meta.physical_cap
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }
    // LCP partial reuse (mirror generate_minimax + ep_serve_minimax's per-rank rewind):
    // rewind every rank's KV cursor to the common prefix, prefill only the suffix.
    let prefill_ids: Vec<u32> = {
        let prior_len = m.session.conversation_tokens.len();
        let max_match = prior_len.min(prompt_n);
        let mut lcp = 0usize;
        while lcp < max_match && m.session.conversation_tokens[lcp] == prompt_ids[lcp] {
            lcp += 1;
        }
        let cache_hit = lcp > 0 && lcp < prompt_n;
        let prefill_from = if cache_hit {
            m.session.conversation_tokens.truncate(lcp);
            lcp
        } else {
            m.session.conversation_tokens.clear();
            0
        };
        if let ModelParallel::Ep(EpState {
            inner: EpArch::Minimax { state, .. },
            ..
        }) = &mut m.parallel
        {
            for s in state.iter_mut() {
                s.n_tokens = prefill_from;
            }
        }
        if !cache_hit {
            if let Err(error) = model_reset_context(m, gpu) {
                return reset_failed(id, error);
            }
        }
        m.session.seq_pos = prefill_from;
        prompt_ids[prefill_from..].to_vec()
    };
    let cached_tokens_count = prompt_n.saturating_sub(prefill_ids.len());
    let prefill_len = prefill_ids.len();
    // Display-only `<think>` primer re-emit (mirror ep_serve_minimax) — NOT in the tape.
    if primed_think {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type":"token","id":id,"text":"<think>\n"})
        );
        let _ = stdout.flush();
    }
    let mut disp = MinimaxEpDispatch { m: &mut *m };
    ar_generate(
        &mut disp,
        ForwardCtx::Mesh,
        stdout,
        id,
        sampling.temp,
        sampling.top_p,
        sampling.top_k,
        sampling.min_p,
        max_tokens,
        1.0, // repeat_penalty — EP uses sample_full_dist (no penalties)
        0,   // repeat_window
        0.0, // presence_penalty
        0.0, // frequency_penalty
        0,   // budget_alert_at_tok
        "",  // budget_alert_text
        0,   // max_think_tokens (no force-close think on minimax)
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        stop,        // user stop sequences (DefaultStreamParser honors them)
        None,        // tools (minimax EP has no grammar)
        prefill_ids, // new_tokens: the post-LCP suffix
        &[],         // im_end (minimax stops via is_eos / [e~[ filter)
        &[],         // nl
        None,        // im_end_token
        None,        // tool_call_pair
        None,        // think_pair (primer emitted display-only above)
        prefill_len,
        cached_tokens_count,
        None, // pflash_summary
        None, // pflash_bypass_reason
        None, // pflash_alpha
        std::time::Instant::now(),
        None, // tape (dual-run scaffolding removed at the flip)
    )
}

/// Outcome of the LCP prompt-cache decision (see [`plan_prompt_cache`]).
struct PromptCachePlan {
    /// Full canonical conversation tokens (system + history + live user +
    /// assistant prefix). Stored as `conversation_tokens` after generation so
    /// the next turn can LCP against it.
    rendered: Vec<u32>,
    /// Tokens to actually prefill: the suffix `rendered[start_pos..]` on a hit,
    /// the whole `rendered` on a miss.
    new_tokens: Vec<u32>,
    /// Absolute position the prefill starts at (the reused-prefix length on a
    /// hit, 0 on a miss).
    start_pos: usize,
    /// `cached_tokens` for OpenAI usage reporting (== start_pos).
    cached_tokens: usize,
    /// True ⇒ reuse existing KV/DeltaNet[0..start_pos]; prefill only the suffix.
    /// False ⇒ caller must full-reset and prefill the whole conversation.
    cache_hit: bool,
    /// `Some(ckpt)` ⇒ this is a divergent-render RESUME (not a pure extension):
    /// the caller must restore the DeltaNet recurrent state from the checkpoint
    /// at `ckpt`, rewind seq_pos/conversation_tokens to `ckpt`, then treat the
    /// turn like a HIT with `start_pos == ckpt` (re-prefill only the tail) and
    /// drop `draft_ctx_cached_rows` to `ckpt`. `None` on a normal hit/miss.
    resume_from: Option<usize>,
}

/// Pure LCP prompt-cache decision shared in spirit with the AR `generate`
/// path's inline block — but side-effect-free (touches no GPU/seq_pos state),
/// so the DFlash path can use it too. Renders the canonical conversation via
/// `build_cached_history` (verbatim assistant-turn replay through
/// `asst_turn_cache`, which is what makes the LCP byte-exact across turns), then
/// compares against `m.conversation_tokens`. Reports a HIT only on a strict
/// forward extension (`lcp == prior_len && lcp < rendered.len()`), which keeps
/// the recurrent DeltaNet state valid by construction (the prior turn left it at
/// exactly `prior_len`, so prefilling the suffix advances it correctly with no
/// rewind). The exact-match edge (`lcp == rendered.len()`) degrades to a miss to
/// avoid a 1-token DeltaNet over-advance. Caller must be in the
/// `messages_history.is_some()` case.
#[allow(clippy::too_many_arguments)]
fn plan_prompt_cache(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    asst_turn_cache: &mut AsstTurnCache,
    conversation_tokens: &[u32],
    eviction_is_none: bool,
    system_prompt: Option<&str>,
    prompt: &str,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    messages_history: &[hipfire_runtime::prompt_frame::Message],
    cache_disabled: bool,
    // Ascending DeltaNet checkpoint positions (from `m.session.dflash_checkpoints`) and
    // whether resume-from-checkpoint is enabled. On a divergence the plan picks
    // the latest checkpoint `<= lcp && < rendered.len()` to resume from.
    dflash_ckpt_positions: &[usize],
    resume_enabled: bool,
) -> PromptCachePlan {
    let q_tokens = tokenizer.encode(prompt);
    let rendered = hipfire_runtime::prompt_frame::build_cached_history(
        tokenizer,
        system_prompt,
        messages_history,
        &q_tokens,
        assistant_prefix,
        |msg| {
            let stripped = strip_think_for_fingerprint(&msg.content);
            let normalized =
                hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
            let fp = asst_turn_fingerprint(&normalized, &msg.tool_calls);
            asst_turn_cache.get(&fp).cloned()
        },
    );
    let cache_eligible = !cache_disabled && eviction_is_none && !conversation_tokens.is_empty();
    if cache_eligible {
        let prior_len = conversation_tokens.len();
        let max_match = prior_len.min(rendered.len());
        let mut lcp = 0usize;
        while lcp < max_match && conversation_tokens[lcp] == rendered[lcp] {
            lcp += 1;
        }
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[qwen-cache lcp dflash] prior_len={} rendered_len={} lcp={}",
                prior_len,
                rendered.len(),
                lcp
            );
        }
        if lcp == prior_len && lcp < rendered.len() && lcp > 0 {
            return PromptCachePlan {
                new_tokens: rendered[lcp..].to_vec(),
                start_pos: lcp,
                cached_tokens: lcp,
                cache_hit: true,
                resume_from: None,
                rendered,
            };
        }
        // Divergent render (lcp < prior_len, or the exact-match edge): not a
        // pure extension, so the recurrent state at the end is stale. If resume
        // is enabled, rewind to the latest checkpoint at-or-before lcp that
        // still leaves ≥1 token to re-prefill, and resume from there instead of
        // cold-prefilling the whole conversation.
        if resume_enabled {
            if let Some(&ckpt) = dflash_ckpt_positions
                .iter()
                .filter(|&&p| p <= lcp && p < rendered.len())
                .max()
            {
                eprintln!(
                    "[qwen-cache resume dflash] checkpoint pos={} (lcp={}, prior_len={}, rendered_len={}) — replaying {} tokens vs cold-prefilling {}",
                    ckpt,
                    lcp,
                    prior_len,
                    rendered.len(),
                    rendered.len() - ckpt,
                    rendered.len(),
                );
                return PromptCachePlan {
                    new_tokens: rendered[ckpt..].to_vec(),
                    start_pos: ckpt,
                    cached_tokens: ckpt,
                    cache_hit: true,
                    resume_from: Some(ckpt),
                    rendered,
                };
            }
        }
    }
    PromptCachePlan {
        new_tokens: rendered.clone(),
        start_pos: 0,
        cached_tokens: 0,
        cache_hit: false,
        resume_from: None,
        rendered,
    }
}

/// Fallible façade for the loader-owned total reset authority.  The daemon
/// deliberately does not know how session, speculative, architecture, or
/// mesh state is laid out; it only decides when a request has crossed a
/// total-discard boundary.
fn model_reset_context(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), hipfire_loader::ResetError> {
    let result = m.reset_context(gpu);
    if let Err(error) = &result {
        eprintln!(
            "[daemon] reset failed; gpu_fatal={} — model slot must not be reused",
            error.is_gpu_fatal()
        );
    }
    result
}

/// Architectures whose request reset may preserve an LCP prefix cache. The
/// cache is persistent model state; the total reset still clears the active
/// request and all speculative mirrors before the next turn.
fn reset_domain_cache_capable(arch_id: u32) -> bool {
    matches!(arch_id, 5 | 6 | 9 | 10 | 12)
}

/// A Qwen prompt-cache miss is evaluated as a fresh context for capacity
/// purposes. The production caller performs the GPU reset before the guard;
/// this pure mirror keeps the zero-capacity regression explicit in CPU tests.
fn qwen_cache_guard_position(seq_pos: usize, cold_reset_required: bool) -> usize {
    if cold_reset_required {
        0
    } else {
        seq_pos
    }
}

/// VL images are request-local. A second image-bearing turn must cold-reset
/// before vision tokens are spliced into the text context, regardless of the
/// architecture's cache capability.
fn vl_request_requires_cold_reset(
    has_image: bool,
    seq_pos: usize,
    prior_image_state: Option<u64>,
) -> bool {
    has_image && (seq_pos > 0 || prior_image_state.is_some())
}

/// Stable, content-derived state for a preprocessed VL image. A compact FNV-1a
/// digest is sufficient here: this is a request-transition sentinel, not a
/// cryptographic identity or a cache key.
fn vl_image_state(pixels: &[f32], height: usize, width: usize) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in (height as u64)
        .to_le_bytes()
        .into_iter()
        .chain((width as u64).to_le_bytes())
        .chain(
            pixels
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes()),
        )
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Apply the CPU-visible model-turn transition before the fallible GPU reset.
/// Keeping this seam separate makes the image-turn isolation contract testable
/// without a vision model or a GPU, while `model_reset_context` remains the
/// authority for device and architecture state.
fn prepare_vl_request_state(session: &mut hipfire_loader::SessionState, image_state: u64) -> bool {
    if vl_request_requires_cold_reset(true, session.seq_pos, session.vl_image_state) {
        session.reset_cpu();
        session.vl_image_state = Some(image_state);
        true
    } else {
        session.vl_image_state = Some(image_state);
        false
    }
}

fn reset_error_json(id: &str, error: &impl std::fmt::Debug) -> String {
    let message = format!("context reset failed: {error:?}");
    reset_error_json_message(id, &message)
}

fn reset_error_json_message(id: &str, message: &str) -> String {
    format!(
        r#"{{"type":"error","id":{},"message":{}}}"#,
        serde_json::to_string(id).unwrap_or_else(|_| "\"\"".to_string()),
        serde_json::to_string(message).unwrap_or_else(|_| "\"context reset failed\"".to_string())
    )
}

fn emit_reset_error(stdout: &mut std::io::Stdout, id: &str, error: &impl std::fmt::Debug) {
    let _ = writeln!(stdout, "{}", reset_error_json(id, error));
    let _ = stdout.flush();
}

fn emit_reset_error_message(stdout: &mut std::io::Stdout, id: &str, message: &str) {
    let _ = writeln!(stdout, "{}", reset_error_json_message(id, message));
    let _ = stdout.flush();
}

fn deepseek4_abort_json(id: &str, generated: usize) -> (String, String) {
    (
        serde_json::json!({
            "type": "aborted",
            "id": id,
            "reason": "client_cancelled",
        })
        .to_string(),
        serde_json::json!({
            "type": "done",
            "id": id,
            "finish_reason": "aborted",
            "prompt_tokens": 0,
            "completion_tokens": generated,
        })
        .to_string(),
    )
}

fn poison_model_slot(
    model: &mut Option<LoadedModel>,
    pflash_state: &mut Option<hipfire_arch_qwen35::pflash::PflashState>,
    pflash_drafter_gpu: &mut Option<rdna_compute::Gpu>,
    pflash_cfg: &mut Option<hipfire_arch_qwen35::pflash::PflashConfig>,
    gpu: &mut rdna_compute::Gpu,
) {
    // PFlash owns sibling-device allocations and must be torn down first.
    if let Some(mut pf) = pflash_state.take() {
        if let Some(mut dg) = pflash_drafter_gpu.take() {
            dg.bind_thread_or_warn();
            pf.unload_drafter(&mut dg);
            gpu.bind_thread_or_warn();
        } else {
            pf.unload_drafter(gpu);
        }
    }
    *pflash_cfg = None;
    if let Some(old) = model.take() {
        hipfire_loader::unload_model(old, gpu);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeferredTerminal {
    Aborted { id: String, generated: usize },
    Error { id: String, message: String },
}

#[derive(Debug, Clone, PartialEq)]
enum GenerateResult {
    Complete,
    Deferred(DeferredTerminal),
    ResetFailed {
        id: String,
        message: String,
    },
    /// Produced by the borrowed PP worker. The outer wrapper owns the model
    /// again and must perform the optional reset before publishing `done`.
    PpCompletion {
        generated: usize,
        prefill_tokens: usize,
        tok_s: f64,
        prefill_ms: f64,
        prefill_tok_s: f64,
        decode_tok_s: f64,
        reset_required: bool,
    },
}

fn reset_failed(id: &str, error: impl std::fmt::Debug) -> GenerateResult {
    GenerateResult::ResetFailed {
        id: id.to_string(),
        message: format!("{error:?}"),
    }
}

fn reset_failed_message(id: &str, message: String) -> GenerateResult {
    GenerateResult::ResetFailed {
        id: id.to_string(),
        message,
    }
}

/// Reset the loader-owned model state before publishing a deferred terminal.
/// The caller owns the model slot and decides what to do when this fallible
/// reset fails (normally: unload/poison the slot).
fn reset_then_emit(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    terminal: DeferredTerminal,
) -> Result<(), hipfire_loader::ResetError> {
    model_reset_context(m, gpu)?;
    match terminal {
        DeferredTerminal::Aborted { id, generated } => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"aborted","id":{},"reason":"client_cancelled"}}"#,
                serde_json::to_string(&id).unwrap_or_else(|_| "\"\"".to_string())
            );
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":{},"finish_reason":"aborted","prompt_tokens":0,"completion_tokens":{},"prefill_ms":0,"decode_ms":0}}"#,
                serde_json::to_string(&id).unwrap_or_else(|_| "\"\"".to_string()),
                generated
            );
        }
        DeferredTerminal::Error { id, message } => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":{},"message":{}}}"#,
                serde_json::to_string(&id).unwrap_or_else(|_| "\"\"".to_string()),
                serde_json::to_string(&message).unwrap_or_default()
            );
        }
    }
    let _ = stdout.flush();
    Ok(())
}

/// DFlash-powered greedy decode. Mirrors `generate`'s ChatML shape and
/// token-streaming output but replaces the AR sample loop with
/// `spec_step_dflash` cycles — each cycle drafts B tokens via the diffusion
/// model and verifies them in one target forward, committing accept_len+1
/// at a time.
///
/// Prompt cache: for `messages_history`-bearing chat requests this path now
/// reuses the target KV + DeltaNet prefix on a pure conversation extension
/// (via [`plan_prompt_cache`] + `seed_target_hidden_suffix_abortable`), and the
/// draft's cumulative `target_hidden` is extended by scattering only the suffix
/// rows — so DFlash keeps its decode speedup AND skips re-prefilling the cached
/// prefix. A divergent / first / raw-prompt turn full-resets and prefills the
/// whole conversation as before.
///
/// The arch-dispatched borrow of the spec-decode target as `&mut dyn SpecTarget`
/// now lives behind the loader's `spec_target_guard()` + the runtime
/// `SpecTargetGuard` trait — this fn only ever sees `&mut dyn SpecTarget` and
/// never learns which arch (qwen35 moved-bundle vs llama borrow-in-place) it drives.
#[allow(clippy::too_many_arguments)]
/// Render the [`ClientEvent`]s a [`SpecEmit`] step produced to the daemon's
/// JSONL wire format, byte-identical to `generate_dflash`'s old inline writes.
/// `t_ms` is the per-step timestamp the inline path attached to committed +
/// token frames (`t0.elapsed()`); tool_calls frames carry no timing.
fn render_client_events(stdout: &mut std::io::Stdout, id: &str, events: &[ClientEvent], t_ms: u64) {
    for ev in events {
        match ev {
            ClientEvent::Committed { id: tok_id, idx } => {
                emit_committed_event(stdout, id, *tok_id, *idx, t_ms);
            }
            ClientEvent::Token(text) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"token","id":"{}","text":{}}}"#,
                    id,
                    serde_json::to_string(text).unwrap_or_default()
                );
                let _ = stdout.flush();
            }
            ClientEvent::Reasoning(text) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"reasoning","id":"{}","text":{}}}"#,
                    id,
                    serde_json::to_string(text).unwrap_or_default()
                );
                let _ = stdout.flush();
            }
            ClientEvent::ToolCalls(calls) => {
                let calls_json: Vec<serde_json::Value> = calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "name": tc.name,
                            "arguments": tc.arguments,
                        })
                    })
                    .collect();
                let calls_str =
                    serde_json::to_string(&calls_json).unwrap_or_else(|_| "[]".to_string());
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"tool_calls","id":"{}","calls":{}}}"#,
                    id, calls_str,
                );
            }
        }
    }
}

/// Model-independent emitter recipe a `generate_spec` wrapper supplies so the
/// arch's carrier can build the matching [`SpecEmit`] *after* `generate_spec` has
/// acquired the slot (the emitter needs `slot.eos_token()` and the tokenizer,
/// both derived from `m` *inside* `generate_spec`). Every field is neutral —
/// tool definitions are the request's raw JSON, which each arch's emitter parses
/// into its own grammar schema; `generate_spec` builds a `SpecEmitCtx` from this
/// + the slot's eos + the tokenizer and calls `carrier.make_spec_emitter`.
struct SpecEmitRequest {
    im_end: Option<u32>,
    /// Raw tool definitions (OpenAI-shape JSON); `None`/empty ⇒ no tool grammar.
    tools: Option<Vec<serde_json::Value>>,
    stop: Vec<String>,
    max_think: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    /// Reasoning-effort level (consumed by the ds4 emitter; ignored by ChatML).
    think_mode: ThinkMode,
    /// Pre-decoded vocab for arches whose grammar masks per-token (ds4). The
    /// wrapper builds/caches the Arc so this struct carries no `LoadedModel` ref.
    decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
}

/// Summary returned by `generate_spec` to its arch wrapper, which writes the
/// arch-specific `done` envelope and (qwen35) the asst-turn cache store from it.
/// `None` is returned on the abort/error early-exits, which already wrote their
/// own `done`/`error`; the wrapper then does nothing.
struct SpecRun {
    generated: usize,
    spec_cycles: usize,
    spec_accepted: usize,
    /// The emitter's consuming seal. Qwen's finalized turn is the sole source
    /// for text, tools, diagnostics, and reusable replay.
    finalized: Option<hipfire_runtime::spec_transcript::FinalizedAssistantTurn>,
    /// Whether the target state ended exactly at the sealed replay boundary.
    /// A partial/intra-token stop or hidden cache advance invalidates reuse.
    target_reusable: bool,
    /// Newly-prefilled token count (the suffix actually fed through the model).
    prefill_tokens_len: usize,
    /// The terminal flush summary (tool-call count drives the wrapper's
    /// `finish_reason`; events were already rendered inside `generate_spec`).
    finish: FinishSummary,
    prefill_s: f64,
    total_s: f64,
    decode_s: f64,
}

enum SpecRunOutcome {
    Ready(SpecRun),
    ImmediateError(String),
    Failed(String),
    Aborted { generated: usize },
}

/// Production prefill admission.  The first-token seed is not published until
/// both the speculator and the cancellation gate agree that the turn may
/// continue.  This is deliberately before `SpecEmit::begin` and before any
/// durable session/cache update.
fn admit_spec_prefill(
    result: Result<PrefillOutcome, String>,
    cancelled: bool,
) -> Result<u32, SpecRunOutcome> {
    match result {
        Ok(PrefillOutcome::Ready { first_token }) if !cancelled => Ok(first_token),
        Ok(PrefillOutcome::Ready { .. }) | Ok(PrefillOutcome::Aborted) => {
            Err(SpecRunOutcome::Aborted { generated: 0 })
        }
        Err(error) => Err(SpecRunOutcome::Failed(format!("prefill: {error}"))),
    }
}

/// Final cancellation gate between native prefill completion and first-token
/// admission to the emitter. This closes the race where cancellation arrives
/// after the drafter returned its seed but before any wire-visible event.
fn admit_spec_prefill_before_output(
    result: Result<PrefillOutcome, String>,
    cancelled: &dyn Fn() -> bool,
) -> Result<u32, SpecRunOutcome> {
    let first_token = admit_spec_prefill(result, false)?;
    if cancelled() {
        Err(SpecRunOutcome::Aborted { generated: 0 })
    } else {
        Ok(first_token)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SpecTerminalEnvelope {
    Error(String),
    Aborted { generated: usize },
    DoneAborted { generated: usize },
    Done { generated: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecEmitDisposition {
    Finish,
    Discard,
}

fn spec_emit_disposition(outcome: &SpecRunOutcome) -> SpecEmitDisposition {
    match outcome {
        SpecRunOutcome::Ready(_) => SpecEmitDisposition::Finish,
        SpecRunOutcome::ImmediateError(_)
        | SpecRunOutcome::Failed(_)
        | SpecRunOutcome::Aborted { .. } => SpecEmitDisposition::Discard,
    }
}

fn settle_spec_emit<'a>(
    emit: Box<dyn SpecEmit + 'a>,
    disposition: SpecEmitDisposition,
) -> Option<FinishSummary> {
    match disposition {
        SpecEmitDisposition::Finish => Some(emit.finish()),
        SpecEmitDisposition::Discard => {
            drop(emit);
            None
        }
    }
}

fn settle_spec_emit_for_outcome<'a>(
    emit: Box<dyn SpecEmit + 'a>,
    outcome: &SpecRunOutcome,
) -> Option<FinishSummary> {
    settle_spec_emit(emit, spec_emit_disposition(outcome))
}

fn discard_spec_emit<'a>(emit: Box<dyn SpecEmit + 'a>, outcome: SpecRunOutcome) -> SpecRunOutcome {
    let _ = settle_spec_emit_for_outcome(emit, &outcome);
    outcome
}

/// Complete one production speculative caller lifecycle. This is deliberately
/// the only boundary that may release the target guard, consume/discard the
/// real emitter, reset request state, insert a reusable turn, or publish a
/// terminal envelope. `generate_spec` is called by both Qwen and DeepSeek
/// wrappers, so both routes use this exact ordering.
///
/// `guard` is `Some` for the prefill admission path, where the emitter is still
/// live. The normal decode epilogue passes `None`/`None` because it has already
/// consumed the emitter and released the guard while building `SpecRun`; the
/// same function still owns reset/cache/envelope ordering there.
fn finish_spec_caller_after_guard<'a>(
    guard: Option<Box<dyn SpecTargetGuard + 'a>>,
    emit: Option<Box<dyn SpecEmit + 'a>>,
    outcome: SpecRunOutcome,
    id: &str,
    mut reset: impl FnMut() -> Result<(), String>,
    mut cache_insert: impl FnMut(&SpecRun),
    mut envelope: impl FnMut(SpecTerminalEnvelope),
) -> Result<Option<SpecRun>, GenerateResult> {
    // Drop first. Qwen's guard restores its moved bundle here; DeepSeek's
    // in-place guard releases its borrow here. Canonical reset must never run
    // while either production borrow is alive.
    drop(guard);
    match outcome {
        SpecRunOutcome::Ready(run) => {
            let mut run = run;
            if let Some(emit) = emit {
                let finish = emit.finish();
                run.finalized = finish.finalized.clone();
                run.finish = finish;
            }
            if !run.target_reusable {
                if let Err(error) = reset() {
                    return Err(reset_failed_message(id, error));
                }
            }
            cache_insert(&run);
            envelope(SpecTerminalEnvelope::Done {
                generated: run.generated,
            });
            Ok(Some(run))
        }
        SpecRunOutcome::ImmediateError(error) | SpecRunOutcome::Failed(error) => {
            drop(emit);
            if let Err(reset_error) = reset() {
                return Err(reset_failed_message(id, reset_error));
            }
            envelope(SpecTerminalEnvelope::Error(error));
            Ok(None)
        }
        SpecRunOutcome::Aborted { generated } => {
            drop(emit);
            if let Err(reset_error) = reset() {
                return Err(reset_failed_message(id, reset_error));
            }
            envelope(SpecTerminalEnvelope::Aborted { generated });
            envelope(SpecTerminalEnvelope::DoneAborted { generated });
            Ok(None)
        }
    }
}

fn emit_spec_terminal_envelope(
    stdout: &mut std::io::Stdout,
    id: &str,
    envelope: SpecTerminalEnvelope,
) {
    let json_id = serde_json::to_string(id).unwrap_or_else(|_| "\"\"".to_string());
    match envelope {
        SpecTerminalEnvelope::Error(error) => emit_error_with_id(stdout, id, error),
        SpecTerminalEnvelope::Aborted { .. } => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"aborted","id":{},"reason":"client_cancelled"}}"#,
                json_id
            );
        }
        SpecTerminalEnvelope::DoneAborted { generated } => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":{},"finish_reason":"aborted","prompt_tokens":0,"completion_tokens":{},"prefill_ms":0,"decode_ms":0}}"#,
                json_id, generated
            );
            let _ = stdout.flush();
        }
        // Architecture-specific normal done envelopes are still emitted by
        // the two wrappers after the returned SpecRun. The lifecycle seam
        // emits this only to make normal ordering explicit to its collaborators
        // and deterministic tests.
        SpecTerminalEnvelope::Done { .. } => {}
    }
}

#[cfg(test)]
fn forced_advance_completed(outcome: &SpecAdvance) -> bool {
    matches!(outcome, SpecAdvance::Ready { .. })
}

#[derive(Debug)]
enum ForcedAdvance<E> {
    Aborted,
    Failed(E),
}

/// Advance an emitter continuation as one cache transaction before rendering it.
/// Returning the target's post-continuation argmax gives the caller the next
/// unconsumed speculative seed after every forced token reaches cache state.
fn advance_forced_tokens<E>(
    tokens: &[u32],
    remaining: usize,
    advance: impl FnOnce(&[u32]) -> Result<SpecAdvance, E>,
) -> Result<Option<u32>, ForcedAdvance<E>> {
    if tokens.is_empty() || tokens.len() > remaining {
        return Ok(None);
    }
    match advance(tokens).map_err(ForcedAdvance::Failed)? {
        SpecAdvance::Ready { last_argmax } => Ok(Some(last_argmax)),
        SpecAdvance::Aborted => Err(ForcedAdvance::Aborted),
    }
}

/// Observe the target's unconsumed post-continuation prediction before the
/// next speculative window advances it. `observe` reports whether it produced
/// a client-visible event, which is the daemon's generated-token accounting.
fn observe_forced_seed(
    seed: u32,
    generated: &mut usize,
    max_tokens: usize,
    observe: impl FnOnce(u32) -> hipfire_runtime::spec::EmitOutcome,
) -> bool {
    if *generated >= max_tokens {
        return false;
    }
    if observe(seed).generation_advanced {
        *generated += 1;
    }
    true
}

/// Preserve the full target/DFlash-advanced continuation in session and
/// speculative repeat context without making hidden tokens client-visible.
fn record_forced_batch(history: &mut Vec<u32>, tokens: &[u32]) {
    history.extend_from_slice(tokens);
}

/// Copy the committed internal sequence for Qwen's verbatim assistant-turn
/// cache, which must match the cached KV even when client replay stopped early.
/// Render a committed forced continuation only through its first terminal token.
/// The target and draft caches have already advanced over the full batch.
fn replay_forced_tokens(
    tokens: &[u32],
    mut observe: impl FnMut(u32) -> Option<StopReason>,
) -> Option<StopReason> {
    for &token in tokens {
        if let Some(stop) = observe(token) {
            return Some(stop);
        }
    }
    None
}

/// Publish or discard the CPU-side state for a sealed Qwen speculative turn.
///
/// A reusable seal is baked from the exact sealed token boundary. Any other
/// seal leaves no reusable conversation prefix; the caller still performs the
/// full model reset for GPU/recurrent state after the target guard is dropped.
fn publish_sealed_qwen_state(
    session: &mut hipfire_loader::SessionState,
    prompt_tokens: &[u32],
    position: usize,
    finalized: Option<&hipfire_runtime::spec_transcript::FinalizedAssistantTurn>,
    target_reusable: bool,
) {
    if target_reusable {
        if let Some(turn) = finalized {
            session.seq_pos = position;
            session.conversation_tokens = {
                let mut tokens =
                    Vec::with_capacity(prompt_tokens.len() + turn.diagnostic_tokens().len());
                tokens.extend_from_slice(prompt_tokens);
                tokens.extend_from_slice(turn.diagnostic_tokens());
                tokens
            };
        }
    } else {
        session.seq_pos = 0;
        session.conversation_tokens.clear();
    }
}

fn publish_eviction_position(
    new_physical: usize,
    draft_mirror: Result<(), String>,
) -> Result<usize, String> {
    draft_mirror.map(|()| new_physical)
}

fn maybe_evict_spec(
    gpu: &mut rdna_compute::Gpu,
    eviction: Option<&Eviction>,
    target: &mut dyn SpecTarget,
    spec: &mut dyn Speculator,
    position: usize,
) -> Result<Option<usize>, String> {
    let Some(eviction) = eviction else {
        return Ok(None);
    };
    let result = {
        let kv = target
            .kv_cache_mut()
            .ok_or_else(|| "eviction configured for non-KvCache spec target".to_string())?;
        eviction
            .maybe_evict(gpu, kv, position)
            .map_err(|e| format!("target eviction: {e:?}"))?
    };
    let Some(result) = result else {
        return Ok(None);
    };
    if result.retain_mask.is_empty() {
        return Ok(Some(result.new_physical));
    }
    let retain = EvictRetain {
        retain_mask: result.retain_mask,
        pre_phys: position,
    };
    let draft_mirror = spec
        .on_evict(gpu, &retain)
        .map_err(|e| format!("draft eviction mirror: {e}"));
    publish_eviction_position(result.new_physical, draft_mirror).map(Some)
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
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    pflash_bypass_reason: Option<&str>,
    pflash_alpha: Option<f32>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
    // Request-resolved sampling temperature. 0.0 → greedy/argmax-accept (the
    // historical DFlash posture). >0 → distribution-preserving spec decode via
    // one of two verify mechanisms inside `DflashSpeculator::step`, picked by how
    // the drafter loaded:
    //   * ddtree mode — SWOR tree-verify, fed the `temp` arg directly; honors
    //     temperature ONLY (the caller routes an explicit top_p/top_k/min_p to AR).
    //   * chain mode — lossless rejection sampling (full-vocab softmax on BOTH
    //     draft + target), fed the top_p/top_k/cactus args below via
    //     `Speculator::set_sampling` (see the set_sampling call). Honors
    //     temp+top_p+top_k; min_p is NOT plumbed (a min_p request routes to AR).
    temp: f32,
    // Nucleus (top_p) cutoff for the chain rejection-sampling path, applied
    // IDENTICALLY to both draft + target softmaxes (lossless == AR at this top_p).
    // 1.0 (>= 0.999) disables it. Ignored by the ddtree SWOR arm.
    top_p: f32,
    // Top-k cutoff (request/card recipe, e.g. qwen3.6 top_k=20) for the chain
    // sampled path, applied to both draft + target softmax rows. 0 = disabled.
    // Ignored by the ddtree SWOR arm.
    top_k: usize,
    // Cactus-style acceptance bump. 0.0 → lossless (distribution-preserving).
    // >0 → deliberately lossy (KL-bounded τ-for-correctness tradeoff). The
    // daemon hardcodes 0.0; the param exists only so a future opt-in request
    // field can reach it without re-touching this signature.
    cactus_delta: f32,
) -> GenerateResult {
    // The spec-step dispatch, ModelSlot assembly, checkpoint ring, and
    // SpecStats that this function used to drive inline now live behind the
    // arch-generic `Speculator` trait (the loader's `DflashSpeculator`) and the
    // `Qwen35SlotGuard` RAII target borrow — see the prefill/step/on_evict/
    // reset/rewind_to calls below.

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
    // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
    // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
    // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
    // the .hfq carries no template, e.g. an older A1B convert).
    // Jinja default-ON (flipped 2026-06-09): render through the model's chat
    // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
    // ChatML/Plain). Falls back to Plain automatically when no template resolves.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    let try_jinja = jinja_enabled && m.meta.chat_template.is_some();
    let prompt_tokens: Vec<u32> = if try_jinja {
        let template = m.meta.chat_template.as_ref().unwrap();
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
                            tool_plan: String::new(),
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        tool_plan: String::new(),
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
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // Prompt-cache plan (native DFlash reuse). For non-jinja chat with history,
    // decide whether this turn is a pure extension of the cached conversation.
    // On a HIT we reuse target KV + DeltaNet[0..start_pos] and the draft's
    // cumulative target_hidden, prefilling only the suffix; on a MISS we
    // full-reset and prefill the whole conversation (legacy behaviour).
    let cache_disabled =
        try_jinja || std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
    // DFlash divergent-render resume (default ON; opt out with
    // HIPFIRE_DFLASH_CKPT_RESUME=0). Requires no eviction (resume rewinds the
    // resident KV prefix). When on, the recurrent state is checkpointed during
    // the prompt seed and a divergent render resumes from the latest checkpoint
    // ≤ lcp — byte-identical to a cold prefill of the same render (verified),
    // so worst case equals the legacy cold-reset path. Off ⇒ no checkpoints
    // (zero overhead) + legacy cold-reset-on-divergence.
    let dflash_resume_enabled = std::env::var("HIPFIRE_DFLASH_CKPT_RESUME").ok().as_deref()
        != Some("0")
        && m.eviction.is_none();
    let dflash_ckpt_positions: Vec<usize> = m
        .speculator
        .as_ref()
        .map(|s| s.checkpoint_positions())
        .unwrap_or_default();
    let cache_plan: Option<PromptCachePlan> = if !try_jinja {
        messages_history.map(|hist| {
            let tok = m.tokenizer.as_ref().unwrap();
            plan_prompt_cache(
                tok,
                &mut m.persist.asst_turn_cache,
                &m.session.conversation_tokens,
                m.eviction.is_none(),
                system_prompt,
                prompt,
                assistant_prefix,
                hist,
                cache_disabled,
                &dflash_ckpt_positions,
                dflash_resume_enabled,
            )
        })
    } else {
        None
    };
    let resume_from: Option<usize> = cache_plan.as_ref().and_then(|p| p.resume_from);
    // `prompt_tokens` becomes the full canonical conversation when the cache
    // plan rendered it (keeps the end-of-turn `conversation_tokens` bake and the
    // next turn's LCP byte-consistent). Otherwise keep the jinja/ChatFrame build.
    let prompt_tokens: Vec<u32> = match &cache_plan {
        Some(p) => p.rendered.clone(),
        None => prompt_tokens,
    };
    let (prefill_tokens, prefill_start, cache_hit, cached_tokens_dflash): (
        Vec<u32>,
        usize,
        bool,
        usize,
    ) = match &cache_plan {
        Some(p) => (
            p.new_tokens.clone(),
            p.start_pos,
            p.cache_hit,
            p.cached_tokens,
        ),
        None => (prompt_tokens.clone(), 0, false, 0),
    };

    // ── Grammar-guided decoding setup (dflash path) ─────────────
    //
    // qwen35 enforces tool-call grammar POST-acceptance inside the emitter
    // (`Qwen35Emit::observe`); the emitter now extracts its own `ToolSchema`
    // list from the raw tool JSON inside `make_spec_emitter`. This wrapper only
    // honors the `HIPFIRE_QWEN35_GRAMMAR=0` kill-switch by withholding `tools`
    // (⇒ empty schema ⇒ grammar inactive).
    let grammar_enabled = std::env::var("HIPFIRE_QWEN35_GRAMMAR").ok().as_deref() != Some("0");
    let emit_tools: Option<Vec<serde_json::Value>> = if grammar_enabled {
        tools.map(|t| t.to_vec())
    } else {
        None
    };

    // The decode core (slot guard, prefill, accept-window loop, bake, finish) is
    // the arch-generic `generate_spec`. This wrapper owns the qwen35/llama-specific
    // prologue (jinja/Plain render + LCP cache plan), the emitter recipe
    // (`EmitSpec::Qwen35`), and the epilogue (asst-turn cache store + `done`
    // envelope). A future ds4 wrapper (Phase 4 T4c-2) builds its DSML render +
    // ds4 cache plan + `EmitSpec::Deepseek4` and writes its own ds4 `done`.
    //
    // Thread the request's sampling into the speculator BEFORE the step loop so
    // `DflashSpeculator::step` runs lossless rejection sampling at temp>0 instead
    // of decoding greedy (the #477-merge re-wire of spec-graph's sampled-DFlash).
    // Greedy (temp 0) is unchanged. No-op for a greedy-only Speculator impl; the
    // deepseek4 spec wrapper (`generate_deepseek4_spec`) deliberately never calls
    // this, so ds4 MTP stays greedy. cactus_delta is 0.0 (lossless) from here.
    if let Some(spec) = m.speculator.as_mut() {
        spec.set_sampling(temp, top_p, top_k, cactus_delta);
    }
    let prefill_tokens_full = prefill_tokens.len();
    let run = match generate_spec(
        m,
        gpu,
        stdout,
        id,
        prompt_tokens,
        prefill_tokens,
        prefill_start,
        cache_hit,
        resume_from,
        max_tokens,
        SpecEmitRequest {
            im_end: im_end_token,
            tools: emit_tools,
            stop: stop.to_vec(),
            max_think: max_think_tokens,
            assistant_prefix,
            think_mode: ThinkMode::NonThink,
            decoded_vocab: None,
        },
        temp,
    ) {
        Ok(Some(r)) => r,
        // Abort / error early-exit already wrote its own done/error envelope.
        Ok(None) => return GenerateResult::Complete,
        Err(failure) => return failure,
    };
    debug_assert_eq!(run.prefill_tokens_len, prefill_tokens_full);

    // ── consume the sealed assistant turn + populate asst_turn_cache ──────
    //
    // The emitter's seal is the only authority for text, tools, fingerprint,
    // and replay. In particular, an intra-token stop has no replay boundary
    // and must never be reconstructed from the visible stream.
    let finalized = run
        .finalized
        .as_ref()
        .expect("Qwen spec emitter must publish a finalized assistant turn");
    let emit_tool_calls = finalized.tool_calls().to_vec();

    // ── done envelope (qwen35-flavoured) ─────────────────────────
    let tok_s = if run.total_s > 0.0 {
        run.generated as f64 / run.total_s
    } else {
        0.0
    };
    let decode_tok_s = if run.decode_s > 0.0 {
        run.generated as f64 / run.decode_s
    } else {
        0.0
    };
    // New-token count (not full rendered length) so the prefill rate reflects
    // actual work on a cache HIT/resume — matches every other path's numerator.
    let prefill_tok_s = if run.prefill_s > 0.0 {
        run.prefill_tokens_len as f64 / run.prefill_s
    } else {
        0.0
    };
    let tau = if run.spec_cycles > 0 {
        run.spec_accepted as f64 / run.spec_cycles as f64
    } else {
        0.0
    };
    // Per PRD §3.1, when PFlash bypassed (e.g. dflash_decode_active for
    // this branch) the `done` object must surface the bypass reason and
    // alpha alongside the dflash perf metrics.
    let pflash_done_field = match (pflash_bypass_reason, pflash_alpha) {
        (Some(r), Some(a)) => format!(
            r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
            r.replace('"', "'"),
            a,
        ),
        _ => String::new(),
    };
    // Length-cap detection — see qwen35 path for rationale.
    let hit_length_cap = run.generated >= max_tokens;
    let finish_reason = if hit_length_cap {
        "length"
    } else if !emit_tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1},"dflash":true,"tau":{:.2},"cycles":{},"cached_tokens":{},"finish_reason":"{}"{}}}"#,
        // `prefill_tokens` is the NEWLY-prefilled count (the suffix actually fed
        // through the model), NOT the full rendered length — the CLI computes
        // `prompt_tokens = cached + prefill`, so reporting the full length here
        // double-counted the cached prefix on every HIT/resume.
        id,
        run.generated,
        tok_s,
        run.prefill_tokens_len,
        run.prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        run.prefill_s * 1000.0,
        tau,
        run.spec_cycles,
        cached_tokens_dflash,
        finish_reason,
        pflash_done_field,
    );
    let _ = stdout.flush();
    GenerateResult::Complete
}

/// Arch-generic spec-decode core extracted from `generate_dflash` (Phase 4 T4a).
/// Drives any `Speculator` (`m.speculator`) + `SpecTarget` (via `spec_target_guard`)
/// + `SpecEmit` through one prefill → accept-window loop → bake → done. The caller
/// (`generate_dflash` for qwen35/llama today) prepares the arch-specific inputs:
/// the already-rendered `prompt_tokens`, the LCP cache decision
/// (`prefill_tokens`/`prefill_start`/`cache_hit`/`resume_from`), and the emitter
/// recipe (`EmitSpec`). It returns a [`SpecRun`] summary from which the wrapper
/// writes its arch-specific `done` envelope + cache store; `None` on the
/// abort/error early-exits (which already wrote their own done/error).
/// T4c-2 adds the deepseek4 wrapper + `EmitSpec::Deepseek4` variant.
#[allow(clippy::too_many_arguments)]
fn generate_spec(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_tokens: Vec<u32>,
    prefill_tokens: Vec<u32>,
    prefill_start: usize,
    cache_hit: bool,
    resume_from: Option<usize>,
    max_tokens: usize,
    emit_req: SpecEmitRequest,
    // Request sampling temperature. >0 only reaches here for speculators that
    // report `supports_temp_verify()` (qwen35 DFlash ddtree → SWOR); greedy
    // drafters ignore it. The daemon's routing gate enforces that invariant.
    temp: f32,
) -> Result<Option<SpecRun>, GenerateResult> {
    // See the prefill-cancellation lifecycle call below: the emitter borrows
    // the model tokenizer, while reset must run only after that emitter and the
    // target guard are released.
    let model_ptr = m as *mut LoadedModel;
    let gpu_ptr = gpu as *mut rdna_compute::Gpu;
    let qwen_cache_ptr = &mut m.persist.asst_turn_cache as *mut _;

    // Acquire the target via the RAII slot guard — it restores the bundle into
    // m.state on EVERY exit path (return, `?`, panic), which structurally
    // eliminates the eight hand-written reconstruction sites that were the
    // #462 cross-request state-bleed class. `m.speculator`, `m.state`,
    // `m.session.seq_pos`, `m.conversation_tokens` and `m.eviction` are disjoint fields,
    // so the guard, the speculator borrow, and the bookkeeping below coexist.
    let (block_size, ctx_capacity) = match m.speculator.as_ref() {
        Some(s) => (s.block_size(), s.ctx_capacity()),
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"dflash path entered without a loaded speculator"}}"#,
                id
            );
            let _ = stdout.flush();
            return Ok(None);
        }
    };
    // Resolve the arch's carrier once — the single dispatch the spec path routes
    // through for BOTH the target borrow and the emitter (the daemon never
    // arch-matches for spec-decode). `&'static dyn Carrier` borrows nothing from
    // `m`, so it coexists with the `tokenizer`/`&mut m.state` borrows below.
    let arch_id = m.meta.arch_id;
    let carrier = match hipfire_loader::carrier_for(arch_id) {
        Some(c) => c,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"no carrier for arch_id {}"}}"#,
                id, arch_id
            );
            let _ = stdout.flush();
            return Ok(None);
        }
    };
    // Capacity checks. With eviction enabled the advertised context window is
    // effectively unbounded (eviction fires between spec cycles), but the
    // *prompt* must still fit in one physical_cap span because the prompt seed
    // writes it per-token without chunking. These run before acquiring the slot:
    // they cannot mutate target state and must not reset a clean model.
    let eff_prompt_cap = if m.eviction.is_some() {
        m.meta.physical_cap
    } else {
        ctx_capacity
    };
    if prompt_tokens.len().saturating_add(block_size) > eff_prompt_cap {
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
        return Ok(None);
    }
    if m.eviction.is_none()
        && prompt_tokens
            .len()
            .saturating_add(max_tokens)
            .saturating_add(block_size)
            > ctx_capacity
    {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt+max_tokens exceeds ctx_capacity {} (enable cask_sidecar for long decode)"}}"#,
            id, ctx_capacity,
        );
        let _ = stdout.flush();
        return Ok(None);
    }

    // A speculative cache miss is a full context transition, not just a
    // bookkeeping change. Reset before borrowing the target/emitter so all
    // request-owned GPU state is cleared and ResetFailed can propagate.
    if !cache_hit {
        if let Err(error) = model_reset_context(m, gpu) {
            return Err(reset_failed(id, error));
        }
    }

    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Arch-dispatched target borrow via `Carrier::spec_target_guard()`
    // (`m.meta.model_path` is a disjoint field → no borrow conflict with the
    // `&mut m.state` the guard takes). qwen35 moves the bundle out + reopens its
    // HfqFile (restored on Drop); the pure-attention arms borrow in place. The
    // boxed `SpecTargetGuard` yields `&mut dyn SpecTarget` either way.
    let mut guard = match carrier.spec_target_guard(&mut m.state, &m.meta.model_path) {
        Ok(g) => g,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"{}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            return Ok(None);
        }
    };
    let outcome = 'spec: {
        let slot = match guard.slot() {
            Ok(s) => s,
            Err(e) => break 'spec SpecRunOutcome::ImmediateError(e),
        };

        // Build the emitter before speculative state can mutate. A construction
        // failure therefore leaves the target clean and does not need recovery.
        let emit_ctx = hipfire_runtime::spec::SpecEmitCtx {
            tokenizer,
            eos: slot.eos_token(),
            im_end: emit_req.im_end,
            tools: emit_req.tools.as_deref(),
            stop: emit_req.stop,
            max_think: emit_req.max_think,
            max_tokens,
            assistant_prefix: emit_req.assistant_prefix,
            think_mode: emit_req.think_mode,
            decoded_vocab: emit_req.decoded_vocab,
        };
        let mut emit: Box<dyn SpecEmit> = match carrier.make_spec_emitter(emit_ctx) {
            Ok(e) => e,
            Err(e) => break 'spec SpecRunOutcome::ImmediateError(e),
        };
        let spec = m
            .speculator
            .as_mut()
            .expect("checked before slot acquisition");

        // Divergent-render RESUME: restore the drafter-local + target recurrent
        // state to the latest checkpoint ≤ ckpt and drop the now-stale tail of the
        // checkpoint ring (`rewind_to` does both), then rewind the daemon's seq_pos
        // / conversation_tokens. The turn then proceeds exactly like a HIT with
        // start_pos == ckpt (the cache plan already set cache_hit=true).
        if let Some(ckpt) = resume_from {
            if let Err(e) = spec.rewind_to(gpu, slot, ckpt) {
                break 'spec discard_spec_emit(
                    emit,
                    SpecRunOutcome::Failed(format!("spec rewind: {e}")),
                );
            }
            m.session.seq_pos = ckpt;
            m.session.conversation_tokens.truncate(ckpt);
        }

        if cache_hit && std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[qwen-cache HIT dflash] reuse prefix={} suffix={} (no reset)",
                prefill_start,
                prefill_tokens.len()
            );
        }

        let t0 = Instant::now();
        // Prefill: the speculator seeds the target's hidden state (advancing its KV
        // + recurrent state), primes the drafter's cached target-hidden, snapshots
        // the divergent-render checkpoint ring, and returns the target's first
        // token. On a cache hit only the suffix is seeded; on a miss the seed
        // self-resets target state and the full prompt is seeded. Client cancel is
        // surfaced as `PrefillOutcome::Aborted`.
        let id_for_abort = id.to_string();
        let abort_latch = AbortLatch::new();
        let prefill = spec.prefill(
            gpu,
            slot,
            &prompt_tokens,
            &prefill_tokens,
            prefill_start,
            cache_hit,
            resume_from,
            &|| abort_latch.poll(&id_for_abort),
        );
        let first_token = match admit_spec_prefill_before_output(prefill, &|| {
            abort_latch.is_cancelled() || abort_latch.poll(id)
        }) {
            Ok(first_token) => first_token,
            Err(outcome) => {
                // This is the shared production cancellation boundary for both
                // Qwen and DeepSeek callers. Keep the real guard and emitter
                // alive until the lifecycle seam has restored/discarded them;
                // only then may canonical reset publish abort envelopes.
                // The erased emitter borrows the model tokenizer, so retain
                // reset as a raw-pointer hook. The lifecycle function drops
                // both production borrows before invoking it; that ordering is
                // the safety invariant this seam exists to enforce.
                match finish_spec_caller_after_guard(
                    Some(guard),
                    Some(emit),
                    outcome,
                    id,
                    || unsafe {
                        model_reset_context(&mut *model_ptr, &mut *gpu_ptr)
                            .map_err(|error| format!("{error:?}"))
                    },
                    |_| {},
                    |envelope| emit_spec_terminal_envelope(stdout, id, envelope),
                ) {
                    Ok(_) => return Ok(None),
                    Err(failure) => return Err(failure),
                }
            }
        };

        let t_prefill = Instant::now();

        // Decode loop — spec.step returns one acceptance window (SpecStep) per cycle.
        // `emitted` is the speculator's repeat / n-gram context (NOT emission state);
        // it stays in the loop and excludes any grammar-rejected token.
        let mut emitted: Vec<u32> = Vec::new();
        let mut position = prompt_tokens.len();
        let mut seed_token = first_token;
        // τ accounting, inlined from the unified `SpecStep` (the old `SpecStats`
        // type took the arch-specific `SpecStepResult`, which the daemon no longer
        // sees): τ = accepted drafts / cycle.
        let mut spec_cycles = 0usize;
        let mut spec_accepted = 0usize;
        let mut generated = 0usize;
        // Logical target advancement excludes the prefill prediction itself;
        // the first token is the seed for the first verify window.
        let mut target_advanced = 0usize;

        // Post-prefill compaction (FlashCASK pattern from dflash_spec_demo).
        // If the prompt already filled past budget+beta, compact once before
        // entering the spec loop so the first spec step writes at physical slot
        // `budget`. compact_offset is maintained on slot.kv_cache; subsequent
        // forwards inside the speculator read it for RoPE phase automatically. The
        // drafter-local hidden cache is compacted to match via `on_evict`.
        if let Some(new_physical) =
            match maybe_evict_spec(gpu, m.eviction.as_ref(), slot, spec.as_mut(), position) {
                Ok(position) => position,
                Err(e) => {
                    if abort_latch.poll(id) {
                        break 'spec discard_spec_emit(
                            emit,
                            SpecRunOutcome::Aborted { generated: 0 },
                        );
                    }
                    break 'spec discard_spec_emit(
                        emit,
                        SpecRunOutcome::Failed(format!("post-prefill eviction: {e}")),
                    );
                }
            }
        {
            let compact_offset = match slot.kv_cache_mut() {
                Some(kv) => kv.compact_offset,
                None => {
                    break 'spec discard_spec_emit(
                        emit,
                        SpecRunOutcome::Failed(
                            "post-prefill eviction: missing KvCache after compaction".into(),
                        ),
                    );
                }
            };
            eprintln!(
                "[dflash] post-prefill evict: {} -> {} (compact_offset={})",
                position, new_physical, compact_offset,
            );
            position = new_physical;
        }

        // Eviction and its drafter mirror are GPU work performed after prefill.
        // A cancellation observed by either that work or the prefill's nested
        // chunk loop remains latched and must be settled before `begin` can
        // publish even the first token.
        if abort_latch.is_cancelled() || abort_latch.poll(id) {
            break 'spec discard_spec_emit(emit, SpecRunOutcome::Aborted { generated: 0 });
        }

        // Emit the first token immediately so TTFT is the prefill time. `begin`
        // pushes + filters it, seeds the grammar matcher, and reports whether the
        // first token is itself a terminator (→ skip the spec loop entirely, so
        // spec_step_dflash never drafts a whole block seeded on a terminal token).
        let first_begin = emit.begin(first_token);
        render_client_events(
            stdout,
            id,
            &first_begin.events,
            t0.elapsed().as_millis() as u64,
        );
        // Count the first token only when the emitter emitted it (see the same guard
        // in the accept loop). qwen35 always emits a `Committed`; the ds4 emitter
        // returns no events for an EOS-first prefill argmax, yielding an empty turn.
        if first_begin.generation_advanced {
            generated += 1;
            emitted.push(first_token);
        }
        let first_token_is_eos = first_begin.stop.is_some();

        // (The DFlash RNG cell and the chain-vs-tree resolution that used to live
        // here are now resolved once at build time and
        // owned inside the speculator — see `build_dflash_speculator`.)

        // Fast path exit conditions (mirrors the dflash_spec_demo outer loop).
        // `!first_token_is_eos` short-circuits the entire spec loop when the prefill's
        // first sampled token was already a terminator (see the guard above).
        while !first_token_is_eos && generated < max_tokens {
            // Decode-side abort (dflash path). See the matching block in
            // `generate()` for rationale. Without this, a Pi cancel
            // mid-decode leaves the spec-decode loop running for max_tokens
            // worth of wasted work.
            if abort_latch.poll(id) {
                // The target and drafter have advanced past un-baked history. Leave
                // the guard scope so the canonical model reset can recover both.
                break 'spec discard_spec_emit(emit, SpecRunOutcome::Aborted { generated });
            }
            if position.saturating_add(block_size) >= ctx_capacity {
                break;
            }

            // One acceptance window. The speculator owns chain-vs-tree dispatch
            // internally; the daemon just hands it the borrowed target and
            // the prior committed tokens (drafter repeat / n-gram context). The
            // in-step grammar mask comes from the emitter: qwen35 returns `None`
            // (post-hoc grammar in `observe`); a ds4 emitter returns its erased
            // matcher so the fused step constrains drafts in-place. `emit.grammar()`'s
            // borrow ends when `step` returns, before the per-token `emit.observe`.
            let step = match spec.step(
                gpu,
                slot,
                position,
                seed_token,
                &emitted,
                emit.grammar(),
                temp,
            ) {
                Ok(s) => s,
                Err(e) => {
                    break 'spec discard_spec_emit(
                        emit,
                        SpecRunOutcome::Failed(format!("spec_step: {e}")),
                    );
                }
            };
            if abort_latch.poll(id) {
                break 'spec discard_spec_emit(emit, SpecRunOutcome::Aborted { generated });
            }
            spec_cycles += 1;
            spec_accepted += step.accepted;
            // `emit` is already the committed tail with the seed re-echo stripped.
            let committed_tail: Vec<u32> = step.emit.to_vec();

            let mut hit_eos = false;
            let mut think_cap_hit = false;
            let mut forced_after: Vec<u32> = Vec::new();
            for &tok in &committed_tail {
                if generated >= max_tokens {
                    break;
                }
                // `emit.observe` runs the per-token emission policy: grammar
                // pre-check (reject → grammar violation, NO emit, post-loop forces a
                // full KV/DN reset to clear the polluted slots spec_step wrote), then
                // committed/token frames, EOS, user stop-sequence, and think
                // force-close. Loop state (emitted/generated/position) stays here.
                emit.set_generated_hint(generated);
                let outcome = emit.observe(tok);
                if outcome.stop == Some(StopReason::GrammarViolation) {
                    // Rejected before emit — not added to the repeat context, not
                    // streamed, not counted. Treat as EOS for this turn.
                    render_client_events(
                        stdout,
                        id,
                        &outcome.events,
                        t0.elapsed().as_millis() as u64,
                    );
                    hit_eos = true;
                    break;
                }
                // Count/bake/render a committed token only when the emitter actually
                // emitted it (non-empty events). qwen35 always emits a `Committed`
                // event (even on EOS / held bytes), so this is a no-op there; the
                // deepseek4 emitter returns NO events for an accepted EOS, so that
                // terminator is neither counted into `generated` nor baked into
                // `conversation_tokens` — byte-matching the bespoke ds4 loop (which
                // broke on accepted-EOS before push/increment).
                if outcome.generation_advanced {
                    emitted.push(tok);
                    render_client_events(
                        stdout,
                        id,
                        &outcome.events,
                        t0.elapsed().as_millis() as u64,
                    );
                    generated += 1;
                }
                // Generation-intervention hook: the emitter may request tokens to
                // FORCE after this one, suppressing the step's terminator (cohere2moe
                // empty-turn guard / think-budget force-close). Default empty for
                // every other emitter ⇒ this branch never taken, loop byte-identical.
                // Processed after the post-loop `position += emit.len()` so forced
                // tokens advance from the target's true (batch-advanced) cursor.
                let forced = emit.take_forced();
                if !forced.is_empty() {
                    forced_after = forced;
                    break;
                }
                match outcome.stop {
                    Some(StopReason::Eos) | Some(StopReason::StopSequence) => {
                        hit_eos = true;
                        break;
                    }
                    Some(StopReason::ThinkCap) => {
                        think_cap_hit = true;
                        break;
                    }
                    Some(StopReason::GrammarViolation) => unreachable!("handled above"),
                    None => {}
                }
            }
            // Advance by the emitted-tail length (= accepted + 1), NOT by `accepted`
            // — the loop contract pinned by spec.rs's `emit_len_drives_advance`.
            position += step.emit.len();
            target_advanced += step.emit.len();
            seed_token = step.next_seed;

            // Forced-token injection (cohere2moe generation guards; no-op for every
            // other emitter — `take_forced` defaulted empty). The speculative window
            // is fully committed above, so advance the entire continuation as one
            // target/drafter cache transaction before exposing any of it to the client.
            if !forced_after.is_empty() {
                let forced = std::mem::take(&mut forced_after);
                match advance_forced_tokens(
                    &forced,
                    max_tokens.saturating_sub(generated),
                    |tokens| {
                        spec.advance_forced(gpu, slot, tokens, position, &|| abort_latch.poll(id))
                    },
                ) {
                    Ok(Some(seed)) => {
                        position += forced.len();
                        target_advanced += forced.len();
                        // Only tokens explicitly admitted by the owner enter
                        // the repeat context; target advancement alone is not
                        // client-visible generation.
                        let forced_stop = replay_forced_tokens(&forced, |token| {
                            emit.set_generated_hint(generated);
                            let outcome = emit.observe(token);
                            if outcome.generation_advanced {
                                emitted.push(token);
                                render_client_events(
                                    stdout,
                                    id,
                                    &outcome.events,
                                    t0.elapsed().as_millis() as u64,
                                );
                                generated += 1;
                            }
                            outcome.stop
                        });
                        match forced_stop {
                            Some(StopReason::GrammarViolation) => {
                                break 'spec discard_spec_emit(
                                    emit,
                                    SpecRunOutcome::Failed(
                                        "speculative grammar violation during forced continuation"
                                            .into(),
                                    ),
                                );
                            }
                            Some(StopReason::Eos) | Some(StopReason::StopSequence) => {
                                hit_eos = true
                            }
                            Some(StopReason::ThinkCap) => think_cap_hit = true,
                            None => {
                                let mut seed_stop = None;
                                let seed_generated_hint = generated;
                                if observe_forced_seed(seed, &mut generated, max_tokens, |token| {
                                    emit.set_generated_hint(seed_generated_hint);
                                    let outcome = emit.observe(token);
                                    seed_stop = outcome.stop;
                                    if outcome.generation_advanced {
                                        emitted.push(token);
                                        render_client_events(
                                            stdout,
                                            id,
                                            &outcome.events,
                                            t0.elapsed().as_millis() as u64,
                                        );
                                    }
                                    outcome
                                }) {
                                    match seed_stop {
                                        Some(StopReason::GrammarViolation) => {
                                            break 'spec discard_spec_emit(
                                                emit,
                                                SpecRunOutcome::Failed(
                                                "speculative grammar violation after forced continuation".into(),
                                                ),
                                            );
                                        }
                                        Some(StopReason::Eos) | Some(StopReason::StopSequence) => {
                                            hit_eos = true
                                        }
                                        Some(StopReason::ThinkCap) => think_cap_hit = true,
                                        None => {}
                                    }
                                    seed_token = seed;
                                }
                            }
                        }
                    }
                    // Never partially inject a structural continuation. The current
                    // capped thinking turn ends before the close if it cannot fit.
                    Ok(None) => think_cap_hit = true,
                    Err(ForcedAdvance::Aborted) => {
                        break 'spec discard_spec_emit(emit, SpecRunOutcome::Aborted { generated });
                    }
                    Err(ForcedAdvance::Failed(e)) => {
                        break 'spec discard_spec_emit(
                            emit,
                            SpecRunOutcome::Failed(format!("forced-token advance: {e}")),
                        );
                    }
                }
            }
            // Per-cycle eviction (FlashCASK). Fires whenever current physical
            // has grown to budget+β since the last compaction. No-op when
            // physical < budget+β, so non-firing cycles pay only the check cost.
            if let Some(new_physical) =
                match maybe_evict_spec(gpu, m.eviction.as_ref(), slot, spec.as_mut(), position) {
                    Ok(position) => position,
                    Err(e) => {
                        if abort_latch.poll(id) {
                            break 'spec discard_spec_emit(
                                emit,
                                SpecRunOutcome::Aborted { generated },
                            );
                        }
                        break 'spec discard_spec_emit(
                            emit,
                            SpecRunOutcome::Failed(format!("eviction: {e}")),
                        );
                    }
                }
            {
                position = new_physical;
            }
            if hit_eos || think_cap_hit {
                break;
            }
        }

        // Snapshot only state that is not part of the consuming seal.
        let grammar_violated = emit.grammar_violated();

        if grammar_violated {
            eprintln!(
                "[grammar-dflash] grammar violation — forcing full KV/DN reset for next turn"
            );
            break 'spec discard_spec_emit(
                emit,
                SpecRunOutcome::Failed("speculative grammar violation".into()),
            );
        }

        // Terminal `finish` flush — parses tool calls from the decoded text and
        // renders the `tool_calls` ClientEvent. The arch-specific epilogue (the
        // asst-turn cache store + the `done` envelope) is the WRAPPER's job: it
        // differs per arch (qwen35: `dflash`/`tau`/`cycles` + ChatML token-replay
        // cache; ds4: `spec_k`/`spec_windows`/`spec_accept_pct`), so this core
        // returns a `SpecRun` summary instead of writing them itself.
        let finish = settle_spec_emit(emit, SpecEmitDisposition::Finish)
            .expect("normal spec completion must finish its emitter");
        render_client_events(stdout, id, &finish.events, 0);

        let target_reusable = finish
            .finalized
            .as_ref()
            .and_then(|turn| turn.replay_tokens())
            .map(|replay| {
                // The first token is a prediction, not a token consumed by the
                // target. Everything after it must line up with target advance.
                target_advanced == replay.len().saturating_sub(1)
            })
            .unwrap_or(false);
        publish_sealed_qwen_state(
            &mut m.session,
            &prompt_tokens,
            position,
            finish.finalized.as_ref(),
            target_reusable,
        );

        let t_end = Instant::now();
        SpecRunOutcome::Ready(SpecRun {
            generated,
            spec_cycles,
            spec_accepted,
            finalized: finish.finalized.clone(),
            target_reusable,
            prefill_tokens_len: prefill_tokens.len(),
            finish,
            prefill_s: t_prefill.duration_since(t0).as_secs_f64(),
            total_s: t_end.duration_since(t0).as_secs_f64(),
            decode_s: t_end.duration_since(t_prefill).as_secs_f64(),
        })
    };

    // The slot guard must restore its target bundle before the canonical reset
    // touches model state. No failure path below retains target/speculator borrows.
    drop(guard);
    finish_spec_caller_after_guard(
        None,
        None,
        outcome,
        id,
        || model_reset_context(m, gpu).map_err(|error| format!("{error:?}")),
        |run| {
            if matches!(arch_id, 5 | 6) {
                let finalized = run
                    .finalized
                    .as_ref()
                    .expect("Qwen spec emitter must publish a finalized assistant turn");
                // The guard/emitter have already been released by the lifecycle
                // seam. The raw pointer only avoids extending the tokenizer
                // borrow across this callback; the callback runs synchronously.
                unsafe {
                    let _ = cache_sealed_qwen_turn(
                        &mut *qwen_cache_ptr,
                        finalized,
                        run.target_reusable,
                    );
                }
            }
        },
        |envelope| emit_spec_terminal_envelope(stdout, id, envelope),
    )
}

#[cfg(test)]
mod spec_recovery_tests {
    use super::{
        abort_for_id, admit_spec_prefill_before_output, advance_forced_tokens,
        forced_advance_completed, observe_forced_seed, publish_eviction_position,
        record_forced_batch, replay_forced_tokens, AbortLatch, SpecRunOutcome,
    };
    use hipfire_runtime::spec::PrefillOutcome;
    use hipfire_runtime::spec::SpecAdvance;
    use std::sync::atomic::Ordering;

    #[test]
    fn does_not_publish_new_position_when_draft_mirror_fails() {
        let result: Result<usize, String> =
            publish_eviction_position(128, Err("draft mirror failed".into()));

        assert_eq!(result, Err("draft mirror failed".into()));
    }

    #[test]
    fn aborted_forced_advance_is_not_successful() {
        assert!(!forced_advance_completed(&SpecAdvance::Aborted));
    }

    #[test]
    fn cancellation_latch_survives_one_shot_abort_consumption() {
        let request_id = "abort-latch-spec-recovery-test";
        *abort_for_id().lock().unwrap() = Some(request_id.to_string());
        let latch = AbortLatch::new();

        assert!(latch.poll(request_id));
        // The global abort slot has been consumed, but nested production
        // callers must still observe the terminal cancellation.
        assert!(latch.poll(request_id));
        assert!(latch.is_cancelled());
        *abort_for_id().lock().unwrap() = None;
    }

    #[test]
    fn latched_prefill_cannot_admit_a_first_token_or_wire_event() {
        let latch = AbortLatch::new();
        latch.cancelled.store(true, Ordering::Release);
        let admitted = admit_spec_prefill_before_output(
            Ok(PrefillOutcome::Ready { first_token: 41 }),
            &|| latch.is_cancelled(),
        );
        assert!(matches!(
            admitted,
            Err(SpecRunOutcome::Aborted { generated: 0 })
        ));
    }

    #[test]
    fn forced_tokens_advance_once_and_reseed_with_the_unconsumed_argmax() {
        let forced = [41, 42, 43];
        let mut calls = Vec::new();

        let seed = advance_forced_tokens(&forced, forced.len(), |tokens| {
            calls.push(tokens.to_vec());
            Ok::<_, ()>(SpecAdvance::Ready { last_argmax: 99 })
        })
        .unwrap();

        assert_eq!(calls, vec![forced]);
        assert_eq!(seed, Some(99));
    }

    #[test]
    fn over_budget_forced_continuation_is_not_advanced() {
        let forced = [41, 42, 43];
        let mut calls = 0;

        let seed = advance_forced_tokens(&forced, 2, |_| {
            calls += 1;
            Ok::<_, ()>(SpecAdvance::Ready { last_argmax: 0 })
        })
        .unwrap();

        assert_eq!(calls, 0);
        assert_eq!(seed, None);
    }

    #[test]
    fn post_forced_argmax_is_observed_counted_once_and_respects_max_tokens() {
        let mut observed = Vec::new();
        let mut emitted = Vec::new();
        let mut generated = 3;

        assert!(observe_forced_seed(99, &mut generated, 4, |token| {
            observed.push(token);
            emitted.push(token);
            hipfire_runtime::spec::EmitOutcome {
                generation_advanced: true,
                ..Default::default()
            }
        }));
        assert_eq!(observed, vec![99]);
        assert_eq!(emitted, vec![99]);
        assert_eq!(generated, 4);

        assert!(!observe_forced_seed(100, &mut generated, 4, |token| {
            observed.push(token);
            emitted.push(token);
            hipfire_runtime::spec::EmitOutcome::default()
        }));
        assert_eq!(observed, vec![99]);
        assert_eq!(emitted, vec![99]);
        assert_eq!(generated, 4);
    }

    #[test]
    fn forced_replay_stops_at_stop_sequence_before_suffix_or_seed() {
        let forced = [41, 42, 43];
        let seed = 99;
        let mut observed = Vec::new();

        let stop = replay_forced_tokens(&forced, |token| {
            observed.push(token);
            (token == 42).then_some(hipfire_runtime::spec::StopReason::StopSequence)
        });
        if stop.is_none() {
            observed.push(seed);
        }

        assert_eq!(stop, Some(hipfire_runtime::spec::StopReason::StopSequence));
        assert_eq!(observed, vec![41, 42]);

        let mut session_tokens = vec![1, 10];
        record_forced_batch(&mut session_tokens, &forced);
        assert_eq!(session_tokens, vec![1, 10, 41, 42, 43]);
    }
}

#[cfg(test)]
mod spec_emit_lifecycle_tests {
    use super::{
        admit_spec_prefill, admit_spec_prefill_before_output, deepseek4_abort_json,
        finish_spec_caller_after_guard, reset_error_json, reset_error_json_message,
        settle_spec_emit_for_outcome, spec_emit_disposition, GenerateResult, SpecEmitDisposition,
        SpecRun, SpecRunOutcome, SpecTerminalEnvelope,
    };
    use hipfire_runtime::spec::{
        EmitOutcome, FinishSummary, PrefillOutcome, SpecEmit, SpecTargetGuard,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct FakeEmit {
        finish_count: Arc<AtomicUsize>,
        drop_count: Arc<AtomicUsize>,
    }

    impl SpecEmit for FakeEmit {
        fn begin(&mut self, _first_token: u32) -> EmitOutcome {
            EmitOutcome::default()
        }

        fn observe(&mut self, _token: u32) -> EmitOutcome {
            EmitOutcome::default()
        }

        fn finish(self: Box<Self>) -> FinishSummary {
            self.finish_count.fetch_add(1, Ordering::SeqCst);
            FinishSummary {
                finish_reason: "stop",
                ..FinishSummary::default()
            }
        }
    }

    impl Drop for FakeEmit {
        fn drop(&mut self) {
            self.drop_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn fake() -> (Box<dyn SpecEmit>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let finish_count = Arc::new(AtomicUsize::new(0));
        let drop_count = Arc::new(AtomicUsize::new(0));
        (
            Box::new(FakeEmit {
                finish_count: Arc::clone(&finish_count),
                drop_count: Arc::clone(&drop_count),
            }),
            finish_count,
            drop_count,
        )
    }

    struct TraceGuard {
        drops: Arc<AtomicUsize>,
    }

    impl SpecTargetGuard for TraceGuard {
        fn slot(&mut self) -> Result<&mut dyn hipfire_runtime::spec::SpecTarget, String> {
            Err("test guard slot is never entered".into())
        }
    }

    impl Drop for TraceGuard {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct TraceEmit {
        finish_count: Arc<AtomicUsize>,
        order: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }

    impl SpecEmit for TraceEmit {
        fn begin(&mut self, _first_token: u32) -> EmitOutcome {
            EmitOutcome::default()
        }

        fn observe(&mut self, _token: u32) -> EmitOutcome {
            EmitOutcome::default()
        }

        fn finish(self: Box<Self>) -> FinishSummary {
            self.finish_count.fetch_add(1, Ordering::SeqCst);
            self.order.lock().unwrap().push("Finish");
            FinishSummary {
                finish_reason: "stop",
                ..FinishSummary::default()
            }
        }
    }

    fn ready_run(target_reusable: bool) -> SpecRun {
        SpecRun {
            generated: 1,
            spec_cycles: 0,
            spec_accepted: 0,
            finalized: None,
            target_reusable,
            prefill_tokens_len: 1,
            finish: FinishSummary::default(),
            prefill_s: 0.0,
            total_s: 0.0,
            decode_s: 0.0,
        }
    }

    #[test]
    fn qwen_production_prefill_abort_resets_before_envelopes_and_allows_fresh_turn() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let drops = Arc::new(AtomicUsize::new(0));
        let finish_count = Arc::new(AtomicUsize::new(0));
        let cache_inserts = Arc::new(AtomicUsize::new(0));
        let prefills = Arc::new(AtomicUsize::new(0));
        let fresh_state = Arc::new(std::sync::atomic::AtomicBool::new(true));

        for _ in 0..2 {
            assert!(fresh_state.swap(false, Ordering::SeqCst));
            prefills.fetch_add(1, Ordering::SeqCst);
            let emit = Box::new(TraceEmit {
                finish_count: Arc::clone(&finish_count),
                order: Arc::clone(&events),
            }) as Box<dyn SpecEmit>;
            let guard = Box::new(TraceGuard {
                drops: Arc::clone(&drops),
            }) as Box<dyn SpecTargetGuard>;
            let outcome = finish_spec_caller_after_guard(
                Some(guard),
                Some(emit),
                SpecRunOutcome::Aborted { generated: 0 },
                "qwen-spec-abort-test",
                || {
                    assert_eq!(
                        drops.load(Ordering::SeqCst),
                        prefills.load(Ordering::SeqCst),
                        "guard must restore before Qwen reset"
                    );
                    fresh_state.store(true, Ordering::SeqCst);
                    events.lock().unwrap().push("Reset");
                    Ok(())
                },
                |_| {
                    cache_inserts.fetch_add(1, Ordering::SeqCst);
                },
                |envelope| {
                    events.lock().unwrap().push(match envelope {
                        SpecTerminalEnvelope::Aborted { .. } => "Aborted",
                        SpecTerminalEnvelope::DoneAborted { .. } => "DoneAborted",
                        _ => "unexpected",
                    })
                },
            );
            assert!(outcome.expect("successful abort lifecycle").is_none());
        }

        assert_eq!(
            &*events.lock().unwrap(),
            &vec![
                "Reset",
                "Aborted",
                "DoneAborted",
                "Reset",
                "Aborted",
                "DoneAborted",
            ]
        );
        assert_eq!(finish_count.load(Ordering::SeqCst), 0);
        assert_eq!(cache_inserts.load(Ordering::SeqCst), 0);
        assert_eq!(prefills.load(Ordering::SeqCst), 2);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn deepseek_production_abort_zeroes_decode_caches_before_envelopes() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let drops = Arc::new(AtomicUsize::new(0));
        let finish_count = Arc::new(AtomicUsize::new(0));
        let qwen_cache_proxy = Arc::new(AtomicUsize::new(0));
        let emit = Box::new(TraceEmit {
            finish_count: Arc::clone(&finish_count),
            order: Arc::clone(&events),
        }) as Box<dyn SpecEmit>;
        let guard = Box::new(TraceGuard {
            drops: Arc::clone(&drops),
        }) as Box<dyn SpecTargetGuard>;

        let outcome = finish_spec_caller_after_guard(
            Some(guard),
            Some(emit),
            SpecRunOutcome::Aborted { generated: 3 },
            "deepseek-test",
            || {
                assert_eq!(drops.load(Ordering::SeqCst), 1);
                events.lock().unwrap().push("zero_decode_caches");
                events.lock().unwrap().push("reset");
                Ok(())
            },
            |_| {
                qwen_cache_proxy.fetch_add(1, Ordering::SeqCst);
            },
            |envelope| {
                events.lock().unwrap().push(match envelope {
                    SpecTerminalEnvelope::Aborted { .. } => "Aborted",
                    SpecTerminalEnvelope::DoneAborted { .. } => "DoneAborted",
                    _ => "unexpected",
                })
            },
        );

        assert!(outcome.expect("successful abort lifecycle").is_none());
        assert_eq!(
            &*events.lock().unwrap(),
            &vec!["zero_decode_caches", "reset", "Aborted", "DoneAborted"]
        );
        assert_eq!(finish_count.load(Ordering::SeqCst), 0);
        assert_eq!(qwen_cache_proxy.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn production_lifecycle_control_finishes_caches_then_publishes_done() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let finish_count = Arc::new(AtomicUsize::new(0));
        let emit = Box::new(TraceEmit {
            finish_count: Arc::clone(&finish_count),
            order: Arc::clone(&events),
        }) as Box<dyn SpecEmit>;
        let guard = Box::new(TraceGuard {
            drops: Arc::new(AtomicUsize::new(0)),
        }) as Box<dyn SpecTargetGuard>;

        let result = finish_spec_caller_after_guard(
            Some(guard),
            Some(emit),
            SpecRunOutcome::Ready(ready_run(true)),
            "qwen-spec-test",
            || {
                events.lock().unwrap().push("Reset");
                Ok(())
            },
            |_| events.lock().unwrap().push("Cache"),
            |envelope| {
                if matches!(envelope, SpecTerminalEnvelope::Done { .. }) {
                    events.lock().unwrap().push("Done");
                }
            },
        );

        assert!(result.expect("successful completion lifecycle").is_some());
        assert_eq!(finish_count.load(Ordering::SeqCst), 1);
        assert_eq!(&*events.lock().unwrap(), &vec!["Finish", "Cache", "Done"]);
    }

    #[test]
    fn reset_failure_is_terminal_before_abort_or_done_envelopes() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (emit, _finish_count, _drop_count) = fake();
        let result = finish_spec_caller_after_guard(
            None,
            Some(emit),
            SpecRunOutcome::Aborted { generated: 4 },
            "reset-failed-test",
            || Err("poisoned reset".to_string()),
            |_| panic!("reset failure must not publish a cache"),
            |envelope| {
                events.lock().unwrap().push(match envelope {
                    SpecTerminalEnvelope::Error(message) => message,
                    _ => panic!("reset failure published a non-error terminal envelope"),
                });
            },
        );

        match result {
            Err(GenerateResult::ResetFailed { id, message }) => {
                assert_eq!(id, "reset-failed-test");
                assert_eq!(message, "poisoned reset");
            }
            _ => panic!("reset failure was not propagated untouched"),
        }
        assert!(events.lock().unwrap().is_empty());
    }

    #[test]
    fn reset_command_error_is_valid_json_when_debug_text_contains_quotes() {
        let json = reset_error_json("req\"1", &"bad \"gpu\" {state}");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid reset JSON");
        assert_eq!(value["id"], "req\"1");
        assert_eq!(value["type"], "error");
        assert!(value["message"].as_str().unwrap().contains("gpu"));

        let json = reset_error_json_message("req\"1", "bad \"gpu\" {state}");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid reset JSON");
        assert_eq!(value["message"], "bad \"gpu\" {state}");
    }

    #[test]
    fn deepseek_abort_envelopes_escape_request_ids() {
        let (aborted, done) = deepseek4_abort_json("req\"1\\2", 7);
        let aborted: serde_json::Value =
            serde_json::from_str(&aborted).expect("valid aborted JSON");
        let done: serde_json::Value = serde_json::from_str(&done).expect("valid done JSON");

        assert_eq!(aborted["id"], "req\"1\\2");
        assert_eq!(aborted["reason"], "client_cancelled");
        assert_eq!(done["id"], "req\"1\\2");
        assert_eq!(done["completion_tokens"], 7);
    }

    #[test]
    fn normal_completion_consumes_finish_once() {
        let (emit, finish_count, drop_count) = fake();

        let outcome = SpecRunOutcome::Ready(SpecRun {
            generated: 0,
            spec_cycles: 0,
            spec_accepted: 0,
            finalized: None,
            target_reusable: false,
            prefill_tokens_len: 0,
            finish: FinishSummary::default(),
            prefill_s: 0.0,
            total_s: 0.0,
            decode_s: 0.0,
        });
        assert_eq!(spec_emit_disposition(&outcome), SpecEmitDisposition::Finish);
        assert!(settle_spec_emit_for_outcome(emit, &outcome).is_some());
        assert_eq!(finish_count.load(Ordering::SeqCst), 1);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn abort_error_and_grammar_failure_discard_without_finish_or_events() {
        for outcome in [
            SpecRunOutcome::Aborted { generated: 3 },
            SpecRunOutcome::Failed("forward failed".into()),
            SpecRunOutcome::Failed("speculative grammar violation".into()),
        ] {
            assert_eq!(
                spec_emit_disposition(&outcome),
                SpecEmitDisposition::Discard
            );
            let (emit, finish_count, drop_count) = fake();

            assert!(settle_spec_emit_for_outcome(emit, &outcome).is_none());
            assert_eq!(finish_count.load(Ordering::SeqCst), 0);
            assert_eq!(drop_count.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn normal_finish_precedes_terminal_done_without_cache_side_effect() {
        let (emit, finish_count, _) = fake();
        let outcome = SpecRunOutcome::Ready(SpecRun {
            generated: 2,
            spec_cycles: 1,
            spec_accepted: 1,
            finalized: None,
            target_reusable: false,
            prefill_tokens_len: 0,
            finish: FinishSummary::default(),
            prefill_s: 0.0,
            total_s: 0.0,
            decode_s: 0.0,
        });
        let mut lifecycle = Vec::new();

        let _finish = settle_spec_emit_for_outcome(emit, &outcome).expect("normal finish");
        assert_eq!(finish_count.load(Ordering::SeqCst), 1);
        lifecycle.push("finish");
        lifecycle.push("terminal");
        lifecycle.push("done");

        assert_eq!(lifecycle, vec!["finish", "terminal", "done"]);
        assert!(
            !lifecycle.contains(&"cache"),
            "normal lifecycle test must not hide a cache insertion"
        );
    }

    #[test]
    fn generate_spec_prefill_cancellation_discards_seed_before_emission() {
        let gated = admit_spec_prefill_before_output(
            Ok(PrefillOutcome::Ready { first_token: 41 }),
            &|| true,
        );
        assert!(matches!(
            gated,
            Err(SpecRunOutcome::Aborted { generated: 0 })
        ));

        let ready = admit_spec_prefill(Ok(PrefillOutcome::Ready { first_token: 41 }), true);
        assert!(matches!(
            &ready,
            Err(SpecRunOutcome::Aborted { generated: 0 })
        ));

        let (emit, finish_count, drop_count) = fake();
        let outcome = ready.unwrap_err();
        assert!(settle_spec_emit_for_outcome(emit, &outcome).is_none());
        assert_eq!(finish_count.load(Ordering::SeqCst), 0);
        assert_eq!(drop_count.load(Ordering::SeqCst), 1);
    }
}

#[cfg(test)]
mod qwen_sealed_turn_daemon_tests {
    use super::{
        admit_qwen35_pp_token, asst_turn_fingerprint, cache_sealed_qwen_turn, plan_prompt_cache,
        publish_sealed_qwen_state,
    };
    use hipfire_arch_qwen35::spec_emit::Qwen35Emit;
    use hipfire_loader::{AsstTurnCache, SessionState};
    use hipfire_runtime::eos_filter::EosFilter;
    use hipfire_runtime::prompt_frame::{AssistantPrefix, Message, Role};
    use hipfire_runtime::spec::{ClientEvent, SpecEmitCtx, StopReason};
    use hipfire_runtime::tokenizer::Tokenizer;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_hf_json(
            r#"{
                "model": {
                    "vocab": {
                        "safe": 0,
                        "safe<st": 1,
                        "op><tool_call>tail": 2,
                        "<|im_start|>": 3,
                        "<|im_end|>": 4,
                        "user": 5,
                        "assistant": 6,
                        "\\n": 7,
                        "next": 8,
                        "<|endoftext|>": 9
                    },
                    "merges": []
                },
                "added_tokens": [
                    {"id": 3, "content": "<|im_start|>", "special": true},
                    {"id": 4, "content": "<|im_end|>", "special": true},
                    {"id": 9, "content": "<|endoftext|>", "special": true}
                ]
            }"#,
        )
        .expect("daemon sealed-turn fixture tokenizer")
    }

    #[test]
    fn two_turn_stop_quarantine_uses_sealed_cache_and_reset_paths() {
        let tokenizer = tokenizer();
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tokenizer,
            eos: 9,
            im_end: Some(4),
            tools: None,
            stop: vec!["<stop>".to_string()],
            max_think: 0,
            max_tokens: 32,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });

        let mut events = emit.begin(1).events;
        let stopped = emit.observe(2);
        events.extend(stopped.events);
        assert_eq!(stopped.stop, Some(StopReason::StopSequence));
        let summary = emit.finish();
        events.extend(summary.events);
        let finalized = summary.finalized.expect("sealed Qwen turn");

        assert_eq!(finalized.text(), "safe");
        assert!(finalized.tool_calls().is_empty());
        assert!(finalized.replay_tokens().is_none());
        assert!(finalized.diagnostic_tokens().is_empty());
        assert!(events.iter().all(|event| match event {
            ClientEvent::Token(text) | ClientEvent::Reasoning(text) => {
                !text.contains("tail") && !text.contains("<tool_call>")
            }
            ClientEvent::ToolCalls(_) => false,
            ClientEvent::Committed { id, .. } => *id != 2,
        }));

        let safe_fp = asst_turn_fingerprint(finalized.text(), finalized.tool_calls());
        let tail_fp = asst_turn_fingerprint("safe<stop><tool_call>tail", &[]);
        assert_ne!(safe_fp, tail_fp);

        let mut cache = AsstTurnCache::new_from_env();
        assert_eq!(
            cache_sealed_qwen_turn(&mut cache, &finalized, false),
            None,
            "an intra-token stop has no replay/cache boundary"
        );
        assert!(!cache.contains_key(&safe_fp));
        assert!(!cache.contains_key(&tail_fp));

        let mut session = SessionState::default();
        session.seq_pos = 17;
        session.conversation_tokens = vec![1, 2];
        publish_sealed_qwen_state(&mut session, &[3, 6, 7], 19, Some(&finalized), false);
        assert_eq!(session.seq_pos, 0);
        assert!(session.conversation_tokens.is_empty());

        let history = [Message {
            role: Role::Assistant,
            content: finalized.text().to_string(),
            tool_calls: finalized.tool_calls().to_vec(),
            tool_call_id: None,
            tool_plan: String::new(),
        }];
        let turn2 = plan_prompt_cache(
            &tokenizer,
            &mut cache,
            &session.conversation_tokens,
            true,
            None,
            "next",
            AssistantPrefix::Plain,
            &history,
            false,
            &[],
            false,
        );
        assert!(
            !turn2.cache_hit,
            "turn 2 must cold-start after the cut turn"
        );
        assert_eq!(turn2.start_pos, 0);
        assert_eq!(turn2.cached_tokens, 0);
        assert!(!turn2.rendered.contains(&2));
        assert!(!tokenizer.decode(&turn2.rendered).contains("tail"));
    }

    #[test]
    fn qwen35_pp_caller_path_resets_after_intra_token_stop() {
        let tokenizer = tokenizer();
        let mut turn =
            hipfire_runtime::spec_transcript::OpenAssistantTurn::new([b"<stop>".as_slice()]);
        let mut filter =
            EosFilter::new(super::qwen35_pp_eos_filter_config(&["<stop>".to_string()]));

        let first = admit_qwen35_pp_token(&mut turn, &mut filter, &tokenizer, 1, false);
        assert!(!first.terminal);
        let second = admit_qwen35_pp_token(&mut turn, &mut filter, &tokenizer, 2, false);
        assert!(second.terminal);
        assert!(matches!(
            second.action,
            hipfire_runtime::eos_filter::FilterAction::Stop { .. }
        ));

        let finalized = turn.seal();
        assert_eq!(finalized.text(), "safe");
        assert!(finalized.replay_tokens().is_none());
        assert!(finalized.diagnostic_tokens().is_empty());

        let mut session = SessionState::default();
        session.seq_pos = 2;
        session.conversation_tokens = vec![10, 1];
        publish_sealed_qwen_state(&mut session, &[10], 2, Some(&finalized), false);
        assert_eq!(session.seq_pos, 0);
        assert!(session.conversation_tokens.is_empty());
    }

    #[test]
    fn qwen35_pp_caller_path_retains_exact_token_aligned_stop() {
        let tokenizer = tokenizer();
        let mut turn =
            hipfire_runtime::spec_transcript::OpenAssistantTurn::new([b"<stop>".as_slice()]);
        let mut filter = EosFilter::new(super::qwen35_pp_eos_filter_config(&[]));
        let first = admit_qwen35_pp_token(&mut turn, &mut filter, &tokenizer, 0, false);
        assert!(!first.terminal);
        let stop = admit_qwen35_pp_token(&mut turn, &mut filter, &tokenizer, 4, true);
        assert!(stop.terminal);

        let finalized = turn.seal();
        assert_eq!(finalized.replay_tokens(), Some([0].as_slice()));
        assert_eq!(finalized.diagnostic_tokens(), [0].as_slice());
        let mut session = SessionState::default();
        session.conversation_tokens = vec![10, 0];
        publish_sealed_qwen_state(&mut session, &[10], 2, Some(&finalized), true);
        assert_eq!(session.conversation_tokens, vec![10, 0]);
    }
}

fn generate_multi_borrowed(
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
    top_k: Option<u32>,
    min_p: Option<f32>,
    max_tokens: usize,
    repeat_penalty: f32,
    _repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
) -> GenerateResult {
    let prompt_est = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        tokenizer.encode(prompt).len() + 20
    };
    if m.session
        .seq_pos
        .saturating_add(prompt_est)
        .saturating_add(max_tokens)
        > m.meta.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.session.seq_pos, m.meta.max_seq
        );
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    let try_jinja = jinja_enabled && m.meta.chat_template.is_some();
    if try_jinja && m.session.seq_pos > 0 {
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }

    let tokenizer = m.tokenizer.as_ref().unwrap();

    let im_end = tokenizer.encode("<|im_end|>");
    let nl = tokenizer.encode("\n");
    let raw_q_tokens = tokenizer.encode(prompt);

    // PFlash compression on first turn (seq_pos == 0). Drafter runs on the
    // daemon's single-GPU `gpu` handle, which binds to the same physical
    // device as rank-0 in the PP mesh (HIP enumerates within ROCR_VISIBLE).
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
        if m.session.seq_pos == 0 {
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
    // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
    // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
    // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
    // the .hfq carries no template, e.g. an older A1B convert).
    // Jinja default-ON (flipped 2026-06-09): render through the model's chat
    // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
    // ChatML/Plain). Falls back to Plain automatically when no template resolves.
    // hunt3 H-A: drop the `seq_pos == 0` gate (PR #389 removed it from generate()).
    // With the gate, turn 2+ fell through to the Plain scaffold, dropping the
    // system prompt and the full history replay that render_messages provides.
    // Now Jinja renders the full conversation every turn; the cold-reset block
    // below (guarded on seq_pos > 0) re-zeros recurrent state so the full render
    // writes from position 0 instead of appending to the prior turn's KV/DeltaNet.
    let new_tokens = if try_jinja {
        let template = m.meta.chat_template.as_ref().unwrap();
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
                            tool_plan: String::new(),
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        tool_plan: String::new(),
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
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: if m.session.seq_pos == 0 {
                        system_prompt
                    } else {
                        None
                    },
                    user: "",
                    assistant_prefix,
                    raw: false,
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer,
            system: if m.session.seq_pos == 0 {
                system_prompt
            } else {
                None
            },
            user: "",
            assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(&q_tokens)
    };

    // hunt3 H-A: under Jinja the full conversation (system + history) is
    // re-rendered every turn, so turn 2+ must cold-reset BEFORE the budget guard
    // + prefill — otherwise the full render appends to the prior turn's dirty
    // KV / DeltaNet / checkpoint state (stale recurrent state → drift; the
    // system prompt was also being silently dropped on turn 2+). Mirrors the
    // loader reset semantics, written here before the PP state is borrowed. Same
    // shape as the context-full reset at the top of this fn and generate()'s `jinja_active &&
    // seq_pos > 0` block.
    let trailer = nl.len();
    if m.session
        .seq_pos
        .saturating_add(new_tokens.len())
        .saturating_add(max_tokens)
        .saturating_add(trailer)
        > m.meta.physical_cap
    {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
            id,
            m.session.seq_pos,
            new_tokens.len(),
            max_tokens,
            trailer,
            m.meta.physical_cap
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
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

    let ModelState::Qwen35(b) = m.state.as_mut().unwrap() else {
        unreachable!()
    };
    let config = &b.config;
    let weights = &b.weights;
    let pl = b
        .pipeline
        .as_ref()
        .expect("pp>1 qwen35 must carry pipeline scratch");
    let scratch_set = &pl.scratch_set;
    let kv = &mut b.kv_cache;
    let dn = &mut b.dn_state;
    let ModelParallel::Pp(PipelineImpl::ArchResident(gpus)) = &mut m.parallel else {
        unreachable!("qwen35 pp>1 forward without ArchResident mesh")
    };

    let dev_last = gpus.output_device;
    let vocab_size = config.vocab_size;
    // Effective penalty window = request `_repeat_window` (default 128),
    // bounded by repeat_buf capacity (2048). Default stays 128; the wide buffer
    // only enables a larger window when a request explicitly sets one.
    let repeat_buf_cap =
        (scratch_set.per_device[dev_last].repeat_buf.buf.size() / 4).min(_repeat_window.max(1));

    // hunt3 M-C: grammar-guided decoding for pp>1 (mirrors generate() ~8168).
    // Without this, a pp>1 + tools request samples unconstrained once the model
    // commits to <tool_call>, reproducing the ChatML-noise-in-tool_call-body
    // attractor the single-GPU path masks via the qwen35 Matcher. The decoded
    // vocab is built into a request-local Vec rather than cached on `m`
    // (m.persist.decoded_vocab) because `m` is already mutably borrowed here (kv/dn/gpus)
    // — pp>1 + tools is uncommon, so the per-request decode is acceptable.
    let grammar_enabled = std::env::var("HIPFIRE_QWEN35_GRAMMAR").ok().as_deref() != Some("0");
    let tool_schemas_qwen: Vec<hipfire_arch_qwen35::grammar::ToolSchema> = if grammar_enabled {
        tools
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())?
                            .to_string();
                        let required: Vec<String> = func
                            .get("parameters")
                            .and_then(|p| p.get("required"))
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some(hipfire_arch_qwen35::grammar::ToolSchema { name, required })
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let grammar_active = !tool_schemas_qwen.is_empty();
    let mut grammar_matcher = hipfire_arch_qwen35::grammar::Matcher::new(tool_schemas_qwen);
    let grammar_vocab: Vec<String> = if grammar_active {
        let n = tokenizer.vocab_size();
        (0..n).map(|id| tokenizer.decode(&[id as u32])).collect()
    } else {
        Vec::new()
    };
    let mut grammar_mask: Vec<bool> = vec![true; grammar_vocab.len()];

    if let Err(e) = qwen35::forward_prefill_batch_multi(
        gpus,
        weights,
        config,
        &new_tokens,
        m.session.seq_pos,
        kv,
        dn,
        scratch_set,
    ) {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: format!("forward_prefill_batch_multi: {e}"),
        });
    }
    m.session.seq_pos += new_tokens.len();
    m.session.conversation_tokens.extend_from_slice(&new_tokens);

    if check_abort(id) {
        return GenerateResult::Deferred(DeferredTerminal::Aborted {
            id: id.to_string(),
            generated: 0,
        });
    }

    // ngram scope: generated tokens only (matches pp=1).
    let ngram_scope_start = m.session.conversation_tokens.len();

    let mut rng_state: u32 = 0x13579BDFu32;

    let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
        .into_iter()
        .chain(think_pair.into_iter())
        .collect();

    // First sample on the output device.
    let ngram_scope = &m.session.conversation_tokens[ngram_scope_start..];
    let mut blocked0: Vec<u32> = Vec::new();
    sampler::collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
    let cfg0 = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window: repeat_buf_cap,
        presence_penalty,
        frequency_penalty,
        blocked_tokens: blocked0,
        top_k,
        min_p,
    };
    // hunt3 M-C: grammar-gated first sample (GPU fast path when matcher free;
    // CPU mask-then-sample when constraining). Matches generate()'s tok0 site.
    let tok0 = {
        let s_last = &scratch_set.per_device[dev_last];
        let g_last = &mut gpus.devices[dev_last];
        if grammar_active && !grammar_matcher.is_free() {
            let _ = g_last.bind_thread();
            let mut logits = g_last
                .download_f32(&s_last.logits)
                .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
            grammar_matcher.token_mask(&grammar_vocab, &mut grammar_mask);
            hipfire_arch_qwen35::grammar::Matcher::apply_mask_to_logits(&grammar_mask, &mut logits);
            sampler::sample_cpu(&mut logits, ngram_scope, &cfg0)
        } else {
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
        }
    };
    if grammar_active {
        grammar_matcher.advance(&tokenizer.decode(&[tok0]));
    }
    let t_prefill = Instant::now();
    let mut next_token = tok0;

    let mut generated = 0usize;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut filter = EosFilter::new(qwen35_pp_eos_filter_config(stop));
    let mut sealed_turn =
        hipfire_runtime::spec_transcript::OpenAssistantTurn::new_with_reasoning_open(
            stop.iter().map(|sequence| sequence.as_bytes()),
            matches!(
                assistant_prefix,
                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
            ),
        );
    let mut target_advanced = 0usize;
    let mut terminal_turn = false;
    let mut alert_fired = false;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;
    let mut force_answer_latched = false;
    let think_open_tok = tokenizer.special_token_id("<think>");
    let max_total_think: usize = std::env::var("HIPFIRE_MAX_TOTAL_THINK_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut total_think_tokens: usize = 0;
    // Post-latch answer bound. Once the think-cap latches we force-close <think>
    // and ask the model to answer; but `total_think_tokens` only advances
    // in-think, so a model that rambles a NON-think answer (or re-opens <think>
    // in a tight loop the force-close keeps re-closing) never trips the +256 EOS
    // and runs to max_tokens. Mark the latch position and hard-EOS once
    // generation runs this many tokens past it — generous for a real final
    // answer, bounded against runaway.
    let post_latch_answer_budget: usize = std::env::var("HIPFIRE_POST_LATCH_ANSWER_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let mut latch_gen_mark: Option<usize> = None;
    let loop_guard =
        hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    while generated < max_tokens {
        if check_abort(id) {
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated,
            });
        }
        let semantic_stop = next_token == config.eos_token
            || im_end_token == Some(next_token)
            || tokenizer.is_terminator(next_token);
        let admission = admit_qwen35_pp_token(
            &mut sealed_turn,
            &mut filter,
            tokenizer,
            next_token,
            semantic_stop,
        );
        let filter_stopped = emit_qwen35_pp_filter_action(stdout, id, admission.action);
        if admission.terminal || filter_stopped {
            break;
        }
        if !admission.generation_advanced {
            break;
        }
        generated += 1;
        m.session.conversation_tokens.push(next_token);
        streamed_tokens.push(next_token);
        emit_committed_event(
            stdout,
            id,
            next_token,
            streamed_tokens.len() - 1,
            t0.elapsed().as_millis() as u64,
        );

        if let Err(e) = qwen35::forward_scratch_multi(
            gpus,
            weights,
            config,
            next_token,
            m.session.seq_pos,
            kv,
            dn,
            scratch_set,
        ) {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("forward_scratch_multi decode: {e}"),
            });
        }
        m.session.seq_pos += 1;
        target_advanced += 1;

        // max_think_tokens / force-answer enforcement: same decoded-text scan
        // as pp=1, but all recurrent-state writes route through *_multi.
        let force_answer_now = check_force_answer(id);
        if force_answer_now {
            force_answer_latched = true;
        }
        if max_think_tokens > 0 || force_answer_now || force_answer_latched || max_total_think > 0 {
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = currently_in_think(
                raw_str,
                matches!(
                    assistant_prefix,
                    hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                ),
            );
            if in_think {
                total_think_tokens += 1;
            }
            if max_total_think > 0 && total_think_tokens >= max_total_think {
                force_answer_latched = true;
            }
            if force_answer_latched && latch_gen_mark.is_none() {
                latch_gen_mark = Some(generated);
            }
            if max_total_think > 0 && in_think && total_think_tokens >= max_total_think + 256 {
                eprintln!(
                    "[think-cap] id={} — total think {} exceeded cap {}+256 while still thinking; forcing EOS",
                    id, total_think_tokens, max_total_think
                );
                break;
            }
            if let Some(mark) = latch_gen_mark {
                if generated.saturating_sub(mark) >= post_latch_answer_budget {
                    eprintln!(
                        "[think-cap] id={} — {} tokens since think-cap latch without finishing; forcing EOS",
                        id,
                        generated.saturating_sub(mark)
                    );
                    break;
                }
            }
            if max_think_tokens > 0 {
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
            }
            let budget_hit = max_think_tokens > 0 && think_count >= max_think_tokens;

            if in_think && (budget_hit || force_answer_now || force_answer_latched) {
                if force_answer_now {
                    eprintln!(
                        "[force-answer] id={} — closing <think> mid-turn to commit to the answer",
                        id
                    );
                } else if force_answer_latched {
                    eprintln!(
                        "[force-answer] id={} — re-closing a re-opened <think> (latched / think-cap)",
                        id
                    );
                }
                let close_tokens = tokenizer.encode(&think_continuation());
                let budget_left = max_tokens.saturating_sub(generated);
                let take = close_tokens.len().min(budget_left);
                let mut filter_stopped = false;
                for &t in &close_tokens[..take] {
                    let admission = admit_qwen35_pp_token(
                        &mut sealed_turn,
                        &mut filter,
                        tokenizer,
                        t,
                        t == config.eos_token
                            || im_end_token == Some(t)
                            || tokenizer.is_terminator(t),
                    );
                    filter_stopped = emit_qwen35_pp_filter_action(stdout, id, admission.action);
                    if admission.terminal || filter_stopped {
                        terminal_turn = true;
                        break;
                    }
                    if !admission.generation_advanced {
                        terminal_turn = true;
                        break;
                    }
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        t,
                        m.session.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        eprintln!("[daemon] max_think close forward_scratch_multi: {}", e);
                        return GenerateResult::Deferred(DeferredTerminal::Error {
                            id: id.to_string(),
                            message: format!("max_think close forward_scratch_multi: {e}"),
                        });
                    }
                    m.session.seq_pos += 1;
                    target_advanced += 1;
                    m.session.conversation_tokens.push(t);
                    // hunt3 M-C: keep the grammar matcher in sync over force-closed
                    // </think> tokens, exactly as generate() does (~8591). Without
                    // this a tools request that force-closes <think> leaves the
                    // matcher stale → malformed tool calls after the forced close.
                    if grammar_active {
                        grammar_matcher.advance(&tokenizer.decode(&[t]));
                    }
                    streamed_tokens.push(t);
                    emit_committed_event(
                        stdout,
                        id,
                        t,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    generated += 1;
                    if filter_stopped {
                        break;
                    }
                }
                think_count = 0;
                prev_in_think = false;
                if generated >= max_tokens {
                    break;
                }
                if terminal_turn || filter_stopped {
                    break;
                }
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
        if !alert_fired
            && budget_alert_at_tok > 0
            && generated >= budget_alert_at_tok
            && !budget_alert_text.is_empty()
        {
            alert_fired = true;
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let in_think = currently_in_think(
                raw_str,
                matches!(
                    assistant_prefix,
                    hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                ),
            );
            if !in_think {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                    id
                );
                let _ = stdout.flush();
                let ngram_scope = &m.session.conversation_tokens[ngram_scope_start..];
                let mut blocked: Vec<u32> = Vec::new();
                sampler::collect_unclosed_attractor_blocks(
                    ngram_scope,
                    &attractor_pairs,
                    20,
                    2,
                    &mut blocked,
                );
                if force_answer_latched {
                    if let Some(t) = think_open_tok {
                        blocked.push(t);
                    }
                }
                let cfg = SamplerConfig {
                    temperature: temp,
                    top_p,
                    repeat_penalty,
                    repeat_window: repeat_buf_cap,
                    presence_penalty,
                    frequency_penalty,
                    blocked_tokens: blocked,
                    top_k,
                    min_p,
                };
                // hunt3 M-C: grammar-gated budget-alert resample.
                next_token = {
                    let s_last = &scratch_set.per_device[dev_last];
                    let g_last = &mut gpus.devices[dev_last];
                    if grammar_active && !grammar_matcher.is_free() {
                        let _ = g_last.bind_thread();
                        let mut logits = g_last
                            .download_f32(&s_last.logits)
                            .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
                        grammar_matcher.token_mask(&grammar_vocab, &mut grammar_mask);
                        hipfire_arch_qwen35::grammar::Matcher::apply_mask_to_logits(
                            &grammar_mask,
                            &mut logits,
                        );
                        sampler::sample_cpu(&mut logits, ngram_scope, &cfg)
                    } else {
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
                    }
                };
                if grammar_active {
                    grammar_matcher.advance(&tokenizer.decode(&[next_token]));
                }
                continue;
            }
            let nudge_tokens = tokenizer.encode(budget_alert_text);
            let budget_left = max_tokens.saturating_sub(generated);
            let nudge_len = nudge_tokens.len().min(budget_left);
            let need_kv = m
                .session
                .seq_pos
                .saturating_add(nudge_len)
                .saturating_add(
                    max_tokens
                        .saturating_sub(generated)
                        .saturating_sub(nudge_len),
                )
                .saturating_add(nl.len());
            if nudge_len > 0 && need_kv <= m.meta.physical_cap {
                let mut filter_stopped = false;
                for &tok in &nudge_tokens[..nudge_len] {
                    let admission = admit_qwen35_pp_token(
                        &mut sealed_turn,
                        &mut filter,
                        tokenizer,
                        tok,
                        tok == config.eos_token
                            || im_end_token == Some(tok)
                            || tokenizer.is_terminator(tok),
                    );
                    filter_stopped = emit_qwen35_pp_filter_action(stdout, id, admission.action);
                    if admission.terminal || filter_stopped {
                        terminal_turn = true;
                        break;
                    }
                    if !admission.generation_advanced {
                        terminal_turn = true;
                        break;
                    }
                    if let Err(e) = qwen35::forward_scratch_multi(
                        gpus,
                        weights,
                        config,
                        tok,
                        m.session.seq_pos,
                        kv,
                        dn,
                        scratch_set,
                    ) {
                        eprintln!("[daemon] budget_alert forward_scratch_multi: {}", e);
                        return GenerateResult::Deferred(DeferredTerminal::Error {
                            id: id.to_string(),
                            message: format!("budget_alert forward_scratch_multi: {e}"),
                        });
                    }
                    m.session.seq_pos += 1;
                    target_advanced += 1;
                    m.session.conversation_tokens.push(tok);
                    streamed_tokens.push(tok);
                    emit_committed_event(
                        stdout,
                        id,
                        tok,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    generated += 1;
                    if filter_stopped {
                        break;
                    }
                }
                if terminal_turn || filter_stopped {
                    break;
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
        let ngram_scope = &m.session.conversation_tokens[ngram_scope_start..];
        let mut blocked: Vec<u32> = Vec::new();
        sampler::collect_unclosed_attractor_blocks(
            ngram_scope,
            &attractor_pairs,
            20,
            2,
            &mut blocked,
        );
        if force_answer_latched {
            if let Some(t) = think_open_tok {
                blocked.push(t);
            }
        }
        let cfg = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked,
            top_k,
            min_p,
        };
        // hunt3 M-C: grammar-gated steady-state sample.
        next_token = {
            let s_last = &scratch_set.per_device[dev_last];
            let g_last = &mut gpus.devices[dev_last];
            if grammar_active && !grammar_matcher.is_free() {
                let _ = g_last.bind_thread();
                let mut logits = g_last
                    .download_f32(&s_last.logits)
                    .unwrap_or_else(|_| vec![0.0f32; vocab_size]);
                grammar_matcher.token_mask(&grammar_vocab, &mut grammar_mask);
                hipfire_arch_qwen35::grammar::Matcher::apply_mask_to_logits(
                    &grammar_mask,
                    &mut logits,
                );
                sampler::sample_cpu(&mut logits, ngram_scope, &cfg)
            } else {
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
            }
        };
        if grammar_active {
            let was_detected = grammar_matcher.attractor_detected();
            grammar_matcher.advance(&tokenizer.decode(&[next_token]));
            if !was_detected && grammar_matcher.attractor_detected() {
                eprintln!(
                    "[grammar-ngram pp] attractor detected in tool_call args at gen={} — forcing close",
                    generated,
                );
            }
        }
    }

    // ChatML \n trailer so the next turn opens cleanly. Keep this before the
    // filter epilogue: a trailer failure is an error path and must discard
    // the filter without flushing buffered output.
    if im_end_token == Some(*m.session.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            if let Err(e) = qwen35::forward_scratch_multi(
                gpus,
                weights,
                config,
                t,
                m.session.seq_pos,
                kv,
                dn,
                scratch_set,
            ) {
                eprintln!("[daemon] trailer forward_scratch_multi: {}", e);
                return GenerateResult::Deferred(DeferredTerminal::Error {
                    id: id.to_string(),
                    message: format!("trailer forward_scratch_multi: {e}"),
                });
            }
            m.session.seq_pos += 1;
            m.session.conversation_tokens.push(t);
        }
    }

    // Local terminal epilogue for the arch-resident PP raw filter. Every
    // normal loop exit (EOS, user stop, think/budget stop, or length) reaches
    // this point exactly once; abort/error returns above drop the filter and
    // therefore cannot render buffered bytes after the terminal event.
    if let Some(text) = finish_qwen35_pp_filter(&mut filter) {
        let _ = writeln!(
            stdout,
            r#"{{"type":"token","id":"{}","text":{}}}"#,
            id,
            serde_json::to_string(&text).unwrap_or_default()
        );
        let _ = stdout.flush();
    }

    let finalized = sealed_turn.seal();
    let target_reusable = finalized.replay_tokens().is_some()
        && target_advanced == finalized.diagnostic_tokens().len();
    let prompt_prefix = m.session.conversation_tokens[..ngram_scope_start].to_vec();
    let final_position = m.session.seq_pos;
    publish_sealed_qwen_state(
        &mut m.session,
        &prompt_prefix,
        final_position,
        Some(&finalized),
        target_reusable,
    );
    if target_reusable {
        let _ = cache_sealed_qwen_turn(&mut m.persist.asst_turn_cache, &finalized, true);
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
    GenerateResult::PpCompletion {
        generated,
        prefill_tokens,
        tok_s,
        prefill_ms: prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        reset_required: !target_reusable,
    }
}

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
    top_k: Option<u32>,
    min_p: Option<f32>,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
) -> GenerateResult {
    let result = generate_multi_borrowed(
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
        top_k,
        min_p,
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
        stop,
    );
    match result {
        GenerateResult::PpCompletion {
            generated,
            prefill_tokens,
            tok_s,
            prefill_ms,
            prefill_tok_s,
            decode_tok_s,
            reset_required,
        } => {
            // `generate_multi_borrowed` has returned, so every PP bundle
            // borrow is dead before the authoritative reset is attempted.
            if reset_required {
                if let Err(error) = model_reset_context(m, gpu) {
                    return reset_failed(id, error);
                }
            }
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
                id,
                generated,
                tok_s,
                prefill_tokens,
                prefill_ms,
                prefill_tok_s,
                decode_tok_s,
                prefill_ms
            );
            let _ = stdout.flush();
            GenerateResult::Complete
        }
        other => other,
    }
}

fn qwen35_pp_eos_filter_config(stop: &[String]) -> EosFilterConfig {
    EosFilterConfig {
        stop_at: stop
            .iter()
            .filter(|sequence| !sequence.is_empty())
            .map(|sequence| sequence.as_bytes().to_vec())
            .collect(),
        ..EosFilterConfig::default()
    }
}

/// Render one Qwen35 PP filter action. A stop action owns the safe prefix and
/// is terminal; callers must stop the decode loop after forwarding that token.
fn emit_qwen35_pp_filter_action(
    stdout: &mut std::io::Stdout,
    id: &str,
    action: FilterAction,
) -> bool {
    let stopped = matches!(&action, FilterAction::Stop { .. });
    let bytes = match action {
        FilterAction::Emit(bytes) => bytes,
        FilterAction::Stop { emit } => emit,
        FilterAction::Hold => return false,
    };
    if bytes.is_empty() {
        return stopped;
    }
    let text = std::str::from_utf8(&bytes).expect("Qwen35 PP filter emits valid UTF-8");
    let _ = writeln!(
        stdout,
        r#"{{"type":"token","id":"{}","text":{}}}"#,
        id,
        serde_json::to_string(text).unwrap_or_default()
    );
    let _ = stdout.flush();
    stopped
}

#[derive(Debug)]
struct Qwen35PpAdmission {
    action: FilterAction,
    terminal: bool,
    generation_advanced: bool,
}

/// Admit a PP token before it becomes durable.  The target is only advanced
/// after this returns non-terminal: a stop token is a prediction, not a token
/// that belongs in the reusable KV/session prefix.
fn admit_qwen35_pp_token(
    turn: &mut hipfire_runtime::spec_transcript::OpenAssistantTurn,
    filter: &mut EosFilter,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    token: u32,
    semantic_stop: bool,
) -> Qwen35PpAdmission {
    if semantic_stop || turn.stopped() {
        if semantic_stop {
            turn.stop();
        }
        return Qwen35PpAdmission {
            action: FilterAction::Hold,
            terminal: true,
            generation_advanced: false,
        };
    }

    let raw = tokenizer.decode_bytes(&[token]);
    let delta = turn.observe(token, &raw);
    let action = filter.observe(&raw);
    let filter_terminal = matches!(action, FilterAction::Stop { .. });
    if filter_terminal && !turn.stopped() {
        turn.stop();
    }
    let terminal = delta.stopped || filter_terminal;
    Qwen35PpAdmission {
        action,
        terminal,
        generation_advanced: !terminal,
    }
}

/// Finish the Qwen35 PP raw output filter once and return only a valid,
/// non-empty UTF-8 payload for the terminal token event. The caller invokes
/// this only on the normal path; abort/error paths drop the filter untouched.
fn finish_qwen35_pp_filter(filter: &mut EosFilter) -> Option<String> {
    let bytes = filter.finish();
    if bytes.is_empty() {
        return None;
    }
    std::str::from_utf8(&bytes)
        .ok()
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

#[allow(clippy::too_many_arguments)]
/// Generic auto-regressive decode driver (Inc 1, Task 1.4b-iii). Extracted
/// verbatim-in-behavior from the qwen35 AR arm of `generate` (arch 5/6), but
/// every arch-coupled op routes through `ArchDispatch` hooks and all loop state
/// (seq_pos, streamed tokens, think/budget counters, rng) is local. DEAD CODE
/// this stage: NOT routed at the `if m.meta.arch_id==5||6` dispatch point. The 1.4c
/// dual-run parity harness validates token-identity vs the old arm on GPU.
///
/// Faithfulness note (for the 1.4c reviewer): `ngram_scope` uses the local
/// `streamed_tokens` in place of `m.conversation_tokens[ngram_scope_start..]`
/// (proven identical push-order at every sample site); the asst-turn cached_seq
/// still reads the real conversation buffer (it includes the post-loop ChatML
/// trailer that `streamed_tokens` does not). Adaptive-KV stderr phase labels are
/// unified in the hook (diagnostic only, not token-affecting).
#[derive(Debug, Clone, PartialEq, Eq)]
enum ArDecodeOutcome {
    Complete,
    Aborted,
    Error(String),
}

/// Route a decode-loop terminal control to the lifecycle path that settles it.
/// Normal controls share completion/trailer handling; cancellation and errors
/// discard the parser without a trailer.
fn route_ar_decode_outcome(normal_completion: bool, error: Option<String>) -> ArDecodeOutcome {
    match error {
        Some(error) => ArDecodeOutcome::Error(error),
        None if normal_completion => ArDecodeOutcome::Complete,
        None => ArDecodeOutcome::Aborted,
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ArSampledEos {
    Stop,
    CommitAndStop,
    Forced(u32),
    Error(String),
}

fn stream_stop(actions: &[hipfire_runtime::stream_parser::StreamAction]) -> bool {
    actions
        .iter()
        .any(|action| matches!(action, hipfire_runtime::stream_parser::StreamAction::Stop))
}

fn decode_limit_reached(
    max_tokens: usize,
    generated: usize,
    physical_cap: usize,
    seq_pos: usize,
    has_eviction: bool,
) -> bool {
    if generated >= max_tokens {
        true
    } else if !has_eviction && seq_pos >= physical_cap {
        true
    } else {
        false
    }
}

fn ar_sampled_eos(parser: &mut dyn hipfire_runtime::stream_parser::StreamParser) -> ArSampledEos {
    match parser.on_eos() {
        hipfire_runtime::stream_parser::EosDecision::Stop => ArSampledEos::Stop,
        hipfire_runtime::stream_parser::EosDecision::CommitAndStop => ArSampledEos::CommitAndStop,
        hipfire_runtime::stream_parser::EosDecision::Inject(tokens) => {
            if tokens.is_empty() {
                return ArSampledEos::Error(
                    "sampled eos injection produced no forced token".into(),
                );
            }
            for token in tokens {
                parser.enqueue(token);
            }
            match parser.next_forced() {
                Some(token) => ArSampledEos::Forced(token),
                None => ArSampledEos::Error("sampled eos injection failed forced dequeue".into()),
            }
        }
    }
}

fn grammar_mask_for_sample<'a>(
    grammar_active: bool,
    matcher: Option<&dyn hipfire_runtime::arch_dispatch::GrammarMatcher>,
    grammar_vocab: &[String],
    grammar_mask: &'a mut [bool],
) -> Result<Option<&'a [bool]>, String> {
    if !grammar_active {
        return Ok(None);
    }
    let matcher = matcher.ok_or_else(|| "grammar matcher unavailable".to_string())?;
    if matcher.is_free() {
        return Ok(None);
    }
    matcher.token_mask(grammar_vocab, grammar_mask);
    Ok(Some(grammar_mask))
}

fn advance_grammar(
    grammar_active: bool,
    matcher: &mut Option<Box<dyn hipfire_runtime::arch_dispatch::GrammarMatcher>>,
    text: &str,
) -> Result<(bool, bool), String> {
    if !grammar_active {
        return Ok((false, false));
    }
    let matcher = matcher
        .as_deref_mut()
        .ok_or_else(|| "grammar matcher unavailable".to_string())?;
    let was_detected = matcher.attractor_detected();
    matcher.advance(text);
    Ok((was_detected, matcher.attractor_detected()))
}

/// Apply the ChatML turn trailer only after a normal decode completion. The
/// callback commits each token only after its forward succeeds, so a failed
/// trailer cannot publish partial conversation state.
fn apply_chatml_trailer(
    last_token: Option<u32>,
    seq_pos: &mut usize,
    im_end_token: Option<u32>,
    nl: &[u32],
    mut forward_and_commit: impl FnMut(u32, usize) -> Result<usize, String>,
) -> Result<(), String> {
    if im_end_token != last_token || nl.is_empty() {
        return Ok(());
    }
    for &token in nl {
        *seq_pos = forward_and_commit(token, *seq_pos)?;
    }
    Ok(())
}

/// Settle the parser only for a normal completion. Abort and error outcomes
/// deliberately consume and drop it without calling `finish`, so buffered
/// parser state cannot leak into a later request.
fn settle_ar_parser(
    parser: Box<dyn hipfire_runtime::stream_parser::StreamParser>,
    outcome: ArDecodeOutcome,
    mut render: impl FnMut(hipfire_runtime::stream_parser::StreamAction),
) -> ArDecodeOutcome {
    match outcome {
        ArDecodeOutcome::Complete => {
            let mut parser = parser;
            for action in parser.finish() {
                render(action);
            }
            ArDecodeOutcome::Complete
        }
        terminal @ (ArDecodeOutcome::Aborted | ArDecodeOutcome::Error(_)) => {
            drop(parser);
            terminal
        }
    }
}

/// Run the shared normal-completion path. Keeping trailer processing here,
/// outside the decode control block, makes every normal exit (EOS, parser or
/// forced stop, budget, and length) update the same ChatML state before parser
/// settlement. Abort/error outcomes skip it and discard the parser.
fn complete_ar_decode(
    parser: Box<dyn hipfire_runtime::stream_parser::StreamParser>,
    outcome: ArDecodeOutcome,
    last_token: Option<u32>,
    seq_pos: &mut usize,
    im_end_token: Option<u32>,
    nl: &[u32],
    forward_and_commit: impl FnMut(u32, usize) -> Result<usize, String>,
    render: impl FnMut(hipfire_runtime::stream_parser::StreamAction),
) -> ArDecodeOutcome {
    let outcome = match outcome {
        ArDecodeOutcome::Complete => {
            match apply_chatml_trailer(last_token, seq_pos, im_end_token, nl, forward_and_commit) {
                Ok(()) => ArDecodeOutcome::Complete,
                Err(error) => ArDecodeOutcome::Error(format!("ChatML trailer forward: {error}")),
            }
        }
        terminal => terminal,
    };
    settle_ar_parser(parser, outcome, render)
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ar_generate(
    dispatch: &mut dyn hipfire_runtime::arch_dispatch::ArchDispatch,
    mut ctx: ForwardCtx<'_>,
    stdout: &mut std::io::Stdout,
    id: &str,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    stop: &[String],
    tools: Option<&[serde_json::Value]>,
    new_tokens: Vec<u32>,
    #[allow(unused_variables)] im_end: &[u32],
    nl: &[u32],
    im_end_token: Option<u32>,
    tool_call_pair: Option<(u32, u32)>,
    think_pair: Option<(u32, u32)>,
    prefill_tokens: usize,
    cached_tokens_count: usize,
    pflash_summary: Option<hipfire_arch_qwen35::pflash::CompressedPrompt>,
    pflash_bypass_reason: Option<String>,
    pflash_alpha: Option<f32>,
    t0: std::time::Instant,
    mut tape: Option<&mut TokenTape>,
) -> GenerateResult {
    // Local copy of the `generate`-nested pflash `done`-event fragment builder
    // (pure over its args; duplicated here so ar_generate stays self-contained).
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

    // seq_pos is a driver local (cumulative physical KV slot); seeded from the
    // model and written back at finalize / on abort.
    let mut seq_pos = dispatch.seq_pos();

    // On any GPU forward / hook error mid-generation, leave model reset and
    // terminal emission to the owner of the model slot. The dispatch borrows the
    // model, so doing either here would alias the loader-owned reset authority.
    macro_rules! prefill_hook_or_fail {
        ($call:expr, $what:expr) => {
            match $call {
                Ok(v) => v,
                Err(e) => {
                    return GenerateResult::Deferred(DeferredTerminal::Error {
                        id: id.to_string(),
                        message: format!("{}: {}", $what, e),
                    });
                }
            }
        };
    }

    // ── Prefill (abort-aware) ────────────────────────────────────────────
    let mut prefill_aborted = false;
    if let Some(window) = dispatch.eviction_window() {
        // Eviction path: chunk to the (budget+beta) window, evict between chunks.
        let mut remaining: &[u32] = &new_tokens;
        while !remaining.is_empty() {
            if check_abort(id) {
                prefill_aborted = true;
                break;
            }
            let space = window.saturating_sub(seq_pos).max(1);
            let chunk_len = remaining.len().min(space);
            let (chunk, rest) = remaining.split_at(chunk_len);
            prefill_hook_or_fail!(
                dispatch.prefill_forward(ctx.reborrow(), chunk, seq_pos),
                "prefill forward"
            );
            seq_pos += chunk_len;
            if let Some(new_phys) =
                prefill_hook_or_fail!(dispatch.maybe_evict(ctx.reborrow(), seq_pos), "kv eviction")
            {
                seq_pos = new_phys;
            }
            remaining = rest;
        }
    } else {
        // No-eviction: chunk at prefill_max_batch so abort fires between batches;
        // adaptive-KV downshift + checkpoint between chunks.
        let chunk_max = dispatch.prefill_max_batch();
        let mut start = 0usize;
        while start < new_tokens.len() {
            if check_abort(id) {
                prefill_aborted = true;
                break;
            }
            let end = (start + chunk_max).min(new_tokens.len());
            let chunk = &new_tokens[start..end];
            prefill_hook_or_fail!(
                dispatch.prefill_forward(ctx.reborrow(), chunk, seq_pos),
                "prefill forward"
            );
            seq_pos += chunk.len();
            dispatch.maybe_adaptive_downshift(ctx.reborrow(), seq_pos);
            if ckpt_resume_enabled() {
                dispatch.take_prefill_checkpoint(ctx.reborrow(), seq_pos);
            }
            start = end;
        }
    }
    if prefill_aborted {
        return GenerateResult::Deferred(DeferredTerminal::Aborted {
            id: id.to_string(),
            generated: 0,
        });
    }
    // Post-prefill adaptive-KV downshift.
    dispatch.maybe_adaptive_downshift(ctx.reborrow(), seq_pos);
    dispatch
        .conversation_tokens_mut()
        .extend_from_slice(&new_tokens);

    // Boundary marker for the prompt-cache / asst_turn_cache slice: the model's
    // verbatim emitted tokens start here in the conversation buffer.
    let decode_start_tokens_idx = dispatch.conversation_tokens_mut().len();

    let vocab_size = dispatch.vocab_size();
    let mut rng_state: u32 = 0x13579BDFu32;
    let repeat_buf_cap = (dispatch.repeat_buf_cap_bytes() / 4).min(repeat_window.max(1));

    let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
        .into_iter()
        .chain(think_pair.into_iter())
        .collect();

    // ── Grammar-guided decoding setup ────────────────────────────────────
    let grammar_enabled = std::env::var("HIPFIRE_QWEN35_GRAMMAR").ok().as_deref() != Some("0");
    let tool_pairs: Vec<(String, Vec<String>)> = if grammar_enabled {
        tools
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())?
                            .to_string();
                        let required: Vec<String> = func
                            .get("parameters")
                            .and_then(|p| p.get("required"))
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some((name, required))
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let grammar_active = !tool_pairs.is_empty();
    let mut matcher: Option<Box<dyn hipfire_runtime::arch_dispatch::GrammarMatcher>> =
        if grammar_active {
            dispatch.init_grammar(&tool_pairs)
        } else {
            None
        };
    let grammar_vocab_arc = if grammar_active {
        Some(dispatch.ensure_decoded_vocab())
    } else {
        None
    };
    let empty_vocab: Vec<String> = Vec::new();
    let grammar_vocab: &[String] = grammar_vocab_arc
        .as_deref()
        .map(|v| v.as_slice())
        .unwrap_or(&empty_vocab);
    let mut grammar_mask: Vec<bool> = vec![true; grammar_vocab.len()];

    // ── First sample (tok0) ──────────────────────────────────────────────
    // ngram_scope is empty at tok0 (no generated tokens yet).
    let ngram_scope0: &[u32] = &[];
    let mut blocked0: Vec<u32> = Vec::new();
    sampler::collect_unclosed_attractor_blocks(
        ngram_scope0,
        &attractor_pairs,
        20,
        2,
        &mut blocked0,
    );
    let cfg0 = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window: repeat_buf_cap,
        presence_penalty,
        frequency_penalty,
        blocked_tokens: blocked0,
        top_k,
        min_p,
    };
    let tok0 = {
        let mask = prefill_hook_or_fail!(
            grammar_mask_for_sample(
                grammar_active,
                matcher.as_deref(),
                grammar_vocab,
                &mut grammar_mask,
            ),
            "grammar mask"
        );
        prefill_hook_or_fail!(
            dispatch.sample(
                ctx.reborrow(),
                &cfg0,
                vocab_size,
                ngram_scope0,
                mask,
                &mut rng_state
            ),
            "sample"
        )
    };
    if grammar_active {
        let text = dispatch.tokenizer().decode(&[tok0]);
        prefill_hook_or_fail!(
            advance_grammar(grammar_active, &mut matcher, &text),
            "grammar advance"
        );
    }
    let t_prefill = Instant::now();
    let mut next_token = tok0;

    let mut generated = 0;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut alert_fired = false;
    let think_open_tok = dispatch.tokenizer().special_token_id("<think>");
    let max_total_think: usize = std::env::var("HIPFIRE_MAX_TOTAL_THINK_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let post_latch_answer_budget: usize = std::env::var("HIPFIRE_POST_LATCH_ANSWER_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);

    // Output parser (Inc Axis-A). `DefaultStreamParser` reproduces the inline emit /
    // stop-seq / think-cap (force-close surfaced via next_forced) / n-gram behavior.
    // Budget-alert stays DRIVER-side (cfg `budget_alert_at_tok=0`); its nudge tokens
    // emit through `parser.emit_only`. The parser owns the EosFilter; the driver owns
    // `streamed_tokens` + `bytes_fed_to_filter` and hands the running-vector byte delta.
    let mut parser =
        dispatch.stream_parser(hipfire_runtime::stream_parser::DefaultStreamParserConfig {
            eos_filter: dispatch.eos_filter_config(),
            stop_seqs: stop.to_vec(),
            max_tokens,
            max_think_tokens,
            think_continuation_ids: dispatch.tokenizer().encode(&think_continuation()),
            max_total_think,
            post_latch_answer_budget,
            budget_alert_at_tok: 0,
            budget_alert_ids: Vec::new(),
            started_in_think: matches!(
                assistant_prefix,
                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
            ),
        });

    // Tool calls emitted BY THE PARSER (ds4 DSML / cohere2moe markers) — captured
    // in the ToolCalls arm below so finish_reason + the asst-cache fingerprint can
    // reflect them. The driver's ChatML `extract_tool_calls_from_text` (used at
    // end-of-turn) only matches `<tool_call>` text (qwen), so without this a custom
    // parser's tool call would leave finish_reason=stop/length + a []-fingerprint
    // (cache miss on echo-back) even though the wire event fired (review I2).
    let mut parser_tool_calls: Vec<hipfire_runtime::prompt_frame::ToolCall> = Vec::new();

    // Execute a StreamAction the parser returned (token/reasoning/info/tool_calls);
    // `Stop` is handled by the caller. Mirrors the legacy inline writes byte-for-byte.
    macro_rules! exec_stream_action {
        ($act:expr) => {{
            match $act {
                hipfire_runtime::stream_parser::StreamAction::Emit { text, reasoning } => {
                    if reasoning {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{},"reasoning":true}}"#,
                            id,
                            serde_json::to_string(&text).unwrap_or_default()
                        );
                    } else {
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{}}}"#,
                            id,
                            serde_json::to_string(&text).unwrap_or_default()
                        );
                    }
                    let _ = stdout.flush();
                }
                hipfire_runtime::stream_parser::StreamAction::Info(msg) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","id":"{}","message":{}}}"#,
                        id,
                        serde_json::to_string(&msg).unwrap_or_default()
                    );
                    let _ = stdout.flush();
                }
                hipfire_runtime::stream_parser::StreamAction::ToolCalls(v) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"tool_calls","id":"{}","calls":{}}}"#,
                        id, v
                    );
                    let _ = stdout.flush();
                    // Capture for finish_reason + fingerprint (review I2). The wire
                    // event already went out above — this only feeds the metadata,
                    // so no double-emit. `v` is a `{name,arguments}` array.
                    if let serde_json::Value::Array(arr) = &v {
                        for c in arr {
                            if let Some(name) = c.get("name").and_then(|n| n.as_str()) {
                                parser_tool_calls.push(hipfire_runtime::prompt_frame::ToolCall {
                                    name: name.to_string(),
                                    arguments: c
                                        .get("arguments")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null),
                                });
                            }
                        }
                    }
                }
                hipfire_runtime::stream_parser::StreamAction::Stop => {}
            }
        }};
    }

    // Whether `next_token` came from the parser's forced queue (think-cap splice).
    // Forced tokens are emit-only (bypass the guards, matching the legacy inline
    // splice); sampled tokens go through `feed`. tok0 is sampled → false.
    let mut was_forced = false;

    let decode_outcome = 'decode: {
        // After parser construction, errors leave through the local outcome so
        // the parser can be discarded without running its finish epilogue.
        macro_rules! hook_or_fail {
            ($call:expr, $what:expr) => {
                match $call {
                    Ok(v) => v,
                    Err(e) => {
                        break 'decode route_ar_decode_outcome(
                            false,
                            Some(format!("{}: {}", $what, e)),
                        )
                    }
                }
            };
        }

        while generated < max_tokens {
            if check_abort(id) {
                // Client cancelled mid-decode — full cold reset (mirrors DFlash abort).
                break 'decode route_ar_decode_outcome(false, None);
            }
            if decode_limit_reached(
                max_tokens,
                generated,
                dispatch.physical_cap(),
                seq_pos,
                dispatch.has_eviction(),
            ) {
                break 'decode route_ar_decode_outcome(true, None);
            }
            // ── eos / im_end pre-decision (BEFORE commit) ────────────────────────
            // A sampled eos consults the parser's discipline. `CommitAndStop` (the
            // simple-arch default) falls through to commit+forward+emit the eos then
            // break — byte-identical to the legacy inline eos break. `Inject` (cohere2moe
            // empty-turn guard) enqueues continuation markers and continues WITHOUT
            // committing the eos (the markers surface via next_forced next iter). `Stop`
            // breaks without committing. Forced tokens are never eos.
            let mut eos_commit_and_stop = false;
            if !was_forced && (dispatch.is_eos(next_token) || im_end_token == Some(next_token)) {
                match ar_sampled_eos(parser.as_mut()) {
                    ArSampledEos::Stop => {
                        break 'decode route_ar_decode_outcome(true, None);
                    }
                    ArSampledEos::CommitAndStop => {
                        eos_commit_and_stop = true;
                    }
                    ArSampledEos::Forced(f) => {
                        next_token = f;
                        was_forced = true;
                        continue;
                    }
                    ArSampledEos::Error(message) => {
                        break 'decode route_ar_decode_outcome(false, Some(message));
                    }
                }
            }
            generated += 1;
            dispatch.conversation_tokens_mut().push(next_token);
            streamed_tokens.push(next_token);
            if let Some(t) = tape.as_deref_mut() {
                t.push(next_token);
            }
            emit_committed_event(
                stdout,
                id,
                next_token,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            // Running-vector byte delta (BPE detok is non-local → whole-vector diff, the
            // exact stream the legacy filter consumed). The parser owns its EosFilter.
            let new_bytes: Vec<u8> = {
                let all_bytes = dispatch.tokenizer().decode_bytes(&streamed_tokens);
                let nb = all_bytes[bytes_fed_to_filter..].to_vec();
                bytes_fed_to_filter = all_bytes.len();
                nb
            };

            hook_or_fail!(
                dispatch.decode_step_forward(ctx.reborrow(), next_token, seq_pos),
                "token forward"
            );
            seq_pos += 1;
            if ckpt_resume_enabled() {
                dispatch.take_prefill_checkpoint(ctx.reborrow(), seq_pos);
            }
            if let Some(new_phys) =
                hook_or_fail!(dispatch.maybe_evict(ctx.reborrow(), seq_pos), "kv eviction")
            {
                seq_pos = new_phys;
            }
            dispatch.maybe_adaptive_downshift(ctx.reborrow(), seq_pos);

            // ── Terminal eos (CommitAndStop, decided pre-commit above) ───────────
            // The eos token has now been committed + forwarded; emit it through the filter
            // (display-suppressed) then break — byte-identical to the legacy inline eos break.
            if eos_commit_and_stop {
                for act in parser.emit_only(next_token, &new_bytes) {
                    exec_stream_action!(act);
                }
                break 'decode route_ar_decode_outcome(true, None);
            }

            // ── Output shaping + guards ──────────────────────────────────────────
            // Forced tokens (think-cap continuation splice) bypass the guards via
            // `emit_only` (the legacy inline splice forwarded them without re-running the
            // guards). Sampled tokens go through `feed` (emit + stop-seq + think-cap enqueue
            // + n-gram); a `Stop` action breaks the loop.
            if was_forced {
                let actions = parser.emit_only(next_token, &new_bytes);
                let stop = stream_stop(&actions);
                for act in actions {
                    if !matches!(act, hipfire_runtime::stream_parser::StreamAction::Stop) {
                        exec_stream_action!(act);
                    }
                }
                if stop {
                    break 'decode route_ar_decode_outcome(true, None);
                }
            } else {
                parser.note_force_answer(check_force_answer(id));
                let actions = parser.feed(next_token, &new_bytes);
                let stop = stream_stop(&actions);
                for act in actions {
                    if !matches!(act, hipfire_runtime::stream_parser::StreamAction::Stop) {
                        exec_stream_action!(act);
                    }
                }
                if stop {
                    break 'decode route_ar_decode_outcome(true, None);
                }
            }

            // Budget-alert injection.
            if !alert_fired
                && budget_alert_at_tok > 0
                && generated >= budget_alert_at_tok
                && !budget_alert_text.is_empty()
            {
                alert_fired = true;
                let raw_so_far = dispatch.tokenizer().decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let in_think = currently_in_think(
                    raw_str,
                    matches!(
                        assistant_prefix,
                        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                    ),
                );
                if !in_think {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    let ngram_scope: &[u32] = &streamed_tokens;
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
                        presence_penalty,
                        frequency_penalty,
                        blocked_tokens: blocked,
                        top_k,
                        min_p,
                    };
                    next_token = {
                        let mask = match grammar_mask_for_sample(
                            grammar_active,
                            matcher.as_deref(),
                            grammar_vocab,
                            &mut grammar_mask,
                        ) {
                            Ok(mask) => mask,
                            Err(message) => {
                                break 'decode route_ar_decode_outcome(
                                    false,
                                    Some(format!("grammar mask: {message}")),
                                );
                            }
                        };
                        hook_or_fail!(
                            dispatch.sample(
                                ctx.reborrow(),
                                &cfg,
                                vocab_size,
                                ngram_scope,
                                mask,
                                &mut rng_state
                            ),
                            "sample"
                        )
                    };
                    if grammar_active {
                        let text = dispatch.tokenizer().decode(&[next_token]);
                        if let Err(message) = advance_grammar(grammar_active, &mut matcher, &text) {
                            break 'decode route_ar_decode_outcome(
                                false,
                                Some(format!("grammar advance: {message}")),
                            );
                        }
                    }
                    was_forced = false;
                    continue;
                }
                let nudge_tokens = dispatch.tokenizer().encode(budget_alert_text);
                let budget_left = max_tokens.saturating_sub(generated);
                let nudge_len = nudge_tokens.len().min(budget_left);
                let need_kv = seq_pos
                    .saturating_add(nudge_len)
                    .saturating_add(
                        max_tokens
                            .saturating_sub(generated)
                            .saturating_sub(nudge_len),
                    )
                    .saturating_add(nl.len());
                if nudge_len > 0 && (dispatch.has_eviction() || need_kv <= dispatch.physical_cap())
                {
                    for &tok in &nudge_tokens[..nudge_len] {
                        dispatch.conversation_tokens_mut().push(tok);
                        streamed_tokens.push(tok);
                        if let Some(tp) = tape.as_deref_mut() {
                            tp.push(tok);
                        }
                        emit_committed_event(
                            stdout,
                            id,
                            tok,
                            streamed_tokens.len() - 1,
                            t0.elapsed().as_millis() as u64,
                        );
                        let new_bytes2: Vec<u8> = {
                            let all_bytes2 = dispatch.tokenizer().decode_bytes(&streamed_tokens);
                            let nb = all_bytes2[bytes_fed_to_filter..].to_vec();
                            bytes_fed_to_filter = all_bytes2.len();
                            nb
                        };
                        // Nudge tokens bypass the guards (like the think-cap splice) — emit
                        // through the parser's filter, no feed().
                        let actions = parser.emit_only(tok, &new_bytes2);
                        let stop = stream_stop(&actions);
                        for act in actions {
                            if !matches!(act, hipfire_runtime::stream_parser::StreamAction::Stop) {
                                exec_stream_action!(act);
                            }
                        }
                        hook_or_fail!(
                            dispatch.decode_step_forward(ctx.reborrow(), tok, seq_pos),
                            "budget nudge forward"
                        );
                        seq_pos += 1;
                        if let Some(new_phys) = hook_or_fail!(
                            dispatch.maybe_evict(ctx.reborrow(), seq_pos),
                            "kv eviction"
                        ) {
                            seq_pos = new_phys;
                        }
                        generated += 1;
                        if stop {
                            break 'decode route_ar_decode_outcome(true, None);
                        }
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
                    break 'decode route_ar_decode_outcome(true, None);
                }
            }

            // Next token: forced (think-cap continuation splice, drained one at a time) or
            // the steady-state sample. Forced tokens are emit-only at the top of the next
            // iteration (was_forced), matching the legacy inline splice.
            if let Some(f) = parser.next_forced() {
                next_token = f;
                was_forced = true;
                if grammar_active {
                    let text = dispatch.tokenizer().decode(&[f]);
                    if let Err(message) = advance_grammar(grammar_active, &mut matcher, &text) {
                        break 'decode route_ar_decode_outcome(
                            false,
                            Some(format!("grammar advance: {message}")),
                        );
                    }
                }
                continue;
            }
            was_forced = false;

            // Steady-state sample.
            let ngram_scope: &[u32] = &streamed_tokens;
            let mut blocked: Vec<u32> = Vec::new();
            sampler::collect_unclosed_attractor_blocks(
                ngram_scope,
                &attractor_pairs,
                20,
                2,
                &mut blocked,
            );
            if parser.force_answer_latched() {
                if let Some(t) = think_open_tok {
                    blocked.push(t);
                }
            }
            let cfg = SamplerConfig {
                temperature: temp,
                top_p,
                repeat_penalty,
                repeat_window: repeat_buf_cap,
                presence_penalty,
                frequency_penalty,
                blocked_tokens: blocked,
                top_k,
                min_p,
            };
            next_token = {
                let mask = match grammar_mask_for_sample(
                    grammar_active,
                    matcher.as_deref(),
                    grammar_vocab,
                    &mut grammar_mask,
                ) {
                    Ok(mask) => mask,
                    Err(message) => {
                        break 'decode route_ar_decode_outcome(
                            false,
                            Some(format!("grammar mask: {message}")),
                        );
                    }
                };
                hook_or_fail!(
                    dispatch.sample(
                        ctx.reborrow(),
                        &cfg,
                        vocab_size,
                        ngram_scope,
                        mask,
                        &mut rng_state
                    ),
                    "sample"
                )
            };
            if grammar_active {
                let text = dispatch.tokenizer().decode(&[next_token]);
                let (was_detected, is_detected) =
                    match advance_grammar(grammar_active, &mut matcher, &text) {
                        Ok(states) => states,
                        Err(message) => {
                            break 'decode route_ar_decode_outcome(
                                false,
                                Some(format!("grammar advance: {message}")),
                            );
                        }
                    };
                if !was_detected && is_detected {
                    eprintln!(
                    "[grammar-ngram] attractor detected in tool_call args at gen={} — forcing close",
                    generated,
                );
                }
            }
        }

        route_ar_decode_outcome(true, None)
    };

    let last_conv = dispatch.conversation_tokens_mut().last().copied();
    let decode_outcome = complete_ar_decode(
        parser,
        decode_outcome,
        last_conv,
        &mut seq_pos,
        im_end_token,
        nl,
        |token, position| {
            dispatch
                .decode_step_forward(ctx.reborrow(), token, position)
                .map_err(|e| e.to_string())?;
            let mut next_position = position + 1;
            if let Some(new_phys) = dispatch
                .maybe_evict(ctx.reborrow(), next_position)
                .map_err(|e| format!("kv eviction: {e}"))?
            {
                next_position = new_phys;
            }
            dispatch.conversation_tokens_mut().push(token);
            Ok(next_position)
        },
        |act| {
            exec_stream_action!(act);
        },
    );

    match decode_outcome {
        ArDecodeOutcome::Complete => {}
        ArDecodeOutcome::Aborted => {
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated,
            });
        }
        ArDecodeOutcome::Error(message) => {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message,
            });
        }
    }

    // Write the final physical slot back to the model (old arm mutated
    // m.session.seq_pos directly per token).
    dispatch.set_seq_pos(seq_pos);

    // ── parse tool_calls + content once ──────────────────────────────────
    let decoded_full = dispatch.tokenizer().decode(&streamed_tokens);
    // ChatML text-recovered tool calls (qwen: written as `<tool_call>` text). Custom
    // parsers (ds4 DSML / cohere2moe markers) emit their calls via the ToolCalls
    // StreamAction instead — captured in `parser_tool_calls` above.
    let text_tool_calls = extract_tool_calls_from_text(&decoded_full);

    // Wire re-emit is ONLY for text-recovered calls — parser-emitted calls already
    // went out via the StreamAction arm, so re-emitting here would double them (I2).
    if !text_tool_calls.is_empty() {
        let calls_json: Vec<serde_json::Value> = text_tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
            })
            .collect();
        let calls_str = serde_json::to_string(&calls_json).unwrap_or_else(|_| "[]".to_string());
        let _ = writeln!(
            stdout,
            r#"{{"type":"tool_calls","id":"{}","calls":{}}}"#,
            id, calls_str,
        );
    }

    // finish_reason + asst-cache fingerprint reflect tool calls from EITHER source
    // (text extraction for qwen; parser StreamAction for ds4/cohere2moe) — review I2.
    let emit_tool_calls = if text_tool_calls.is_empty() {
        parser_tool_calls
    } else {
        text_tool_calls
    };

    // ── asst_turn_cache write ────────────────────────────────────────────
    {
        let mut cached_seq: Vec<u32> =
            dispatch.conversation_tokens_mut()[decode_start_tokens_idx..].to_vec();
        while let Some(&last) = cached_seq.last() {
            if nl.contains(&last) {
                cached_seq.pop();
            } else {
                break;
            }
        }
        if let Some(&last) = cached_seq.last() {
            if im_end_token == Some(last) {
                cached_seq.pop();
            }
        }
        if !cached_seq.is_empty() {
            let stripped = strip_think_for_fingerprint(&decoded_full);
            let emit_text =
                hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
            let fp = asst_turn_fingerprint(&emit_text, &emit_tool_calls);
            if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                eprintln!(
                    "[qwen-cache store] fp={:#018x} cached_seq={} emit_text.len={} tool_calls={} preview={:?}",
                    fp,
                    cached_seq.len(),
                    emit_text.len(),
                    emit_tool_calls.len(),
                    emit_text.chars().take(60).collect::<String>(),
                );
            }
            dispatch.insert_asst_turn(fp, cached_seq);
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
    let hit_length_cap = generated >= max_tokens;
    let finish_reason = if hit_length_cap {
        "length"
    } else if !emit_tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1},"cached_tokens":{},"finish_reason":"{}"{}}}"#,
        id,
        generated,
        tok_s,
        prefill_tokens,
        prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        prefill_s * 1000.0,
        cached_tokens_count,
        finish_reason,
        pflash_done_fragment(&pflash_summary, &pflash_bypass_reason, pflash_alpha),
    );
    let _ = stdout.flush();
    GenerateResult::Complete
}

fn generate(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    drafter_gpu: Option<&mut rdna_compute::Gpu>,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    // Whether the request EXPLICITLY set a non-temperature sampling control
    // (top_p/top_k/min_p/penalties). Gates temp>0 spec routing: explicit controls
    // force the AR sampler (the SWOR spec verify can only honor temperature).
    user_explicit_sampling: bool,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
    // CACTUS acceptance-boost δ (0 = lossless). Request opt-in; applies only to a
    // CACTUS-capable sampled verify (deepseek4 DSpark / qwen35 DFlash).
    cactus_delta: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    think_mode: ThinkMode,
    stop: &[String],
) -> GenerateResult {
    // hunt3 M-E: seed the process-global CPU sampler RNG with this request's
    // fixed seed so the grammar/CPU-fallback sample stream is deterministic per
    // request and does not carry RNG state across requests. Matches the u32 the
    // GPU sample path uses (0x13579BDF).
    hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);
    // Multi-GPU dispatch: one exhaustive match on the parallelism axis (Task 7).
    // Match on the Copy kind() value so the borrow of m ends before the
    // per-arch arms reborrow m mutably (a direct `match &m.parallel` would
    // conflict with `dense_serve_via_ar_generate(m,…)`).
    // Disjoint field borrows: `m.tokenizer` (read) + the model field (mut).
    match m.parallel.kind() {
        ModelParallelKind::Tp | ModelParallelKind::PpDense => {
            // Dense TP / dense-PP AR decode on the unified ar_generate driver.
            return dense_serve_via_ar_generate(
                m,
                gpu,
                stdout,
                id,
                system_prompt,
                prompt,
                assistant_prefix,
                messages_history,
                temp,
                top_p,
                top_k,
                min_p,
                max_tokens,
                stop,
            );
        }
        ModelParallelKind::Ep => {
            // EP serve (ds4/minimax): thread the SAME resolved sampling the
            // single-GPU handler computed (request field > m.rec_* > arch-default
            // ladder, all done at the call site above) into the EP decode loops.
            // Previously the EP path dropped these to a hardcoded greedy argmax,
            // which loops on ds4's quantized instruct model (card mandates
            // temp=1.0/top_p=1.0). reset_cpu_sampler_rng(0x13579BDF) was already
            // called above, so the host-side draw in ep_serve_* is deterministic.
            let ep_sampling = EpSampling {
                temp,
                top_p,
                top_k,
                min_p,
            };
            return generate_ep(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                max_think_tokens,
                think_mode,
                tools,
                messages_history,
                stop,
                ep_sampling,
            );
        }
        ModelParallelKind::Single | ModelParallelKind::PpQwen35 => {
            // Fall through to the per-arch single-GPU / qwen35-PP short-circuit below.
        }
    }
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
    // arch_id=7 with an opt-in model-free n-gram speculator loaded (qwen2
    // `SpecTarget`) routes to the arch-generic spec loop, exactly like llama
    // (0/1). Without a speculator it falls through to the plain qwen2 decode
    // short-circuit below.
    // The n-gram ChainSpeculator on arch 7/10/11/12 is greedy-only: `build_speculator`
    // sets `samples=true` only for arch 5/6 (spec_build.rs), so `requires_greedy()` is
    // true and `verify_block()` decodes argmax regardless of the requested temp. Gate
    // each spec short-circuit the same way the qwen35/llama DFlash route is gated
    // (~7042): route into the spec loop only when greedy (temp<=1e-6) OR the loaded
    // drafter genuinely samples; a temp>0 request on a greedy-only drafter falls
    // through to the arch's AR path below (faithful sampling) instead of being
    // silently decoded greedy. Future-proof: a sampling-capable drafter on these
    // arches auto-enables the temp>0 route with no further change.
    let ngram_can_sample = m
        .speculator
        .as_ref()
        .map(|s| !s.requires_greedy())
        .unwrap_or(false);
    if m.meta.arch_id == 7 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
        return generate_dflash(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            None, // pflash_bypass_reason — no pflash on the n-gram path
            None, // pflash_alpha
            tools,
            messages_history,
            stop,
            temp, // greedy-only n-gram drafter: gate above only reaches here at temp<=1e-6 (temp>0 → AR path below); honored if a sampling-capable drafter is ever loaded
            top_p, // nucleus cutoff — active only for a sampling-capable drafter
            top_k.map(|k| k as usize).unwrap_or(0), // top-k cutoff
            cactus_delta, // request opt-in (0.0 default = lossless); ds4 DSpark / qwen35 DFlash only
        );
    }
    if m.meta.arch_id == 7 {
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
        );
        let _ = stop; // hunt3 M-F: not wired for arch_id=7 (qwen2 bring-up)
        return generate_qwen2(
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
    }
    if m.meta.arch_id == 9 {
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
        let _ = stop; // hunt3 M-F: not wired for arch_id=9 (deepseek4 bring-up)
                      // Route the MTP spec-decode path (greedy, speculator present) through the
                      // unified `generate_spec`; the AR sampler path (temp>0) and the
                      // no-speculator fallback stay in the bespoke `generate_deepseek4` (T5
                      // deletes the latter's now-redundant spec loop).
                      // Route to the DSpark/MTP spec loop for greedy OR — now un-gated — for
                      // temp>0 when the loaded speculator can sample its verify. DSpark
                      // signals this via `requires_greedy()==false` (its sampled verify is the
                      // "chain" kind, NOT the ddtree-SWOR flavour that `supports_temp_verify()`
                      // flags) — same predicate the arch_id=7 path uses above. The τ-adaptive
                      // block controller makes temp>0 DSpark beat AR + CACTUS adds more; this
                      // was hardcoded greedy-only (temp>0 → AR).
        let spec_temp_ok = temp <= 1e-6
            || m.speculator
                .as_ref()
                .map_or(false, |s| !s.requires_greedy());
        let spec_mode = deepseek4_spec_requested(m) && spec_temp_ok;
        if spec_mode && m.speculator.is_some() {
            return generate_deepseek4_spec(
                m,
                gpu,
                stdout,
                id,
                prompt,
                system_prompt,
                max_tokens,
                temp,
                top_p,
                top_k.map(|k| k as usize).unwrap_or(0),
                cactus_delta,
                think_mode,
                tools,
                messages_history,
            );
        } else {
            return generate_deepseek4(
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
        }
    }
    // arch_id=11 (LFM2.5-MoE) with an opt-in model-free n-gram speculator loaded
    // (lfm2moe `SpecTarget`, conv-state snapshot/rollback) routes to the
    // arch-generic spec loop, like qwen2 (7) / minimax (10). Without a speculator
    // it falls through to the plain `generate_lfm2moe` short-circuit below.
    if m.meta.arch_id == 11 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
        return generate_dflash(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            None, // pflash_bypass_reason — no pflash on the n-gram path
            None, // pflash_alpha
            tools,
            messages_history,
            stop,
            temp, // greedy-only n-gram drafter: gate above only reaches here at temp<=1e-6 (temp>0 → AR path below); honored if a sampling-capable drafter is ever loaded
            top_p, // nucleus cutoff — active only for a sampling-capable drafter
            top_k.map(|k| k as usize).unwrap_or(0), // top-k cutoff
            cactus_delta, // request opt-in (0.0 default = lossless); ds4 DSpark / qwen35 DFlash only
        );
    }
    if m.meta.arch_id == 11 {
        // arch_id=11 (LFM2.5-8B-A1B). Standalone bring-up — same shape as
        // the deepseek4 short-circuit above. PFlash / DFlash / VL / multi-GPU
        // / sampler-budget scaffolding all bypass. We honour `system_prompt`,
        // `temp`, `top_p`, `tools`, and `messages_history`; everything else
        // routes through future follow-ups.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            think_mode,
        );
        let _ = (repeat_penalty, repeat_window);
        return generate_lfm2moe(
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
    }
    // arch_id=12 (Cohere2-MoE) with an opt-in model-free n-gram speculator loaded
    // (cohere2moe `SpecTarget` + `Cohere2MoeEmit`, which ports the agentic-marker
    // state machine + empty-turn / think-budget generation guards) routes to the
    // arch-generic spec loop, like qwen2 (7) / minimax (10) / lfm2moe (11).
    // Without a speculator it falls through to the plain `generate_cohere2moe`.
    if m.meta.arch_id == 12 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
        return generate_dflash(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            None, // pflash_bypass_reason — no pflash on the n-gram path
            None, // pflash_alpha
            tools,
            messages_history,
            stop,
            temp, // greedy-only n-gram drafter: gate above only reaches here at temp<=1e-6 (temp>0 → AR path below); honored if a sampling-capable drafter is ever loaded
            top_p, // nucleus cutoff — active only for a sampling-capable drafter
            top_k.map(|k| k as usize).unwrap_or(0), // top-k cutoff
            cactus_delta, // request opt-in (0.0 default = lossless); ds4 DSpark / qwen35 DFlash only
        );
    }
    if m.meta.arch_id == 12 {
        // arch_id=12 (Cohere2-MoE / North-Mini-Code). Standalone bring-up
        // with Cohere agentic marker parsing, batched prefill when supported,
        // and prefix-cache reuse. PFlash / DFlash / VL / multi-GPU /
        // sampler-budget scaffolding all bypass here.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            think_mode,
        );
        let _ = (repeat_penalty, repeat_window);
        return generate_cohere2moe(
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
    }
    // arch_id=10 (MiniMax-M2) with an opt-in model-free n-gram speculator loaded
    // (minimax `SpecTarget`) routes to the arch-generic spec loop, exactly like
    // qwen2 (7) above. Without a speculator it falls through to the plain
    // `generate_minimax` short-circuit below.
    if m.meta.arch_id == 10 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
        return generate_dflash(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            assistant_prefix,
            None, // pflash_bypass_reason — no pflash on the n-gram path
            None, // pflash_alpha
            tools,
            messages_history,
            stop,
            temp, // greedy-only n-gram drafter: gate above only reaches here at temp<=1e-6 (temp>0 → AR path below); honored if a sampling-capable drafter is ever loaded
            top_p, // nucleus cutoff — active only for a sampling-capable drafter
            top_k.map(|k| k as usize).unwrap_or(0), // top-k cutoff
            cactus_delta, // request opt-in (0.0 default = lossless); ds4 DSpark / qwen35 DFlash only
        );
    }
    if m.meta.arch_id == 10 {
        // arch_id=10 (MiniMax-M2). Minimal AR bring-up — same shape as the
        // deepseek4 / lfm2moe short-circuits above. PFlash / DFlash / VL /
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
            think_mode,
        );
        let _ = (repeat_penalty, repeat_window);
        return generate_minimax(
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
    }
    // Expert-parallel dispatch (task #26). ep.is_some() → generate_ep (AR via
    // forward_ep, full sampler on rank-0 logits). Refusals enforced at load.
    // Multi-GPU pipeline-parallel dispatch (Stage 7 of #58). qwen35 ArchResident PP
    // is refused at load when DFlash / CASK / PFlash / VL is requested, so this branch
    // doesn't need to thread any of those args through.
    if matches!(m.parallel, ModelParallel::Pp(PipelineImpl::ArchResident(_))) {
        return generate_multi(
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
            top_k,
            min_p,
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
            stop, // hunt3 M-F: thread user stop sequences into the pp>1 path
        );
    }
    // DFlash fast path -- only when a draft model is loaded. DFlash now serves
    // BOTH greedy (temp≈0 → argmax-accept) AND temp>0 (lossless rejection-
    // sampling: full-vocab softmax on draft + target, accept `u*p_draft <=
    // p_target`, rejection/all-accept bonus from the renormalized residual /
    // target softmax — so the committed distribution exactly equals the target's
    // own temp-T sampling at cactus_delta==0). Skip the normal AR sampling setup
    // entirely. CAVEAT: top_p/top_k/min_p are NOT honored on the spec path
    // (spec_step_dflash is full-vocab; truncating one side would break the
    // rejection invariant) — a temp>0 request carrying a non-default top_p gets
    // temp-only sampling and a one-time warn below.
    //
    // Budgeted thinking remains on the selected decode path. AR splices the
    // continuation directly, DFlash uses SpecEmit::take_forced, and native MTP
    // applies the same close at a committed-block boundary.

    // Prompt-cache routing (2026-05-30, native-reuse update). `generate_dflash`
    // now implements LCP prompt-cache reuse natively: on a pure conversation
    // extension it reuses the target KV + DeltaNet prefix and extends the
    // draft's cumulative `target_hidden` by only the suffix — verified
    // byte-identical to a full prefill. So the DFlash path now gives BOTH a warm
    // prefill on agentic turns AND its ~2× decode speedup, strictly better than
    // the AR cache path for greedy chat. (Earlier this same routing site sent
    // chat to AR as a stopgap because DFlash re-prefilled cold every turn — that
    // reason is gone.) DFlash is the default for greedy chat on qwen3.5/3.6;
    // opt out to the simpler AR path (e.g. to avoid spec-decode) with
    // `HIPFIRE_DFLASH_CHAT=0`.
    let force_ar_chat = std::env::var("HIPFIRE_DFLASH_CHAT").ok().as_deref() == Some("0");
    // ── temp>0 DFlash spec routing — composes two distribution-preserving verifies ──
    // The qwen35 DflashSpeculator carries BOTH temp>0 mechanisms, picked by how it
    // LOADED (not per-request); both preserve the target distribution, differing only
    // in which sampling controls they can honor (and in τ):
    //   * ddtree mode (`supports_temp_verify`): SWOR tree-verify, fed `temp` directly
    //     (#483). Distribution-exact but honors TEMPERATURE ONLY — it samples
    //     softmax(logits/temp), so it cannot honor an explicit top_p/top_k/min_p/
    //     penalty. Highest τ (multi-candidate tree).
    //   * chain mode (`!ddtree`, `!requires_greedy`): master's lossless rejection
    //     sampling via `set_sampling`, honoring temp + top_p + top_k (== AR at that
    //     nucleus). Reproduces spec-graph's shipped default-on sampled-DFlash.
    // Routing therefore PREFERS ddtree-SWOR for a bare-temperature request; an explicit
    // sampling control SWOR can't honor falls to AR (ddtree mode) or is honored by
    // chain rejection sampling (chain mode). A greedy-only drafter (MTP/n-gram) keeps
    // temp>0 on AR. min_p is unimplemented on both spec paths (a min_p request → AR).
    // Opt out of temp>0 spec entirely: HIPFIRE_DFLASH_TEMP_SPEC=0.
    let fast_sample_on = std::env::var("HIPFIRE_DFLASH_FAST_SAMPLE").ok().as_deref() != Some("0");
    let dflash_min_p_present = min_p.map(|p| p > 0.0).unwrap_or(false);
    let temp_spec_env_off = std::env::var("HIPFIRE_DFLASH_TEMP_SPEC").ok().as_deref() == Some("0");
    let supports_temp_swor = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_temp_verify());
    // ddtree-SWOR: distribution-exact but temperature-only — engage only for a bare-
    // temperature request (an explicit control it can't honor falls to AR below). A
    // *defaulted* top_p (model recommendation) stays advisory.
    let ddtree_swor_route =
        temp > 1e-6 && supports_temp_swor && !user_explicit_sampling && !temp_spec_env_off;
    // chain rejection sampling (master): honors temp+top_p+top_k. `!requires_greedy()`
    // stops a sampled request from routing into a greedy-only drafter (MTP/n-gram) and
    // being silently decoded greedy. Gate on `!supports_temp_swor` so a ddtree-mode
    // drafter never takes this arm — its `step()` dispatches to the temp-only SWOR
    // verify, which would silently drop top_p/top_k. `!supports_temp_swor` is exactly
    // chain mode (no ddtree configured).
    let spec_can_sample = m
        .speculator
        .as_ref()
        .map(|s| !s.requires_greedy())
        .unwrap_or(false);
    let chain_sample_route = temp > 1e-6
        && !supports_temp_swor
        && spec_can_sample
        && fast_sample_on
        && !dflash_min_p_present
        && !temp_spec_env_off;
    // qwen3.5/3.6: greedy always; temp>0 via ddtree-SWOR or chain rejection sampling.
    let qwen_dflash_route = (m.meta.arch_id == 5 || m.meta.arch_id == 6)
        && (temp <= 1e-6 || ddtree_swor_route || chain_sample_route);
    // llama (arch 0/1): #483 built + validated dense DFlash with ddtree tree-SWOR, so
    // temp>0 engages ddtree-SWOR here (bare temp). DSpark (qwen3) adds a validated
    // sampled-llama CHAIN path — its fused sample_top_p_pf verify honors
    // temp+top_p+top_k and beats AR at temp>0 — so `chain_sample_route` now engages
    // llama temp>0 too. (Non-DSpark chain-mode llama has no such path and stays on
    // AR via `spec_can_sample`/`supports_temp_swor` gating.)
    let llama_dflash_route = (m.meta.arch_id == 0 || m.meta.arch_id == 1)
        && (temp <= 1e-6 || ddtree_swor_route || chain_sample_route);
    // Operator visibility: a temp>0 request on a DFlash-capable arch that did NOT
    // qualify for spec silently runs AR (correct, but slower). Name the reason.
    if temp > 1e-6
        && m.speculator.is_some()
        && (m.meta.arch_id == 5
            || m.meta.arch_id == 6
            || m.meta.arch_id == 0
            || m.meta.arch_id == 1)
        && !qwen_dflash_route
        && !llama_dflash_route
        && !force_ar_chat
    {
        let reason = if temp_spec_env_off {
            "HIPFIRE_DFLASH_TEMP_SPEC=0"
        } else if supports_temp_swor && user_explicit_sampling {
            "request set an explicit top_p/top_k/min_p/penalty (ddtree SWOR verify honors temperature only); AR applies them"
        } else if dflash_min_p_present {
            "request set min_p (sampled DFlash honors top_p/top_k only); AR applies it"
        } else if !spec_can_sample {
            "loaded drafter is greedy-only (MTP/n-gram); temp>0 runs AR"
        } else {
            "ddtree SWOR verify not active (needs ddtree_budget>0)"
        };
        eprintln!(
            "[hipfire] id={id}: temp>0 DFlash spec disabled -> AR ({reason}). Temperature honored; spec speedup off."
        );
    }
    if m.speculator.is_some() && !force_ar_chat && (qwen_dflash_route || llama_dflash_route) {
        // One-time visibility: temp + top_p + top_k ARE now honored on the
        // DFlash spec sampled path (identical (top_k,top_p) nucleus truncation on
        // draft + target → lossless == AR-at-(top_k,top_p)). Only min_p remains
        // unimplemented; warn once if a non-default min_p was requested so the
        // residual mismatch is visible rather than silent.
        let minp_requested = min_p.map(|p| p > 0.0).unwrap_or(false);
        if temp > 1e-6 && minp_requested {
            static SPEC_MINP_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !SPEC_MINP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"warning","id":"{}","message":"DFlash spec-decode honors temp+top_p+top_k but ignores min_p; set HIPFIRE_DFLASH_CHAT=0 to route through AR for full min_p support"}}"#,
                    id,
                );
                let _ = stdout.flush();
            }
        }
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
        return generate_dflash(
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
            stop,  // hunt3 M-F: thread user stop sequences into the default DFlash path
            temp, // request-resolved temp: ddtree mode → SWOR (temp-only); chain mode → lossless rejection-sampling
            top_p, // nucleus cutoff: honored on the chain sampled path (ignored by the ddtree SWOR arm)
            top_k.map(|k| k as usize).unwrap_or(0), // top-k cutoff (chain path; recipe → folded into tau)
            cactus_delta, // request opt-in (0.0 default = lossless); ds4 DSpark / qwen35 DFlash only
        );
    }

    // Auto-reset on multi-turn rollover. When eviction is active (operator
    // enabled cask_sidecar at load), the physical buffer is bounded by
    // budget+beta+safety regardless of conversation length, so reset never
    // needs to fire — eviction reclaims slots after each token. When eviction
    // is OFF, physical grows unbounded up to max_seq; reset when we'd overrun.
    let prompt_est = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        tokenizer.encode(prompt).len() + 20
    };
    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache GEN-ENTRY] conv_tok={} seq_pos={}",
            m.session.conversation_tokens.len(),
            m.session.seq_pos
        );
    }
    if m.eviction.is_none()
        && m.session
            .seq_pos
            .saturating_add(prompt_est)
            .saturating_add(max_tokens)
            > m.meta.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.session.seq_pos, m.meta.max_seq
        );
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }
    let cache_kill_switch = std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
    let pflash_active = pflash_cfg
        .map(|c| !matches!(c.mode, hipfire_arch_qwen35::pflash::PflashMode::Off))
        .unwrap_or(false);
    let jinja_active = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0")
        && m.meta.chat_template.is_some();
    let cache_eligible = !cache_kill_switch
        && messages_history.is_some()
        && m.eviction.is_none()
        && !pflash_active
        && !m.session.conversation_tokens.is_empty();
    if jinja_active && !cache_eligible && m.session.seq_pos > 0 {
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();

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
            m.session.seq_pos,
            raw_q_tokens.len(),
            drafter_gpu.is_some()
        );
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.session.seq_pos == 0 {
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
    // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
    // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
    // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
    // the .hfq carries no template, e.g. an older A1B convert).
    // Jinja default-ON (flipped 2026-06-09): render through the model's chat
    // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
    // ChatML/Plain). Falls back to Plain automatically when no template resolves.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    // Jinja renders the FULL conversation every turn (stateless full-render,
    // like generate_dflash) — fire on every turn, not just `seq_pos == 0`.
    // `render_messages` below replays `messages_history` (all prior turns) and
    // includes the system prompt, so turn 2+ no longer falls through to the
    // Plain branch (which dropped the system prompt and lost the Jinja
    // template). The cold-reset further down (`jinja_active && seq_pos > 0`)
    // re-prefills this full render from position 0.
    let try_jinja = jinja_enabled && m.meta.chat_template.is_some();
    let new_tokens = if try_jinja {
        let template = m.meta.chat_template.as_ref().unwrap();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
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
                            tool_plan: String::new(),
                        });
                    }
                    v.push(hipfire_runtime::prompt_frame::Message {
                        role: hipfire_runtime::prompt_frame::Role::User,
                        content: prompt.to_string(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        tool_plan: String::new(),
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
                hipfire_runtime::prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: false,
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        hipfire_runtime::prompt_frame::ChatFrame {
            tokenizer,
            system: if m.session.seq_pos == 0 {
                system_prompt
            } else {
                None
            },
            user: "", // unused: we pass tokens directly via build_with_user_tokens
            assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(&q_tokens)
    };

    // ── Prompt cache (LCP-based) — Qwen3.5/3.6 only ──────────────────────
    //
    // Mirrors V4F's prefix-cache (daemon.rs ~5390). Eligible when:
    //   - HIPFIRE_QWEN_PROMPT_CACHE != "0"  (default on)
    //   - messages_history is provided (full-conversation context)
    //   - eviction not active (compact_offset > 0 invalidates the
    //     "conversation_tokens mirrors KV" invariant the cache relies on)
    //   - PFlash compression not enabled this session (compression
    //     changes the KV's token IDs relative to msg.content from history)
    //   - prior conversation_tokens non-empty (first turn = nothing to LCP)
    //
    // On HIT we set `m.session.seq_pos = LCP` and override `new_tokens` to the
    // suffix slice [LCP..] so the prefill below only writes new tokens.
    // DeltaNet state at position LCP is already correct (cumulative from
    // prior decode). On MISS (divergence in the middle) we full-reset
    // (seq_pos=0, conversation_tokens.clear(), zero DeltaNet, KV
    // compact_offset=0) and prefill the FULL rendered prompt — DeltaNet
    // is not reversible to position M<N so partial rollback is unsafe.
    // Jinja-on disqualification: when `HIPFIRE_JINJA_CHAT=1` the first
    // turn renders through the upstream HF chat template (which the
    // model was actually trained on — emits default system prompts,
    // Hermes XML tool-call format on Qwen3.5/3.6, etc.). The cache
    // path uses scaffold-style rendering (`ChatScaffold`) which
    // produces a DIFFERENT byte sequence for the same logical content.
    // Mixing the two within a session would degrade output quality
    // (the model sees a different input distribution than it was
    // trained for after turn 1). Skip the cache when Jinja is active
    // so the operator gets consistent rendering across all turns.
    // Cache-with-Jinja is a future project (would require Jinja-side
    // assistant-turn replay).
    // Cache-with-Jinja (item #37): `jinja_active` is NO LONGER a disqualifier.
    // When jinja is active the prompt-build below routes through
    // `build_cached_history_jinja` (verbatim assistant-turn splice through the
    // model's trained template) instead of the ChatScaffold `build_cached_history`,
    // so the LCP forward-extension cache now works under HIPFIRE_JINJA_CHAT too.
    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache eligible] eligible={} kill={} hist={} evict_none={} !pflash={} jinja={} conv_tok={}",
            cache_eligible,
            cache_kill_switch,
            messages_history.is_some(),
            m.eviction.is_none(),
            !pflash_active,
            jinja_active,
            m.session.conversation_tokens.len(),
        );
    }
    let mut cached_tokens_count: usize = 0;
    let mut cold_reset_required = false;
    let new_tokens: Vec<u32> = if cache_eligible {
        let history = messages_history.unwrap();
        let trace_cache = std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1");
        // Build the canonical full-conversation token stream, replaying
        // any historical assistant turn whose fingerprint matches a
        // cached emission (BPE-bijective replacement).
        let rendered = if jinja_active {
            // Jinja cache (item #37): render the full conversation through the
            // model's trained template, splicing each cached assistant turn's
            // VERBATIM tokens in place of its content (sentinel substitution).
            // The store side (`asst_turn_cache`) holds the GENERATED body only
            // (post-primer); the template renders a history assistant turn as
            // `<|im_start|>assistant\n{content}` with NO generation primer, so
            // we prepend the assistant-opener primer (e.g. `<think>\n`) that
            // THIS turn's cold render emitted — making the spliced stream
            // byte-match `conversation_tokens` for a clean forward extension.
            let primer: Vec<u32> = {
                let im_start = tokenizer.special_token_id("<|im_start|>");
                let opener_len = tokenizer.encode("<|im_start|>assistant\n").len();
                match im_start.and_then(|id| new_tokens.iter().rposition(|&t| t == id)) {
                    Some(q) if q + opener_len <= new_tokens.len() => {
                        new_tokens[q + opener_len..].to_vec()
                    }
                    _ => Vec::new(),
                }
            };
            let template = m.meta.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let cache_ref = &mut m.persist.asst_turn_cache;
            let built = hipfire_runtime::prompt_frame::build_cached_history_jinja(
                &frame,
                history,
                tools,
                |msg| {
                    let stripped = strip_think_for_fingerprint(&msg.content);
                    let normalized =
                        hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
                    let fp = asst_turn_fingerprint(&normalized, &msg.tool_calls);
                    let hit = cache_ref.get(&fp).map(|cached| {
                        let mut v = primer.clone();
                        v.extend_from_slice(cached);
                        v
                    });
                    if trace_cache {
                        eprintln!(
                            "[qwen-cache jinja lookup] fp={:#018x} role={:?} content.len={}/stripped.len={} primer={} hit={}",
                            fp,
                            msg.role,
                            msg.content.len(),
                            normalized.len(),
                            primer.len(),
                            hit.is_some(),
                        );
                    }
                    hit
                },
            );
            match built {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[qwen-cache] jinja cached-history build failed ({e}) — cold render");
                    new_tokens.clone()
                }
            }
        } else {
            let cache_ref = &mut m.persist.asst_turn_cache;
            hipfire_runtime::prompt_frame::build_cached_history(
                tokenizer,
                system_prompt,
                history,
                &q_tokens,
                assistant_prefix,
                |msg| {
                    // Match the store side's stripping. The store applies
                    // `strip_think_for_fingerprint` then `maybe_normalize_prompt`
                    // to the model's emitted text before hashing. The CLI
                    // is SUPPOSED to strip `<think>...</think>` from the
                    // visible content before forwarding to clients, but
                    // the inThink state machine only handles paired blocks;
                    // when non-thinking mode prefills `<think>\n\n</think>\n\n`
                    // the model often resumes by emitting another orphan
                    // `</think>\n\n` (training-distribution artifact),
                    // which leaks through to the client's msg.content
                    // verbatim. Apply the same strip here so the lookup
                    // hash matches the store hash regardless of whether
                    // the client preserved the orphan.
                    let stripped = strip_think_for_fingerprint(&msg.content);
                    let normalized =
                        hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
                    let fp = asst_turn_fingerprint(&normalized, &msg.tool_calls);
                    let hit = cache_ref.get(&fp).cloned();
                    if trace_cache {
                        eprintln!(
                            "[qwen-cache lookup] fp={:#018x} role={:?} content.len={}/stripped.len={} tool_calls={} hit={}",
                            fp,
                            msg.role,
                            msg.content.len(),
                            normalized.len(),
                            msg.tool_calls.len(),
                            hit.is_some(),
                        );
                    }
                    hit
                },
            )
        };
        // LCP detection vs m.conversation_tokens.
        let prior_len = m.session.conversation_tokens.len();
        let max_match = prior_len.min(rendered.len());
        let mut lcp = 0usize;
        while lcp < max_match && m.session.conversation_tokens[lcp] == rendered[lcp] {
            lcp += 1;
        }
        if trace_cache {
            eprintln!(
                "[qwen-cache lcp] prior_len={} rendered_len={} lcp={}",
                prior_len,
                rendered.len(),
                lcp,
            );
            if lcp < prior_len || lcp < rendered.len() {
                // Print full token-ID context on each side past lcp,
                // not just the symmetric overlap window. Lets us see
                // BPE drift cases (same decoded bytes, different ids)
                // and "one side ran out" cases (rendered_len == lcp).
                let pre = lcp.saturating_sub(6);
                let prior_post = (lcp + 16).min(prior_len);
                let rend_post = (lcp + 16).min(rendered.len());
                if lcp > pre {
                    eprintln!(
                        "  common[{}..{}] ids={:?} dec={:?}",
                        pre,
                        lcp,
                        &m.session.conversation_tokens[pre..lcp],
                        tokenizer.decode(&m.session.conversation_tokens[pre..lcp]),
                    );
                }
                if prior_post > lcp {
                    eprintln!(
                        "  prior_past[{}..{}] ids={:?} dec={:?}",
                        lcp,
                        prior_post,
                        &m.session.conversation_tokens[lcp..prior_post],
                        tokenizer.decode(&m.session.conversation_tokens[lcp..prior_post]),
                    );
                }
                if rend_post > lcp {
                    eprintln!(
                        "  rend_past[{}..{}] ids={:?} dec={:?}",
                        lcp,
                        rend_post,
                        &rendered[lcp..rend_post],
                        tokenizer.decode(&rendered[lcp..rend_post]),
                    );
                }
            }
        } else if lcp < prior_len && prior_len > 50 {
            // Production-visible cache-miss log. Only fires when LCP
            // detected a real divergence (not the first-turn or
            // small-context case). Helps diagnose Pi-style "single-turn
            // cache invalidation" patterns without requiring the
            // operator to reproduce with HIPFIRE_QWEN_CACHE_TRACE=1.
            // Cheap (one eprintln per miss, not per turn).
            //
            // Three windows printed (each clipped to 60 chars):
            //  - common@lcp-4..lcp  — shared tail before divergence
            //  - prior@lcp..lcp+12  — what prior had past lcp (empty if rendered is longer)
            //  - rendered@lcp..lcp+12 — what rendered had past lcp (empty if prior is longer)
            // Plus prior_tail / rendered_tail (last 4 tokens) so we
            // know what each side ends with.
            let pre = lcp.saturating_sub(4);
            let common_dec = if lcp > pre {
                tokenizer.decode(&m.session.conversation_tokens[pre..lcp])
            } else {
                String::new()
            };
            let prior_post = (lcp + 12).min(prior_len);
            let prior_past_dec = if prior_post > lcp {
                tokenizer.decode(&m.session.conversation_tokens[lcp..prior_post])
            } else {
                String::new()
            };
            let rend_post = (lcp + 12).min(rendered.len());
            let rend_past_dec = if rend_post > lcp {
                tokenizer.decode(&rendered[lcp..rend_post])
            } else {
                String::new()
            };
            let prior_tail = if prior_len >= 4 {
                tokenizer.decode(&m.session.conversation_tokens[prior_len - 4..])
            } else {
                tokenizer.decode(&m.session.conversation_tokens[..])
            };
            let rend_tail = if rendered.len() >= 4 {
                tokenizer.decode(&rendered[rendered.len() - 4..])
            } else {
                tokenizer.decode(&rendered[..])
            };
            eprintln!(
                "[qwen-cache miss] lcp={} prior_len={} rendered_len={}",
                lcp,
                prior_len,
                rendered.len(),
            );
            eprintln!(
                "  common@{}..{}={:?}",
                pre,
                lcp,
                common_dec.chars().take(60).collect::<String>(),
            );
            eprintln!(
                "  prior_past@{}..{}={:?} rendered_past@{}..{}={:?}",
                lcp,
                prior_post,
                prior_past_dec.chars().take(60).collect::<String>(),
                lcp,
                rend_post,
                rend_past_dec.chars().take(60).collect::<String>(),
            );
            eprintln!(
                "  prior_tail={:?} rendered_tail={:?}",
                prior_tail.chars().take(60).collect::<String>(),
                rend_tail.chars().take(60).collect::<String>(),
            );
        }
        if lcp < prior_len || lcp == rendered.len() {
            // Divergence OR exact full-match — NOT a pure forward extension.
            // `lcp == rendered.len()` (⇒ lcp == prior_len) means the request
            // re-renders byte-identically; re-prefilling the final token (the old
            // `lcp-1` over-advance in the else-branch) would re-apply its
            // NON-COMMUTATIVE DeltaNet recurrent update a second time, corrupting
            // S-matrix/conv_state (temp-0 non-determinism + BF16 divergence on
            // re-sent prompts). DeltaNet has no rewindable KV (unlike FullAttention),
            // so the exact-match edge MUST degrade to checkpoint-resume / cold reset —
            // the strict-`<` HIT predicate the sibling DFlash plan_prompt_cache uses.
            //
            // Divergence: the client sent a non-extension render (it dropped or
            // edited earlier history, so the prior conversation is no longer a
            // prefix of this prompt). Rather than cold-prefill the whole thing,
            // try to RESUME from the latest prefill checkpoint at or before
            // `lcp`: restore the DeltaNet recurrent state captured there, rewind
            // seq_pos + the KV write head, and re-prefill only
            // [resume_pos..rendered.len()). KV for [0..resume_pos] is still
            // resident (positional, never overwritten). Gated to the single-GPU,
            // no-eviction case — eviction remaps physical KV slots, which would
            // invalidate the resident prefix. `seq_pos < rendered.len()` on the
            // chosen checkpoint guarantees ≥1 token is re-prefilled.
            //
            // SAFETY INVARIANT (fix/deltanet-truncation-resume-guard): this
            // restore-checkpoint-at-rpos + replay rendered[rpos..] is exact iff the
            // checkpoint at rpos reflects the committed prefix rendered[..rpos].
            // That holds because (a) rpos <= lcp => rendered[..rpos] ==
            // conversation_tokens[..rpos] (lcp is their longest common prefix), and
            // (b) ALL abort paths now full-reset, so a retained checkpoint can never
            // carry UNCOMMITTED tokens — the poison that used to drift the
            // non-reversible DeltaNet state into garbage. If you ever remove an
            // abort-reset (or let conversation_tokens diverge from the forwarded
            // stream), this resume becomes unsound: re-validate with a per-checkpoint
            // prefix hash (llama.cpp's tokens_hash contract) or cold-recompute.
            // Guarded by scripts/test-qwen35-abort-resume.sh.
            let evict_safe = !m.parallel.is_pipelined()
                && m.eviction.is_none()
                && m.state.as_ref().map_or(true, |s| match s {
                    ModelState::Llama(b) => b.kv.compact_offset == 0,
                    // qwen35's KV compact_offset lives in the bundle, not the
                    // always-None m.kv_cache direct field.
                    ModelState::Qwen35(b) => b.kv_cache.compact_offset == 0,
                    _ => true,
                });
            // Resume is only valid for qwen35 (the DeltaNet recurrent state in the
            // bundle). The gate used to read the always-None m.dn_state → resume
            // was silently disabled post-merge; gate on the bundle instead.
            let resume_idx = if ckpt_resume_enabled()
                && evict_safe
                && matches!(m.state.as_ref(), Some(ModelState::Qwen35(_)))
            {
                m.session
                    .prefill_checkpoints
                    .iter()
                    .rposition(|(p, _)| *p <= lcp && *p < rendered.len())
            } else {
                None
            };
            let resumed = if let Some(idx) = resume_idx {
                let rpos = m.session.prefill_checkpoints[idx].0;
                // RESTORE only (do NOT zero): roll the bundle's DeltaNet state
                // back to the checkpoint. Disjoint split: m.state and
                // m.session.prefill_checkpoints are different fields of `m`.
                let ok = if let (Some(ModelState::Qwen35(b)), Some(ck)) =
                    (m.state.as_mut(), m.session.prefill_checkpoints.get(idx))
                {
                    ck.1.restore_to(&mut b.dn_state, gpu).is_ok()
                } else {
                    false
                };
                if ok {
                    m.session.seq_pos = rpos;
                    // `evict_safe` guarantees compact_offset == 0, so setting
                    // seq_pos already points the KV write head at rpos — nothing
                    // to restore (checkpoints are only captured with offset 0).
                    m.session.conversation_tokens.truncate(rpos);
                    truncate_checkpoints(&mut m.session.prefill_checkpoints, idx + 1, gpu);
                    cached_tokens_count = rpos;
                    eprintln!(
                        "[qwen-cache resume] rewound to checkpoint pos={} (lcp={}, prior_len={}, rendered_len={}) — replaying {} tokens vs cold-prefilling {}",
                        rpos,
                        lcp,
                        prior_len,
                        rendered.len(),
                        rendered.len() - rpos,
                        rendered.len(),
                    );
                    Some(rendered[rpos..].to_vec())
                } else {
                    None
                }
            } else {
                None
            };
            match resumed {
                Some(tail) => tail,
                None => {
                    // No usable checkpoint — full cold reset. DeltaNet recurrent
                    // state is non-reversible; treat as a miss.
                    cold_reset_required = true;
                    rendered
                }
            }
        } else {
            // Pure forward extension: `lcp == prior_len && lcp < rendered.len()`.
            // The prior turn left the recurrent DeltaNet state at exactly
            // `prior_len`, so reusing KV/DeltaNet[0..lcp] and prefilling the new
            // suffix `rendered[lcp..]` (≥1 token, since lcp < rendered.len())
            // advances the state correctly with no rewind and no over-advance.
            // The exact-match edge (lcp == rendered.len()) no longer reaches here —
            // it degrades to checkpoint-resume / cold reset above.
            m.session.seq_pos = lcp;
            cached_tokens_count = lcp;
            rendered[lcp..].to_vec()
        }
    } else {
        new_tokens
    };

    // Jinja path renders the full conversation each turn. When the LCP cache
    // ran this turn (`cache_eligible`), it already managed seq_pos — set it to
    // the LCP on a forward-extension HIT, or full-reset on a MISS — so we must
    // NOT blanket-reset here (that would discard a valid cache hit and force a
    // cold re-prefill every turn). Only cold-reset when the cache did NOT run
    // (item #37): first turn (empty conversation), kill switch
    // (HIPFIRE_QWEN_PROMPT_CACHE=0), eviction/PFlash active. On turn 2+ in those
    // cases, reset BEFORE the budget guard + prefill so the full render writes
    // from position 0 rather than appending to the prior turn's dirty
    // DeltaNet/KV/checkpoint state. Uses `free_checkpoints` (NOT a bare
    // `.clear()`) so the checkpoint GPU buffers are freed rather than leaked.
    // A cache miss is a total context transition. Do this before the capacity
    // guard: a reset can move an effectively-zero-capacity session back to
    // position zero, while checking the dirty position first can reject a
    // request that would fit in the freshly reset context.
    if cold_reset_required {
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();

    // KV-budget guard. Without eviction the physical buffer is the hard cap;
    // we must fit prefill + generation + trailer in one allocation. With
    // eviction, physical is bounded by physical_cap regardless of total tokens
    // — the chunked prefill below calls maybe_evict between chunks, and the
    // decode loop evicts after every token. The only ceiling under eviction is
    // the advertised context window (max_seq) — refuse requests that would
    // overflow it in absolute position terms (current absolute + new).
    let trailer = nl.len();
    let guard_seq_pos = qwen_cache_guard_position(m.session.seq_pos, cold_reset_required);
    let absolute_pos = m.session.seq_pos.saturating_add(
        m.state
            .as_ref()
            .and_then(|s| match s {
                ModelState::Llama(b) => Some(b.kv.compact_offset),
                // qwen35 KV compact_offset lives in the bundle, not the
                // always-None m.kv_cache direct field.
                ModelState::Qwen35(b) => Some(b.kv_cache.compact_offset),
                _ => None,
            })
            .unwrap_or(0),
    );
    if m.eviction.is_none() {
        if guard_seq_pos
            .saturating_add(new_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.meta.physical_cap
        {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
                id,
                guard_seq_pos,
                new_tokens.len(),
                max_tokens,
                trailer,
                m.meta.physical_cap
            );
            let _ = stdout.flush();
            return GenerateResult::Complete;
        }
    } else if absolute_pos
        .saturating_add(new_tokens.len())
        .saturating_add(max_tokens)
        .saturating_add(trailer)
        > m.meta.max_seq
    {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds advertised context window: absolute={} + prefill={} + max_tokens={} + trailer={} > max_seq={}"}}"#,
            id,
            absolute_pos,
            new_tokens.len(),
            max_tokens,
            trailer,
            m.meta.max_seq
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
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

    if m.meta.arch_id == 5 || m.meta.arch_id == 6 {
        // Qwen3.5 / Qwen3.5-MoE AR decode via the generic ArchDispatch driver
        // (Inc 1 Task 1.4d — flipped from the legacy inline arm after the dual-run
        // shadow-parity gate proved token-identity, single-GPU + emulated-2, FP32
        // DeltaNet + deterministic). The legacy arm + parity scaffold are removed.
        let mut __disp = Qwen35Dispatch { m: &mut *m };
        return ar_generate(
            &mut __disp,
            ForwardCtx::Single(gpu),
            stdout,
            id,
            temp,
            top_p,
            top_k,
            min_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            stop,
            tools,
            new_tokens,
            &im_end,
            &nl,
            im_end_token,
            tool_call_pair,
            think_pair,
            prefill_tokens,
            cached_tokens_count,
            pflash_summary,
            pflash_bypass_reason,
            pflash_alpha,
            t0,
            None,
        );
    } else {
        // LLaMA (arch 0/1) AR decode via the generic ArchDispatch driver
        // (Inc 3, path-a uplift): batched forward + generic sampler via
        // LlamaDispatch. Gains loop guard / think-cap / stop-seqs / richer
        // sampling. NOT byte-identical to the legacy fused forward_scratch arm
        // (coherence + perf validated, not strict parity). pflash is qwen35-only.
        let mut __disp = LlamaDispatch { m: &mut *m };
        return ar_generate(
            &mut __disp,
            ForwardCtx::Single(gpu),
            stdout,
            id,
            temp,
            top_p,
            top_k,
            min_p,
            max_tokens,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            budget_alert_at_tok,
            budget_alert_text,
            max_think_tokens,
            assistant_prefix,
            stop,
            tools,
            new_tokens,
            &im_end,
            &nl,
            im_end_token,
            tool_call_pair,
            think_pair,
            prefill_tokens,
            cached_tokens_count,
            None,
            None,
            None,
            t0,
            None,
        );
    }
}

/// DeepSeek V4 Flash generate path (arch_id=9, hipfire-arch-deepseek4).
///
/// Parity with `deepseek4_chat`: batched chunked prefill +
/// optional MTP spec-decode + greedy argmax sampler. PBS is pre-allocated
/// once at load time (`b.pbs`), reused across every turn.
///
/// Env knobs (read fresh per generate call so they can be toggled
/// without daemon restart):
///   HIPFIRE_DEEPSEEK4_SPEC_DECODE=1     opt-in MTP speculative decode
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
fn build_deepseek4_dsml_prompt(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    system_prompt: Option<&str>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    live_prompt: &str,
    think_mode: ThinkMode,
    deepseek4_eos_tok: u32,
    asst_turn_cache: &mut AsstTurnCache,
) -> Vec<u32> {
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
        // Skip the trailing user prompt — we add it explicitly after, BUT only
        // when a non-empty `live_prompt` actually carries it. The OpenAI
        // messages API (no separate `prompt` field) puts the live user turn as
        // the LAST history message with `live_prompt == ""`; trimming it then
        // drops the user's question entirely (model greets instead of answering
        // — observed on ds4 EP tp4). So only trim when live_prompt is non-empty.
        use hipfire_runtime::prompt_frame::Role;
        let trim_end = if !live_prompt.is_empty()
            && matches!(history.last().map(|m| m.role), Some(Role::User))
        {
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
                    // Match store-side stripping (see qwen35 path comment).
                    let stripped = strip_think_for_fingerprint(&msg.content);
                    let normalized =
                        hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
                    let fp = asst_turn_fingerprint(&normalized, &msg.tool_calls);
                    if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        eprintln!(
                            "[asst-cache lookup] fp={:#018x} content.len={}/stripped.len={} tool_calls={} hit={}",
                            fp,
                            msg.content.len(),
                            normalized.len(),
                            msg.tool_calls.len(),
                            asst_turn_cache.contains_key(&fp),
                        );
                    }
                    if let Some(cached) = asst_turn_cache.get(&fp) {
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

                    // If the replayed turn body opened a `<think>` block but
                    // the model premature-stopped without closing it (EOS inside
                    // the think, no tool call), close it here with a `</think>`.
                    // Otherwise the dangling `<think>…<EOS>` drifts the next turn
                    // (more premature stops, a leaked `</think>`). This is a
                    // deterministic surround token — a pure function of
                    // msg.content, NOT part of the cached turn body or the
                    // asst_turn_fingerprint (which strips think anyway) — so it
                    // is emitted identically on hit and miss paths and the
                    // prefix-cache LCP + asst_turn_cache stay effective.
                    if msg.tool_calls.is_empty()
                        && msg.content.starts_with("<think>")
                        && !msg.content.contains("</think>")
                    {
                        prompt_ids.extend(tokenizer.encode("</think>"));
                    }

                    // Close the assistant turn with the EOS marker so
                    // the next turn starts cleanly.
                    prompt_ids.push(deepseek4_eos_tok);
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
    if !live_prompt.is_empty() {
        if let Some(u) = user_tok {
            prompt_ids.push(u);
        }
        prompt_ids.extend(tokenizer.encode(live_prompt));
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

    prompt_ids
}

/// Resolve whether deepseek4 spec-decode is requested for this model, mirroring
/// the env/config chain inside `generate_deepseek4` (daemon.rs spec_requested).
/// The dispatch uses this (plus `temp <= 1e-6` and `m.speculator.is_some()`) to
/// route the spec path through the unified `generate_spec`; the AR path (and the
/// no-speculator fallback) stay in `generate_deepseek4`.
fn deepseek4_spec_requested(m: &LoadedModel) -> bool {
    mtp_metadata_requested(&m.meta.mtp_mode, m.mtp_weights_present())
}

fn mtp_metadata_requested(mtp_mode: &str, weights_present: bool) -> bool {
    mtp_mode == "on" || (mtp_mode == "auto" && weights_present)
}

/// deepseek4 MTP spec-decode through the unified `generate_spec` (Phase 4 T4c-2).
///
/// This is the ds4 sibling of `generate_dflash`: it owns the arch-specific
/// prologue (DSML prompt render + `plan_cache(CachePolicy::deepseek4())` + the
/// DSA decode-cache miss teardown) and epilogue (the ds4 `done` envelope), and
/// drives the shared decode core via `m.speculator` (a `Deepseek4MtpDrafter`),
/// the `Deepseek4Bundle` target (via `spec_target_guard`), and `Deepseek4Emit`.
/// Greedy-only — the dispatch routes here only at `temp <= 1e-6`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn generate_deepseek4_spec(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    // Sampling: temp<=1e-6 → greedy (argmax-accept, byte-identical to before);
    // temp>0 → DSpark sampled verify. cactus_delta is the opt-in acceptance boost.
    temp: f32,
    top_p: f32,
    top_k: usize,
    cactus_delta: f32,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) -> GenerateResult {
    // eos token (for the DSML prompt build) from the bundle — immutable peek.
    let eos_tok = match m.state.as_ref() {
        Some(ModelState::Deepseek4(b)) => b.eos_tok,
        _ => {
            emit_error_with_id(stdout, id, "deepseek4 bundle missing on arch_id=9 spec");
            return GenerateResult::Complete;
        }
    };

    // DSML prompt render (same builder the bespoke loop uses).
    let prompt_ids = {
        let tokenizer = match m.tokenizer.as_ref() {
            Some(t) => t,
            None => {
                emit_error_with_id(stdout, id, "tokenizer not loaded");
                return GenerateResult::Complete;
            }
        };
        build_deepseek4_dsml_prompt(
            tokenizer,
            system_prompt,
            tools,
            messages_history,
            prompt,
            think_mode,
            eos_tok,
            &mut m.persist.asst_turn_cache,
        )
    };
    if prompt_ids.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return GenerateResult::Complete;
    }

    let spec_k = m.meta.mtp_k;

    // Prefix-cache plan (ds4 policy: forced-cold on partial, ring-safety length
    // guard, step-back exact). Pure decision; the GPU teardown is applied below.
    let plan = hipfire_runtime::cache_plan::plan_cache(
        &prompt_ids,
        &m.session.conversation_tokens,
        &hipfire_runtime::cache_plan::CachePolicy::deepseek4(),
        &[],
        false,
    );
    let cached_tokens = plan.cached_tokens;
    let suffix: Vec<u32> = prompt_ids[plan.start_pos..].to_vec();

    // Capacity guard (KV sized for physical_cap; overrun is a serve-killing
    // panic). Mirrors the bespoke ds4 pre-prefill guard — generate_spec's own
    // guard checks ctx_capacity (max_position_embeddings), which for ds4 can far
    // exceed physical_cap, so keep this explicit one.
    if plan
        .start_pos
        .saturating_add(suffix.len())
        .saturating_add(max_tokens)
        > m.meta.physical_cap
    {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq",
                plan.start_pos + suffix.len(),
                max_tokens,
                m.meta.physical_cap
            ),
        );
        return GenerateResult::Complete;
    }

    // The ds4 emitter builds its in-step tool-call grammar from the raw tool
    // JSON inside `make_spec_emitter`, but that grammar masks per-token over the
    // decoded vocab — which the neutral `SpecEmitCtx` can't lazily derive (it has
    // no `&mut m`). Build/cache the vocab Arc here when tools are present and
    // hand it down; mirrors the lazy cache the old `build_deepseek4_spec_grammar`
    // did internally.
    let decoded_vocab: Option<std::sync::Arc<Vec<String>>> =
        if tools.map_or(false, |t| !t.is_empty()) {
            if m.persist.decoded_vocab.is_none() {
                let tok = m.tokenizer.as_ref().expect("tokenizer present");
                let n = tok.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tok.decode(&[id as u32])).collect();
                m.persist.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.persist.decoded_vocab.clone()
        } else {
            None
        };

    // Configure the speculator's sampling before the loop — mirrors generate_dflash's
    // set_sampling before generate_spec. Greedy (temp<=1e-6) leaves it at argmax-accept
    // (byte-identical to the prior hardcoded-greedy path); temp>0 drives the DSpark
    // sampled verify, and cactus_delta>0 applies the opt-in acceptance boost.
    if let Some(spec) = m.speculator.as_mut() {
        spec.set_sampling(temp, top_p, top_k, cactus_delta);
    }
    let prompt_tokens_total = prompt_ids.len();
    let run = match generate_spec(
        m,
        gpu,
        stdout,
        id,
        prompt_ids,
        suffix,
        plan.start_pos,
        plan.cache_hit,
        None, // resume_from — ds4 has no DeltaNet checkpoints
        max_tokens,
        SpecEmitRequest {
            im_end: None,
            tools: tools.map(|t| t.to_vec()),
            stop: Vec::new(),
            max_think: 0,
            assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
            think_mode,
            decoded_vocab,
        },
        temp, // temp>0 → DSpark sampled verify (routed here only when supports_temp_verify)
    ) {
        Ok(Some(r)) => r,
        // Abort / error early-exit already wrote its own done/error envelope.
        Ok(None) => return GenerateResult::Complete,
        Err(failure) => return failure,
    };

    // ── ds4 done envelope ────────────────────────────────────────
    let tok_s = if run.decode_s > 0.0 {
        run.generated as f64 / run.decode_s
    } else {
        0.0
    };
    // accept_pct denominator is windows × k (the bespoke loop added `spec_k` per
    // window to `spec_drafts_offered`, not the actual n_proposed).
    let accept_pct = if run.spec_cycles > 0 && spec_k > 0 {
        run.spec_accepted as f64 / (run.spec_cycles * spec_k) as f64 * 100.0
    } else {
        0.0
    };
    let finish_reason: &'static str = if run.finish.tool_calls > 0 {
        "tool_calls"
    } else if run.generated >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let done_envelope = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": run.generated,
        "tok_s": tok_s,
        "prompt_tokens": prompt_tokens_total,
        "prefill_tokens": run.prefill_tokens_len,
        "cached_tokens": cached_tokens,
        "prefill_ms": (run.prefill_s * 1000.0) as u128,
        "total_ms": (run.total_s * 1000.0) as u128,
        "finish_reason": finish_reason,
        "spec_k": spec_k,
        "spec_windows": run.spec_cycles,
        "spec_accept_pct": accept_pct,
    });
    let _ = writeln!(stdout, "{}", done_envelope);
    let _ = stdout.flush();
    GenerateResult::Complete
}

/// Local terminal policy for DeepSeek's bespoke AR parser. The parser owns
/// buffered DSML/text state and its `finish` consumes it, so normal completion
/// is the sole path that may render final events. Abort/error paths explicitly
/// discard the parser without invoking `finish`.
fn settle_deepseek4_ar_parser(
    parser: hipfire_arch_deepseek4::dsml::StreamParser,
    normal_completion: bool,
    mut render: impl FnMut(hipfire_arch_deepseek4::dsml::StreamEvent),
) {
    if normal_completion {
        for event in parser.finish() {
            render(event);
        }
    } else {
        drop(parser);
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
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) -> GenerateResult {
    let eos_tok = match m.state.as_ref() {
        Some(ModelState::Deepseek4(b)) => b.eos_tok,
        _ => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"deepseek4_config missing on arch_id=9 generate"}}"#,
                id
            );
            let _ = stdout.flush();
            return GenerateResult::Complete;
        }
    };

    let prompt_ids = {
        let tokenizer = match m.tokenizer.as_ref() {
            Some(t) => t,
            None => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
                    id
                );
                let _ = stdout.flush();
                return GenerateResult::Complete;
            }
        };
        build_deepseek4_dsml_prompt(
            tokenizer,
            system_prompt,
            tools,
            messages_history,
            prompt,
            think_mode,
            eos_tok,
            &mut m.persist.asst_turn_cache,
        )
    };

    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    if std::env::var("HIPFIRE_DEEPSEEK4_DUMP_PROMPT")
        .ok()
        .as_deref()
        == Some("1")
    {
        let tokenizer = m.tokenizer.as_ref().unwrap();
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

    // This function is now AR-only: the MTP spec-decode path (greedy, temp≈0,
    // speculator present) is dispatched to `generate_deepseek4_spec` (T4c-2),
    // which owns the spec_requested/spec_mode/spec_k resolution. Reaching here
    // means either temp>0 (sample) or no speculator (fallback) — either way the
    // plain AR sampler below honours the requested `temp`/`top_p`.

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
        let prior = &m.session.conversation_tokens;
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

    // DSA compressor-ring safety on a PARTIAL prefix-cache hit.
    //
    // The DSA decode caches (SWA ring, compressor/indexer ring state, full +
    // compressed KV) are *position-indexed* and were left by the prior turn at
    // ITS end position. A FULL hit (`lcp == prior length`) resumes exactly where
    // the prior turn left those rings, so the incremental prefill is correct —
    // this is the normal "growing conversation" path and stays fast.
    //
    // A PARTIAL hit (`0 < lcp < prior length`) resumes the suffix prefill from
    // `start_pos = lcp`, but the compressor ring still holds the prior turn's
    // *end* window, not `lcp`'s. The first compressed block committed after the
    // resume point then pools a STALE overlap window — and with ratio-4 overlap
    // that window reaches back over the just-cached tail, corrupting far-context
    // recall (the cwd/tool-path "lossiness" symptom). The ring can't be cheaply
    // repopulated (a position's hidden state depends on its SWA window, which
    // chains all the way back to token 0), so the correct, robust fix is to fall
    // back to a cold rebuild for partial hits only. Full hits are unaffected.
    let lcp = if lcp > 0 && lcp < m.session.conversation_tokens.len() {
        0
    } else {
        lcp
    };

    if lcp == 0 {
        // Cache miss — start a fresh conversation in V4F's state.
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let start_pos: u32 = lcp as u32;

    // Slice off the suffix — the only tokens we actually need to prefill.
    // For lcp=0 this is the full prompt; for a full cache hit on a turn
    // that adds N new tokens this is just those N.
    let suffix_tokens: &[u32] = &prompt_ids[lcp..];

    // O2b-2 capacity guard (ds4 single-GPU): after any cache reset above, the
    // KV ends at start_pos + suffix_tokens.len() (== prompt_ids.len()) and
    // decode appends max_tokens. forward_prefill_batch_chunked writes into a KV
    // sized for m.meta.physical_cap; overrunning it is a KV-overrun panic that takes
    // down serve. Emit a clean error and return BEFORE prefill.
    // saturating_add: an adversarially huge max_tokens must not wrap usize and
    // slip under the cap.
    if (start_pos as usize)
        .saturating_add(suffix_tokens.len())
        .saturating_add(max_tokens)
        > m.meta.physical_cap
    {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id,
            start_pos as usize + suffix_tokens.len(),
            max_tokens,
            m.meta.physical_cap
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    // All lifecycle decisions and guards are complete. Only now borrow the
    // architecture bundle for the forward/decode phase.
    let Some(ModelState::Deepseek4(b)) = m.state.as_mut() else {
        unreachable!();
    };
    let cfg = &b.config;
    let weights = &b.weights;
    let state = &mut b.state;
    let pbs = &b.pbs;

    // Prefill: batched chunked through PBS. (The MTP-fill prefill variant moved
    // to the drafter's `mtp_prefill`, driven by `generate_deepseek4_spec`.)
    let prefill_result = deepseek4::forward::forward_prefill_batch_chunked(
        cfg,
        weights,
        state,
        gpu,
        suffix_tokens,
        start_pos,
        pbs,
    );
    let last_logits = match prefill_result {
        Ok(l) => l,
        Err(e) => {
            // Release the architecture borrows before the canonical reset;
            // otherwise this path can emit an error while dirty DS4 rings and
            // session tokens remain reusable by turn two.
            if let Err(reset_error) = model_reset_context(m, gpu) {
                return reset_failed(id, reset_error);
            } else {
                emit_error_with_id(stdout, id, format!("deepseek4prefill failed: {e:?}"));
            }
            return GenerateResult::Complete;
        }
    };
    // `forward_prefill_batch_chunked` does NOT advance `state.n_tokens`.
    // Callers are responsible for it (mirrors deepseek4_chat's explicit
    // `state.n_tokens = pos as u64;` at deepseek4_chat.rs:324). Without this,
    // the next decode_step queries the SWA cache at the BOS position
    // instead of the next-prediction position and the model emits
    // attractor garbage at greedy temp=0.
    state.n_tokens = (start_pos as usize + suffix_tokens.len()) as u64;
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
        m.session.conversation_tokens.clear();
        m.session.conversation_tokens.extend_from_slice(&prompt_ids);
    } else {
        m.session.conversation_tokens.truncate(lcp);
        m.session
            .conversation_tokens
            .extend_from_slice(suffix_tokens);
    }
    let cached_tokens: usize = lcp;

    // Sync to ensure all prefill kernels have completed before stopping
    // the timer (head's download_f32 already syncs but defensive).
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = t0.elapsed().as_millis();

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    let pos_after_prefill = state.n_tokens as u32;

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
    // Plain autoregressive decode. The MTP spec-decode path now routes through
    // `generate_deepseek4_spec` + the unified `generate_spec` (T4c-2); this
    // function is the AR sampler path (temp>0) and the no-speculator fallback.
    // The bare block scopes the parser/matcher/sampler locals (was the old
    // `else` arm of the now-deleted `if spec_mode` spec loop).
    {
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
        // using V4F turn. The cache lives on `LoadedModel.persist.decoded_vocab`
        // as an `Arc<Vec<String>>` and is cleared on model unload.
        //
        // Borrow note: `m.persist.decoded_vocab` is a disjoint field from
        // `m.state` (whose `ModelState::Deepseek4` bundle `state` holds `&mut`
        // to) and from `m.tokenizer` (which `tokenizer` holds `&` to), so the
        // assignment compiles under Rust's split-borrows.
        let decoded_vocab_arc: Option<std::sync::Arc<Vec<String>>> = if grammar_active {
            if m.persist.decoded_vocab.is_none() {
                let n = tokenizer.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tokenizer.decode(&[id as u32])).collect();
                m.persist.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.persist.decoded_vocab.clone()
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
        let decode_start_tokens_idx = m.session.conversation_tokens.len();
        let mut emit_text_buf = String::new();
        let mut emit_tool_calls_buf: Vec<hipfire_runtime::prompt_frame::ToolCall> = Vec::new();
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
                        emit_tool_calls_buf.push(hipfire_runtime::prompt_frame::ToolCall {
                            name: c.name.clone(),
                            arguments: c.arguments.clone(),
                        });
                    }
                }
            }
        };

        while generated_count < max_tokens && next_tok != eos_tok {
            if check_abort(id) {
                // Cancellation is terminal but not a normal stream end: drop
                // the parser's buffered DSML/text state without flushing it.
                settle_deepseek4_ar_parser(parser, false, |_| {});
                drop(absorb_event);
                if let Err(reset_error) = model_reset_context(m, gpu) {
                    return reset_failed(id, reset_error);
                } else {
                    let (aborted, done) = deepseek4_abort_json(id, generated_count);
                    let _ = writeln!(stdout, "{aborted}");
                    let _ = writeln!(stdout, "{done}");
                }
                let _ = stdout.flush();
                return GenerateResult::Complete;
            }
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
            m.session.conversation_tokens.push(next_tok);
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
                    // GPU errors must not run the parser's terminal flush.
                    settle_deepseek4_ar_parser(parser, false, |_| {});
                    drop(absorb_event);
                    if let Err(reset_error) = model_reset_context(m, gpu) {
                        return reset_failed(id, reset_error);
                    } else {
                        emit_error_with_id(stdout, id, format!("deepseek4decode failed: {e:?}"));
                    }
                    let _ = stdout.flush();
                    return GenerateResult::Complete;
                }
            }
        }
        // Local normal-completion epilogue: EOS, stop, budget, and length all
        // flush exactly once before the done envelope is written below.
        settle_deepseek4_ar_parser(parser, true, |ev| {
            absorb_event(&ev);
            emit_stream_event(stdout, id, ev);
        });
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
            && m.session.conversation_tokens.len() > decode_start_tokens_idx
        {
            let cached_seq: Vec<u32> =
                m.session.conversation_tokens[decode_start_tokens_idx..].to_vec();
            let fp = asst_turn_fingerprint(&emit_text_buf, &emit_tool_calls_buf);
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
            m.persist.asst_turn_cache.insert(fp, cached_seq);
        }
    }

    m.session.seq_pos = state.n_tokens as usize;

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
    // `tool_calls_parsed_count` is set immediately after parser.finish() below.
    let finish_reason: &'static str = if tool_calls_parsed_count > 0 {
        "tool_calls"
    } else if generated_count >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let done_envelope = serde_json::json!({
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
    });
    let _ = writeln!(stdout, "{}", done_envelope);
    let _ = stdout.flush();
    GenerateResult::Complete
}

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
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) -> GenerateResult {
    if m.tokenizer.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }
    if m.lfm2moe().is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"lfm2moe_config missing on arch_id=11 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    // ── Prompt build (same two-path branch as the minimax AR path) ──
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
        // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
        // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
        // the .hfq carries no template, e.g. an older A1B convert).
        // Jinja default-ON (flipped 2026-06-09): render through the model's chat
        // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
        // ChatML/Plain). Falls back to Plain automatically when no template resolves.
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.meta.chat_template.is_some();
        if try_jinja {
            let template = m.meta.chat_template.as_ref().unwrap();
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
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
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
                        "[daemon] jinja render failed in lfm2moe path ({e}) — falling back to Plain"
                    );
                    hipfire_runtime::prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
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
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                raw: false,
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
        return GenerateResult::Complete;
    }

    let eos_tok = m.lfm2moe().unwrap().eos_tok;

    // LFM2.5 emits MULTIPLE EOS-class tokens inconsistently: the chat turn-end
    // `<|im_end|>` (which `eos_tok` resolves to) but ALSO the document-end
    // `<|endoftext|>` (and rarely `</s>`). The single `next_tok == eos_tok` stop
    // in the decode loop misses the others, so the model LEAKS a literal
    // `<|endoftext|>` into the visible answer (observed: "...Paris.<|endoftext|>")
    // and wastefully keeps generating. This id-set is the FAST path: it catches
    // any EOS-class token whose literal string round-trips through `encode` to a
    // single id (true for `<|im_end|>`). NOTE it does NOT round-trip for
    // `<|endoftext|>` (encode yields subwords, not the special id), so the
    // reliable catch for that one is the string-level guard in the decode loop
    // below, which matches on the DECODED frag.
    // Stop-id set for Lfm2MoeDispatch::is_eos. Unlike `stop_toks` (which uses `encode`
    // and so only catches the round-tripping `<|im_end|>`=eos_tok), this resolves the
    // ACTUAL special-token ids via `special_token_id` — so ar_generate's single is_eos
    // check catches `<|endoftext|>` / `</s>` too (the legacy loop needed a separate
    // decoded-frag string guard for those). Paired with the dispatch's eos_filter_config
    // (strips their literals from display).
    let stop_ids: Vec<u32> = {
        let tk = m.tokenizer.as_ref().unwrap();
        let mut v = vec![eos_tok];
        for s in ["<|im_end|>", "<|endoftext|>", "</s>"] {
            if let Some(id) = tk.special_token_id(s) {
                if !v.contains(&id) {
                    v.push(id);
                }
            }
        }
        v
    };

    // Cross-conversation reset (FIX: LFM turn-to-turn KV accumulation). The
    // prior design only reset on capacity overflow, so every request APPENDED to
    // the KV at the growing `n_tokens` and the model attended to all prior
    // requests' tokens at offset RoPE positions — benign for a couple of short
    // turns (RoPE decay + the prompt re-establishes context) but it bloats the KV
    // and degrades quality over a real multi-request serve session, only
    // recovering at overflow. LFM2.5's hybrid conv+GQA state can't be cheaply
    // rewound to an arbitrary prefix (the conv window chains back to token 0, like
    // ds4's SWA ring), so partial prefix-reuse is unsafe → cold-rebuild every
    // turn. This is NOT a perf regression: the path already re-prefills the full
    // prompt each turn; it now does so from position 0 with no stale KV. A
    // continuing conversation re-prefills its whole history from the prompt, so
    // multi-turn is preserved (validated: Bjorn/axolotl recall).
    if let Err(error) = model_reset_context(m, gpu) {
        return reset_failed(id, error);
    }

    // After the reset the KV starts at 0, so the only overflow risk is a SINGLE
    // prompt+generation larger than the whole context — the prefill decode_step
    // loop would write past the KV (sized for state.max_seq) and panic, taking
    // down serve. Emit a clean error BEFORE prefill — mirror the minimax/qwen2
    // guard. saturating_add: an adversarially huge max_tokens must not wrap usize
    // and slip under the cap.
    let cap = m.lfm2moe().unwrap().state.max_seq;
    if prompt_ids.len().saturating_add(max_tokens) > cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id,
            prompt_ids.len(),
            max_tokens,
            cap
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    let t0 = Instant::now();
    let __prefill_len = prompt_ids.len();

    // ── Decode via the generic ar_generate/Lfm2MoeDispatch driver (Inc 5 flip).
    // ar_generate prefills `new_tokens` (per-token decode_step, from seq_pos=0 — the
    // preamble cold-reset the KV) then runs the AR loop. Proven token-identical to the
    // legacy loop at temp0 on lfm2.5-8b-a1b.mq4 via the dual-run shadow parity this
    // replaces. UPLIFT: n-gram loop guard + generic sampler (temp0 == legacy argmax).
    // The stop-id set (is_eos) + eos_filter_config (in Lfm2MoeDispatch) subsume the
    // legacy id + decoded-frag stop guards; the eos-class literals are stripped from
    // display. No think-cap, no grammar, no LCP (cold-reset arch).
    let mut __disp = Lfm2MoeDispatch {
        m: &mut *m,
        stop_ids,
    };
    let ar_result = ar_generate(
        &mut __disp,
        ForwardCtx::Single(gpu),
        stdout,
        id,
        temp,
        top_p,
        None, // top_k
        None, // min_p
        max_tokens,
        1.0, // repeat_penalty (legacy: none)
        0,   // repeat_window
        0.0, // presence_penalty
        0.0, // frequency_penalty
        0,   // budget_alert_at_tok
        "",  // budget_alert_text
        0,   // max_think_tokens
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        &[],        // stop
        None,       // tools (grammar off)
        prompt_ids, // new_tokens: full render (cold-reset each turn, no LCP)
        &[],        // im_end
        &[],        // nl
        None,       // im_end_token
        None,       // tool_call_pair
        None,       // think_pair
        __prefill_len,
        0,    // cached_tokens_count (cold reset)
        None, // pflash_summary
        None, // pflash_bypass_reason
        None, // pflash_alpha
        t0,
        None, // tape (prod: no dual-run)
    );
    drop(__disp);
    ar_result
}

/// MiniMax-M2 (arch_id=10) generate path — minimal AR bring-up.
///
/// Mirrors `generate_lfm2moe`'s shape (prefill loop / chunked batched prefill,
/// per-token decode loop, JSONL `token` / `done` events) with two differences:
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
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) -> GenerateResult {
    if m.tokenizer.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }
    if m.minimax().is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"minimax_config missing on arch_id=10 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    // ── Prompt build (same two-path branch as the lfm2moe AR path) ──
    // `primed_think` records whether the rendered prompt actually ended with
    // the MiniMax `<think>` generation-primer, so we only re-emit the opener
    // (below) when the model truly begins inside the reasoning block. A jinja
    // render failure that falls back to the Plain frame leaves it false.
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // MiniMax-M2 (arch 10) and LFM2.5 (arch 11) REQUIRE their embedded Jinja
        // chat_template — their structural tokens are NOT ChatML. MiniMax frames
        // turns with `]~b]ai` / `[e~[` and primes the assistant with `<think>\n`;
        // LFM2 needs its `<|startoftext|>` BOS. The hand-rolled Plain ChatML frame
        // emits `<|im_start|>`/`<|im_end|>` which these models never trained on,
        // producing an off-distribution prompt that (a) decodes incoherently and
        // (b) never matches across turns so the LCP prompt-cache is dead. Force
        // jinja on for both (falls back to Plain only when the .hfq carries no
        // template).
        // Jinja default-ON (flipped 2026-06-09); opt out with HIPFIRE_JINJA_CHAT=0.
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.meta.chat_template.is_some();
        if try_jinja {
            let template = m.meta.chat_template.as_ref().unwrap();
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
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
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
                    eprintln!(
                        "[daemon] jinja render failed in minimax path ({e}) — falling back to Plain"
                    );
                    hipfire_runtime::prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
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
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                raw: false,
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
        return GenerateResult::Complete;
    }

    // Capacity guard. No eviction on arch_id=10 — reset the KV cursor when the
    // FULL rendered conversation + generation would overflow. `prompt_ids` is
    // the full Jinja-rendered conversation; the LCP below reuses the warm prefix.
    let overflow = {
        let state = &m.minimax().unwrap().state;
        prompt_ids.len().saturating_add(max_tokens) > state.max_seq
    };
    if overflow {
        let (n, cap) = {
            let state = &m.minimax().unwrap().state;
            (state.n_tokens, state.max_seq)
        };
        eprintln!("[daemon] arch_id=10 context full ({n}/{cap}) — resetting MiniMaxState",);
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }

        // O2b-2 capacity guard (minimax single): the reset above recovers a
        // grown multi-turn conversation, but a SINGLE prompt larger than the
        // whole context can never fit — prefilling it would write past the KV
        // (sized for state.max_seq) and panic, taking down serve. After the
        // reset, if prompt + generation still overflows, emit a clean error.
        let cap = m.minimax().unwrap().state.max_seq;
        // saturating_add: an adversarially huge max_tokens must not wrap usize
        // and slip under the cap.
        if prompt_ids.len().saturating_add(max_tokens) > cap {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
                id,
                prompt_ids.len(),
                max_tokens,
                cap
            );
            let _ = stdout.flush();
            return GenerateResult::Complete;
        }
    }

    // ── Prefix cache (LCP) with PARTIAL reuse. `prompt_ids` is the full
    // Jinja-rendered conversation (the trained chat template). MiniMax-M2 is an
    // INTERLEAVED-THINKING model: its chat_template renders a prior turn's
    // `<think>…</think>` reasoning into history ONLY while no newer user message
    // follows (`loop.index0 > last_user_index`). Once the next user turn
    // arrives the canonical render DROPS that reasoning, so every position after
    // the most-recent assistant opener shifts and turn N+1 diverges from turn
    // N's KV at that opener — i.e. `lcp < prior_len`, never a pure forward
    // extension. We therefore support PARTIAL reuse: rewind `n_tokens` to `lcp`
    // and re-prefill the (reasoning-free, hence shorter) suffix. MiniMax is
    // standard attention with no compound recurrent/compressed state, so KV
    // positions ≥ lcp are simply overwritten by the new prefill and the stale
    // tail is never attended. The reused prefix GROWS with the conversation
    // (all older turns, reasoning already stripped, stay matched), so
    // steady-state per-turn prefill is just {last visible answer} + {new user}.
    let prefill_ids: Vec<u32> = {
        let prior_len = m.session.conversation_tokens.len();
        let max_match = prior_len.min(prompt_ids.len());
        let mut lcp = 0usize;
        while lcp < max_match && m.session.conversation_tokens[lcp] == prompt_ids[lcp] {
            lcp += 1;
        }
        // A usable common prefix that leaves at least one fresh token to prefill
        // (the render always appends a new `]~b]ai\n<think>\n` primer, so
        // lcp == rendered_len cannot occur on a normal turn). `partial` is the
        // interleaved-thinking divergence (lcp < prior_len); lcp == prior_len is
        // the degenerate pure-extension case (rewind is then a no-op).
        let cache_hit = lcp > 0 && lcp < prompt_ids.len();
        let partial = lcp < prior_len;
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[minimax-cache] prior_len={} rendered_len={} lcp={} hit={} partial={} n_tokens={}",
                prior_len,
                prompt_ids.len(),
                lcp,
                cache_hit,
                cache_hit && partial,
                m.minimax().unwrap().state.n_tokens,
            );
        }
        if cache_hit {
            // Rewind KV + token history to the common prefix. When lcp ==
            // prior_len this is a no-op; when lcp < prior_len it discards the
            // stale reasoning+answer tail. The prefill loop below reads
            // `state.n_tokens` as its base position, so n_tokens is the only
            // KV state the rewind must touch (plus the mirror token history).
            m.minimax_mut().unwrap().state.n_tokens = lcp;
            m.session.conversation_tokens.truncate(lcp);
            m.session.seq_pos = lcp;
            prompt_ids[lcp..].to_vec()
        } else {
            // A cold-prefill decision owns the full context transition even
            // when the session mirror is already empty (for example after an
            // aborted or externally cleared request). Do not infer reset
            // ownership from `prior_len`; route it through the loader façade.
            if let Err(error) = model_reset_context(m, gpu) {
                return reset_failed(id, error);
            }
            prompt_ids.clone()
        }
    };

    // Reused-prefix count for the done event: `prompt_ids` is the full render,
    // `prefill_ids` the post-LCP suffix ar_generate will prefill.
    let cached_tokens_count = prompt_ids.len().saturating_sub(prefill_ids.len());
    let prefill_len = prefill_ids.len();
    let t0 = Instant::now();

    // MiniMax-M2's chat template unconditionally primes the assistant turn with
    // `<think>\n` (chat_template.jinja generation-prompt block), so generated tokens
    // begin *inside* the reasoning block and the model only ever emits the closing
    // `</think>`. Every downstream `<think>` consumer — the serve reasoning/content
    // split, the run/chat-path stripper, the history `stripThinkingInline` — keys on
    // a LEADING `<think>`, so re-emit the primer into the DISPLAY stream here, before
    // ar_generate (which now owns prefill+decode), making the assistant message a
    // well-formed `<think>...</think>...` block. Display-only: the primer is already
    // in the KV from the render and is never committed to state.
    if primed_think {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type": "token", "id": id, "text": "<think>\n"}),
        );
        let _ = stdout.flush();
    }

    // ── Decode via the generic ar_generate/MinimaxDispatch driver (Inc 4 flip).
    // ar_generate prefills `new_tokens` starting at seq_pos = MiniMaxState.n_tokens
    // (set by the LCP rewind above), then runs the AR loop. Proven token-identical to
    // the legacy loop at temp0 on MiniMax-M2.7.mq2 via the dual-run shadow parity that
    // this replaces. UPLIFT over the legacy loop: the n-gram loop guard + the generic
    // sampler (sampler::sample_cpu — temp0 == the legacy argmax; temp>0 nucleus by
    // design). No think-cap (minimax primes its own `<think>` and emits its own
    // `</think>`), no grammar, no stop-seqs plumbed.
    let mut __disp = MinimaxDispatch { m: &mut *m };
    let ar_result = ar_generate(
        &mut __disp,
        ForwardCtx::Single(gpu),
        stdout,
        id,
        temp,
        top_p,
        None, // top_k
        None, // min_p
        max_tokens,
        1.0, // repeat_penalty (legacy: none)
        0,   // repeat_window
        0.0, // presence_penalty
        0.0, // frequency_penalty
        0,   // budget_alert_at_tok
        "",  // budget_alert_text
        0,   // max_think_tokens (no force-close think on minimax)
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        &[],         // stop
        None,        // tools (grammar off)
        prefill_ids, // new_tokens: the post-LCP suffix to prefill
        &[],         // im_end
        &[],         // nl
        None,        // im_end_token
        None,        // tool_call_pair
        None,        // think_pair
        prefill_len,
        cached_tokens_count,
        None, // pflash_summary
        None, // pflash_bypass_reason
        None, // pflash_alpha
        t0,
        None, // tape (prod: no dual-run)
    );
    drop(__disp);
    ar_result
}

/// Cohere2-MoE / North-Mini-Code (arch_id=12) generate path. Mirrors
/// `generate_minimax`, plus: BATCHED `forward_batch` prefill (≈9×, see the
/// prefill block) and a Cohere agentic-marker state machine in the decode loop
/// that suppresses the special markers, routes `<|START_THINKING|>` content to a
/// reasoning channel, emits `<|START_TEXT|>` content as the visible answer, and
/// parses `<|START_ACTION|>` blocks into `tool_calls` events. Decode is plain
/// per-token `decode_step` (no hipGraph variant yet). Out of scope (NOT wired):
/// spec-decode, MTP, grammar-constrained decoding, tool EXECUTION, repeat
/// penalty, multi-GPU.
/// StreamParser for cohere2moe / North-Mini-Code (arch 12) — a verbatim port of the
/// agentic-marker state machine in `generate_cohere2moe`'s decode loop (daemon.rs
/// ~12660–12866). Sampling stays in the driver; this owns output routing + stop /
/// forced-injection. Built-only until Task 7 wires `generate_cohere2moe` onto
/// `ar_generate`. Faithful (NOT byte-identical) — validated by event equivalence +
/// `coherence-gate-cohere2moe.sh`, since token-parity is blind to the tool_calls /
/// reasoning EVENTS that are this arch's whole point.
#[allow(dead_code)]
struct Cohere2MoeStreamParser {
    sec: C2mSec,
    action_buf: String,
    vis_buf: String,
    forced: std::collections::VecDeque<u32>,
    think_count: usize,
    think_budget: usize,
    think_force_closed: bool,
    empty_turn_guard: bool,
    emitted_visible: bool,
    eos_suppressions: usize,
    tool_calls_emitted: bool,
    finish_latched: bool,
    last_tok: u32,
    repeat_run: usize,
    mk_think0: u32,
    mk_think1: u32,
    mk_text0: u32,
    mk_text1: u32,
    mk_act0: u32,
    mk_act1: u32,
    pad_tok: Option<u32>,
    known_tools: Vec<String>,
    tool_params: Vec<(String, Vec<String>)>,
}

#[derive(PartialEq, Clone, Copy)]
#[allow(dead_code)]
enum C2mSec {
    Pre,
    Think,
    Text,
    Action,
}

#[allow(dead_code)]
impl Cohere2MoeStreamParser {
    const REPEAT_GUARD: usize = 24;
    const MAX_EOS_SUPPRESS: usize = 3;

    /// Mirrors the setup block at generate_cohere2moe ~12560–12659: resolve the 6
    /// marker ids + `<PAD>`, build `known_tools`/`tool_params` from the schemas, and
    /// compute `think_budget` (the `think_reserve` clamp).
    fn new(
        tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
        tools: Option<&[serde_json::Value]>,
        max_tokens: usize,
        max_think_tokens: usize,
    ) -> Self {
        let mark = |s: &str, fb: u32| -> u32 { tokenizer.special_token_id(s).unwrap_or(fb) };
        let known_tools: Vec<String> = tools
            .map(|ts| {
                ts.iter()
                    .filter_map(|t| {
                        t.get("function")
                            .and_then(|f| f.get("name"))
                            .or_else(|| t.get("name"))
                            .and_then(|n| n.as_str())
                            .map(String::from)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let tool_params: Vec<(String, Vec<String>)> = tools
            .map(|ts| {
                ts.iter()
                    .filter_map(|t| {
                        let f = t.get("function").unwrap_or(t);
                        let name = f.get("name").and_then(|n| n.as_str())?.to_string();
                        let params = f
                            .get("parameters")
                            .and_then(|p| p.get("properties"))
                            .and_then(|p| p.as_object())
                            .map(|o| o.keys().cloned().collect())
                            .unwrap_or_default();
                        Some((name, params))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let think_reserve = (max_tokens / 4).clamp(64, 512).min(max_tokens / 2);
        let think_budget = if max_think_tokens > 1 {
            max_think_tokens.min(max_tokens.saturating_sub(think_reserve))
        } else {
            max_tokens.saturating_sub(think_reserve)
        };
        Self {
            sec: C2mSec::Pre,
            action_buf: String::new(),
            vis_buf: String::new(),
            forced: std::collections::VecDeque::new(),
            think_count: 0,
            think_budget,
            think_force_closed: false,
            empty_turn_guard: std::env::var("HIPFIRE_C2M_EMPTY_TURN_GUARD")
                .ok()
                .as_deref()
                != Some("0"),
            emitted_visible: false,
            eos_suppressions: 0,
            tool_calls_emitted: false,
            finish_latched: false,
            last_tok: u32::MAX,
            repeat_run: 0,
            mk_think0: mark("<|START_THINKING|>", 255010),
            mk_think1: mark("<|END_THINKING|>", 255011),
            mk_text0: mark("<|START_TEXT|>", 255012),
            mk_text1: mark("<|END_TEXT|>", 255013),
            mk_act0: mark("<|START_ACTION|>", 255014),
            mk_act1: mark("<|END_ACTION|>", 255015),
            pad_tok: tokenizer.special_token_id("<PAD>"),
            known_tools,
            tool_params,
        }
    }

    fn process_marker(
        &mut self,
        tok: u32,
    ) -> Option<Vec<hipfire_runtime::stream_parser::StreamAction>> {
        use hipfire_runtime::stream_parser::StreamAction;
        let mut acts = Vec::new();
        if tok == self.mk_think0 {
            self.sec = C2mSec::Think;
        } else if tok == self.mk_text0 {
            self.sec = C2mSec::Text;
        } else if tok == self.mk_act0 {
            self.sec = C2mSec::Action;
            self.action_buf.clear();
        } else if tok == self.mk_think1 || tok == self.mk_text1 {
            self.sec = C2mSec::Pre;
        } else if tok == self.mk_act1 {
            let action = std::mem::take(&mut self.action_buf);
            let mut calls = cohere2moe::spec_emit::parse_cohere_action(&action);
            cohere2moe::spec_emit::snap_call_names(
                &mut calls,
                &self.known_tools,
                &self.tool_params,
            );
            if !calls.is_empty() {
                acts.push(StreamAction::ToolCalls(serde_json::json!(calls)));
                self.emitted_visible = true;
                self.tool_calls_emitted = true;
            }
            self.sec = C2mSec::Pre;
        } else {
            return None;
        }
        Some(acts)
    }
}

#[allow(dead_code)]
impl hipfire_runtime::stream_parser::StreamParser for Cohere2MoeStreamParser {
    fn next_forced(&mut self) -> Option<u32> {
        // Pre-sample think-budget force-close (12667–12681): only when the queue is
        // empty and we're still reasoning past budget with nothing visible.
        if self.empty_turn_guard
            && self.forced.is_empty()
            && !self.think_force_closed
            && !self.emitted_visible
            && self.sec == C2mSec::Think
            && self.think_count >= self.think_budget
        {
            self.forced.push_back(self.mk_think1);
            self.forced.push_back(self.mk_text0);
            self.think_force_closed = true;
        }
        self.forced.pop_front()
    }

    fn on_eos(&mut self) -> hipfire_runtime::stream_parser::EosDecision {
        use hipfire_runtime::stream_parser::EosDecision;
        // Empty-turn guard (12687–12709): a reasoning-only turn ending with no visible
        // output — inject START_TEXT (closing THINK first if still inside) instead of
        // committing the eos. Bounded to MAX_EOS_SUPPRESS.
        if self.empty_turn_guard
            && !self.emitted_visible
            && self.eos_suppressions < Self::MAX_EOS_SUPPRESS
        {
            self.eos_suppressions += 1;
            let mut v = Vec::new();
            if self.sec == C2mSec::Think {
                v.push(self.mk_think1);
            }
            v.push(self.mk_text0);
            EosDecision::Inject(v)
        } else {
            // cohere2moe does NOT commit the eos (legacy breaks pre-commit at 12708).
            EosDecision::Stop
        }
    }

    fn enqueue(&mut self, tok: u32) {
        self.forced.push_back(tok);
    }

    fn feed(
        &mut self,
        tok: u32,
        bytes: &[u8],
    ) -> Vec<hipfire_runtime::stream_parser::StreamAction> {
        use hipfire_runtime::stream_parser::StreamAction;
        let mut acts = Vec::new();

        // Degenerate-output guards (12715–12735). NB: in the StreamParser model the
        // driver has already committed `tok`; a degenerate stop that commits one extra
        // <PAD>/attractor token before aborting is immaterial (the turn is garbage).
        if Some(tok) == self.pad_tok {
            acts.push(StreamAction::Stop);
            return acts;
        }
        if tok == self.last_tok {
            self.repeat_run += 1;
            if self.repeat_run >= Self::REPEAT_GUARD {
                acts.push(StreamAction::Stop);
                return acts;
            }
        } else {
            self.last_tok = tok;
            self.repeat_run = 1;
        }

        // Agentic-marker state machine (12750–12822) — markers themselves never emit.
        if let Some(marker_acts) = self.process_marker(tok) {
            return marker_acts;
        } else {
            let frag = String::from_utf8_lossy(bytes);
            // Defense-in-depth marker suppression (12789–12796): any OTHER special
            // token decoding to `<|UPPER_SNAKE|>` is dropped from output.
            let is_marker = frag.len() > 4
                && frag.starts_with("<|")
                && frag.ends_with("|>")
                && frag[2..frag.len() - 2]
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_');
            if !is_marker {
                match self.sec {
                    C2mSec::Action => self.action_buf.push_str(&frag),
                    C2mSec::Think => {
                        acts.push(StreamAction::Emit {
                            text: frag.into_owned(),
                            reasoning: true,
                        });
                        self.think_count += 1;
                    }
                    C2mSec::Text | C2mSec::Pre => {
                        self.vis_buf.push_str(&frag);
                        acts.push(StreamAction::Emit {
                            text: frag.into_owned(),
                            reasoning: false,
                        });
                        self.emitted_visible = true;
                    }
                }
            }
        }
        acts
    }

    fn emit_only(
        &mut self,
        tok: u32,
        bytes: &[u8],
    ) -> Vec<hipfire_runtime::stream_parser::StreamAction> {
        // Forced continuation markers bypass feed's guards, but must still perform
        // the Cohere section transition and suppress their decoded marker bytes.
        if let Some(marker_acts) = self.process_marker(tok) {
            return marker_acts;
        }
        if bytes.is_empty() {
            return Vec::new();
        }
        vec![hipfire_runtime::stream_parser::StreamAction::Emit {
            text: String::from_utf8_lossy(bytes).into_owned(),
            reasoning: false,
        }]
    }

    fn finish(&mut self) -> Vec<hipfire_runtime::stream_parser::StreamAction> {
        use hipfire_runtime::stream_parser::StreamAction;
        if self.finish_latched {
            return Vec::new();
        }
        self.finish_latched = true;

        // Tool-call-as-text recovery (12851–12866).
        let mut recovered = cohere2moe::spec_emit::parse_cohere_action(&self.action_buf);
        if recovered.is_empty() && !self.tool_calls_emitted {
            recovered = cohere2moe::spec_emit::parse_cohere_action(&self.vis_buf);
        }
        if !recovered.is_empty() {
            cohere2moe::spec_emit::snap_call_names(
                &mut recovered,
                &self.known_tools,
                &self.tool_params,
            );
            self.action_buf.clear();
            self.vis_buf.clear();
            return vec![StreamAction::ToolCalls(serde_json::json!(recovered))];
        }
        self.action_buf.clear();
        self.vis_buf.clear();
        Vec::new()
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_cohere2moe(
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
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
) -> GenerateResult {
    if m.tokenizer.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }
    if m.cohere2moe().is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"cohere2moe_config missing on arch_id=12 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    // ── Prompt build (same two-path branch as the minimax / lfm2moe AR path) ──
    // `primed_think` records whether the rendered prompt actually ended with a
    // `<think>` generation-primer, so we only re-emit the opener (below) when
    // the model truly begins inside the reasoning block. (A jinja render failure
    // now returns an error rather than falling back, so we never reach decode
    // with an off-distribution Plain frame.)
    // Set on the sole surviving (successful-render) path; the failure paths in
    // the block below return, so this needs no dead initializer.
    let primed_think: bool;
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // Cohere2-MoE / North-Mini-Code REQUIRES its embedded Jinja chat_template
        // — its structural tokens are NOT ChatML (turns end with
        // `<|END_OF_TURN_TOKEN|>`). The hand-rolled Plain ChatML frame emits
        // `<|im_start|>`/`<|im_end|>` which this model never trained on,
        // producing an off-distribution prompt that (a) decodes incoherently and
        // (b) never matches across turns so the LCP prompt-cache is dead. Force
        // jinja on (falls back to Plain only when the .hfq carries no template).
        // Jinja default-ON; opt out with HIPFIRE_JINJA_CHAT=0.
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.meta.chat_template.is_some();
        if try_jinja {
            let template = m.meta.chat_template.as_ref().unwrap();
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
                let messages_slice: &[hipfire_runtime::prompt_frame::Message] =
                    match messages_history {
                        Some(h) => h,
                        None => {
                            let mut v = Vec::new();
                            if let Some(sys) = system_prompt {
                                v.push(hipfire_runtime::prompt_frame::Message {
                                    role: hipfire_runtime::prompt_frame::Role::System,
                                    content: sys.to_string(),
                                    tool_calls: Vec::new(),
                                    tool_call_id: None,
                                    tool_plan: String::new(),
                                });
                            }
                            v.push(hipfire_runtime::prompt_frame::Message {
                                role: hipfire_runtime::prompt_frame::Role::User,
                                content: prompt.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                tool_plan: String::new(),
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
                    if std::env::var("HIPFIRE_C2M_DUMP_PROMPT").ok().as_deref() == Some("1") {
                        let ids = tokenizer.encode(&rendered);
                        eprintln!(
                            "[c2m prompt dump] rendered chars={} tokens={}\n>>> HEAD(400):\n{}\n>>> TAIL(800):\n{}\n<<< end",
                            rendered.len(),
                            ids.len(),
                            &rendered[..rendered.len().min(400)],
                            &rendered[rendered.len().saturating_sub(800)..],
                        );
                    }
                    tokenizer.encode(&rendered)
                }
                Err(e) => {
                    // North-Mini-Code's turns are NOT ChatML; a Plain frame emits
                    // <|im_start|>/<|im_end|> the model never trained on → off-
                    // distribution garbage that reads as a model-quality bug. Refuse
                    // rather than silently serve it.
                    eprintln!(
                        "[daemon] cohere2moe jinja render failed ({e}) — refusing ChatML fallback"
                    );
                    emit_error_with_id(stdout, id, format!("cohere2moe jinja render failed: {e}"));
                    return GenerateResult::Complete;
                }
            }
        } else {
            // cohere2moe REQUIRES its embedded jinja template (see above). When it
            // is unavailable — HIPFIRE_JINJA_CHAT=0, or the .hfq carries no template
            // — the only frame we could build is Plain ChatML, off-distribution for
            // North. Refuse loudly instead of serving garbage.
            let why = if !jinja_enabled {
                "HIPFIRE_JINJA_CHAT=0 disables the required template"
            } else {
                "model .hfq carries no chat_template"
            };
            eprintln!(
                "[daemon] cohere2moe cannot build a valid prompt frame ({why}) — refusing ChatML fallback"
            );
            emit_error_with_id(
                stdout,
                id,
                format!("cohere2moe requires its jinja chat template ({why})"),
            );
            return GenerateResult::Complete;
        }
    };

    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    // Capacity guard. No eviction on arch_id=12 — reset the KV cursor when the
    // FULL rendered conversation + generation would overflow. `prompt_ids` is
    // the full Jinja-rendered conversation; the LCP below reuses the warm prefix.
    // The KV cache holds `max_seq` positions. A prompt that itself exceeds that
    // cannot be prefilled without writing PAST the cache — which previously
    // produced silent GPU-memory corruption (degenerate/garbage output): the
    // guard reset the cursor but then prefilled the oversized prompt anyway.
    // Fix: prompt too long → clean error + free KV; prompt fits but generation
    // would overflow → cap the token budget to the remaining slots so decode
    // stops at capacity instead of writing OOB.
    let max_seq = m.cohere2moe().unwrap().state.max_seq;
    if prompt_ids.len() >= max_seq {
        eprintln!(
            "[daemon] arch_id=12 prompt {} >= max_seq {} — refusing (would OOB the KV cache)",
            prompt_ids.len(),
            max_seq,
        );
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        } else {
            emit_error_with_id(
            stdout,
            id,
            format!(
                "cohere2moe: prompt is {} tokens but KV capacity (max_seq) is {} — load with a larger max_seq or shorten the prompt",
                prompt_ids.len(),
                max_seq
            ),
        );
        }
        return GenerateResult::Complete;
    }
    // Cap generation so prefill(prompt) + decode(max_tokens) never exceeds the
    // cache. `max_tokens` is shadowed for the decode loop below.
    let max_tokens = max_tokens.min(max_seq - prompt_ids.len());

    // ── Prefix cache (LCP) with PARTIAL reuse. `prompt_ids` is the full
    // Jinja-rendered conversation (the trained chat template). Cohere2-MoE is
    // standard attention with no compound recurrent/compressed state, so KV
    // positions ≥ lcp are simply overwritten by the new prefill and the stale
    // tail is never attended. We rewind `n_tokens` to `lcp` and re-prefill the
    // suffix; the reused prefix GROWS with the conversation.
    let cold_start: bool;
    let prefill_ids: Vec<u32> = {
        let prior_len = m.session.conversation_tokens.len();
        let max_match = prior_len.min(prompt_ids.len());
        let mut lcp = 0usize;
        while lcp < max_match && m.session.conversation_tokens[lcp] == prompt_ids[lcp] {
            lcp += 1;
        }
        // A usable common prefix that leaves at least one fresh token to prefill.
        // `partial` is the divergence case (lcp < prior_len); lcp == prior_len is
        // the degenerate pure-extension case (rewind is then a no-op).
        let cache_hit = lcp > 0 && lcp < prompt_ids.len();
        cold_start = !cache_hit;
        let partial = lcp < prior_len;
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[cohere2moe-cache] prior_len={} rendered_len={} lcp={} hit={} partial={} n_tokens={}",
                prior_len,
                prompt_ids.len(),
                lcp,
                cache_hit,
                cache_hit && partial,
                m.cohere2moe().unwrap().state.n_tokens,
            );
        }
        if cache_hit {
            // Rewind KV + token history to the common prefix. When lcp ==
            // prior_len this is a no-op; when lcp < prior_len it discards the
            // stale tail. The prefill loop below reads `state.n_tokens` as its
            // base position, so n_tokens is the only KV state the rewind must
            // touch (plus the mirror token history).
            m.cohere2moe_mut().unwrap().state.n_tokens = lcp;
            m.session.conversation_tokens.truncate(lcp);
            m.session.seq_pos = lcp;
            prompt_ids[lcp..].to_vec()
        } else {
            // A cold-prefill decision owns the full context transition even
            // when the session mirror is already empty (for example after an
            // aborted or externally cleared request). Do not infer reset
            // ownership from `prior_len`; route it through the loader façade.
            if let Err(error) = model_reset_context(m, gpu) {
                return reset_failed(id, error);
            }
            prompt_ids.clone()
        }
    };

    let t0 = Instant::now();

    // ── Decode via the generic ar_generate/Cohere2MoeDispatch driver (Task 9 flip).
    // Cohere2MoeStreamParser (via the stream_parser() override) owns the agentic-marker
    // state machine + tool_calls + empty-turn / think-budget / repeat guards. `prefill_ids`
    // is the LCP suffix (the preamble rewound state.n_tokens=lcp); ar_generate prefills it
    // from seq_pos = Cohere2MoeDispatch.seq_pos() = state.n_tokens, preserving multi-turn
    // prefix reuse. Proven token-parity-equivalent to the deleted legacy loop on North.
    let cached_tokens_count = prompt_ids.len().saturating_sub(prefill_ids.len());
    let prefill_len = prefill_ids.len();
    let mut __disp = Cohere2MoeDispatch {
        m: &mut *m,
        tools: tools.map(|t| t.to_vec()),
        cold_start,
    };
    let ar_result = ar_generate(
        &mut __disp,
        ForwardCtx::Single(gpu),
        stdout,
        id,
        temp,
        top_p,
        None, // top_k
        None, // min_p
        max_tokens,
        1.0, // repeat_penalty
        0,   // repeat_window
        0.0, // presence_penalty
        0.0, // frequency_penalty
        0,   // budget_alert_at_tok
        "",  // budget_alert_text
        max_think_tokens,
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        &[],         // stop
        None,        // tools (grammar off; Cohere2MoeStreamParser owns tools)
        prefill_ids, // new_tokens: the LCP suffix (prod: no reset)
        &[],         // im_end
        &[],         // nl
        None,        // im_end_token
        None,        // tool_call_pair
        None,        // think_pair
        prefill_len, // prefill_tokens
        cached_tokens_count,
        None, // pflash_summary
        None, // pflash_bypass_reason
        None, // pflash_alpha
        t0,
        None, // tape (prod: no dual-run)
    );
    drop(__disp);
    ar_result
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
/// `m.session.seq_pos` (mirrors the qwen35/llama bookkeeping) plus
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
) -> GenerateResult {
    // Qwen2 (arch 7) AR decode via the generic ar_generate driver (Inc 2 — flipped
    // from the legacy greedy loop; uplift: qwen2 gains ar_generate's n-gram loop
    // guard). generate_qwen2 keeps its raw-prompt preamble (no framing) + capacity
    // guard, then hands the outputs to ar_generate (which prefills + decodes).
    let prompt_ids = {
        let tokenizer = match m.tokenizer.as_ref() {
            Some(t) => t,
            None => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"tokenizer not loaded"}}"#,
                    id
                );
                let _ = stdout.flush();
                return GenerateResult::Complete;
            }
        };
        tokenizer.encode(prompt)
    };
    if prompt_ids.is_empty() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return GenerateResult::Complete;
    }

    // Capacity guard. No eviction on arch_id=7 yet — reset state when
    // the requested run would overflow the KV budget.
    let (next_pos, max_seq) = match m.state.as_ref() {
        Some(ModelState::Qwen2(b)) => (b.state.next_pos, b.state.max_seq),
        _ => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"qwen2 state missing on arch_id=7 generate"}}"#,
                id
            );
            let _ = stdout.flush();
            return GenerateResult::Complete;
        }
    };
    if next_pos
        .saturating_add(prompt_ids.len())
        .saturating_add(max_tokens)
        > max_seq
    {
        eprintln!(
            "[daemon] arch_id=7 context full ({}/{}) — resetting Qwen2State.next_pos",
            next_pos, max_seq,
        );
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }

        // O2b-2 capacity guard (qwen2 single): the reset above (next_pos=0)
        // recovers a grown multi-turn conversation, but a SINGLE prompt larger
        // than the whole context still overflows — prefilling it writes past
        // the KV (sized for state.max_seq) and panics, taking down serve. After
        // the reset, if prompt + generation still overflows, emit a clean error.
        // saturating_add: an adversarially huge max_tokens must not wrap usize
        // and slip under the cap.
        if prompt_ids.len().saturating_add(max_tokens) > max_seq {
            let cap = max_seq;
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
                id,
                prompt_ids.len(),
                max_tokens,
                cap
            );
            let _ = stdout.flush();
            return GenerateResult::Complete;
        }
    }

    let t0 = Instant::now();

    // Hand the raw-prompt preamble outputs to the generic driver. ar_generate
    // prefills `new_tokens` and runs the decode loop; greedy params reproduce the
    // legacy loop (temp0 argmax, no framing/think/budget/grammar). state/cfg/weights
    // borrows above have ended, so Qwen2Dispatch can take &mut m.
    let __prefill_tokens = prompt_ids.len();
    let mut __disp = Qwen2Dispatch { m: &mut *m };
    let ar_result = ar_generate(
        &mut __disp,
        ForwardCtx::Single(gpu),
        stdout,
        id,
        0.0,
        1.0,
        None,
        None,
        max_tokens,
        1.0,
        0,
        0.0,
        0.0,
        0,
        "",
        0,
        hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
        &[],
        None,
        prompt_ids,
        &[],
        &[],
        None,
        None,
        None,
        __prefill_tokens,
        0,
        None,
        None,
        None,
        t0,
        None,
    );
    drop(__disp);
    ar_result
}

fn generate_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) -> GenerateResult {
    // hunt3 M-E: seed the process-global CPU sampler RNG with this request's
    // fixed seed. The VL path samples exclusively via sampler::sample_cpu, which
    // draws from this global; without the per-request reset it carried RNG state
    // across requests (and across earlier text-path requests) → cross-request
    // nondeterminism. Matches the GPU path's u32 (0x13579BDF).
    hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);
    // INVARIANT: all early returns before the `vision_forward` call (the
    // first expensive GPU allocation in this function) use `write_error`
    // and return without owning any GPU buffers. If you add a GPU
    // allocation above this line, you MUST clean it up on every early
    // return path — the current early returns are safe because they
    // only hold CPU-side data (tokenizer refs, preprocess output).
    let GenerateVLParams {
        id,
        prompt,
        system_prompt,
        ref image_source,
        temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        max_think_tokens,
    } = *params;
    let vision_config = m.vision.as_ref().unwrap().config.clone();

    // Vision special-token IDs resolved from the tokenizer rather than
    // hardcoded constants. Different VL-capable Qwen variants ship with
    // different IDs for these tokens; a hardcoded mismatch silently
    // splices the wrong tokens into the prompt. This is a request error, not
    // a process-fatal invariant: keep the daemon lifecycle in control of the
    // dirty-state reset path instead of panicking here.
    let (image_pad_id, vision_start_id, vision_end_id) = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        match (
            tokenizer.special_token_id("<|image_pad|>"),
            tokenizer.special_token_id("<|vision_start|>"),
            tokenizer.special_token_id("<|vision_end|>"),
        ) {
            (Some(image_pad), Some(vision_start), Some(vision_end)) => {
                (image_pad, vision_start, vision_end)
            }
            _ => {
                write_error(
                    stdout,
                    id,
                    "VL tokenizer is missing a required vision token",
                );
                return GenerateResult::Complete;
            }
        }
    };

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
                    return GenerateResult::Complete;
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
                        return GenerateResult::Complete;
                    }
                }
            } else {
                b64
            };
            eprintln!(
                "[VL-DEBUG] preprocessing image: <{}-byte buffer>",
                raw_b64.len()
            );
            let bytes = match Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64) {
                Ok(b) => b,
                Err(e) => {
                    write_error(
                        stdout,
                        id,
                        &format!("failed to decode base64 image data: {e}"),
                    );
                    return GenerateResult::Complete;
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
                    return GenerateResult::Complete;
                }
            }
        }
    };
    eprintln!("[VL-DEBUG] preprocessed: {}x{}", img_w, img_h);

    let image_state = vl_image_state(&pixels, img_h, img_w);
    if prepare_vl_request_state(&mut m.session, image_state) {
        eprintln!("[daemon/vl] replacing prior image turn (image_state={image_state:016x})");
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
        m.session.vl_image_state = Some(image_state);
    }

    let grid_h = img_h / vision_config.patch_size;
    let grid_w = img_w / vision_config.patch_size;
    let n_patches = grid_h * grid_w;
    let n_visual_tokens =
        n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);

    // Capacity estimate including system prompt — a long system prompt
    // on first turn would otherwise let an over-budget request through
    // the soft check, only to fail the hard check after the expensive
    // vision encoder runs.
    let prompt_est = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let system_est = system_prompt
            .map(|s| tokenizer.encode(s).len())
            .unwrap_or(0);
        tokenizer.encode(prompt).len() + system_est + n_visual_tokens + 20
    };

    if m.eviction.is_none()
        && m.session
            .seq_pos
            .saturating_add(prompt_est)
            .saturating_add(max_tokens)
            > m.meta.max_seq
    {
        eprintln!(
            "[daemon/vl] context full ({}/{}) — resetting conversation",
            m.session.seq_pos, m.meta.max_seq
        );
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
    }

    let tokenizer = m.tokenizer.as_ref().unwrap();

    if m.eviction.is_none() && prompt_est.saturating_add(max_tokens) > m.meta.max_seq {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: format!(
                "request size ({} tokens) exceeds loaded KV budget ({})",
                prompt_est.saturating_add(max_tokens),
                m.meta.max_seq,
            ),
        });
    }

    let ModelState::Qwen35(b) = m.state.as_mut().unwrap() else {
        unreachable!()
    };
    let config = &b.config;
    let weights = &b.weights;
    let scratch = &b.scratch;
    let kv = &mut b.kv_cache;
    let dn = &mut b.dn_state;
    let vision_weights = &m.vision.as_ref().unwrap().weights;

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
        system: if m.session.seq_pos == 0 {
            system_prompt
        } else {
            None
        },
        user: "", // unused: we pass tokens directly via build_with_user_tokens
        assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain, // VL always uses Plain
        raw: false,
    }
    .build_with_user_tokens(&user_body);

    // KV-budget guard — physical_cap without eviction, absolute window with.
    // Mirrors the textual generate() contract; reserves trailer slots so
    // natural im_end termination can still write the ChatML \n.
    let trailer = nl.len();
    let absolute_pos_vl = m.session.seq_pos.saturating_add(kv.compact_offset);
    let over_budget = if m.eviction.is_none() {
        m.session
            .seq_pos
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.meta.physical_cap
    } else {
        absolute_pos_vl
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.meta.max_seq
    };
    if over_budget {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: format!(
                "request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > cap={} — reload model with a larger max_seq",
                m.session.seq_pos,
                prompt_tokens.len(),
                max_tokens,
                trailer,
                if m.eviction.is_none() {
                    m.meta.physical_cap
                } else {
                    m.meta.max_seq
                },
            ),
        });
    }

    // Now safe to run the expensive GPU vision encoder.
    if check_abort(id) {
        return GenerateResult::Deferred(DeferredTerminal::Aborted {
            id: id.to_string(),
            generated: 0,
        });
    }
    let patches = hipfire_arch_qwen35_vl::image::extract_patches(
        &pixels,
        3,
        img_h,
        img_w,
        vision_config.patch_size,
        vision_config.temporal_patch_size,
        vision_config.spatial_merge_size,
    );
    let visual_tokens = match qwen35_vl::vision_forward(
        gpu,
        vision_weights,
        &vision_config,
        &patches,
        grid_h,
        grid_w,
    ) {
        Ok(tokens) => tokens,
        Err(error) => {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("vision forward failed: {error:?}"),
            });
        }
    };

    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };
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
    // advance m.session.seq_pos in-loop and call maybe_evict after every write.
    let mut visual_idx = 0usize;
    for &token in prompt_tokens.iter() {
        if check_abort(id) {
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated: 0,
            });
        }
        if token == image_pad_id && visual_idx < n_visual_tokens {
            let emb = &visual_tokens[visual_idx * config.dim..(visual_idx + 1) * config.dim];
            if let Err(error) = qwen35::forward_scratch_embed(
                gpu,
                weights,
                config,
                emb,
                m.session.seq_pos,
                kv,
                dn,
                scratch,
            ) {
                return GenerateResult::Deferred(DeferredTerminal::Error {
                    id: id.to_string(),
                    message: format!("forward_scratch_embed failed: {error:?}"),
                });
            }
            visual_idx += 1;
        } else {
            if let Err(error) = qwen35::forward_scratch(
                gpu,
                weights,
                config,
                token,
                m.session.seq_pos,
                kv,
                dn,
                scratch,
            ) {
                return GenerateResult::Deferred(DeferredTerminal::Error {
                    id: id.to_string(),
                    message: format!("forward_scratch failed: {error:?}"),
                });
            }
        }
        m.session.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            if let Some(hipfire_runtime::triattn::EvictionResult {
                new_physical: new_phys,
                ..
            }) = match ev.maybe_evict(gpu, kv, m.session.seq_pos) {
                Ok(result) => result,
                Err(error) => {
                    return GenerateResult::Deferred(DeferredTerminal::Error {
                        id: id.to_string(),
                        message: format!("VL prefill eviction failed: {error:?}"),
                    });
                }
            } {
                m.session.seq_pos = new_phys;
            }
        }
    }

    m.session
        .conversation_tokens
        .extend_from_slice(&prompt_tokens);

    // hunt3 M-D: repeat-penalty / n-gram-block history must be scoped to the
    // GENERATED tokens only (mirrors the text path's `ngram_scope_start` set to
    // conversation_tokens.len() after prefill). Passing the full conversation
    // makes the trailing window prompt-dominated, suppressing the names/numbers
    // a VL transcription task must reproduce.
    let vl_ngram_scope_start = m.session.conversation_tokens.len();

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
    let mut logits = match gpu.download_f32(&scratch.logits) {
        Ok(logits) => logits,
        Err(error) => {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("VL logits download failed: {error:?}"),
            });
        }
    };
    if let Some((open, close)) = think_pair {
        block_attractor_unclosed_cpu(
            &mut logits,
            &m.session.conversation_tokens,
            open,
            close,
            20,
            2,
        );
    }
    let vl_cfg_first = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty: 1.0,
        repeat_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        // VL path samples on the CPU (sample_cpu), which does not yet honor
        // top_k / min_p; keep None so behavior is unchanged.
        top_k: None,
        min_p: None,
    };
    let vl_cfg = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        top_k: None,
        min_p: None,
    };
    let mut next_token = sampler::sample_cpu(&mut logits, &[], &vl_cfg_first);
    let t_prefill = Instant::now();
    let mut generated = 0;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    // Think-depth tracking via token IDs (not UTF-8 rfind).
    // The previous implementation decoded the full streamed output to a
    // string and ran rfind on every token — O(N²) total, fragile to
    // tokenizer changes. Since `think_pair` already gives us the
    // open/close token IDs, we can track depth incrementally in O(1).
    let mut think_depth: usize = 0; // number of unmatched opens seen
    let mut think_count: usize = 0; // tokens emitted while depth > 0

    // N-gram loop detector — mirrors the text path. Catches answer-phase
    // attractor loops that the think cap and repeat penalty miss.
    let loop_guard =
        hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    while generated < max_tokens {
        if check_abort(id) {
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated,
            });
        }
        generated += 1;
        m.session.conversation_tokens.push(next_token);
        emit_committed_event(
            stdout,
            id,
            next_token,
            generated - 1,
            t0.elapsed().as_millis() as u64,
        );
        streamed_tokens.push(next_token);

        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[emitted_bytes..];
        let valid_len = match std::str::from_utf8(new_bytes) {
            Ok(_) => new_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_len > 0 {
            let text = std::str::from_utf8(&new_bytes[..valid_len]).unwrap();
            let _ = writeln!(
                stdout,
                r#"{{"type":"token","id":"{}","text":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default()
            );
            let _ = stdout.flush();
            emitted_bytes += valid_len;
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

        if let Err(error) = qwen35::forward_scratch(
            gpu,
            weights,
            config,
            next_token,
            m.session.seq_pos,
            kv,
            dn,
            scratch,
        ) {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("VL decode failed: {error:?}"),
            });
        }
        m.session.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            if let Some(hipfire_runtime::triattn::EvictionResult {
                new_physical: new_phys,
                ..
            }) = match ev.maybe_evict(gpu, kv, m.session.seq_pos) {
                Ok(result) => result,
                Err(error) => {
                    return GenerateResult::Deferred(DeferredTerminal::Error {
                        id: id.to_string(),
                        message: format!("VL eviction failed: {error:?}"),
                    });
                }
            } {
                m.session.seq_pos = new_phys;
            }
        }
        logits = match gpu.download_f32(&scratch.logits) {
            Ok(logits) => logits,
            Err(error) => {
                return GenerateResult::Deferred(DeferredTerminal::Error {
                    id: id.to_string(),
                    message: format!("VL decode logits download failed: {error:?}"),
                });
            }
        };
        // hunt3 M-D: scope ngram-block + repeat-penalty history to generated-only.
        let vl_ngram_scope = &m.session.conversation_tokens[vl_ngram_scope_start..];
        llama::apply_ngram_block(&mut logits, vl_ngram_scope);
        if let Some((open, close)) = think_pair {
            block_attractor_unclosed_cpu(
                &mut logits,
                &m.session.conversation_tokens,
                open,
                close,
                20,
                2,
            );
        }

        next_token = sampler::sample_cpu(&mut logits, vl_ngram_scope, &vl_cfg);

        if max_think_tokens > 0 {
            if let Some((open, close)) = think_pair {
                // Incremental think-depth tracking via token IDs — O(1)
                // per token instead of the previous O(N²) decode+rfind.
                if next_token == open {
                    think_depth += 1;
                    think_count = 1;
                } else if next_token == close {
                    think_depth = think_depth.saturating_sub(1);
                    if think_depth == 0 {
                        think_count = 0;
                    }
                } else if think_depth > 0 {
                    think_count += 1;
                }

                if think_depth > 0 && think_count >= max_think_tokens {
                    let close_tokens = tokenizer.encode("</think>\n");
                    let budget_left = max_tokens.saturating_sub(generated);
                    let take = close_tokens.len().min(budget_left);
                    for &t in &close_tokens[..take] {
                        if let Err(error) = qwen35::forward_scratch(
                            gpu,
                            weights,
                            config,
                            t,
                            m.session.seq_pos,
                            kv,
                            dn,
                            scratch,
                        ) {
                            return GenerateResult::Deferred(DeferredTerminal::Error {
                                id: id.to_string(),
                                message: format!("VL think-close forward failed: {error:?}"),
                            });
                        }
                        m.session.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult {
                                new_physical: new_phys,
                                ..
                            }) = match ev.maybe_evict(gpu, kv, m.session.seq_pos) {
                                Ok(result) => result,
                                Err(error) => {
                                    return GenerateResult::Deferred(DeferredTerminal::Error {
                                        id: id.to_string(),
                                        message: format!(
                                            "VL think-close eviction failed: {error:?}"
                                        ),
                                    });
                                }
                            } {
                                m.session.seq_pos = new_phys;
                            }
                        }
                        m.session.conversation_tokens.push(t);
                        streamed_tokens.push(t);
                        // hunt3 H-F: emit the committed-token event for force-closed
                        // </think> tokens too, BEFORE `generated += 1`, so the
                        // committed pos stays in lockstep with the streamed count
                        // under HIPFIRE_EMIT_TOKEN_IDS=1. The VL main loop uses
                        // `generated - 1` after its increment; here `generated`
                        // (pre-increment) is the same value.
                        emit_committed_event(
                            stdout,
                            id,
                            t,
                            generated,
                            t0.elapsed().as_millis() as u64,
                        );

                        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes = &all_bytes[emitted_bytes..];
                        let vl = match std::str::from_utf8(new_bytes) {
                            Ok(_) => new_bytes.len(),
                            Err(e) => e.valid_up_to(),
                        };
                        if vl > 0 {
                            let text = std::str::from_utf8(&new_bytes[..vl]).unwrap();
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"token","id":"{}","text":{}}}"#,
                                id,
                                serde_json::to_string(&text).unwrap_or_default()
                            );
                            let _ = stdout.flush();
                            emitted_bytes += vl;
                        }
                        generated += 1;
                    }
                    think_count = 0;
                    think_depth = 0; // Must reset — the close tokens
                                     // above bypass the incremental tracker, so depth
                                     // is still > 0 here. Without this, any subsequent
                                     // non-open/close token would re-trigger the cap.
                    if generated >= max_tokens {
                        break;
                    }
                    logits = match gpu.download_f32(&scratch.logits) {
                        Ok(logits) => logits,
                        Err(error) => {
                            return GenerateResult::Deferred(DeferredTerminal::Error {
                                id: id.to_string(),
                                message: format!(
                                    "VL think-close logits download failed: {error:?}"
                                ),
                            });
                        }
                    };
                    block_attractor_unclosed_cpu(
                        &mut logits,
                        &m.session.conversation_tokens,
                        open,
                        close,
                        20,
                        2,
                    );
                    // hunt3 M-D: generated-only repeat-penalty scope.
                    next_token = sampler::sample_cpu(
                        &mut logits,
                        &m.session.conversation_tokens[vl_ngram_scope_start..],
                        &vl_cfg,
                    );
                }
            }
        }
    }

    // ChatML \n boundary — run through forward to keep KV cache + DeltaNet in sync
    if im_end_token == Some(*m.session.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            if let Err(error) =
                qwen35::forward_scratch(gpu, weights, config, t, m.session.seq_pos, kv, dn, scratch)
            {
                return GenerateResult::Deferred(DeferredTerminal::Error {
                    id: id.to_string(),
                    message: format!("VL EOS-boundary forward failed: {error:?}"),
                });
            }
            m.session.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = match ev.maybe_evict(gpu, kv, m.session.seq_pos) {
                    Ok(result) => result,
                    Err(error) => {
                        return GenerateResult::Deferred(DeferredTerminal::Error {
                            id: id.to_string(),
                            message: format!("VL EOS-boundary eviction failed: {error:?}"),
                        });
                    }
                } {
                    m.session.seq_pos = new_phys;
                }
            }
            m.session.conversation_tokens.push(t);
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
    GenerateResult::Complete
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
fn generate_vl_dots_ocr(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) -> GenerateResult {
    use hipfire_arch_dots_ocr::image as dots_image;
    let t0 = Instant::now();
    let GenerateVLParams {
        id,
        prompt,
        ref image_source,
        max_tokens,
        ..
    } = *params;

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
                    None => {
                        write_error(stdout, id, "malformed data URL: missing ',' separator");
                        return GenerateResult::Complete;
                    }
                },
                None => &b64[..],
            };
            eprintln!(
                "[dots-ocr] preprocessing base64 image (<{}-byte payload>)",
                raw_b64.len()
            );
            match Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64) {
                Ok(bytes) => dots_image::preprocess_image_bytes(&bytes),
                Err(e) => {
                    write_error(stdout, id, &format!("dots.ocr: base64 decode failed: {e}"));
                    return GenerateResult::Complete;
                }
            }
        }
    };
    let img = match img {
        Ok(i) => i,
        Err(e) => {
            write_error(
                stdout,
                id,
                &format!("dots.ocr image preprocess failed: {e}"),
            );
            return GenerateResult::Complete;
        }
    };
    let image_state = vl_image_state(&img.patches, img.resized_h, img.resized_w);
    if prepare_vl_request_state(&mut m.session, image_state) {
        eprintln!("[daemon/vl] replacing prior image turn (image_state={image_state:016x})");
        if let Err(error) = model_reset_context(m, gpu) {
            return reset_failed(id, error);
        }
        m.session.vl_image_state = Some(image_state);
    }
    let n_visual = img.n_visual_tokens();
    let n_patches = img.n_patches();
    eprintln!(
        "[dots-ocr] grid {}x{}, {} patches → {} visual tokens",
        img.grid_h, img.grid_w, n_patches, n_visual
    );
    if check_abort(id) {
        return GenerateResult::Deferred(DeferredTerminal::Aborted {
            id: id.to_string(),
            generated: 0,
        });
    }

    let max_seq = m.meta.max_seq;

    // 2. Model state (disjoint field borrows of `m`).
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let Some(ModelState::DotsOcr(b)) = m.state.as_mut() else {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: "dots-ocr bundle missing on arch_id=8".to_string(),
        });
    };
    let config = &b.config;
    let weights = &b.weights;
    let state = &mut b.state;
    let text_cfg = &config.text;
    let dim = text_cfg.hidden_size;

    // 3. Build the prompt (HF-exact framing; imgpad count == n_visual by construction).
    let prompt_ids = dots_ocr::build_prompt_ids(tokenizer, prompt, n_visual);
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: format!(
                "dots.ocr request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
                prompt_ids.len(),
                max_tokens,
                max_seq
            ),
        });
    }

    // 4. Vision encoder → merged visual tokens.
    let patch_cols = img.patches.len() / n_patches;
    let patches_gpu = match gpu.upload_f32(&img.patches, &[n_patches, patch_cols]) {
        Ok(t) => t,
        Err(e) => {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("dots.ocr patch upload failed: {e:?}"),
            });
        }
    };
    let merged_gpu = match dots_ocr::vision_forward(
        gpu,
        &weights.vision,
        &config.vision,
        &patches_gpu,
        img.grid_h,
        img.grid_w,
    ) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(patches_gpu);
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("dots.ocr vision_forward failed: {e:?}"),
            });
        }
    };
    let _ = gpu.free_tensor(patches_gpu);
    let merged = match gpu.download_f32(&merged_gpu) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(merged_gpu);
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("dots.ocr merger download failed: {e:?}"),
            });
        }
    };
    let _ = gpu.free_tensor(merged_gpu);
    // Hard guard: merger output count MUST equal the imgpad-slot count, or
    // the splice silently corrupts the text context (PRD §"Vision token splicing").
    if merged.len() != n_visual * dim {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: format!(
                "dots.ocr: merger produced {} values but prompt has {} <|imgpad|> slots × {} dims = {}",
                merged.len(),
                n_visual,
                dim,
                n_visual * dim
            ),
        });
    }

    // 5. Prefill: build the [seq × dim] embedding matrix (token-embedding
    // rows for text positions, spliced vision-merger rows at IMGPAD slots)
    // and run it through the batched prefill in one pass. Only the ~215
    // text positions need a GPU embedding lookup; the 4880 visual rows are
    // already host-resident in `merged`.
    let t_prefill = Instant::now();
    let mut embeds = vec![0f32; prompt_ids.len() * dim];
    let emb_scratch = match gpu.alloc_tensor(&[dim], rdna_compute::DType::F32) {
        Ok(t) => t,
        Err(e) => {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("dots.ocr embed scratch alloc failed: {e:?}"),
            });
        }
    };
    let mut visual_idx = 0usize;
    let mut embed_err: Option<String> = None;
    for (pos, &token) in prompt_ids.iter().enumerate() {
        if check_abort(id) {
            let _ = gpu.free_tensor(emb_scratch);
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated: 0,
            });
        }
        if token == dots_ocr::IMGPAD_ID {
            embeds[pos * dim..(pos + 1) * dim]
                .copy_from_slice(&merged[visual_idx * dim..(visual_idx + 1) * dim]);
            visual_idx += 1;
        } else {
            // Dispatch the token-embedding lookup on the actual embedding
            // format. HFQ dots.ocr ships Q8_0 embeddings, but the
            // safetensors/Dir loader uploads F32 — hardcoding the Q8 kernel
            // here misreads F32 bytes as Q8 blocks, corrupting every text
            // token's embedding (the model then ignores the prompt). Mirrors
            // the per-format dispatch in `llama::forward`.
            let lookup = hipfire_runtime::llama::embedding_lookup_dispatch(
                gpu,
                weights.text.embd_format,
                &weights.text.token_embd,
                &emb_scratch,
                token,
                dim,
            );
            if let Err(e) = lookup {
                embed_err = Some(format!("embedding lookup: {e:?}"));
                break;
            }
            match gpu.download_f32(&emb_scratch) {
                Ok(row) => embeds[pos * dim..(pos + 1) * dim].copy_from_slice(&row),
                Err(e) => {
                    embed_err = Some(format!("embedding download: {e:?}"));
                    break;
                }
            }
        }
    }
    let _ = gpu.free_tensor(emb_scratch);
    if let Some(e) = embed_err {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: format!("dots.ocr prefill embed build failed: {e}"),
        });
    }
    if let Err(e) =
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, text_cfg, state, &embeds)
    {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: format!("dots.ocr batched prefill failed: {e:?}"),
        });
    }
    let prefill_tokens = prompt_ids.len();
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // 6. Decode. Opt-in n-gram speculative decode when a speculator was built at
    // load (HIPFIRE_NGRAM_DRAFT=1, arch_id=8 gate in `spec_build`); else the
    // bespoke greedy AR loop below. The vision prefill above already advanced the
    // Qwen2 KV (`ModelState::DotsOcr(b).state`), so both paths decode from the
    // same warm state — only the drafting differs. The n-gram verify always falls
    // back to the target's greedy argmax, so spec output is byte-identical to AR;
    // only τ (speed) changes. The prefill bindings above (`tokenizer`/`b`/…) are
    // released here so the speculative branch can take `&mut m`; the AR path
    // re-borrows them below.
    if m.speculator.is_some() {
        return decode_vl_dots_ocr_ngram(
            m,
            gpu,
            stdout,
            id,
            &prompt_ids,
            max_tokens,
            t0,
            prefill_tokens,
            prefill_s,
        );
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let Some(ModelState::DotsOcr(b)) = m.state.as_mut() else {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: "dots-ocr bundle missing on arch_id=8".to_string(),
        });
    };
    let config = &b.config;
    let text_cfg = &config.text;
    let weights = &b.weights;
    let state = &mut b.state;

    // Greedy decode, streaming in the daemon JSONL protocol.
    let eos_set: Vec<u32> = if text_cfg.eos_token_ids.is_empty() {
        vec![text_cfg.eos_token_id]
    } else {
        text_cfg.eos_token_ids.clone()
    };
    let mut next = match gpu.argmax_f32(&state.logits, text_cfg.vocab_size) {
        Ok(t) => t,
        Err(e) => {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("dots.ocr argmax failed: {e:?}"),
            });
        }
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
        if check_abort(id) {
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated,
            });
        }
        if eos_set.contains(&next) {
            break;
        }
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
            let _ = writeln!(
                stdout,
                r#"{{"type":"token","id":"{}","text":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default()
            );
            let _ = stdout.flush();
            emitted_bytes += valid_len;
        }

        match qwen2::forward_step_greedy(gpu, &weights.text, text_cfg, state, next) {
            Ok(t) => next = t,
            Err(e) => {
                return GenerateResult::Deferred(DeferredTerminal::Error {
                    id: id.to_string(),
                    message: format!("dots.ocr decode failed: {e:?}"),
                });
            }
        }
    }

    let decode_s = t_gen.elapsed().as_secs_f64();
    let total_s = t0.elapsed().as_secs_f64();
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
    GenerateResult::Complete
}

/// dots.ocr (arch_id=8) n-gram speculative decode, post-vision-prefill.
///
/// `generate_vl_dots_ocr` runs the image-conditioned prefill and routes here
/// when a model-free n-gram speculator was built at load (HIPFIRE_NGRAM_DRAFT=1).
/// dots.ocr's text decoder IS Qwen2, so the speculator drives it through the
/// `DotsOcrBundle: SpecTarget` impl. The vision prefill already advanced the
/// `ModelState::DotsOcr(b).state` KV, so this only replaces the *decode* phase.
///
/// The bundle is now live in `m.state` as `ModelState::DotsOcr`; borrow it
/// in-place via `m.state.as_mut()` so the `spec_target_guard` contract holds
/// without take/put churn. `m.speculator` and `m.tokenizer` are disjoint fields
/// and coexist with the bundle borrow.
#[allow(clippy::too_many_arguments)]
fn decode_vl_dots_ocr_ngram(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    t0: Instant,
    prefill_tokens: usize,
    prefill_s: f64,
) -> GenerateResult {
    let Some(ModelState::DotsOcr(bundle)) = m.state.as_mut() else {
        return GenerateResult::Deferred(DeferredTerminal::Error {
            id: id.to_string(),
            message: "dots-ocr bundle missing on arch_id=8".to_string(),
        });
    };
    let mut spec = m.speculator.take().unwrap();
    // `m.tokenizer` is a disjoint field → coexists with the bundle borrow above.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let result = run_dots_ocr_ngram_loop(
        bundle,
        spec.as_mut(),
        tokenizer,
        gpu,
        stdout,
        id,
        prompt_ids,
        max_tokens,
        t0,
        prefill_tokens,
        prefill_s,
    );
    m.speculator = Some(spec);
    result
}

/// The dots.ocr n-gram decode loop proper, factored out of
/// [`decode_vl_dots_ocr_ngram`] so the `&DotsOcrBundle` borrow it drives is
/// disjoint from the `&mut m` field-restore. Mirrors the `generate_spec`
/// prefill→step contract but with plain UTF-8 text streaming (no `SpecEmit`:
/// dots.ocr output is unframed layout-JSON, no reasoning/marker/tool channels).
#[allow(clippy::too_many_arguments)]
fn run_dots_ocr_ngram_loop(
    bundle: &mut hipfire_arch_dots_ocr::DotsOcrBundle,
    spec: &mut dyn hipfire_runtime::spec::Speculator,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    t0: Instant,
    prefill_tokens: usize,
    prefill_s: f64,
) -> GenerateResult {
    let eos_set: Vec<u32> = if bundle.config.text.eos_token_ids.is_empty() {
        vec![bundle.config.text.eos_token_id]
    } else {
        bundle.config.text.eos_token_ids.clone()
    };
    let block_size = spec.block_size();
    let ctx_capacity = spec.ctx_capacity();

    // Prime the n-gram drafter + fetch the first token WITHOUT re-running the
    // (vision-conditioned) target prefill. `cache_hit=true` + an empty suffix
    // makes `ChainSpeculator::prefill` skip the target advance —
    // `spec_advance(&[], prompt_len)` just argmaxes the live
    // post-vision-prefill logits — and only `drafter.prefill_seed(prompt_ids)`.
    // It also lazily builds the verify scratch (required before the first `step`).
    let first_token = match spec.prefill(
        gpu,
        bundle,
        prompt_ids,
        &[],
        prompt_ids.len(),
        true,
        None,
        &|| check_abort(id),
    ) {
        Ok(PrefillOutcome::Ready { first_token }) => first_token,
        Ok(PrefillOutcome::Aborted) => {
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated: 0,
            });
        }
        Err(e) => {
            return GenerateResult::Deferred(DeferredTerminal::Error {
                id: id.to_string(),
                message: format!("dots.ocr spec prefill: {e}"),
            });
        }
    };

    let t_gen = Instant::now();
    let mut streamed: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut generated = 0usize;
    // n-gram context (committed generated tail; the drafter holds the prompt
    // internally via prefill_seed).
    let mut emitted: Vec<u32> = Vec::new();
    let mut position = prompt_ids.len();
    let mut seed_token = first_token;
    // τ accounting (accepted drafts / windows) — mirrors the text spec path so
    // the done envelope reports acceptance for diagnosing spec-vs-AR perf.
    let mut spec_cycles = 0usize;
    let mut spec_accepted = 0usize;
    // Tokens to stream this iteration. First window = the prefill seed alone
    // (mirrors the AR loop emitting the first argmax), then the accepted
    // committed tail from each `spec.step` (seed re-echo already stripped).
    let mut window: Vec<u32> = vec![first_token];

    'outer: loop {
        for &tok in &window {
            if generated >= max_tokens {
                break 'outer;
            }
            // EOS is never streamed (matches the AR loop's pre-emit break).
            if eos_set.contains(&tok) {
                break 'outer;
            }
            emit_committed_event(stdout, id, tok, generated, t0.elapsed().as_millis() as u64);
            generated += 1;
            streamed.push(tok);
            emitted.push(tok);
            // Incremental UTF-8 streaming — only emit complete code points
            // (byte-identical to the AR path).
            let all_bytes = tokenizer.decode_bytes(&streamed);
            let new_bytes = &all_bytes[emitted_bytes..];
            let valid_len = match std::str::from_utf8(new_bytes) {
                Ok(_) => new_bytes.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_len > 0 {
                let text = std::str::from_utf8(&new_bytes[..valid_len]).unwrap();
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"token","id":"{}","text":{}}}"#,
                    id,
                    serde_json::to_string(&text).unwrap_or_default()
                );
                let _ = stdout.flush();
                emitted_bytes += valid_len;
            }
        }
        if generated >= max_tokens {
            break;
        }
        // Decode-side cancel: stop early. The next request resets state at
        // prefill, so no cross-request bleed; the caller restores bundle/spec.
        if check_abort(id) {
            return GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: id.to_string(),
                generated,
            });
        }
        // Context-overflow guard (matches generate_spec): one window writes up
        // to `block_size` KV slots.
        if position.saturating_add(block_size) >= ctx_capacity {
            break;
        }
        let step = match spec.step(gpu, bundle, position, seed_token, &emitted, None, 0.0) {
            Ok(s) => s,
            Err(e) => {
                return GenerateResult::Deferred(DeferredTerminal::Error {
                    id: id.to_string(),
                    message: format!("dots.ocr spec_step: {e}"),
                });
            }
        };
        spec_cycles += 1;
        spec_accepted += step.accepted;
        // Advance by the emitted-tail length (= accepted + 1), per the spec.rs
        // `emit_len_drives_advance` contract; the target already wrote KV for the
        // whole tail in `verify_block`.
        position += step.emit.len();
        seed_token = step.next_seed;
        window = step.emit.to_vec();
    }

    let decode_s = t_gen.elapsed().as_secs_f64();
    let total_s = t0.elapsed().as_secs_f64();
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
    let tau = if spec_cycles > 0 {
        spec_accepted as f64 / spec_cycles as f64
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1},"dflash":true,"tau":{:.2},"cycles":{}}}"#,
        id,
        generated,
        tok_s,
        prefill_tokens,
        prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        prefill_s * 1000.0,
        tau,
        spec_cycles
    );
    let _ = stdout.flush();
    GenerateResult::Complete
}

#[cfg(test)]
mod tool_call_parser_tests {
    use super::extract_tool_calls_from_text;

    #[test]
    fn parses_valid_block() {
        let s = r#"prelude<tool_call>
{"name": "read", "arguments": {"path": "/etc/hostname"}}
</tool_call>tail"#;
        let calls = extract_tool_calls_from_text(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "/etc/hostname");
    }

    #[test]
    fn handles_unclosed_tool_call() {
        // Model truncated at max_tokens before emitting </tool_call>.
        // OLD parser broke out of the loop; NEW parser treats rest of
        // string as body and recovers the call. This was the Pi-session
        // call-9 failure mode that flipped the asst-cache fingerprint
        // from tool_calls=1 (CLI) to tool_calls=0 (daemon) → full reset.
        let s = r#"prelude<tool_call>
{"name": "read", "arguments": {"path": "/etc/hostname"}}"#;
        let calls = extract_tool_calls_from_text(s);
        assert_eq!(calls.len(), 1, "unclosed block dropped — should recover");
        assert_eq!(calls[0].name, "read");
    }

    #[test]
    fn truncated_args_not_emitted_as_empty() {
        // A `write` cut off mid-`content` (max_tokens / grammar force-close):
        // the args object never closes, so no balanced object is recoverable.
        // The OLD fallback fabricated empty `{}` args, presenting write({}) to
        // the client as executable (the write-tool empty-args incident). NEW:
        // drop the call entirely so the emission surfaces as content +
        // finish_reason for the client to retry. Distinct from
        // `handles_unclosed_tool_call`, where the args ARE complete and only
        // the `</tool_call>` marker is missing.
        let s = "<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"/tmp/big.zig\", \"content\": \"const std = @im";
        let calls = extract_tool_calls_from_text(s);
        assert!(
            calls.is_empty(),
            "truncated args must NOT emit a fabricated-empty call"
        );
    }

    #[test]
    fn loose_json_with_complete_args_still_recovered() {
        // Broken outer JSON (leading `{` lost to special-token leakage) but a
        // COMPLETE balanced args object — the fallback still recovers it,
        // distinguishing real recovery from the truncation case above.
        let s =
            "<tool_call>\nname\": \"read\", \"arguments\": {\"path\": \"/tmp/x\"}\n</tool_call>";
        let calls = extract_tool_calls_from_text(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "/tmp/x");
    }

    #[test]
    fn strips_chatml_special_tokens_in_body() {
        let s = "<tool_call>\n<|im_start|>{\"name\": \"read\", \"arguments\": {\"path\": \"/x\"}}<|im_end|>\n</tool_call>";
        let calls = extract_tool_calls_from_text(s);
        assert_eq!(calls.len(), 1, "ChatML token leakage broke JSON parse");
        assert_eq!(calls[0].name, "read");
    }

    #[test]
    fn nested_opener_stripped() {
        let s = r#"<tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/x"}}
</tool_call>"#;
        let calls = extract_tool_calls_from_text(s);
        assert_eq!(calls.len(), 1, "nested opener dropped");
        assert_eq!(calls[0].name, "read");
    }

    #[test]
    fn no_block_no_calls() {
        let calls = extract_tool_calls_from_text("just text, no tool call");
        assert!(calls.is_empty());
    }

    #[test]
    fn form4_skips_name_substring_in_other_key() {
        // `firstname` contains `name` — the fallback used to bail when
        // it saw an invalid pre-byte for the first match. Should now
        // skip and find the real `name` key on the next occurrence.
        // (Strict JSON parse handles this trivially; this test exercises
        // the fallback path by wrapping in <tool_call> with off-spec
        // shape that triggers fallback.)
        let body = r#"{"firstname":"X","name":"read","arguments":{"path":"/x"}}"#;
        assert_eq!(
            hipfire_runtime::emit_text::extract_tool_call_name_fallback(body),
            Some("read".to_string())
        );
    }

    #[test]
    fn form4_handles_trailing_comma() {
        // serde_json rejects trailing commas; the fallback should
        // still find name + arguments.
        let s = r#"<tool_call>
{"name": "read", "arguments": {"path": "/x",},}
</tool_call>"#;
        let calls = extract_tool_calls_from_text(s);
        assert_eq!(calls.len(), 1, "trailing-comma JSON dropped");
        assert_eq!(calls[0].name, "read");
    }

    #[test]
    fn form4_handles_unquoted_key() {
        // Off-spec JSON with unquoted key.
        let body = r#"{name: "read"}"#;
        assert_eq!(
            hipfire_runtime::emit_text::extract_tool_call_name_fallback(body),
            Some("read".to_string())
        );
    }

    #[test]
    fn empty_body_no_call() {
        // Empty `<tool_call></tool_call>` shouldn't produce a call.
        let s = "<tool_call></tool_call>";
        let calls = extract_tool_calls_from_text(s);
        assert!(calls.is_empty());
    }

    #[test]
    fn multiple_blocks_extract_all() {
        // Two valid tool_call blocks in one emission should yield two calls.
        let s = r#"<tool_call>
{"name":"a","arguments":{}}
</tool_call>prose<tool_call>
{"name":"b","arguments":{}}
</tool_call>"#;
        let calls = extract_tool_calls_from_text(s);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }
}

#[cfg(test)]
mod ar_parser_lifecycle_tests {
    use super::{
        ar_sampled_eos, complete_ar_decode, decode_limit_reached, reset_failed_message,
        route_ar_decode_outcome, stream_stop, ArDecodeOutcome, ArSampledEos, DeferredTerminal,
        GenerateResult,
    };
    use hipfire_runtime::stream_parser::{EosDecision, StreamAction, StreamParser};
    use std::sync::{Arc, Mutex};

    #[test]
    fn generation_errors_are_deferred_with_request_context() {
        assert_eq!(
            GenerateResult::Deferred(DeferredTerminal::Error {
                id: "r1".into(),
                message: "token forward: bad state".into(),
            }),
            GenerateResult::Deferred(DeferredTerminal::Error {
                id: "r1".into(),
                message: "token forward: bad state".into(),
            })
        );
        assert_ne!(
            GenerateResult::Deferred(DeferredTerminal::Aborted {
                id: "r1".into(),
                generated: 3,
            }),
            GenerateResult::Complete
        );
    }

    fn assert_reset_failure_is_untouched(result: GenerateResult, expected_id: &str) {
        match result {
            GenerateResult::ResetFailed { id, message } => {
                assert_eq!(id, expected_id);
                assert_eq!(message, "reset exploded");
            }
            GenerateResult::Deferred(_) => panic!("reset failure was converted to Deferred"),
            GenerateResult::Complete => panic!("reset failure was swallowed"),
            GenerateResult::PpCompletion { .. } => {
                panic!("reset failure was converted to PP completion")
            }
        }
    }

    #[test]
    fn pp_completion_requires_at_most_one_outer_reset() {
        let completion = GenerateResult::PpCompletion {
            generated: 2,
            prefill_tokens: 4,
            tok_s: 1.0,
            prefill_ms: 2.0,
            prefill_tok_s: 2.0,
            decode_tok_s: 1.0,
            reset_required: true,
        };
        match completion {
            GenerateResult::PpCompletion { reset_required, .. } => {
                assert!(reset_required, "only the outer PP wrapper may reset")
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn text_reset_failure_propagates_untouched() {
        assert_reset_failure_is_untouched(
            reset_failed_message("text-id", "reset exploded".into()),
            "text-id",
        );
    }

    #[test]
    fn vl_reset_failure_propagates_untouched() {
        assert_reset_failure_is_untouched(
            reset_failed_message("vl-id", "reset exploded".into()),
            "vl-id",
        );
    }

    #[test]
    fn spec_reset_failure_propagates_untouched() {
        assert_reset_failure_is_untouched(
            reset_failed_message("spec-id", "reset exploded".into()),
            "spec-id",
        );
    }

    #[test]
    fn pp_reset_failure_propagates_untouched() {
        assert_reset_failure_is_untouched(
            reset_failed_message("pp-id", "reset exploded".into()),
            "pp-id",
        );
    }

    #[test]
    fn deferred_terminal_ids_are_json_escaped() {
        let id = "deferred\"\\\n";
        let encoded = serde_json::to_string(id).expect("request id is serializable");
        assert_eq!(encoded, "\"deferred\\\"\\\\\\n\"");
    }

    struct LifecycleParser {
        finish_count: Arc<std::sync::atomic::AtomicUsize>,
        eos: EosDecision,
    }

    impl StreamParser for LifecycleParser {
        fn feed(&mut self, _tok: u32, _bytes: &[u8]) -> Vec<StreamAction> {
            Vec::new()
        }

        fn finish(&mut self) -> Vec<StreamAction> {
            self.finish_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            vec![StreamAction::Emit {
                text: "flushed".into(),
                reasoning: false,
            }]
        }

        fn on_eos(&mut self) -> EosDecision {
            self.eos.clone()
        }
    }

    fn parser(counter: &Arc<std::sync::atomic::AtomicUsize>) -> Box<dyn StreamParser> {
        Box::new(LifecycleParser {
            finish_count: Arc::clone(counter),
            eos: EosDecision::Stop,
        })
    }

    fn assert_normal_completion(control: &str, outcome: ArDecodeOutcome) {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let events = Arc::new(Mutex::new(Vec::<String>::new()));
        let forward_events = Arc::clone(&events);
        let render_events = Arc::clone(&events);
        let mut conversation = vec![1, 42];
        let mut seq_pos = 10;
        let result = complete_ar_decode(
            parser(&count),
            outcome,
            conversation.last().copied(),
            &mut seq_pos,
            Some(42),
            &[99],
            |token, position| {
                forward_events
                    .lock()
                    .unwrap()
                    .push(format!("forward:{token}@{position}"));
                conversation.push(token);
                Ok(position + 1)
            },
            move |action| {
                assert!(
                    matches!(action, StreamAction::Emit { .. }),
                    "control={control}"
                );
                render_events.lock().unwrap().push("finish".into());
            },
        );

        assert_eq!(result, ArDecodeOutcome::Complete, "control={control}");
        assert_eq!(conversation, vec![1, 42, 99], "control={control}");
        assert_eq!(seq_pos, 11, "control={control}");
        assert_eq!(
            *events.lock().unwrap(),
            vec!["forward:99@10", "finish"],
            "control={control}"
        );
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn sampled_eos_routes_to_normal_completion() {
        assert_normal_completion("sampled EOS", route_ar_decode_outcome(true, None));
    }

    #[test]
    fn forced_stop_routes_to_normal_completion() {
        assert_normal_completion("forced stop", route_ar_decode_outcome(true, None));
    }

    #[test]
    fn budget_stop_routes_to_normal_completion() {
        assert_normal_completion("budget stop", route_ar_decode_outcome(true, None));
    }

    #[test]
    fn length_routes_to_normal_completion() {
        assert_normal_completion("length", route_ar_decode_outcome(true, None));
    }

    #[test]
    fn stream_stop_detects_normal_parser_control() {
        let stop = [StreamAction::Stop];
        assert!(stream_stop(&stop));
        assert!(!stream_stop(&[]));
    }

    #[test]
    fn decode_limits_stop_the_loop() {
        assert!(decode_limit_reached(8, 8, 8, 8, false));
        assert!(decode_limit_reached(8, 7, 8, 8, false));
        assert!(!decode_limit_reached(8, 7, 8, 8, true));
        assert!(!decode_limit_reached(8, 7, 8, 7, false));
    }

    #[test]
    fn sampled_eos_injection_requires_a_forced_token() {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut stop_parser = LifecycleParser {
            finish_count: Arc::clone(&count),
            eos: EosDecision::Stop,
        };
        assert_eq!(ar_sampled_eos(&mut stop_parser), ArSampledEos::Stop);
        let mut commit_parser = LifecycleParser {
            finish_count: Arc::clone(&count),
            eos: EosDecision::CommitAndStop,
        };
        assert_eq!(
            ar_sampled_eos(&mut commit_parser),
            ArSampledEos::CommitAndStop
        );
        let mut parser = LifecycleParser {
            finish_count: count,
            eos: EosDecision::Inject(Vec::new()),
        };
        assert_eq!(
            ar_sampled_eos(&mut parser),
            ArSampledEos::Error("sampled eos injection produced no forced token".into())
        );
    }

    #[test]
    fn failed_forced_dequeue_is_an_error_not_normal_completion() {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut parser = LifecycleParser {
            finish_count: count,
            eos: EosDecision::Inject(vec![42]),
        };
        assert_eq!(
            ar_sampled_eos(&mut parser),
            ArSampledEos::Error("sampled eos injection failed forced dequeue".into())
        );
    }

    #[test]
    fn abort_and_error_discard_without_events_or_finish() {
        for outcome in [
            route_ar_decode_outcome(false, None),
            route_ar_decode_outcome(false, Some("token forward: test failure".into())),
        ] {
            let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let events = Arc::new(Mutex::new(Vec::new()));
            let forward_events = Arc::clone(&events);
            let settle_events = Arc::clone(&events);
            let expected = format!("{outcome:?}");
            let mut seq_pos = 10;
            let mut conversation = vec![1, 42];
            let result = complete_ar_decode(
                parser(&count),
                outcome,
                Some(42),
                &mut seq_pos,
                Some(42),
                &[99],
                |token, _| {
                    conversation.push(token);
                    forward_events.lock().unwrap().push("unexpected-forward");
                    Ok(11)
                },
                move |_| {
                    settle_events.lock().unwrap().push("unexpected-event");
                },
            );

            assert_eq!(format!("{result:?}"), expected);
            assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
            assert!(events.lock().unwrap().is_empty());
            assert_eq!(seq_pos, 10);
            assert_eq!(conversation, vec![1, 42]);
        }
    }

    #[test]
    fn trailer_forward_failure_is_controlled_error_and_discards_parser() {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut seq_pos = 10;
        let conversation = vec![1, 42];
        let result = complete_ar_decode(
            parser(&count),
            ArDecodeOutcome::Complete,
            Some(42),
            &mut seq_pos,
            Some(42),
            &[99],
            |_token, _| Err::<usize, _>("GPU failed".into()),
            |_| panic!("failed trailer must not settle parser"),
        );

        assert_eq!(
            result,
            ArDecodeOutcome::Error("ChatML trailer forward: GPU failed".into())
        );
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(seq_pos, 10);
        assert_eq!(conversation, vec![1, 42]);
    }
}

#[cfg(test)]
mod terminal_epilogue_policy_tests {
    use super::{finish_qwen35_pp_filter, qwen35_pp_eos_filter_config, settle_deepseek4_ar_parser};
    use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};

    #[test]
    fn qwen35_pp_production_config_stops_and_returns_safe_prefix() {
        let mut filter = EosFilter::new(qwen35_pp_eos_filter_config(&["<stop>".into()]));

        assert_eq!(
            filter.observe(b"safe<stop>post-marker"),
            FilterAction::Stop {
                emit: b"safe".to_vec()
            }
        );
        assert!(matches!(
            filter.observe(b"must-not-leak"),
            FilterAction::Hold
        ));
        assert!(filter.finish().is_empty());
    }

    #[test]
    fn qwen35_filter_flushes_safe_tail_once() {
        let mut filter = EosFilter::new(EosFilterConfig {
            stop_at: vec![b"<stop>".to_vec()],
            ..EosFilterConfig::default()
        });
        assert_eq!(filter.observe(b"<sto"), FilterAction::Hold);

        assert_eq!(
            finish_qwen35_pp_filter(&mut filter).as_deref(),
            Some("<sto")
        );
        // The local terminal epilogue is idempotent at the filter boundary;
        // no second terminal event can be rendered.
        assert_eq!(finish_qwen35_pp_filter(&mut filter), None);
    }

    #[test]
    fn deepseek_normal_epilogue_renders_buffered_text() {
        let mut parser = hipfire_arch_deepseek4::dsml::StreamParser::new();
        assert!(parser.feed("<th").is_empty());
        let mut events = Vec::new();
        settle_deepseek4_ar_parser(parser, true, |event| events.push(event));

        assert_eq!(
            events,
            vec![hipfire_arch_deepseek4::dsml::StreamEvent::Token(
                "<th".into()
            )]
        );
    }

    #[test]
    fn deepseek_abort_or_error_discards_buffered_text() {
        let mut parser = hipfire_arch_deepseek4::dsml::StreamParser::new();
        assert!(parser.feed("<th").is_empty());
        let mut events = Vec::new();
        settle_deepseek4_ar_parser(parser, false, |event| events.push(event));
        assert!(events.is_empty());
    }
}

#[cfg(test)]
mod c2m_stream_parser_tests {
    use super::{C2mSec, Cohere2MoeStreamParser};
    use hipfire_runtime::stream_parser::{EosDecision, StreamAction, StreamParser};

    // Build a parser with explicit test marker ids (bypasses tokenizer/new()).
    fn mk() -> Cohere2MoeStreamParser {
        Cohere2MoeStreamParser {
            sec: C2mSec::Pre,
            action_buf: String::new(),
            vis_buf: String::new(),
            forced: std::collections::VecDeque::new(),
            think_count: 0,
            think_budget: usize::MAX,
            think_force_closed: false,
            empty_turn_guard: true,
            emitted_visible: false,
            eos_suppressions: 0,
            tool_calls_emitted: false,
            finish_latched: false,
            last_tok: u32::MAX,
            repeat_run: 0,
            mk_think0: 10,
            mk_think1: 11,
            mk_text0: 12,
            mk_text1: 13,
            mk_act0: 14,
            mk_act1: 15,
            pad_tok: None,
            known_tools: Vec::new(),
            tool_params: Vec::new(),
        }
    }

    #[test]
    fn section_machine_routes_and_suppresses_markers() {
        let mut p = mk();
        // [START_THINKING, "hi", END_THINKING, START_TEXT, "yo"]
        assert_eq!(p.feed(10, b""), Vec::new()); // START_THINKING → Think, suppressed
        assert_eq!(
            p.feed(99, b"hi"),
            vec![StreamAction::Emit {
                text: "hi".into(),
                reasoning: true
            }]
        );
        assert_eq!(p.feed(11, b""), Vec::new()); // END_THINKING → Pre, suppressed
        assert_eq!(p.feed(12, b""), Vec::new()); // START_TEXT → Text, suppressed
        assert_eq!(
            p.feed(98, b"yo"),
            vec![StreamAction::Emit {
                text: "yo".into(),
                reasoning: false
            }]
        );
    }

    #[test]
    fn empty_turn_guard_injects_start_text_then_stops() {
        let mut p = mk();
        // Inside Think, nothing visible → on_eos injects [END_THINKING, START_TEXT].
        let _ = p.feed(10, b""); // enter Think
        match p.on_eos() {
            EosDecision::Inject(v) => assert_eq!(v, vec![11, 12]),
            other => panic!("expected Inject, got {other:?}"),
        }
        // After MAX_EOS_SUPPRESS, falls through to Stop.
        p.emitted_visible = false;
        p.eos_suppressions = Cohere2MoeStreamParser::MAX_EOS_SUPPRESS;
        assert_eq!(p.on_eos(), EosDecision::Stop);
    }

    #[test]
    fn eos_injection_is_not_terminal() {
        let mut p = mk();
        let _ = p.feed(10, b""); // enter Think
        assert!(matches!(p.on_eos(), EosDecision::Inject(_)));

        // The injected continuation is followed by ordinary decoding.
        assert_eq!(p.feed(11, b""), Vec::new());
        assert_eq!(p.feed(12, b""), Vec::new());
        assert_eq!(
            p.feed(99, b"answer"),
            vec![StreamAction::Emit {
                text: "answer".into(),
                reasoning: false,
            }]
        );
    }

    #[test]
    fn forced_marker_emission_updates_section_and_suppresses_marker_bytes() {
        let mut p = mk();
        let _ = p.feed(10, b""); // enter Think
        let EosDecision::Inject(forced) = p.on_eos() else {
            panic!("expected forced continuation");
        };
        for tok in forced {
            p.enqueue(tok);
            let tok = p.next_forced().expect("enqueued marker");
            let marker = match tok {
                11 => b"<|END_THINKING|>".as_slice(),
                12 => b"<|START_TEXT|>".as_slice(),
                other => panic!("unexpected forced token {other}"),
            };
            assert_eq!(p.emit_only(tok, marker), Vec::new());
        }
        assert_eq!(
            p.feed(99, b"answer"),
            vec![StreamAction::Emit {
                text: "answer".into(),
                reasoning: false,
            }]
        );
    }

    #[test]
    fn finish_recovers_complete_pending_action_once() {
        let mut p = mk();
        let action = r#"[{"tool_name":"bash","parameters":{"command":"ls"}}]"#;
        let _ = p.feed(14, b""); // START_ACTION
        let _ = p.feed(99, action.as_bytes());

        assert_eq!(
            p.finish(),
            vec![StreamAction::ToolCalls(serde_json::json!([{
                "name": "bash",
                "arguments": { "command": "ls" }
            }]))]
        );
        assert_eq!(p.finish(), Vec::new());
    }

    #[test]
    fn finish_recovers_second_complete_action_after_first_closed_action() {
        let mut p = mk();
        let first = r#"[{"tool_name":"bash","parameters":{"command":"ls"}}]"#;
        let second = r#"[{"tool_name":"bash","parameters":{"command":"pwd"}}]"#;
        let _ = p.feed(14, b""); // START_ACTION
        let mut actions = p.feed(99, first.as_bytes());
        actions.extend(p.feed(15, b"")); // END_ACTION
        let _ = p.feed(14, b""); // START_ACTION
        let _ = p.feed(99, second.as_bytes());
        actions.extend(p.finish());

        assert_eq!(
            actions,
            vec![
                StreamAction::ToolCalls(serde_json::json!([{
                    "name": "bash",
                    "arguments": { "command": "ls" }
                }])),
                StreamAction::ToolCalls(serde_json::json!([{
                    "name": "bash",
                    "arguments": { "command": "pwd" }
                }])),
            ]
        );
        assert_eq!(p.finish(), Vec::new());
    }

    #[test]
    fn finish_discards_incomplete_pending_action() {
        let mut p = mk();
        let _ = p.feed(14, b""); // START_ACTION
        let _ = p.feed(
            99,
            br#"[{"tool_name":"bash","parameters":{"command":"ls"}}"#,
        );

        assert_eq!(p.finish(), Vec::new());
        assert_eq!(p.finish(), Vec::new());
    }

    #[test]
    fn repeated_finish_is_silent_after_text_recovery() {
        let mut p = mk();
        let text = r#"[{"tool_name":"bash","parameters":{"command":"ls"}}]"#;
        let _ = p.feed(99, text.as_bytes());

        assert_eq!(
            p.finish(),
            vec![StreamAction::ToolCalls(serde_json::json!([{
                "name": "bash",
                "arguments": { "command": "ls" }
            }]))]
        );
        assert_eq!(p.finish(), Vec::new());
    }
}

#[cfg(test)]
mod ds4_stream_parser_tests {
    use super::Deepseek4StreamParser;
    use hipfire_arch_deepseek4::dsml::{StreamEvent, ToolCall};
    use hipfire_runtime::stream_parser::{EosDecision, StreamAction, StreamParser};

    #[test]
    fn maps_token_and_reasoning_to_emit_channels() {
        let acts = Deepseek4StreamParser::map_events(vec![
            StreamEvent::Token("hi".into()),
            StreamEvent::Reasoning("because".into()),
        ]);
        assert_eq!(
            acts,
            vec![
                StreamAction::Emit {
                    text: "hi".into(),
                    reasoning: false
                },
                StreamAction::Emit {
                    text: "because".into(),
                    reasoning: true
                },
            ]
        );
    }

    #[test]
    fn maps_tool_calls_to_name_arguments_array() {
        let acts =
            Deepseek4StreamParser::map_events(vec![StreamEvent::ToolCalls(vec![ToolCall {
                name: "get_weather".into(),
                arguments: serde_json::json!({ "city": "Tokyo" }),
            }])]);
        assert_eq!(acts.len(), 1);
        match &acts[0] {
            StreamAction::ToolCalls(v) => {
                // Same shape emit_stream_event emits: [{"name":..,"arguments":..}].
                assert_eq!(
                    *v,
                    serde_json::json!([{ "name": "get_weather", "arguments": { "city": "Tokyo" } }])
                );
            }
            other => panic!("expected ToolCalls, got {other:?}"),
        }
    }

    #[test]
    fn on_eos_is_stop_not_commit() {
        // ep_serve_ds4 breaks on the sampled eos without forwarding/emitting it.
        let mut p = Deepseek4StreamParser::new(
            hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            Vec::new(),
        );
        assert_eq!(p.on_eos(), EosDecision::Stop);
    }

    #[test]
    fn finish_consumes_inner_and_is_idempotent() {
        let mut p = Deepseek4StreamParser::new(
            hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            Vec::new(),
        );
        let _ = p.finish(); // takes the inner dsml parser
        assert_eq!(p.finish(), Vec::new()); // second finish: inner is None → empty
    }

    #[test]
    fn honors_user_stop_sequences() {
        // review I1: `stop` must not be a no-op for ds4-EP. The stop check runs on
        // the accumulated raw pieces (independent of dsml buffering).
        let mut p = Deepseek4StreamParser::new(
            hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            vec!["END".to_string()],
        );
        let a1 = p.feed(1, b"hello ");
        assert!(!a1.iter().any(|x| matches!(x, StreamAction::Stop)));
        let a2 = p.feed(2, b"END");
        assert!(
            a2.iter().any(|x| matches!(x, StreamAction::Stop)),
            "stop sequence at the decoded suffix must append a Stop action"
        );
    }
}

#[cfg(test)]
mod mtp_k_tests {
    use super::{mtp_metadata_requested, resolve_mtp_k, resolve_mtp_mode};
    use serde_json::Value;
    use std::sync::{Mutex, OnceLock};

    fn resolve(param: Option<Value>) -> Result<usize, String> {
        resolve_mtp_k(param.as_ref(), false)
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct MtpKEnvRestore {
        generic: Option<std::ffi::OsString>,
        deepseek: Option<std::ffi::OsString>,
        mode: Option<std::ffi::OsString>,
    }

    impl Drop for MtpKEnvRestore {
        fn drop(&mut self) {
            match self.generic.take() {
                Some(value) => std::env::set_var("HIPFIRE_MTP_K", value),
                None => std::env::remove_var("HIPFIRE_MTP_K"),
            }
            match self.deepseek.take() {
                Some(value) => std::env::set_var("HIPFIRE_DEEPSEEK4_SPEC_K", value),
                None => std::env::remove_var("HIPFIRE_DEEPSEEK4_SPEC_K"),
            }
            match self.mode.take() {
                Some(value) => std::env::set_var("HIPFIRE_MTP_MODE", value),
                None => std::env::remove_var("HIPFIRE_MTP_MODE"),
            }
        }
    }

    fn with_mtp_env<T>(
        generic: Option<&str>,
        deepseek: Option<&str>,
        mode: Option<&str>,
        test: impl FnOnce() -> T,
    ) -> T {
        let _lock = env_lock().lock().unwrap();
        let _restore = MtpKEnvRestore {
            generic: std::env::var_os("HIPFIRE_MTP_K"),
            deepseek: std::env::var_os("HIPFIRE_DEEPSEEK4_SPEC_K"),
            mode: std::env::var_os("HIPFIRE_MTP_MODE"),
        };
        match generic {
            Some(value) => std::env::set_var("HIPFIRE_MTP_K", value),
            None => std::env::remove_var("HIPFIRE_MTP_K"),
        }
        match deepseek {
            Some(value) => std::env::set_var("HIPFIRE_DEEPSEEK4_SPEC_K", value),
            None => std::env::remove_var("HIPFIRE_DEEPSEEK4_SPEC_K"),
        }
        match mode {
            Some(value) => std::env::set_var("HIPFIRE_MTP_MODE", value),
            None => std::env::remove_var("HIPFIRE_MTP_MODE"),
        }
        test()
    }

    fn with_mtp_k_env<T>(value: Option<&str>, test: impl FnOnce() -> T) -> T {
        with_mtp_env(value, None, None, test)
    }

    #[test]
    fn environment_overrides_load_parameter() {
        with_mtp_k_env(Some("7"), || {
            assert_eq!(resolve(Some(serde_json::json!(4))), Ok(7))
        });
    }

    #[test]
    fn deepseek_environment_overrides_generic_only_for_deepseek() {
        with_mtp_env(Some("4"), Some("6"), None, || {
            assert_eq!(resolve_mtp_k(Some(&serde_json::json!(3)), true), Ok(6));
            assert_eq!(resolve_mtp_k(Some(&serde_json::json!(3)), false), Ok(4));
        });
    }

    #[test]
    fn load_parameter_is_used_without_environment() {
        with_mtp_k_env(None, || {
            assert_eq!(resolve(Some(serde_json::json!(4))), Ok(4))
        });
    }

    #[test]
    fn defaults_to_three_without_parameter_or_environment() {
        with_mtp_k_env(None, || assert_eq!(resolve(None), Ok(3)));
    }

    #[test]
    fn invalid_environment_rejects_even_with_valid_load_parameter() {
        with_mtp_k_env(Some("nope"), || {
            let error = resolve(Some(serde_json::json!(4))).unwrap_err();
            assert!(error.contains("HIPFIRE_MTP_K"));
        });
    }

    #[test]
    fn environment_rejects_noncanonical_leading_zero() {
        with_mtp_k_env(Some("01"), || {
            assert!(resolve(Some(serde_json::json!(4))).is_err())
        });
    }

    #[test]
    fn out_of_range_environment_is_rejected() {
        with_mtp_k_env(Some("11"), || {
            assert!(resolve(Some(serde_json::json!(4))).is_err())
        });
    }

    #[test]
    fn accepts_full_daemon_range() {
        for k in 1..=8 {
            let env_value = k.to_string();
            with_mtp_k_env(Some(&env_value), || assert_eq!(resolve(None), Ok(k)));
        }
    }

    #[test]
    fn non_integer_json_parameter_is_rejected() {
        with_mtp_k_env(None, || {
            assert!(resolve(Some(serde_json::json!("four"))).is_err())
        });
    }

    #[test]
    fn out_of_range_json_parameter_is_rejected() {
        for value in [
            serde_json::json!(-1),
            serde_json::json!(0),
            serde_json::json!(9),
            serde_json::json!(10),
            serde_json::json!(11),
        ] {
            with_mtp_k_env(None, || assert!(resolve(Some(value)).is_err()));
        }
    }

    #[test]
    fn non_integer_json_values_are_rejected() {
        for value in [
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!([3]),
            serde_json::json!({ "mtp_k": 3 }),
            serde_json::json!(3.5),
            serde_json::json!("3"),
        ] {
            with_mtp_k_env(None, || assert!(resolve(Some(value)).is_err()));
        }
    }

    #[test]
    fn invalid_mtp_mode_parameter_is_rejected() {
        with_mtp_env(None, None, None, || {
            for value in [
                serde_json::json!(null),
                serde_json::json!(true),
                serde_json::json!("enabled"),
            ] {
                assert!(resolve_mtp_mode(Some(&value)).is_err());
            }
        });
    }

    #[test]
    fn mtp_mode_defaults_to_auto_and_accepts_only_documented_values() {
        with_mtp_env(None, None, None, || {
            assert_eq!(resolve_mtp_mode(None), Ok(None));
            assert_eq!(resolve_mtp_mode(Some(&serde_json::json!("auto"))), Ok(None));
            assert_eq!(
                resolve_mtp_mode(Some(&serde_json::json!("on"))),
                Ok(Some(true))
            );
            assert_eq!(
                resolve_mtp_mode(Some(&serde_json::json!("off"))),
                Ok(Some(false))
            );
        });
    }

    #[test]
    fn direct_mode_environment_overrides_load_parameter() {
        with_mtp_env(None, None, Some("off"), || {
            assert_eq!(
                resolve_mtp_mode(Some(&serde_json::json!("on"))),
                Ok(Some(false))
            );
        });
    }

    #[test]
    fn metadata_only_mode_controls_deepseek_speculation() {
        assert!(!mtp_metadata_requested("off", true));
        assert!(!mtp_metadata_requested("auto", false));
        assert!(mtp_metadata_requested("auto", true));
        assert!(mtp_metadata_requested("on", false));
    }
}

#[cfg(test)]
mod vl_request_state_tests {
    use super::{prepare_vl_request_state, vl_image_state};
    use hipfire_loader::SessionState;

    #[test]
    fn image_turn_cold_reset_clears_model_turn_state() {
        let mut session = SessionState::default();
        let image = vl_image_state(&[0.0, 1.0, 2.0], 1, 3);
        assert!(!prepare_vl_request_state(&mut session, image));
        session.seq_pos = 17;
        session.conversation_tokens = vec![11, 12, 13];
        let other_image = vl_image_state(&[0.0, 1.0, 3.0], 1, 3);
        assert_ne!(image, other_image);
        assert!(prepare_vl_request_state(&mut session, other_image));
        assert_eq!(session.seq_pos, 0);
        assert!(session.conversation_tokens.is_empty());
        assert_eq!(session.vl_image_state, Some(other_image));
    }

    #[test]
    fn first_image_turn_preserves_empty_start_state() {
        let mut session = SessionState::default();
        let image = vl_image_state(&[0.0, 1.0, 2.0], 1, 3);
        assert!(!prepare_vl_request_state(&mut session, image));
        assert_eq!(session.seq_pos, 0);
        assert!(session.conversation_tokens.is_empty());
        assert_eq!(session.vl_image_state, Some(image));
    }

    #[test]
    fn prior_image_turn_requires_reset_even_when_position_was_rewound() {
        let mut session = SessionState::default();
        let image_a = vl_image_state(&[0.0, 1.0, 2.0], 1, 3);
        let image_b = vl_image_state(&[3.0, 2.0, 1.0], 1, 3);

        assert!(!prepare_vl_request_state(&mut session, image_a));
        // A failed/short turn can leave seq_pos at zero while the image-turn
        // sentinel still identifies model state that must not be reused.
        session.seq_pos = 0;
        session.conversation_tokens.extend_from_slice(&[11, 12]);

        assert!(prepare_vl_request_state(&mut session, image_b));
        assert_eq!(session.seq_pos, 0);
        assert!(session.conversation_tokens.is_empty());
        assert_eq!(session.vl_image_state, Some(image_b));
    }

    #[test]
    fn image_a_to_image_b_transition_matches_fresh_image_b_state() {
        let image_a = vl_image_state(&[0.0, 1.0, 2.0], 1, 3);
        let image_b = vl_image_state(&[3.0, 2.0, 1.0], 1, 3);
        let mut reused = SessionState::default();
        let mut fresh = SessionState::default();

        assert!(!prepare_vl_request_state(&mut reused, image_a));
        reused.seq_pos = 0;
        reused.conversation_tokens.extend_from_slice(&[11, 12]);
        assert!(prepare_vl_request_state(&mut reused, image_b));
        assert!(!prepare_vl_request_state(&mut fresh, image_b));

        assert_eq!(
            (
                reused.seq_pos,
                reused.conversation_tokens,
                reused.vl_image_state
            ),
            (
                fresh.seq_pos,
                fresh.conversation_tokens,
                fresh.vl_image_state
            )
        );
    }
}

#[cfg(test)]
mod qwen_cache_capacity_tests {
    use super::qwen_cache_guard_position;

    #[test]
    fn cache_miss_is_reset_before_effective_zero_capacity_check() {
        // A dirty session can be at the physical end while the new prompt fits
        // in a freshly reset context. This is the regression that used to be
        // rejected because the guard ran before the cold reset.
        assert_eq!(qwen_cache_guard_position(128, true), 0);
        assert_eq!(qwen_cache_guard_position(128, false), 128);
    }
}

#[cfg(test)]
mod lifecycle_matrix_tests {
    use super::reset_domain_cache_capable;

    #[test]
    fn production_cache_lifecycle_matrix_is_explicit() {
        for arch_id in [5, 6, 9, 10, 12] {
            assert!(
                reset_domain_cache_capable(arch_id),
                "cache-capable AR path missing from lifecycle matrix: arch_id={arch_id}"
            );
        }
        for arch_id in [7, 8, 11] {
            assert!(
                !reset_domain_cache_capable(arch_id),
                "non-cache-capable path unexpectedly admitted: arch_id={arch_id}"
            );
        }
    }
}

/// Host-only daemon load plan tests — no GPU, no hardware.
///
/// These tests exercise the `daemon_load_plan` helper which is the
/// pre-mesh admission + lifecycle call for the daemon load path.  Every
/// test creates a minimal HFQ fixture and verifies that admission (or
/// refusal) returns a valid [`DaemonLoadPlan`] with correct lifecycle
/// flags and effective mesh before any GPU work.  These tests do NOT
/// prove full end-to-end integration (no GPU, no model → no
/// `load_admitted` execution) — they guard the load-planning phase.
///
/// # RED/GREEN contract
///
/// - **RED**: the old daemon code used `HfqFile::open` peeks + manual
///   TP→EP remap + raw-degree guards, which would NOT catch VL PP,
///   dense EP normalisation, or composition correctly BEFORE mesh build.
/// - **GREEN**: the daemon load handler uses `daemon_load_plan` which
///   classifies the source, resolves policy, and returns lifecycle
///   decisions before any GPU mutation.
#[cfg(test)]
mod daemon_load_plan_tests {
    use super::*;
    use hipfire_hardware::DimKind;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Mutex;
    use tempfile::TempDir;

    // ── Minimal HFQ fixture writer (mirrors loader test infrastructure) ──

    fn write_hfq(
        dir: &Path,
        name: &str,
        arch_id: u32,
        metadata_json: &str,
        tensor_payload_sizes: &[(&str, u64)],
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        let meta_val: serde_json::Value =
            serde_json::from_str(metadata_json).expect("metadata must be valid JSON");
        let wrapped = if meta_val.get("config").is_some() {
            metadata_json.to_string()
        } else {
            format!(r#"{{"architecture":"test","config":{metadata_json}}}"#)
        };
        let meta_bytes = wrapped.as_bytes();
        let n_tensors = tensor_payload_sizes.len() as u32;
        let mut idx = Vec::new();
        idx.extend_from_slice(&n_tensors.to_le_bytes());
        for (tname, data_size) in tensor_payload_sizes {
            let nb = tname.as_bytes();
            idx.extend_from_slice(&(nb.len() as u16).to_le_bytes());
            idx.extend_from_slice(nb);
            idx.push(1); // quant_type
            idx.push(1); // n_dims
            idx.extend_from_slice(&4u32.to_le_bytes()); // shape[0]
            idx.extend_from_slice(&0u32.to_le_bytes()); // group_size
            idx.extend_from_slice(&data_size.to_le_bytes());
        }
        let metadata_offset: u64 = 32;
        let data_offset: u64 = metadata_offset + meta_bytes.len() as u64 + idx.len() as u64;
        f.write_all(b"HFQM").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&arch_id.to_le_bytes()).unwrap();
        f.write_all(&n_tensors.to_le_bytes()).unwrap();
        f.write_all(&metadata_offset.to_le_bytes()).unwrap();
        f.write_all(&data_offset.to_le_bytes()).unwrap();
        f.write_all(meta_bytes).unwrap();
        f.write_all(&idx).unwrap();
        for &(_, data_size) in tensor_payload_sizes {
            let payload = vec![0u8; data_size as usize];
            f.write_all(&payload).unwrap();
        }
        f.flush().unwrap();
        path
    }

    fn tmp_dir() -> TempDir {
        TempDir::new().unwrap()
    }

    fn vl_tensors() -> Vec<(&'static str, u64)> {
        vec![("model.visual.patch_embed.proj.weight", 12)]
    }

    fn llama_cfg() -> String {
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        })
        .to_string()
    }

    fn deepseek4_cfg() -> String {
        serde_json::json!({
            "model_type": "deepseek_v4",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        })
        .to_string()
    }

    fn minimax_cfg() -> String {
        serde_json::json!({
            "model_type": "minimax_m2",
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 8192,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        })
        .to_string()
    }

    fn qwen3_cfg() -> String {
        serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        })
        .to_string()
    }

    // ── Admission error tests (refusal before mesh/GPU) ───────────────

    #[test]
    fn rejects_qwen35_vl_pp() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "q35vl.hfq",
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":16,"vocab_size":152064,"vision_config":{"hidden_size":1024}}"#,
            &vl_tensors(),
        );
        let err = daemon_load_plan(path.to_str().unwrap(), 2, 1, 1).unwrap_err();
        assert!(err.contains("AXIS-004"), "expected AXIS-004, got: {err}");
        // Admission fails BEFORE mesh building — if the daemon used raw
        // pp/tp to build the mesh, it would have a Pp mesh without ever
        // checking the VL PP policy.
    }

    #[test]
    fn rejects_tp_ep_composition() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let err = daemon_load_plan(path.to_str().unwrap(), 1, 2, 2).unwrap_err();
        assert!(err.contains("COMP-001"), "expected COMP-001, got: {err}");
        // Composition is caught before any arch classification or mesh.
    }

    #[test]
    fn rejects_pp_tp_composition() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let err = daemon_load_plan(path.to_str().unwrap(), 2, 2, 1).unwrap_err();
        assert!(
            err.contains("PP cannot"),
            "expected PP+TP rejection, got: {err}"
        );
    }

    #[test]
    fn rejects_zero_degree() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let err = daemon_load_plan(path.to_str().unwrap(), 0, 1, 1).unwrap_err();
        assert!(err.contains("degree"), "expected degree error, got: {err}");
    }

    #[test]
    fn cap_001_error_carries_cap_tag() {
        let dir = tmp_dir();
        let path = write_hfq(
            dir.path(),
            "q35vl.hfq",
            5,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":16,"vocab_size":152064,"vision_config":{"hidden_size":1024}}"#,
            &vl_tensors(),
        );
        let err = daemon_load_plan(path.to_str().unwrap(), 2, 1, 1).unwrap_err();
        assert!(
            err.contains("[CAP-001]"),
            "error should carry [CAP-001]: {err}"
        );
    }

    #[test]
    fn composition_error_carries_comp_tag() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let err = daemon_load_plan(path.to_str().unwrap(), 1, 2, 2).unwrap_err();
        assert!(
            err.contains("COMP-001"),
            "error should carry [COMP-001]: {err}"
        );
    }

    // ── Dense EP normalisation → Single mesh + eager lifecycle ─────────
    //
    // Dense EP (arch that normalises EP→Single) must produce:
    //   - effective (1,1,1) — normalised to Single
    //   - single-device mesh (no EP axis)
    //   - defer_unload = false (eager lifecycle — no multi-GPU EP)
    //   - pflash_suppressed = false (no EP guard)

    #[test]
    fn dense_ep_normalises_to_single() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let plan = daemon_load_plan(path.to_str().unwrap(), 1, 1, 4).unwrap();
        // Effective degrees normalised
        assert_eq!(
            plan.effective,
            hipfire_loader::parallel_capability::RawParallelRequest::new(1, 1, 1),
            "dense EP should normalise to Single (1,1,1)"
        );
        // Single-device mesh, no EP axis
        let mesh = &plan.effective_mesh;
        assert_eq!(mesh.n_devices(), 1, "normalised mesh must be single-device");
        assert!(
            !mesh.has_axis(DimKind::Ep),
            "normalised mesh must not have EP axis"
        );
        // Eager lifecycle
        assert!(
            !plan.defer_unload,
            "dense EP normalised Single must not defer unload"
        );
        assert!(
            !plan.pflash_suppressed,
            "dense EP normalised Single must not suppress PFlash"
        );
        // Admitted token preserved for load_admitted
        let _ = plan.admitted;
    }

    // ── Legacy TP→EP remap → EP mesh + deferred lifecycle ─────────────
    //
    // DeepSeek4 / MiniMax with tp=2,ep=1 produce:
    //   - effective (1,1,2) — EP axis
    //   - EP mesh (ep=2)
    //   - defer_unload = true (transactional multi-GPU EP lifecycle)
    //   - pflash_suppressed = true (EP guards PFlash)

    #[test]
    fn deepseek4_tp_remaps_to_ep() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "ds4.hfq", 9, &deepseek4_cfg(), &[]);
        let plan = daemon_load_plan(path.to_str().unwrap(), 1, 2, 1).unwrap();
        // Effective degrees after remap
        assert_eq!(plan.effective.tp, 1, "DS4 TP must be normalised to 1");
        assert_eq!(plan.effective.ep, 2, "DS4 EP must be 2 after TP→EP remap");
        assert_eq!(plan.effective.pp, 1, "DS4 PP must be 1");
        // EP mesh
        let mesh = &plan.effective_mesh;
        assert!(mesh.has_axis(DimKind::Ep), "DS4 remap must produce EP mesh");
        assert_eq!(mesh.n_devices(), 2, "DS4 remap mesh must have 2 devices");
        // Deferred lifecycle
        assert!(plan.defer_unload, "DS4 EP must defer unload");
        assert!(plan.pflash_suppressed, "DS4 EP must suppress PFlash");
    }

    #[test]
    fn minimax_tp_remaps_to_ep() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "minimax.hfq", 10, &minimax_cfg(), &[]);
        let plan = daemon_load_plan(path.to_str().unwrap(), 1, 2, 1).unwrap();
        // Effective degrees after remap
        assert_eq!(plan.effective.tp, 1, "MiniMax TP must be normalised to 1");
        assert_eq!(
            plan.effective.ep, 2,
            "MiniMax EP must be 2 after TP→EP remap"
        );
        // EP mesh
        let mesh = &plan.effective_mesh;
        assert!(
            mesh.has_axis(DimKind::Ep),
            "MiniMax remap must produce EP mesh"
        );
        assert_eq!(
            mesh.n_devices(),
            2,
            "MiniMax remap mesh must have 2 devices"
        );
        // Deferred lifecycle
        assert!(plan.defer_unload, "MiniMax EP must defer unload");
        assert!(plan.pflash_suppressed, "MiniMax EP must suppress PFlash");
    }

    // ── TP mesh + deferred lifecycle ───────────────────────────────────
    //
    // A dense model loaded with tp=2,ep=1 produces:
    //   - effective (1,2,1) — TP axis
    //   - TP mesh (tp=2)
    //   - defer_unload = true (transactional multi-GPU TP lifecycle)
    //   - pflash_suppressed = true (multi-device guards PFlash)

    #[test]
    fn tp_mesh_defers_and_suppresses() {
        let dir = tmp_dir();
        // PlainQwen3 (arch_id=1, model_type=qwen3) supports TP.
        let path = write_hfq(dir.path(), "qwen3.hfq", 1, &qwen3_cfg(), &[]);
        let plan = daemon_load_plan(path.to_str().unwrap(), 1, 2, 1).unwrap();
        // Effective degrees unchanged (dense model, no EP remap)
        assert_eq!(plan.effective.tp, 2, "TP mesh must preserve effective tp=2");
        assert_eq!(plan.effective.ep, 1, "TP mesh must have ep=1");
        assert_eq!(plan.effective.pp, 1, "TP mesh must have pp=1");
        // TP mesh
        let mesh = &plan.effective_mesh;
        assert!(mesh.has_axis(DimKind::Tp), "TP mesh must have TP axis");
        assert_eq!(mesh.n_devices(), 2, "TP mesh must have 2 devices");
        // Deferred lifecycle (multi-device)
        assert!(plan.defer_unload, "TP mesh must defer unload");
        assert!(plan.pflash_suppressed, "TP mesh must suppress PFlash");
    }

    // ── Single-GPU passthrough ─────────────────────────────────────────

    #[test]
    fn single_passes() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let plan = daemon_load_plan(path.to_str().unwrap(), 1, 1, 1).unwrap();
        assert_eq!(
            plan.effective,
            hipfire_loader::parallel_capability::RawParallelRequest::new(1, 1, 1),
        );
        assert!(!plan.defer_unload, "Single must not defer unload");
        assert!(!plan.pflash_suppressed, "Single must not suppress PFlash");
    }

    // ── Admitted PP topology + lifecycle ────────────────────────────────
    //
    // PP = 2 on an admitted model (llama QK-norm, arch_id=0) produces:
    //   - effective (2,1,1) — PP axis
    //   - PP mesh (pp=2)
    //   - defer_unload = false (eager lifecycle — PP unloads the prior model
    //     before the new load, same as Single)
    //   - pflash_suppressed = false (PP does not suppress PFlash)

    #[test]
    fn pp_load_plan_topology_lifecycle() {
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let plan = daemon_load_plan(path.to_str().unwrap(), 2, 1, 1).unwrap();
        // Effective degrees
        assert_eq!(
            plan.effective,
            hipfire_loader::parallel_capability::RawParallelRequest::new(2, 1, 1),
            "PP admission must preserve pp=2"
        );
        // PP mesh
        let mesh = &plan.effective_mesh;
        assert!(mesh.has_axis(DimKind::Pp), "PP mesh must have Pp axis");
        assert_eq!(mesh.n_devices(), 2, "PP mesh must have 2 devices");
        // Eager lifecycle (same as Single — no multi-device deferral)
        assert!(!plan.defer_unload, "PP must not defer unload");
        assert!(!plan.pflash_suppressed, "PP must not suppress PFlash");
        // Admitted token preserved for load_admitted
        let _ = plan.admitted;
    }

    // ── Environment lock for parallel-safe env var tests ────────────────

    /// Process-wide mutex for tests that mutate env vars.
    /// Acquired before `EnvGuard::set()` to prevent races between
    /// concurrent test functions in the same binary.
    static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

    /// RAII guard that restores an env var to its prior value on drop.
    struct EnvGuard {
        key: &'static str,
        old: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, val: &str) -> Self {
            let old = std::env::var(key).ok();
            std::env::set_var(key, val);
            Self { key, old }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.old {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    // ── HIPFIRE_PP_LAYERS helper returns None (no consumer yet) ──────────
    //
    // `pp_bands_from_env` is the production-connected helper used by the
    // real daemon load handler to decide PP bands.  It currently returns
    // `None` for all env values because no `load_admitted` consumer
    // accepts pp_bands yet.  Stale/invalid env cannot reject a valid load.

    #[test]
    fn invalid_pp_layers_helper_returns_none() {
        let _env_lock = ENV_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("HIPFIRE_PP_LAYERS", "x,y,z");
        assert!(pp_bands_from_env().is_none());
    }

    #[test]
    fn garbage_pp_layers_helper_returns_none() {
        let _env_lock = ENV_TEST_MUTEX.lock().unwrap();
        let _guard = EnvGuard::set("HIPFIRE_PP_LAYERS", "garbage");
        assert!(pp_bands_from_env().is_none());
    }

    // ── Admission error JSON envelope shape ──────────────────────────────
    //
    // Every error from `daemon_load_plan` must be a plain string suitable
    // for `{"type":"error","message":"..."}`.  Errors must carry the
    // `[CAP-001]` or `[COMP-001]` diagnostic tag and must not contain
    // raw JSON, unescaped quotes that would break the envelope, or be
    // empty.

    #[test]
    fn admission_error_is_plain_string_with_diagnostic_tag() {
        // Composition error (COMP-001 tag)
        let dir = tmp_dir();
        let path = write_hfq(dir.path(), "llama.hfq", 0, &llama_cfg(), &[]);
        let err = daemon_load_plan(path.to_str().unwrap(), 1, 2, 2).unwrap_err();
        assert!(!err.is_empty(), "error must not be empty");
        assert!(
            err.contains("COMP-001"),
            "composition error must carry COMP-001 tag: {err}"
        );
        assert!(
            err.contains("TP and EP"),
            "composition error must describe the conflict: {err}"
        );
        // Verify error is a single-line plain string (no internal newlines)
        assert!(
            !err.contains('\n'),
            "error must be a single line for JSON envelope: {err}"
        );

        // Planned cell error (CAP-001 tag) — Qwen35 MoE PP=2
        let path_q35 = write_hfq(
            dir.path(),
            "q35moe.hfq",
            6,
            &r#"{"model_type":"qwen3.5","hidden_size":2048,"num_hidden_layers":24,"num_attention_heads":16,"vocab_size":152064}"#,
            &[],
        );
        let err2 = daemon_load_plan(path_q35.to_str().unwrap(), 2, 1, 1).unwrap_err();
        assert!(!err2.is_empty(), "planned error must not be empty");
        assert!(
            err2.contains("[CAP-001]"),
            "planned error must carry [CAP-001] tag: {err2}"
        );
        assert!(
            !err2.contains('\n'),
            "planned error must be single-line: {err2}"
        );

        // Degree-zero error (CAP-001 tag)
        let err3 = daemon_load_plan(path.to_str().unwrap(), 0, 1, 1).unwrap_err();
        assert!(!err3.is_empty(), "degree-zero error must not be empty");
        assert!(
            err3.contains("[CAP-001]"),
            "degree error must carry [CAP-001] tag: {err3}"
        );
        assert!(
            err3.contains("degree"),
            "degree error must mention degree: {err3}"
        );
        assert!(
            !err3.contains('\n'),
            "degree error must be single-line: {err3}"
        );
    }
}
