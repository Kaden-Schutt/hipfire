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
// Used by generate_qwen35_mtp (native-MTP serve path, merged from spec-graph):
// it manually re-packs the Qwen35 bundle on every exit + re-opens the HFQ mmap.
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::emit_text::{currently_in_think, extract_tool_calls_from_text};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Instant;

use hipfire_hardware::DimKind;
use hipfire_loader::{AsstTurnCache, EpArch, EpState, LoadedModel, ModelState};
use hipfire_runtime::spec::{
    ClientEvent, EvictRetain, FinishSummary, PrefillOutcome, SpecEmit, StopReason,
};

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

// ─── Dual-run shadow-parity harness (Inc 1, god-struct-collapse) ──────────────
//
// `archdispatch_parity_enabled` gates a shadow second-pass through the
// refactored ArchDispatch path (Task 1.4 wires this into generation).
// `TokenTape` accumulates committed token IDs for one pass; `assert_token_parity`
// compares two tapes and panics with a precise divergence report on mismatch.
// The `--self-check-parity` CLI branch exercises these without a GPU.

#[allow(dead_code)]
fn archdispatch_parity_enabled() -> bool {
    std::env::var("HIPFIRE_ARCHDISPATCH_PARITY")
        .map(|v| v == "1")
        .unwrap_or(false)
}

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
        // Direct field read — mirrors every `m.arch_id` branch in the daemon.
        self.m.arch_id
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

    fn reset(&mut self, gpu: &mut rdna_compute::Gpu) {
        // Exact existing helper (daemon.rs:4105): handles both the pp>1
        // multi-GPU DeltaNet memset path and the single-GPU path; also
        // resets kv_cache.compact_offset. No logic duplicated here.
        reset_qwen35_recurrent(self.m, gpu);
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
        gpu: &mut rdna_compute::Gpu,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
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
        gpu: &mut rdna_compute::Gpu,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
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
            .map(|(name, required)| hipfire_arch_qwen35::grammar::ToolSchema {
                name: name.clone(),
                required: required.clone(),
            })
            .collect();
        Some(Box::new(Qwen35GrammarMatcher(
            hipfire_arch_qwen35::grammar::Matcher::new(schemas),
        )))
    }

    fn maybe_evict(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        seq_pos: usize,
    ) -> Result<Option<usize>, String> {
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

    fn maybe_adaptive_downshift(&mut self, gpu: &mut rdna_compute::Gpu, seq_pos: usize) {
        let m = &mut *self.m;
        let Some(ModelState::Qwen35(b)) = m.state.as_mut() else {
            return;
        };
        let kv = &mut b.kv_cache;
        // Stderr phase-label is unified here (the arm distinguishes prefill /
        // post-prefill / decode); logging is diagnostic and NOT part of token
        // parity (assert_token_parity compares committed token IDs only).
        if let Some(ad) = m.kv_adaptive.as_mut() {
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

    fn take_prefill_checkpoint(&mut self, gpu: &mut rdna_compute::Gpu, seq_pos: usize) {
        let m = &mut *self.m;
        let Some(ModelState::Qwen35(b)) = m.state.as_mut() else {
            return;
        };
        speculative::take_dn_checkpoint(
            &mut m.prefill_checkpoints,
            &mut b.dn_state,
            gpu,
            seq_pos,
            ckpt_interval(),
            ckpt_max(),
        );
    }

    fn abort_zero_recurrent(&mut self, gpu: &mut rdna_compute::Gpu) {
        let m = &mut *self.m;
        if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
            for s in &b.dn_state.s_matrices {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &b.dn_state.s_scales {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &b.dn_state.conv_states {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            b.kv_cache.compact_offset = 0;
        }
        // Co-resident Llama KV reset (arm does the same defensive no-op).
        if let Some(ModelState::Llama(b)) = m.state.as_mut() {
            b.kv.compact_offset = 0;
        }
    }

    fn sample(
        &self,
        gpu: &mut rdna_compute::Gpu,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
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
        self.m.seq_pos
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        self.m.seq_pos = seq_pos;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.conversation_tokens
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

    fn free_prefill_checkpoints(&mut self, gpu: &mut rdna_compute::Gpu) {
        free_checkpoints(&mut self.m.prefill_checkpoints, gpu);
    }

    fn ensure_decoded_vocab(&mut self) -> std::sync::Arc<Vec<String>> {
        if self.m.decoded_vocab.is_none() {
            let v: Vec<String> = {
                let tok = self.m.tokenizer.as_ref().unwrap();
                let n = tok.vocab_size();
                (0..n).map(|id| tok.decode(&[id as u32])).collect()
            };
            self.m.decoded_vocab = Some(std::sync::Arc::new(v));
        }
        self.m.decoded_vocab.clone().unwrap()
    }

    fn has_eviction(&self) -> bool {
        self.m.eviction.is_some()
    }

    fn physical_cap(&self) -> usize {
        self.m.physical_cap
    }

    fn eviction_window(&self) -> Option<usize> {
        self.m.eviction.as_ref().map(|ev| ev.budget() + ev.beta())
    }

    fn insert_asst_turn(&mut self, fp: u64, seq: Vec<u32>) {
        self.m.asst_turn_cache.insert(fp, seq);
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
        self.m.arch_id
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

    fn reset(&mut self, gpu: &mut rdna_compute::Gpu) {
        // Mirrors generate_qwen2's overflow-reset: Qwen2State cursor + daemon
        // bookkeeping. No GPU buffers to free (KV is overwritten on re-prefill).
        let _ = gpu;
        if let Some(ModelState::Qwen2(b)) = self.m.state.as_mut() {
            b.state.reset();
        }
        self.m.seq_pos = 0;
        self.m.conversation_tokens.clear();
    }

    fn prefill_forward(
        &mut self,
        gpu: &mut rdna_compute::Gpu,
        chunk: &[u32],
        seq_pos: usize,
    ) -> Result<(), String> {
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
        gpu: &mut rdna_compute::Gpu,
        token: u32,
        seq_pos: usize,
    ) -> Result<(), String> {
        let _ = seq_pos;
        let ModelState::Qwen2(b) = self.m.state.as_mut().ok_or("no state")? else {
            return Err("decode_step_forward: not a qwen2 bundle".into());
        };
        qwen2::forward_step(gpu, &b.weights, &b.config, &mut b.state, token)
            .map_err(|e| format!("qwen2 forward_step (decode): {e:?}"))
    }

    fn sample(
        &self,
        gpu: &mut rdna_compute::Gpu,
        cfg: &hipfire_runtime::sampler::SamplerConfig,
        vocab_size: usize,
        ngram_scope: &[u32],
        grammar_mask: Option<&[bool]>,
        rng_state: &mut u32,
    ) -> Result<u32, String> {
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
        self.m.seq_pos
    }

    fn set_seq_pos(&mut self, seq_pos: usize) {
        // Adopt the arch cursor as authority (== generate_qwen2's finalize
        // `m.seq_pos = state.next_pos`; equal to the driver's local for a
        // single-turn run, but exact for multi-turn).
        let _ = seq_pos;
        let np = if let Some(ModelState::Qwen2(b)) = self.m.state.as_ref() {
            b.state.next_pos
        } else {
            return;
        };
        self.m.seq_pos = np;
    }

    fn conversation_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.m.conversation_tokens
    }

    fn vocab_size(&self) -> usize {
        if let Some(ModelState::Qwen2(b)) = self.m.state.as_ref() {
            b.config.vocab_size
        } else {
            0
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Drain + free a DeltaNet checkpoint ring. `DeviceBuffer` has no `Drop`, so a
/// bare `Vec::clear()` orphans each snapshot's GPU buffers — the per-reset leak
/// that OOMs long-lived serves (hipMalloc-OOM after ~N independent requests).
/// Routes every drop through `DeltaNetSnapshot::free_gpu`.
fn free_checkpoints(
    cks: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    gpu: &mut rdna_compute::Gpu,
) {
    for (_, snap) in cks.drain(..) {
        snap.free_gpu(gpu);
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
            eprintln!("Kernel precompilation finished with {errors} failure(s) — the missing kernels will JIT on first use.");
        } else {
            eprintln!("Kernel precompilation done.");
        }
        return;
    }

    // --self-check-parity: verify the dual-run shadow-parity harness. Constructs
    // two identical TokenTapes, asserts parity (no panic expected), then exits 0.
    // No GPU or model needed. Used by CI / task harness to validate Inc 1.
    if args.iter().any(|a| a == "--self-check-parity") {
        let mut tape_a = TokenTape::default();
        let mut tape_b = TokenTape::default();
        for tok in [1u32, 42, 100, 999] {
            tape_a.push(tok);
            tape_b.push(tok);
        }
        assert_token_parity(&tape_a, &tape_b, "self-check");
        println!("parity self-check OK");
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

    while let Ok(daemon_msg) = msg_rx.recv() {
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
                // FIX #1 (transactional EP load): the unload of the prior model
                // is deferred for the EP (tp>1) path until AFTER the new load
                // succeeds, so a partial EP load failure leaves the prior model
                // intact (and load_model_ep's staging guard frees the partial
                // ranks). For the single-GPU / pp path the prior model is
                // unloaded eagerly here as before (load_model uses the daemon's
                // `gpu` directly, so it can't be deferred without a major
                // refactor). The multi-GPU SHARD degree (expert-parallel `ep` OR
                // tensor-parallel `tp`) is parsed authoritatively below; peek the
                // max here (either shard path defers the prior-model unload).
                let load_tp = {
                    let peek = |k: &str| {
                        msg.get("params")
                            .and_then(|p| p.get(k))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1) as usize
                    };
                    peek("ep").max(peek("tp"))
                };
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
                        hipfire_loader::unload_model(m, &mut gpu);
                    }
                }

                let path = msg.get("model").and_then(|v| v.as_str()).unwrap_or("");
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

                // MTP speculative decode config. `mtp_mode` gates weight
                // discovery at load time (off=skip, on=error-if-missing,
                // auto=scan+log). `mtp_k` sets the draft window size.
                let mtp_mode = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_mode"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto")
                    .to_string();
                let mtp_k: usize = msg
                    .get("params")
                    .and_then(|p| p.get("mtp_k"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as usize;

                // Model-free n-gram speculator config, forwarded by the CLI after
                // resolving the `speculation` selector + legacy knobs through the
                // config ladder (env > flag > per-model > global). `ngram_draft`
                // is the per-load enable; `ngram_k`/`ngram_min_count` tune the
                // drafter. The loader applies env-wins over these (so a directly
                // driven daemon with `HIPFIRE_NGRAM_DRAFT=1` still works). Absent
                // params leave the fields `None` → loader defaults / env.
                let spec_cfg = hipfire_runtime::loader_api::SpecLoadCfg {
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
                // Multi-GPU sharding, EP↔TP disentangled (PB-TP5 prep):
                //   `ep` = expert-parallel (Ep axis, MoE routed experts, load_model_ep),
                //   `tp` = tensor-parallel (Tp axis, dense row/col, load_model_tp).
                // Back-compat: a legacy `tp>1` on an EP-capable MoE arch (9/10)
                // means EP (the `--tp` serve flag historically drove EP), so `tp`
                // reads as "shard across N GPUs; the arch picks the axis".
                let tp = msg
                    .get("params")
                    .and_then(|p| p.get("tp"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let ep = msg
                    .get("params")
                    .and_then(|p| p.get("ep"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                // Peek the arch to route a legacy `tp` correctly (MoE → EP).
                let moe_arch = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(
                    msg.get("model").and_then(|v| v.as_str()).unwrap_or(""),
                ))
                .map(|h| matches!(h.arch_id, 9 | 10))
                .unwrap_or(false);
                let (ep, tp) = if ep > 1 {
                    (ep, 1) // explicit expert-parallel
                } else if tp > 1 && moe_arch {
                    (tp, 1) // legacy: --tp on a MoE arch == expert-parallel
                } else {
                    (1, tp) // dense: real tensor-parallel (Tp axis)
                };
                if (ep > 1 || tp > 1) && pp > 1 {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"tp/ep (tensor/expert-parallel) and pp (pipeline-parallel) are mutually exclusive; set only one."}}"#
                    );
                    let _ = stdout.flush();
                    continue;
                }
                if (ep > 1 || tp > 1) && draft_path.is_some() {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","message":"EP/TP serving (ep>1 or tp>1) does not support DFlash drafters in v1; reload without a draft."}}"#
                    );
                    let _ = stdout.flush();
                    continue;
                }
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

                // One mesh from the (already remapped + guarded) scalars. The
                // remap forces ep/tp mutually exclusive and the guard rejects
                // (ep|tp)>1 with pp>1, so at most one axis is >1 here — routing
                // is byte-identical to the old ep>1 / tp>1 / pp chain.
                // emulate=None is load-bearing: passing Some(_) would auto-promote
                // a plain serve on a HIPFIRE_EMULATE_GPUS box into EP-2.
                let mesh = hipfire_runtime::config::resolve_mesh(pp, tp, ep, None);
                // Dense llama-family (arch 0/1) + Pp axis → the P-C driver-owned
                // PP path (PpModel). qwen35 (5/6) keeps its hand-coded multi-path
                // via load_model below.
                let dense_llama_pp = mesh.has_axis(DimKind::Pp)
                    && hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(path))
                        .map(|h| matches!(h.arch_id, 0 | 1))
                        .unwrap_or(false);
                // Ragged PP bands (qwen35 arm only): parse + length-validate at
                // the edge. EP/TP are excluded by the earlier mutual-exclusion
                // guard, and dense llama-PP ignores HIPFIRE_PP_LAYERS (uniform).
                let pp_bands: Option<Vec<usize>> = if pp > 1 && !dense_llama_pp {
                    match hipfire_runtime::config::parse_pp_layers(
                        std::env::var("HIPFIRE_PP_LAYERS").ok(),
                        pp,
                    ) {
                        Ok(bands) => bands,
                        Err(msg) => {
                            let _ = writeln!(
                                stdout,
                                r#"{{"type":"error","message":"{}"}}"#,
                                msg
                            );
                            let _ = stdout.flush();
                            continue;
                        }
                    }
                } else {
                    None
                };
                let loaded = if mesh.has_axis(DimKind::Ep) {
                    hipfire_loader::load_model_ep(path, max_seq, &mesh)
                } else if mesh.has_axis(DimKind::Tp) {
                    hipfire_loader::load_model_tp(path, max_seq, &mesh)
                } else if dense_llama_pp {
                    hipfire_loader::load_model_pp(path, max_seq, &mesh)
                } else {
                    // NOT single-GPU only: this arm still carries qwen35 (arch 5/6)
                    // pp>1 pipeline-parallel serving. qwen35 PP: mesh carries pp
                    // via size_of(Pp); pp_bands carries the ragged HIPFIRE_PP_LAYERS
                    // split, parsed + length-validated at the edge above.
                    hipfire_loader::load_model(
                        path,
                        max_seq,
                        draft_path.as_deref(),
                        kv_mode_override.as_deref(),
                        kv_adaptive_override.as_deref(),
                        state_quant_override.as_deref(),
                        &cask,
                        &mesh,
                        pp_bands.as_deref(),
                        spec_cfg,
                        &mut gpu,
                    )
                };
                match loaded {
                    Ok(mut m) => {
                        // FIX #1 (deferred EP unload): the new EP model loaded
                        // successfully — NOW it's safe to free the prior model
                        // (single-GPU/pp models were already unloaded eagerly
                        // above; this branch only fires for the deferred tp>1
                        // path). The prior model's PFlash drafter (pflash_state)
                        // is part of that prior model, so it's torn down here in
                        // the same drainer-before-unload order used elsewhere:
                        // unload_drafter queues the drafter tensors into the
                        // pool, then unload_model drains it.
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
                            if let Some(old) = model.take() {
                                hipfire_loader::unload_model(old, &mut gpu);
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
                            _ => "qwen3",
                        };
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
                        let cache_capable = matches!(m.arch_id, 5 | 6 | 9 | 10 | 12);
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
                        // EP guard (load_tp > 1): the EP path serves through
                        // `generate_ep`, which bypasses PFlash entirely (the
                        // EP archs ds4/minimax refuse/ignore PFlash drafters).
                        // Loading a drafter here would just pin GPU memory it
                        // never reads until unload, so skip the load outright.
                        // Warn once if the operator actually supplied a drafter
                        // so the silent no-op is visible.
                        if load_tp > 1 {
                            if pflash_drafter.is_some() && pflash_mode_str != "off" {
                                eprintln!(
                                    "[pflash] WARN: ignoring PFlash drafter on EP (tp={}) model \
                                     — generate_ep bypasses PFlash; drafter would only waste GPU memory",
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
                // the .hfq `generation_config` (m.rec_temperature/m.rec_top_p,
                // populated at load time via HfqFile::recommended_sampling) take
                // precedence over this ladder; an explicit per-request field
                // (set below via `msg.get(...)`) overrides both. The CLI's
                // curated registry `recommended_settings` reach this handler as
                // explicit request fields (CLI explicit-send guard), so they sit
                // above the .hfq layer on that path.
                let (arch_default_temp, arch_default_top_p) = if m.arch_id == 11 {
                    // LFM2.5 (11): Liquid's model card recommends temperature=0.1,
                    // top_k=50, repetition_penalty=1.05. The daemon sampler is
                    // temp + top_p + repeat_penalty (no user-facing top_k — the
                    // sample_top_p kernel's top-K is a fixed candidate gather), so
                    // we apply temp=0.1 + rep=1.05 (set below) and keep a tight
                    // top_p=0.80; at temp 0.1 the top_k-vs-top_p choice is near
                    // moot (the distribution is already peaked).
                    (0.1_f64, 0.80_f64)
                } else if m.arch_id == 9 {
                    // DeepSeek V4 Flash (9): MTP spec-decode is greedy-only and,
                    // since the k=2 + K+1 shared-accept-core work, ~3× faster than
                    // AR (measured 81.7% accept / 24 vs 7.8 tok/s on the
                    // deepseek4-mtp code bench). Default to temp=0 so an omitted
                    // `temperature` gets that spec speedup; explicit temp>0 is
                    // still honored and routes to the AR sampler (spec is
                    // greedy-only). Was 1.0 to dodge block-level attractors at low
                    // temp — coherence-gate-deepseek4-mtp re-validates greedy.
                    (0.0_f64, 1.0_f64)
                } else if m.arch_id == 10 {
                    // MiniMax-M2 (10): quantized instruct model that falls into
                    // block-level attractors at lower temperatures — keep the
                    // card-recommended temp=1.0/top_p=1.0.
                    (1.0_f64, 1.0_f64)
                } else if m.arch_id == 12 {
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
                let default_repeat_penalty = if m.arch_id == 11 { 1.05_f64 } else { 1.0_f64 };
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
                let is_dots_ocr = m.arch_id == 8;
                let has_vl = m.vision_config.is_some() || is_dots_ocr;

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
                        free_checkpoints(&mut m.prefill_checkpoints, &mut gpu);
                        free_checkpoints(&mut m.dflash_checkpoints, &mut gpu);
                        // The DFlash checkpoint ring now lives inside the
                        // speculator (m.dflash_checkpoints is vestigial/empty),
                        // so free THAT ring on conversation reset too — else its
                        // GPU snapshots persist until the next prefill-miss.
                        if let Some(s) = m.speculator.as_mut() {
                            s.reset(&mut gpu);
                        }
                        // qwen35(-vl) recurrent state lives in the bundle
                        // (ModelState::Qwen35), not the always-None
                        // m.dn_state/m.kv_cache direct fields.
                        reset_qwen35_recurrent(m, &mut gpu);
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
                        if let Some(ref mut ad) = m.kv_adaptive {
                            ad.reset();
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
                    );
                }
            }

            "reset" => {
                // Reset conversation state without unloading the model.
                if let Some(ref mut m) = model {
                    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
                        eprintln!("[qwen-cache RESET] daemon received reset — clearing conversation_tokens (was {})", m.conversation_tokens.len());
                    }
                    model_reset_context(m, &mut gpu);
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
                    .map(|m| match m.arch_id {
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
                if m.pp > 1 || m.ep.is_some() || m.tp.is_some() {
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
                if n.saturating_add(32) > m.physical_cap {
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
                // prefill-on-top-of-prior-state. qwen35 recurrent state lives in
                // the bundle (ModelState::Qwen35), not the always-None m.dn_state.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                reset_qwen35_recurrent(m, &mut gpu);
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

                // Flush any residual GPU work so it doesn't bleed into the
                // measured interval, then time forward_prefill_batch + a
                // trailing device_synchronize so we capture actual GPU
                // completion (kernel launches are async by default).
                let _ = gpu.hip.device_synchronize();
                let t0 = Instant::now();
                let run_ok = if m.arch_id == 5 || m.arch_id == 6 {
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
                } else if m.arch_id == 7 {
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
                } else if m.arch_id == 9 {
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
                } else if m.arch_id == 11 {
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
                } else if m.arch_id == 10 {
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
                } else if m.arch_id == 12 {
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

                // Reset state AFTER measurement — we've written N KV slots and a
                // DeltaNet state that the next real request must not inherit.
                // qwen35 recurrent state lives in the bundle (ModelState::Qwen35),
                // not the always-None m.dn_state.
                m.seq_pos = 0;
                m.conversation_tokens.clear();
                reset_qwen35_recurrent(m, &mut gpu);
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

/// A dense llama-family model served by the shared [`generate_dense`] loop: a
/// per-token forward + logits over its own multi-GPU state. Implemented by both
/// `TpModel` (Tp axis, PB-TP5) and `PpModel` (Pp axis, P-C) so the daemon's decode
/// loop is agnostic to which parallelism axis the model is served on. (Inherent
/// methods are called via the fully-qualified path so the trait forwarders don't
/// self-recurse.)
trait DenseServed {
    fn forward_token(&mut self, token: u32, pos: usize) -> Result<(), String>;
    fn logits(&mut self) -> Result<Vec<f32>, String>;
    fn eos_token(&self) -> u32;
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
    fn eos_token(&self) -> u32 {
        hipfire_runtime::tp_serve::TpModel::eos_token(self)
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
    fn eos_token(&self) -> u32 {
        hipfire_runtime::pp_serve::PpModel::eos_token(self)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<(), String> {
        hipfire_runtime::pp_serve::PpModel::prefill(self, tokens)
    }
}

/// Dense multi-GPU serve (PB-TP5 / P-C). Serves a llama-family model over ANY
/// parallelism axis (`TpModel` or `PpModel` via [`DenseServed`]): ChatFrame-render
/// the prompt, prefill it token-by-token, then greedy-or-sampled decode. Prefills
/// `prefill_tokens` at `[start_pos..]` (cache miss ⇒ start_pos 0, batched; cache
/// hit ⇒ per-token suffix over the cached KV prefix) and returns the generated
/// token ids for the caller's `conversation_tokens` bake. Lean by design:
/// no spec-decode / PFlash / eviction / grammar / tools. Validated argmax-exact vs
/// single-GPU (`tp_decode_parity` / `pp_decode_parity`).
#[allow(clippy::too_many_arguments)]
fn generate_dense<M: DenseServed>(
    model: &mut M,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    stdout: &mut std::io::Stdout,
    id: &str,
    prefill_tokens: &[u32],
    start_pos: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: Option<f32>,
    max_tokens: usize,
    stop: &[String],
) -> Option<Vec<u32>> {
    use hipfire_runtime::sampler::{self, SamplerConfig};
    let t0 = std::time::Instant::now();
    let prefill_n = prefill_tokens.len();
    let eos = model.eos_token();

    macro_rules! fail {
        ($e:expr) => {{
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":{}}}"#,
                id,
                serde_json::to_string(&format!("dense serve: {}", $e)).unwrap_or_default()
            );
            let _ = stdout.flush();
            return None;
        }};
    }

    // Prefill. Cache miss (start_pos==0): batched fast path. Cache hit
    // (start_pos>0): per-token suffix over the cached KV prefix [0..start_pos]
    // — forward_token attends over the retained prefix, so this is exact.
    let prefill_res = if start_pos == 0 {
        model.prefill(prefill_tokens)
    } else {
        let mut r = Ok(());
        for (i, &t) in prefill_tokens.iter().enumerate() {
            if let Err(e) = model.forward_token(t, start_pos + i) {
                r = Err(e);
                break;
            }
        }
        r
    };
    if let Err(e) = prefill_res {
        fail!(e);
    }
    let t_prefill = std::time::Instant::now();
    let prefill_ms = t_prefill.duration_since(t0).as_secs_f64() * 1000.0;

    let cfg = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty: 1.0,
        repeat_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        top_k,
        min_p,
    };
    let mut logits = match model.logits() {
        Ok(l) => l,
        Err(e) => fail!(e),
    };
    let mut history: Vec<u32> = Vec::new();
    // Tokens actually materialized into the KV (each pushed right after its
    // forward_token). On a max_tokens/stop break the last emitted token is in
    // `history` but was NOT forwarded, so returning `history` would over-count
    // conversation_tokens by one vs the dense KV (a #462-class mirror skew).
    let mut materialized: Vec<u32> = Vec::new();
    let mut next = sampler::sample_cpu(&mut logits, &history, &cfg);
    let mut generated = 0usize;
    let mut pos = start_pos + prefill_n;
    loop {
        if next == eos || tokenizer.is_terminator(next) {
            break;
        }
        let text = tokenizer.decode(&[next]);
        let _ = writeln!(
            stdout,
            r#"{{"type":"token","id":"{}","text":{}}}"#,
            id,
            serde_json::to_string(&text).unwrap_or_default()
        );
        let _ = stdout.flush();
        history.push(next);
        generated += 1;
        if generated >= max_tokens {
            break;
        }
        if !stop.is_empty() {
            let suffix = tokenizer.decode(&history);
            if stop
                .iter()
                .any(|s| !s.is_empty() && suffix.ends_with(s.as_str()))
            {
                break;
            }
        }
        if let Err(e) = model.forward_token(next, pos) {
            fail!(e);
        }
        materialized.push(next);
        pos += 1;
        logits = match model.logits() {
            Ok(l) => l,
            Err(e) => fail!(e),
        };
        next = sampler::sample_cpu(&mut logits, &history, &cfg);
    }

    let decode_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let decode_tok_s = if decode_ms > 0.0 {
        generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"decode_tok_s":{:.1}}}"#,
        id, generated, decode_tok_s, prefill_n, prefill_ms, decode_tok_s
    );
    let _ = stdout.flush();
    Some(materialized)
}

fn generate_ep(
    m: &mut LoadedModel,
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
) {
    // ── Canonical multi-turn render via the arch's trained chat_template
    // (ds4/minimax). Mirrors generate_minimax: `messages_history` (the full
    // conversation, live user last) → render_messages with `tools` threaded;
    // falls back to a synthesized [system?, user] turn when no history is
    // supplied. The trim_blocks/lstrip_blocks env (prompt_frame) keeps the
    // structural prefix history-invariant so the EP LCP cache below can hit.
    // `primed_think` records whether the render ended on the MiniMax `<think>`
    // generation primer (re-emitted display-only in ep_serve_minimax). ──
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = if m.arch_id == 9 {
        primed_think = false;
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let eos_tok = m.deepseek4_eos_tok;
        build_deepseek4_dsml_prompt(
            tokenizer,
            system_prompt,
            tools,
            messages_history,
            prompt,
            think_mode,
            eos_tok,
            &mut m.asst_turn_cache,
        )
    } else {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        if let Some(template) = m.chat_template.as_ref() {
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
                    return;
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
            m.arch_id,
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
        return;
    }
    let eos_tok = if m.arch_id == 10 {
        // MiniMax EP state lives in `m.ep`, not `m.state`, so `minimax()` is
        // None here — read the EP eos carried on LoadedModel (set at load).
        m.minimax_eos_tok
    } else {
        m.deepseek4_eos_tok
    };
    match m.arch_id {
        10 => ep_serve_minimax(
            m,
            stdout,
            id,
            &prompt_ids,
            eos_tok,
            max_tokens,
            stop,
            primed_think,
            sampling,
        ),
        _ => ep_serve_ds4(
            m,
            stdout,
            id,
            &prompt_ids,
            eos_tok,
            max_tokens,
            think_mode,
            tools,
            stop,
            sampling,
        ),
    }
}

/// Stream a token JSON event; returns true if a stop sequence is now satisfied.
fn ep_emit_token(
    stdout: &mut std::io::Stdout,
    id: &str,
    piece: &str,
    text_acc: &mut String,
    stop: &[String],
) -> bool {
    text_acc.push_str(piece);
    let _ = writeln!(
        stdout,
        r#"{{"type":"token","id":"{}","text":{}}}"#,
        id,
        serde_json::to_string(piece).unwrap_or_else(|_| "\"\"".to_string())
    );
    let _ = stdout.flush();
    stop.iter().any(|s| !s.is_empty() && text_acc.ends_with(s))
}

fn ep_emit_done(
    stdout: &mut std::io::Stdout,
    id: &str,
    generated: usize,
    prompt_n: usize,
    prefill_ms: f64,
    decode_ms: f64,
) {
    let decode_tok_s = if decode_ms > 0.0 {
        generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_ms > 0.0 {
        prompt_n as f64 / (prefill_ms / 1000.0)
    } else {
        0.0
    };
    eprintln!("[daemon] EP generate done: {generated} tok, {decode_tok_s:.1} tok/s");
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1}}}"#,
        id, generated, decode_tok_s, prompt_n, prefill_ms, prefill_tok_s, decode_tok_s, prefill_ms
    );
    let _ = stdout.flush();
}

/// FIX #3 (ep-no-abort): reset every EP rank's KV cursor to a clean state and
/// clear the shared conversation history. The mid-generation per-rank KV /
/// position is advanced past the (un-committed) `conversation_tokens`, so the
/// next turn must cold-start; both arch states expose `reset()` which rewinds
/// the per-rank token cursor (`n_tokens`). Mirrors the single-GPU abort reset
/// (`m.seq_pos = 0; m.conversation_tokens.clear();`).
fn ep_reset_after_abort(m: &mut LoadedModel) {
    if let Some(ep) = m.ep.as_mut() {
        match &mut ep.inner {
            EpArch::Ds4 { state, .. } => {
                for s in state.iter_mut() {
                    s.reset();
                }
            }
            EpArch::Minimax { state, .. } => {
                for s in state.iter_mut() {
                    s.reset();
                }
            }
        }
    }
    m.seq_pos = 0;
    m.conversation_tokens.clear();
}

/// FIX #3: emit the standard `aborted` + `done(finish_reason=aborted)` event
/// pair (mirrors the single-GPU AR abort path) then reset EP state.
fn ep_emit_abort(
    stdout: &mut std::io::Stdout,
    id: &str,
    m: &mut LoadedModel,
    completion_tokens: usize,
) {
    let _ = writeln!(
        stdout,
        r#"{{"type":"aborted","id":"{}","reason":"client_cancelled"}}"#,
        id
    );
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":{},"prefill_ms":0,"decode_ms":0}}"#,
        id, completion_tokens
    );
    let _ = stdout.flush();
    ep_reset_after_abort(m);
}

/// ds4 EP prefill + greedy decode.
fn ep_serve_ds4(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    eos_tok: u32,
    max_tokens: usize,
    think_mode: ThinkMode,
    tools: Option<&[serde_json::Value]>,
    stop: &[String],
    sampling: EpSampling,
) {
    use hipfire_arch_deepseek4::dsml::StreamEvent;
    use std::time::Instant;

    let prompt_n = prompt_ids.len();

    // O2b-2 capacity guard (ds4 EP): this path replays the full prompt from
    // position 0 every turn (no LCP reuse), so the absolute KV span is
    // prompt_n + max_tokens. Without eviction the EP state KV was allocated
    // for `m.physical_cap` (== max_seq at load). Overrunning it drives
    // forward_ep past the KV buffer → corruption/panic (serve-wide crash).
    // Emit a clean error and return BEFORE prefill — mirror the qwen35 guard.
    // saturating_add: an adversarially huge max_tokens must not wrap usize and
    // slip under the cap.
    if prompt_n.saturating_add(max_tokens) > m.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id, prompt_n, max_tokens, m.physical_cap
        );
        let _ = stdout.flush();
        return;
    }

    // ── Cross-conversation reset (FIX: ds4 EP turn-to-turn contamination) ──
    // ds4 EP replays the full prompt from position 0 every turn (no LCP reuse —
    // see the capacity-guard comment above), so a CONTINUING conversation
    // re-prefills its whole history straight from the prompt; nothing in the EP
    // state is reusable across turns, so we reset unconditionally here. This is
    // the EP analogue of the single-GPU cache-miss reset in generate_deepseek4:
    // `reset()` alone only rewinds n_tokens — the position-indexed decode caches
    // (SWA ring, compressed/full KV, indexer scratch) retain the PRIOR turn's
    // residue and bleed into the next conversation (observed: turn 2 echoing
    // turn 1's answer) unless explicitly zeroed. Do it per rank on its own
    // device. Without this, ep_serve_ds4 only ever reset on abort (and even that
    // path, ep_reset_after_abort, omitted zero_decode_caches).
    if let Some(ep) = m.ep.as_mut() {
        let EpState { gpus, inner } = ep;
        if let EpArch::Ds4 { state, .. } = inner {
            for (rank, s) in state.iter_mut().enumerate() {
                let g = &mut gpus.devices[rank];
                let _ = g.bind_thread();
                s.reset();
                s.zero_decode_caches(g);
                g.invalidate_graph_state();
            }
        }
    }
    m.seq_pos = 0;
    m.conversation_tokens.clear();

    let mut parser = match think_mode {
        ThinkMode::High | ThinkMode::Max => deepseek4::dsml::StreamParser::new_in_think(),
        ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
    };
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
    let decoded_vocab_arc: Option<std::sync::Arc<Vec<String>>> = if grammar_active {
        if m.decoded_vocab.is_none() {
            let tokenizer = m.tokenizer.as_ref().unwrap();
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
    let mut emit_text_buf = String::new();
    let mut emit_tool_calls_buf: Vec<hipfire_runtime::prompt_frame::ToolCall> = Vec::new();
    let mut absorb_event = |ev: &StreamEvent| match ev {
        StreamEvent::Token(t) => emit_text_buf.push_str(t),
        StreamEvent::Reasoning(_) => {}
        StreamEvent::ToolCalls(calls) => {
            for c in calls {
                emit_tool_calls_buf.push(hipfire_runtime::prompt_frame::ToolCall {
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                });
            }
        }
    };

    let t_prefill = Instant::now();
    // FIX #1 (ep-prefill-abort): set when check_abort fires inside the prefill
    // loop. Declared outside the borrow scope so the post-loop abort guard can
    // read it after the `gpus`/`state` borrow is dropped.
    let mut aborted_in_prefill = false;
    {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Ds4 {
            config,
            weights,
            state,
            partials,
            partials_i64,
        } = inner
        else {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"EP arch mismatch (expected ds4)"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        };
        for (pos, &t) in prompt_ids.iter().enumerate() {
            // FIX #1 (ep-prefill-abort): check the cancel signal at the TOP of
            // every prefill iteration, not just after the loop. A long prompt
            // (thousands of tokens) means the post-loop check below would still
            // run the entire multi-GPU prefill before honoring a cancel. Mirror
            // the decode loop: on abort, emit aborted+done, reset KV cursors,
            // and stop. We must drop the `gpus`/`state` borrow before calling
            // `ep_emit_abort` (which re-borrows `m.ep`), so break out and let
            // the post-loop guard fire — but set the abort flag is consumed by
            // check_abort, so call it here and short-circuit via a flag.
            if check_abort(id) {
                // Drop the EpState borrow by breaking; the post-loop guard
                // re-checks via a sentinel. Simpler: emit + return is blocked
                // by the borrow, so we set `aborted` and break.
                aborted_in_prefill = true;
                break;
            }
            if let Err(e) = deepseek4::forward::forward_ep(
                gpus,
                weights,
                config,
                state,
                partials,
                partials_i64,
                t,
                pos as u32,
            ) {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"forward_ep prefill: {}"}}"#,
                    id,
                    format!("{e}").replace('"', "'")
                );
                let _ = stdout.flush();
                return;
            }
        }
    }
    // FIX #1 / FIX #3 (ep-no-abort): a client cancel during the (potentially
    // long) prefill should stop here instead of running the whole decode loop.
    // `aborted_in_prefill` is set when check_abort fired mid-loop (it already
    // consumed the signal, so we don't re-call check_abort); the post-loop
    // check_abort catches a cancel that arrived after the final iteration.
    // Mirror the single-GPU paths: emit aborted+done and reset every rank's KV
    // cursor.
    if aborted_in_prefill || check_abort(id) {
        ep_emit_abort(stdout, id, m, 0);
        return;
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let mut logits = {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Ds4 { state, .. } = inner else {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"EP arch mismatch (expected ds4)"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        };
        let _ = gpus.devices[0].bind_thread();
        match state[0].logits.as_ref() {
            // FIX #4 (ep-download-swallow): on a download failure, emit a JSON
            // error and STOP — never `unwrap_or_default()` into an all-zero
            // logits vec (argmax → token 0, an undetectable corruption).
            Some(l) => match gpus.devices[0].download_f32(l) {
                Ok(v) => v,
                Err(e) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","id":"{}","message":"EP first-logits download failed: {}"}}"#,
                        id,
                        format!("{e:?}").replace('"', "'")
                    );
                    let _ = stdout.flush();
                    return;
                }
            },
            None => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"EP logits unset after prefill"}}"#,
                    id
                );
                let _ = stdout.flush();
                return;
            }
        }
    };

    let t_decode = Instant::now();
    let mut generated = 0usize;
    let mut pos = prompt_n;
    let mut text_acc = String::new();
    let mut local_emitted_ids: Vec<u32> = Vec::new();
    while generated < max_tokens {
        // FIX #3 (ep-no-abort): client cancel mid-decode → emit aborted+done,
        // reset EP cursors, stop. Without this a Pi/CLI cancel leaves the EP
        // decode loop running for the full max_tokens of wasted multi-GPU work.
        if check_abort(id) {
            ep_emit_abort(stdout, id, m, generated);
            return;
        }
        if grammar_active && !matcher.is_free() {
            matcher.token_mask(decoded_vocab, &mut grammar_mask);
            deepseek4::grammar::Matcher::apply_mask_to_logits(&grammar_mask, &mut logits);
        }
        // Host-side sampler over the downloaded f32 logits (temp → top_k →
        // top_p → min_p → seeded draw, temp<=1e-6 = argmax). RNG seeded once
        // per request via reset_cpu_sampler_rng(0x13579BDF) in generate().
        let next = hipfire_runtime::llama::sample_full_dist(
            &logits,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
        );
        if next == eos_tok {
            break;
        }
        let piece = m.tokenizer.as_ref().unwrap().decode(&[next]);
        for ev in parser.feed(&piece) {
            absorb_event(&ev);
            emit_stream_event(stdout, id, ev);
        }
        emit_committed_event(
            stdout,
            id,
            next,
            generated,
            t_decode.elapsed().as_millis() as u64,
        );
        let _ = stdout.flush();
        if grammar_active {
            matcher.advance(&piece);
        }
        local_emitted_ids.push(next);
        text_acc.push_str(&piece);
        generated += 1;
        if stop.iter().any(|s| !s.is_empty() && text_acc.ends_with(s)) {
            break;
        }
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Ds4 {
            config,
            weights,
            state,
            partials,
            partials_i64,
        } = inner
        else {
            break;
        };
        if let Err(e) = deepseek4::forward::forward_ep(
            gpus,
            weights,
            config,
            state,
            partials,
            partials_i64,
            next,
            pos as u32,
        ) {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"forward_ep decode: {}"}}"#,
                id,
                format!("{e}").replace('"', "'")
            );
            return;
        }
        pos += 1;
        let _ = gpus.devices[0].bind_thread();
        // FIX #4 (ep-download-swallow): explicit error handling on the per-token
        // logits download — emit a JSON error and stop, never feed a zeroed
        // (token-0) logits vec.
        logits = match state[0].logits.as_ref() {
            Some(l) => match gpus.devices[0].download_f32(l) {
                Ok(v) => v,
                Err(e) => {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","id":"{}","message":"EP decode logits download failed: {}"}}"#,
                        id,
                        format!("{e:?}").replace('"', "'")
                    );
                    let _ = stdout.flush();
                    return;
                }
            },
            None => break,
        };
    }
    for ev in parser.finish() {
        absorb_event(&ev);
        emit_stream_event(stdout, id, ev);
    }
    let _ = stdout.flush();
    drop(absorb_event);

    let finish_reason: &'static str = if !emit_tool_calls_buf.is_empty() {
        "tool_calls"
    } else if generated >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let have_replayable_payload =
        !emit_text_buf.trim().is_empty() || !emit_tool_calls_buf.is_empty();
    if have_replayable_payload && generated > 0 && !local_emitted_ids.is_empty() {
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
                local_emitted_ids.len(),
            );
        }
        m.asst_turn_cache.insert(fp, local_emitted_ids);
    }

    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let decode_tok_s = if decode_ms > 0.0 {
        generated as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };
    let prefill_tok_s = if prefill_ms > 0.0 {
        prompt_n as f64 / (prefill_ms / 1000.0)
    } else {
        0.0
    };
    eprintln!("[daemon] EP generate done: {generated} tok, {decode_tok_s:.1} tok/s");
    let done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": decode_tok_s,
        "prefill_tokens": prompt_n,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "ttft_ms": prefill_ms,
        "finish_reason": finish_reason,
    });
    let _ = writeln!(stdout, "{}", done);
    let _ = stdout.flush();
}

/// MiniMax-M2 EP prefill + greedy decode (mirror of ep_serve_ds4, MiniMax types).
/// Carries the single-GPU prefix cache to EP: an LCP over the shared
/// `conversation_tokens` rewinds every rank's KV cursor to the common prefix
/// and re-prefills only the divergent suffix (interleaved-thinking partial
/// reuse — see generate_minimax for the full rationale). `primed_think`
/// re-emits the MiniMax `<think>\n` opener display-only for a well-formed turn.
#[allow(clippy::too_many_arguments)]
fn ep_serve_minimax(
    m: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    eos_tok: u32,
    max_tokens: usize,
    stop: &[String],
    primed_think: bool,
    sampling: EpSampling,
) {
    use std::time::Instant;
    let prompt_n = prompt_ids.len();

    // O2b-2 capacity guard (minimax EP): even with LCP reuse the KV ends up
    // holding [0, prompt_n) after prefill, then decode appends max_tokens, so
    // the absolute span is prompt_n + max_tokens. The EP state KV was allocated
    // for `m.physical_cap` (== max_seq at load); overrunning it writes past the
    // per-rank KV buffer → corruption/panic (serve-wide crash). Emit a clean
    // error and return BEFORE any state mutation — mirror the qwen35 guard.
    // saturating_add: an adversarially huge max_tokens must not wrap usize and
    // slip under the cap.
    if prompt_n.saturating_add(max_tokens) > m.physical_cap {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id, prompt_n, max_tokens, m.physical_cap
        );
        let _ = stdout.flush();
        return;
    }

    // ── LCP partial reuse. The per-rank KV holds [0, prior_total) from last
    // turn; `conversation_tokens` mirrors it. Rewind n_tokens to the common
    // prefix and re-prefill the (reasoning-free, shorter) suffix; MiniMax is
    // standard attention so KV ≥ lcp is overwritten and the stale tail is
    // never attended. lcp == 0 ⇒ cold prefill. ──
    let prefill_from: usize = {
        let prior_len = m.conversation_tokens.len();
        let max_match = prior_len.min(prompt_n);
        let mut lcp = 0usize;
        while lcp < max_match && m.conversation_tokens[lcp] == prompt_ids[lcp] {
            lcp += 1;
        }
        let cache_hit = lcp > 0 && lcp < prompt_n;
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[minimax-ep-cache] prior_len={} rendered_len={} lcp={} hit={} partial={}",
                prior_len,
                prompt_n,
                lcp,
                cache_hit,
                cache_hit && lcp < prior_len,
            );
        }
        if cache_hit {
            m.conversation_tokens.truncate(lcp);
            lcp
        } else {
            m.conversation_tokens.clear();
            0
        }
    };
    // Rewind every rank's KV cursor to the reuse point.
    {
        let EpState { inner, .. } = m.ep.as_mut().unwrap();
        if let EpArch::Minimax { state, .. } = inner {
            for s in state.iter_mut() {
                s.n_tokens = prefill_from;
            }
        }
    }

    // ── Prefill the suffix [prefill_from, prompt_n) across ranks. ──
    let t_prefill = Instant::now();
    // FIX #1 (ep-prefill-abort): set when check_abort fires inside the prefill
    // loop. Declared outside the borrow scope so the abort guard below can read
    // it after the `gpus`/`state` borrow is dropped.
    let mut aborted_in_prefill = false;
    // Track how many suffix tokens were actually prefilled so that, on a
    // mid-prefill abort, we don't mirror un-prefilled tokens into
    // `conversation_tokens` (which would desync the cache from KV state).
    let mut prefilled_n = 0usize;
    {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Minimax {
            config,
            weights,
            state,
            partials,
            partials_i64,
        } = inner
        else {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"EP arch mismatch (expected minimax)"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        };
        for (i, &t) in prompt_ids[prefill_from..].iter().enumerate() {
            // FIX #1 (ep-prefill-abort): honor a client cancel at the TOP of
            // every prefill iteration, not just after the loop. Without this a
            // long prompt runs the full multi-GPU prefill before the post-loop
            // check fires. check_abort consumes the signal, so record it in a
            // flag and break; the borrow on `m.ep` is dropped before we call
            // ep_emit_abort below.
            if check_abort(id) {
                aborted_in_prefill = true;
                break;
            }
            let pos = (prefill_from + i) as u32;
            if let Err(e) = minimax::forward::forward_ep(
                gpus,
                weights,
                config,
                state,
                partials,
                partials_i64,
                t,
                pos,
            ) {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"forward_ep prefill: {}"}}"#,
                    id,
                    format!("{e}").replace('"', "'")
                );
                let _ = stdout.flush();
                return;
            }
            prefilled_n = i + 1;
        }
    }
    // Mirror only the actually-prefilled suffix into conversation_tokens (the
    // prefix is kept). On a mid-prefill abort, ep_emit_abort resets every
    // rank's KV cursor, so leaving conversation_tokens at the prefix keeps the
    // cache consistent with KV state.
    for &t in &prompt_ids[prefill_from..prefill_from + prefilled_n] {
        m.conversation_tokens.push(t);
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    // FIX #1 / FIX #3 (ep-no-abort): client cancel during prefill → stop
    // cleanly. `aborted_in_prefill` already consumed the signal mid-loop; the
    // post-loop check_abort catches a cancel that arrived after the last token.
    if aborted_in_prefill || check_abort(id) {
        ep_emit_abort(stdout, id, m, 0);
        return;
    }

    // MiniMax primes the assistant with `<think>\n`; re-emit display-only so the
    // assistant message is a well-formed think block (parity with single-GPU).
    if primed_think {
        let _ = writeln!(
            stdout,
            "{}",
            serde_json::json!({"type":"token","id":id,"text":"<think>\n"})
        );
        let _ = stdout.flush();
    }

    let mut logits = {
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Minimax { state, .. } = inner else {
            return;
        };
        let _ = gpus.devices[0].bind_thread();
        // FIX #4 (ep-download-swallow): explicit error handling — never feed a
        // zeroed (token-0) logits vec on a download failure.
        match gpus.devices[0].download_f32(&state[0].logits) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"EP first-logits download failed: {}"}}"#,
                    id,
                    format!("{e:?}").replace('"', "'")
                );
                let _ = stdout.flush();
                return;
            }
        }
    };
    let t_decode = Instant::now();
    let mut generated = 0usize;
    let mut pos = prompt_n;
    let mut text_acc = String::new();
    while generated < max_tokens {
        // FIX #3 (ep-no-abort): client cancel mid-decode → emit aborted+done,
        // reset EP cursors, stop.
        if check_abort(id) {
            ep_emit_abort(stdout, id, m, generated);
            return;
        }
        // Host-side sampler over downloaded f32 logits (temp → top_k → top_p →
        // min_p → seeded draw, temp<=1e-6 = argmax). MiniMax's card carries
        // top_k=40, threaded here via sampling.top_k. RNG seeded per request in
        // generate() (reset_cpu_sampler_rng).
        let next = hipfire_runtime::llama::sample_full_dist(
            &logits,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
        );
        if next == eos_tok {
            break;
        }
        let piece = m.tokenizer.as_ref().unwrap().decode(&[next]);
        generated += 1;
        m.conversation_tokens.push(next);
        if ep_emit_token(stdout, id, &piece, &mut text_acc, stop) {
            break;
        }
        let EpState { gpus, inner } = m.ep.as_mut().unwrap();
        let EpArch::Minimax {
            config,
            weights,
            state,
            partials,
            partials_i64,
        } = inner
        else {
            break;
        };
        if let Err(e) = minimax::forward::forward_ep(
            gpus,
            weights,
            config,
            state,
            partials,
            partials_i64,
            next,
            pos as u32,
        ) {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"forward_ep decode: {}"}}"#,
                id,
                format!("{e}").replace('"', "'")
            );
            return;
        }
        pos += 1;
        let _ = gpus.devices[0].bind_thread();
        // FIX #4 (ep-download-swallow): explicit error handling on the per-token
        // download — emit a JSON error and stop, never feed a zeroed (token-0)
        // logits vec.
        logits = match gpus.devices[0].download_f32(&state[0].logits) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"EP decode logits download failed: {}"}}"#,
                    id,
                    format!("{e:?}").replace('"', "'")
                );
                let _ = stdout.flush();
                return;
            }
        };
    }
    ep_emit_done(
        stdout,
        id,
        generated,
        prompt_n,
        prefill_ms,
        t_decode.elapsed().as_secs_f64() * 1000.0,
    );
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
    // Ascending DeltaNet checkpoint positions (from `m.dflash_checkpoints`) and
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
                    ckpt, lcp, prior_len, rendered.len(), rendered.len() - ckpt, rendered.len(),
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

/// Zero qwen3.5 (arch 5/6) recurrent DeltaNet + KV state IN PLACE for a fresh
/// turn. The state lives in the bundle (ModelState::Qwen35), NOT the always-None
/// direct fields m.dn_state/m.kv_cache — sourcing from the bundle (no hoist →
/// no double-free). Mirrors the per-LA-device memset for pp>1. No-op off qwen35.
fn reset_qwen35_recurrent(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) {
    if m.pp > 1 {
        if let (Some(ModelState::Qwen35(b)), Some(gpus), Some(la)) = (
            m.state.as_ref(),
            m.pp_gpus.as_mut(),
            m.pp_dn_la_to_device.as_ref(),
        ) {
            let dn = &b.dn_state;
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
    } else if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
        let dn = &b.dn_state;
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
    if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
        b.kv_cache.compact_offset = 0;
    }
}

/// Full conversation-state reset: clears CPU cursor + token history, frees GPU
/// checkpoint rings, resets the speculator ring, zeros per-arch recurrent /
/// KV state, and rewinds the adaptive-KV controller. This is the single
/// canonical site for "start a fresh conversation without unloading the model"
/// semantics — mirrors the "reset" command handler (lines ~2406-2491). All
/// per-arch arms are no-ops when that arch is not loaded, so calling this
/// function on any LoadedModel is always safe.
fn model_reset_context(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) {
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    free_checkpoints(&mut m.prefill_checkpoints, gpu);
    free_checkpoints(&mut m.dflash_checkpoints, gpu);
    if let Some(s) = m.speculator.as_mut() {
        s.reset(gpu);
    }
    reset_qwen35_recurrent(m, gpu);
    if let Some(ModelState::Llama(b)) = m.state.as_mut() {
        b.kv.compact_offset = 0;
    }
    if let Some(ref mut s) = m.qwen2_state {
        s.reset();
    }
    if let Some(b) = m.qwen2_mut() {
        b.state.reset();
    }
    if let Some(b) = m.deepseek4_mut() {
        b.state.reset();
        gpu.invalidate_graph_state();
    }
    if let Some(b) = m.lfm2moe_mut() {
        let _ = b.state.reset(gpu);
    }
    if let Some(b) = m.minimax_mut() {
        b.state.reset();
    }
    if let Some(ref mut ad) = m.kv_adaptive {
        ad.reset();
    }
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
    /// Full committed stream (from the emitter) for the wrapper's cache store.
    /// Empty for emitters that don't track it (deepseek4 — its spec path stores
    /// no asst-turn cache).
    streamed_tokens: Vec<u32>,
    /// Newly-prefilled token count (the suffix actually fed through the model).
    prefill_tokens_len: usize,
    /// The terminal flush summary (tool-call count drives the wrapper's
    /// `finish_reason`; events were already rendered inside `generate_spec`).
    finish: FinishSummary,
    prefill_s: f64,
    total_s: f64,
    decode_s: f64,
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
) {
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
                &mut m.asst_turn_cache,
                &m.conversation_tokens,
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
        Some(r) => r,
        // Abort / error early-exit already wrote its own done/error envelope.
        None => return,
    };
    debug_assert_eq!(run.prefill_tokens_len, prefill_tokens_full);

    // ── parse tool_calls + populate asst_turn_cache ──────────────
    //
    // The terminal `finish` flush (inside generate_spec) already rendered the
    // `tool_calls` event. The asst-turn cache fingerprint below is daemon
    // bookkeeping that needs its own parse + the decoded text, so it recomputes
    // from the streamed tokens (mirrors the qwen35 non-dflash path so a
    // dflash-emitted asst turn is reusable via verbatim token replay).
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let decoded_full = tokenizer.decode(&run.streamed_tokens);
    let emit_tool_calls = extract_tool_calls_from_text(&decoded_full);

    // Trim trailing `<|im_end|>` + newline from streamed_tokens so the
    // cached body slots cleanly between the assistant_prefix and the
    // im_end+nl trailer that `build_cached_history` re-adds on replay
    // (mirrors qwen35 cache writer).
    let nl_token = tokenizer.encode("\n");
    let nl_set: std::collections::HashSet<u32> = nl_token.iter().copied().collect();
    let mut cached_seq: Vec<u32> = run.streamed_tokens.clone();
    while let Some(&last) = cached_seq.last() {
        if nl_set.contains(&last) {
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
        let emit_text = hipfire_runtime::tokenizer::maybe_normalize_prompt(&stripped).into_owned();
        let fp = asst_turn_fingerprint(&emit_text, &emit_tool_calls);
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[qwen-cache store dflash] fp={:#018x} cached_seq={} emit_text.len={} tool_calls={} preview={:?}",
                fp, cached_seq.len(), emit_text.len(), emit_tool_calls.len(),
                emit_text.chars().take(60).collect::<String>(),
            );
        }
        m.asst_turn_cache.insert(fp, cached_seq);
    }

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
) -> Option<SpecRun> {
    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Acquire the target via the RAII slot guard — it restores the bundle into
    // m.state on EVERY exit path (return, `?`, panic), which structurally
    // eliminates the eight hand-written reconstruction sites that were the
    // #462 cross-request state-bleed class. `m.speculator`, `m.state`,
    // `m.seq_pos`, `m.conversation_tokens` and `m.eviction` are disjoint fields,
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
            return None;
        }
    };
    // Resolve the arch's carrier once — the single dispatch the spec path routes
    // through for BOTH the target borrow and the emitter (the daemon never
    // arch-matches for spec-decode). `&'static dyn Carrier` borrows nothing from
    // `m`, so it coexists with the `tokenizer`/`&mut m.state` borrows below.
    let arch_id = m.arch_id;
    let carrier = match hipfire_loader::carrier_for(arch_id) {
        Some(c) => c,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"no carrier for arch_id {}"}}"#,
                id, arch_id
            );
            let _ = stdout.flush();
            return None;
        }
    };
    // Arch-dispatched target borrow via `Carrier::spec_target_guard()`
    // (`m.model_path` is a disjoint field → no borrow conflict with the
    // `&mut m.state` the guard takes). qwen35 moves the bundle out + reopens its
    // HfqFile (restored on Drop); the pure-attention arms borrow in place. The
    // boxed `SpecTargetGuard` yields `&mut dyn SpecTarget` either way.
    let mut guard = match carrier.spec_target_guard(&mut m.state, &m.model_path) {
        Ok(g) => g,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"{}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            return None;
        }
    };
    let spec = m.speculator.as_mut().unwrap();

    // Divergent-render RESUME: restore the drafter-local + target recurrent
    // state to the latest checkpoint ≤ ckpt and drop the now-stale tail of the
    // checkpoint ring (`rewind_to` does both), then rewind the daemon's seq_pos
    // / conversation_tokens. The turn then proceeds exactly like a HIT with
    // start_pos == ckpt (the cache plan already set cache_hit=true).
    if let Some(ckpt) = resume_from {
        let slot = match guard.slot() {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"{}"}}"#,
                    id, e
                );
                let _ = stdout.flush();
                return None;
            }
        };
        let _ = spec.rewind_to(gpu, slot, ckpt);
        m.seq_pos = ckpt;
        m.conversation_tokens.truncate(ckpt);
    }

    if !cache_hit {
        // Fresh target state — full prefill from position 0. The DeltaNet
        // recurrent state is zeroed by the prefill seed itself
        // (`seed_target_hidden_from_prompt_abortable` calls `target.reset_state`,
        // which also zeroes s_ef_residual — more complete than the memset loop
        // the old inline path ran here), so only the daemon-side position
        // bookkeeping remains.
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    } else if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache HIT dflash] reuse prefix={} suffix={} (no reset)",
            prefill_start,
            prefill_tokens.len()
        );
    }

    let t0 = Instant::now();
    // Capacity checks. With eviction enabled the advertised context window is
    // effectively unbounded (eviction fires between spec cycles), but the
    // *prompt* must still fit in one physical_cap span because the prompt seed
    // writes it per-token without chunking. Error returns just `return` — the
    // slot guard restores the bundle into m.state on the way out.
    let eff_prompt_cap = if m.eviction.is_some() {
        m.physical_cap
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
        return None;
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
        return None;
    }

    // Prefill: the speculator seeds the target's hidden state (advancing its KV
    // + recurrent state), primes the drafter's cached target-hidden, snapshots
    // the divergent-render checkpoint ring, and returns the target's first
    // token. On a cache hit only the suffix is seeded; on a miss the seed
    // self-resets target state and the full prompt is seeded. Client cancel is
    // surfaced as `PrefillOutcome::Aborted`.
    let id_for_abort = id.to_string();
    let slot = match guard.slot() {
        Ok(s) => s,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"{}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            return None;
        }
    };
    let prefill_outcome = spec.prefill(
        gpu,
        slot,
        &prompt_tokens,
        &prefill_tokens,
        prefill_start,
        cache_hit,
        resume_from,
        &|| check_abort(&id_for_abort),
    );
    let first_token = match prefill_outcome {
        Ok(PrefillOutcome::Ready { first_token }) => first_token,
        Ok(PrefillOutcome::Aborted) => {
            // Full state reset on abort: zero the target recurrent state + KV
            // eviction offset (reset_recurrent, which also clears s_ef_residual)
            // and free the speculator's checkpoint ring, then emit aborted+done
            // for the CLI's drain loop. The slot guard restores the bundle on
            // the way out.
            slot.reset_recurrent(gpu);
            spec.reset(gpu);
            m.seq_pos = 0;
            m.conversation_tokens.clear();
            let _ = writeln!(
                stdout,
                r#"{{"type":"aborted","id":"{}","reason":"client_cancelled"}}"#,
                id
            );
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":0,"prefill_ms":0,"decode_ms":0}}"#,
                id
            );
            let _ = stdout.flush();
            return None;
        }
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"prefill: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            return None;
        }
    };

    let t_prefill = Instant::now();

    // Per-token emission (EosFilter byte stream + grammar + think force-close +
    // stop-sequence match) lives behind the arch-generic `SpecEmit` seam. The
    // emitter is built HERE (not by the caller) because it needs `slot.eos_token()`
    // (only available after the slot guard) AND borrows the tokenizer (derived
    // from `m`). The wrapper supplies the model-independent recipe as an owned
    // `SpecEmitRequest`; the arch's carrier turns it into the concrete
    // `Box<dyn SpecEmit>` (extracting its own grammar schema from `tools`).
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
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"{}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            return None;
        }
    };

    // Decode loop — spec.step returns one acceptance window (SpecStep) per cycle.
    // `emitted` is the speculator's repeat / n-gram context (NOT emission state);
    // it stays in the loop and excludes any grammar-rejected token.
    let mut emitted: Vec<u32> = vec![first_token];
    let mut position = prompt_tokens.len();
    let mut seed_token = first_token;
    // τ accounting, inlined from the unified `SpecStep` (the old `SpecStats`
    // type took the arch-specific `SpecStepResult`, which the daemon no longer
    // sees): τ = accepted drafts / cycle.
    let mut spec_cycles = 0usize;
    let mut spec_accepted = 0usize;
    let mut generated = 0usize;

    // Post-prefill compaction (FlashCASK pattern from dflash_spec_demo).
    // If the prompt already filled past budget+beta, compact once before
    // entering the spec loop so the first spec step writes at physical slot
    // `budget`. compact_offset is maintained on slot.kv_cache; subsequent
    // forwards inside the speculator read it for RoPE phase automatically. The
    // drafter-local hidden cache is compacted to match via `on_evict`.
    if let Some(ref ev) = m.eviction {
        // Eviction is only ever configured for KvCache-backed arches
        // (qwen35/llama); a `Qwen2State`-backed target never reaches here.
        let kv = slot
            .kv_cache_mut()
            .expect("eviction configured ⇒ KvCache-backed spec target");
        if let Some(res) = ev.maybe_evict(gpu, kv, position).unwrap() {
            let pre_phys = position;
            let compact_offset = slot.kv_cache_mut().unwrap().compact_offset;
            eprintln!(
                "[dflash] post-prefill evict: {} -> {} (compact_offset={})",
                pre_phys, res.new_physical, compact_offset,
            );
            position = res.new_physical;
            if !res.retain_mask.is_empty() {
                let _ = spec.on_evict(
                    gpu,
                    &EvictRetain {
                        retain_mask: res.retain_mask,
                        pre_phys,
                    },
                );
            }
        }
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
    if !first_begin.events.is_empty() {
        generated += 1;
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
        if check_abort(id) {
            // Mid-decode cancel: the KV/DeltaNet are advanced past the
            // (un-baked) conversation_tokens, so the next turn must cold-start.
            // Zero the target recurrent state + KV eviction offset and free the
            // speculator's checkpoint ring; the slot guard restores the bundle
            // into m.state on return. Without this, stale mid-decode recurrent
            // state corrupts the next generation (drift → premature EOS).
            slot.reset_recurrent(gpu);
            spec.reset(gpu);
            m.seq_pos = 0;
            m.conversation_tokens.clear();
            let _ = writeln!(
                stdout,
                r#"{{"type":"aborted","id":"{}","reason":"client_cancelled"}}"#,
                id
            );
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":{},"prefill_ms":0,"decode_ms":0}}"#,
                id, generated
            );
            let _ = stdout.flush();
            return None;
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
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"error","id":"{}","message":"spec_step: {}"}}"#,
                    id, e
                );
                let _ = stdout.flush();
                break;
            }
        };
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
                render_client_events(stdout, id, &outcome.events, t0.elapsed().as_millis() as u64);
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
            if !outcome.events.is_empty() {
                emitted.push(tok);
                render_client_events(stdout, id, &outcome.events, t0.elapsed().as_millis() as u64);
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
        seed_token = step.next_seed;

        // Forced-token injection (cohere2moe generation guards; no-op for every
        // other emitter — `take_forced` defaulted empty). The target already
        // batch-advanced over the whole committed tail, so `position` is now its
        // true cursor: advance it over each forced token, re-feed the token
        // through the emitter (a marker transitions its state machine + emits a
        // Committed event), set the next draft seed to it, and continue WITHOUT
        // honoring the suppressed terminator. Bounded by the emitter's own
        // re-entry guard (e.g. MAX_EOS_SUPPRESS) so forcing always terminates.
        if !forced_after.is_empty() {
            for ft in std::mem::take(&mut forced_after) {
                if let Err(e) =
                    slot.spec_advance(gpu, &[ft], position, false, &|| check_abort(id), None)
                {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"error","id":"{}","message":"forced-token advance: {}"}}"#,
                        id, e
                    );
                    let _ = stdout.flush();
                    hit_eos = true;
                    break;
                }
                position += 1;
                emit.set_generated_hint(generated);
                let fo = emit.observe(ft);
                if !fo.events.is_empty() {
                    emitted.push(ft);
                    render_client_events(stdout, id, &fo.events, t0.elapsed().as_millis() as u64);
                    generated += 1;
                }
                seed_token = ft;
            }
        }
        // Per-cycle eviction (FlashCASK). Fires whenever current physical
        // has grown to budget+β since the last compaction. No-op when
        // physical < budget+β, so non-firing cycles pay only the check cost.
        if let Some(ref ev) = m.eviction {
            // Eviction ⇒ KvCache-backed target (qwen35/llama); never qwen2.
            let kv = slot
                .kv_cache_mut()
                .expect("eviction configured ⇒ KvCache-backed spec target");
            if let Some(res) = ev.maybe_evict(gpu, kv, position).unwrap() {
                let pre_phys = position;
                position = res.new_physical;
                if !res.retain_mask.is_empty() {
                    let _ = spec.on_evict(
                        gpu,
                        &EvictRetain {
                            retain_mask: res.retain_mask,
                            pre_phys,
                        },
                    );
                }
            }
        }
        if hit_eos || think_cap_hit {
            break;
        }
    }

    // Snapshot the emitter's state needed by the post-loop bookkeeping before
    // the terminal `finish` consumes it: the full streamed-token stream (for the
    // asst-turn cache) and whether a committed token tripped the grammar matcher.
    let streamed_tokens = emit.streamed_tokens().to_vec();
    let grammar_violated = emit.grammar_violated();

    m.seq_pos = position;
    // Bake the FULL conversation (prefill + decode) into conversation_tokens
    // so subsequent turns can compute LCP against it. Previously this stored
    // only the decoded portion (`emitted`), making the next non-dflash turn
    // full-reset because no system/user prefix was present.
    m.conversation_tokens = {
        let mut v = Vec::with_capacity(prompt_tokens.len() + emitted.len());
        v.extend_from_slice(&prompt_tokens);
        v.extend_from_slice(&emitted);
        v
    };

    // Grammar-violation cleanup: the speculator wrote KV + DN state for the
    // rejected token(s) before the post-acceptance grammar check saw them.
    // Those slots are now poisoned — force a full reset (target recurrent state
    // + KV eviction offset via reset_recurrent, plus the drafter's checkpoint
    // ring via spec.reset) so the next request starts clean. The user pays the
    // prefill cost on the retry but never sees the bad tokens; see Pi turn-12.
    if grammar_violated {
        eprintln!("[grammar-dflash] grammar violation — forcing full KV/DN reset for next turn");
        slot.reset_recurrent(gpu);
        spec.reset(gpu);
        m.conversation_tokens.clear();
        m.seq_pos = 0;
    }

    // Restore the target bundle into m.state via the slot guard's Drop, before
    // the wrapper's end-of-turn cache bookkeeping (which no longer needs the slot).
    drop(guard);

    // Terminal `finish` flush — parses tool calls from the decoded text and
    // renders the `tool_calls` ClientEvent. The arch-specific epilogue (the
    // asst-turn cache store + the `done` envelope) is the WRAPPER's job: it
    // differs per arch (qwen35: `dflash`/`tau`/`cycles` + ChatML token-replay
    // cache; ds4: `spec_k`/`spec_windows`/`spec_accept_pct`), so this core
    // returns a `SpecRun` summary instead of writing them itself.
    let finish = emit.finish();
    render_client_events(stdout, id, &finish.events, 0);

    let t_end = Instant::now();
    Some(SpecRun {
        generated,
        spec_cycles,
        spec_accepted,
        streamed_tokens,
        prefill_tokens_len: prefill_tokens.len(),
        finish,
        prefill_s: t_prefill.duration_since(t0).as_secs_f64(),
        total_s: t_end.duration_since(t0).as_secs_f64(),
        decode_s: t_end.duration_since(t_prefill).as_secs_f64(),
    })
}

/// Qwen3.5/3.6 native-MTP (NextN) speculative decode serve path.
///
/// Analog of [`generate_deepseek4`]'s spec branch and [`generate_dflash`], but
/// drafts via the Qwen MTP head ([`hipfire_arch_qwen35::mtp_spec`]) instead of
/// the diffusion drafter. The proven-durable config (27B-3.6 genre sweep, all
/// genres ≥1.15× AR, lossless): K=3, p_min=0.4, compressed-serial.
///
/// Call sequence (mirrors `mtp_only_demo`):
///   1. cold-reset trunk DeltaNet/KV (v1 is single-turn — no LCP cache),
///   2. `prefill_trunk_and_mtp_cache` (trunk batched prefill + MTP private KV
///      fill over the prompt positions),
///   3. seed token = trunk argmax at the last prefill position,
///   4. loop `spec_step_mtp_compressed_serial` (the production path), committing
///      `result.committed` and advancing `cur_pos += result.advance` each cycle.
///
/// Per-request lifecycle (state-bleed guard, the serve-multiturn class): the
/// `MtpSpecState` (which owns the trunk DN snapshot + the MTP-private KV cache)
/// is allocated FRESH at the top of this function and freed at EVERY exit — so
/// no recurrent MTP state survives between requests. The trunk's persistent DN
/// state lives in the bundle and is cold-zeroed at the start of each request
/// (mirrors `generate_dflash`'s `!cache_hit` reset). The MTP head itself is
/// persistent (loaded once on `LoadedModel.qwen35_mtp_head`).
///
/// Gated behind opt-in: this is only reached when `HIPFIRE_QWEN_MTP=1` AND the
/// head is present (see the dispatch site in `generate`), so the DEFAULT serve
/// path (DFlash/AR) is unchanged.
#[allow(clippy::too_many_arguments)]
fn generate_qwen35_mtp(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    stop: &[String],
    // Request-resolved temp. 0.0 → greedy/argmax-match (p_min confidence cutoff
    // applies). >0 → lossless residual-acceptance sampling (set_sampling below;
    // mutually exclusive with p_min). Reached with temp>0 only when the dispatch
    // gate saw HIPFIRE_MTP_SAMPLED=1 and a temp+top_p-only request.
    temp: f32,
    // Nucleus (top_p) cutoff for the sampled MTP path. Applied to BOTH draft +
    // target nuclei (independent per-side truncation → lossless == AR-at-top_p).
    // Ignored on the greedy path. <= 0.0 is treated as disabled (1.0).
    top_p: f32,
    // Top-k cutoff for the sampled MTP path (request- or card-recommended; e.g.
    // qwen3.6 A3B ships top_k=20). Applied to BOTH draft + target nuclei
    // alongside top_p, so it stays lossless == AR-at-(top_k,top_p). None / 0 →
    // disabled (top_p-only). Ignored on the greedy path.
    top_k: Option<u32>,
    // Nucleus min-prob floor for the sampled MTP path: keep tokens with prob >=
    // min_p * max_prob, folded into the GPU kernel's tau alongside top_p/top_k and
    // applied to BOTH draft + target nuclei → lossless == AR-at-(min_p,top_k,top_p).
    // 0.0 = disabled. Ignored on the greedy path.
    min_p: f32,
) {
    use hipfire_arch_qwen35::mtp_head::MtpKvMode;
    use hipfire_arch_qwen35::mtp_spec::{self, MtpSpecState};
    use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};

    // ── Resolve the proven-durable MTP config ──────────────────────────
    // K defaults to 3 (max_n); p_min defaults to 0.4. Both env-overridable so
    // the GPU-validation thread can sweep. `set_p_min(0.4)` is applied below
    // unconditionally (the MtpSpecState::new default p_min is arch-derived; we
    // pin the proven value for the serve path and let HIPFIRE_MTP_P_MIN win).
    let max_n: usize = std::env::var("HIPFIRE_MTP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|k| (1..=8).contains(k))
        .unwrap_or(3);
    let p_min: f32 = std::env::var("HIPFIRE_MTP_P_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|p: &f32| (0.0..=1.0).contains(p))
        .unwrap_or(0.4);

    let tokenizer = match m.tokenizer.as_ref() {
        Some(t) => t,
        None => {
            emit_error_with_id(stdout, id, "tokenizer not loaded");
            return;
        }
    };

    // ── Prompt build (ChatML / jinja, same two-path branch as DFlash) ───
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
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
                eprintln!("[daemon] jinja render failed in mtp path ({e}) — falling back to Plain");
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

    if prompt_tokens.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return;
    }

    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // ── Cold-reset the trunk recurrent state (v1 = single-turn) ─────────
    // Like generate_dflash's !cache_hit branch: zero the bundle's DeltaNet
    // recurrent state + reset the KV write head so a fresh prompt prefills
    // from position 0 over clean buffers. (No LCP prompt-cache in v1.)
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
        let dn = &b.dn_state;
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
    if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
        b.kv_cache.compact_offset = 0;
    }

    // ── Take the bundle → build a transient ModelSlot ───────────────────
    // The mtp_spec helpers take `&mut ModelSlot`; the daemon owns the qwen35
    // pieces in the bundle. Take them, build the slot, run, put them back at
    // every exit (mirrors generate_dflash). The HfqFile is re-opened (mmap,
    // few µs) because ModelSlot carries it; the MTP path doesn't read it.
    let Qwen35Bundle {
        config: orig_config,
        weights,
        scratch,
        kv_cache,
        dn_state,
    } = match m.state.take() {
        Some(ModelState::Qwen35(b)) => b,
        _ => {
            emit_error_with_id(stdout, id, "qwen35 MTP serve: model state is not Qwen35");
            return;
        }
    };
    let target_config = orig_config.clone();
    let hfq = match HfqFile::open(Path::new(&m.model_path)) {
        Ok(h) => h,
        Err(e) => {
            emit_error_with_id(stdout, id, format!("reopen model: {e}"));
            m.state = Some(ModelState::Qwen35(Qwen35Bundle {
                config: orig_config,
                weights,
                scratch,
                kv_cache,
                dn_state,
            }));
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
        dspark_extract_layers: Vec::new(),
    };

    // Helper closure analog: every early return must put the bundle back. We
    // do it inline (Rust closures can't move `target` piecewise + keep using
    // `m`), so each exit re-packs Qwen35Bundle from target's fields.
    let eos_token = target.config.eos_token;
    let dim = target.config.dim;
    let vocab = target.config.vocab_size;

    // Capacity guard: worst case per cycle writes max_n+1 verify slots before
    // the rollback truncates. seq budget must hold prompt + max*(max_n+1).
    let max_seq_total = m.physical_cap;
    if prompt_tokens
        .len()
        .saturating_add(max_tokens.saturating_mul(max_n + 1))
        .saturating_add(16)
        > max_seq_total
    {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "prompt ({}) + max ({}) × (max_n+1) ({}) exceeds context capacity {} — reload with a larger max_seq",
                prompt_tokens.len(),
                max_tokens,
                max_n + 1,
                max_seq_total
            ),
        );
        m.state = Some(ModelState::Qwen35(Qwen35Bundle {
            config: orig_config,
            weights: target.weights,
            scratch: target.scratch,
            kv_cache: target.kv_cache,
            dn_state: target.dn_state,
        }));
        return;
    }

    // ── Allocate a FRESH per-request MtpSpecState (no cross-request bleed) ─
    let head = m
        .qwen35_mtp_head
        .as_ref()
        .expect("generate_qwen35_mtp reached without a loaded MTP head — dispatch gate is wrong");
    // Compressed (cvs) draft head? Copy the Option out now so we don't hold a
    // borrow of `head`/`m` past the state setup — the compressed-logits scratch
    // alloc below needs &mut state + &mut gpu.
    let cvs_opt = head.weights.compressed_vocab_size;
    let kv_mode = MtpKvMode::Q8;
    let mut state =
        match MtpSpecState::new_for_slot_with_kv_mode(gpu, &target, head, max_n, kv_mode) {
            Ok(s) => s,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("alloc MtpSpecState: {e:?}"));
                m.state = Some(ModelState::Qwen35(Qwen35Bundle {
                    config: orig_config,
                    weights: target.weights,
                    scratch: target.scratch,
                    kv_cache: target.kv_cache,
                    dn_state: target.dn_state,
                }));
                return;
            }
        };
    // Compressed-serial (cvs) head needs its compressed-logits scratch allocated
    // up front (mtp_only_demo does this; the daemon previously only set up the
    // full-vocab path, so a cvs sidecar panicked inside spec_step). Full-vocab
    // heads (cvs == None) skip this entirely.
    if let Some(cvs) = cvs_opt {
        if let Err(e) = state.mtp_scratch.ensure_compressed_logits(gpu, cvs) {
            emit_error_with_id(stdout, id, format!("alloc logits_compressed: {e:?}"));
            return;
        }
        if let Err(e) = state.ensure_compressed_lm_logits(gpu, cvs) {
            emit_error_with_id(stdout, id, format!("alloc mtp_lm_logits_compressed: {e:?}"));
            return;
        }
    }
    if temp > 1e-6 {
        // Sampled MTP. temp/top_p/top_k AND the sampling min_p are honored (folded
        // into the GPU nucleus tau). The draft-chain p_min now ALSO composes with
        // sampling via DraftMode::SampledPMin (sampled draft + early chain-cutoff
        // on the head's raw top prob — lossless, the tau lever for sampled MTP), so
        // we feed the resolved p_min through instead of clearing it. p_min<=0
        // (HIPFIRE_MTP_P_MIN=0) → plain DraftMode::Sampled (no chain pruning).
        state.set_p_min(p_min);
        // set_sampling asserts top_p in (0,1]; a request top_p of 0.0 means
        // "disabled", so clamp it to 1.0 (the no-nucleus sentinel).
        let top_p_eff = if top_p > 0.0 { top_p.min(1.0) } else { 1.0 };
        state.set_sampling(
            mtp_spec::MtpSamplingConfig {
                temp,
                top_k: top_k.map(|k| k as usize).unwrap_or(0),
                top_p: top_p_eff,
                min_p,
            },
            42, // deterministic seed for v1 (reproducible for the coherence battery; a per-request seed is a follow-up)
        );
    } else {
        state.set_p_min(p_min); // greedy MTP confidence cutoff
    }

    let _ = (dim, vocab); // dims sanity-checked inside MtpSpecState::new

    let t0 = Instant::now();

    // ── Prefill: trunk batched + MTP private-KV fill (trunk-spine) ──────
    let head = m.qwen35_mtp_head.as_ref().unwrap();
    let prefill_res = mtp_spec::prefill_trunk_and_mtp_cache(
        gpu,
        &mut target,
        head,
        &mut state,
        &prompt_tokens,
        0,
    );
    if let Err(e) = prefill_res {
        emit_error_with_id(stdout, id, format!("mtp prefill: {e:?}"));
        state.free_gpu(gpu);
        m.state = Some(ModelState::Qwen35(Qwen35Bundle {
            config: orig_config,
            weights: target.weights,
            scratch: target.scratch,
            kv_cache: target.kv_cache,
            dn_state: target.dn_state,
        }));
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = t0.elapsed().as_millis();

    // Seed token = trunk argmax at the last prefill position. prefill left
    // target.scratch.logits holding the post-prompt logits.
    let seed_logits = match gpu.download_f32(&target.scratch.logits) {
        Ok(v) => v,
        Err(e) => {
            emit_error_with_id(stdout, id, format!("download seed logits: {e:?}"));
            state.free_gpu(gpu);
            m.state = Some(ModelState::Qwen35(Qwen35Bundle {
                config: orig_config,
                weights: target.weights,
                scratch: target.scratch,
                kv_cache: target.kv_cache,
                dn_state: target.dn_state,
            }));
            return;
        }
    };
    let seed_token = seed_logits
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

    // ── Decode loop ─────────────────────────────────────────────────────
    let t_prefill = Instant::now();
    let mut emitted: Vec<u32> = vec![seed_token];
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter = EosFilter::new(EosFilterConfig::default());
    let mut last_committed = seed_token;
    let mut cur_pos = prompt_tokens.len();
    let mut generated = 0usize;
    let mut cycles = 0usize;
    let mut accepted_total = 0usize;
    let mut think_count: usize = 0;
    let mut prev_in_think = false;

    // Emit the seed token first (TTFT = prefill).
    streamed_tokens.push(seed_token);
    emit_committed_event(
        stdout,
        id,
        seed_token,
        streamed_tokens.len() - 1,
        t0.elapsed().as_millis() as u64,
    );
    {
        let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
            if let Ok(text) = std::str::from_utf8(&text_bytes) {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"token","id":"{}","text":{}}}"#,
                    id,
                    serde_json::to_string(&text).unwrap_or_default()
                );
                let _ = stdout.flush();
            }
        }
    }
    generated += 1;

    let seed_is_eos = seed_token == eos_token
        || im_end_token == Some(seed_token)
        || tokenizer.is_terminator(seed_token);

    let mut step_error: Option<String> = None;
    while !seed_is_eos && generated < max_tokens {
        if check_abort(id) {
            break;
        }
        if cur_pos + max_n + 1 >= max_seq_total {
            break;
        }

        let result = match mtp_spec::spec_step_mtp_compressed_serial(
            gpu,
            &mut target,
            head,
            &mut state,
            cur_pos,
            last_committed,
            eos_token,
        ) {
            Ok(r) => r,
            Err(e) => {
                step_error = Some(format!("{e:?}"));
                break;
            }
        };
        cycles += 1;
        accepted_total += result.accept_count;

        let mut hit_eos = false;
        let mut think_cap_hit = false;
        for &tok in &result.committed {
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
            if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                if let Ok(text) = std::str::from_utf8(&text_bytes) {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"token","id":"{}","text":{}}}"#,
                        id,
                        serde_json::to_string(&text).unwrap_or_default()
                    );
                    let _ = stdout.flush();
                }
            }
            generated += 1;
            if tok == eos_token || im_end_token == Some(tok) || tokenizer.is_terminator(tok) {
                hit_eos = true;
                break;
            }
            if !stop.is_empty() {
                let decoded_suffix = tokenizer.decode(&streamed_tokens);
                if stop.iter().any(|s| decoded_suffix.ends_with(s.as_str())) {
                    hit_eos = true;
                    break;
                }
            }
            if max_think_tokens > 0 {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let in_think = currently_in_think(
                    raw_str,
                    matches!(
                        assistant_prefix,
                        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                    ),
                );
                if in_think && !prev_in_think {
                    think_count = 0;
                }
                if in_think {
                    think_count += 1;
                }
                prev_in_think = in_think;
                if in_think && think_count >= max_think_tokens {
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
        last_committed = match result.committed.last() {
            Some(&t) => t,
            None => break, // defensive: spec_step always commits ≥ 1
        };
        cur_pos += result.advance;
        if result.hit_eos || hit_eos || think_cap_hit {
            break;
        }
    }

    let t_end = Instant::now();

    // ── Free the per-request MtpSpecState + put the bundle back ─────────
    // CRITICAL (state-bleed guard): free_gpu releases the MTP-private KV cache
    // + the trunk DN snapshot, so the NEXT request starts with no residue.
    state.free_gpu(gpu);
    // The trunk's mid-decode DN/KV is advanced past the (un-baked) prompt; the
    // next request cold-resets at the top of this fn (and the DFlash/AR paths
    // reset on their own !cache_hit branch), so we leave it as-is here. Bake an
    // empty conversation tracker (v1 is single-turn / no LCP reuse).
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    m.state = Some(ModelState::Qwen35(Qwen35Bundle {
        config: orig_config,
        weights: target.weights,
        scratch: target.scratch,
        kv_cache: target.kv_cache,
        dn_state: target.dn_state,
    }));

    if let Some(e) = step_error {
        emit_error_with_id(stdout, id, format!("mtp spec_step: {e}"));
        return;
    }

    // ── Done envelope ──────────────────────────────────────────────────
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
    // τ = tokens committed per cycle (excludes the seed). > 1.0 = MTP speedup.
    let tau = if cycles > 0 {
        (emitted.len().saturating_sub(1)) as f64 / cycles as f64
    } else {
        0.0
    };
    let _ = accepted_total;
    let hit_length_cap = generated >= max_tokens;
    let finish_reason = if hit_length_cap { "length" } else { "stop" };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{},"mtp":true,"tau":{:.2},"cycles":{},"cached_tokens":0,"finish_reason":"{}"}}"#,
        id,
        generated,
        tok_s,
        prompt_tokens.len(),
        prefill_ms,
        prefill_tok_s,
        decode_tok_s,
        prefill_ms,
        tau,
        cycles,
        finish_reason,
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
) {
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.seq_pos
        .saturating_add(prompt_est)
        .saturating_add(max_tokens)
        > m.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // qwen35 recurrent state lives in the bundle (ModelState::Qwen35), not
        // the always-None m.dn_state/m.kv_cache. Inlined (disjoint field access)
        // because a `&tokenizer` borrow of `m` is live here; covers both the
        // pp>1 per-LA-device path and the single-GPU path.
        if m.pp > 1 {
            if let (Some(ModelState::Qwen35(b)), Some(ref mut gpus), Some(ref la)) = (
                m.state.as_ref(),
                m.pp_gpus.as_mut(),
                m.pp_dn_la_to_device.as_ref(),
            ) {
                let dn = &b.dn_state;
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
        } else if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
            let dn = &b.dn_state;
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
        if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(ad) = m.kv_adaptive.as_mut() {
            ad.reset();
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
    // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
    // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
    // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
    // the .hfq carries no template, e.g. an older A1B convert).
    // Jinja default-ON (flipped 2026-06-09): render through the model's chat
    // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
    // ChatML/Plain). Falls back to Plain automatically when no template resolves.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    // hunt3 H-A: drop the `seq_pos == 0` gate (PR #389 removed it from generate()).
    // With the gate, turn 2+ fell through to the Plain scaffold, dropping the
    // system prompt and the full history replay that render_messages provides.
    // Now Jinja renders the full conversation every turn; the cold-reset block
    // below (guarded on seq_pos > 0) re-zeros recurrent state so the full render
    // writes from position 0 instead of appending to the prior turn's KV/DeltaNet.
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let new_tokens = if try_jinja {
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
                    system: if m.seq_pos == 0 { system_prompt } else { None },
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
            system: if m.seq_pos == 0 { system_prompt } else { None },
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
    // `reset_pp_uncommitted_state!` semantics, written inline because that macro
    // is defined later (after kv/dn/gpus are borrowed). Same shape as the
    // context-full reset at the top of this fn and generate()'s `jinja_active &&
    // seq_pos > 0` block.
    if try_jinja && m.seq_pos > 0 {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // qwen35 recurrent state lives in the bundle (ModelState::Qwen35), not
        // the always-None m.dn_state/m.kv_cache. Covers pp>1 + single-GPU.
        if m.pp > 1 {
            if let (Some(ModelState::Qwen35(b)), Some(ref mut gpus), Some(ref la)) = (
                m.state.as_ref(),
                m.pp_gpus.as_mut(),
                m.pp_dn_la_to_device.as_ref(),
            ) {
                let dn = &b.dn_state;
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
        } else if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
            let dn = &b.dn_state;
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
        if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(ModelState::Llama(b)) = m.state.as_mut() {
            b.kv.compact_offset = 0;
        }
    }

    let trailer = nl.len();
    if m.seq_pos
        .saturating_add(new_tokens.len())
        .saturating_add(max_tokens)
        .saturating_add(trailer)
        > m.physical_cap
    {
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

    let ModelState::Qwen35(b) = m.state.as_mut().unwrap() else {
        unreachable!()
    };
    let config = &b.config;
    let weights = &b.weights;
    let scratch_set = m.pp_scratch_set.as_ref().unwrap();
    let kv = &mut b.kv_cache;
    let dn = &mut b.dn_state;
    let gpus = m.pp_gpus.as_mut().unwrap();
    let dn_la_to_device = m.pp_dn_la_to_device.as_ref().unwrap();

    macro_rules! reset_pp_uncommitted_state {
        () => {{
            m.seq_pos = 0;
            m.conversation_tokens.clear();
            free_checkpoints(&mut m.prefill_checkpoints, gpu);
            free_checkpoints(&mut m.dflash_checkpoints, gpu);
            for (i, s) in dn.s_matrices.iter().enumerate() {
                let g = &mut gpus.devices[dn_la_to_device[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            for (i, s) in dn.s_scales.iter().enumerate() {
                let g = &mut gpus.devices[dn_la_to_device[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            for (i, s) in dn.conv_states.iter().enumerate() {
                let g = &mut gpus.devices[dn_la_to_device[i] as usize];
                let _ = g.bind_thread();
                let _ = g.hip.memset(&s.buf, 0, s.buf.size());
            }
            kv.compact_offset = 0;
            if let Some(ModelState::Llama(b)) = m.state.as_mut() {
                b.kv.compact_offset = 0;
            }
        }};
    }

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
    // (m.decoded_vocab) because `m` is already mutably borrowed here (kv/dn/gpus)
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
        m.seq_pos,
        kv,
        dn,
        scratch_set,
    ) {
        // hunt3 M-A: a partial-band prefill failure leaves DeltaNet partially
        // advanced; without resetting, the next cold turn prefills over dirty
        // recurrent state (drift). Mirror both abort paths, which already reset.
        reset_pp_uncommitted_state!();
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

    if check_abort(id) {
        reset_pp_uncommitted_state!();
        let _ = writeln!(
            stdout,
            r#"{{"type":"aborted","id":"{}","reason":"client_cancelled"}}"#,
            id
        );
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":0,"prefill_ms":0,"decode_ms":0}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

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
    let mut bytes_fed_to_filter = 0usize;
    let mut filter = EosFilter::new(EosFilterConfig::default());
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
            reset_pp_uncommitted_state!();
            let _ = writeln!(
                stdout,
                r#"{{"type":"aborted","id":"{}","reason":"client_cancelled"}}"#,
                id
            );
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":{},"prefill_ms":0,"decode_ms":0}}"#,
                id, generated
            );
            let _ = stdout.flush();
            return;
        }
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
        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
            let text = std::str::from_utf8(&text_bytes).unwrap();
            let _ = writeln!(
                stdout,
                r#"{{"type":"token","id":"{}","text":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default()
            );
            let _ = stdout.flush();
        }

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
            // hunt3 M-A: a decode-step failure leaves DeltaNet advanced past the
            // (un-baked) conversation_tokens; reset so the next cold turn starts
            // clean. Mirrors both abort paths.
            reset_pp_uncommitted_state!();
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
        if im_end_token == Some(next_token) {
            break;
        }
        if tokenizer.is_terminator(next_token) {
            break;
        }

        // hunt3 M-F: user stop-sequence match against the decoded output suffix
        // (pp>1 multi-GPU path). Mirrors the AR generate() loop; matches the
        // full decoded text so a stop string spanning a token boundary is
        // caught. A plain break exits the `while generated < max_tokens` loop
        // (this path's `done` event carries no finish_reason field, so there is
        // no reason to resolve — terminating generation is the contract). Gated
        // behind `!stop.is_empty()` so the common path pays nothing.
        if !stop.is_empty() {
            let decoded_suffix = tokenizer.decode(&streamed_tokens);
            if stop.iter().any(|s| decoded_suffix.ends_with(s.as_str())) {
                break;
            }
        }

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
                eprintln!("[think-cap] id={} — total think {} exceeded cap {}+256 while still thinking; forcing EOS", id, total_think_tokens, max_total_think);
                break;
            }
            if let Some(mark) = latch_gen_mark {
                if generated.saturating_sub(mark) >= post_latch_answer_budget {
                    eprintln!("[think-cap] id={} — {} tokens since think-cap latch without finishing; forcing EOS", id, generated.saturating_sub(mark));
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
                    eprintln!("[force-answer] id={} — re-closing a re-opened <think> (latched / think-cap)", id);
                }
                let close_tokens = tokenizer.encode(&think_continuation());
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
                    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
                    let new_bytes = &all_bytes[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes.len();
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                        let text = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{}}}"#,
                            id,
                            serde_json::to_string(&text).unwrap_or_default()
                        );
                        let _ = stdout.flush();
                    }
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
                let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
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
                .seq_pos
                .saturating_add(nudge_len)
                .saturating_add(
                    max_tokens
                        .saturating_sub(generated)
                        .saturating_sub(nudge_len),
                )
                .saturating_add(nl.len());
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
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes2) {
                        let t = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{}}}"#,
                            id,
                            serde_json::to_string(&t).unwrap_or_default()
                        );
                        let _ = stdout.flush();
                    }
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
/// Generic auto-regressive decode driver (Inc 1, Task 1.4b-iii). Extracted
/// verbatim-in-behavior from the qwen35 AR arm of `generate` (arch 5/6), but
/// every arch-coupled op routes through `ArchDispatch` hooks and all loop state
/// (seq_pos, streamed tokens, think/budget counters, rng) is local. DEAD CODE
/// this stage: NOT routed at the `if m.arch_id==5||6` dispatch point. The 1.4c
/// dual-run parity harness validates token-identity vs the old arm on GPU.
///
/// Faithfulness note (for the 1.4c reviewer): `ngram_scope` uses the local
/// `streamed_tokens` in place of `m.conversation_tokens[ngram_scope_start..]`
/// (proven identical push-order at every sample site); the asst-turn cached_seq
/// still reads the real conversation buffer (it includes the post-loop ChatML
/// trailer that `streamed_tokens` does not). Adaptive-KV stderr phase labels are
/// unified in the hook (diagnostic only, not token-affecting).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn ar_generate(
    dispatch: &mut dyn hipfire_runtime::arch_dispatch::ArchDispatch,
    gpu: &mut rdna_compute::Gpu,
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
) {
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
            dispatch.prefill_forward(gpu, chunk, seq_pos).unwrap();
            seq_pos += chunk_len;
            if let Some(new_phys) = dispatch.maybe_evict(gpu, seq_pos).unwrap() {
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
            dispatch.prefill_forward(gpu, chunk, seq_pos).unwrap();
            seq_pos += chunk.len();
            dispatch.maybe_adaptive_downshift(gpu, seq_pos);
            if ckpt_resume_enabled() {
                dispatch.take_prefill_checkpoint(gpu, seq_pos);
            }
            start = end;
        }
    }
    if prefill_aborted {
        dispatch.abort_zero_recurrent(gpu);
        seq_pos = 0;
        dispatch.set_seq_pos(0);
        dispatch.conversation_tokens_mut().clear();
        dispatch.free_prefill_checkpoints(gpu);
        let _ = writeln!(
            stdout,
            r#"{{"type":"aborted","id":"{}","reason":"client_cancelled"}}"#,
            id
        );
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":0,"prefill_ms":0,"decode_ms":0}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }
    // Post-prefill adaptive-KV downshift.
    dispatch.maybe_adaptive_downshift(gpu, seq_pos);
    dispatch.conversation_tokens_mut().extend_from_slice(&new_tokens);

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
    sampler::collect_unclosed_attractor_blocks(ngram_scope0, &attractor_pairs, 20, 2, &mut blocked0);
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
        let mask: Option<&[bool]> = if grammar_active && !matcher.as_ref().unwrap().is_free() {
            matcher
                .as_ref()
                .unwrap()
                .token_mask(grammar_vocab, &mut grammar_mask);
            Some(&grammar_mask)
        } else {
            None
        };
        dispatch
            .sample(gpu, &cfg0, vocab_size, ngram_scope0, mask, &mut rng_state)
            .unwrap()
    };
    if grammar_active {
        let text = dispatch.tokenizer().decode(&[tok0]);
        matcher.as_mut().unwrap().advance(&text);
    }
    let t_prefill = Instant::now();
    let mut next_token = tok0;

    let mut generated = 0;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter = EosFilter::new(EosFilterConfig::default());
    let mut alert_fired = false;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;
    let mut force_answer_latched = false;
    let think_open_tok = dispatch.tokenizer().special_token_id("<think>");
    let max_total_think: usize = std::env::var("HIPFIRE_MAX_TOTAL_THINK_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut total_think_tokens: usize = 0;
    let post_latch_answer_budget: usize = std::env::var("HIPFIRE_POST_LATCH_ANSWER_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let mut latch_gen_mark: Option<usize> = None;

    let loop_guard =
        hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    while generated < max_tokens {
        if check_abort(id) {
            // Client cancelled mid-decode — full cold reset (mirrors DFlash abort).
            seq_pos = 0;
            dispatch.set_seq_pos(0);
            dispatch.conversation_tokens_mut().clear();
            dispatch.free_prefill_checkpoints(gpu);
            dispatch.abort_zero_recurrent(gpu);
            let _ = writeln!(
                stdout,
                r#"{{"type":"aborted","id":"{}","reason":"client_cancelled"}}"#,
                id
            );
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","finish_reason":"aborted","prompt_tokens":0,"completion_tokens":{},"prefill_ms":0,"decode_ms":0}}"#,
                id, generated
            );
            let _ = stdout.flush();
            return;
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
        let all_bytes = dispatch.tokenizer().decode_bytes(&streamed_tokens);
        let new_bytes = &all_bytes[bytes_fed_to_filter..];
        bytes_fed_to_filter = all_bytes.len();
        if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
            let text = std::str::from_utf8(&text_bytes).unwrap();
            let _ = writeln!(
                stdout,
                r#"{{"type":"token","id":"{}","text":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default()
            );
            let _ = stdout.flush();
        }

        dispatch
            .decode_step_forward(gpu, next_token, seq_pos)
            .unwrap();
        seq_pos += 1;
        if ckpt_resume_enabled() {
            dispatch.take_prefill_checkpoint(gpu, seq_pos);
        }
        if let Some(new_phys) = dispatch.maybe_evict(gpu, seq_pos).unwrap() {
            seq_pos = new_phys;
        }
        dispatch.maybe_adaptive_downshift(gpu, seq_pos);

        // Arch-specific eos/terminator set (qwen35 = eos||terminator; qwen2 =
        // eos_token_ids set). im_end + stop-seqs stay driver-generic below.
        if dispatch.is_eos(next_token) {
            break;
        }
        if im_end_token == Some(next_token) {
            break;
        }

        if !stop.is_empty() {
            let decoded_suffix = dispatch.tokenizer().decode(&streamed_tokens);
            if stop.iter().any(|s| decoded_suffix.ends_with(s.as_str())) {
                break;
            }
        }

        // max_think_tokens / total-think / force-answer enforcement.
        let force_answer_now = check_force_answer(id);
        if force_answer_now {
            force_answer_latched = true;
        }
        if max_think_tokens > 0 || force_answer_now || force_answer_latched || max_total_think > 0 {
            let raw_so_far = dispatch.tokenizer().decode_bytes(&streamed_tokens);
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
                eprintln!("[think-cap] id={} — total think {} exceeded cap {}+256 while still thinking; forcing EOS", id, total_think_tokens, max_total_think);
                break;
            }
            if let Some(mark) = latch_gen_mark {
                if generated.saturating_sub(mark) >= post_latch_answer_budget {
                    eprintln!("[think-cap] id={} — {} tokens since think-cap latch without finishing; forcing EOS", id, generated.saturating_sub(mark));
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
                    eprintln!("[force-answer] id={} — closing <think> mid-turn to commit to the answer", id);
                } else if force_answer_latched {
                    eprintln!("[force-answer] id={} — re-closing a re-opened <think> (latched / think-cap)", id);
                }
                let close_tokens = dispatch.tokenizer().encode(&think_continuation());
                let budget_left = max_tokens.saturating_sub(generated);
                let take = close_tokens.len().min(budget_left);
                for &t in &close_tokens[..take] {
                    dispatch.decode_step_forward(gpu, t, seq_pos).unwrap();
                    seq_pos += 1;
                    if let Some(new_phys) = dispatch.maybe_evict(gpu, seq_pos).unwrap() {
                        seq_pos = new_phys;
                    }
                    dispatch.conversation_tokens_mut().push(t);
                    if grammar_active {
                        let text = dispatch.tokenizer().decode(&[t]);
                        matcher.as_mut().unwrap().advance(&text);
                    }
                    streamed_tokens.push(t);
                    if let Some(tp) = tape.as_deref_mut() {
                        tp.push(t);
                    }
                    emit_committed_event(
                        stdout,
                        id,
                        t,
                        streamed_tokens.len() - 1,
                        t0.elapsed().as_millis() as u64,
                    );
                    let all_bytes = dispatch.tokenizer().decode_bytes(&streamed_tokens);
                    let new_bytes = &all_bytes[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes.len();
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                        let text = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{}}}"#,
                            id,
                            serde_json::to_string(&text).unwrap_or_default()
                        );
                        let _ = stdout.flush();
                    }
                    generated += 1;
                }
                think_count = 0;
                prev_in_think = false;
                if generated >= max_tokens {
                    break;
                }
            }
        }

        // N-gram loop detector.
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
                    let mask: Option<&[bool]> =
                        if grammar_active && !matcher.as_ref().unwrap().is_free() {
                            matcher
                                .as_ref()
                                .unwrap()
                                .token_mask(grammar_vocab, &mut grammar_mask);
                            Some(&grammar_mask)
                        } else {
                            None
                        };
                    dispatch
                        .sample(gpu, &cfg, vocab_size, ngram_scope, mask, &mut rng_state)
                        .unwrap()
                };
                if grammar_active {
                    let text = dispatch.tokenizer().decode(&[next_token]);
                    matcher.as_mut().unwrap().advance(&text);
                }
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
            if nudge_len > 0 && (dispatch.has_eviction() || need_kv <= dispatch.physical_cap()) {
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
                    let all_bytes2 = dispatch.tokenizer().decode_bytes(&streamed_tokens);
                    let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                    bytes_fed_to_filter = all_bytes2.len();
                    if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes2) {
                        let t = std::str::from_utf8(&text_bytes).unwrap();
                        let _ = writeln!(
                            stdout,
                            r#"{{"type":"token","id":"{}","text":{}}}"#,
                            id,
                            serde_json::to_string(&t).unwrap_or_default()
                        );
                        let _ = stdout.flush();
                    }
                    dispatch.decode_step_forward(gpu, tok, seq_pos).unwrap();
                    seq_pos += 1;
                    if let Some(new_phys) = dispatch.maybe_evict(gpu, seq_pos).unwrap() {
                        seq_pos = new_phys;
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
            if generated >= max_tokens {
                break;
            }
        }

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
        next_token = {
            let mask: Option<&[bool]> = if grammar_active && !matcher.as_ref().unwrap().is_free() {
                matcher
                    .as_ref()
                    .unwrap()
                    .token_mask(grammar_vocab, &mut grammar_mask);
                Some(&grammar_mask)
            } else {
                None
            };
            dispatch
                .sample(gpu, &cfg, vocab_size, ngram_scope, mask, &mut rng_state)
                .unwrap()
        };
        if grammar_active {
            let text = dispatch.tokenizer().decode(&[next_token]);
            let was_detected = matcher.as_ref().unwrap().attractor_detected();
            matcher.as_mut().unwrap().advance(&text);
            if !was_detected && matcher.as_ref().unwrap().attractor_detected() {
                eprintln!(
                    "[grammar-ngram] attractor detected in tool_call args at gen={} — forcing close",
                    generated,
                );
            }
        }
    }

    // ChatML \n trailer after <|im_end|>.
    let last_conv = dispatch
        .conversation_tokens_mut()
        .last()
        .copied()
        .unwrap_or(0);
    if im_end_token == Some(last_conv) && !nl.is_empty() {
        for &t in nl {
            dispatch.decode_step_forward(gpu, t, seq_pos).unwrap();
            seq_pos += 1;
            if let Some(new_phys) = dispatch.maybe_evict(gpu, seq_pos).unwrap() {
                seq_pos = new_phys;
            }
            dispatch.conversation_tokens_mut().push(t);
        }
    }

    // Write the final physical slot back to the model (old arm mutated
    // m.seq_pos directly per token).
    dispatch.set_seq_pos(seq_pos);

    // ── parse tool_calls + content once ──────────────────────────────────
    let decoded_full = dispatch.tokenizer().decode(&streamed_tokens);
    let emit_tool_calls = extract_tool_calls_from_text(&decoded_full);

    if !emit_tool_calls.is_empty() {
        let calls_json: Vec<serde_json::Value> = emit_tool_calls
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
) {
    // hunt3 M-E: seed the process-global CPU sampler RNG with this request's
    // fixed seed so the grammar/CPU-fallback sample stream is deterministic per
    // request and does not carry RNG state across requests. Matches the u32 the
    // GPU sample path uses (0x13579BDF).
    hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);
    // Dense multi-GPU serve (PB-TP5 Tp / P-C Pp): route to the shared generate_dense
    // BEFORE any arch short-circuit / EP. The model lives in `m.tp` (TpModel) or
    // `m.pp_dense` (PpModel) over its own Gpus; the single-GPU arch fields are None.
    // Disjoint field borrows: `m.tokenizer` (read) + the model field (mut).
    if m.tp.is_some() {
        // Multi-turn KV reuse at parity with the single-GPU llama path: LCP the
        // rendered conversation vs conversation_tokens; pure-KV ⇒ no DeltaNet
        // checkpoints, cold-prefill on divergence (empty ckpts + resume off).
        let hist: &[hipfire_runtime::prompt_frame::Message] = messages_history.unwrap_or(&[]);
        let cache_disabled =
            std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
        let tok = m.tokenizer.as_ref().expect("dense serve: tokenizer");
        let plan = plan_prompt_cache(
            tok,
            &mut m.asst_turn_cache,
            &m.conversation_tokens,
            m.eviction.is_none(),
            system_prompt,
            prompt,
            assistant_prefix,
            hist,
            cache_disabled,
            &[],
            false,
        );
        let rendered = plan.rendered;
        let new_tokens = plan.new_tokens;
        let start_pos = plan.start_pos;
        let model = m.tp.as_mut().expect("dense serve: TpModel");
        let gen = generate_dense(
            model,
            tok,
            stdout,
            id,
            &new_tokens,
            start_pos,
            temp,
            top_p,
            top_k,
            min_p,
            max_tokens,
            stop,
        );
        match gen {
            Some(g) => {
                let mut v = rendered;
                v.extend_from_slice(&g);
                m.conversation_tokens = v;
            }
            None => m.conversation_tokens.clear(),
        }
        return;
    }
    if m.pp_dense.is_some() {
        let hist: &[hipfire_runtime::prompt_frame::Message] = messages_history.unwrap_or(&[]);
        let cache_disabled =
            std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
        let tok = m.tokenizer.as_ref().expect("dense serve: tokenizer");
        let plan = plan_prompt_cache(
            tok,
            &mut m.asst_turn_cache,
            &m.conversation_tokens,
            m.eviction.is_none(),
            system_prompt,
            prompt,
            assistant_prefix,
            hist,
            cache_disabled,
            &[],
            false,
        );
        let rendered = plan.rendered;
        let new_tokens = plan.new_tokens;
        let start_pos = plan.start_pos;
        let model = m.pp_dense.as_mut().expect("dense serve: PpModel");
        let gen = generate_dense(
            model,
            tok,
            stdout,
            id,
            &new_tokens,
            start_pos,
            temp,
            top_p,
            top_k,
            min_p,
            max_tokens,
            stop,
        );
        match gen {
            Some(g) => {
                let mut v = rendered;
                v.extend_from_slice(&g);
                m.conversation_tokens = v;
            }
            None => m.conversation_tokens.clear(),
        }
        return;
    }
    // Expert-parallel (task #26): route to generate_ep BEFORE any arch
    // short-circuit (generate_qwen2/_deepseek4/...), since EP mode leaves the
    // single-GPU arch fields (q35_*/deepseek4_*) None — the per-arch paths
    // would unwrap-panic / error on the missing config.
    if m.ep.is_some() {
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
        generate_ep(
            m,
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
        return;
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
    if m.arch_id == 7 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
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
        return;
    }
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
        );
        let _ = stop; // hunt3 M-F: not wired for arch_id=7 (qwen2 bring-up)
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
        let spec_temp_ok =
            temp <= 1e-6 || m.speculator.as_ref().map_or(false, |s| !s.requires_greedy());
        let spec_mode = deepseek4_spec_requested(m) && spec_temp_ok;
        if spec_mode && m.speculator.is_some() {
            generate_deepseek4_spec(
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
        }
        return;
    }
    // arch_id=11 (LFM2.5-MoE) with an opt-in model-free n-gram speculator loaded
    // (lfm2moe `SpecTarget`, conv-state snapshot/rollback) routes to the
    // arch-generic spec loop, like qwen2 (7) / minimax (10). Without a speculator
    // it falls through to the plain `generate_lfm2moe` short-circuit below.
    if m.arch_id == 11 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
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
        return;
    }
    if m.arch_id == 11 {
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
    // arch_id=12 (Cohere2-MoE) with an opt-in model-free n-gram speculator loaded
    // (cohere2moe `SpecTarget` + `Cohere2MoeEmit`, which ports the agentic-marker
    // state machine + empty-turn / think-budget generation guards) routes to the
    // arch-generic spec loop, like qwen2 (7) / minimax (10) / lfm2moe (11).
    // Without a speculator it falls through to the plain `generate_cohere2moe`.
    if m.arch_id == 12 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
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
        return;
    }
    if m.arch_id == 12 {
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
        generate_cohere2moe(
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
    // arch_id=10 (MiniMax-M2) with an opt-in model-free n-gram speculator loaded
    // (minimax `SpecTarget`) routes to the arch-generic spec loop, exactly like
    // qwen2 (7) above. Without a speculator it falls through to the plain
    // `generate_minimax` short-circuit below.
    if m.arch_id == 10 && m.speculator.is_some() && (temp <= 1e-6 || ngram_can_sample) {
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
        );
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
        return;
    }
    if m.arch_id == 10 {
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
    // Expert-parallel dispatch (task #26). ep.is_some() → generate_ep (AR via
    // forward_ep, full sampler on rank-0 logits). Refusals enforced at load.
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
        return;
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
    // Exception: thinking-on + max_think_tokens currently needs the AR path.
    // DFlash's budget cap can close/strip the think span but does not yet
    // continue into visible answer text after the forced close. AR already
    // splices </think> through KV and continues generation, so route budgeted
    // thinking requests there until DFlash continuation is implemented.
    let budgeted_thinking_needs_ar = max_think_tokens > 0
        && !matches!(
            assistant_prefix,
            hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
        );

    // ── Qwen3.5/3.6 native-MTP serve path (opt-in) ─────────────────────
    //
    // Routed BEFORE the DFlash fast path so an operator who opts in gets MTP
    // (durable on prose, ≥1.15× AR every genre; proven 27B-3.6 K=3 p_min=0.4
    // compressed-serial). Gated tightly so the DEFAULT path (DFlash/AR) is
    // unchanged: requires the env opt-in `HIPFIRE_QWEN_MTP=1`, a loaded MTP
    // head (`m.qwen35_mtp_head`), greedy (temp≈0; MTP serve v1 is greedy/
    // argmax-match), a qwen3.5/3.6 trunk (arch 5/6), single-GPU, and no
    // budgeted-thinking (which needs the AR </think> splice). Anything not
    // satisfying ALL of these falls through to the existing DFlash/AR routing.
    //
    // SAMPLED (temp>0) MTP is now distribution-preserving and reachable here,
    // but gated behind an opt-in env flag until the temp>0 coherence battery
    // validates it. The arch layer's F5 DFlash-convention fix (mtp_spec.rs:
    // independent per-side nuclei + sample_residual) makes the temp>0 branch
    // lossless: draft + target each truncate to their own top_p nucleus, the
    // accept ratio is computed against the truncated distributions, and the
    // bonus is drawn from the renormalized residual (no longer the lossy
    // un-truncated / sample_top_p(trunk) posture). Gating:
    //   * greedy (temp <= 1e-6) → ALWAYS routes to MTP (default, unchanged), or
    //   * sampled (temp > 0) → routes to MTP only when `HIPFIRE_MTP_SAMPLED=1`.
    //     temp + top_p + top_k + min_p are all plumbed through and honored on the
    //     sampled MTP path (top_k+top_p lossless on both nuclei; min_p applied via
    //     the mtp_spec cutoff), so no sampling knob forces a fall-through here.
    // With HIPFIRE_MTP_SAMPLED unset, a temp>0 MTP-opted-in request still falls
    // through to DFlash (lossless sampled-spec) or AR exactly as before.
    let qwen_mtp_opt_in = std::env::var("HIPFIRE_QWEN_MTP").ok().as_deref() == Some("1");
    let mtp_sampled_on = std::env::var("HIPFIRE_MTP_SAMPLED").ok().as_deref() == Some("1");
    // Sampled MTP honors temp + top_p + top_k: the residual-accept sampler
    // applies the SAME top_k+top_p nucleus to BOTH the draft and target sides
    // (see mtp_sampled_accept / the draft truncation), so it stays lossless ==
    // AR-at-(top_k,top_p). min_p is ALSO plumbed through (min_p.unwrap_or(0.0)
    // below → generate_qwen35_mtp → the mtp_spec min_p cutoff) and honored, not
    // carved out. top_k + min_p flow through to generate_qwen35_mtp below and on
    // into the nuclei — this is what lets a model whose card recommends top_k
    // (qwen3.6 A3B ships top_k=20) keep its recipe AND get the MTP speedup,
    // instead of silently dropping top_k/min_p or losing MTP.
    if qwen_mtp_opt_in
        && m.qwen35_mtp_head.is_some()
        && (temp <= 1e-6 || mtp_sampled_on)
        && (m.arch_id == 5 || m.arch_id == 6)
        && !budgeted_thinking_needs_ar
    {
        generate_qwen35_mtp(
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
            stop,
            temp,  // request-resolved temp (0.0 greedy / >0 lossless residual-accept sampling)
            top_p, // nucleus cutoff: honored on the sampled MTP path
            top_k, // top-k cutoff: honored on the sampled MTP path (both nuclei)
            min_p.unwrap_or(0.0), // min_p floor: honored on the sampled MTP nuclei
        );
        // Silence unused-variable warnings for AR-only / DFlash-only knobs the
        // MTP serve path does not consume.
        let _ = (
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
            pflash_cfg,
            think_mode,
        );
        return;
    }
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
    let qwen_dflash_route = (m.arch_id == 5 || m.arch_id == 6)
        && (temp <= 1e-6 || ddtree_swor_route || chain_sample_route);
    // llama (arch 0/1): #483 built + validated dense DFlash with ddtree tree-SWOR, so
    // temp>0 engages ddtree-SWOR here (bare temp). DSpark (qwen3) adds a validated
    // sampled-llama CHAIN path — its fused sample_top_p_pf verify honors
    // temp+top_p+top_k and beats AR at temp>0 — so `chain_sample_route` now engages
    // llama temp>0 too. (Non-DSpark chain-mode llama has no such path and stays on
    // AR via `spec_can_sample`/`supports_temp_swor` gating.)
    let llama_dflash_route = (m.arch_id == 0 || m.arch_id == 1)
        && (temp <= 1e-6 || ddtree_swor_route || chain_sample_route);
    // Operator visibility: a temp>0 request on a DFlash-capable arch that did NOT
    // qualify for spec silently runs AR (correct, but slower). Name the reason.
    if temp > 1e-6
        && m.speculator.is_some()
        && (m.arch_id == 5 || m.arch_id == 6 || m.arch_id == 0 || m.arch_id == 1)
        && !qwen_dflash_route
        && !llama_dflash_route
        && !budgeted_thinking_needs_ar
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
    if m.speculator.is_some()
        && !budgeted_thinking_needs_ar
        && !force_ar_chat
        && (qwen_dflash_route || llama_dflash_route)
    {
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
            stop,  // hunt3 M-F: thread user stop sequences into the default DFlash path
            temp, // request-resolved temp: ddtree mode → SWOR (temp-only); chain mode → lossless rejection-sampling
            top_p, // nucleus cutoff: honored on the chain sampled path (ignored by the ddtree SWOR arm)
            top_k.map(|k| k as usize).unwrap_or(0), // top-k cutoff (chain path; recipe → folded into tau)
            cactus_delta, // request opt-in (0.0 default = lossless); ds4 DSpark / qwen35 DFlash only
        );
        // Silence unused-variable warnings for the params DFlash doesn't
        // consume. top_p IS now applied to the spec sampling (nucleus on both
        // draft + target). repeat penalties are AR-only sampling knobs;
        // pflash_state is bypassed on the DFlash decode path.
        let _ = (
            repeat_penalty,
            repeat_window,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
        );
        return;
    }

    // Auto-reset on multi-turn rollover. When eviction is active (operator
    // enabled cask_sidecar at load), the physical buffer is bounded by
    // budget+beta+safety regardless of conversation length, so reset never
    // needs to fire — eviction reclaims slots after each token. When eviction
    // is OFF, physical grows unbounded up to max_seq; reset when we'd overrun.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache GEN-ENTRY] conv_tok={} seq_pos={}",
            m.conversation_tokens.len(),
            m.seq_pos
        );
    }
    if m.eviction.is_none()
        && m.seq_pos
            .saturating_add(prompt_est)
            .saturating_add(max_tokens)
            > m.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // Free the speculator's (relocated) checkpoint ring on reset — this AR
        // path is reachable by a DFlash-capable model (temp>0 / budgeted-think /
        // HIPFIRE_DFLASH_CHAT=0), so its drafter state must not survive here.
        if let Some(s) = m.speculator.as_mut() {
            s.reset(gpu);
        }
        // Zero DeltaNet state on reset. qwen35 recurrent state lives in the
        // bundle (ModelState::Qwen35), not the always-None m.dn_state/m.kv_cache.
        // Inlined (disjoint field access) because a `&tokenizer` borrow of `m`
        // is live here.
        if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
            let dn = &b.dn_state;
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
        if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(ModelState::Llama(b)) = m.state.as_mut() {
            b.kv.compact_offset = 0;
        }
        if let Some(ad) = m.kv_adaptive.as_mut() {
            ad.reset();
        }
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
            m.seq_pos,
            raw_q_tokens.len(),
            drafter_gpu.is_some()
        );
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        if m.seq_pos == 0 {
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
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
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
            system: if m.seq_pos == 0 { system_prompt } else { None },
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
    // On HIT we set `m.seq_pos = LCP` and override `new_tokens` to the
    // suffix slice [LCP..] so the prefill below only writes new tokens.
    // DeltaNet state at position LCP is already correct (cumulative from
    // prior decode). On MISS (divergence in the middle) we full-reset
    // (seq_pos=0, conversation_tokens.clear(), zero DeltaNet, KV
    // compact_offset=0) and prefill the FULL rendered prompt — DeltaNet
    // is not reversible to position M<N so partial rollback is unsafe.
    let cache_kill_switch = std::env::var("HIPFIRE_QWEN_PROMPT_CACHE").ok().as_deref() == Some("0");
    let pflash_active = pflash_cfg
        .map(|c| !matches!(c.mode, hipfire_arch_qwen35::pflash::PflashMode::Off))
        .unwrap_or(false);
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
    let jinja_active = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0")
        && m.chat_template.is_some();
    // Cache-with-Jinja (item #37): `jinja_active` is NO LONGER a disqualifier.
    // When jinja is active the prompt-build below routes through
    // `build_cached_history_jinja` (verbatim assistant-turn splice through the
    // model's trained template) instead of the ChatScaffold `build_cached_history`,
    // so the LCP forward-extension cache now works under HIPFIRE_JINJA_CHAT too.
    let cache_eligible = !cache_kill_switch
        && messages_history.is_some()
        && m.eviction.is_none()
        && !pflash_active
        && !m.conversation_tokens.is_empty();
    if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[qwen-cache eligible] eligible={} kill={} hist={} evict_none={} !pflash={} jinja={} conv_tok={}",
            cache_eligible, cache_kill_switch, messages_history.is_some(),
            m.eviction.is_none(), !pflash_active, jinja_active, m.conversation_tokens.len(),
        );
    }
    let mut cached_tokens_count: usize = 0;
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
            let template = m.chat_template.as_ref().unwrap();
            let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let cache_ref = &mut m.asst_turn_cache;
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
                            fp, msg.role, msg.content.len(), normalized.len(), primer.len(), hit.is_some(),
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
            let cache_ref = &mut m.asst_turn_cache;
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
                            fp, msg.role, msg.content.len(), normalized.len(),
                            msg.tool_calls.len(), hit.is_some(),
                        );
                    }
                    hit
                },
            )
        };
        // LCP detection vs m.conversation_tokens.
        let prior_len = m.conversation_tokens.len();
        let max_match = prior_len.min(rendered.len());
        let mut lcp = 0usize;
        while lcp < max_match && m.conversation_tokens[lcp] == rendered[lcp] {
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
                        &m.conversation_tokens[pre..lcp],
                        tokenizer.decode(&m.conversation_tokens[pre..lcp]),
                    );
                }
                if prior_post > lcp {
                    eprintln!(
                        "  prior_past[{}..{}] ids={:?} dec={:?}",
                        lcp,
                        prior_post,
                        &m.conversation_tokens[lcp..prior_post],
                        tokenizer.decode(&m.conversation_tokens[lcp..prior_post]),
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
                tokenizer.decode(&m.conversation_tokens[pre..lcp])
            } else {
                String::new()
            };
            let prior_post = (lcp + 12).min(prior_len);
            let prior_past_dec = if prior_post > lcp {
                tokenizer.decode(&m.conversation_tokens[lcp..prior_post])
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
                tokenizer.decode(&m.conversation_tokens[prior_len - 4..])
            } else {
                tokenizer.decode(&m.conversation_tokens[..])
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
            let evict_safe = m.pp <= 1
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
                m.prefill_checkpoints
                    .iter()
                    .rposition(|(p, _)| *p <= lcp && *p < rendered.len())
            } else {
                None
            };
            let resumed = if let Some(idx) = resume_idx {
                let rpos = m.prefill_checkpoints[idx].0;
                // RESTORE only (do NOT zero): roll the bundle's DeltaNet state
                // back to the checkpoint. Disjoint split: m.state and
                // m.prefill_checkpoints are different fields of `m`.
                let ok = if let (Some(ModelState::Qwen35(b)), Some(ck)) =
                    (m.state.as_mut(), m.prefill_checkpoints.get(idx))
                {
                    ck.1.restore_to(&mut b.dn_state, gpu).is_ok()
                } else {
                    false
                };
                if ok {
                    m.seq_pos = rpos;
                    // `evict_safe` guarantees compact_offset == 0, so setting
                    // seq_pos already points the KV write head at rpos — nothing
                    // to restore (checkpoints are only captured with offset 0).
                    m.conversation_tokens.truncate(rpos);
                    truncate_checkpoints(&mut m.prefill_checkpoints, idx + 1, gpu);
                    cached_tokens_count = rpos;
                    eprintln!(
                        "[qwen-cache resume] rewound to checkpoint pos={} (lcp={}, prior_len={}, rendered_len={}) — replaying {} tokens vs cold-prefilling {}",
                        rpos, lcp, prior_len, rendered.len(), rendered.len() - rpos, rendered.len(),
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
                    // state is non-reversible; treat as a miss. Inlined (not
                    // `full_reset_cold`) because a `&tokenizer` borrow of `m` is
                    // live here; these are disjoint field accesses. qwen35 state
                    // lives in the bundle (ModelState::Qwen35), not the always-None
                    // m.dn_state/m.kv_cache.
                    m.seq_pos = 0;
                    m.conversation_tokens.clear();
                    free_checkpoints(&mut m.prefill_checkpoints, gpu);
                    if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
                        let dn = &b.dn_state;
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
                    if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
                        b.kv_cache.compact_offset = 0;
                    }
                    if let Some(ModelState::Llama(b)) = m.state.as_mut() {
                        b.kv.compact_offset = 0;
                    }
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
            m.seq_pos = lcp;
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
    if jinja_active && !cache_eligible && m.seq_pos > 0 {
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // Free the speculator's (relocated) checkpoint ring on reset — this AR
        // path is reachable by a DFlash-capable model.
        if let Some(s) = m.speculator.as_mut() {
            s.reset(gpu);
        }
        // qwen35 recurrent state lives in the bundle (ModelState::Qwen35), not
        // the always-None m.dn_state/m.kv_cache. Inlined (disjoint field access)
        // because a `&tokenizer` borrow of `m` is live here.
        if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
            let dn = &b.dn_state;
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
        if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(ModelState::Llama(b)) = m.state.as_mut() {
            b.kv.compact_offset = 0;
        }
    }

    // KV-budget guard. Without eviction the physical buffer is the hard cap;
    // we must fit prefill + generation + trailer in one allocation. With
    // eviction, physical is bounded by physical_cap regardless of total tokens
    // — the chunked prefill below calls maybe_evict between chunks, and the
    // decode loop evicts after every token. The only ceiling under eviction is
    // the advertised context window (max_seq) — refuse requests that would
    // overflow it in absolute position terms (current absolute + new).
    let trailer = nl.len();
    let absolute_pos = m.seq_pos.saturating_add(
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
        if m.seq_pos
            .saturating_add(new_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.physical_cap
        {
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
    } else if absolute_pos
        .saturating_add(new_tokens.len())
        .saturating_add(max_tokens)
        .saturating_add(trailer)
        > m.max_seq
    {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds advertised context window: absolute={} + prefill={} + max_tokens={} + trailer={} > max_seq={}"}}"#,
            id,
            absolute_pos,
            new_tokens.len(),
            max_tokens,
            trailer,
            m.max_seq
        );
        let _ = stdout.flush();
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

    if m.arch_id == 5 || m.arch_id == 6 {
        // Qwen3.5 / Qwen3.5-MoE AR decode via the generic ArchDispatch driver
        // (Inc 1 Task 1.4d — flipped from the legacy inline arm after the dual-run
        // shadow-parity gate proved token-identity, single-GPU + emulated-2, FP32
        // DeltaNet + deterministic). The legacy arm + parity scaffold are removed.
        let mut __disp = Qwen35Dispatch { m: &mut *m };
        ar_generate(
            &mut __disp,
            gpu,
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
        // LLaMA path -- multi-turn aware
        let ModelState::Llama(b) = m.state.as_mut().unwrap() else {
            unreachable!()
        };
        let config = &b.config;
        let weights = &b.weights;
        let scratch = &b.scratch;
        let kv = &mut b.kv;

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
        let mut filter = EosFilter::new(EosFilterConfig::default());

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
            if let FilterAction::Emit(text_bytes) = filter.observe(new_bytes) {
                let text = std::str::from_utf8(&text_bytes).unwrap();
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"token","id":"{}","text":{}}}"#,
                    id,
                    serde_json::to_string(&text).unwrap_or_default()
                );
                let _ = stdout.flush();
            }

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
                            fp, msg.content.len(), normalized.len(),
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
    std::env::var("HIPFIRE_DEEPSEEK4_SPEC_DECODE")
        .ok()
        .map(|v| v == "1")
        .unwrap_or_else(|| match std::env::var("HIPFIRE_MTP_MODE").ok().as_deref() {
            Some("on") => true,
            Some("off") => false,
            _ => m.mtp_mode == "on" || (m.mtp_mode == "auto" && m.mtp_weights_present),
        })
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
) {
    // eos token (for the DSML prompt build) from the bundle — immutable peek.
    let eos_tok = match m.state.as_ref() {
        Some(ModelState::Deepseek4(b)) => b.eos_tok,
        _ => {
            emit_error_with_id(stdout, id, "deepseek4 bundle missing on arch_id=9 spec");
            return;
        }
    };

    // DSML prompt render (same builder the bespoke loop uses).
    let prompt_ids = {
        let tokenizer = match m.tokenizer.as_ref() {
            Some(t) => t,
            None => {
                emit_error_with_id(stdout, id, "tokenizer not loaded");
                return;
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
            &mut m.asst_turn_cache,
        )
    };
    if prompt_ids.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return;
    }

    // spec_k: env chain → 2 (the deepseek4-specific default; see generate_deepseek4).
    let spec_k: usize = std::env::var("HIPFIRE_DEEPSEEK4_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            std::env::var("HIPFIRE_MTP_K")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(2);

    // Prefix-cache plan (ds4 policy: forced-cold on partial, ring-safety length
    // guard, step-back exact). Pure decision; the GPU teardown is applied below.
    let plan = hipfire_runtime::cache_plan::plan_cache(
        &prompt_ids,
        &m.conversation_tokens,
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
        > m.physical_cap
    {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq",
                plan.start_pos + suffix.len(),
                max_tokens,
                m.physical_cap
            ),
        );
        return;
    }

    // DSA decode-cache miss teardown (the part NOT done by the drafter's
    // cache-miss `state.reset()` or generate_spec's seq_pos/conversation clear):
    // zero the position-indexed rings + invalidate the captured decode graph so a
    // fresh conversation reproduces a freshly-launched daemon's clean state.
    if !plan.cache_hit {
        if let Some(ModelState::Deepseek4(b)) = m.state.as_mut() {
            b.state.zero_decode_caches(gpu);
        }
        gpu.invalidate_graph_state();
    }

    // The ds4 emitter builds its in-step tool-call grammar from the raw tool
    // JSON inside `make_spec_emitter`, but that grammar masks per-token over the
    // decoded vocab — which the neutral `SpecEmitCtx` can't lazily derive (it has
    // no `&mut m`). Build/cache the vocab Arc here when tools are present and
    // hand it down; mirrors the lazy cache the old `build_deepseek4_spec_grammar`
    // did internally.
    let decoded_vocab: Option<std::sync::Arc<Vec<String>>> =
        if tools.map_or(false, |t| !t.is_empty()) {
            if m.decoded_vocab.is_none() {
                let tok = m.tokenizer.as_ref().expect("tokenizer present");
                let n = tok.vocab_size();
                let v: Vec<String> = (0..n).map(|id| tok.decode(&[id as u32])).collect();
                m.decoded_vocab = Some(std::sync::Arc::new(v));
            }
            m.decoded_vocab.clone()
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
        Some(r) => r,
        // Abort / error early-exit already wrote its own done/error envelope.
        None => return,
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
    // `pbs` is a disjoint field from `m.state`, so it borrows independently of
    // the bundle's `&mut state` below.
    let pbs = m
        .deepseek4_pbs
        .as_ref()
        .expect("deepseek4_pbs missing on arch_id=9 generate");
    // The single-GPU ds4 bundle (config/weights/state/eos) lives in
    // `ModelState::Deepseek4`. Field-borrow it disjointly so `cfg`/`weights`
    // (shared) and `state` (`&mut`) are live simultaneously, exactly as the
    // forward path needs.
    let Some(ModelState::Deepseek4(b)) = m.state.as_mut() else {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"deepseek4_config missing on arch_id=9 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    };
    let cfg = &b.config;
    let weights = &b.weights;
    let state = &mut b.state;
    let eos_tok = b.eos_tok;

    let prompt_ids = build_deepseek4_dsml_prompt(
        tokenizer,
        system_prompt,
        tools,
        messages_history,
        prompt,
        think_mode,
        eos_tok,
        &mut m.asst_turn_cache,
    );

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
    let lcp = if lcp > 0 && lcp < m.conversation_tokens.len() {
        0
    } else {
        lcp
    };

    if lcp == 0 {
        // Cache miss — start a fresh conversation in V4F's state.
        state.reset();
        // reset() only rewinds n_tokens; the position-indexed decode caches
        // (SWA ring, compressed/full KV, indexer scratch) still hold the prior
        // turn's residue, which bleeds into this fresh conversation's forward
        // and makes greedy output drift turn-to-turn (the "recall/tool-calls
        // unreliable" symptom). Zero them so a fresh conversation reproduces a
        // freshly-launched daemon's clean, deterministic state.
        state.zero_decode_caches(gpu);
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

    // O2b-2 capacity guard (ds4 single-GPU): after any cache reset above, the
    // KV ends at start_pos + suffix_tokens.len() (== prompt_ids.len()) and
    // decode appends max_tokens. forward_prefill_batch_chunked writes into a KV
    // sized for m.physical_cap; overrunning it is a KV-overrun panic that takes
    // down serve. Emit a clean error and return BEFORE prefill.
    // saturating_add: an adversarially huge max_tokens must not wrap usize and
    // slip under the cap.
    if (start_pos as usize)
        .saturating_add(suffix_tokens.len())
        .saturating_add(max_tokens)
        > m.physical_cap
    {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
            id,
            start_pos as usize + suffix_tokens.len(),
            max_tokens,
            m.physical_cap
        );
        let _ = stdout.flush();
        return;
    }

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
            emit_error_with_id(stdout, id, format!("deepseek4prefill failed: {e:?}"));
            return;
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
        // using V4F turn. The cache lives on `LoadedModel.decoded_vocab`
        // as an `Arc<Vec<String>>` and is cleared on model unload.
        //
        // Borrow note: `m.decoded_vocab` is a disjoint field from
        // `m.state` (whose `ModelState::Deepseek4` bundle `state` holds `&mut`
        // to) and from `m.tokenizer` (which `tokenizer` holds `&` to), so the
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
    if m.lfm2moe().is_none() {
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
        // LFM2.5 (arch_id 11) REQUIRES its embedded Jinja chat_template — the
        // hand-rolled Plain ChatML path omits LFM2's `<|startoftext|>` BOS and
        // produces garbage. Force jinja on for arch 11 (falls back to Plain only if
        // the .hfq carries no template, e.g. an older A1B convert).
        // Jinja default-ON (flipped 2026-06-09): render through the model's chat
        // template for ALL arches; opt out with HIPFIRE_JINJA_CHAT=0 (hand-rolled
        // ChatML/Plain). Falls back to Plain automatically when no template resolves.
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
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
                    eprintln!("[daemon] jinja render failed in lfm2moe path ({e}) — falling back to Plain");
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
        return;
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
    let stop_toks: Vec<u32> = {
        let tk = m.tokenizer.as_ref().unwrap();
        let mut v = vec![eos_tok];
        for s in ["<|endoftext|>", "</s>", "<|im_end|>"] {
            let ids = tk.encode(s);
            if ids.len() == 1 && !v.contains(&ids[0]) {
                v.push(ids[0]);
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
    let _ = m.lfm2moe_mut().unwrap().state.reset(gpu);
    m.seq_pos = 0;
    m.conversation_tokens.clear();

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
        return;
    }

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token. The LAST decode_step's logits
    // are the predictions for the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let b = m.lfm2moe_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
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
        if stop_toks.contains(&next_tok) {
            break;
        }

        let frag = {
            let tokenizer = m.tokenizer.as_ref().unwrap();
            tokenizer.decode(&[next_tok])
        };
        // String-level EOS-class guard. The id-based `stop_toks` above misses
        // `<|endoftext|>` because encoding the literal STRING doesn't round-trip
        // to the special-token id (it yields subwords), so the real token id is
        // never in the set. The daemon decodes one token at a time, so the
        // leaking turn-end token arrives as its own frag — catch it on the
        // decoded text and stop WITHOUT emitting (was: "...Paris.<|endoftext|>").
        if matches!(frag.trim(), "<|endoftext|>" | "</s>" | "<|im_end|>") {
            break;
        }
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
            let b = m.lfm2moe_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
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

    m.seq_pos = m.lfm2moe().unwrap().state.n_tokens;

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
    if m.minimax().is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"minimax_config missing on arch_id=10 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
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
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
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
                    eprintln!("[daemon] jinja render failed in minimax path ({e}) — falling back to Plain");
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
        return;
    }

    let eos_tok = m.minimax().unwrap().eos_tok;

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
        m.minimax_mut().unwrap().state.reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();

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
            return;
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
    let prefill_ids: Vec<u32> =
        {
            let prior_len = m.conversation_tokens.len();
            let max_match = prior_len.min(prompt_ids.len());
            let mut lcp = 0usize;
            while lcp < max_match && m.conversation_tokens[lcp] == prompt_ids[lcp] {
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
                prior_len, prompt_ids.len(), lcp, cache_hit, cache_hit && partial,
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
                m.conversation_tokens.truncate(lcp);
                m.seq_pos = lcp;
                prompt_ids[lcp..].to_vec()
            } else {
                if prior_len > 0 {
                    m.minimax_mut().unwrap().state.reset();
                    m.seq_pos = 0;
                    m.conversation_tokens.clear();
                }
                prompt_ids.clone()
            }
        };

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token, or chunked batched prefill.
    // Disjoint field borrows of `m` (config / weights / state) let us also
    // push to `m.conversation_tokens` in the same scope. The LAST forward's
    // logits are the predictions for the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let b = m.minimax_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        // Batched prefill: process the prompt in chunks of <=64 tokens through
        // the batched verify forward (one weight read per chunk vs one
        // decode_step per token) → much lower TTFT. Validated byte-identical to
        // the sequential path (cosine 1.0). DEFAULT ON when every layer's expert
        // dtypes have batched kernels; the pre-check routes unsupported tiers
        // (MQ3-Lloyd etc.) to the sequential path to avoid a mid-pass error.
        // Force off with HIPFIRE_MINIMAX_BATCH_PREFILL=0.
        let batch_prefill = std::env::var_os("HIPFIRE_MINIMAX_BATCH_PREFILL")
            .map_or(true, |v| v != "0")
            && minimax::forward::forward_batch_supported(weights);
        if batch_prefill && !prefill_ids.is_empty() {
            let mut pos = state.n_tokens;
            for chunk in prefill_ids.chunks(64) {
                match minimax::forward::forward_batch(cfg, weights, state, gpu, chunk, pos) {
                    Ok(logits) => last_logits = logits,
                    Err(e) => {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("minimax batch prefill failed: {e:?}"),
                        );
                        return;
                    }
                }
                pos += chunk.len();
            }
        } else {
            let mut position = state.n_tokens as u32;
            for &tok in &prefill_ids {
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
    }
    for &tok in &prefill_ids {
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
            let b = m.minimax_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            // hipGraph decode (opt-in via HIPFIRE_MINIMAX_GRAPH=1, default eager
            // — measured only +1.0% on gfx1151). First call warms up eager, then
            // captures + replays.
            minimax::forward::decode_step_with_graph(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("minimax decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.minimax().unwrap().state.n_tokens;

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

/// Cohere2-MoE / North-Mini-Code (arch_id=12) generate path. Mirrors
/// `generate_minimax`, plus: BATCHED `forward_batch` prefill (≈9×, see the
/// prefill block) and a Cohere agentic-marker state machine in the decode loop
/// that suppresses the special markers, routes `<|START_THINKING|>` content to a
/// reasoning channel, emits `<|START_TEXT|>` content as the visible answer, and
/// parses `<|START_ACTION|>` blocks into `tool_calls` events. Decode is plain
/// per-token `decode_step` (no hipGraph variant yet). Out of scope (NOT wired):
/// spec-decode, MTP, grammar-constrained decoding, tool EXECUTION, repeat
/// penalty, multi-GPU.
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
    if m.cohere2moe().is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"cohere2moe_config missing on arch_id=12 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
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
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
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
                            rendered.len(), ids.len(),
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
                    return;
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
            eprintln!("[daemon] cohere2moe cannot build a valid prompt frame ({why}) — refusing ChatML fallback");
            emit_error_with_id(
                stdout,
                id,
                format!("cohere2moe requires its jinja chat template ({why})"),
            );
            return;
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

    let eos_tok = m.cohere2moe().unwrap().eos_tok;

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
        emit_error_with_id(
            stdout,
            id,
            format!(
                "cohere2moe: prompt is {} tokens but KV capacity (max_seq) is {} — load with a larger max_seq or shorten the prompt",
                prompt_ids.len(),
                max_seq
            ),
        );
        let _ = m.cohere2moe_mut().unwrap().state.reset(gpu);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        return;
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
    let prefill_ids: Vec<u32> = {
        let prior_len = m.conversation_tokens.len();
        let max_match = prior_len.min(prompt_ids.len());
        let mut lcp = 0usize;
        while lcp < max_match && m.conversation_tokens[lcp] == prompt_ids[lcp] {
            lcp += 1;
        }
        // A usable common prefix that leaves at least one fresh token to prefill.
        // `partial` is the divergence case (lcp < prior_len); lcp == prior_len is
        // the degenerate pure-extension case (rewind is then a no-op).
        let cache_hit = lcp > 0 && lcp < prompt_ids.len();
        let partial = lcp < prior_len;
        if std::env::var("HIPFIRE_QWEN_CACHE_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[cohere2moe-cache] prior_len={} rendered_len={} lcp={} hit={} partial={} n_tokens={}",
                prior_len, prompt_ids.len(), lcp, cache_hit, cache_hit && partial,
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
            m.conversation_tokens.truncate(lcp);
            m.seq_pos = lcp;
            prompt_ids[lcp..].to_vec()
        } else {
            if prior_len > 0 {
                let _ = m.cohere2moe_mut().unwrap().state.reset(gpu);
                m.seq_pos = 0;
                m.conversation_tokens.clear();
            }
            prompt_ids.clone()
        }
    };

    let t0 = Instant::now();

    // ── Prefill: BATCHED `forward_batch` (~9× the per-token path) when the
    // expert tier supports the indexed-MoE GEMV (MQ4/MQ6), chunked at 256 (the
    // WMMA Q8-projection + grouped-MoE sweet spot). `forward_batch` writes KV at
    // [start_pos..start_pos+b] and its attention covers the cached prefix
    // [0..start_pos], so it composes with the prompt cache (start_pos = lcp).
    // Q8/F16 expert tiers (no indexed kernel) fall back to per-token decode_step.
    // The LAST forward's logits predict the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let b = m.cohere2moe_mut().unwrap();
        let cfg = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        if cohere2moe::forward::forward_batch_supported(weights) && prefill_ids.len() > 1 {
            let mut i = 0;
            while i < prefill_ids.len() {
                let end = (i + 256).min(prefill_ids.len());
                let start_pos = state.n_tokens;
                match cohere2moe::forward::forward_batch(
                    cfg,
                    weights,
                    state,
                    gpu,
                    &prefill_ids[i..end],
                    start_pos,
                ) {
                    Ok(logits) => last_logits = logits,
                    Err(e) => {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("cohere2moe batched prefill failed: {e:?}"),
                        );
                        return;
                    }
                }
                i = end;
            }
        } else {
            let mut position = state.n_tokens as u32;
            for &tok in &prefill_ids {
                match cohere2moe::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                    Ok(logits) => last_logits = logits,
                    Err(e) => {
                        emit_error_with_id(stdout, id, format!("cohere2moe prefill failed: {e:?}"));
                        return;
                    }
                }
                position += 1;
            }
        }
    }
    for &tok in &prefill_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = t0.elapsed().as_millis();

    // Re-emit a leading `<think>\n` opener into the token stream (display-only,
    // not pushed to state) when the rendered prompt primed the assistant turn
    // inside a reasoning block, so downstream `<think>` consumers see a
    // well-formed block. No-op for templates that don't prime think.
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

    // Cohere agentic markers are SPECIAL TOKENS. Resolve their ids once (scoped
    // borrow so the loop can still mutate `m`). This model emits <|START_TEXT|>
    // for its response (NOT the template's <|START_RESPONSE|>, which isn't a
    // real special token here — verified empirically).
    let (mk_think0, mk_think1, mk_text0, mk_text1, mk_act0, mk_act1) = {
        let tk = m.tokenizer.as_ref().unwrap();
        // `encode` SPLITS these added tokens (they round-trip on DECODE only),
        // so resolve content→id via special_token_id, with North-Mini-Code's
        // fixed marker ids as the fallback.
        let mark = |s: &str, fb: u32| -> u32 { tk.special_token_id(s).unwrap_or(fb) };
        (
            mark("<|START_THINKING|>", 255010),
            mark("<|END_THINKING|>", 255011),
            mark("<|START_TEXT|>", 255012),
            mark("<|END_TEXT|>", 255013),
            mark("<|START_ACTION|>", 255014),
            mark("<|END_ACTION|>", 255015),
        )
    };
    #[derive(PartialEq, Clone, Copy)]
    enum Sec {
        Pre,
        Think,
        Text,
        Action,
    }
    let mut sec = Sec::Pre;
    let mut action_buf = String::new();

    // Degenerate-output guards. The long-context forward can collapse into a
    // single-token attractor — observed in a Pi session where a ~30K-token
    // prompt drove the model to emit `<PAD>` until the client aborted ~9.5 min
    // later. These stop the decode promptly instead of hanging. They do NOT mask
    // the underlying forward bug: a fired guard MEANS the forward produced
    // garbage (long-context attention/KV) and is logged loudly so it's caught.
    let pad_tok = m.tokenizer.as_ref().unwrap().special_token_id("<PAD>");
    let mut last_tok: u32 = u32::MAX;
    let mut repeat_run: usize = 0;
    const REPEAT_GUARD: usize = 24; // consecutive-identical-token attractor

    // Empty-turn guard: North sometimes emits <|END_THINKING|> then
    // <|END_OF_TURN_TOKEN|> with no response or action, surfacing as a turn with
    // reasoning only and empty visible content (the model ends after <think>
    // without returning a result). Track whether anything visible (a text token
    // or a tool_call) was produced; if EOS arrives before that, mask it and
    // re-sample so the model is forced into <|START_TEXT|>/<|START_ACTION|> and
    // actually returns content. Opt out with HIPFIRE_C2M_EMPTY_TURN_GUARD=0.
    let empty_turn_guard = std::env::var("HIPFIRE_C2M_EMPTY_TURN_GUARD")
        .ok()
        .as_deref()
        != Some("0");
    let mut emitted_visible = false;
    let mut eos_suppressions = 0usize;
    const MAX_EOS_SUPPRESS: usize = 3;

    // Think-budget force-close (mechanism #2): a heavy reasoner can out-think its
    // token budget and return reasoning only (it hits max_tokens still inside
    // <think>). Reserve room for an answer; honor an explicit max_think_tokens
    // when the client sets one. When thinking reaches the budget with nothing
    // visible yet, inject <|END_THINKING|> + <|START_TEXT|> so the model closes
    // thinking and answers within the reserve instead of hitting the cap empty.
    let think_reserve = (max_tokens / 4).clamp(64, 512).min(max_tokens / 2);
    let think_budget = if max_think_tokens > 1 {
        max_think_tokens.min(max_tokens.saturating_sub(think_reserve))
    } else {
        max_tokens.saturating_sub(think_reserve)
    };
    let mut think_count = 0usize;
    let mut think_force_closed = false;
    let mut forced_toks: std::collections::VecDeque<u32> = std::collections::VecDeque::new();

    // Tool-calling robustness against non-Cohere harnesses: known tool names (to
    // snap a verbose/hallucinated name back to the real tool) and a buffer of the
    // visible output (to recover a tool call the model wrote as TEXT instead of
    // via <|START_ACTION|>). Handles both {function:{name}} (OpenAI) and {name}.
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
    // name -> [parameter names], for snapping glitched argument keys.
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
    let mut tool_calls_emitted = false;
    let mut vis_buf = String::new();

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    loop {
        if generated_count >= max_tokens {
            break;
        }
        if empty_turn_guard
            && forced_toks.is_empty()
            && !think_force_closed
            && !emitted_visible
            && sec == Sec::Think
            && think_count >= think_budget
        {
            eprintln!(
                "[cohere2moe] think-budget guard: force-closing thinking at {think_count} think-tok \
                 (budget {think_budget}, max_tokens {max_tokens}) — forcing an answer"
            );
            forced_toks.push_back(mk_think1);
            forced_toks.push_back(mk_text0);
            think_force_closed = true;
        }
        let mut next_tok = match forced_toks.pop_front() {
            Some(t) => t,
            None => deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng),
        };
        if next_tok == eos_tok {
            if empty_turn_guard && !emitted_visible && eos_suppressions < MAX_EOS_SUPPRESS {
                // Reasoning-only turn in progress — the model is ending after
                // <think> with nothing visible. FORCE a <|START_TEXT|> continuation
                // rather than re-sampling the EOS-masked distribution: re-sampling
                // drew from the low-probability tail (where garbage / other-language
                // tokens live), emitting a garbage first token that then derailed the
                // turn (the "veen hier" / "ماعopilot" glitches). Injecting the marker
                // puts the model in the response section so its NEXT token is sampled
                // normally (same as the think-budget force-close, which is coherent).
                // Close thinking first if we somehow took EOS while still inside it.
                eos_suppressions += 1;
                eprintln!(
                    "[cohere2moe] empty-turn guard: forcing START_TEXT (EOS after thinking, no \
                     visible output, #{eos_suppressions}/{MAX_EOS_SUPPRESS}) at gen {generated_count}"
                );
                if sec == Sec::Think {
                    forced_toks.push_back(mk_think1);
                }
                forced_toks.push_back(mk_text0);
                next_tok = forced_toks.pop_front().unwrap();
            } else {
                break;
            }
        }
        // Degenerate-output guards (see above). `<PAD>` is never a valid
        // generation token, and any token repeating REPEAT_GUARD× in a row is an
        // attractor. Either means the forward collapsed — stop before emitting
        // or decoding further, and log so the real bug isn't silently swallowed.
        if Some(next_tok) == pad_tok {
            eprintln!(
                "[cohere2moe] DEGENERATE OUTPUT: <PAD> (id {next_tok}) emitted at gen {generated_count}, \
                 ctx={} — forward collapse (long-context attention/KV bug); stopping",
                m.cohere2moe().map(|b| b.state.n_tokens).unwrap_or(0)
            );
            break;
        }
        if next_tok == last_tok {
            repeat_run += 1;
            if repeat_run >= REPEAT_GUARD {
                eprintln!(
                    "[cohere2moe] DEGENERATE OUTPUT: token {next_tok} repeated {repeat_run}× at gen {generated_count} \
                     (attractor) — forward collapse; stopping"
                );
                break;
            }
        } else {
            last_tok = next_tok;
            repeat_run = 1;
        }
        // Probe-mode token-id stream (HIPFIRE_EMIT_TOKEN_IDS=1). Was wired into
        // the qwen35/deepseek4 generate loops but NOT here, so coherence_probe
        // and token-id detectors silently saw nothing for north-mini-code.
        emit_committed_event(
            stdout,
            id,
            next_tok,
            generated_count,
            decode_t0.elapsed().as_millis() as u64,
        );
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        // Agentic-marker state machine — markers themselves are never emitted.
        if next_tok == mk_think0 {
            sec = Sec::Think;
        } else if next_tok == mk_text0 {
            sec = Sec::Text;
        } else if next_tok == mk_act0 {
            sec = Sec::Action;
            action_buf.clear();
        } else if next_tok == mk_think1 || next_tok == mk_text1 {
            sec = Sec::Pre;
        } else if next_tok == mk_act1 {
            // End of an action block → parse the JSON array into tool_calls,
            // snapping any verbose/hallucinated tool name to a real tool.
            let mut calls = cohere2moe::spec_emit::parse_cohere_action(&action_buf);
            cohere2moe::spec_emit::snap_call_names(&mut calls, &known_tools, &tool_params);
            if !calls.is_empty() {
                let _ = writeln!(
                    stdout,
                    "{}",
                    serde_json::json!({"type": "tool_calls", "id": id, "calls": calls}),
                );
                let _ = stdout.flush();
                emitted_visible = true;
                tool_calls_emitted = true;
            }
            sec = Sec::Pre;
        } else {
            // Build the fragment through serde_json so arbitrary UTF-8 can't
            // corrupt the JSONL line.
            let frag = {
                let tokenizer = m.tokenizer.as_ref().unwrap();
                tokenizer.decode(&[next_tok])
            };
            // Defense-in-depth: never emit a Cohere structural marker into
            // visible output / the action buffer. The ID state machine above
            // handles the 6 THINKING/TEXT/ACTION markers; this catches any OTHER
            // special token the model might emit (START_OF_TURN_TOKEN,
            // CHATBOT_TOKEN, START_TOOL_RESULT, …) — each decodes to a full
            // `<|MARKER|>`. The token is still fed to decode_step below; only its
            // emit is dropped, so a state-machine miss can never leak a marker.
            let is_marker = frag.len() > 4
                && frag.starts_with("<|")
                && frag.ends_with("|>")
                && frag[2..frag.len() - 2]
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_');
            if is_marker {
                // suppressed
            } else {
                match sec {
                    Sec::Action => action_buf.push_str(&frag),
                    Sec::Think => {
                        // Reasoning channel: tagged so clients can fold it; the CLI
                        // (ignoring unknown fields) shows it inline.
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({"type": "token", "id": id, "text": frag, "reasoning": true}),
                        );
                        let _ = stdout.flush();
                        think_count += 1;
                    }
                    Sec::Text | Sec::Pre => {
                        vis_buf.push_str(&frag);
                        let _ = writeln!(
                            stdout,
                            "{}",
                            serde_json::json!({"type": "token", "id": id, "text": frag}),
                        );
                        let _ = stdout.flush();
                        emitted_visible = true;
                    }
                }
            }
        }

        // Advance one step on the freshly sampled token (plain eager decode —
        // no hipGraph variant on this arch yet).
        let step = {
            let b = m.cohere2moe_mut().unwrap();
            let cfg = &b.config;
            let weights = &b.weights;
            let state = &mut b.state;
            let position = state.n_tokens as u32;
            cohere2moe::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("cohere2moe decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.cohere2moe().unwrap().state.n_tokens;

    // Tool-call-as-text recovery: if the model never emitted a <|START_ACTION|>
    // block but wrote a tool-call JSON array (`[{tool_name, parameters}]`) as
    // visible text — which happens when a non-Cohere harness primes it with a
    // generic tool-call format — parse it out and emit it as a real tool_calls
    // event so the harness can actually execute it.
    if !tool_calls_emitted {
        let mut recovered = cohere2moe::spec_emit::parse_cohere_action(&vis_buf);
        if !recovered.is_empty() {
            cohere2moe::spec_emit::snap_call_names(&mut recovered, &known_tools, &tool_params);
            eprintln!(
                "[cohere2moe] recovered {} tool_call(s) written as text (model skipped <|START_ACTION|>)",
                recovered.len()
            );
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::json!({"type": "tool_calls", "id": id, "calls": recovered}),
            );
            let _ = stdout.flush();
        }
    }

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
    // Qwen2 (arch 7) AR decode via the generic ar_generate driver (Inc 2 — flipped
    // from the legacy greedy loop; uplift: qwen2 gains ar_generate's n-gram loop
    // guard). generate_qwen2 keeps its raw-prompt preamble (no framing) + capacity
    // guard, then hands the outputs to ar_generate (which prefills + decodes).
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
    let state_ref = match m.state.as_mut() {
        Some(ModelState::Qwen2(b)) => b,
        _ => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"qwen2 state missing on arch_id=7 generate"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        }
    };
    let state = &mut state_ref.state;

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
    if state
        .next_pos
        .saturating_add(prompt_ids.len())
        .saturating_add(max_tokens)
        > state.max_seq
    {
        eprintln!(
            "[daemon] arch_id=7 context full ({}/{}) — resetting Qwen2State.next_pos",
            state.next_pos, state.max_seq,
        );
        state.reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();

        // O2b-2 capacity guard (qwen2 single): the reset above (next_pos=0)
        // recovers a grown multi-turn conversation, but a SINGLE prompt larger
        // than the whole context still overflows — prefilling it writes past
        // the KV (sized for state.max_seq) and panics, taking down serve. After
        // the reset, if prompt + generation still overflows, emit a clean error.
        // saturating_add: an adversarially huge max_tokens must not wrap usize
        // and slip under the cap.
        if prompt_ids.len().saturating_add(max_tokens) > state.max_seq {
            let cap = state.max_seq;
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq"}}"#,
                id,
                prompt_ids.len(),
                max_tokens,
                cap
            );
            let _ = stdout.flush();
            return;
        }
    }

    let t0 = Instant::now();

    // Hand the raw-prompt preamble outputs to the generic driver. ar_generate
    // prefills `new_tokens` and runs the decode loop; greedy params reproduce the
    // legacy loop (temp0 argmax, no framing/think/budget/grammar). state/cfg/weights
    // borrows above have ended, so Qwen2Dispatch can take &mut m.
    let __prefill_tokens = prompt_ids.len();
    let mut __disp = Qwen2Dispatch { m: &mut *m };
    ar_generate(
        &mut __disp,
        gpu,
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
}

fn generate_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
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
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let vision_config = m.vision_config.as_ref().unwrap();

    // Vision special-token IDs resolved from the tokenizer rather than
    // hardcoded constants. Different VL-capable Qwen variants ship with
    // different IDs for these tokens; a hardcoded mismatch silently
    // splices the wrong tokens into the prompt. Required at load time —
    // panic loudly here so the failure is at first-VL-request, not after
    // a successful but wrong forward pass.
    let image_pad_id = tokenizer
        .special_token_id("<|image_pad|>")
        .unwrap_or_else(|| panic!("VL tokenizer missing <|image_pad|> special token"));
    let vision_start_id = tokenizer
        .special_token_id("<|vision_start|>")
        .unwrap_or_else(|| panic!("VL tokenizer missing <|vision_start|> special token"));
    let vision_end_id = tokenizer
        .special_token_id("<|vision_end|>")
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
    let n_visual_tokens =
        n_patches / (vision_config.spatial_merge_size * vision_config.spatial_merge_size);

    // Capacity estimate including system prompt — a long system prompt
    // on first turn would otherwise let an over-budget request through
    // the soft check, only to fail the hard check after the expensive
    // vision encoder runs.
    let system_est = system_prompt
        .map(|s| tokenizer.encode(s).len())
        .unwrap_or(0);
    let prompt_est = tokenizer.encode(prompt).len() + system_est + n_visual_tokens + 20;

    if m.eviction.is_none()
        && m.seq_pos
            .saturating_add(prompt_est)
            .saturating_add(max_tokens)
            > m.max_seq
    {
        eprintln!(
            "[daemon/vl] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // Free the speculator's (relocated) checkpoint ring on reset.
        if let Some(s) = m.speculator.as_mut() {
            s.reset(gpu);
        }
        // VL is qwen35-vl (arch 5/8); its recurrent state lives in the bundle
        // (ModelState::Qwen35), not the always-None m.dn_state/m.kv_cache.
        // Inlined (disjoint field access) because a `&tokenizer` borrow of `m`
        // is live here.
        if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
            let dn = &b.dn_state;
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
        if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(ad) = m.kv_adaptive.as_mut() {
            ad.reset();
        }
    }

    if m.eviction.is_none() && prompt_est.saturating_add(max_tokens) > m.max_seq {
        write_error(
            stdout,
            id,
            &format!(
                "request size ({} tokens) exceeds loaded KV budget ({})",
                prompt_est.saturating_add(max_tokens),
                m.max_seq,
            ),
        );
        return;
    }

    let ModelState::Qwen35(b) = m.state.as_mut().unwrap() else {
        unreachable!()
    };
    let config = &b.config;
    let weights = &b.weights;
    let scratch = &b.scratch;
    let kv = &mut b.kv_cache;
    let dn = &mut b.dn_state;
    let vision_weights = m.vision_weights.as_ref().unwrap();

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
    let absolute_pos_vl = m.seq_pos.saturating_add(kv.compact_offset);
    let over_budget = if m.eviction.is_none() {
        m.seq_pos
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.physical_cap
    } else {
        absolute_pos_vl
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.max_seq
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
        &pixels,
        3,
        img_h,
        img_w,
        vision_config.patch_size,
        vision_config.temporal_patch_size,
        vision_config.spatial_merge_size,
    );
    let visual_tokens =
        qwen35_vl::vision_forward(gpu, vision_weights, vision_config, &patches, grid_h, grid_w)
            .expect("vision forward failed");

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
            if let Some(hipfire_runtime::triattn::EvictionResult {
                new_physical: new_phys,
                ..
            }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
            {
                m.seq_pos = new_phys;
            }
        }
    }

    m.conversation_tokens.extend_from_slice(&prompt_tokens);

    // hunt3 M-D: repeat-penalty / n-gram-block history must be scoped to the
    // GENERATED tokens only (mirrors the text path's `ngram_scope_start` set to
    // conversation_tokens.len() after prefill). Passing the full conversation
    // makes the trailing window prompt-dominated, suppressing the names/numbers
    // a VL transcription task must reproduce.
    let vl_ngram_scope_start = m.conversation_tokens.len();

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
        generated += 1;
        m.conversation_tokens.push(next_token);
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

        qwen35::forward_scratch(gpu, weights, config, next_token, m.seq_pos, kv, dn, scratch)
            .unwrap();
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            if let Some(hipfire_runtime::triattn::EvictionResult {
                new_physical: new_phys,
                ..
            }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
            {
                m.seq_pos = new_phys;
            }
        }
        logits = gpu.download_f32(&scratch.logits).unwrap();
        // hunt3 M-D: scope ngram-block + repeat-penalty history to generated-only.
        let vl_ngram_scope = &m.conversation_tokens[vl_ngram_scope_start..];
        llama::apply_ngram_block(&mut logits, vl_ngram_scope);
        if let Some((open, close)) = think_pair {
            block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
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
                        qwen35::forward_scratch(
                            gpu, weights, config, t, m.seq_pos, kv, dn, scratch,
                        )
                        .unwrap();
                        m.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult {
                                new_physical: new_phys,
                                ..
                            }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
                            {
                                m.seq_pos = new_phys;
                            }
                        }
                        m.conversation_tokens.push(t);
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
                    logits = gpu.download_f32(&scratch.logits).unwrap();
                    block_attractor_unclosed_cpu(
                        &mut logits,
                        &m.conversation_tokens,
                        open,
                        close,
                        20,
                        2,
                    );
                    // hunt3 M-D: generated-only repeat-penalty scope.
                    next_token = sampler::sample_cpu(
                        &mut logits,
                        &m.conversation_tokens[vl_ngram_scope_start..],
                        &vl_cfg,
                    );
                }
            }
        }
    }

    // ChatML \n boundary — run through forward to keep KV cache + DeltaNet in sync
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch).unwrap();
            m.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = ev.maybe_evict(gpu, kv, m.seq_pos).unwrap()
                {
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
) {
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
                        return;
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
                    return;
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
            return;
        }
    };
    let n_visual = img.n_visual_tokens();
    let n_patches = img.n_patches();
    eprintln!(
        "[dots-ocr] grid {}x{}, {} patches → {} visual tokens",
        img.grid_h, img.grid_w, n_patches, n_visual
    );

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
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
        write_error(stdout, id, &format!(
            "dots.ocr request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
            prompt_ids.len(), max_tokens, max_seq));
        return;
    }

    // 4. Vision encoder → merged visual tokens.
    let patch_cols = img.patches.len() / n_patches;
    let patches_gpu = match gpu.upload_f32(&img.patches, &[n_patches, patch_cols]) {
        Ok(t) => t,
        Err(e) => {
            write_error(stdout, id, &format!("dots.ocr patch upload failed: {e:?}"));
            return;
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
            write_error(
                stdout,
                id,
                &format!("dots.ocr vision_forward failed: {e:?}"),
            );
            return;
        }
    };
    let _ = gpu.free_tensor(patches_gpu);
    let merged = match gpu.download_f32(&merged_gpu) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(merged_gpu);
            write_error(
                stdout,
                id,
                &format!("dots.ocr merger download failed: {e:?}"),
            );
            return;
        }
    };
    let _ = gpu.free_tensor(merged_gpu);
    // Hard guard: merger output count MUST equal the imgpad-slot count, or
    // the splice silently corrupts the text context (PRD §"Vision token splicing").
    if merged.len() != n_visual * dim {
        write_error(
            stdout,
            id,
            &format!(
            "dots.ocr: merger produced {} values but prompt has {} <|imgpad|> slots × {} dims = {}",
            merged.len(), n_visual, dim, n_visual * dim),
        );
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
        Err(e) => {
            write_error(
                stdout,
                id,
                &format!("dots.ocr embed scratch alloc failed: {e:?}"),
            );
            return;
        }
    };
    let mut visual_idx = 0usize;
    let mut embed_err: Option<String> = None;
    for (pos, &token) in prompt_ids.iter().enumerate() {
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
        write_error(
            stdout,
            id,
            &format!("dots.ocr prefill embed build failed: {e}"),
        );
        return;
    }
    if let Err(e) =
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, text_cfg, state, &embeds)
    {
        write_error(
            stdout,
            id,
            &format!("dots.ocr batched prefill failed: {e:?}"),
        );
        return;
    }
    let prefill_tokens = prompt_ids.len();
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    // 6. Decode. Opt-in n-gram speculative decode when a speculator was built at
    // load (HIPFIRE_NGRAM_DRAFT=1, arch_id=8 gate in `spec_build`); else the
    // bespoke greedy AR loop below. The vision prefill above already advanced the
    // shared Qwen2 KV (`m.qwen2_state`), so both paths decode from the same warm
    // state — only the drafting differs. The n-gram verify always falls back to
    // the target's greedy argmax, so spec output is byte-identical to AR; only τ
    // (speed) changes. The prefill bindings above (`tokenizer`/`config`/`state`/…)
    // are released here so the speculative branch can take `&mut m`; the AR path
    // re-borrows them below.
    if m.speculator.is_some() {
        decode_vl_dots_ocr_ngram(
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
        return;
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = m.dots_ocr_config.as_ref().unwrap();
    let text_cfg = &config.text;
    let weights = m.dots_ocr_weights.as_ref().unwrap();
    let state = m.qwen2_state.as_mut().unwrap();

    // Greedy decode, streaming in the daemon JSONL protocol.
    let eos_set: Vec<u32> = if text_cfg.eos_token_ids.is_empty() {
        vec![text_cfg.eos_token_id]
    } else {
        text_cfg.eos_token_ids.clone()
    };
    let mut next = match gpu.argmax_f32(&state.logits, text_cfg.vocab_size) {
        Ok(t) => t,
        Err(e) => {
            write_error(stdout, id, &format!("dots.ocr argmax failed: {e:?}"));
            return;
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
                write_error(stdout, id, &format!("dots.ocr decode failed: {e:?}"));
                return;
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
}

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
) {
    use hipfire_arch_dots_ocr::DotsOcrBundle;
    // Move the live decoder state into a SpecTarget bundle; restored on return.
    let mut bundle = DotsOcrBundle {
        config: m.dots_ocr_config.take().unwrap(),
        weights: m.dots_ocr_weights.take().unwrap(),
        state: m.qwen2_state.take().unwrap(),
    };
    let mut spec = m.speculator.take().unwrap();
    // `m.tokenizer` is a disjoint field → coexists with the takes above and the
    // restore below; the loop never touches `m`.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    run_dots_ocr_ngram_loop(
        &mut bundle,
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
    m.dots_ocr_config = Some(bundle.config);
    m.dots_ocr_weights = Some(bundle.weights);
    m.qwen2_state = Some(bundle.state);
    m.speculator = Some(spec);
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
) {
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
    // `spec_advance(&[], prompt_len, reset=false)` just argmaxes the live
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
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","tokens":0,"tok_s":0.0,"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":0.0,"decode_tok_s":0.0,"ttft_ms":{:.1}}}"#,
                id,
                prefill_tokens,
                prefill_s * 1000.0,
                prefill_s * 1000.0
            );
            let _ = stdout.flush();
            return;
        }
        Err(e) => {
            write_error(stdout, id, &format!("dots.ocr spec prefill: {e}"));
            return;
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
            break;
        }
        // Context-overflow guard (matches generate_spec): one window writes up
        // to `block_size` KV slots.
        if position.saturating_add(block_size) >= ctx_capacity {
            break;
        }
        let step = match spec.step(gpu, bundle, position, seed_token, &emitted, None, 0.0) {
            Ok(s) => s,
            Err(e) => {
                write_error(stdout, id, &format!("dots.ocr spec_step: {e}"));
                break;
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
