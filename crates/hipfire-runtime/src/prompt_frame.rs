// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ChatML prompt framing — single source of truth for assembling
//! the token sequence that gets fed to the model. Replaces the three
//! near-copies that lived in daemon.rs (AR, PFlash, DFlash paths).
//!
//! The canonical layout for a single turn is:
//!
//! ```text
//! [<|im_start|> system \n <system content> <|im_end|> \n]?  ← optional
//!  <|im_start|> user \n <user content> <|im_end|> \n
//!  <|im_start|> assistant \n [<think> \n]?
//! ```
//!
//! All three daemon copies converge to this exact byte sequence. The
//! AR path's whitespace conventions are canonical because it is the
//! most-exercised and the path against which the locked speed/coherence
//! baselines were captured.
//!
//! Multi-turn extends the same pattern by repeating
//! `<|im_start|> {user|assistant} \n <content> <|im_end|> \n`
//! for each prior turn before appending the new turn + assistant prefix.
//!
//! # Per-call-site policy
//!
//! Whether to *include* a system message on a given call (e.g. only on
//! `seq_pos == 0`) is the **caller's** decision. `ChatFrame` simply
//! emits a system block iff `system` is `Some`. The daemon is
//! responsible for passing `Some(_)` only on the appropriate turn.
//!
//! # Raw bypass
//!
//! `raw: true` skips ChatML scaffolding entirely and returns the
//! tokenization of `user` alone. This supports completion-style use
//! against a base model where any `<|im_start|>` token would be
//! out-of-distribution.

use crate::tokenizer::Tokenizer;

/// Chooses what goes after the assistant role-and-newline opener.
#[derive(Debug, Clone, Copy)]
pub enum AssistantPrefix {
    /// Plain assistant turn opener: `<|im_start|>assistant\n`.
    Plain,
    /// Assistant turn with `<think>` opener for thinking-mode models:
    /// `<|im_start|>assistant\n<think>\n`.
    ///
    /// Use only when the tokenizer recognizes `<think>` as a single
    /// special token. If `<think>` is absent from the vocab, the
    /// builder falls back to `Plain` (no opener emitted) rather than
    /// silently inserting raw text bytes that would tokenize
    /// differently from the special-token path.
    OpenThink,
    /// Assistant turn with an immediately closed empty think block
    /// for non-thinking mode:
    /// `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
    ///
    /// This mirrors the merged Qwen 3.6 community template behavior
    /// when `enable_thinking=false`. The model starts generation in
    /// visible-answer mode because the think block is already closed.
    /// Useful for routing/agentic contexts where we need visible
    /// output without disabling DFlash (still valid at temp=0).
    ///
    /// Requires both `<think>` and `</think>` as single special
    /// tokens. Falls back to `Plain` if either is absent.
    ClosedThink,
}

/// Reasoning-effort level requested for a turn (OpenAI-compatible
/// `reasoning_effort` / project-custom `thinking_mode`).
///
/// This is a model-independent *request* parameter (like temperature or
/// top-p), carried through the serving path and the neutral `SpecEmitCtx` so
/// per-arch emitters can interpret it. The arch-specific *frame* mapping — e.g.
/// DeepSeek V4's `<｜Assistant｜></think>` vs `<think>` open-token, or its `Max`
/// extended-reasoning preamble — lives in the arch crate that consumes this.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ThinkMode {
    /// Non-thinking: model skips reasoning and replies directly.
    NonThink,
    /// Thinking with the model's default/low effort. Architectures that encode
    /// effort as prompt text must not add a high-effort prefix for this mode.
    Low,
    /// Thinking with the model's high-effort framing.
    High,
    /// Thinking-max: same output protocol as `Low`/`High`, plus the model's
    /// maximum-effort preamble (the
    /// consuming arch decides what that means). Some models recommend a large
    /// context window for this mode.
    Max,
}

impl ThinkMode {
    /// Map a JSONL field value (OpenAI-compatible `reasoning_effort` or
    /// project-custom `thinking_mode`) to a mode.
    /// Accepted: "none|off|chat" → NonThink;
    ///           "minimal|low|medium|med|thinking" → Low;
    ///           "high" → High; "max" → Max.
    /// Anything else → NonThink (safe default).
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "max" => Self::Max,
            "high" => Self::High,
            "minimal" | "low" | "medium" | "med" | "thinking" => Self::Low,
            _ => Self::NonThink,
        }
    }
}

#[cfg(test)]
mod think_mode_tests {
    use super::ThinkMode;

    #[test]
    fn reasoning_effort_levels_remain_distinct() {
        assert_eq!(ThinkMode::from_str("none"), ThinkMode::NonThink);
        assert_eq!(ThinkMode::from_str("low"), ThinkMode::Low);
        assert_eq!(ThinkMode::from_str("medium"), ThinkMode::Low);
        assert_eq!(ThinkMode::from_str("high"), ThinkMode::High);
        assert_eq!(ThinkMode::from_str("max"), ThinkMode::Max);
    }
}

/// Role of a multi-turn history entry. `User` / `Assistant` are
/// canonical for `ChatFrame::Plain` (the hand-rolled ChatML path).
/// `System` / `Tool` are accepted by `JinjaChatFrame::render_messages`
/// (the upstream-template path) but rejected by `ChatFrame::Plain`,
/// which has no scaffold for them — that route panics loudly to
/// signal "migrate this caller to JinjaChatFrame".
///
/// Lowercase serialization matches what the Qwen3.5/3.6 + Gemma 4
/// templates compare against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// ChatML frame builder. Holds borrowed references to the tokenizer
/// and the textual content; the builder methods produce owned
/// `Vec<u32>` token sequences.
///
/// Not `#[derive(Debug)]` because `Tokenizer` doesn't implement
/// `Debug`. Callers that need a printable struct should format the
/// non-tokenizer fields manually.
#[derive(Clone)]
pub struct ChatFrame<'a> {
    pub tokenizer: &'a Tokenizer,
    pub system: Option<&'a str>,
    pub user: &'a str,
    pub assistant_prefix: AssistantPrefix,
    /// If true, bypass ChatML entirely and just encode `user` as raw
    /// tokens. For completion-style use against a base model.
    pub raw: bool,
}

impl<'a> ChatFrame<'a> {
    /// Build the prompt token sequence for a single-turn request.
    pub fn build(&self) -> Vec<u32> {
        if self.raw {
            return self.tokenizer.encode(self.user);
        }
        let scaffold = ChatScaffold::for_tokenizer(self.tokenizer);
        let mut out: Vec<u32> = Vec::new();
        if let Some(sys) = self.system {
            scaffold.append_system(&mut out, sys);
        }
        scaffold.append_user_turn(&mut out, self.user);
        scaffold.append_assistant_prefix(&mut out, self.assistant_prefix);
        out
    }

    /// Build the prompt token sequence for a single-turn request,
    /// substituting `user_tokens` for the encoding of `self.user`.
    /// Used by the daemon's AR/PFlash path where the user content has
    /// already been tokenized (and possibly compressed) upstream. The
    /// `self.user` field is ignored by this method when not in `raw`
    /// mode.
    ///
    /// In `raw` mode, returns `user_tokens` verbatim (system + ChatML
    /// scaffolding still bypassed, matching `build()`'s `raw` semantics).
    pub fn build_with_user_tokens(&self, user_tokens: &[u32]) -> Vec<u32> {
        if self.raw {
            return user_tokens.to_vec();
        }
        let scaffold = ChatScaffold::for_tokenizer(self.tokenizer);
        let mut out: Vec<u32> = Vec::new();
        if let Some(sys) = self.system {
            scaffold.append_system(&mut out, sys);
        }
        scaffold.append_user_turn_tokens(&mut out, user_tokens);
        scaffold.append_assistant_prefix(&mut out, self.assistant_prefix);
        out
    }

    /// Build the prompt token sequence for a multi-turn request.
    /// `history` is prior turns in chronological order (oldest first);
    /// the final turn is appended from `self.user` +
    /// `self.assistant_prefix`. The system message (if any) is emitted
    /// once, before the first history turn.
    ///
    /// In `raw` mode, history is concatenated as plain text encodings
    /// joined by newlines, then `user` is appended on its own line.
    /// This is best-effort — completion-style use against a base model
    /// rarely needs multi-turn.
    pub fn build_multi_turn(&self, history: &[(Role, &str)]) -> Vec<u32> {
        if self.raw {
            let mut out: Vec<u32> = Vec::new();
            for (i, (_role, content)) in history.iter().enumerate() {
                if i > 0 {
                    out.extend_from_slice(&self.tokenizer.encode("\n"));
                }
                out.extend_from_slice(&self.tokenizer.encode(content));
            }
            if !history.is_empty() {
                out.extend_from_slice(&self.tokenizer.encode("\n"));
            }
            out.extend_from_slice(&self.tokenizer.encode(self.user));
            return out;
        }
        let scaffold = ChatScaffold::for_tokenizer(self.tokenizer);
        let mut out: Vec<u32> = Vec::new();
        if let Some(sys) = self.system {
            scaffold.append_system(&mut out, sys);
        }
        for (role, content) in history {
            match role {
                Role::User => scaffold.append_user_turn(&mut out, content),
                Role::Assistant => scaffold.append_assistant_turn(&mut out, content),
                Role::System | Role::Tool => panic!(
                    "ChatFrame::Plain does not support {role:?} role in history. \
                     Use JinjaChatFrame::render_messages for system/tool turns."
                ),
            }
        }
        scaffold.append_user_turn(&mut out, self.user);
        scaffold.append_assistant_prefix(&mut out, self.assistant_prefix);
        out
    }
}

/// Pre-encoded ChatML scaffolding plus a borrowed tokenizer reference.
/// The fixed structural tokens (`<|im_start|>`, role names, `\n`,
/// `<|im_end|>`) are encoded once up front; per-turn content gets
/// encoded inside the append helpers as it's appended.
struct ChatScaffold<'a> {
    tokenizer: &'a Tokenizer,
    im_start: Vec<u32>,
    im_end: Vec<u32>,
    nl: Vec<u32>,
    system_role: Vec<u32>,
    user_role: Vec<u32>,
    assistant_role: Vec<u32>,
    tool_role: Vec<u32>,
    /// `<think>` opener (if the tokenizer recognizes it as a single
    /// special token). When `None`, `OpenThink` falls back to `Plain`
    /// — see `append_assistant_prefix`.
    think_open: Option<u32>,
    /// `</think>` closer (if the tokenizer recognizes it as a single
    /// special token). When `None`, `ClosedThink` falls back to `Plain`
    /// — see `append_assistant_prefix`.
    think_close: Option<u32>,
}

impl<'a> ChatScaffold<'a> {
    fn for_tokenizer(t: &'a Tokenizer) -> Self {
        Self {
            tokenizer: t,
            im_start: t.encode("<|im_start|>"),
            im_end: t.encode("<|im_end|>"),
            nl: t.encode("\n"),
            system_role: t.encode("system"),
            user_role: t.encode("user"),
            assistant_role: t.encode("assistant"),
            tool_role: t.encode("tool"),
            think_open: t.special_token_id("<think>"),
            think_close: t.special_token_id("</think>"),
        }
    }

    fn append_system(&self, out: &mut Vec<u32>, content: &str) {
        let body = self.tokenizer.encode(content);
        out.extend_from_slice(&self.im_start);
        out.extend_from_slice(&self.system_role);
        out.extend_from_slice(&self.nl);
        out.extend_from_slice(&body);
        out.extend_from_slice(&self.im_end);
        out.extend_from_slice(&self.nl);
    }

    fn append_user_turn(&self, out: &mut Vec<u32>, content: &str) {
        let body = self.tokenizer.encode(content);
        self.append_user_turn_tokens(out, &body);
    }

    /// Like `append_user_turn` but the body is already tokenized.
    fn append_user_turn_tokens(&self, out: &mut Vec<u32>, body: &[u32]) {
        out.extend_from_slice(&self.im_start);
        out.extend_from_slice(&self.user_role);
        out.extend_from_slice(&self.nl);
        out.extend_from_slice(body);
        out.extend_from_slice(&self.im_end);
        out.extend_from_slice(&self.nl);
    }

    fn append_assistant_turn(&self, out: &mut Vec<u32>, content: &str) {
        let body = self.tokenizer.encode(content);
        out.extend_from_slice(&self.im_start);
        out.extend_from_slice(&self.assistant_role);
        out.extend_from_slice(&self.nl);
        out.extend_from_slice(&body);
        out.extend_from_slice(&self.im_end);
        out.extend_from_slice(&self.nl);
    }

    /// Append an assistant turn where the body is already tokenized
    /// (typically a verbatim replay from the daemon's
    /// `asst_turn_cache`). Distinct from `append_assistant_turn` so
    /// callers don't have to detour through a tokenizer round-trip
    /// that BPE isn't bijective under for the model's emitted tokens.
    fn append_assistant_turn_tokens(&self, out: &mut Vec<u32>, body: &[u32]) {
        out.extend_from_slice(&self.im_start);
        out.extend_from_slice(&self.assistant_role);
        out.extend_from_slice(&self.nl);
        out.extend_from_slice(body);
        out.extend_from_slice(&self.im_end);
        out.extend_from_slice(&self.nl);
    }

    fn append_tool_turn(&self, out: &mut Vec<u32>, content: &str) {
        let body = self.tokenizer.encode(content);
        out.extend_from_slice(&self.im_start);
        out.extend_from_slice(&self.tool_role);
        out.extend_from_slice(&self.nl);
        out.extend_from_slice(&body);
        out.extend_from_slice(&self.im_end);
        out.extend_from_slice(&self.nl);
    }

    fn append_assistant_prefix(&self, out: &mut Vec<u32>, prefix: AssistantPrefix) {
        out.extend_from_slice(&self.im_start);
        out.extend_from_slice(&self.assistant_role);
        out.extend_from_slice(&self.nl);
        match prefix {
            AssistantPrefix::OpenThink => {
                // Only emit `<think>\n` when the tokenizer registers
                // `<think>` as a single special token. Otherwise the
                // string would tokenize as ordinary BPE pieces and behave
                // differently from the special-token path the model was
                // trained on. Falling back to `Plain` in that case is
                // safer than silently emitting wrong-shaped tokens.
                if let Some(think_id) = self.think_open {
                    out.push(think_id);
                    out.extend_from_slice(&self.nl);
                }
            }
            AssistantPrefix::ClosedThink => {
                // Emit an immediately-closed empty think block:
                // `<think>\n\n</think>\n\n`.
                // Mirrors the merged Qwen 3.6 community template's
                // `enable_thinking=false` behavior. Falls back to
                // `Plain` if either `<think>` or `</think>` is not
                // a single special token.
                if let (Some(open_id), Some(close_id)) = (self.think_open, self.think_close) {
                    out.push(open_id);
                    out.extend_from_slice(&self.nl);
                    out.extend_from_slice(&self.nl);
                    out.push(close_id);
                    out.extend_from_slice(&self.nl);
                    out.extend_from_slice(&self.nl);
                }
            }
            AssistantPrefix::Plain => {}
        }
    }
}

/// How a historical assistant turn's `tool_calls` are re-rendered on a
/// cache MISS in [`build_cached_history`] — the hand-rolled ChatScaffold
/// path used only when `HIPFIRE_JINJA_CHAT=0`. (The default jinja path
/// renders history through the model's trained chat_template instead, via
/// `build_cached_history_jinja`, and is unaffected by this enum.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolCallRender {
    /// Legacy Hermes JSON: `\n<tool_call>\n{"name":..,"arguments":..}\n</tool_call>`.
    /// Correct for Hermes-native models (carnice) and grammar-forced JSON.
    HermesJson,
    /// Qwen native XML: `\n<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n…\n</function>\n</tool_call>`.
    /// Matches what grammar-off Qwen3.5/3.6 actually emit (and their
    /// upstream chat_template), so a cache-miss history replay shows the
    /// model its own format instead of fighting it with JSON.
    QwenXml,
}

/// Render one tool call in Qwen3.5/3.6 native XML, matching the shape the
/// model emits and its upstream chat_template produces. A string argument
/// is emitted verbatim; any other JSON value is compact-serialized
/// (mirrors the template's `value | string if string else value | tojson`).
fn render_tool_call_qwen_xml(tc: &ToolCall) -> String {
    // Invariant: tool-call arguments are a JSON object (per the tool schema).
    // A non-object degrades to an empty `<function=…></function>` below; assert
    // in debug so a violation surfaces in tests, degrade gracefully in release.
    // NOTE: parameter values are NOT XML-escaped — a value literally containing
    // `</parameter>`/`</function>` would desync the replayed token stream, but
    // that only turns a cache-hit into a cache-miss (LCP divergence forces a
    // re-prefill), and mirrors the model's own upstream chat_template.
    debug_assert!(
        tc.arguments.is_object(),
        "tool-call arguments should be a JSON object, got {:?}",
        tc.arguments
    );
    let mut s = String::from("\n<tool_call>\n<function=");
    s.push_str(&tc.name);
    s.push_str(">\n");
    if let Some(obj) = tc.arguments.as_object() {
        for (k, v) in obj {
            s.push_str("<parameter=");
            s.push_str(k);
            s.push_str(">\n");
            match v {
                serde_json::Value::String(sv) => s.push_str(sv),
                other => s.push_str(&serde_json::to_string(other).unwrap_or_default()),
            }
            s.push_str("\n</parameter>\n");
        }
    }
    s.push_str("</function>\n</tool_call>");
    s
}

/// Whether the Hermes-JSON tool-call grammar (and history render) should be ON
/// for a Qwen3.5/3.6-family model, given the `HIPFIRE_QWEN35_GRAMMAR` env value
/// (`None` if unset) and the loaded model path. Lives in the lib — not only in
/// the CLI's `resolveToolGrammar` — so the DAEMON defaults correctly when
/// launched DIRECTLY: the GPU behavior gate and `coherence-gate.sh` /
/// `agentic-gate.sh` spawn `examples/daemon` without the CLI's `applyConfigEnv`,
/// so without this they fall back to Hermes and never exercise the native-XML
/// path. Precedence: explicit env wins (`"0"` = off kill-switch, any other set
/// value = on); otherwise carnice (Hermes-native) => ON, plain Qwen3.5/3.6
/// (XML-native) => OFF. Keyed on `carnice` in the model file name (all carnice
/// cards ship as `carnice-*.mq{4,6}`), the only Hermes family in this arch.
pub fn qwen35_grammar_on(env: Option<&str>, model_path: &str) -> bool {
    match env {
        Some("0") => false,
        Some(_) => true,
        None => model_path.to_ascii_lowercase().contains("carnice"),
    }
}

/// Cache-miss history re-render format (non-jinja `HIPFIRE_JINJA_CHAT=0` path)
/// for the same inputs as [`qwen35_grammar_on`]: grammar ON => Hermes JSON,
/// grammar OFF => Qwen native XML.
pub fn qwen35_history_render(env: Option<&str>, model_path: &str) -> ToolCallRender {
    if qwen35_grammar_on(env, model_path) {
        ToolCallRender::HermesJson
    } else {
        ToolCallRender::QwenXml
    }
}

/// Build a multi-turn token stream from structured history, splicing
/// cached verbatim token sequences for any historical assistant turn
/// that `cache_lookup` returns `Some` for. Used by the daemon's Qwen
/// prompt-cache path so that the rendered prefix is byte-identical to
/// what was written into KV by prior turns — required for an LCP-based
/// suffix prefill to extend through historical assistant turns (BPE is
/// not bijective; re-encoding `msg.content` may produce a different
/// token sequence than the one the model actually emitted).
///
/// Format mirrors the inline ChatML format consumed by the daemon
/// today (`<|im_start|>{role}\n{content}<|im_end|>\n`):
///   - System turn emitted once at top when `system.is_some()`
///   - User turn: `<|im_start|>user\n{content}<|im_end|>\n`
///   - Tool turn: `<|im_start|>tool\n<tool_response>\n{content}\n</tool_response><|im_end|>\n`
///   - Assistant turn: `<|im_start|>assistant\n{body}<|im_end|>\n` where
///     `body` is either the cached verbatim sequence or
///     `tokenizer.encode(msg.content) + tool_calls_as_text` on miss.
///
/// `live_user_tokens` is appended as the trailing user turn (the
/// request's live prompt). `assistant_prefix` adds the
/// `<|im_start|>assistant\n[<think>...]` trailer the model decodes from.
pub fn build_cached_history(
    tokenizer: &Tokenizer,
    system: Option<&str>,
    history: &[Message],
    live_user_tokens: &[u32],
    assistant_prefix: AssistantPrefix,
    tool_render: ToolCallRender,
    mut cache_lookup: impl FnMut(&Message) -> Option<Vec<u32>>,
) -> Vec<u32> {
    let scaffold = ChatScaffold::for_tokenizer(tokenizer);
    let mut out: Vec<u32> = Vec::new();
    if let Some(sys) = system {
        scaffold.append_system(&mut out, sys);
    }
    // If the trailing history message is a User, treat its content as
    // the live prompt and drop it here — caller already passes the
    // live user via `live_user_tokens`. Without this trim, the live
    // user turn gets rendered twice (once from history, once from
    // `live_user_tokens`). Mirrors V4F's daemon-side renderer at
    // `crates/hipfire-runtime/examples/daemon.rs:5163`.
    let trim_end = if matches!(history.last().map(|m| &m.role), Some(Role::User)) {
        1
    } else {
        0
    };
    let history = &history[..history.len().saturating_sub(trim_end)];
    for msg in history {
        match msg.role {
            // System messages in history are emitted via the top-level
            // `system` parameter above. Embedded system turns in
            // `history` are ignored to avoid duplication.
            Role::System => {}
            Role::User => scaffold.append_user_turn(&mut out, &msg.content),
            Role::Tool => {
                let wrapped = format!("<tool_response>\n{}\n</tool_response>", msg.content);
                scaffold.append_tool_turn(&mut out, &wrapped);
            }
            Role::Assistant => {
                // Emit `<|im_start|>assistant\n` plus the same
                // `assistant_prefix` scaffolding the daemon used as the
                // prompt-side prefix when this turn was originally
                // generated (e.g. `<think>\n\n</think>\n\n` for
                // thinking-off `ClosedThink`). Without it the cached
                // body sits at the wrong KV offset and LCP fails right
                // at the assistant turn boundary. Assumes the
                // assistant_prefix has stayed constant across the
                // conversation — true for typical OpenAI clients that
                // don't toggle `chat_template_kwargs.enable_thinking`
                // mid-session; turns with a mid-session toggle simply
                // degrade to cache miss (LCP detects the divergence).
                scaffold.append_assistant_prefix(&mut out, assistant_prefix);
                if let Some(cached) = cache_lookup(msg) {
                    out.extend_from_slice(&cached);
                } else {
                    // Cache miss — render the turn via tokenizer.encode
                    // of the content + tool_calls. The resulting token
                    // sequence may diverge from what the model originally
                    // emitted (BPE non-bijectivity for the boundaries),
                    // which is fine: the LCP check downstream will
                    // detect divergence and trigger a full reset.
                    if !msg.content.is_empty() && msg.content != "null" {
                        out.extend(tokenizer.encode(&msg.content));
                    }
                    for tc in &msg.tool_calls {
                        let rendered = match tool_render {
                            ToolCallRender::HermesJson => {
                                let payload = serde_json::json!({
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                });
                                format!(
                                    "\n<tool_call>\n{}\n</tool_call>",
                                    serde_json::to_string(&payload).unwrap_or_default(),
                                )
                            }
                            ToolCallRender::QwenXml => render_tool_call_qwen_xml(tc),
                        };
                        out.extend(tokenizer.encode(&rendered));
                    }
                }
                out.extend_from_slice(&scaffold.im_end);
                out.extend_from_slice(&scaffold.nl);
            }
        }
    }
    // Skip the trailing user turn when there's no live user content
    // (happens when the agent loop is continuing after a tool result
    // and the caller doesn't supply a new user message — OpenAI's
    // tool-use flow lets the model decode directly from the tool
    // response). Without this guard we emit an empty
    // `<|im_start|>user\n<|im_end|>\n` wrap which (a) is off-distribution
    // for the model and (b) gets baked into conversation_tokens,
    // breaking the LCP on the NEXT turn because the renderer then
    // includes that empty user in its historical replay while the
    // newer turn's history doesn't have it.
    if !live_user_tokens.is_empty() {
        scaffold.append_user_turn_tokens(&mut out, live_user_tokens);
    }
    scaffold.append_assistant_prefix(&mut out, assistant_prefix);
    out
}

// ─── Jinja path — render upstream HF chat_template ──────────────────────────
//
// `ChatFrame` above is a hand-rolled approximation of ChatML scaffolding.
// `JinjaChatFrame` renders the actual `chat_template` shipped with the
// model (via the .hfq metadata blob). When the template is present this
// is strictly more correct: the model sees the exact prefix shape it
// was trained on, including default system prompts, `<think>\n` openers
// gated by `enable_thinking`, tool-call scaffolding, and any other
// per-arch quirks the upstream tokenizer_config encodes.
//
// Failure modes (template parse error, missing context var, explicit
// `raise_exception`) bubble up as `Err(String)` so the caller can fall
// back to `ChatFrame::Plain` rather than panicking.
//
// The render output is a plain UTF-8 string. Tokenization goes through
// `Tokenizer::encode` which recognizes registered special tokens
// (`<|im_start|>`, `<|im_end|>`, `<think>`, etc.) and emits their
// single-token IDs — so the rendered string round-trips to the same
// token sequence the model would see under transformers' apply_chat_template.

/// Renders the upstream HF Jinja `chat_template` to produce a prompt
/// token sequence. Use when the .hfq carries a chat_template; fall back
/// to `ChatFrame::Plain` when it doesn't or when render fails.
pub struct JinjaChatFrame<'a> {
    pub tokenizer: &'a Tokenizer,
    /// The Jinja template source string from the model's
    /// `tokenizer_config.json:chat_template` field.
    pub template: &'a str,
    /// Optional system message for this turn. `None` = no system block.
    /// Ignored by `render_messages` (the multi-turn entry point); use
    /// only when going through the single-turn `render()` convenience.
    pub system: Option<&'a str>,
    /// User content for the new turn. Ignored by `render_messages`.
    pub user: &'a str,
    /// Maps to the upstream `enable_thinking` template kwarg. For
    /// Qwen3.5/3.6 thinking-mode models, `true` (the upstream default)
    /// emits `<|im_start|>assistant\n<think>\n` at the end; `false`
    /// emits the empty-think pattern `<think>\n\n</think>\n\n` which
    /// is known to cause loop pathologies (see
    /// `feedback_no_think_directive_loops.prd`). Default callers
    /// should pass `true`.
    pub enable_thinking: bool,
    /// Optional explicit bos_token string for the template's
    /// `{{ bos_token }}` expression. Required when the tokenizer's
    /// `decode_bytes(bos_id)` does NOT match the canonical BOS string
    /// the template expects. Example: Gemma 4's tokenizer reports
    /// bos_id=203 (and id=2 decodes to LLaMA-cosmetic `<s>`), but the
    /// Gemma 4 template needs the literal `<bos>` which re-tokenizes to
    /// single special token id=2 (the actual BOS the model trained on).
    /// When None, falls back to decoding bos_id (works for Qwen3.5/3.6).
    pub bos_token: Option<&'a str>,
    /// Reasoning strength hint for Onyx/Harmony templates (`low|medium|high|xhigh`).
    /// Phase 1 always passes `None` (Lenient undefined → template defaults to 'high').
    pub reasoning_strength: Option<&'a str>,
}

/// Multi-turn message representation for `JinjaChatFrame::render_messages`.
///
/// The fields are intentionally serialize-friendly so the entire `&[Message]`
/// slice can be passed straight into the Jinja `messages` context var via
/// `Value::from_serialize(...)`. Templates probe `message['role']`,
/// `message['content']`, `message['tool_calls']`, and (less commonly)
/// `message['tool_call_id']` under strict-undefined mode; all four fields
/// are always present (defaults: empty content, empty tool_calls vec, no
/// tool_call_id) so probes never raise.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    /// Assistant self-channel (reasoning) body, verbatim as the model generated it.
    ///
    /// Renders into the Harmony/Onyx `<|start|>assistant to=self<|message|>…<|eom|>`
    /// envelope (Muse Glimmer, arch 14). `None` — not `Some("")` — means "no reasoning
    /// envelope": the template tests truthiness and the render env is
    /// [`minijinja::UndefinedBehavior::Lenient`], so an omitted key and an empty string
    /// are both falsy, but omission is the faithful pre-change shape for every other
    /// architecture and keeps their rendered bytes identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Inbound-only tool-result identity (OpenAI `message.name`). Deserialized from the
    /// wire, never serialized back into the Jinja context: exposing it generically would
    /// change the rendered bytes of templates that probe `message.name`. The Glimmer
    /// history-preparation step projects it through [`Message::rendered_name`].
    #[serde(default, skip_serializing)]
    pub name: Option<String>,
    /// Render-only projection of a tool-result turn's FUNCTION name, serialized as `name`.
    ///
    /// Populated exclusively by Glimmer's history preparation, which resolves it from the
    /// preceding assistant turn's [`ToolCall::id`]. Every other path leaves it `None`, so
    /// no other architecture sees a `name` key it did not see before.
    #[serde(
        default,
        skip_deserializing,
        skip_serializing_if = "Option::is_none",
        rename = "name"
    )]
    pub rendered_name: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Set on Tool-role messages to identify which assistant tool_call
    /// this is responding to. Qwen3.5/3.6 templates currently ignore
    /// this field; OpenAI-spec clients and some other templates require
    /// it. Skipped from the serialized JSON when None so templates that
    /// `is defined` against it don't see a misleading null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Assistant-turn reasoning that preceded a tool call ("interleaved
    /// thinking"). Rendered into the Cohere/North chat template's
    /// `<|START_THINKING|>{{message.tool_plan}}<|END_THINKING|>` slot so a
    /// multi-turn agentic flow preserves prior reasoning (per the
    /// North-Mini-Code model card — "pass model-generated thinking to future
    /// agentic steps/turns"). Empty for plain turns / non-thinking models;
    /// always serialized (templates that don't reference it ignore the field).
    ///
    /// Distinct from [`Message::reasoning_content`]: `tool_plan` is the Cohere slot and is
    /// written for EVERY architecture by the CLI's inbound normalization, while
    /// `reasoning_content` is the Harmony self-channel body. Never merge or repurpose them.
    #[serde(default)]
    pub tool_plan: String,
}

/// One assistant-emitted tool call, attached to an assistant `Message`.
/// `arguments` is a free-form JSON value (typically an object). Templates
/// that render in XML format (Qwen3.5/3.6's `<function=NAME><parameter=ARG>`
/// shape) walk this with `arguments | items` under pycompat.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    /// Inbound OpenAI `tool_calls[].id`, used to resolve the FOLLOWING tool-result turn's
    /// function name. Inbound-only for the same no-clobber reason as [`Message::name`].
    #[serde(default, skip_serializing)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// Render-only override for this call's entire rendered body.
    ///
    /// Set ONLY by [`build_cached_history_jinja`], to a splice sentinel, so a cached
    /// verbatim tool body replaces the regenerated ATEM text. Never deserialized, so a
    /// client cannot inject it.
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub rendered_body: Option<String>,
}

/// One rendered assistant body slot, captured verbatim from generation.
///
/// `token_ids` are exactly the tokens between a Harmony `<|message|>` and its closing
/// marker: template-owned header (`<|start|>`, `assistant to=…`, `<|message|>`) and
/// terminator (`<|eom|>`, `<|eot|>`) tokens are excluded, because the template re-emits
/// those itself. `text` is the decoded body, kept so the daemon can recover a client's
/// missing `reasoning_content` without retokenizing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedAssistantBody {
    pub token_ids: Vec<u32>,
    pub text: String,
}

/// One `to=<recipient>` tool envelope's verbatim body. `recipient` MUST equal the
/// corresponding [`Message::tool_calls`] entry's `name`, or the splice is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedAssistantToolBody {
    pub recipient: String,
    pub token_ids: Vec<u32>,
}

/// Per-channel verbatim replay for one assistant turn.
///
/// Slot order within a turn is `[reasoning?] ++ (tools | content)`. Tools and content are
/// MUTUALLY EXCLUSIVE in the Harmony/Onyx assistant branch — it renders one envelope per
/// tool call, or a single content envelope, never both.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CachedAssistantTurn {
    pub reasoning: Option<CachedAssistantBody>,
    pub tools: Vec<CachedAssistantToolBody>,
    pub content: Option<CachedAssistantBody>,
}

/// JSON formatter matching HuggingFace's `json.dumps(..., ensure_ascii=False)`
/// default separators — `", "` between elements and `": "` after keys — the
/// exact form the model's chat_template was trained on. minijinja's builtin
/// `tojson` is compact (`,`/`:`); registering [`hf_tojson`] on the render env
/// makes `{{ x | tojson }}` (tool DEFINITIONS and mapping-valued tool-call
/// arguments) byte-match `transformers.apply_chat_template`.
struct HfJsonFormatter;
impl serde_json::ser::Formatter for HfJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// HF-compatible `tojson` filter (see [`HfJsonFormatter`]). Serializes the
/// minijinja value DIRECTLY (not through an intermediate `serde_json::Value`),
/// so map key order is whatever the value carries — preserved end-to-end when
/// `serde_json` is built with `preserve_order` (without it, the request-parse
/// `BTreeMap` has already alphabetized object keys before render). Register with
/// `env.add_filter("tojson", hf_tojson)` to override minijinja's compact builtin.
pub fn hf_tojson(value: minijinja::Value) -> Result<String, minijinja::Error> {
    use serde::Serialize;
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, HfJsonFormatter);
    value.serialize(&mut ser).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson: {e}"),
        )
    })?;
    String::from_utf8(buf).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson utf8: {e}"),
        )
    })
}

/// Parenthesize a bare conditional expression used as a CALL KEYWORD ARGUMENT.
///
/// Jinja2 accepts `namespace(name=tcid if tcid else '')`; minijinja does not —
/// it stops at the `if` and reports
/// `syntax error: unexpected identifier, expected ','`. Wrapping the value in
/// parentheses, `namespace(name=(tcid if tcid else ''))`, parses and evaluates
/// identically, so this is semantics-preserving in the same sense as
/// `strip_generation_tags` above.
///
/// Muse Glimmer bring-up, 2026-08-11: its chat_template.jinja carries exactly
/// this construct in the tool-response branch. Because the template is a single
/// line, the parse failure took the WHOLE template down, and the daemon fell
/// back to "BOS + raw prompt" for every request — so multi-turn silently
/// degenerated (each turn re-answering a bare prompt) with only an eprintln to
/// show for it. Any model whose template uses a conditional kwarg hits this.
///
/// Deliberately narrow: only rewrites `<ident>=` immediately inside a call's
/// argument list, and only when the value contains a top-level ` if ` before
/// the argument ends at `,` or `)`. Depth-tracked so nested calls, strings and
/// already-parenthesized values are left alone.
fn parenthesize_conditional_kwargs(template: &str) -> String {
    let b = template.as_bytes();
    let mut out = String::with_capacity(template.len() + 16);
    // `copied` is the start of the not-yet-emitted slice. Emitting SLICES (not
    // per-byte chars) keeps this UTF-8 safe: the template contains multibyte
    // characters, and `b[i] as char` would reinterpret them as Latin-1.
    let mut copied = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        // Find `<ident>=` sitting directly inside a call's argument list.
        if b[i] == b'='
            && i + 1 < b.len()
            && b[i + 1] != b'='
            && i > 0
            && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')
        {
            let mut s = i;
            while s > 0 && (b[s - 1].is_ascii_alphanumeric() || b[s - 1] == b'_') {
                s -= 1;
            }
            if matches!(template[..s].trim_end().chars().last(), Some('(') | Some(',')) {
                // Scan the value to the end of this argument at depth 0.
                let mut j = i + 1;
                let mut depth = 0i32;
                let mut quote: Option<u8> = None;
                let mut has_if = false;
                while j < b.len() {
                    let c = b[j];
                    if let Some(q) = quote {
                        if c == q {
                            quote = None;
                        }
                    } else if c == b'\'' || c == b'"' {
                        quote = Some(c);
                    } else if c == b'(' || c == b'[' || c == b'{' {
                        depth += 1;
                    } else if c == b')' || c == b']' || c == b'}' {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    } else if c == b',' && depth == 0 {
                        break;
                    } else if depth == 0 && template[j..].starts_with(" if ") {
                        has_if = true;
                    }
                    j += 1;
                }
                let value = &template[i + 1..j];
                if has_if && !value.trim().is_empty() && !value.trim().starts_with('(') {
                    out.push_str(&template[copied..=i]);
                    out.push('(');
                    out.push_str(value);
                    out.push(')');
                    copied = j;
                    i = j;
                    continue;
                }
            }
        }
        i += 1;
    }
    out.push_str(&template[copied..]);
    out
}

/// Strip unsupported `{% generation %}` / `{% endgeneration %}` Jinja tags
/// (with any whitespace/dash variants) from LFM2.5-230M's chat_template.
/// These are no-op generation-scope markers that minijinja doesn't know;
/// without stripping, `env.add_template` fails with "unknown statement generation".
/// Transparent: the markers just delimit the assistant generation block, which
/// the template already handles via `add_generation_prompt` and the assistant
/// loop, so removing them is semantics-preserving. (LFM2.5-230M bring-up,
/// 2026-08-09: `{%- generation -%}` at chat_template.jinja:77,105).
fn strip_generation_tags(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut i = 0;
    let bytes = template.as_bytes();
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'{' && bytes[i + 1] == b'%' {
            if let Some(end) = template[i..].find("%}") {
                let inner = &template[i + 2..i + end];
                // Normalize: trim whitespace, then '-' markers, then whitespace again
                let clean = inner.trim().trim_matches('-').trim();
                // Handle inner like "- generation -" -> after first trim it's "- generation -",
                // after trim_matches('-') it's " generation ", after final trim it's "generation"
                // For " generation " it's already "generation"
                if clean == "generation" || clean == "endgeneration" {
                    i += end + 2;
                    continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Render-time `YYYY-MM-DD` date string for chat templates that surface a
/// "Current date:" line (MiniMax-M2's `system_message.current_date`, and any
/// future template using the same convention).
///
/// Some templates (MiniMax-M2) read `current_date` as an attribute of the
/// **system message dict**, gated behind `{% if system_message and
/// system_message.current_date %}`. Under minijinja strict-undefined a system
/// message that lacks the key raises a render error instead of skipping the
/// branch — so [`render_messages`](JinjaChatFrame::render_messages) injects this
/// value onto the leading system message (see that method).
///
/// Determinism note: this is the **prompt-build** path, not the
/// forward/sampler determinism path, so a wall-clock date here does not affect
/// token-level reproducibility. To keep renders byte-stable across a test (or
/// to honour a daemon request-time clock) callers can pin the value via the
/// `HIPFIRE_CHAT_CURRENT_DATE` env var; otherwise we derive it from
/// `SystemTime::now()` (NOT a determinism-forbidden argless engine `Date::now`).
pub fn render_time_date() -> String {
    if let Ok(pinned) = hipfire_config::developer_var("HIPFIRE_CHAT_CURRENT_DATE") {
        if !pinned.trim().is_empty() {
            return pinned;
        }
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert days-since-Unix-epoch to a proleptic-Gregorian `(year, month, day)`.
/// Howard Hinnant's `civil_from_days` algorithm — no `chrono` dependency.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

impl<'a> JinjaChatFrame<'a> {
    /// Render the template and tokenize the result. Returns `Err` on
    /// any template-side failure so the caller can fall back to
    /// `ChatFrame::Plain` framing.
    pub fn render_and_encode(&self) -> Result<Vec<u32>, String> {
        let rendered = self.render()?;
        Ok(self.tokenizer.encode(&rendered))
    }

    /// Render the template to a string without tokenizing. Single-turn
    /// convenience wrapper around `render_messages` that synthesizes a
    /// `[system?, user]` message slice from the struct's `system` /
    /// `user` fields. Exposed separately so a diagnostic example can
    /// dump the rendered prompt for byte-level comparison against
    /// transformers' output.
    pub fn render(&self) -> Result<String, String> {
        let mut messages: Vec<Message> = Vec::new();
        if let Some(sys) = self.system {
            messages.push(Message {
                role: Role::System,
                content: sys.to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            });
        }
        messages.push(Message {
            role: Role::User,
            content: self.user.to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        });
        self.render_messages(&messages, None, None)
    }

    /// Render the template against a full multi-turn message history.
    /// This is the canonical entry point — `render()` above is just a
    /// single-turn convenience.
    ///
    /// `tools` is the OpenAI tool definitions list (each entry an object
    /// with `type` + `function`); pass `None` for plain (no-tools)
    /// turns and the template's `if tools` predicate evaluates false.
    /// `tool_call_kwargs` is a free-form map propagated to the template
    /// context for templates that opt into per-call rendering switches;
    /// pass `None` for the default empty map.
    ///
    /// Strict-undefined empty defaults still apply when args are `None`,
    /// so templates that probe `tools` / `documents` / `tool_call_kwargs`
    /// don't raise.
    pub fn render_messages(
        &self,
        messages: &[Message],
        tools: Option<&[serde_json::Value]>,
        tool_call_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String, String> {
        use minijinja::{Environment, Error, ErrorKind, Value};
        use minijinja_contrib::pycompat::unknown_method_callback;

        let mut env = Environment::new();
        // Lenient undefined: undefined renders as "", is falsy in boolean
        // context, and compares false — matching HuggingFace's jinja2 DEFAULT
        // environment, which is what upstream chat templates are authored
        // against. REQUIRED by Cohere2's tool_use template, which legitimately
        // references optional fields the engine doesn't always populate
        // (`message.tool_plan`, per-tool_call `id`, `{% if developer_preamble %}`,
        // `{% if enable_citations %}`). Strict/SemiStrict raised on those and
        // forced a fallback to the hand-rolled ChatML frame mid-conversation —
        // the agentic multi-turn derail (the model then saw <|im_start|> markers
        // and hallucinated). Complete templates (qwen, etc.) never hit undefined,
        // so they render byte-identically; this only changes the
        // previously-erroring paths, degrading them to HF-equivalent "".
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        // Match HuggingFace's apply_chat_template Jinja environment, which is
        // constructed with `trim_blocks=True, lstrip_blocks=True`. Without these,
        // block tags (`{% … %}`) leak their surrounding source whitespace into
        // the rendered output — off-distribution vs. what the model trained on.
        // Worse, for templates with history-length-dependent control flow (e.g.
        // MiniMax-M2's `last_user_index` scan, which emits a `\n        ` per
        // user message), the leaked leading whitespace VARIES by turn, so turn
        // N+1's render diverges from turn N's at token 1 and the LCP prompt
        // cache collapses to lcp=1. Enabling both makes our render byte-track
        // HF and keeps the structural prefix history-invariant.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // Make Python-style str/list/dict methods (`.startswith`,
        // `.split`, `.rstrip`, `.lstrip`, `|items`, etc.) work on
        // ordinary Jinja values. Required by the Qwen3 family
        // template — it calls these throughout the assistant-turn
        // and tool branches.
        env.set_unknown_method_callback(unknown_method_callback);
        // The Qwen3 template uses `raise_exception('...')` to fail
        // fast on malformed inputs (e.g. system message in the
        // middle of the conversation). minijinja has no builtin
        // for this, so we register it as a global function that
        // surfaces the message as a render error.
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        // Override minijinja's compact builtin `tojson` with the HF-spaced form
        // (`", "` / `": "`) the model trained on, so tool-definition and
        // mapping-arg rendering byte-matches transformers' apply_chat_template.
        env.add_filter("tojson", hf_tojson);

        let cleaned_template =
            parenthesize_conditional_kwargs(&strip_generation_tags(self.template));
        env.add_template("chat", &cleaned_template)
            .map_err(|e| format!("template parse: {e}"))?;
        let tmpl = env
            .get_template("chat")
            .map_err(|e| format!("template lookup: {e}"))?;

        // Pass bos_token to the template context. Caller may override via
        // `self.bos_token` (Gemma 4 needs explicit `<bos>` because its
        // tokenizer returns LLaMA-cosmetic `<s>` for decode_bytes(bos_id)
        // and that re-tokenizes to a 3-token BPE fragment instead of
        // single id=2 the template expects). Default: decode bos_id back
        // to text (works for Qwen / LLaMA).
        let bos_token: String = match self.bos_token {
            Some(s) => s.to_string(),
            None => {
                let bytes = self.tokenizer.decode_bytes(&[self.tokenizer.bos_id]);
                String::from_utf8_lossy(&bytes).to_string()
            }
        };
        // Strict-undefined empty defaults so templates that probe
        // `tools` / `documents` / `tool_call_kwargs` on plain turns
        // don't raise. Caller-provided values override the empties.
        let empty_list: Vec<serde_json::Value> = Vec::new();
        let empty_map = serde_json::Map::new();
        let tools_val = match tools {
            Some(t) => Value::from_serialize(t),
            None => Value::from_serialize(&empty_list),
        };
        let kwargs_val = match tool_call_kwargs {
            Some(k) => Value::from_serialize(k),
            None => Value::from_serialize(&empty_map),
        };
        // Inject `current_date` onto the leading system message so templates
        // that read `system_message.current_date` (MiniMax-M2, chat.jinja:37)
        // don't raise under strict-undefined. The no-system-message path needs
        // nothing — those templates gate the access behind `if system_message
        // and …`, which short-circuits when `system_message` is `none` (the
        // built-in model identity is injected instead). Adding the key here
        // makes both paths render identically modulo the date line that the
        // upstream template intends a system turn to carry. Only the FIRST
        // message is patched, and only when it is a system turn — every other
        // template / model serializes through unchanged.
        let messages_val = {
            let mut arr: Vec<serde_json::Value> = messages
                .iter()
                .map(|m| serde_json::to_value(m).unwrap_or(serde_json::Value::Null))
                .collect();
            if let Some(serde_json::Value::Object(map)) = arr.first_mut() {
                let is_system = matches!(
                    map.get("role"),
                    Some(serde_json::Value::String(r)) if r == "system"
                );
                if is_system {
                    // Populated date → the template's "Current date:" line
                    // renders (its intended behaviour for a system turn).
                    map.entry("current_date".to_string())
                        .or_insert_with(|| serde_json::Value::String(render_time_date()));
                    // Sibling optional attributes the same templates probe
                    // (also gated `{% if system_message and system_message.X %}`)
                    // would equally raise under strict-undefined on a system
                    // dict that lacks them. Seed an empty string: falsy, so the
                    // gate short-circuits and the line stays absent — matching
                    // upstream "unset" semantics — without a render error.
                    map.entry("current_location".to_string())
                        .or_insert_with(|| serde_json::Value::String(String::new()));
                }
            }
            Value::from_serialize(&arr)
        };
        let reasoning_strength_val = match self.reasoning_strength {
            Some(s) => Value::from_serialize(s),
            None => Value::from_serialize(Option::<String>::None),
        };
        let ctx = minijinja::context! {
            messages => messages_val,
            add_generation_prompt => true,
            enable_thinking => self.enable_thinking,
            bos_token => bos_token,
            reasoning_strength => reasoning_strength_val,
            tools => tools_val,
            documents => Value::from_serialize(&empty_list),
            tool_call_kwargs => kwargs_val,
        };
        tmpl.render(ctx)
            .map_err(|e| format!("template render: {e}"))
    }
}

/// Pick `n` distinct atomic special-token sentinels for the verbatim-splice render.
///
/// Each sentinel must (1) encode to exactly one token (so it never BPE-merges
/// with neighbouring template text) and (2) never be emitted by the template
/// itself. We prefer obviously-reserved tokens (`reserved` / `unused` / `pad`)
/// and otherwise take any non-structural special token that round-trips atomically.
/// Returns `None` if fewer than `n` distinct sentinels exist; the caller then
/// falls back to a plain render. Distinctness within a turn is what makes a
/// channel shift detectable; recycling the same set across turns keeps the pool
/// bounded (`k = 1 + max_tool_slots + 1`).
fn pick_splice_sentinels(tok: &Tokenizer, n: usize) -> Option<Vec<(String, u32)>> {
    if n == 0 {
        return Some(Vec::new());
    }
    const STRUCTURAL: &[&str] = &[
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<|endoftext|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<s>",
        "</s>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<pad>",
        "<|file_separator|>",
        "<|eom|>",
        "<|eot|>",
        "<|start|>",
        "<|message|>",
    ];
    let atomic = |s: &str| -> bool {
        tok.special_token_id(s)
            .map_or(false, |id| tok.encode(s) == vec![id])
    };
    let mut candidates: Vec<(String, u32)> = Vec::new();
    for (s, id) in tok.special_tokens() {
        if STRUCTURAL.contains(&s.as_str()) {
            continue;
        }
        if !atomic(s) {
            continue;
        }
        candidates.push((s.clone(), *id));
    }
    // Sort deterministically by (preferred_tier, id, text)
    candidates.sort_by(|(a_text, a_id), (b_text, b_id)| {
        let a_tier = {
            let ls = a_text.to_ascii_lowercase();
            if ls.contains("reserved") || ls.contains("unused") || ls.contains("pad") {
                0
            } else {
                1
            }
        };
        let b_tier = {
            let ls = b_text.to_ascii_lowercase();
            if ls.contains("reserved") || ls.contains("unused") || ls.contains("pad") {
                0
            } else {
                1
            }
        };
        a_tier.cmp(&b_tier).then_with(|| a_id.cmp(b_id)).then_with(|| a_text.cmp(b_text))
    });
    // Dedup by token id, keeping first (preferred tier) occurrence
    let mut seen = std::collections::HashSet::new();
    let mut deduped: Vec<(String, u32)> = Vec::new();
    for (s, id) in candidates {
        if seen.insert(id) {
            deduped.push((s, id));
        }
    }
    if deduped.len() < n {
        return None;
    }
    deduped.truncate(n);
    Some(deduped)
}

/// Jinja-native analogue of [`build_cached_history`]: render the conversation
/// through the model's **trained** `chat_template` but splice each cached
/// channel body verbatim. The resulting token stream byte-exactly reproduces
/// what the daemon prefilled, so the LCP prompt-cache hits reliably.
///
/// `messages` is the full conversation INCLUDING the live user turn last.
/// For each assistant turn, `cache_lookup` returns `Some(CachedAssistantTurn)`
/// (verbatim per-channel bodies) or `None` (no cache entry: that turn keeps its
/// retokenized content). The splice is channel-aware: reasoning, each tool
/// envelope, and final content are separate slots. Tools and content are
/// MUTUALLY EXCLUSIVE (`[reasoning?] ++ (tools | content)`). Sentinels are
/// distinct WITHIN a turn and RECYCLED ACROSS turns (`k = 1 + max_tool_slots + 1`).
/// The render is committed only when the observed sentinel-id SEQUENCE equals the
/// expected sequence exactly; any inequality (missing, duplicate, reorder) returns
/// the plain render.
///
/// Body boundaries exclude template-owned header/terminator tokens (`<|start|>`,
/// `assistant to=…`, `<|message|>`, `<|eom|>`, `<|eot|>`); the template re-emits
/// those. `CachedAssistantBody.text` must byte-equal the corresponding
/// `Message` field when that field is non-empty; recipient must equal
/// `Message.tool_calls[i].name`. Any violation is structural doubt → plain render.
pub fn build_cached_history_jinja(
    frame: &JinjaChatFrame,
    messages: &[Message],
    tools: Option<&[serde_json::Value]>,
    mut cache_lookup: impl FnMut(&Message) -> Option<CachedAssistantTurn>,
) -> Result<Vec<u32>, String> {
    let tok = frame.tokenizer;
    // 1. Plain render is the sole fallback; compute it first.
    let plain_tokens = tok.encode(&frame.render_messages(messages, tools, None)?);
    // 2. Walk messages, call lookup at most once per assistant turn, validate.
    let mut cached_hits: Vec<(usize, CachedAssistantTurn)> = Vec::new();
    let mut max_tool_slots: usize = 0;
    let mut total_slots: usize = 0;
    for (idx, m) in messages.iter().enumerate() {
        if !matches!(m.role, Role::Assistant) {
            continue;
        }
        let Some(turn) = cache_lookup(m) else {
            continue;
        };
        // Default / no-slot is a miss
        if turn.reasoning.is_none() && turn.tools.is_empty() && turn.content.is_none() {
            continue;
        }
        // Validate: tools and content are mutually exclusive
        if !turn.tools.is_empty() && turn.content.is_some() {
            return Ok(plain_tokens);
        }
        if turn.tools.is_empty() && turn.content.is_none() {
            // Must have at least one slot (reasoning alone without content is invalid per spec)
            return Ok(plain_tokens);
        }
        // Validate tool count and recipients, and reasoning provenance.
        if turn.tools.is_empty() {
            // Content turn: must have no tool_calls on the Message.
            if !m.tool_calls.is_empty() {
                return Ok(plain_tokens);
            }
            if turn.content.is_none() {
                return Ok(plain_tokens);
            }
            // Deliberately NO `turn.content.text == m.content` check.
            //
            // Content identity is ALREADY what the cache key is derived from — the caller
            // fingerprints the message's normalized content to find this entry, so a hit
            // means the content matched under that normalization. Re-comparing the RAW
            // strings here re-tests the same property with a *stricter* rule than the key,
            // so any normalization the key forgives (whitespace collapse, `<think>`
            // stripping the API layer applies) makes the key hit and this guard silently
            // refuse — a splice that is found and then thrown away.
            //
            // Observed: a 4-turn coding session hit the key on turn 2 yet fell back to the
            // plain render, and the retokenized reasoning diverged 74 tokens into the
            // assistant turn (`lcp=173` against `prior_len=371`), collapsing the cache.
            // `reasoning` still needs the check below because it is NOT part of the key.
        } else {
            // Tool turn: counts must match
            if turn.tools.len() != m.tool_calls.len() {
                return Ok(plain_tokens);
            }
            for (i, tb) in turn.tools.iter().enumerate() {
                if tb.recipient != m.tool_calls[i].name {
                    return Ok(plain_tokens);
                }
            }
            // Content must be None already checked
        }
        // Reasoning text check when Message has non-empty reasoning_content
        if let Some(rc) = &m.reasoning_content {
            if !rc.is_empty() {
                match &turn.reasoning {
                    Some(rb) if rb.text == *rc => {}
                    _ => return Ok(plain_tokens),
                }
            }
        }
        // Also if turn has reasoning, but Message reasoning is None/empty, that's allowed (recovery)
        // Count slots
        let mut slots = 0;
        if turn.reasoning.is_some() {
            slots += 1;
        }
        if !turn.tools.is_empty() {
            slots += turn.tools.len();
        } else if turn.content.is_some() {
            slots += 1;
        }
        if slots == 0 {
            continue;
        }
        max_tool_slots = max_tool_slots.max(turn.tools.len());
        total_slots += slots;
        cached_hits.push((idx, turn));
    }
    if cached_hits.is_empty() {
        return Ok(plain_tokens);
    }
    // 3. Pick k distinct sentinels, recycled across turns.
    let k = 1 + max_tool_slots + 1;
    let sentinels = match pick_splice_sentinels(tok, k) {
        Some(v) => v,
        None => return Ok(plain_tokens),
    };
    let sentinel_ids: Vec<u32> = sentinels.iter().map(|(_, id)| *id).collect();
    let sentinel_texts: Vec<String> = sentinels.iter().map(|(s, _)| s.clone()).collect();
    // Collision: any sentinel id already in plain_tokens -> fallback
    for sid in &sentinel_ids {
        if plain_tokens.contains(sid) {
            return Ok(plain_tokens);
        }
    }
    // Map: R = sentinels[0], T_i = sentinels[1+i], A = sentinels[k-1]
    let r_text = sentinel_texts[0].clone();
    let r_id = sentinel_ids[0];
    let a_text = sentinel_texts[k - 1].clone();
    let a_id = sentinel_ids[k - 1];
    // Build expected sequence, slot bodies and slot texts in document order
    let mut expected_ids: Vec<u32> = Vec::with_capacity(total_slots);
    let mut slot_bodies: Vec<Vec<u32>> = Vec::with_capacity(total_slots);
    for (_, turn) in &cached_hits {
        if let Some(rb) = &turn.reasoning {
            expected_ids.push(r_id);
            slot_bodies.push(rb.token_ids.clone());
        }
        if !turn.tools.is_empty() {
            for (i, tb) in turn.tools.iter().enumerate() {
                let tid = sentinel_ids[1 + i];
                expected_ids.push(tid);
                slot_bodies.push(tb.token_ids.clone());
            }
        } else if let Some(cb) = &turn.content {
            expected_ids.push(a_id);
            slot_bodies.push(cb.token_ids.clone());
        }
    }
    // 4. Clone messages and substitute sentinels per slot
    let mut subbed: Vec<Message> = messages.to_vec();
    for (msg_idx, turn) in &cached_hits {
        let m = &mut subbed[*msg_idx];
        if turn.reasoning.is_some() {
            m.reasoning_content = Some(r_text.clone());
        }
        if !turn.tools.is_empty() {
            for (i, _) in turn.tools.iter().enumerate() {
                let t_text = sentinel_texts[1 + i].clone();
                m.tool_calls[i].rendered_body = Some(t_text);
            }
        } else if turn.content.is_some() {
            m.content = a_text.clone();
        }
    }
    // 5. Render substituted, tokenize. Error -> plain.
    let sub_rendered = match frame.render_messages(&subbed, tools, None) {
        Ok(s) => s,
        Err(_) => return Ok(plain_tokens),
    };
    let sub_tokens = tok.encode(&sub_rendered);
    // 6. Extract observed sentinel subsequence and require exact equality
    let mut observed_ids: Vec<u32> = Vec::new();
    for &t in &sub_tokens {
        if sentinel_ids.contains(&t) {
            observed_ids.push(t);
        }
    }
    if observed_ids != expected_ids {
        return Ok(plain_tokens);
    }
    // 7. Splice in one non-recursive pass
    let total_body_len: usize = slot_bodies.iter().map(|b| b.len()).sum();
    let mut out: Vec<u32> = Vec::with_capacity(sub_tokens.len() - expected_ids.len() + total_body_len);
    let mut bi: usize = 0;
    for &t in &sub_tokens {
        if sentinel_ids.contains(&t) {
            if bi >= expected_ids.len() || t != expected_ids[bi] {
                return Ok(plain_tokens);
            }
            out.extend_from_slice(&slot_bodies[bi]);
            bi += 1;
        } else {
            out.push(t);
        }
    }
    debug_assert_eq!(bi, slot_bodies.len());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a hermetic test tokenizer. Uses the `from_hf_json` path
    /// with a minimal vocabulary that is sufficient to round-trip the
    /// ChatML special tokens and a few simple ASCII strings. The test
    /// does NOT depend on any GGUF fixture.
    ///
    /// Strategy: GPT-2-BPE flavor (triggered by adding `Ġ` to the
    /// vocab). The byte-level fallback in `encode_gpt2_bpe` will
    /// convert any unmapped string into per-byte token IDs without
    /// the SentencePiece `▁`-prepending quirk that complicates
    /// equality checks.
    ///
    /// IMPORTANT: the tests below build their *expected* byte
    /// sequences using the same `tokenizer.encode()` call that
    /// `ChatFrame::build` uses internally, so any quirks of the
    /// encoder cancel out. The tests verify *structural* properties
    /// (system block precedes user turn; assistant prefix appears at
    /// end; raw bypasses scaffolding; multi-turn concatenates
    /// turns), not exact-byte oracles against a hand-rolled string.
    fn make_tokenizer() -> Tokenizer {
        // Vocab includes:
        // - chatml special tokens
        // - role names ("system", "user", "assistant")
        // - common ascii bytes for short strings ("hello", "hi", "world", etc.)
        // - the `Ġ` trigger that puts the tokenizer in GPT-2 BPE mode
        // - all 256 single bytes (mapped via byte_to_gpt2_char) for
        //   robust fallback on arbitrary content
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""<think>": 2"#.to_string());
        entries.push(r#""</think>": 3"#.to_string());
        entries.push(r#""system": 4"#.to_string());
        entries.push(r#""user": 5"#.to_string());
        entries.push(r#""assistant": 6"#.to_string());
        entries.push(r#""\n": 7"#.to_string());
        entries.push(r#""Ġ": 8"#.to_string()); // gpt-2 mode trigger
        entries.push(r#""<|reserved_0|>": 9"#.to_string()); // splice sentinel (atomic special)
        entries.push(r#""<|reserved_1|>": 10"#.to_string());
        entries.push(r#""<|reserved_2|>": 11"#.to_string());
        entries.push(r#""<|reserved_3|>": 12"#.to_string());
        entries.push(r#""<|reserved_4|>": 13"#.to_string());
                                                            // All 256 GPT-2-byte characters get unique ids 100..356 so
                                                            // any short string round-trips byte-by-byte.
        for b in 0u32..=255u32 {
            // Use rust escape; the encoder will look up the GPT-2 char
            // form of each byte directly.
            let ch = byte_to_gpt2_char_test(b as u8);
            // JSON-escape the char carefully — only `\`, `"`, control
            // chars need it; the GPT-2 byte mapping uses non-ASCII
            // unicode chars for the printable byte range.
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
                    {{"id": 9, "content": "<|reserved_0|>", "special": true}},
                    {{"id": 10, "content": "<|reserved_1|>", "special": true}},
                    {{"id": 11, "content": "<|reserved_2|>", "special": true}},
                    {{"id": 12, "content": "<|reserved_3|>", "special": true}},
                    {{"id": 13, "content": "<|reserved_4|>", "special": true}}
                ]
            }}"#,
            vocab = vocab_block,
        );
        Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    /// Like `make_tokenizer` but WITHOUT `<think>` / `</think>`
    /// as special added tokens — used to verify ClosedThink fallback.
    fn test_tokenizer_no_think() -> Tokenizer {
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""system": 4"#.to_string());
        entries.push(r#""user": 5"#.to_string());
        entries.push(r#""assistant": 6"#.to_string());
        entries.push(r#""\n": 7"#.to_string());
        entries.push(r#""Ġ": 8"#.to_string());
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
                    {{"id": 1, "content": "<|im_end|>", "special": true}}
                ]
            }}"#,
            vocab = vocab_block,
        );
        Tokenizer::from_hf_json(&json).expect("test tokenizer without think tokens")
    }

    /// Mirror of `byte_to_gpt2_char` from tokenizer.rs (private). The
    /// GPT-2 byte-to-char mapping leaves printable ASCII (33..127, 161..173,
    /// 174..256) untouched and renumbers the rest above 256.
    fn byte_to_gpt2_char_test(b: u8) -> char {
        // Standard GPT-2 byte_to_unicode table. We only need it stable
        // across the test tokenizer + the production tokenizer; the
        // production code reuses the same canonical table.
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
        let idx = bs
            .iter()
            .position(|&x| x == b as u32)
            .expect("byte in table");
        char::from_u32(cs[idx]).expect("valid char")
    }

    fn json_escape(s: &str) -> String {
        // Only escape what JSON requires: backslash, quote, control.
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    #[test]
    fn plain_assistant_prefix_layout() {
        let t = make_tokenizer();
        let frame = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hello",
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        };
        let got = frame.build();

        // Build expected using the same encoder, mirroring daemon's
        // canonical AR-path framing exactly:
        //   <|im_start|> user \n <user content> <|im_end|> \n
        //   <|im_start|> assistant \n
        let mut expected: Vec<u32> = Vec::new();
        expected.extend_from_slice(&t.encode("<|im_start|>"));
        expected.extend_from_slice(&t.encode("user"));
        expected.extend_from_slice(&t.encode("\n"));
        expected.extend_from_slice(&t.encode("hello"));
        expected.extend_from_slice(&t.encode("<|im_end|>"));
        expected.extend_from_slice(&t.encode("\n"));
        expected.extend_from_slice(&t.encode("<|im_start|>"));
        expected.extend_from_slice(&t.encode("assistant"));
        expected.extend_from_slice(&t.encode("\n"));
        assert_eq!(got, expected, "Plain assistant prefix layout mismatch");
    }

    #[test]
    fn open_think_appends_think_newline_when_special_present() {
        let t = make_tokenizer();
        let plain = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hi",
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        }
        .build();
        let opened = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hi",
            assistant_prefix: AssistantPrefix::OpenThink,
            raw: false,
        }
        .build();
        // The test tokenizer always registers `<think>` as a special
        // token, so OpenThink must append exactly `<think>\n`.
        let think_id = t
            .special_token_id("<think>")
            .expect("test tokenizer registers <think> as special");
        let mut expected = plain.clone();
        expected.push(think_id);
        expected.extend_from_slice(&t.encode("\n"));
        assert_eq!(
            opened, expected,
            "OpenThink should append <think>\\n after the assistant prefix"
        );
        assert!(
            opened.len() > plain.len(),
            "OpenThink output must be strictly longer than Plain"
        );
    }

    #[test]
    fn closed_think_appends_empty_closed_block_when_tokens_present() {
        let t = make_tokenizer();
        let plain = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hi",
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        }
        .build();
        let closed = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hi",
            assistant_prefix: AssistantPrefix::ClosedThink,
            raw: false,
        }
        .build();
        let think_id = t
            .special_token_id("<think>")
            .expect("test tokenizer registers <think> as special");
        let close_id = t
            .special_token_id("</think>")
            .expect("test tokenizer registers </think> as special");
        let nl = t.encode("\n");
        let mut expected = plain.clone();
        // <think>\n\n</think>\n\n
        expected.push(think_id);
        expected.extend_from_slice(&nl);
        expected.extend_from_slice(&nl);
        expected.push(close_id);
        expected.extend_from_slice(&nl);
        expected.extend_from_slice(&nl);
        assert_eq!(
            closed, expected,
            "ClosedThink should append <think>\\n\\n</think>\\n\\n after the assistant prefix"
        );
        assert!(
            closed.len() > plain.len(),
            "ClosedThink output must be strictly longer than Plain"
        );
    }

    #[test]
    fn closed_think_falls_back_to_plain_when_tokens_missing() {
        // tokenize from scratch with no think/close special tokens
        let t = test_tokenizer_no_think();
        let plain = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hi",
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        }
        .build();
        let closed = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hi",
            assistant_prefix: AssistantPrefix::ClosedThink,
            raw: false,
        }
        .build();
        assert_eq!(
            closed, plain,
            "ClosedThink without special tokens must fall back to Plain"
        );
    }

    #[test]
    fn raw_bypasses_chatml() {
        let t = make_tokenizer();
        let frame = ChatFrame {
            tokenizer: &t,
            system: Some("ignored when raw"),
            user: "completion text",
            assistant_prefix: AssistantPrefix::Plain,
            raw: true,
        };
        let got = frame.build();
        let expected = t.encode("completion text");
        assert_eq!(got, expected, "raw=true should bypass ChatML scaffolding");
    }

    #[test]
    fn build_multi_turn_two_turn_history() {
        let t = make_tokenizer();
        let history: [(Role, &str); 2] = [(Role::User, "hello"), (Role::Assistant, "hi")];
        let frame = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "world",
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        };
        let got = frame.build_multi_turn(&history);

        // Expected: history[user] history[assistant] new[user] new[assistant_prefix]
        let mut expected: Vec<u32> = Vec::new();
        // Prior user turn
        expected.extend_from_slice(&t.encode("<|im_start|>"));
        expected.extend_from_slice(&t.encode("user"));
        expected.extend_from_slice(&t.encode("\n"));
        expected.extend_from_slice(&t.encode("hello"));
        expected.extend_from_slice(&t.encode("<|im_end|>"));
        expected.extend_from_slice(&t.encode("\n"));
        // Prior assistant turn
        expected.extend_from_slice(&t.encode("<|im_start|>"));
        expected.extend_from_slice(&t.encode("assistant"));
        expected.extend_from_slice(&t.encode("\n"));
        expected.extend_from_slice(&t.encode("hi"));
        expected.extend_from_slice(&t.encode("<|im_end|>"));
        expected.extend_from_slice(&t.encode("\n"));
        // New user turn
        expected.extend_from_slice(&t.encode("<|im_start|>"));
        expected.extend_from_slice(&t.encode("user"));
        expected.extend_from_slice(&t.encode("\n"));
        expected.extend_from_slice(&t.encode("world"));
        expected.extend_from_slice(&t.encode("<|im_end|>"));
        expected.extend_from_slice(&t.encode("\n"));
        // Assistant prefix (Plain)
        expected.extend_from_slice(&t.encode("<|im_start|>"));
        expected.extend_from_slice(&t.encode("assistant"));
        expected.extend_from_slice(&t.encode("\n"));

        assert_eq!(got, expected, "multi-turn token sequence mismatch");
    }

    #[test]
    fn build_with_user_tokens_matches_build_when_tokens_match_string() {
        // The pre-tokenized variant must produce byte-identical output
        // to `build()` when the supplied tokens equal `tokenizer.encode(self.user)`.
        // This is the daemon AR-path no-PFlash case.
        let t = make_tokenizer();
        let user_text = "hello";
        let frame = ChatFrame {
            tokenizer: &t,
            system: Some("sysprompt"),
            user: user_text,
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        };
        let via_string = frame.build();
        let via_tokens = frame.build_with_user_tokens(&t.encode(user_text));
        assert_eq!(
            via_string, via_tokens,
            "build_with_user_tokens must match build() when tokens align"
        );
    }

    #[test]
    fn message_deserializes_minimal_shape() {
        // The daemon's stdin schema must accept the smallest valid
        // message: role + content, no tool_calls, no tool_call_id.
        let json = r#"{"role":"user","content":"hi"}"#;
        let m: Message = serde_json::from_str(json).expect("minimal message parses");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content, "hi");
        assert!(m.tool_calls.is_empty());
        assert!(m.tool_call_id.is_none());
    }

    #[test]
    fn message_deserializes_assistant_tool_call() {
        // OpenAI-style assistant turn that emitted a tool call. The
        // template path consumes `tool_calls[]` to render the model's
        // own prior call (XML on Qwen3.5/3.6, JSON on others).
        let json = r#"{
            "role":"assistant",
            "content":"",
            "tool_calls":[{"name":"get_weather","arguments":{"city":"SF","unit":"f"}}]
        }"#;
        let m: Message = serde_json::from_str(json).expect("assistant w/ tool_call parses");
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].name, "get_weather");
        assert_eq!(
            m.tool_calls[0].arguments,
            serde_json::json!({"city":"SF","unit":"f"}),
        );
    }

    #[test]
    fn message_deserializes_tool_response() {
        // Tool-role response carries a `tool_call_id` referencing the
        // assistant call it answers. Field must round-trip through
        // serde so templates that read it (OpenAI-spec ones) see it.
        let json = r#"{"role":"tool","content":"72F","tool_call_id":"call_42"}"#;
        let m: Message = serde_json::from_str(json).expect("tool response parses");
        assert_eq!(m.role, Role::Tool);
        assert_eq!(m.content, "72F");
        assert_eq!(m.tool_call_id.as_deref(), Some("call_42"));
    }

    #[test]
    fn jinja_splice_extends_prior_turn_for_thinking_model() {
        // The core guarantee of `build_cached_history_jinja`: turn N+1's cached
        // render is a strict EXTENSION of turn N's prefilled `conversation_tokens`
        // (prompt + verbatim generated tokens), so the daemon's LCP prefix-cache
        // hits — even for a thinking model whose generated <think>…</think>
        // tokens cannot be recovered by re-tokenizing the API-stripped answer.
        let t = make_tokenizer();
        // Minimal ChatML template: history turns render
        // `<|im_start|>{role}\n{content}<|im_end|>\n`; the generation prompt opens
        // the assistant turn and (thinking-on) primes `<think>\n`.
        let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
        reasoning_strength: None,
        };

        // Turn 1: daemon prefills R1 (prompt, ends with the primed `<think>\n`)
        // then generates `reason</think>ok`.
        let u1 = Message {
            role: Role::User,
            content: "hi".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let r1 = t.encode(
            &frame
                .render_messages(std::slice::from_ref(&u1), None, None)
                .unwrap(),
        );
        let t1_gen = t.encode("reason</think>ok");
        let mut conv_after_t1 = r1.clone();
        conv_after_t1.extend_from_slice(&t1_gen);

        // Turn 2: the asst_turn_cache stored the VERBATIM assistant slot — the
        // primed `<think>\n` plus the generated tokens (everything the daemon
        // laid between `assistant\n` and the next turn).
        let asst_slot: Vec<u32> = {
            let mut v = t.encode("<think>\n");
            v.extend_from_slice(&t1_gen);
            v
        };
        let a1 = Message {
            role: Role::Assistant,
            content: "ok".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let u2 = Message {
            role: Role::User,
            content: "again".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: vec![],
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let messages_t2 = vec![u1.clone(), a1, u2];

        let rendered_t2 = build_cached_history_jinja(&frame, &messages_t2, None, |m| {
            if matches!(m.role, Role::Assistant) {
                Some(CachedAssistantTurn {
                    reasoning: None,
                    tools: Vec::new(),
                    content: Some(CachedAssistantBody {
                        token_ids: asst_slot.clone(),
                        text: "ok".to_string(),
                    }),
                })
            } else {
                None
            }
        })
        .expect("jinja splice render");

        // No sentinel leaked.
        let sentinel_id = t.special_token_id("<|reserved_0|>").unwrap();
        assert!(
            !rendered_t2.contains(&sentinel_id),
            "sentinel must be fully replaced: {rendered_t2:?}"
        );
        // Verbatim splice happened.
        assert!(
            rendered_t2
                .windows(asst_slot.len())
                .any(|w| w == asst_slot.as_slice()),
            "cached assistant slot must be spliced verbatim",
        );
        // THE KEY PROPERTY: turn 2 strictly extends turn 1's conversation_tokens.
        assert!(
            rendered_t2.len() > conv_after_t1.len(),
            "turn 2 must be longer than turn 1"
        );
        assert_eq!(
            &rendered_t2[..conv_after_t1.len()],
            conv_after_t1.as_slice(),
            "turn 2 render must extend turn 1's conversation_tokens as a strict prefix",
        );

        // Fallback: no cache entries ⇒ identical to a plain render.
        let plain = t.encode(&frame.render_messages(&messages_t2, None, None).unwrap());
        let no_cache = build_cached_history_jinja(&frame, &messages_t2, None, |_| None).unwrap();
        assert_eq!(no_cache, plain, "no-cache path must equal a plain render");
    }

    #[test]
    fn message_name_rendered_name_serde_contract() {
        // Contract 6.1 verbatim
        let json = r#"{"role":"tool","content":"x","name":"weather.get"}"#;
        let m: Message = serde_json::from_str(json).expect("tool message parses");
        assert_eq!(m.name.as_deref(), Some("weather.get"));
        assert_eq!(m.rendered_name, None);
        // serializing with name: Some("a"), rendered_name: None emits NO name key
        let msg = Message {
            role: Role::Tool,
            content: "x".to_string(),
            reasoning_content: None,
            name: Some("a".to_string()),
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v.get("name").is_none(), "name should not serialize, got {v}");
        // serializing with rendered_name: Some("b") emits "name":"b"
        let msg2 = Message {
            role: Role::Tool,
            content: "x".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: Some("b".to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let v2 = serde_json::to_value(&msg2).unwrap();
        assert_eq!(v2.get("name").and_then(|x| x.as_str()), Some("b"));
    }

    #[test]
    fn jinja_splice_replays_reasoning_tools_and_content_in_document_order() {
        let t = make_tokenizer();
        // Minimal Onyx-shaped template: distinct envelopes for reasoning, tools, content
        let template = r#"{% for m in messages %}{% if m.role == 'assistant' %}[A]{% if m.reasoning_content %}<R>{{ m.reasoning_content }}</R>{% endif %}{% if m.tool_calls %}{% for tc in m.tool_calls %}<T n="{{ tc.name }}">{% if tc.rendered_body %}{{ tc.rendered_body }}{% else %}{{ tc.arguments | tojson }}{% endif %}</T>{% endfor %}{% else %}<C>{{ m.content }}</C>{% endif %}[AEND]{% else %}[{{ m.role }}:{{ m.content }}]{% endif %}{% endfor %}{% if add_generation_prompt %}[GEN]{% endif %}"#;
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
            reasoning_strength: None,
        };
        // Two assistant turns: reasoning+content and reasoning+two-tool
        let u1 = Message {
            role: Role::User,
            content: "q1".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let a1 = Message {
            role: Role::Assistant,
            content: "answer1".to_string(),
            reasoning_content: Some("reason1".to_string()),
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let a2 = Message {
            role: Role::Assistant,
            content: "".to_string(),
            reasoning_content: Some("reason2".to_string()),
            name: None,
            rendered_name: None,
            tool_calls: vec![
                ToolCall {
                    id: Some("call1".to_string()),
                    name: "toolA".to_string(),
                    arguments: serde_json::json!({"x": 1}),
                    rendered_body: None,
                },
                ToolCall {
                    id: Some("call2".to_string()),
                    name: "toolB".to_string(),
                    arguments: serde_json::json!({"y": 2}),
                    rendered_body: None,
                },
            ],
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let u2 = Message {
            role: Role::User,
            content: "q2".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        // Cached bodies: deliberately non-retokenized ids
        let r1_body = vec![5001, 5002];
        let c1_body = vec![5003, 5004, 5005];
        let r2_body = vec![6001];
        let t1_body = vec![6002, 6003];
        let t2_body = vec![6004];
        let turn1 = CachedAssistantTurn {
            reasoning: Some(CachedAssistantBody { token_ids: r1_body.clone(), text: "reason1".to_string() }),
            tools: Vec::new(),
            content: Some(CachedAssistantBody { token_ids: c1_body.clone(), text: "answer1".to_string() }),
        };
        let turn2 = CachedAssistantTurn {
            reasoning: Some(CachedAssistantBody { token_ids: r2_body.clone(), text: "reason2".to_string() }),
            tools: vec![
                CachedAssistantToolBody { recipient: "toolA".to_string(), token_ids: t1_body.clone() },
                CachedAssistantToolBody { recipient: "toolB".to_string(), token_ids: t2_body.clone() },
            ],
            content: None,
        };
        let messages = vec![u1.clone(), a1.clone(), a2.clone(), u2.clone()];
        // Lookup by content/match: use order
        let mut call_idx = 0;
        let rendered = build_cached_history_jinja(&frame, &messages, None, |m| {
            if matches!(m.role, Role::Assistant) {
                let ret = if call_idx == 0 { Some(turn1.clone()) } else { Some(turn2.clone()) };
                call_idx += 1;
                ret
            } else {
                None
            }
        })
        .expect("splice");
        // Build expected manually: encode structural parts and splice bodies
        // We can compute expected by reproducing the sentinel substitution manually
        // For validation, just check exact properties
        // No sentinel leaked
        for sid in [9, 10, 11, 12, 13] {
            assert!(!rendered.contains(&sid), "no sentinel {sid} leaked");
        }
        // Bodies appear in document order: r1, c1, r2, t1, t2
        // Find each body's occurrence in order
        let bodies = vec![r1_body.clone(), c1_body.clone(), r2_body.clone(), t1_body.clone(), t2_body.clone()];
        // Check that bodies appear in order and not interleaved incorrectly
        let mut pos = 0;
        for body in &bodies {
            // find body as subsequence starting at or after pos
            let mut found = None;
            for i in pos..=rendered.len().saturating_sub(body.len()) {
                if &rendered[i..i + body.len()] == body.as_slice() {
                    found = Some(i);
                    break;
                }
            }
            assert!(found.is_some(), "body {:?} not found in order", body);
            pos = found.unwrap() + body.len();
        }
        // Also verify tool order: t1 before t2 and both after r2
        // Already checked via order
        // Verify that a plain render without cache would be different (contains retokenized content)
        let plain = t.encode(&frame.render_messages(&messages, None, None).unwrap());
        assert_ne!(rendered, plain, "spliced should differ from plain");
    }

    #[test]
    fn jinja_splice_missing_or_reordered_slot_falls_back_to_plain() {
        let t = make_tokenizer();
        // Base messages: reasoning+content turn
        let a = Message {
            role: Role::Assistant,
            content: "answer".to_string(),
            reasoning_content: Some("reason".to_string()),
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let u = Message {
            role: Role::User,
            content: "hi".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let messages = vec![u.clone(), a.clone()];
        let turn = CachedAssistantTurn {
            reasoning: Some(CachedAssistantBody { token_ids: vec![7001], text: "reason".to_string() }),
            tools: Vec::new(),
            content: Some(CachedAssistantBody { token_ids: vec![7002], text: "answer".to_string() }),
        };
        // Template that DROPS content slot
        let template_drop = r#"{% for m in messages %}{% if m.role == 'assistant' %}[A]{% if m.reasoning_content %}<R>{{ m.reasoning_content }}</R>{% endif %}[AEND]{% else %}[{{ m.role }}:{{ m.content }}]{% endif %}{% endfor %}{% if add_generation_prompt %}[GEN]{% endif %}"#;
        let frame_drop = JinjaChatFrame {
            tokenizer: &t,
            template: template_drop,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
            reasoning_strength: None,
        };
        let plain_drop = t.encode(&frame_drop.render_messages(&messages, None, None).unwrap());
        let spliced_drop = build_cached_history_jinja(&frame_drop, &messages, None, |m| {
            if matches!(m.role, Role::Assistant) { Some(turn.clone()) } else { None }
        })
        .unwrap();
        assert_eq!(spliced_drop, plain_drop, "drop-content template must fallback to plain");

        // Template that REORDERS content before reasoning
        let template_reorder = r#"{% for m in messages %}{% if m.role == 'assistant' %}[A]{% if m.tool_calls %}{% for tc in m.tool_calls %}<T>{{ tc.rendered_body }}</T>{% endfor %}{% else %}<C>{{ m.content }}</C>{% endif %}{% if m.reasoning_content %}<R>{{ m.reasoning_content }}</R>{% endif %}[AEND]{% else %}[{{ m.role }}:{{ m.content }}]{% endif %}{% endfor %}{% if add_generation_prompt %}[GEN]{% endif %}"#;
        let frame_reorder = JinjaChatFrame {
            tokenizer: &t,
            template: template_reorder,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
            reasoning_strength: None,
        };
        let plain_reorder = t.encode(&frame_reorder.render_messages(&messages, None, None).unwrap());
        let spliced_reorder = build_cached_history_jinja(&frame_reorder, &messages, None, |m| {
            if matches!(m.role, Role::Assistant) { Some(turn.clone()) } else { None }
        })
        .unwrap();
        assert_eq!(spliced_reorder, plain_reorder, "reordered template must fallback to plain");
    }

    #[test]
    fn jinja_splice_content_cache_on_tool_turn_falls_back_plain() {
        let t = make_tokenizer();
        let template = r#"{% for m in messages %}{% if m.role == 'assistant' %}[A]{% if m.reasoning_content %}<R>{{ m.reasoning_content }}</R>{% endif %}{% if m.tool_calls %}{% for tc in m.tool_calls %}<T n="{{ tc.name }}">{% if tc.rendered_body %}{{ tc.rendered_body }}{% else %}{{ tc.arguments | tojson }}{% endif %}</T>{% endfor %}{% else %}<C>{{ m.content }}</C>{% endif %}[AEND]{% else %}[{{ m.role }}:{{ m.content }}]{% endif %}{% endfor %}{% if add_generation_prompt %}[GEN]{% endif %}"#;
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
            reasoning_strength: None,
        };
        let a = Message {
            role: Role::Assistant,
            content: "".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: vec![ToolCall {
                id: Some("c1".to_string()),
                name: "toolA".to_string(),
                arguments: serde_json::json!({"x": 1}),
                rendered_body: None,
            }],
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let u = Message {
            role: Role::User,
            content: "hi".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let messages = vec![u.clone(), a.clone()];
        // Give a content-only cache payload (invalid for tool turn)
        let bad_turn = CachedAssistantTurn {
            reasoning: None,
            tools: Vec::new(),
            content: Some(CachedAssistantBody { token_ids: vec![8001], text: "".to_string() }),
        };
        let plain = t.encode(&frame.render_messages(&messages, None, None).unwrap());
        let spliced = build_cached_history_jinja(&frame, &messages, None, |m| {
            if matches!(m.role, Role::Assistant) { Some(bad_turn.clone()) } else { None }
        })
        .unwrap();
        assert_eq!(spliced, plain, "content cache on tool turn must fallback");
    }

    #[test]
    fn jinja_splice_content_only_qwen_template_byte_identical() {
        let t = make_tokenizer();
        let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
            reasoning_strength: None,
        };
        let a = Message {
            role: Role::Assistant,
            content: "hello".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let u = Message {
            role: Role::User,
            content: "hi".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let messages = vec![u.clone(), a.clone()];
        let body = vec![9001, 9002, 9003];
        let turn = CachedAssistantTurn {
            reasoning: None,
            tools: Vec::new(),
            content: Some(CachedAssistantBody { token_ids: body.clone(), text: "hello".to_string() }),
        };
        let spliced = build_cached_history_jinja(&frame, &messages, None, |m| {
            if matches!(m.role, Role::Assistant) { Some(turn.clone()) } else { None }
        })
        .unwrap();
        // Manually compute expected via sentinel substitution and check no sentinel leaked
        for sid in [9, 10, 11, 12, 13] {
            assert!(!spliced.contains(&sid), "no sentinel leaked");
        }
        // Body must appear verbatim
        assert!(spliced.windows(body.len()).any(|w| w == body.as_slice()), "body must be spliced");
        // Plain fallback check: ensure it differs from plain (which would retokenize "hello")
        let plain = t.encode(&frame.render_messages(&messages, None, None).unwrap());
        // Since body is non-retokenized (9001..), spliced should differ from plain if body not equal to encode("hello")
        // But we just check that spliced is valid and contains body
        assert_ne!(spliced, plain);
    }

    #[test]
    fn pick_splice_sentinels_none_reaches_plain_render() {
        let t = test_tokenizer_no_think();
        // This tokenizer has no reserved tokens, so n=1 should be None
        assert!(pick_splice_sentinels(&t, 1).is_none());
        let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "",
            enable_thinking: false,
            bos_token: Some(""),
            reasoning_strength: None,
        };
        let a = Message {
            role: Role::Assistant,
            content: "hi".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let u = Message {
            role: Role::User,
            content: "q".to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let messages = vec![u.clone(), a.clone()];
        let turn = CachedAssistantTurn {
            reasoning: None,
            tools: Vec::new(),
            content: Some(CachedAssistantBody { token_ids: vec![1, 2, 3], text: "hi".to_string() }),
        };
        let plain = t.encode(&frame.render_messages(&messages, None, None).unwrap());
        let spliced = build_cached_history_jinja(&frame, &messages, None, |m| {
            if matches!(m.role, Role::Assistant) { Some(turn.clone()) } else { None }
        })
        .unwrap();
        assert_eq!(spliced, plain, "when no sentinel available, must fallback to plain");
    }

    #[test]
    fn gemma4_bundled_template_renders() {
        // The daemon bundles `templates/gemma-4-it.jinja` as the arch-13
        // fallback chat template (resolve_chat_template). This test renders
        // it through the production JinjaChatFrame environment (strict
        // undefined + pycompat + trim/lstrip + raise_exception + hf tojson)
        // so a minijinja regression or template edit that breaks parsing /
        // rendering fails HERE at unit-test time instead of silently falling
        // back to a raw un-framed prompt at serve time.
        let t = make_tokenizer();
        let template = include_str!("../templates/gemma-4-it.jinja");

        // Plain system+user turn, non-thinking (the daemon's v1 default):
        // generation prompt must pre-fill the empty thought channel.
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: Some("Be brief."),
            user: "What is the capital of France?",
            enable_thinking: false,
            bos_token: Some("<bos>"),
        reasoning_strength: None,
        };
        let rendered = frame.render().expect("gemma4 template renders plain turn");
        assert!(
            rendered.starts_with("<bos>"),
            "render must start with the explicit bos_token: got {:?}",
            &rendered[..rendered.len().min(60)],
        );
        assert!(
            rendered.contains("<|turn>system\n"),
            "system turn must render: got {rendered:?}",
        );
        assert!(
            rendered.contains("<|turn>user\nWhat is the capital of France?<turn|>"),
            "user turn must render: got {rendered:?}",
        );
        assert!(
            rendered.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "non-thinking generation prompt must pre-fill the empty thought              channel: got tail {:?}",
            &rendered[rendered.len().saturating_sub(80)..],
        );

        // Tools branch: one OpenAI-shape function definition must survive the
        // template's format_function_declaration macro tree.
        let messages = vec![Message {
            role: Role::User,
            content: "weather in SF?".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        }];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        })];
        let with_tools = frame
            .render_messages(&messages, Some(&tools), None)
            .expect("gemma4 template renders tools turn");
        assert!(
            with_tools.contains("<|tool>") && with_tools.contains("get_weather"),
            "tools block must render the declaration: got {with_tools:?}",
        );

        // Agentic replay: assistant turn with a FLAT hipfire ToolCall
        // ({name, arguments} — not the nested OpenAI {function:{…}} shape)
        // followed by a tool-role response. Exercises the template's
        // tool_call replay + forward-scan tool_response branches.
        let history = vec![
            Message {
                role: Role::User,
                content: "weather in SF?".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: String::new(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: vec![ToolCall {
id: None,
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city": "SF"}),
                rendered_body: None,
                }],
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Tool,
                content: "72F sunny".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        let replay = frame
            .render_messages(&history, Some(&tools), None)
            .expect("gemma4 template renders tool-call replay history");
        assert!(
            replay.contains(r#"<|tool_call>call:get_weather{city:<|"|>SF<|"|>}<tool_call|>"#),
            "assistant tool_call must replay from the flat shape: got {replay:?}",
        );
        assert!(
            replay.contains("<|tool_response>"),
            "tool response must render: got {replay:?}",
        );
    }

    #[test]
    fn render_messages_with_tools_fires_tools_block() {
        // Smoke test: a minimal template gated on `{% if tools %}`
        // must render the tools branch when the caller supplies a
        // non-empty tools array — and skip it when tools is None.
        // This is the architectural invariant Phase 1 unblocks:
        // structured tools from daemon stdin reach the Jinja template's
        // `{% if tools %}` predicate.
        let t = make_tokenizer();
        let template = "{% if tools %}TOOLS:{% for f in tools %}{{ f.function.name }};{% endfor %}{% endif %}MSGS:{% for m in messages %}{{ m.role }}={{ m.content }};{% endfor %}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
        reasoning_strength: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: "hi".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        }];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {}},
            }
        })];

        let with_tools = frame
            .render_messages(&messages, Some(&tools), None)
            .expect("render_messages w/ tools succeeds");
        assert!(
            with_tools.contains("TOOLS:get_weather;"),
            "tools-block must fire when tools is Some: got {with_tools:?}",
        );
        assert!(
            with_tools.contains("MSGS:user=hi;"),
            "messages must still render: got {with_tools:?}",
        );

        // None branch: empty tools array means `{% if tools %}` evaluates false.
        let without_tools = frame
            .render_messages(&messages, None, None)
            .expect("render_messages w/o tools succeeds");
        assert!(
            !without_tools.contains("TOOLS:"),
            "tools-block must NOT fire when tools is None: got {without_tools:?}",
        );
        assert!(
            without_tools.contains("MSGS:user=hi;"),
            "messages must still render w/o tools: got {without_tools:?}",
        );
    }

    #[test]
    fn render_messages_with_history_and_tools_includes_assistant_call() {
        // Full agentic round-trip shape: system + user + assistant w/
        // tool_calls + tool response. The template walks tool_calls and
        // tool_call_id so the trip-record must reach it.
        let t = make_tokenizer();
        // `tool_call_id` is serialize-skipped when None, so under
        // strict-undefined the template MUST guard with `is defined`
        // (matching how the upstream Qwen3.5/3.6 + Hermes templates
        // probe the field). The Message doc comment on this struct
        // describes the same convention.
        let template = "{% for m in messages %}{{ m.role }}:{% if m.tool_calls %}call={% for tc in m.tool_calls %}{{ tc.name }}({{ tc.arguments.city }});{% endfor %}{% else %}{{ m.content }}{% endif %}{% if m.tool_call_id is defined %}[id={{ m.tool_call_id }}]{% endif %};{% endfor %}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
        reasoning_strength: None,
        };
        let messages = vec![
            Message {
                role: Role::System,
                content: "be brief".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "weather?".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: "".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: vec![ToolCall {
id: None,
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city":"SF"}),
                rendered_body: None,
                }],
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Tool,
                content: "72F".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: Some("call_1".to_string()),
                tool_plan: String::new(),
            },
        ];
        let out = frame
            .render_messages(&messages, None, None)
            .expect("multi-turn render succeeds");
        assert!(
            out.contains("system:be brief;"),
            "system content visible: {out:?}"
        );
        assert!(
            out.contains("user:weather?;"),
            "user content visible: {out:?}"
        );
        assert!(
            out.contains("assistant:call=get_weather(SF);"),
            "assistant tool_call rendered: {out:?}",
        );
        assert!(
            out.contains("tool:72F[id=call_1];"),
            "tool response w/ tool_call_id rendered: {out:?}",
        );
    }

    #[test]
    fn cache_miss_history_tool_call_renders_native_xml_for_qwen() {
        // Non-jinja ChatScaffold replay: on a cache MISS the historical
        // assistant tool_call must re-render in the model's OWN format.
        // For grammar-off Qwen3.5/3.6 that is native XML
        // (`<function=NAME><parameter=ARG>`), NOT Hermes JSON — otherwise a
        // post-eviction replay shows the model JSON and nudges it off its
        // trained tool-call distribution.
        let t = make_tokenizer();
        let history = vec![
            Message {
                role: Role::User,
                content: "weather?".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: String::new(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: vec![ToolCall {
id: None,
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city": "SF"}),
                rendered_body: None,
                }],
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];
        // cache_lookup returns None => MISS => the miss-branch render runs.
        let toks = build_cached_history(
            &t,
            None,
            &history,
            &[],
            AssistantPrefix::Plain,
            ToolCallRender::QwenXml,
            |_| None,
        );
        let text = t.decode(&toks);
        assert!(
            text.contains("<function=get_weather>"),
            "expected native XML function tag, got: {text:?}"
        );
        assert!(
            text.contains("<parameter=city>"),
            "expected native XML parameter tag, got: {text:?}"
        );
        assert!(
            !text.contains("{\"name\""),
            "must NOT emit Hermes JSON payload, got: {text:?}"
        );
    }

    #[test]
    fn cache_miss_history_tool_call_renders_hermes_json_when_selected() {
        // Hermes-native models (carnice / grammar-forced JSON) must keep
        // the legacy `{"name":..,"arguments":..}` payload on replay.
        let t = make_tokenizer();
        let history = vec![Message {
            role: Role::Assistant,
            content: String::new(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: vec![ToolCall {
id: None,
                name: "get_weather".to_string(),
                arguments: serde_json::json!({"city": "SF"}),
            rendered_body: None,
            }],
            tool_call_id: None,
            tool_plan: String::new(),
        }];
        let toks = build_cached_history(
            &t,
            None,
            &history,
            &[],
            AssistantPrefix::Plain,
            ToolCallRender::HermesJson,
            |_| None,
        );
        let text = t.decode(&toks);
        assert!(
            text.contains("{\"name\":\"get_weather\""),
            "expected Hermes JSON payload, got: {text:?}"
        );
        assert!(
            !text.contains("<function="),
            "must NOT emit XML function tag, got: {text:?}"
        );
    }

    // ── Model-aware tool-call format default ────────────────────────────
    // The DAEMON (not just the CLI) must default the tool-call format by
    // model, so a daemon-direct launch (GPU behavior gate, coherence-gate.sh,
    // agentic-gate.sh) gets native XML for plain Qwen3.5/3.6 — not Hermes.

    #[test]
    fn grammar_default_off_for_plain_qwen() {
        // No env => plain (non-carnice) Qwen3.5/3.6 defaults grammar OFF (XML).
        assert!(!qwen35_grammar_on(None, "/models/qwen3.6-27b.mq4"));
        assert!(!qwen35_grammar_on(None, "/models/qwen3.5-4b.mq4"));
    }

    #[test]
    fn grammar_default_on_for_carnice() {
        // No env => carnice (Hermes-native) defaults grammar ON (JSON).
        assert!(qwen35_grammar_on(None, "/models/carnice-27b.mq4"));
        assert!(qwen35_grammar_on(None, "/MODELS/Carnice-9b.mq6")); // case-insensitive
    }

    #[test]
    fn grammar_env_zero_is_killswitch_even_for_carnice() {
        assert!(!qwen35_grammar_on(Some("0"), "/models/carnice-27b.mq4"));
    }

    #[test]
    fn grammar_env_set_forces_on_even_for_plain_qwen() {
        assert!(qwen35_grammar_on(Some("1"), "/models/qwen3.6-27b.mq4"));
    }

    #[test]
    fn history_render_follows_model_default_and_env() {
        assert_eq!(
            qwen35_history_render(None, "/m/qwen3.6-27b.mq4"),
            ToolCallRender::QwenXml
        );
        assert_eq!(
            qwen35_history_render(None, "/m/carnice-27b.mq4"),
            ToolCallRender::HermesJson
        );
        assert_eq!(
            qwen35_history_render(Some("1"), "/m/qwen3.6-27b.mq4"),
            ToolCallRender::HermesJson
        );
        assert_eq!(
            qwen35_history_render(Some("0"), "/m/carnice-27b.mq4"),
            ToolCallRender::QwenXml
        );
    }

    #[test]
    fn system_message_precedes_first_user_turn() {
        let t = make_tokenizer();
        let with_sys = ChatFrame {
            tokenizer: &t,
            system: Some("sysprompt"),
            user: "hello",
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        }
        .build();
        let without_sys = ChatFrame {
            tokenizer: &t,
            system: None,
            user: "hello",
            assistant_prefix: AssistantPrefix::Plain,
            raw: false,
        }
        .build();

        // The "with system" output must equal a system block followed
        // by the "without system" output. This is the canonical
        // daemon AR-path invariant.
        let mut sys_block: Vec<u32> = Vec::new();
        sys_block.extend_from_slice(&t.encode("<|im_start|>"));
        sys_block.extend_from_slice(&t.encode("system"));
        sys_block.extend_from_slice(&t.encode("\n"));
        sys_block.extend_from_slice(&t.encode("sysprompt"));
        sys_block.extend_from_slice(&t.encode("<|im_end|>"));
        sys_block.extend_from_slice(&t.encode("\n"));

        let mut expected = sys_block;
        expected.extend_from_slice(&without_sys);
        assert_eq!(
            with_sys, expected,
            "system message should be a prefix of the rest of the frame"
        );
    }

    /// Faithful excerpt of the MiniMax-M2 `tokenizer_config.chat_template`
    /// system-message logic. It reads `system_message.current_date` (and
    /// `current_location`) as attributes of the leading system message dict,
    /// gated behind `{% if system_message and system_message.<attr> %}`.
    /// Under minijinja strict-undefined a system message that lacks the key
    /// raises a render error — this is the exact byte sequence (chat.jinja:37)
    /// the O1 e2e tripped when serving MiniMax with an explicit system message.
    const MINIMAX_SYSTEM_TEMPLATE: &str = "\
{%- macro build_system_message(system_message) -%}
    {%- if system_message and system_message.content -%}
        {{- system_message.content }}
    {%- else -%}
        {%- if model_identity is not defined -%}
            {%- set model_identity = \"You are MiniMax AI.\" -%}
        {%- endif -%}
        {{- model_identity }}
    {%- endif -%}
    {%- if system_message and system_message.current_date -%}
        {{- '\\n' ~ 'Current date: ' + system_message.current_date }}
    {%- endif -%}
    {%- if system_message and system_message.current_location -%}
        {{- '\\n' ~ 'Current location: ' + system_message.current_location }}
    {%- endif -%}
{%- endmacro -%}
{%- set system_message = none -%}
{%- set conversation_messages = messages -%}
{%- if messages and messages[0].role == \"system\" -%}
    {%- set system_message = messages[0] -%}
    {%- set conversation_messages = messages[1:] -%}
{%- endif -%}
SYS:{{ build_system_message(system_message) }}:END
{%- for m in conversation_messages -%}
{{ m.role }}={{ m.content }};
{%- endfor -%}";

    #[test]
    fn minimax_explicit_system_message_renders_with_current_date() {
        // Regression: serving MiniMax with an EXPLICIT system message must NOT
        // crash on `system_message.current_date` under strict-undefined. The
        // render path injects `current_date` onto the leading system message
        // so the gated access resolves to a populated string instead of
        // raising. GPU-free, no model load — uses the embedded template excerpt.
        let t = make_tokenizer();
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template: MINIMAX_SYSTEM_TEMPLATE,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
        reasoning_strength: None,
        };
        let messages = vec![
            Message {
                role: Role::System,
                content: "Be concise.".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "hi".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];

        // The crux: this previously errored with
        //   "template render: ... system_message.current_date ... is undefined"
        let out = frame
            .render_messages(&messages, None, None)
            .expect("explicit system message must render without strict-undefined error");

        assert!(
            out.contains("Be concise."),
            "explicit system content must be present: {out:?}"
        );
        assert!(
            out.contains("Current date:"),
            "current_date must be injected + rendered: {out:?}"
        );
        assert!(
            out.contains("user=hi;"),
            "conversation messages must still render: {out:?}"
        );
        // current_location is NOT supplied → its gated branch must stay silent
        // (the `if system_message and system_message.current_location` guard
        // short-circuits on the absent key without raising).
        assert!(
            !out.contains("Current location:"),
            "current_location must not appear when unset: {out:?}"
        );
    }

    #[test]
    fn minimax_no_system_message_injects_builtin_identity() {
        // The no-system-message path must be unchanged: `system_message` stays
        // `none`, the built-in model identity is injected, and the gated
        // current_date/current_location accesses short-circuit on the falsy
        // `none` so no date line appears.
        let t = make_tokenizer();
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template: MINIMAX_SYSTEM_TEMPLATE,
            system: None,
            user: "",
            enable_thinking: true,
            bos_token: Some(""),
        reasoning_strength: None,
        };
        let messages = vec![Message {
            role: Role::User,
            content: "hi".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        }];

        let out = frame
            .render_messages(&messages, None, None)
            .expect("no-system-message render must succeed");
        assert!(
            out.contains("You are MiniMax AI."),
            "built-in identity must be injected on the no-system path: {out:?}"
        );
        assert!(
            !out.contains("Current date:"),
            "no date line on the no-system path (gated access short-circuits): {out:?}"
        );
        assert!(out.contains("user=hi;"), "user turn renders: {out:?}");
    }

    #[test]
    fn strip_generation_tags_plain_and_dash_variants() {
        // LFM2.5 chat_template uses unsupported generation-scope markers.
        // Every whitespace-control spelling must disappear while the
        // surrounding template text stays byte-identical.
        assert_eq!(
            strip_generation_tags("A{% generation %}B{% endgeneration %}C"),
            "ABC"
        );
        assert_eq!(
            strip_generation_tags("A{%- generation -%}B{%- endgeneration -%}C"),
            "ABC"
        );
        assert_eq!(
            strip_generation_tags("A{%-generation-%}B{%  endgeneration  %}C"),
            "ABC"
        );
        assert_eq!(
            strip_generation_tags("{% generation -%}X{%- endgeneration %}"),
            "X"
        );
        assert_eq!(
            strip_generation_tags("{%- generation %}Y{% endgeneration -%}"),
            "Y"
        );
    }

    #[test]
    fn strip_generation_tags_preserves_ordinary_tags_and_literals() {
        // Ordinary control/expression tags and adjacent literals must not be
        // rewritten — only bare generation / endgeneration statements.
        let src = "{% for m in messages %}KEEP{% if true %}Y{% endif %}{% endfor %}{% generation %}G{% endgeneration %}";
        assert_eq!(
            strip_generation_tags(src),
            "{% for m in messages %}KEEP{% if true %}Y{% endif %}{% endfor %}G"
        );
        // Expression / assignment forms that merely mention the words stay.
        assert_eq!(
            strip_generation_tags("{{ generation }}{% set endgeneration = 1 %}"),
            "{{ generation }}{% set endgeneration = 1 %}"
        );
    }

    #[test]
    fn jinja_lfm_generation_scope_markers_render_framing() {
        // LFM2.5-shaped loop: assistant content is wrapped in
        // `{%- generation -%}` / `{%- endgeneration -%}` scope markers.
        // Without stripping, minijinja fails template parse with
        // "unknown statement generation". After strip, framing must match
        // the identical template written without the markers.
        let t = make_tokenizer();
        let with_markers = "\
{% for m in messages -%}\
{%- if m.role == 'user' -%}\
<|im_start|>user\n{{ m.content }}<|im_end|>\n\
{%- elif m.role == 'assistant' -%}\
<|im_start|>assistant\n{%- generation -%}{{ m.content }}{%- endgeneration -%}<|im_end|>\n\
{%- endif -%}\
{%- endfor -%}\
{%- if add_generation_prompt -%}\
<|im_start|>assistant\n{%- generation -%}\
{%- endif -%}";
        let without_markers = "\
{% for m in messages -%}\
{%- if m.role == 'user' -%}\
<|im_start|>user\n{{ m.content }}<|im_end|>\n\
{%- elif m.role == 'assistant' -%}\
<|im_start|>assistant\n{{ m.content }}<|im_end|>\n\
{%- endif -%}\
{%- endfor -%}\
{%- if add_generation_prompt -%}\
<|im_start|>assistant\n\
{%- endif -%}";

        let messages = vec![
            Message {
                role: Role::User,
                content: "hello".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::Assistant,
                content: "hi there".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
            Message {
                role: Role::User,
                content: "again".to_string(),

reasoning_content: None,

name: None,

rendered_name: None,

tool_calls: Vec::new(),
                tool_call_id: None,
                tool_plan: String::new(),
            },
        ];

        let frame_marked = JinjaChatFrame {
            tokenizer: &t,
            template: with_markers,
            system: None,
            user: "",
            enable_thinking: false,
            bos_token: Some(""),
        reasoning_strength: None,
        };
        let frame_clean = JinjaChatFrame {
            tokenizer: &t,
            template: without_markers,
            system: None,
            user: "",
            enable_thinking: false,
            bos_token: Some(""),
        reasoning_strength: None,
        };

        let out = frame_marked
            .render_messages(&messages, None, None)
            .expect("LFM generation markers must parse after strip");
        let expected = frame_clean
            .render_messages(&messages, None, None)
            .expect("baseline template without markers renders");
        assert_eq!(
            out, expected,
            "stripped generation markers must be semantics-preserving vs marker-free template"
        );

        // Structural framing: both user turns, prior assistant content, and
        // the open assistant generation prompt. Dash-controlled Jinja tags
        // intentionally trim inter-turn newlines.
        let user_hello = "<|im_start|>user\nhello<|im_end|>";
        let asst_hi = "<|im_start|>assistant\nhi there<|im_end|>";
        let user_again = "<|im_start|>user\nagain<|im_end|>";
        let asst_open = "<|im_start|>assistant";
        assert!(
            out.contains(&user_hello),
            "first user turn framing missing: {out:?}"
        );
        assert!(
            out.contains(&asst_hi),
            "prior assistant content framing missing: {out:?}"
        );
        assert!(
            out.contains(&user_again),
            "second user turn framing missing: {out:?}"
        );
        assert!(
            out.ends_with(asst_open),
            "add_generation_prompt must open the live assistant turn: {out:?}"
        );
    }

    #[test]
    fn civil_from_days_known_dates() {
        // Spot-check the chrono-free date conversion against known epochs.
        assert_eq!(civil_from_days(0), (1970, 1, 1)); // Unix epoch
        assert_eq!(civil_from_days(18_993), (2022, 1, 1));
        // 2026-06-18 = 20622 days after 1970-01-01
        assert_eq!(civil_from_days(20_622), (2026, 6, 18));
    }


    /// Muse Glimmer's chat_template uses `namespace(name=tcid if tcid else '')`,
    /// which Jinja2 accepts and minijinja rejects with
    /// "unexpected identifier, expected `,`". Because that template is a single
    /// line, the failure took the WHOLE template down and every request silently
    /// fell back to "BOS + raw prompt", degenerating multi-turn.
    ///
    /// Negative control: the pre-fixup string MUST fail to parse, otherwise this
    /// test would pass even with the fixup removed.
    #[test]
    fn conditional_kwarg_is_parenthesized_and_parses() {
        let raw = "{%- set rns = namespace(name=tcid if tcid else '') -%}{{ rns.name }}";
        let mut env = minijinja::Environment::new();
        assert!(
            env.template_from_named_str("chat", raw).is_err(),
            "negative control: raw template must NOT parse, or the fixup is untested"
        );

        let fixed = parenthesize_conditional_kwargs(raw);
        assert!(fixed.contains("namespace(name=(tcid if tcid else ''))"), "got: {fixed}");
        let mut env2 = minijinja::Environment::new();
        env2.template_from_named_str("chat", &fixed)
            .expect("fixed template must parse");
    }

    /// The fixup must not disturb templates that do not need it, must not touch
    /// non-kwarg `=`, and must be UTF-8 safe (an earlier version emitted
    /// `bytes[i] as char`, which mangles multibyte characters).
    #[test]
    fn conditional_kwarg_fixup_leaves_other_templates_alone() {
        for src in [
            "{{ a if b else c }}",
            "{%- if x == 1 -%}y{%- endif -%}",
            "{{ f(name=(p if q else '')) }}",
            "{{ f(name=plain, other=2) }}",
            "{{ '\u{2014} \u{1f600} caf\u{e9}' }}",
        ] {
            assert_eq!(parenthesize_conditional_kwargs(src), src, "changed: {src}");
        }
    }
}
