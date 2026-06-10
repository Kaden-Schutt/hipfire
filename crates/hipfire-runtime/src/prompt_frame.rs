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
                        let payload = serde_json::json!({
                            "name": tc.name,
                            "arguments": tc.arguments,
                        });
                        let rendered = format!(
                            "\n<tool_call>\n{}\n</tool_call>",
                            serde_json::to_string(&payload).unwrap_or_default(),
                        );
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
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Set on Tool-role messages to identify which assistant tool_call
    /// this is responding to. Qwen3.5/3.6 templates currently ignore
    /// this field; OpenAI-spec clients and some other templates require
    /// it. Skipped from the serialized JSON when None so templates that
    /// `is defined` against it don't see a misleading null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// One assistant-emitted tool call, attached to an assistant `Message`.
/// `arguments` is a free-form JSON value (typically an object). Templates
/// that render in XML format (Qwen3.5/3.6's `<function=NAME><parameter=ARG>`
/// shape) walk this with `arguments | items` under pycompat.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
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
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("tojson: {e}"))
    })?;
    String::from_utf8(buf).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson utf8: {e}"),
        )
    })
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
                tool_calls: Vec::new(),
                tool_call_id: None,
            });
        }
        messages.push(Message {
            role: Role::User,
            content: self.user.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
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
        // Strict-undefined: a missing context variable raises Err instead of
        // silently rendering empty/partial output. Without this, malformed
        // prompts could propagate to the model unnoticed (Codex review on
        // PR #175 flagged this; we apply it here in the same port).
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
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

        env.add_template("chat", self.template)
            .map_err(|e| format!("template parse: {e}"))?;
        let tmpl = env.get_template("chat")
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
        let ctx = minijinja::context! {
            messages => Value::from_serialize(messages),
            add_generation_prompt => true,
            enable_thinking => self.enable_thinking,
            bos_token => bos_token,
            tools => tools_val,
            documents => Value::from_serialize(&empty_list),
            tool_call_kwargs => kwargs_val,
        };
        tmpl.render(ctx).map_err(|e| format!("template render: {e}"))
    }
}

/// Pick an atomic special-token sentinel for the verbatim-splice render.
///
/// The sentinel must (1) encode to exactly one token (so it never BPE-merges
/// with neighbouring template text — every structural token then stays
/// byte-identical to a pure render) and (2) never be emitted by the template
/// itself (so its post-render occurrence count equals the number of spliced
/// assistant turns). We prefer obviously-reserved tokens (`reserved` / `unused`
/// / `pad` in the name) and otherwise take any non-structural special token
/// that round-trips atomically. Returns `None` when the tokenizer exposes no
/// usable sentinel — the caller then falls back to a plain (retokenized) render.
fn pick_splice_sentinel(tok: &Tokenizer) -> Option<String> {
    // Tokens the chat templates emit structurally — never use these as a
    // sentinel (their post-render count wouldn't equal the spliced-turn count).
    const STRUCTURAL: &[&str] = &[
        "<|im_start|>", "<|im_end|>", "<think>", "</think>",
        "<|endoftext|>", "<|begin_of_text|>", "<|end_of_text|>",
        "<s>", "</s>", "<bos>", "<eos>", "<unk>", "<pad>", "<|file_separator|>",
    ];
    let atomic = |s: &str| -> bool {
        tok.special_token_id(s).map_or(false, |id| tok.encode(s) == vec![id])
    };
    // First pass: obviously-reserved scratch tokens.
    for (s, _id) in tok.special_tokens() {
        if STRUCTURAL.contains(&s.as_str()) { continue; }
        let ls = s.to_ascii_lowercase();
        if (ls.contains("reserved") || ls.contains("unused") || ls.contains("pad"))
            && atomic(s)
        {
            return Some(s.clone());
        }
    }
    // Second pass: any non-structural special token that round-trips atomically.
    for (s, _id) in tok.special_tokens() {
        if STRUCTURAL.contains(&s.as_str()) { continue; }
        if atomic(s) { return Some(s.clone()); }
    }
    None
}

/// Jinja-native analogue of [`build_cached_history`]: render the conversation
/// through the model's **trained** `chat_template` (no hand-rolled
/// `ChatScaffold`), but splice the VERBATIM generated tokens of each cached
/// assistant turn in place of its content. The resulting token stream
/// byte-exactly reproduces what the daemon prefilled when that turn was
/// generated, so the downstream LCP prompt-cache hits reliably even for
/// thinking models — whose generated `<think>…</think>` tokens cannot be
/// recovered by re-tokenizing the API-stripped visible content (the exact
/// failure mode that makes a plain re-render miss at the assistant boundary).
///
/// `messages` is the full conversation INCLUDING the live user turn last (the
/// template's `add_generation_prompt` then appends the assistant opener). For
/// each assistant turn, `cache_lookup` returns `Some(verbatim_tokens)` — the
/// tokens that occupied that turn's content slot in `conversation_tokens` — or
/// `None` (no cache entry: that turn keeps its retokenized content, which only
/// costs a safe LCP miss at/after it).
///
/// Mechanism: substitute each cached assistant turn's content with an atomic
/// special-token sentinel, render via [`JinjaChatFrame::render_messages`],
/// tokenize, then replace each sentinel token with the cached tokens. The
/// substitution is verified (sentinel occurs exactly once per cached turn);
/// any mismatch — or no usable sentinel — falls back to a plain render so the
/// result is always a valid (if uncached) token stream.
pub fn build_cached_history_jinja(
    frame: &JinjaChatFrame,
    messages: &[Message],
    tools: Option<&[serde_json::Value]>,
    mut cache_lookup: impl FnMut(&Message) -> Option<Vec<u32>>,
) -> Result<Vec<u32>, String> {
    let tok = frame.tokenizer;
    let plain = |f: &JinjaChatFrame| -> Result<Vec<u32>, String> {
        Ok(tok.encode(&f.render_messages(messages, tools, None)?))
    };
    let sentinel = match pick_splice_sentinel(tok) {
        Some(s) => s,
        None => return plain(frame),
    };
    let sentinel_id = match tok.special_token_id(&sentinel) {
        Some(id) => id,
        None => return plain(frame),
    };

    // Build a messages copy where each cached assistant turn's content is the
    // sentinel; collect the cached token vectors in document order.
    let mut cached: Vec<Vec<u32>> = Vec::new();
    let mut subbed: Vec<Message> = Vec::with_capacity(messages.len());
    for m in messages {
        if matches!(m.role, Role::Assistant) {
            if let Some(toks) = cache_lookup(m) {
                cached.push(toks);
                subbed.push(Message {
                    role: Role::Assistant,
                    content: sentinel.clone(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
                continue;
            }
        }
        subbed.push(m.clone());
    }
    if cached.is_empty() {
        return plain(frame);
    }

    let toks = tok.encode(&frame.render_messages(&subbed, tools, None)?);
    // Safety: the sentinel must appear exactly once per cached turn. If the
    // template dropped/duplicated a turn, or the sentinel merged with adjacent
    // text, splicing would corrupt the stream — fall back to a plain render.
    if toks.iter().filter(|&&t| t == sentinel_id).count() != cached.len() {
        return plain(frame);
    }
    let mut out: Vec<u32> = Vec::with_capacity(toks.len() + cached.iter().map(|c| c.len()).sum::<usize>());
    let mut k = 0usize;
    for &t in &toks {
        if t == sentinel_id {
            out.extend_from_slice(&cached[k]);
            k += 1;
        } else {
            out.push(t);
        }
    }
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
                    {{"id": 9, "content": "<|reserved_0|>", "special": true}}
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
        let idx = bs.iter().position(|&x| x == b as u32).expect("byte in table");
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
        let think_id = t.special_token_id("<think>")
            .expect("test tokenizer registers <think> as special");
        let mut expected = plain.clone();
        expected.push(think_id);
        expected.extend_from_slice(&t.encode("\n"));
        assert_eq!(opened, expected, "OpenThink should append <think>\\n after the assistant prefix");
        assert!(opened.len() > plain.len(), "OpenThink output must be strictly longer than Plain");
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
        let think_id = t.special_token_id("<think>")
            .expect("test tokenizer registers <think> as special");
        let close_id = t.special_token_id("</think>")
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
        assert_eq!(closed, expected, "ClosedThink should append <think>\\n\\n</think>\\n\\n after the assistant prefix");
        assert!(closed.len() > plain.len(), "ClosedThink output must be strictly longer than Plain");
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
        assert_eq!(closed, plain, "ClosedThink without special tokens must fall back to Plain");
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
        let history: [(Role, &str); 2] =
            [(Role::User, "hello"), (Role::Assistant, "hi")];
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
        assert_eq!(via_string, via_tokens, "build_with_user_tokens must match build() when tokens align");
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
            tokenizer: &t, template, system: None, user: "",
            enable_thinking: true, bos_token: Some(""),
        };

        // Turn 1: daemon prefills R1 (prompt, ends with the primed `<think>\n`)
        // then generates `reason</think>ok`.
        let u1 = Message { role: Role::User, content: "hi".to_string(), tool_calls: vec![], tool_call_id: None };
        let r1 = t.encode(&frame.render_messages(std::slice::from_ref(&u1), None, None).unwrap());
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
        let a1 = Message { role: Role::Assistant, content: "ok".to_string(), tool_calls: vec![], tool_call_id: None };
        let u2 = Message { role: Role::User, content: "again".to_string(), tool_calls: vec![], tool_call_id: None };
        let messages_t2 = vec![u1.clone(), a1, u2];

        let rendered_t2 = build_cached_history_jinja(
            &frame, &messages_t2, None,
            |m| if matches!(m.role, Role::Assistant) { Some(asst_slot.clone()) } else { None },
        ).expect("jinja splice render");

        // No sentinel leaked.
        let sentinel_id = t.special_token_id("<|reserved_0|>").unwrap();
        assert!(!rendered_t2.contains(&sentinel_id), "sentinel must be fully replaced: {rendered_t2:?}");
        // Verbatim splice happened.
        assert!(
            rendered_t2.windows(asst_slot.len()).any(|w| w == asst_slot.as_slice()),
            "cached assistant slot must be spliced verbatim",
        );
        // THE KEY PROPERTY: turn 2 strictly extends turn 1's conversation_tokens.
        assert!(rendered_t2.len() > conv_after_t1.len(), "turn 2 must be longer than turn 1");
        assert_eq!(
            &rendered_t2[..conv_after_t1.len()], conv_after_t1.as_slice(),
            "turn 2 render must extend turn 1's conversation_tokens as a strict prefix",
        );

        // Fallback: no cache entries ⇒ identical to a plain render.
        let plain = t.encode(&frame.render_messages(&messages_t2, None, None).unwrap());
        let no_cache = build_cached_history_jinja(&frame, &messages_t2, None, |_| None).unwrap();
        assert_eq!(no_cache, plain, "no-cache path must equal a plain render");
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
        };
        let messages = vec![Message {
            role: Role::User,
            content: "hi".to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
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
        };
        let messages = vec![
            Message {
                role: Role::System,
                content: "be brief".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: "weather?".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            Message {
                role: Role::Assistant,
                content: "".to_string(),
                tool_calls: vec![ToolCall {
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city":"SF"}),
                }],
                tool_call_id: None,
            },
            Message {
                role: Role::Tool,
                content: "72F".to_string(),
                tool_calls: Vec::new(),
                tool_call_id: Some("call_1".to_string()),
            },
        ];
        let out = frame
            .render_messages(&messages, None, None)
            .expect("multi-turn render succeeds");
        assert!(out.contains("system:be brief;"), "system content visible: {out:?}");
        assert!(out.contains("user:weather?;"), "user content visible: {out:?}");
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
}
