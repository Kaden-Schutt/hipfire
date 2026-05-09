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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
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
#[derive(Clone, Debug, serde::Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Set on Tool-role messages to identify which assistant tool_call
    /// this is responding to. Qwen3.5/3.6 templates currently ignore
    /// this field; OpenAI-spec clients and some other templates require
    /// it. Skipped from the serialized JSON when None so templates that
    /// `is defined` against it don't see a misleading null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// One assistant-emitted tool call, attached to an assistant `Message`.
/// `arguments` is a free-form JSON value (typically an object). Templates
/// that render in XML format (Qwen3.5/3.6's `<function=NAME><parameter=ARG>`
/// shape) walk this with `arguments | items` under pycompat.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
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
                    {{"id": 3, "content": "</think>", "special": true}}
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
