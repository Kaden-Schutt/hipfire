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

use std::path::Path;

/// Resolved model chat template plus the source selected by the load-time
/// precedence chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedChatTemplate {
    pub template: String,
    pub source: ChatTemplateSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateSource {
    EnvFile(String),
    PerModelFile(String),
    Embedded,
}

/// Resolve the chat template to use for a loaded model.
///
/// Precedence:
/// 1. `HIPFIRE_CHAT_TEMPLATE_FILE`, when set and readable.
/// 2. Per-model file at `$HOME/.hipfire/templates/<model-basename>.j2`.
/// 3. Embedded model template.
///
/// Read failures on override files are non-fatal and fall through to the next
/// source, matching the daemon's historical behavior.
pub fn resolve_chat_template(
    model_path: &str,
    embedded_template: Option<String>,
) -> Option<ResolvedChatTemplate> {
    if let Ok(env_path) = std::env::var("HIPFIRE_CHAT_TEMPLATE_FILE") {
        if !env_path.is_empty() {
            match std::fs::read_to_string(&env_path) {
                Ok(template) => {
                    return Some(ResolvedChatTemplate {
                        template,
                        source: ChatTemplateSource::EnvFile(env_path),
                    });
                }
                Err(e) => {
                    eprintln!(
                        "[chat_template] HIPFIRE_CHAT_TEMPLATE_FILE={env_path} failed to read ({e}); falling through"
                    );
                }
            }
        }
    }

    if let Some(home) = std::env::var_os("HOME") {
        let basename = Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !basename.is_empty() {
            let per_model = Path::new(&home)
                .join(".hipfire")
                .join("templates")
                .join(format!("{basename}.j2"));
            if per_model.is_file() {
                match std::fs::read_to_string(&per_model) {
                    Ok(template) => {
                        return Some(ResolvedChatTemplate {
                            template,
                            source: ChatTemplateSource::PerModelFile(
                                per_model.display().to_string(),
                            ),
                        });
                    }
                    Err(e) => eprintln!(
                        "[chat_template] per-model file {} failed to read ({e}); falling through",
                        per_model.display()
                    ),
                }
            }
        }
    }

    embedded_template.map(|template| ResolvedChatTemplate {
        template,
        source: ChatTemplateSource::Embedded,
    })
}

pub fn log_resolved_chat_template_source(source: &ChatTemplateSource) {
    match source {
        ChatTemplateSource::EnvFile(path) => {
            eprintln!("[chat_template] using HIPFIRE_CHAT_TEMPLATE_FILE={path}");
        }
        ChatTemplateSource::PerModelFile(path) => {
            eprintln!("[chat_template] using per-model override {path}");
        }
        ChatTemplateSource::Embedded => {
            eprintln!("[chat_template] using HFQ-embedded tokenizer_config.chat_template");
        }
    }
}

/// Minimal tokenizer surface needed by prompt framing.
///
/// Keeping this trait in `hipfire-prompt` lets the prompt crate own chat
/// rendering without depending back on `hipfire-runtime`.
pub trait PromptTokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn special_token_id(&self, content: &str) -> Option<u32>;
    fn special_tokens(&self) -> &[(String, u32)] {
        &[]
    }
    fn bos_token_text(&self) -> String;
}

/// Chooses what goes after the assistant role-and-newline opener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl AssistantPrefix {
    pub fn from_label(label: Option<&str>) -> Self {
        match label.unwrap_or("plain") {
            "open_think" => Self::OpenThink,
            "closed_think" => Self::ClosedThink,
            _ => Self::Plain,
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::OpenThink => "open_think",
            Self::ClosedThink => "closed_think",
        }
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
/// Not `#[derive(Debug)]` because tokenizer implementations don't have to
/// implement
/// `Debug`. Callers that need a printable struct should format the
/// non-tokenizer fields manually.
#[derive(Clone)]
pub struct ChatFrame<'a> {
    pub tokenizer: &'a dyn PromptTokenizer,
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
    tokenizer: &'a dyn PromptTokenizer,
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
    fn for_tokenizer(t: &'a dyn PromptTokenizer) -> Self {
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
    pub tokenizer: &'a dyn PromptTokenizer,
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

/// Stable canonical JSON representation for prompt-identity fingerprints.
///
/// Objects emit keys in lexical order recursively; arrays preserve order.
/// This is intentionally narrower than a full JSON canonicalization standard:
/// it only needs to make semantically identical tool-call argument objects hash
/// identically when source-side insertion order differs.
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical_json(value, &mut out);
    out
}

fn write_canonical_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string()))
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (idx, item) in arr.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            out.push('{');
            for (idx, key) in keys.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(*key).unwrap_or_else(|_| "\"\"".to_string()));
                out.push(':');
                write_canonical_json(&map[*key], out);
            }
            out.push('}');
        }
    }
}

/// Stable fingerprint for assistant-turn prompt-history identity.
///
/// Pure text turns hash trimmed content. Tool-call turns hash only the
/// canonicalized tool calls, matching OpenAI-compatible clients that often send
/// `content: null` or an empty string for assistant messages with tool calls.
pub fn assistant_turn_fingerprint(content: &str, tool_calls: &[ToolCall]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h = DefaultHasher::new();
    "assistant".hash(&mut h);
    if tool_calls.is_empty() {
        content.trim().hash(&mut h);
    }
    for tool_call in tool_calls {
        tool_call.name.hash(&mut h);
        canonical_json(&tool_call.arguments).hash(&mut h);
    }
    h.finish()
}

pub fn openai_chat_role_to_prompt_role(role: &str) -> Option<Role> {
    match role {
        "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

pub fn openai_chat_content_to_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

pub fn openai_chat_message_to_prompt_message(
    role: &str,
    content: Option<&serde_json::Value>,
) -> Option<Message> {
    Some(Message {
        role: openai_chat_role_to_prompt_role(role)?,
        content: openai_chat_content_to_text(content),
        tool_calls: Vec::new(),
        tool_call_id: None,
    })
}

pub fn openai_chat_messages_to_prompt_messages<'a, I>(messages: I) -> Vec<Message>
where
    I: IntoIterator<Item = (&'a str, Option<&'a serde_json::Value>)>,
{
    messages
        .into_iter()
        .filter_map(|(role, content)| openai_chat_message_to_prompt_message(role, content))
        .collect()
}

pub fn openai_chat_last_user_prompt<'a, I>(messages: I) -> String
where
    I: IntoIterator<Item = (&'a str, Option<&'a serde_json::Value>)>,
{
    let mut last = None;
    for (role, content) in messages {
        if role == "user" {
            last = Some(openai_chat_content_to_text(content));
        }
    }
    last.unwrap_or_default()
}

/// Remove HF `{% generation %}` / `{% endgeneration %}` tags (with optional
/// whitespace-control dashes and the line they sit on) from a chat template.
/// These mark the assistant-token span for training-data masking and emit
/// nothing, so dropping the markers keeps inference rendering byte-identical —
/// while letting minijinja (which has no `generation` block tag) parse the
/// template instead of erroring. No-op for templates without them.
fn strip_generation_tags(template: &str) -> String {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE
        .get_or_init(|| regex::Regex::new(r"[ \t]*\{%-?\s*(?:end)?generation\s*-?%\}\n?").unwrap());
    re.replace_all(template, "").into_owned()
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
pub fn hf_tojson(
    value: minijinja::Value,
    kwargs: minijinja::value::Kwargs,
) -> Result<String, minijinja::Error> {
    use serde::Serialize;
    let _ = kwargs.get::<Option<bool>>("ensure_ascii");
    let _ = kwargs.get::<Option<i64>>("indent");
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

        // Chainable-undefined (NOT Strict): chat templates are authored for
        // Jinja2's lenient semantics and routinely PROBE optional fields —
        // `system_message.current_date`, `.current_location`, `tools`,
        // `documents`, etc. Under Strict, probing a key a caller didn't set
        // raises, and the whole render fails → silent Plain fallback. That's
        // exactly what broke MiniMax-M2 under an agent (Hermes) that sends a
        // system message without `current_date`: `{% if system_message and
        // system_message.current_date %}` errored at template line 37. Chainable
        // matches Jinja2 (missing keys → falsy/undefined, chained access on
        // undefined stays undefined) while still surfacing hard render errors
        // (raise_exception, syntax). The required context vars (messages, tools,
        // add_generation_prompt, bos_token) are always provided below, so the
        // PR #175 "missing required var" concern doesn't regress.
        let mut env = Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Chainable);
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
        // (`", "` / `": "`) the model trained on. Some HF templates pass kwargs
        // such as `ensure_ascii=False`; `hf_tojson` accepts and ignores them.
        env.add_filter("tojson", hf_tojson);

        // Strip HF `{% generation %}` / `{% endgeneration %}` training-mask
        // tags (and their whitespace-control `{%- … -%}` variants). minijinja
        // has no `generation` block tag, so a template that uses them (e.g.
        // LFM2.5) fails to parse and the caller silently falls back to Plain
        // framing. These tags only delimit the assistant-token span for
        // training-data masking — they emit nothing — so removing the markers
        // (including their own line) leaves the rendered output byte-identical
        // for inference. No-op for templates that don't use them (Qwen/MiniMax).
        let sanitized = strip_generation_tags(self.template);
        env.add_template_owned("chat", sanitized)
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
            None => self.tokenizer.bos_token_text(),
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
        tmpl.render(ctx)
            .map_err(|e| format!("template render: {e}"))
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
fn pick_splice_sentinel(tok: &dyn PromptTokenizer) -> Option<String> {
    // Tokens the chat templates emit structurally — never use these as a
    // sentinel (their post-render count wouldn't equal the spliced-turn count).
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
    ];
    let atomic = |s: &str| -> bool {
        tok.special_token_id(s)
            .map_or(false, |id| tok.encode(s) == vec![id])
    };
    // First pass: obviously-reserved scratch tokens.
    for (s, _id) in tok.special_tokens() {
        if STRUCTURAL.contains(&s.as_str()) {
            continue;
        }
        let ls = s.to_ascii_lowercase();
        if (ls.contains("reserved") || ls.contains("unused") || ls.contains("pad")) && atomic(s) {
            return Some(s.clone());
        }
    }
    // Second pass: any non-structural special token that round-trips atomically.
    for (s, _id) in tok.special_tokens() {
        if STRUCTURAL.contains(&s.as_str()) {
            continue;
        }
        if atomic(s) {
            return Some(s.clone());
        }
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
    let mut out: Vec<u32> =
        Vec::with_capacity(toks.len() + cached.iter().map(|c| c.len()).sum::<usize>());
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
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvSnapshot {
        hipfire_chat_template_file: Option<std::ffi::OsString>,
        home: Option<std::ffi::OsString>,
    }

    impl EnvSnapshot {
        fn capture() -> Self {
            Self {
                hipfire_chat_template_file: std::env::var_os("HIPFIRE_CHAT_TEMPLATE_FILE"),
                home: std::env::var_os("HOME"),
            }
        }

        fn restore(self) {
            unsafe {
                match self.hipfire_chat_template_file {
                    Some(value) => std::env::set_var("HIPFIRE_CHAT_TEMPLATE_FILE", value),
                    None => std::env::remove_var("HIPFIRE_CHAT_TEMPLATE_FILE"),
                }
                match self.home {
                    Some(value) => std::env::set_var("HOME", value),
                    None => std::env::remove_var("HOME"),
                }
            }
        }
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "hipfire-prompt-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    struct TestTokenizer {
        include_think: bool,
        special_tokens: Vec<(String, u32)>,
    }

    impl PromptTokenizer for TestTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            let specials = [
                ("<|im_start|>", 0),
                ("<|im_end|>", 1),
                ("<think>", 2),
                ("</think>", 3),
                ("system", 4),
                ("user", 5),
                ("assistant", 6),
                ("\n", 7),
                ("<|reserved_0|>", 8),
            ];
            let mut out = Vec::new();
            let mut i = 0usize;
            while i < text.len() {
                let rest = &text[i..];
                if let Some((s, id)) = specials
                    .iter()
                    .filter(|(s, _)| *s != "<think>" && *s != "</think>" || self.include_think)
                    .find(|(s, _)| rest.starts_with(*s))
                {
                    out.push(*id);
                    i += s.len();
                    continue;
                }
                let ch = rest.chars().next().expect("non-empty rest");
                let mut buf = [0u8; 4];
                for b in ch.encode_utf8(&mut buf).as_bytes() {
                    out.push(100 + u32::from(*b));
                }
                i += ch.len_utf8();
            }
            out
        }

        fn special_token_id(&self, content: &str) -> Option<u32> {
            match content {
                "<think>" if self.include_think => Some(2),
                "</think>" if self.include_think => Some(3),
                "<|reserved_0|>" => Some(8),
                _ => None,
            }
        }

        fn special_tokens(&self) -> &[(String, u32)] {
            &self.special_tokens
        }

        fn bos_token_text(&self) -> String {
            "<bos>".to_string()
        }
    }

    fn make_tokenizer() -> TestTokenizer {
        TestTokenizer {
            include_think: true,
            special_tokens: vec![("<|reserved_0|>".to_string(), 8)],
        }
    }

    fn test_tokenizer_no_think() -> TestTokenizer {
        TestTokenizer {
            include_think: false,
            special_tokens: vec![("<|reserved_0|>".to_string(), 8)],
        }
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
    fn strip_generation_tags_removes_markers_keeps_body() {
        // HF training-mask tags (and their whitespace-control variants) are
        // dropped; the body between them and everything else is untouched.
        let tpl = "a\n{%- generation -%}\nBODY\n{%- endgeneration -%}\nb\n{% generation %}X{% endgeneration %}c";
        let got = strip_generation_tags(tpl);
        assert!(!got.contains("generation"), "tags not stripped: {got:?}");
        assert!(got.contains("BODY"), "inner body dropped: {got:?}");
        assert!(
            got.contains('X') && got.contains('c'),
            "non-dashed body dropped: {got:?}"
        );
        // A template with no generation tags is returned unchanged.
        let plain = "{{ bos_token }}{%- for m in messages -%}{{ m.role }}{%- endfor -%}";
        assert_eq!(strip_generation_tags(plain), plain);
    }

    #[test]
    fn jinja_render_tolerates_generation_tags() {
        // A minimal template that uses `{% generation %}` around the assistant
        // body — minijinja has no such tag, so without the strip this fails to
        // parse. With the strip it renders the assistant body normally.
        let t = make_tokenizer();
        let template = "{%- for message in messages -%}\
            {{- '<|im_start|>' + message.role + '\\n' -}}\
            {%- if message.role == 'assistant' -%}{%- generation -%}{{- message.content -}}{%- endgeneration -%}\
            {%- else -%}{{- message.content -}}{%- endif -%}\
            {{- '<|im_end|>\\n' -}}\
            {%- endfor -%}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "hi",
            enable_thinking: false,
            bos_token: Some("<|im_start|>"),
        };
        let msgs = vec![
            Message {
                role: Role::User,
                content: "hi".into(),
                tool_calls: vec![],
                tool_call_id: None,
            },
            Message {
                role: Role::Assistant,
                content: "yo".into(),
                tool_calls: vec![],
                tool_call_id: None,
            },
        ];
        let rendered = frame
            .render_messages(&msgs, None, None)
            .expect("template with generation tags must render after strip");
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n"
        );
    }

    #[test]
    fn jinja_tojson_accepts_ensure_ascii_kwarg() {
        // MiniMax-M2's template calls `tojson(ensure_ascii=False)`; minijinja's
        // builtin rejects the kwarg → render fails → silent Plain fallback (the
        // model then never emits its native tool format). Our override must
        // accept the kwarg and emit raw JSON.
        let t = make_tokenizer();
        let template = "{%- for m in messages -%}{{- m.content -}}{%- endfor -%}{{- tools | tojson(ensure_ascii=False) -}}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "hi",
            enable_thinking: false,
            bos_token: Some(""),
        };
        let msgs = vec![Message {
            role: Role::User,
            content: "hi".into(),
            tool_calls: vec![],
            tool_call_id: None,
        }];
        let tools =
            vec![serde_json::json!({"type": "function", "function": {"name": "get_weather"}})];
        let rendered = frame
            .render_messages(&msgs, Some(&tools), None)
            .expect("tojson(ensure_ascii=False) must render, not fall back");
        assert!(
            rendered.contains("\"get_weather\""),
            "tool json missing: {rendered}"
        );
    }

    #[test]
    fn jinja_chainable_tolerates_missing_optional_field() {
        // Chat templates probe optional message fields (e.g. MiniMax-M2's
        // `system_message.current_date`). Under Strict-undefined that raised and
        // forced a Plain fallback (broke MiniMax under Hermes); Chainable treats a
        // missing key as falsy like Jinja2, so the probe is a no-op.
        let t = make_tokenizer();
        let template = "{%- for m in messages -%}\
            {%- if m.current_date -%}D:{{ m.current_date }}{%- endif -%}\
            {{- m.content -}}\
            {%- endfor -%}";
        let frame = JinjaChatFrame {
            tokenizer: &t,
            template,
            system: None,
            user: "hi",
            enable_thinking: false,
            bos_token: Some(""),
        };
        let msgs = vec![
            Message {
                role: Role::System,
                content: "S".into(),
                tool_calls: vec![],
                tool_call_id: None,
            },
            Message {
                role: Role::User,
                content: "U".into(),
                tool_calls: vec![],
                tool_call_id: None,
            },
        ];
        let rendered = frame
            .render_messages(&msgs, None, None)
            .expect("probing a missing optional message field must not raise");
        assert_eq!(rendered, "SU");
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
    fn openai_chat_helpers_build_prompt_messages_and_last_user_fallback() {
        let messages = vec![
            ("system", Some(serde_json::json!("be brief"))),
            ("user", Some(serde_json::json!("first"))),
            ("assistant", Some(serde_json::json!("ok"))),
            ("ignored", Some(serde_json::json!("drop me"))),
            (
                "user",
                Some(serde_json::json!({"type": "text", "text": "second"})),
            ),
        ];

        let prompt_messages = openai_chat_messages_to_prompt_messages(
            messages
                .iter()
                .map(|(role, content)| (*role, content.as_ref())),
        );
        assert_eq!(prompt_messages.len(), 4);
        assert_eq!(prompt_messages[0].role, Role::System);
        assert_eq!(prompt_messages[0].content, "be brief");
        assert_eq!(prompt_messages[2].role, Role::Assistant);
        assert_eq!(prompt_messages[3].role, Role::User);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&prompt_messages[3].content)
                .expect("structured prompt message json"),
            serde_json::json!({"type": "text", "text": "second"})
        );

        let fallback = openai_chat_last_user_prompt(
            messages
                .iter()
                .map(|(role, content)| (*role, content.as_ref())),
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&fallback)
                .expect("structured fallback prompt json"),
            serde_json::json!({"type": "text", "text": "second"})
        );
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
    fn assistant_prefix_labels_match_daemon_wire_policy() {
        assert_eq!(AssistantPrefix::from_label(None), AssistantPrefix::Plain);
        assert_eq!(
            AssistantPrefix::from_label(Some("plain")),
            AssistantPrefix::Plain
        );
        assert_eq!(
            AssistantPrefix::from_label(Some("open_think")),
            AssistantPrefix::OpenThink
        );
        assert_eq!(
            AssistantPrefix::from_label(Some("closed_think")),
            AssistantPrefix::ClosedThink
        );
        assert_eq!(
            AssistantPrefix::from_label(Some("unknown")),
            AssistantPrefix::Plain
        );
        assert_eq!(AssistantPrefix::ClosedThink.as_label(), "closed_think");
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
    fn canonical_json_sorts_object_keys_recursively() {
        let left = serde_json::json!({
            "b": [{"z": 1, "a": true}],
            "a": {"d": null, "c": "x"},
        });
        let right = serde_json::json!({
            "a": {"c": "x", "d": null},
            "b": [{"a": true, "z": 1}],
        });

        assert_eq!(canonical_json(&left), canonical_json(&right));
        assert_eq!(
            canonical_json(&left),
            r#"{"a":{"c":"x","d":null},"b":[{"a":true,"z":1}]}"#
        );
    }

    #[test]
    fn chat_template_resolution_prefers_env_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snapshot = EnvSnapshot::capture();
        let root = temp_dir("env-template");
        let env_file = root.join("env.j2");
        let home = root.join("home");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(&env_file, "env-template").unwrap();
        unsafe {
            std::env::set_var("HIPFIRE_CHAT_TEMPLATE_FILE", &env_file);
            std::env::set_var("HOME", &home);
        }

        let resolved =
            resolve_chat_template("/models/qwen3.5-9b-mq4.hfq", Some("embedded".to_string()))
                .expect("template");

        assert_eq!(resolved.template, "env-template");
        assert_eq!(
            resolved.source,
            ChatTemplateSource::EnvFile(env_file.display().to_string())
        );
        snapshot.restore();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn chat_template_resolution_falls_back_to_per_model_then_embedded() {
        let _guard = ENV_LOCK.lock().unwrap();
        let snapshot = EnvSnapshot::capture();
        let root = temp_dir("per-model-template");
        let home = root.join("home");
        let templates = home.join(".hipfire").join("templates");
        std::fs::create_dir_all(&templates).unwrap();
        let per_model = templates.join("qwen3.5-9b-mq4.hfq.j2");
        std::fs::write(&per_model, "per-model-template").unwrap();
        unsafe {
            std::env::set_var("HIPFIRE_CHAT_TEMPLATE_FILE", root.join("missing.j2"));
            std::env::set_var("HOME", &home);
        }

        let resolved =
            resolve_chat_template("/models/qwen3.5-9b-mq4.hfq", Some("embedded".to_string()))
                .expect("template");

        assert_eq!(resolved.template, "per-model-template");
        assert_eq!(
            resolved.source,
            ChatTemplateSource::PerModelFile(per_model.display().to_string())
        );

        std::fs::remove_file(&per_model).unwrap();
        let resolved =
            resolve_chat_template("/models/qwen3.5-9b-mq4.hfq", Some("embedded".to_string()))
                .expect("embedded template");
        assert_eq!(resolved.template, "embedded");
        assert_eq!(resolved.source, ChatTemplateSource::Embedded);
        snapshot.restore();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn assistant_turn_fingerprint_matches_prompt_history_identity_policy() {
        let text_a = assistant_turn_fingerprint(" answer \n", &[]);
        let text_b = assistant_turn_fingerprint("answer", &[]);
        let text_c = assistant_turn_fingerprint("different", &[]);
        assert_eq!(text_a, text_b);
        assert_ne!(text_a, text_c);

        let calls_a = vec![ToolCall {
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "/tmp/a", "opts": {"tail": 5, "raw": false}}),
        }];
        let calls_b = vec![ToolCall {
            name: "read_file".to_string(),
            arguments: serde_json::json!({"opts": {"raw": false, "tail": 5}, "path": "/tmp/a"}),
        }];
        assert_eq!(
            assistant_turn_fingerprint("Let me inspect that.", &calls_a),
            assistant_turn_fingerprint("", &calls_b)
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
        };

        // Turn 1: daemon prefills R1 (prompt, ends with the primed `<think>\n`)
        // then generates `reason</think>ok`.
        let u1 = Message {
            role: Role::User,
            content: "hi".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
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
            tool_calls: vec![],
            tool_call_id: None,
        };
        let u2 = Message {
            role: Role::User,
            content: "again".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        };
        let messages_t2 = vec![u1.clone(), a1, u2];

        let rendered_t2 = build_cached_history_jinja(&frame, &messages_t2, None, |m| {
            if matches!(m.role, Role::Assistant) {
                Some(asst_slot.clone())
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
