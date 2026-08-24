// SPDX-License-Identifier: Apache-2.0
//! Calibration corpus construction — the fix for the chat-template defect.
//!
//! Measured 2026-08-16: the shipped calibration corpus contained **zero**
//! chat-template tokens (`grep -c "<|im_start|>" == 0`), while bartowski's
//! v6 recipe renders tool-calling + reasoning conversations through the
//! target model's own chat template with `--parse-special`. We collected
//! excellent statistics over a distribution the model is never deployed in.
//!
//! This module owns:
//! * rendering conversations through a Jinja chat template (`render_conversations`)
//! * joining prose and rendered conversations into a single calibration text (`build`)
//! * the tripwire that identified the defect (`count_special_tokens` + `audit`)

use crate::{ArtifactId, QuantError, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Specification for building a calibration corpus.
#[derive(Debug, Clone)]
pub struct CorpusSpec {
    pub name: String,
    pub prose: Vec<PathBuf>,
    pub conversations: Vec<PathBuf>,
    pub chat_template: Option<PathBuf>,
    pub separator: String,
}

/// The built corpus and its content-addressed identity.
#[derive(Debug, Clone)]
pub struct BuiltCorpus {
    pub text: String,
    pub id: ArtifactId,
    pub n_prose: usize,
    pub n_conversations: usize,
    pub special_token_count: usize,
}

/// Audit result for a corpus text.
#[derive(Debug, Clone)]
pub struct CorpusAudit {
    pub special_token_count: usize,
    pub has_chat_structure: bool,
    pub bytes: usize,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Special-token counting
// ---------------------------------------------------------------------------

/// Count ChatML-style special tokens.
///
/// Counts every `<|...|>` span. This includes the canonical
/// `<|im_start|>`, `<|im_end|>`, and any other `<|...|>` marker such as
/// `<|tool_calls_section_begin|>` etc.
///
/// This is the tripwire: the shipped corpus scores **0** here.
pub fn count_special_tokens(text: &str) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < text.len() {
        let Some(rel_start) = text[i..].find("<|") else {
            break;
        };
        let abs_start = i + rel_start;
        let search_from = abs_start + 2;
        if search_from > text.len() {
            break;
        }
        let Some(rel_end) = text[search_from..].find("|>") else {
            break;
        };
        count += 1;
        i = search_from + rel_end + 2;
    }
    count
}

/// Audit a corpus text for chat structure.
pub fn audit(text: &str) -> CorpusAudit {
    let special_token_count = count_special_tokens(text);
    let has_chat_structure = special_token_count > 0;
    let bytes = text.len();
    let mut warnings = Vec::new();
    if special_token_count == 0 {
        warnings.push(
            "calibration corpus contains no chat-template tokens; activation statistics will not reflect deployment distribution".to_string(),
        );
    }
    CorpusAudit {
        special_token_count,
        has_chat_structure,
        bytes,
        warnings,
    }
}

// ---------------------------------------------------------------------------
// Conversation rendering
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
struct Doc {
    #[serde(default)]
    recipe: Option<String>,
    #[serde(default)]
    n_conversations: Option<usize>,
    conversations: Vec<Conversation>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
struct Conversation {
    #[serde(default)]
    kept_index: Option<serde_json::Value>,
    #[serde(default)]
    scenario_category: Option<String>,
    #[serde(default)]
    source: Option<String>,
    messages: Vec<serde_json::Value>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
}

/// Normalize `tool_calls[].function.arguments` that arrived as a JSON string
/// into a JSON object so Qwen-style templates can index into it as a dict.
///
/// The v6 recipe stores arguments as a dict, but some encoders serialize that
/// dict to a string. Templates such as `{{ message.tool_calls[0].function.arguments.city }}`
/// require the dict form, so we parse string values that look like JSON objects.
fn normalize_arguments_in_messages(messages: &mut [serde_json::Value]) {
    for msg in messages.iter_mut() {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        let Some(tcs) = obj.get_mut("tool_calls") else {
            continue;
        };
        let Some(arr) = tcs.as_array_mut() else {
            continue;
        };
        for tc in arr.iter_mut() {
            let Some(tc_obj) = tc.as_object_mut() else {
                continue;
            };
            let Some(func) = tc_obj.get_mut("function") else {
                continue;
            };
            let Some(func_obj) = func.as_object_mut() else {
                continue;
            };
            let Some(args) = func_obj.get_mut("arguments") else {
                continue;
            };
            if let Some(s) = args.as_str() {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                    if parsed.is_object() {
                        *args = parsed;
                    }
                }
            }
        }
    }
}

/// Render each conversation through `template_src`.
///
/// `json` is the full content of a v6 `v6_conversations.json` file.
/// `template_src` is the Jinja source of the target model's chat template.
///
/// Binds `messages`, `tools`, and `add_generation_prompt = false`.
///
/// Returns `QuantError::Malformed` with the failing conversation index on
/// parse or render failure.
pub fn render_conversations(json: &str, template_src: &str) -> Result<Vec<String>> {
    let doc: Doc = serde_json::from_str(json)
        .map_err(|e| QuantError::Malformed(format!("conversations parse: {e}")))?;

    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );

    let tmpl = env
        .template_from_str(template_src)
        .map_err(|e| QuantError::Malformed(format!("template parse: {e}")))?;

    let mut out = Vec::with_capacity(doc.conversations.len());
    for (idx, mut conv) in doc.conversations.into_iter().enumerate() {
        normalize_arguments_in_messages(&mut conv.messages);

        let messages_val = minijinja::Value::from_serialize(&conv.messages);
        let tools_val = match &conv.tools {
            Some(v) => minijinja::Value::from_serialize(v),
            None => minijinja::Value::from_serialize(&serde_json::Value::Array(Vec::new())),
        };
        let ctx = minijinja::context! {
            messages => messages_val,
            tools => tools_val,
            add_generation_prompt => false,
        };
        let rendered = tmpl
            .render(ctx)
            .map_err(|e| QuantError::Malformed(format!("conversation {idx}: {e}")))?;
        out.push(rendered);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Corpus build
// ---------------------------------------------------------------------------

/// Build a calibration corpus from `spec`.
///
/// * Reads each prose file verbatim (UTF-8).
/// * When both `spec.conversations` is non-empty and `spec.chat_template` is
///   `Some`, reads the template and renders each conversation file through it.
/// * Joins all pieces with `spec.separator`.
/// * Counts special tokens in the final text and content-addresses the output
///   with SHA-256 (no file needs to exist on disk).
pub fn build(spec: &CorpusSpec) -> Result<BuiltCorpus> {
    let mut parts: Vec<String> = Vec::new();
    for p in &spec.prose {
        let content = std::fs::read_to_string(p)?;
        parts.push(content);
    }
    let n_prose = spec.prose.len();

    let mut n_conversations: usize = 0;
    if !spec.conversations.is_empty() {
        if let Some(tmpl_path) = &spec.chat_template {
            let template_src = std::fs::read_to_string(tmpl_path)?;
            for conv_path in &spec.conversations {
                let json = std::fs::read_to_string(conv_path)?;
                let rendered = render_conversations(&json, &template_src)?;
                n_conversations += rendered.len();
                parts.extend(rendered);
            }
        }
    }

    let text = parts.join(&spec.separator);
    let special_token_count = count_special_tokens(&text);

    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let sha256 = format!("{:x}", hasher.finalize());
    let bytes = text.len() as u64;
    let id = ArtifactId {
        path: spec.name.clone(),
        sha256,
        bytes,
    };

    Ok(BuiltCorpus {
        text,
        id,
        n_prose,
        n_conversations,
        special_token_count,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn qwen_chatml_template() -> &'static str {
        "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    }

    fn conversations_json_one_user_one_assistant() -> String {
        serde_json::json!({
            "recipe": "v6",
            "n_conversations": 1,
            "conversations": [
                {
                    "kept_index": 0,
                    "scenario_category": "test",
                    "source": "unit",
                    "messages": [
                        {"role": "user", "content": "hello user content"},
                        {"role": "assistant", "content": "hello assistant content"}
                    ],
                    "tools": []
                }
            ]
        })
        .to_string()
    }

    #[test]
    fn render_chatml_contains_markers_and_both_contents() {
        let json = conversations_json_one_user_one_assistant();
        let tmpl = qwen_chatml_template();
        let rendered = render_conversations(&json, tmpl).expect("render should succeed");
        assert_eq!(rendered.len(), 1);
        let r = &rendered[0];
        assert!(r.contains("<|im_start|>"), "missing <|im_start|> in {r:?}");
        assert!(r.contains("<|im_end|>"), "missing <|im_end|> in {r:?}");
        assert!(r.contains("hello user content"), "user content missing");
        assert!(
            r.contains("hello assistant content"),
            "assistant content missing"
        );
        assert!(
            count_special_tokens(r) >= 4,
            "expected at least 4 markers, got {}",
            count_special_tokens(r)
        );
    }

    #[test]
    fn render_tool_calls_arguments_as_indexable_dict() {
        let tmpl = "{% for message in messages %}{% if message.tool_calls %}{{ message.tool_calls[0].function.arguments.city }}{% endif %}{% endfor %}";
        let json = serde_json::json!({
            "recipe": "v6",
            "n_conversations": 1,
            "conversations": [
                {
                    "kept_index": 0,
                    "scenario_category": "tool",
                    "source": "unit",
                    "messages": [
                        {"role": "user", "content": "weather in Paris?"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": {"city": "Paris"}
                                    }
                                }
                            ]
                        }
                    ],
                    "tools": [
                        {"type": "function", "function": {"name": "get_weather", "description": "get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
                    ]
                }
            ]
        })
        .to_string();
        let rendered = render_conversations(&json, tmpl).expect("tool_calls render failed");
        assert_eq!(rendered.len(), 1);
        assert!(
            rendered[0].contains("Paris"),
            "expected 'Paris' from dict indexing, got {:?}",
            rendered[0]
        );
    }

    #[test]
    fn render_reasoning_content_and_tool_role() {
        let tmpl = qwen_chatml_template();
        let json = serde_json::json!({
            "recipe": "v6",
            "n_conversations": 1,
            "conversations": [
                {
                    "kept_index": 0,
                    "scenario_category": "reasoning",
                    "source": "unit",
                    "messages": [
                        {"role": "user", "content": "explain and call tool"},
                        {
                            "role": "assistant",
                            "content": "the answer",
                            "reasoning_content": "let me think step by step",
                            "tool_calls": [
                                {
                                    "id": "call_42",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": {"query": "rust"}
                                    }
                                }
                            ]
                        },
                        {
                            "role": "tool",
                            "content": "search result: rust is great",
                            "tool_call_id": "call_42"
                        }
                    ],
                    "tools": [
                        {"type": "function", "function": {"name": "search", "description": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
                    ]
                }
            ]
        })
        .to_string();
        let rendered = render_conversations(&json, tmpl).expect("reasoning/tool render failed");
        assert_eq!(rendered.len(), 1);
        assert!(rendered[0].contains("explain and call tool"));
        assert!(rendered[0].contains("the answer"));
        assert!(rendered[0].contains("tool"));
    }

    #[test]
    fn count_raw_prose_zero() {
        let prose = "This is raw prose with no chat markers. Just plain English text.\nAnother paragraph of ordinary prose without any special tokens.";
        assert_eq!(
            count_special_tokens(prose),
            0,
            "raw prose must score 0 — this is the measured property of the shipped corpus"
        );
    }

    #[test]
    fn count_chatml_exact() {
        let chat = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld<|im_end|>";
        assert_eq!(count_special_tokens(chat), 4);
        let with_tool = "<|im_start|>user\nhello<|im_end|><|tool_call_begin|>foo<|tool_call_end|>";
        assert_eq!(count_special_tokens(with_tool), 4);
        let empty = "";
        assert_eq!(count_special_tokens(empty), 0);
        let incomplete = "<|im_start without closing";
        assert_eq!(count_special_tokens(incomplete), 0);
        let generic = "<|a|><|b|><|c|>";
        assert_eq!(count_special_tokens(generic), 3);
    }

    #[test]
    fn audit_zero_warning_fires() {
        let prose = "plain prose no markers";
        let a = audit(prose);
        assert_eq!(a.special_token_count, 0);
        assert!(!a.has_chat_structure);
        assert_eq!(a.bytes, prose.len());
        assert!(
            a.warnings.iter().any(|w| w.contains("calibration corpus contains no chat-template tokens; activation statistics will not reflect deployment distribution")),
            "expected warning on zero-special-token corpus, got {:?}",
            a.warnings
        );
    }

    #[test]
    fn audit_rendered_chat_no_warning() {
        let chat = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld<|im_end|>";
        let a = audit(chat);
        assert_eq!(a.special_token_count, 4);
        assert!(a.has_chat_structure);
        assert!(
            a.warnings.is_empty(),
            "chat corpus must not warn, got {:?}",
            a.warnings
        );
    }

    #[test]
    fn build_over_tempfile_prose_plus_conversations() {
        let dir = tempdir().expect("tempdir");
        let prose1 = dir.path().join("p1.txt");
        let prose2 = dir.path().join("p2.txt");
        fs::write(&prose1, "hello prose one").expect("write p1");
        fs::write(&prose2, "second piece").expect("write p2");

        let conv_path = dir.path().join("conv.json");
        let conv_json = serde_json::json!({
            "recipe": "v6",
            "n_conversations": 2,
            "conversations": [
                {
                    "kept_index": 0,
                    "scenario_category": "test",
                    "source": "unit",
                    "messages": [
                        {"role": "user", "content": "hi from conv 0"},
                        {"role": "assistant", "content": "reply 0"}
                    ],
                    "tools": []
                },
                {
                    "kept_index": 1,
                    "scenario_category": "test",
                    "source": "unit",
                    "messages": [
                        {"role": "user", "content": "hi from conv 1"},
                        {"role": "assistant", "content": "reply 1"}
                    ],
                    "tools": []
                }
            ]
        })
        .to_string();
        fs::write(&conv_path, &conv_json).expect("write conv");

        let tmpl_path = dir.path().join("template.jinja");
        let tmpl = qwen_chatml_template();
        fs::write(&tmpl_path, tmpl).expect("write template");

        let spec = CorpusSpec {
            name: "test-corpus".to_string(),
            prose: vec![prose1.clone(), prose2.clone()],
            conversations: vec![conv_path.clone()],
            chat_template: Some(tmpl_path.clone()),
            separator: "\n---\n".to_string(),
        };

        let built = build(&spec).expect("build");
        assert_eq!(built.n_prose, 2);
        assert_eq!(built.n_conversations, 2);
        assert!(built.text.contains("hello prose one"));
        assert!(built.text.contains("second piece"));
        assert!(built.text.contains("hi from conv 0"));
        assert!(built.text.contains("hi from conv 1"));
        assert!(built.text.contains("\n---\n"), "separator missing");
        let sep_count = built.text.matches("\n---\n").count();
        assert_eq!(
            sep_count, 3,
            "expected 3 separators, got {sep_count} in {:?}",
            built.text
        );
        assert!(built.special_token_count > 0);
        assert_eq!(built.id.bytes, built.text.len() as u64);
        assert_eq!(built.id.path, "test-corpus");

        let built2 = build(&spec).expect("second build");
        assert_eq!(built.id.sha256, built2.id.sha256, "sha256 must be stable");

        fs::write(&prose1, "changed content entirely").expect("overwrite p1");
        let built3 = build(&spec).expect("third build");
        assert_ne!(
            built.id.sha256, built3.id.sha256,
            "sha256 must change when input changes"
        );

        assert_eq!(built3.id.sha256.len(), 64);
        assert!(built3.id.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            built3.id.sha256,
            built3.id.sha256.to_ascii_lowercase(),
            "sha256 must be lowercase"
        );
    }

    #[test]
    fn build_prose_only_no_template() {
        let dir = tempdir().expect("tempdir");
        let prose = dir.path().join("p.txt");
        fs::write(&prose, "just prose").expect("write");
        let conv_path = dir.path().join("conv.json");
        let conv_json = serde_json::json!({
            "recipe": "v6",
            "n_conversations": 1,
            "conversations": [
                {
                    "kept_index": 0,
                    "scenario_category": "test",
                    "source": "unit",
                    "messages": [{"role":"user","content":"hello"}],
                    "tools": []
                }
            ]
        })
        .to_string();
        fs::write(&conv_path, conv_json).expect("write conv");
        let spec = CorpusSpec {
            name: "prose-only".to_string(),
            prose: vec![prose.clone()],
            conversations: vec![conv_path],
            chat_template: None,
            separator: "\n".to_string(),
        };
        let built = build(&spec).expect("build prose-only");
        assert_eq!(built.n_prose, 1);
        assert_eq!(built.n_conversations, 0);
        assert!(built.text.contains("just prose"));
        assert_eq!(built.special_token_count, 0);
    }

    #[test]
    fn render_malformed_json_errors() {
        let bad = "{ not valid json at all }";
        let tmpl = qwen_chatml_template();
        let err = render_conversations(bad, tmpl).expect_err("should fail on malformed json");
        match err {
            QuantError::Malformed(msg) => {
                assert!(!msg.is_empty(), "malformed message empty");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn render_broken_template_errors() {
        let json = conversations_json_one_user_one_assistant();
        let broken = "{% for %}";
        let err = render_conversations(&json, broken).expect_err("should fail on broken template");
        match err {
            QuantError::Malformed(msg) => {
                assert!(!msg.is_empty());
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn render_invalid_conversation_structure_malformed() {
        let raising_tmpl = "{% for m in messages %}{% if m.content == 'trigger' %}{{ raise_exception('bad conversation') }}{% endif %}{{ m.content }}{% endfor %}";
        let json = serde_json::json!({
            "recipe": "v6",
            "n_conversations": 1,
            "conversations": [
                {
                    "kept_index": 0,
                    "scenario_category": "test",
                    "source": "unit",
                    "messages": [{"role":"user","content":"trigger"}],
                    "tools": []
                }
            ]
        })
        .to_string();
        let err = render_conversations(&json, raising_tmpl)
            .expect_err("should fail with raise_exception");
        match err {
            QuantError::Malformed(msg) => {
                assert!(
                    msg.contains("conversation 0"),
                    "error must name conversation index, got {msg:?}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }
}
