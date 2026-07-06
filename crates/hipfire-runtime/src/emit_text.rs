// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Model-independent parsing of *emitted* assistant text — think-span state and
//! Hermes/ChatML `<tool_call>` extraction.
//!
//! These helpers used to live in the `daemon` example, where the per-arch
//! `SpecEmit` impls call them during the decode loop. They are pure
//! string-processing over the ChatML conventions (`<think>…</think>`,
//! `<tool_call>{json}</tool_call>`) that every Qwen/LLaMA-family arch shares, so
//! they belong in the runtime where the arch crates' emitters can reach them —
//! an example cannot be imported by the arch crates that depend on this crate.

/// Whether the model is currently inside an open `<think>` span, from the
/// generated text so far plus whether thinking was opened via the assistant
/// prefix. `assistant_prefix=open_think` injects the `<think>` opener into the
/// PROMPT, so it never shows up in the generated stream: without
/// `started_in_think` the `(None, None)` case reads as "not thinking", the
/// `max_think_tokens` force-close never fires, and a model that out-thinks its
/// budget runs away to `max_tokens`. Centralises the scan used by every
/// force-close / budget-alert site so they stay consistent.
pub fn currently_in_think(raw_str: &str, started_in_think: bool) -> bool {
    match (raw_str.rfind("<think>"), raw_str.rfind("</think>")) {
        (Some(o), Some(c)) => o > c, // both present: in-think iff opener is latest
        (Some(_), None) => true,     // generated opener, not yet closed
        (None, Some(_)) => false,    // closed (e.g. a prompt-injected opener) → answering
        (None, None) => started_in_think, // no tags generated yet → trust the prompt prefix
    }
}

/// Extract Hermes/ChatML `<tool_call>{json}</tool_call>` calls from generated
/// text. Tolerant of truncation (unclosed `<tool_call>`), ChatML special-token
/// leakage, and stacked/nested openers (MQ4 #111 attractor shapes). Mirrors the
/// CLI's `parseToolCalls` / `parseOneToolCall`.
pub fn extract_tool_calls_from_text(s: &str) -> Vec<crate::prompt_frame::ToolCall> {
    let mut out: Vec<crate::prompt_frame::ToolCall> = Vec::new();
    let mut search_pos = 0;
    while let Some(open_rel) = s[search_pos..].find("<tool_call>") {
        let body_start = search_pos + open_rel + "<tool_call>".len();
        // Unclosed `<tool_call>` — model hit max_tokens or truncated;
        // treat the rest of the string as the body. CLI parser does
        // the same via the `<tool_call>\s*(.*)` regex branch. Without
        // this, a truncated emit stores `tool_calls=0`, the CLI on the
        // wire parses `tool_calls=1`, and the asst-turn fingerprint
        // mismatches on echo-back → cache miss.
        let (body_end, advance) = match s[body_start..].find("</tool_call>") {
            Some(i) => (body_start + i, body_start + i + "</tool_call>".len()),
            None => (s.len(), s.len()),
        };
        let body_raw = &s[body_start..body_end];
        // Sanitize ChatML special-token leakage (mirrors CLI's
        // parseOneToolCall: cli/index.ts:2273-2278). qwen3.6:27b
        // occasionally glues `<|im_start|>` / `<|im_end|>` / etc. into
        // the JSON body when the tokenizer's special-token boundary
        // catches the JSON key opener.
        let body_clean: String = body_raw
            .replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|im_sep|>", "");
        // Strip nested `<tool_call>` openers (MQ4 attractor: model
        // stacks 1-2 nested openers before the JSON body lands).
        let mut body_stripped = body_clean.trim_start();
        while body_stripped.starts_with("<tool_call>") {
            body_stripped = body_stripped["<tool_call>".len()..].trim_start();
        }
        let body = body_stripped.trim();
        if !body.is_empty() {
            // Form 1: strict JSON parse
            let mut parsed: Option<(String, serde_json::Value)> = None;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(body) {
                let name = val
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !name.is_empty() {
                    let arguments = val
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                    parsed = Some((name, arguments));
                }
            }
            // Form 4 (regex fallback): when JSON parse fails, recover
            // name + arguments via a relaxed key-delimiter pattern.
            // Mirrors cli/index.ts:2287-2295.
            if parsed.is_none() {
                if let Some(name) = extract_tool_call_name_fallback(body) {
                    if let Some(arguments) = extract_tool_call_arguments_fallback(body) {
                        // Recovered a complete, strict-valid args object.
                        parsed = Some((name, arguments));
                    } else if tool_call_args_object_complete(body) {
                        // The args object is present and brace-balanced but not
                        // strict JSON (trailing comma, unquoted key, …) — a
                        // model formatting glitch, not a truncation. Preserve
                        // the call by name with empty args (legacy behavior).
                        parsed = Some((name, serde_json::Value::Object(Default::default())));
                    }
                    // else: NO balanced args object — the call was cut off
                    // mid-value by `max_tokens` or a grammar force-close.
                    // Dropping it (rather than fabricating empty `{}`) keeps a
                    // broken call from being delivered as executable: the
                    // client would otherwise invoke e.g. `write({})` and fail
                    // schema validation (the write-tool empty-args incident).
                    // The truncated emission instead surfaces as content +
                    // finish_reason so the client retries.
                }
            }
            if let Some((name, arguments)) = parsed {
                out.push(crate::prompt_frame::ToolCall { name, arguments });
            }
        }
        search_pos = advance;
        if advance == s.len() {
            break;
        }
    }
    out
}

/// Relaxed name extraction: matches `"name": "X"` (or `'name': 'X'`,
/// or with an opening quote replaced by a special-token boundary —
/// `name": "X"`). Mirrors CLI Form 4 regex in `parseOneToolCall`.
///
/// Walks the string looking for `name` substring occurrences. For each,
/// validates the byte before it is a JSON key-position char ({ , " ' or
/// whitespace) — false matches like `firstname` get skipped and the
/// walk continues. First valid `name: "value"` match wins.
pub fn extract_tool_call_name_fallback(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut search_from = 0usize;
    while let Some(idx_rel) = s[search_from..].find("name") {
        let abs = search_from + idx_rel;
        // Advance search anchor past this "name" regardless of outcome
        // so the next iteration looks for the next occurrence.
        let after_name = abs + "name".len();
        search_from = after_name;
        // Key-position check: byte before `name` must be a JSON key
        // boundary char. Skips false matches like the `name` substring
        // inside `firstname` / `lastname` / etc.
        let pre = if abs == 0 { b' ' } else { bytes[abs - 1] };
        let pre_ok = matches!(pre, b'{' | b',' | b' ' | b'\n' | b'\t' | b'"' | b'\'');
        if !pre_ok {
            continue;
        }
        let mut j = after_name;
        // Skip optional closing quote on the key.
        if j < bytes.len() && (bytes[j] == b'"' || bytes[j] == b'\'') {
            j += 1;
        }
        // Skip whitespace before `:`.
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        // Require `:`.
        if j >= bytes.len() || bytes[j] != b':' {
            continue;
        }
        j += 1;
        // Skip whitespace after `:`.
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        // Require opening quote for the value.
        if j >= bytes.len() || (bytes[j] != b'"' && bytes[j] != b'\'') {
            continue;
        }
        let q = bytes[j];
        j += 1;
        let val_start = j;
        while j < bytes.len() && bytes[j] != q {
            j += 1;
        }
        if j >= bytes.len() {
            continue;
        }
        let name = &s[val_start..j];
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
        {
            continue;
        }
        return Some(name.to_string());
    }
    None
}

/// Best-effort `arguments` extraction: find the first balanced `{...}`
/// after the `arguments`-style key, parse it as JSON. Returns None if
/// no balanced object is found or the object isn't valid JSON.
pub fn extract_tool_call_arguments_fallback(s: &str) -> Option<serde_json::Value> {
    let key_idx = s.find("arguments")?;
    let tail = &s[key_idx + "arguments".len()..];
    // Skip key terminator + colon + whitespace
    let mut chars = tail.char_indices().peekable();
    while let Some(&(_, c)) = chars.peek() {
        if c == '"' || c == '\'' || c == ':' || c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
    let obj_rel_start = chars.next().map(|(i, _)| i)?;
    let obj_start = key_idx + "arguments".len() + obj_rel_start;
    let after_key = &s[obj_start..];
    // Need to find the opening brace
    let brace_off = after_key.find('{')?;
    let abs_start = obj_start + brace_off;
    // Walk to find the matching close brace
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    let mut k = abs_start;
    while k < bytes.len() {
        let ch = bytes[k];
        if in_str {
            if escape {
                escape = false;
            } else if ch == b'\\' {
                escape = true;
            } else if ch == b'"' {
                in_str = false;
            }
        } else if ch == b'"' {
            in_str = true;
        } else if ch == b'{' {
            depth += 1;
        } else if ch == b'}' {
            depth -= 1;
            if depth == 0 {
                let slice = &s[abs_start..=k];
                return serde_json::from_str(slice).ok();
            }
        }
        k += 1;
    }
    None
}

/// True iff a brace-balanced `{...}` object exists after the `arguments`
/// key — i.e. the args object is COMPLETE (not truncated), regardless of
/// whether it is strict-valid JSON. Distinguishes a model formatting glitch
/// (trailing comma / unquoted key — keep the call) from a generation cut off
/// mid-args (drop the call). Mirrors the brace walk in
/// [`extract_tool_call_arguments_fallback`] but stops at the matching close
/// brace without requiring valid JSON.
pub fn tool_call_args_object_complete(s: &str) -> bool {
    let key_idx = match s.find("arguments") {
        Some(i) => i,
        None => return false,
    };
    let after_key = &s[key_idx + "arguments".len()..];
    let brace_off = match after_key.find('{') {
        Some(i) => i,
        None => return false,
    };
    let abs_start = key_idx + "arguments".len() + brace_off;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    let mut k = abs_start;
    while k < bytes.len() {
        let ch = bytes[k];
        if in_str {
            if escape {
                escape = false;
            } else if ch == b'\\' {
                escape = true;
            } else if ch == b'"' {
                in_str = false;
            }
        } else if ch == b'"' {
            in_str = true;
        } else if ch == b'{' {
            depth += 1;
        } else if ch == b'}' {
            depth -= 1;
            if depth == 0 {
                return true;
            }
        }
        k += 1;
    }
    false
}
