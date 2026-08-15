// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Prompt-frame helpers — architecture-neutral Jinja/ChatFrame rendering.
//!
//! Relocated verbatim from `crates/hipfire-runtime/examples/daemon.rs` (wave 3).

/// Pure helper: derive Jinja `enable_thinking` and `reasoning_effort` from the
/// exact raw request effort and the `max_think` cap. Do not lowercase and do
/// not drop empty strings. Absence/`auto` => undefined; `none`/`off`/`chat`
/// => disabled+undefined; all other exact strings pass unchanged.
pub fn qwen_jinja_reasoning(
    raw_effort: Option<&str>,
    max_think_tokens: usize,
) -> (bool, Option<String>) {
    let is_disable = matches!(raw_effort, Some("none") | Some("off") | Some("chat"));
    let enable = max_think_tokens != 1 && !is_disable;
    if !enable {
        return (false, None);
    }
    match raw_effort {
        None | Some("auto") => (true, None),
        Some(s) => (true, Some(s.to_string())),
    }
}

/// Stateless prompt rendering for a batch lane, reusing the production
/// `ChatFrame`/`JinjaChatFrame` path. Called with `seq_pos=0`, no tools/
/// messages/PFlash, retains `started_in_think` for barrier gating.
/// Plain fallback on Jinja render failure is preserved only when no explicit
/// `reasoning_effort` was supplied; explicit effort render errors are
/// surfaced as `Err` (request validation) instead of hidden by Plain.
pub fn batch_render_prompt_tokens(
    prompt: &str,
    system: Option<&str>,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    chat_template: Option<&String>,
    max_think_tokens: usize,
    messages_history: Option<&[hipfire_runtime::prompt_frame::Message]>,
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
) -> Result<(Vec<u32>, bool), String> {
    debug_assert!(!enable_thinking || max_think_tokens != 1);
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    let try_jinja = jinja_enabled && chat_template.is_some();
    let q_tokens = tokenizer.encode(prompt);
    let system_prompt = system;
    let mut started_in_think = matches!(
        assistant_prefix,
        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
    );
    let new_tokens = if try_jinja {
        let template = chat_template.unwrap();
        let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
            tokenizer,
            template,
            system: system_prompt,
            user: prompt,
            enable_thinking,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort,
        };
        let render_result = if let Some(messages) = messages_history {
            frame.render_messages(messages, None, None)
        } else {
            frame.render()
        };
        match render_result {
            Ok(rendered) => {
                started_in_think = crate::emit::render_tail_opens_think(&rendered);
                tokenizer.encode(&rendered)
            }
            Err(e) => {
                if reasoning_effort.is_some() {
                    return Err(e);
                }
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
            system: system_prompt,
            user: "",
            assistant_prefix,
            raw: false,
        }
        .build_with_user_tokens(&q_tokens)
    };
    Ok((new_tokens, started_in_think))
}
