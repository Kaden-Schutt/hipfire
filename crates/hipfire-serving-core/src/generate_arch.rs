// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-arch text generate paths for the non-qwen35 families.
//!
//! One `generate_*` per family — DeepSeek V4 Flash (DSML tool-call streaming),
//! Qwen2, MiniMax-M2, and LFM2.5-MoE — each running its own prefill + per-token
//! sample/stream loop against that arch's `forward_step`/`decode_step`. The
//! qwen35 AR/DFlash/MTP paths and the shared `generate` dispatcher stay in
//! `main.rs` for now. Extracted verbatim from the former `main.rs` monolith (no
//! behavior change); items called from `main.rs` are `pub`.

use std::io::Write;
use std::time::Instant;

use hipfire_arch_deepseek4 as deepseek4;
#[cfg(feature = "arch-lfm2moe")]
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_prompt as prompt_frame;
use hipfire_runtime::arch::{
    decode_loop_with_timing, DecodeLoopTiming, GenerateCtx, ServingBackend, SimpleAr,
};

use crate::events::{emit_committed_event, emit_error_with_id, emit_stream_event};
use crate::evidence::write_daemon_runtime_oneshot_evidence;
use crate::model::{effective_raw, LoadedModel};
use crate::request::ThinkMode;

/// DeepSeek V4 Flash generate path: prefill via the batched scratch, then a
/// per-token decode loop that parses the model's DSML stream into
/// token/reasoning/tool-call events ([`emit_stream_event`]). Honors think-mode
/// and the optional MTP spec-decode head.
pub fn generate_deepseek4(
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
    messages_history: Option<&[prompt_frame::Message]>,
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
    let cfg = match m.deepseek4_config.as_ref() {
        Some(c) => c,
        None => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"deepseek4_config missing on arch_id=9 generate"}}"#,
                id
            );
            let _ = stdout.flush();
            return;
        }
    };
    let weights = m
        .deepseek4_weights
        .as_ref()
        .expect("deepseek4_weights missing on arch_id=9 generate");
    let pbs = m
        .deepseek4_pbs
        .as_ref()
        .expect("deepseek4_pbs missing on arch_id=9 generate");
    let state = m
        .deepseek4_state
        .as_mut()
        .expect("deepseek4_state missing on arch_id=9 generate");
    let eos_tok = m.deepseek4_eos_tok;

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
        // Skip the trailing user prompt — we add it explicitly after.
        // Heuristic: if last message is role=user, treat its content as
        // the live prompt and drop it here.
        use prompt_frame::Role;
        let trim_end = if matches!(history.last().map(|m| m.role), Some(Role::User)) {
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
                    let fp =
                        prompt_frame::assistant_turn_fingerprint(&msg.content, &msg.tool_calls);
                    if std::env::var("HIPFIRE_DEEPSEEK4_CACHE_TRACE")
                        .ok()
                        .as_deref()
                        == Some("1")
                    {
                        eprintln!(
                            "[asst-cache lookup] fp={:#018x} content.len={} tool_calls={} hit={}",
                            fp,
                            msg.content.len(),
                            msg.tool_calls.len(),
                            m.asst_turn_cache.contains_key(&fp),
                        );
                    }
                    if let Some(cached) = m.asst_turn_cache.get(&fp) {
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

                    // Close the assistant turn with the EOS marker so
                    // the next turn starts cleanly.
                    prompt_ids.push(m.deepseek4_eos_tok);
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
    if !prompt.is_empty() {
        if let Some(u) = user_tok {
            prompt_ids.push(u);
        }
        prompt_ids.extend(tokenizer.encode(prompt));
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

    // Triaged config resolution for MTP speculative decode.
    // Priority: 1. legacy env var → 2. generic env var → 3. stored config → default.
    let spec_mode = std::env::var("HIPFIRE_DEEPSEEK4_SPEC_DECODE")
        .ok()
        .map(|v| v == "1")
        .unwrap_or_else(|| match std::env::var("HIPFIRE_MTP_MODE").ok().as_deref() {
            Some("on") => true,
            Some("off") => false,
            _ => m.mtp_mode == "on" || (m.mtp_mode == "auto" && m.mtp_weights_present),
        });
    let spec_k: usize = std::env::var("HIPFIRE_DEEPSEEK4_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            std::env::var("HIPFIRE_MTP_K")
                .ok()
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(m.mtp_k);

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

    if lcp == 0 {
        // Cache miss — start a fresh conversation in V4F's state.
        state.reset();
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

    // Prefill: batched chunked through PBS. If spec_mode, also fill the
    // MTP layer's SWA cache (prefill_with_mtp_fill) so the first
    // draft step sees a populated MTP history.
    let prefill_result = if spec_mode {
        deepseek4::forward::prefill_with_mtp_fill(
            cfg,
            weights,
            state,
            gpu,
            pbs,
            suffix_tokens,
            start_pos,
        )
    } else {
        deepseek4::forward::forward_prefill_batch_chunked(
            cfg,
            weights,
            state,
            gpu,
            suffix_tokens,
            start_pos,
            pbs,
        )
    };
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
    // attractor garbage at greedy temp=0. The MTP-fill prefill DOES
    // advance internally (forward.rs:7453), so we only need to update
    // for the plain-prefill branch.
    if !spec_mode {
        state.n_tokens = (start_pos as usize + suffix_tokens.len()) as u64;
    }
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
    let mut spec_windows: u64 = 0;
    let mut spec_drafts_offered: u64 = 0;
    let mut spec_drafts_accepted: u64 = 0;

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
    if spec_mode {
        // Spec-decode loop. The verifier picks argmax (greedy) so accept
        // semantics stay deterministic. When tools are present, thread
        // the same DSML grammar matcher through the MTP draft and main
        // verifier logits, then parse the emitted stream into tool_calls
        // events just like the plain decode loop.
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
        let mut parser = match think_mode {
            ThinkMode::High | ThinkMode::Max => deepseek4::dsml::StreamParser::new_in_think(),
            ThinkMode::NonThink => deepseek4::dsml::StreamParser::new(),
        };
        let mut matcher = deepseek4::grammar::Matcher::new(tool_schemas);
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
        let mut emit_tool_calls_buf: Vec<prompt_frame::ToolCall> = Vec::new();
        use hipfire_arch_deepseek4::dsml::StreamEvent;
        let mut absorb_event = |ev: &StreamEvent| {
            if let StreamEvent::ToolCalls(calls) = ev {
                for c in calls {
                    emit_tool_calls_buf.push(prompt_frame::ToolCall {
                        name: c.name.clone(),
                        arguments: c.arguments.clone(),
                    });
                }
            }
        };

        let mut spec_last_token = deepseek4::spec_decode::logits_argmax(&last_logits) as u32;
        let mut spec_last_position = pos_after_prefill;
        let mut last_hidden_ref = state.mtp_last_hidden.as_ref().map(|t| t as *const _);
        'outer: while generated_count < max_tokens {
            let lh: Option<&rdna_compute::GpuTensor> = unsafe {
                last_hidden_ref.and_then(|p| (p as *const rdna_compute::GpuTensor).as_ref())
            };
            let r = match if grammar_active {
                deepseek4::spec_decode::speculative_decode_step_with_pbs_grammar(
                    cfg,
                    weights,
                    state,
                    gpu,
                    pbs,
                    spec_last_token,
                    spec_last_position,
                    lh,
                    spec_k,
                    &mut matcher,
                    decoded_vocab,
                    &mut grammar_mask,
                )
            } else {
                deepseek4::spec_decode::speculative_decode_step_with_pbs(
                    cfg,
                    weights,
                    state,
                    gpu,
                    pbs,
                    spec_last_token,
                    spec_last_position,
                    lh,
                    spec_k,
                )
            } {
                Ok(r) => r,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("deepseek4spec-decode failed: {e:?}"));
                    let _ = stdout.flush();
                    return;
                }
            };
            spec_windows += 1;
            spec_drafts_offered += spec_k as u64;
            spec_drafts_accepted += r.n_accepted as u64;

            for &t in &r.accepted_tokens {
                if generated_count >= max_tokens || t == eos_tok {
                    break 'outer;
                }
                let frag = tokenizer.decode(&[t]);
                if grammar_active {
                    for ev in parser.feed(&frag) {
                        absorb_event(&ev);
                        emit_stream_event(stdout, id, ev);
                    }
                } else {
                    // Build through serde_json so `id` (user-supplied) and
                    // `frag` (model-generated UTF-8 with possible `"`/`\`)
                    // can't corrupt the JSONL line.
                    let envelope = serde_json::json!({
                        "type": "token",
                        "id": id,
                        "text": frag,
                    });
                    let _ = writeln!(stdout, "{}", envelope);
                }
                emit_committed_event(
                    stdout,
                    id,
                    t,
                    generated_count,
                    decode_t0.elapsed().as_millis() as u64,
                );
                let _ = stdout.flush();
                m.conversation_tokens.push(t);
                generated_count += 1;
            }
            if let Some(&t) = r.accepted_tokens.last() {
                spec_last_position += r.accepted_tokens.len() as u32;
                spec_last_token = t;
            }
            last_hidden_ref = state.mtp_last_hidden.as_ref().map(|t| t as *const _);
        }
        if grammar_active {
            for ev in parser.finish() {
                absorb_event(&ev);
                emit_stream_event(stdout, id, ev);
            }
            let _ = stdout.flush();
            drop(absorb_event);
            tool_calls_parsed_count = emit_tool_calls_buf.len();
        }
    } else {
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
        // `m.deepseek4_state` (which `state` holds `&mut` to) and from
        // `m.tokenizer` (which `tokenizer` holds `&` to), so the
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
        let mut emit_tool_calls_buf: Vec<prompt_frame::ToolCall> = Vec::new();
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
                        emit_tool_calls_buf.push(prompt_frame::ToolCall {
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
            let fp = prompt_frame::assistant_turn_fingerprint(&emit_text_buf, &emit_tool_calls_buf);
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
    // `tool_calls_parsed_count` is set inside the non-spec branch
    // immediately after parser.finish(); spec_mode leaves it at 0.
    let finish_reason: &'static str = if tool_calls_parsed_count > 0 {
        "tool_calls"
    } else if generated_count >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let done_envelope = if spec_mode {
        let accept_pct = if spec_drafts_offered > 0 {
            spec_drafts_accepted as f64 / spec_drafts_offered as f64 * 100.0
        } else {
            0.0
        };
        serde_json::json!({
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
            "spec_k": spec_k,
            "spec_windows": spec_windows,
            "spec_accept_pct": accept_pct,
        })
    } else {
        serde_json::json!({
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
        })
    };
    let _ = writeln!(stdout, "{}", done_envelope);
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
/// P3.1: now routed through `ServingBackend::serve` (the shared
/// `run_simple_ar` / `decode_loop` seam, mirroring `generate_gemma3`). The
/// per-token prefill/decode loop and streaming live inside `Qwen2Backend`.
pub fn generate_qwen2(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    _system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
) {
    // P3.1: route through the shared `ServingBackend::serve` seam
    // (`run_simple_ar` → tokenize → prefill → `decode_loop`), mirroring
    // `generate_gemma3`. `decode_loop` honors `repeat_penalty`; greedy otherwise.
    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded".to_string());
        return;
    }
    if m.qwen2_backend.is_none() {
        emit_error_with_id(
            stdout,
            id,
            "qwen2 backend not loaded (arch 7 not active)".to_string(),
        );
        return;
    }
    // Disjoint field borrows: tokenizer (shared) + backend (mut).
    let tok = m.tokenizer.as_ref().unwrap();
    let backend = m.qwen2_backend.as_mut().unwrap();

    let no_images: [&[u8]; 0] = [];
    let mut ctx = GenerateCtx {
        id,
        prompt,
        temperature: temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        max_think_tokens: 0,
        stop_sequences: &[],
        images: &no_images,
        sink: stdout,
    };
    let result = backend.serve(gpu, tok, &mut ctx);
    // `ctx` mutably borrows `stdout`; drop before reusing it for errors.
    drop(ctx);
    if let Err(e) = result {
        emit_error_with_id(stdout, id, format!("qwen2 serve: {e}"));
    }
}

/// LLaMA / Mistral / plain-Qwen3 (arch_id 0/1) generate path — routes through the
/// `ServingBackend` seam (P3.2). Unlike qwen2, llama needs chat-framing, so this
/// builds `prompt_tokens` (the model's jinja `chat_template` when
/// `HIPFIRE_JINJA_CHAT=1`, else the hand-rolled `ChatFrame` scaffold honoring
/// `assistant_prefix` / raw-completion) — the same framing the qwen35-shared
/// `generate()` applied — then prefills those tokens and runs the shared
/// `decode_loop` (full temperature/top-p sampling via P3.3). Fast paths
/// (DFlash/MTP/tools-execution) are out of scope here; correctness first.
/// nemotron_h (arch_id 14) / pure Mamba-2 (arch_id 15) generate path — the same
/// dense-AR `ServingBackend` seam as `generate_llama`, driving the Mamba-capable
/// `NemotronModel` backend. Frames the prompt (jinja `chat_template` when
/// `HIPFIRE_JINJA_CHAT=1`, else the hand-rolled `ChatFrame`), prefills the
/// framed tokens (which builds per-block recurrent/KV state), then runs the
/// shared `decode_loop`. Fast paths are out of scope; correctness first.
#[allow(clippy::too_many_arguments)]
pub fn generate_nemotron(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    evidence_dir: Option<&str>,
) {
    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded".to_string());
        return;
    }
    if m.nemotron_backend.is_none() {
        emit_error_with_id(
            stdout,
            id,
            "mamba-capable backend not loaded (arch 14/15 not active)".to_string(),
        );
        return;
    }

    // Frame the prompt up front (releases the shared `m` borrows before the
    // backend `&mut` borrow below). Same scaffold as generate_llama.
    let raw = effective_raw(m);
    let prompt_tokens: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // nemotron_h ships a correct ChatML jinja template (`<|im_start|>` /
        // `<|im_end|>`), so default to it when present (opt out with
        // HIPFIRE_JINJA_CHAT=0). The hand-rolled Plain ChatFrame is the fallback.
        let try_jinja = m.chat_template.is_some()
            && std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let rendered = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(mh) => mh,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
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
            match rendered {
                Ok(text) => tokenizer.encode(&text),
                Err(e) => {
                    eprintln!("[daemon] nemotron jinja render failed ({e}) — Plain fallback");
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix,
                        raw,
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix,
                raw,
            }
            .build()
        }
    };
    if prompt_tokens.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after framing".to_string());
        return;
    }

    let tok = m.tokenizer.as_ref().unwrap();
    let backend = m.nemotron_backend.as_mut().unwrap();
    let eos = backend.eos_token();

    let prefill_t0 = Instant::now();
    if let Err(e) = SimpleAr::prefill(backend, gpu, &prompt_tokens) {
        emit_error_with_id(stdout, id, format!("nemotron prefill: {e}"));
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;
    let n = prompt_tokens.len();
    let no_images: [&[u8]; 0] = [];
    let mut ctx = GenerateCtx {
        id,
        prompt: "", // unused: prefill already consumed the framed tokens
        temperature: temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        max_think_tokens: 0,
        stop_sequences: &[],
        images: &no_images,
        sink: stdout,
    };
    let result = decode_loop_with_timing(
        gpu,
        backend,
        tok,
        eos,
        &mut ctx,
        n,
        n,
        DecodeLoopTiming {
            prefill_ms: Some(prefill_ms),
        },
    );
    drop(ctx);
    match result {
        Ok(outcome) => {
            if let (Some(dir), Some(prefill_ms), Some(decode_ms)) =
                (evidence_dir, outcome.prefill_ms, outcome.decode_ms)
            {
                write_daemon_runtime_oneshot_evidence(
                    dir,
                    m,
                    gpu,
                    id,
                    outcome.prompt_tokens,
                    outcome.tokens_generated,
                    prefill_ms / 1000.0,
                    decode_ms / 1000.0,
                    outcome.ttft_ms.unwrap_or(prefill_ms),
                );
            }
        }
        Err(e) => emit_error_with_id(stdout, id, format!("nemotron decode: {e}")),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_zaya(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    evidence_dir: Option<&str>,
) {
    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded".to_string());
        return;
    }
    if m.zaya_backend.is_none() {
        emit_error_with_id(
            stdout,
            id,
            "zaya backend not loaded (arch 16 not active)".to_string(),
        );
        return;
    }

    // Frame the prompt up front (releases the shared `m` borrows before the
    // backend `&mut` borrow below). Same scaffold as generate_llama.
    let raw = effective_raw(m);
    let prompt_tokens: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        // nemotron_h ships a correct ChatML jinja template (`<|im_start|>` /
        // `<|im_end|>`), so default to it when present (opt out with
        // HIPFIRE_JINJA_CHAT=0). The hand-rolled Plain ChatFrame is the fallback.
        let try_jinja = m.chat_template.is_some()
            && std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let rendered = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(mh) => mh,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
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
            match rendered {
                Ok(text) => tokenizer.encode(&text),
                Err(e) => {
                    eprintln!("[daemon] zaya jinja render failed ({e}) — Plain fallback");
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix,
                        raw,
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix,
                raw,
            }
            .build()
        }
    };
    if prompt_tokens.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after framing".to_string());
        return;
    }

    let tok = m.tokenizer.as_ref().unwrap();
    let backend = m.zaya_backend.as_mut().unwrap();
    let eos = backend.eos_token();

    let prefill_t0 = Instant::now();
    if let Err(e) = SimpleAr::prefill(backend, gpu, &prompt_tokens) {
        emit_error_with_id(stdout, id, format!("zaya prefill: {e}"));
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;
    let n = prompt_tokens.len();
    let no_images: [&[u8]; 0] = [];
    let mut ctx = GenerateCtx {
        id,
        prompt: "", // unused: prefill already consumed the framed tokens
        temperature: temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        max_think_tokens: 0,
        stop_sequences: &[],
        images: &no_images,
        sink: stdout,
    };
    let result = decode_loop_with_timing(
        gpu,
        backend,
        tok,
        eos,
        &mut ctx,
        n,
        n,
        DecodeLoopTiming {
            prefill_ms: Some(prefill_ms),
        },
    );
    drop(ctx);
    match result {
        Ok(outcome) => {
            if let (Some(dir), Some(prefill_ms), Some(decode_ms)) =
                (evidence_dir, outcome.prefill_ms, outcome.decode_ms)
            {
                write_daemon_runtime_oneshot_evidence(
                    dir,
                    m,
                    gpu,
                    id,
                    outcome.prompt_tokens,
                    outcome.tokens_generated,
                    prefill_ms / 1000.0,
                    decode_ms / 1000.0,
                    outcome.ttft_ms.unwrap_or(prefill_ms),
                );
            }
        }
        Err(e) => emit_error_with_id(stdout, id, format!("zaya decode: {e}")),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_llama(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    evidence_dir: Option<&str>,
) {
    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded".to_string());
        return;
    }
    if m.llama_backend.is_none() {
        emit_error_with_id(
            stdout,
            id,
            "llama backend not loaded (arch 0/1 not active)".to_string(),
        );
        return;
    }

    // Build the framed prompt tokens up front (releases the shared `m` borrows
    // before the backend `&mut` borrow below).
    let raw = effective_raw(m);
    let prompt_tokens: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let try_jinja = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1")
            && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let rendered = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(mh) => mh,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
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
            match rendered {
                Ok(text) => tokenizer.encode(&text),
                Err(e) => {
                    eprintln!("[daemon] llama jinja render failed ({e}) — Plain fallback");
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix,
                        raw,
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix,
                raw,
            }
            .build()
        }
    };
    if prompt_tokens.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after framing".to_string());
        return;
    }

    // Disjoint field borrows: tokenizer (shared) + backend (mut).
    let tok = m.tokenizer.as_ref().unwrap();
    let backend = m.llama_backend.as_mut().unwrap();
    let eos = backend.eos_token();

    // Pre-tokenized prefill (the framing already produced tokens), then the
    // shared streaming/sampling decode loop.
    let prefill_t0 = Instant::now();
    if let Err(e) = SimpleAr::prefill(backend, gpu, &prompt_tokens) {
        emit_error_with_id(stdout, id, format!("llama prefill: {e}"));
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;
    let n = prompt_tokens.len();
    let no_images: [&[u8]; 0] = [];
    let mut ctx = GenerateCtx {
        id,
        prompt: "", // unused: prefill already consumed the framed tokens
        temperature: temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        max_think_tokens: 0,
        stop_sequences: &[],
        images: &no_images,
        sink: stdout,
    };
    let result = decode_loop_with_timing(
        gpu,
        backend,
        tok,
        eos,
        &mut ctx,
        n,
        n,
        DecodeLoopTiming {
            prefill_ms: Some(prefill_ms),
        },
    );
    drop(ctx);
    match result {
        Ok(outcome) => {
            if let (Some(dir), Some(prefill_ms), Some(decode_ms)) =
                (evidence_dir, outcome.prefill_ms, outcome.decode_ms)
            {
                write_daemon_runtime_oneshot_evidence(
                    dir,
                    m,
                    gpu,
                    id,
                    outcome.prompt_tokens,
                    outcome.tokens_generated,
                    prefill_ms / 1000.0,
                    decode_ms / 1000.0,
                    outcome.ttft_ms.unwrap_or(prefill_ms),
                );
            }
        }
        Err(e) => emit_error_with_id(stdout, id, format!("llama decode: {e}")),
    }
}

/// MiniMax-M2 (arch_id=10) generate path — minimal AR bring-up.
///
/// Mirrors `generate_qwen2`'s shape (prefill = per-token loop, decode =
/// per-token loop, JSONL `token` / `done` events) with two differences:
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
/// MiniMax-M2 generate path: per-token prefill + decode over
/// `minimax::forward::decode_step` (Mixtral-style MoE), streaming events.
pub fn generate_minimax(
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
    messages_history: Option<&[prompt_frame::Message]>,
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
    if m.minimax_config.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"minimax_config missing on arch_id=10 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    // ── Prompt build (same two-path branch as the qwen35 AR path) ──
    // `primed_think` records whether the rendered prompt actually ended with
    // the MiniMax `<think>` generation-primer, so we only re-emit the opener
    // (below) when the model truly begins inside the reasoning block. A jinja
    // render failure that falls back to the Plain frame leaves it false.
    let mut primed_think = false;
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(h) => h,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
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
                Ok(rendered) => {
                    primed_think = rendered.trim_end().ends_with("<think>");
                    tokenizer.encode(&rendered)
                }
                Err(e) => {
                    eprintln!("[daemon] jinja render failed in minimax path ({e}) — falling back to Plain");
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                        raw: effective_raw(m),
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                raw: effective_raw(m),
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

    let eos_tok = m.minimax_eos_tok;

    // Capacity guard. No eviction on arch_id=10 — reset the KV cursor when
    // the requested run would overflow the budget. (max_seq + n_tokens live
    // on the state.)
    let overflow = {
        let state = m.minimax_state.as_ref().unwrap();
        state.n_tokens + prompt_ids.len() + max_tokens > state.max_seq
    };
    if overflow {
        let (n, cap) = {
            let state = m.minimax_state.as_ref().unwrap();
            (state.n_tokens, state.max_seq)
        };
        eprintln!("[daemon] arch_id=10 context full ({n}/{cap}) — resetting MiniMaxState",);
        m.minimax_state.as_mut().unwrap().reset();
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }

    let t0 = Instant::now();

    // ── Prefill: decode_step per prompt token. Disjoint field borrows of
    // `m` (config / weights / state) let us also push to
    // `m.conversation_tokens` in the same scope (same pattern as
    // generate_qwen2). The LAST decode_step's logits are the predictions
    // for the first generated token. ──
    let mut last_logits: Vec<f32> = Vec::new();
    {
        let cfg = m.minimax_config.as_ref().unwrap();
        let weights = m.minimax_weights.as_ref().unwrap();
        let state = m.minimax_state.as_mut().unwrap();
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_ids {
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
    for &tok in &prompt_ids {
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
            let cfg = m.minimax_config.as_ref().unwrap();
            let weights = m.minimax_weights.as_ref().unwrap();
            let state = m.minimax_state.as_mut().unwrap();
            let position = state.n_tokens as u32;
            minimax::forward::decode_step(cfg, weights, state, gpu, next_tok, position)
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("minimax decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = m.minimax_state.as_ref().unwrap().n_tokens;

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

/// LFM2.5-MoE (arch_id=11) generate path — minimal AR bring-up.
///
/// Prefill routes through the arch-local batched prefill path when eligible
/// (`HIPFIRE_PREFILL_BATCHED=0` falls back inside the arch crate); decode stays
/// per-token. Out of scope (and not wired): spec-decode, MTP, grammar,
/// tool-call parsing/execution, repeat penalty, multi-GPU, eviction/prefix-cache.
#[cfg(feature = "arch-lfm2moe")]
#[allow(clippy::too_many_arguments)]
/// LFM2.5-MoE generate path: prompt prefill + decode over the hybrid
/// conv/attention + top-4 MoE `lfm2moe` forward, streaming events. Gated on the
/// `arch-lfm2moe` feature.
pub fn generate_lfm2moe(
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
    messages_history: Option<&[prompt_frame::Message]>,
    prefill_already_done: bool,
    prefilled_prompt_tokens: Option<usize>,
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
    if m.lfm2moe_config.is_none() {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"lfm2moe_config missing on arch_id=11 generate"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }
    if m.lfm2_dflash.is_some() && temp <= 1e-6 {
        generate_lfm2moe_dflash(
            m,
            gpu,
            stdout,
            id,
            prompt,
            system_prompt,
            max_tokens,
            max_think_tokens,
            tools,
            messages_history,
            prefill_already_done,
            prefilled_prompt_tokens,
        );
        let _ = top_p;
        return;
    }

    // ── Prompt build (same two-path branch as the minimax AR path) ──
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(h) => h,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
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
                    eprintln!("[daemon] jinja render failed in lfm2moe path ({e}) — falling back to Plain");
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                        raw: effective_raw(m),
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                raw: effective_raw(m),
            }
            .build()
        }
    };

    if prompt_ids.is_empty() && !prefill_already_done {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"empty prompt after tokenize"}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    let eos_tok = m.lfm2moe_eos_tok;

    // Capacity guard. Without eviction the physical buffer is the hard cap.
    // With CASK/TriAttention active, `n_tokens` is the physical cursor and
    // `compact_offset` carries the logical prefix already compacted away.
    let overflow = {
        let state = m.lfm2moe_state.as_ref().unwrap();
        let current = if m.eviction.is_some() {
            state.n_tokens + state.kv.compact_offset
        } else {
            state.n_tokens
        };
        let prefill_budget = if prefill_already_done {
            0
        } else {
            prompt_ids.len()
        };
        current + prefill_budget + max_tokens > state.max_seq
    };
    if overflow {
        let (n, logical, cap) = {
            let state = m.lfm2moe_state.as_ref().unwrap();
            (
                state.n_tokens,
                state.n_tokens + state.kv.compact_offset,
                state.max_seq,
            )
        };
        if prefill_already_done {
            emit_error_with_id(
                stdout,
                id,
                format!(
                    "prefill_already_done request exceeds LFM2 context: logical={} + max_tokens={} > max_seq={}",
                    logical, max_tokens, cap
                ),
            );
            return;
        }
        eprintln!(
            "[daemon] arch_id=11 context full (physical={n} logical={logical}/{cap}) — resetting Lfm2MoeState",
        );
        let _ = m.lfm2moe_state.as_mut().unwrap().reset(gpu);
        m.seq_pos = 0;
        m.conversation_tokens.clear();
    }

    let t0 = Instant::now();

    // ── Prefill. The returned logits are the predictions for the first
    // generated token. ──
    let mut last_logits: Vec<f32>;
    let prefill_ms: u128;
    if prefill_already_done {
        let current_position = {
            let state = m
                .lfm2moe_state
                .as_ref()
                .expect("lfm2moe_state missing on arch_id=11 generate");
            state.n_tokens + state.kv.compact_offset
        };
        let expected_position = prefilled_prompt_tokens.unwrap_or(prompt_ids.len());
        if expected_position == 0 {
            emit_error_with_id(
                stdout,
                id,
                "prefill_already_done requested for LFM2 without a positive prefilled prompt token count",
            );
            return;
        }
        if current_position != expected_position {
            emit_error_with_id(
                stdout,
                id,
                format!(
                    "prefill_already_done requested but active LFM2 session position {} does not match expected prefilled prompt token count {}",
                    current_position, expected_position
                ),
            );
            return;
        }
        let state = m
            .lfm2moe_state
            .as_ref()
            .expect("lfm2moe_state missing on arch_id=11 generate");
        last_logits = match gpu.download_f32(&state.logits) {
            Ok(logits) => logits,
            Err(e) => {
                emit_error_with_id(
                    stdout,
                    id,
                    format!("lfm2moe prefilled logits download failed: {e:?}"),
                );
                return;
            }
        };
        if last_logits.is_empty() {
            emit_error_with_id(stdout, id, "lfm2moe prefilled logits were empty");
            return;
        }
        prefill_ms = 0;
    } else {
        let cfg = m.lfm2moe_config.as_ref().unwrap();
        let weights = m.lfm2moe_weights.as_ref().unwrap();
        let state = m.lfm2moe_state.as_mut().unwrap();
        if let Some(ref ev) = m.eviction {
            let mut logits = Vec::new();
            for &tok in &prompt_ids {
                let position = state.n_tokens as u32;
                match lfm2moe::forward::decode_step(cfg, weights, state, gpu, tok, position) {
                    Ok(next_logits) => logits = next_logits,
                    Err(e) => {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("lfm2moe serial prefill failed: {e:?}"),
                        );
                        return;
                    }
                }
                match ev.maybe_evict(gpu, &mut state.kv, state.n_tokens) {
                    Ok(Some(hipfire_runtime::triattn::EvictionResult { new_physical, .. })) => {
                        state.n_tokens = new_physical;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        emit_error_with_id(
                            stdout,
                            id,
                            format!("lfm2moe prefill eviction failed: {e:?}"),
                        );
                        return;
                    }
                }
            }
            last_logits = logits;
        } else {
            match lfm2moe::forward::prefill_batch(cfg, weights, state, gpu, &prompt_ids) {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    emit_error_with_id(stdout, id, format!("lfm2moe prefill failed: {e:?}"));
                    return;
                }
            }
        }
        for &tok in &prompt_ids {
            m.conversation_tokens.push(tok);
        }
        prefill_ms = t0.elapsed().as_millis();
    }

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
        if next_tok == eos_tok {
            break;
        }

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

        let step = {
            let cfg = m.lfm2moe_config.as_ref().unwrap();
            let weights = m.lfm2moe_weights.as_ref().unwrap();
            let state = m.lfm2moe_state.as_mut().unwrap();
            let position = state.n_tokens as u32;
            let step = lfm2moe::forward::decode_step(cfg, weights, state, gpu, next_tok, position);
            if step.is_ok() {
                if let Some(ref ev) = m.eviction {
                    match ev.maybe_evict(gpu, &mut state.kv, state.n_tokens) {
                        Ok(Some(hipfire_runtime::triattn::EvictionResult {
                            new_physical, ..
                        })) => {
                            state.n_tokens = new_physical;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            emit_error_with_id(
                                stdout,
                                id,
                                format!("lfm2moe decode eviction failed: {e:?}"),
                            );
                            return;
                        }
                    }
                }
            }
            step
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm2moe decode failed: {e:?}"));
                return;
            }
        }
    }

    m.seq_pos = {
        let state = m.lfm2moe_state.as_ref().unwrap();
        state.n_tokens + state.kv.compact_offset
    };

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

#[cfg(feature = "arch-lfm2moe")]
#[allow(clippy::too_many_arguments)]
fn generate_lfm2moe_dflash(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    prefill_already_done: bool,
    prefilled_prompt_tokens: Option<usize>,
) {
    if m.eviction.is_some() {
        emit_error_with_id(
            stdout,
            id,
            "LFM2 DFlash does not support CASK/TriAttention eviction yet",
        );
        return;
    }
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
        let try_jinja = jinja_enabled && m.chat_template.is_some();
        if try_jinja {
            let template = m.chat_template.as_ref().unwrap();
            let frame = prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: prompt,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
            };
            let render_result = if tools.is_some() || messages_history.is_some() {
                let synthesized: Vec<prompt_frame::Message>;
                let messages_slice: &[prompt_frame::Message] = match messages_history {
                    Some(h) => h,
                    None => {
                        let mut v = Vec::new();
                        if let Some(sys) = system_prompt {
                            v.push(prompt_frame::Message {
                                role: prompt_frame::Role::System,
                                content: sys.to_string(),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                            });
                        }
                        v.push(prompt_frame::Message {
                            role: prompt_frame::Role::User,
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
                    eprintln!(
                        "[daemon] jinja render failed in lfm2moe dflash path ({e}) - falling back to Plain"
                    );
                    prompt_frame::ChatFrame {
                        tokenizer,
                        system: system_prompt,
                        user: prompt,
                        assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                        raw: effective_raw(m),
                    }
                    .build()
                }
            }
        } else {
            prompt_frame::ChatFrame {
                tokenizer,
                system: system_prompt,
                user: prompt,
                assistant_prefix: prompt_frame::AssistantPrefix::Plain,
                raw: effective_raw(m),
            }
            .build()
        }
    };
    if prompt_ids.is_empty() && !prefill_already_done {
        emit_error_with_id(stdout, id, "empty prompt after tokenize");
        return;
    }
    let expected_prefilled_position = if prefill_already_done {
        let expected = prefilled_prompt_tokens.unwrap_or(prompt_ids.len());
        if expected == 0 {
            emit_error_with_id(
                stdout,
                id,
                "prefill_already_done requested for LFM2 DFlash without a positive prefilled prompt token count",
            );
            return;
        }
        Some(expected)
    } else {
        None
    };
    if max_tokens == 0 {
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":0,"tok_s":0.0,"prefill_ms":0,"total_ms":0,"dflash":true,"cycles":0,"accepted":0,"accept_rate":0.0}}"#,
            id
        );
        let _ = stdout.flush();
        return;
    }

    let (ctx_capacity, block_size) = {
        let df = m.lfm2_dflash.as_ref().unwrap();
        (df.ctx_capacity, df.block_size)
    };
    let target_capacity = m
        .lfm2moe_state
        .as_ref()
        .map(|s| s.max_seq)
        .unwrap_or(ctx_capacity);
    let usable_capacity = ctx_capacity.min(target_capacity);
    let starting_position = expected_prefilled_position.unwrap_or(prompt_ids.len());
    if starting_position.saturating_add(block_size) > usable_capacity {
        emit_error_with_id(
            stdout,
            id,
            format!(
                "LFM2 DFlash position ({}) + block ({}) exceeds context capacity {}",
                starting_position, block_size, usable_capacity
            ),
        );
        return;
    }

    let eos_tok = m.lfm2moe_eos_tok;
    let t0 = Instant::now();
    let first_token = if let Some(expected_position) = expected_prefilled_position {
        let current_position = {
            let state = m.lfm2moe_state.as_ref().unwrap();
            state.n_tokens + state.kv.compact_offset
        };
        if current_position != expected_position {
            emit_error_with_id(
                stdout,
                id,
                format!(
                    "prefill_already_done requested but active LFM2 DFlash session position {} does not match expected prefilled prompt token count {}",
                    current_position, expected_position
                ),
            );
            return;
        }
        {
            let df = m.lfm2_dflash.as_mut().unwrap();
            let row_floats = df.draft_config.num_extract() * df.draft_config.hidden;
            let expected_hidden = match expected_position.checked_mul(row_floats) {
                Some(v) => v,
                None => {
                    emit_error_with_id(
                        stdout,
                        id,
                        "lfm2moe dflash prefilled hidden history size overflow",
                    );
                    return;
                }
            };
            if df.target_hidden_host.len() != expected_hidden {
                emit_error_with_id(
                    stdout,
                    id,
                    format!(
                        "lfm2moe dflash prefilled hidden history has {} floats, expected {} for position {}",
                        df.target_hidden_host.len(),
                        expected_hidden,
                        expected_position
                    ),
                );
                return;
            }
            df.draft_scratch.reset_upload_tracking();
        }
        let logits = {
            let state = m.lfm2moe_state.as_ref().unwrap();
            match gpu.download_f32(&state.logits) {
                Ok(logits) => logits,
                Err(e) => {
                    emit_error_with_id(
                        stdout,
                        id,
                        format!("lfm2moe dflash prefilled logits download failed: {e:?}"),
                    );
                    return;
                }
            }
        };
        if logits.is_empty() {
            emit_error_with_id(stdout, id, "lfm2moe dflash prefilled logits were empty");
            return;
        }
        lfm2_argmax(&logits)
    } else {
        let cfg = m.lfm2moe_config.as_ref().unwrap();
        let weights = m.lfm2moe_weights.as_ref().unwrap();
        let state = m.lfm2moe_state.as_mut().unwrap();
        let df = m.lfm2_dflash.as_mut().unwrap();
        if let Err(e) = state.reset(gpu) {
            emit_error_with_id(stdout, id, format!("lfm2moe dflash reset failed: {e}"));
            return;
        }
        df.target_hidden_host.clear();
        df.draft_scratch.reset_upload_tracking();
        let mut capture = match lfm2moe::forward::Lfm2HiddenCapture::new(
            cfg.num_hidden_layers,
            cfg.hidden_size,
            df.draft_config.target_layer_ids.clone(),
        ) {
            Ok(capture) => capture,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm2moe dflash capture: {e}"));
                return;
            }
        };
        let logits_per_pos = match lfm2moe::forward::prefill_batch_with_hidden_logits(
            cfg,
            weights,
            state,
            gpu,
            &prompt_ids,
            &mut capture,
        ) {
            Ok(logits) => logits,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm2moe dflash prefill failed: {e}"));
                return;
            }
        };
        df.target_hidden_host
            .extend_from_slice(&capture.take_rows());
        if state.n_tokens != prompt_ids.len() {
            emit_error_with_id(
                stdout,
                id,
                format!(
                    "lfm2moe dflash prefill ended at {}, expected {}",
                    state.n_tokens,
                    prompt_ids.len()
                ),
            );
            return;
        }
        match logits_per_pos.chunks_exact(cfg.vocab_size).last() {
            Some(row) => lfm2_argmax(row),
            None => {
                emit_error_with_id(stdout, id, "lfm2moe dflash prefill returned no logits");
                return;
            }
        }
    };
    let prefill_ms = if prefill_already_done {
        0
    } else {
        t0.elapsed().as_millis()
    };
    if !prefill_already_done {
        m.conversation_tokens.clear();
        m.conversation_tokens.extend_from_slice(&prompt_ids);
    }

    if first_token == eos_tok
        || m.tokenizer
            .as_ref()
            .map(|t| t.is_terminator(first_token))
            .unwrap_or(false)
    {
        let total_ms = t0.elapsed().as_millis().max(1);
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{}","tokens":0,"tok_s":0.0,"prefill_ms":{},"total_ms":{},"dflash":true,"cycles":0,"accepted":0,"accept_rate":0.0}}"#,
            id, prefill_ms, total_ms
        );
        let _ = stdout.flush();
        return;
    }

    let emit_token = |stdout: &mut std::io::Stdout,
                      id: &str,
                      token: u32,
                      ordinal: usize,
                      elapsed_ms: u64,
                      tokenizer: &hipfire_model::tokenizer::Tokenizer| {
        let frag = tokenizer.decode(&[token]);
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        emit_committed_event(stdout, id, token, ordinal, elapsed_ms);
    };

    {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        emit_token(
            stdout,
            id,
            first_token,
            0,
            t0.elapsed().as_millis() as u64,
            tokenizer,
        );
    }
    m.conversation_tokens.push(first_token);

    let decode_t0 = Instant::now();
    let mut generated_count = 1usize;
    let mut position = starting_position;
    let mut seed_token = first_token;
    let mut cycles = 0usize;
    let mut accepted_total = 0usize;
    let mut drafted_total = 0usize;

    while generated_count < max_tokens {
        if position.saturating_add(block_size) > usable_capacity {
            break;
        }
        let step = {
            let cfg = m.lfm2moe_config.as_ref().unwrap();
            let weights = m.lfm2moe_weights.as_ref().unwrap();
            let state = m.lfm2moe_state.as_mut().unwrap();
            let df = m.lfm2_dflash.as_mut().unwrap();
            lfm2moe::spec_step_dflash(
                gpu,
                weights,
                cfg,
                state,
                &df.draft_weights,
                &df.draft_config,
                &mut df.draft_scratch,
                &mut df.target_hidden_host,
                &mut df.target_snap,
                position,
                seed_token,
                None,
                None,
            )
        };
        let step = match step {
            Ok(step) => step,
            Err(e) => {
                emit_error_with_id(stdout, id, format!("lfm2moe dflash spec_step failed: {e}"));
                break;
            }
        };
        cycles += 1;
        accepted_total += step.accepted;
        drafted_total += step.drafted.len().saturating_sub(1);

        let mut hit_eos = false;
        for &tok in step.committed.iter().skip(1) {
            if generated_count >= max_tokens {
                break;
            }
            let terminator = {
                let tokenizer = m.tokenizer.as_ref().unwrap();
                tok == eos_tok || tokenizer.is_terminator(tok)
            };
            if terminator {
                hit_eos = true;
                break;
            }
            {
                let tokenizer = m.tokenizer.as_ref().unwrap();
                emit_token(
                    stdout,
                    id,
                    tok,
                    generated_count,
                    t0.elapsed().as_millis() as u64,
                    tokenizer,
                );
            }
            m.conversation_tokens.push(tok);
            generated_count += 1;
        }
        position += step.advance;
        seed_token = step.bonus_token;
        if hit_eos {
            break;
        }
    }

    m.seq_pos = m
        .lfm2moe_state
        .as_ref()
        .map(|s| s.n_tokens)
        .unwrap_or(position);
    let decode_ms = decode_t0.elapsed().as_millis().max(1);
    let total_ms = t0.elapsed().as_millis().max(1);
    let tok_s = (generated_count as f64 * 1000.0) / decode_ms as f64;
    let accept_rate = if drafted_total > 0 {
        accepted_total as f64 / drafted_total as f64
    } else {
        0.0
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.2},"prefill_ms":{},"total_ms":{},"dflash":true,"cycles":{},"accepted":{},"accept_rate":{:.3}}}"#,
        id, generated_count, tok_s, prefill_ms, total_ms, cycles, accepted_total, accept_rate,
    );
    let _ = stdout.flush();
}

#[cfg(feature = "arch-lfm2moe")]
fn lfm2_argmax(row: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &value) in row.iter().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn framed_gemma3_prompt(prompt: &str, system_prompt: Option<&str>) -> String {
    let mut framed = String::from("<bos><start_of_turn>user\n");
    if let Some(sys) = system_prompt.filter(|s| !s.is_empty()) {
        // gemma3 has no system role — HF folds system content into the user turn.
        framed.push_str(sys);
        framed.push_str("\n\n");
    }
    framed.push_str(prompt);
    framed.push_str("<end_of_turn>\n<start_of_turn>model\n");
    framed
}

/// Gemma3 text (arch_id=12, e.g. medgemma-*-text) generate path.
///
/// Frames the gemma chat prompt (bos + user turn + model turn; `<bos>` /
/// `<start_of_turn>` / `<end_of_turn>` are registered specials that round-trip
/// through `tok.encode`, so the framed text reproduces the gemma chat template),
/// times prefill, then runs the shared decode loop. Greedy; `repeat_penalty` is
/// honored by `decode_loop`. No vision, tools, or think-budget on this path.
#[allow(clippy::too_many_arguments)]
pub fn generate_gemma3(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
) {
    let framed = framed_gemma3_prompt(prompt, system_prompt);

    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded".to_string());
        return;
    }
    if m.gemma3_text.is_none() {
        emit_error_with_id(
            stdout,
            id,
            "gemma3 backend not loaded (arch 12 not active)".to_string(),
        );
        return;
    }
    // Disjoint field borrows: tokenizer (shared) + backend (mut).
    let tok = m.tokenizer.as_ref().unwrap();
    let backend = m.gemma3_text.as_mut().unwrap();
    let eos = tok
        .special_token_id("<end_of_turn>")
        .unwrap_or_else(|| backend.eos_token());
    let prompt_tokens = tok.encode(&framed);
    if prompt_tokens.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after framing".to_string());
        return;
    }

    let prefill_t0 = Instant::now();
    if let Err(e) = SimpleAr::prefill(backend, gpu, &prompt_tokens) {
        emit_error_with_id(stdout, id, format!("gemma3 prefill: {e}"));
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;

    let no_images: [&[u8]; 0] = [];
    let mut ctx = GenerateCtx {
        id,
        prompt: "",
        temperature: temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        max_think_tokens: 0,
        stop_sequences: &[],
        images: &no_images,
        sink: stdout,
    };
    let n = prompt_tokens.len();
    let result = decode_loop_with_timing(
        gpu,
        backend,
        tok,
        eos,
        &mut ctx,
        n,
        n,
        DecodeLoopTiming {
            prefill_ms: Some(prefill_ms),
        },
    );
    // `ctx` mutably borrows `stdout`; drop before reusing it for errors.
    drop(ctx);
    if let Err(e) = result {
        emit_error_with_id(stdout, id, format!("gemma3 serve: {e}"));
    }
}

/// Gemma3-VL text-only generate path (arch_id=13, e.g. full MedGemma).
///
/// The image-bearing daemon route lives in `generate_vl_gemma3`; this handles
/// ordinary text requests against the multimodal artifact by reusing the same
/// Gemma3 chat framing as arch_id=12 and passing an empty image slice into the
/// VL backend's `ServingBackend::serve`.
#[allow(clippy::too_many_arguments)]
pub fn generate_gemma3_vl_text(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
) {
    let framed = framed_gemma3_prompt(prompt, system_prompt);

    if m.tokenizer.is_none() {
        emit_error_with_id(stdout, id, "tokenizer not loaded".to_string());
        return;
    }
    if m.gemma3_vl.is_none() {
        emit_error_with_id(
            stdout,
            id,
            "gemma3-vl backend not loaded (arch 13 not active)".to_string(),
        );
        return;
    }

    let tok = m.tokenizer.as_ref().unwrap();
    let backend = m.gemma3_vl.as_mut().unwrap();
    let eos = tok
        .special_token_id("<end_of_turn>")
        .unwrap_or_else(|| backend.eos_token());
    let prompt_tokens = tok.encode(&framed);
    if prompt_tokens.is_empty() {
        emit_error_with_id(stdout, id, "empty prompt after framing".to_string());
        return;
    }

    let prefill_t0 = Instant::now();
    if let Err(e) = SimpleAr::prefill(backend, gpu, &prompt_tokens) {
        emit_error_with_id(stdout, id, format!("gemma3-vl prefill: {e}"));
        return;
    }
    let _ = gpu.hip.device_synchronize();
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;

    let no_images: [&[u8]; 0] = [];
    let mut ctx = GenerateCtx {
        id,
        prompt: "",
        temperature: temp,
        top_p,
        max_tokens,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        max_think_tokens: 0,
        stop_sequences: &[],
        images: &no_images,
        sink: stdout,
    };
    let n = prompt_tokens.len();
    let result = decode_loop_with_timing(
        gpu,
        backend,
        tok,
        eos,
        &mut ctx,
        n,
        n,
        DecodeLoopTiming {
            prefill_ms: Some(prefill_ms),
        },
    );
    drop(ctx);
    if let Err(e) = result {
        emit_error_with_id(stdout, id, format!("gemma3-vl serve: {e}"));
    }
}
