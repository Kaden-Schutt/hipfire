// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The Qwen3.5 / Qwen3 (llama) text generate paths — the daemon's default
//! autoregressive decode.
//!
//! `generate` is the central AR text path the request loop calls for the
//! qwen35/llama families: prompt framing, prefill, the per-token
//! sample/EOS-filter/loop-guard/stream loop, multi-turn KV + eviction, and
//! optional PFlash prefill compression. The qwen35 spec-decode fast paths layer
//! on top: `generate_mtp` (MTP head), `generate_dflash` (DFlash diffusion
//! drafter + DDTree), and `generate_multi` (multi-GPU pipeline-parallel).
//! Extracted verbatim from the former `main.rs` monolith (no behavior change);
//! items called from `main.rs` are `pub`.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_generate::eos_filter::EosFilter;
use hipfire_generate::loop_guard::StopReason;
use hipfire_generate::sampler::{collect_unclosed_attractor_blocks, SamplerConfig};
use hipfire_model::is_qwen35_family_arch_id;
use hipfire_prompt as prompt_frame;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::sampler;

use crate::events::{emit_committed_event, emit_filter_action, write_error};
use crate::evidence::{
    write_daemon_moe_router_evidence, write_daemon_runtime_oneshot_evidence,
    DaemonMoeRouterHistogramGuard,
};
#[cfg(feature = "arch-lfm2moe")]
use crate::generate_arch::generate_lfm2moe;
use crate::generate_arch::{generate_deepseek4, generate_minimax, generate_qwen2};
use crate::model::{effective_raw, LoadedModel};
use crate::output_filter::chat_output_filter;
use crate::output_filter::{chat_output_filter_from_profile, loop_guard_from_runtime_config};
use crate::request::ThinkMode;
use crate::session::{qwen35_restore_or_error, Qwen35RequestSessionState};

/// MTP (Multi-Token Prediction) spec-decode generate path for Qwen3.5/3.6.
///
/// Selected when a qwen35 model carries a co-trained MTP head (bundled
/// `-mq4+mtp.hfq` or sibling `.mtp.hfq`), `mtp_mode != "off"`, and no DFlash
/// drafter is loaded. Uses the **non-tree** `mtp_spec::spec_step_mtp` (linear
/// autoregressive K-token draft) because the FP32 DeltaNet state the small
/// models run on does not support tree-mode replay (see TODO.md /
/// `qwen35::default_state_quant`). The MTP head is lazy-loaded from
/// `m.model_path` per request and freed at function end (cache-on-slot is a
/// future optimization).
#[allow(clippy::too_many_arguments)]
pub fn generate_mtp(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
) {
    use hipfire_arch_qwen35::mtp_head::{self, MtpKvMode};
    use hipfire_arch_qwen35::mtp_spec::{self, MtpSpecState};
    use hipfire_arch_qwen35::qwen35;
    use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};

    let tokenizer = m.tokenizer.as_ref().unwrap();

    // Prompt build mirrors generate_dflash: Jinja chat template when enabled,
    // else the hand-rolled ChatFrame::Plain scaffold.
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let prompt_tokens: Vec<u32> = if try_jinja {
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
        match render_result {
            Ok(rendered) => tokenizer.encode(&rendered),
            Err(e) => {
                eprintln!("[daemon] jinja render failed in mtp path ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
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
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build()
    };

    let im_end = tokenizer.encode("<|im_end|>");
    let im_end_token = if im_end.len() == 1 {
        Some(im_end[0])
    } else {
        None
    };

    // Fresh target state — MTP runs its own prefill from position 0.
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    {
        let dn = m.dn_state.as_ref().unwrap();
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

    // Assemble a transient ModelSlot, mirroring generate_dflash's take/putback.
    let target_config = m.q35_config.as_ref().unwrap().clone();
    let weights = m.q35_weights.take().expect("q35 weights");
    let kv_cache = m.kv_cache.take().expect("kv cache");
    let dn_state = m.dn_state.take().expect("dn state");
    let scratch = m.q35_scratch.take().expect("q35 scratch");
    macro_rules! putback {
        ($t:expr) => {{
            m.q35_weights = Some($t.weights);
            m.kv_cache = Some($t.kv_cache);
            m.dn_state = Some($t.dn_state);
            m.q35_scratch = Some($t.scratch);
        }};
    }
    let hfq = match HfqFile::open(Path::new(&m.model_path)) {
        Ok(h) => h,
        Err(e) => {
            write_error(stdout, id, &format!("reopen model: {e}"));
            m.q35_weights = Some(weights);
            m.kv_cache = Some(kv_cache);
            m.dn_state = Some(dn_state);
            m.q35_scratch = Some(scratch);
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
    };

    let max_seq_total = m.physical_cap.max(m.max_seq);
    let max_n = m.mtp_k.clamp(1, 8);
    let eos_token = target.config.eos_token;
    let dim = target.config.dim;

    if prompt_tokens.len() + max_tokens * (max_n + 1) + 16 > max_seq_total {
        write_error(
            stdout,
            id,
            &format!(
                "prompt ({}) + max_tokens ({}) × (max_n+1) won't fit in max_seq {}",
                prompt_tokens.len(),
                max_tokens,
                max_seq_total
            ),
        );
        putback!(target);
        return;
    }

    // Load the MTP head from the bundled trailer (or a sibling .mtp.hfq).
    let head = match mtp_head::load_mtp_head_bundled(Path::new(&m.model_path), gpu, max_seq_total) {
        Ok(Some(h)) => h,
        Ok(None) => {
            // No bundled trailer — try a sibling "<stem>.mtp.hfq".
            let sidecar = m.model_path.replace(".hfq", ".mtp.hfq");
            match mtp_head::load_mtp_head(Path::new(&sidecar), gpu, max_seq_total) {
                Ok(h) => h,
                Err(e) => {
                    write_error(stdout, id, &format!("load mtp head: {e}"));
                    putback!(target);
                    return;
                }
            }
        }
        Err(e) => {
            write_error(stdout, id, &format!("load bundled mtp head: {e}"));
            putback!(target);
            return;
        }
    };

    let kv_mode = m
        .q35_kv_mode
        .as_deref()
        .and_then(|s| MtpKvMode::parse(s).ok())
        .unwrap_or(MtpKvMode::Q8);
    let mut state =
        match MtpSpecState::new_for_slot_with_kv_mode(gpu, &target, &head, max_n, kv_mode) {
            Ok(s) => s,
            Err(e) => {
                write_error(stdout, id, &format!("alloc MtpSpecState: {e}"));
                head.free_gpu(gpu);
                putback!(target);
                return;
            }
        };

    let t0 = Instant::now();

    // Run prefill + spec-decode loop; centralize cleanup after.
    // Returns (hit_eos, generated, cycles, accepted_total, decode_secs).
    let run: Result<(bool, usize, usize, usize, f64), String> = (|| {
        // Prefill the prompt through the batched WMMA path.
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            &prompt_tokens,
            0,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| format!("prefill forward_prefill_batch: {e:?}"))?;

        // Snapshot trunk's post-output-norm hidden at the last prefill position.
        state
            .capture_prev_hidden_from_scratch_tmp(gpu, &target.scratch.tmp, dim)
            .map_err(|e| format!("capture prev_hidden: {e:?}"))?;

        // Seed token = argmax of trunk logits at the last prefill position.
        let logits0 = gpu
            .download_f32(&target.scratch.logits)
            .map_err(|e| format!("download seed logits: {e:?}"))?;
        let mut seed_token = 0u32;
        let mut best = f32::NEG_INFINITY;
        for (i, &v) in logits0.iter().enumerate() {
            if v > best {
                best = v;
                seed_token = i as u32;
            }
        }

        // Streaming state (mirrors generate_dflash).
        let mut streamed_tokens: Vec<u32> = Vec::new();
        let mut bytes_fed_to_filter = 0usize;
        let mut filter = chat_output_filter(m, request_stop_sequences);
        let mut generated = 0usize;
        let mut think_count: usize = 0;
        let mut prev_in_think: bool = false;

        // Helper closure semantics inlined: stream one committed token, return
        // (hit_eos, think_cap_hit).
        let emit_token = |stdout: &mut std::io::Stdout,
                          tok: u32,
                          streamed_tokens: &mut Vec<u32>,
                          bytes_fed_to_filter: &mut usize,
                          filter: &mut EosFilter,
                          think_count: &mut usize,
                          prev_in_think: &mut bool|
         -> (bool, bool) {
            streamed_tokens.push(tok);
            emit_committed_event(
                stdout,
                id,
                tok,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            let all_bytes = tokenizer.decode_bytes(streamed_tokens);
            let new_bytes = &all_bytes[*bytes_fed_to_filter..];
            *bytes_fed_to_filter = all_bytes.len();
            if emit_filter_action(stdout, id, filter.observe(new_bytes)) {
                return (true, false);
            }
            let hit_eos =
                tok == eos_token || im_end_token == Some(tok) || tokenizer.is_terminator(tok);
            let mut think_cap_hit = false;
            if !hit_eos && max_think_tokens > 0 {
                let raw = tokenizer.decode_bytes(streamed_tokens);
                let raw_str = std::str::from_utf8(&raw).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think && !*prev_in_think {
                    *think_count = 0;
                }
                if in_think {
                    *think_count += 1;
                }
                *prev_in_think = in_think;
                if in_think && *think_count >= max_think_tokens {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"token","id":"{}","text":"</think>\n"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    think_cap_hit = true;
                }
            }
            (hit_eos, think_cap_hit)
        };

        // Emit the seed token first.
        let (mut hit_eos, mut think_cap_hit) = emit_token(
            stdout,
            seed_token,
            &mut streamed_tokens,
            &mut bytes_fed_to_filter,
            &mut filter,
            &mut think_count,
            &mut prev_in_think,
        );
        generated += 1;

        let mut last_committed = seed_token;
        let mut cur_pos = prompt_tokens.len();
        let mut cycles = 0usize;
        let mut accepted_total = 0usize;
        let t_decode = Instant::now();

        while !hit_eos && !think_cap_hit && generated < max_tokens {
            if cur_pos + max_n + 1 >= max_seq_total {
                break;
            }
            let result = mtp_spec::spec_step_mtp(
                gpu,
                &mut target,
                &head,
                &mut state,
                cur_pos,
                last_committed,
                eos_token,
            )
            .map_err(|e| format!("spec_step_mtp: {e:?}"))?;
            cycles += 1;
            accepted_total += result.accept_count;

            // result.committed already EXCLUDES the seed (unlike DFlash).
            for &tok in &result.committed {
                if generated >= max_tokens {
                    break;
                }
                let (eos, cap) = emit_token(
                    stdout,
                    tok,
                    &mut streamed_tokens,
                    &mut bytes_fed_to_filter,
                    &mut filter,
                    &mut think_count,
                    &mut prev_in_think,
                );
                generated += 1;
                if eos {
                    hit_eos = true;
                    break;
                }
                if cap {
                    think_cap_hit = true;
                    break;
                }
            }
            if let Some(&last) = result.committed.last() {
                last_committed = last;
            }
            cur_pos += result.advance;
            if result.hit_eos {
                hit_eos = true;
            }
        }
        Ok((
            hit_eos,
            generated,
            cycles,
            accepted_total,
            t_decode.elapsed().as_secs_f64(),
        ))
    })();

    // Cleanup: free MTP state + head, put trunk pieces back on the model.
    state.free_gpu(gpu);
    head.free_gpu(gpu);
    putback!(target);

    match run {
        Ok((_hit_eos, generated, cycles, accepted_total, decode_secs)) => {
            let tok_s = generated as f64 / decode_secs.max(1e-9);
            // τ = committed tokens per spec cycle (excludes the prefill seed).
            let tau = if cycles > 0 {
                generated.saturating_sub(1) as f64 / cycles as f64
            } else {
                0.0
            };
            let _ = writeln!(
                stdout,
                r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"decode_tok_s":{:.1},"mtp":true,"tau":{:.2},"cycles":{},"accepted":{},"max_n":{}}}"#,
                id, generated, tok_s, tok_s, tau, cycles, accepted_total, max_n,
            );
            let _ = stdout.flush();
        }
        Err(e) => {
            write_error(stdout, id, &e);
        }
    }
}

/// DFlash-powered greedy decode. Mirrors `generate`'s ChatML shape and
/// token-streaming output but replaces the AR sample loop with
/// `spec_step_dflash` cycles — each cycle drafts B tokens via the diffusion
/// drafter and verifies them in one target forward, committing accept_len+1 at
/// a time. With DDTree enabled it uses the tree-verify path instead of the
/// linear chain. Single-turn: resets target state at entry (stateless
/// chat-completions contract).
pub fn generate_dflash(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    pflash_bypass_reason: Option<&str>,
    pflash_alpha: Option<f32>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
) {
    use hipfire_arch_qwen35::speculative::{
        spec_step_ddtree_batched, spec_step_ddtree_path_c, spec_step_dflash, ModelSlot,
        ModelSlotConfig, Phase2Snapshots, SpecStats,
    };

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
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.chat_template.is_some();
    let prompt_tokens: Vec<u32> = if try_jinja {
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
                Some(m) => m,
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
                    "[daemon] jinja render failed in dflash path ({e}) — falling back to Plain"
                );
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: prompt,
                    assistant_prefix,
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
            assistant_prefix,
            raw: effective_raw(m),
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

    // Fresh target state — DFlash seed_target_hidden_from_prompt does its own
    // full prefill, so we reset first to avoid double-accounting.
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    {
        let dn = m.dn_state.as_ref().unwrap();
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
    let chat_template_profile = m.chat_template_profile.clone();
    let df = m.dflash.as_mut().unwrap();
    df.target_hidden_host.clear();
    df.draft_scratch.reset_upload_tracking();

    // Assemble a transient ModelSlot for the spec helpers — they both take
    // `&mut ModelSlot`. We own the pieces on LoadedModel individually, so
    // take them, build the ModelSlot, run, then put them back.
    //
    // ModelSlot needs its own HfqFile field but spec_step_dflash doesn't
    // actually touch it. Reopening via mmap is essentially free (few µs).
    let target_config = m.q35_config.as_ref().unwrap().clone();
    let weights = m.q35_weights.take().expect("q35 weights");
    let kv_cache = m.kv_cache.take().expect("kv cache");
    let dn_state = m.dn_state.take().expect("dn state");
    let scratch = m.q35_scratch.take().expect("q35 scratch");
    let hfq = match HfqFile::open(Path::new(&m.model_path)) {
        Ok(h) => h,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"reopen model: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            m.q35_weights = Some(weights);
            m.kv_cache = Some(kv_cache);
            m.dn_state = Some(dn_state);
            m.q35_scratch = Some(scratch);
            return;
        }
    };
    let slot_config = ModelSlotConfig::default();
    let mut target = ModelSlot {
        name: String::from("target"),
        hfq,
        config: target_config,
        weights,
        kv_cache,
        dn_state,
        scratch,
        slot_config,
    };

    let t0 = Instant::now();
    let ctx_capacity = df.ctx_capacity;
    // Capacity checks. With eviction enabled the advertised context window is
    // effectively unbounded (eviction fires between spec cycles), but the
    // *prompt* must still fit in one physical_cap span because
    // seed_target_hidden_from_prompt writes it per-token without chunking.
    let eff_prompt_cap = if m.eviction.is_some() {
        m.physical_cap
    } else {
        ctx_capacity
    };
    if prompt_tokens.len() + df.block_size > eff_prompt_cap {
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
        m.q35_weights = Some(target.weights);
        m.kv_cache = Some(target.kv_cache);
        m.dn_state = Some(target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }
    if m.eviction.is_none() && prompt_tokens.len() + max_tokens + df.block_size > ctx_capacity {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prompt+max_tokens exceeds ctx_capacity {} (enable cask_sidecar for long decode)"}}"#,
            id, ctx_capacity,
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        m.kv_cache = Some(target.kv_cache);
        m.dn_state = Some(target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }

    // Seed target_hidden via the demo's helper — runs a per-token prefill
    // with hidden extraction into hidden_rb, then downloads prompt-length
    // worth of rows into target_hidden_host. The draft's first forward
    // uses these as context.
    if let Err(e) = speculative::seed_target_hidden_from_prompt(
        gpu,
        &mut target,
        &mut df.hidden_rb,
        &mut df.target_hidden_host,
        &prompt_tokens,
    ) {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"prefill: {}"}}"#,
            id, e
        );
        let _ = stdout.flush();
        m.q35_weights = Some(target.weights);
        m.kv_cache = Some(target.kv_cache);
        m.dn_state = Some(target.dn_state);
        m.q35_scratch = Some(target.scratch);
        return;
    }
    // Prime the draft's GPU target_hidden buffer from the prompt rows so the
    // first spec step can skip the CPU→GPU upload of the whole context.
    if let Err(e) = speculative::scatter_hidden_block_to_interleaved(
        gpu,
        &df.hidden_rb,
        &df.draft_scratch.target_hidden,
        0,
        prompt_tokens.len(),
        prompt_tokens.len(),
    ) {
        eprintln!("[dflash] scatter failed: {e} — falling back to per-cycle upload");
    }
    df.draft_scratch.uploaded_target_hidden_rows = prompt_tokens.len();
    df.draft_scratch.target_hidden_abs_positions = (0..prompt_tokens.len() as i32).collect();

    // First emit = target's argmax at the final prompt position. seed_target_hidden
    // already ran the per-token forward for every prompt token; its scratch.logits
    // holds the post-prompt logits.
    let first_logits = match gpu.download_f32(&target.scratch.logits) {
        Ok(v) => v,
        Err(e) => {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"download logits: {}"}}"#,
                id, e
            );
            let _ = stdout.flush();
            m.q35_weights = Some(target.weights);
            m.kv_cache = Some(target.kv_cache);
            m.dn_state = Some(target.dn_state);
            m.q35_scratch = Some(target.scratch);
            return;
        }
    };
    let first_token = first_logits
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

    let t_prefill = Instant::now();

    // Decode loop — spec_step_dflash returns a committed batch per cycle.
    let mut emitted: Vec<u32> = vec![first_token];
    let mut streamed_tokens: Vec<u32> = Vec::new();
    // `bytes_fed_to_filter` is the index into the freshly-decoded byte
    // stream past which we have not yet handed bytes to the filter.
    // The filter owns UTF-8 boundary buffering and any future arch
    // quirks (Gemma 4 marker holdback, strip-think, byte-level stop_at);
    // see crates/engine/src/eos_filter.rs.
    let mut bytes_fed_to_filter = 0usize;
    let mut filter =
        chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);
    let mut position = prompt_tokens.len();
    let mut seed_token = first_token;
    let mut stats = SpecStats::new(df.block_size);
    // max_think_tokens enforcement state (mirrors the AR path).
    let mut think_count: usize = 0;
    let mut prev_in_think = false;
    let mut generated = 0usize;

    // Post-prefill compaction (FlashCASK pattern from dflash_spec_demo).
    // If the prompt already filled past budget+beta, compact once before
    // entering the spec loop so the first spec_step writes at physical slot
    // `budget`. compact_offset is maintained on target.kv_cache; subsequent
    // forwards inside spec_step_dflash read it for RoPE phase automatically.
    if let Some(ref ev) = m.eviction {
        if let Some(res) = ev.maybe_evict(gpu, &mut target.kv_cache, position).unwrap() {
            let pre_phys = position;
            eprintln!(
                "[dflash] post-prefill evict: {} -> {} (compact_offset={})",
                pre_phys, res.new_physical, target.kv_cache.compact_offset,
            );
            position = res.new_physical;
            if !res.retain_mask.is_empty() {
                let _ = speculative::apply_eviction_retain_to_draft(
                    gpu,
                    &mut df.draft_scratch,
                    &res.retain_mask,
                    df.draft_config.num_extract(),
                    df.draft_config.hidden,
                    pre_phys,
                );
                speculative::compact_target_hidden_host(
                    &mut df.target_hidden_host,
                    &res.retain_mask,
                    df.draft_config.num_extract(),
                    df.draft_config.hidden,
                );
            }
        }
    }

    // Emit the first token immediately so TTFT is the prefill time.
    streamed_tokens.push(first_token);
    emit_committed_event(
        stdout,
        id,
        first_token,
        streamed_tokens.len() - 1,
        t0.elapsed().as_millis() as u64,
    );
    let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
    let new_bytes = &all_bytes[bytes_fed_to_filter..];
    bytes_fed_to_filter = all_bytes.len();
    let first_filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));
    generated += 1;

    // First-token EOS guard. The first token is already emitted above; if
    // it is itself a terminator, do not seed another drafted/verified block.
    // The committed-tail check inside the loop applies the same terminator
    // test to every subsequent token.
    let first_token_is_eos = first_token == target.config.eos_token
        || im_end_token == Some(first_token)
        || tokenizer.is_terminator(first_token);

    let mut rng_state: u64 = 0x13579BDFu64;

    // Resolve `HIPFIRE_DDTREE_PATH_C` ONCE before the decode loop. The
    // previous version re-read the env-var on every spec cycle which
    // is microseconds of waste on a hot path. Validate eagerly: invalid
    // values fall back to spec_step_ddtree_batched (the documented
    // behavior) but warn so misconfigurations don't fail silently.
    //
    // Only meaningful when DDTree itself is enabled (HIPFIRE_DDTREE_BUDGET).
    // `phase1` runs Step 1 only (linear main-path verify); `phase2` adds
    // the lazy branch FA-only re-verify (Steps 2+3). See
    // `docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd`.
    let path_c_mode_owned: Option<&'static str> = match std::env::var("HIPFIRE_DDTREE_PATH_C").ok()
    {
        None => None,
        Some(s) if s.is_empty() => None,
        Some(s) if s == "phase1" => Some("phase1"),
        Some(s) if s == "phase2" => Some("phase2"),
        Some(s) => {
            if df.ddtree.is_some() {
                eprintln!(
                    "[hipfire-daemon] HIPFIRE_DDTREE_PATH_C={:?} is not 'phase1' or 'phase2'. \
                     Falling back to spec_step_ddtree_batched.",
                    s
                );
            }
            None
        }
    };

    // Fast path exit conditions (mirrors the dflash_spec_demo outer loop).
    while !first_filter_stop && !first_token_is_eos && generated < max_tokens {
        if position + df.block_size >= ctx_capacity {
            break;
        }

        // Dispatch: when DDTree is configured (HIPFIRE_DDTREE_BUDGET set
        // at startup), route through `spec_step_ddtree_batched`. Otherwise
        // keep the existing chain-mode `spec_step_dflash` path. The two
        // produce the same `SpecStepResult` shape so the rest of the loop
        // is unchanged. Note: `spec_step_ddtree_batched` is greedy-only
        // (temp=0); the daemon currently runs at 0.0_f32 so this matches.
        let path_c_mode = path_c_mode_owned;
        let step_result = if let Some(dd) = df.ddtree.as_mut() {
            if path_c_mode == Some("phase1") || path_c_mode == Some("phase2") {
                let phase2_snaps = if path_c_mode == Some("phase2") {
                    Some(Phase2Snapshots {
                        parent_pre_snap: &mut dd.path_c_parent_pre_snap,
                        main_end_snap: &mut dd.path_c_main_end_snap,
                    })
                } else {
                    None
                };
                spec_step_ddtree_path_c(
                    gpu,
                    &mut target,
                    &df.draft_weights,
                    &df.draft_config,
                    &mut df.draft_scratch,
                    &mut df.hidden_rb,
                    &mut df.target_hidden_host,
                    &mut df.target_snap,
                    &mut df.gdn_tape,
                    &df.verify_scratch,
                    position,
                    seed_token,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                    phase2_snaps,
                )
            } else {
                spec_step_ddtree_batched(
                    gpu,
                    &mut target,
                    &df.draft_weights,
                    &df.draft_config,
                    &mut df.draft_scratch,
                    &mut df.hidden_rb,
                    &mut df.target_hidden_host,
                    &mut df.target_snap,
                    &mut dd.post_seed_snap,
                    &mut df.gdn_tape,
                    &dd.scratch,
                    &df.verify_scratch,
                    position,
                    seed_token,
                    None, // ctx_slice = full history
                    dd.budget,
                    dd.topk,
                )
            }
        } else {
            spec_step_dflash(
                gpu,
                &mut target,
                &df.draft_weights,
                &df.draft_config,
                &mut df.draft_scratch,
                &mut df.hidden_rb,
                &mut df.target_hidden_host,
                &mut df.target_snap,
                &df.verify_scratch,
                position,
                seed_token,
                None, // ctx_slice = full history
                Some(&mut df.gdn_tape),
                0.0_f32, // temperature
                &mut rng_state,
                None, // block_size override
                None, // ngram_cache
                &emitted,
                0.0_f32, // cactus_delta
                None,    // pld_spine
                1.0_f32, // repeat_penalty (off)
                0,       // repeat_window
            )
        };
        let step = match step_result {
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
        stats.record(&step);
        let committed_tail: Vec<u32> = step.committed.iter().skip(1).copied().collect();

        let mut hit_eos = false;
        let mut think_cap_hit = false;
        for &tok in &committed_tail {
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
            if emit_filter_action(stdout, id, filter.observe(new_bytes)) {
                hit_eos = true;
                break;
            }
            generated += 1;
            if tok == target.config.eos_token
                || im_end_token == Some(tok)
                || tokenizer.is_terminator(tok)
            {
                hit_eos = true;
                break;
            }

            // max_think_tokens enforcement (mirrors the AR path). Track
            // <think>/<⁄think> in decoded text and count tokens inside.
            if max_think_tokens > 0 {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if in_think && !prev_in_think {
                    think_count = 0;
                }
                if in_think {
                    think_count += 1;
                }
                prev_in_think = in_think;

                if in_think && think_count >= max_think_tokens {
                    // Force-close: emit </think>\n and break out of this batch.
                    // Unlike the AR path we can't splice into the KV cache mid-
                    // spec-cycle, so we just stream the close text and break.
                    // The next request will start fresh.
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
        let rollback = speculative::spec_rollback_parity_decision_for_step(position, &step);
        if !rollback.allow_single_session {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"DFlash rollback parity guard failed: {}"}}"#,
                id, rollback.reason
            );
            let _ = stdout.flush();
            break;
        }
        debug_assert!(!rollback.allow_multi_request_verify_batch);
        position = rollback.next_position;
        seed_token = step.bonus_token;
        // Per-cycle eviction (FlashCASK). Fires whenever current physical
        // has grown to budget+β since the last compaction. No-op when
        // physical < budget+β, so non-firing cycles pay only the check cost.
        if let Some(ref ev) = m.eviction {
            if let Some(res) = ev.maybe_evict(gpu, &mut target.kv_cache, position).unwrap() {
                let pre_phys = position;
                position = res.new_physical;
                if !res.retain_mask.is_empty() {
                    let _ = speculative::apply_eviction_retain_to_draft(
                        gpu,
                        &mut df.draft_scratch,
                        &res.retain_mask,
                        df.draft_config.num_extract(),
                        df.draft_config.hidden,
                        pre_phys,
                    );
                    speculative::compact_target_hidden_host(
                        &mut df.target_hidden_host,
                        &res.retain_mask,
                        df.draft_config.num_extract(),
                        df.draft_config.hidden,
                    );
                }
            }
        }
        if hit_eos || think_cap_hit {
            break;
        }
    }

    // Put target state back on LoadedModel so the next request sees fresh
    // (reset) state. We zero DN/kv on entry anyway, but we still need the
    // ownership back.
    m.q35_weights = Some(target.weights);
    m.kv_cache = Some(target.kv_cache);
    m.dn_state = Some(target.dn_state);
    m.q35_scratch = Some(target.scratch);
    m.seq_pos = position;
    m.conversation_tokens = emitted.clone();

    let t_end = Instant::now();
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
    let tau = if stats.cycles > 0 {
        stats.accepted_tokens as f64 / stats.cycles as f64
    } else {
        0.0
    };
    // Per PRD §3.1, when PFlash bypassed (e.g. dflash_decode_active for
    // this branch) the `done` object must surface the bypass reason and
    // alpha alongside the dflash perf metrics. Build a small fragment
    // when both are available; otherwise empty for back-compat.
    let pflash_done_field = match (pflash_bypass_reason, pflash_alpha) {
        (Some(r), Some(a)) => format!(
            r#","pflash":{{"bypass_reason":"{}","alpha":{:.6}}}"#,
            r.replace('"', "'"),
            a,
        ),
        _ => String::new(),
    };
    let _ = writeln!(
        stdout,
        r#"{{"type":"done","id":"{}","tokens":{},"tok_s":{:.1},"prefill_tokens":{},"prefill_ms":{:.1},"prefill_tok_s":{:.1},"decode_tok_s":{:.1},"ttft_ms":{:.1},"dflash":true,"tau":{:.2},"cycles":{}{}}}"#,
        id,
        generated,
        tok_s,
        prompt_tokens.len(),
        prefill_s * 1000.0,
        prefill_tok_s,
        decode_tok_s,
        prefill_s * 1000.0,
        tau,
        stats.cycles,
        pflash_done_field,
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
/// Multi-GPU pipeline-parallel generate (`pp > 1`): drives the per-stage
/// `Gpus` orchestrator + `Qwen35ScratchSet` through prefill and the per-token
/// decode loop, streaming tokens. Single-session; DFlash/MTP/CASK/VL are refused
/// into this path at load time.
pub fn generate_multi(
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
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    request_stop_sequences: &[String],
) {
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    if m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        if let (Some(ref dn), Some(ref mut gpus), Some(ref la)) = (
            m.dn_state.as_ref(),
            m.pp_gpus.as_mut(),
            m.pp_dn_la_to_device.as_ref(),
        ) {
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
        if let Some(kv) = m.kv_cache.as_mut() {
            kv.compact_offset = 0;
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
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let try_jinja = jinja_enabled && m.seq_pos == 0 && m.chat_template.is_some();
    let new_tokens = if try_jinja {
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
                Some(m) => m,
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
                eprintln!("[daemon] jinja render failed in pp path ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: if m.seq_pos == 0 { system_prompt } else { None },
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: if m.seq_pos == 0 { system_prompt } else { None },
            user: "",
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build_with_user_tokens(&q_tokens)
    };

    let trailer = nl.len();
    if m.seq_pos + new_tokens.len() + max_tokens + trailer > m.physical_cap {
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
    let chat_template_profile = m.chat_template_profile.clone();

    let config = m.q35_config.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch_set = m.pp_scratch_set.as_ref().unwrap();
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();
    let gpus = m.pp_gpus.as_mut().unwrap();

    let dev_last = gpus.output_device;
    let vocab_size = config.vocab_size;
    let repeat_buf_cap =
        (scratch_set.per_device[dev_last].repeat_buf.buf.size() / 4).min(repeat_window);

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
    collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
    let cfg0 = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window: repeat_buf_cap,
        presence_penalty,
        frequency_penalty,
        blocked_tokens: blocked0,
    };
    let tok0 = {
        let s_last = &scratch_set.per_device[dev_last];
        let g_last = &mut gpus.devices[dev_last];
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
    };
    let t_prefill = Instant::now();
    let mut next_token = tok0;

    let mut generated = 0usize;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut bytes_fed_to_filter = 0usize;
    let mut filter =
        chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);
    let mut alert_fired = false;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;
    let loop_guard = loop_guard_from_runtime_config();

    while generated < max_tokens {
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
        let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

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
        if filter_stop {
            break;
        }
        if im_end_token == Some(next_token) {
            break;
        }
        if tokenizer.is_terminator(next_token) {
            break;
        }

        // max_think_tokens enforcement: same decoded-text scan as pp=1.
        if max_think_tokens > 0 {
            let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
            let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
            let open_idx = raw_str.rfind("<think>");
            let close_idx = raw_str.rfind("</think>");
            let in_think = match (open_idx, close_idx) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                _ => false,
            };
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

            if in_think && think_count >= max_think_tokens {
                let close_tokens = tokenizer.encode("</think>\n");
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
                    let _ = emit_filter_action(stdout, id, filter.observe(new_bytes));
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
        if let Some(StopReason::NgramRepeat { count, .. }) = loop_guard.check(&streamed_tokens) {
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
            let in_think = match (raw_str.rfind("<think>"), raw_str.rfind("</think>")) {
                (Some(o), Some(c)) => o > c,
                (Some(_), None) => true,
                _ => false,
            };
            if !in_think {
                let _ = writeln!(
                    stdout,
                    r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                    id
                );
                let _ = stdout.flush();
                let ngram_scope = &m.conversation_tokens[ngram_scope_start..];
                let mut blocked: Vec<u32> = Vec::new();
                collect_unclosed_attractor_blocks(
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
                };
                next_token = {
                    let s_last = &scratch_set.per_device[dev_last];
                    let g_last = &mut gpus.devices[dev_last];
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
                };
                continue;
            }
            let nudge_tokens = tokenizer.encode(budget_alert_text);
            let budget_left = max_tokens.saturating_sub(generated);
            let nudge_len = nudge_tokens.len().min(budget_left);
            let need_kv = m.seq_pos + nudge_len + (max_tokens - generated - nudge_len) + nl.len();
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
                    let _ = emit_filter_action(stdout, id, filter.observe(new_bytes2));
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
        collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked);
        let cfg = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked,
        };
        next_token = {
            let s_last = &scratch_set.per_device[dev_last];
            let g_last = &mut gpus.devices[dev_last];
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
        };
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
/// The default autoregressive text generate path for the qwen35 / qwen3 (llama)
/// families, and the central text dispatcher: frames the prompt (chat template
/// or raw), prefills (optionally PFlash-compressed), and runs the per-token
/// sample → EOS-filter → loop-guard → stream loop with multi-turn KV + eviction.
/// Delegates to the spec-decode fast paths ([`generate_mtp`], [`generate_dflash`],
/// [`generate_multi`]) and to the non-qwen35 arch paths ([`generate_deepseek4`]
/// etc.) when the loaded model calls for them.
pub fn generate(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    drafter_gpu: Option<&mut rdna_compute::Gpu>,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    repeat_penalty: f32,
    repeat_window: usize,
    presence_penalty: f32,
    frequency_penalty: f32,
    budget_alert_at_tok: usize,
    budget_alert_text: &str,
    max_think_tokens: usize,
    assistant_prefix: prompt_frame::AssistantPrefix,
    pflash_state: Option<&mut hipfire_arch_qwen35::pflash::PflashState>,
    pflash_cfg: Option<&hipfire_arch_qwen35::pflash::PflashConfig>,
    tools: Option<&[serde_json::Value]>,
    messages_history: Option<&[prompt_frame::Message]>,
    think_mode: ThinkMode,
    prefill_already_done: bool,
    prefilled_prompt_tokens: Option<usize>,
    request_stop_sequences: &[String],
    evidence_dir: Option<&str>,
) {
    // Seed the process-global CPU sampler RNG for this request. CPU fallback and
    // grammar/VL-style sampling should not inherit RNG state from prior requests.
    hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);

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
            prefill_already_done,
        );
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
        return;
    }
    if m.arch_id == 10 {
        // arch_id=10 (MiniMax-M2). Minimal AR bring-up — same shape as the
        // qwen2 / deepseek4 short-circuits above. PFlash / DFlash / VL /
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
            repeat_penalty,
            repeat_window,
            think_mode,
        );
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
    #[cfg(feature = "arch-lfm2moe")]
    if m.arch_id == 11 {
        // arch_id=11 (LFM2.5-MoE). Minimal AR bring-up — same shape as the
        // qwen2 / deepseek4 / minimax short-circuits above. PFlash / DFlash /
        // VL / multi-GPU / sampler-budget / grammar / tools-execution all
        // bypass. We honour `system_prompt`, `temp`, `top_p`, and (via
        // JinjaChatFrame) `messages_history` + `tools` rendering; spec-decode
        // / MTP / grammar are out of scope for the scaffold.
        let _ = (
            budget_alert_at_tok,
            budget_alert_text,
            assistant_prefix,
            pflash_state,
            pflash_cfg,
            repeat_penalty,
            repeat_window,
            think_mode,
        );
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
            request_stop_sequences,
        );
        return;
    }
    // DFlash fast path -- only when a draft model is loaded AND temperature is
    // effectively 0 (DFlash is greedy-only in this integration). Skip the
    // normal AR sampling setup entirely.
    //
    // Exception: thinking-on + max_think_tokens currently needs the AR path.
    // DFlash's budget cap can close/strip the think span but does not yet
    // continue into visible answer text after the forced close. AR already
    // splices </think> through KV and continues generation, so route budgeted
    // thinking requests there until DFlash continuation is implemented.
    let budgeted_thinking_needs_ar = max_think_tokens > 0
        && !matches!(assistant_prefix, prompt_frame::AssistantPrefix::ClosedThink);
    if prefill_already_done && m.dflash.is_some() {
        write_error(
            stdout,
            id,
            "prefill_already_done is not supported on DFlash-loaded models",
        );
        return;
    }
    if prefill_already_done && pflash_state.is_some() {
        write_error(
            stdout,
            id,
            "prefill_already_done is not supported when PFlash state is loaded",
        );
        return;
    }

    if m.dflash.is_some()
        && temp <= 1e-6
        && is_qwen35_family_arch_id(m.arch_id)
        && !budgeted_thinking_needs_ar
    {
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
            request_stop_sequences,
        );
        // Silence unused-variable warnings for the params DFlash doesn't
        // consume (top_p / repeat penalties are AR-only sampling knobs;
        // pflash_state is bypassed on the DFlash decode path).
        let _ = (
            top_p,
            repeat_penalty,
            repeat_window,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
        );
        return;
    }

    // MTP spec-decode: qwen35 model with a co-trained MTP head, no DFlash
    // drafter, greedy, mtp_mode enabled. Uses the non-tree spec_step_mtp
    // (FP32 DeltaNet state is tree-incompatible — see generate_mtp / TODO.md).
    if m.dflash.is_none()
        && m.mtp_weights_present
        && m.mtp_mode != "off"
        && temp <= 1e-6
        && is_qwen35_family_arch_id(m.arch_id)
        && !prefill_already_done
        && !budgeted_thinking_needs_ar
    {
        generate_mtp(
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
            request_stop_sequences,
        );
        let _ = (
            top_p,
            repeat_penalty,
            repeat_window,
            budget_alert_at_tok,
            budget_alert_text,
            pflash_state,
        );
        return;
    }

    let is_qwen35_ar = is_qwen35_family_arch_id(m.arch_id);
    let mut q35_session = if is_qwen35_ar {
        match Qwen35RequestSessionState::take_from_loaded(m, gpu) {
            Ok(session) => Some(session),
            Err(e) => {
                write_error(stdout, id, &format!("qwen35 request session state: {e}"));
                return;
            }
        }
    } else {
        None
    };

    // Auto-reset on multi-turn rollover. When eviction is active (operator
    // enabled cask_sidecar at load), the physical buffer is bounded by
    // budget+beta+safety regardless of conversation length, so reset never
    // needs to fire — eviction reclaims slots after each token. When eviction
    // is OFF, physical grows unbounded up to max_seq; reset when we'd overrun.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let prompt_est = tokenizer.encode(prompt).len() + 20;
    let current_seq_pos = q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos);
    if !prefill_already_done
        && m.eviction.is_none()
        && current_seq_pos + prompt_est + max_tokens > m.max_seq
    {
        eprintln!(
            "[daemon] context full ({}/{}) — resetting conversation",
            current_seq_pos, m.max_seq
        );
        if let Some(session) = q35_session.as_mut() {
            session.reset(gpu);
        } else {
            m.seq_pos = 0;
            m.conversation_tokens.clear();
            if let Some(kv) = m.llama_kv.as_mut() {
                kv.compact_offset = 0;
            }
        }
    }

    // `nl` is needed for the trailer write after natural <|im_end|>
    // termination; `im_end` derives the EOS-check token id. Other
    // ChatML scaffolding tokens are now built inside hipfire-prompt.
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
    pub fn pflash_done_fragment(
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
            q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos),
            raw_q_tokens.len(),
            drafter_gpu.is_some()
        );
    }
    let q_tokens = if let (Some(state), Some(cfg)) = (pflash_state, pflash_cfg) {
        let seq_pos = q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos);
        if seq_pos == 0 {
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
    let jinja_enabled = std::env::var("HIPFIRE_JINJA_CHAT").ok().as_deref() == Some("1");
    let seq_pos_for_prompt = if prefill_already_done {
        0
    } else {
        q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos)
    };
    let try_jinja = jinja_enabled && seq_pos_for_prompt == 0 && m.chat_template.is_some();
    let new_tokens = if try_jinja {
        let template = m.chat_template.as_ref().unwrap();
        let frame = prompt_frame::JinjaChatFrame {
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
            let synthesized: Vec<prompt_frame::Message>;
            let messages_slice: &[prompt_frame::Message] = match messages_history {
                Some(m) => m,
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
                eprintln!("[daemon] jinja render failed ({e}) — falling back to Plain");
                prompt_frame::ChatFrame {
                    tokenizer,
                    system: system_prompt,
                    user: "",
                    assistant_prefix,
                    raw: effective_raw(m),
                }
                .build_with_user_tokens(&q_tokens)
            }
        }
    } else {
        prompt_frame::ChatFrame {
            tokenizer,
            system: if seq_pos_for_prompt == 0 {
                system_prompt
            } else {
                None
            },
            user: "", // unused: we pass tokens directly via build_with_user_tokens
            assistant_prefix,
            raw: effective_raw(m),
        }
        .build_with_user_tokens(&q_tokens)
    };

    // KV-budget guard. Without eviction the physical buffer is the hard cap;
    // we must fit prefill + generation + trailer in one allocation. With
    // eviction, physical is bounded by physical_cap regardless of total tokens
    // — the chunked prefill below calls maybe_evict between chunks, and the
    // decode loop evicts after every token. The only ceiling under eviction is
    // the advertised context window (max_seq) — refuse requests that would
    // overflow it in absolute position terms (current absolute + new).
    let trailer = nl.len();
    let current_seq_pos = q35_session.as_ref().map(|s| s.seq_pos).unwrap_or(m.seq_pos);
    let budget_prefill_tokens = if prefill_already_done {
        0
    } else {
        new_tokens.len()
    };
    let absolute_pos = if let Some(session) = q35_session.as_ref() {
        session.seq_pos + session.kv_cache.compact_offset
    } else {
        m.seq_pos + m.llama_kv.as_ref().map(|kv| kv.compact_offset).unwrap_or(0)
    };
    if m.eviction.is_none() {
        if current_seq_pos + budget_prefill_tokens + max_tokens + trailer > m.physical_cap {
            let _ = writeln!(
                stdout,
                r#"{{"type":"error","id":"{}","message":"request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > physical_cap={} — reload model with a larger max_seq"}}"#,
                id, current_seq_pos, budget_prefill_tokens, max_tokens, trailer, m.physical_cap
            );
            let _ = stdout.flush();
            if let Some(session) = q35_session.take() {
                qwen35_restore_or_error(stdout, id, m, gpu, session);
            }
            return;
        }
    } else if absolute_pos + budget_prefill_tokens + max_tokens + trailer > m.max_seq {
        let _ = writeln!(
            stdout,
            r#"{{"type":"error","id":"{}","message":"request exceeds advertised context window: absolute={} + prefill={} + max_tokens={} + trailer={} > max_seq={}"}}"#,
            id, absolute_pos, budget_prefill_tokens, max_tokens, trailer, m.max_seq
        );
        let _ = stdout.flush();
        if let Some(session) = q35_session.take() {
            qwen35_restore_or_error(stdout, id, m, gpu, session);
        }
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

    if is_qwen35_family_arch_id(m.arch_id) {
        // Qwen3.5 / Qwen3.5-MoE — multi-turn: prefill only the NEW turn tokens,
        // continuing from session.seq_pos (KV cache + DeltaNet state are cumulative)
        let mut session = q35_session.take().expect("qwen35 request session state");
        if prefill_already_done {
            let current_position = session.seq_pos + session.kv_cache.compact_offset;
            let expected_position = prefilled_prompt_tokens.unwrap_or(new_tokens.len());
            if current_position != expected_position {
                write_error(
                    stdout,
                    id,
                    &format!(
                        "prefill_already_done requested but active session position {} does not match expected prefilled prompt token count {}",
                        current_position,
                        expected_position
                    ),
                );
                qwen35_restore_or_error(stdout, id, m, gpu, session);
                return;
            }
        }
        let config = m.q35_config.as_ref().unwrap();
        let weights = m.q35_weights.as_ref().unwrap();
        let scratch = m.q35_scratch.as_ref().unwrap();
        let kv = &mut session.kv_cache;
        let dn = &mut session.dn_state;
        let moe_router_histogram = DaemonMoeRouterHistogramGuard::start(evidence_dir, config);

        // Prefill this turn's tokens via the batched prefill entry point.
        // On gfx11+ for MQ4/HFQ4/MQ6/HFQ6 weights this hits the WMMA GEMM
        // fast path; other archs fall back to dp2 / FP16-packed / scalar
        // variants. The one sequential hotspot inside is the gated_delta_net
        // Q8 state update (N sequential per-token calls per LA layer, byte-
        // exact with decode to keep the quality gate green).
        //
        // Note: forward_prefill_batch launches HIP kernels asynchronously.
        // The t_prefill mark below lives AFTER the first sample_top_p, whose
        // D2H readback of tok0 forces a device sync — that's the point at
        // which the first token is actually ready to stream. Placing the
        // mark earlier captures CPU-dispatch time, which under-reports
        // prefill by a large factor (prefill_tok_s ~5–10× too optimistic).
        //
        // Under eviction: chunk prefill to the (budget+beta) eviction window
        // and call `maybe_evict` between chunks so physical never exceeds
        // physical_cap. Chunk size caps out at physical capacity available —
        // when physical is at post-evict `budget`, a full `beta`-sized chunk
        // can run before the next eviction fires.
        if !prefill_already_done {
            if let Some(ref ev) = m.eviction {
                let window = ev.budget() + ev.beta();
                let mut remaining: &[u32] = &new_tokens;
                while !remaining.is_empty() {
                    let space = window.saturating_sub(session.seq_pos).max(1);
                    let chunk_len = remaining.len().min(space);
                    let (chunk, rest) = remaining.split_at(chunk_len);
                    qwen35::forward_prefill_batch(
                        gpu,
                        weights,
                        config,
                        chunk,
                        session.seq_pos,
                        kv,
                        dn,
                        scratch,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap();
                    session.seq_pos += chunk_len;
                    if let Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                    {
                        session.seq_pos = new_phys;
                    }
                    remaining = rest;
                }
            } else {
                qwen35::forward_prefill_batch(
                    gpu,
                    weights,
                    config,
                    &new_tokens,
                    session.seq_pos,
                    kv,
                    dn,
                    scratch,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
                session.seq_pos += new_tokens.len();
            }
            session.conversation_tokens.extend_from_slice(&new_tokens);
        }

        // ngram scope for the repeat penalty: ONLY generated tokens (never the
        // prompt). Prior design included the user's prompt as an anti-loop
        // anchor, but that penalizes the very tokens we're asked to recall
        // (names, numbers, facts) under MQ4/MQ6 quantizations that are more
        // RP-sensitive than llama.cpp's Q4_K. First sample: empty scope (no
        // generated tokens yet); subsequent samples: generated-so-far only.
        let ngram_scope_start = if prefill_already_done {
            session
                .conversation_tokens
                .len()
                .saturating_sub(session.prefilled_generated_suffix_len)
        } else {
            session.conversation_tokens.len()
        };
        session.prefilled_generated_suffix_len = 0;

        // Generate. GPU-side sampling eliminates per-token logits download +
        // CPU softmax + CPU repeat penalty. Closes the 2× gap between raw
        // bench throughput and daemon throughput.
        //
        // Kernel signature reads `repeat_tokens[0..repeat_window]`, so we
        // only need to upload the tokens that will actually be read — no
        // need to clear the buffer between calls. The upload is on the same
        // stream as the sample kernel launch, so the copy and compute pipeline
        // naturally.
        let vocab_size = config.vocab_size;
        let mut rng_state: u32 = 0x13579BDFu32;
        let repeat_buf_cap = (scratch.repeat_buf.buf.size() / 4).min(repeat_window);

        // Build the list of paired (open, close) attractor pairs once;
        // collect_unclosed_attractor_blocks decides per-call
        // which openers (if any) trip the depth threshold.
        let attractor_pairs: Vec<(u32, u32)> = tool_call_pair
            .into_iter()
            .chain(think_pair.into_iter())
            .collect();

        // First sample: use conversation so far as scope.
        let ngram_scope = &session.conversation_tokens[ngram_scope_start..];
        // #111 attractor block: empty `ngram_scope` on first sample (no
        // generated tokens yet), so the unclosed-depth is always 0 and
        // `blocked` is empty. Still call collect_* for symmetry with
        // the loop body, in case a future change moves this block into
        // a multi-step warmup.
        let mut blocked0: Vec<u32> = Vec::new();
        collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked0);
        let cfg0 = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty,
            // Window is bounded by the GPU repeat_buf capacity. Pre-PR3 code did this
            // bound by setting `scope_start = len - repeat_buf_cap`
            // and passing `scope.len()` to the kernel; we let
            // sampler::sample do the same `min(window, buf_cap)`
            // internally.
            repeat_window: repeat_buf_cap,
            presence_penalty,
            frequency_penalty,
            blocked_tokens: blocked0,
        };
        let tok0 = sampler::sample(
            gpu,
            &scratch.logits,
            &scratch.sample_buf,
            &scratch.repeat_buf,
            vocab_size,
            ngram_scope,
            &cfg0,
            &mut rng_state,
        );
        // First token is ready (sample_top_p's D2H forces GPU sync). This is
        // the user-observable "time to first token" boundary — prefill above,
        // decode loop below.
        let t_prefill = Instant::now();
        let mut next_token = tok0;

        let mut generated = 0;
        let mut streamed_tokens: Vec<u32> = Vec::new();
        // `bytes_fed_to_filter` is the index into the freshly-decoded
        // byte stream past which we have not yet handed bytes to the
        // filter. The filter owns UTF-8 boundary buffering and any
        // future arch quirks (Gemma 4 marker holdback, strip-think,
        // byte-level stop_at); see crates/engine/src/eos_filter.rs.
        let mut bytes_fed_to_filter = 0usize;
        let mut filter = chat_output_filter(m, request_stop_sequences);
        let mut alert_fired = false;
        // max_think_tokens enforcement state. think_count increments only
        // while we observe ourselves to be inside a `<think>...</think>`
        // block via the same decoded-text scan budget_alert uses. When the
        // cap is hit we splice "</think>\n" into the stream (KV write +
        // stdout emit + advance generated) so the model finishes thinking
        // and commits to an answer with the remaining max_tokens budget.
        // Re-armable: if the model later opens another <think> in the same
        // turn (rare) the counter resets and the cap re-fires.
        let mut think_count: usize = 0;
        let mut prev_in_think: bool = false;

        // N-gram loop detector: track 4-gram token sequences. When any
        // 4-gram repeats more than `ngram_loop_threshold` times in the
        // last `ngram_window` tokens, force EOS. This catches answer-phase
        // repetition loops that the think cap and repeat penalty miss.
        // Operates on token IDs (no decode overhead).
        // Implementation lives in `hipfire-generate` loop_guard; defaults read from
        // HIPFIRE_NGRAM_LOOP_THRESHOLD (default 8, 0 = disabled) and
        // HIPFIRE_NGRAM_WINDOW (default 256). See loop_guard.rs.
        let loop_guard = loop_guard_from_runtime_config();

        // `while` instead of `for 0..max_tokens` so budget-alert injection
        // (which increments `generated` beyond the iteration count) can't
        // push generated past max_tokens: each loop start rechecks the cap.
        while generated < max_tokens {
            generated += 1;
            session.conversation_tokens.push(next_token);
            streamed_tokens.push(next_token);
            emit_committed_event(
                stdout,
                id,
                next_token,
                streamed_tokens.len() - 1,
                t0.elapsed().as_millis() as u64,
            );
            // Incremental UTF-8 + filter routing: feed only the new
            // bytes since last call, let the filter buffer any partial
            // codepoint or marker prefix until disambiguated.
            let all_bytes = tokenizer.decode_bytes(&streamed_tokens);
            let new_bytes = &all_bytes[bytes_fed_to_filter..];
            bytes_fed_to_filter = all_bytes.len();
            let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

            // Write this token's K/V to the cache FIRST so the next turn
            // always starts from a fully-written context. Breaking before
            // forward_scratch used to leave a hole at the im_end/eos
            // position — the next turn then attended over zero-init K/V
            // at that slot.
            //
            // Under eviction, session.seq_pos is the *physical* write slot; we
            // advance and call maybe_evict immediately so the next write
            // never overruns physical_cap. compact_offset bookkeeping on
            // the cache itself keeps RoPE phase correct across evictions.
            if let Err(e) = qwen35::forward_scratch(
                gpu,
                weights,
                config,
                next_token,
                session.seq_pos,
                kv,
                dn,
                scratch,
            ) {
                write_error(
                    stdout,
                    id,
                    &format!("qwen35 decode forward_scratch failed: {e:?}"),
                );
                qwen35_restore_or_error(stdout, id, m, gpu, session);
                return;
            }
            session.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                if let Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                {
                    session.seq_pos = new_phys;
                }
            }
            if filter_stop {
                break;
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

            // max_think_tokens enforcement. Track whether we're inside an
            // open <think>...</think> block and how many tokens we've
            // emitted there. When the cap is hit, splice "</think>\n" into
            // the stream (KV write + stdout emit + advance generated) so
            // the model commits to an answer with the remaining budget.
            // Same decoded-text scan budget_alert uses; counter is
            // incremented per-iteration only when we're still inside.
            if max_think_tokens > 0 {
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let open_idx = raw_str.rfind("<think>");
                let close_idx = raw_str.rfind("</think>");
                let in_think = match (open_idx, close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
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

                if in_think && think_count >= max_think_tokens {
                    // Force-close. Encode the close sequence and run each
                    // token through the KV write + emit path the same way
                    // a normally-sampled token does. This ensures the
                    // model's next sample is conditioned on having "said"
                    // </think>\n itself, instead of seeing a hidden-state
                    // discontinuity. Respect max_tokens — clip the close
                    // sequence if not enough room remains and bail.
                    let close_tokens = tokenizer.encode("</think>\n");
                    let budget_left = max_tokens.saturating_sub(generated);
                    let take = close_tokens.len().min(budget_left);
                    for &t in &close_tokens[..take] {
                        qwen35::forward_scratch(
                            gpu,
                            weights,
                            config,
                            t,
                            session.seq_pos,
                            kv,
                            dn,
                            scratch,
                        )
                        .unwrap();
                        session.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult {
                                new_physical: new_phys,
                                ..
                            }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                            {
                                session.seq_pos = new_phys;
                            }
                        }
                        session.conversation_tokens.push(t);
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
                        let _ = emit_filter_action(stdout, id, filter.observe(new_bytes));
                        generated += 1;
                    }
                    think_count = 0;
                    prev_in_think = false;
                    if generated >= max_tokens {
                        break;
                    }
                }
            }

            // N-gram loop detector: check if any 4-gram in the recent window
            // repeats excessively. When detected, emit an info message and
            // force EOS to prevent wasting the remaining token budget on
            // repetitive output. Logic lives in `hipfire-generate` loop_guard.
            if let Some(StopReason::NgramRepeat { count, .. }) = loop_guard.check(&streamed_tokens)
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

            // Budget-alert injection: once we hit the configured token count,
            // splice the nudge text into the stream. Tokens are emitted to
            // stdout (so the client sees them) AND forward-fed through the KV
            // cache (so the model's next sample is conditioned on having
            // "said" them itself). Injected tokens count against `max_tokens`
            // — we never exceed the caller's requested budget — so we clip
            // the nudge if not enough room remains, and break out of the
            // outer loop if the budget is fully spent after injection.
            if !alert_fired
                && budget_alert_at_tok > 0
                && generated >= budget_alert_at_tok
                && !budget_alert_text.is_empty()
            {
                alert_fired = true;
                // Only inject while the model is inside an open <think> block.
                // The whole point of the feature is to nudge the model's
                // reasoning; firing past </think> just graffities the visible
                // answer with a system-alert string. Check the raw decoded
                // text rather than token IDs since <think> tokenizes as a
                // multi-token sequence in Qwen3.5's vocab.
                let raw_so_far = tokenizer.decode_bytes(&streamed_tokens);
                let raw_str = std::str::from_utf8(&raw_so_far).unwrap_or("");
                let think_open_idx = raw_str.rfind("<think>");
                let think_close_idx = raw_str.rfind("</think>");
                let in_think = match (think_open_idx, think_close_idx) {
                    (Some(o), Some(c)) => o > c,
                    (Some(_), None) => true,
                    _ => false,
                };
                if !in_think {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"info","id":"{}","message":"budget_alert skipped: not inside an open <think> block"}}"#,
                        id
                    );
                    let _ = stdout.flush();
                    // Fall through — resample next token as normal
                    let ngram_scope = &session.conversation_tokens[ngram_scope_start..];
                    let mut blocked: Vec<u32> = Vec::new();
                    collect_unclosed_attractor_blocks(
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
                    };
                    next_token = sampler::sample(
                        gpu,
                        &scratch.logits,
                        &scratch.sample_buf,
                        &scratch.repeat_buf,
                        vocab_size,
                        ngram_scope,
                        &cfg,
                        &mut rng_state,
                    );
                    continue;
                }
                let nudge_tokens = tokenizer.encode(budget_alert_text);
                let budget_left = max_tokens.saturating_sub(generated);
                let nudge_len = nudge_tokens.len().min(budget_left);
                // KV headroom check — don't run past physical_cap. If we don't
                // have room for the clipped nudge, skip entirely rather than
                // emit a partial nudge that poisons the trajectory. Under
                // eviction the physical check is trivially satisfied (budget
                // always holds post-evict), but we still respect the check for
                // the non-eviction path.
                let need_kv =
                    session.seq_pos + nudge_len + (max_tokens - generated - nudge_len) + nl.len();
                if nudge_len > 0 && (m.eviction.is_some() || need_kv <= m.physical_cap) {
                    for &tok in &nudge_tokens[..nudge_len] {
                        session.conversation_tokens.push(tok);
                        streamed_tokens.push(tok);
                        emit_committed_event(
                            stdout,
                            id,
                            tok,
                            streamed_tokens.len() - 1,
                            t0.elapsed().as_millis() as u64,
                        );
                        // Emit the injected token's text to stdout so the client
                        // sees it as part of the stream (will be inside <think>
                        // if that's the current state, and get stripped client-
                        // side just like any other think token).
                        let all_bytes2 = tokenizer.decode_bytes(&streamed_tokens);
                        let new_bytes2 = &all_bytes2[bytes_fed_to_filter..];
                        bytes_fed_to_filter = all_bytes2.len();
                        let _ = emit_filter_action(stdout, id, filter.observe(new_bytes2));
                        if let Err(e) = qwen35::forward_scratch(
                            gpu,
                            weights,
                            config,
                            tok,
                            session.seq_pos,
                            kv,
                            dn,
                            scratch,
                        ) {
                            write_error(
                                stdout,
                                id,
                                &format!("qwen35 budget-alert forward_scratch failed: {e:?}"),
                            );
                            qwen35_restore_or_error(stdout, id, m, gpu, session);
                            return;
                        }
                        session.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            if let Some(hipfire_runtime::triattn::EvictionResult {
                                new_physical: new_phys,
                                ..
                            }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                            {
                                session.seq_pos = new_phys;
                            }
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
                // Respect max_tokens: if injection used the remainder, bail
                // before sampling another model token.
                if generated >= max_tokens {
                    break;
                }
            }

            // Decide which paired-opener tokens (if any) trip the depth
            // threshold over a 20-token window. #111 attractor block —
            // cheap when not tripped, ~5 µs per blocked token when
            // tripped (single 4-byte H2D into the logits buffer
            // performed inside sampler::sample).
            let ngram_scope = &session.conversation_tokens[ngram_scope_start..];
            let mut blocked: Vec<u32> = Vec::new();
            collect_unclosed_attractor_blocks(ngram_scope, &attractor_pairs, 20, 2, &mut blocked);
            let cfg = SamplerConfig {
                temperature: temp,
                top_p,
                repeat_penalty,
                repeat_window: repeat_buf_cap,
                presence_penalty,
                frequency_penalty,
                blocked_tokens: blocked,
            };
            // GPU sample: reads scratch.logits (already on GPU), writes
            // token+rng to scratch.sample_buf. Blocks only on the 8-byte
            // D2H readback inside sampler::sample.
            next_token = sampler::sample(
                gpu,
                &scratch.logits,
                &scratch.sample_buf,
                &scratch.repeat_buf,
                vocab_size,
                ngram_scope,
                &cfg,
                &mut rng_state,
            );
        }
        // session.seq_pos is already the "next physical write slot" — advanced
        // per-token in the decode loop above, and evicted back down to
        // `budget` whenever maybe_evict fired. No post-loop fix-up needed.

        // ChatML requires \n after <|im_end|>. Run it through forward so KV cache
        // and DeltaNet state stay in sync with seq_pos.
        if im_end_token == Some(*session.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty()
        {
            for &t in &nl {
                if let Err(e) = qwen35::forward_scratch(
                    gpu,
                    weights,
                    config,
                    t,
                    session.seq_pos,
                    kv,
                    dn,
                    scratch,
                ) {
                    write_error(
                        stdout,
                        id,
                        &format!("qwen35 ChatML newline forward_scratch failed: {e:?}"),
                    );
                    qwen35_restore_or_error(stdout, id, m, gpu, session);
                    return;
                }
                session.seq_pos += 1;
                if let Some(ref ev) = m.eviction {
                    if let Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    }) = ev.maybe_evict(gpu, kv, session.seq_pos).unwrap()
                    {
                        session.seq_pos = new_phys;
                    }
                }
                session.conversation_tokens.push(t);
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
        if let Some(dir) = evidence_dir {
            write_daemon_runtime_oneshot_evidence(
                dir,
                m,
                gpu,
                id,
                prefill_tokens,
                generated,
                prefill_s,
                decode_s,
                prefill_s * 1000.0,
            );
            if let Some(hist) = moe_router_histogram.take() {
                write_daemon_moe_router_evidence(dir, m, id, hist);
            }
        }
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
        qwen35_restore_or_error(stdout, id, m, gpu, session);
    } else {
        // Qwen3 / LLaMA path -- multi-turn aware
        let chat_template_profile = m.chat_template_profile.clone();
        let config = m.llama_config.as_ref().unwrap();
        let weights = m.llama_weights.as_ref().unwrap();
        let scratch = m.llama_scratch.as_ref().unwrap();
        let kv = m.llama_kv.as_mut().unwrap();

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
        let mut filter =
            chat_output_filter_from_profile(chat_template_profile.as_ref(), request_stop_sequences);

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
            let filter_stop = emit_filter_action(stdout, id, filter.observe(new_bytes));

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
            if filter_stop {
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
        if let Some(dir) = evidence_dir {
            write_daemon_runtime_oneshot_evidence(
                dir,
                m,
                gpu,
                id,
                prefill_tokens,
                generated,
                prefill_s,
                decode_s,
                prefill_s * 1000.0,
            );
        }
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
