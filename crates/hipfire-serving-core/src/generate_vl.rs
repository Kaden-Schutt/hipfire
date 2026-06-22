// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision-language generate paths: image + prompt → encoder → text decode.
//!
//! `generate_vl` drives the Qwen3.5-VL family (SigLIP-style tower +
//! `qwen35_vl::vision_forward` spliced into the qwen35 text decoder);
//! `generate_vl_dots_ocr` drives the dots.ocr family (Qwen2 text decoder + its
//! own vision tower). Both decode the image, build the multimodal prompt, and
//! run the per-token sample/stream loop. Extracted verbatim from the former
//! `main.rs` monolith (no behavior change); items called from `main.rs` are
//! `pub`.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use base64::Engine;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35_vl::{image, qwen35_vl};
use hipfire_generate::loop_guard::StopReason;
use hipfire_generate::sampler::SamplerConfig;
use hipfire_generate::{GenerateVLParams, ImageSource};
use hipfire_prompt as prompt_frame;
use hipfire_runtime::arch::GenerateCtx;
use hipfire_runtime::sampler;

use crate::events::{emit_committed_event, write_error};
use crate::model::{effective_raw, LoadedModel};
use crate::output_filter::{block_attractor_unclosed_cpu, loop_guard_from_runtime_config};

/// Qwen3.5-VL generate path: decode the image (base64 or path), run the vision
/// tower (`qwen35_vl::vision_forward`), splice the image embeddings into the
/// qwen35 text prompt, then run the per-token sample/stream loop (EOS filter +
/// loop guard + attractor blocking), emitting `token`/`done` events.
pub fn generate_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    // Keep host-side VL sampling deterministic per request instead of carrying
    // the global CPU sampler state across daemon calls.
    hipfire_runtime::sampler::reset_cpu_sampler_rng(0x13579BDF);

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
        encode_only: _, // qwen35-vl path always decodes
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

    if m.eviction.is_none() && m.seq_pos + prompt_est + max_tokens > m.max_seq {
        eprintln!(
            "[daemon/vl] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        if let Some(ref dn) = m.dn_state {
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
        if let Some(kv) = m.kv_cache.as_mut() {
            kv.compact_offset = 0;
        }
    }

    if m.eviction.is_none() && prompt_est + max_tokens > m.max_seq {
        write_error(
            stdout,
            id,
            &format!(
                "request size ({} tokens) exceeds loaded KV budget ({})",
                prompt_est + max_tokens,
                m.max_seq,
            ),
        );
        return;
    }

    let config = m.q35_config.as_ref().unwrap();
    let vision_weights = m.vision_weights.as_ref().unwrap();
    let weights = m.q35_weights.as_ref().unwrap();
    let scratch = m.q35_scratch.as_ref().unwrap();
    // Compute the raw-prompt flag before the mutable kv/dn borrows below.
    let vl_raw = effective_raw(m);
    let kv = m.kv_cache.as_mut().unwrap();
    let dn = m.dn_state.as_mut().unwrap();

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

    let prompt_tokens = prompt_frame::ChatFrame {
        tokenizer,
        system: if m.seq_pos == 0 { system_prompt } else { None },
        user: "", // unused: we pass tokens directly via build_with_user_tokens
        assistant_prefix: prompt_frame::AssistantPrefix::Plain, // VL always uses Plain
        raw: vl_raw,
    }
    .build_with_user_tokens(&user_body);

    // KV-budget guard — physical_cap without eviction, absolute window with.
    // Mirrors the textual generate() contract; reserves trailer slots so
    // natural im_end termination can still write the ChatML \n.
    let trailer = nl.len();
    let absolute_pos_vl = m.seq_pos + kv.compact_offset;
    let over_budget = if m.eviction.is_none() {
        m.seq_pos + prompt_tokens.len() + max_tokens + trailer > m.physical_cap
    } else {
        absolute_pos_vl + prompt_tokens.len() + max_tokens + trailer > m.max_seq
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
    };
    let vl_cfg = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
    };
    let mut next_token = sampler::sample_cpu(&mut logits, &[], &vl_cfg_first);
    let t_prefill = Instant::now();
    let mut generated = 0;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut think_count: usize = 0;
    let mut prev_in_think: bool = false;

    // N-gram loop detector — mirrors the text path. Catches answer-phase
    // attractor loops that the think cap and repeat penalty miss.
    let loop_guard = loop_guard_from_runtime_config();

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

        if let Some(StopReason::NgramRepeat { count, .. }) = loop_guard.check(&streamed_tokens) {
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
        hipfire_runtime::sampler::apply_ngram_block(&mut logits, &m.conversation_tokens);
        if let Some((open, close)) = think_pair {
            block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
        }

        next_token = sampler::sample_cpu(&mut logits, &m.conversation_tokens, &vl_cfg);

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
                    qwen35::forward_scratch(gpu, weights, config, t, m.seq_pos, kv, dn, scratch)
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
                prev_in_think = false;
                if generated >= max_tokens {
                    break;
                }
                logits = gpu.download_f32(&scratch.logits).unwrap();
                if let Some((open, close)) = think_pair {
                    block_attractor_unclosed_cpu(
                        &mut logits,
                        &m.conversation_tokens,
                        open,
                        close,
                        20,
                        2,
                    );
                }
                next_token = sampler::sample_cpu(&mut logits, &m.conversation_tokens, &vl_cfg);
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
/// dots.ocr generate path: same shape as [`generate_vl`] but for the dots.ocr
/// family — its own vision tower feeding a Qwen2 text decoder
/// (`qwen2::forward_step*`).
pub fn generate_vl_dots_ocr(
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
    if prompt_ids.len() + max_tokens > max_seq {
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
            // dots.ocr text weights are Q8_0 (q8.hfq).
            if let Err(e) =
                gpu.embedding_lookup_q8(&weights.text.token_embd, &emb_scratch, token, dim)
            {
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

    // 6. Greedy decode, streaming in the daemon JSONL protocol.
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

/// Decode the multimodal inputs of a gemma3-vl request into owned, raw encoded
/// image bytes — one entry per image/frame, in prompt order. The bytes are what
/// `Gemma3VlBackend::serve` consumes (each goes through `preprocess_image_bytes`
/// → SigLIP), so they are the *encoded* container bytes (PNG/JPEG), not pixels.
///
/// Routing (in precedence order): a `video` path (or an `image` path that
/// `hipfire_media::is_video`) expands to up to `max_frames` uniformly-sampled PNG
/// frames in slice order; a non-empty `images` list reads each path as one frame
/// (true multi-image — the clean multi-frame case with distinct images); a still
/// `image` path reads as one frame; a base64 payload (optional `data:…;base64,`
/// prefix stripped) decodes as one frame. `max_frames == 0` means "all frames".
/// Precedence: `video` > `images[]` > `image` > `image_base64`.
pub fn decode_vl_frames(
    image: Option<&str>,
    images: &[&str],
    image_base64: Option<&str>,
    video: Option<&str>,
    max_frames: usize,
) -> Result<Vec<Vec<u8>>, String> {
    if let Some(vp) = video {
        return hipfire_media::decode_frames(Path::new(vp), max_frames);
    }
    if !images.is_empty() {
        let mut frames = Vec::with_capacity(images.len());
        for ip in images {
            let p = Path::new(ip);
            // A video in the list still expands to its frames, concatenated in
            // list order with the other images.
            if hipfire_media::is_video(p) {
                frames.extend(hipfire_media::decode_frames(p, max_frames)?);
            } else {
                frames.push(
                    std::fs::read(p).map_err(|e| format!("gemma3-vl: read image {ip}: {e}"))?,
                );
            }
        }
        return Ok(frames);
    }
    if let Some(ip) = image {
        let p = Path::new(ip);
        if hipfire_media::is_video(p) {
            return hipfire_media::decode_frames(p, max_frames);
        }
        let bytes = std::fs::read(p).map_err(|e| format!("gemma3-vl: read image {ip}: {e}"))?;
        return Ok(vec![bytes]);
    }
    if let Some(b64) = image_base64 {
        let raw = match b64.strip_prefix("data:") {
            Some(rest) => rest
                .split_once(',')
                .map(|(_, after)| after)
                .ok_or_else(|| "malformed data URL: missing ',' separator".to_string())?,
            None => b64,
        };
        let bytes = Engine::decode(&base64::engine::general_purpose::STANDARD, raw)
            .map_err(|e| format!("gemma3-vl: base64 decode failed: {e}"))?;
        return Ok(vec![bytes]);
    }
    Err("gemma3-vl: no image/video provided".to_string())
}

/// Gemma3-VL (medgemma, arch_id=13) generate path.
///
/// Builds the gemma3 chat-framed prompt with one `<start_of_image>` marker per
/// frame (HF wraps every image as `\n\n<start_of_image>\n\n`). Each frame's
/// projected rows are resolved through the **vision-embedding cache** (Goal 1):
/// `xxh64(frame bytes)` namespaced by the model + vision-config identity → on a
/// hit the SigLIP + projector encode is skipped entirely; on a miss the frame is
/// encoded via [`Gemma3VlBackend::encode_image`] and inserted. The concatenated
/// rows are spliced + decoded by [`Gemma3VlBackend::serve_with_embeds`], which
/// streams the daemon's exact `token`/`done` schema through `decode_loop`.
///
/// `frames` is daemon-decoded (see [`decode_vl_frames`]): one raw encoded image
/// per image/video-frame, in order. `params.image_source` is unused here (the
/// bytes arrive via `frames`); the rest of `params` supplies id/prompt/system/
/// sampling. Decode uses `decode_loop` with `params.repeat_penalty` (the daemon
/// defaults arch 13 to 1.3), which breaks the near-duplicate-frame attractor.
pub fn generate_vl_gemma3(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
    frames: &[Vec<u8>],
) {
    let id = params.id;

    // Frame the prompt. `<bos>` / `<start_of_turn>` / `<end_of_turn>` /
    // `<start_of_image>` all round-trip through `tok.encode` (they are registered
    // special tokens), so the whole chat frame can be expressed as text and the
    // backend's `tok.encode(ctx.prompt)` reproduces the example's token stream.
    let mut framed = String::from("<bos><start_of_turn>user\n");
    if let Some(sys) = params.system_prompt.filter(|s| !s.is_empty()) {
        // gemma3 has no system role — HF folds system content into the user turn.
        framed.push_str(sys);
        framed.push_str("\n\n");
    }
    for _ in 0..frames.len() {
        framed.push_str("\n\n<start_of_image>\n\n");
    }
    framed.push_str(params.prompt);
    framed.push_str("<end_of_turn>\n<start_of_turn>model\n");

    // Disjoint field borrows: tokenizer (shared) + backend (mut).
    if m.tokenizer.is_none() {
        write_error(stdout, id, "gemma3-vl: tokenizer not loaded");
        return;
    }
    if m.gemma3_vl.is_none() {
        write_error(
            stdout,
            id,
            "gemma3-vl: backend not loaded (arch 13 not active)",
        );
        return;
    }
    // Model identity for the cache namespace (captured before the backend borrow).
    let model_path = m.model_path.clone();
    let tok = m.tokenizer.as_ref().unwrap();
    let backend = m.gemma3_vl.as_mut().unwrap();

    let th = backend.vl_cfg.text_hidden_size;
    let mm = backend.vl_cfg.mm_tokens_per_image;
    let n_images = frames.len();

    // Vision-embedding cache (Goal 1): key = xxh64(frame bytes) namespaced by the
    // model + vision-config identity, so cached projected rows never alias across
    // models/towers. A hit skips SigLIP + projector entirely for that frame.
    let cache = open_vision_cache();
    let namespace = {
        let vc = &backend.vl_cfg;
        format!(
            "{model_path}|gemma3vl|img{}|p{}|mm{}|th{}",
            vc.vision.image_size, vc.vision.patch_size, vc.mm_tokens_per_image, vc.text_hidden_size
        )
    };

    let mut img_embeds: Vec<f32> = Vec::with_capacity(n_images * mm * th);
    let mut hits = 0usize;
    for frame in frames {
        let key = cache
            .as_ref()
            .map(|_| hipfire_vision_cache::CacheKey::new(&namespace, frame.as_slice()));
        // Probe: on a hit, splice the cached rows and skip the encode.
        if let (Some(c), Some(k)) = (cache.as_ref(), key.as_ref()) {
            match c.get(k) {
                Ok(Some(emb)) => {
                    img_embeds.extend_from_slice(&emb.data);
                    hits += 1;
                    continue;
                }
                Ok(None) => {}
                Err(e) => eprintln!("[gemma3-vl] vision cache get failed: {e}"),
            }
        }
        // Miss (or cache disabled): encode through SigLIP + projector, then insert.
        let rows = match backend.encode_image(gpu, frame.as_slice()) {
            Ok(r) => r,
            Err(e) => {
                write_error(stdout, id, &format!("gemma3-vl encode: {e}"));
                return;
            }
        };
        if let (Some(c), Some(k)) = (cache.as_ref(), key.as_ref()) {
            let emb = hipfire_vision_cache::CachedEmbedding::new(mm, th, rows.clone());
            if let Err(e) = c.insert(k, &emb) {
                eprintln!("[gemma3-vl] vision cache insert failed: {e}");
            }
        }
        img_embeds.extend_from_slice(&rows);
    }
    if let Some(c) = cache.as_ref() {
        let s = c.stats();
        eprintln!(
            "[gemma3-vl] vision cache: {hits}/{n_images} frame(s) hit (lifetime hits={}, misses={})",
            s.hits, s.misses
        );
    }

    // Cache-warm mode: the encode + cache insert above is all we need; skip the
    // (expensive, per-token) LM decode and report the frames processed.
    if params.encode_only {
        let _ = writeln!(
            stdout,
            r#"{{"type":"done","id":"{id}","tokens":0,"frames":{n_images},"cache_hits":{hits},"encode_only":true}}"#
        );
        let _ = stdout.flush();
        return;
    }

    // `serve_with_embeds` consumes `img_embeds` directly and ignores `ctx.images`.
    let no_images: [&[u8]; 0] = [];
    let mut ctx = GenerateCtx {
        id,
        prompt: &framed,
        temperature: params.temp,
        top_p: params.top_p,
        max_tokens: params.max_tokens,
        repeat_penalty: params.repeat_penalty,
        repeat_window: params.repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        max_think_tokens: params.max_think_tokens,
        stop_sequences: &[],
        images: &no_images,
        sink: stdout,
    };
    let result = backend.serve_with_embeds(gpu, tok, &mut ctx, &img_embeds, n_images);
    // `ctx` mutably borrows `stdout`; drop it before reusing `stdout` for errors.
    drop(ctx);
    if let Err(e) = result {
        write_error(stdout, id, &format!("gemma3-vl serve: {e}"));
    }
}

/// Open the vision-embedding cache from the environment, or `None` when disabled
/// / unopenable. `HIPFIRE_VISION_CACHE=0` disables it;
/// `HIPFIRE_VISION_CACHE_DIR` overrides the path (default
/// `${HIPFIRE_DIR:-$HOME/.hipfire}/cache/vision`);
/// `HIPFIRE_VISION_CACHE_MAX_BYTES` sets the byte budget (default 4 GiB).
fn open_vision_cache() -> Option<hipfire_vision_cache::VisionCache> {
    if std::env::var("HIPFIRE_VISION_CACHE").ok().as_deref() == Some("0") {
        return None;
    }
    let dir = std::env::var("HIPFIRE_VISION_CACHE_DIR").unwrap_or_else(|_| {
        let base = std::env::var("HIPFIRE_DIR").unwrap_or_else(|_| {
            format!(
                "{}/.hipfire",
                std::env::var("HOME").unwrap_or_else(|_| ".".into())
            )
        });
        format!("{base}/cache/vision")
    });
    let max_bytes = std::env::var("HIPFIRE_VISION_CACHE_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(4 * 1024 * 1024 * 1024);
    match hipfire_vision_cache::VisionCache::open(&dir, max_bytes) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[gemma3-vl] vision cache disabled (open '{dir}' failed: {e})");
            None
        }
    }
}

#[cfg(test)]
mod gemma3_vl_tests {
    use super::decode_vl_frames;
    use base64::Engine;

    #[test]
    fn decode_vl_frames_errors_without_input() {
        let err = decode_vl_frames(None, &[], None, None, 0).unwrap_err();
        assert!(err.contains("no image/video"), "got: {err}");
    }

    #[test]
    fn decode_vl_frames_decodes_base64_single_frame() {
        let payload = b"\x89PNG\r\n\x1a\n-pretend-bytes";
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let frames = decode_vl_frames(None, &[], Some(&b64), None, 0).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], payload);
    }

    #[test]
    fn decode_vl_frames_strips_data_url_prefix() {
        let payload = b"jpeg-ish";
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let data_url = format!("data:image/png;base64,{b64}");
        let frames = decode_vl_frames(None, &[], Some(&data_url), None, 0).unwrap();
        assert_eq!(frames[0], payload);
    }

    #[test]
    fn decode_vl_frames_rejects_malformed_data_url() {
        // `data:` prefix but no comma separator.
        let err = decode_vl_frames(None, &[], Some("data:image/png;base64"), None, 0).unwrap_err();
        assert!(err.contains("data URL"), "got: {err}");
    }

    #[test]
    fn decode_vl_frames_reads_still_image_path_as_one_frame() {
        let dir = std::env::temp_dir().join(format!("hfvl-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("frame.png");
        std::fs::write(&p, b"not-really-png-but-bytes").unwrap();
        let frames = decode_vl_frames(Some(p.to_str().unwrap()), &[], None, None, 0).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], b"not-really-png-but-bytes");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decode_vl_frames_errors_on_missing_image_path() {
        let err = decode_vl_frames(Some("/no/such/frame.png"), &[], None, None, 0).unwrap_err();
        assert!(err.contains("read image"), "got: {err}");
    }

    #[test]
    fn decode_vl_frames_reads_images_list_in_order() {
        let dir = std::env::temp_dir().join(format!("hfvl-multi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p0 = dir.join("a.png");
        let p1 = dir.join("b.png");
        std::fs::write(&p0, b"image-a").unwrap();
        std::fs::write(&p1, b"image-b").unwrap();
        let list = [p0.to_str().unwrap(), p1.to_str().unwrap()];
        // `images[]` takes precedence over a single `image`, preserving order.
        let frames = decode_vl_frames(Some("/ignored"), &list, None, None, 0).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], b"image-a");
        assert_eq!(frames[1], b"image-b");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
