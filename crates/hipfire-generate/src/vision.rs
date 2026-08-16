// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision and OCR generation — Qwen3.5-VL and dots.ocr.
//!
//! Per-architecture generation bodies lifted verbatim from `crates/hipfire-daemon/src/main.rs`
//! (wave 5 / D3). See `lib.rs` for layering rationale.

use base64::Engine;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_engine::emit::{emit_active_attempt_error, emit_qwen_ar_cancelled, write_error};
use hipfire_engine::scheduler::block_attractor_unclosed_cpu;
use hipfire_engine::terminal::{
    active_attempt_id, await_client_terminal_commit, check_abort, emit_staged_terminal_done,
    ClientTerminalDecision,
};
use hipfire_loader::{LoadedModel, ModelState};
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_runtime::spec::{PrefillOutcome, Speculator};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// ── Local verbatim copies of daemon-shared helpers ───────────────────────────
// `free_checkpoints` and `emit_committed_event` are shared across all generate
// paths in the daemon (25 and 18 call sites). They cannot live in
// `hipfire-engine` (the former touches `hipfire_arch_qwen35::speculative::DeltaNetSnapshot`,
// an arch type) and the generate crate cannot depend back on the daemon.
// This local copy keeps the moved VL bodies byte-identical without introducing a
// circular dependency. The daemon retains its own identical copy.

fn free_checkpoints(
    cks: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    gpu: &mut rdna_compute::Gpu,
) {
    for (_, snap) in cks.drain(..) {
        snap.free_gpu(gpu);
    }
}

fn emit_committed_event(
    stdout: &mut (impl std::io::Write + ?Sized),
    id: &str,
    tok_id: u32,
    pos: usize,
    t_ms: u64,
) {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var("HIPFIRE_EMIT_TOKEN_IDS").ok().as_deref() == Some("1"));
    if !*ENABLED {
        return;
    }
    // Build through `serde_json::json!` for the same reason
    // `emit_error_with_id` does: `id` is user-supplied and a single `"`
    // or `\` in it would corrupt the line, breaking the client's JSONL
    // parser for every subsequent event on the same connection.
    let envelope = serde_json::json!({
        "type": "committed",
        "id": id,
        "tok_id": tok_id,
        "pos": pos,
        "t_ms": t_ms,
        "attempt_id": active_attempt_id(),
    });
    let _ = writeln!(stdout, "{}", envelope);
}


pub enum ImageSource<'a> {
    Path(&'a str),
    Base64(&'a str),
}

pub struct GenerateVLParams<'a> {
    pub id: &'a str,
    pub prompt: &'a str,
    pub system_prompt: Option<&'a str>,
    pub image_source: ImageSource<'a>,
    pub temp: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub max_think_tokens: usize,
    pub assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
}

pub fn vl_no_eviction_kv_cap(physical_cap: usize, max_seq: usize, adaptive_engaged: bool) -> usize {
    if adaptive_engaged {
        max_seq
    } else {
        physical_cap
    }
}

pub(crate) fn vl_cold_reset_uncommitted(
    gpu: &mut rdna_compute::Gpu,
    dn: &qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
) {
    for s in &dn.s_matrices {
        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
    }
    for s in &dn.s_scales {
        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
    }
    for s in &dn.conv_states {
        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
    }
    for s in &dn.s_ef_residual {
        let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
    }
    kv.compact_offset = 0;
    if let Some(ad) = kv_adaptive.as_mut() {
        if !ad.is_poisoned() {
            ad.reset_with_cache(gpu, kv);
        }
    }
    *seq_pos = 0;
    conversation_tokens.clear();
    free_checkpoints(prefill_checkpoints, gpu);
}

pub(crate) fn vl_adaptive_downshift_fail_closed(
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    gpu: &mut rdna_compute::Gpu,
    kv: &mut hipfire_runtime::llama::KvCache,
    dn: &qwen35::DeltaNetState,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    stdout: &mut std::io::Stdout,
    id: &str,
    phase: &str,
) -> bool {
    let Some(ad) = kv_adaptive.as_mut() else {
        return false;
    };
    let committed = *seq_pos;
    match ad.maybe_downshift(gpu, kv, committed) {
        Ok(applied) => {
            for step in &applied {
                eprintln!(
                    "[adaptive-kv] downshift @ pos {} ({}): {:?} (K={:?} V={:?})",
                    committed, phase, step, ad.cur_k, ad.cur_v
                );
            }
            false
        }
        Err(e) => {
            eprintln!(
                "[adaptive-kv] maybe_downshift error @ pos {} ({}): {:?} — poisoning model",
                committed, phase, e
            );
            // maybe_downshift already poisons on partial failure; cold-reset
            // leaves poison sticky (reset_with_cache skipped when poisoned).
            vl_cold_reset_uncommitted(
                gpu,
                dn,
                kv,
                kv_adaptive,
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
            );
            write_error(
                stdout,
                id,
                &format!("adaptive KV transition failed during {phase}: {e}"),
            );
            true
        }
    }
}

pub(crate) fn vl_forward_fail(
    stdout: &mut std::io::Stdout,
    id: &str,
    phase: &str,
    err: impl std::fmt::Display,
    gpu: &mut rdna_compute::Gpu,
    dn: &qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
) {
    vl_cold_reset_uncommitted(
        gpu,
        dn,
        kv,
        kv_adaptive,
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
    );
    write_error(stdout, id, &format!("VL {phase}: {err}"));
}

pub(crate) fn build_vl_mrope_ctx(
    prompt_ids: &[u32],
    image_pad_id: u32,
    n_visual: usize,
    grid_h: usize,
    grid_w: usize,
    spatial_merge_size: usize,
    base: usize,
    config: &qwen35::Qwen35Config,
) -> Option<qwen35::MropeCtx> {
    if n_visual == 0 || spatial_merge_size == 0 {
        return None;
    }
    let bail = |why: &str| -> Option<qwen35::MropeCtx> {
        eprintln!("[daemon/vl] mrope disabled ({why}) — falling back to 1D positions");
        None
    };
    // Cross-turn cursor continuity is NOT modelled: this context is built per
    // request with prompt positions shifted by `base`, but HF would resume a
    // later turn at `previous_max + 1` (i.e. `base` + the earlier turn's
    // rope_delta), not at `base`. The multi-image-pad bail below only inspects
    // THIS turn's `prompt_ids`, so it cannot catch an image in an earlier turn.
    //
    // The generate handler at daemon.rs:2434 force-resets `m.seq_pos = 0` (and
    // clears `conversation_tokens`) whenever a VL request arrives with
    // `seq_pos > 0` — "Force a reset so VL always starts from a clean KV
    // state." So `base` is always 0 for a VL-with-image request and this guard
    // is expected not to fire. It is here so that if that upstream reset is
    // ever moved, weakened, or bypassed by a new VL entry point, we fail loudly
    // to the 1D path instead of silently mis-positioning every token after the
    // image.
    if base > 0 {
        return bail("base > 0: cross-turn mrope cursor continuity not modelled");
    }
    let Some(start) = prompt_ids.iter().position(|&t| t == image_pad_id) else {
        // The daemon splices these pads itself a few lines above the call
        // site, so `n_visual > 0` with no pad in the prompt is a real
        // inconsistency, not an ordinary text-only request.
        return bail("no <|image_pad|> in the prompt despite n_visual > 0");
    };
    if start + n_visual > prompt_ids.len() {
        return bail("image span runs past the prompt");
    }
    // The span must be exactly one contiguous run of `n_visual` pads.
    if !prompt_ids[start..start + n_visual]
        .iter()
        .all(|&t| t == image_pad_id)
    {
        return bail("image-pad run is not contiguous");
    }
    if prompt_ids[start + n_visual..].contains(&image_pad_id) {
        return bail("more than one image-pad run (multi-image not wired)");
    }
    // Merged grid must account for exactly the spliced visual tokens —
    // otherwise `build_mrope_positions` pushes a different count than the
    // prompt has and every downstream index is off.
    let merged = (grid_h / spatial_merge_size) * (grid_w / spatial_merge_size);
    if merged != n_visual {
        return bail(&format!(
            "merged grid {merged} != spliced visual tokens {n_visual}"
        ));
    }

    let spans = [hipfire_arch_qwen35_vl::mrope::ImageSpan {
        start,
        len: n_visual,
        grid_h,
        grid_w,
    }];
    let built = hipfire_arch_qwen35_vl::mrope::build_mrope_positions(
        prompt_ids.len(),
        &spans,
        spatial_merge_size,
    );
    // Post-condition the library does not assert for us.
    if built.positions.len() != prompt_ids.len() {
        return bail(&format!(
            "build_mrope_positions returned {} positions for {} tokens",
            built.positions.len(),
            prompt_ids.len()
        ));
    }

    let base_i = base as i32;
    let positions: Vec<[i32; 3]> = built
        .positions
        .iter()
        .map(|p| [p[0] + base_i, p[1] + base_i, p[2] + base_i])
        .collect();
    eprintln!(
        "[daemon/vl] mrope: span start={start} len={n_visual} grid={grid_h}x{grid_w} \
         merge={spatial_merge_size} base={base} rope_delta={} section={:?}",
        built.rope_delta, config.mrope_section
    );
    Some(qwen35::MropeCtx::new(
        config,
        base,
        positions,
        built.rope_delta,
    ))
}

pub fn generate_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    // hunt3 M-E: seed the process-global CPU sampler RNG with this request's
    // fixed seed. The VL path samples exclusively via sampler::sample_cpu, which
    // draws from this global; without the per-request reset it carried RNG state
    // across requests (and across earlier text-path requests) → cross-request
    // nondeterminism. Matches the GPU path's u32 (0x13579BDF).
    hipfire_runtime::llama::reset_cpu_sampler_rng(0x13579BDF);
    // INVARIANT: all early returns before the `vision_forward` call (the
    // first expensive GPU allocation in this function) use `write_error`
    // and return without owning any GPU buffers. If you add a GPU
    // allocation above this line, you MUST clean it up on every early
    // return path — the current early returns are safe because they
    // only hold CPU-side data (tokenizer refs, preprocess output).
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
        assistant_prefix,
    } = *params;
    // Adaptive KV poison is sticky until unload/reload. Refuse VL generation so a
    // partial tier transition cannot continue writing into mixed-tier state.
    // Mirror generate() — reset preserves poison, so VL must refuse independently.
    if let Some(ad) = m.kv_adaptive.as_ref() {
        if ad.is_poisoned() {
            let reason = ad
                .poison_reason()
                .unwrap_or("adaptive KV is poisoned; unload/reload required");
            write_error(stdout, id, reason);
            return;
        }
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let vision_config = m.vision_config().unwrap().clone();

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

    if m.eviction.is_none()
        && m.seq_pos
            .saturating_add(prompt_est)
            .saturating_add(max_tokens)
            > m.max_seq
    {
        eprintln!(
            "[daemon/vl] context full ({}/{}) — resetting conversation",
            m.seq_pos, m.max_seq
        );
        m.seq_pos = 0;
        m.conversation_tokens.clear();
        free_checkpoints(&mut m.prefill_checkpoints, gpu);
        free_checkpoints(&mut m.dflash_checkpoints, gpu);
        // Free the speculator's (relocated) checkpoint ring on reset.
        if let Some(s) = m.speculator.as_mut() {
            if let Err(e) = s.reset(gpu) {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("vision context reset failed: {e}"),
                    "gpu",
                    true,
                    false,
                );
                return;
            }
        }
        // VL is qwen35-vl (arch 5/8); its recurrent state lives in the bundle
        // (ModelState::Qwen35), not the always-None m.dn_state/m.kv_cache.
        // Inlined (disjoint field access) because a `&tokenizer` borrow of `m`
        // is live here.
        if let Some(ModelState::Qwen35(b)) = m.state.as_ref() {
            let dn = &b.dn_state;
            for s in &dn.s_matrices {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_scales {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.conv_states {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
            for s in &dn.s_ef_residual {
                let _ = gpu.hip.memset(&s.buf, 0, s.buf.size());
            }
        }
        if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
            b.kv_cache.compact_offset = 0;
        }
        if let Some(ad) = m.kv_adaptive.as_mut() {
            if let Some(ModelState::Qwen35(b)) = m.state.as_mut() {
                ad.reset_with_cache(gpu, &mut b.kv_cache);
            } else {
                ad.reset();
            }
        }
    }

    if m.eviction.is_none() && prompt_est.saturating_add(max_tokens) > m.max_seq {
        write_error(
            stdout,
            id,
            &format!(
                "request size ({} tokens) exceeds loaded KV budget ({})",
                prompt_est.saturating_add(max_tokens),
                m.max_seq,
            ),
        );
        return;
    }

    let ModelState::Qwen35(b) = m.state.as_mut().unwrap() else {
        unreachable!()
    };
    let config = &b.config;
    let weights = &b.weights;
    let scratch = &b.scratch;
    let kv = &mut b.kv_cache;
    let dn = &mut b.dn_state;
    let vision_weights = b.vision_weights.as_ref().unwrap();

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

    let prompt_tokens = hipfire_runtime::prompt_frame::ChatFrame {
        tokenizer,
        system: if m.seq_pos == 0 { system_prompt } else { None },
        user: "", // unused: we pass tokens directly via build_with_user_tokens
        assistant_prefix,
        raw: false,
    }
    .build_with_user_tokens(&user_body);

    // KV-budget guard — tier-aware without eviction, absolute window with.
    // Adaptive admits against max_seq (floor-tier guarantee); non-adaptive keeps
    // physical_cap. Reserves trailer slots so natural im_end can write ChatML \n.
    let trailer = nl.len();
    let absolute_pos_vl = m.seq_pos.saturating_add(kv.compact_offset);
    let adaptive_engaged = m.kv_adaptive.is_some();
    let no_evict_cap = vl_no_eviction_kv_cap(m.physical_cap, m.max_seq, adaptive_engaged);
    let over_budget = if m.eviction.is_none() {
        m.seq_pos
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > no_evict_cap
    } else {
        absolute_pos_vl
            .saturating_add(prompt_tokens.len())
            .saturating_add(max_tokens)
            .saturating_add(trailer)
            > m.max_seq
    };
    if over_budget {
        write_error(stdout, id, &format!(
            "request exceeds loaded KV budget: seq_pos={} + prefill={} + max_tokens={} + trailer={} > cap={} — reload model with a larger max_seq",
            m.seq_pos, prompt_tokens.len(), max_tokens, trailer,
            if m.eviction.is_none() { no_evict_cap } else { m.max_seq },
        ));
        return;
    }

    // 3D mrope positions for this request. Built from the image span we just
    // spliced, BEFORE any GPU work so a validation bail is cheap.
    //
    // Disabled while eviction is armed: TriAttention renumbers physical slots
    // mid-prefill (`m.seq_pos = new_phys`), and `MropeCtx::positions` is indexed
    // by physical position. Rather than silently mis-indexing, that
    // configuration keeps today's 1D behavior.
    let mrope_ctx = if m.eviction.is_some() {
        if n_visual_tokens > 0 {
            eprintln!("[daemon/vl] mrope disabled (eviction armed) — falling back to 1D positions");
        }
        None
    } else {
        build_vl_mrope_ctx(
            &prompt_tokens,
            image_pad_id,
            n_visual_tokens,
            grid_h,
            grid_w,
            vision_config.spatial_merge_size,
            m.seq_pos,
            config,
        )
    };
    let mrope = mrope_ctx.as_ref();

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
    let visual_tokens = match qwen35_vl::vision_forward(
        gpu,
        vision_weights,
        &vision_config,
        &patches,
        grid_h,
        grid_w,
    ) {
        Ok(v) => v,
        Err(e) => {
            vl_forward_fail(
                stdout,
                id,
                "vision_forward",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
    };

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
    // advance m.seq_pos in-loop and call maybe_evict / maybe_downshift after
    // every committed write. Lazy VMM map/growth failures and adaptive
    // transition errors are request-scoped (no panic, no later token emit).
    let mut visual_idx = 0usize;
    for &token in prompt_tokens.iter() {
        if token == image_pad_id && visual_idx < n_visual_tokens {
            let emb = &visual_tokens[visual_idx * config.dim..(visual_idx + 1) * config.dim];
            if let Err(e) = qwen35::forward_scratch_embed_mrope(
                gpu, weights, config, emb, m.seq_pos, kv, dn, scratch, mrope,
            ) {
                vl_forward_fail(
                    stdout,
                    id,
                    "forward_scratch_embed (prefill)",
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
            visual_idx += 1;
        } else if let Err(e) = qwen35::forward_scratch_mrope(
            gpu, weights, config, token, m.seq_pos, kv, dn, scratch, mrope,
        ) {
            vl_forward_fail(
                stdout,
                id,
                "forward_scratch (prefill)",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            match ev.maybe_evict(gpu, kv, m.seq_pos) {
                Ok(Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                })) => {
                    m.seq_pos = new_phys;
                }
                Ok(None) => {}
                Err(e) => {
                    vl_forward_fail(
                        stdout,
                        id,
                        "maybe_evict (prefill)",
                        e,
                        gpu,
                        dn,
                        kv,
                        &mut m.kv_adaptive,
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                    );
                    return;
                }
            }
        }
        // Adaptive KV: downshift BETWEEN prefill tokens the moment the
        // start-tier buffer fills so a long multi-chunk visual+text prompt
        // cannot overflow current-stride capacity before decode begins.
        if vl_adaptive_downshift_fail_closed(
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            gpu,
            kv,
            dn,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            stdout,
            id,
            "vl-prefill",
        ) {
            return;
        }
    }

    m.conversation_tokens.extend_from_slice(&prompt_tokens);

    // Adaptive KV: post-prefill catch-up before first sample/decode write.
    if vl_adaptive_downshift_fail_closed(
        &mut m.kv_adaptive,
        &mut m.seq_pos,
        gpu,
        kv,
        dn,
        &mut m.conversation_tokens,
        &mut m.prefill_checkpoints,
        stdout,
        id,
        "vl-post-prefill",
    ) {
        return;
    }

    // hunt3 M-D: repeat-penalty / n-gram-block history must be scoped to the
    // GENERATED tokens only (mirrors the text path's `ngram_scope_start` set to
    // conversation_tokens.len() after prefill). Passing the full conversation
    // makes the trailing window prompt-dominated, suppressing the names/numbers
    // a VL transcription task must reproduce.
    let vl_ngram_scope_start = m.conversation_tokens.len();

    // Generate. CPU-side sampling — VL path predates the GPU sampler
    // and downloads logits each step:
    //   - first sample: top-p only (no repeat penalty);
    //   - subsequent samples: repeat penalty, then top-p sample.
    //
    // Unlike ordinary text generation, do not apply the positional 3..6-gram
    // ban here. OCR/layout output legitimately repeats table and markup
    // sequences. The configured LoopGuard remains available for pathological
    // full-loop termination, and the text paths retain their n-gram policies.
    //
    // Attractor-block uses CPU-side mutation of the downloaded logits
    // vector (`block_attractor_unclosed_cpu`) instead of the previous
    // GPU memcpy + redownload — saves a full vocab-sized DMA per token.
    let mut logits = match gpu.download_f32(&scratch.logits) {
        Ok(v) => v,
        Err(e) => {
            vl_forward_fail(
                stdout,
                id,
                "download_f32 (post-prefill)",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
    };
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
        // VL path samples on the CPU (sample_cpu), which does not yet honor
        // top_k / min_p; keep None so behavior is unchanged.
        top_k: None,
        min_p: None,
    };
    let vl_cfg = SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty,
        repeat_window,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        top_k: None,
        min_p: None,
    };
    let mut next_token = sampler::sample_cpu(&mut logits, &[], &vl_cfg_first);
    let t_prefill = Instant::now();
    let mut generated = 0;
    let mut streamed_tokens: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    // Think-depth tracking via token IDs (not UTF-8 rfind).
    // The previous implementation decoded the full streamed output to a
    // string and ran rfind on every token — O(N²) total, fragile to
    // tokenizer changes. Since `think_pair` already gives us the
    // open/close token IDs, we can track depth incrementally in O(1).
    let mut think_depth: usize = 0; // number of unmatched opens seen
    let mut think_count: usize = 0; // tokens emitted while depth > 0

    // N-gram loop detector — mirrors the text path. Catches answer-phase
    // attractor loops that the think cap and repeat penalty miss.
    let loop_guard =
        hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get());

    while generated < max_tokens {
        // Commit KV for this sampled token BEFORE any client-visible emit so a
        // lazy VMM map/growth failure cannot stream an uncommitted token.
        // Order: forward → seq_pos++ → evict → downshift → then
        // generated/conversation/committed/token text. On failure: cold-reset
        // + request error only (no failed token). Terminators break after a
        // successful commit (same as AR).
        if let Err(e) = qwen35::forward_scratch_mrope(
            gpu, weights, config, next_token, m.seq_pos, kv, dn, scratch, mrope,
        ) {
            vl_forward_fail(
                stdout,
                id,
                "forward_scratch (decode)",
                e,
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
            );
            return;
        }
        m.seq_pos += 1;
        if let Some(ref ev) = m.eviction {
            match ev.maybe_evict(gpu, kv, m.seq_pos) {
                Ok(Some(hipfire_runtime::triattn::EvictionResult {
                    new_physical: new_phys,
                    ..
                })) => {
                    m.seq_pos = new_phys;
                }
                Ok(None) => {}
                Err(e) => {
                    vl_forward_fail(
                        stdout,
                        id,
                        "maybe_evict (decode)",
                        e,
                        gpu,
                        dn,
                        kv,
                        &mut m.kv_adaptive,
                        &mut m.seq_pos,
                        &mut m.conversation_tokens,
                        &mut m.prefill_checkpoints,
                    );
                    return;
                }
            }
        }
        if vl_adaptive_downshift_fail_closed(
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            gpu,
            kv,
            dn,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            stdout,
            id,
            "vl-decode",
        ) {
            return;
        }

        generated += 1;
        m.conversation_tokens.push(next_token);
        streamed_tokens.push(next_token);
        emit_committed_event(
            stdout,
            id,
            next_token,
            generated - 1,
            t0.elapsed().as_millis() as u64,
        );

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
                r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default(),
                active_attempt_id()
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

        if let Some(hipfire_runtime::loop_guard::StopReason::NgramRepeat { count, .. }) =
            loop_guard.check(&streamed_tokens)
        {
            let window_len = loop_guard.window_len(streamed_tokens.len());
            let _ = writeln!(
                stdout,
                r#"{{"type":"info","id":"{}","message":"ngram loop detected (4gram repeated {}× in last {} tokens) — forcing EOS"}}"#,
                id, count, window_len,
            );
            let _ = stdout.flush();
            break;
        }

        logits = match gpu.download_f32(&scratch.logits) {
            Ok(v) => v,
            Err(e) => {
                vl_forward_fail(
                    stdout,
                    id,
                    "download_f32 (decode)",
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
        };
        // hunt3 M-D: scope repeat-penalty history to generated-only.
        // Exact transcription legitimately repeats HTML/Markdown n-grams
        // (`<tr>`, `<td>`, table delimiters, boilerplate). Hard no-repeat
        // blocking corrupts those outputs by forcing a lower-ranked token
        // whenever a 3..6-gram recurs. Keep the ordinary configured repeat
        // penalty in `sample_cpu`, but do not mutate VL logits with an
        // unconditional no-repeat constraint.
        let vl_ngram_scope = &m.conversation_tokens[vl_ngram_scope_start..];
        if let Some((open, close)) = think_pair {
            block_attractor_unclosed_cpu(&mut logits, &m.conversation_tokens, open, close, 20, 2);
        }

        next_token = sampler::sample_cpu(&mut logits, vl_ngram_scope, &vl_cfg);

        if max_think_tokens > 0 {
            if let Some((open, close)) = think_pair {
                // Incremental think-depth tracking via token IDs — O(1)
                // per token instead of the previous O(N²) decode+rfind.
                if next_token == open {
                    think_depth += 1;
                    think_count = 1;
                } else if next_token == close {
                    think_depth = think_depth.saturating_sub(1);
                    if think_depth == 0 {
                        think_count = 0;
                    }
                } else if think_depth > 0 {
                    think_count += 1;
                }

                if think_depth > 0 && think_count >= max_think_tokens {
                    let close_tokens = tokenizer.encode("</think>\n");
                    let budget_left = max_tokens.saturating_sub(generated);
                    let take = close_tokens.len().min(budget_left);
                    for &t in &close_tokens[..take] {
                        // KV write before any emit — same contract as main decode.
                        if let Err(e) = qwen35::forward_scratch_mrope(
                            gpu, weights, config, t, m.seq_pos, kv, dn, scratch, mrope,
                        ) {
                            vl_forward_fail(
                                stdout,
                                id,
                                "forward_scratch (vl-think-close)",
                                e,
                                gpu,
                                dn,
                                kv,
                                &mut m.kv_adaptive,
                                &mut m.seq_pos,
                                &mut m.conversation_tokens,
                                &mut m.prefill_checkpoints,
                            );
                            return;
                        }
                        m.seq_pos += 1;
                        if let Some(ref ev) = m.eviction {
                            match ev.maybe_evict(gpu, kv, m.seq_pos) {
                                Ok(Some(hipfire_runtime::triattn::EvictionResult {
                                    new_physical: new_phys,
                                    ..
                                })) => {
                                    m.seq_pos = new_phys;
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    vl_forward_fail(
                                        stdout,
                                        id,
                                        "maybe_evict (vl-think-close)",
                                        e,
                                        gpu,
                                        dn,
                                        kv,
                                        &mut m.kv_adaptive,
                                        &mut m.seq_pos,
                                        &mut m.conversation_tokens,
                                        &mut m.prefill_checkpoints,
                                    );
                                    return;
                                }
                            }
                        }
                        if vl_adaptive_downshift_fail_closed(
                            &mut m.kv_adaptive,
                            &mut m.seq_pos,
                            gpu,
                            kv,
                            dn,
                            &mut m.conversation_tokens,
                            &mut m.prefill_checkpoints,
                            stdout,
                            id,
                            "vl-think-close",
                        ) {
                            return;
                        }
                        m.conversation_tokens.push(t);
                        streamed_tokens.push(t);
                        // hunt3 H-F: emit the committed-token event for force-closed
                        // </think> tokens too, BEFORE `generated += 1`, so the
                        // committed pos stays in lockstep with the streamed count
                        // under HIPFIRE_EMIT_TOKEN_IDS=1. The VL main loop uses
                        // `generated - 1` after its increment; here `generated`
                        // (pre-increment) is the same value.
                        emit_committed_event(
                            stdout,
                            id,
                            t,
                            generated,
                            t0.elapsed().as_millis() as u64,
                        );

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
                                r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                                id,
                                serde_json::to_string(&text).unwrap_or_default(),
                                active_attempt_id()
                            );
                            let _ = stdout.flush();
                            emitted_bytes += vl;
                        }
                        generated += 1;
                    }
                    think_count = 0;
                    think_depth = 0; // Must reset — the close tokens
                                     // above bypass the incremental tracker, so depth
                                     // is still > 0 here. Without this, any subsequent
                                     // non-open/close token would re-trigger the cap.
                    if generated >= max_tokens {
                        break;
                    }
                    logits = match gpu.download_f32(&scratch.logits) {
                        Ok(v) => v,
                        Err(e) => {
                            vl_forward_fail(
                                stdout,
                                id,
                                "download_f32 (vl-think-close)",
                                e,
                                gpu,
                                dn,
                                kv,
                                &mut m.kv_adaptive,
                                &mut m.seq_pos,
                                &mut m.conversation_tokens,
                                &mut m.prefill_checkpoints,
                            );
                            return;
                        }
                    };
                    block_attractor_unclosed_cpu(
                        &mut logits,
                        &m.conversation_tokens,
                        open,
                        close,
                        20,
                        2,
                    );
                    // hunt3 M-D: generated-only repeat-penalty scope.
                    next_token = sampler::sample_cpu(
                        &mut logits,
                        &m.conversation_tokens[vl_ngram_scope_start..],
                        &vl_cfg,
                    );
                }
            }
        }
    }

    // ChatML \n boundary — run through forward to keep KV cache + DeltaNet in sync
    if im_end_token == Some(*m.conversation_tokens.last().unwrap_or(&0)) && !nl.is_empty() {
        for &t in &nl {
            if let Err(e) = qwen35::forward_scratch_mrope(
                gpu, weights, config, t, m.seq_pos, kv, dn, scratch, mrope,
            ) {
                vl_forward_fail(
                    stdout,
                    id,
                    "forward_scratch (vl-trailer)",
                    e,
                    gpu,
                    dn,
                    kv,
                    &mut m.kv_adaptive,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                );
                return;
            }
            m.seq_pos += 1;
            if let Some(ref ev) = m.eviction {
                match ev.maybe_evict(gpu, kv, m.seq_pos) {
                    Ok(Some(hipfire_runtime::triattn::EvictionResult {
                        new_physical: new_phys,
                        ..
                    })) => {
                        m.seq_pos = new_phys;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        vl_forward_fail(
                            stdout,
                            id,
                            "maybe_evict (vl-trailer)",
                            e,
                            gpu,
                            dn,
                            kv,
                            &mut m.kv_adaptive,
                            &mut m.seq_pos,
                            &mut m.conversation_tokens,
                            &mut m.prefill_checkpoints,
                        );
                        return;
                    }
                }
            }
            if vl_adaptive_downshift_fail_closed(
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                gpu,
                kv,
                dn,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                stdout,
                id,
                "vl-trailer",
            ) {
                return;
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
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}

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
    let config = match &m.state {
        Some(hipfire_loader::ModelState::DotsOcr(b)) => b.config.clone(),
        _ => unreachable!("generate_vl_dots_ocr requires ModelState::DotsOcr"),
    };
    let text_cfg = config.text.clone();
    let dim = text_cfg.hidden_size;
    // Weights/state via raw pointers to allow owned config while keeping disjoint borrows.
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle = match &mut m.state {
        Some(hipfire_loader::ModelState::DotsOcr(b)) => b as *mut _,
        _ => unreachable!(),
    };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };
    // 3. Build the prompt (HF-exact framing; imgpad count == n_visual by construction).
    let prompt_ids = dots_ocr::build_prompt_ids(tokenizer, prompt, n_visual);
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
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
            // Dispatch the token-embedding lookup on the actual embedding
            // format. HFQ dots.ocr ships Q8_0 embeddings, but the
            // safetensors/Dir loader uploads F32 — hardcoding the Q8 kernel
            // here misreads F32 bytes as Q8 blocks, corrupting every text
            // token's embedding (the model then ignores the prompt). Mirrors
            // the per-format dispatch in `llama::forward`.
            let lookup = hipfire_runtime::llama::embedding_lookup_dispatch(
                gpu,
                weights.text.embd_format,
                &weights.text.token_embd,
                &emb_scratch,
                token,
                dim,
            );
            if let Err(e) = lookup {
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
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, &text_cfg, state, &embeds)
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

    // 6. Decode. Opt-in n-gram speculative decode when a speculator was built at
    // load (HIPFIRE_NGRAM_DRAFT=1, arch_id=8 gate in `spec_build`); else the
    // bespoke greedy AR loop below. The vision prefill above already advanced the
    // dots-ocr Qwen2 state (`ModelState::DotsOcr`), so both paths decode from the
    // same warm state — only the drafting differs. The n-gram verify always falls back to
    // the target's greedy argmax, so spec output is byte-identical to AR; only τ
    // (speed) changes. The prefill bindings above (`tokenizer`/`config`/`state`/…)
    // are released here so the speculative branch can take `&mut m`; the AR path
    // re-borrows them below.
    if m.speculator.is_some() {
        decode_vl_dots_ocr_ngram(
            m,
            gpu,
            stdout,
            id,
            &prompt_ids,
            max_tokens,
            t0,
            prefill_tokens,
            prefill_s,
        );
        return;
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = match &m.state {
        Some(hipfire_loader::ModelState::DotsOcr(b)) => b.config.clone(),
        _ => unreachable!(),
    };
    let text_cfg = config.text.clone();
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle = match &mut m.state {
        Some(hipfire_loader::ModelState::DotsOcr(b)) => b as *mut _,
        _ => unreachable!(),
    };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };
    // Greedy decode, streaming in the daemon JSONL protocol.
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
                r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default(),
                active_attempt_id()
            );
            let _ = stdout.flush();
            emitted_bytes += valid_len;
        }

        match qwen2::forward_step_greedy(gpu, &weights.text, &text_cfg, state, next) {
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
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}

pub fn decode_vl_dots_ocr_ngram(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    t0: Instant,
    prefill_tokens: usize,
    prefill_s: f64,
) {
    use hipfire_arch_dots_ocr::DotsOcrBundle;
    use hipfire_loader::ModelState;
    // Move the live decoder state into a SpecTarget bundle; restored on return.
    let mut bundle = match m.state.take().unwrap() {
        ModelState::DotsOcr(b) => b,
        _ => unreachable!("decode_vl_dots_ocr_ngram requires ModelState::DotsOcr"),
    };
    let mut spec = m.speculator.take().unwrap();
    // `m.tokenizer` is a disjoint field → coexists with the takes above and the
    // restore below; the loop never touches `m`.
    let tokenizer = m.tokenizer.as_ref().unwrap();
    run_dots_ocr_ngram_loop(
        &mut bundle,
        spec.as_mut(),
        tokenizer,
        gpu,
        stdout,
        id,
        prompt_ids,
        max_tokens,
        t0,
        prefill_tokens,
        prefill_s,
    );
    m.state = Some(ModelState::DotsOcr(bundle));
    m.speculator = Some(spec);
}
pub fn run_dots_ocr_ngram_loop(
    bundle: &mut hipfire_arch_dots_ocr::DotsOcrBundle,
    spec: &mut dyn hipfire_runtime::spec::Speculator,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
    t0: Instant,
    prefill_tokens: usize,
    prefill_s: f64,
) {
    let eos_set: Vec<u32> = if bundle.config.text.eos_token_ids.is_empty() {
        vec![bundle.config.text.eos_token_id]
    } else {
        bundle.config.text.eos_token_ids.clone()
    };
    let block_size = spec.block_size();
    let ctx_capacity = spec.ctx_capacity();

    // Prime the n-gram drafter + fetch the first token WITHOUT re-running the
    // (vision-conditioned) target prefill. `cache_hit=true` + an empty suffix
    // makes `ChainSpeculator::prefill` skip the target advance —
    // `spec_advance(&[], prompt_len, reset=false)` just argmaxes the live
    // post-vision-prefill logits — and only `drafter.prefill_seed(prompt_ids)`.
    // It also lazily builds the verify scratch (required before the first `step`).
    let first_token = match spec.prefill(
        gpu,
        bundle,
        prompt_ids,
        &[],
        prompt_ids.len(),
        true,
        None,
        &|| check_abort(id),
    ) {
        Ok(PrefillOutcome::Ready { first_token }) => first_token,
        Ok(PrefillOutcome::Aborted) => {
            // Client cancel during n-gram prefill: cancel lifecycle only
            // (no success done / commit_ready).
            emit_qwen_ar_cancelled(stdout, id, 0);
            return;
        }
        Err(e) => {
            write_error(stdout, id, &format!("dots.ocr spec prefill: {e}"));
            return;
        }
    };

    let t_gen = Instant::now();
    let mut streamed: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut generated = 0usize;
    // n-gram context (committed generated tail; the drafter holds the prompt
    // internally via prefill_seed).
    let mut emitted: Vec<u32> = Vec::new();
    let mut position = prompt_ids.len();
    let mut seed_token = first_token;
    // τ accounting (accepted drafts / windows) — mirrors the text spec path so
    // the done envelope reports acceptance for diagnosing spec-vs-AR perf.
    let mut spec_cycles = 0usize;
    let mut spec_accepted = 0usize;
    // Tokens to stream this iteration. First window = the prefill seed alone
    // (mirrors the AR loop emitting the first argmax), then the accepted
    // committed tail from each `spec.step` (seed re-echo already stripped).
    let mut window: Vec<u32> = vec![first_token];

    'outer: loop {
        for &tok in &window {
            if generated >= max_tokens {
                break 'outer;
            }
            // EOS is never streamed (matches the AR loop's pre-emit break).
            if eos_set.contains(&tok) {
                break 'outer;
            }
            emit_committed_event(stdout, id, tok, generated, t0.elapsed().as_millis() as u64);
            generated += 1;
            streamed.push(tok);
            emitted.push(tok);
            // Incremental UTF-8 streaming — only emit complete code points
            // (byte-identical to the AR path).
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
                    r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                    id,
                    serde_json::to_string(&text).unwrap_or_default(),
                    active_attempt_id()
                );
                let _ = stdout.flush();
                emitted_bytes += valid_len;
            }
        }
        if generated >= max_tokens {
            break;
        }
        // Decode-side cancel: stop early. The next request resets state at
        // prefill, so no cross-request bleed; the caller restores bundle/spec.
        if check_abort(id) {
            break;
        }
        // Context-overflow guard (matches generate_spec): one window writes up
        // to `block_size` KV slots.
        if position.saturating_add(block_size) >= ctx_capacity {
            break;
        }
        let max_emit = max_tokens.saturating_sub(generated);
        let step = match spec.step(
            gpu, bundle, position, seed_token, &emitted, None, 0.0, max_emit,
        ) {
            Ok(s) => s,
            Err(e) => {
                write_error(stdout, id, &format!("dots.ocr spec_step: {e}"));
                break;
            }
        };
        spec_cycles += 1;
        spec_accepted += step.accepted;
        // Advance by the emitted-tail length (= accepted + 1), per the spec.rs
        // `emit_len_drives_advance` contract; the target already wrote KV for the
        // whole tail in `verify_block`.
        position += step.emit.len();
        seed_token = step.next_seed;
        window = step.emit.to_vec();
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
    let tau = if spec_cycles > 0 {
        spec_accepted as f64 / spec_cycles as f64
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "dflash": true,
        "tau": (tau * 100.0).round() / 100.0,
        "cycles": spec_cycles,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}

pub fn generate_dots_ocr_text(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    id: &str,
    prompt: &str,
    _system_prompt: Option<&str>,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
) {
    let _ = (temp, top_p); // greedy decode for now; sampling left for future work
    let t0 = Instant::now();

    let max_seq = m.max_seq;

    // Model state (disjoint field borrows of `m`).
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let config = match &m.state {
        Some(hipfire_loader::ModelState::DotsOcr(b)) => b.config.clone(),
        _ => unreachable!(),
    };
    let text_cfg = config.text.clone();
    let dim = text_cfg.hidden_size;
    // Weights/state via raw pointers to allow owned config while keeping disjoint borrows.
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle = match &mut m.state {
        Some(hipfire_loader::ModelState::DotsOcr(b)) => b as *mut _,
        _ => unreachable!(),
    };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };
    // Tokenize the text prompt directly (no image tokens).
    let prompt_ids = tokenizer.encode(prompt);
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
        write_error(stdout, id, &format!(
            "dots.ocr text request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
            prompt_ids.len(), max_tokens, max_seq));
        return;
    }

    // Prefill: build the [seq × dim] embedding matrix via per-token
    // embedding lookup dispatch, then run through batched prefill.
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
    let mut embed_err: Option<String> = None;
    for (pos, &token) in prompt_ids.iter().enumerate() {
        let lookup = hipfire_runtime::llama::embedding_lookup_dispatch(
            gpu,
            weights.text.embd_format,
            &weights.text.token_embd,
            &emb_scratch,
            token,
            dim,
        );
        if let Err(e) = lookup {
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
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, &text_cfg, state, &embeds)
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

    // Greedy decode, streaming in the daemon JSONL protocol.
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
                r#"{{"type":"token","id":"{}","text":{},"attempt_id":{}}}"#,
                id,
                serde_json::to_string(&text).unwrap_or_default(),
                active_attempt_id()
            );
            let _ = stdout.flush();
            emitted_bytes += valid_len;
        }

        match qwen2::forward_step_greedy(gpu, &weights.text, &text_cfg, state, next) {
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
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated,
        "tok_s": (tok_s * 10.0).round() / 10.0,
        "prefill_tokens": prefill_tokens,
        "prefill_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "prefill_tok_s": (prefill_tok_s * 10.0).round() / 10.0,
        "decode_tok_s": (decode_tok_s * 10.0).round() / 10.0,
        "ttft_ms": ((prefill_s * 1000.0) * 10.0).round() / 10.0,
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => emit_staged_terminal_done(stdout, &pending_done),
        ClientTerminalDecision::Abort => {}
    }
}

