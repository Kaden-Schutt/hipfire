// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision and OCR generation — Qwen3.5-VL and dots.ocr.
//!
//! Per-architecture generation bodies lifted verbatim from `crates/hipfire-daemon/src/main.rs`
//! (wave 5 / D3). See `lib.rs` for layering rationale.

use crate::ar::emit_active_route_error;
use base64::Engine;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative;
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_engine::emit::{emit_reasoning_token, emit_visible_token};
use hipfire_engine::scheduler::block_attractor_unclosed_cpu;
use hipfire_engine::terminal::{
    active_attempt_id, await_client_terminal_commit, check_abort, ClientTerminalDecision,
};
use hipfire_loader::LoadedModel;
use hipfire_runtime::emit_text::{ThinkOutputRouter, ThinkRouteEvent};
use hipfire_runtime::eos_filter::{EosFilter, FilterAction};
use hipfire_runtime::sampler::{self, SamplerConfig};
use hipfire_runtime::spec::{PrefillOutcome, Speculator};
use std::any::Any;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
fn emit_active_attempt_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    emit_active_route_error(stdout, id, message, class, retryable, rolled_back);
}
fn write_error(stdout: &mut impl std::io::Write, id: &str, message: &str) {
    crate::ar::emit_active_route_error(stdout, Some(id), message, "internal", false, false);
}

// ── Local copy of the daemon-shared emission helper ──────────────────────────
// `emit_committed_event` is shared across all generate paths in the daemon,
// but the generate crate cannot depend back on the daemon. Keeping this tiny
// copy here avoids introducing a circular dependency.

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

/// Feed one decode-step's new bytes through the v2 typed emission contract:
/// EosFilter (UTF-8 boundary + EOT marker suppression) → ThinkOutputRouter
/// (think-channel split) → `token` / `reasoning` envelopes.
///
/// Arch 5/6 advertises `gen_start.contract_version = 2`, under which the CLI
/// appends token text to `content` VERBATIM — no marker scan (complete.rs
/// SemanticEventFold). The producer therefore owns think-splitting and marker
/// suppression; before this helper the VL loop emitted every decoded byte as
/// a bare token envelope, so image turns shipped the whole think block plus
/// literal `</think>` / `<|im_end|>` markers inside `content` with zero
/// `reasoning_content` (2026-08-27 ledger, post-thinking garble).
///
/// Mirrors `ar::qwen_ar_observe_and_route` minus the ToolOutputRouter: VL
/// image turns carry no tool contract, so reserved-looking text stays visible
/// instead of being buffered or failing closed.
fn vl_route_decode_text(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think: &mut ThinkOutputRouter,
    new_bytes: &[u8],
) -> bool {
    let (bytes, is_stop) = match filter.observe(new_bytes) {
        FilterAction::Emit(b) => (b, false),
        FilterAction::EmitAndStop(b) => (b, true),
        FilterAction::Hold => return false,
        FilterAction::Stop => return true,
    };
    if let Ok(text) = std::str::from_utf8(&bytes) {
        if !text.is_empty() {
            let mut routed = Vec::new();
            think.push_into(text, &mut routed);
            for ev in routed {
                match ev {
                    ThinkRouteEvent::Reasoning(t) => emit_reasoning_token(stdout, id, &t),
                    ThinkRouteEvent::Content(t) => emit_visible_token(stdout, id, &t),
                }
            }
        }
    }
    is_stop
}

fn vl_finish_think_routing(
    stdout: &mut impl std::io::Write,
    id: &str,
    filter: &mut EosFilter,
    think: &mut ThinkOutputRouter,
) {
    let pending = filter.flush_pending();
    if !pending.is_empty() {
        if let Ok(text) = std::str::from_utf8(&pending) {
            if !text.is_empty() {
                let mut routed = Vec::new();
                think.push_into(text, &mut routed);
                for ev in routed {
                    match ev {
                        ThinkRouteEvent::Reasoning(t) => emit_reasoning_token(stdout, id, &t),
                        ThinkRouteEvent::Content(t) => emit_visible_token(stdout, id, &t),
                    }
                }
            }
        }
    }
    let mut routed = Vec::new();
    think.finish_into(&mut routed);
    for ev in routed {
        match ev {
            ThinkRouteEvent::Reasoning(t) => emit_reasoning_token(stdout, id, &t),
            ThinkRouteEvent::Content(t) => emit_visible_token(stdout, id, &t),
        }
    }
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
    /// Per-request sampler seed (see `hipfire_engine::request_seed_for`).
    pub seed: u32,
}

pub fn vl_no_eviction_kv_cap(physical_cap: usize, max_seq: usize, adaptive_engaged: bool) -> usize {
    if adaptive_engaged {
        max_seq
    } else {
        physical_cap
    }
}

#[allow(clippy::too_many_arguments)]
fn vl_reset_live(
    gpu: &mut rdna_compute::Gpu,
    dn: &mut qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    speculator: &mut Option<Box<dyn Speculator>>,
) -> crate::common::RollbackEpilogue {
    let mut reset_target = |gpu: &mut rdna_compute::Gpu| {
        dn.reset(gpu)
            .map_err(|e| format!("VL recurrent reset: {e}"))?;
        kv.compact_offset = 0;
        if let Some(ad) = kv_adaptive.as_mut() {
            if !ad.is_poisoned() {
                ad.reset_with_cache(gpu, kv);
            }
        }
        Ok(())
    };
    let spec = speculator
        .as_deref_mut()
        .map(|spec| spec as &mut dyn Speculator);
    crate::common::production_fail_closed_rollback_live_with_target(
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
        dflash_checkpoints,
        asst_turn_cache,
        gpu,
        &mut reset_target,
        spec,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vl_adaptive_downshift_fail_closed(
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    gpu: &mut rdna_compute::Gpu,
    dn: &mut qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    speculator: &mut Option<Box<dyn Speculator>>,
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
            let ep = vl_reset_live(
                gpu,
                dn,
                kv,
                kv_adaptive,
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
                dflash_checkpoints,
                asst_turn_cache,
                speculator,
            );
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                &format!("adaptive KV transition failed during {phase}: {e}"),
                "gpu",
                true,
                &ep,
            );
            true
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vl_forward_fail(
    stdout: &mut std::io::Stdout,
    id: &str,
    phase: &str,
    err: impl std::fmt::Display,
    gpu: &mut rdna_compute::Gpu,
    dn: &mut qwen35::DeltaNetState,
    kv: &mut hipfire_runtime::llama::KvCache,
    kv_adaptive: &mut Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    speculator: &mut Option<Box<dyn Speculator>>,
) {
    let message = format!("VL {phase}: {err}");
    let ep = vl_reset_live(
        gpu,
        dn,
        kv,
        kv_adaptive,
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
        dflash_checkpoints,
        asst_turn_cache,
        speculator,
    );
    crate::common::emit_fail_closed_error(stdout, Some(id), &message, "gpu", true, &ep);
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
#[allow(clippy::too_many_arguments)]
fn dots_reset_live(
    target_reset: &mut dyn FnMut(&mut rdna_compute::Gpu) -> Result<(), String>,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    speculator: &mut Option<Box<dyn Speculator>>,
    gpu: &mut rdna_compute::Gpu,
) -> crate::common::RollbackEpilogue {
    let spec = speculator
        .as_deref_mut()
        .map(|spec| spec as &mut dyn Speculator);
    crate::common::production_fail_closed_rollback_live_with_target(
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
        dflash_checkpoints,
        asst_turn_cache,
        gpu,
        target_reset,
        spec,
    )
}
#[allow(clippy::too_many_arguments)]
fn dots_reset_state_live(
    state: &mut hipfire_arch_qwen2::qwen2::Qwen2State,
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
    speculator: &mut Option<Box<dyn Speculator>>,
    gpu: &mut rdna_compute::Gpu,
) -> crate::common::RollbackEpilogue {
    let mut target_reset = |_: &mut rdna_compute::Gpu| {
        state.reset();
        Ok(())
    };
    dots_reset_live(
        &mut target_reset,
        seq_pos,
        conversation_tokens,
        prefill_checkpoints,
        dflash_checkpoints,
        asst_turn_cache,
        speculator,
        gpu,
    )
}

pub fn generate_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    // Stream-contract opener. MUST be the first event on this request's
    // stream: the HTTP CLI's StreamContractGate rejects any later event that
    // arrives without a preceding gen_start for this id — which stranded
    // image turns after the encoder finished ("no response bytes", wedged
    // slot; 2026-08-27 ledger finding b). Text-path generate() has emitted
    // this since the e99583afa-class fixes.
    // started_in_think mirrors the ChatFrame builder's own conditions: the
    // `<think>` opener lands in the prompt only for AssistantPrefix::OpenThink
    // AND a tokenizer that carries the special token (the builder falls back
    // to Plain otherwise). v2 consumers ignore the field, but the envelope
    // stays honest for any contract that consults it.
    let started_in_think = matches!(
        params.assistant_prefix,
        hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
    ) && m
        .tokenizer
        .as_ref()
        .is_some_and(|t| t.special_token_id("<think>").is_some());
    crate::ar::emit_generation_start(
        crate::ar::GenerationRoute::QwenAr,
        stdout,
        params.id,
        started_in_think,
    );
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
        seed,
    } = *params;
    // hunt3 M-E: seed the process-global CPU sampler RNG per request. The VL
    // path samples exclusively via sampler::sample_cpu, which draws from this
    // global; without the per-request reset it carried RNG state across
    // requests (and across earlier text-path requests) → cross-request
    // nondeterminism. Seeded by hipfire-engine::request_seed_for (wire `seed`
    // wins, else attempt key + counter), matching the sequential text path.
    hipfire_runtime::llama::reset_cpu_sampler_rng(seed);
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
        let reset = match m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            Some(bundle) => vl_reset_live(
                gpu,
                &mut bundle.dn_state,
                &mut bundle.kv_cache,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            ),
            None => crate::common::RollbackEpilogue {
                rolled_back: false,
                context: Some("VL bundle missing during context reset".to_string()),
            },
        };
        if !reset.rolled_back {
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                "vision context reset failed",
                "gpu",
                true,
                &reset,
            );
            return;
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

    let Some(b) = m.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) else {
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
            let ep = vl_reset_live(
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            );
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                &format!("VL vision_forward: {e}"),
                "gpu",
                true,
                &ep,
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
        // The encoder phase cannot be interrupted mid-kernel.  Poll before
        // the first token and reset before publishing the cancellation pair.
        if check_abort(id) {
            let ep = vl_reset_live(
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
            return;
        }
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
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    &mut m.speculator,
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
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
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
                        &mut m.dflash_checkpoints,
                        &mut m.asst_turn_cache,
                        &mut m.speculator,
                    );
                    return;
                }
            }
        }
        if vl_adaptive_downshift_fail_closed(
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            gpu,
            dn,
            kv,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            stdout,
            id,
            "vl-prefill",
        ) {
            return;
        }
    }

    m.conversation_tokens.extend_from_slice(&prompt_tokens);
    if vl_adaptive_downshift_fail_closed(
        &mut m.kv_adaptive,
        &mut m.seq_pos,
        gpu,
        dn,
        kv,
        &mut m.conversation_tokens,
        &mut m.prefill_checkpoints,
        &mut m.dflash_checkpoints,
        &mut m.asst_turn_cache,
        &mut m.speculator,
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
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
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
    // Typed-emission state for the v2 stream contract (see
    // `vl_route_decode_text`): EosFilter owns UTF-8 boundaries + EOT marker
    // suppression, ThinkOutputRouter owns the reasoning/content channel
    // split. `emitted_bytes` above becomes the bytes-FED-to-filter cursor —
    // the filter holds back partial UTF-8/marker tails internally, so the
    // old valid-prefix arithmetic is gone.
    let mut vl_filter = EosFilter::new(crate::ar::qwen_ar_eos_filter_config());
    let mut vl_think = ThinkOutputRouter::new(started_in_think);
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

    'vl_generate: while generated < max_tokens {
        // Decode-side client-cancel poll — same canonical-terminal rule as
        // the prefill poll above; partial per-call state (seq_pos,
        // conversation_tokens) is reclaimed by the next dispatch's
        // non-zero-seq_pos reset, matching the dots.ocr cancel path.
        if check_abort(id) {
            let ep = vl_reset_live(
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            return;
        }
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
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
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
                        &mut m.dflash_checkpoints,
                        &mut m.asst_turn_cache,
                        &mut m.speculator,
                    );
                    return;
                }
            }
        }
        if vl_adaptive_downshift_fail_closed(
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            gpu,
            dn,
            kv,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
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
        emitted_bytes = all_bytes.len();
        if vl_route_decode_text(stdout, id, &mut vl_filter, &mut vl_think, new_bytes) {
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
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    &mut m.speculator,
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
                                &mut m.dflash_checkpoints,
                                &mut m.asst_turn_cache,
                                &mut m.speculator,
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
                                        &mut m.dflash_checkpoints,
                                        &mut m.asst_turn_cache,
                                        &mut m.speculator,
                                    );
                                    return;
                                }
                            }
                        }
                        if vl_adaptive_downshift_fail_closed(
                            &mut m.kv_adaptive,
                            &mut m.seq_pos,
                            gpu,
                            dn,
                            kv,
                            &mut m.conversation_tokens,
                            &mut m.prefill_checkpoints,
                            &mut m.dflash_checkpoints,
                            &mut m.asst_turn_cache,
                            &mut m.speculator,
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
                        emitted_bytes = all_bytes.len();
                        // Same typed routing as the main decode site: the
                        // forced `</think>` closer is consumed by the router
                        // (channel flips to content), never emitted literally.
                        if vl_route_decode_text(
                            stdout,
                            id,
                            &mut vl_filter,
                            &mut vl_think,
                            new_bytes,
                        ) {
                            generated += 1;
                            break 'vl_generate;
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
                                &mut m.dflash_checkpoints,
                                &mut m.asst_turn_cache,
                                &mut m.speculator,
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
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    &mut m.speculator,
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
                            &mut m.dflash_checkpoints,
                            &mut m.asst_turn_cache,
                            &mut m.speculator,
                        );
                        return;
                    }
                }
            }
            if vl_adaptive_downshift_fail_closed(
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                gpu,
                dn,
                kv,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                stdout,
                id,
                "vl-trailer",
            ) {
                return;
            }
            m.conversation_tokens.push(t);
        }
    }

    if check_abort(id) {
        let ep = vl_reset_live(
            gpu,
            dn,
            kv,
            &mut m.kv_adaptive,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
        );
        crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
        return;
    }
    // Flush any trailing partial think marker as ordinary text in its
    // current channel (text-AR finish parity) before the terminal.
    vl_finish_think_routing(stdout, id, &mut vl_filter, &mut vl_think);
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
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done(stdout, id, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            let ep = vl_reset_live(
                gpu,
                dn,
                kv,
                &mut m.kv_adaptive,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
        }
    }
}

pub fn generate_vl_dots_ocr(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    use hipfire_arch_dots_ocr::image as dots_image;
    // Stream-contract opener — same HTTP-gate rationale as generate_vl above.
    crate::ar::emit_generation_start(
        crate::ar::GenerationRoute::DotsOcr,
        stdout,
        params.id,
        false,
    );
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

    // 2. Resolve the text configuration and build the prompt before taking
    // any mutable model borrow.  The reset below is the loader-owned boundary
    // for every Dots OCR turn.
    let config = m.dots_ocr().unwrap().config.clone();
    let text_cfg = config.text.clone();
    let dim = text_cfg.hidden_size;
    let prompt_ids = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        dots_ocr::build_prompt_ids(tokenizer, prompt, n_visual)
    };
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
        write_error(stdout, id, &format!(
            "dots.ocr request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
            prompt_ids.len(), max_tokens, max_seq));
        return;
    }
    if check_abort(id) {
        crate::ar::emit_active_route_cancel(stdout, id, 0);
        return;
    }
    if let Err(error) = m.reset_context(gpu) {
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            "dots.ocr context reset failed",
            "gpu",
            true,
            &crate::common::RollbackEpilogue {
                rolled_back: false,
                context: Some(error),
            },
        );
        return;
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    // Weights/state via raw pointers to allow owned config while keeping
    // disjoint borrows.  The loader reset above has already released all
    // previous request state.
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle =
        match m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        }) {
            Some(b) => b as *mut _,
            None => unreachable!(),
        };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };

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
        || check_abort(id),
    ) {
        Ok(Some(t)) => t,
        Ok(None) => {
            let _ = gpu.free_tensor(patches_gpu);
            crate::ar::emit_active_route_cancel(stdout, id, 0);
            return;
        }
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
        if check_abort(id) {
            let _ = gpu.free_tensor(emb_scratch);
            let ep = dots_reset_state_live(
                state,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                gpu,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
            return;
        }
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
    if check_abort(id) {
        let ep = dots_reset_state_live(
            state,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            gpu,
        );
        crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
        return;
    }
    if let Some(e) = embed_err {
        let ep = dots_reset_state_live(
            state,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            gpu,
        );
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            &format!("dots.ocr prefill embed build failed: {e}"),
            "gpu",
            true,
            &ep,
        );
        return;
    }
    if let Err(e) =
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, &text_cfg, state, &embeds)
    {
        let ep = dots_reset_state_live(
            state,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            gpu,
        );
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            &format!("dots.ocr batched prefill failed: {e:?}"),
            "gpu",
            true,
            &ep,
        );
        return;
    }
    if check_abort(id) {
        let ep = dots_reset_state_live(
            state,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            gpu,
        );
        crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
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
    let config = m.dots_ocr().unwrap().config.clone();
    let text_cfg = config.text.clone();
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle =
        match m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        }) {
            Some(b) => b as *mut _,
            None => unreachable!(),
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
            let ep = dots_reset_state_live(
                state,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                gpu,
            );
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                &format!("dots.ocr argmax failed: {e:?}"),
                "gpu",
                true,
                &ep,
            );
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
        if check_abort(id) {
            let ep = dots_reset_state_live(
                state,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                gpu,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            return;
        }
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
                let ep = dots_reset_state_live(
                    state,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    &mut m.speculator,
                    gpu,
                );
                crate::common::emit_fail_closed_error(
                    stdout,
                    Some(id),
                    &format!("dots.ocr decode failed: {e:?}"),
                    "gpu",
                    true,
                    &ep,
                );
                return;
            }
        }
    }

    if check_abort(id) {
        let ep = dots_reset_state_live(
            state,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            gpu,
        );
        crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
        return;
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
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done(stdout, id, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            let ep = dots_reset_state_live(
                state,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                gpu,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
        }
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
    // Move the live decoder state into a SpecTarget bundle; restored on return.
    let mut bundle = *(m.state.take().unwrap() as Box<dyn std::any::Any>)
        .downcast::<DotsOcrBundle>()
        .unwrap();
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
        &mut m.seq_pos,
        &mut m.conversation_tokens,
        &mut m.prefill_checkpoints,
        &mut m.dflash_checkpoints,
        &mut m.asst_turn_cache,
    );
    m.state = Some(Box::new(bundle));
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
    seq_pos: &mut usize,
    conversation_tokens: &mut Vec<u32>,
    prefill_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    dflash_checkpoints: &mut Vec<(usize, speculative::DeltaNetSnapshot)>,
    asst_turn_cache: &mut hipfire_loader::AsstTurnCache,
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
            let mut reset_target = |gpu: &mut rdna_compute::Gpu| {
                hipfire_runtime::spec::SpecTarget::reset_recurrent(bundle, gpu)
            };
            let ep = crate::common::production_fail_closed_rollback_live_with_target(
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
                dflash_checkpoints,
                asst_turn_cache,
                gpu,
                &mut reset_target,
                Some(spec),
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
            return;
        }
        Err(e) => {
            let mut reset_target = |gpu: &mut rdna_compute::Gpu| {
                hipfire_runtime::spec::SpecTarget::reset_recurrent(bundle, gpu)
            };
            let ep = crate::common::production_fail_closed_rollback_live_with_target(
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
                dflash_checkpoints,
                asst_turn_cache,
                gpu,
                &mut reset_target,
                Some(spec),
            );
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                &format!("dots.ocr spec prefill: {e}"),
                "gpu",
                true,
                &ep,
            );
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
        // Decode-side cancel: emit the canonical cancelled pair and stop —
        // falling through to the done handshake on an aborted attempt strands
        // the serve admission guard (2026-08-27 ledger finding-c class; same
        // rule as the prefill-cancel site above). The caller restores
        // bundle/spec state on return; the next request resets at prefill.
        if check_abort(id) {
            let mut reset_target = |gpu: &mut rdna_compute::Gpu| {
                hipfire_runtime::spec::SpecTarget::reset_recurrent(bundle, gpu)
            };
            let ep = crate::common::production_fail_closed_rollback_live_with_target(
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
                dflash_checkpoints,
                asst_turn_cache,
                gpu,
                &mut reset_target,
                Some(spec),
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            return;
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
                let mut reset_target = |gpu: &mut rdna_compute::Gpu| {
                    hipfire_runtime::spec::SpecTarget::reset_recurrent(bundle, gpu)
                };
                let ep = crate::common::production_fail_closed_rollback_live_with_target(
                    seq_pos,
                    conversation_tokens,
                    prefill_checkpoints,
                    dflash_checkpoints,
                    asst_turn_cache,
                    gpu,
                    &mut reset_target,
                    Some(spec),
                );
                crate::common::emit_fail_closed_error(
                    stdout,
                    Some(id),
                    &format!("dots.ocr spec_step: {e}"),
                    "gpu",
                    true,
                    &ep,
                );
                return;
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
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done(stdout, id, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            let mut reset_target = |gpu: &mut rdna_compute::Gpu| {
                hipfire_runtime::spec::SpecTarget::reset_recurrent(bundle, gpu)
            };
            let ep = crate::common::production_fail_closed_rollback_live_with_target(
                seq_pos,
                conversation_tokens,
                prefill_checkpoints,
                dflash_checkpoints,
                asst_turn_cache,
                gpu,
                &mut reset_target,
                Some(spec),
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
        }
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
    crate::ar::emit_generation_start(crate::ar::GenerationRoute::DotsOcr, stdout, id, false);
    let _ = (temp, top_p); // greedy decode for now; sampling left for future work
    let t0 = Instant::now();

    let max_seq = m.max_seq;
    let config = m.dots_ocr().unwrap().config.clone();
    let text_cfg = config.text.clone();
    let dim = text_cfg.hidden_size;
    let prompt_ids = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        tokenizer.encode(prompt)
    };
    if prompt_ids.len().saturating_add(max_tokens) > max_seq {
        write_error(stdout, id, &format!(
            "dots.ocr text request ({} prompt + {} gen) exceeds KV budget ({}); reload with a larger --max-seq",
            prompt_ids.len(), max_tokens, max_seq));
        return;
    }
    if check_abort(id) {
        crate::ar::emit_active_route_cancel(stdout, id, 0);
        return;
    }
    if let Err(error) = m.reset_context(gpu) {
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            "dots.ocr text context reset failed",
            "gpu",
            true,
            &crate::common::RollbackEpilogue {
                rolled_back: false,
                context: Some(error),
            },
        );
        return;
    }
    let tokenizer = m.tokenizer.as_ref().unwrap();
    let bundle_ptr: *mut hipfire_arch_dots_ocr::DotsOcrBundle =
        match m.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_dots_ocr::DotsOcrBundle>()
        }) {
            Some(b) => b as *mut _,
            None => unreachable!(),
        };
    let weights = unsafe { &(*bundle_ptr).weights };
    let state = unsafe { &mut (*bundle_ptr).state };
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
        let ep = dots_reset_state_live(
            state,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            gpu,
        );
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            &format!("dots.ocr prefill embed build failed: {e}"),
            "gpu",
            true,
            &ep,
        );
        return;
    }
    if let Err(e) =
        qwen2::forward_prefill_batch_embeds(gpu, &weights.text, &text_cfg, state, &embeds)
    {
        let ep = dots_reset_state_live(
            state,
            &mut m.seq_pos,
            &mut m.conversation_tokens,
            &mut m.prefill_checkpoints,
            &mut m.dflash_checkpoints,
            &mut m.asst_turn_cache,
            &mut m.speculator,
            gpu,
        );
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            &format!("dots.ocr batched prefill failed: {e:?}"),
            "gpu",
            true,
            &ep,
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
            let ep = dots_reset_state_live(
                state,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                gpu,
            );
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                &format!("dots.ocr argmax failed: {e:?}"),
                "gpu",
                true,
                &ep,
            );
            return;
        }
    };
    let t_gen = Instant::now();
    let mut streamed: Vec<u32> = Vec::new();
    let mut emitted_bytes = 0usize;
    let mut generated = 0usize;

    while generated < max_tokens {
        if check_abort(id) {
            let ep = dots_reset_state_live(
                state,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                gpu,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
            return;
        }
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
                let ep = dots_reset_state_live(
                    state,
                    &mut m.seq_pos,
                    &mut m.conversation_tokens,
                    &mut m.prefill_checkpoints,
                    &mut m.dflash_checkpoints,
                    &mut m.asst_turn_cache,
                    &mut m.speculator,
                    gpu,
                );
                crate::common::emit_fail_closed_error(
                    stdout,
                    Some(id),
                    &format!("dots.ocr decode failed: {e:?}"),
                    "gpu",
                    true,
                    &ep,
                );
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
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done(stdout, id, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            let ep = dots_reset_state_live(
                state,
                &mut m.seq_pos,
                &mut m.conversation_tokens,
                &mut m.prefill_checkpoints,
                &mut m.dflash_checkpoints,
                &mut m.asst_turn_cache,
                &mut m.speculator,
                gpu,
            );
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated, &ep);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{vl_finish_think_routing, vl_route_decode_text};
    use hipfire_runtime::emit_text::ThinkOutputRouter;
    use hipfire_runtime::eos_filter::EosFilter;

    /// Drive byte chunks through the VL typed-emission pipeline and collect
    /// (channel, text) wire events parsed back from the JSONL lines.
    fn drive(started_in_think: bool, chunks: &[&[u8]]) -> Vec<(String, String)> {
        let mut filter = EosFilter::new(crate::ar::qwen_ar_eos_filter_config());
        let mut think = ThinkOutputRouter::new(started_in_think);
        let mut out = Vec::new();
        for chunk in chunks {
            let _ = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, chunk);
        }
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out = String::from_utf8_lossy(&out);
        out.lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|v| {
                (
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect()
    }

    fn joined(events: &[(String, String)], channel: &str) -> String {
        events
            .iter()
            .filter(|(t, _)| t == channel)
            .map(|(_, s)| s.as_str())
            .collect()
    }

    #[test]
    fn thinking_turn_splits_reasoning_from_content() {
        // OpenThink prefix: generation starts inside think (no opener in the
        // generated bytes); closer flips to content; trailing EOT suppressed.
        let events = drive(
            true,
            &[
                b"The user wants a description.\n\n",
                b"</think>\n\n",
                b"A Shiba Inu dog naps on its back.",
                b"<|im_end|>",
            ],
        );
        assert_eq!(
            joined(&events, "reasoning"),
            "The user wants a description.\n\n"
        );
        assert_eq!(
            joined(&events, "token"),
            "A Shiba Inu dog naps on its back."
        );
        let all = events.iter().map(|(_, s)| s.clone()).collect::<String>();
        assert!(!all.contains("</think>"), "closer leaked: {all:?}");
        assert!(!all.contains("<|im_end|>"), "EOT leaked: {all:?}");
    }

    #[test]
    fn marker_split_across_chunks_still_routes() {
        // `</thi` + `nk>` arriving as separate decode deltas (token boundary)
        // must not leak a partial marker as visible text.
        let events = drive(true, &[b"plan</thi", b"nk>\nanswer", b"<|im_end|>"]);
        assert_eq!(joined(&events, "reasoning"), "plan");
        assert_eq!(joined(&events, "token"), "answer");
    }

    #[test]
    fn plain_prefix_spontaneous_think_markers_route() {
        // assistant_prefix=Plain and the model opens think itself.
        let events = drive(
            false,
            &[
                b"<think>private</think>",
                b"\n\nvisible answer",
                b"<|im_end|>",
            ],
        );
        assert_eq!(joined(&events, "reasoning"), "private");
        assert_eq!(joined(&events, "token"), "visible answer");
    }

    #[test]
    fn no_think_turn_is_pure_content() {
        let events = drive(false, &[b"Pointy teeths wow.", b"<|im_end|>"]);
        assert!(joined(&events, "reasoning").is_empty());
        assert_eq!(joined(&events, "token"), "Pointy teeths wow.");
    }

    #[test]
    fn hostile_request_id_stays_json_escaped() {
        // The old hand-rolled envelope spliced `id` into the JSON unescaped;
        // the typed emitters must survive a quote-bearing id.
        let mut filter = EosFilter::new(crate::ar::qwen_ar_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let _ = vl_route_decode_text(&mut out, "bad\"id\\", &mut filter, &mut think, b"text");
        vl_finish_think_routing(&mut out, "bad\"id\\", &mut filter, &mut think);
        let out = String::from_utf8_lossy(&out);
        for line in out.lines() {
            assert!(
                serde_json::from_str::<serde_json::Value>(line).is_ok(),
                "{line}"
            );
        }
    }

    #[test]
    fn ordinary_emit_returns_false() {
        let mut filter = EosFilter::new(crate::ar::qwen_ar_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let stopped = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, b"hello");
        assert!(!stopped, "ordinary emit should return false");
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out_s = String::from_utf8_lossy(&out);
        assert!(out_s.contains("hello"), "payload should emit: {out_s:?}");
    }

    #[test]
    fn stop_marker_returns_true_and_emits_preceding_text_without_leakage() {
        let mut filter = EosFilter::new(crate::ar::qwen_ar_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let payload = b"visible answer<|im_end|>";
        let stopped = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, payload);
        assert!(stopped, "EOT marker should signal stop");
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out_s = String::from_utf8_lossy(&out);
        assert!(
            out_s.contains("visible answer"),
            "preceding text should emit: {out_s:?}"
        );
        assert!(
            !out_s.contains("<|im_end|>"),
            "marker must not leak: {out_s:?}"
        );
    }

    #[test]
    fn trailing_held_prefix_flushed_at_finish() {
        // ` <` is a partial prefix of `<|im_end|>`; EosFilter holds it mid-stream
        // and only at finish should it be treated as ordinary prose and routed
        // through ThinkOutputRouter.
        let mut filter = EosFilter::new(crate::ar::qwen_ar_eos_filter_config());
        let mut think = ThinkOutputRouter::new(false);
        let mut out = Vec::new();
        let stopped = vl_route_decode_text(&mut out, "t-id", &mut filter, &mut think, b"hello <");
        assert!(!stopped);
        // Before finish, the held `<` should not yet be emitted.
        let mid = String::from_utf8_lossy(&out).to_string();
        let mid_events: Vec<(String, String)> = mid
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|v| {
                (
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect();
        let mid_text = joined(&mid_events, "token");
        assert_eq!(
            mid_text, "hello ",
            "partial prefix should be held, not emitted yet: {mid_text:?}"
        );
        vl_finish_think_routing(&mut out, "t-id", &mut filter, &mut think);
        let out_s = String::from_utf8_lossy(&out);
        let events: Vec<(String, String)> = out_s
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|v| {
                (
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    v.get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                )
            })
            .collect();
        assert_eq!(
            joined(&events, "token"),
            "hello <",
            "held prefix should flush at finish"
        );
    }
}

// ── LFM2-VL (arch 11) ────────────────────────────────────────────────────────
//
// SigLIP2-NaFlex tower + projector executed by `hipfire_arch_lfm2_vl`; the
// text loop, stream contract, sampling, and terminal handshake are copied
// from `dense::generate_lfm2moe` so an image turn behaves exactly like a
// long-prompted text turn once the visual features are spliced. See
// docs/specs/2026-08-27-lfm2-vl-vision-runtime.md.

use hipfire_arch_lfm2_vl as lfm2vl;

/// Expand the single `<image>` id in `prompt_ids` into the marker structure
/// HF's processor produces (narrow spec §1.5): `<|image_start|>`,
/// per-tile `<|img_row_R_col_C|>` + 256 placeholders, `<|img_thumbnail|>` +
/// thumbnail placeholders, `<|image_end|>` (markers only when present in
/// this tokenizer; multi-tile REQUIRES its markers and fails closed).
fn expand_image_placeholders(
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    image_token_id: u32,
    prepared: &lfm2vl::Prepared,
    cfg: &lfm2vl::VisionConfig,
) -> Result<Vec<u32>, String> {
    let start_count = prompt_ids.iter().filter(|&&t| t == image_token_id).count();
    if start_count != 1 {
        return Err(format!(
            "expected exactly one <image> token in rendered prompt, found {start_count} — \
             the artifact tokenizer is missing the VL added-token; requantize or use a \
             VL-capable tokenizer"
        ));
    }
    let special = |s: &str| -> Option<u32> { tokenizer.special_token_id(s) };
    let multi_tile = prepared.grid_rows > 1 || prepared.grid_cols > 1;

    let mut body: Vec<u32> = Vec::new();
    let start_id = special("<|image_start|>");
    let end_id = special("<|image_end|>");

    let tiles: Vec<(usize, usize, usize)> = if multi_tile {
        // row-major tile list aligned with Prepared.sub_images ordering.
        prepared
            .sub_images
            .iter()
            .take(prepared.grid_rows * prepared.grid_cols)
            .enumerate()
            .map(|(i, s)| {
                (
                    i / prepared.grid_cols + 1,
                    i % prepared.grid_cols + 1,
                    cfg.tokens_for_grid(s.gh(cfg), s.gw(cfg)),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    // HF's processor ALWAYS wraps the placeholder run in <|image_start|> …
    // <|image_end|> (single-tile included — pinned source §1.5), so both
    // markers are unconditionally required from the tokenizer.
    body.push(start_id.ok_or("tokenizer missing <|image_start|>")?);

    if multi_tile {
        for &(row, col, ntok) in &tiles {
            let marker = special(&format!("<|img_row_{row}_col_{col}|>"))
                .ok_or_else(|| format!("tokenizer missing <|img_row_{row}_col_{col}|>"))?;
            body.push(marker);
            for _ in 0..ntok {
                body.push(image_token_id);
            }
        }
        if cfg.use_thumbnail {
            let thumb_marker = special("<|img_thumbnail|>")
                .ok_or("multi-tile image requires <|img_thumbnail|> but tokenizer lacks it")?;
            body.push(thumb_marker);
            for _ in 0..cfg.tokens_for_grid(
                prepared.sub_images.last().unwrap().gh(cfg),
                prepared.sub_images.last().unwrap().gw(cfg),
            ) {
                body.push(image_token_id);
            }
        }
    } else {
        for _ in 0..cfg.tokens_for_grid(
            prepared.sub_images[0].gh(cfg),
            prepared.sub_images[0].gw(cfg),
        ) {
            body.push(image_token_id);
        }
    }
    body.push(end_id.ok_or("tokenizer missing <|image_end|>")?);

    // splice over the single <image>
    let idx = prompt_ids
        .iter()
        .position(|&t| t == image_token_id)
        .expect("position checked above");
    let mut out = Vec::with_capacity(prompt_ids.len() + body.len() - 1);
    out.extend_from_slice(&prompt_ids[..idx]);
    out.extend_from_slice(&body);
    out.extend_from_slice(&prompt_ids[idx + 1..]);

    let total = out.iter().filter(|&&t| t == image_token_id).count();
    if total != prepared.total_tokens(cfg) {
        return Err(format!(
            "placeholder/feature mismatch after expansion: {total} placeholders vs {} projected tokens",
            prepared.total_tokens(cfg)
        ));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_lfm2_vl(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    params: &GenerateVLParams,
) {
    let GenerateVLParams {
        id,
        prompt,
        system_prompt,
        ref image_source,
        temp,
        top_p,
        max_tokens,
        max_think_tokens,
        seed,
        ..
    } = *params;
    crate::ar::emit_generation_start(crate::ar::GenerationRoute::LfmAr, stdout, id, false);
    if m.tokenizer.is_none() {
        emit_active_attempt_error(
            stdout,
            Some(id),
            "tokenizer not loaded",
            "validation",
            false,
            false,
        );
        return;
    }
    let vision_cfg = match m.lfm2_vision() {
        Some((vc, _)) => vc.clone(),
        None => {
            emit_active_attempt_error(
                stdout,
                Some(id),
                "model has no vision encoder",
                "validation",
                false,
                false,
            );
            return;
        }
    };

    // Full-turn clock: preprocess + tower encode + prefill + decode. The
    // tower dominates image turns (~7–10 s of the ~22 s wall on gfx1101),
    // so a total that excludes it would misreport the turn.
    let t_turn = Instant::now();

    // ── Preprocess (CPU decode + resize/split/thumbnail) ──
    let prepared = match image_source {
        ImageSource::Path(path) => {
            lfm2vl::load_and_preprocess(std::path::Path::new(path), &vision_cfg)
        }
        ImageSource::Base64(b64) => {
            let raw_b64 = if let Some(rest) = b64.strip_prefix("data:") {
                match rest.split_once(',') {
                    Some((_, after)) => after,
                    None => {
                        emit_active_attempt_error(
                            stdout,
                            Some(id),
                            "malformed data URL: missing ',' separator",
                            "validation",
                            false,
                            false,
                        );
                        return;
                    }
                }
            } else {
                b64
            };
            match Engine::decode(&base64::engine::general_purpose::STANDARD, raw_b64) {
                Ok(bytes) => lfm2vl::load_and_preprocess_from_bytes(&bytes, &vision_cfg),
                Err(e) => Err(format!("failed to decode base64 image data: {e}")),
            }
        }
    };
    let prepared = match prepared {
        Ok(p) => p,
        Err(e) => {
            emit_active_attempt_error(stdout, Some(id), &e, "validation", false, false);
            return;
        }
    };
    let n_visual_tokens = prepared.total_tokens(&vision_cfg);
    eprintln!(
        "[daemon/vl-lfm2] image preprocessed: {} sub-image(s), {}x{} grid(s), {} visual tokens",
        prepared.sub_images.len(),
        prepared.grid_rows,
        prepared.grid_cols,
        n_visual_tokens
    );

    // ── Prompt build: normal ChatML jinja frame with a literal <image> ──
    let image_token_id = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        match tokenizer.special_token_id("<image>") {
            Some(t) => t,
            None => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    "tokenizer has no <image> token — not a VL tokenizer",
                    "validation",
                    false,
                    false,
                );
                return;
            }
        }
    };
    let prompt_ids: Vec<u32> = {
        let tokenizer = m.tokenizer.as_ref().unwrap();
        let template = m.chat_template.as_ref();
        let user_content = format!("<image>{prompt}");
        let rendered = match template {
            Some(template) => hipfire_runtime::prompt_frame::JinjaChatFrame {
                tokenizer,
                template,
                system: system_prompt,
                user: &user_content,
                enable_thinking: max_think_tokens != 1,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: None,
            }
            .render(),
            None => Err("no chat template embedded in artifact".to_string()),
        };
        match rendered {
            Ok(text) => tokenizer.encode(&text),
            Err(e) => {
                emit_active_attempt_error(
                    stdout,
                    Some(id),
                    &format!("vl prompt render failed: {e}"),
                    "validation",
                    false,
                    false,
                );
                return;
            }
        }
    };
    let prompt_ids = match expand_image_placeholders(
        m.tokenizer.as_ref().unwrap(),
        &prompt_ids,
        image_token_id,
        &prepared,
        &vision_cfg,
    ) {
        Ok(p) => p,
        Err(e) => {
            emit_active_attempt_error(stdout, Some(id), &e, "validation", false, false);
            return;
        }
    };

    // Capacity guard BEFORE GPU work. VL forces the cold-reset path (below),
    // so seq_pos is always 0 here.
    let cap = m.lfm2moe().unwrap().state.max_seq;
    if prompt_ids.len().saturating_add(max_tokens) > cap {
        emit_active_attempt_error(
            stdout,
            Some(id),
            &format!(
                "prompt exceeds context capacity: prompt={} (incl. {} visual) + max_tokens={} > capacity={}",
                prompt_ids.len(),
                n_visual_tokens,
                max_tokens,
                cap
            ),
            "context_length",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }

    // Cross-conversation cold reset through the common LFM attestation adapter.
    // Every non-success exit after this point must reset first and publish one
    // terminal reflecting the rollback outcome. A failed reset itself is
    // fail-closed: no token stream has been emitted yet, so the single error
    // terminal carries rolled_back=false and its context.
    let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
    if !ep.rolled_back {
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            "LFM2 session reset failed before prefill",
            "gpu",
            true,
            &ep,
        );
        return;
    }

    // ── Vision encode → projected [n_visual, hidden] rows ──
    let visual_tokens = match m.lfm2_vision() {
        Some((_, vw)) => lfm2vl::vision_forward(gpu, vw, &vision_cfg, &prepared),
        None => {
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_fail_closed_error(
                stdout,
                Some(id),
                "model has no vision encoder",
                "internal",
                false,
                &ep,
            );
            return;
        }
    };
    let visual_tokens = match visual_tokens {
        Ok(v) => v,
        Err(e) => {
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_fail_closed_error(stdout, Some(id), &e, "gpu", true, &ep);
            return;
        }
    };
    // Release-mode fail-closed length check (the narrow spec §2.3 claims
    // "fails loud BEFORE splicing"): debug_assert compiles out in release,
    // and a mismatch would otherwise panic the daemon on the emb slice below
    // instead of erroring the request. Unreachable while the splitter and
    // the tower share tokens_for_grid, but the splice must not trust that
    // by accident.
    if visual_tokens.len() != n_visual_tokens * vision_cfg.out_hidden_size {
        let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
        crate::common::emit_fail_closed_error(
            stdout,
            Some(id),
            &format!(
                "vision/projector row mismatch: {} floats for {} tokens × {} dims",
                visual_tokens.len(),
                n_visual_tokens,
                vision_cfg.out_hidden_size
            ),
            "gpu",
            true,
            &ep,
        );
        return;
    }

    // EOS-class stop set (verbatim rationale lives at the text path copy).
    let stop_toks: Vec<u32> = {
        let tk = m.tokenizer.as_ref().unwrap();
        let eos_tok = m.lfm2moe().unwrap().eos_tok;
        let mut v = vec![eos_tok];
        for s in ["<|endoftext|>", "</s>", "<|im_end|>"] {
            let ids = tk.encode(s);
            if ids.len() == 1 && !v.contains(&ids[0]) {
                v.push(ids[0]);
            }
        }
        v
    };

    // ── Prefill with visual embeddings spliced at placeholder positions ──
    // Every exit is fail-closed: the live bundle borrow is released before
    // the common adapter attests the rollback, then exactly one terminal
    // reflecting that attestation is published.
    let mut last_logits: Vec<f32> = Vec::new();
    let prefill_t0 = Instant::now();
    let mut prefill_aborted = false;
    let mut prefill_err: Option<String> = None;
    {
        let b = m.lfm2moe_mut().unwrap();
        let cfg_t = &b.config;
        let weights = &b.weights;
        let state = &mut b.state;
        let dim = vision_cfg.out_hidden_size;
        let mut vis_idx = 0usize;
        let mut position = state.n_tokens as u32;
        for &tok in &prompt_ids {
            if check_abort(id) {
                prefill_aborted = true;
                break;
            }
            let res = if tok == image_token_id && vis_idx < n_visual_tokens {
                let emb = &visual_tokens[vis_idx * dim..(vis_idx + 1) * dim];
                vis_idx += 1;
                hipfire_arch_lfm2moe::forward::prefill_embed_step(
                    cfg_t, weights, state, gpu, emb, position,
                )
            } else {
                hipfire_arch_lfm2moe::forward::decode_step(
                    cfg_t, weights, state, gpu, tok, position,
                )
            };
            match res {
                Ok(logits) => last_logits = logits,
                Err(e) => {
                    prefill_err = Some(format!("lfm2-vl prefill failed: {e:?}"));
                    break;
                }
            }
            position += 1;
        }
    }
    if let Some(msg) = prefill_err {
        let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
        crate::common::emit_fail_closed_error(stdout, Some(id), &msg, "gpu", true, &ep);
        return;
    }
    if prefill_aborted || check_abort(id) {
        let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
        crate::common::emit_spec_cancel_after_rollback(stdout, id, 0, &ep);
        return;
    }
    for &tok in &prompt_ids {
        m.conversation_tokens.push(tok);
    }
    let prefill_ms = prefill_t0.elapsed().as_millis().max(1);

    // ── Decode loop (identical sampling/stop semantics to generate_lfm2moe) ──
    // Per-request sampler seed from hipfire-engine::request_seed_for (wire
    // `seed` wins, else attempt key + counter). Matches generate_vl / text
    // paths — never wall-clock entropy.
    let mut rng = hipfire_arch_deepseek4::sampling::Xorshift::new(seed as u64);

    let mut generated_count: usize = 0;
    let decode_t0 = Instant::now();
    let mut decode_err: Option<String> = None;
    loop {
        if check_abort(id) {
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
            return;
        }
        if generated_count >= max_tokens {
            break;
        }
        let next_tok =
            hipfire_arch_deepseek4::sampling::sample_token(&last_logits, temp, 0, top_p, &mut rng);
        if stop_toks.contains(&next_tok) {
            break;
        }
        let frag = m.tokenizer.as_ref().unwrap().decode(&[next_tok]);
        if matches!(
            frag.trim(),
            "<|endoftext|>" | "</s>" | "<|im_end|>" | "<|startoftext|>"
        ) {
            break;
        }
        let envelope = serde_json::json!({
            "type": "token",
            "id": id,
            "text": frag,
            "attempt_id": active_attempt_id(),
        });
        let _ = writeln!(stdout, "{}", envelope);
        let _ = stdout.flush();
        m.conversation_tokens.push(next_tok);
        generated_count += 1;

        if check_abort(id) {
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
            return;
        }
        let step = {
            let b = m.lfm2moe_mut().unwrap();
            let position = b.state.n_tokens as u32;
            hipfire_arch_lfm2moe::forward::decode_step(
                &b.config,
                &b.weights,
                &mut b.state,
                gpu,
                next_tok,
                position,
            )
        };
        match step {
            Ok(logits) => last_logits = logits,
            Err(e) => {
                decode_err = Some(format!("lfm2-vl decode failed: {e:?}"));
                break;
            }
        }
    }
    if let Some(msg) = decode_err {
        let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
        crate::common::emit_fail_closed_error(stdout, Some(id), &msg, "gpu", true, &ep);
        return;
    }
    let decode_ms = decode_t0.elapsed().as_millis().max(1);

    // Post-loop abort latch — MANDATORY before the two-phase commit
    // handshake. If the client cancelled/disconnected mid-decode,
    // `await_client_terminal_commit` would block forever waiting for a
    // commit that can never arrive and wedge the single slot (the exact
    // failure recorded in the 2026-08-27 serve ledger). Emits the CANONICAL
    // cancelled-terminal pair via the attested adapter — serve's stream
    // reader only releases an HTTP handler on the recognized terminal
    // dialect, so a raw custom event here would hold the admission guard
    // forever.
    if check_abort(id) {
        let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
        crate::common::emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
        return;
    }

    m.seq_pos = m.lfm2moe().unwrap().state.n_tokens;

    let tok_s: f64 = if generated_count > 0 {
        (generated_count as f64 * 1000.0) / decode_ms as f64
    } else {
        0.0
    };
    let pending_done = serde_json::json!({
        "type": "done",
        "id": id,
        "tokens": generated_count,
        "tok_s": (tok_s * 100.0).round() / 100.0,
        "prefill_ms": prefill_ms,
        "total_ms": t_turn.elapsed().as_millis().max(1),
        "attempt_id": active_attempt_id(),
    });
    match await_client_terminal_commit(stdout, id, &pending_done) {
        ClientTerminalDecision::Commit => {
            crate::ar::emit_active_route_done(stdout, id, &pending_done)
        }
        ClientTerminalDecision::Abort => {
            // Same release contract as the post-loop latch: the terminal pair
            // must be the recognized wire dialect or serve holds its
            // admission guard forever. Attest rollback first.
            let ep = crate::common::production_fail_closed_rollback(m, gpu, None, None);
            crate::common::emit_spec_cancel_after_rollback(stdout, id, generated_count, &ep);
        }
    }
}

#[cfg(test)]
mod lfm2_vl_lifecycle_tests {
    use crate::ar::{generation_route_adapter, GenerationRoute};
    use crate::common::{
        emit_fail_closed_error, emit_spec_cancel_after_rollback, RollbackEpilogue,
    };
    use hipfire_engine::terminal::{
        activate_terminal_control, clear_terminal_control, set_active_attempt_id,
    };
    use std::sync::{Mutex, OnceLock};

    fn terminal_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn parse_lines(buf: &[u8]) -> Vec<serde_json::Value> {
        String::from_utf8_lossy(buf)
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .collect()
    }

    fn with_lfm_route<F: FnOnce(&mut Vec<u8>)>(
        id: &str,
        attempt: u64,
        f: F,
    ) -> Vec<serde_json::Value> {
        let _guard = terminal_lock();
        clear_terminal_control();
        activate_terminal_control(id, attempt);
        set_active_attempt_id(attempt);
        // GenerationRoute::LfmAr adapter is installed by emit_generation_start in production;
        // for unit tests we set it explicitly so the attested helpers route through the same adapter.
        crate::ar::set_generation_route(GenerationRoute::LfmAr);
        let mut out = Vec::new();
        // Ensure gen_start is claimed so the terminal can be published — mirrors production order
        // where every non-success LFM exit occurs after gen_start.
        crate::ar::emit_generation_start(GenerationRoute::LfmAr, &mut out, id, false);
        let start_len = out.len();
        f(&mut out);
        let lines = parse_lines(&out);
        // Gen_start is present; return only terminals after it so assertions count exactly one semantic owner.
        // Keep helper to validate that gen_start was not duplicated or lost.
        assert_eq!(
            lines
                .first()
                .and_then(|v| v.get("type"))
                .and_then(|v| v.as_str()),
            Some("gen_start"),
            "LFM route must emit gen_start first"
        );
        let tail = parse_lines(&out[start_len..]);
        clear_terminal_control();
        set_active_attempt_id(0);
        tail
    }

    #[test]
    fn initial_reset_failure_publishes_one_fail_closed_error() {
        let id = "lfm-vl-initial-reset";
        let tail = with_lfm_route(id, 11, |out| {
            let ep = RollbackEpilogue {
                rolled_back: false,
                context: Some(
                    "LFM2 session reset failed before prefill: device_synchronize: HipError(42)"
                        .to_string(),
                ),
            };
            emit_fail_closed_error(
                out,
                Some(id),
                "LFM2 session reset failed before prefill",
                "gpu",
                true,
                &ep,
            );
        });
        assert_eq!(
            tail.len(),
            1,
            "initial reset failure must publish exactly one terminal"
        );
        let err = &tail[0];
        assert_eq!(err.get("type").and_then(|v| v.as_str()), Some("error"));
        assert_eq!(
            err.get("rolled_back").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert!(
            err.get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .contains("device_synchronize"),
            "fail-closed error must carry rollback context: {err:?}"
        );
    }

    #[test]
    fn initial_reset_failure_with_attested_rollback_still_fail_closed_but_rolled_back_true() {
        // Defensive: if the adapter ever reports rolled_back true for an initial failure, the error still carries rolled_back=true.
        let id = "lfm-vl-initial-reset-ok";
        let tail = with_lfm_route(id, 12, |out| {
            let ep = RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            emit_fail_closed_error(
                out,
                Some(id),
                "LFM2 session reset failed before prefill",
                "gpu",
                true,
                &ep,
            );
        });
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].get("type").and_then(|v| v.as_str()), Some("error"));
        assert_eq!(
            tail[0].get("rolled_back").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn partial_prefill_abort_attested_publishes_single_cancel_pair() {
        let id = "lfm-vl-prefill-abort";
        let tail = with_lfm_route(id, 13, |out| {
            let ep = RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            emit_spec_cancel_after_rollback(out, id, 0, &ep);
        });
        // Cancel via the LFM adapter is the fold-compatible aborted + done pair — exactly one semantic owner.
        let types: Vec<_> = tail
            .iter()
            .filter_map(|v| v.get("type").and_then(|v| v.as_str()))
            .collect();
        assert!(
            types.contains(&"aborted"),
            "attested cancel must emit aborted: {types:?}"
        );
        // The done is the second line of the pair; count aborted + error as semantic owners.
        let aborted = types.iter().filter(|&&t| t == "aborted").count();
        let done = types.iter().filter(|&&t| t == "done").count();
        let error = types.iter().filter(|&&t| t == "error").count();
        assert_eq!(aborted, 1, "exactly one aborted: {types:?}");
        assert_eq!(error, 0, "attested abort must not emit error: {types:?}");
        // done is present as the commit's second record (fold keeps aborted+done as one owner)
        assert!(done <= 1, "at most one done with aborted: {types:?}");
    }

    #[test]
    fn partial_prefill_abort_unattested_publishes_one_fail_closed_error() {
        let id = "lfm-vl-prefill-abort-fail";
        let tail = with_lfm_route(id, 14, |out| {
            let ep = RollbackEpilogue {
                rolled_back: false,
                context: Some("device_synchronize: HipError(99)".to_string()),
            };
            emit_spec_cancel_after_rollback(out, id, 0, &ep);
        });
        assert_eq!(
            tail.len(),
            1,
            "unattested abort must publish exactly one terminal"
        );
        assert_eq!(tail[0].get("type").and_then(|v| v.as_str()), Some("error"));
        assert_eq!(
            tail[0].get("rolled_back").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert!(
            tail[0]
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .contains("fail-closed"),
            "unattested cancel must be fail-closed: {tail:?}"
        );
    }

    #[test]
    fn forward_error_prefill_publishes_one_fail_closed_error_reflecting_rollback() {
        let id = "lfm-vl-forward-prefill";
        // Attested rollback: error with rolled_back true
        let tail_ok = with_lfm_route(id, 15, |out| {
            let ep = RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            emit_fail_closed_error(
                out,
                Some(id),
                "lfm2-vl prefill failed: HipError(1)",
                "gpu",
                true,
                &ep,
            );
        });
        assert_eq!(tail_ok.len(), 1);
        assert_eq!(
            tail_ok[0].get("rolled_back").and_then(|v| v.as_bool()),
            Some(true)
        );
        // Unattested: error with context, rolled_back false
        let tail_fail = with_lfm_route(id, 16, |out| {
            let ep = RollbackEpilogue {
                rolled_back: false,
                context: Some("Architecture: VL recurrent reset: HipError(2); GraphsAndSynchronize: device_synchronize: HipError(3)".to_string()),
            };
            emit_fail_closed_error(
                out,
                Some(id),
                "lfm2-vl prefill failed: HipError(1)",
                "gpu",
                true,
                &ep,
            );
        });
        assert_eq!(tail_fail.len(), 1);
        assert_eq!(
            tail_fail[0].get("rolled_back").and_then(|v| v.as_bool()),
            Some(false)
        );
        assert!(tail_fail[0]
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("Architecture"));
    }

    #[test]
    fn forward_error_decode_publishes_one_fail_closed_error() {
        let id = "lfm-vl-forward-decode";
        let tail = with_lfm_route(id, 17, |out| {
            let ep = RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            emit_fail_closed_error(
                out,
                Some(id),
                "lfm2-vl decode failed: HipError(7)",
                "gpu",
                true,
                &ep,
            );
        });
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].get("type").and_then(|v| v.as_str()), Some("error"));
        // No second terminal — exactly once.
        assert_eq!(
            tail.iter()
                .filter(|v| v.get("type").and_then(|x| x.as_str()) == Some("done"))
                .count(),
            0
        );
    }

    #[test]
    fn post_decode_abort_attested_publishes_single_cancel() {
        let id = "lfm-vl-post-decode-abort";
        let tail = with_lfm_route(id, 18, |out| {
            let ep = RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            emit_spec_cancel_after_rollback(out, id, 42, &ep);
        });
        let types: Vec<_> = tail
            .iter()
            .filter_map(|v| v.get("type").and_then(|v| v.as_str()))
            .collect();
        assert!(
            types.contains(&"aborted"),
            "post-decode attested abort must emit aborted"
        );
        assert_eq!(
            tail.iter()
                .filter(|v| v.get("type").and_then(|x| x.as_str()) == Some("error"))
                .count(),
            0
        );
    }

    #[test]
    fn post_decode_abort_unattested_is_fail_closed_error() {
        let id = "lfm-vl-post-decode-abort-fail";
        let tail = with_lfm_route(id, 19, |out| {
            let ep = RollbackEpilogue {
                rolled_back: false,
                context: Some("GraphsAndSynchronize: device_synchronize: HipError(5)".to_string()),
            };
            emit_spec_cancel_after_rollback(out, id, 5, &ep);
        });
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].get("type").and_then(|v| v.as_str()), Some("error"));
        assert_eq!(
            tail[0].get("rolled_back").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn handshake_abort_attested_publishes_single_cancel() {
        let id = "lfm-vl-handshake-abort";
        let tail = with_lfm_route(id, 20, |out| {
            let ep = RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            emit_spec_cancel_after_rollback(out, id, 7, &ep);
        });
        let types: Vec<_> = tail
            .iter()
            .filter_map(|v| v.get("type").and_then(|v| v.as_str()))
            .collect();
        assert!(types.contains(&"aborted"));
        // Handshake abort must not leak a second terminal after the pair.
        assert!(types.iter().filter(|&&t| t == "aborted").count() == 1);
    }

    #[test]
    fn handshake_abort_unattested_is_fail_closed() {
        let id = "lfm-vl-handshake-abort-fail";
        let tail = with_lfm_route(id, 21, |out| {
            let ep = RollbackEpilogue {
                rolled_back: false,
                context: Some("Checkpoints: free".to_string()),
            };
            emit_spec_cancel_after_rollback(out, id, 7, &ep);
        });
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].get("type").and_then(|v| v.as_str()), Some("error"));
    }

    #[test]
    fn every_lfm_non_success_exit_is_exactly_one_terminal() {
        let _guard = terminal_lock();
        // Cross-cut: initial-reset, prefill-abort, forward-error, post-decode-abort, handshake-abort
        // must each be exactly one semantic terminal, and a second claim is rejected.
        let id = "lfm-vl-exactly-once";
        let attempt = 22;
        clear_terminal_control();
        activate_terminal_control(id, attempt);
        set_active_attempt_id(attempt);
        crate::ar::set_generation_route(GenerationRoute::LfmAr);
        let mut out = Vec::new();
        crate::ar::emit_generation_start(GenerationRoute::LfmAr, &mut out, id, false);
        let adapter = generation_route_adapter(GenerationRoute::LfmAr).unwrap();
        let start_lines = parse_lines(&out);
        assert_eq!(start_lines.len(), 1);
        // First terminal wins.
        let ep_ok = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_spec_cancel_after_rollback(&mut out, id, 0, &ep_ok);
        let after_first = parse_lines(&out);
        // Subsequent fail-closed error must be suppressed by the exactly-once latch.
        let ep_fail = RollbackEpilogue {
            rolled_back: false,
            context: Some("late".to_string()),
        };
        emit_fail_closed_error(&mut out, Some(id), "late error", "gpu", true, &ep_fail);
        // Also try a second cancel via the adapter directly — must be ignored.
        let mut late = Vec::new();
        adapter.emit_terminal(&mut late, id, attempt, crate::ar::RouteTerminal::Cancel);
        out.extend(late);
        let final_lines = parse_lines(&out);
        // Only one semantic owner after gen_start: the first cancel's aborted(+done) counts as one.
        let tail = &final_lines[start_lines.len()..];
        let aborted = tail
            .iter()
            .filter(|v| v.get("type").and_then(|x| x.as_str()) == Some("aborted"))
            .count();
        let done = tail
            .iter()
            .filter(|v| v.get("type").and_then(|x| x.as_str()) == Some("done"))
            .count();
        let error = tail
            .iter()
            .filter(|v| v.get("type").and_then(|x| x.as_str()) == Some("error"))
            .count();
        // aborted+done is one owner, error would be a second — must be 1 owner total.
        let owners = if aborted == 1 { 1 } else { error };
        assert_eq!(owners, 1, "LFM non-success must be exactly one terminal, got aborted={aborted} done={done} error={error}");
        assert_eq!(
            final_lines.len(),
            after_first.len(),
            "late terminals must be suppressed"
        );
        clear_terminal_control();
        set_active_attempt_id(0);
    }

    #[test]
    fn terminal_claim_requires_exact_request_and_attempt() {
        let _guard = terminal_lock();
        // Stale or mismatched (request_id, attempt_id) must not claim a terminal.
        let id = "lfm-vl-claim-key";
        let attempt = 30;
        clear_terminal_control();
        activate_terminal_control(id, attempt);
        set_active_attempt_id(attempt);
        crate::ar::set_generation_route(GenerationRoute::LfmAr);
        let mut out = Vec::new();
        crate::ar::emit_generation_start(GenerationRoute::LfmAr, &mut out, id, false);
        let adapter = generation_route_adapter(GenerationRoute::LfmAr).unwrap();
        // Correct key succeeds.
        let ep = RollbackEpilogue {
            rolled_back: true,
            context: None,
        };
        emit_spec_cancel_after_rollback(&mut out, id, 1, &ep);
        let after_correct = out.clone();
        // Stale attempt id — must be rejected (no new bytes).
        let mut stale = Vec::new();
        adapter.emit_terminal(
            &mut stale,
            id,
            attempt + 1,
            crate::ar::RouteTerminal::Cancel,
        );
        // Wrong request id — must be rejected.
        let mut wrong_id = Vec::new();
        adapter.emit_terminal(
            &mut wrong_id,
            "other-id",
            attempt,
            crate::ar::RouteTerminal::Error,
        );
        assert!(stale.is_empty(), "stale attempt must not claim terminal");
        assert!(
            wrong_id.is_empty(),
            "wrong request_id must not claim terminal"
        );
        // Exactly-once latch still holds — second correct claim is also rejected.
        let mut second = Vec::new();
        adapter.emit_terminal(&mut second, id, attempt, crate::ar::RouteTerminal::Done);
        assert!(
            second.is_empty(),
            "second claim after terminal must be suppressed"
        );
        // Original output unchanged.
        assert_eq!(out, after_correct);
        clear_terminal_control();
        set_active_attempt_id(0);
    }
}
