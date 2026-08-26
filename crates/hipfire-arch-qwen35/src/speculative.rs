// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Speculative decoding infrastructure for hipfire.
//!
//! Phase 1: holds target + draft model slots side-by-side on a single shared
//! `Gpu`. The actual speculative decode loop (draft → verify → accept) lives
//! in `spec_loop` once Phase 2 lands. For now, each slot just supports
//! independent forward passes so we can validate that loading two models at
//! once works and that both produce coherent output.
//!
//! Both slots share the same `Gpu` instance — HIP kernels run serialized on
//! the default stream, and the MQ rotation scratch buffers on `Gpu` are reused
//! across calls. This is correct as long as we never have two in-flight GEMVs
//! on different models sharing the same MQ scratch (which we won't, since
//! speculative decode serializes draft-generate then target-verify).

use crate::carrier::Qwen35Bundle;
use crate::dflash_verify_pm4::{
    fingerprint_u64, DflashVerifyBinding, DflashVerifyPm4, DflashVerifyPm4Phase, DflashVerifyRoute,
    DflashVerifyWindow,
};
use crate::qwen35::{self, DeltaNetState, Qwen35Config, Qwen35Scratch, Qwen35Weights};
use hip_bridge::{DeviceBuffer, HipResult, Stream};
use hipfire_runtime::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, KvCache};
use hipfire_runtime::tokenizer::{Tokenizer, TokenizerError};
use rdna_compute::{Gpu, GpuTensor};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// #397 Ship 5.3: route a single spec-decode (DFlash) batched GEMM through
/// [`GemmFamily::run_key`](hipfire_dispatch::families::gemm::GemmFamily::run_key)
/// against an *explicit* dispatcher-entry [`KernelKey`].
///
/// Behavior-preserving migration primitive for the draft/verify lm_head GEMM
/// call sites in this module — the spec-decode analogue of qwen35.rs's
/// `run_plain_gemm_key`. Passing the dispatcher-entry key
/// (`GemmQ8_0BatchedChunked`, `GemmQ8_0Batched`, `GemmHfq4G256`,
/// `GemmHfq4G256BatchedLmhead`, `GemmHfq3G256BatchedLmhead`,
/// `GemmHfq6G256BatchedLmhead`) makes `run_key` dispatch to the IDENTICAL
/// `gpu.gemm_*` method the direct call used, so each method's own internal arch
/// routing (WMMA for batch>1 on gfx11/gfx12, dp4a on gfx906, fp16/scalar
/// fallback otherwise) is preserved byte-for-byte on every (dtype × arch ×
/// shape). All keys used here are registered `ArchPredicate::Always`, so
/// `run_key`'s registry check never rejects on a supported build. `resolve()`
/// is deliberately NOT used (it would front-run the kernel's internal dispatch
/// with a dtype-keyed WMMA preference and could diverge on some arches). The
/// weight buffer, `x`, output `y`, and m/k/n are passed in the IDENTICAL order
/// the prior direct call used.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_spec_gemm_key(
    gpu: &mut Gpu,
    key: hipfire_dispatch::types::KernelKey,
    w_buf: &GpuTensor,
    w_dtype: rdna_compute::DType,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemm::GemmParams;
    use hipfire_dispatch::families::gemv::WeightRef;
    use std::cell::RefCell;
    // Cache the DispatchCtx per thread instead of rebuilding it on every call.
    // `DispatchCtx::new` lowers the immutable process snapshot into FeatureFlags
    // plus ArchCaps and ResourceManager; its own doc says it is "resolved once
    // ... and shared immutably across all dispatch calls". This helper is hit
    // ~170×/generation by the DFlash verify+draft
    // lm_head loop, so reconstructing the ctx per call cost ~9 tok/s at constant
    // τ (gfx1201, 27B AWQ DFlash). `gemm_family()` is already a OnceLock
    // singleton; this brings the ctx in line. Keyed on `gpu.arch` so a thread
    // that drives a different arch (multi-GPU) rebuilds rather than reusing stale.
    thread_local! {
        static SPEC_CTX: RefCell<Option<(String, DispatchCtx)>> = const { RefCell::new(None) };
    }
    let w = WeightRef {
        buf: w_buf,
        dtype: w_dtype,
        m,
        k,
        row_stride: k,
        rotation: None,
        awq_scale: None,
    };
    let params = GemmParams {
        w: &w,
        x,
        y,
        batch_size: n,
    };
    SPEC_CTX.with(|cell| {
        let needs_rebuild = {
            let slot = cell.borrow();
            slot.as_ref().map_or(true, |(arch, _)| arch != &gpu.arch)
        };
        if needs_rebuild {
            let fresh = DispatchCtx::new(gpu);
            *cell.borrow_mut() = Some((gpu.arch.clone(), fresh));
        }
        let slot = cell.borrow();
        let ctx = &slot.as_ref().unwrap().1;
        hipfire_runtime::llama::gemm_family()
            .run_key(key, ctx, gpu, &params)
            .map_err(hip_bridge::HipError::from)
    })
}

fn dflash_gemm_q8_lmhead(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    if gpu.flags.dflash_q8_lmhead_wmma {
        // #397 Ship 5.3: GemmQ8_0BatchedChunked routes to the identical
        // gpu.gemm_q8_0_batched_chunked the prior direct call used.
        return run_spec_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
            &w_out.buf,
            w_out.gpu_dtype,
            x,
            y,
            w_out.m,
            w_out.k,
            n,
        );
    }

    const Q8_LM_MAX: usize = 64;
    let mut chunk_start = 0usize;
    while chunk_start < n {
        let chunk_end = (chunk_start + Q8_LM_MAX).min(n);
        let chunk_n = chunk_end - chunk_start;
        let x_chunk = x.sub_offset(chunk_start * w_out.k, chunk_n * w_out.k);
        let y_chunk = y.sub_offset(chunk_start * w_out.m, chunk_n * w_out.m);
        // #397 Ship 5.3: GemmQ8_0Batched routes to the identical
        // gpu.gemm_q8_0_batched the prior direct call used.
        run_spec_gemm_key(
            gpu,
            hipfire_dispatch::types::KernelKey::GemmQ8_0Batched,
            &w_out.buf,
            w_out.gpu_dtype,
            &x_chunk,
            &y_chunk,
            w_out.m,
            w_out.k,
            chunk_n,
        )?;
        chunk_start = chunk_end;
    }
    Ok(())
}

fn dflash_moe_verify_graph_lmhead_enabled_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        None => true,
    }
}

fn dflash_moe_verify_graph_lmhead_eligible(
    num_experts: usize,
    want_full_logits: bool,
    tree_verify_present: bool,
    env_value: Option<&str>,
) -> bool {
    // MoE-only: dense DFlash keeps the old forward-only verify graph. The
    // extended graph is also greedy-only because sampling needs full logits.
    num_experts > 0
        && !want_full_logits
        && !tree_verify_present
        && dflash_moe_verify_graph_lmhead_enabled_from_env_value(env_value)
}

fn dflash_moe_draft_ffn_graph_eligible(
    num_experts: usize,
    ctx_slice_present: bool,
    pld_present: bool,
    use_temp_sampling: bool,
    env_value: Option<&str>,
) -> bool {
    // The draft FFN graph captures fixed-B per-layer work. Keep it off for
    // dense, PLD, ctx-slice diagnostics, and sampling until each path is
    // separately benched/coherence-gated.
    num_experts > 0
        && !ctx_slice_present
        && !pld_present
        && !use_temp_sampling
        && dflash_moe_verify_graph_lmhead_enabled_from_env_value(env_value)
}

/// Whether `HIPFIRE_VERIFY_GRAPH` admits the HipGraph verify route for this
/// (arch, target lm_head dtype) pair.
///
/// `dc105ea64` newly admitted MQ2/3/4/5/6G256V2 to gfx1100 batched WMMA, and
/// `verify_graph_ok` shares that eligibility via `prefill_batch_pbs_eligible`.
/// Graph-off direct and forced-blob direct full-model V2 fixtures pass on
/// gfx1100, but two graph-on V2 campaigns lost the endpoint. This is a
/// default-off graph capability quarantine for exact gfx1100 + V2 only;
/// direct batched WMMA remains on. `HIPFIRE_VERIFY_GRAPH=1` opts back in
/// diagnostically; `=0` force-offs everywhere; gfx1151/gfx12 and non-V2
/// gfx1100 stay default-on.
fn dflash_verify_graph_env_eligible(
    arch: &str,
    output_dtype: rdna_compute::DType,
    env_value: Option<&str>,
) -> bool {
    if env_value == Some("0") {
        return false;
    }
    let is_mq_v2 = matches!(
        output_dtype,
        rdna_compute::DType::MQ4G256V2
            | rdna_compute::DType::MQ6G256V2
            | rdna_compute::DType::MQ5G256V2
            | rdna_compute::DType::MQ3G256V2
            | rdna_compute::DType::MQ2G256V2
    );
    if arch == "gfx1100" && is_mq_v2 {
        return env_value == Some("1");
    }
    true
}

fn dflash_batched_lm_head_supported(dtype: rdna_compute::DType) -> bool {
    matches!(
        dtype,
        rdna_compute::DType::Q8_0
            | rdna_compute::DType::HFQ4G256
            | rdna_compute::DType::MQ4G256
            | rdna_compute::DType::MQ4G256V2
            | rdna_compute::DType::MQ6G256V2
            | rdna_compute::DType::MQ5G256V2
            | rdna_compute::DType::MQ3G256V2
            | rdna_compute::DType::MQ2G256V2
            | rdna_compute::DType::MQ3G256
            | rdna_compute::DType::HFQ6G256
            | rdna_compute::DType::MQ6G256
    )
}

fn dflash_enqueue_verify_lm_head(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    final_hidden: &GpuTensor,
    verify_scratch: &VerifyScratch,
    b: usize,
    vocab: usize,
) -> HipResult<()> {
    let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
    match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            dflash_gemm_q8_lmhead(gpu, w_out, final_hidden, &logits_batch, b)?;
        }
        rdna_compute::DType::HFQ4G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                final_hidden,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ4G256 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ4G256V2 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ4 v2 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            gpu.gemm_mq4g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ6G256V2 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ6V2 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            gpu.gemm_mq6g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ5G256V2 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ5V2 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            gpu.gemm_mq5g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ3G256V2 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ3V2 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            gpu.gemm_mq3g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ2G256V2 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ2V2 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            gpu.gemm_mq2g256v2_batched_lmhead(
                &w_out.buf,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ3G256 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ3 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::HFQ6G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                final_hidden,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        rdna_compute::DType::MQ6G256 => {
            assert!(
                b * w_out.k <= verify_scratch.max_n * verify_scratch.hidden_k,
                "verify_scratch.rot undersized for MQ6 lm_head: b*k={} > max_n*hidden_k={}",
                b * w_out.k,
                verify_scratch.max_n * verify_scratch.hidden_k
            );
            let rot = verify_scratch.rot.sub_offset(0, b * w_out.k);
            llama::rotate_x_mq_batched_for(gpu, w_out, final_hidden, &rot, w_out.k, b)?;
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rot,
                &logits_batch,
                w_out.m,
                w_out.k,
                b,
            )?;
        }
        other => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("DFlash verify graph lm_head unsupported dtype {other:?}"),
            ));
        }
    }

    Ok(())
}

fn dflash_enqueue_verify_lm_head_argmax(
    gpu: &mut Gpu,
    w_out: &llama::WeightTensor,
    final_hidden: &GpuTensor,
    verify_scratch: &VerifyScratch,
    b: usize,
    vocab: usize,
) -> HipResult<()> {
    dflash_enqueue_verify_lm_head(gpu, w_out, final_hidden, verify_scratch, b, vocab)?;
    let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
    let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
    gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)
}

fn dflash_download_verify_argmax(
    gpu: &Gpu,
    verify_scratch: &VerifyScratch,
    b: usize,
) -> HipResult<Vec<u32>> {
    let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
    let mut host_idx = vec![0i32; b];
    {
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, b * 4) };
        gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
    }
    Ok(host_idx.into_iter().map(|idx| idx as u32).collect())
}

/// Fold a DFlash2 candidate-selector proposal into the chain draft buffers.
///
/// Greedy: tokens only — no full-vocab D2H. Temperature: each sparse q row
/// is materialized into a dense `draft_softmaxes` vector (zeros outside the
/// candidate set) and `draft_probs_at_drafted` is taken from
/// `selected_probabilities`, so the existing host rejection/residual path
/// stays mathematically exact. Request top_p/top_k is NOT applied to this q.
/// `proposal.probabilities` is the flattened normalized selector q over every
/// top-K candidate row when temperature>0 (not raw unary logits); greedy may
/// return `None` probabilities.
fn apply_dflash2_selector_proposal(
    proposal: dflash::DflashCandidateProposal,
    vocab: usize,
    use_temp_sampling: bool,
    drafted: &mut Vec<u32>,
    draft_softmaxes: &mut Vec<Vec<f32>>,
    draft_probs_at_drafted: &mut Vec<f32>,
) -> HipResult<()> {
    let rows = proposal.tokens.len();
    let top_k = proposal.top_k;
    // Flattened length guards — silently dropping tail masses would make the
    // sparse residual subtract an incomplete q and corrupt the bonus distribution.
    if proposal.candidates.len() != rows * top_k {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "selector proposal candidates len {} != rows {} * top_k {}",
                proposal.candidates.len(),
                rows,
                top_k
            ),
        ));
    }
    if use_temp_sampling {
        let probs = proposal.probabilities.as_ref().ok_or_else(|| {
            hip_bridge::HipError::new(0, "selector proposal missing probabilities for temp>0")
        })?;
        if probs.len() != rows * top_k {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "selector proposal probabilities len {} != rows {} * top_k {}",
                    probs.len(),
                    rows,
                    top_k
                ),
            ));
        }
        let sel = proposal.selected_probabilities.as_ref().ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                "selector proposal missing selected_probabilities for temp>0",
            )
        })?;
        if sel.len() != rows {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "selector proposal selected_probabilities len {} != rows {}",
                    sel.len(),
                    rows
                ),
            ));
        }
        drafted.extend_from_slice(&proposal.tokens);
        draft_softmaxes.reserve(rows);
        draft_probs_at_drafted.reserve(rows);
        for i in 0..rows {
            let mut row = vec![0f32; vocab];
            let off = i * top_k;
            for j in 0..top_k {
                let cand = proposal.candidates[off + j] as usize;
                let q = probs[off + j];
                if cand < vocab {
                    row[cand] = q;
                } else {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!("selector candidate id {} out of vocab {}", cand, vocab),
                    ));
                }
            }
            draft_softmaxes.push(row);
            draft_probs_at_drafted.push(sel[i]);
        }
        Ok(())
    } else {
        // Greedy: no q materialization, but still validate candidate shape and
        // that probabilities is None (or absent); selected may be present as 1.0.
        if proposal.probabilities.is_some() {
            // Greedy may return None probabilities per contract; if Some, still
            // require correct shape but do not materialize.
            let probs = proposal.probabilities.as_ref().unwrap();
            if probs.len() != rows * top_k {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "selector proposal greedy probabilities len {} != rows {} * top_k {}",
                        probs.len(),
                        rows,
                        top_k
                    ),
                ));
            }
        }
        if let Some(sel) = &proposal.selected_probabilities {
            if sel.len() != rows {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "selector proposal greedy selected_probabilities len {} != rows {}",
                        sel.len(),
                        rows
                    ),
                ));
            }
        }
        drafted.extend_from_slice(&proposal.tokens);
        Ok(())
    }
}

/// Task #93 Phase B seed-prediction oracle counters.
///
/// Three proxies, all derived from data the draft already computes (zero
/// extra device work). For each cycle:
///   - REJ_BOUNDARY: `drafted[accept_len + 1] == bonus_token` (PRD's "naive"
///     proxy — argmax at rejection position). Zero-by-construction when
///     `accept_len < b - 1` because the accept loop broke precisely because
///     those didn't match. Reported anyway to document the dead-end.
///   - TAIL: `drafted[b - 1] == bonus_token`. Draft's final-position argmax.
///     Gives a non-zero signal. If the usual case is "target's bonus happens
///     at position b-1 because accept_len = b-2", this proxy catches those.
///   - ANYPOS: `bonus_token ∈ drafted[1..b]`. Upper bound of any position-
///     based single-guess proxy. Useful as a ceiling.
///
/// FULLACCEPT counts cycles where `accept_len == b - 1` (full acceptance —
/// draft has no native prediction at position `b`, so REJ_BOUNDARY is
/// undefined and TAIL/ANYPOS are the only candidates there).
///
static SEED_ORACLE_TOTAL: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_REJ_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_TAIL_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_ANYPOS_MATCH: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_FULLACCEPT: AtomicU64 = AtomicU64::new(0);
static SEED_ORACLE_ACCEPT_LEN_SUM: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default)]
pub struct SeedOracleStats {
    pub total: u64,
    pub rej_match: u64,
    pub tail_match: u64,
    pub anypos_match: u64,
    pub full_accept: u64,
    pub accept_len_sum: u64,
}

/// Snapshot the process-global seed-oracle counters.
pub fn read_seed_oracle_stats() -> SeedOracleStats {
    SeedOracleStats {
        total: SEED_ORACLE_TOTAL.load(Ordering::Relaxed),
        rej_match: SEED_ORACLE_REJ_MATCH.load(Ordering::Relaxed),
        tail_match: SEED_ORACLE_TAIL_MATCH.load(Ordering::Relaxed),
        anypos_match: SEED_ORACLE_ANYPOS_MATCH.load(Ordering::Relaxed),
        full_accept: SEED_ORACLE_FULLACCEPT.load(Ordering::Relaxed),
        accept_len_sum: SEED_ORACLE_ACCEPT_LEN_SUM.load(Ordering::Relaxed),
    }
}

/// Zero all seed-oracle counters. Call before a fresh generation run.
pub fn reset_seed_oracle_stats() {
    SEED_ORACLE_TOTAL.store(0, Ordering::Relaxed);
    SEED_ORACLE_REJ_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_TAIL_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_ANYPOS_MATCH.store(0, Ordering::Relaxed);
    SEED_ORACLE_FULLACCEPT.store(0, Ordering::Relaxed);
    SEED_ORACLE_ACCEPT_LEN_SUM.store(0, Ordering::Relaxed);
}

/// DDTree meta-verifier pruner telemetry: per-cycle tree-size histogram.
/// `cycle_count` = cycles observed; `total_nodes` = sum of tree.num_nodes()
/// across cycles; `max_nodes` / `min_nodes` = range observed.
static DDTREE_META_CYCLES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_TOTAL_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MAX_NODES: AtomicU64 = AtomicU64::new(0);
static DDTREE_META_MIN_NODES: AtomicU64 = AtomicU64::new(u64::MAX);

pub fn record_ddtree_meta_nodes(n: usize) {
    let n64 = n as u64;
    DDTREE_META_CYCLES.fetch_add(1, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.fetch_add(n64, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.fetch_max(n64, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.fetch_min(n64, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DdtreeMetaStats {
    pub cycles: u64,
    pub total_nodes: u64,
    pub max_nodes: u64,
    pub min_nodes: u64,
}

pub fn read_ddtree_meta_stats() -> DdtreeMetaStats {
    let c = DDTREE_META_CYCLES.load(Ordering::Relaxed);
    DdtreeMetaStats {
        cycles: c,
        total_nodes: DDTREE_META_TOTAL_NODES.load(Ordering::Relaxed),
        max_nodes: DDTREE_META_MAX_NODES.load(Ordering::Relaxed),
        min_nodes: if c == 0 {
            0
        } else {
            DDTREE_META_MIN_NODES.load(Ordering::Relaxed)
        },
    }
}

pub fn reset_ddtree_meta_stats() {
    DDTREE_META_CYCLES.store(0, Ordering::Relaxed);
    DDTREE_META_TOTAL_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MAX_NODES.store(0, Ordering::Relaxed);
    DDTREE_META_MIN_NODES.store(u64::MAX, Ordering::Relaxed);
}

/// Which KV cache layout to use when allocating a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// INT8 co-located K and V (default).
    Q8,
    /// Asym4: rotated 4-bit K + Q8 V (smaller than Q8, higher-fidelity than asym3).
    Asym4,
    /// Asym3: rotated 3-bit K + Q8 V. ~2.7× less KV BW than Q8, tightly-tuned
    /// kernel for the hot FA attention path. Good choice for long-context verify.
    Asym3,
    /// Asym2: rotated 2-bit K + Q8 V. Smallest but most lossy.
    Asym2,
    /// Fwht4: signed-FWHT-rotated 4-bit K + Q8 V. Byte-identical storage to
    /// Asym4 but with a Hadamard rotation (matches MQ4's weight-quant trick).
    /// Centroid LUTs were always Lloyd-Max-fit for post-FWHT N(0, 1/128) per
    /// turbo_common.h:13 — Fwht4 finally uses them on the distribution they
    /// were calibrated for. Opt-in via `--kv-mode fwht4`.
    Fwht4,
    /// Fwht3: signed-FWHT-256 rotated 3-bit K + Q8 V. Byte-identical storage to
    /// Asym3 (the canonical default). Single-pass 256-element FWHT — the
    /// natural fit for asym3's existing layout (8 dims/thread). Empirical
    /// prose-τ win on 3.5-27b at the 4-bit tier suggests the 3-bit tier
    /// should benefit even more from rotation. Opt-in via `--kv-mode fwht3`.
    Fwht3,
    /// Fwht2: signed-FWHT-128 rotated 2-bit K + Q8 V. Byte-identical storage
    /// to Asym2. 2-pass-over-128 structure matches fwht4. Highest theoretical
    /// leverage tier — Asym2 is doc'd "most lossy" and 2-bit centroid quant
    /// suffers most from outliers. Opt-in via `--kv-mode fwht2`.
    Fwht2,
}

impl Default for KvMode {
    fn default() -> Self {
        KvMode::Q8
    }
}

/// Configuration for loading a single model slot.
#[derive(Debug, Clone)]
pub struct ModelSlotConfig {
    pub max_seq: usize,
    pub kv_mode: KvMode,
    pub repeat_window: usize,
    pub state_quant: qwen35::StateQuant,
}

impl Default for ModelSlotConfig {
    fn default() -> Self {
        Self {
            max_seq: 2048,
            kv_mode: KvMode::Q8,
            repeat_window: 128,
            state_quant: qwen35::StateQuant::Q8,
        }
    }
}

/// A single loaded Qwen3.5 model with its own KV cache, DeltaNet state, and
/// forward-pass scratch. The `Gpu` is borrowed, not owned — multiple slots
/// share one `Gpu` instance.
pub struct ModelSlot {
    pub name: String,
    pub hfq: HfqFile,
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    pub scratch: Qwen35Scratch,
    pub slot_config: ModelSlotConfig,
    /// DSpark (EAGLE-3) residual-hidden extract-layer ids. Empty for every
    /// non-DSpark path (AR / n-gram / DFlash / MTP), which keeps `new_spec_scratch`
    /// building a `num_extract = 0` no-op hidden ring — byte-identical to the
    /// pre-DSpark behaviour. A DSpark drafter populates this via
    /// `set_dflash_extract_layers` / `capture_seed_main_hidden` so the per-window
    /// verify captures hidden at exactly the sidecar's layer ids.
    pub dspark_extract_layers: Vec<usize>,
    /// Vision tower carried through the slot guard so `Qwen35Bundle` round-trips
    /// without loss. The slot never drives `vision_forward`; it just parks the
    /// tower while the bundle is out of `ModelState`.
    pub vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    pub vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
}

impl ModelSlot {
    /// Assemble the spec-decode target slot from a live [`Qwen35Bundle`] that was
    /// parked out of `ModelState`, opening the model's `HfqFile` (the mmap handle
    /// `ModelSlot` carries but the spec kernels never read).
    ///
    /// **Fallible WITHOUT loss:** on a reopen failure the bundle is handed BACK in
    /// the `Err` so the caller's RAII guard can re-park it and restore it into
    /// `ModelState` on `Drop` — consuming the bundle on the error path would drop
    /// it and leave `m.state == None`, reintroducing the #462 cross-request
    /// state-bleed class this seam exists to prevent. This is the only piece of
    /// `Qwen35Bundle` field knowledge the loader's guard previously inlined; it
    /// lives here so the loader never names the bundle's fields.
    pub fn from_bundle(bundle: Qwen35Bundle, path: &Path) -> Result<Self, (Qwen35Bundle, String)> {
        let hfq = match HfqFile::open(path) {
            Ok(h) => h,
            Err(e) => return Err((bundle, format!("reopen model: {e}"))),
        };
        let Qwen35Bundle {
            config,
            weights,
            scratch,
            kv_cache,
            dn_state,
            kv_adaptive: _,
            pp_scratch_set,
            vision_config,
            vision_weights,
            qwen35_decode_batch,
        } = bundle;
        debug_assert!(
            pp_scratch_set.is_none(),
            "ModelSlot::from_bundle: pp_scratch_set must be None (pp>1 never enters spec slot)"
        );
        let _ = pp_scratch_set;
        debug_assert!(
            qwen35_decode_batch.is_none(),
            "ModelSlot::from_bundle: qwen35_decode_batch must be None (batch staging is outside spec slot)"
        );
        let _ = qwen35_decode_batch;
        Ok(Self {
            name: String::from("target"),
            hfq,
            config,
            weights,
            kv_cache,
            dn_state,
            scratch,
            slot_config: ModelSlotConfig::default(),
            dspark_extract_layers: Vec::new(),
            vision_config,
            vision_weights,
        })
    }

    /// Disassemble the slot back into a [`Qwen35Bundle`] — the `HfqFile` mmap,
    /// `name`, and `slot_config` drop here; the five live pieces return to the
    /// bundle. The inverse of [`from_bundle`](Self::from_bundle); the loader's
    /// guard calls this on `Drop` to restore `ModelState`.
    pub fn into_bundle(self) -> Qwen35Bundle {
        Qwen35Bundle {
            config: self.config,
            weights: self.weights,
            scratch: self.scratch,
            kv_cache: self.kv_cache,
            dn_state: self.dn_state,
            // Controller lives on LoadedModel, not ModelSlot.
            kv_adaptive: None,
            pp_scratch_set: None,
            vision_config: self.vision_config,
            vision_weights: self.vision_weights,
            qwen35_decode_batch: None,
        }
    }
}

impl ModelSlot {
    /// for all allocations. `name` is a human-readable label used in logs.
    pub fn load(
        gpu: &mut Gpu,
        path: &Path,
        name: impl Into<String>,
        slot_config: ModelSlotConfig,
    ) -> HipResult<Self> {
        let name = name.into();
        let mut hfq = HfqFile::open(path).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("open {} ({}): {}", path.display(), name, e))
        })?;
        let config = qwen35::config_from_hfq(&hfq).map_err(|e| {
            hip_bridge::HipError::new(
                0,
                &format!(
                    "invalid Qwen3.5 config in {} ({}): {e}",
                    path.display(),
                    name
                ),
            )
        })?;
        let mut src = qwen35::HfqSource::new(&mut hfq, &config);
        let layout = qwen35::Layout::single(config.n_layers);
        let weights = qwen35::load_weights(&mut src, std::slice::from_mut(gpu), &layout)?;

        // For hybrid arches (Qwen 3.5 = 48 DeltaNet LinearAttention + 16
        // FullAttention out of 64 total), only the FullAttention layers need
        // a KV cache slot. The LinearAttention layers carry their own state
        // via DeltaNetState (`new_with_quant` below) and never write to
        // kv_cache.k_gpu / .v_gpu. Pre-2026-05-15 the KV constructor
        // allocated full K/V slots for ALL layers regardless of type — at
        // ctx=64K that's ~5 GB of dead allocation on 27B. The `_filtered`
        // constructors take a `is_kv_layer` slice and substitute a
        // 1-element placeholder for non-KV layers. Indexing by absolute
        // layer_idx is preserved.
        let is_kv_layer: Vec<bool> = config
            .layer_types
            .iter()
            .map(|t| *t == qwen35::LayerType::FullAttention)
            .collect();

        // Honor the caller's requested KV cache mode. Default is Q8 for
        // backwards-compat, but DFlash verify is KV-bandwidth sensitive at
        // longer contexts — asym3/asym4 cut the verify attention cost.
        let kv_cache = match slot_config.kv_mode {
            KvMode::Q8 => KvCache::new_gpu_q8_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym4 => KvCache::new_gpu_asym4_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym3 => KvCache::new_gpu_asym3_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Asym2 => KvCache::new_gpu_asym2_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Fwht4 => KvCache::new_gpu_fwht4_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Fwht3 => KvCache::new_gpu_fwht3_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
            KvMode::Fwht2 => KvCache::new_gpu_fwht2_filtered(
                gpu,
                &is_kv_layer,
                config.n_kv_heads,
                config.head_dim,
                slot_config.max_seq,
            )?,
        };

        let dn_state = DeltaNetState::new_with_quant(gpu, &config, slot_config.state_quant)?;
        let scratch = Qwen35Scratch::new(gpu, &config, slot_config.repeat_window)?;

        Ok(Self {
            name,
            hfq,
            config,
            weights,
            kv_cache,
            dn_state,
            scratch,
            slot_config,
            dspark_extract_layers: Vec::new(),
            vision_config: None,
            vision_weights: None,
        })
    }

    /// Load the tokenizer from this slot's HFQ metadata. Each slot technically
    /// carries its own tokenizer; callers should validate that two slots'
    /// tokenizers are compatible via `Tokenizer::is_compatible_with` before
    /// sharing. Returns the underlying `TokenizerError` on failure so callers
    /// can surface specific diagnostics (e.g. `MissingMergeResult` from a
    /// truncated quantizer output) rather than a generic "no tokenizer"
    /// message — see #203.
    pub fn load_tokenizer(&self) -> Result<Tokenizer, TokenizerError> {
        Tokenizer::from_hfq_metadata(&self.hfq.metadata_json)
    }

    /// Single-token forward pass. Writes logits into `self.scratch.logits`.
    pub fn forward(&mut self, gpu: &mut Gpu, token: u32, pos: usize) -> HipResult<()> {
        gpu.graphs.ar_graph_eligible = false; // spec re-seed: never the plain-AR graph
        qwen35::forward_scratch(
            gpu,
            &self.weights,
            &self.config,
            token,
            pos,
            &mut self.kv_cache,
            &mut self.dn_state,
            &self.scratch,
        )
    }

    /// Reset the DeltaNet recurrent state (S/scale/conv + default-on EF residual)
    /// and zero the KV write head. Does NOT shrink the KV allocation — callers
    /// track `seq_pos` separately.
    pub fn reset_state(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        // Canonical path: `DeltaNetState::reset` is stream-aware and zeroes
        // s_ef_residual alongside S/scale/conv. Hand-rolled memset loops here
        // previously omitted EF and left residual noise across cold resets.
        self.dn_state.reset(gpu)?;
        self.kv_cache.compact_offset = 0;
        Ok(())
    }
}

/// A pair of target + draft slots sharing one `Gpu` and one tokenizer.
///
/// Phase 1 just carries both slots. Phase 2+ adds the `spec_decode_step`
/// method for the verify-and-accept loop.
pub struct SpecPair {
    pub target: ModelSlot,
    pub draft: ModelSlot,
    pub tokenizer: Tokenizer,
}

impl SpecPair {
    /// Load target and draft from separate HFQ files on the same `Gpu`.
    /// Validates that the two models share a compatible tokenizer before
    /// returning — speculative decode requires identical vocab + token IDs.
    pub fn load(
        gpu: &mut Gpu,
        target_path: &Path,
        draft_path: &Path,
        target_cfg: ModelSlotConfig,
        draft_cfg: ModelSlotConfig,
    ) -> HipResult<Self> {
        let target = ModelSlot::load(gpu, target_path, "target", target_cfg)?;
        let draft = ModelSlot::load(gpu, draft_path, "draft", draft_cfg)?;

        let target_tok = target.load_tokenizer().map_err(|e| {
            hip_bridge::HipError::new(0, &format!("target tokenizer load failed: {e}"))
        })?;
        let draft_tok = draft.load_tokenizer().map_err(|e| {
            hip_bridge::HipError::new(0, &format!("draft tokenizer load failed: {e}"))
        })?;

        if target_tok.vocab_size() != draft_tok.vocab_size() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer mismatch: target vocab={}, draft vocab={}. \
                     Speculative decode requires identical vocabularies.",
                    target_tok.vocab_size(),
                    draft_tok.vocab_size()
                ),
            ));
        }

        // Sanity-check a round-trip on a common string — catches vocab-size
        // match but token-ID mismatch (different BPE merges producing same
        // vocab count).
        let probe = "<|im_start|>user\nHello world\n<|im_end|>";
        let a = target_tok.encode(probe);
        let b = draft_tok.encode(probe);
        if a != b {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "tokenizer merge rules diverge: target={:?}, draft={:?}. \
                     Speculative decode requires identical tokenization.",
                    &a, &b
                ),
            ));
        }

        Ok(Self {
            target,
            draft,
            tokenizer: target_tok,
        })
    }

    /// Run a minimal smoke test: 8 forward passes on each slot with a dummy
    /// token sequence, ensuring neither model crashes and the logits buffers
    /// contain finite values. Returns `(target_ok, draft_ok)`.
    pub fn smoke_test(&mut self, gpu: &mut Gpu) -> HipResult<(bool, bool)> {
        // Token ID 1 is a safe placeholder for both Qwen3 and Qwen3.5; the
        // smoke test only checks that the forward pass runs without crashing
        // and produces finite logits.
        let probe_token: u32 = 1;
        for pos in 0..8 {
            self.target.forward(gpu, probe_token, pos)?;
        }
        for pos in 0..8 {
            self.draft.forward(gpu, probe_token, pos)?;
        }
        let target_logits = gpu.download_f32(&self.target.scratch.logits)?;
        let draft_logits = gpu.download_f32(&self.draft.scratch.logits)?;
        let target_ok = target_logits.iter().take(1024).all(|x| x.is_finite());
        let draft_ok = draft_logits.iter().take(1024).all(|x| x.is_finite());

        // Reset both after the smoke test so the caller starts from a clean
        // state at seq_pos=0.
        self.target.reset_state(gpu)?;
        self.draft.reset_state(gpu)?;

        Ok((target_ok, draft_ok))
    }
}

/// Result of one speculative decode step.
#[derive(Debug, Clone)]
pub struct SpecStepResult {
    /// Number of draft tokens accepted (0..=k).
    pub accepted: usize,
    /// Target's next-token prediction at the first rejection point (or after
    /// all drafted tokens if accepted == k). Appended to `committed`.
    pub bonus_token: u32,
    /// The full sequence of tokens the draft proposed this cycle.
    pub drafted: Vec<u32>,
    /// The tokens actually committed to both models: `drafted[..accepted]`
    /// followed by `bonus_token`. Always non-empty (length = accepted + 1).
    pub committed: Vec<u32>,
}

/// Backing storage for a DeltaNetState snapshot. Holds device buffers sized
/// to match the source state's tensors. Allocate once per slot, reuse across
/// all speculative cycles.
///
/// Includes the default-on Q8 error-feedback residual (`s_ef_residual`) when
/// present. Empty when EF is off (`HIPFIRE_DN_STATE_EF=0`) or non-Q8 quant —
/// save/restore/free then no-op over that vector, matching the live state.
pub struct DeltaNetSnapshot {
    s_matrix_bufs: Vec<DeviceBuffer>,
    s_scale_bufs: Vec<DeviceBuffer>,
    conv_state_bufs: Vec<DeviceBuffer>,
    /// F16 per-element EF residual backups; `len == state.s_ef_residual.len()`.
    s_ef_residual_bufs: Vec<DeviceBuffer>,
}

impl DeltaNetSnapshot {
    /// Allocate backup buffers matching `state`'s shapes (incl. EF residual).
    pub fn new_for(gpu: &mut Gpu, state: &DeltaNetState) -> HipResult<Self> {
        let mut s_matrix_bufs = Vec::with_capacity(state.s_matrices.len());
        for t in &state.s_matrices {
            s_matrix_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        let mut s_scale_bufs = Vec::with_capacity(state.s_scales.len());
        for t in &state.s_scales {
            s_scale_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        let mut conv_state_bufs = Vec::with_capacity(state.conv_states.len());
        for t in &state.conv_states {
            conv_state_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        let mut s_ef_residual_bufs = Vec::with_capacity(state.s_ef_residual.len());
        for t in &state.s_ef_residual {
            s_ef_residual_bufs.push(gpu.hip.malloc(t.buf.size())?);
        }
        Ok(Self {
            s_matrix_bufs,
            s_scale_bufs,
            conv_state_bufs,
            s_ef_residual_bufs,
        })
    }

    /// Number of EF residual backup buffers (0 when EF is off).
    #[inline]
    pub fn s_ef_len(&self) -> usize {
        self.s_ef_residual_bufs.len()
    }

    /// Copy live state → backup (S/scale/conv + EF residual).
    pub fn save_from(&mut self, state: &DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        for (dst, src) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        for (dst, src) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        for (dst, src) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        for (dst, src) in self
            .s_ef_residual_bufs
            .iter()
            .zip(state.s_ef_residual.iter())
        {
            gpu.hip.memcpy_dtod(dst, &src.buf, src.buf.size())?;
        }
        Ok(())
    }

    /// Async copy live state → backup on `stream`.
    ///
    /// Caller owns cross-stream ordering. MTP trunk-spine uses this as an
    /// opt-in experiment to overlap DN snapshot copy with proposal work.
    pub fn save_from_async_on(
        &mut self,
        state: &DeltaNetState,
        gpu: &Gpu,
        stream: &Stream,
    ) -> HipResult<()> {
        for (dst, src) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        for (dst, src) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        for (dst, src) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        for (dst, src) in self
            .s_ef_residual_bufs
            .iter()
            .zip(state.s_ef_residual.iter())
        {
            gpu.hip
                .memcpy_dtod_async_at(dst, 0, &src.buf, 0, src.buf.size(), stream)?;
        }
        Ok(())
    }

    /// Copy backup → live state (rewinds recurrent + EF residual to the snapshot).
    pub fn restore_to(&self, state: &mut DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        for (src, dst) in self.s_matrix_bufs.iter().zip(state.s_matrices.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        for (src, dst) in self.s_scale_bufs.iter().zip(state.s_scales.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        for (src, dst) in self.conv_state_bufs.iter().zip(state.conv_states.iter()) {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        for (src, dst) in self
            .s_ef_residual_bufs
            .iter()
            .zip(state.s_ef_residual.iter())
        {
            gpu.hip.memcpy_dtod(&dst.buf, src, src.size())?;
        }
        Ok(())
    }

    /// Free the backup GPU buffers, consuming the snapshot. `DeviceBuffer` has
    /// no `Drop`, so a bare `Vec::clear()`/`truncate()` on a checkpoint ring
    /// orphans this device memory — the source of the per-reset GPU-memory leak
    /// that OOMs long-lived serves (a fresh `hipMalloc` per reset, never freed).
    /// Every site that drops a snapshot must route through here.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for b in self.s_matrix_bufs {
            let _ = gpu.hip.free(b);
        }
        for b in self.s_scale_bufs {
            let _ = gpu.hip.free(b);
        }
        for b in self.conv_state_bufs {
            let _ = gpu.hip.free(b);
        }
        for b in self.s_ef_residual_bufs {
            let _ = gpu.hip.free(b);
        }
    }
}

/// A series of `n_slots` `DeltaNetSnapshot` slots, used by the tape-replay
/// rollback path. After each verify forward step writes its post-state into
/// the next slot, `restore_from(accept_len + 1)` jumps the live DN state
/// to exactly `start + accept_len + 1` positions of advance — no replay
/// loop needed.
///
/// VRAM cost: `n_slots × (one DeltaNetSnapshot)`. For Qwen3.5-4B and
/// `n_slots = B + 1 = 17`, that's roughly 100 MB; for 9B it scales with
/// the hybrid layer count.
pub struct DeltaNetTape {
    pub slots: Vec<DeltaNetSnapshot>,
}

/// Innovation tape for the GatedDeltaNet recurrence. During a batched verify
/// forward we capture the per-LA-layer pre-conv1d `qkv` projection and the
/// post-sigmoid `(α, β)` for every block position. On rollback we replay
/// conv1d + QK-norm + repeat-interleave + GDN for `accept_len + 1` steps
/// against the pre-verify DN snapshot — advancing both S-state AND
/// conv_state correctly, no full target re-run needed.
///
/// Why pre-conv1d qkv instead of post-conv1d (q, k, v): conv_state is a
/// recurrent buffer advanced by conv1d_silu_split. If we skipped conv1d on
/// replay the next verify would see a stale conv_state reflecting the
/// previous full-B aborted trajectory rather than the accepted prefix —
/// small numerical drift that empirically halves τ on our 4B hybrid target.
/// Running conv1d from the captured qkv advances conv_state to the right
/// place.
pub struct GdnTape {
    pub max_n: usize,
    pub qkv_dim: usize,
    pub v_dim: usize,
    pub k_dim: usize,
    pub n_v_heads: usize,
    pub n_key_heads: usize,
    pub value_head_dim: usize,
    pub key_head_dim: usize,
    /// Per-LA-layer [max_n × qkv_dim] F32 — raw qkvza projection output.
    pub qkv_bufs: Vec<GpuTensor>,
    /// Per-LA-layer [max_n × n_v_heads] F32 — post-sigmoid_alpha_gate.
    pub alpha_bufs: Vec<GpuTensor>,
    pub beta_bufs: Vec<GpuTensor>,
    /// Replay scratch (shared across layers — serial replay is fine).
    pub q_raw_scratch: GpuTensor, // [max_n × k_dim]
    pub k_raw_scratch: GpuTensor, // [max_n × k_dim]
    pub v_scratch: GpuTensor,     // [max_n × v_dim]
    pub q_scratch: GpuTensor,     // [max_n × v_dim] (post repeat-interleave)
    pub k_scratch: GpuTensor,     // [max_n × v_dim]
    pub attn_scratch: GpuTensor,  // [max_n × v_dim]
}

impl GdnTape {
    pub fn new_for_config(
        gpu: &mut Gpu,
        config: &qwen35::Qwen35Config,
        max_n: usize,
    ) -> HipResult<Self> {
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let n_v_heads = config.linear_num_value_heads;
        let n_key_heads = config.linear_num_key_heads;
        let n_la_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == qwen35::LayerType::LinearAttention)
            .count();

        let mut qkv_bufs = Vec::with_capacity(n_la_layers);
        let mut alpha_bufs = Vec::with_capacity(n_la_layers);
        let mut beta_bufs = Vec::with_capacity(n_la_layers);
        for _ in 0..n_la_layers {
            qkv_bufs.push(gpu.alloc_tensor(&[max_n * qkv_dim], rdna_compute::DType::F32)?);
            alpha_bufs.push(gpu.alloc_tensor(&[max_n * n_v_heads], rdna_compute::DType::F32)?);
            beta_bufs.push(gpu.alloc_tensor(&[max_n * n_v_heads], rdna_compute::DType::F32)?);
        }

        Ok(Self {
            max_n,
            qkv_dim,
            v_dim,
            k_dim,
            n_v_heads,
            n_key_heads,
            value_head_dim: config.linear_value_head_dim,
            key_head_dim: config.linear_key_head_dim,
            qkv_bufs,
            alpha_bufs,
            beta_bufs,
            q_raw_scratch: gpu.alloc_tensor(&[max_n * k_dim], rdna_compute::DType::F32)?,
            k_raw_scratch: gpu.alloc_tensor(&[max_n * k_dim], rdna_compute::DType::F32)?,
            v_scratch: gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
            q_scratch: gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
            k_scratch: gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
            attn_scratch: gpu.alloc_tensor(&[max_n * v_dim], rdna_compute::DType::F32)?,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self
            .qkv_bufs
            .into_iter()
            .chain(self.alpha_bufs.into_iter())
            .chain(self.beta_bufs.into_iter())
        {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.q_raw_scratch);
        let _ = gpu.free_tensor(self.k_raw_scratch);
        let _ = gpu.free_tensor(self.v_scratch);
        let _ = gpu.free_tensor(self.q_scratch);
        let _ = gpu.free_tensor(self.k_scratch);
        let _ = gpu.free_tensor(self.attn_scratch);
    }

    /// Replay the full LA sub-pipeline (conv1d + qk-l2norm + repeat-interleave +
    /// GDN recurrence) for `n_steps` across all LinearAttention layers. Advances
    /// both `dn_state.s_matrices`/`s_scales` AND `dn_state.conv_states` by
    /// exactly `n_steps` single-token updates. Caller must have restored the
    /// DN snapshot to the pre-verify point before calling this.
    ///
    /// Graph-capture path (OPT-IN with HIPFIRE_REPLAY_GRAPH=1): per distinct
    /// n_steps, the first call runs direct as a warmup, the second captures a
    /// hipGraph, and subsequent calls replay the graph. Eligibility:
    /// gpu.active_stream must be Some (so a verify-graph path that already
    /// created one has run first in this cycle).
    ///
    /// MEASURED NULL RESULT (2026-04-21, 27B HumanEval @ accept≈10):
    /// 77.27 → 77.41 tok/s (+0.18 %, noise). τ and mean_committed byte-exact
    /// across A/B. Replay's ~192 kernel launches per cycle add ~0.3 ms of
    /// dispatch API time out of a 130 ms cycle — graphing them saves that
    /// 0.3 ms but cycle cost lives in GDN kernel execution time (scales
    /// linearly with n_steps). Kept as opt-in infrastructure for future
    /// launch-overhead-dominated workloads (smaller models, finer kernels).
    pub fn replay_gdn(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &mut qwen35::DeltaNetState,
        n_steps: usize,
    ) -> HipResult<()> {
        let graph_enabled = hipfire_config::developer_var("HIPFIRE_REPLAY_GRAPH")
            .ok()
            .as_deref()
            == Some("1");
        let can_graph = graph_enabled && gpu.active_stream.is_some();

        if can_graph && gpu.graphs.replay_has_graph(n_steps) {
            return gpu.graphs.replay_graph_launch(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                n_steps,
            );
        }

        if can_graph && gpu.graphs.replay_needs_warmup(n_steps) {
            self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps)?;
            gpu.graphs.replay_mark_warmup_done(n_steps);
            return Ok(());
        }

        if can_graph {
            gpu.graphs.begin_replay_graph_capture(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                n_steps,
            )?;
            let r = self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps);
            if r.is_ok() {
                gpu.graphs.end_replay_graph_capture(
                    &gpu.hip,
                    gpu.device_id,
                    gpu.active_stream.as_ref().unwrap(),
                )?;
                // Same pattern as verify_graph: hipStreamBeginCapture records
                // without executing, so launch once here to apply this cycle's
                // state updates.
                gpu.graphs.replay_graph_launch(
                    &gpu.hip,
                    gpu.device_id,
                    gpu.active_stream.as_ref().unwrap(),
                    n_steps,
                )?;
                return Ok(());
            } else {
                let _ = gpu
                    .hip
                    .stream_end_capture(gpu.active_stream.as_ref().unwrap());
                gpu.graphs.capture_mode = false;
                gpu.graphs.capture_blobs.clear();
                return r;
            }
        }

        self.replay_gdn_inner(gpu, weights, config, dn_state, n_steps)
    }

    /// Direct kernel path — the original `replay_gdn` body, retained as a
    /// helper so both the graph-warmup first call and the non-graph fallback
    /// share one implementation.
    fn replay_gdn_inner(
        &self,
        gpu: &mut Gpu,
        weights: &qwen35::Qwen35Weights,
        config: &qwen35::Qwen35Config,
        dn_state: &mut qwen35::DeltaNetState,
        n_steps: usize,
    ) -> HipResult<()> {
        assert!(
            n_steps <= self.max_n,
            "replay_gdn: n_steps {n_steps} > max_n"
        );
        let n_v_heads = self.n_v_heads;
        let n_key_heads = self.n_key_heads;
        let hd = self.key_head_dim;
        let v_dim = self.v_dim;
        let k_dim = self.k_dim;
        let value_head_dim = self.value_head_dim;
        let mut la_idx = 0usize;

        for (layer_idx, lt) in config.layer_types.iter().enumerate() {
            if *lt != qwen35::LayerType::LinearAttention {
                continue;
            }
            let conv_weight = match &weights.layers[layer_idx] {
                qwen35::LayerWeights::DeltaNet(l) => &l.conv_weight,
                qwen35::LayerWeights::DeltaNetMoe(l) => &l.conv_weight,
                _ => unreachable!("LA layer type mismatch in replay_gdn"),
            };

            // 1. conv1d + SiLU + split — advances conv_state, writes
            //    (q_raw, k_raw, v) into scratch.
            gpu.conv1d_silu_split_f32_n(
                &self.q_raw_scratch,
                &self.k_raw_scratch,
                &self.v_scratch,
                &self.qkv_bufs[la_idx],
                conv_weight,
                &dn_state.conv_states[la_idx],
                k_dim,
                v_dim,
                n_steps,
            )?;

            // 2. L2 norm(Q) + L2 norm(K) + scale(Q).
            gpu.fused_qk_l2_norm_scale_f32_batched(
                &self.q_raw_scratch,
                &self.k_raw_scratch,
                n_key_heads,
                hd,
                1.0 / (hd as f32).sqrt(),
                config.norm_eps,
                n_steps,
            )?;

            // 3. Repeat-interleave if GQA.
            if n_key_heads < n_v_heads {
                let ratio = n_v_heads / n_key_heads;
                gpu.repeat_interleave_qk_f32_batched(
                    &self.q_raw_scratch,
                    &self.k_raw_scratch,
                    &self.q_scratch,
                    &self.k_scratch,
                    n_key_heads,
                    ratio,
                    hd,
                    n_steps,
                )?;
            } else {
                let bytes = n_steps * k_dim * 4;
                gpu.hip.memcpy_dtod_at(
                    &self.q_scratch.buf,
                    0,
                    &self.q_raw_scratch.buf,
                    0,
                    bytes,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &self.k_scratch.buf,
                    0,
                    &self.k_raw_scratch.buf,
                    0,
                    bytes,
                )?;
            }

            // 4. GDN recurrence — advances S_state.
            match dn_state.quant {
                qwen35::StateQuant::FP32 => gpu.gated_delta_net_f32_batch_seq(
                    &self.q_scratch,
                    &self.k_scratch,
                    &self.v_scratch,
                    &self.alpha_bufs[la_idx],
                    &self.beta_bufs[la_idx],
                    &dn_state.s_matrices[la_idx],
                    &self.attn_scratch,
                    n_steps,
                    n_v_heads,
                    value_head_dim,
                )?,
                qwen35::StateQuant::Q8 => gpu.gated_delta_net_q8_batch_seq(
                    &self.q_scratch,
                    &self.k_scratch,
                    &self.v_scratch,
                    &self.alpha_bufs[la_idx],
                    &self.beta_bufs[la_idx],
                    &dn_state.s_matrices[la_idx],
                    &dn_state.s_scales[la_idx],
                    &self.attn_scratch,
                    n_steps,
                    n_v_heads,
                    value_head_dim,
                    dn_state.ef_residual(la_idx),
                )?,
                qwen35::StateQuant::Q4 => gpu.gated_delta_net_q4(
                    &self.q_scratch,
                    &self.k_scratch,
                    &self.v_scratch,
                    &self.alpha_bufs[la_idx],
                    &self.beta_bufs[la_idx],
                    &dn_state.s_matrices[la_idx],
                    &dn_state.s_scales[la_idx],
                    &self.attn_scratch,
                    n_steps,
                    n_v_heads,
                    value_head_dim,
                )?,
            }

            la_idx += 1;
        }
        Ok(())
    }
}

impl DeltaNetTape {
    pub fn new_for(gpu: &mut Gpu, state: &DeltaNetState, n_slots: usize) -> HipResult<Self> {
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(DeltaNetSnapshot::new_for(gpu, state)?);
        }
        Ok(Self { slots })
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn save_at(&mut self, slot: usize, state: &DeltaNetState, gpu: &mut Gpu) -> HipResult<()> {
        self.slots[slot].save_from(state, gpu)
    }

    pub fn restore_from(
        &self,
        slot: usize,
        state: &mut DeltaNetState,
        gpu: &mut Gpu,
    ) -> HipResult<()> {
        self.slots[slot].restore_to(state, gpu)
    }
}

/// Compute the DFlash target-layer extraction indices for a model of
/// `num_target_layers` layers. Matches the `build_target_layer_ids` function in
/// the DFlash reference implementation:
///
/// ```text
/// start = 1
/// end   = num_target_layers - 3        # 29 for num_target_layers=32
/// step  = (end - start) / (num_extract - 1)
/// layers[i] = round(start + i * step)  # for i in 0..num_extract
/// ```
///
/// For Qwen3.5-9B (32 layers) and 5 extraction layers this returns
/// `[1, 8, 15, 22, 29]`, matching the hard-coded indices in the HuggingFace
/// `z-lab/Qwen3.5-9B-DFlash` config.
pub fn dflash_extract_layer_ids(num_target_layers: usize, num_extract: usize) -> Vec<usize> {
    if num_extract == 0 {
        return Vec::new();
    }
    if num_extract == 1 {
        return vec![1];
    }
    let start: f32 = 1.0;
    let end: f32 = (num_target_layers as i32 - 3).max(1) as f32;
    let step = (end - start) / (num_extract as f32 - 1.0);
    (0..num_extract)
        .map(|i| (start + i as f32 * step).round() as usize)
        .collect()
}

/// Ring buffer holding the most recent `max_positions` of hidden state
/// extractions from the target model's forward pass. Each of the `extract_layers`
/// entries is a `[max_positions, hidden_dim]` f32 GPU tensor. `head` is the
/// position that the NEXT write will land at (0..max_positions). `written` is
/// the total cumulative number of writes, used to tell full vs partial buffer.
///
/// For DFlash, the draft model pulls a contiguous slice ending at the most
/// recent position to use as context KV input.
/// Persistent scratch for `spec_step_ddtree_batched` — eliminates the
/// per-cycle alloc/free churn that dominated early-benchmark wall-clock
/// time. Allocated once at session start, sized for the maximum tree we
/// may see.
///
/// Contents:
/// - `attn_bias`: `[max_n × max_n]` f32 additive bias buffer. `max_n =
///   1 + tree_budget`. Per cycle the caller uploads `big_n × big_n`
///   floats into its head — unused tail space is irrelevant because the
///   FA kernel reads at `global_bid * block_cols + col` with `block_cols
///   = big_n`.
///
/// Callers pass this by `&mut` to `spec_step_ddtree_batched`. It's OK
/// to over-allocate (max_n larger than any cycle's actual tree) — the
/// per-cycle cost is only the htod of the current cycle's mask bytes.
pub struct DdtreeScratch {
    pub max_n: usize,
    pub attn_bias: GpuTensor,
    /// Per-slot parent index consumed by tree-aware LA kernels when
    /// `HIPFIRE_DDTREE_TREE_LA=1`. `[max_n]` i32, uploaded fresh each
    /// cycle via `memcpy_htod` before calling `verify_dflash_block_tree`.
    /// Allocated as Raw bytes (4 × max_n) since there's no i32 DType.
    pub parent_indices: GpuTensor,
}

impl DdtreeScratch {
    /// Allocate for a worst-case tree of `max_budget` non-root nodes.
    pub fn new(gpu: &mut Gpu, max_budget: usize) -> HipResult<Self> {
        let max_n = 1 + max_budget;
        let attn_bias = gpu.alloc_tensor(&[max_n * max_n], rdna_compute::DType::F32)?;
        let parent_indices = gpu.alloc_tensor(&[max_n * 4], rdna_compute::DType::Raw)?;

        Ok(Self {
            max_n,
            attn_bias,
            parent_indices,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.attn_bias);
        let _ = gpu.free_tensor(self.parent_indices);
    }
}

/// Persistent per-decode-cycle scratch for the target verify pass and the
/// draft lm_head. Prior to 2026-04-16 these were allocated fresh every cycle
/// inside `verify_dflash_block_inner` and `spec_step_dflash` — ~8 hipMalloc
/// + hipFree pairs per cycle (biggest are 16 MB logits buffers). The HIP
/// allocator is 50–200 µs per call, so per-cycle overhead was 0.5–1.5 ms
/// just in allocator churn. Preallocating once at session start removes
/// the churn with no correctness impact.
///
/// `max_n` must be ≥ max of (verify block size, tree-verify node count).
/// The demo sizes it to `max(block_size, 1 + tree_budget)` to cover both
/// the vanilla DFlash and DDTree paths.
pub struct VerifyScratch {
    pub max_n: usize,
    pub dim: usize,
    pub vocab: usize,
    pub hidden_k: usize,
    /// Post-output-norm hidden from the target forward, [max_n × dim] F32.
    /// Drives the per-position lm_head GEMM.
    pub final_hidden: GpuTensor,
    /// Scratch logits from target + draft lm_head, [max_n × vocab] F32.
    /// Reused across target verify (n=B) and draft lm_head (n=B-1).
    pub logits: GpuTensor,
    /// FWHT-rotated hidden for MQ4 lm_head path, [max_n × hidden_k] F32.
    /// Allocated unconditionally; unused on non-MQ4 targets.
    pub rot: GpuTensor,
    /// Argmax output for greedy path, [max_n] f32 (treated as i32 host-side).
    pub argmax: GpuTensor,
    /// Persistent per-layer batch scratch for `qwen35::forward_prefill_batch`.
    /// Sized to `max_n`, so `verify_dflash_block` processes each block in a
    /// single chunk without the ~25 hipMalloc/hipFree pairs the in-function
    /// allocation would incur. Present whenever the caller passes a config
    /// to `VerifyScratch::with_prefill`. Absent (None) for the legacy
    /// constructor — `forward_prefill_batch` then falls back to allocating
    /// its own scratch.
    pub prefill_batch: Option<qwen35::PrefillBatchScratch>,
}

impl VerifyScratch {
    pub fn new(
        gpu: &mut Gpu,
        max_n: usize,
        dim: usize,
        vocab: usize,
        hidden_k: usize,
    ) -> HipResult<Self> {
        Ok(Self {
            max_n,
            dim,
            vocab,
            hidden_k,
            final_hidden: gpu.alloc_tensor(&[max_n * dim], rdna_compute::DType::F32)?,
            logits: gpu.alloc_tensor(&[max_n * vocab], rdna_compute::DType::F32)?,
            rot: gpu.alloc_tensor(&[max_n * hidden_k], rdna_compute::DType::F32)?,
            argmax: gpu.alloc_tensor(&[max_n], rdna_compute::DType::F32)?,
            prefill_batch: None,
        })
    }

    /// Like `new`, but also allocates a persistent `PrefillBatchScratch`
    /// sized to `max_n`. Use this for DFlash verify where the same block
    /// scratch is reused every cycle — drops ~25 tensor alloc/free pairs
    /// per cycle (measured ~3-5 ms/cycle on 27B Qwen3.5 where the per-call
    /// allocation dominated verify wall-time).
    pub fn with_prefill(
        gpu: &mut Gpu,
        max_n: usize,
        dim: usize,
        vocab: usize,
        hidden_k: usize,
        config: &qwen35::Qwen35Config,
    ) -> HipResult<Self> {
        let mut s = Self::new(gpu, max_n, dim, vocab, hidden_k)?;
        s.prefill_batch = Some(qwen35::PrefillBatchScratch::new(gpu, config, max_n)?);
        Ok(s)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.final_hidden);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.rot);
        let _ = gpu.free_tensor(self.argmax);
        if let Some(pbs) = self.prefill_batch {
            pbs.free_gpu(gpu);
        }
    }
}

pub struct HiddenStateRingBuffer {
    pub layer_bufs: Vec<GpuTensor>,
    pub extract_layers: Vec<usize>,
    pub max_positions: usize,
    pub hidden_dim: usize,
    pub head: usize,
    pub written: usize,
    /// Per-extract-layer staging buffer, shape `[max_batch × hidden_dim]`.
    /// Captured kernels (verify forward) write here with FIXED offsets so
    /// their captured pointers don't bake in a per-cycle `head`. After the
    /// graph returns, `commit_staging_to_ring` scatters staging → `layer_bufs`
    /// at the current head (outside the captured region, head-aware).
    pub staging_bufs: Vec<GpuTensor>,
    /// Max rows a single staging write can hold — sized to the maximum batch
    /// the caller ever passes to `write_rows_to_staging`. For DFlash verify
    /// this is `budget + 1`.
    pub max_batch: usize,
}

impl HiddenStateRingBuffer {
    /// Physical ring slot holding block-local row `r` of the most recent
    /// `block_size`-row block — the same mapping
    /// [`scatter_hidden_block_to_interleaved`] walks, exposed so callers that
    /// need to address block rows directly (DDTree post-accept gather) do not
    /// re-derive the ring arithmetic.
    pub fn block_row_slot(&self, block_size: usize, r: usize) -> usize {
        let r_skip = block_size.saturating_sub(self.max_positions);
        let start_slot =
            (self.head + self.max_positions - (block_size - r_skip)) % self.max_positions;
        (start_slot + r.saturating_sub(r_skip)) % self.max_positions
    }

    /// Allocate GPU ring buffer for explicit extraction layer list.
    ///
    /// `extract_layers` must be in ascending order; validation of range is the
    /// caller's responsibility ( `load_dflash_state` checks against
    /// `num_target_layers` ).
    pub fn new_for_layers(
        gpu: &mut Gpu,
        extract_layers: &[usize],
        hidden_dim: usize,
        max_positions: usize,
        max_batch: usize,
    ) -> HipResult<Self> {
        let num_extract = extract_layers.len();
        let mut layer_bufs = Vec::with_capacity(num_extract);
        let mut staging_bufs = Vec::with_capacity(num_extract);
        for _ in 0..num_extract {
            layer_bufs
                .push(gpu.alloc_tensor(&[max_positions * hidden_dim], rdna_compute::DType::F32)?);
            staging_bufs
                .push(gpu.alloc_tensor(&[max_batch * hidden_dim], rdna_compute::DType::F32)?);
        }
        Ok(Self {
            layer_bufs,
            extract_layers: extract_layers.to_vec(),
            max_positions,
            hidden_dim,
            head: 0,
            written: 0,
            staging_bufs,
            max_batch,
        })
    }

    /// Allocate GPU ring buffer for `num_extract` target layers.
    ///
    /// `max_batch` sizes the staging buffers used by the graph-capture path.
    /// Typical value for DFlash verify is `budget + 1`.
    pub fn new(
        gpu: &mut Gpu,
        num_target_layers: usize,
        num_extract: usize,
        hidden_dim: usize,
        max_positions: usize,
        max_batch: usize,
    ) -> HipResult<Self> {
        let extract_layers = dflash_extract_layer_ids(num_target_layers, num_extract);
        Self::new_for_layers(gpu, &extract_layers, hidden_dim, max_positions, max_batch)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.layer_bufs {
            let _ = gpu.free_tensor(t);
        }
        for t in self.staging_bufs {
            let _ = gpu.free_tensor(t);
        }
    }

    /// If `target_layer_idx` matches one of the extraction layers, return the
    /// index into `layer_bufs`/`extract_layers` for that layer. Otherwise None.
    #[inline]
    pub fn extract_slot(&self, target_layer_idx: usize) -> Option<usize> {
        self.extract_layers
            .iter()
            .position(|&l| l == target_layer_idx)
    }

    /// Copy `x` (shape `[hidden_dim]`) into the ring buffer slot for the given
    /// extraction layer at the CURRENT head position. Call once per extracted
    /// layer per forward pass, then `advance_head()` at the end of the forward
    /// to move to the next slot.
    pub fn write_at_head(&self, gpu: &mut Gpu, extract_idx: usize, x: &GpuTensor) -> HipResult<()> {
        let offset = self.head * self.hidden_dim * 4;
        gpu.hip.memcpy_dtod_at(
            &self.layer_bufs[extract_idx].buf,
            offset,
            &x.buf,
            0,
            self.hidden_dim * 4,
        )
    }

    /// Advance the write head. Call once per forward pass, AFTER all layer
    /// extractions for this position have been written.
    #[inline]
    pub fn advance_head(&mut self) {
        self.head = (self.head + 1) % self.max_positions;
        self.written += 1;
    }

    /// Advance the write head by `n`. Used by the batched prefill path after
    /// writing N rows per extract layer in a single dispatch.
    #[inline]
    pub fn advance_head_by(&mut self, n: usize) {
        self.head = (self.head + n) % self.max_positions;
        self.written += n;
    }

    /// Copy `n` contiguous rows from `src` (shape `[n × hidden_dim]` row-major)
    /// into the ring buffer slot for the given extraction layer, starting at
    /// the CURRENT head position. Handles the ring-buffer wrap: if head + n
    /// exceeds max_positions, the write splits into a head→end + 0→tail pair.
    /// Call this once per extracted layer per batched forward, then advance
    /// the head by `n` via `advance_head_by(n)` at the end.
    pub fn write_rows_at_head(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        src: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        let row_bytes = self.hidden_dim * 4;
        let head = self.head;
        let max_pos = self.max_positions;
        if head + n <= max_pos {
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                head * row_bytes,
                &src.buf,
                0,
                n * row_bytes,
            )?;
        } else {
            let first = max_pos - head;
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                head * row_bytes,
                &src.buf,
                0,
                first * row_bytes,
            )?;
            gpu.hip.memcpy_dtod_at(
                &self.layer_bufs[extract_idx].buf,
                0,
                &src.buf,
                first * row_bytes,
                (n - first) * row_bytes,
            )?;
        }
        Ok(())
    }

    /// Write `n` contiguous rows from `src` into the staging buffer for the
    /// given extraction layer at FIXED offset 0. Safe to call inside a
    /// hipGraph stream capture: the captured memcpy node bakes in the
    /// staging pointer (which is stable across cycles), not a per-cycle head.
    ///
    /// Callers must call `commit_staging_to_ring(n)` after the forward
    /// returns (outside the captured region) to scatter staging → `layer_bufs`
    /// at the current head, then advance the head.
    pub fn write_rows_to_staging(
        &self,
        gpu: &mut Gpu,
        extract_idx: usize,
        src: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        // Hard assert (not debug_assert): a violation silently overran the
        // staging buffer in release and surfaced as a cryptic d2d-bounds panic
        // deep in ffi.rs. Fail loud and named instead.
        assert!(
            n <= self.max_batch,
            "write_rows_to_staging: n {} > staging max_batch {} — staging buffer \
             too small for this prefill chunk",
            n,
            self.max_batch
        );
        let row_bytes = self.hidden_dim * 4;
        let bytes = n * row_bytes;
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.memcpy_dtod_async_at(
                &self.staging_bufs[extract_idx].buf,
                0,
                &src.buf,
                0,
                bytes,
                stream,
            )
        } else {
            gpu.hip
                .memcpy_dtod_at(&self.staging_bufs[extract_idx].buf, 0, &src.buf, 0, bytes)
        }
    }

    /// Scatter staging buffers into `layer_bufs` at the current head, handling
    /// ring wrap, then advance the head by `n`. Must be called AFTER the
    /// forward (outside any captured region) — uses the current `head` to
    /// compute destination offsets, which would be baked wrong in a replayed
    /// graph.
    ///
    /// When `gpu.active_stream` is Some, we first sync the stream (so the
    /// captured forward's staging writes are complete) then use sync D2D
    /// for the scatter. This matches the existing sync-memcpy semantics the
    /// rest of the engine relies on for ordering with null-stream consumers
    /// (e.g. the draft forward's D2H of hidden rows after this commit).
    pub fn commit_staging_to_ring(&mut self, gpu: &mut Gpu, n: usize) -> HipResult<()> {
        let row_bytes = self.hidden_dim * 4;
        let head = self.head;
        let max_pos = self.max_positions;

        // If running under an explicit stream (graph capture path), wait
        // for the captured writes to complete before the scatter so we
        // don't read uninitialized staging.
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        }

        for ei in 0..self.layer_bufs.len() {
            if head + n <= max_pos {
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf,
                    head * row_bytes,
                    &self.staging_bufs[ei].buf,
                    0,
                    n * row_bytes,
                )?;
            } else {
                let first = max_pos - head;
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf,
                    head * row_bytes,
                    &self.staging_bufs[ei].buf,
                    0,
                    first * row_bytes,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &self.layer_bufs[ei].buf,
                    0,
                    &self.staging_bufs[ei].buf,
                    first * row_bytes,
                    (n - first) * row_bytes,
                )?;
            }
        }
        self.head = (head + n) % max_pos;
        self.written += n;
        Ok(())
    }

    /// Reset to empty (head=0, written=0). GPU buffers are not zeroed; stale
    /// data is simply unreadable because `written < max_positions`.
    pub fn reset(&mut self) {
        self.head = 0;
        self.written = 0;
    }
}

/// Single-pass argmax for token sampling. Not SIMD-optimized — the logit
/// vector is downloaded once per verify step so the CPU scan cost is
/// negligible relative to GEMV work.
#[inline]
fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature-scaled softmax. Writes into `out` (reused across calls to
/// avoid per-position allocation in the rejection-sampling hot loop).
#[inline]
pub(crate) fn softmax_temp_into(logits: &[f32], temp: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(logits.len());
    let inv_t = 1.0 / temp;
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let s = v * inv_t;
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v * inv_t - max).exp();
        out.push(e);
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    for p in out.iter_mut() {
        *p *= inv_sum;
    }
}

/// Apply a nucleus (top_p) truncation to a normalized probability row IN PLACE,
/// given the per-row threshold `tau_cut` and kept mass `Z` produced by the GPU
/// `softmax_temp_topp_batched_f32` kernel:
///
///   p'(x) = if p(x) >= tau_cut { p(x) / Z } else { 0 }
///
/// After this the row sums to 1 (it was divided by the kept mass Z), so
/// `sample_categorical` / `sample_residual` operate on a proper distribution.
///
/// When `tau_cut == 0.0` (the kernel's top_p>=1.0 guard) this is the IDENTITY
/// (every p >= 0 is kept, Z == 1) → byte-equivalent to no truncation.
///
/// This reproduces the AR nucleus cut in `mtp_spec.rs` (sort desc, accumulate,
/// cut at the FIRST prefix whose cumulative >= top_p INCLUSIVE, renorm by kept
/// mass) in threshold form. Exact float ties at the boundary are kept together
/// here (`>=`), whereas AR cuts by sort position — a measure-zero divergence for
/// real logits.
#[inline]
pub(crate) fn apply_topp_trunc(row: &mut [f32], tau_cut: f32, z: f32) {
    if tau_cut <= 0.0 {
        // top_p disabled (or single-token row already at full mass): identity.
        return;
    }
    let inv_z = 1.0f32 / z.max(f32::MIN_POSITIVE);
    for p in row.iter_mut() {
        if *p >= tau_cut {
            *p *= inv_z;
        } else {
            *p = 0.0;
        }
    }
}

/// Host-side nucleus (top_p) truncation of an ALREADY-normalized softmax row,
/// IN PLACE, matching the AR rule in `mtp_spec.rs:307-325`: sort descending,
/// accumulate, cut at the FIRST token whose cumulative prob >= top_p (inclusive
/// crossing token), renormalize the kept set by its mass. Used on the non-fast
/// DFlash temp arms (which compute the softmax on host, not GPU) so top_p is
/// honored on the WHOLE temp path, not only under HIPFIRE_DFLASH_FAST_SAMPLE.
///
/// `top_p >= 1.0` is a no-op (identity). Produces the same kept-set and the same
/// renormalized values as `apply_topp_trunc` would given the GPU `tau_cut/Z`,
/// modulo the boundary-tie convention (this version keeps AR's sort-position
/// tie-break exactly; `apply_topp_trunc` keeps all boundary-equal tokens).
pub(crate) fn apply_host_nucleus(row: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    // Indices sorted by probability descending (stable for tie reproducibility).
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f64;
    let mut kept_mass = 0.0f64;
    let mut cutoff = order.len();
    for (rank, &idx) in order.iter().enumerate() {
        cum += row[idx] as f64;
        kept_mass += row[idx] as f64;
        if cum >= top_p as f64 {
            cutoff = rank + 1;
            break;
        }
    }
    let inv = if kept_mass > 0.0 {
        1.0f64 / kept_mass
    } else {
        0.0
    };
    // Zero out the dropped tail, renorm the kept head.
    for (rank, &idx) in order.iter().enumerate() {
        if rank < cutoff {
            row[idx] = (row[idx] as f64 * inv) as f32;
        } else {
            row[idx] = 0.0;
        }
    }
}

/// Host-side top-k truncation of an ALREADY-normalized softmax row, IN PLACE.
/// Keeps the `k` highest-probability tokens, zeros the rest, and renormalizes
/// the kept mass to 1. `k == 0` or `k >= row.len()` is a no-op (identity).
#[inline]
pub(crate) fn apply_host_topk(row: &mut [f32], k: usize) {
    if k == 0 || k >= row.len() {
        return;
    }
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept_mass = 0.0f64;
    for &idx in order.iter().take(k) {
        kept_mass += row[idx] as f64;
    }
    if kept_mass <= 0.0 {
        return;
    }
    let inv = 1.0f64 / kept_mass;
    for (rank, &idx) in order.iter().enumerate() {
        if rank < k {
            row[idx] = (row[idx] as f64 * inv) as f32;
        } else {
            row[idx] = 0.0;
        }
    }
}

/// Draw a categorical sample from `probs` given uniform u ∈ [0, 1).
#[inline]
pub(crate) fn sample_categorical(probs: &[f32], u: f32) -> u32 {
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Draw from (p_target − p_draft)₊, renormalized. Used on rejection to
/// sample the "corrective" bonus token in speculative rejection sampling
/// (Chen & Leviathan 2023, algorithm 1).
#[inline]
pub(crate) fn sample_residual(p_target: &[f32], p_draft: &[f32], u: f32) -> u32 {
    let mut sum = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 0.0 {
        // Degenerate case (p_draft >= p_target everywhere). Should not
        // happen in practice if a rejection was just drawn. Fall back to
        // argmax of p_target.
        return argmax_u32(p_target);
    }
    let u_scaled = u * sum;
    let mut acc = 0.0f32;
    for i in 0..p_target.len() {
        let d = p_target[i] - p_draft[i];
        if d > 0.0 {
            acc += d;
            if u_scaled < acc {
                return i as u32;
            }
        }
    }
    (p_target.len() - 1) as u32
}

// `NgramCache` / `PldMatcher` / `PldMatch` moved to `hipfire_runtime::spec` so
// the arch-generic `NgramSpeculator` can use them without an arch-crate
// dependency. Re-exported here so `spec_step_dflash` (and the dflash_spec_demo)
// keep referring to them unqualified.
pub use hipfire_runtime::spec::{NgramCache, PldMatch, PldMatcher};

/// Small, fast RNG for per-cycle sampling u ∈ [0, 1). Xorshift64*; deterministic
/// given the seed, cheap enough to inline into the B-rejection loop.
#[inline]
pub(crate) fn xorshift_next_unit(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Top 24 bits for a reasonable float mantissa; divide by 2^24.
    ((s >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Aggregated metrics for a sequence of speculative decode steps.
#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Total number of speculative cycles run.
    pub cycles: usize,
    /// Total number of tokens committed (sum of committed.len() across cycles).
    pub committed_tokens: usize,
    /// Total number of draft tokens accepted (sum of `accepted`).
    pub accepted_tokens: usize,
    /// Per-cycle acceptance count histogram, indexed by accepted count
    /// (0..=k). `acceptance_hist[i]` = number of cycles where exactly `i`
    /// draft tokens were accepted.
    pub acceptance_hist: Vec<usize>,
}

impl SpecStats {
    pub fn new(k: usize) -> Self {
        Self {
            cycles: 0,
            committed_tokens: 0,
            accepted_tokens: 0,
            acceptance_hist: vec![0; k + 1],
        }
    }

    pub fn record(&mut self, step: &SpecStepResult) {
        self.cycles += 1;
        self.committed_tokens += step.committed.len();
        self.accepted_tokens += step.accepted;
        if step.accepted < self.acceptance_hist.len() {
            self.acceptance_hist[step.accepted] += 1;
        }
    }

    /// Mean accepted draft tokens per cycle. This is τ from the Leviathan paper.
    pub fn tau(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.cycles as f32
        }
    }

    /// Mean committed tokens per cycle (tau + 1 on average, since each
    /// cycle always commits one bonus token).
    pub fn mean_committed(&self) -> f32 {
        if self.cycles == 0 {
            0.0
        } else {
            self.committed_tokens as f32 / self.cycles as f32
        }
    }
}

/// One speculative decode step (greedy, Leviathan verify-and-accept).
/// Operates on separate `target` and `draft` `ModelSlot` handles so the
/// caller can keep them owned in top-level variables.
///
/// Preconditions:
/// - Both `target.scratch.logits` and `draft.scratch.logits` contain the
///   logits for position `pos` (from the previous commit or prompt prefill).
/// - `target_snap` / `draft_snap` are preallocated via `DeltaNetSnapshot::new_for`.
/// - `k >= 1` is the speculation count.
///
/// Postconditions:
/// - Both slots' state advances to `pos + committed.len()`, and their
///   `scratch.logits` contain logits at the new position.
/// - Returns a `SpecStepResult` describing how many draft tokens were
///   accepted, the bonus token, and the full committed sequence.
///
/// Naive sequential verification: runs the target on each drafted token one
/// at a time. Phase 5 replaces the inner loop with a single batched prefill.
pub fn spec_step_greedy(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft: &mut ModelSlot,
    pos: usize,
    k: usize,
    target_snap: &mut DeltaNetSnapshot,
    draft_snap: &mut DeltaNetSnapshot,
) -> HipResult<SpecStepResult> {
    assert!(k >= 1, "speculation count k must be ≥ 1");

    // Snapshot both models' recurrent state at position `pos` so we can
    // rewind after verification and commit the final accepted prefix.
    target_snap.save_from(&target.dn_state, gpu)?;
    draft_snap.save_from(&draft.dn_state, gpu)?;

    // Target's current logits (at position `pos`) are used to verify
    // drafted[0]. Capture before anything trashes them.
    let target_logits_at_pos: Vec<f32> = gpu.download_f32(&target.scratch.logits)?;

    // Draft k tokens. drafted[0] samples from draft's current logits (which
    // are also for position `pos`). drafted[i] samples from the logits
    // produced by draft.forward(drafted[i-1], pos+i-1).
    let mut drafted: Vec<u32> = Vec::with_capacity(k);
    {
        let first_logits = gpu.download_f32(&draft.scratch.logits)?;
        drafted.push(argmax_u32(&first_logits));
    }
    for i in 0..k {
        draft.forward(gpu, drafted[i], pos + i)?;
        if i + 1 < k {
            let logits = gpu.download_f32(&draft.scratch.logits)?;
            drafted.push(argmax_u32(&logits));
        }
    }

    // Verification: run the target on each drafted token, collect logits.
    // target_mid_logits[i] = target's prediction at position pos+i+1.
    let mut target_mid_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
    for i in 0..k {
        target.forward(gpu, drafted[i], pos + i)?;
        target_mid_logits.push(gpu.download_f32(&target.scratch.logits)?);
    }
    // Acceptance:
    //   drafted[0] verified by target_logits_at_pos  (logits at pos)
    //   drafted[i] (i >= 1) verified by target_mid_logits[i-1] (logits at pos+i)
    let mut accepted: usize = 0;
    if !target_logits_at_pos.is_empty() && argmax_u32(&target_logits_at_pos) == drafted[0] {
        accepted = 1;
        for i in 1..k {
            if argmax_u32(&target_mid_logits[i - 1]) == drafted[i] {
                accepted += 1;
            } else {
                break;
            }
        }
    }

    // Bonus token = target's prediction at position pos+accepted.
    let bonus_logits: &[f32] = if accepted == 0 {
        &target_logits_at_pos
    } else {
        &target_mid_logits[accepted - 1]
    };
    let bonus_token = argmax_u32(bonus_logits);

    // Commit = accepted draft prefix + bonus.
    let mut committed: Vec<u32> = Vec::with_capacity(accepted + 1);
    committed.extend_from_slice(&drafted[..accepted]);
    committed.push(bonus_token);

    // Restore both models' state and replay the committed sequence so both
    // slots end at `pos + committed.len()` with correct logits.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    draft_snap.restore_to(&mut draft.dn_state, gpu)?;
    for (i, &tok) in committed.iter().enumerate() {
        target.forward(gpu, tok, pos + i)?;
        draft.forward(gpu, tok, pos + i)?;
    }

    Ok(SpecStepResult {
        accepted,
        bonus_token,
        drafted,
        committed,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash-specific target-side verify
// ═══════════════════════════════════════════════════════════════════════════

/// Output of a DFlash target verify step.
pub struct DflashVerifyOutput {
    /// Target argmax token at each of the B positions. argmax_per_pos[i]
    /// is what the target would greedy-decode at absolute position
    /// `start_pos + i` given the preceding context plus `draft_tokens[0..i]`.
    pub argmax_per_pos: Vec<u32>,
    /// Full logits downloaded for every position, concatenated row-major
    /// as `[B * vocab_size]`. Only populated when `want_full_logits=true`
    /// (i.e. temperature sampling). Empty otherwise — greedy decode
    /// uses GPU argmax and ships just B × 4 bytes to the host.
    pub logits_per_pos: Vec<f32>,
}

fn dflash_use_gdn_tape_replay(caller_supplied_tape: bool, verify_populates_tape: bool) -> bool {
    caller_supplied_tape && verify_populates_tape
}

/// Run the target on `draft_tokens` (length B) positions starting at
/// `start_pos`. Advances `target.kv_cache` and `target.dn_state` by B
/// positions. Writes B hidden-state rows into `hidden_rb` (ring head
/// advances B times). Returns downloaded logits + argmax per position.
///
/// Fast path (0.1.7 batched verify): one `forward_prefill_batch` call
/// over all B tokens with hidden extraction + per-token post-output-norm
/// hidden capture. Then B sequential `weight_gemv`s against the target's
/// lm_head to get per-position logits. The batched layer-level kernels
/// amortize launch overhead across all B tokens; the lm_head still loops
/// because a batched Q8/MQ4 lm_head GEMM isn't wired yet (task #13).
///
/// Fallback: when the batched path is ineligible (non-MQ weights,
/// non-Q8/asym KV cache, N < MIN_BATCH), `forward_prefill_batch` routes
/// to the per-token loop using `forward_scratch_with_hidden`, so hidden
/// extraction still works.
pub fn verify_dflash_block(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    verify_scratch: &VerifyScratch,
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        want_full_logits,
        None,
        verify_scratch,
        false, // chain path always needs argmax
        None,  // no retained-PM4 route on the generic wrapper
    )
}

/// Tree-verify variant of `verify_dflash_block`. Pass the linearized
/// `(positions, attn_bias)` built from a `DdTree` and this runs the whole
/// tree through a single batched forward — per-position argmax at slot i
/// corresponds to target's prediction after tree node i (slot 0 = after
/// seed / tree root).
///
/// Note: `gdn_tape` captured from a tree verify records innovations in
/// linearization order, NOT commit order. Callers that need to advance
/// GDN state to a specific committed path should either (a) re-verify
/// the committed linear prefix with `verify_dflash_block` (no tree) and
/// capture tape on that, matching `spec_step_ddtree`'s pattern, or (b)
/// implement a slot-reordering replay (not currently available).
pub fn verify_dflash_block_tree(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    tree_verify: qwen35::TreeVerifyCtx<'_>,
    verify_scratch: &VerifyScratch,
    // D9: when the caller will use the SWOR walk (which derives accepted
    // indices from the 68-byte walk-result D2H, not from argmax_per_pos),
    // skip the big_n × 4 byte argmax download entirely — the returned
    // argmax_per_pos will be empty. Pass `false` for the greedy path.
    skip_argmax_d2h: bool,
) -> HipResult<DflashVerifyOutput> {
    verify_dflash_block_inner(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        want_full_logits,
        Some(tree_verify),
        verify_scratch,
        skip_argmax_d2h,
        None, // tree verify never takes the retained-PM4 route
    )
}

fn verify_dflash_block_inner(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    tree_verify: Option<qwen35::TreeVerifyCtx<'_>>,
    verify_scratch: &VerifyScratch,
    // D9: skip the big_n × 4 argmax D2H when the caller will use SWOR walk
    // (which gets accepted indices from the 68-byte walk-result D2H instead).
    skip_argmax_d2h: bool,
    // Retained-PM4 route for this window, or `None` for the shipping path.
    // Only `verify_dflash_block_retained` ever passes `Some`, and only for the
    // fixed B=16 chain verify.
    mut retained: Option<&mut RetainedCtx<'_>>,
) -> HipResult<DflashVerifyOutput> {
    let b = draft_tokens.len();
    let vocab = target.config.vocab_size;
    let dim = target.config.dim;
    let required_tokens = start_pos.checked_add(b).ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!("DFlash verify KV token range overflow ({start_pos} + {b})"),
        )
    })?;
    // Verify replay bypasses the regular qwen35 forward wrappers.
    target
        .kv_cache
        .ensure_mapped_capacity(gpu, required_tokens)?;

    assert!(
        b <= verify_scratch.max_n,
        "verify_scratch max_n {} < b {}",
        verify_scratch.max_n,
        b
    );
    assert_eq!(verify_scratch.dim, dim, "verify_scratch dim mismatch");
    assert_eq!(verify_scratch.vocab, vocab, "verify_scratch vocab mismatch");

    // Views into the persistent scratch — no per-cycle allocation. Sized to
    // the actual current `b` (≤ max_n) so downstream kernels see the right
    // shapes. sub_offset returns a non-owning view; do NOT free these.
    let final_hidden = verify_scratch.final_hidden.sub_offset(0, b * dim);
    let tree_verify_present = tree_verify.is_some();
    let moe_lmhead_graph_env =
        hipfire_config::developer_var("HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD").ok();
    let moe_lmhead_graph_ok =
        dflash_moe_verify_graph_lmhead_eligible(
            target.config.num_experts,
            want_full_logits,
            tree_verify_present,
            moe_lmhead_graph_env.as_deref(),
        ) && dflash_batched_lm_head_supported(target.weights.output.gpu_dtype);

    // Graph-capture path eligibility. The captured forward bakes in:
    //   - N (the batch size) — via kernel grid dims
    //   - kernel selection + layer-type branches (dispatched once at capture)
    //   - weight/bias/buffer pointers (stable across cycles)
    // Per-cycle inputs (tokens, positions, kv_cache contents, dn_state contents,
    // hidden_rb staging dest) are read from device buffers whose *contents*
    // change between replays — the captured graph reads the current bytes.
    //
    // Eligibility is narrow: HFQ4G256/Q8_0 embedding (uploads via pbs.tokens),
    // tree_ok_for_graph, pbs is Some, and `qwen35::prefill_batch_pbs_eligible`
    // (shared with the forward / spec_step tape gate). When that shared
    // predicate is false — e.g. qt44/qt45 on non-gfx12 — skip capture entirely
    // and take the direct `forward_prefill_batch_with_pbs_opts` route, which
    // performs the correct per-token fallback instead of erroring inside
    // `forward_prefill_batch_single_chunk_captured_opts`.
    // `gdn_tape` is safe because verify is single-chunk → tape_offset=0 always
    // → captured node's dst offset is correct across cycles.
    //
    // Default-on for eligible models (2026-04-21 smoke on 27B MQ4 Qwen3.5
    // showed +14 % tok/s 25.6→29.2, wall-per-cycle 89→80 ms via coalescing
    // verify kernels into one graph replay and saving ~1.3 ms of per-cycle
    // launch overhead). Opt out with HIPFIRE_VERIFY_GRAPH=0.
    // Tree-verify was historically excluded (tree_verify.is_none()) because
    // the tree-attention mask varies per cycle. In theory mask +
    // parent_indices live in fixed `ddtree_scratch` buffers that the caller
    // repopulates via uncaptured memcpy_htod before each graph replay, so
    // the graph's kernels would read fresh data every cycle.
    //
    // DIAGNOSTIC ONLY — known broken 2026-04-24 (commit 480e51e +
    // A/B bench ee0bedf-followup). 3-run median on 27B MQ4 asym3 b12-k2:
    //   code     τ 7.08 → 4.51 (-36 %)   tok/s 110 → 80.1 (-27 %)
    //   prose    τ 2.50 → 3.58 (+43 %)   tok/s 45.8 → 60.2 (+31 %, noisy)
    //   instr    τ 2.19 → 1.77 (-19 %)   tok/s 47.6 → 35.6 (-25 %)
    // Coherence-gate-dflash passes (no attractors), so it's a τ bug, not a
    // correctness bug. Suspect: a scalar kernarg or intra-forward memcpy
    // inside captured region bakes in first-cycle state; when tree shape
    // varies, acceptance collapses on code (high-variance trees) but
    // coincidentally holds on prose (more uniform trees). Needs root-cause
    // dive: most likely candidates are GDN tape-offset scalar kernargs or
    // the parent_indices-driven conv1d path. DO NOT ENABLE in production.
    //
    // Gate kept live so the next session can bisect without re-plumbing.
    let tree_graph_enabled = hipfire_config::developer_var("HIPFIRE_VERIFY_GRAPH_TREE")
        .ok()
        .as_deref()
        == Some("1");
    if tree_graph_enabled && tree_verify_present {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[verify-graph-tree] WARN: HIPFIRE_VERIFY_GRAPH_TREE=1 is DIAGNOSTIC ONLY — known τ regression on code/instruct. Do not use for production benchmarks."
            );
        }
    }
    let tree_ok_for_graph = !tree_verify_present || tree_graph_enabled;
    // NB: NO Q8-long-context skip-gate here. The captured single-chunk verify
    // forward is capture-safe at any physical_cap for Q8 (non-asym) KV:
    // forward_prefill_chunk routes max_ctx_len > 8192 to the tiled
    // attention_flash_q8_0_tile_batched (O(1) LDS, no per-position malloc), so there
    // is no malloc-in-capture and no per-ctx LDS scaling — see the comment in
    // forward_prefill_batch_single_chunk_captured in qwen35.rs. The former
    // `VERIFY_GRAPH_Q8_CTX_LIMIT = 15000` skip (re-introduced by #481, obsolete once
    // the tiled crossover landed 2026-06-09) is intentionally NOT reinstated: it
    // would skip a verify-graph that now replays fine and forfeit the long-context
    // spec speedup.
    let moe_router_logits_present = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|pbs| pbs.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let pbs_eligible = qwen35::prefill_batch_pbs_eligible(
        &target.weights,
        &target.config,
        &target.dn_state,
        b,
        gpu.arch.as_str(),
        moe_router_logits_present,
    );
    // See `dflash_verify_graph_env_eligible`: gfx1100 + MQ*V2 lm_head is
    // default-off for HipGraph only (dc105ea64 newly made V2 graph-eligible).
    let verify_graph_env = hipfire_config::developer_var("HIPFIRE_VERIFY_GRAPH").ok();
    let verify_graph_ok = dflash_verify_graph_env_eligible(
        gpu.arch.as_str(),
        target.weights.output.gpu_dtype,
        verify_graph_env.as_deref(),
    ) && tree_ok_for_graph
        && matches!(
            target.weights.embd_format,
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G256
                | hipfire_runtime::llama::EmbeddingFormat::Q8_0,
        )
        && verify_scratch.prefill_batch.is_some()
        && pbs_eligible;

    // Per-cycle timing for verify-graph A/B diagnostic
    // (HIPFIRE_VERIFY_GRAPH_TIMING=1). Two device-sync points bracket the
    // forward + lm_head; the recorded mode tag distinguishes replay vs
    // warmup-direct vs first-capture vs no-graph-eligible.
    let vg_timing = hipfire_config::developer_var("HIPFIRE_VERIFY_GRAPH_TIMING")
        .ok()
        .as_deref()
        == Some("1");
    let mut vg_mode = "direct";
    let vg_t0 = if vg_timing {
        gpu.hip.device_synchronize()?;
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut graph_includes_lmhead_argmax = false;

    // Retained-PM4 route. When the caller handed us a live route for this
    // window it owns the forward outright, and the HipGraph selection below is
    // skipped so the tape never records a graph launch.
    let retained_selected = retained
        .as_ref()
        .map(|ctx| ctx.selected)
        .unwrap_or(DflashVerifyRoute::HipAuto);
    let retained_active = !matches!(retained_selected, DflashVerifyRoute::HipAuto);
    let batch_result = if retained_active {
        vg_mode = "retained";
        let ctx = retained
            .as_mut()
            .expect("retained_active implies a retained context");
        run_retained_verify_forward(
            gpu,
            target,
            draft_tokens,
            start_pos,
            hidden_rb,
            &final_hidden,
            gdn_tape,
            verify_scratch,
            ctx,
        )
    } else if verify_graph_ok {
        let pbs = verify_scratch.prefill_batch.as_ref().unwrap();
        debug_assert!(b <= pbs.max_batch);
        // Pre-capture: pre-upload inputs and ensure a stream exists. memcpy_htod
        // runs on the host/null-stream side and is NOT captured.
        qwen35::upload_prefill_batch_inputs(gpu, pbs, draft_tokens, start_pos)?;
        // 39aa358: tree verify also needs the depth-based RoPE angles on
        // device before replay (the captured RoPE kernel reads
        // pbs.rope_positions; pbs.positions stays the flat KV slot array).
        if let Some(tv) = tree_verify.as_ref() {
            debug_assert_eq!(tv.positions.len(), b, "tree RoPE positions length");
            let rope_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tv.positions.as_ptr() as *const u8, b * 4) };
            gpu.hip.memcpy_htod(&pbs.rope_positions.buf, rope_bytes)?;
        }
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create()?);
        }
        if gpu.graphs.verify_has_graph(b) {
            vg_mode = "replay";
            graph_includes_lmhead_argmax =
                moe_lmhead_graph_ok && gpu.graphs.verify_graph_has_lmhead_argmax(b);
            // Replay path: kernels read pbs.tokens/pbs.positions/dn_state/
            // kv_cache contents that were freshly updated above + upstream.
            gpu.graphs.verify_graph_launch(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                b,
            )?;
            Ok(())
        } else if gpu.graphs.verify_needs_warmup(b) {
            vg_mode = "warmup";
            // Warmup for this b: run direct so kernel JIT and any lazy scratch
            // allocations (e.g., MQ signs/x_rot/x_q8, FP16 shadow) happen
            // outside any captured region. Capturing a JIT + scratch-malloc
            // hits "hipMalloc not permitted under stream capture" the first
            // time any kernel is compiled inline. One warmup per distinct b.
            //
            // A successful launch sequence is not yet a successful warmup:
            // launches are asynchronous. Synchronize the active stream before
            // admitting this B into the capture branch. Otherwise an async
            // kernel fault can leave `warmed_up` set and make the next call
            // capture a path whose warmup never completed.
            let r = qwen35::forward_prefill_batch_single_chunk_captured_opts(
                gpu,
                &target.weights,
                &target.config,
                draft_tokens,
                start_pos,
                &mut target.kv_cache,
                &mut target.dn_state,
                &target.scratch,
                pbs,
                Some(hidden_rb),
                Some(&final_hidden),
                gdn_tape,
                tree_verify,
                false, // DFlash computes all verify logits from final_hidden below
            );
            r.and_then(|_| {
                gpu.hip.stream_synchronize(
                    gpu.active_stream
                        .as_ref()
                        .expect("verify warmup requires an active stream"),
                )?;
                gpu.graphs.verify_mark_warmup_done(b);
                eprintln!(
                    "[verify-graph] warmup for B={} complete — capture next cycle at this B",
                    b
                );
                Ok(())
            })
        } else {
            vg_mode = "capture";
            // Capture path: first call at this B after warmup.
            let capture_lmhead_argmax = moe_lmhead_graph_ok;
            gpu.graphs.begin_verify_graph_capture(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
                b,
            )?;
            let r = qwen35::forward_prefill_batch_single_chunk_captured_opts(
                gpu,
                &target.weights,
                &target.config,
                draft_tokens,
                start_pos,
                &mut target.kv_cache,
                &mut target.dn_state,
                &target.scratch,
                pbs,
                Some(hidden_rb),
                Some(&final_hidden),
                gdn_tape,
                tree_verify,
                false, // DFlash computes all verify logits from final_hidden below
            );
            let r = if r.is_ok() && capture_lmhead_argmax {
                r.and_then(|_| {
                    dflash_enqueue_verify_lm_head_argmax(
                        gpu,
                        &target.weights.output,
                        &final_hidden,
                        verify_scratch,
                        b,
                        vocab,
                    )
                })
            } else {
                r
            };
            if r.is_ok() {
                let blob_count = gpu.graphs.capture_blobs.len();
                gpu.graphs.end_verify_graph_capture(
                    &gpu.hip,
                    gpu.device_id,
                    gpu.active_stream.as_ref().unwrap(),
                )?;
                // Under `hipStreamBeginCapture`, kernels + memcpys on the
                // captured stream are RECORDED, not executed. final_hidden
                // and hidden_rb staging are left stale. Launching the graph
                // once here makes this cycle's forward actually run so lm_head
                // reads fresh data. DN state double-advance (if any future
                // HIP version does execute during capture) is washed out by
                // target_snap.restore_to after verify returns. KV cache
                // double-write writes the same data to the same positions.
                //
                // The launch is asynchronous. Complete this one-time first
                // execution before admitting the graph into the replay cache:
                // if execution fails, destroy every per-B entry so a retry
                // cannot enqueue the failed graph (or another graph recorded
                // against the now-faulted stream state).
                let first_launch = gpu
                    .graphs
                    .verify_graph_launch(
                        &gpu.hip,
                        gpu.device_id,
                        gpu.active_stream.as_ref().unwrap(),
                        b,
                    )
                    .and_then(|_| {
                        gpu.hip
                            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
                    });
                if let Err(err) = first_launch {
                    gpu.graphs.verify_graph_destroy_all(&gpu.hip, gpu.device_id);
                    return Err(err);
                }
                if capture_lmhead_argmax {
                    gpu.graphs.verify_mark_graph_lmhead_argmax(b);
                    graph_includes_lmhead_argmax = true;
                }
                eprintln!(
                    "[verify-graph] captured for B={} with {} blobs (cache size: {})",
                    b,
                    blob_count,
                    gpu.graphs.verify_graph_count(),
                );
            } else {
                // If capture failed, tear down the partial capture so we fall
                // back to the direct path next cycle cleanly.
                let _ = gpu
                    .hip
                    .stream_end_capture(gpu.active_stream.as_ref().unwrap());
                gpu.graphs.capture_mode = false;
                gpu.graphs.capture_blobs.clear();
            }
            r
        }
    } else {
        qwen35::forward_prefill_batch_with_pbs_opts(
            gpu,
            &target.weights,
            &target.config,
            draft_tokens,
            start_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            Some(hidden_rb),
            Some(&final_hidden),
            gdn_tape,
            tree_verify,
            verify_scratch.prefill_batch.as_ref(),
            None,  // mask_override: speculative verify path doesn't use the MTP probe hook
            None,  // max_layer: DFlash verify always runs the full stack
            false, // DFlash computes all verify logits from final_hidden below
        )
    };

    // Commit hidden_rb staging to the ring (outside any captured region).
    // The captured forward wrote to staging[0..b*h]; this scatter places
    // those rows at the current head and advances head by b. Under the
    // graph path we manually drive this because the non-graph chunk loop
    // (forward_prefill_batch_with_pbs) that usually calls it was bypassed.
    // The retained route runs the same capture-safe single-chunk body as the
    // graph path, so it owes the same one external staging commit.
    if (retained_active || verify_graph_ok) && batch_result.is_ok() {
        hidden_rb.commit_staging_to_ring(gpu, b)?;
    }
    // Tree mode at topk>1 REQUIRES this sync. Without it τ degrades badly
    // (e.g. budget=60 topk=8 drops 7.0 → 3.3; 9B asym3 2026-04-14). topk=1
    // is fine without the sync (byte-exact with baseline DFlash either way).
    // Root cause suspected: siblings at the same tree depth produce
    // duplicate entries in `positions[]`, so `kv_cache_write` dispatches
    // multiple batch rows targeting the same cache slot — the async write
    // order lets a subsequent attention kernel read a partially-committed
    // slot. Fix TODO: either serialize within-kernel per-slot, or ensure
    // the "winning" sibling's write happens last.
    //
    // D16: narrowed from device_synchronize() (full device drain, ~3–5 ms
    // on PCIe, ~50–200 µs on UMA) to stream_synchronize (scoped to the
    // active stream only). Semantically identical on hipfire's single-stream
    // setup, but does not stall work enqueued on other streams. The ordering
    // guarantee — all prior KV writes complete before the next attention
    // kernel reads the slot — is preserved because all ops ride the same
    // stream. Caller (spec_step_ddtree_batched) ensures active_stream is
    // Some before reaching here; the fallback keeps the old full drain for
    // the rare path where it isn't set.
    if batch_result.is_ok() && tree_verify.is_some() {
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        } else {
            gpu.hip.device_synchronize()?;
        }
    }
    batch_result?;

    // Per-position lm_head. Fast paths in priority order:
    //   Q8_0      → DFlash-scoped Q8 lm_head dispatcher (gfx12 WMMA by default).
    //   MQ4G256   → batched rotate + gemm_hfq4g256 (one launch + one D2H).
    //   HFQ4G256  → batched gemm_hfq4g256 directly.
    //   else      → B sequential weight_gemv calls + B downloads (legacy).
    let w_out = &target.weights.output;
    let mut logits_per_pos: Vec<f32> = Vec::with_capacity(b * vocab);
    let mut argmax_per_pos: Vec<u32> = Vec::with_capacity(b);

    let try_batched = dflash_batched_lm_head_supported(w_out.gpu_dtype);

    if graph_includes_lmhead_argmax {
        debug_assert!(!want_full_logits);
        // The MoE-only extended verify graph already enqueued lm_head+argmax.
        // Mirror MTP's graph shape: keep device work captured, then perform
        // the single small D2H read after graph launch.
        argmax_per_pos = dflash_download_verify_argmax(gpu, verify_scratch, b)?;
    } else if try_batched {
        let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
        // Q8_0 routes through the DFlash lm_head helper so the gfx12 WMMA
        // arm can be coherence-gated independently of MTP. Set
        // HIPFIRE_DFLASH_Q8_LMHEAD_WMMA=0 to force the legacy scalar chunks.
        // MQ4/HFQ4/HFQ6/MQ6 kernels have no 64-row cap and take the
        // single-shot path.
        dflash_enqueue_verify_lm_head(gpu, w_out, &final_hidden, verify_scratch, b, vocab)?;
        if want_full_logits {
            // Rejection-sampling path needs full target distribution.
            // Cost: B × vocab × 4 bytes D2H per verify (~15 MB at B=16 × 248K).
            let host_logits = gpu.download_f32(&logits_batch)?;
            for i in 0..b {
                let row = &host_logits[i * vocab..(i + 1) * vocab];
                argmax_per_pos.push(argmax_u32(row));
            }
            logits_per_pos = host_logits;
        } else if skip_argmax_d2h {
            // D9: SWOR verify path — the accept walk uses the 68-byte walk-result
            // D2H (D13), NOT argmax_per_pos. Skip this D2H entirely; argmax_per_pos
            // stays empty. Still enqueue the GPU argmax kernel so verify_scratch.argmax
            // is populated if some other consumer ever needs it (none currently).
            let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)?;
            // argmax_per_pos intentionally left empty — caller must not read it.
        } else {
            // GPU-side batched argmax. Writes B i32 indices; we download just
            // 4*B bytes instead of the full B×vocab logits. Saves ~15 MB of
            // PCIe D2H per verify on the 4B Q8 lm_head (~3-5 ms/iter).
            let argmax_buf = verify_scratch.argmax.sub_offset(0, b);
            gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, b)?;
            argmax_per_pos = dflash_download_verify_argmax(gpu, verify_scratch, b)?;
        }
        // Greedy path doesn't need `logits_per_pos`; leave empty to avoid
        // the 15 MB D2H. If temp>0 sampling is added later, reinstate the
        // download or sample on-GPU.
    } else {
        // Fallback: B sequential GEMVs.
        for i in 0..b {
            let hidden_row = final_hidden.sub_offset(i * dim, dim);
            llama::weight_gemv(
                gpu,
                &target.weights.output,
                &hidden_row,
                &target.scratch.logits,
            )?;
            let row = gpu.download_f32(&target.scratch.logits)?;
            debug_assert_eq!(row.len(), vocab);
            argmax_per_pos.push(argmax_u32(&row));
            logits_per_pos.extend_from_slice(&row);
        }
    }

    if let Some(t0) = vg_t0 {
        gpu.hip.device_synchronize()?;
        eprintln!(
            "[vg-time] B={} mode={} elapsed_us={}",
            b,
            vg_mode,
            t0.elapsed().as_micros()
        );
    }

    Ok(DflashVerifyOutput {
        argmax_per_pos,
        logits_per_pos,
    })
}

/// Retained-PM4 execution context for one DFlash verify window.
///
/// Built by [`verify_dflash_block_retained`] and threaded into
/// [`verify_dflash_block_inner`], which owns the forward. `selected` is the
/// route [`DflashVerifyPm4::plan_route`] chose for this window;
/// [`DflashVerifyRoute::HipAuto`] means the retained machinery stands down and
/// the shipping path runs unchanged.
struct RetainedCtx<'a> {
    state: &'a mut DflashVerifyPm4,
    binding: DflashVerifyBinding,
    selected: DflashVerifyRoute,
}

/// Why a recording capture stopped short.
///
/// The distinction is the whole safety argument: `Forward` means the model body
/// itself failed and there is no valid result for this cycle, while the other
/// three happen after a synchronized successful body and therefore only cost us
/// the route.
enum CaptureFailure {
    Forward(hip_bridge::HipError),
    Record(String),
    Contract(String),
    Prepare(String),
}

/// How far a successful recording window got.
///
/// The first recording only calibrates: a prepared route may retain a scalar
/// kernarg solely when a second recording at a different position proves the
/// scalar tracks the position.
enum CaptureStep {
    Calibrated(rdna_compute::replay::RecordedKernargSnapshot),
    Ready {
        identity: rdna_compute::replay::PreparedReplayIdentity,
        position_bindings: usize,
    },
}

fn retained_hip_error(reason: &str) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, reason)
}

/// Fingerprint every shape, dtype, extraction id, and captured allocation base
/// the retained tape depends on. Bases and shapes only — never buffer contents,
/// which change every cycle by design.
fn dflash_verify_fingerprint(
    target: &ModelSlot,
    hidden_rb: &HiddenStateRingBuffer,
    verify_scratch: &VerifyScratch,
    gdn_tape: Option<&GdnTape>,
) -> u64 {
    let mut values: Vec<u64> = Vec::new();
    values.push(target.config.dim as u64);
    values.push(target.config.n_layers as u64);
    values.push(target.config.vocab_size as u64);
    values.push(target.weights.output.gpu_dtype as u64);
    values.push(verify_scratch.max_n as u64);
    values.push(verify_scratch.dim as u64);
    values.push(verify_scratch.vocab as u64);
    values.push(verify_scratch.hidden_k as u64);
    values.push(verify_scratch.final_hidden.buf.as_ptr() as u64);
    values.push(verify_scratch.logits.buf.as_ptr() as u64);
    values.push(verify_scratch.rot.buf.as_ptr() as u64);
    values.push(verify_scratch.argmax.buf.as_ptr() as u64);
    if let Some(pbs) = verify_scratch.prefill_batch.as_ref() {
        values.push(pbs.max_batch as u64);
        values.push(pbs.x_batch.buf.as_ptr() as u64);
        values.push(pbs.rope_positions.buf.as_ptr() as u64);
    }
    values.push(hidden_rb.max_positions as u64);
    values.push(hidden_rb.hidden_dim as u64);
    values.push(hidden_rb.max_batch as u64);
    for &layer_id in &hidden_rb.extract_layers {
        values.push(layer_id as u64);
    }
    // Only the staging buffers are captured. `layer_bufs` are written by the
    // head-dependent commit, which stays outside the tape.
    for staging in &hidden_rb.staging_bufs {
        values.push(staging.buf.as_ptr() as u64);
    }
    for s_matrix in &target.dn_state.s_matrices {
        values.push(s_matrix.buf.as_ptr() as u64);
    }
    if let Some(tape) = gdn_tape {
        values.push(tape.max_n as u64);
        values.push(tape.qkv_dim as u64);
        values.push(tape.v_dim as u64);
        for buf in tape
            .qkv_bufs
            .iter()
            .chain(tape.alpha_bufs.iter())
            .chain(tape.beta_bufs.iter())
        {
            values.push(buf.buf.as_ptr() as u64);
        }
    }
    fingerprint_u64(&values)
}

/// Build the admission binding for the current window.
///
/// `max_position` is the smaller of the hidden ring capacity and the KV
/// physical capacity: a window past it re-primes rather than replaying dynamic
/// geometry that was prepared too small.
fn build_dflash_verify_binding(
    b: usize,
    arch: &str,
    target: &ModelSlot,
    hidden_rb: &HiddenStateRingBuffer,
    verify_scratch: &VerifyScratch,
    gdn_tape: Option<&GdnTape>,
) -> DflashVerifyBinding {
    let fingerprint = dflash_verify_fingerprint(target, hidden_rb, verify_scratch, gdn_tape);
    // The layout generation moves whenever the KV slab is remapped or resized.
    // Expected VMM growth may keep the same reserved base, so capacity is
    // folded in alongside every per-layer allocation base.
    let kv = &target.kv_cache;
    let mut layout: Vec<u64> = Vec::with_capacity(2 + kv.k_gpu.len() * 2);
    layout.push(kv.physical_cap as u64);
    layout.push(kv.max_seq as u64);
    for (k, v) in kv.k_gpu.iter().zip(kv.v_gpu.iter()) {
        layout.push(k.buf.as_ptr() as u64);
        layout.push(v.buf.as_ptr() as u64);
    }
    let layout_generation = fingerprint_u64(&layout);
    DflashVerifyBinding::new(
        b,
        arch,
        fingerprint,
        layout_generation,
        hidden_rb.max_positions.min(kv.physical_cap),
    )
}

/// The capture-safe single-chunk body, run directly with no graph and no
/// recording. Shared by every retained route so the recorded body and the
/// fallback body are literally the same call.
#[allow(clippy::too_many_arguments)]
fn dflash_direct_verify_forward(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    final_hidden: &GpuTensor,
    gdn_tape: Option<&mut GdnTape>,
    pbs: &qwen35::PrefillBatchScratch,
) -> HipResult<()> {
    qwen35::forward_prefill_batch_single_chunk_captured_opts(
        gpu,
        &target.weights,
        &target.config,
        draft_tokens,
        start_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        pbs,
        Some(hidden_rb),
        Some(final_hidden),
        gdn_tape,
        None,
        false, // DFlash computes all verify logits from final_hidden
    )
}

/// Execute one retained-route verify window and report the outcome to the
/// route state machine.
///
/// Host token/position upload happens here, before any recording, and never
/// enters the indirect buffer. lm-head, argmax, accept, and the staging commit
/// all stay with the caller.
#[allow(clippy::too_many_arguments)]
fn run_retained_verify_forward(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    final_hidden: &GpuTensor,
    gdn_tape: Option<&mut GdnTape>,
    verify_scratch: &VerifyScratch,
    ctx: &mut RetainedCtx<'_>,
) -> HipResult<()> {
    let pbs = verify_scratch.prefill_batch.as_ref().ok_or_else(|| {
        retained_hip_error("retained DFlash verify requires a persistent PrefillBatchScratch")
    })?;
    qwen35::upload_prefill_batch_inputs(gpu, pbs, draft_tokens, start_pos)?;
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    match ctx.selected {
        DflashVerifyRoute::HipAuto => Err(retained_hip_error(
            "run_retained_verify_forward called for a HipAuto window",
        )),
        DflashVerifyRoute::PrimeDirect => {
            let result = dflash_direct_verify_forward(
                gpu,
                target,
                draft_tokens,
                start_pos,
                hidden_rb,
                final_hidden,
                gdn_tape,
                pbs,
            );
            if result.is_ok() {
                ctx.state.note_prime_success(ctx.binding.clone());
            }
            result
        }
        DflashVerifyRoute::CaptureRecord => {
            let max_position = ctx.binding.max_position;
            swap_retained_controller(gpu, ctx.state)?;
            let began = gpu.replay.begin_capture();
            if let Err(reason) = began {
                // Nothing was recorded and nothing ran: restore the ordinary
                // controller and produce this cycle's result on direct HIP.
                swap_retained_controller(gpu, ctx.state)?;
                ctx.state
                    .poison(format!("DFlash retained begin capture: {reason}"));
                return dflash_direct_verify_forward(
                    gpu,
                    target,
                    draft_tokens,
                    start_pos,
                    hidden_rb,
                    final_hidden,
                    gdn_tape,
                    pbs,
                );
            }
            // A prepared route may only retain a kernarg scalar that provably
            // tracks the decode position, and one recording cannot prove that.
            // The first eligible window records a calibration tape; the next one
            // at a different position differences against it.
            let earlier = ctx
                .state
                .calibration()
                .map(|(snapshot, at)| (snapshot.clone(), at));
            let outcome = (|| -> Result<CaptureStep, CaptureFailure> {
                dflash_direct_verify_forward(
                    gpu,
                    target,
                    draft_tokens,
                    start_pos,
                    hidden_rb,
                    final_hidden,
                    gdn_tape,
                    pbs,
                )
                .map_err(CaptureFailure::Forward)?;
                gpu.hip
                    .device_synchronize()
                    .map_err(CaptureFailure::Forward)?;
                gpu.replay.finish_capture().map_err(|reason| {
                    CaptureFailure::Record(format!("finish capture: {reason}"))
                })?;
                let Some((earlier_snapshot, earlier_position)) = earlier else {
                    return Ok(CaptureStep::Calibrated(
                        gpu.replay.snapshot_recorded_kernargs(),
                    ));
                };
                if start_pos <= earlier_position {
                    // Differencing needs a strictly later position; keep the
                    // fresher recording and wait for generation to advance.
                    return Ok(CaptureStep::Calibrated(
                        gpu.replay.snapshot_recorded_kernargs(),
                    ));
                }
                let position_bindings = gpu
                    .replay
                    .synthesize_position_bindings(&earlier_snapshot, earlier_position, start_pos)
                    .map_err(|reason| {
                        CaptureFailure::Contract(format!("position binding synthesis: {reason}"))
                    })?;
                let launches = gpu.replay.recorded_launches().len();
                gpu.replay
                    .probe_aql_contracts(gpu.device_id as usize)
                    .map_err(|reason| {
                        CaptureFailure::Contract(format!("AQL contract probe: {reason}"))
                    })?;
                gpu.replay.set_prepared_max_position(max_position);
                gpu.replay
                    .prepare_pm4_prefix_calibrated(gpu.device_id as usize, launches)
                    .map_err(|reason| CaptureFailure::Prepare(format!("PM4 prepare: {reason}")))?;
                gpu.replay
                    .prepared_route_identity()
                    .map(|identity| CaptureStep::Ready {
                        identity,
                        position_bindings,
                    })
                    .ok_or_else(|| {
                        CaptureFailure::Prepare(
                            "prepared PM4 route reported no identity".to_string(),
                        )
                    })
            })();
            swap_retained_controller(gpu, ctx.state)?;
            match outcome {
                Ok(CaptureStep::Calibrated(snapshot)) => {
                    ctx.state.note_calibration_capture(snapshot, start_pos);
                    Ok(())
                }
                Ok(CaptureStep::Ready {
                    identity,
                    position_bindings,
                }) => {
                    ctx.state.note_capture();
                    ctx.state.note_position_bindings(position_bindings);
                    ctx.state.note_ready(ctx.binding.clone(), identity);
                    Ok(())
                }
                Err(CaptureFailure::Forward(error)) => {
                    // The body failed mid-capture: KV/DeltaNet/staging may be
                    // half-written, so there is nothing to salvage here.
                    ctx.state
                        .poison(format!("DFlash retained capture body failed: {error:?}"));
                    Err(error)
                }
                Err(CaptureFailure::Record(reason)) => {
                    ctx.state.poison(reason);
                    Ok(())
                }
                Err(CaptureFailure::Contract(reason)) => {
                    ctx.state.note_capture();
                    ctx.state.note_contract_failure(reason);
                    Ok(())
                }
                Err(CaptureFailure::Prepare(reason)) => {
                    ctx.state.note_capture();
                    ctx.state.note_prepare_failure(reason);
                    Ok(())
                }
            }
        }
        DflashVerifyRoute::Pm4 => {
            swap_retained_controller(gpu, ctx.state)?;
            let replayed = unsafe { gpu.replay.replay_pm4_checked(start_pos) };
            swap_retained_controller(gpu, ctx.state)?;
            match replayed {
                Ok(_) => {
                    ctx.state.note_replay_success(start_pos);
                    Ok(())
                }
                Err(failure) => {
                    let quiescence = failure.quiescence;
                    ctx.state
                        .note_replay_failure(start_pos, quiescence, failure.error.clone());
                    Err(retained_hip_error(&format!(
                        "DFlash retained PM4 replay failed at position {start_pos} \
                         (quiescence {quiescence:?}): {}",
                        failure.error
                    )))
                }
            }
        }
    }
}

/// Swap the dedicated retained controller in or out of `gpu.replay`.
///
/// The ordinary-AR controller must never record or replay the DFlash tape, and
/// the DFlash controller must never own the AR route, so the swap is always
/// paired around exactly the recorded or replayed body.
fn swap_retained_controller(gpu: &mut Gpu, state: &mut DflashVerifyPm4) -> HipResult<()> {
    let controller = state.controller_mut().ok_or_else(|| {
        retained_hip_error("retained DFlash verify route has no dedicated controller")
    })?;
    std::mem::swap(&mut gpu.replay, controller);
    Ok(())
}

/// Retained-route entry point for the fixed B=16 DFlash2 chain verify.
///
/// Plans the route for this window, then defers to the ordinary
/// [`verify_dflash_block_inner`] so lm-head, argmax, and the staging commit run
/// exactly once and in their existing order regardless of which forward ran.
#[allow(clippy::too_many_arguments)]
pub fn verify_dflash_block_retained(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_tokens: &[u32],
    start_pos: usize,
    hidden_rb: &mut HiddenStateRingBuffer,
    gdn_tape: Option<&mut GdnTape>,
    want_full_logits: bool,
    verify_scratch: &VerifyScratch,
    route: &mut DflashVerifyPm4,
) -> HipResult<DflashVerifyOutput> {
    let b = draft_tokens.len();
    let binding = build_dflash_verify_binding(
        b,
        gpu.arch.as_str(),
        target,
        hidden_rb,
        verify_scratch,
        gdn_tape.as_deref(),
    );
    let selected = {
        let window = DflashVerifyWindow {
            batch: b,
            tree: false,
            want_full_logits,
            position: start_pos,
            binding: &binding,
        };
        route.plan_route(&window)
    };
    let mut ctx = RetainedCtx {
        state: route,
        binding,
        selected,
    };
    verify_dflash_block_inner(
        gpu,
        target,
        draft_tokens,
        start_pos,
        hidden_rb,
        gdn_tape,
        want_full_logits,
        None,
        verify_scratch,
        false,
        Some(&mut ctx),
    )
}

/// Download extracted target hidden states for the most recent B positions
/// from `hidden_rb` and concat them into a flat `[B × num_extract × hidden]`
/// host vector in the order expected by `dflash::draft_forward` (per-position,
/// then per-extract-layer).
///
/// Caller typically slices this by `[0..accept_len+1]` of the position
/// dimension when appending to the cumulative target_hidden buffer used
/// by subsequent draft forwards.
///
/// Partial-download path (2026-04-16): downloads only the B most recent
/// rows per layer via `memcpy_dtoh` of the exact slice needed, handling
/// the ring-buffer wrap as two segments when necessary. Prior version
/// downloaded the full `max_pos × hidden` per layer (~170 MB at ctx=2048
/// × hidden=4096 × 5 layers); this cuts per-cycle D2H to the useful
/// `B × hidden × 5 × 4` bytes (~1.3 MB). For a math prompt at ctx=1024
/// this saves ~7 ms/cycle of PCIe + sync overhead.
/// GPU-side scatter of the FIRST `n_rows` rows of the most recently written
/// `block_size` slots of `hidden_rb` into a flat dst tensor laid out as
/// `[max_ctx × num_extract × hidden]` (interleaved per-position).
///
/// Semantics mirror `download_hidden_block(b)` followed by a slice of the
/// first `rows_to_keep * num_extract * hidden` f32s — the spec_step caller
/// pattern. The ring has just been advanced by `block_size` (by a verify
/// forward); the slots written in THAT verify occupy
/// `[head − block_size, head)` (mod max_pos). We copy the first `n_rows`
/// of those to rows `[dst_row_offset, dst_row_offset + n_rows)` of dst.
///
/// For each row r in 0..n_rows:
/// - ring slot = (head − block_size + r) mod max_pos
/// - dst row   = (dst_row_offset + r)
/// - For each extract layer `ext`: D2D copy hidden×4 bytes from
///   `hidden_rb.layer_bufs[ext][slot × hidden ..]` to
///   `dst[(dst_row × num_extract + ext) × hidden ..]`.
///
/// When called after `seed_target_hidden_from_prompt` (no prior block), the
/// caller passes `block_size = n_rows = prompt_len` and `dst_row_offset = 0`.
///
/// Replaces the previous D2H-then-H2D roundtrip via `target_hidden_host`
/// Vec<f32> + `draft_forward`'s upload for the common ctx_slice=None path.
/// Eliminates 5 blocking D2H sync points per spec step (one per extract
/// layer) and the follow-on per-cycle H2D upload; remains about 80 small
/// async D2D enqueues per cycle (ne × n_rows ≤ 5 × 16), which the stream
/// dispatcher handles in ~200 µs of CPU time with zero cross-device waits.
pub fn scatter_hidden_block_to_interleaved(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    n_rows: usize,
    dst_modulus: usize,
) -> HipResult<()> {
    assert!(
        n_rows <= block_size,
        "scatter: n_rows {n_rows} > block_size {block_size}"
    );
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let head = hidden_rb.head;
    let written = hidden_rb.written;
    assert!(
        block_size <= written.max(max_pos),
        "scatter: block_size {block_size} > written {written}"
    );
    let row_bytes = hidden * 4;
    // The ring physically holds only the last `max_pos` rows of a logical
    // block longer than that — skip the fallen-off prefix (windowed draft
    // mode: SWA layers never read out-of-window rows; the last layer's
    // long-reach fill is the post-seed backfill's job). Legacy callers pass
    // block_size <= max_pos ⇒ r_skip = 0, identical behaviour.
    let r_skip = block_size.saturating_sub(max_pos);
    let start_slot = (head + max_pos - (block_size - r_skip)) % max_pos;

    for r in r_skip..n_rows {
        let slot = (start_slot + (r - r_skip)) % max_pos;
        // dst_row_offset is an absolute row index; windowed-mode dst rings
        // address it by slot (row % modulus). Legacy passes usize::MAX.
        let dst_row = if dst_modulus == usize::MAX {
            dst_row_offset + r
        } else {
            (dst_row_offset + r) % dst_modulus
        };
        let dst_row_base_bytes = dst_row * num_extract * row_bytes;
        for ext in 0..num_extract {
            let src_offset_bytes = slot * row_bytes;
            let dst_offset_bytes = dst_row_base_bytes + ext * row_bytes;
            gpu.hip.memcpy_dtod_at(
                &dst.buf,
                dst_offset_bytes,
                &hidden_rb.layer_bufs[ext].buf,
                src_offset_bytes,
                row_bytes,
            )?;
        }
    }
    Ok(())
}

pub fn download_hidden_block(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    b: usize,
) -> HipResult<Vec<f32>> {
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let written = hidden_rb.written;

    // Figure out which ring positions hold the most recent B writes.
    // `head` points to where the NEXT write will land. After B advances,
    // the most recent B sit at ring slots (head - B) mod max_pos ..
    // (head - 1) mod max_pos.
    assert!(
        b <= written,
        "verify must have written at least B rows to ring buffer"
    );
    let head = hidden_rb.head;
    let start_slot = (head + max_pos - b) % max_pos;
    let row_bytes = hidden * 4;

    // Per layer, download only the B needed rows (not the full ring).
    // If start_slot + b <= max_pos: one contiguous segment.
    // Otherwise: two segments (head→end + 0→tail).
    //
    // Each layer's B-row slice lands at [ext × b × hidden] in layer_data_flat.
    let mut layer_data_flat = vec![0f32; num_extract * b * hidden];
    for ext in 0..num_extract {
        let src_buf = &hidden_rb.layer_bufs[ext].buf;
        let dst_offset_floats = ext * b * hidden;
        if start_slot + b <= max_pos {
            // Single contiguous copy.
            let dst_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat.as_mut_ptr().add(dst_offset_floats) as *mut u8,
                    b * row_bytes,
                )
            };
            gpu.hip
                .memcpy_dtoh_at(dst_bytes, src_buf, start_slot * row_bytes)?;
        } else {
            // Two-segment ring wrap: tail of buffer, then head.
            let first_rows = max_pos - start_slot;
            let second_rows = b - first_rows;
            let dst_first_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat.as_mut_ptr().add(dst_offset_floats) as *mut u8,
                    first_rows * row_bytes,
                )
            };
            gpu.hip
                .memcpy_dtoh_at(dst_first_bytes, src_buf, start_slot * row_bytes)?;
            let dst_second_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    layer_data_flat
                        .as_mut_ptr()
                        .add(dst_offset_floats + first_rows * hidden)
                        as *mut u8,
                    second_rows * row_bytes,
                )
            };
            gpu.hip.memcpy_dtoh_at(dst_second_bytes, src_buf, 0)?;
        }
    }

    // Rearrange into per-position-then-per-extract-layer order.
    // layer_data_flat is [ext × b × hidden]; we want [b × ext × hidden].
    let mut out: Vec<f32> = Vec::with_capacity(b * num_extract * hidden);
    for pi in 0..b {
        for ext in 0..num_extract {
            let src_off = (ext * b + pi) * hidden;
            out.extend_from_slice(&layer_data_flat[src_off..src_off + hidden]);
        }
    }

    debug_assert_eq!(out.len(), b * num_extract * hidden);
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// DFlash spec step — one speculative decode iteration
// ═══════════════════════════════════════════════════════════════════════════

/// One DFlash speculative iteration. Given a previously-accepted token at
/// `position - 1` (the "seed" for block_output_ids[0]) and a cumulative
/// `target_hidden_host` buffer of shape `[position × num_extract × hidden]`,
/// runs the draft to fill B-1 mask slots, verifies against the target,
/// commits the accepted prefix plus a bonus target token, and rewinds the
/// target's DeltaNet state so only `accept_len + 1` forwards are reflected.
///
/// Returns `SpecStepResult` describing accepted draft count, bonus token,
/// drafted proposals, and the full committed sequence (length accept+2:
/// `[seed_token, draft[..accept_len], posterior[accept_len]]` — note the
/// seed_token is ALSO committed here because it was the bonus token from
/// the PREVIOUS iteration and still needs the target forward at its
/// position). Callers append `committed[1..]` to the output token stream
/// (the seed was already emitted).
///
/// Side effects:
/// - Appends `accept_len + 1` positions × `num_extract × hidden` floats to
///   `target_hidden_host`.
/// - Advances target's KV cache and DeltaNet state by `accept_len + 1`
///   positions. Draft has no persistent state.
///
/// Preconditions:
/// - `target_hidden_host.len() == position × num_extract × hidden` (set up
///   by `seed_target_hidden_from_prompt`).
/// - `position ≤ draft_scratch.max_ctx_len`.
/// - `draft_cfg.block_size ≤ draft_scratch.max_block_size`.
///
/// `ctx_slice`: if `Some(N)`, the draft only sees the most recent `N` rows
/// of `target_hidden_host` (with RoPE positions `[position-N..position+B)`).
/// Use this for accept-rate bisect experiments — if training-time context
/// was shorter than inference-time, truncation may help. `None` uses the
/// full cumulative context (the default, distribution-preserving path).
#[allow(clippy::too_many_arguments)]
pub fn spec_step_dflash(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    gdn_tape: Option<&mut GdnTape>,
    temp: f32,
    // Nucleus (top_p) cutoff applied IDENTICALLY to both the draft and target
    // softmax rows on the temp sampling path. 1.0 (or >= 0.999) disables it →
    // the path is byte-equivalent to plain temp-T full-vocab sampling. Lossless
    // == AR-at-this-top_p holds via the TARGET truncation (Leviathan marginal
    // preservation); the draft is truncated too for parity + acceptance
    // efficiency. See apply_topp_trunc / apply_host_nucleus.
    top_p: f32,
    // Top-k cutoff, applied identically to draft + target softmax rows on the
    // sampled path (folded into tau by the GPU kernel: tau = max(tau_p, tau_k)).
    // 0 = disabled (top_p-only). Lossless == AR-at-(top_k,top_p), same cut both
    // sides.
    top_k: usize,
    rng_state: &mut u64,
    block_size_override: Option<usize>,
    ngram_cache: Option<&NgramCache>,
    prev_committed: &[u32],
    cactus_delta: f32,
    pld_spine: Option<&[u32]>,
    repeat_penalty: f32,
    repeat_window: usize,
    // Max accepted drafts this window (`emit.len() == accepted + 1` after the
    // bonus). `None` = uncapped (bench/demo). Clamped **before** hidden/KV/
    // DeltaNet/drafter commit so post-hoc `cap_emit` is defense only.
    max_accept: Option<usize>,
    // Retained-PM4 verify route owned by the speculator. `None` keeps the
    // shipping HIP/HipGraph path byte-for-byte.
    verify_pm4: Option<&mut DflashVerifyPm4>,
) -> HipResult<SpecStepResult> {
    let selector_mode = draft_weights.has_candidate_selector();
    if selector_mode {
        if pld_spine.is_some() {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: PLD spine rewrite is not supported with DFlash2 candidate selector",
            ));
        }
        if ngram_cache.is_some() {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: ngram-cache rewrite is not supported with DFlash2 candidate selector",
            ));
        }
        if cactus_delta != 0.0 {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: CACTUS is not supported with DFlash2 candidate selector",
            ));
        }
        if repeat_penalty != 1.0 {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: repeat penalty is not supported with DFlash2 candidate selector",
            ));
        }
        if hipfire_runtime::config::get()
            .dflash_ngram_block
            .unwrap_or(false)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "selector mode: ngram-block rewriting is not supported with DFlash2 candidate selector",
            ));
        }
    }
    // Effective block size for THIS step. Usually `draft_cfg.block_size`
    // (what the draft was trained at, 16 for Qwen3.5-*-DFlash) but a caller
    // doing adaptive-B based on rolling τ can shrink to save per-iter cost.
    //
    // When `pld_spine` is Some, shrink b to 1+pld.len() (capped at requested)
    // so we don't run off the end of the PLD continuation. PLD-supplied
    // spines are often shorter than the trained B; the paper caps at 8.
    let requested_b = block_size_override.unwrap_or(draft_cfg.block_size);
    let b = match pld_spine {
        Some(pld) => (1 + pld.len()).min(requested_b).max(2),
        None => requested_b,
    };
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;

    // Ensure active_stream is set before any draft/verify work so memset_async
    // and stream-ordered launches have a non-null stream to ride on. Without
    // this, the lm_head pre-zero memsets in dispatch.rs:4475/4545 fall through
    // to the sync hipMemset path (~46 hot calls/cycle on 27B).
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    assert!(b >= 2, "dflash block size must be ≥ 2");
    // `target_hidden_host` is only authoritative on the ctx_slice=Some path,
    // where it backs the CPU slice handed to draft_forward. On the default
    // ctx_slice=None path the data lives on GPU in draft_scratch.target_hidden
    // (populated by D2D scatter, no CPU shadow). Only enforce the length
    // invariant when we actually read it.
    if ctx_slice.is_some() {
        assert_eq!(
            target_hidden_host.len(),
            position * ne * h,
            "target_hidden_host size mismatches position"
        );
    }

    // HIPFIRE_SPEC_PHASES=1: per-cycle phase breakdown. Inserts a
    // device_synchronize at each phase boundary so the wall-clock reflects
    // ACTUAL GPU completion (not CPU enqueue of async work). Perf-heavy —
    // use only for diagnostics. When disabled, zero cost beyond a handful
    // of Instant::now() calls.
    static PHASE_ON_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let phase_on = *PHASE_ON_ENV.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_SPEC_PHASES")
            .ok()
            .as_deref()
            == Some("1")
    });
    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_spec_start = std::time::Instant::now();
    let t_phase = t_spec_start;

    // ── 1. block_output_ids seeded with prev bonus at [0], masks at [1..B] ──
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // Draft state: either synthesized from a PLD spine (Goose §4.3 bypass
    // mode — deterministic, skips the DFlash forward) or produced by the
    // DFlash draft forward pass below. Declared out here so the post-draft
    // common code (ngram gating, target verify, rejection) sees the same
    // `drafted` / `draft_softmaxes` / `draft_probs_at_drafted` regardless
    // of draft source.
    let mut drafted: Vec<u32> = vec![seed_token];
    let mut draft_probs_at_drafted: Vec<f32> = Vec::new();
    let mut draft_softmaxes: Vec<Vec<f32>> = Vec::new();
    let use_temp_sampling = temp > 0.0;
    // Truncation active when either top_p non-default OR top_k enabled (vocab-gated).
    // Host applies top-k before nucleus; GPU uses combined softmax_temp_topp whenever active.
    let trunc_active = use_temp_sampling && (top_p < 0.999 || (top_k > 0 && top_k < vocab));
    let topp_active = trunc_active;
    let rp_active = repeat_penalty > 1.0 && !use_temp_sampling;
    // HIPFIRE_DFLASH_NGRAM_BLOCK=1: apply llama::apply_ngram_block to every
    // host-path row in BOTH draft and target argmax paths. Bans the next
    // token after any 3/4/5/6-gram repeat (NEG_INFINITY logit). Matches the
    // production-path defense in daemon/run/infer for the AR sampler.
    // Forces the per-row host download even when RP is off (extra D2H per
    // cycle); off-by-default for that reason.
    let ngram_block_active = !use_temp_sampling
        && hipfire_runtime::config::get()
            .dflash_ngram_block
            .unwrap_or(false);
    // HIPFIRE_DFLASH_LOGIT_DUMP=1: per-cycle diagnostic. Forces the host-logits
    // path and prints, at the acceptance boundary, the target top1/top2 margin
    // and the logit gap to the drafted (rejected) token. A tiny gap at the
    // first divergence position is the signature of a sub-ULP numerics
    // tie-break (noise), vs a large gap = a real forward divergence. Greedy
    // only; zero cost when unset.
    static LOGIT_DUMP_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let logit_dump_active = !use_temp_sampling
        && *LOGIT_DUMP_ENV.get_or_init(|| {
            hipfire_config::developer_var("HIPFIRE_DFLASH_LOGIT_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
    let host_path_active = rp_active || ngram_block_active || logit_dump_active;
    // HIPFIRE_DFLASH_FAST_SAMPLE (default ON, opt out with =0): in the temp>0
    // sampled path, compute the per-row softmax (and the top_p nucleus cutoff) on
    // the GPU instead of the host `exp` loop over the full vocab. The host RNG,
    // accept test, residual sampler, and CACTUS math are UNCHANGED — they just
    // read the GPU-produced probabilities instead of host-recomputed ones. The
    // D2H transfer size is identical (full-vocab probs vs full-vocab logits);
    // what moves to the GPU is the ~31×vocab `exp`/max/normalize work per
    // cycle. PARITY: distribution-only, NOT byte-identical (GPU tree-reduction
    // vs host sequential sum can differ at the last ULP and rarely flip a
    // borderline `u*p_d <= p_t` accept) — validated coherent across genres
    // (no attractors), so default-on. Greedy path (temp==0) is never affected.
    let fast_sample_active = use_temp_sampling && gpu.flags.dflash_fast_sample;
    let draft_ffn_graph_env =
        hipfire_config::developer_var("HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH").ok();
    let draft_ffn_graph = dflash_moe_draft_ffn_graph_eligible(
        target.config.num_experts,
        ctx_slice.is_some(),
        pld_spine.is_some(),
        use_temp_sampling,
        draft_ffn_graph_env.as_deref(),
    );

    // C8: device tensors kept alive from draft-sample through verify-accept.
    // Only populated when fast_sample_active AND the batched GEMM path runs.
    // Otherwise stays None and the host accept loop falls back to draft_softmaxes.
    let mut c8_draft_probs_dev: Option<GpuTensor> = None;
    let mut c8_draft_tau_dev: Option<GpuTensor> = None;
    let mut c8_draft_z_dev: Option<GpuTensor> = None;
    let mut c8_draft_tokens_dev: Option<GpuTensor> = None;
    let mut c8_draft_p_at_token_dev: Option<GpuTensor> = None;

    if let Some(pld) = pld_spine {
        // PLD spine path: drafted tokens come from context-suffix match.
        // At temp>0, draft "probability" at each PLD token is 1.0 — PLD is
        // context-deterministic, not a softmax. The rejection math below
        // computes residual from (target_probs − draft_probs)+ normalized,
        // and with draft one-hot at tok, the residual pulls correctly from
        // target minus just that single-position overclaim.
        for i in 0..b - 1 {
            drafted.push(pld[i]);
        }
        if use_temp_sampling {
            draft_probs_at_drafted.reserve(b - 1);
            draft_softmaxes.reserve(b - 1);
            for i in 0..b - 1 {
                let mut probs = vec![0f32; vocab];
                probs[pld[i] as usize] = 1.0;
                draft_softmaxes.push(probs);
                draft_probs_at_drafted.push(1.0);
            }
        }
    } else {
        // ── 2. noise_embedding = target.embed_tokens(block) written directly
        // into draft_scratch.x on GPU (no host round-trip). Target and draft
        // share the same Gpu, so the embedding lookup can target the draft's
        // scratch buffer. Avoids 16 × D2H + one H2D per iter (~1 ms saved).
        for (i, &tok) in block.iter().enumerate() {
            let dst = draft_scratch.x.sub_offset(i * h, h);
            match target.weights.embd_format {
                hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
                    gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
                }
                hipfire_runtime::llama::EmbeddingFormat::HFQ4G128 => {
                    gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
                }
                hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
                    gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
                }
                hipfire_runtime::llama::EmbeddingFormat::F32 => {
                    gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
                }
                _ => panic!("dflash: unsupported target embedding format for noise lookup"),
            }
        }

        // ── 3. Position arrays + optional context slice ─────────────────────
        // Q positions: the absolute positions of the block slots,
        //   [position + compact_offset .. position + B + compact_offset).
        // K positions by default: absolute positions of all populated target_hidden
        // rows (potentially non-contiguous after a TriAttention eviction), then
        // the same block slots.
        //
        // Pre-eviction: positions are contiguous [0..position+B), so the
        // abs_positions vec contains [0..position) and this matches the old
        // behaviour byte-for-byte.
        // Post-eviction: abs_positions contains the subset retained by the last
        // FA layer's top-B mask, paired with the correct pre-eviction absolute
        // positions so draft RoPE aligns with target.
        //
        // If `ctx_slice = Some(N)` is set, restrict the draft's context view to
        // the last `N` rows of target_hidden_host, with RoPE positions
        // [position-N..position+B). Eviction-aware abs positions are not tracked
        // on this diagnostic path — callers using it don't expect FlashCASK.
        let effective_ctx_len = match ctx_slice {
            Some(n) => n.min(position),
            None => draft_scratch.thlog.abs_positions().len().min(position),
        };
        let ctx_start = position - effective_ctx_len;
        let co = target.kv_cache.compact_offset as i32;
        let positions_q: Vec<i32> =
            ((position as i32 + co)..(position as i32 + b as i32 + co)).collect();
        let positions_k: Vec<i32> = if ctx_slice.is_some() {
            // Diagnostic path: keep legacy contiguous layout. abs_positions isn't
            // tracked here and eviction isn't supported with ctx_slice anyway.
            (ctx_start as i32..(position + b) as i32).collect()
        } else {
            let mut v = Vec::with_capacity(effective_ctx_len + b);
            let th_abs = draft_scratch.thlog.abs_positions();
            let start_idx = th_abs.len().saturating_sub(effective_ctx_len);
            v.extend_from_slice(&th_abs[start_idx..]);
            for p in 0..b {
                v.push(position as i32 + p as i32 + co);
            }
            v
        };

        // Slice target_hidden_host to the last effective_ctx_len rows. When
        // ctx_slice is None, this is a no-op (ctx_start = 0). Row stride is
        // num_extract × hidden = ne * h.
        //
        // Fast path (ctx_slice == None, 2026-04-16): `draft_scratch.target_hidden`
        // is already populated via D2D scatter at the END of the previous cycle
        // (or seed_target_hidden_from_prompt for the first cycle). We pass
        // `target_hidden = None` to draft_forward so it skips the H2D upload
        // entirely — kills the per-cycle CPU roundtrip.
        //
        // ctx_slice=Some(N) still goes through the CPU shadow (target_hidden_host
        // Vec) because its moving-window semantics don't map onto the append-only
        // GPU buffer without an extra D2D shuffle. It's a diagnostic path anyway.
        let (th_arg, _th_offset): (Option<&[f32]>, usize) = if ctx_slice.is_some() {
            let th_offset = ctx_start * ne * h;
            (Some(&target_hidden_host[th_offset..]), th_offset)
        } else {
            (None, 0)
        };

        // ── 4. draft_forward ────────────────────────────────────────────────
        // noise_embedding = None: we wrote embeddings directly into
        // draft_scratch.x above via D2D (no host round-trip).
        dflash::draft_forward_opts(
            gpu,
            draft_weights,
            draft_cfg,
            None,
            th_arg,
            &positions_q,
            &positions_k,
            b,
            effective_ctx_len,
            draft_scratch,
            draft_ffn_graph,
        )?;

        // ── 5. Apply target.lm_head to draft hidden positions 1..B ──────────
        // Fast path: a single batched GEMM against target.weights.output over
        // (B-1) hidden rows at once. Drops lm_head from ~40 ms (B-1 serial
        // weight_gemv + downloads) to ~8 ms (one batched GEMM + one download)
        // for MQ4/HFQ4 lm_heads. Falls back to the per-row loop when the
        // output weight dtype isn't covered by the batched gemm dispatch.
        //
        // Temperature-sampling mode (temp > 0): we must DOWNLOAD the full
        // (B-1, vocab) draft logits, softmax + sample + record p_draft[token]
        // for later rejection acceptance. The greedy GPU-argmax path is kept
        // intact for temp == 0 so we don't regress that case.
        let w_out = &target.weights.output;
        let use_batched_gemm = matches!(
            w_out.gpu_dtype,
            rdna_compute::DType::Q8_0
                | rdna_compute::DType::HFQ4G256
                | rdna_compute::DType::MQ4G256
                | rdna_compute::DType::MQ4G256V2
                | rdna_compute::DType::MQ6G256V2
                | rdna_compute::DType::MQ5G256V2
                | rdna_compute::DType::MQ3G256V2
                | rdna_compute::DType::MQ2G256V2
                | rdna_compute::DType::MQ3G256
                | rdna_compute::DType::HFQ6G256
                | rdna_compute::DType::MQ6G256,
        );
        if use_batched_gemm {
            // Unified batched path: one GEMM over B-1 rows, GPU-side argmax,
            // download just (B-1) × 4 bytes of indices.
            //
            // Reuses `verify_scratch.logits` and `.rot` — same buffers the target
            // verify uses. Draft calls this BEFORE verify in the cycle, so
            // there's no aliasing. The verify call overwrites these buffers
            // afterward. Avoids 2-3 hipMalloc/Free pairs per cycle.
            let batch = b - 1;
            assert!(
                batch <= verify_scratch.max_n,
                "verify_scratch max_n {} < draft batch {}",
                verify_scratch.max_n,
                batch
            );
            let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
            let logits_batch = verify_scratch.logits.sub_offset(0, batch * vocab);

            match w_out.gpu_dtype {
                rdna_compute::DType::Q8_0 => {
                    dflash_gemm_q8_lmhead(gpu, w_out, &hidden_rows, &logits_batch, batch)?;
                }
                rdna_compute::DType::HFQ4G256 => {
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &hidden_rows,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ4G256 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    // AWQ-aware rotation; same rationale as the target-verify
                    // arms above.
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq4G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ4G256V2 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ4 v2 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    gpu.gemm_mq4g256v2_batched_lmhead(
                        &w_out.buf,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ6G256V2 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ6V2 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    gpu.gemm_mq6g256v2_batched_lmhead(
                        &w_out.buf,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ5G256V2 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ5V2 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    gpu.gemm_mq5g256v2_batched_lmhead(
                        &w_out.buf,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ3G256V2 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ3V2 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    gpu.gemm_mq3g256v2_batched_lmhead(
                        &w_out.buf,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ2G256V2 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ2V2 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    gpu.gemm_mq2g256v2_batched_lmhead(
                        &w_out.buf,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ3G256 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ3 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::HFQ6G256 => {
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &hidden_rows,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                rdna_compute::DType::MQ6G256 => {
                    assert!(
                        batch * h <= verify_scratch.max_n * verify_scratch.hidden_k,
                        "verify_scratch.rot undersized for MQ6 draft lm_head"
                    );
                    let rotated = verify_scratch.rot.sub_offset(0, batch * h);
                    llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch)?;
                    run_spec_gemm_key(
                        gpu,
                        hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                        &w_out.buf,
                        w_out.gpu_dtype,
                        &rotated,
                        &logits_batch,
                        w_out.m,
                        w_out.k,
                        batch,
                    )?;
                }
                _ => unreachable!(),
            }

            if selector_mode {
                // Selector q is proposal-local: seed anchor this cycle, then
                // sequential tokens. Greedy stays on-device (small top-K D2H
                // inside propose_candidates_device). Temperature materializes
                // sparse q into draft_softmaxes and does NOT take the GPU
                // full-vocab fast-accept path. Request top_p/top_k is not
                // applied to selector q; target p still uses them at verify.
                let uniforms = if use_temp_sampling {
                    Some(
                        (0..batch)
                            .map(|_| xorshift_next_unit(rng_state))
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    None
                };
                let proposal = dflash::propose_candidates_device(
                    gpu,
                    draft_weights,
                    draft_scratch,
                    &hidden_rows,
                    &logits_batch,
                    batch,
                    seed_token,
                    temp,
                    uniforms.as_deref(),
                )?;
                apply_dflash2_selector_proposal(
                    proposal,
                    vocab,
                    use_temp_sampling,
                    &mut drafted,
                    &mut draft_softmaxes,
                    &mut draft_probs_at_drafted,
                )?;
            } else if use_temp_sampling && fast_sample_active {
                // C8 GPU-sample path: softmax stays device-resident; only
                // draft_tokens + draft_p_at_token (batch×8 bytes) come back.
                // draft_probs_dev is kept alive in c8_draft_probs_dev until
                // chain_accept_spec_f32 consumes it after verify.
                let probs_dev = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
                let (tau_dev, z_dev) = if topp_active {
                    let tau_d = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
                    let z_d = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
                    gpu.softmax_temp_topp_batched_into_f32(
                        &logits_batch,
                        &probs_dev,
                        &tau_d,
                        &z_d,
                        vocab,
                        batch,
                        temp,
                        top_p,
                        top_k,
                        0.0, // min_p: DFlash min_p parity is the follow-up; off here
                    )?;
                    (tau_d, z_d)
                } else {
                    gpu.softmax_temp_batched_into_f32(
                        &logits_batch,
                        &probs_dev,
                        vocab,
                        batch,
                        temp,
                    )?;
                    // topp inactive: kernel expects tau=0 (no truncation) and z=1
                    // (inv_z=1 → eff_prob returns p unchanged). Use zeros() for tau
                    // (0.0 bit-pattern = 0x00000000) and fill_f32 for z.
                    let tau_d = gpu.zeros(&[batch], rdna_compute::DType::F32)?;
                    let z_d = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
                    gpu.fill_f32(&z_d, 1.0f32)?;
                    (tau_d, z_d)
                };
                // GPU categorical sample per row: writes draft_tokens + draft_p_at_token.
                let tok_dev = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?; // i32 via f32 slot
                let pat_dev = gpu.alloc_tensor(&[batch], rdna_compute::DType::F32)?;
                let seed_u32 = (*rng_state >> 32) as u32 ^ (*rng_state as u32);
                gpu.batched_categorical_sample_f32(
                    &probs_dev, &tau_dev, &z_dev, &tok_dev, &pat_dev, vocab, batch, seed_u32,
                )?;
                // Download only tokens + probs: batch×8 bytes total.
                let mut raw_tok = vec![0i32; batch];
                {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(raw_tok.as_mut_ptr() as *mut u8, batch * 4)
                    };
                    gpu.hip.memcpy_dtoh(bytes, &tok_dev.buf)?;
                }
                let raw_pat = gpu.download_f32(&pat_dev)?;
                // Keep pat_dev alive on device for chain_accept_spec_f32.
                for i in 0..batch {
                    drafted.push(raw_tok[i] as u32);
                    draft_probs_at_drafted.push(raw_pat[i]);
                }
                // Advance host rng_state by batch steps to maintain entropy across
                // cycles (GPU uses its own LCG; host state must change per cycle).
                for _ in 0..batch {
                    xorshift_next_unit(rng_state);
                }
                // Stash device tensors for chain_accept_spec_f32 in verify step.
                c8_draft_probs_dev = Some(probs_dev);
                c8_draft_tau_dev = Some(tau_dev);
                c8_draft_z_dev = Some(z_dev);
                c8_draft_tokens_dev = Some(tok_dev);
                c8_draft_p_at_token_dev = Some(pat_dev);
                // draft_probs_at_drafted populated above; draft_softmaxes NOT
                // populated (not needed when GPU accept kernel runs).
            } else if use_temp_sampling {
                // Full D2H of (B-1)×vocab logits, CPU softmax+sample.
                let host_logits = gpu.download_f32(&logits_batch)?;
                debug_assert_eq!(host_logits.len(), batch * vocab);
                draft_softmaxes.reserve(batch);
                for i in 0..batch {
                    let row = &host_logits[i * vocab..(i + 1) * vocab];
                    let mut probs = Vec::with_capacity(vocab);
                    softmax_temp_into(row, temp, &mut probs);
                    // Host truncation: top-k before nucleus, matching GPU combined cut.
                    if top_k > 0 && top_k < vocab {
                        apply_host_topk(&mut probs, top_k);
                    }
                    if top_p < 0.999 {
                        apply_host_nucleus(&mut probs, top_p);
                    }
                    let u = xorshift_next_unit(rng_state);
                    let t = sample_categorical(&probs, u);
                    draft_probs_at_drafted.push(probs[t as usize]);
                    drafted.push(t);
                    draft_softmaxes.push(probs);
                }
            } else if host_path_active {
                // RP / n-gram-block path: apply per-row penalties before argmax
                // so draft and target pick from the same reshaped distribution.
                // Keeps spec-decode aligned (τ doesn't collapse from mismatched
                // argmaxes — both sides see identical -inf logits on banned toks).
                let host_logits = gpu.download_f32(&logits_batch)?;
                debug_assert_eq!(host_logits.len(), batch * vocab);
                let mut row = vec![0f32; vocab];
                for i in 0..batch {
                    row.copy_from_slice(&host_logits[i * vocab..(i + 1) * vocab]);
                    if rp_active {
                        llama::apply_repeat_penalty(
                            &mut row,
                            prev_committed,
                            repeat_window,
                            repeat_penalty,
                        );
                    }
                    if ngram_block_active {
                        llama::apply_ngram_block(&mut row, prev_committed);
                    }
                    drafted.push(argmax_u32(&row));
                }
            } else {
                // GPU argmax over (B-1) rows — one kernel, small D2H.
                let argmax_buf = verify_scratch.argmax.sub_offset(0, batch);
                gpu.argmax_f32_batched(&logits_batch, &argmax_buf, vocab, batch)?;
                let mut host_idx = vec![0i32; batch];
                {
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(host_idx.as_mut_ptr() as *mut u8, batch * 4)
                    };
                    gpu.hip.memcpy_dtoh(bytes, &argmax_buf.buf)?;
                }
                for &idx in &host_idx {
                    drafted.push(idx as u32);
                }
            }
        } else if selector_mode {
            // lm-head fallback: still go through propose_candidates_host so
            // greedy is not an independent argmax and sampled q stays sparse.
            let batch = b - 1;
            let mut host_logits = Vec::with_capacity(batch * vocab);
            for i in 1..b {
                let hidden_row = draft_scratch.x.sub_offset(i * h, h);
                llama::weight_gemv(gpu, w_out, &hidden_row, &target.scratch.logits)?;
                let logits = gpu.download_f32(&target.scratch.logits)?;
                debug_assert_eq!(logits.len(), vocab);
                host_logits.extend_from_slice(&logits);
            }
            let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
            let uniforms = if use_temp_sampling {
                Some(
                    (0..batch)
                        .map(|_| xorshift_next_unit(rng_state))
                        .collect::<Vec<f32>>(),
                )
            } else {
                None
            };
            let proposal = dflash::propose_candidates_host(
                gpu,
                draft_weights,
                draft_scratch,
                &hidden_rows,
                &host_logits,
                batch,
                seed_token,
                temp,
                uniforms.as_deref(),
            )?;
            apply_dflash2_selector_proposal(
                proposal,
                vocab,
                use_temp_sampling,
                &mut drafted,
                &mut draft_softmaxes,
                &mut draft_probs_at_drafted,
            )?;
        } else {
            // Fallback: per-row weight_gemv loop.
            for i in 1..b {
                let hidden_row = draft_scratch.x.sub_offset(i * h, h);
                llama::weight_gemv(gpu, w_out, &hidden_row, &target.scratch.logits)?;
                let logits = gpu.download_f32(&target.scratch.logits)?;
                debug_assert_eq!(logits.len(), vocab);
                if use_temp_sampling {
                    let mut probs = Vec::with_capacity(vocab);
                    softmax_temp_into(&logits, temp, &mut probs);
                    if top_k > 0 && top_k < vocab {
                        apply_host_topk(&mut probs, top_k);
                    }
                    if top_p < 0.999 {
                        apply_host_nucleus(&mut probs, top_p);
                    }
                    let u = xorshift_next_unit(rng_state);
                    let t = sample_categorical(&probs, u);
                    draft_probs_at_drafted.push(probs[t as usize]);
                    drafted.push(t);
                    draft_softmaxes.push(probs);
                } else if host_path_active {
                    let mut row = logits.clone();
                    if rp_active {
                        llama::apply_repeat_penalty(
                            &mut row,
                            prev_committed,
                            repeat_window,
                            repeat_penalty,
                        );
                    }
                    if ngram_block_active {
                        llama::apply_ngram_block(&mut row, prev_committed);
                    }
                    drafted.push(argmax_u32(&row));
                } else {
                    drafted.push(argmax_u32(&logits));
                }
            }
        }
    } // close else (DFlash draft path)

    for i in 1..b {
        block[i] = drafted[i];
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_draft_end = std::time::Instant::now();

    // ── 5b. N-gram override (DFlash path only) ───────────────────────────
    // When an n-gram cache is supplied, walk the block left-to-right. For
    // each position i, look up the bigram (block[i-2], block[i-1]) → t. If
    // the cache has a high-enough count for t, override block[i] with t.
    // Chained: subsequent lookups use the (possibly-overridden) prior
    // tokens. Chained overrides only "compound" when the cache captures
    // multi-step patterns (e.g. boilerplate phrases, code indentation).
    //
    // Cost: two HashMap lookups per block position = microseconds.
    //
    // Limitation: dflash's draft_forward already ran against the ORIGINAL
    // draft argmax block; overrides don't feed back into the draft. So
    // downstream positions' target-hidden cross-attention was computed
    // against the un-overridden block. In practice this doesn't matter
    // because the per-position target attention at verify time reruns
    // anyway — what matters is target's argmax at position i versus
    // block[i+1] (the override).
    // Skip bigram override when PLD is the draft source: per Goose §3,
    // PLD tokens have 2–18× higher acceptance than bigram (TR) tokens
    // (median 6×). Overriding PLD with a bigram guess strictly lowers τ.
    if pld_spine.is_none() {
        if let Some(ng) = ngram_cache {
            if prev_committed.len() >= 2 {
                let mut a = prev_committed[prev_committed.len() - 2];
                let mut bb = seed_token;
                for i in 1..b {
                    if let Some((tok, _cnt)) = ng.predict(a, bb) {
                        block[i] = tok;
                        // Also reflect the override in `drafted` so the committed
                        // sequence reported back to the caller matches what was
                        // actually verified against the target.
                        drafted[i] = tok;
                    }
                    a = bb;
                    bb = block[i];
                }
            }
        }
    }

    // ── 6. Snapshot DeltaNet pre-verify, run verify (advances state by B) ─
    //
    // If a GdnTape is supplied, the verify forward also records the
    // per-LA-layer (q, k, v, α, β) innovation tape so the rollback can
    // replay just the GDN recurrence for `accept+1` steps without
    // re-running the target.
    target_snap.save_from(&target.dn_state, gpu)?;
    // Mutable variable to allow both verify capture + rollback replay usage.
    let moe_router_logits_present = verify_scratch
        .prefill_batch
        .as_ref()
        .map(|pbs| pbs.moe_router_logits_batch.is_some())
        .unwrap_or(true);
    let verify_populates_tape = qwen35::prefill_batch_pbs_eligible(
        &target.weights,
        &target.config,
        &target.dn_state,
        b,
        gpu.arch.as_str(),
        moe_router_logits_present,
    );
    let use_tape_replay = dflash_use_gdn_tape_replay(gdn_tape.is_some(), verify_populates_tape);
    let mut gdn_tape_opt = if use_tape_replay { gdn_tape } else { None };

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_verify_start = std::time::Instant::now();
    // Full target logits are D2H'd to the host for rejection sampling, RP, or
    // n-gram block. Under FAST_SAMPLE the rejection sampler reads the
    // GPU-softmaxed target probs directly from the resident
    // `verify_scratch.logits` buffer, so the host full-logit download + host
    // argmax are skipped — that's the target-side half of the cost cut.
    let want_full_logits = (use_temp_sampling && !fast_sample_active) || host_path_active;
    let verify_out = match verify_pm4 {
        Some(route) => {
            let replay_failures_before = route.counters().replay_failures;
            match verify_dflash_block_retained(
                gpu,
                target,
                &block,
                position,
                hidden_rb,
                gdn_tape_opt.as_deref_mut(),
                want_full_logits,
                verify_scratch,
                route,
            ) {
                Ok(out) => out,
                Err(error) => {
                    // Fail closed when quiescence was never proven: a PM4 body
                    // may still be executing against KV, DeltaNet state, and
                    // hidden staging, so restoring or re-running would race it.
                    if matches!(route.phase(), DflashVerifyPm4Phase::Quarantined { .. }) {
                        return Err(error);
                    }
                    // Not a retained-execution failure — an ordinary model
                    // error propagates exactly as it does on the HIP path.
                    if route.counters().replay_failures == replay_failures_before {
                        return Err(error);
                    }
                    // Proven-quiescent replay failure. Rewind the recurrent
                    // state to the pre-window snapshot and redo the window on
                    // the ordinary route; KV slots, hidden staging,
                    // final_hidden, and the GDN tape are all overwritten by the
                    // retry, and the ring is committed exactly once because the
                    // failed attempt never committed.
                    target_snap.restore_to(&mut target.dn_state, gpu)?;
                    route.note_safe_hip_retry();
                    verify_dflash_block(
                        gpu,
                        target,
                        &block,
                        position,
                        hidden_rb,
                        gdn_tape_opt.as_deref_mut(),
                        want_full_logits,
                        verify_scratch,
                    )?
                }
            }
        }
        None => verify_dflash_block(
            gpu,
            target,
            &block,
            position,
            hidden_rb,
            gdn_tape_opt.as_deref_mut(),
            want_full_logits,
            verify_scratch,
        )?,
    };

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_verify_end = std::time::Instant::now();

    // ── 7. Acceptance ──────────────────────────────────────────────────
    //
    // Greedy path: longest prefix where block[i+1] == argmax_per_pos[i].
    //   bonus = argmax_per_pos[accept_len].
    //
    // Rejection-sampling path (temp > 0):
    //   For each i in 0..B-1:
    //     t = block[i+1] (draft sampled this at position start+i+1)
    //     p_d = draft_softmax[i][t]
    //     p_t = target_softmax[i][t]  (softmax of verify logits row i, same temp)
    //     u = rng
    //     accept if u * p_d < p_t
    //     else: rejected → bonus = sample from (p_target - p_draft)+
    //   If all accepted → bonus = sample from target_softmax[B-1].
    let mut accept_len = 0usize;
    let mut bonus_token;
    if use_temp_sampling {
        // When the GPU sampling path ran on the draft side AND we have the
        // required device tensors, run chain_accept_spec_f32 for the full
        // accept loop (C8b: eliminates the ~9 MB target probs D2H + host loop).
        // Otherwise fall through to the host loop (PLD, fallback per-row path).
        // GPU chain_accept only returns the final boundary draw — when the
        // remaining budget can bind mid-window, force the host path so we can
        // clamp accept_len and re-draw the boundary sample from the target row.
        let budget_binds = matches!(max_accept, Some(m) if m < b - 1);
        let gpu_accept = !selector_mode
            && fast_sample_active
            && !budget_binds
            && c8_draft_probs_dev.is_some()
            && c8_draft_tokens_dev.is_some()
            && c8_draft_p_at_token_dev.is_some();

        if gpu_accept {
            // ── C8 GPU accept path ──────────────────────────────────────────
            // verify_scratch.logits[0..b*vocab] contains the target logits for
            // all b positions (b = draft_batch + 1; last row = bonus position).
            // Softmax them into tgt_probs_dev (kept device-resident).
            let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
            let tgt_probs_dev = gpu.alloc_tensor(&[b * vocab], rdna_compute::DType::F32)?;
            let (tau_t_dev, z_t_dev) = if topp_active {
                let tau_t = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
                let z_t = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
                gpu.softmax_temp_topp_batched_into_f32(
                    &logits_batch,
                    &tgt_probs_dev,
                    &tau_t,
                    &z_t,
                    vocab,
                    b,
                    temp,
                    top_p,
                    top_k,
                    0.0,
                )?;
                (tau_t, z_t)
            } else {
                gpu.softmax_temp_batched_into_f32(&logits_batch, &tgt_probs_dev, vocab, b, temp)?;
                let tau_t = gpu.zeros(&[b], rdna_compute::DType::F32)?;
                let z_t = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
                gpu.fill_f32(&z_t, 1.0f32)?;
                (tau_t, z_t)
            };

            let dft_probs_dev = c8_draft_probs_dev.as_ref().unwrap();
            let dft_tok_dev = c8_draft_tokens_dev.as_ref().unwrap();
            let dft_pat_dev = c8_draft_p_at_token_dev.as_ref().unwrap();
            let tau_d_dev = c8_draft_tau_dev.as_ref().unwrap();
            let z_d_dev = c8_draft_z_dev.as_ref().unwrap();

            let out_dev = gpu.alloc_tensor(&[4], rdna_compute::DType::F32)?; // 4×i32 via f32 slot
                                                                             // kernel's b parameter = number of draft comparison positions = b-1
            let draft_b = b - 1;
            let rng_seed = (*rng_state >> 32) as u32 ^ (*rng_state as u32);
            gpu.chain_accept_spec_f32(
                &tgt_probs_dev,
                dft_probs_dev,
                dft_tok_dev,
                dft_pat_dev,
                &tau_t_dev,
                &z_t_dev,
                tau_d_dev,
                z_d_dev,
                &out_dev,
                draft_b,
                vocab,
                rng_seed,
                cactus_delta,
            )?;

            // Download 16 bytes: {accept_len, bonus_token, rejected_at, new_rng}
            let mut out_raw = [0i32; 4];
            {
                let bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(out_raw.as_mut_ptr() as *mut u8, 16) };
                gpu.hip.memcpy_dtoh(bytes, &out_dev.buf)?;
            }
            accept_len = out_raw[0] as usize;
            bonus_token = out_raw[1] as u32;
            // Advance host rng_state once so future cycles seed differently.
            // GPU consumed its own LCG draws; host state just needs to change.
            xorshift_next_unit(rng_state);

            let _ = gpu.free_tensor(out_dev);
            let _ = gpu.free_tensor(tgt_probs_dev);
            let _ = gpu.free_tensor(tau_t_dev);
            let _ = gpu.free_tensor(z_t_dev);
        } else {
            // ── Host accept path (PLD / fallback per-row) ──────────────────
            debug_assert_eq!(draft_softmaxes.len(), b - 1);
            // FAST_SAMPLE: compute the per-row target softmax on the GPU once for
            // all B rows and download the probs. The accept loop below then reads
            // `target_probs` from this buffer instead of calling the host
            // `softmax_temp_into` — identical RNG/accept/residual/CACTUS math,
            // distribution-parity probs. The verify logits are still resident in
            // `verify_scratch.logits[0..b*vocab]` (verify enqueued lm_head into it
            // and only ran GPU-argmax afterwards, which does not overwrite it).
            let mut fast_tgt_tau: Option<Vec<f32>> = None;
            let mut fast_tgt_z: Option<Vec<f32>> = None;
            let fast_tgt_probs: Option<Vec<f32>> = if fast_sample_active {
                let logits_batch = verify_scratch.logits.sub_offset(0, b * vocab);
                let probs_gpu = gpu.alloc_tensor(&[b * vocab], rdna_compute::DType::F32)?;
                if topp_active {
                    let tau_gpu = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
                    let z_gpu = gpu.alloc_tensor(&[b], rdna_compute::DType::F32)?;
                    gpu.softmax_temp_topp_batched_into_f32(
                        &logits_batch,
                        &probs_gpu,
                        &tau_gpu,
                        &z_gpu,
                        vocab,
                        b,
                        temp,
                        top_p,
                        top_k,
                        0.0,
                    )?;
                    fast_tgt_tau = Some(gpu.download_f32(&tau_gpu)?);
                    fast_tgt_z = Some(gpu.download_f32(&z_gpu)?);
                    let _ = gpu.free_tensor(tau_gpu);
                    let _ = gpu.free_tensor(z_gpu);
                } else {
                    gpu.softmax_temp_batched_into_f32(&logits_batch, &probs_gpu, vocab, b, temp)?;
                }
                let host = gpu.download_f32(&probs_gpu)?;
                let _ = gpu.free_tensor(probs_gpu);
                debug_assert_eq!(host.len(), b * vocab);
                Some(host)
            } else {
                debug_assert_eq!(verify_out.logits_per_pos.len(), b * vocab);
                None
            };
            let tgt_logits = &verify_out.logits_per_pos;
            let mut target_probs = Vec::with_capacity(vocab);
            let mut rejected_bonus: Option<u32> = None;
            // CACTUS (Hao & Mou 2026, arXiv:2604.04987 Corollary 5).
            let use_cactus = cactus_delta > 0.0;
            for i in 0..b - 1 {
                // Pre-commit budget: stop accepting drafts once emit would
                // exceed max_emit (emit = accept_len + 1). Bonus is re-drawn
                // from the target row at the clamped boundary below.
                if let Some(m) = max_accept {
                    if accept_len >= m {
                        break;
                    }
                }
                if let Some(fast) = &fast_tgt_probs {
                    target_probs.clear();
                    target_probs.extend_from_slice(&fast[i * vocab..(i + 1) * vocab]);
                    if let (Some(tau), Some(z)) = (&fast_tgt_tau, &fast_tgt_z) {
                        apply_topp_trunc(&mut target_probs, tau[i], z[i]);
                    }
                } else {
                    softmax_temp_into(
                        &tgt_logits[i * vocab..(i + 1) * vocab],
                        temp,
                        &mut target_probs,
                    );
                    if top_k > 0 && top_k < vocab {
                        apply_host_topk(&mut target_probs, top_k);
                    }
                    if top_p < 0.999 {
                        apply_host_nucleus(&mut target_probs, top_p);
                    }
                }
                let t = block[i + 1] as usize;
                let p_d = draft_probs_at_drafted[i].max(f32::MIN_POSITIVE);
                let p_t = target_probs[t];
                let accept_prob = if use_cactus {
                    let bump = (2.0 * cactus_delta * p_t * (1.0 - p_t)).max(0.0).sqrt();
                    (p_t + bump).min(1.0)
                } else {
                    p_t
                };
                let u = xorshift_next_unit(rng_state);
                if u * p_d <= accept_prob {
                    accept_len += 1;
                } else {
                    if use_cactus {
                        let qn = p_t.clamp(0.0, 1.0);
                        let gamma_star = accept_prob;
                        if qn >= 1.0 - 1e-6 {
                            for v in target_probs.iter_mut() {
                                *v = 0.0;
                            }
                            target_probs[t] = 1.0;
                        } else {
                            let scale = (1.0 - gamma_star) / (1.0 - qn);
                            for (j, v) in target_probs.iter_mut().enumerate() {
                                *v = if j == t { gamma_star } else { scale * *v };
                            }
                        }
                    }
                    let u2 = xorshift_next_unit(rng_state);
                    rejected_bonus = Some(sample_residual(&target_probs, &draft_softmaxes[i], u2));
                    break;
                }
            }
            bonus_token = if let Some(b) = rejected_bonus {
                b
            } else {
                // Full-accept OR budget-bound: draw bonus from the target row
                // at the boundary (accept_len). Never substitute argmax.
                let i = accept_len.min(b - 1);
                if let Some(fast) = &fast_tgt_probs {
                    target_probs.clear();
                    target_probs.extend_from_slice(&fast[i * vocab..(i + 1) * vocab]);
                } else {
                    softmax_temp_into(
                        &tgt_logits[i * vocab..(i + 1) * vocab],
                        temp,
                        &mut target_probs,
                    );
                    if top_k > 0 && top_k < vocab {
                        apply_host_topk(&mut target_probs, top_k);
                    }
                    if top_p < 0.999 {
                        apply_host_nucleus(&mut target_probs, top_p);
                    }
                }
                let u = xorshift_next_unit(rng_state);
                sample_categorical(&target_probs, u)
            };
        }
    } else {
        // Greedy path. If RP or n-gram-block is active, re-derive argmax per
        // row after applying penalties to the full target logits (requires
        // want_full_logits). `prev_committed` carries the emitted history
        // used as the penalty / block window.
        let argmax_per_pos: std::borrow::Cow<'_, [u32]> = if host_path_active {
            let tgt_logits = &verify_out.logits_per_pos;
            debug_assert_eq!(tgt_logits.len(), b * vocab);
            let mut out: Vec<u32> = Vec::with_capacity(b);
            let mut row = vec![0f32; vocab];
            for i in 0..b {
                row.copy_from_slice(&tgt_logits[i * vocab..(i + 1) * vocab]);
                if rp_active {
                    llama::apply_repeat_penalty(
                        &mut row,
                        prev_committed,
                        repeat_window,
                        repeat_penalty,
                    );
                }
                if ngram_block_active {
                    llama::apply_ngram_block(&mut row, prev_committed);
                }
                out.push(argmax_u32(&row));
            }
            std::borrow::Cow::Owned(out)
        } else {
            std::borrow::Cow::Borrowed(verify_out.argmax_per_pos.as_slice())
        };
        // Shared greedy accept-prefix (eos=None: DFlash never early-stops on EOS
        // here — the daemon handles EOS downstream). drafts = block[1..b],
        // target_pick = the (repeat-penalty / n-gram-adjusted) argmax;
        // bonus = target_pick[accept_len].
        let acc = hipfire_runtime::spec::accept_greedy_prefix(&block[1..b], &argmax_per_pos, None);
        accept_len = acc.accepted;
        bonus_token = *acc.committed.last().expect("eos=None yields a bonus");
        // Pre-commit budget: clamp drafts so emit (= accept + 1) ≤ max_emit,
        // then re-pick bonus = argmax at the clamped boundary (temp-0 correct).
        if let Some(m) = max_accept {
            if accept_len > m {
                accept_len = m;
                bonus_token = argmax_per_pos[accept_len];
            }
        }

        if logit_dump_active {
            // Inspect the acceptance boundary. If accept_len < b-1 the block
            // was rejected at position accept_len (drafted loser = block[accept_len+1]);
            // otherwise the whole block accepted and we report the bonus slot.
            let tgt = &verify_out.logits_per_pos;
            let rej_i = accept_len.min(b - 1);
            let row = &tgt[rej_i * vocab..(rej_i + 1) * vocab];
            let (mut top1_idx, mut top1, mut top2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
            for (j, &v) in row.iter().enumerate() {
                if v > top1 {
                    top2 = top1;
                    top1 = v;
                    top1_idx = j;
                } else if v > top2 {
                    top2 = v;
                }
            }
            let rejected = accept_len < b - 1;
            let rej_tok: i64 = if rejected {
                block[accept_len + 1] as i64
            } else {
                -1
            };
            let (rej_logit, gap, rej_rank) = if rejected {
                let rv = row[rej_tok as usize];
                let rank = row.iter().filter(|&&v| v > rv).count() + 1;
                (rv, top1 - rv, rank as i64)
            } else {
                (f32::NAN, f32::NAN, -1)
            };
            eprintln!(
                "DFLDUMP pos={} b={} accept={} rejected={} top1_tok={} top1={:.5} top2={:.5} margin={:.5} rej_tok={} rej_logit={:.5} gap={:.5} rej_rank={} bonus={}",
                position, b, accept_len, rejected as u8, top1_idx, top1, top2,
                top1 - top2, rej_tok, rej_logit, gap, rej_rank, bonus_token
            );
        }
    }

    // Free C8 device tensors now that accept is resolved.
    if let Some(t) = c8_draft_probs_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_tau_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_z_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_tokens_dev {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = c8_draft_p_at_token_dev {
        let _ = gpu.free_tensor(t);
    }

    // ── 7b. Seed-prediction oracle (Task #93 Phase B) ───────────────────
    // Three position-based proxies for the next cycle's `seed_token`
    // (= this cycle's `bonus_token`). See comment at top of file for the
    // reasoning — the PRD's "naive argmax at rejection boundary" proxy is
    // 0 % by construction (the accept loop broke precisely there), which
    // we measure as REJ_MATCH to document the dead-end. TAIL_MATCH and
    // ANYPOS_MATCH are the actually-usable ceilings.
    let rej_proxy: Option<u32> = if accept_len + 1 < b {
        Some(drafted[accept_len + 1])
    } else {
        None
    };
    let tail_proxy: u32 = drafted[b - 1];
    let anypos_hit: bool = drafted[1..b].iter().any(|&t| t == bonus_token);
    let rej_hit: bool = rej_proxy == Some(bonus_token);
    let tail_hit: bool = tail_proxy == bonus_token;
    SEED_ORACLE_TOTAL.fetch_add(1, Ordering::Relaxed);
    SEED_ORACLE_ACCEPT_LEN_SUM.fetch_add(accept_len as u64, Ordering::Relaxed);
    if rej_hit {
        SEED_ORACLE_REJ_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if tail_hit {
        SEED_ORACLE_TAIL_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if anypos_hit {
        SEED_ORACLE_ANYPOS_MATCH.fetch_add(1, Ordering::Relaxed);
    }
    if rej_proxy.is_none() {
        SEED_ORACLE_FULLACCEPT.fetch_add(1, Ordering::Relaxed);
    }
    if hipfire_config::developer_var("HIPFIRE_DFLASH_SEED_ORACLE")
        .ok()
        .as_deref()
        == Some("1")
    {
        let s = read_seed_oracle_stats();
        let denom = s.total.max(1) as f32;
        eprintln!(
            "[seed-oracle] cycle: accept_len={} b={} bonus={} rej={:?}/{} tail={}/{} anypos={} fullacc={} | cum rej={:.3} tail={:.3} anypos={:.3} mean_accept={:.2}",
            accept_len, b, bonus_token, rej_proxy, rej_hit,
            tail_proxy, tail_hit, anypos_hit, rej_proxy.is_none(),
            s.rej_match as f32 / denom,
            s.tail_match as f32 / denom,
            s.anypos_match as f32 / denom,
            s.accept_len_sum as f32 / denom,
        );
    }

    // ── 8. Committed sequence ───────────────────────────────────────────
    // committed[0] is the seed_token (already emitted by prev iter). The
    // caller's output stream appends committed[1..]. We include seed in
    // committed because target KV/state must be at position seed+accept_len+1
    // after this step.
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for i in 0..accept_len {
        committed.push(drafted[i + 1]);
    }
    committed.push(bonus_token);
    let committed_count = committed.len();
    debug_assert_eq!(committed_count, accept_len + 2);

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_accept_end = std::time::Instant::now();

    // ── 9. Append accepted target hidden rows to target_hidden_host ─────
    // Verify wrote B rows into hidden_rb. We keep the first accept_len+1
    // (= committed_count - 1) because the last committed token (bonus) is
    // ALREADY reflected in target state + will get its hidden captured on
    // the NEXT verify when it's forwarded as block[0].
    //
    // Wait: bonus_token is placed at position `position + accept_len + 1`.
    // Its hidden was captured at ring slot (verify start + accept_len),
    // which corresponds to the B-th verify forward position = position +
    // accept_len. That's the bonus position if we identify it correctly.
    //
    // Actually every verify position writes one hidden row. Position i of
    // the B-verify corresponds to absolute position `position + i`, so:
    //   block[0] hidden captured at ring slot (head - B + 0) → pos=position
    //   block[1] hidden captured at ring slot (head - B + 1) → pos=position+1
    //   ...
    //   block[accept_len] hidden captured → pos=position+accept_len (THIS is the last committed before bonus)
    //   block[accept_len+1] hidden captured → pos=position+accept_len+1 (this would be bonus; but target's prediction at that slot is what drove the bonus choice)
    //
    // The bonus token is what target WOULD predict at position+accept_len+1
    // given the B-verify input. Its hidden was NOT captured at that
    // position — the hidden at that slot is for `block[accept_len+1]`, a
    // REJECTED draft token's target forward. We can't use that hidden for
    // the committed bonus token.
    //
    // Resolution: DON'T append bonus-token hidden here. Next iter's
    // verify will forward the bonus token at its position (position +
    // committed_count - 1) as its new block[0], capturing proper hidden
    // and target state there. Committed_count - 1 rows appended here
    // covers positions [position..position + committed_count - 2] =
    // [position..position + accept_len]. Bonus at position+accept_len+1
    // sits in no-man's land — its hidden will materialize on next iter.
    //
    // This matches the reference's `target_hidden = ...[:, :accept_len+1, :]`
    // pattern which slices the verify's hidden output to accept_len+1
    // rows — NOT accept_len+2.
    let rows_to_keep = accept_len + 1;
    if ctx_slice.is_some() {
        // ctx_slice path: CPU shadow still required for the window slice.
        let hidden_block = download_hidden_block(gpu, hidden_rb, b)?;
        target_hidden_host.extend_from_slice(&hidden_block[..rows_to_keep * ne * h]);
    } else {
        // Fast path: scatter straight from hidden_rb into draft scratch on GPU.
        // No D2H, no CPU reshape, no next-cycle H2D.
        //
        // Verify just wrote B slots to hidden_rb; we want the first
        // `rows_to_keep` (= accept+1) of those. Pass block_size=b so the
        // scatter function aligns to the verify-block origin, not the
        // ring tail.
        scatter_hidden_block_to_interleaved(
            gpu,
            hidden_rb,
            &draft_scratch.target_hidden,
            position,
            b,
            rows_to_keep,
            draft_scratch.ctx_modulus(),
        )?;
        // Keep draft_forward's incremental-upload tracker in sync so any future
        // ctx_slice=Some call in the same session doesn't try to re-upload what
        // GPU already has, and track the absolute positions of the appended
        // rows (logical `position..position+rows_to_keep` plus the current
        // target KV compact_offset — zero pre-eviction, non-zero after) for the
        // next cycle's `positions_k` construction.
        let co = target.kv_cache.compact_offset as i32;
        draft_scratch
            .thlog
            .append_committed(position, rows_to_keep, co);
    }

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_scatter_end = std::time::Instant::now();

    // ── 10. Rewind DeltaNet + replay committed tokens ────────────────────
    // After verify, target state reflects B forwards. We need it to reflect
    // `committed_count - 1 = accept_len + 1` forwards (the seed + accepted
    // draft tokens). The bonus token is NOT replayed — it will be
    // block[0] of the next iter. This keeps the invariant that before each
    // verify, target state is at position `start` (= pre-verify position).
    target_snap.restore_to(&mut target.dn_state, gpu)?;

    if phase_on {
        gpu.hip.device_synchronize()?;
    }
    let t_restore_end = std::time::Instant::now();
    // Tape-replay path (0.1.7 perf): if a GdnTape was captured during verify,
    // replay the GatedDeltaNet recurrence for (accept+1) steps using the
    // recorded (q, k, v, α, β) tuples — no full-target re-run needed. The
    // FullAttention layers don't need explicit rewind because the next
    // verify (starting at position + accept + 1) will overwrite their KV
    // cache slots [position + accept + 1 .. position + accept + 1 + B),
    // which subsumes the previously-written [position..position + B) range.
    //
    // Fallback (no tape): batched forward_prefill_batch over (accept+1)
    // tokens, same as the prior version — re-runs the full target but one
    // batched call instead of (accept+1) sequential decodes.
    if let Some(tape) = gdn_tape_opt.as_deref() {
        tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
    } else {
        let replay_tokens = &committed[..accept_len + 1];
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            replay_tokens,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            None,
            None,
            None,
            None,
        )?;
    }
    // Target state is now at position + accept_len + 1. KV cache has
    // written K/V at positions [position..position+accept_len]. The bonus
    // token's K/V will be written on the next iter's verify (at position
    // `position + accept_len + 1`) as part of that iter's block[0] forward.

    if phase_on {
        gpu.hip.device_synchronize()?;
        let t_end = std::time::Instant::now();
        let us_draft = t_draft_end.duration_since(t_spec_start).as_micros();
        let us_ngram = t_verify_start.duration_since(t_draft_end).as_micros();
        let us_verify = t_verify_end.duration_since(t_verify_start).as_micros();
        let us_accept = t_accept_end.duration_since(t_verify_end).as_micros();
        let us_scatter = t_scatter_end.duration_since(t_accept_end).as_micros();
        let us_restore = t_restore_end.duration_since(t_scatter_end).as_micros();
        let us_replay = t_end.duration_since(t_restore_end).as_micros();
        let us_total = t_end.duration_since(t_spec_start).as_micros();
        eprintln!(
            "[phase] B={} accept={} draft={}µs ngram={}µs verify={}µs \
             cmpr={}µs scatter={}µs restore={}µs replay={}µs | total={}µs",
            b,
            accept_len,
            us_draft,
            us_ngram,
            us_verify,
            us_accept,
            us_scatter,
            us_restore,
            us_replay,
            us_total,
        );
    }
    let _ = (
        t_phase,
        t_draft_end,
        t_verify_start,
        t_verify_end,
        t_accept_end,
        t_scatter_end,
        t_restore_end,
    );

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Run the DFlash draft forward + lm_head, return the raw per-position draft
/// logits as a host `Vec<f32>` of length `(b - 1) * vocab`.
///
/// Shared factor-out of the draft-producing half of spec_step_dflash — used by
/// spec_step_ddtree to feed Algorithm 1 with per-position top-K. The vanilla
/// DFlash path doesn't call this because it takes the argmax/softmax directly
/// on GPU (smaller D2H); the tree path needs raw logits for top-K + log-norm.
///
/// Leaves `draft_scratch.x` populated with draft hidden rows, so callers that
/// also want argmax for diagnostics can walk those rows afterward (not used
/// here). Does NOT advance the target KV cache or DeltaNet state — only the
/// draft forward runs.
#[cfg(feature = "deltanet")]
fn run_dflash_draft_for_logits(
    gpu: &mut Gpu,
    target: &ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    target_hidden_host: &[f32],
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    b: usize,
) -> HipResult<Vec<f32>> {
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;
    assert!(b >= 2, "dflash draft: b must be ≥ 2");

    // Block: [seed, mask, mask, ...].
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;

    // Step 1: D2D embedding lookup per block slot (parallels spec_step_dflash).
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("ddtree draft: unsupported target embedding format"),
        }
    }

    // Step 2: Positions + optional ctx_slice (identical to spec_step_dflash).
    let effective_ctx_len = match ctx_slice {
        Some(n) => n.min(position),
        None => position,
    };
    let ctx_start = position - effective_ctx_len;
    let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
    let positions_k: Vec<i32> = (ctx_start as i32..(position + b) as i32).collect();
    let th_offset = ctx_start * ne * h;
    let th_slice: &[f32] = &target_hidden_host[th_offset..];

    // Step 3: Draft forward (fills draft_scratch.x with per-position draft
    // hidden rows).
    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        Some(th_slice),
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // Step 4: Apply target.lm_head to draft hidden rows [1..B). Same batched
    // GEMM paths as spec_step_dflash. Unlike the vanilla path we download
    // the full (B-1) × vocab logits so the tree builder can compute top-K.
    let batch = b - 1;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    let logits_batch = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
    let w_out = &target.weights.output;

    let gemm_result = match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            dflash_gemm_q8_lmhead(gpu, w_out, &hidden_rows, &logits_batch, batch)
        }
        rdna_compute::DType::HFQ4G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ4G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            // AWQ-aware rotation; see the target-verify dispatch above for the
            // rationale (numerically identical when `w_out.awq_scale` is None).
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ4G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq4g256v2_batched_lmhead(
                &w_out.buf,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ6G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq6g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ5G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq5g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq3g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ2G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq2g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::HFQ6G256 => {
            // Phase A.4: HFQ6 lm_head batched via gemm_hfq6g256_batched_lmhead
            // (which zeros Y then dispatches the dp4a residual on gfx906 or
            // WMMA / FP16 fallbacks elsewhere).
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ6G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256/MQ4G256V2/MQ6G256V2/MQ5G256V2/MQ3G256V2/MQ2G256V2/MQ3G256/HFQ6G256/MQ6G256)",
        )),
    };
    if let Err(e) = gemm_result {
        let _ = gpu.free_tensor(logits_batch);
        return Err(e);
    }

    let host_logits = match gpu.download_f32(&logits_batch) {
        Ok(v) => v,
        Err(e) => {
            let _ = gpu.free_tensor(logits_batch);
            return Err(e);
        }
    };
    let _ = gpu.free_tensor(logits_batch);
    debug_assert_eq!(host_logits.len(), batch * vocab);
    Ok(host_logits)
}

/// Like `run_dflash_draft_for_logits` but does the top-K + log-sum-exp
/// ON GPU via `topk_logsumexp_batched_f32`, returning only the top-K
/// tokens and log-probs per row. Used by `spec_step_ddtree_batched` to
/// skip the ~20 ms CPU sort and the 15 MB logits D2H.
///
/// Returns `(top_tokens, top_log_probs)` each of size `(b-1) * k` in
/// row-major order (same convention as `ddtree::topk_from_logits`).
#[allow(clippy::too_many_arguments)]
/// Upload an `i32` slice as a device buffer (no I32 DType — raw bytes; kernels
/// read it as `int*`).
fn upload_i32(gpu: &Gpu, data: &[i32]) -> HipResult<GpuTensor> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    gpu.upload_raw(bytes, &[data.len()])
}

/// Run the fused on-device SWOR tree-verify walk and return the CPU-shaped
/// `(accepted_node_indices, bonus_token)`. Builds the device metadata
/// (slot→depth, draw-ordered candidates, child-of-candidate adjacency), launches
/// `ddtree_swor_walk_f32` against the device-resident target + draft logits, and
/// reads back only the tiny result. No full-vocab work on the host.
#[allow(clippy::too_many_arguments)]
fn swor_walk_gpu(
    gpu: &mut Gpu,
    tree: &hipfire_runtime::ddtree::DdTree,
    target_logits: &GpuTensor,
    draft_logits: &GpuTensor,
    pos_cands: &[u32],
    num_pos: usize,
    k: usize,
    vocab: usize,
    temp: f32,
    seed: u64,
) -> HipResult<(Vec<usize>, u32)> {
    let n_slots = 1 + tree.nodes.len();
    let mut slot_depth = vec![0i32; n_slots];
    for s in 1..n_slots {
        slot_depth[s] = tree.nodes[s - 1].depth as i32;
    }
    let mut child_of_cand = vec![-1i32; n_slots * k];
    for s in 0..n_slots {
        let depth = slot_depth[s] as usize;
        if depth >= num_pos {
            continue;
        }
        for r in 0..k {
            let token = pos_cands[depth * k + r];
            if let Some(&ci) = tree.child_maps[s].get(&token) {
                child_of_cand[s * k + r] = ci as i32;
            }
        }
    }
    let pos_cands_i32: Vec<i32> = pos_cands.iter().map(|&t| t as i32).collect();
    let t_pcand = upload_i32(gpu, &pos_cands_i32)?;
    let t_depth = upload_i32(gpu, &slot_depth)?;
    let t_child = upload_i32(gpu, &child_of_cand)?;
    let t_pres = gpu.alloc_tensor(&[vocab], rdna_compute::DType::F32)?;
    let t_qpos = gpu.alloc_tensor(&[vocab], rdna_compute::DType::F32)?;
    let t_out = gpu.alloc_tensor(&[2 + num_pos], rdna_compute::DType::F32)?;
    gpu.ddtree_swor_walk_f32(
        target_logits,
        draft_logits,
        &t_pcand,
        &t_depth,
        &t_child,
        &t_pres,
        &t_qpos,
        &t_out,
        temp,
        k,
        vocab,
        n_slots,
        num_pos,
        seed,
    )?;
    let raw = gpu.download_f32(&t_out)?;
    let _ = gpu.free_tensor(t_pcand);
    let _ = gpu.free_tensor(t_depth);
    let _ = gpu.free_tensor(t_child);
    let _ = gpu.free_tensor(t_pres);
    let _ = gpu.free_tensor(t_qpos);
    let _ = gpu.free_tensor(t_out);
    let accept_len = (raw[0].to_bits() as i32).max(0) as usize;
    let bonus = raw[1].to_bits() as u32;
    let mut accepted = Vec::with_capacity(accept_len);
    for i in 0..accept_len {
        accepted.push((raw[2 + i].to_bits() as i32) as usize);
    }
    Ok((accepted, bonus))
}

fn run_dflash_draft_for_topk_gpu(
    gpu: &mut Gpu,
    target: &ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    // Stage 1b: `None` = GPU-resident path (scratch.target_hidden already
    // populated via D2D scatter; thlog tracks the live rows). `Some(slice)`
    // = ctx_slice diagnostic path (host shadow, always full H2D upload).
    target_hidden_host: Option<&[f32]>,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    b: usize,
    k: usize,
    // `Some((temp, rng))` ⇒ draw the `k` children per position WITHOUT replacement
    // from the draft softmax at `temp` (Gumbel-top-k = exact SWOR sampling), and
    // return their TRUE per-position log-probs. This makes the tree's candidates
    // genuine draft samples, which is what the q-exploiting SWOR verify needs to
    // be distribution-exact. `None` ⇒ the default deterministic GPU top-k.
    sample: Option<(f32, &mut u64)>,
) -> HipResult<(Vec<u32>, Vec<f32>, Option<GpuTensor>)> {
    // The 3rd return is the draft logits kept ON DEVICE (`[batch·vocab]`),
    // populated only in `sample` mode — the fused GPU SWOR walk softmaxes them
    // for its residual, so they must NOT be freed here. `None` on the top-k path;
    // the SWOR caller frees the tensor after the walk.
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    let vocab = target.config.vocab_size;
    let mask_token = draft_cfg.mask_token_id;
    assert!(b >= 2, "dflash draft: b must be ≥ 2");
    assert!(k >= 1 && k <= 8, "topk k={} must be in [1, 8]", k);

    // Step 1-3: identical to run_dflash_draft_for_logits — embed, positions,
    // draft forward. Duplicating the small glue to avoid a refactor risk;
    // this path is shipped after. (Could factor out, but the savings is <50
    // lines and the call site is stable.)
    let mut block: Vec<u32> = vec![mask_token; b];
    block[0] = seed_token;
    for (i, &tok) in block.iter().enumerate() {
        let dst = draft_scratch.x.sub_offset(i * h, h);
        match target.weights.embd_format {
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&target.weights.token_embd, &dst, tok, h)?
            }
            hipfire_runtime::llama::EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&target.weights.token_embd, &dst, tok, h)?
            }
            _ => panic!("ddtree draft: unsupported target embedding format"),
        }
    }
    // Stage 1b: positions and hidden-source depend on whether we are using the
    // GPU-resident path (target_hidden_host=None) or the ctx_slice host path.
    let (positions_q, positions_k, th_arg, effective_ctx_len) =
        if let Some(host_slice) = target_hidden_host {
            // ctx_slice diagnostic path: host shadow, simple contiguous positions.
            let effective_ctx_len = match ctx_slice {
                Some(n) => n.min(position),
                None => position,
            };
            let ctx_start = position - effective_ctx_len;
            let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
            let positions_k: Vec<i32> = (ctx_start as i32..(position + b) as i32).collect();
            let th_offset = ctx_start * ne * h;
            (
                positions_q,
                positions_k,
                Some(&host_slice[th_offset..]),
                effective_ctx_len,
            )
        } else {
            // GPU-resident path: scratch.target_hidden already contains all rows
            // via D2D scatter (Stage 1); thlog tracks uploaded_rows + abs_positions.
            // Mirror the chain path (spec_step_dflash ~line 3157): use thlog's
            // eviction-aware abs_positions for k-positions, pass None to skip H2D.
            let effective_ctx_len = draft_scratch.thlog.abs_positions().len().min(position);
            let ctx_start = position - effective_ctx_len;
            let positions_q: Vec<i32> = (position as i32..(position + b) as i32).collect();
            let th_abs = draft_scratch.thlog.abs_positions();
            let start_idx = th_abs.len().saturating_sub(effective_ctx_len);
            let mut positions_k: Vec<i32> = Vec::with_capacity(effective_ctx_len + b);
            positions_k.extend_from_slice(&th_abs[start_idx..]);
            for p in 0..b {
                positions_k.push(position as i32 + p as i32);
            }
            let _ = ctx_start; // not used in this branch
            (positions_q, positions_k, None, effective_ctx_len)
        };

    dflash::draft_forward(
        gpu,
        draft_weights,
        draft_cfg,
        None,
        th_arg,
        &positions_q,
        &positions_k,
        b,
        effective_ctx_len,
        draft_scratch,
    )?;

    // Step 4: lm_head → [batch × vocab] logits (GPU-resident).
    let batch = b - 1;
    let hidden_rows = draft_scratch.x.sub_offset(h, batch * h);
    let logits_batch = gpu.alloc_tensor(&[batch * vocab], rdna_compute::DType::F32)?;
    let w_out = &target.weights.output;
    let gemm_result = match w_out.gpu_dtype {
        rdna_compute::DType::Q8_0 => {
            dflash_gemm_q8_lmhead(gpu, w_out, &hidden_rows, &logits_batch, batch)
        }
        rdna_compute::DType::HFQ4G256 => {
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ4G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            // AWQ-aware rotation for the target lm_head when an AWQ
            // sidecar is attached. Sister of the spec-verify dispatch above.
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ4G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq4g256v2_batched_lmhead(
                &w_out.buf,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ6G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq6g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ5G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq5g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq3g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ2G256V2 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = gpu.gemm_mq2g256v2_batched_lmhead(&w_out.buf, &rotated, &logits_batch, w_out.m, w_out.k, batch);
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::MQ3G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq3G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        rdna_compute::DType::HFQ6G256 => {
            // Phase A.4: HFQ6 lm_head batched.
            run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &hidden_rows,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            )
        }
        rdna_compute::DType::MQ6G256 => {
            let rotated = gpu.alloc_tensor(&[batch * h], rdna_compute::DType::F32)?;
            let r1 = llama::rotate_x_mq_batched_for(gpu, w_out, &hidden_rows, &rotated, h, batch);
            if let Err(e) = r1 {
                let _ = gpu.free_tensor(rotated);
                let _ = gpu.free_tensor(logits_batch);
                return Err(e);
            }
            let r2 = run_spec_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmHfq6G256BatchedLmhead,
                &w_out.buf,
                w_out.gpu_dtype,
                &rotated,
                &logits_batch,
                w_out.m,
                w_out.k,
                batch,
            );
            let _ = gpu.free_tensor(rotated);
            r2
        }
        _ => Err(hip_bridge::HipError::new(
            0,
            "ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256/MQ4G256V2/MQ6G256V2/MQ5G256V2/MQ3G256V2/MQ2G256V2/MQ3G256/HFQ6G256/MQ6G256)",
        )),
    };
    if let Err(e) = gemm_result {
        let _ = gpu.free_tensor(logits_batch);
        return Err(e);
    }

    // Step 5a: Gumbel-top-k SWOR sampling (q-exploiting verify). Download the
    // draft logits and, per position, draw k tokens WITHOUT replacement from
    // softmax(logits/temp) via the Gumbel-top-k trick (top-k by `logit/temp +
    // Gumbel(0,1)` is an exact size-k SWOR sample), returning each token's TRUE
    // per-position log-prob `log_softmax(logits/temp)[token]`. The candidates are
    // then genuine draft samples — the precondition the SWOR verify needs.
    if let Some((temp, rng)) = sample {
        // Device-side Gumbel-top-k SWOR sampling: NO [B×vocab] D2H. Draw the k
        // draw-ordered candidates per position + their true log-q on the GPU; only
        // B×k come back for the CPU tree build. `logits_batch` stays resident for
        // the fused SWOR walk (returned below).
        let seed = *rng;
        xorshift_next_unit(rng);
        let idx_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
        let logp_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
        gpu.ddtree_gumbel_topk_batched_f32(
            &logits_batch,
            &idx_gpu,
            &logp_gpu,
            vocab,
            k,
            batch,
            temp,
            seed,
        )?;
        let idx_host = gpu.download_f32(&idx_gpu)?;
        let top_log_probs = gpu.download_f32(&logp_gpu)?;
        let _ = gpu.free_tensor(idx_gpu);
        let _ = gpu.free_tensor(logp_gpu);
        let top_tokens: Vec<u32> = idx_host.iter().map(|f| f.to_bits()).collect();
        return Ok((top_tokens, top_log_probs, Some(logits_batch)));
    }

    // Step 5: GPU top-K + log-sum-exp. Writes [batch × k] indices + log-probs.
    let topk_idx_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
    let topk_val_gpu = gpu.alloc_tensor(&[batch * k], rdna_compute::DType::F32)?;
    let topk_result = gpu.topk_logsumexp_batched_f32(
        &logits_batch,
        &topk_idx_gpu,
        &topk_val_gpu,
        vocab,
        k,
        batch,
    );
    let _ = gpu.free_tensor(logits_batch);
    if let Err(e) = topk_result {
        let _ = gpu.free_tensor(topk_idx_gpu);
        let _ = gpu.free_tensor(topk_val_gpu);
        return Err(e);
    }

    // Step 6: D2H just the top-K outputs (tiny — 8 × 15 × 4 = 480 bytes for k=8).
    let mut idx_host: Vec<i32> = vec![0i32; batch * k];
    let mut val_host: Vec<f32> = vec![0f32; batch * k];
    let idx_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(idx_host.as_mut_ptr() as *mut u8, batch * k * 4) };
    let val_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(val_host.as_mut_ptr() as *mut u8, batch * k * 4) };
    gpu.hip.memcpy_dtoh(idx_bytes, &topk_idx_gpu.buf)?;
    gpu.hip.memcpy_dtoh(val_bytes, &topk_val_gpu.buf)?;
    let _ = gpu.free_tensor(topk_idx_gpu);
    let _ = gpu.free_tensor(topk_val_gpu);

    let top_tokens: Vec<u32> = idx_host.into_iter().map(|x| x as u32).collect();
    Ok((top_tokens, val_host, None))
}

/// Enumerate all root-to-leaf paths in a DdTree. Returns paths as Vec<Vec<usize>>
/// where each inner Vec is the sequence of node indices from the first
/// child-of-root (depth 1) down to a leaf. Leaves are nodes with no children
/// in the tree; if the tree is empty (N=0) this returns a single empty path.
fn enumerate_paths(tree: &hipfire_runtime::ddtree::DdTree) -> Vec<Vec<usize>> {
    if tree.nodes.is_empty() {
        return vec![Vec::new()];
    }
    let mut leaves: Vec<usize> = Vec::new();
    for i in 0..tree.nodes.len() {
        let slot = i + 1;
        if tree.child_maps[slot].is_empty() {
            leaves.push(i);
        }
    }
    let mut paths: Vec<Vec<usize>> = Vec::with_capacity(leaves.len());
    for &leaf_idx in &leaves {
        let mut path: Vec<usize> = Vec::new();
        let mut cur: i32 = leaf_idx as i32;
        while cur >= 0 {
            path.push(cur as usize);
            cur = tree.nodes[cur as usize].parent_index;
        }
        path.reverse();
        paths.push(path);
    }
    paths
}

/// DDTree speculative step (Ringel & Romano 2026, our hybrid-arch port).
///
/// Flow per cycle:
///   1. Run DFlash draft, download raw (B-1) × vocab logits.
///   2. CPU top-K + log-norm per row → per-position (tokens, log-probs).
///   3. Algorithm 1: best-first heap builds up to `tree_budget` tree nodes.
///   4. Snapshot target state (pre-seed). Forward seed once to get posterior[0]
///      and the post-seed branch point; snapshot post-seed state.
///   5. For each root-to-leaf path in the tree, forward each node sequentially
///      through `forward_scratch`; on first visit of a node slot, record its
///      target argmax as `posterior[slot]`. Restore post-seed state between paths.
///   6. Greedy walk: follow target's argmax down the tree to the longest
///      accepted path + bonus token.
///   7. Restore to pre-seed, re-forward (seed + accepted path) with hidden
///      capture so the next cycle's DFlash draft has valid target_hidden_host.
///
/// Cost per cycle: O(N) target forwards where N is the node budget (paper
/// uses 60; we default to `draft_cfg.block_size` = 16 for a cheaper spike).
/// That's ~5× the batched-verify cost of spec_step_dflash; no batched tree
/// attention on hybrid arch would change that, but per-path verify is the
/// correctness-first path (LA state is not polluted across branches).
///
/// Temp=0 only for now — rejection-sampling / CACTUS integration is deferred
/// until the greedy signal looks promising. Paper's DDTree numbers are
/// temp=0 too, so this matches the reference setup.
#[cfg(feature = "deltanet")]
pub fn spec_step_ddtree(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    post_seed_snap: &mut DeltaNetSnapshot,
    gdn_tape: &mut GdnTape,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    tree_budget: usize,
    tree_topk: usize,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let vocab = target.config.vocab_size;
    let h = draft_cfg.hidden;
    let ne = draft_cfg.num_extract();
    assert!(b >= 2, "spec_step_ddtree: block_size must be ≥ 2");
    assert_eq!(
        target_hidden_host.len(),
        position * ne * h,
        "target_hidden_host size mismatches position"
    );
    assert!(
        tree_topk >= 1 && tree_topk <= vocab,
        "tree_topk must be in [1, vocab]"
    );
    if draft_weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(
            0,
            "DFlash2 candidate selector is chain-only; DDTree independent-marginal verify is disabled",
        ));
    }

    // ── 1. Run DFlash draft, download raw logits ─────────────────────────
    let draft_logits = run_dflash_draft_for_logits(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        target_hidden_host,
        position,
        seed_token,
        ctx_slice,
        b,
    )?;

    // ── 2. Per-position top-K + log-normalize (CPU) ───────────────────────
    let (top_tokens, top_log_probs) =
        hipfire_runtime::ddtree::topk_from_logits(&draft_logits, b - 1, vocab, tree_topk);

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    // HIPFIRE_DDTREE_LOGW_CUTOFF=<f32> enables the meta-verifier pruner: stop
    // heap expansion when the next candidate's cumulative log-probability
    // drops below -cutoff. Per-cycle dynamic budget. Disabled (= 0.0 or
    // unset) preserves the fixed-budget behaviour.
    let tree = hipfire_runtime::ddtree::build_ddtree_tree_bounded(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        0, // min_nodes=0: pure-cutoff DDTree build
        tree_budget,
        gpu.flags.ddtree_logw_cutoff_value(),
    );
    record_ddtree_meta_nodes(tree.num_nodes());

    // Edge case: empty tree (shouldn't happen if budget≥1 and b≥2, but guard).
    // With zero nodes there's nothing to verify — just forward seed, sample,
    // commit. Mirrors the behavior of a B=2 DFlash cycle.
    // Note: `forward_scratch_with_hidden` runs the final rmsnorm + lm_head
    // internally and leaves the next-token logits in `scratch.logits` — do
    // NOT call weight_gemv again on scratch.x (that's pre-rmsnorm hidden
    // and produces incorrect logits).
    if tree.nodes.is_empty() {
        target_snap.save_from(&target.dn_state, gpu)?;
        qwen35::forward_scratch_with_hidden(
            gpu,
            &target.weights,
            &target.config,
            seed_token,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            hidden_rb,
        )?;
        let logits0 = gpu.download_f32(&target.scratch.logits)?;
        let bonus = argmax_u32(&logits0);
        let hidden_block = download_hidden_block(gpu, hidden_rb, 1)?;
        target_hidden_host.extend_from_slice(&hidden_block[..1 * ne * h]);
        return Ok(SpecStepResult {
            accepted: 0,
            bonus_token: bonus,
            drafted: vec![seed_token],
            committed: vec![seed_token, bonus],
        });
    }

    // ── 4. Snapshot pre-seed target state ─────────────────────────────────
    //
    // We verify each root-to-leaf path via `verify_dflash_block` starting
    // from the pre-seed state — this is the same batched target forward
    // DFlash uses for its verify, so we stay byte-exact with the non-tree
    // path. Between paths we restore pre-seed (both DN and KV cache; KV
    // overwrites happen naturally because each verify writes to the same
    // position range starting at `position`).
    target_snap.save_from(&target.dn_state, gpu)?;
    // post_seed_snap is allocated by the caller but unused in this path —
    // kept in the signature so the API stays compatible with potentially
    // sharing-the-seed-forward optimizations in a later rev. Suppress the
    // unused warning without asking the caller to annotate.
    let _ = &post_seed_snap;

    let mut posterior: Vec<u32> = vec![0; 1 + tree.num_nodes()];
    let mut posterior_set: Vec<bool> = vec![false; 1 + tree.num_nodes()];

    // ── 5. Per-path verify via verify_dflash_block ───────────────────────
    //
    // For each root-to-leaf path, run the batched target verify on
    // [seed_token, path_tokens...]. verify_dflash_block gives us argmax
    // per position via the same code path as spec_step_dflash, which
    // guarantees no numerical drift vs baseline at temp=0. Per-node
    // posterior records are first-visit-wins — all paths traversing the
    // same ancestor produce the same argmax at that ancestor's slot.
    let paths = enumerate_paths(&tree);
    for path in &paths {
        // Build verify block: [seed] + path_tokens.
        let mut verify_block: Vec<u32> = Vec::with_capacity(1 + path.len());
        verify_block.push(seed_token);
        for &ni in path {
            verify_block.push(tree.nodes[ni].token);
        }

        // Restore pre-seed state before each verify. DN state via snapshot;
        // KV cache self-overwrites at positions [position, position+N).
        target_snap.restore_to(&mut target.dn_state, gpu)?;

        // NOTE: verify_dflash_block takes &mut HiddenStateRingBuffer (not
        // Option); we pass our buffer but its writes get clobbered by the
        // step-8 replay. That's fine — we only read hidden_rb in step 9
        // after the replay. Path verifies DO advance the ring buffer head
        // but the final replay brings it right back.
        let verify_out = verify_dflash_block(
            gpu,
            target,
            &verify_block,
            position,
            hidden_rb,
            None,
            false, // want_full_logits=false — greedy only for now
            verify_scratch,
        )?;

        // verify_out.argmax_per_pos has length N = verify_block.len().
        // argmax_per_pos[i] = target's predicted NEXT token at position
        // `position + i`. That's:
        //   i=0          → prediction after seed = what should match block[1]
        //                  = posterior at root slot
        //   i=1..N-1     → prediction after node at path-position i-1
        //                  = posterior at path[i-1]'s slot
        // (We don't use argmax_per_pos[N-1] because we'd need a child of
        // the leaf, which the tree doesn't have — greedy walk stops there.)
        if !posterior_set[0] {
            posterior[0] = verify_out.argmax_per_pos[0];
            posterior_set[0] = true;
        }
        for (i, &ni) in path.iter().enumerate() {
            let slot = ni + 1;
            if !posterior_set[slot] && i + 1 < verify_out.argmax_per_pos.len() {
                posterior[slot] = verify_out.argmax_per_pos[i + 1];
                posterior_set[slot] = true;
            }
        }
    }

    // ── 6. Greedy walk: longest accepted path + bonus ─────────────────────
    let (accepted_node_indices, bonus_token) =
        hipfire_runtime::ddtree::follow_verified_tree(&tree, &posterior);
    let accept_len = accepted_node_indices.len();

    // ── 7. Build committed + drafted sequences ────────────────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for &ni in &accepted_node_indices {
        committed.push(tree.nodes[ni].token);
    }
    committed.push(bonus_token);

    let mut drafted: Vec<u32> = Vec::with_capacity(accept_len + 1);
    drafted.push(seed_token);
    for &ni in &accepted_node_indices {
        drafted.push(tree.nodes[ni].token);
    }

    // ── 8. Tape-capturing verify on the committed path, then tape replay ─
    //
    // The tape records per-LA-layer (q, k, v, α, β) innovations for the
    // tokens it processes. Replaying the tape then advances DN state
    // through THOSE tokens. So the tape MUST be captured from a verify
    // whose block contains the actual committed tokens — any divergence
    // (e.g., capturing from the top-1 chain when the tree accepted a
    // rank>0 branch) feeds wrong LA updates into the next cycle's state.
    //
    // For topk=1 the tree's only path IS the top-1 chain, so committed
    // (length accept_len+1) is a prefix of the full-B DFlash block; we
    // still verify at full B here to stay batch-size-identical with the
    // DFlash baseline, then replay just the first accept_len+1 tape
    // steps. That path is byte-exact with baseline.
    //
    // For topk>1 the committed path may contain branch tokens that don't
    // appear in dflash_block's top-1 chain. In that case we fall back to
    // running the tape capture over the committed path directly — not
    // batch-size-equal to DFlash but tokens-correct. Some cross-cycle
    // numerical drift vs baseline is the tradeoff; output should remain
    // a valid target-greedy sequence.
    let topk1_is_committed_prefix = accept_len > 0
        && committed[1..=accept_len]
            .iter()
            .enumerate()
            .all(|(d, &tok)| tok == top_tokens[d * tree_topk]);
    let tape_block: Vec<u32> = if topk1_is_committed_prefix || accept_len == 0 {
        // Safe to use full-B top-1 block (byte-exact with DFlash path).
        let mut vb: Vec<u32> = Vec::with_capacity(b);
        vb.push(seed_token);
        for d in 0..(b - 1) {
            vb.push(top_tokens[d * tree_topk]);
        }
        vb
    } else {
        // Accepted a branch — verify over the committed tokens to get
        // correct LA innovations.
        committed[..accept_len + 1].to_vec()
    };
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    let _tape_verify = verify_dflash_block(
        gpu,
        target,
        &tape_block,
        position,
        hidden_rb,
        Some(gdn_tape),
        false,
        verify_scratch,
    )?;
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    gdn_tape.replay_gdn(
        gpu,
        &target.weights,
        &target.config,
        &mut target.dn_state,
        accept_len + 1,
    )?;
    // Target state is now at position + accept_len + 1. Bonus token's state
    // is deferred to next cycle's block[0], matching spec_step_dflash.

    // ── 9. Append (1 + accept_len) hidden rows to target_hidden_host ─────
    //
    // The tape-capturing verify wrote `tape_block.len()` rows to hidden_rb.
    // We want the FIRST (accept_len + 1) — positions [position, position +
    // accept_len] of the verified block. download_hidden_block returns the
    // most-recent N rows in order, so pulling tape_block.len() rows and
    // slicing to accept_len+1 grabs the right prefix.
    let hidden_rows_written = tape_block.len();
    let hidden_block = download_hidden_block(gpu, hidden_rb, hidden_rows_written)?;
    let rows_to_keep = accept_len + 1;
    target_hidden_host.extend_from_slice(&hidden_block[..rows_to_keep * ne * h]);

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Committed-order row moves for a DDTree post-accept gather, in BLOCK-LOCAL
/// row space (row 0 is the seed slot and never moves).
///
/// Committed row `i + 1` must end up holding linearization slot
/// `accepted_node_indices[i] + 1`. Consecutive moves with the same
/// source→destination delta are coalesced into one run. A spine accept yields
/// an empty list, which is exactly the existing fast path.
fn ddtree_commit_row_runs(accepted_node_indices: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for (i, &ni) in accepted_node_indices.iter().enumerate() {
        let (dst, src) = (i + 1, ni + 1);
        if dst == src {
            continue;
        }
        match runs.last_mut() {
            Some((d, s, n)) if *d + *n == dst && *s + *n == src => *n += 1,
            _ => runs.push((dst, src, 1)),
        }
    }
    runs
}

/// Move rows within a single position-addressed buffer, in place.
///
/// `moves` are `(dst_row, src_row, n_rows)` in PHYSICAL row space, ordered by
/// ascending `dst_row`.
///
/// No scratch buffer is required, and that is a property of the linearization
/// rather than luck: slots are assigned in depth order, so the i-th accepted
/// node can never occupy a slot below `i` (`src >= dst` for every move). Writing
/// row `d` can therefore only alias the source of a LATER move, whose source
/// `s' >= d' > d` — every source is read before anything overwrites it. Distinct
/// block rows map to distinct physical rows, so this also holds when the caller
/// maps through a ring modulus.
fn move_rows_in_place(
    gpu: &Gpu,
    buf: &GpuTensor,
    row_bytes: usize,
    moves: &[(usize, usize, usize)],
) -> HipResult<()> {
    for &(dst, src, n) in moves {
        let (dst_off, src_off, size) = (dst * row_bytes, src * row_bytes, n * row_bytes);
        match gpu.active_stream.as_ref() {
            // Stream-ordered: the moves serialize against each other and against
            // the verify that produced the rows, with no host-side drain.
            Some(stream) => gpu
                .hip
                .memcpy_dtod_async_at(&buf.buf, dst_off, &buf.buf, src_off, size, stream)?,
            None => gpu
                .hip
                .memcpy_dtod_at(&buf.buf, dst_off, &buf.buf, src_off, size)?,
        }
    }
    Ok(())
}

/// DDTree post-accept gather: relocate the accepted chain's rows from their
/// linearization slots into the committed linear slots.
///
/// The tree verify already wrote every node's K/V at its OWN slot (commit
/// 39aa358 decoupled the two: RoPE rotates at DEPTH positions while KV writes
/// use flat physical slots), and the i-th accepted node sits at depth `i + 1`,
/// so its baked-in RoPE phase already equals its committed linear position.
/// The bytes are correct where they sit — only the row INDEX is wrong when the
/// greedy walk detoured off the spine. Moving them is therefore equivalent to
/// (and cheaper than) re-running the committed prefix through a second verify.
///
/// Covers all three row-addressed structures the rest of the cycle consumes:
/// the target KV cache (per full-attention layer, plus the separate scale planes
/// in `quant_int8` mode), the GDN tape (per linear-attention layer), and the
/// extracted-hidden ring. Afterwards the cycle is indistinguishable from a spine
/// accept, so every downstream consumer takes its fast path unchanged.
fn ddtree_gather_committed_rows(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    gdn_tape: &GdnTape,
    hidden_rb: &HiddenStateRingBuffer,
    position: usize,
    block_size: usize,
    accepted_node_indices: &[usize],
) -> HipResult<()> {
    let runs = ddtree_commit_row_runs(accepted_node_indices);
    if runs.is_empty() {
        return Ok(());
    }

    // ── target KV cache ───────────────────────────────────────────────────
    // Row stride is derived from the allocation rather than recomputed per
    // quant mode: every K/V layout here is `physical_cap` rows of equal size,
    // and re-deriving it would duplicate five mode-specific formulas.
    let cap = target.kv_cache.physical_cap.max(1);
    let kv_moves: Vec<(usize, usize, usize)> = runs
        .iter()
        .map(|&(d, s, n)| (position + d, position + s, n))
        .collect();
    for li in 0..target.config.layer_types.len() {
        if target.config.layer_types[li] != qwen35::LayerType::FullAttention {
            continue;
        }
        for plane in [
            target.kv_cache.k_gpu.get(li),
            target.kv_cache.v_gpu.get(li),
            target.kv_cache.k_scales.get(li),
            target.kv_cache.v_scales.get(li),
        ]
        .into_iter()
        .flatten()
        {
            let bytes = plane.byte_size();
            if bytes == 0 {
                continue;
            }
            move_rows_in_place(gpu, plane, bytes / cap, &kv_moves)?;
        }
    }

    // ── GDN tape (block-local rows) ───────────────────────────────────────
    let gate_row_bytes = gdn_tape.n_v_heads * 4;
    for la in 0..gdn_tape.qkv_bufs.len() {
        move_rows_in_place(gpu, &gdn_tape.qkv_bufs[la], gdn_tape.qkv_dim * 4, &runs)?;
        move_rows_in_place(gpu, &gdn_tape.alpha_bufs[la], gate_row_bytes, &runs)?;
        move_rows_in_place(gpu, &gdn_tape.beta_bufs[la], gate_row_bytes, &runs)?;
    }

    // ── extracted-hidden ring ─────────────────────────────────────────────
    // Ring slots wrap, so these go row-by-row through the ring mapping instead
    // of as coalesced runs (the no-scratch argument above survives the wrap
    // because distinct block rows map to distinct slots).
    let ring_moves: Vec<(usize, usize, usize)> = runs
        .iter()
        .flat_map(|&(d, s, n)| (0..n).map(move |k| (d + k, s + k, 1)))
        .map(|(d, s, n)| {
            (
                hidden_rb.block_row_slot(block_size, d),
                hidden_rb.block_row_slot(block_size, s),
                n,
            )
        })
        .collect();
    let hidden_row_bytes = hidden_rb.hidden_dim * 4;
    for buf in &hidden_rb.layer_bufs {
        move_rows_in_place(gpu, buf, hidden_row_bytes, &ring_moves)?;
    }
    Ok(())
}

/// Batched tree-verify counterpart of `spec_step_ddtree`. Replaces the
/// per-path DFS with a single `verify_dflash_block_tree` call using the
/// FA tree-attention mask infrastructure (commits 835aa46 / f0ee980 /
/// 704bf11). Same return value and side-effect semantics as the per-path
/// version — callers swap the two transparently.
///
/// Correctness notes:
///
/// - **FA side (tree-exact):** each tree node's Q attends only to its
///   ancestors + prompt. The mask is -inf on non-ancestor in-block keys
///   so exp-sum collapses, matching per-path DFS argmaxes exactly at
///   temp=0.
/// - **GDN side (linear-replay approximation):** in the tree forward
///   the recurrent GDN kernel advances state sequentially through the
///   linearized token order `[seed, n0, n1, ...]`. For siblings at the
///   same tree depth this cross-contaminates state — node b's S-state
///   update sees node a's innovations even though they're alternatives,
///   not sequential. At `topk=1` the tree is a pure chain so state
///   advance is identical to DFlash (byte-exact). At `topk>1` the FA
///   posteriors are still correct (ancestor attention), but the GDN
///   contribution to each node's hidden has small drift vs per-path DFS.
/// - **Tape/commit path (correct):** we do a SECOND verify on the
///   committed prefix (no tree) for tape capture. That tape is byte-
///   exact with per-path DFS's committed-prefix verify, so LA state
///   advances correctly after the cycle completes.
///
#[allow(clippy::too_many_arguments)]
pub fn spec_step_ddtree_batched(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    draft_weights: &DflashWeights,
    draft_cfg: &DflashConfig,
    draft_scratch: &mut DflashScratch,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    target_snap: &mut DeltaNetSnapshot,
    post_seed_snap: &mut DeltaNetSnapshot,
    gdn_tape: &mut GdnTape,
    scratch: &DdtreeScratch,
    verify_scratch: &VerifyScratch,
    position: usize,
    seed_token: u32,
    ctx_slice: Option<usize>,
    tree_budget: usize,
    tree_topk: usize,
    // Temperature for the verify-side acceptance. temp == 0 → greedy argmax
    // walk (follow_verified_tree); temp > 0 → distribution-preserving naive
    // tree sampling (sample_verified_tree), which needs the full per-slot
    // target logits and consumes `rng_state` (xorshift, shared with the linear
    // DFlash sampler convention).
    temp: f32,
    rng_state: &mut u64,
    // Max accepted tree nodes this window (`emit.len() == accepted + 1`).
    // `None` = uncapped. Clamped before tape replay / hidden / KV commit.
    max_accept: Option<usize>,
) -> HipResult<SpecStepResult> {
    let b = draft_cfg.block_size;
    let vocab = target.config.vocab_size;
    assert!(b >= 2, "spec_step_ddtree_batched: block_size must be ≥ 2");
    // Stage 1b: target_hidden_host is no longer maintained in the GPU-resident
    // default path (ctx_slice=None). The length-invariant assert is removed
    // to avoid false failures. The ctx_slice=Some path still uses the host Vec.
    assert!(
        tree_topk >= 1 && tree_topk <= vocab,
        "tree_topk must be in [1, vocab]"
    );
    if draft_weights.has_candidate_selector() {
        return Err(hip_bridge::HipError::new(
            0,
            "DFlash2 candidate selector is chain-only; DDTree independent-marginal verify is disabled",
        ));
    }

    // D16: ensure active_stream is set before any work so memset_async and
    // the stream-scoped sync in verify_dflash_block_inner have a non-null
    // stream to ride on. Mirrors the identical setup in spec_step_dflash
    // (~line 2968) — da2753e pattern.
    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }

    // Unused in the batched path (no per-path DFS), kept in signature for
    // API compatibility with `spec_step_ddtree` so callers can switch by
    // flipping a single fn pointer.
    let _ = &post_seed_snap;

    // `DDTREE_TIMING=1` prints per-cycle breakdown: draft / topk / build /
    // pre_verify / verify. Used to diagnose where the wall-clock goes.
    // `DDTREE_TIMING=1` prints per-cycle breakdown: draft+topk / build /
    // pre_verify / verify. The draft and top-K are fused into one GPU-
    // resident path now — no separate timer.
    let debug_tm = std::env::var("DDTREE_TIMING").is_ok();
    let t_all = std::time::Instant::now();

    // Verify-scheme selector. SWOR (q-exploiting Sequoia/SpecTr, distribution-
    // exact) is the ONLY temp>0 verify path — the naive-sampling fallback
    // (HIPFIRE_DDTREE_VERIFY=naive) was removed in D8 because it required
    // a ~37 MB/cycle full-logits D2H and SWOR is strictly superior.
    // temp=0 → greedy argmax walk; temp>0 → on-GPU SWOR.
    let use_swor = temp > 0.0;

    // ── 1+2. GPU-resident draft + per-row top-K + log-sum-exp ────────────
    // Keeps logits on device; returns only (b-1) × k indices + log-probs
    // to the host. Replaces the prior 15 MB D2H + CPU sort pair (~34 ms)
    // with an on-device top-K (~µs) plus a ~480 byte D2H. SWOR mode instead
    // Gumbel-top-k samples the children (download + CPU sample).
    //
    // Stage 1b: Pass `None` as target_hidden_host in the default production
    // path (ctx_slice=None) so draft_forward skips the H2D upload — the
    // GPU-resident scratch.target_hidden is already populated by the previous
    // cycle's D2D scatter (or the prefill scatter). The ctx_slice diagnostic
    // path (Some(n)) still uses the CPU host shadow.
    let th_host_arg: Option<&[f32]> = if ctx_slice.is_none() {
        None // GPU-resident: scratch.target_hidden populated via D2D scatter
    } else {
        Some(target_hidden_host.as_slice())
    };
    let (top_tokens, top_log_probs, draft_logits_dev) = run_dflash_draft_for_topk_gpu(
        gpu,
        target,
        draft_weights,
        draft_cfg,
        draft_scratch,
        th_host_arg,
        position,
        seed_token,
        ctx_slice,
        b,
        tree_topk,
        if use_swor {
            Some((temp, &mut *rng_state))
        } else {
            None
        },
    )?;
    // SWOR verify needs the draw-ordered candidates per position; `top_tokens` is
    // already in Gumbel-draw order (rank 0 = first drawn) in sample mode.
    let swor_pos_cands = top_tokens.clone();

    let t_draft = t_all.elapsed();
    let t_topk = t_draft; // fused with draft now

    // ── 3. Build the DDTree ───────────────────────────────────────────────
    // HIPFIRE_DDTREE_LOGW_CUTOFF=<f32> enables the meta-verifier pruner: stop
    // heap expansion when the next candidate's cumulative log-probability
    // drops below -cutoff. Per-cycle dynamic budget. Disabled (= 0.0 or
    // unset) preserves the fixed-budget behaviour.
    let tree = hipfire_runtime::ddtree::build_ddtree_tree_bounded(
        &top_tokens,
        &top_log_probs,
        b - 1,
        tree_topk,
        0, // min_nodes=0: pure-cutoff DDTree build
        tree_budget,
        gpu.flags.ddtree_logw_cutoff_value(),
    );
    record_ddtree_meta_nodes(tree.num_nodes());

    let t_build = t_all.elapsed();

    // Empty-tree shortcut (identical to spec_step_ddtree's path).
    if tree.nodes.is_empty() {
        target_snap.save_from(&target.dn_state, gpu)?;
        qwen35::forward_scratch_with_hidden(
            gpu,
            &target.weights,
            &target.config,
            seed_token,
            position,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            hidden_rb,
        )?;
        let logits0 = gpu.download_f32(&target.scratch.logits)?;
        let bonus = argmax_u32(&logits0);
        // D2D scatter 1 row into draft_scratch.target_hidden, then keep
        // target_hidden_host length-consistent for the next cycle's assert.
        scatter_hidden_block_to_interleaved(
            gpu,
            hidden_rb,
            &draft_scratch.target_hidden,
            position,
            1,
            1,
            draft_scratch.ctx_modulus(),
        )?;
        let co = target.kv_cache.compact_offset as i32;
        draft_scratch.thlog.append_committed(position, 1, co);
        // Stage 1b: no CPU download needed — GPU buffer is authoritative.
        return Ok(SpecStepResult {
            accepted: 0,
            bonus_token: bonus,
            drafted: vec![seed_token],
            committed: vec![seed_token, bonus],
        });
    }

    // ── 4. Linearize the tree into (tokens, positions, mask_host, parents) ─
    //
    // mask_host is computed here for the HIPFIRE_DDTREE_ASSERT_MASK=1 dual-
    // path check (byte-equality proof vs GPU build). In normal operation it
    // is not uploaded — the GPU kernel (step 5) rebuilds it on-device.
    let (verify_tokens, verify_positions, mask_host, parent_host) =
        hipfire_runtime::ddtree::linearize_tree_with_parents(&tree, seed_token, position as u32);
    let big_n = verify_tokens.len();
    debug_assert_eq!(big_n, 1 + tree.num_nodes());
    debug_assert_eq!(parent_host.len(), big_n);

    assert!(
        big_n <= scratch.max_n,
        "tree big_n {} exceeds scratch.max_n {} (increase DdtreeScratch size)",
        big_n,
        scratch.max_n,
    );

    // ── 5. Upload parent_indices; build attn_bias mask on-GPU (Stage 3a) ─
    //
    // D5: parent_indices H2D (244 B, stays — needed for tree-aware LA and
    // for the mask kernel itself).  D4 (attn_bias H2D, ~15 KB/cycle) is
    // eliminated: instead of uploading mask_host we launch
    // ddtree_build_attn_mask_f32 which walks the parent chain per-thread
    // and writes scratch.attn_bias directly on-device.
    //
    // parent_indices is always uploaded (needed both by the mask kernel and,
    // when ddtree_tree_la=true, by the tree-aware GDN kernels).
    let use_tree_la = gpu.flags.ddtree_tree_la;
    {
        let parent_bytes = unsafe {
            std::slice::from_raw_parts(parent_host.as_ptr() as *const u8, parent_host.len() * 4)
        };
        gpu.hip
            .memcpy_htod(&scratch.parent_indices.buf, parent_bytes)?;
    }

    // Build attn_bias on GPU from the now-resident parent_indices. The sub-
    // offset view is used by the FA kernel (step 7) to bound-check big_n²
    // reads.  The kernel writes exactly big_n² floats at the head of
    // scratch.attn_bias; tail is never read by the FA kernel.
    gpu.ddtree_build_attn_mask_f32(&scratch.parent_indices, &scratch.attn_bias, big_n)?;

    // ── 5b. Byte-equality assert (HIPFIRE_DDTREE_ASSERT_MASK=1) ──────────
    //
    // Dual-path proof: download the GPU mask and compare byte-for-byte with
    // the host mask_host computed by linearize_tree_with_parents above.
    // Off by default (costs one D2H per cycle).
    if hipfire_config::developer_var("HIPFIRE_DDTREE_ASSERT_MASK")
        .ok()
        .as_deref()
        == Some("1")
    {
        // Synchronize so the kernel has finished writing before the D2H.
        gpu.hip.device_synchronize()?;
        let n_floats = big_n * big_n;
        let mut gpu_mask = vec![0.0f32; n_floats];
        let gpu_mask_bytes = unsafe {
            std::slice::from_raw_parts_mut(gpu_mask.as_mut_ptr() as *mut u8, n_floats * 4)
        };
        gpu.hip
            .memcpy_dtoh(gpu_mask_bytes, &scratch.attn_bias.buf)?;
        // mask_host is big_n*big_n; gpu_mask is big_n*big_n.
        assert_eq!(
            mask_host.len(),
            n_floats,
            "ASSERT_MASK: mask_host.len() mismatch"
        );
        for idx in 0..n_floats {
            let h = mask_host[idx].to_bits();
            let g = gpu_mask[idx].to_bits();
            assert_eq!(
                h, g,
                "ASSERT_MASK: mismatch at flat index {idx} (row={}, col={}): host={:08x} gpu={:08x}",
                idx / big_n,
                idx % big_n,
                h,
                g,
            );
        }
        eprintln!(
            "[DDTREE_ASSERT_MASK] big_n={big_n} mask byte-identical ({} floats)",
            n_floats
        );
    }

    // ── 6. Snapshot pre-seed target state ─────────────────────────────────
    target_snap.save_from(&target.dn_state, gpu)?;

    // ── 7. Tree verify: single batched forward with tree-attention mask ──
    //
    // Key optimization: pass `gdn_tape` INTO the tree verify so GDN
    // innovations get captured in the linear tree-traversal order. For the
    // topk=1 (or topk>1 where the accepted path coincides with the top-1
    // linear chain) case, the committed path is a contiguous prefix of the
    // linear order — so replaying `tape[0..accept_len+1]` advances LA state
    // correctly and we save an entire forward pass. For topk>1 paths that
    // diverge from the linear prefix, we fall back to a second verify over
    // the committed tokens (step 10 below).
    //
    // argmax_per_pos[i] = target's argmax prediction at slot i in the
    // linearization, i.e. what comes AFTER the token at that slot.
    // Sub-offset view sized to the exact big_n × big_n the current tree needs.
    // scratch.attn_bias is sized for the worst case (max_n² = (1+max_budget)²),
    // but when the actual tree is smaller (e.g. topk=1 linear-chain trees
    // don't fill max_budget), forward_prefill_batch's assert rejects the
    // oversized buffer. The kernel only ever reads up to big_n² floats via
    // `tree_bias[row × block_cols + col]`, so a view is equivalent and keeps
    // the assert semantics meaningful.
    let attn_bias_view = scratch.attn_bias.sub_offset(0, big_n * big_n);
    // Parent-indices sub-view sized to big_n (one i32 per slot; stored as
    // 4 × big_n raw bytes). Only populated when HIPFIRE_DDTREE_TREE_LA=1.
    let parent_view = scratch.parent_indices.sub_offset(0, big_n * 4);
    let ctx = qwen35::TreeVerifyCtx {
        positions: &verify_positions,
        attn_bias: &attn_bias_view,
        parent_indices: if use_tree_la {
            Some(&parent_view)
        } else {
            None
        },
    };
    let t_pre_verify = t_all.elapsed();
    // D8: `want_full_logits` was `temp > 0.0 && !use_swor` (the naive path,
    // gated by HIPFIRE_DDTREE_VERIFY=naive). Since use_swor is now the only
    // temp>0 path (the naive flag and its ~37 MB/cycle D2H have been removed),
    // want_full_logits is permanently false. Greedy: GPU-argmax + tiny D2H.
    // SWOR: no host logits needed; verify_scratch.logits stays device-resident.
    let verify_out = verify_dflash_block_tree(
        gpu,
        target,
        &verify_tokens,
        position,
        hidden_rb,
        Some(gdn_tape),
        false, // want_full_logits: always false; D8 naive path removed
        ctx,
        verify_scratch,
        use_swor, // D9: SWOR uses 68-byte walk result, not argmax_per_pos
    )?;
    let t_post_verify = t_all.elapsed();

    // ── 8. Accept walk: longest accepted path + bonus ─────────────────────
    // temp=0 → greedy argmax walk (follow_verified_tree).
    // temp>0 → q-exploiting Sequoia/SpecTr SWOR walk (use_swor, distribution-
    //   exact). The naive-sampling fallback (HIPFIRE_DDTREE_VERIFY=naive) has
    //   been removed (D8): it required a ~37 MB/cycle full-logits D2H and is
    //   superseded by SWOR which achieves the same distribution-preservation
    //   on-device. All paths return the same (accepted_node_indices, bonus_token)
    //   shape; step 10's divergent-path commit handles non-linear accepted paths.
    let (mut accepted_node_indices, mut bonus_token) = if use_swor {
        // Fully on-device fused SWOR walk: target logits from verify scratch,
        // draft logits kept on device — no full-vocab host work, no q D2H.
        let target_dev = verify_scratch.logits.sub_offset(0, big_n * vocab);
        let draft_dev = draft_logits_dev
            .as_ref()
            .expect("use_swor ⇒ draft kept its device logits");
        let seed = *rng_state;
        xorshift_next_unit(rng_state);
        swor_walk_gpu(
            gpu,
            &tree,
            &target_dev,
            draft_dev,
            &swor_pos_cands,
            b - 1,
            tree_topk,
            vocab,
            temp,
            seed,
        )?
    } else {
        // Greedy (temp=0): follow argmax at each tree node.
        hipfire_runtime::ddtree::follow_verified_tree(&tree, &verify_out.argmax_per_pos)
    };
    // The kept draft logits are no longer needed once the walk has run.
    if let Some(t) = draft_logits_dev {
        let _ = gpu.free_tensor(t);
    }
    // Pre-commit budget: emit = accept_len + 1 ≤ max_emit. Truncate the walk
    // BEFORE tape replay / hidden scatter / DN advance. When the walk went
    // past the budget, the next accepted node token is itself a genuine
    // target draw (greedy match or SWOR sample) — promote it to bonus.
    // Greedy boundary can equivalently re-read argmax_per_pos[slot].
    if let Some(m) = max_accept {
        if accepted_node_indices.len() > m {
            if use_swor {
                bonus_token = tree.nodes[accepted_node_indices[m]].token;
            } else {
                let bonus_slot = if m == 0 {
                    0
                } else {
                    accepted_node_indices[m - 1] + 1
                };
                bonus_token = verify_out.argmax_per_pos[bonus_slot];
            }
            accepted_node_indices.truncate(m);
        }
    }
    let accept_len = accepted_node_indices.len();

    // ── 9. Build committed + drafted sequences ────────────────────────────
    let mut committed: Vec<u32> = Vec::with_capacity(accept_len + 2);
    committed.push(seed_token);
    for &ni in &accepted_node_indices {
        committed.push(tree.nodes[ni].token);
    }
    committed.push(bonus_token);

    let mut drafted: Vec<u32> = Vec::with_capacity(accept_len + 1);
    drafted.push(seed_token);
    for &ni in &accepted_node_indices {
        drafted.push(tree.nodes[ni].token);
    }

    // ── 10. Tape/hidden path selection ────────────────────────────────────
    //
    // Fast path: accepted tree nodes occupy linear slots [0, 1, 2, ...,
    // accept_len - 1] in the tree. Their tokens are in linear-order
    // positions [1, 2, ..., accept_len] of the tape (slot 0 = seed). The
    // tree tape captures innovations at those same linear slots, so
    // `replay_gdn(accept_len + 1)` is exact with DFlash.
    //
    // Fast path is ALWAYS the case at topk=1 (tree is a chain, accepted
    // indices are [0, 1, 2, ...]). At topk>1 it holds iff the greedy walk
    // picked the rank-0 child at every accepted step (no sibling detour).
    //
    // Slow path (topk>1 detour): re-capture tape on the committed prefix
    // with a second verify, then replay as before. Costs +1 forward but
    // keeps LA state byte-correct.
    // HIPFIRE_DDTREE_FORCE_SLOW=1: force the slow (re-verify) path even when
    // committed path == linearization prefix. Diagnostic — quantifies the
    // cost of always re-running the committed tokens through a non-tree
    // verify to fix KV cache entries at committed slots (topk>1 siblings
    // at same depth otherwise race and the LAST write wins regardless of
    // which sibling was committed).
    let force_slow = hipfire_config::developer_var("HIPFIRE_DDTREE_FORCE_SLOW")
        .ok()
        .as_deref()
        == Some("1");
    let spine_accept = accepted_node_indices
        .iter()
        .enumerate()
        .all(|(i, &ni)| ni == i);
    let fast_tape_ok = !force_slow && spine_accept;
    // Per-cycle fast/slow accounting. HIPFIRE_DDTREE_TAPE_DUMP=1 emits a
    // per-cycle line to stderr; useful to quantify how often the slow-path
    // 2nd verify fires at a given topk / workload. Aggregate stats are
    // printed by dflash_spec_demo at end-of-generation via this thread-local.
    thread_local! {
        static DDTREE_FAST_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static DDTREE_SLOW_COUNT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    if fast_tape_ok {
        DDTREE_FAST_COUNT.with(|c| c.set(c.get() + 1));
    } else if !force_slow {
        DDTREE_SLOW_COUNT.with(|c| c.set(c.get() + 1));
    }
    if hipfire_config::developer_var("HIPFIRE_DDTREE_TAPE_DUMP")
        .ok()
        .as_deref()
        == Some("1")
    {
        let fast = DDTREE_FAST_COUNT.with(|c| c.get());
        let slow = DDTREE_SLOW_COUNT.with(|c| c.get());
        eprintln!(
            "[ddtree-tape] cycle: fast_tape_ok={} accept_len={} spine_accept={} tree_la={} (cumulative fast={}/slow={})",
            fast_tape_ok, accept_len, spine_accept, use_tree_la, fast, slow,
        );
    }
    // True when the block's rows sit in committed order (either the accept
    // followed the spine, or the gather below put them there), which is what
    // lets the hidden commit take its GPU-resident fast path.
    let committed_row_order;
    let hidden_rows_written;
    if fast_tape_ok {
        // Tape already captured in tree verify. Restore + replay directly.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = big_n;
        committed_row_order = true;
    } else if !force_slow {
        // Post-accept gather instead of a second forward: the tree verify
        // already produced every accepted row at its own linearization slot,
        // with the correct baked-in RoPE phase (the i-th accepted node has
        // depth i+1, so its phase IS its committed position). Relocating those
        // rows into the committed slots reproduces what the re-verify below
        // would compute, without the ~40-50 ms forward that fired on ~31% of
        // cycles at topk=2.
        ddtree_gather_committed_rows(
            gpu,
            target,
            gdn_tape,
            hidden_rb,
            position,
            big_n,
            &accepted_node_indices,
        )?;
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = big_n;
        committed_row_order = true;
    } else {
        // Slow path (non-spine accept): re-verify the committed prefix to
        // get a linear-order tape AND correctly RoPE'd K written to committed
        // slots. ~40-50 ms cost on 27B.
        let tape_block: Vec<u32> = committed[..accept_len + 1].to_vec();
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        let _tape_verify = verify_dflash_block(
            gpu,
            target,
            &tape_block,
            position,
            hidden_rb,
            Some(gdn_tape),
            false,
            verify_scratch,
        )?;
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = tape_block.len();
        // The re-verify wrote committed-order rows itself.
        committed_row_order = true;
    }

    // ── 11. D2D-scatter committed rows into draft_scratch.target_hidden ─
    // Mirror the chain path (spec_step_dflash ~line 3911): scatter only the
    // accept_len+1 committed rows directly from hidden_rb into the GPU-resident
    // draft_scratch.target_hidden buffer, then call thlog.append_committed so
    // the next cycle's draft_forward sees prev==l and skips the H2D upload.
    // We still download the same rows to CPU to maintain the target_hidden_host
    // length invariant (checked at cycle entry, line 4992-4995).
    //
    // Fast-tape: hidden_rb holds big_n rows in linearization order; the first
    // rows_to_keep are the committed prefix (spine_accept == true guarantees
    // linear order). Pass block_size=big_n so scatter aligns to ring origin.
    //
    // Slow-tape: the 2nd verify (above) wrote exactly accept_len+1 rows in
    // committed order to hidden_rb. Pass block_size=rows_to_keep.
    //
    // Every branch above leaves hidden_rb in committed row order — the accept
    // followed the spine, the post-accept gather relocated the rows, or the
    // force-slow re-verify wrote them directly — so this is unconditionally the
    // GPU-resident scatter. (The old host-gather arm that read
    // `accepted_node_indices` on the CPU is gone: it maintained only
    // `target_hidden_host`, skipping both the GPU scatter and the thlog append,
    // which would leave the drafter with an unwritten row hole.)
    let rows_to_keep = accept_len + 1;
    debug_assert!(
        committed_row_order,
        "hidden_rb rows must be in committed order before the scatter"
    );
    // Fast/gathered: block_size=big_n, n_rows=rows_to_keep (committed prefix).
    // Force-slow: block_size=rows_to_keep=hidden_rows_written (linear order).
    let block_size = hidden_rows_written;
    scatter_hidden_block_to_interleaved(
        gpu,
        hidden_rb,
        &draft_scratch.target_hidden,
        position,
        block_size,
        rows_to_keep,
        draft_scratch.ctx_modulus(),
    )?;
    let co = target.kv_cache.compact_offset as i32;
    draft_scratch
        .thlog
        .append_committed(position, rows_to_keep, co);
    // Stage 1b: GPU buffer (scratch.target_hidden) is now authoritative — the
    // D2D scatter above populated it. No CPU download needed; the next cycle's
    // draft_forward receives None and skips H2D entirely. target_hidden_host is
    // intentionally NOT updated (it's unused in the GPU-resident path).

    if debug_tm {
        let total = t_all.elapsed();
        eprintln!(
            "[ddtree-tm] draft={:.2}ms topk={:.2}ms build={:.2}ms pre_verify={:.2}ms verify={:.2}ms total={:.2}ms  (N={} accept={})",
            t_draft.as_secs_f64() * 1000.0,
            (t_topk - t_draft).as_secs_f64() * 1000.0,
            (t_build - t_topk).as_secs_f64() * 1000.0,
            (t_pre_verify - t_build).as_secs_f64() * 1000.0,
            (t_post_verify - t_pre_verify).as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
            big_n, accept_len,
        );
    }

    Ok(SpecStepResult {
        accepted: accept_len,
        bonus_token,
        drafted,
        committed,
    })
}

/// Seed `target_hidden_host` from the prompt by running the target over
/// each prompt token one at a time with hidden-state extraction enabled.
/// This is a slow but correct MVP path — the target already ran a fast
/// prefill earlier; this exists only to populate `hidden_rb` + host vec
/// with the prompt's layer-selected hidden states.
///
/// Callers with a fast-path prefill that already populates `hidden_rb`
/// should skip this and just call `download_hidden_block(hidden_rb, len)`
/// instead. For MVP we eat the redundant work because it's a one-shot
/// cost at session start.
/// Snapshot the DeltaNet recurrent state into a bounded ring `cks` (pairs of
/// `(seq_pos, snapshot)`) when `interval` tokens have elapsed since the last
/// one. Shared by BOTH the AR `generate` and the DFlash prompt-cache paths to
/// enable resume-from-checkpoint on a divergent client render (see the daemon's
/// `generate` divergence branch + `generate_dflash`). Oldest evicted at `cap`
/// (buffers reused — no realloc churn after warmup). Cheap: one device-to-device
/// memcpy of the recurrent S/scale/conv buffers; no KV copy (FullAttention KV is
/// positional and stays resident, so resume only restores the recurrent state).
/// Gating (resume enabled / no eviction) is the caller's responsibility.
pub fn take_dn_checkpoint(
    cks: &mut Vec<(usize, DeltaNetSnapshot)>,
    dn: &DeltaNetState,
    gpu: &mut Gpu,
    pos: usize,
    interval: usize,
    cap: usize,
) {
    if pos == 0 || cap == 0 {
        return;
    }
    match cks.last().map(|(p, _)| *p) {
        Some(p) if pos < p + interval => return,
        Some(p) if p == pos => return,
        _ => {}
    }
    let mut snap = if cks.len() >= cap {
        cks.remove(0).1
    } else {
        match DeltaNetSnapshot::new_for(gpu, dn) {
            Ok(s) => s,
            Err(_) => return,
        }
    };
    if snap.save_from(dn, gpu).is_err() {
        return;
    }
    cks.push((pos, snap));
}

pub fn seed_target_hidden_from_prompt(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    prompt_tokens: &[u32],
) -> HipResult<()> {
    // Reset target state to avoid double-prefill of the same context.
    target.reset_state(gpu)?;
    // Fast path: one batched prefill populates hidden_rb + KV + dn_state in a
    // single forward, instead of N per-token forwards. On 9B MQ4 with a
    // 6.2k-token prompt this drops prompt ingest from ~51s (121 tok/s) to
    // a few seconds, which is the primary cost of an agent's first turn.
    // `forward_prefill_batch` itself falls back to per-token internally if
    // the KV quant mode / batch size aren't on the fast path, so the call
    // is always safe — the effective cadence just varies.
    qwen35::forward_prefill_batch(
        gpu,
        &target.weights,
        &target.config,
        prompt_tokens,
        0,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        Some(hidden_rb),
        None,
        None,
        None,
    )?;
    // Gather the just-written rows from the ring buffer.
    let block = download_hidden_block(gpu, hidden_rb, prompt_tokens.len())?;
    target_hidden_host.extend_from_slice(&block);
    Ok(())
}

/// Abortable variant of `seed_target_hidden_from_prompt`. Manually
/// chunks the prefill at [`qwen35::PREFILL_MAX_BATCH`] boundaries and
/// calls `abort_check` between chunks. Returns `Ok(true)` if aborted
/// (state has been fully reset — caller should NOT continue with
/// decode), `Ok(false)` on normal completion. The chunked path matches
/// the kernel-internal sub-batch size, so per-chunk throughput is the
/// same as the one-shot variant; the only overhead is one
/// `download_hidden_block` per chunk (host-side memcpy of ~5 MB).
///
/// Used by the daemon's `generate_dflash` to honor client-side
/// cancellation on long-context retries (cache-miss scenarios where
/// the full conversation must be re-prefilled from scratch).
#[allow(clippy::too_many_arguments)]
pub fn seed_target_hidden_from_prompt_abortable(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    hidden_rb: &mut HiddenStateRingBuffer,
    target_hidden_host: &mut Vec<f32>,
    prompt_tokens: &[u32],
    abort_check: &dyn Fn() -> bool,
    // Optional DeltaNet checkpoint ring for divergent-render resume. When
    // `Some`, the recurrent state is snapshotted every `ckpt_interval` tokens
    // (bounded at `ckpt_cap`). `None` ⇒ no checkpointing (zero overhead).
    mut checkpoints: Option<&mut Vec<(usize, DeltaNetSnapshot)>>,
    ckpt_interval: usize,
    ckpt_cap: usize,
) -> HipResult<bool> {
    target.reset_state(gpu)?;
    target_hidden_host.clear();
    if let Some(cks) = checkpoints.as_deref_mut() {
        // fresh cold prefill ⇒ stale checkpoints no longer valid; free their GPU buffers
        for (_, snap) in cks.drain(..) {
            snap.free_gpu(gpu);
        }
    }
    let chunk_max = qwen35::PREFILL_MAX_BATCH;
    let mut seq_pos: usize = 0;
    while seq_pos < prompt_tokens.len() {
        if abort_check() {
            let _ = target.reset_state(gpu);
            target_hidden_host.clear();
            if let Some(cks) = checkpoints.as_deref_mut() {
                for (_, snap) in cks.drain(..) {
                    snap.free_gpu(gpu);
                }
            }
            return Ok(true);
        }
        let end = (seq_pos + chunk_max).min(prompt_tokens.len());
        let chunk = &prompt_tokens[seq_pos..end];
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            chunk,
            seq_pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            Some(hidden_rb),
            None,
            None,
            None,
        )?;
        let block = download_hidden_block(gpu, hidden_rb, chunk.len())?;
        target_hidden_host.extend_from_slice(&block);
        seq_pos = end;
        if let Some(cks) = checkpoints.as_deref_mut() {
            take_dn_checkpoint(cks, &target.dn_state, gpu, seq_pos, ckpt_interval, ckpt_cap);
        }
    }
    Ok(false)
}

/// Incremental prompt seed for the DFlash prompt cache: prefill ONLY the
/// `suffix` tokens starting at absolute position `start_pos`, WITHOUT resetting
/// target KV / DeltaNet state. Used when a turn is a pure extension of the
/// cached conversation (LCP == prior length) — the target KV[0..start_pos] and
/// the recurrent DeltaNet state are already correct from the prior turn, so we
/// only advance them through the new suffix. `hidden_rb` is left holding the
/// suffix's extracted hidden rows so the caller can scatter them into the
/// draft's cumulative `target_hidden` at row offset `start_pos` (the draft's
/// projection cache, keyed on `draft_ctx_cached_rows`, then projects only the
/// new rows — same delta path decode already uses).
///
/// Correctness rests on the same invariant the AR `generate` cache relies on:
/// `forward_prefill_batch` at a nonzero `seq_pos` continues the hybrid
/// (FullAttention KV + DeltaNet recurrent) forward exactly as if the prefix had
/// just been prefilled, because the recurrent state is naturally at the end of
/// the prior conversation (pure extension — no rewind). Returns `Ok(true)` if
/// aborted mid-prefill (state left as-is; caller must full-reset & retry),
/// `Ok(false)` on completion.
#[allow(clippy::too_many_arguments)]
pub fn seed_target_hidden_suffix_abortable(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    hidden_rb: &mut HiddenStateRingBuffer,
    suffix: &[u32],
    start_pos: usize,
    abort_check: &dyn Fn() -> bool,
    // Optional DeltaNet checkpoint ring (see from_prompt variant). Lets a HIT
    // or a resume keep adding checkpoints as the conversation grows, so a later
    // divergence resumes from a recent point rather than the initial prefill.
    mut checkpoints: Option<&mut Vec<(usize, DeltaNetSnapshot)>>,
    ckpt_interval: usize,
    ckpt_cap: usize,
) -> HipResult<bool> {
    let chunk_max = qwen35::PREFILL_MAX_BATCH;
    let mut off: usize = 0;
    let mut pos = start_pos;
    while off < suffix.len() {
        if abort_check() {
            return Ok(true);
        }
        let end = (off + chunk_max).min(suffix.len());
        let chunk = &suffix[off..end];
        qwen35::forward_prefill_batch(
            gpu,
            &target.weights,
            &target.config,
            chunk,
            pos,
            &mut target.kv_cache,
            &mut target.dn_state,
            &target.scratch,
            Some(hidden_rb),
            None,
            None,
            None,
        )?;
        pos += chunk.len();
        off = end;
        if let Some(cks) = checkpoints.as_deref_mut() {
            take_dn_checkpoint(cks, &target.dn_state, gpu, pos, ckpt_interval, ckpt_cap);
        }
    }
    Ok(false)
}

/// Mirror a TriAttention KV eviction into the DFlash draft's GPU-resident
/// `target_hidden` and `target_hidden_abs_positions`, so the draft's cross-
/// attention sees the same subset of context target now has.
///
/// `retain_mask` is the source-position retain selection returned by
/// `EvictionCtx::maybe_evict` (ascending, length == budget). An empty
/// `retain_mask` is a no-op — the caller should have skipped calling this
/// (CASK m-fold path returns empty because merged slots don't map cleanly
/// to a single source position).
///
/// Implementation: download the relevant `physical` rows of `target_hidden`,
/// reorder to `budget` rows on the host per `retain_mask`, upload back. Runs
/// at eviction cadence (~once per β decoded tokens) so the PCIe round-trip
/// is amortized — perf impact is small relative to the τ recovery.
///
/// Post-conditions (all via `draft_scratch.thlog.rebuild_after_eviction`):
/// - `thlog.abs_positions()` has exactly `budget` entries, each pulled from
///   `retain_mask[i]` of the pre-eviction abs_positions.
/// - `draft_scratch.target_hidden` GPU slots [0..budget) hold the retained
///   rows in ascending source order.
/// - `thlog.uploaded_rows() == budget` so the next draft_forward sees the
///   compacted layout as already-uploaded (and the projection cache is
///   invalidated).
pub fn apply_eviction_retain_to_draft(
    gpu: &mut rdna_compute::Gpu,
    draft_scratch: &mut dflash::DflashScratch,
    retain_mask: &[u32],
    ne: usize,
    h: usize,
    physical: usize,
) -> HipResult<()> {
    if retain_mask.is_empty() {
        return Ok(());
    }
    let row_floats = ne * h;
    // Download only the populated prefix of target_hidden. `alloc_tensor`
    // is sized to max_ctx_len — we just need `physical` rows.
    let mut host = vec![0f32; physical * row_floats];
    {
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                host.as_mut_ptr() as *mut u8,
                host.len() * std::mem::size_of::<f32>(),
            )
        };
        gpu.hip
            .memcpy_dtoh(bytes, &draft_scratch.target_hidden.buf)?;
    }
    let budget = retain_mask.len();
    let mut compacted = Vec::with_capacity(budget * row_floats);
    let mut new_abs = Vec::with_capacity(budget);
    for &src_idx in retain_mask {
        let s = src_idx as usize;
        let row = &host[s * row_floats..(s + 1) * row_floats];
        compacted.extend_from_slice(row);
        new_abs.push(
            *draft_scratch
                .thlog
                .abs_positions()
                .get(s)
                .expect("retain_mask index out of range for abs_positions"),
        );
    }
    let dst_bytes = budget * row_floats * std::mem::size_of::<f32>();
    let compacted_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(compacted.as_ptr() as *const u8, dst_bytes) };
    gpu.hip
        .memcpy_htod(&draft_scratch.target_hidden.buf, compacted_bytes)?;
    // Replace the row layout with the compacted positions and invalidate the
    // per-layer k_ctx/v_ctx projection cache (indexed by the pre-eviction row
    // layout, now stale — rebuilt on the next draft_forward; one slow cycle per
    // eviction is fine). `rebuild_after_eviction` sets the upload watermark to
    // `new_abs.len()` (== budget) so the two cursors can't desync.
    draft_scratch.thlog.rebuild_after_eviction(new_abs);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType;

    /// A spine accept needs no row motion — this is what keeps the existing
    /// fast path free of any gather work.
    #[test]
    fn commit_runs_empty_on_spine_accept() {
        assert!(ddtree_commit_row_runs(&[]).is_empty());
        assert!(ddtree_commit_row_runs(&[0]).is_empty());
        assert!(ddtree_commit_row_runs(&[0, 1, 2, 3]).is_empty());
    }

    /// Committed row `i+1` must take linearization slot `accepted[i]+1`, and a
    /// contiguous stretch sharing one delta must coalesce into a single copy —
    /// per-row copies would still be correct but cost one D2D launch each,
    /// across every KV plane and tape buffer.
    #[test]
    fn commit_runs_coalesce_uniform_delta() {
        // Detour at depth 1: slots 2,3,4 carry committed rows 1,2,3 (delta +1).
        assert_eq!(ddtree_commit_row_runs(&[1, 2, 3]), vec![(1, 2, 3)]);
        // Spine prefix stays put; only the tail shifts.
        assert_eq!(ddtree_commit_row_runs(&[0, 1, 3, 4]), vec![(3, 4, 2)]);
    }

    /// Two detours with different deltas cannot share a run.
    #[test]
    fn commit_runs_split_on_delta_change() {
        // rows 1,2 <- slots 2,3 (delta +1); row 3 <- slot 5 (delta +2).
        assert_eq!(
            ddtree_commit_row_runs(&[1, 2, 4]),
            vec![(1, 2, 2), (3, 5, 1)]
        );
    }

    /// The in-place, scratch-free move is only sound because every move reads a
    /// row at or after the row it writes, with runs ordered by ascending
    /// destination. Assert that invariant over every monotonically increasing
    /// accept pattern up to a small bound.
    #[test]
    fn commit_runs_never_write_below_a_later_source() {
        fn check(accepted: &[usize]) {
            let runs = ddtree_commit_row_runs(accepted);
            let mut prev_dst_end = 0usize;
            for &(dst, src, n) in &runs {
                assert!(src >= dst, "src {src} < dst {dst} would need scratch");
                assert!(dst >= prev_dst_end, "runs must ascend by destination");
                prev_dst_end = dst + n;
            }
        }
        // Accepted node indices are strictly increasing (root-to-leaf walk over
        // a depth-ordered linearization), so enumerate those shapes.
        for a in 0..6 {
            for b in (a + 1)..7 {
                for c in (b + 1)..8 {
                    check(&[a, b, c]);
                }
            }
        }
    }

    /// CPU mirror of the GPU `softmax_temp_topp_batched_f32` nucleus phase:
    /// bisect tau over [0, p_max] for the inclusive crossing threshold and
    /// recompute Z = mass(tau) exactly. Used only by the parity test below.
    fn cpu_tau_cut_z(probs: &[f32], top_p: f32) -> (f32, f32) {
        if top_p >= 1.0 {
            return (0.0, 1.0);
        }
        let pmax = probs.iter().cloned().fold(0.0f32, f32::max);
        let mass = |tau: f32| -> f32 { probs.iter().filter(|&&p| p >= tau).sum() };
        let (mut lo, mut hi) = (0.0f32, pmax);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if mass(mid) >= top_p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let tau = lo;
        (tau, mass(tau))
    }

    #[test]
    fn apply_topp_trunc_matches_ar_nucleus_cut() {
        // A row with a clear nucleus boundary (no exact ties): the GPU-style
        // threshold cut must reproduce the AR sort-and-cut nucleus exactly
        // (same kept set, same renormalized values).
        let base = [0.40f32, 0.25, 0.20, 0.10, 0.05];
        for &top_p in &[0.5f32, 0.6, 0.85, 0.95, 0.99] {
            // AR reference (sort desc, cumulative >= top_p inclusive, renorm).
            let mut ar = base.to_vec();
            apply_host_nucleus(&mut ar, top_p);

            // GPU-style: compute tau_cut/Z by bisection, then host truncate.
            let (tau, z) = cpu_tau_cut_z(&base, top_p);
            let mut gpu_style = base.to_vec();
            apply_topp_trunc(&mut gpu_style, tau, z);

            // Same kept set.
            for i in 0..base.len() {
                assert_eq!(
                    ar[i] == 0.0,
                    gpu_style[i] == 0.0,
                    "kept-set mismatch at idx {i}, top_p {top_p}: ar={:?} gpu={:?}",
                    ar,
                    gpu_style
                );
            }
            // Same renormalized values for kept tokens.
            for i in 0..base.len() {
                assert!(
                    (ar[i] - gpu_style[i]).abs() < 1e-5,
                    "value mismatch at idx {i}, top_p {top_p}: ar={} gpu={}",
                    ar[i],
                    gpu_style[i]
                );
            }
            // Renormalized row sums to 1.
            let s: f32 = gpu_style.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row should renorm to 1, got {s}");
        }
    }

    #[test]
    fn apply_topp_trunc_identity_when_disabled() {
        let base = [0.4f32, 0.3, 0.2, 0.1];
        let mut row = base.to_vec();
        // tau_cut == 0.0 (the kernel's top_p>=1.0 guard) → identity.
        apply_topp_trunc(&mut row, 0.0, 1.0);
        assert_eq!(row, base.to_vec());
        // apply_host_nucleus with top_p>=1.0 → identity too.
        let mut row2 = base.to_vec();
        apply_host_nucleus(&mut row2, 1.0);
        assert_eq!(row2, base.to_vec());
    }

    #[test]
    fn apply_topp_trunc_single_token_nucleus() {
        // Very tight top_p keeps only the top token; Z == p_max.
        let base = [0.7f32, 0.2, 0.07, 0.03];
        let (tau, z) = cpu_tau_cut_z(&base, 0.5);
        let mut row = base.to_vec();
        apply_topp_trunc(&mut row, tau, z);
        assert!((row[0] - 1.0).abs() < 1e-5, "single-token nucleus → 1.0");
        assert_eq!(&row[1..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn dflash_gdn_tape_replay_uses_actual_verify_eligibility() {
        assert!(dflash_use_gdn_tape_replay(true, true));
        assert!(!dflash_use_gdn_tape_replay(false, true));
        assert!(!dflash_use_gdn_tape_replay(true, false));
    }

    #[test]
    fn dflash_extended_verify_graph_is_moe_greedy_only() {
        assert!(dflash_moe_verify_graph_lmhead_eligible(
            256, false, false, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            0, false, false, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            256, true, false, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            256, false, true, None
        ));
        assert!(!dflash_moe_verify_graph_lmhead_eligible(
            256,
            false,
            false,
            Some("0")
        ));
    }

    #[test]
    fn dflash_draft_ffn_graph_is_moe_plain_dflash_only() {
        assert!(dflash_moe_draft_ffn_graph_eligible(
            256, false, false, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            0, false, false, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256, true, false, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256, false, true, false, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256, false, false, true, None
        ));
        assert!(!dflash_moe_draft_ffn_graph_eligible(
            256,
            false,
            false,
            false,
            Some("0")
        ));
    }

    #[test]
    fn dflash_verify_graph_env_quarantines_gfx1100_mq_v2() {
        use rdna_compute::DType;

        // gfx1100 + every V2 width: default-off; =0 force-off; =1 diagnostic opt-in.
        for dtype in [
            DType::MQ2G256V2,
            DType::MQ3G256V2,
            DType::MQ4G256V2,
            DType::MQ5G256V2,
            DType::MQ6G256V2,
        ] {
            assert!(!dflash_verify_graph_env_eligible("gfx1100", dtype, None));
            assert!(!dflash_verify_graph_env_eligible(
                "gfx1100",
                dtype,
                Some("0")
            ));
            assert!(dflash_verify_graph_env_eligible(
                "gfx1100",
                dtype,
                Some("1")
            ));
        }

        // gfx1100 + legacy quant remains default-on.
        assert!(dflash_verify_graph_env_eligible(
            "gfx1100",
            DType::MQ3G256,
            None
        ));

        // Other arches keep default-on for V2; =0 still force-off.
        assert!(dflash_verify_graph_env_eligible(
            "gfx1151",
            DType::MQ3G256V2,
            None
        ));
        assert!(!dflash_verify_graph_env_eligible(
            "gfx1151",
            DType::MQ3G256V2,
            Some("0")
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1151",
            DType::MQ3G256V2,
            Some("1")
        ));
        assert!(dflash_verify_graph_env_eligible(
            "gfx1201",
            DType::MQ3G256V2,
            None
        ));
    }

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok()
    }

    /// Tiny hand-built DeltaNetState with one EF residual buffer.
    fn tiny_dn_with_ef(gpu: &mut Gpu) -> DeltaNetState {
        DeltaNetState {
            s_matrices: vec![gpu.zeros(&[8], DType::F32).expect("s")],
            s_scales: vec![gpu.zeros(&[1], DType::F32).expect("scale")],
            conv_states: vec![gpu.zeros(&[4], DType::F32).expect("conv")],
            s_ef_residual: vec![gpu.zeros(&[8], DType::F16).expect("ef")],
            quant: qwen35::StateQuant::Q8,
        }
    }

    fn read_f16_as_f32(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
        // Use logical element count from shape; device buf may be padded.
        let n: usize = t.shape.iter().product();
        let mut bytes = vec![0u8; t.buf.size()];
        gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).expect("dtoh f16");
        (0..n)
            .map(|i| {
                let h = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                hipfire_runtime::llama::f16_to_f32(h)
            })
            .collect()
    }

    fn write_f16_from_f32(gpu: &Gpu, t: &GpuTensor, vals: &[f32]) {
        let n: usize = t.shape.iter().product();
        assert_eq!(vals.len(), n, "logical F16 element count");
        let mut bytes = vec![0u8; t.buf.size()];
        for (i, &v) in vals.iter().enumerate() {
            let h = hipfire_runtime::llama::f32_to_f16(v);
            bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
        }
        gpu.hip.memcpy_htod(&t.buf, &bytes).expect("htod f16");
    }

    /// Production behavior: dirty EF residual, save/restore via DeltaNetSnapshot,
    /// and confirm restore recovers the saved values (not the dirtied ones).
    #[test]
    fn deltanet_snapshot_save_restore_includes_ef_residual() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut state = tiny_dn_with_ef(&mut gpu);
        assert_eq!(state.s_ef_residual.len(), 1);

        let saved_vals = [1.0f32, -0.5, 0.25, 2.0, -1.0, 0.125, 0.75, -0.25];
        write_f16_from_f32(&gpu, &state.s_ef_residual[0], &saved_vals);

        let mut snap = DeltaNetSnapshot::new_for(&mut gpu, &state).expect("snap alloc");
        assert_eq!(
            snap.s_ef_len(),
            state.s_ef_residual.len(),
            "snapshot EF count must track live residual"
        );
        snap.save_from(&state, &mut gpu).expect("save");

        // Dirty live EF after save.
        let dirty = [9.0f32; 8];
        write_f16_from_f32(&gpu, &state.s_ef_residual[0], &dirty);
        let after_dirty = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        assert!(
            after_dirty.iter().all(|&v| (v - 9.0).abs() < 1e-2),
            "precondition: live EF dirtied"
        );

        snap.restore_to(&mut state, &mut gpu).expect("restore");
        let restored = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        for (i, (&got, &want)) in restored.iter().zip(saved_vals.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-2,
                "EF[{i}] restore mismatch: got {got} want {want}"
            );
        }

        snap.free_gpu(&mut gpu);
        state.free_gpu(&mut gpu);
    }

    /// Production behavior: DeltaNetState::reset zeroes a dirtied EF residual
    /// (the path ModelSlot::reset_state / reset_recurrent must use).
    #[test]
    fn deltanet_reset_clears_dirtied_ef_residual() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut state = tiny_dn_with_ef(&mut gpu);
        write_f16_from_f32(&gpu, &state.s_ef_residual[0], &[3.5f32; 8]);
        let before = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        assert!(
            before.iter().any(|&v| v.abs() > 1.0),
            "precondition: EF dirty"
        );

        state.reset(&mut gpu);
        let after = read_f16_as_f32(&gpu, &state.s_ef_residual[0]);
        for (i, &v) in after.iter().enumerate() {
            assert!(v.abs() < 1e-3, "EF[{i}] must be zero after reset, got {v}");
        }
        state.free_gpu(&mut gpu);
    }
}
