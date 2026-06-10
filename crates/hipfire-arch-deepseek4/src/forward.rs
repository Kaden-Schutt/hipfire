// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 forward pass — skeleton.
//!
//! Layout-only: the function signatures and per-layer call sequence
//! are locked in; the bodies are `unimplemented!` until each piece
//! gets wired. Reading this file gives a future implementer (or
//! reviewer) the entire decode-step flow at a glance.
//!
//! The seven GPU-validated kernels referenced here:
//!   - `gpu.hc_compute_control`       (Phase 3)
//!   - `gpu.hc_sinkhorn_4x4`          (Phase 3)
//!   - `gpu.hc_mix_4stream`           (Phase 3)
//!   - `gpu.indexer_compressed_k_score`  (Phase 2)
//!   - `gpu.indexer_top_k`            (Phase 2)
//!   - `gpu.indexer_kv_gather`        (Phase 2)
//!   - `gpu.rope_tail_halfsplit`      (Phase 4)
//!
//! Existing hipfire-runtime kernels reused (no DeepSeek V4-specific impl):
//!   - RMSNorm
//!   - Quantized GEMV (MQ-family) for Q-LoRA, KV, O-LoRA, experts
//!   - Embedding lookup, lm_head matmul, sampler

use crate::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    self, ForwardBindings, OpBinding, OpFlavor, SuperOp, SuperOpKind,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::{DType, Gpu, GpuTensor};

/// OnceLock-cached env-var lookups for the DeepSeek V4 decode hot path. Each
/// `std::env::var` is a syscall (~1μs) — at 43 layers × ~5 lookups per
/// layer in the un-cached code that was ~200μs/token of pure syscall
/// overhead. Each helper reads the env once and atomic-loads thereafter.
mod env_cache {
    use std::sync::OnceLock;

    fn flag_one(name: &'static str) -> bool {
        std::env::var(name).ok().as_deref() == Some("1")
    }

    /// `HIPFIRE_DEEPSEEK4_MOE` — default ON. Opt out with "0" for diagnostic
    /// shared-only-FFN runs. Without routed-expert dispatch the model is
    /// architecturally broken (DeepSeek V4 is MoE), so leaving this off was only
    /// useful during initial bring-up before the forward path consumed the
    /// expert blobs.
    pub(super) fn moe_on() -> bool {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var("HIPFIRE_DEEPSEEK4_MOE").ok().as_deref() != Some("0"))
    }
    /// `HIPFIRE_DEEPSEEK4_SKIP_FFN` — diagnostic: zero ffn_out to isolate attn growth.
    pub(super) fn skip_ffn() -> bool {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| flag_one("HIPFIRE_DEEPSEEK4_SKIP_FFN"))
    }
    /// `HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS` — cap on the compressed-KV scan length.
    pub(super) fn max_compress_pos() -> usize {
        static V: OnceLock<usize> = OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2048)
        })
    }
    /// `HIPFIRE_DEEPSEEK4_ATTN` — when "pos0", attn_stub uses the diagnostic
    /// pos-0 attention path instead of SWA. Default false (i.e. use SWA).
    pub(super) fn attn_pos0() -> bool {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var("HIPFIRE_DEEPSEEK4_ATTN").ok().as_deref() == Some("pos0"))
    }
    /// `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` — default ON since 2026-05-21: route
    /// the MTP output (step 8 of mtp_forward) through head-HC mix using
    /// `mtp.0.hc_head_fn / hc_head_base / hc_head_scale`. Mirrors the
    /// main model's final_norm_and_head head-HC reduction. Without
    /// this, MTP step 8 reads only stream 0 — discarding 75% of HC
    /// signal at the head boundary (same architectural pattern as the
    /// input-side HC fix shipped in 82224ad). Opt out with =0 for
    /// debugging or pre-fix-compat builds.
    pub(super) fn mtp_head_hc_on() -> bool {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| std::env::var("HIPFIRE_DEEPSEEK4_MTP_HEAD_HC").ok().as_deref() != Some("0"))
    }
}

/// DeepSeek V4 GEMV dispatch: switch kernel based on weight dtype.
///
/// - `DType::MQ4G256` (default DeepSeek V4 non-expert quant): consume FWHT-rotated
///   input via `gemv_mq4g256_prerotated`. This is the existing fast path.
/// - `DType::F32` (set by `--non-expert-f16` quantizer flag, F16 source
///   converted to F32 on upload): consume plain RMSNorm'd input (no FWHT)
///   via `gemv_f32`. Used to faithfully reproduce antirez/ds4's PROVEN
///   recipe of keeping compressor / indexer / attn projections at F16
///   precision.
///
/// Caller passes BOTH the FWHT-rotated and plain inputs; helper picks
/// whichever the weight needs. `m` and `k` are passed-through for the
/// MQ4 path only — gemv_f32 derives them from the weight's shape.
/// True if `gemv_auto` for this weight dtype will read the FWHT-rotated
/// input (`x_rotated` arg), false if it only reads the plain input.
/// DeepSeek V4's mq2lloyd-q8 build has F16/Q8 everywhere except the routed
/// MoE experts (which take a separate path) — meaning most decode-path
/// rotations into `ffn_x_rot` / `silu_rot` / `q_lat_rot` / etc. are
/// DEAD WORK (kernel runs, output never read). Use this to skip them.
#[inline]
pub(crate) fn weight_needs_fwht(weight: &GpuTensor) -> bool {
    hipfire_dispatch::types::dtype_needs_rotation(weight.dtype)
}

fn gemv_auto(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Result<(), String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::WeightRef;

    let gemv = hipfire_runtime::llama::gemv_family();
    let ctx = DispatchCtx::new(gpu);
    let x = if weight_needs_fwht(weight) { x_rotated } else { x_plain };
    let wr = WeightRef { buf: weight, dtype: weight.dtype, m, k, row_stride: 0, rotation: None, awq_scale: None };
    gemv.run_auto(&ctx, gpu, &wr, x, y)
        .map_err(|e| format!("gemv dispatch: {e}"))
}

/// Batched twin of `gemv_auto` for Phase B2 chunk forward.
///
/// Same dispatch shape but each call processes `batch_size` inputs against
/// a single weight matrix. Output `y` is row-major `[batch_size, m]` —
/// matches what concatenating `batch_size` sequential gemv_auto outputs
/// would produce.
///
/// Inputs:
///   - `x_rotated_batch`: `[batch_size, k]` FWHT-rotated (consumed by the
///     MQ4 path only)
///   - `x_plain_batch`:   `[batch_size, k]` plain RMSNorm'd (consumed by
///     the F32 and Q8 paths)
///
/// Backed by the existing GEMM-batched kernels:
///   - F32  → `gemm_f32_batched` (M_kernel=batch, N_kernel=output_dim)
///   - Q8_0 → `gemm_q8_0_batched_chunked` (handles batch > 64 via internal
///            sub-batching; same MAX_BATCH=64 as the underlying kernel)
///   - Raw (MQ4G256) → `gemm_hfq4g256` (consumes pre-rotated x)
///
/// At batch_size == 1 each path reduces to the equivalent of one
/// sequential gemv_auto call against the same weight; per-row outputs
/// match within FMA-order ε.
#[allow(dead_code, clippy::too_many_arguments)]
fn gemv_auto_batched(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated_batch: &GpuTensor,
    x_plain_batch: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch_size: usize,
) -> Result<(), String> {
    gemv_auto_batched_wmma(
        gpu,
        weight,
        x_rotated_batch,
        x_plain_batch,
        y,
        m,
        k,
        batch_size,
        /*x_f16_scratch=*/ None,
    )
}

/// `gemv_auto_batched` plus an opt-in WMMA path. When `x_f16_scratch`
/// Determinism-bisection helper: when `HIPFIRE_DEEPSEEK4_DUMP_STATE=<dir>` is
/// set, this writes the entire device buffer to `<dir>/<tag>.bin` after a
/// device-sync. Two same-seed runs with different output dirs can then be
/// compared with `cmp -l a/x.bin b/x.bin` to find the first byte that
/// differs — pinpointing which kernel introduces non-determinism.
fn dump_buf(gpu: &mut Gpu, tag: &str, buf: &rdna_compute::GpuTensor) {
    let dir = match std::env::var("HIPFIRE_DEEPSEEK4_DUMP_STATE") {
        Ok(d) => d,
        Err(_) => return,
    };
    let _ = gpu.hip.device_synchronize();
    let n = buf.byte_size();
    let mut host = vec![0u8; n];
    if gpu.hip.memcpy_dtoh(&mut host, &buf.buf).is_ok() {
        let path = format!("{dir}/{tag}.bin");
        if let Err(e) = std::fs::write(&path, &host) {
            eprintln!("[dump_buf] write {path}: {e}");
        }
    }
}

fn gemv_auto_batched_wmma(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated_batch: &GpuTensor,
    x_plain_batch: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch_size: usize,
    x_f16_scratch: Option<&GpuTensor>,
) -> Result<(), String> {
    match weight.dtype {
        DType::F32 => {
            if std::env::var("HIPFIRE_DEEPSEEK4_F32_TRACE").is_ok() {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static N: AtomicUsize = AtomicUsize::new(0);
                let c = N.fetch_add(1, Ordering::Relaxed);
                if c < 8 {
                    eprintln!(
                        "[F32_TRACE #{c}] m={m} k={k} B={batch_size} weight.shape={:?}",
                        weight.shape
                    );
                }
            }
            gpu.gemm_f32_register_tiled(weight, x_plain_batch, y, m, k, batch_size)
                .map_err(|e| format!("gemm_f32_register_tiled: {e:?}"))
        }
        DType::Q8_0 => {
            let wmma_on = std::env::var("HIPFIRE_DEEPSEEK4_Q8_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if wmma_on && gpu.arch_caps.is_rdna4() {
                // RDNA4 (gfx12): upstream-tuned gating (unchanged).
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    let opt_out = std::env::var("HIPFIRE_DEEPSEEK4_Q8_4W")
                        .as_deref() == Ok("0");
                    let use_4w = !opt_out
                        && batch_size >= 256
                        && m >= 4096
                        && m % 64 == 0
                        && k % 32 == 0
                        && batch_size % 64 == 0;
                    if use_4w {
                        return gpu
                            .gemm_q8_0_wmma_4w(weight, scratch, y, m, k, batch_size)
                            .map_err(|e| format!("gemm_q8_0_wmma_4w: {e:?}"));
                    }
                    return gpu
                        .gemm_q8_0_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_q8_0_wmma: {e:?}"));
                }
            } else if wmma_on && gpu.arch_caps.has_wmma() && m % 64 == 0 && k % 32 == 0 {
                // gfx11 / RDNA3.5 (gfx1151) Q8_0 WMMA prefill. The activation
                // is pre-converted to F16 in `scratch`; the kernels honor the
                // F16 dtype (no re-convert). 4-warp 64×64-tile kernel for
                // batch%64==0 (~12% over single-warp 16×16; weight-bandwidth-
                // bound); HIPFIRE_DEEPSEEK4_Q8_4W=0 forces single-warp.
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    let opt_out_4w =
                        std::env::var("HIPFIRE_DEEPSEEK4_Q8_4W").as_deref() == Ok("0");
                    if !opt_out_4w && batch_size >= 64 && batch_size % 64 == 0 {
                        return gpu
                            .gemm_q8_0_wmma_4w(weight, scratch, y, m, k, batch_size)
                            .map_err(|e| format!("gemm_q8_0_wmma_4w: {e:?}"));
                    }
                    return gpu
                        .gemm_q8_0_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_q8_0_wmma: {e:?}"));
                }
            }
            gpu.gemm_q8_0_batched_chunked(weight, x_plain_batch, y, m, k, batch_size)
                .map_err(|e| format!("gemm_q8_0_batched_chunked: {e:?}"))
        }
        DType::F16 => {
            // gfx12/RDNA4: route through the VALIDATED gfx12 f16 WMMA kernel
            // `gemm_f16_wmma_mb8` (takes F32 X directly, has a known-good
            // `_gfx12` port) rather than `gemm_f16_x_f16_wmma`'s gfx12 port.
            // Same math (Y[b,m]=Σ_k W[m,k]·X[b,k], f16 WMMA). On gfx11 keep the
            // original f16×f16 path (caller-converted X scratch).
            if gpu.arch_caps.has_wmma_w32_gfx12() {
                return gpu
                    .gemm_f16_wmma_mb8(weight, x_plain_batch, y, m, k, batch_size)
                    .map_err(|e| format!("gemm_f16_wmma_mb8 (gfx12 f16): {e:?}"));
            }
            if let Some(scratch) = x_f16_scratch {
                let n = (batch_size * k) as i64;
                gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                    .map_err(|e| format!("convert_f32_to_f16 (F16 weight): {e:?}"))?;
                gpu.gemm_f16_x_f16_wmma(weight, scratch, y, m, k, batch_size)
                    .map_err(|e| format!("gemm_f16_x_f16_wmma: {e:?}"))
            } else {
                Err("F16 weight requires WMMA path with x_f16_scratch".to_string())
            }
        }
        _ => {
            let wmma_on = std::env::var("HIPFIRE_DEEPSEEK4_HFQ4_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if wmma_on {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_rotated_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (HFQ4 WMMA): {e:?}"))?;
                    return gpu
                        .gemm_hfq4g256_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_hfq4g256_wmma: {e:?}"));
                }
            }
            gpu.gemm_hfq4g256(weight, x_rotated_batch, y, m, k, batch_size)
                .map_err(|e| format!("gemm_hfq4g256: {e:?}"))
        }
    }
}

/// DeepSeek V4 Compressor decode step (phase 3b scaffold — not yet wired).
///
/// Implements the upstream `Compressor.forward` decode case
/// (start_pos != 0):
///
///   kv = wkv @ x_rotated     [coff * head_dim]
///   score = wgate @ x_rotated [coff * head_dim]
///   score += ape[pos % ratio]
///   kv_state[ratio + pos%ratio]    = kv     (overlap=true)
///   score_state[ratio + pos%ratio] = score
///   if (pos+1) % ratio == 0:
///     overlap_concat → [2*ratio, head_dim]  for kv and score
///     softmax_pool   → [head_dim] compressed
///     rmsnorm (compressor.norm)
///     if is_indexer: tail RoPE (compress_rope_theta = 160000)
///     kv_cache[pos // ratio] = compressed
///     shift kv_state[:ratio] = kv_state[ratio:]  (and score_state)
///
/// Parameterized by `is_indexer`:
///   - false → main attn compressor; head_dim = cfg.head_dim = 512;
///     no RoPE on output; targets `state._indexer[l].main_*`
///   - true  → indexer's sub-compressor; head_dim = idx_head_dim = 128;
///     applies tail RoPE with cfg.compress_rope_theta;
///     targets `state._indexer[l].indexer_*`
///
/// TODO: implement (kernels ready: compressor_softmax_pool_f32 +
/// compressor_overlap_concat_f32). See `docs/plans/deepseek4-next-
/// session.md` for the precise step-by-step.
#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    x_rotated: &GpuTensor,
    position: u32,
    is_indexer: bool,
) -> Result<(), String> {
    compressor_forward_impl(
        cfg, weights, state, gpu, layer_idx, x_rotated, position, is_indexer,
        /*pre_batched=*/ None,
    )
}

/// Variant of `compressor_forward` that uses pre-batched wkv/wgate
/// outputs computed once per (layer, compressor) for all B positions
/// in a chunk. Skips the per-position GEMVs entirely; the caller is
/// responsible for running gemv_auto_batched on the full tmp/tmp_plain
/// batch and providing the resulting (kv, score) buffers with a
/// per-position offset into the [B, proj_dim] view.
#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward_prebatched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
    is_indexer: bool,
    kv_batch: &GpuTensor,
    score_batch: &GpuTensor,
    batch_offset: usize,
) -> Result<(), String> {
    let null_x = state
        .tmp
        .as_ref()
        .ok_or_else(|| format!("compressor_forward_prebatched: state.tmp missing l{layer_idx}"))?
        .sub_offset(0, cfg.hidden_size);
    compressor_forward_impl(
        cfg,
        weights,
        state,
        gpu,
        layer_idx,
        &null_x,
        position,
        is_indexer,
        Some((kv_batch, score_batch, batch_offset)),
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward_impl(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    x_rotated: &GpuTensor,
    position: u32,
    is_indexer: bool,
    pre_batched: Option<(&GpuTensor, &GpuTensor, usize)>,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    if ratio == 0 {
        return Ok(());
    }
    if is_indexer && ratio != 4 {
        return Ok(());
    }

    let overlap = ratio == 4;
    let coff: usize = if overlap { 2 } else { 1 };
    let head_dim = if is_indexer {
        cfg.index_head_dim
    } else {
        cfg.head_dim
    };
    let proj_dim = coff * head_dim;
    let state_rows = coff * ratio; // 8 for ratio=4 overlap, 128 for ratio=128

    // Pick weights based on which compressor (main vs indexer).
    let (wkv, wgate, norm, ape) = if is_indexer {
        (
            layer
                .indexer_compressor_wkv
                .as_ref()
                .ok_or_else(|| format!("idx_comp_wkv l{layer_idx}"))?,
            layer
                .indexer_compressor_wgate
                .as_ref()
                .ok_or_else(|| format!("idx_comp_wgate l{layer_idx}"))?,
            layer
                .indexer_compressor_norm
                .as_ref()
                .ok_or_else(|| format!("idx_comp_norm l{layer_idx}"))?,
            layer
                .indexer_compressor_ape
                .as_ref()
                .ok_or_else(|| format!("idx_comp_ape l{layer_idx}"))?,
        )
    } else {
        (
            layer
                .compressor_wkv
                .as_ref()
                .ok_or_else(|| format!("comp_wkv l{layer_idx}"))?,
            layer
                .compressor_wgate
                .as_ref()
                .ok_or_else(|| format!("comp_wgate l{layer_idx}"))?,
            layer
                .compressor_norm
                .as_ref()
                .ok_or_else(|| format!("comp_norm l{layer_idx}"))?,
            layer
                .compressor_ape
                .as_ref()
                .ok_or_else(|| format!("comp_ape l{layer_idx}"))?,
        )
    };

    let max_compressed: usize = std::env::var("HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    // Lazy-allocate state buffers per (layer, compressor-type).
    {
        let l_state = &mut state._indexer[layer_idx];
        if is_indexer {
            if l_state.indexer_kv_state.is_none() {
                l_state.indexer_kv_state = Some(
                    gpu.zeros(&[state_rows, proj_dim], DType::F32)
                        .map_err(|e| format!("alloc idx kv_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.indexer_score_state.is_none() {
                l_state.indexer_score_state = Some(
                    // -inf init: unfilled pool slots (e.g. block 0's missing
                    // overlap prev-window) must get zero softmax weight, per the
                    // reference `score_state = torch.full(-inf)`.
                    gpu.full_f32(&[state_rows, proj_dim], f32::NEG_INFINITY)
                        .map_err(|e| format!("alloc idx score_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.indexer_kv_cache.is_none() {
                l_state.indexer_kv_cache = Some(
                    gpu.zeros(&[max_compressed, head_dim], DType::F32)
                        .map_err(|e| format!("alloc idx kv_cache l{layer_idx}: {e:?}"))?,
                );
            }
        } else {
            if l_state.main_kv_state.is_none() {
                l_state.main_kv_state = Some(
                    gpu.zeros(&[state_rows, proj_dim], DType::F32)
                        .map_err(|e| format!("alloc main kv_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.main_score_state.is_none() {
                l_state.main_score_state = Some(
                    // -inf init (reference `score_state = torch.full(-inf)`):
                    // unfilled overlap slots get zero softmax weight.
                    gpu.full_f32(&[state_rows, proj_dim], f32::NEG_INFINITY)
                        .map_err(|e| format!("alloc main score_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.main_kv_cache.is_none() {
                l_state.main_kv_cache = Some(
                    gpu.zeros(&[max_compressed, head_dim], DType::F32)
                        .map_err(|e| format!("alloc main kv_cache l{layer_idx}: {e:?}"))?,
                );
            }
        }
    }

    // Per-step scratch — lazy-alloc on layer's IndexerLayerState.
    {
        let l_state = &mut state._indexer[layer_idx];
        if l_state.comp_kv_buf.is_none() {
            l_state.comp_kv_buf = Some(
                gpu.alloc_tensor(&[proj_dim], DType::F32)
                    .map_err(|e| format!("alloc comp_kv_buf l{layer_idx}: {e:?}"))?,
            );
        }
        if l_state.comp_score_buf.is_none() {
            l_state.comp_score_buf = Some(
                gpu.alloc_tensor(&[proj_dim], DType::F32)
                    .map_err(|e| format!("alloc comp_score_buf l{layer_idx}: {e:?}"))?,
            );
        }
        if overlap && l_state.comp_concat_kv.is_none() {
            l_state.comp_concat_kv = Some(
                gpu.alloc_tensor(&[2 * ratio, head_dim], DType::F32)
                    .map_err(|e| format!("alloc comp_concat_kv l{layer_idx}: {e:?}"))?,
            );
        }
        if overlap && l_state.comp_concat_score.is_none() {
            l_state.comp_concat_score = Some(
                gpu.alloc_tensor(&[2 * ratio, head_dim], DType::F32)
                    .map_err(|e| format!("alloc comp_concat_score l{layer_idx}: {e:?}"))?,
            );
        }
    }

    let hidden = cfg.hidden_size;
    let pos = position as usize;
    let slot = if overlap {
        ratio + pos % ratio
    } else {
        pos % ratio
    };

    // 1. kv = wkv @ x_rotated; score = wgate @ x_rotated
    //    Dispatch: MQ4 path uses x_rotated (FWHT'd); F16 path uses
    //    tmp_plain (plain RMSNorm, no FWHT — see q_lora step 1b).
    // If pre_batched is Some, the caller has already run the GEMVs
    // for all B positions; we just point kv/score at the b-th slice.
    let owned_kv_buf;
    let owned_score_buf;
    let (kv_buf, score_buf) = if let Some((kv_b, score_b, b_off)) = pre_batched {
        owned_kv_buf = kv_b.sub_offset(b_off * proj_dim, proj_dim);
        owned_score_buf = score_b.sub_offset(b_off * proj_dim, proj_dim);
        (&owned_kv_buf, &owned_score_buf)
    } else {
        let kvb = state._indexer[layer_idx].comp_kv_buf.as_ref().unwrap();
        let scb = state._indexer[layer_idx].comp_score_buf.as_ref().unwrap();
        let tmp_plain = state.tmp_plain.as_ref().ok_or_else(|| {
            format!("comp l{layer_idx}: tmp_plain missing (q_lora must run first)")
        })?;
        gemv_auto(gpu, wkv, x_rotated, tmp_plain, kvb, proj_dim, hidden)?;
        gemv_auto(gpu, wgate, x_rotated, tmp_plain, scb, proj_dim, hidden)?;
        (kvb, scb)
    };

    // 2. score += ape[pos % ratio]
    //
    // APE (Absolute Position Encoding) is stored as F32 on device after
    // `upload_global_f16_as_f32` at load time. Shape: [ratio, proj_dim].
    // The row at index `pos % ratio` is the positional bias for this
    // slot within the current compression window. Adding it to `score`
    // BEFORE the softmax-pool is what lets the pool distinguish slot 0
    // from slot 1/.../ratio-1 — without it the pool is content-only and
    // distant-token recall degrades to fuzzy paraphrasing
    // (`mariozechner` → `marioze`, `v20.19.6` → `v20.19.20`, etc.). This
    // was a known TODO that has now landed.
    // DIAGNOSTIC: disabled while debugging illegal-memory-access crash.
    // The APE load to F32 stays — only the add is gated.
    // The per-layer scratch buffers (comp_kv_buf / comp_score_buf) are
    // lazy-alloced at the *first* call's proj_dim. For ratio=4 layers we
    // call compressor_forward twice per layer (main then indexer), and
    // the indexer's proj_dim is smaller — but the score buffer already
    // exists at the main proj_dim. `score_buf.numel()` therefore over-
    // states the live length; we must clamp to `proj_dim` (the GEMV
    // write length) so add_inplace_f32 doesn't run past the ape row.
    let ape_row_idx = pos % ratio;
    let ape_row = ape.sub_offset(ape_row_idx * proj_dim, proj_dim);
    let score_view = score_buf.sub_offset(0, proj_dim);
    gpu.add_inplace_f32(&score_view, &ape_row)
        .map_err(|e| format!("comp ape add l{layer_idx}: {e:?}"))?;

    // Stage-bisect dump: HIPFIRE_COMP_DUMP="<pos>,<layer>" prints each
    // pipeline stage's output fingerprint at that (position, layer) so the
    // first cross-arch divergent op can be identified. Diagnostic only.
    let comp_dump_here = std::env::var("HIPFIRE_COMP_DUMP")
        .ok()
        .and_then(|s| {
            let mut it = s.split(',');
            let p: u32 = it.next()?.trim().parse().ok()?;
            let l: usize = it.next()?.trim().parse().ok()?;
            Some((p, l))
        })
        .map(|(p, l)| p == position && l == layer_idx)
        .unwrap_or(false);
    let comp_dbg = |gpu: &Gpu, name: &str, t: &GpuTensor, n: usize| {
        if !comp_dump_here {
            return;
        }
        let _ = gpu.hip.device_synchronize();
        if let Ok(v) = gpu.download_f32(t) {
            let l2: f64 = v.iter().take(n).map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            let head: Vec<String> = v.iter().take(6).map(|x| format!("{x:.6e}")).collect();
            eprintln!(
                "COMPDUMP l{layer_idx} pos={position} idx={is_indexer} {name}: l2={l2:.9e} head={}",
                head.join(",")
            );
        }
    };
    comp_dbg(&*gpu, "kv_buf(gemv)", kv_buf, proj_dim);
    comp_dbg(&*gpu, "score_buf(gemv+ape)", score_buf, proj_dim);

    // Compressor commit + compress pipeline.
    //
    // Two paths share the same dataflow but differ in how slot indices
    // reach the kernels:
    //
    // - `pre_batched=Some` (prefill batched per-position fallback):
    //   slot indices are baked into memcpy_dtod_auto offsets host-side.
    //   `compressed_slot >= max_compressed` and `(pos+1) % ratio != 0`
    //   short-circuit via host-side return.
    //
    // - `pre_batched=None` (decode, captured under HIP graphs):
    //   slot indices are read from `state.attn_state_buf`. ring_slot lives
    //   at offset 6 (ratio=4) or 8 (ratio=128); commit_slot at offset 7
    //   (ratio=4) or 9 (ratio=128). The buf-variant kernels early-return
    //   on commit_slot < 0, so the captured graph can include every
    //   commit kernel at every replay and they no-op on non-commit
    //   positions.
    let l_state = &state._indexer[layer_idx];
    let kv_state = if is_indexer {
        l_state.indexer_kv_state.as_ref().unwrap()
    } else {
        l_state.main_kv_state.as_ref().unwrap()
    };
    let score_state = if is_indexer {
        l_state.indexer_score_state.as_ref().unwrap()
    } else {
        l_state.main_score_state.as_ref().unwrap()
    };
    let kv_cache = if is_indexer {
        l_state.indexer_kv_cache.as_ref().unwrap()
    } else {
        l_state.main_kv_cache.as_ref().unwrap()
    };

    // Per-layer compressor rope pos comes from the pre-computed pos_array.
    // Slot 1 = main_comp_rope_pos, slot 2 = indexer_comp_rope_pos.
    let rope_pos_slot = if is_indexer { 2 } else { 1 };
    let pos_buf = pos_slot(state, layer_idx, rope_pos_slot)?;

    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) = if is_indexer {
        (
            cfg.compress_rope_theta,
            1.0_f32,
            0.0_f32,
            1.0_f32,
            0.0_f32,
            0.0_f32,
        )
    } else {
        layer_rope_params(cfg, layer.compress_ratio)
    };
    let do_rope = true;

    // Capture attn_state_buf slot views BEFORE borrowing l_state — we
    // need a non-overlapping immutable borrow of state.
    let attn_buf_view = if pre_batched.is_some() {
        None
    } else {
        let attn_buf = state.attn_state_buf.as_ref().ok_or_else(|| {
            format!(
                "comp l{layer_idx}: attn_state_buf missing (precompute_attn_state must run first)"
            )
        })?;
        let (ring_off, commit_off) = if ratio == 4 {
            (6usize, 7usize)
        } else {
            (8usize, 9usize)
        };
        Some((
            attn_buf.sub_offset(ring_off, 1),
            attn_buf.sub_offset(commit_off, 1),
        ))
    };

    if pre_batched.is_some() {
        // ---- Prebatched prefill path (host-side gating, memcpy ring writes) ----
        let kv_dst = kv_state.sub_offset(slot * proj_dim, proj_dim);
        let score_dst = score_state.sub_offset(slot * proj_dim, proj_dim);
        gpu.memcpy_dtod_auto(&kv_dst.buf, &kv_buf.buf, proj_dim * 4)
            .map_err(|e| format!("comp kv-store l{layer_idx}: {e:?}"))?;
        gpu.memcpy_dtod_auto(&score_dst.buf, &score_buf.buf, proj_dim * 4)
            .map_err(|e| format!("comp score-store l{layer_idx}: {e:?}"))?;

        let should_compress = (pos + 1).is_multiple_of(ratio);
        if !should_compress {
            return Ok(());
        }
        let compressed_slot = pos / ratio;
        if compressed_slot >= max_compressed {
            return Ok(());
        }
        let kv_cache_slot = kv_cache.sub_offset(compressed_slot * head_dim, head_dim);

        if overlap {
            let concat_kv = l_state.comp_concat_kv.as_ref().unwrap();
            let concat_score = l_state.comp_concat_score.as_ref().unwrap();
            gpu.compressor_overlap_concat_f32(kv_state, concat_kv, ratio as i32, head_dim as i32)
                .map_err(|e| format!("comp concat_kv l{layer_idx}: {e:?}"))?;
            gpu.compressor_overlap_concat_f32(
                score_state,
                concat_score,
                ratio as i32,
                head_dim as i32,
            )
            .map_err(|e| format!("comp concat_score l{layer_idx}: {e:?}"))?;
            gpu.compressor_softmax_pool_f32(
                concat_kv,
                concat_score,
                &kv_cache_slot,
                (2 * ratio) as i32,
                head_dim as i32,
            )
            .map_err(|e| format!("comp pool l{layer_idx}: {e:?}"))?;
        } else {
            gpu.compressor_softmax_pool_f32(
                kv_state,
                score_state,
                &kv_cache_slot,
                ratio as i32,
                head_dim as i32,
            )
            .map_err(|e| format!("comp pool no-overlap l{layer_idx}: {e:?}"))?;
        }
        gpu.rmsnorm_f32(&kv_cache_slot, norm, &kv_cache_slot, cfg.rms_norm_eps)
            .map_err(|e| format!("comp rmsnorm l{layer_idx}: {e:?}"))?;
        if do_rope {
            if is_indexer {
                gpu.rope_tail_interleaved(
                    &kv_cache_slot,
                    &kv_cache_slot,
                    &pos_buf,
                    1,
                    0,
                    head_dim as i32,
                    cfg.qk_rope_head_dim as i32,
                    cfg.compress_rope_theta,
                )
                .map_err(|e| format!("comp rope l{layer_idx}: {e:?}"))?;
            } else {
                gpu.rope_tail_yarn_interleaved(
                    &kv_cache_slot,
                    &kv_cache_slot,
                    &pos_buf,
                    1,
                    0,
                    head_dim as i32,
                    cfg.qk_rope_head_dim as i32,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    corr_low,
                    corr_high,
                    /*inverse=*/ 0,
                )
                .map_err(|e| format!("comp main rope l{layer_idx}: {e:?}"))?;
            }
        }
        if overlap {
            let shift_bytes = ratio * proj_dim * 4;
            let src_view = kv_state.sub_offset(ratio * proj_dim, ratio * proj_dim);
            let dst_view = kv_state.sub_offset(0, ratio * proj_dim);
            gpu.memcpy_dtod_auto(&dst_view.buf, &src_view.buf, shift_bytes)
                .map_err(|e| format!("comp kv_state shift l{layer_idx}: {e:?}"))?;
            let src_view = score_state.sub_offset(ratio * proj_dim, ratio * proj_dim);
            let dst_view = score_state.sub_offset(0, ratio * proj_dim);
            gpu.memcpy_dtod_auto(&dst_view.buf, &src_view.buf, shift_bytes)
                .map_err(|e| format!("comp score_state shift l{layer_idx}: {e:?}"))?;
        }
        return Ok(());
    }

    // ---- Decode / graph-captured path (state-buffer-driven slots) ----
    let (ring_slot_buf, commit_slot_buf) =
        attn_buf_view.expect("attn_buf_view populated when !pre_batched.is_some()");

    // Ring write — unconditional within graph, no-op on -1 sentinel.
    gpu.state_ring_write_f32_buf(kv_buf, kv_state, &ring_slot_buf, proj_dim as i32)
        .map_err(|e| format!("comp ring write kv l{layer_idx}: {e:?}"))?;
    gpu.state_ring_write_f32_buf(score_buf, score_state, &ring_slot_buf, proj_dim as i32)
        .map_err(|e| format!("comp ring write score l{layer_idx}: {e:?}"))?;
    comp_dbg(&*gpu, "kv_state(ring)", kv_state, state_rows * proj_dim);
    comp_dbg(&*gpu, "score_state(ring)", score_state, state_rows * proj_dim);

    // Compress event — concat (overlap only) is unconditional within graph;
    // pool/rmsnorm/rope/shift all sentinel-gate on commit_slot_buf.
    if overlap {
        let concat_kv = l_state.comp_concat_kv.as_ref().unwrap();
        let concat_score = l_state.comp_concat_score.as_ref().unwrap();
        gpu.compressor_overlap_concat_f32(kv_state, concat_kv, ratio as i32, head_dim as i32)
            .map_err(|e| format!("comp concat_kv l{layer_idx}: {e:?}"))?;
        gpu.compressor_overlap_concat_f32(score_state, concat_score, ratio as i32, head_dim as i32)
            .map_err(|e| format!("comp concat_score l{layer_idx}: {e:?}"))?;
        comp_dbg(&*gpu, "concat_kv", concat_kv, 2 * ratio * head_dim);
        comp_dbg(&*gpu, "concat_score", concat_score, 2 * ratio * head_dim);
        gpu.compressor_softmax_pool_f32_buf(
            concat_kv,
            concat_score,
            kv_cache,
            &commit_slot_buf,
            (2 * ratio) as i32,
            head_dim as i32,
        )
        .map_err(|e| format!("comp pool buf l{layer_idx}: {e:?}"))?;
    } else {
        gpu.compressor_softmax_pool_f32_buf(
            kv_state,
            score_state,
            kv_cache,
            &commit_slot_buf,
            ratio as i32,
            head_dim as i32,
        )
        .map_err(|e| format!("comp pool buf no-overlap l{layer_idx}: {e:?}"))?;
    }
    let commit_row = kv_cache.sub_offset((pos / ratio) * head_dim, head_dim);
    comp_dbg(&*gpu, "kv_cache(pool)", &commit_row, head_dim);
    gpu.rmsnorm_f32_at_slot_buf(
        kv_cache,
        norm,
        &commit_slot_buf,
        head_dim as i32,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("comp rmsnorm buf l{layer_idx}: {e:?}"))?;
    comp_dbg(&*gpu, "kv_cache(rmsnorm)", &commit_row, head_dim);
    if do_rope {
        gpu.rope_tail_yarn_interleaved_at_slot_buf(
            kv_cache,
            &pos_buf,
            &commit_slot_buf,
            head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            corr_low,
            corr_high,
        )
        .map_err(|e| format!("comp rope buf l{layer_idx}: {e:?}"))?;
    }
    comp_dbg(&*gpu, "kv_cache(rope)", &commit_row, head_dim);
    if overlap {
        gpu.state_overlap_shift_f32_buf(kv_state, &commit_slot_buf, ratio as i32, proj_dim as i32)
            .map_err(|e| format!("comp kv_state shift buf l{layer_idx}: {e:?}"))?;
        gpu.state_overlap_shift_f32_buf(
            score_state,
            &commit_slot_buf,
            ratio as i32,
            proj_dim as i32,
        )
        .map_err(|e| format!("comp score_state shift buf l{layer_idx}: {e:?}"))?;
    }

    let _ = (position, max_compressed); // consumed via attn_state_buf
    Ok(())
}

/// Batched compressor commit + compress for a whole chunk of B
/// positions in a single layer (Phase A, 2026-05-20).
///
/// Replaces the per-batch-position loop of `compressor_forward_prebatched`
/// when `start_pos % R == 0` (aligned chunks). For ratio=4 layers
/// at B=64, this collapses ~256 launches/layer (64 × 2 ring writes
/// + 16 × 6 compress-event kernels) into ~6 batched launches:
///   - 1× compressor_compress_aligned_batched_f32
///   - 1× rmsnorm_batched (on N_events × head_dim)
///   - 1× rope_tail_(yarn_)interleaved_batched
///   - 1× memcpy_dtod to update ring state for next chunk
///
/// For the no-event case (B < R, e.g. ratio=128 layers at B=64):
///   - 1× compressor_ring_write_batched_f32
///
/// Bisect at B=1 must remain byte-eq vs the per-position path.
#[allow(dead_code, clippy::too_many_arguments)]
fn compressor_forward_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    start_pos: u32,
    batch_size: usize,
    is_indexer: bool,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    if ratio == 0 {
        return Ok(());
    }
    if is_indexer && ratio != 4 {
        return Ok(());
    }

    let overlap = ratio == 4;
    let coff: usize = if overlap { 2 } else { 1 };
    let head_dim = if is_indexer {
        cfg.index_head_dim
    } else {
        cfg.head_dim
    };
    let proj_dim = coff * head_dim;
    let state_rows = coff * ratio;

    let max_compressed: usize = std::env::var("HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    // Lazy-alloc state buffers (mirror compressor_forward_impl exactly).
    {
        let l_state = &mut state._indexer[layer_idx];
        if is_indexer {
            if l_state.indexer_kv_state.is_none() {
                l_state.indexer_kv_state = Some(
                    gpu.zeros(&[state_rows, proj_dim], DType::F32)
                        .map_err(|e| format!("alloc idx kv_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.indexer_score_state.is_none() {
                l_state.indexer_score_state = Some(
                    // -inf init: unfilled pool slots (e.g. block 0's missing
                    // overlap prev-window) must get zero softmax weight, per the
                    // reference `score_state = torch.full(-inf)`.
                    gpu.full_f32(&[state_rows, proj_dim], f32::NEG_INFINITY)
                        .map_err(|e| format!("alloc idx score_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.indexer_kv_cache.is_none() {
                l_state.indexer_kv_cache = Some(
                    gpu.zeros(&[max_compressed, head_dim], DType::F32)
                        .map_err(|e| format!("alloc idx kv_cache l{layer_idx}: {e:?}"))?,
                );
            }
        } else {
            if l_state.main_kv_state.is_none() {
                l_state.main_kv_state = Some(
                    gpu.zeros(&[state_rows, proj_dim], DType::F32)
                        .map_err(|e| format!("alloc main kv_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.main_score_state.is_none() {
                l_state.main_score_state = Some(
                    // -inf init (reference `score_state = torch.full(-inf)`):
                    // unfilled overlap slots get zero softmax weight.
                    gpu.full_f32(&[state_rows, proj_dim], f32::NEG_INFINITY)
                        .map_err(|e| format!("alloc main score_state l{layer_idx}: {e:?}"))?,
                );
            }
            if l_state.main_kv_cache.is_none() {
                l_state.main_kv_cache = Some(
                    gpu.zeros(&[max_compressed, head_dim], DType::F32)
                        .map_err(|e| format!("alloc main kv_cache l{layer_idx}: {e:?}"))?,
                );
            }
        }
    }

    // Select the right wkv/wgate-output buffer + ring state + cache.
    let (kv_batch_full, score_batch_full) = if is_indexer {
        (&pbs.comp_idx_kv_batch, &pbs.comp_idx_score_batch)
    } else {
        (&pbs.comp_main_kv_batch, &pbs.comp_main_score_batch)
    };
    let norm = if is_indexer {
        layer
            .indexer_compressor_norm
            .as_ref()
            .ok_or_else(|| format!("idx_comp_norm l{layer_idx}"))?
    } else {
        layer
            .compressor_norm
            .as_ref()
            .ok_or_else(|| format!("comp_norm l{layer_idx}"))?
    };
    let ape = if is_indexer {
        layer
            .indexer_compressor_ape
            .as_ref()
            .ok_or_else(|| format!("idx_comp_ape l{layer_idx}"))?
    } else {
        layer
            .compressor_ape
            .as_ref()
            .ok_or_else(|| format!("comp_ape l{layer_idx}"))?
    };

    // Apply per-slot APE to the batched score buffer. This MUST happen
    // before any kernel that reads score_batch_full (compress, ring-write,
    // or state-update memcpy) — those kernels consume the APE-applied
    // scores, mirroring the sequential per-position path in
    // `compressor_forward_impl`.
    //
    // The batched score buffer is allocated at `[max_batch, 2 * head_dim]`
    // but the GEMV writes `proj_dim` floats per slot (head_dim for
    // ratio=128, 2*head_dim for ratio=4 overlap). The APE add reads the
    // same `proj_dim` columns of each slot, so the unused tail half
    // (ratio=128 layers only) stays untouched.
    gpu.compressor_add_ape_batched_f32(
        score_batch_full,
        ape,
        batch_size as i32,
        proj_dim as i32,
        ratio as i32,
        start_pos as i32,
    )
    .map_err(|e| format!("comp ape batched l{layer_idx}: {e:?}"))?;

    let slot_base = (start_pos as usize) % ratio;
    // first chunk position whose absolute (p+1) % R == 0:
    // first_event_chunk_pos = R - 1 - slot_base.
    let first_event_chunk_pos = if slot_base == 0 {
        ratio - 1
    } else {
        ratio - slot_base - 1
    };
    let n_events = if first_event_chunk_pos < batch_size {
        (batch_size - first_event_chunk_pos).div_ceil(ratio)
    } else {
        0
    };

    let aligned = slot_base == 0;
    let compressed_slot_base = (start_pos as usize) / ratio;

    // Check kv_cache capacity for this chunk's events.
    let n_events_capped = if compressed_slot_base + n_events > max_compressed {
        max_compressed.saturating_sub(compressed_slot_base)
    } else {
        n_events
    };

    // ALIGNED PATH: B*R-aligned chunk start, do batched compress.
    if aligned && n_events_capped > 0 {
        let kv_state = if is_indexer {
            state._indexer[layer_idx].indexer_kv_state.as_ref().unwrap()
        } else {
            state._indexer[layer_idx].main_kv_state.as_ref().unwrap()
        };
        let score_state = if is_indexer {
            state._indexer[layer_idx]
                .indexer_score_state
                .as_ref()
                .unwrap()
        } else {
            state._indexer[layer_idx].main_score_state.as_ref().unwrap()
        };
        let kv_cache = if is_indexer {
            state._indexer[layer_idx].indexer_kv_cache.as_ref().unwrap()
        } else {
            state._indexer[layer_idx].main_kv_cache.as_ref().unwrap()
        };

        // `prev_kv` / `prev_score` for event 0 = first R rows of ring state.
        // For overlap=1: ring rows 0..R hold the prior chunk's last NEW window
        //   (FIRST half is the OLD-contribution; SECOND half unused).
        // For chunk 0 (start_pos=0): ring state is zeros — correct: OLD == 0.
        let prev_kv = kv_state.sub_offset(0, ratio * proj_dim);
        let prev_score = score_state.sub_offset(0, ratio * proj_dim);

        let kv_cache_out =
            kv_cache.sub_offset(compressed_slot_base * head_dim, n_events_capped * head_dim);

        gpu.compressor_compress_aligned_batched_f32(
            &prev_kv,
            &prev_score,
            kv_batch_full,
            score_batch_full,
            &kv_cache_out,
            ratio as i32,
            head_dim as i32,
            n_events_capped as i32,
            if overlap { 1 } else { 0 },
            batch_size as i32,
        )
        .map_err(|e| format!("compressor_compress_aligned_batched l{layer_idx}: {e:?}"))?;

        // RMSNorm batched over n_events × head_dim.
        gpu.rmsnorm_batched(
            &kv_cache_out,
            norm,
            &kv_cache_out,
            n_events_capped,
            head_dim,
            cfg.rms_norm_eps,
        )
        .map_err(|e| format!("comp rmsnorm batched l{layer_idx}: {e:?}"))?;

        // Tail RoPE batched. Per event we want a per-event position.
        // Build the position array on host and upload once.
        // See note in `update_pos_array_host` — "start" matches reference
        // ds4 (`comp_pos = pos + 1 - ratio`). Default to that; "mid" / "end"
        // remain available via env var for diagnostic A/B.
        let rope_pos_mode = std::env::var("HIPFIRE_DEEPSEEK4_COMP_ROPE_POS")
            .ok()
            .unwrap_or_else(|| "start".to_string());
        let positions_host: Vec<i32> = (0..n_events_capped)
            .map(|k| {
                let absolute_event_pos = first_event_chunk_pos + k * ratio + (start_pos as usize);
                if is_indexer {
                    // Indexer always uses start-of-window.
                    (absolute_event_pos / ratio * ratio) as i32
                } else {
                    match rope_pos_mode.as_str() {
                        "end" => absolute_event_pos as i32,
                        "mid" => ((absolute_event_pos / ratio * ratio) + ratio / 2) as i32,
                        _ => (absolute_event_pos / ratio * ratio) as i32,
                    }
                }
            })
            .collect();
        // Use the existing pbs.positions field as scratch (it's [max_batch] F32).
        // We need at least n_events_capped slots. n_events_capped <= max_batch
        // because each event consumes R positions of input. Safe.
        let pos_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n_events_capped * 4)
        };
        gpu.memcpy_htod_auto(&pbs.comp_positions.buf, pos_bytes)
            .map_err(|e| format!("htod comp positions l{layer_idx}: {e:?}"))?;

        if is_indexer {
            gpu.rope_tail_interleaved_batched(
                &kv_cache_out,
                &kv_cache_out,
                &pbs.comp_positions,
                1,
                0,
                head_dim as i32,
                cfg.qk_rope_head_dim as i32,
                cfg.compress_rope_theta,
                n_events_capped as i32,
            )
            .map_err(|e| format!("comp idx rope batched l{layer_idx}: {e:?}"))?;
        } else {
            let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
                layer_rope_params(cfg, layer.compress_ratio);
            gpu.rope_tail_yarn_interleaved_batched(
                &kv_cache_out,
                &kv_cache_out,
                &pbs.comp_positions,
                1,
                0,
                head_dim as i32,
                cfg.qk_rope_head_dim as i32,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                corr_low,
                corr_high,
                /*inverse=*/ 0,
                n_events_capped as i32,
            )
            .map_err(|e| format!("comp main rope batched l{layer_idx}: {e:?}"))?;
        }

        // Update ring state for next chunk: kv_state[0..R] ← last NEW window's
        // positions from kv_batch_full. For overlap=1 the last NEW window is
        // chunk positions [(n_events - 1) * R + first_event_chunk_pos - R + 1
        // .. n_events * R + first_event_chunk_pos]. With aligned (slot_base=0,
        // first_event_chunk_pos = R-1), that simplifies to
        // chunk positions [(n_events - 1) * R .. n_events * R - 1].
        //
        // For overlap=0 (ratio=128), no shift-state needed for the next chunk:
        // the ring still holds in-progress NEW positions (which the no-event
        // path will scatter). But here n_events > 0 only happens for
        // overlap=1 at our typical B=64 ratio=4 case.
        if overlap {
            let last_new_start_b = (n_events_capped - 1) * ratio;
            // Source slice: kv_batch_full[last_new_start_b..last_new_start_b + R]
            let src_kv = kv_batch_full.sub_offset(last_new_start_b * proj_dim, ratio * proj_dim);
            let src_score =
                score_batch_full.sub_offset(last_new_start_b * proj_dim, ratio * proj_dim);
            let dst_kv = kv_state.sub_offset(0, ratio * proj_dim);
            let dst_score = score_state.sub_offset(0, ratio * proj_dim);
            let bytes = ratio * proj_dim * 4;
            gpu.memcpy_dtod_auto(&dst_kv.buf, &src_kv.buf, bytes)
                .map_err(|e| format!("comp state update kv l{layer_idx}: {e:?}"))?;
            gpu.memcpy_dtod_auto(&dst_score.buf, &src_score.buf, bytes)
                .map_err(|e| format!("comp state update score l{layer_idx}: {e:?}"))?;
        }

        return Ok(());
    }

    // NO-EVENT PATH (n_events == 0): just scatter all B positions into the
    // ring state for the next chunk to pick up.
    // Also covers the non-aligned case as a safe fallback for now.
    if !aligned || n_events_capped == 0 {
        let kv_state = if is_indexer {
            state._indexer[layer_idx].indexer_kv_state.as_ref().unwrap()
        } else {
            state._indexer[layer_idx].main_kv_state.as_ref().unwrap()
        };
        let score_state = if is_indexer {
            state._indexer[layer_idx]
                .indexer_score_state
                .as_ref()
                .unwrap()
        } else {
            state._indexer[layer_idx].main_score_state.as_ref().unwrap()
        };

        gpu.compressor_ring_write_batched_f32(
            kv_batch_full,
            score_batch_full,
            kv_state,
            score_state,
            batch_size as i32,
            proj_dim as i32,
            ratio as i32,
            slot_base as i32,
            if overlap { 1 } else { 0 },
        )
        .map_err(|e| format!("comp ring write batched l{layer_idx}: {e:?}"))?;

        // If aligned but n_events==0 (impossible by construction), or
        // non-aligned (we should add per-position compress-event handling
        // for any events that DO fire in this chunk). For our DeepSeek V4 bench
        // start_pos is always a multiple of B which is a multiple of 4,
        // so this path is hit only for ratio=128 layers at B<128. No
        // compress events to handle.
        if !aligned && n_events_capped > 0 {
            return Err(format!(
                "compressor_forward_batched: non-aligned chunks with compress events \
                 not yet supported (l{layer_idx}, start_pos={start_pos}, B={batch_size}, ratio={ratio})"
            ));
        }
    }

    Ok(())
}

/// DeepSeek V4 indexer scoring + top-K selection (phase 4b).
///
/// Run after `compressor_forward(is_indexer=true)` for layers with
/// `compress_ratio == 4`. Produces `state._indexer[l].topk_idx_indices`,
/// the indices into `indexer_kv_cache` that the modified main attention
/// (phase 5) will gather K/V from.
///
/// Pipeline:
///   q_idx     = indexer_wq_b @ q_lat_rot                  → [H, D]
///   tail-rope on q_idx (compress_rope_theta, current pos) → [H, D]
///   idx_w     = indexer_weights_proj @ state.tmp          → [H]
///   scores[n] = sum_h relu(q_idx[h] · K_cache[n]) * idx_w[h]
///   topk      = top-K(scores) — combined, not per-head
///
/// Returns the actual number of compressed slots scored (0 means no
/// scoring possible because the cache is still empty at this pos).
#[allow(dead_code, clippy::too_many_arguments)]
fn indexer_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    position: u32,
) -> Result<usize, String> {
    let layer = weights.resolve_layer(layer_idx);
    if layer.compress_ratio != 4 {
        return Ok(0);
    }

    let h = cfg.index_n_heads;
    let d = cfg.index_head_dim;
    let k = cfg.index_topk;
    let pos = position as usize;
    let ratio = 4usize;

    // Compressed-slot count = number of writes already committed.
    // Writes happen when `(pos+1) % ratio == 0`. Just-finished pos:
    //   n_filled = (pos + 1) / ratio  (integer)
    let n_filled = (pos + 1) / ratio;
    // HIP-graphs note: the host-side `if n_filled == 0 { return 0 }`
    // early return was removed. The buf-variant kernels handle n=0
    // gracefully (relu_score writes -inf sentinels, top_k writes -1
    // sentinels, downstream gather kernels skip -1 idx). Always
    // running the kernels means the captured graph contains them
    // whether warmup hit them or not, fixing graph replay at early
    // positions.
    let max_compressed = env_cache::max_compress_pos();
    let n = n_filled.min(max_compressed);

    let wq_b = layer
        .indexer_wq_b
        .as_ref()
        .ok_or_else(|| format!("idx_wq_b l{layer_idx}"))?;
    let weights_proj = layer
        .indexer_weights_proj
        .as_ref()
        .ok_or_else(|| format!("idx_weights_proj l{layer_idx}"))?;

    // Lazy-alloc scratch on this layer's indexer state.
    {
        let l_state = &mut state._indexer[layer_idx];
        if l_state.q_idx.is_none() {
            l_state.q_idx = Some(
                gpu.alloc_tensor(&[h, d], DType::F32)
                    .map_err(|e| format!("alloc q_idx l{layer_idx}: {e:?}"))?,
            );
        }
        if l_state.idx_weights.is_none() {
            l_state.idx_weights = Some(
                gpu.alloc_tensor(&[h], DType::F32)
                    .map_err(|e| format!("alloc idx_weights l{layer_idx}: {e:?}"))?,
            );
        }
        if l_state.index_score.is_none() {
            l_state.index_score = Some(
                gpu.alloc_tensor(&[max_compressed], DType::F32)
                    .map_err(|e| format!("alloc index_score l{layer_idx}: {e:?}"))?,
            );
        }
        if l_state.topk_idx_indices.is_none() {
            l_state.topk_idx_indices = Some(
                gpu.alloc_tensor(&[k], DType::F32)
                    .map_err(|e| format!("alloc topk_idx l{layer_idx}: {e:?}"))?,
            );
        }
    }

    // 1. q_idx = wq_b @ q_lat_rot   (MQ4 prerotated GEMV: M = H*D, K = q_lora_rank)
    let q_lat = state
        .q_lat
        .as_ref()
        .ok_or_else(|| "indexer: q_lat not allocated".to_string())?;
    let q_lat_rot = state
        .q_lat_rot
        .as_ref()
        .ok_or_else(|| "indexer: q_lat_rot not allocated".to_string())?;
    let q_idx = state._indexer[layer_idx].q_idx.as_ref().unwrap();
    gemv_auto(gpu, wq_b, q_lat_rot, q_lat, q_idx, h * d, cfg.q_lora_rank)?;

    // 2. Tail RoPE on q_idx with compress_rope_theta (matching is_indexer=true
    //    K-side compressor's RoPE). Use main `pos_buf` (already holds current
    //    position from apply_tail_rope). qk_rope_head_dim applies on each head.
    let pos_buf = state
        .pos_buf
        .as_ref()
        .ok_or_else(|| "indexer: pos_buf missing".to_string())?;
    gpu.rope_tail_interleaved(
        q_idx,
        q_idx,
        pos_buf,
        h as i32,
        0,
        d as i32,
        cfg.qk_rope_head_dim as i32,
        cfg.compress_rope_theta,
    )
    .map_err(|e| format!("idx rope l{layer_idx}: {e:?}"))?;

    // 3. idx_w = weights_proj @ state.tmp  → [H]
    let tmp = state
        .tmp
        .as_ref()
        .ok_or_else(|| "indexer: state.tmp missing".to_string())?;
    let tmp_plain = state
        .tmp_plain
        .as_ref()
        .ok_or_else(|| "indexer: tmp_plain missing".to_string())?;
    let idx_w = state._indexer[layer_idx].idx_weights.as_ref().unwrap();
    gemv_auto(gpu, weights_proj, tmp, tmp_plain, idx_w, h, cfg.hidden_size)?;

    // 4. Score: combined relu-weighted dot products.
    // HIP-graphs-safe: read N (n_compressed_4) from attn_state_buf[2]
    // instead of baking it as i32 kernarg + sub_offset(0, n*d) view.
    // We pass the FULL kv_cache and scores pointers; the buf kernel
    // bounds work to the first N positions and writes -inf to out-of-
    // range scores so top_k_buf ignores them.
    let kv_cache = state._indexer[layer_idx]
        .indexer_kv_cache
        .as_ref()
        .ok_or_else(|| "indexer: kv_cache missing".to_string())?;
    let scores = state._indexer[layer_idx].index_score.as_ref().unwrap();
    let attn_buf = state
        .attn_state_buf
        .as_ref()
        .ok_or_else(|| "indexer: attn_state_buf missing".to_string())?;
    let n_buf = attn_buf.sub_offset(2, 1); // n_compressed_4
    let k_buf = attn_buf.sub_offset(4, 1); // k_active_4
    gpu.indexer_relu_score_f32_buf(
        q_idx,
        kv_cache,
        idx_w,
        scores,
        &n_buf,
        max_compressed as i32,
        h as i32,
        d as i32,
    )
    .map_err(|e| format!("idx score buf l{layer_idx}: {e:?}"))?;

    // 5. Top-K: read N + K from device buffers.
    let topk = state._indexer[layer_idx].topk_idx_indices.as_ref().unwrap();
    gpu.indexer_top_k_buf(
        scores,
        topk,
        &n_buf,
        &k_buf,
        /*n_idx_heads=*/ 1,
        max_compressed as i32,
        k as i32,
    )
    .map_err(|e| format!("idx top_k buf l{layer_idx}: {e:?}"))?;
    let _ = n; // legacy host-computed; not used after migration

    Ok(n)
}

/// Single-token decode step. Takes the token id of the previous
/// position, returns the logits over `vocab_size`.
///
/// Caller is responsible for sampler integration and KV-state
/// advancement.
#[allow(unused_variables, dead_code)]
pub fn decode_step(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    // HIP-graphs prerequisite: lift the ~130 per-token pos_buf
    // `memcpy_htod` calls out of the per-layer code into a single
    // bulk write at decode-step entry. Per-layer kernels then read
    // their slot via `pos_slot(state, layer_idx, slot)`.
    precompute_positions(cfg, state, gpu, position)?;
    // Stage current token_id to device for the GPU hash-router
    // (consumed by `hash_router_normalize_f32_buf` on hash layers).
    precompute_token_id(state, gpu, token_id)?;

    // 1. Token embedding → initial residual streams.
    //    DeepSeek V4 uses `hc_mult = 4` parallel streams. Init pattern is
    //    [embed, 0, 0, 0] (paper-specified; verify against the DeepSeek V4
    //    reference code before optimising).
    init_residual_streams(cfg, weights, state, gpu, token_id)?;

    let _ = decode_step_body(cfg, weights, state, gpu, token_id, position)?;
    let logits = state.logits.as_ref().unwrap();
    gpu.download_f32(logits)
        .map_err(|e| format!("download logits: {e:?}"))
}

/// HIP-graphs-aware decode_step. Opt-in via `HIPFIRE_DEEPSEEK4_GRAPH=1`.
///
/// Three-state machine driven by `state.ar_forward_warmed_up` and
/// `gpu.graphs.graph_exec`:
///   1. !warmed_up                   → direct dispatch (warmup so JIT
///                                       and lazy alloc happen out of
///                                       the captured region), set flag
///   2. warmed_up && no graph        → wrap layer loop + head in
///                                       `begin_graph_capture`/`end_graph_capture`,
///                                       instantiate, run it once
///   3. graph already instantiated   → update `pos_array_host[]` on
///                                       the host (stable Box source),
///                                       `graph_launch()` re-runs the
///                                       captured ops which re-read
///                                       pos_array_host; download logits
///
/// Returns logits same as `decode_step`. Falls back to plain
/// `decode_step` when `HIPFIRE_DEEPSEEK4_GRAPH` is unset / "0".
pub fn decode_step_with_graph(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    use std::sync::OnceLock;
    // State-dependent kernargs (SWA slot/n_valid, indexer n_compressed/k_active,
    // compressor ring/commit slots) all live in `state.attn_state_buf` and
    // `state.pos_array_device` device buffers now. The captured graph re-reads
    // those on every replay → byte-equivalent against direct dispatch out to
    // 200+ steps on gfx1151 (graph_drift_check). Default ON for RDNA3+
    // (gfx11xx/gfx12xx) where graph capture is mature; opt out with
    // `HIPFIRE_DEEPSEEK4_GRAPH=0`. Force on for older archs with
    // `HIPFIRE_DEEPSEEK4_GRAPH=1` (untested — beware kernarg-bake regressions).
    static GRAPH_OPT_ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_override =
        *GRAPH_OPT_ENV.get_or_init(
            || match std::env::var("HIPFIRE_DEEPSEEK4_GRAPH").ok().as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            },
        );
    let graph_on = env_override.unwrap_or_else(|| {
        let a = gpu.arch.as_str();
        a.starts_with("gfx11") || a.starts_with("gfx12")
    });
    // Note: prior to bc6353e the hash-routed MoE path did a d2h of
    // router scores inside the layer body — that broke HIP graph
    // capture. Replaced by `hash_router_normalize_f32_buf` which
    // reads token_id from a device buffer (staged by
    // `precompute_token_id` at decode entry), so MoE+hash layers
    // are now graph-safe and no guard is needed.
    if !graph_on {
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // ── Warmup phase: direct dispatch, no capture ──────────────────
    if !state.ar_forward_warmed_up {
        state.ar_forward_warmed_up = true;
        return decode_step(cfg, weights, state, gpu, token_id, position);
    }

    // From here on we need an explicit stream for capture/replay.
    if gpu.active_stream.is_none() {
        let s = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("decode_step_with_graph: stream_create: {e:?}"))?;
        gpu.active_stream = Some(s);
    }

    // Embedding lookup and pos-array host write run OUTSIDE the captured
    // region. token_id is baked into the embedding kernel arg, so capture
    // would lock the graph to a single token. Pos-array host source must
    // be a stable `Box<[i32]>` — the captured memcpy re-reads it on each
    // replay. We update those host bytes BEFORE launching the graph.
    init_residual_streams(cfg, weights, state, gpu, token_id)?;

    if gpu.graphs.graph_exec.is_none() {
        // ── Capture phase ──────────────────────────────────────────
        // precompute_positions + precompute_token_id are called INSIDE
        // the capture so the captured memcpy nodes re-read their stable
        // host sources on each replay.
        gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("begin_graph_capture: {e:?}"))?;
        precompute_positions(cfg, state, gpu, position)?;
        precompute_token_id(state, gpu, token_id)?;
        let _ = decode_step_body(cfg, weights, state, gpu, token_id, position)?;
        gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("end_graph_capture: {e:?}"))?;
        // Captured kernels were RECORDED, not executed. Launch the
        // freshly-instantiated graph once so this position's forward
        // actually runs and `state.logits` gets fresh values.
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("graph_launch (capture-end): {e:?}"))?;
        eprintln!(
            "[DeepSeek V4 hipGraph] captured forward — {} kernarg blobs retained",
            gpu.graphs.capture_blobs.len()
        );
    } else {
        // ── Replay phase ───────────────────────────────────────────
        // Host-only update of the stable pos_array_host[], attn_state
        // _host[], and token_id_host[]. The captured memcpy nodes
        // re-read these bytes on graph_launch and propagate them to
        // the device-side pos_array_device / attn_state_buf /
        // token_id_buf which all per-layer kernels read.
        update_pos_array_host(cfg, state, position);
        // attn_state depends on state.n_tokens BEFORE increment (the
        // current position being processed). decode_step normally
        // increments state.n_tokens at the END of the body, so replay
        // sees the right pre-increment value.
        update_attn_state_host(cfg, state, state.n_tokens as u32);
        update_token_id_host(state, token_id);
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
            .map_err(|e| format!("graph_launch (replay): {e:?}"))?;
        state.n_tokens += 1;
    }

    // Logits download is outside the captured region (sync memcpy_dtoh
    // on the null stream — completes after the captured kernels finish
    // because the captured stream is observed by the device).
    let logits = state.logits.as_ref().unwrap();
    gpu.download_f32(logits)
        .map_err(|e| format!("download logits (graph path): {e:?}"))
}

/// Update `state.attn_state_host = [slot, n_valid]` and copy to the
/// device buffer. Called from `precompute_positions` (which is itself
/// inside the captured region during graph capture) so the captured
/// memcpy node re-reads the stable host source on every replay.
///
/// `slot = state.n_tokens % sliding_window`
/// `n_valid = min(state.n_tokens + 1, sliding_window)`
///
/// Layer-independent: all 43 layers read the same two values.
pub(crate) fn precompute_attn_state(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    if state.attn_state_buf.is_none() {
        state.attn_state_buf = Some(
            gpu.alloc_tensor(&[10], DType::F32)
                .map_err(|e| format!("alloc attn_state_buf: {e:?}"))?,
        );
    }
    if state.attn_state_host.is_none() {
        state.attn_state_host = Some(Box::new([0i32; 10]));
    }
    fill_attn_state_host(cfg, state, state.n_tokens as u32);
    let host = state.attn_state_host.as_ref().unwrap();
    let dev = state.attn_state_buf.as_ref().unwrap();
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, 10 * 4) };
    gpu.memcpy_htod_auto(&dev.buf, bytes)
        .map_err(|e| format!("htod attn_state: {e:?}"))
}

/// Internal helper: fill `state.attn_state_host[0..10]` from `position`
/// using DeepSeek V4's compress-ratio + index_topk constants. Used by both
/// `precompute_attn_state` (decode entry) and `update_attn_state_host`
/// (graph replay path).
fn fill_attn_state_host(cfg: &DeepseekV4Config, state: &mut DeepseekV4State, position: u32) {
    let win = cfg.sliding_window as i32;
    let topk = cfg.index_topk as i32; // DeepSeek V4: 512
    let pos = position as i32;
    let swa_slot = pos % win;
    let n_valid_swa = (pos + 1).min(win);
    let n_compressed_4 = (pos + 1) / 4;
    let n_compressed_128 = (pos + 1) / 128;
    let k_active_4 = topk.min(n_compressed_4);
    let k_active_128 = topk.min(n_compressed_128);
    // Compressor ring/commit slots. For overlap=true (ratio=4 in DeepSeek V4),
    // the state ring is sized [2*ratio, proj_dim] and writes go to the
    // second half: `ring + ratio + (pos % ratio)`. Commit slot is
    // pos/ratio at commit positions, -1 otherwise (commit kernels
    // early-return on -1).
    let ring_slot_4 = 4 + (pos % 4);
    let max_compressed = env_cache::max_compress_pos() as i32;
    let commit_slot_4 = if (pos + 1) % 4 == 0 {
        let s = pos / 4;
        if s < max_compressed {
            s
        } else {
            -1
        }
    } else {
        -1
    };
    let ring_slot_128 = pos % 128; // overlap=false (ratio=128)
    let commit_slot_128 = if (pos + 1) % 128 == 0 {
        let s = pos / 128;
        if s < max_compressed {
            s
        } else {
            -1
        }
    } else {
        -1
    };
    let host = state
        .attn_state_host
        .as_mut()
        .expect("fill_attn_state_host: attn_state_host not initialised");
    host[0] = swa_slot;
    host[1] = n_valid_swa;
    host[2] = n_compressed_4;
    host[3] = n_compressed_128;
    host[4] = k_active_4;
    host[5] = k_active_128;
    host[6] = ring_slot_4;
    host[7] = commit_slot_4;
    host[8] = ring_slot_128;
    host[9] = commit_slot_128;
}

/// Update host-only `attn_state_host[]` (no device copy). Used by the
/// HIP-graphs replay path — the captured memcpy node re-reads this
/// buffer when graph_launch fires.
pub(crate) fn update_attn_state_host(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    position: u32,
) {
    fill_attn_state_host(cfg, state, position);
}

/// Host-only update of `state.pos_array_host[]` for the given position.
/// Used by the HIP-graphs replay path; the captured memcpy node will
/// re-read these bytes when `graph_launch` runs.
pub(crate) fn update_pos_array_host(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    position: u32,
) {
    let pos_array_host = state.pos_array_host.as_mut().expect(
        "update_pos_array_host: pos_array_host not initialised (call precompute_positions first)",
    );
    fill_pos_array_host(cfg, pos_array_host, position);
}

/// Shared host-side fill of the per-layer `[qk_pos, main_comp_rope_pos,
/// indexer_comp_rope_pos]` triples. Called by both `precompute_positions`
/// (initial alloc + htod path) and `update_pos_array_host` (graph-replay
/// host-only path).
///
/// Reference ds4 uses `comp_pos = pos + 1 - ratio` at compress events
/// (i.e. start of the just-closed window). Equivalent to
/// `pos / ratio * ratio` when `(pos+1) % ratio == 0`, which is exactly
/// when an event fires. "start" matches the reference; "mid" / "end"
/// remain available for diagnostic A/B via `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS`.
///
/// Why one helper: prior to this refactor `precompute_positions` and
/// `update_pos_array_host` carried independently-edited copies of this
/// loop with DIFFERENT defaults (capture path: "mid", replay path:
/// "start"). The captured graph then read one rope_pos at capture time
/// and a different value at replay time, drifting compressor RoPE
/// across the capture/replay boundary.
fn fill_pos_array_host(cfg: &DeepseekV4Config, pos_array_host: &mut [i32], position: u32) {
    let comp_rope_mode = std::env::var("HIPFIRE_DEEPSEEK4_COMP_ROPE_POS").ok();
    let comp_rope_mode = comp_rope_mode.as_deref();
    for layer_idx in 0..=cfg.num_hidden_layers {
        let ratio = if layer_idx < cfg.num_hidden_layers {
            cfg.compress_ratios[layer_idx] as usize
        } else {
            0
        };
        let base = layer_idx * POS_SLOTS_PER_LAYER;
        pos_array_host[base] = position as i32;
        if ratio > 0 {
            let main_rope_pos: i32 = match comp_rope_mode {
                Some("end") => position as i32,
                Some("mid") => (((position as usize) / ratio * ratio) + ratio / 2) as i32,
                _ => ((position as usize) / ratio * ratio) as i32,
            };
            let indexer_rope_pos = ((position as usize) / ratio * ratio) as i32;
            pos_array_host[base + 1] = main_rope_pos;
            pos_array_host[base + 2] = indexer_rope_pos;
        } else {
            pos_array_host[base + 1] = 0;
            pos_array_host[base + 2] = 0;
        }
    }
}

/// Captured-region body of `decode_step`: the per-layer forward loop
/// + final norm + head. Token-id-dependent embedding lookup and
/// position-array setup must be done BEFORE calling this — they are
/// non-graph-safe (token_id is kernarg, position-array htod source must
/// stay alive across replays).
///
/// `position` is still passed through so position-derived sizing logic
/// (e.g. `n_filled = (pos + 1) / ratio` in `indexer_forward`) gets the
/// real value — these are HOST computations that select which slots of
/// the captured kernel to read, not kernarg-side position writes.
///
/// Public so the HIP-graphs capture/replay wrapper can call it directly.
pub fn decode_step_body(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let skip_ffn = env_cache::skip_ffn();

    // #397 Ship 6 — forward-as-pipeline. The per-layer decode routes through the
    // super-op executor (run_layer_program) by DEFAULT; HIPFIRE_FORWARD_LOWERED=0
    // opts back to the hand loop. Validated byte-identical on hipx (the only box
    // deepseek4 fits) in both plain AR and MTP spec-decode modes.
    if ds4_forward_lowered_enabled() {
        return decode_step_body_lowered(cfg, weights, state, gpu, token_id, position);
    }

    // 2. Per-layer forward.
    for layer_idx in 0..cfg.num_hidden_layers {
        let layer = weights.resolve_layer(layer_idx);
        let _l_state = &mut state._indexer[layer_idx];
        let _l_attn = &mut state._attention[layer_idx];

        // ── 2a. Attention block ───────────────────────────────────────
        //
        // mHC pre-step + full mHC mix (paper-faithful) — DISABLED.
        // Even with proper sigmoid/exp+Sinkhorn/2σ/input-mapping
        // implementation, 43 layers cumulative additions overflow f32
        // because we don't apply the small-init learnable α scalars
        // (hc_*_scale [3] in the paper, initialised to small values).
        // Wire those into hc_compute_control as `α · (X · W) + base`
        // and retry. For now: pipeline runs HC-disabled producing
        // bounded but architecturally-trivial logits.
        // Real mHC with corrected kernels (F32 throughout for residuals).
        mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/ true)?;
        q_lora(cfg, weights, state, gpu, layer_idx)?;

        // (Q-LoRA call moved above into the fused RMSNorm + GEMV step.)

        // iii. Joint KV: wkv @ tmp → kv [head_dim = 512] (tied K=V).
        kv_joint(cfg, weights, state, gpu, layer_idx)?;

        // iv. Tail-only RoPE on Q and KV.
        //     Apply rotation on last `qk_rope_head_dim = 64` of each
        //     head's 512 dims.
        //     SWA ring write deferred (needs swa state alloc per layer).
        apply_tail_rope(cfg, weights, state, gpu, position, layer_idx)?;

        // iv. Indexer path (only when compress_ratio > 0):
        //     a. Compressor: x @ compressor.wkv → idx_qk
        //        x @ compressor.wgate → idx_v (per DeepSeek V4 structure)
        //        x normalised by compressor.norm
        //     b. Apply compress_rope on idx_q (freq_base = compress_rope_theta = 160000)
        //     c. If position % compress_ratio == 0: append idx_k to k_idx_compressed cache
        //     d. `gpu.indexer_compressed_k_score(q_idx, k_idx_cache, scores, ...)`
        //     e. `gpu.indexer_top_k(scores, top_indices, ..., k = index_topk = 512)`
        //     f. dedup top_indices across heads (UNION strategy per paper)
        //     g. `gpu.indexer_kv_gather(k_main_cache, v_main_cache, unique_indices, ...)`
        //
        //     When compress_ratio == 0: skip; attention reads SWA only.
        //
        // DeepSeek V4 compressor + indexer (antirez-faithful default behavior):
        // Always run for ratio>0 layers. Antirez ds4 runs compressor
        // unconditionally for compressed layers and the indexer for
        // ratio==4 layers (ds4.c:7505-7555).
        if layer.compress_ratio > 0 {
            let tmp_view = {
                let t = state.tmp.as_ref().unwrap();
                t.sub_offset(0, t.numel())
            };
            compressor_forward(
                cfg, weights, state, gpu, layer_idx, &tmp_view, position,
                /*is_indexer=*/ false,
            )?;
            if layer.compress_ratio == 4 {
                compressor_forward(
                    cfg, weights, state, gpu, layer_idx, &tmp_view, position,
                    /*is_indexer=*/ true,
                )?;
                let _n = indexer_forward(cfg, weights, state, gpu, layer_idx, position)?;
            }
        }

        // v + vi. Main attention + O-LoRA — STUB.
        attn_stub(cfg, weights, state, gpu, layer_idx)?;

        hc_attn_mix(cfg, weights, state, gpu, layer_idx)?;

        // ── 2b. FFN block ─────────────────────────────────────────────
        mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/ false)?;
        if !skip_ffn {
            ffn_stub(cfg, weights, state, gpu, layer_idx)?;
            if layer_idx < cfg.num_hash_layers {
                ffn_hash_routed(cfg, weights, state, gpu, layer_idx, token_id, None)?;
            } else {
                ffn_routed(cfg, weights, state, gpu, layer_idx, None)?;
            }
        } else {
            // Diagnostic: zero ffn_out to isolate attn contribution to growth.
            if state.ffn_out.is_none() {
                state.ffn_out = Some(
                    gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                        .map_err(|e| format!("alloc ffn_out: {e:?}"))?,
                );
            }
            let ffn_out = state.ffn_out.as_ref().unwrap();
            gpu.hip
                .memset(&ffn_out.buf, 0, ffn_out.byte_size())
                .map_err(|e| format!("memset ffn_out: {e:?}"))?;
        }
        hc_ffn_mix(cfg, weights, state, gpu, layer_idx)?;
    }

    // 3. Final norm + LM head. The head-HC mix INSIDE final_norm_and_head
    //    now ALSO captures head_hc_out into state.mtp_last_hidden — that's
    //    the value DeepSeek V4 MTP expects as h_n (post-head-HC-mix, pre-output-norm).
    //    The previous "capture stream 0 before final_norm_and_head" pattern
    //    was wrong on HC models — MTP saw 1 of 4 streams instead of the
    //    actual hidden the main model uses for its own prediction.
    //    Note: DeepSeek V4 has head-level HC (hc_head_base/fn/scale).
    //    For minimal forward: skip the head-HC mix (TODO: head HC
    //    likely projects 4 streams → 1 then applies head_weight)
    //    and just run final norm + standard lm_head.
    final_norm_and_head(cfg, weights, state, gpu)?;

    // Leave logits in `state.logits` for the caller to download. The
    // download is intentionally outside `decode_step_body` so the
    // captured-graph path can place it AFTER `graph_launch` (capturing
    // a sync `memcpy_dtoh` into the captured stream causes wave-reads
    // of stale buffers).
    state.n_tokens += 1;
    Ok(Vec::new())
}

// ─────────────────────────────────────────────────────────────────────────
// #397 Ship 6 — forward-as-pipeline: deepseek4 lowered decode.
//
// deepseek4's decode_step_body is already a sequence of named block fns, so the
// lowering is coarse (minimax-style): every layer is [Attend, Moe], where the
// Attend handler replays the whole attention block (mhc_pre + q_lora + kv_joint +
// tail_rope + conditional compressor/indexer + attn_stub + hc_attn_mix) and the
// Moe handler the whole FFN block (mhc_pre + ffn_stub + hash|score-routed +
// hc_ffn_mix). The per-layer conditionals (compress_ratio, hash vs score) live
// INSIDE the handlers, so it's one variant — the compressor/indexer/HC ops are
// bundled in the coarse handlers (not separate Escape super-ops; Escape stays a
// reserved extension point if per-op remap/fusion is ever wanted). ADDITIVE: the
// hand loop is untouched and reachable via HIPFIRE_FORWARD_LOWERED=0; the lowered
// path is DEFAULT-ON after hipx byte-parity (plain AR + MTP spec-decode).
// ─────────────────────────────────────────────────────────────────────────

/// Attention block (replays decode_step_body's attn arm verbatim). HC residual
/// streams + KV/compressor/indexer state are threaded through `state`.
fn ds4_attn_block(
    cfg: &DeepseekV4Config, weights: &DeepseekV4Weights, state: &mut DeepseekV4State,
    gpu: &mut Gpu, layer_idx: usize, position: u32,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/ true)?;
    q_lora(cfg, weights, state, gpu, layer_idx)?;
    kv_joint(cfg, weights, state, gpu, layer_idx)?;
    apply_tail_rope(cfg, weights, state, gpu, position, layer_idx)?;
    if layer.compress_ratio > 0 {
        let tmp_view = {
            let t = state.tmp.as_ref().unwrap();
            t.sub_offset(0, t.numel())
        };
        compressor_forward(cfg, weights, state, gpu, layer_idx, &tmp_view, position, /*is_indexer=*/ false)?;
        if layer.compress_ratio == 4 {
            compressor_forward(cfg, weights, state, gpu, layer_idx, &tmp_view, position, /*is_indexer=*/ true)?;
            let _n = indexer_forward(cfg, weights, state, gpu, layer_idx, position)?;
        }
    }
    attn_stub(cfg, weights, state, gpu, layer_idx)?;
    hc_attn_mix(cfg, weights, state, gpu, layer_idx)
}

/// FFN block (replays decode_step_body's FFN arm verbatim).
fn ds4_moe_block(
    cfg: &DeepseekV4Config, weights: &DeepseekV4Weights, state: &mut DeepseekV4State,
    gpu: &mut Gpu, layer_idx: usize, token_id: u32, skip_ffn: bool,
) -> Result<(), String> {
    // Non-EP: routed experts combine into `state.ffn_out` (alongside the
    // shared expert seeded by `ffn_stub`), and the HC mix folds ffn_out into
    // `residual_streams` in the same call.
    ds4_moe_block_core(
        cfg, weights, state, gpu, layer_idx, token_id, skip_ffn, None, /*do_mix=*/ true,
    )
}

/// MoE block core, parameterized for expert-parallel (EP).
///
/// - `routed_out = Some(partial)` redirects the routed-expert combine into a
///   zeroed per-rank partial (`partial = Σ_owned w_k · expert_k`), while the
///   SHARED expert (`ffn_stub`) still writes `state.ffn_out` replicated on
///   every rank. The cross-rank all-reduce of `partial` (in the EP executor)
///   then sums the routed contributions; `ds4_ep_add_into_residual` does
///   `ffn_out += all_reduced_partial` so each rank ends with `shared + routed`.
/// - `do_mix = false` defers `hc_ffn_mix` to AFTER the all-reduce (the mix
///   can't run until the full FFN output is assembled).
///
/// `routed_out = None, do_mix = true` is the byte-identical single-GPU path.
fn ds4_moe_block_core(
    cfg: &DeepseekV4Config, weights: &DeepseekV4Weights, state: &mut DeepseekV4State,
    gpu: &mut Gpu, layer_idx: usize, token_id: u32, skip_ffn: bool,
    routed_out: Option<&GpuTensor>, do_mix: bool,
) -> Result<(), String> {
    mhc_pre(cfg, weights, state, gpu, layer_idx, /*is_attn=*/ false)?;
    if !skip_ffn {
        ffn_stub(cfg, weights, state, gpu, layer_idx)?;
        if layer_idx < cfg.num_hash_layers {
            ffn_hash_routed(cfg, weights, state, gpu, layer_idx, token_id, routed_out)?;
        } else {
            ffn_routed(cfg, weights, state, gpu, layer_idx, routed_out)?;
        }
    } else {
        if state.ffn_out.is_none() {
            state.ffn_out = Some(
                gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                    .map_err(|e| format!("alloc ffn_out: {e:?}"))?,
            );
        }
        let ffn_out = state.ffn_out.as_ref().unwrap();
        gpu.hip
            .memset(&ffn_out.buf, 0, ffn_out.byte_size())
            .map_err(|e| format!("memset ffn_out: {e:?}"))?;
    }
    if do_mix {
        hc_ffn_mix(cfg, weights, state, gpu, layer_idx)?;
    }
    Ok(())
}

/// Per-layer execution context for the lowered decode path (rebuilt each layer).
struct Deepseek4Bindings<'a> {
    cfg: &'a DeepseekV4Config,
    weights: &'a DeepseekV4Weights,
    state: &'a mut DeepseekV4State,
    layer_idx: usize,
    position: u32,
    token_id: u32,
    skip_ffn: bool,
}

impl<'a> ForwardBindings for Deepseek4Bindings<'a> {
    fn run_attend(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        ds4_attn_block(self.cfg, self.weights, self.state, gpu, self.layer_idx, self.position)
            .map_err(DispatchError::Hip)
    }
    fn run_moe(&mut self, gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        ds4_moe_block(self.cfg, self.weights, self.state, gpu, self.layer_idx, self.token_id, self.skip_ffn)
            .map_err(DispatchError::Hip)
    }
    fn run_moe_ep(
        &mut self,
        gpu: &mut Gpu,
        _ctx: &DispatchCtx,
        _op: &OpBinding,
        routed_out: &GpuTensor,
        _skip_shared: bool,
    ) -> Result<(), DispatchError> {
        // EP: run mhc_pre + the SHARED expert (ffn_stub, replicated into
        // state.ffn_out on every rank) + the ROUTED experts redirected into the
        // zeroed `routed_out` partial. `hc_ffn_mix` is DEFERRED (do_mix=false)
        // to `ep_add_into_residual`, which runs after the cross-rank all-reduce
        // assembles the full routed output. `skip_shared` is intentionally
        // ignored: DeepSeek's shared expert lives in ffn_out (outside the
        // all-reduced partial), so replicating it per rank is correct — it is
        // never summed across ranks.
        ds4_moe_block_core(
            self.cfg,
            self.weights,
            self.state,
            gpu,
            self.layer_idx,
            self.token_id,
            self.skip_ffn,
            Some(routed_out),
            /*do_mix=*/ false,
        )
        .map_err(DispatchError::Hip)
    }
    fn ep_add_into_residual(&mut self, gpu: &mut Gpu, partial: &GpuTensor) -> Result<(), DispatchError> {
        // ffn_out (shared, replicated) += all-reduced routed partial → full FFN
        // output, then run the deferred HC mix to fold it into residual_streams.
        {
            let ffn_out = self
                .state
                .ffn_out
                .as_ref()
                .ok_or_else(|| DispatchError::Hip("ep_add_into_residual: ffn_out unset".into()))?;
            gpu.add_inplace_f32(ffn_out, partial)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }
        hc_ffn_mix(self.cfg, self.weights, self.state, gpu, self.layer_idx)
            .map_err(DispatchError::Hip)
    }
    fn run_proj(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no Proj super-op".into()))
    }
    fn run_residual_gemv(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no ResidualGemv super-op".into()))
    }
    fn run_norm(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no Norm super-op".into()))
    }
    fn run_conv(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no Conv super-op".into()))
    }
    fn run_recurrent(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding) -> Result<(), DispatchError> {
        Err(DispatchError::Hip("deepseek4 has no Recurrent super-op".into()))
    }
    fn run_escape(&mut self, _gpu: &mut Gpu, _ctx: &DispatchCtx, _op: &OpBinding, kind: superop::EscapeKind) -> Result<(), DispatchError> {
        Err(DispatchError::Hip(format!("deepseek4 has no Escape super-op ({kind:?})")))
    }
}

#[inline]
fn ds4_superop(kind: SuperOpKind) -> SuperOp {
    SuperOp {
        kind,
        binding: OpBinding { key: None, weights: Vec::new(), scratch: Vec::new(), flavor: OpFlavor::None },
    }
}

/// deepseek4 has ONE layer shape ([Attend, Moe]); the per-layer conditionals are
/// inside the handlers. Pure → unit-testable.
fn ds4_lower_program() -> superop::LayerProgram {
    vec![ds4_superop(SuperOpKind::Attend), ds4_superop(SuperOpKind::Moe)]
}

/// Cached HIPFIRE_FORWARD_LOWERED toggle for deepseek4 (default ON, matching
/// qwen35/lfm2/minimax; set =0 to fall back to the hand loop). Flipped on after
/// hipx byte-parity in both plain AR and MTP spec-decode modes.
fn ds4_forward_lowered_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("HIPFIRE_FORWARD_LOWERED").ok().as_deref() != Some("0"))
}

/// Lowered (#397 Ship 6) per-layer decode loop + final norm/head. Behaviorally
/// equivalent to decode_step_body's hand loop (validated via FORWARD_LOWERED=0-vs-=1
/// token-text md5 on hipx). Logits left in state.logits (caller downloads).
fn decode_step_body_lowered(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let skip_ffn = env_cache::skip_ffn();
    let ctx = DispatchCtx::new(gpu);
    let program = ds4_lower_program();
    for layer_idx in 0..cfg.num_hidden_layers {
        let mut bind = Deepseek4Bindings {
            cfg,
            weights,
            state: &mut *state,
            layer_idx,
            position,
            token_id,
            skip_ffn,
        };
        superop::run_layer_program(gpu, &ctx, &program, &mut bind)
            .map_err(|e| format!("ds4 L{layer_idx}: lowered run_layer_program: {e}"))?;
    }
    final_norm_and_head(cfg, weights, state, gpu)?;
    state.n_tokens += 1;
    Ok(Vec::new())
}

// ───────────────────────── Ship 6 substrate-EP (DeepSeek-V4) ─────────────────
//
// Mirror of the qwen35 / MiniMax EP wiring. DeepSeek packs all routed experts
// into ONE blob per projection (too big to load-then-free on a 32 GB card), so
// sharding is done at LOAD time: `DeepseekV4::load_weights_sharded(.., shard,
// rank)` uploads only the rank-owned experts (non-owned → zeroed gate_up dummy).
// UNLIKE MiniMax, DeepSeek has a SHARED expert (ffn_stub) and the HC FFN mix:
//   - the shared expert stays replicated in `state.ffn_out` (every rank),
//   - only the ROUTED combine crosses ranks (redirected into the per-rank
//     partial, all-reduced), and
//   - `hc_ffn_mix` is DEFERRED to `ep_add_into_residual` (runs after the
//     all-reduce assembles `ffn_out = shared + routed`).
// See `Deepseek4Bindings::run_moe_ep` / `ep_add_into_residual` + `ds4_moe_block_core`.
// MLA attention (latent KV) is replicated per rank → no attention-sharding seam.

/// EP (Ship 6 substrate-EP) replicated N-rank decode forward for ONE token.
///
/// Mirror of `decode_step` + `decode_step_body_lowered`, fanned across
/// `gpus.devices.len()` ranks: every rank replicates embed / positions /
/// token-id / residual-stream init and the per-layer `[Attend, Moe]` program
/// (Attend replicated, Moe all-reduce-EP'd) via
/// [`hipfire_runtime::ep::run_layer_program_ep`], then final norm + head run on
/// rank 0 → `state_per_rank[0].logits` (caller downloads). Every device must
/// have an `active_stream` ([`hipfire_runtime::ep::ensure_rank_streams`]); peer
/// access enabled for the fast peer-direct all-reduce.
#[allow(clippy::too_many_arguments)]
pub fn forward_ep(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    partials: &[GpuTensor],
    token: u32,
    position: u32,
) -> Result<(), String> {
    let n = gpus.devices.len();
    assert_eq!(weights_per_rank.len(), n, "ds4 forward_ep: weights_per_rank len");
    assert_eq!(state_per_rank.len(), n, "ds4 forward_ep: state_per_rank len");
    assert_eq!(partials.len(), n, "ds4 forward_ep: partials len");
    let hidden = cfg.hidden_size;
    let skip_ffn = env_cache::skip_ffn();

    // 1. Per-rank embed + position + token-id staging + residual-stream init
    //    (replicated, deterministic functions of the token → bit-identical
    //    across ranks). Mirrors `decode_step`'s preamble.
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("ds4 forward_ep bind {r}: {e:?}"))?;
        precompute_positions(cfg, &mut state_per_rank[r], &mut gpus.devices[r], position)?;
        precompute_token_id(&mut state_per_rank[r], &mut gpus.devices[r], token)?;
        init_residual_streams(cfg, &weights_per_rank[r], &mut state_per_rank[r], &mut gpus.devices[r], token)?;
    }

    // 2. Per-layer EP program (Attend replicated; Moe all-reduce-EP'd). Rebuild
    //    the N per-rank bindings each layer (disjoint `iter_mut` mutable state
    //    borrows), exactly like the single-GPU lowered loop advances per layer.
    let timing = std::env::var("HIPFIRE_EP_DECODE_TIMING").is_ok();
    // Divergence-localization dump: HIPFIRE_EP_DUMP_POS="0,64,...,302" prints a
    // per-(position, layer, rank) fingerprint of the residual streams so EP
    // forwards can be compared across tp counts / arches. Diagnostic only.
    let dump_pos_hit = std::env::var("HIPFIRE_EP_DUMP_POS")
        .ok()
        .map(|s| s.split(',').any(|x| x.trim().parse::<u32>() == Ok(position)))
        .unwrap_or(false);
    let t_layers = std::time::Instant::now();
    let program = ds4_lower_program();
    for l in 0..cfg.num_hidden_layers {
        {
            let mut binds: Vec<Deepseek4Bindings> = Vec::with_capacity(n);
            for (r, st) in state_per_rank.iter_mut().enumerate() {
                binds.push(Deepseek4Bindings {
                    cfg,
                    weights: &weights_per_rank[r],
                    state: st,
                    layer_idx: l,
                    position,
                    token_id: token,
                    skip_ffn,
                });
            }
            hipfire_runtime::ep::run_layer_program_ep(gpus, binds.as_mut_slice(), partials, &program, hidden)
                .map_err(|e| format!("ds4 forward_ep run_layer_program_ep L{l}: {e}"))?;
        }
        if dump_pos_hit {
            for r in 0..n {
                gpus.devices[r]
                    .bind_thread()
                    .map_err(|e| format!("ds4 EPDUMP bind {r}: {e:?}"))?;
                gpus.devices[r]
                    .hip
                    .device_synchronize()
                    .map_err(|e| format!("ds4 EPDUMP sync {r}: {e:?}"))?;
                if let Some(t) = state_per_rank[r].residual_streams.as_ref() {
                    let v = gpus.devices[r].download_f32(t).unwrap_or_default();
                    let l2: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    let mut h: u64 = 0xcbf29ce484222325;
                    for &x in &v {
                        for b in x.to_bits().to_le_bytes() {
                            h ^= b as u64;
                            h = h.wrapping_mul(0x100000001b3);
                        }
                    }
                    eprintln!(
                        "EPDUMP pos={position} layer={l} rank={r} l2={l2:.9e} fnv=0x{h:016x} f0={:.6e} f1={:.6e}",
                        v.first().copied().unwrap_or(0.0),
                        v.get(1).copied().unwrap_or(0.0),
                    );
                }
                // Deeper DSA-path dump (rank 0 only): compressor caches, indexer
                // scores, and selected top-k indices — discriminates a
                // systematically-divergent compressor kernel from near-tie
                // top-k chaos. HIPFIRE_EP_DUMP_IDX=1 to enable.
                if r == 0
                    && std::env::var("HIPFIRE_EP_DUMP_IDX").ok().as_deref() == Some("1")
                {
                    let fp = |gpu: &mut rdna_compute::Gpu,
                              t: &Option<rdna_compute::GpuTensor>|
                     -> String {
                        match t {
                            Some(t) => match gpu.download_f32(t) {
                                Ok(v) => {
                                    let l2: f64 =
                                        v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                                    let mut h: u64 = 0xcbf29ce484222325;
                                    for &x in &v {
                                        for b in x.to_bits().to_le_bytes() {
                                            h ^= b as u64;
                                            h = h.wrapping_mul(0x100000001b3);
                                        }
                                    }
                                    format!("l2={l2:.9e} fnv=0x{h:016x}")
                                }
                                Err(_) => "dl-err".to_string(),
                            },
                            None => "none".to_string(),
                        }
                    };
                    let idx = &state_per_rank[0]._indexer[l];
                    let score_fp = fp(&mut gpus.devices[0], &idx.index_score);
                    let ikv_fp = fp(&mut gpus.devices[0], &idx.indexer_kv_cache);
                    let mkv_fp = fp(&mut gpus.devices[0], &idx.main_kv_cache);
                    let topk_head: String = match idx.topk_idx_indices.as_ref() {
                        Some(t) => match gpus.devices[0].download_f32(t) {
                            Ok(v) => v
                                .iter()
                                .take(24)
                                .map(|x| (x.to_bits() as i32).to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                            Err(_) => "dl-err".to_string(),
                        },
                        None => "none".to_string(),
                    };
                    eprintln!(
                        "EPIDX pos={position} layer={l} score[{score_fp}] ikv[{ikv_fp}] mkv[{mkv_fp}] topk={topk_head}"
                    );
                }
            }
        }
    }

    // 3. Final norm + head on rank 0 → state_per_rank[0].logits.
    {
        gpus.devices[0]
            .bind_thread()
            .map_err(|e| format!("ds4 forward_ep bind0: {e:?}"))?;
        final_norm_and_head(cfg, &weights_per_rank[0], &mut state_per_rank[0], &mut gpus.devices[0])?;
    }

    let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
    // 4. Sync every rank (work ran on active_streams; host logits read races otherwise).
    let t_sync = std::time::Instant::now();
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("ds4 forward_ep sync bind {r}: {e:?}"))?;
        gpus.devices[r]
            .hip
            .device_synchronize()
            .map_err(|e| format!("ds4 forward_ep sync {r}: {e:?}"))?;
    }
    if timing {
        eprintln!(
            "EP-DECODE-TIMING: layers(host)={layers_ms:.2} ms  final-sync(gpu)={:.2} ms",
            t_sync.elapsed().as_secs_f64() * 1000.0,
        );
    }
    for s in state_per_rank.iter_mut() {
        s.n_tokens += 1;
    }
    Ok(())
}

/// DeepSeek V4 Multi-Token Prediction (MTP) forward step — DeepSeek V3 §4.
///
/// Predicts the **next-next** token given:
///   - `h_n`         : hidden state at absolute position N (the output of
///                     the main forward at that position, before the head)
///   - `next_token`  : the token that was emitted at position N+1
///   - `position`    : absolute position N+1 (used by tail-RoPE)
///
/// Output: logits over the vocab for position N+2.
///
/// Architecture (from `mtp.0.*` weights in DeepSeek V4-MTP HFQ files):
/// ```text
/// e_norm     = enorm(embed_lookup(next_token))
/// h_norm     = hnorm(h_n)
/// x_in       = e_proj @ e_norm + h_proj @ h_norm         (Q8F16 GEMVs)
/// x_attn     = attention(attn_norm(x_in))   + x_in        (SWA-only — no compressor)
/// x_ffn      = ffn(ffn_norm(x_attn))        + x_attn      (shared + routed MoE)
/// h_n_plus_1 = mtp_final_norm(x_ffn)
/// logits     = shared_head @ h_n_plus_1                   (reuses main lm_head)
/// ```
///
/// The MTP layer has NO compressor and NO indexer (verified against the
/// safetensors tensor table: only standard attn + FFN weights). Its
/// attention block is the SWA-only path (same as a hash-routed main
/// layer's attention).
///
/// **Status**: M1+M2 (weights ingest) are landed; the standard layer
/// block (attn + FFN with MTP weights) is still pending — the existing
/// per-layer helpers (`q_lora`, `kv_joint`, `attn_stub`, ...) all read
/// `weights.layers[layer_idx]` and need refactoring to accept a
/// `&DeepseekV4LayerWeights` parameter so they can run against
/// `weights.mtp_layer`. Filling in that refactor is M3-complete; it
/// will land alongside validation against the new HFQ that contains
/// the MTP layer.
///
/// Until then this function returns a clear error so callers can stub
/// out the spec-decode path without false-positive bring-up.
pub fn mtp_forward(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    h_n: &GpuTensor,
    next_token: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let mtp_layer_idx = cfg.num_hidden_layers;
    // Steps 0–6: validate MTP weights + embed/norm/HC plumbing + attention.
    mtp_pre_ffn(cfg, weights, state, gpu, h_n, next_token, position)?;
    // FFN block (== ds4_moe_block_core at the MTP layer: mhc_pre(ffn) + shared
    // ffn_stub + routed ffn_routed + hc_ffn_mix). Single-GPU: routed combines
    // into ffn_out alongside the shared expert; the mix folds it.
    mhc_pre(cfg, weights, state, gpu, mtp_layer_idx, /*is_attn=*/ false)?;
    ffn_stub(cfg, weights, state, gpu, mtp_layer_idx)?;
    ffn_routed(cfg, weights, state, gpu, mtp_layer_idx, None)?;
    hc_ffn_mix(cfg, weights, state, gpu, mtp_layer_idx)?;
    // Step 7: capture full HC residual → mtp_last_hidden (chaining input).
    mtp_capture_hidden(cfg, state, gpu)?;
    // SKIP_HEAD short-circuit (prefill MTP-fill: only the SWA write matters).
    if std::env::var("HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD").ok().as_deref() == Some("1") {
        return Ok(Vec::new());
    }
    // Steps 8–9: final norm + lm_head + download.
    mtp_head(cfg, weights, state, gpu)
}

/// Steps 0–6 of the MTP forward: validate MTP weights, embed `next_token`,
/// rmsnorm both inputs, populate the `[hc_mult, hidden]` residual streams via
/// the HC plumbing, and run the MTP-layer attention block (up to `hc_attn_mix`).
/// Shared by [`mtp_forward`] (single-GPU) and [`mtp_forward_ep`] (per rank,
/// replicated — only the FFN routed experts are sharded under EP).
fn mtp_pre_ffn(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    h_n: &GpuTensor,
    next_token: u32,
    position: u32,
) -> Result<(), String> {
    // ── 0. Validate MTP weights are present ────────────────────────────
    let mtp = weights.mtp_layer.as_ref().ok_or_else(|| {
        "mtp_forward: weights.mtp_layer is None — \
            re-quantize DeepSeek V4 with --format deepseek4-q8-mtp to include the \
            mtp.0.* tensors, then HIPFIRE_DEEPSEEK4_LOAD_MTP=1 at load time. \
            Files without an MTP layer cannot run spec-decode."
            .to_string()
    })?;
    let mtp_enorm = mtp
        .mtp_enorm
        .as_ref()
        .ok_or("mtp_forward: mtp_enorm missing")?;
    let mtp_hnorm = mtp
        .mtp_hnorm
        .as_ref()
        .ok_or("mtp_forward: mtp_hnorm missing")?;
    let mtp_e_proj = mtp
        .mtp_e_proj
        .as_ref()
        .ok_or("mtp_forward: mtp_e_proj missing")?;
    let mtp_h_proj = mtp
        .mtp_h_proj
        .as_ref()
        .ok_or("mtp_forward: mtp_h_proj missing")?;

    // Defensive: step 4 below passes `dummy_rotated` aliasing the OTHER
    // norm scratch (not a real FWHT rotation). That's safe for Q8_0 /
    // F16 / F32 dtypes since gemv_auto reads only x_plain on those
    // paths. For MQ4 (Raw dtype) gemv_auto reads x_rotated → we'd feed
    // garbage and produce silent NaN cascades. Reject upfront with a
    // clear message; if someone wants MQ4 MTP they need to plumb proper
    // rotated buffers through step 4.
    for (name, t) in [("mtp_e_proj", mtp_e_proj), ("mtp_h_proj", mtp_h_proj)] {
        match t.dtype {
            DType::F32 | DType::F16 | DType::Q8_0 => {}
            other => {
                return Err(format!(
                    "mtp_forward: {name} dtype {other:?} unsupported — step 4 \
                 only plumbs plain input (no FWHT rotation). Add rotated \
                 buffers or re-quant MTP at Q8F16 / F16."
                ))
            }
        }
    }

    // Full-HC plumbing: h_n must be the previous position's complete
    // [hc_mult, hidden] residual stream (per antirez/ds4 reference). The
    // legacy `[hidden]` shape (stream 0 only) is rejected — it produces
    // the broken ~50% acceptance path. final_norm_and_head and the
    // verify-pass capture in spec_decode now populate the full stream.
    let expected_shape = [cfg.hc_mult, cfg.hidden_size];
    if h_n.shape != expected_shape && h_n.numel() != cfg.hc_mult * cfg.hidden_size {
        return Err(format!(
            "mtp_forward: h_n shape {:?} != [hc_mult={}, hidden_size={}]",
            h_n.shape, cfg.hc_mult, cfg.hidden_size,
        ));
    }
    if cfg.num_nextn_predict_layers == 0 {
        return Err("mtp_forward: cfg.num_nextn_predict_layers == 0; MTP not enabled".to_string());
    }

    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    // MTP layer occupies the slot just past the main layers; resolve_layer
    // routes the per-layer helpers below to `weights.mtp_layer`.
    let mtp_layer_idx = cfg.num_hidden_layers;

    // ── 1. Lazy state allocation ───────────────────────────────────────
    if state.embed_scratch.is_none() {
        state.embed_scratch = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc embed_scratch: {e:?}"))?,
        );
    }
    if state.residual_streams.is_none() {
        let t = gpu
            .zeros(&[hc_mult, hidden], DType::F32)
            .map_err(|e| format!("alloc residual_streams: {e:?}"))?;
        state.residual_streams = Some(t);
    }
    if state.tmp.is_none() {
        state.tmp = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc tmp: {e:?}"))?,
        );
    }
    if state.mtp_e_norm_scratch.is_none() {
        state.mtp_e_norm_scratch = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc mtp_e_norm_scratch: {e:?}"))?,
        );
    }
    // mtp_h_norm_scratch holds the per-HC-row rmsnorm output, sized
    // [hc_mult, hidden] for the full-HC pipeline. Realloc if shape grew
    // from a legacy [hidden] allocation.
    let h_norm_len = hc_mult * hidden;
    let h_norm_needs_realloc = state
        .mtp_h_norm_scratch
        .as_ref()
        .map(|t| t.numel() != h_norm_len)
        .unwrap_or(true);
    if h_norm_needs_realloc {
        state.mtp_h_norm_scratch = Some(
            gpu.alloc_tensor(&[hc_mult, hidden], DType::F32)
                .map_err(|e| format!("alloc mtp_h_norm_scratch: {e:?}"))?,
        );
    }
    if state.logits.is_none() {
        state.logits = Some(
            gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)
                .map_err(|e| format!("alloc logits: {e:?}"))?,
        );
    }

    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or("mtp_forward: token_embd not uploaded")?;

    // ── 2. Embed next_token → embed_scratch [hidden] ───────────────────
    {
        let embed_scratch = state.embed_scratch.as_ref().unwrap();
        gpu.embedding_lookup_q8(token_embd, embed_scratch, next_token, hidden)
            .map_err(|e| format!("mtp embedding_lookup_q8: {e:?}"))?;
    }

    // ── 3. RMSNorm both inputs ─────────────────────────────────────────
    // e_norm = mtp_enorm(embed)              → mtp_e_norm_scratch [hidden]
    // h_norm = mtp_hnorm(h_n) per HC row    → mtp_h_norm_scratch [hc_mult, hidden]
    {
        let embed_scratch = state.embed_scratch.as_ref().unwrap();
        let e_out = state.mtp_e_norm_scratch.as_ref().unwrap();
        gpu.rmsnorm_f32(embed_scratch, mtp_enorm, e_out, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp rmsnorm_e: {e:?}"))?;
    }
    {
        let h_out = state.mtp_h_norm_scratch.as_ref().unwrap();
        gpu.rmsnorm_batched(h_n, mtp_hnorm, h_out, hc_mult, hidden, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp rmsnorm_h batched: {e:?}"))?;
    }

    // ── 4. Populate residual_streams from full HC plumbing ─────────────
    // Per antirez/ds4 reference:
    //   1. x_e   = mtp_e_proj @ e_norm                  (single [hidden])
    //   2. residual_streams[h] = mtp_h_proj @ h_norm[h] for each HC row h
    //   3. residual_streams[h] += x_e                  (broadcast e_proj)
    //
    // gemv_auto dispatches on the weight's GpuTensor.dtype:
    //   - Q8F16 → gemv_q8_0   (plain input)
    //   - F16   → gemv_f16_xf32 / gemm_f16_x_f16_wmma  (plain input)
    //   - MQ4   → gemv_mq4g256_prerotated              (rotated input)
    //
    // For MQ4 (Raw) MTP weights we'd need FWHT-rotated norm outputs; the
    // upfront dtype check (above) rejects MQ4 e_proj/h_proj for that
    // reason. With Q8/F16/F32 the `x_rotated` argument is unused; we pass
    // mtp_h_norm_scratch (any tensor of correct size) as a dummy.
    {
        let e_norm = state.mtp_e_norm_scratch.as_ref().unwrap();
        let dummy_rotated = state.mtp_h_norm_scratch.as_ref().unwrap();
        let tmp = state.tmp.as_ref().unwrap();
        gemv_auto(gpu, mtp_e_proj, dummy_rotated, e_norm, tmp, hidden, hidden)?;
    }
    {
        let h_norm_full = state.mtp_h_norm_scratch.as_ref().unwrap();
        let streams = state.residual_streams.as_ref().unwrap();
        let dummy_rotated = state.mtp_e_norm_scratch.as_ref().unwrap();
        // Per-HC-row h_proj. mtp_h_proj is the same [hidden, hidden]
        // weight matrix for every row; the inputs differ (per-row h_norm).
        // Tried batched GEMM (B=hc_mult=4) for weight-load amortization —
        // measured 5% SLOWER (17.07 → 16.05 tok/s at K=3) because the
        // batched-chunked Q8 path has setup overhead that beats the
        // amortization at B=4. Keep the per-row loop.
        for h in 0..hc_mult {
            let h_norm_row = h_norm_full.sub_offset(h * hidden, hidden);
            let dst_row = streams.sub_offset(h * hidden, hidden);
            gemv_auto(
                gpu,
                mtp_h_proj,
                dummy_rotated,
                &h_norm_row,
                &dst_row,
                hidden,
                hidden,
            )?;
        }
    }
    // ── 5. Broadcast-add x_e (in state.tmp) into every HC row ─────────
    {
        let streams = state.residual_streams.as_ref().unwrap();
        let src = state.tmp.as_ref().unwrap();
        for h in 0..hc_mult {
            let row = streams.sub_offset(h * hidden, hidden);
            gpu.add_inplace_f32(&row, src)
                .map_err(|e| format!("mtp x_e broadcast-add stream {h}: {e:?}"))?;
        }
    }

    // ── 6. Standard layer block at layer_idx = num_hidden_layers ───────
    // All per-layer helpers below call `weights.resolve_layer(layer_idx)`
    // internally, which routes to `weights.mtp_layer`. The MTP layer has
    // NO compressor/indexer (compress_ratio = 0 by construction), and is
    // NOT a hash layer (mtp_layer_idx >= num_hash_layers), so we use the
    // standard MoE router (`ffn_routed`) rather than `ffn_hash_routed`.
    mhc_pre(
        cfg,
        weights,
        state,
        gpu,
        mtp_layer_idx,
        /*is_attn=*/ true,
    )?;
    q_lora(cfg, weights, state, gpu, mtp_layer_idx)?;
    kv_joint(cfg, weights, state, gpu, mtp_layer_idx)?;
    apply_tail_rope(cfg, weights, state, gpu, position, mtp_layer_idx)?;
    // (No compressor / indexer for MTP — compress_ratio == 0.)
    attn_stub(cfg, weights, state, gpu, mtp_layer_idx)?;
    hc_attn_mix(cfg, weights, state, gpu, mtp_layer_idx)?;
    Ok(())
}

/// Step 7 of the MTP forward: capture the full `[hc_mult, hidden]` residual
/// stream into `state.mtp_last_hidden` (the chaining input to the next MTP
/// iteration). Shared by [`mtp_forward`] and [`mtp_forward_ep`].
fn mtp_capture_hidden(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    // ── 7. Capture FULL [hc_mult, hidden] residual stream for chaining ─
    // Subsequent MTP iterations consume this as their h_n input. The
    // full-HC capture matches the antirez/ds4 reference pattern; legacy
    // stream-0-only capture is what pinned K=2 accept at ~50%.
    {
        let stream_len = hc_mult * hidden;
        let need_realloc = state
            .mtp_last_hidden
            .as_ref()
            .map(|t| t.numel() != stream_len)
            .unwrap_or(true);
        if need_realloc {
            state.mtp_last_hidden = Some(
                gpu.alloc_tensor(&[hc_mult, hidden], DType::F32)
                    .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?,
            );
        }
        let streams = state.residual_streams.as_ref().unwrap();
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &streams.buf, stream_len * 4)
            .map_err(|e| format!("capture full HC → mtp_last_hidden: {e:?}"))?;
    }
    Ok(())
}

/// Step 8 of the MTP forward: final norm (stream-0 or head-HC mix) + lm_head →
/// `state.logits` (NO download). Mirrors the main-model `final_norm_and_head`.
/// Shared by [`mtp_head`] (single-GPU, adds the download) and [`mtp_forward_ep`]
/// (rank 0 only, downloads after an all-ranks sync). The `MTP_SKIP_HEAD`
/// short-circuit lives in the callers (it must skip this entirely).
fn mtp_head_compute(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    let mtp = weights
        .mtp_layer
        .as_ref()
        .ok_or("mtp_head: weights.mtp_layer is None")?;
    let mtp_final = mtp
        .mtp_final_norm
        .as_ref()
        .ok_or("mtp_head: mtp_final_norm missing")?;
    let head = weights.head.as_ref().ok_or("mtp_head: head not uploaded")?;
    // ── 8. final_norm + lm_head → logits ──────────────────────────────
    // Two paths (mirrors the main-model `final_norm_and_head`):
    //   - default (legacy): stream 0 → mtp_final_norm → lm_head.
    //   - HIPFIRE_DEEPSEEK4_MTP_HEAD_HC=1: head-HC mix(streams, mtp.0.hc_head_*)
    //     → mtp_final_norm → lm_head (reduces [hc_mult, hidden] → [hidden]).
    if state.final_norm.is_none() {
        state.final_norm = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc final_norm: {e:?}"))?,
        );
    }
    if state.final_norm_rot.is_none() {
        state.final_norm_rot = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc final_norm_rot: {e:?}"))?,
        );
    }
    // Run head-HC mix or legacy stream-0 path; result lands in
    // `state.final_norm` via rmsnorm.
    let use_head_hc = env_cache::mtp_head_hc_on()
        && mtp.mtp_hc_head_fn.is_some()
        && mtp.mtp_hc_head_base.is_some();
    if use_head_hc {
        if state.head_hc_pre.is_none() {
            state.head_hc_pre = Some(
                gpu.alloc_tensor(&[hc_mult], DType::F32)
                    .map_err(|e| format!("alloc head_hc_pre (mtp): {e:?}"))?,
            );
        }
        if state.head_hc_out.is_none() {
            state.head_hc_out = Some(
                gpu.alloc_tensor(&[hidden], DType::F32)
                    .map_err(|e| format!("alloc head_hc_out (mtp): {e:?}"))?,
            );
        }
        let streams = state.residual_streams.as_ref().unwrap();
        let head_hc_pre = state.head_hc_pre.as_ref().unwrap();
        let head_hc_out = state.head_hc_out.as_ref().unwrap();
        let hc_head_fn = mtp.mtp_hc_head_fn.as_ref().unwrap();
        let hc_head_base = mtp.mtp_hc_head_base.as_ref().unwrap();
        let x_dim = hidden * hc_mult;
        gpu.hc_head_compute_pre(
            streams,
            hc_head_fn,
            hc_head_base,
            head_hc_pre,
            hc_mult as i32,
            x_dim as i32,
            mtp.mtp_hc_head_scale,
            cfg.rms_norm_eps,
            cfg.hc_eps,
        )
        .map_err(|e| format!("mtp hc_head_compute_pre: {e:?}"))?;
        gpu.hc_input_map_4stream(head_hc_pre, streams, head_hc_out, hidden as i32)
            .map_err(|e| format!("mtp hc_input_map: {e:?}"))?;
        let final_norm = state.final_norm.as_ref().unwrap();
        gpu.rmsnorm_f32(head_hc_out, mtp_final, final_norm, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp final rmsnorm (head-HC): {e:?}"))?;
    } else {
        let streams = state.residual_streams.as_ref().unwrap();
        let stream0 = streams.sub_offset(0, hidden);
        let final_norm = state.final_norm.as_ref().unwrap();
        gpu.rmsnorm_f32(&stream0, mtp_final, final_norm, cfg.rms_norm_eps)
            .map_err(|e| format!("mtp final rmsnorm (stream0): {e:?}"))?;
    }
    let final_norm = state.final_norm.as_ref().unwrap();
    let final_norm_rot = state.final_norm_rot.as_ref().unwrap();
    if weight_needs_fwht(head) {
        gpu.rotate_x_mq(final_norm, final_norm_rot, hidden)
            .map_err(|e| format!("mtp rotate final: {e:?}"))?;
    }
    {
        let logits = state.logits.as_ref().unwrap();
        gemv_auto(
            gpu,
            head,
            final_norm_rot,
            final_norm,
            logits,
            cfg.vocab_size,
            hidden,
        )?;
    }

    Ok(())
}

/// Single-GPU MTP head: [`mtp_head_compute`] + download logits → host `Vec`.
/// EP (`mtp_forward_ep`) uses `mtp_head_compute` directly + an all-ranks sync
/// before the caller downloads, to avoid racing the head GEMV on `active_stream`.
fn mtp_head(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<Vec<f32>, String> {
    mtp_head_compute(cfg, weights, state, gpu)?;
    let logits = state.logits.as_ref().unwrap();
    gpu.download_f32(logits)
        .map_err(|e| format!("mtp download logits: {e:?}"))
}

/// EP (Ship 6 substrate-EP) MTP **draft** forward across N ranks for ONE
/// next-next prediction — the spec-decode drafter under expert parallelism.
///
/// Mirror of [`mtp_forward`], fanned across `gpus.devices`: the MTP-specific
/// pre-FFN (embed / norm / HC plumbing + attention) runs replicated per rank,
/// the MTP-layer FFN runs through the SAME EP executor as the main layers
/// (shared `ffn_stub` replicated in `state.ffn_out`; the 256 routed experts
/// sharded → all-reduced partial; `hc_ffn_mix` deferred to
/// `ep_add_into_residual`), the residual capture runs per rank, and the head
/// runs on rank 0. Returns rank 0's downloaded logits (over the next-next
/// vocab). `mtp_last_hidden` is updated per rank (replicated) for chaining.
///
/// `h_n_per_rank[r]` is rank r's previous-position full `[hc_mult, hidden]`
/// residual stream (replicated; the chaining input) — it MUST be a buffer
/// DISTINCT from `state_per_rank[r].residual_streams` (pre-FFN reads `h_n` then
/// overwrites `residual_streams`). Every device needs an `active_stream`
/// ([`hipfire_runtime::ep::ensure_rank_streams`]) + peer access for the
/// all-reduce.
#[allow(clippy::too_many_arguments)]
pub fn mtp_forward_ep(
    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
    weights_per_rank: &[DeepseekV4Weights],
    cfg: &DeepseekV4Config,
    state_per_rank: &mut [DeepseekV4State],
    partials: &[GpuTensor],
    h_n_per_rank: &[GpuTensor],
    next_token: u32,
    position: u32,
) -> Result<Vec<f32>, String> {
    let n = gpus.devices.len();
    assert_eq!(weights_per_rank.len(), n, "mtp_forward_ep: weights_per_rank len");
    assert_eq!(state_per_rank.len(), n, "mtp_forward_ep: state_per_rank len");
    assert_eq!(partials.len(), n, "mtp_forward_ep: partials len");
    assert_eq!(h_n_per_rank.len(), n, "mtp_forward_ep: h_n_per_rank len");
    let hidden = cfg.hidden_size;
    let mtp_layer_idx = cfg.num_hidden_layers;

    // 1. Per-rank pre-FFN (embed/norm/HC + attention), replicated. attn_stub
    //    reads state.n_tokens for the MTP-layer SWA ring slot → set it to
    //    `position` per rank (matches spec_decode's bookkeeping).
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("mtp_forward_ep bind {r}: {e:?}"))?;
        state_per_rank[r].n_tokens = position as u64;
        mtp_pre_ffn(
            cfg,
            &weights_per_rank[r],
            &mut state_per_rank[r],
            &mut gpus.devices[r],
            &h_n_per_rank[r],
            next_token,
            position,
        )?;
    }

    // 2. MTP-layer FFN via the EP executor: a single [Moe] program at
    //    layer_idx = mtp_layer_idx. run_moe_ep = mhc_pre(ffn) + shared ffn_stub
    //    + routed ffn_routed→partial; the executor all-reduces the partial;
    //    ep_add_into_residual = ffn_out += partial, then hc_ffn_mix.
    {
        let program = vec![ds4_superop(SuperOpKind::Moe)];
        let mut binds: Vec<Deepseek4Bindings> = Vec::with_capacity(n);
        for (r, st) in state_per_rank.iter_mut().enumerate() {
            binds.push(Deepseek4Bindings {
                cfg,
                weights: &weights_per_rank[r],
                state: st,
                layer_idx: mtp_layer_idx,
                position,
                token_id: next_token,
                skip_ffn: false,
            });
        }
        hipfire_runtime::ep::run_layer_program_ep(gpus, binds.as_mut_slice(), partials, &program, hidden)
            .map_err(|e| format!("mtp_forward_ep run_layer_program_ep: {e}"))?;
    }

    // 3. Per-rank capture (residual_streams → mtp_last_hidden), replicated.
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("mtp_forward_ep cap bind {r}: {e:?}"))?;
        mtp_capture_hidden(cfg, &mut state_per_rank[r], &mut gpus.devices[r])?;
    }

    // 4. Head COMPUTE on rank 0 (no download — drained by the all-ranks sync).
    gpus.devices[0]
        .bind_thread()
        .map_err(|e| format!("mtp_forward_ep head bind0: {e:?}"))?;
    mtp_head_compute(cfg, &weights_per_rank[0], &mut state_per_rank[0], &mut gpus.devices[0])?;

    // 5. Sync every rank, then download rank 0's logits.
    for r in 0..n {
        gpus.devices[r]
            .bind_thread()
            .map_err(|e| format!("mtp_forward_ep sync bind {r}: {e:?}"))?;
        gpus.devices[r]
            .hip
            .device_synchronize()
            .map_err(|e| format!("mtp_forward_ep sync {r}: {e:?}"))?;
    }
    gpus.devices[0]
        .bind_thread()
        .map_err(|e| format!("mtp_forward_ep dl bind0: {e:?}"))?;
    let logits = state_per_rank[0]
        .logits
        .as_ref()
        .ok_or("mtp_forward_ep: rank0 logits unset")?;
    gpus.devices[0]
        .download_f32(logits)
        .map_err(|e| format!("mtp_forward_ep download logits: {e:?}"))
}

/// Batched twin of `mtp_forward` — processes `batch_size` MTP positions
/// in a single pass through the MTP layer block (Phase A4, 2026-05-22).
///
/// Inputs:
/// - `h_n_streams`: `[batch_size, hc_mult, hidden]` — the per-batch full
///   HC residual streams from the main forward's `pbs.streams_batch`.
/// - `next_tokens`: `[batch_size]` — the next-position tokens (T_{i+1}).
/// - `start_pos`: absolute position of the first batch slot.
///
/// Post-state:
/// - `pbs.streams_batch` contains the per-batch MTP-layer output residuals.
/// - `state.mtp_last_hidden` contains the LAST batch position's MTP
///   output stream (the chaining input to subsequent spec-decode windows).
/// - `state._attention[mtp_layer_idx]` SWA cache has slots for the
///   processed positions written.
///
/// Skips lm_head + logits d2h (only the SWA-fill purpose is exercised).
///
/// At batch_size == 1 this is byte-equivalent to `mtp_forward` modulo
/// FP reduction-order noise inherent to the batched kernels.
#[allow(clippy::too_many_arguments)]
pub fn mtp_forward_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    h_n_streams: &GpuTensor,
    next_tokens: &[u32],
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    if batch_size == 0 {
        return Err("mtp_forward_batched: batch_size == 0".to_string());
    }
    if batch_size > pbs.max_batch {
        return Err(format!(
            "mtp_forward_batched: batch_size {batch_size} > pbs.max_batch {}",
            pbs.max_batch
        ));
    }
    if next_tokens.len() != batch_size {
        return Err(format!(
            "mtp_forward_batched: next_tokens.len {} != batch_size {batch_size}",
            next_tokens.len()
        ));
    }
    let mtp = weights
        .mtp_layer
        .as_ref()
        .ok_or_else(|| "mtp_forward_batched: weights.mtp_layer is None".to_string())?;
    let mtp_enorm = mtp
        .mtp_enorm
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_enorm missing")?;
    let mtp_hnorm = mtp
        .mtp_hnorm
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_hnorm missing")?;
    let mtp_e_proj = mtp
        .mtp_e_proj
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_e_proj missing")?;
    let mtp_h_proj = mtp
        .mtp_h_proj
        .as_ref()
        .ok_or("mtp_forward_batched: mtp_h_proj missing")?;
    for (name, t) in [("mtp_e_proj", mtp_e_proj), ("mtp_h_proj", mtp_h_proj)] {
        match t.dtype {
            DType::F32 | DType::F16 | DType::Q8_0 => {}
            other => {
                return Err(format!(
                    "mtp_forward_batched: {name} dtype {other:?} unsupported"
                ))
            }
        }
    }
    if cfg.num_nextn_predict_layers == 0 {
        return Err("mtp_forward_batched: cfg.num_nextn_predict_layers == 0".to_string());
    }

    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;
    let stream_len = hc_mult * hidden;
    let mtp_layer_idx = cfg.num_hidden_layers;

    // Lazy alloc state.mtp_last_hidden.
    if state.mtp_last_hidden.is_none() {
        state.mtp_last_hidden = Some(
            gpu.alloc_tensor(&[hc_mult, hidden], DType::F32)
                .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?,
        );
    }

    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or("mtp_forward_batched: token_embd not uploaded")?;

    // ── 1. Upload next_tokens [batch_size] ─────────────────────────────
    let tokens_host: Vec<i32> = next_tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(tokens_host.as_ptr() as *const u8, batch_size * 4) };
    gpu.memcpy_htod_auto(&pbs.mtp_tokens_batch.buf, token_bytes)
        .map_err(|e| format!("mtp_forward_batched htod tokens: {e:?}"))?;

    // ── 2. Batched embed → pbs.mtp_embed_batch ─────────────────────────
    gpu.embedding_lookup_q8_batched(
        token_embd,
        &pbs.mtp_embed_batch,
        &pbs.mtp_tokens_batch,
        batch_size,
        hidden,
    )
    .map_err(|e| format!("mtp embedding_lookup_q8_batched: {e:?}"))?;

    // ── 3. Batched RMSNorm both inputs ─────────────────────────────────
    // e_norm = mtp_enorm(embed_batch) → mtp_e_norm_batch [B, hidden]
    gpu.rmsnorm_batched(
        &pbs.mtp_embed_batch,
        mtp_enorm,
        &pbs.mtp_e_norm_batch,
        batch_size,
        hidden,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("mtp rmsnorm_e batched: {e:?}"))?;
    // h_norm = mtp_hnorm(h_n_streams) per (batch, HC row) — treat as
    // batch_size * hc_mult rows of length hidden.
    gpu.rmsnorm_batched(
        h_n_streams,
        mtp_hnorm,
        &pbs.mtp_h_norm_batch,
        batch_size * hc_mult,
        hidden,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("mtp rmsnorm_h batched: {e:?}"))?;

    // ── 4. Batched e_proj GEMV: mtp_e_norm_batch → mtp_x_e_batch ───────
    // dummy_rotated is unused for F32/F16/Q8 weight dtypes (guarded above).
    gemv_auto_batched_wmma(
        gpu,
        mtp_e_proj,
        &pbs.mtp_h_norm_batch,
        &pbs.mtp_e_norm_batch,
        &pbs.mtp_x_e_batch,
        hidden,
        hidden,
        batch_size,
        None,
    )?;

    // ── 5. Batched h_proj GEMV — flatten (B, hc_mult) as one batch dim.
    // Input mtp_h_norm_batch [B * hc_mult, hidden] → output streams_batch
    // [B * hc_mult, hidden]. mtp_h_proj is the same weight for every
    // (batch, HC) row.
    gemv_auto_batched_wmma(
        gpu,
        mtp_h_proj,
        &pbs.mtp_e_norm_batch,
        &pbs.mtp_h_norm_batch,
        &pbs.streams_batch,
        hidden,
        hidden,
        batch_size * hc_mult,
        None,
    )?;

    // ── 6. Broadcast-add x_e_b into every HC row of streams_batch_b ───
    // streams_batch[b][h] += mtp_x_e_batch[b] for h in 0..hc_mult, b in 0..B.
    for b in 0..batch_size {
        let x_e_b = pbs.mtp_x_e_batch.sub_offset(b * hidden, hidden);
        for h in 0..hc_mult {
            let off = b * stream_len + h * hidden;
            let row = pbs.streams_batch.sub_offset(off, hidden);
            gpu.add_inplace_f32(&row, &x_e_b)
                .map_err(|e| format!("mtp x_e add b={b} h={h}: {e:?}"))?;
        }
    }

    // ── 7. Populate per-batch positions + attn_state for the MTP layer.
    //   Positions: start_pos + b.
    //   attn_state: slot = (start_pos + b) % swa_window; n_valid = min(start_pos + b + 1, swa_window).
    precompute_positions_batched(cfg, pbs, gpu, start_pos, batch_size)?;
    precompute_attn_state_batched(cfg, pbs, gpu, start_pos, batch_size)?;

    // ── 8. Standard batched layer block at layer_idx = mtp_layer_idx ──
    // The MTP layer has compress_ratio = 0 so attention_block_batched_swa_only
    // is the right path. Hash routing is N/A (mtp_layer_idx >= num_hash_layers).
    let n = batch_size;
    mhc_pre_batched(
        cfg,
        weights,
        pbs,
        gpu,
        mtp_layer_idx,
        /*is_attn=*/ true,
        n,
    )?;
    q_lora_batched(cfg, weights, pbs, &pbs.hc_x_in_batch, gpu, mtp_layer_idx, n)?;
    kv_joint_batched(cfg, weights, pbs, gpu, mtp_layer_idx, n)?;
    apply_tail_rope_batched(cfg, weights, pbs, gpu, mtp_layer_idx, n)?;
    attention_block_batched_swa_only(cfg, weights, state, pbs, gpu, mtp_layer_idx, start_pos, n)?;
    hc_attn_mix_batched(cfg, pbs, gpu, n)?;
    mhc_pre_batched(
        cfg,
        weights,
        pbs,
        gpu,
        mtp_layer_idx,
        /*is_attn=*/ false,
        n,
    )?;
    // ffn_batched takes `tokens` for the hash-routed path; MTP layer is
    // not hash-routed (mtp_layer_idx >= num_hash_layers), so the value
    // is ignored. Pass an empty slice.
    let tokens_dummy: &[u32] = &[];
    ffn_batched(cfg, weights, pbs, gpu, mtp_layer_idx, n, tokens_dummy)?;
    hc_ffn_mix_batched(cfg, pbs, gpu, n)?;

    // ── 9. Capture the LAST batch position's residual stream → mtp_last_hidden.
    //    Subsequent spec-decode windows read from this.
    {
        let last_off = (batch_size - 1) * stream_len;
        let last_slice = pbs.streams_batch.sub_offset(last_off, stream_len);
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &last_slice.buf, stream_len * 4)
            .map_err(|e| format!("mtp d2d streams[last] → mtp_last_hidden: {e:?}"))?;
    }

    Ok(())
}

/// FFN block (partial — shared expert only; routed experts pending).
///
/// DeepSeek V4 has one shared expert + 256 routed experts (top-6 selected
/// per token). The shared expert is a standard SwiGLU:
///   gate = x @ shared_w1   [moe_intermediate=2048]
///   up   = x @ shared_w3   [moe_intermediate]
///   silu_gated = silu(gate) * up
///   out  = silu_gated @ shared_w2   [hidden]
///
/// Then x_ffn = shared_out + routed_scaling_factor * routed_out.
/// Routed_out is currently 0 (router/expert dispatch pending), so
/// ffn_out = shared_out.
fn ffn_stub(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let shared_w1 = layer.shared_w1.as_ref().unwrap();
    let shared_w2 = layer.shared_w2.as_ref().unwrap();
    let shared_w3 = layer.shared_w3.as_ref().unwrap();
    let hc_x_in = state.hc_x_in.as_ref().unwrap();

    let im = cfg.moe_intermediate_size;
    if state.ffn_out.is_none() {
        state.ffn_out = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc ffn_out: {e:?}"))?,
        );
    }
    if state.ffn_x_rot.is_none() {
        state.ffn_x_rot = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc ffn_x_rot: {e:?}"))?,
        );
    }
    if state.ffn_gate.is_none() {
        state.ffn_gate = Some(
            gpu.alloc_tensor(&[im], DType::F32)
                .map_err(|e| format!("alloc ffn_gate: {e:?}"))?,
        );
    }
    if state.ffn_up.is_none() {
        state.ffn_up = Some(
            gpu.alloc_tensor(&[im], DType::F32)
                .map_err(|e| format!("alloc ffn_up: {e:?}"))?,
        );
    }
    if state.ffn_silu_rot.is_none() {
        state.ffn_silu_rot = Some(
            gpu.alloc_tensor(&[im], DType::F32)
                .map_err(|e| format!("alloc ffn_silu_rot: {e:?}"))?,
        );
    }
    if state.ffn_x_plain.is_none() {
        state.ffn_x_plain = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc ffn_x_plain: {e:?}"))?,
        );
    }

    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let ffn_x_plain = state.ffn_x_plain.as_ref().unwrap();
    let gate = state.ffn_gate.as_ref().unwrap();
    let up = state.ffn_up.as_ref().unwrap();
    let silu_rot = state.ffn_silu_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    // Skip FWHT rotations when downstream weight dtype doesn't need
    // them (Q8/F16/F32 paths read x_plain). For deepseek4-q8-mtp this skips
    // ~2-3 rotation kernels per layer per token.
    //
    // CORRECTNESS: the routed-MoE path (ffn_routed) ALSO reads
    // ffn_x_rot — routed experts at MQ2-Lloyd consume FWHT-rotated
    // input. So we must keep the gate/up rotation alive when MoE is
    // on (default; opt out with HIPFIRE_DEEPSEEK4_MOE=0), regardless of shared
    // weight dtype.
    let moe_will_run = env_cache::moe_on();
    let gate_up_need_fwht =
        moe_will_run || weight_needs_fwht(shared_w1) || weight_needs_fwht(shared_w3);
    let down_needs_fwht = weight_needs_fwht(shared_w2);

    // 1. RMSNorm (+ optional FWHT). When BOTH rot and plain outputs are
    //    needed (common case: MoE on OR shared_w1/w3 are MQ4), use the
    //    fused single-launch variant that writes both. Saves one launch
    //    + the duplicate sum-of-squares pass.
    if gate_up_need_fwht {
        gpu.fused_rmsnorm_rotate_mq_plain(
            hc_x_in,
            ffn_norm,
            ffn_x_rot,
            ffn_x_plain,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .map_err(|e| format!("fused_rmsnorm_rotate_mq_plain ffn layer {layer_idx}: {e:?}"))?;
    } else {
        // Pure plain path (no MoE AND shared_w1/w3 not MQ4): only need
        // ffn_x_plain.
        gpu.rmsnorm_f32(hc_x_in, ffn_norm, ffn_x_plain, cfg.rms_norm_eps)
            .map_err(|e| format!("rmsnorm_f32 ffn-side plain l{layer_idx}: {e:?}"))?;
    }

    // 2. gate = x @ shared_w1
    gemv_auto(
        gpu,
        shared_w1,
        ffn_x_rot,
        ffn_x_plain,
        gate,
        im,
        cfg.hidden_size,
    )?;

    // 3. up = x @ shared_w3
    gemv_auto(
        gpu,
        shared_w3,
        ffn_x_rot,
        ffn_x_plain,
        up,
        im,
        cfg.hidden_size,
    )?;

    // 4-5. DeepSeek V4 SwiGLU with swiglu_limit clamp, optionally fused with
    //      the FWHT rotation when shared_w2 is MQ4. The fused kernel
    //      saves one launch + the 8 KB intermediate write/read of
    //      `gate`. cfg.swiglu_limit = 10.0 on DeepSeek V4. Same Expert class
    //      used for shared and routed in upstream model.py.
    if down_needs_fwht {
        gpu.deepseek4_fused_silu_mul_clamp_mq_rotate(gate, up, silu_rot, im, cfg.swiglu_limit)
            .map_err(|e| format!("deepseek4_fused_silu_mul_clamp_mq_rotate layer {layer_idx}: {e:?}"))?;
    } else {
        gpu.deepseek4_silu_mul_clamp_f32(gate, up, gate, cfg.swiglu_limit)
            .map_err(|e| format!("deepseek4_silu_mul_clamp layer {layer_idx}: {e:?}"))?;
    }

    // 6. ffn_out = silu_rot @ shared_w2 (down: [hidden, im])
    // shared_w2: rotated path uses silu_rot (FWHT'd), plain path uses
    // `gate` itself (post-silu_mul, no FWHT).
    gemv_auto(gpu, shared_w2, silu_rot, gate, ffn_out, cfg.hidden_size, im)?;

    Ok(())
}

/// Routed-expert dispatch (DeepSeek V4 top-6 MoE). Accumulates `routed_scaling
/// _factor · Σ_k w_k · expert_{idx_k}(ffn_x_rot)` into `ffn_out`
/// (which already holds the shared-expert output from `ffn_stub`).
///
/// Gated on HIPFIRE_DEEPSEEK4_MOE != "0" (default ON) AND expert blobs present
/// (uploaded by default; opt out with HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0) AND
/// layer is score-routed
/// (layer_idx >= num_hash_layers). Hash-routed layers 0..3 fall back
/// to shared-only (tid2eid lookup table is skipped at quant time).
///
/// Math (per upstream `inference/model.py:Gate.forward` and `Expert.
/// forward`):
///   scores = sqrt(softplus(gate.weight @ x))             [n_exp]
///   indices = topk(scores + bias, k=6)[1]                [k]   ← +bias for selection
///   weights = scores[indices]                            [k]   ← unbiased scores for weights
///   weights /= weights.sum(); weights *= route_scale     [k]
///   for each (idx, w) in (indices, weights):
///     gate_e = w1[idx] @ x                  ← clamp to swiglu_limit (skipped)
///     up_e   = w3[idx] @ x                  ← clamp to ±swiglu_limit (skipped)
///     e_out  = w2[idx] @ (silu(gate_e) * up_e * w)
///     ffn_out += e_out * routed_scaling_factor
fn ffn_routed(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    routed_out: Option<&GpuTensor>,
) -> Result<(), String> {
    if !env_cache::moe_on() {
        return Ok(());
    }
    if layer_idx < cfg.num_hash_layers {
        // Hash routing — tid2eid table skipped at quant time, no expert
        // selection possible. Shared expert alone for these layers.
        return Ok(());
    }
    let layer = weights.resolve_layer(layer_idx);
    if layer.expert_gate_up_blob.is_none() || layer.expert_w2_blob.is_none() {
        return Ok(()); // experts not uploaded; nothing to dispatch
    }

    // 1. Run router: compute unbiased scores on-device. DeepSeek V4's selection
    //    uses BIASED scores while the routing weights use UNBIASED scores
    //    (per upstream model.py: Gate.forward). The GPU top-K kernel
    //    `deepseek4_moe_topk_bias_aware_f32` handles this two-score semantic in
    //    one launch, eliminating the per-layer D2H/CPU/H2D round-trip.
    moe_route(cfg, weights, state, gpu, layer_idx)?;

    let k = cfg.num_experts_per_tok;
    let n_exp = cfg.n_routed_experts;
    let _ = n_exp;
    let im = cfg.moe_intermediate_size;
    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();
    // Route-scale: rarely-overridden; one-shot env read at first call.
    use std::sync::OnceLock;
    static ROUTE_SCALE: OnceLock<f32> = OnceLock::new();
    let route_scale_override: f32 = *ROUTE_SCALE.get_or_init(|| {
        std::env::var("HIPFIRE_DEEPSEEK4_ROUTE_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.2)
    });

    if layer.expert_gate_up_blob.is_some() {
        // Fused MoE dispatch: 2 indexed kernels (gate_up + down) plus
        // k_top per-expert silu_clamp+rotate. Replaces the per-expert
        // k=0..6 × 3 GEMV loop (18 launches → 14 launches per layer).
        // The bigger win is GPU utilisation: grid Y dim spans all k_top
        // experts so the GEMVs run in parallel rather than serially.
        let k_top = k;
        // Lazy-alloc scratch.
        if state.moe_topk_indices.is_none() {
            state.moe_topk_indices = Some(
                gpu.alloc_tensor(&[k_top], DType::F32)
                    .map_err(|e| format!("alloc moe_topk_indices: {e:?}"))?,
            );
        }
        if state.moe_topk_weights.is_none() {
            state.moe_topk_weights = Some(
                gpu.alloc_tensor(&[k_top], DType::F32)
                    .map_err(|e| format!("alloc moe_topk_weights: {e:?}"))?,
            );
        }
        if state.moe_gate_batch.is_none() {
            state.moe_gate_batch = Some(
                gpu.alloc_tensor(&[k_top, im], DType::F32)
                    .map_err(|e| format!("alloc moe_gate_batch: {e:?}"))?,
            );
        }
        if state.moe_up_batch.is_none() {
            state.moe_up_batch = Some(
                gpu.alloc_tensor(&[k_top, im], DType::F32)
                    .map_err(|e| format!("alloc moe_up_batch: {e:?}"))?,
            );
        }
        if state.moe_rot_batch.is_none() {
            state.moe_rot_batch = Some(
                gpu.alloc_tensor(&[k_top, im], DType::F32)
                    .map_err(|e| format!("alloc moe_rot_batch: {e:?}"))?,
            );
        }
        // [k_top × hidden] per-expert down outputs for the deterministic
        // (atomic-free) combine in run_moe_decode_bias_aware (default on;
        // HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC=0 uses the atomic path).
        if state.moe_down_expert_outputs.is_none() {
            state.moe_down_expert_outputs = Some(
                gpu.alloc_tensor(&[k_top, cfg.hidden_size], DType::F32)
                    .map_err(|e| format!("alloc moe_down_expert_outputs: {e:?}"))?,
            );
        }
        let topk_idx_dev = state.moe_topk_indices.as_ref().unwrap();
        let topk_w_dev = state.moe_topk_weights.as_ref().unwrap();
        // GPU top-K: bias-aware select + normalize + route_scale in one
        // launch, outputs straight into topk_idx_dev / topk_w_dev.
        let scores_dev = state.router_scores.as_ref().unwrap();
        let bias_dev = layer
            .gate_bias
            .as_ref()
            .ok_or_else(|| format!("ffn_routed l{layer_idx}: gate_bias missing"))?;
        let gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().unwrap();
        let w2_ptrs = layer.expert_w2_ptrs.as_ref().unwrap();
        let gate_batch = state.moe_gate_batch.as_ref().unwrap();
        let up_batch = state.moe_up_batch.as_ref().unwrap();
        let rot_batch = state.moe_rot_batch.as_ref().unwrap();
        let down_expanded = state.moe_down_expert_outputs.as_ref().unwrap();

        // Bias-aware top-k select + the routed MQ2-Lloyd experts now run through
        // the centralized MoE family (Ship 4.3): bias-aware top-k -> indexed
        // gate_up -> batched silu*mul*clamp -> batched FWHT rotate -> indexed
        // down with route-scaled residual accumulation into ffn_out. The router
        // GEMV + sqrt_softplus (moe_route, above) and the shared expert
        // (ffn_stub) stay model-owned; ffn_stub must have seeded ffn_out before
        // this accumulates into it.
        //
        // EP: `routed_out = Some(partial)` redirects the route-scaled
        // accumulation into a zeroed per-rank partial (so partial holds ONLY
        // this rank's owned-expert routed contribution); the shared expert
        // stays in `state.ffn_out` (replicated). The accumulation kernel does
        // `out += ...`, so a zeroed partial yields exactly the routed sum.
        let out_target = routed_out.unwrap_or(ffn_out);
        let moe_params = hipfire_dispatch::families::moe::MoeBiasAwareParams {
            hidden: cfg.hidden_size,
            mi: im,
            k_top,
            n_exp: cfg.n_routed_experts,
            route_scale: route_scale_override,
            swiglu_limit: cfg.swiglu_limit,
            batch_size: 1,
            x_rot: ffn_x_rot,
            ffn_out: out_target,
            scores: scores_dev,
            gate_bias: bias_dev,
            expert_gate_up_ptrs: gate_up_ptrs,
            expert_down_ptrs: w2_ptrs,
            topk_indices: topk_idx_dev,
            topk_weights: topk_w_dev,
            gate_batch,
            up_batch,
            rot_batch,
            down_expanded,
        };
        hipfire_runtime::llama::moe_family()
            .run_bias_aware(gpu, &moe_params)
            .map_err(|e| format!("ffn_routed l{layer_idx} dispatch: {e}"))?;

        return Ok(());
    }

    // Per-expert fallback path is no longer reachable: separate w1/w3
    // blobs are no longer uploaded (only the combined gate_up blob).
    let _ = route_scale_override;
    Err(format!(
        "deepseek4: layer {layer_idx} has no separate w1/w3 blobs (only \
         combined gate_up). Rebuild the loader with separate-blob uploads."
    ))
}

/// Hash-routed FFN dispatch (DeepSeek V4 layers 0..num_hash_layers = 0..3).
///
/// Per upstream DeepSeek V4 (model.py:Gate.forward, model.py:587-606):
///   if self.hash:
///     indices = self.tid2eid[input_ids]          [k]   ← static lookup
///   else:
///     indices = scores.topk(k)[1]
///   weights = original_scores.gather(1, indices) [k]   ← from unbiased scores
///   weights /= weights.sum();  weights *= route_scale
///
/// So we still need the gate.weight GEMV to get scores for the weight
/// values — only the SELECTION is static. The dispatch loop is otherwise
/// identical to `ffn_routed`.
///
/// Same env gate (`HIPFIRE_DEEPSEEK4_MOE != "0"`, default ON) and blob-presence guard.
fn ffn_hash_routed(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    token_id: u32,
    routed_out: Option<&GpuTensor>,
) -> Result<(), String> {
    if !env_cache::moe_on() {
        return Ok(());
    }
    let layer = weights.resolve_layer(layer_idx);
    if layer.expert_gate_up_blob.is_none() || layer.expert_w2_blob.is_none() {
        return Ok(());
    }
    if layer.tid2eid_host.is_empty() {
        // tid2eid not in the HFQ (pre-FP4-fix quant skipped it). Fall back
        // to shared-only on this layer.
        return Ok(());
    }

    // Compute scores (unbiased) on-device for the weight values.
    moe_route(cfg, weights, state, gpu, layer_idx)?;

    let k = cfg.num_experts_per_tok;
    let n_exp = cfg.n_routed_experts;

    // Bounds check on token_id (host-side; tid2eid_dev shape == tid2eid_host).
    let row = (token_id as usize) * k;
    if row + k > layer.tid2eid_host.len() {
        return Err(format!(
            "hash l{layer_idx}: token_id {token_id} out of tid2eid range \
             ({} entries)",
            layer.tid2eid_host.len()
        ));
    }

    let im = cfg.moe_intermediate_size;
    let ffn_x_rot = state.ffn_x_rot.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();
    let route_scale_override: f32 = std::env::var("HIPFIRE_DEEPSEEK4_ROUTE_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.2);
    let k_top = k;

    // Lazy-alloc moe scratch (shared with ffn_routed via state).
    if state.moe_topk_indices.is_none() {
        state.moe_topk_indices = Some(
            gpu.alloc_tensor(&[k_top], DType::F32)
                .map_err(|e| format!("alloc moe_topk_indices hash: {e:?}"))?,
        );
    }
    if state.moe_topk_weights.is_none() {
        state.moe_topk_weights = Some(
            gpu.alloc_tensor(&[k_top], DType::F32)
                .map_err(|e| format!("alloc moe_topk_weights hash: {e:?}"))?,
        );
    }
    if state.moe_gate_batch.is_none() {
        state.moe_gate_batch = Some(
            gpu.alloc_tensor(&[k_top, im], DType::F32)
                .map_err(|e| format!("alloc moe_gate_batch hash: {e:?}"))?,
        );
    }
    if state.moe_up_batch.is_none() {
        state.moe_up_batch = Some(
            gpu.alloc_tensor(&[k_top, im], DType::F32)
                .map_err(|e| format!("alloc moe_up_batch hash: {e:?}"))?,
        );
    }
    if state.moe_rot_batch.is_none() {
        state.moe_rot_batch = Some(
            gpu.alloc_tensor(&[k_top, im], DType::F32)
                .map_err(|e| format!("alloc moe_rot_batch hash: {e:?}"))?,
        );
    }

    let topk_idx_dev = state.moe_topk_indices.as_ref().unwrap();
    let topk_w_dev = state.moe_topk_weights.as_ref().unwrap();
    let scores = state.router_scores.as_ref().unwrap();

    // GPU-side hash-router lookup + normalize + scale. Replaces the
    // d2h(scores) + host gather + h2d(topk_idx, topk_w) round-trip.
    // Prefer the `_buf` variant (reads token_id from device) so the
    // captured HIP graph re-reads token_id on every replay. Falls
    // back to the kernarg variant or host gather if prerequisites
    // (tid2eid_dev, token_id_buf) are missing.
    if let Some(tid2eid_dev) = layer.tid2eid_dev.as_ref() {
        if let Some(token_id_buf) = state.token_id_buf.as_ref() {
            gpu.hash_router_normalize_f32_buf(
                tid2eid_dev,
                scores,
                token_id_buf,
                topk_idx_dev,
                topk_w_dev,
                n_exp as i32,
                k as i32,
                route_scale_override,
            )
            .map_err(|e| format!("hash_router_normalize_buf hash l{layer_idx}: {e:?}"))?;
        } else {
            gpu.hash_router_normalize_f32(
                tid2eid_dev,
                scores,
                topk_idx_dev,
                topk_w_dev,
                token_id as i32,
                n_exp as i32,
                k as i32,
                route_scale_override,
            )
            .map_err(|e| format!("hash_router_normalize hash l{layer_idx}: {e:?}"))?;
        }
    } else {
        // Fallback: d2h + host gather + h2d.
        let scores_host = gpu
            .download_f32(scores)
            .map_err(|e| format!("d2h scores hash l{layer_idx}: {e:?}"))?;
        let topk_ids: Vec<u32> = layer.tid2eid_host[row..row + k]
            .iter()
            .map(|&i| i.min((n_exp - 1) as u32))
            .collect();
        let wts = match gather_normalized_weights(&scores_host, &topk_ids) {
            Some(w) => w,
            None => return Ok(()),
        };
        let idx_i32: Vec<i32> = topk_ids.iter().map(|&x| x as i32).collect();
        let idx_bytes: Vec<u8> = idx_i32.iter().flat_map(|i| i.to_le_bytes()).collect();
        gpu.memcpy_htod_auto(&topk_idx_dev.buf, &idx_bytes)
            .map_err(|e| format!("htod topk_indices hash l{layer_idx}: {e:?}"))?;
        let w_scaled: Vec<f32> = wts.iter().map(|&w| w * route_scale_override).collect();
        let w_bytes: Vec<u8> = w_scaled.iter().flat_map(|w| w.to_le_bytes()).collect();
        gpu.memcpy_htod_auto(&topk_w_dev.buf, &w_bytes)
            .map_err(|e| format!("htod topk_weights hash l{layer_idx}: {e:?}"))?;
    }

    let gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().unwrap();
    let w2_ptrs = layer.expert_w2_ptrs.as_ref().unwrap();
    let gate_batch = state.moe_gate_batch.as_ref().unwrap();
    let up_batch = state.moe_up_batch.as_ref().unwrap();
    let rot_batch = state.moe_rot_batch.as_ref().unwrap();

    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        gate_up_ptrs,
        topk_idx_dev,
        ffn_x_rot,
        gate_batch,
        up_batch,
        2 * im,
        cfg.hidden_size,
        k_top,
    )
    .map_err(|e| format!("fused gate_up hash l{layer_idx}: {e:?}"))?;

    gpu.deepseek4_silu_mul_clamp_f32_batched(
        gate_batch,
        up_batch,
        gate_batch,
        im,
        k_top,
        cfg.swiglu_limit,
    )
    .map_err(|e| format!("deepseek4_silu_mul_clamp batched hash l{layer_idx}: {e:?}"))?;
    gpu.rotate_x_mq_batched(gate_batch, rot_batch, im, k_top)
        .map_err(|e| format!("rotate batched hash l{layer_idx}: {e:?}"))?;

    // EP: redirect the route-scaled accumulation into the zeroed partial
    // (routed-only) instead of state.ffn_out (shared+routed). The down kernel
    // accumulates `out += w_k · down_k`, so a zeroed partial yields exactly
    // this rank's owned routed contribution.
    let out_target = routed_out.unwrap_or(ffn_out);
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        w2_ptrs,
        topk_idx_dev,
        topk_w_dev,
        rot_batch,
        out_target,
        cfg.hidden_size,
        im,
        k_top,
    )
    .map_err(|e| format!("fused down hash l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// HC FFN mix — same pattern as `hc_attn_mix` but with `hc_ffn_*`
/// tensors and `ffn_out` as transform_out.
fn hc_ffn_mix(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let _ = (weights, layer_idx);
    let streams = state.residual_streams.as_ref().unwrap();
    let ffn_out = state.ffn_out.as_ref().unwrap();

    // Same reasoning as hc_attn_mix: mhc_pre(is_attn=false) has
    // already populated state.hc_c with the FFN block's post and comb
    // (α-scaled, sigmoid'd, sinkhorn'd). Just consume them.
    let post_view = state.hc_c.as_ref().unwrap().sub_offset(4, 4);
    let comb_view = state.hc_c.as_ref().unwrap().sub_offset(8, 16);

    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(
        streams,
        &comb_view,
        &post_view,
        ffn_out,
        streams_out,
        cfg.hidden_size as i32,
    )
    .map_err(|e| format!("hc_mix_4stream ffn: {e:?}"))?;

    let bytes = cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&streams.buf, &streams_out.buf, bytes)
        .map_err(|e| format!("d2d hc_ffn_mix → streams: {e:?}"))?;
    Ok(())
}

/// We were previously taking ONLY stream 0 for the head — discarding 75%
/// of the model's output state. This wires the full HC mix.
/// Steps 1–4 of the head pipeline (head-HC mix, MTP h_n capture, final
/// RMSNorm, and the FWHT rotation for an MQ4 head), leaving the pre-lm_head
/// activation in `state.final_norm` (and `state.final_norm_rot` when the head
/// needs FWHT). Split out of `final_norm_and_head` so the batched verify path
/// can run this cheap per-position prologue K times, then issue ONE batched
/// lm_head GEMV — reading the `[vocab, hidden]` weight once instead of K times.
fn final_norm_compute(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    let output_norm = weights
        .output_norm
        .as_ref()
        .ok_or_else(|| "output_norm not uploaded".to_string())?;
    let head = weights
        .head
        .as_ref()
        .ok_or_else(|| "head not uploaded".to_string())?;
    let hc_head_fn = weights
        .hc_head_fn
        .as_ref()
        .ok_or_else(|| "hc_head_fn not uploaded".to_string())?;
    let hc_head_base = weights
        .hc_head_base
        .as_ref()
        .ok_or_else(|| "hc_head_base not uploaded".to_string())?;
    let streams = state.residual_streams.as_ref().unwrap();

    if state.final_norm.is_none() {
        state.final_norm = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc final_norm: {e:?}"))?,
        );
    }
    if state.final_norm_rot.is_none() {
        state.final_norm_rot = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc final_norm_rot: {e:?}"))?,
        );
    }
    if state.head_hc_pre.is_none() {
        state.head_hc_pre = Some(
            gpu.alloc_tensor(&[cfg.hc_mult], DType::F32)
                .map_err(|e| format!("alloc head_hc_pre: {e:?}"))?,
        );
    }
    if state.head_hc_out.is_none() {
        state.head_hc_out = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc head_hc_out: {e:?}"))?,
        );
    }

    let final_norm = state.final_norm.as_ref().unwrap();
    let final_norm_rot = state.final_norm_rot.as_ref().unwrap();
    let head_hc_pre = state.head_hc_pre.as_ref().unwrap();
    let head_hc_out = state.head_hc_out.as_ref().unwrap();

    // 1. Head HC: compute pre[hc_mult] = sigmoid((hc_head_fn @ x_flat * rsqrt) * scale + base) + eps
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    gpu.hc_head_compute_pre(
        streams,
        hc_head_fn,
        hc_head_base,
        head_hc_pre,
        cfg.hc_mult as i32,
        x_dim as i32,
        weights.hc_head_scale,
        cfg.rms_norm_eps,
        cfg.hc_eps,
    )
    .map_err(|e| format!("hc_head_compute_pre: {e:?}"))?;

    // 2. Head HC combine: head_hc_out[d] = sum_h pre[h] * streams[h, d]
    gpu.hc_input_map_4stream(head_hc_pre, streams, head_hc_out, cfg.hidden_size as i32)
        .map_err(|e| format!("hc_input_map (head): {e:?}"))?;

    // 2.5. Capture h_n for downstream MTP / spec-decode.
    //
    // DeepSeek V4 MTP consumes the FULL [hc_mult, hidden] HC stream of the
    // previous position, not stream 0 alone (per antirez/ds4 reference
    // `metal_graph_eval_mtp_draft_from_hc`, ds4.c:12852). The prior
    // stream-0-only capture discarded 75% of the HC signal and pinned
    // K=2 acceptance at ~50%.
    let mtp_hidden_len = cfg.hc_mult * cfg.hidden_size;
    let mtp_needs_realloc = state
        .mtp_last_hidden
        .as_ref()
        .map(|t| t.numel() != mtp_hidden_len)
        .unwrap_or(true);
    if mtp_needs_realloc {
        state.mtp_last_hidden = Some(
            gpu.alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc mtp_last_hidden in final_norm_and_head: {e:?}"))?,
        );
    }
    {
        let dst = state.mtp_last_hidden.as_ref().unwrap();
        gpu.memcpy_dtod_auto(&dst.buf, &streams.buf, mtp_hidden_len * 4)
            .map_err(|e| format!("capture full HC streams → mtp_last_hidden: {e:?}"))?;
    }

    // 3. RMSNorm of the combined stream output.
    gpu.rmsnorm_f32(head_hc_out, output_norm, final_norm, cfg.rms_norm_eps)
        .map_err(|e| format!("final rmsnorm_f32: {e:?}"))?;

    // 4. FWHT-rotate for MQ4 GEMV — skip if lm_head is Q8/F16/F32.
    if weight_needs_fwht(head) {
        gpu.rotate_x_mq(final_norm, final_norm_rot, cfg.hidden_size)
            .map_err(|e| format!("rotate_x_mq final_norm: {e:?}"))?;
    }

    Ok(())
}

/// Full per-position head: `final_norm_compute` followed by the lm_head GEMV
/// into `state.logits`. Behaviour unchanged from before the split.
fn final_norm_and_head(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
) -> Result<(), String> {
    final_norm_compute(cfg, weights, state, gpu)?;

    let head = weights
        .head
        .as_ref()
        .ok_or_else(|| "head not uploaded".to_string())?;
    if state.logits.is_none() {
        state.logits = Some(
            gpu.alloc_tensor(&[cfg.vocab_size], DType::F32)
                .map_err(|e| format!("alloc logits: {e:?}"))?,
        );
    }
    let final_norm = state.final_norm.as_ref().unwrap();
    let final_norm_rot = state.final_norm_rot.as_ref().unwrap();
    let logits = state.logits.as_ref().unwrap();

    // lm_head GEMV. F16 path uses un-rotated final_norm.
    gemv_auto(
        gpu,
        head,
        final_norm_rot,
        final_norm,
        logits,
        cfg.vocab_size,
        cfg.hidden_size,
    )?;

    Ok(())
}

/// Step 6: Single-position attention (position-0 degenerate case).
///
/// DeepSeek V4's attention with `o_groups = 8` means the 64 query heads
/// are reduced over groups of 8 heads → 8 grouped outputs each of
/// `head_dim = 512`, yielding `[8 * 512 = 4096]` = hidden directly.
/// No separate O-projection needed (wo_a/wo_b's role TBD per paper).
///
/// For position-0 (no past KV history), each query head attends
/// only to the current token's K/V. softmax over 1 position = 1.0,
/// so attn_per_head = V. With o_groups grouping: each of 8 groups
/// sums 8 identical V vectors → attn_per_group = 8 * V.
///
/// Output `[hidden = o_groups * head_dim]`: 8 copies of V (each
/// scaled by 8 due to the in-group sum), giving [8*V, 8*V, ..., 8*V].
///
/// This handles position 0. For position > 0 we need SWA cache +
/// real Q·K·V over history — pending.
fn attn_stub(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    // Final attention contribution: shape [hidden]. Consumed by hc_attn_mix.
    if state.attn_out.is_none() {
        state.attn_out = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc attn_out: {e:?}"))?,
        );
    }
    // Raw attention output [n_heads, head_dim] — kernel writes here.
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let n_heads_head_dim = n_heads * head_dim;
    if state.attn_out_raw.is_none() {
        state.attn_out_raw = Some(
            gpu.alloc_tensor(&[n_heads, head_dim], DType::F32)
                .map_err(|e| format!("alloc attn_out_raw: {e:?}"))?,
        );
    }
    if state.attn_out_raw_rot.is_none() {
        state.attn_out_raw_rot = Some(
            gpu.alloc_tensor(&[n_heads_head_dim], DType::F32)
                .map_err(|e| format!("alloc attn_out_raw_rot: {e:?}"))?,
        );
    }
    let n_groups = cfg.o_groups;
    let o_lora_rank = cfg.o_lora_rank;
    let groups_o_lora = n_groups * o_lora_rank;
    if state.wo_a_out.is_none() {
        state.wo_a_out = Some(
            gpu.alloc_tensor(&[groups_o_lora], DType::F32)
                .map_err(|e| format!("alloc wo_a_out: {e:?}"))?,
        );
    }
    if state.wo_a_out_rot.is_none() {
        state.wo_a_out_rot = Some(
            gpu.alloc_tensor(&[groups_o_lora], DType::F32)
                .map_err(|e| format!("alloc wo_a_out_rot: {e:?}"))?,
        );
    }

    // SWA is now the production default. Pos-0 path retained only as a
    // diagnostic/regression-check escape hatch via HIPFIRE_DEEPSEEK4_ATTN=pos0.
    let use_swa = !env_cache::attn_pos0();

    let q = state.q.as_ref().unwrap();
    let kv = state.kv.as_ref().unwrap();
    let attn_out_raw = state.attn_out_raw.as_ref().unwrap();
    let layer = weights.resolve_layer(layer_idx);
    let attn_sink = layer
        .attn_sink
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_sink not uploaded"))?;

    if !use_swa {
        // Pos-0 attention (default). Each step independent.
        gpu.deepseek4_attn_pos0(
            q,
            kv,
            attn_sink,
            attn_out_raw,
            n_heads as i32,
            head_dim as i32,
            n_groups as i32,
        )
        .map_err(|e| format!("deepseek4_attn_pos0: {e:?}"))?;
    } else {
        // SWA path.
        let n_kv = cfg.num_key_value_heads;
        let win = cfg.sliding_window;
        {
            let attn = &mut state._attention[layer_idx];
            if attn.swa_k.is_none() {
                attn.swa_k = Some(
                    gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                        .map_err(|e| format!("alloc swa_k l{layer_idx}: {e:?}"))?,
                );
            }
            if attn.swa_v.is_none() {
                attn.swa_v = Some(
                    gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                        .map_err(|e| format!("alloc swa_v l{layer_idx}: {e:?}"))?,
                );
            }
        }
        let pos = state.n_tokens as usize;
        // slot/n_valid live in `state.attn_state_buf` (slot at offset 0,
        // n_valid at offset 1), populated by precompute_attn_state at
        // decode_step entry. The _buf kernel variant reads slot from
        // the device buffer, so the captured launch picks up the new
        // position on every graph replay without re-capture.
        let slot_buf = state
            .attn_state_buf
            .as_ref()
            .ok_or_else(|| {
                "attn_state_buf missing (precompute_positions must run first)".to_string()
            })?
            .sub_offset(0, 1);
        {
            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
            gpu.swa_ring_write_f32_buf(
                kv,
                swa_k,
                &slot_buf,
                n_kv as i32,
                head_dim as i32,
                win as i32,
            )
            .map_err(|e| format!("swa_k write: {e:?}"))?;
            gpu.swa_ring_write_f32_buf(
                kv,
                swa_v,
                &slot_buf,
                n_kv as i32,
                head_dim as i32,
                win as i32,
            )
            .map_err(|e| format!("swa_v write: {e:?}"))?;
        }
        let n_valid = (pos + 1).min(win) as i32;

        // Antirez-faithful mixed attention (ds4.c:7559-7566):
        //   ratio == 0 (dense): plain SWA attention over raw_kv
        //   ratio  > 0 (compressed): JOINT softmax over raw_kv + main_kv_cache
        //     ratio == 4: indexer top-K selects which compressor entries
        //     ratio == 128: no indexer, attend to ALL compressor entries
        //
        // Both compressor and raw entries share ONE softmax with the
        // attn_sink as an extra implicit drain entry. The compressed
        // cache contains the model's "coarse memory" — even at small pos
        // (within SWA window) the compressor cache provides DIFFERENT
        // signal than raw KV (compressed entries are softmax-pooled
        // wkv outputs with compressor.norm + RoPE applied; raw KV is the
        // per-position post-kv_norm post-RoPE K=V).
        //
        let do_mixed =
            layer.compress_ratio > 0 && state._indexer[layer_idx].main_kv_cache.is_some();

        if do_mixed {
            let topk_max = cfg.index_topk;
            if state._attention[layer_idx].gathered_k.is_none() {
                state._attention[layer_idx].gathered_k = Some(
                    gpu.zeros(&[n_kv, head_dim, topk_max], DType::F32)
                        .map_err(|e| format!("alloc gathered_k l{layer_idx}: {e:?}"))?,
                );
            }
            // n_compressed / k_active values for the current position are
            // pre-computed into state.attn_state_buf (slots 2-5) by
            // precompute_attn_state. Select the right slot based on
            // layer.compress_ratio so the captured graph reads the right
            // host-updated value on every replay.
            //   ratio=4  → n_compressed at slot 2, k_active at slot 4
            //   ratio=128 → n_compressed at slot 3, k_active at slot 5
            let attn_buf = state
                .attn_state_buf
                .as_ref()
                .ok_or_else(|| "attn_state_buf missing".to_string())?;
            let (n_compressed_buf, k_active_buf) = if layer.compress_ratio == 4 {
                (attn_buf.sub_offset(2, 1), attn_buf.sub_offset(4, 1))
            } else {
                (attn_buf.sub_offset(3, 1), attn_buf.sub_offset(5, 1))
            };

            let use_topk_gather =
                layer.compress_ratio == 4 && state._indexer[layer_idx].topk_idx_indices.is_some();
            if use_topk_gather {
                // ratio=4 path: indexer top-K gather. Launch with fixed
                // grid = topk_max so capture sees a constant grid; lanes
                // past K_buf[0] early-return.
                let topk_idx = state._indexer[layer_idx].topk_idx_indices.as_ref().unwrap();
                let main_kv_cache = state._indexer[layer_idx].main_kv_cache.as_ref().unwrap();
                let gathered_k = state._attention[layer_idx].gathered_k.as_ref().unwrap();
                gpu.deepseek4_topk_kv_gather_f32_buf(
                    main_kv_cache,
                    topk_idx,
                    gathered_k,
                    &k_active_buf,
                    &n_compressed_buf,
                    topk_max as i32,
                    head_dim as i32,
                    topk_max as i32,
                    0,
                    1.0,
                )
                .map_err(|e| format!("mixed gather (idx,buf) l{layer_idx}: {e:?}"))?;
            } else {
                // ratio=128 (or fallback): identity gather over first K rows.
                let main_kv_cache = state._indexer[layer_idx].main_kv_cache.as_ref().unwrap();
                let gathered_k = state._attention[layer_idx].gathered_k.as_ref().unwrap();
                gpu.deepseek4_topk_kv_gather_identity_f32_buf(
                    main_kv_cache,
                    gathered_k,
                    &k_active_buf,
                    topk_max as i32,
                    head_dim as i32,
                    topk_max as i32,
                )
                .map_err(|e| format!("mixed gather (all,buf) l{layer_idx}: {e:?}"))?;
            }

            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
            let gathered_k = state._attention[layer_idx].gathered_k.as_ref().unwrap();
            let n_valid_buf = attn_buf.sub_offset(1, 1);
            // Joint softmax: scores = Q·K for [swa_k, gathered_k, attn_sink],
            // single normalization, V = swa_v + gathered_v (K=V tied, so
            // we pass gathered_k as V too). n_valid_swa + n_active_topk
            // come from the device-side attn_state_buf.
            gpu.deepseek4_attn_swa_topk_f32_buf(
                q,
                swa_k,
                swa_v,
                gathered_k,
                gathered_k,
                attn_sink,
                attn_out_raw,
                &n_valid_buf,
                &k_active_buf,
                n_heads as i32,
                head_dim as i32,
                win as i32,
                topk_max as i32,
            )
            .map_err(|e| format!("deepseek4_attn_swa_topk_buf l{layer_idx}: {e:?}"))?;
            let _ = n_valid; // legacy host-computed value not used after migration
        } else {
            let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
            let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
            // HIP-graphs-safe: n_valid comes from attn_state_buf[1]
            // (populated by precompute_attn_state at decode_step entry).
            // The legacy `gpu.deepseek4_attn_swa(...n_valid kernarg...)` would
            // bake n_valid at capture time → broken on graph replay.
            let n_valid_buf = state
                .attn_state_buf
                .as_ref()
                .ok_or_else(|| "attn_state_buf missing".to_string())?
                .sub_offset(1, 1);
            let _ = n_valid; // legacy host-computed value; not used after migration
            gpu.deepseek4_attn_swa_buf(
                q,
                swa_k,
                swa_v,
                attn_sink,
                attn_out_raw,
                &n_valid_buf,
                n_heads as i32,
                head_dim as i32,
                n_groups as i32,
                win as i32,
            )
            .map_err(|e| format!("deepseek4_attn_swa_buf: {e:?}"))?;
        }
    }

    // Inverse tail RoPE on attn_out_raw. Same YaRN params as the forward
    // apply_tail_rope so the rotation cancels correctly across attention.
    // Antirez `layer_forward_self_one` does the matching:
    //   rope_tail_layer_inplace(q,     ..., pos, il, false)  // forward
    //   rope_tail_layer_inplace(heads, ..., pos, il, true)   // inverse
    // (ds4.c:7868, 7874)
    let pos_buf = state
        .pos_buf
        .as_ref()
        .ok_or_else(|| "pos_buf not allocated".to_string())?;
    {
        let layer = weights.resolve_layer(layer_idx);
        let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
            layer_rope_params(cfg, layer.compress_ratio);
        gpu.rope_tail_yarn_interleaved(
            attn_out_raw,
            attn_out_raw,
            pos_buf,
            n_heads as i32,
            0,
            head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            corr_low,
            corr_high,
            /*inverse=*/ 1,
        )
        .map_err(|e| format!("rope_tail_yarn_interleaved (inverse) l{layer_idx}: {e:?}"))?;
    }

    // O-LoRA projection: wo_a per-group + wo_b.
    //   wo_a: [n_groups * o_lora_rank, heads_per_group * head_dim] MQ4
    //         = [8 * 1024, 8 * 512] = [8192, 4096]
    //   Per group g: y_g [o_lora_rank=1024] = wo_a_g [1024, 4096] @ x_g [4096]
    //   wo_b: [hidden, n_groups * o_lora_rank] MQ4 = [4096, 8192]
    //   y [hidden=4096] = wo_b @ wo_a_out_rot [8192]
    let wo_a = layer
        .wo_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_a missing"))?;
    let wo_b = layer
        .wo_b
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_b missing"))?;
    let attn_out_raw_rot = state.attn_out_raw_rot.as_ref().unwrap();
    let wo_a_out = state.wo_a_out.as_ref().unwrap();
    let wo_a_out_rot = state.wo_a_out_rot.as_ref().unwrap();
    let final_attn_out = state.attn_out.as_ref().unwrap();

    // FWHT-rotate per-group slices of attn_out_raw (k=heads_per_group*head_dim).
    let heads_per_group = n_heads / n_groups;
    let per_group_in = heads_per_group * head_dim;
    let per_group_elems = o_lora_rank * per_group_in;
    // Per-group byte stride depends on wo_a's dtype:
    //   MQ4G256 (Raw):     136 bytes per 256 elements
    //   Q8_0:               34 bytes per 32 elements
    //   F32 (F16-source):   4 bytes per element (handled via sub_offset's
    //                       built-in size scaling — pass elem count)
    let per_group_wa_bytes_raw = (per_group_elems / 256) * 136;
    let per_group_wa_bytes_q8 = (per_group_elems / 32) * 34;

    // FWHT-rotate all 8 group slices in one batched launch. attn_out_raw
    // is contiguous [n_groups, per_group_in] so grid.y=n_groups indexes
    // each group at stride per_group_in. Skip when wo_a is Q8/F16
    // (gemv_auto reads x_plain in those paths, not x_rotated).
    let wo_a_needs_fwht = weight_needs_fwht(wo_a);
    let wo_b_needs_fwht = weight_needs_fwht(wo_b);
    if wo_a_needs_fwht {
        gpu.rotate_x_mq_batched(attn_out_raw, attn_out_raw_rot, per_group_in, n_groups)
            .map_err(|e| format!("rotate attn_out batched l{layer_idx}: {e:?}"))?;
    }

    for g in 0..n_groups {
        let raw_view = attn_out_raw.sub_offset(g * per_group_in, per_group_in);
        let rot_view = attn_out_raw_rot.sub_offset(g * per_group_in, per_group_in);
        // Dtype-aware sub-view for wo_a's per-group slice.
        let wo_a_view = match wo_a.dtype {
            DType::F32 => {
                // sub_offset handles size scaling for F32 (size=4). Result
                // is 1D; gemv_f32 expects 2D [m, k] so we mutate the shape.
                let mut v = wo_a.sub_offset(g * per_group_elems, per_group_elems);
                v.shape = vec![o_lora_rank, per_group_in];
                v
            }
            DType::Q8_0 => wo_a.sub_offset(g * per_group_wa_bytes_q8, per_group_wa_bytes_q8),
            _ => wo_a.sub_offset(g * per_group_wa_bytes_raw, per_group_wa_bytes_raw),
        };
        let out_view = wo_a_out.sub_offset(g * o_lora_rank, o_lora_rank);
        // Dispatch per dtype. F32/Q8 use plain raw_view; MQ4 uses rot_view.
        gemv_auto(
            gpu,
            &wo_a_view,
            &rot_view,
            &raw_view,
            &out_view,
            o_lora_rank,
            per_group_in,
        )?;
    }

    // FWHT-rotate wo_a_out then wo_b GEMV → final_attn_out [hidden].
    // wo_b path: F32/Q8 use plain wo_a_out; MQ4 uses wo_a_out_rot.
    if wo_b_needs_fwht {
        gpu.rotate_x_mq(wo_a_out, wo_a_out_rot, groups_o_lora)
            .map_err(|e| format!("rotate wo_a_out l{layer_idx}: {e:?}"))?;
    }
    gemv_auto(
        gpu,
        wo_b,
        wo_a_out_rot,
        wo_a_out,
        final_attn_out,
        cfg.hidden_size,
        groups_o_lora,
    )?;

    Ok(())
}

/// DeepSeek V4 MoE router: scores and top-K expert selection.
///
/// For score-routed layers (l >= num_hash_layers = 3 on DeepSeek V4):
///   1. logits = gate.weight @ ffn_input  [256]  (MQ4G256 GEMV, M=256, K=hidden)
///   2. logits += gate.bias
///   3. scores = sqrt(softplus(logits))   [256]  (DeepSeek V4 affinity)
///   4. topk_indices = top_k(scores, k=6)        (reuses indexer_top_k)
///
/// For hash-routed layers (l < 3): use the static `tid2eid` lookup
/// table. Currently SKIPPED at quantize time, so hash-routed layers
/// fall back to shared expert only.
///
/// Output lives in state.router_scores and state.topk_indices. The
/// expert-dispatch step reads topk_indices, fetches per-expert weights
/// from `layer.expert_w{1,2,3}` (uploaded by default; opt out with
/// `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS=0`),
/// and accumulates weighted expert outputs into ffn_out.
/// Gather routing weights at the given indices from the (unbiased) scores,
/// then normalize to sum to 1. Returns `None` if the sum is non-positive.
fn gather_normalized_weights(scores: &[f32], indices: &[u32]) -> Option<Vec<f32>> {
    let mut wts: Vec<f32> = indices
        .iter()
        .map(|&i| *scores.get(i as usize).unwrap_or(&0.0))
        .collect();
    let s: f32 = wts.iter().sum();
    if s <= 0.0 {
        return None;
    }
    for w in wts.iter_mut() {
        *w /= s;
    }
    Some(wts)
}

fn moe_route(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    // Hash-routed and score-routed layers BOTH need router_scores for the
    // per-token expert weights (upstream DeepSeek V4 gathers unbiased scores at
    // tid2eid indices for hash layers, top-K for score layers). The split
    // was: score layers ALSO use gate.bias for bias-aware selection. So
    // gate.weight + sqrt_softplus is shared; gate.bias is optional.
    let layer = weights.resolve_layer(layer_idx);
    let gate_w = layer
        .gate_weight
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} gate.weight missing"))?;
    let _gate_b = layer.gate_bias.as_ref(); // None for hash layers; unused here

    let n_exp = cfg.n_routed_experts;
    if state.router_scores.is_none() {
        state.router_scores = Some(
            gpu.alloc_tensor(&[n_exp], DType::F32)
                .map_err(|e| format!("alloc router_scores: {e:?}"))?,
        );
    }
    let scores = state.router_scores.as_ref().unwrap();
    // Note: this function used to also write `state.topk_indices` via a
    // single-threaded selection-sort kernel. That output was never read
    // (the GPU bias-aware top-K in `ffn_routed` overwrites the real
    // expert indices into `state.moe_topk_indices`), so the call has
    // been removed — pure wasted work. The `topk_indices` allocation is
    // kept lazily-None for backward compat with any external readers.
    let _ = state.topk_indices.as_ref();

    // Upstream DeepSeek V4 gates on the POST-ffn_norm input (same x that
    // shared/routed experts see). ffn_x_rot is FWHT(ffn_norm(hc_x_in));
    // ffn_x_plain is the un-rotated version. Both are populated in
    // ffn_stub which runs before us.
    //
    // gemv_auto dispatches on the gate weight's dtype: MQ4 path consumes
    // ffn_x_rot, Q8_0 / F16 paths consume ffn_x_plain. Switching from the
    // hardcoded gemv_mq4g256_prerotated call lets the router work with
    // any quant of `gate.weight` — needed by deepseek4-q8-mtp (Q8F16) and
    // future formats. Using raw hc_x_in (as before ffn_stub landed)
    // caused scores to scale with stream magnitude, biasing selection.
    let ffn_x_rot = state
        .ffn_x_rot
        .as_ref()
        .ok_or_else(|| "ffn_x_rot not allocated — moe_route must run after ffn_stub".to_string())?;
    let ffn_x_plain = state.ffn_x_plain.as_ref().ok_or_else(|| {
        "ffn_x_plain not allocated — moe_route must run after ffn_stub".to_string()
    })?;

    // logits = gate.weight @ x  (dispatch on gate.weight dtype)
    gemv_auto(
        gpu,
        gate_w,
        ffn_x_rot,
        ffn_x_plain,
        scores,
        n_exp,
        cfg.hidden_size,
    )?;

    // logits += gate.bias (bias is F16, scores is F32 — need a kernel
    // for f16-bias-add. Skip for now; bias is small magnitude).
    let _ = _gate_b;

    // scores = sqrt(softplus(logits))
    gpu.sqrt_softplus_f32(scores)
        .map_err(|e| format!("sqrt_softplus layer {layer_idx}: {e:?}"))?;
    let _ = layer_idx;

    Ok(())
}

/// mHC pre-step: compute c = X · W_fn + base [24], split into
/// Ã/B̃/C̃, apply sigmoid/exp+Sinkhorn/2σ, then compute
/// state.hc_x_in = A_l · streams (the input mapping).
///
/// After this runs, the layer's transform (attn or FFN) reads
/// hc_x_in as its [hidden]-shaped input.
fn mhc_pre(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
    is_attn: bool,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let (hc_fn, hc_base) = if is_attn {
        (
            layer.hc_attn_fn.as_ref().unwrap(),
            layer.hc_attn_base.as_ref().unwrap(),
        )
    } else {
        (
            layer.hc_ffn_fn.as_ref().unwrap(),
            layer.hc_ffn_base.as_ref().unwrap(),
        )
    };
    let streams = state.residual_streams.as_ref().unwrap();

    if state.hc_x_in.is_none() {
        state.hc_x_in = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc hc_x_in: {e:?}"))?,
        );
    }
    if state.hc_c.is_none() {
        state.hc_c = Some(
            gpu.alloc_tensor(&[24], DType::F32)
                .map_err(|e| format!("alloc hc_c: {e:?}"))?,
        );
    }

    let n_ctrl = 24;
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    let c_view = state.hc_c.as_ref().unwrap().sub_offset(0, n_ctrl);

    // c = streams · W_fn + base
    gpu.hc_compute_control(
        streams,
        hc_fn,
        hc_base,
        &c_view,
        n_ctrl as i32,
        x_dim as i32,
    )
    .map_err(|e| format!("hc_compute_control layer {layer_idx}: {e:?}"))?;

    // Apply α^pre/res/post scaling (paper eqs 3-5): rescales c so
    // c[i] = α[seg(i)] · (X · W) + (1 - α[seg(i)]) · base[i].
    // α small → static-bias-dominated (initial training behavior).
    let hc_scale = if is_attn {
        layer.hc_attn_scale.as_ref().unwrap()
    } else {
        layer.hc_ffn_scale.as_ref().unwrap()
    };
    gpu.hc_apply_alpha(&c_view, hc_scale, hc_base)
        .map_err(|e| format!("hc_apply_alpha layer {layer_idx}: {e:?}"))?;

    // Upstream DeepSeek V4 mixes layout: [pre(4), post(4), comb(16)] at
    // offsets [0, 4, 8]. The 24-element c[] follows the same ordering
    // since c = α·(hc_fn @ x · rsqrt) + base maintains row order.
    //
    // PRE (4-dim, sigmoid + eps): per-stream INPUT-mapping weights;
    //   y[d] = sum_h pre[h] * x[h, d]. Used by hc_input_map_4stream.
    //
    // Antirez ds4 (ds4.c:4202): `pre[i] = sigmoid(...) + DS4_HC_EPS`
    // where DS4_HC_EPS = 1e-6 (matches our cfg.hc_eps). The eps is tiny
    // but applied uniformly across all 4 streams — its omission shifts
    // every stream by zero in the limit so quality is unchanged here,
    // kept aligned for clarity.
    let pre_view = state.hc_c.as_ref().unwrap().sub_offset(0, 4);
    // FUSED pre + post sigmoid+scale: one kernel launch replaces three
    // (sigmoid(pre), sigmoid(post), scale(post)). hc_c[0..4] gets
    // sigmoid + hc_eps; hc_c[4..8] gets post_scale * sigmoid;
    // hc_c[8..24] left for the sinkhorn pass below.
    //
    // Default post_scale = 1.5: empirical optimum under mixed attention
    // + YaRN. Antirez hardcodes 2.0; the 0.5 delta is plausibly MQ2-
    // Lloyd vs IQ2_XXS+Q2_K quant noise compensation. Env override:
    // HIPFIRE_DEEPSEEK4_POST_SCALE.
    use std::sync::OnceLock;
    static POST_SCALE: OnceLock<f32> = OnceLock::new();
    let post_scale = *POST_SCALE.get_or_init(|| {
        std::env::var("HIPFIRE_DEEPSEEK4_POST_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.5)
    });
    let hc_c_full = state.hc_c.as_ref().unwrap();
    gpu.hc_pre_post_sigmoid_scale_f32(hc_c_full, cfg.hc_eps, post_scale)
        .map_err(|e| format!("hc_pre_post_sigmoid_scale layer {layer_idx}: {e:?}"))?;
    let _post_view = hc_c_full.sub_offset(4, 4);

    // COMB (16-dim → 4x4): cross-stream combining matrix, Sinkhorn-
    //   normalized to be doubly stochastic.
    let comb_view = state.hc_c.as_ref().unwrap().sub_offset(8, 16);
    gpu.hc_sinkhorn_4x4(&comb_view, cfg.hc_eps, cfg.hc_sinkhorn_iters as i32)
        .map_err(|e| format!("hc_sinkhorn_4x4 layer {layer_idx}: {e:?}"))?;

    // Input mapping: hc_x_in = sum_h pre[h] · streams[h, :]
    let hc_x_in = state.hc_x_in.as_ref().unwrap();
    gpu.hc_input_map_4stream(&pre_view, streams, hc_x_in, cfg.hidden_size as i32)
        .map_err(|e| format!("hc_input_map layer {layer_idx}: {e:?}"))?;

    Ok(())
}

/// Step 8 (attention block): full manifold-constrained Hyper-Connection mix.
///
/// Per DeepSeek_V4.pdf §2.2:
///   c     = α · (X · W_fn) + base                  [24]
///   Ã,B̃,C̃ = c[0..4], c[4..20], c[20..24]
///   A_l   = σ(Ã_l)                                  [4]    (input mapping)
///   B_l   = Sinkhorn(exp(B̃_l))                     [4,4]  (residual matrix)
///   C_l   = 2σ(C̃_l)                                 [4]    (output mapping)
///   x_in  = A_l · X_l                               [hidden]   (NOT YET — uses stream0)
///   y     = F_l(x_in)
///   X_l+1 = B_l · X_l + C_l · y
///
/// Currently `α · X · W_fn` is computed without the α scaling, and
/// the input mapping `A·X` is stubbed (transform input = stream 0 not
/// the weighted-sum across streams). These approximations make HC
/// numerically non-canonical but kept bounded by the doubly-stochastic
/// B and bounded-magnitude C.
fn hc_attn_mix(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let _ = weights; // post/comb already in state.hc_c from mhc_pre
    let _ = layer_idx;
    let streams = state.residual_streams.as_ref().unwrap();
    let attn_out = state.attn_out.as_ref().unwrap();

    // Reuse the post and comb values that mhc_pre already computed
    // and saved into state.hc_c (with the correct α scaling applied
    // via hc_apply_alpha + sigmoid + sinkhorn). No need to recompute
    // — same input, same weights, no intervening writes to hc_c.
    let post_view = state.hc_c.as_ref().unwrap().sub_offset(4, 4);
    let comb_view = state.hc_c.as_ref().unwrap().sub_offset(8, 16);

    // X_{l+1} = comb · X_l + post · attn_out
    let streams_out = state.q.as_ref().unwrap();
    gpu.hc_mix_4stream(
        streams,
        &comb_view,
        &post_view,
        attn_out,
        streams_out,
        cfg.hidden_size as i32,
    )
    .map_err(|e| format!("hc_mix_4stream layer: {e:?}"))?;

    let bytes = cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&streams.buf, &streams_out.buf, bytes)
        .map_err(|e| format!("d2d hc_mix → streams: {e:?}"))?;
    Ok(())
}

/// Step 5 (attention block): Tail-only RoPE on Q and KV.
///
/// DeepSeek V4's `qk_rope_head_dim = 64` of `head_dim = 512`. Only the last
/// 64 dims of each head's 512-dim vector get rotated; the first 448
/// are pass-through. Same rotation applies to KV's 512-dim vector
/// (treated as 1 head).
///
/// Uses `rope_tail_halfsplit_f32` with DeepSeek V4's `rope_theta = 10000`.
/// YaRN correction dim: per-dim-pair index at which the high-vs-low
/// frequency split happens. Matches antirez ds4's `rope_yarn_corr_dim`.
fn rope_yarn_corr_dim(n_dims: u32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    n_dims as f32 * ((n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln())
        / (2.0 * base.ln())
}

/// Per-layer RoPE parameters: returns (freq_base, freq_scale, ext_factor,
/// attn_factor, corr_low, corr_high). Mirrors antirez's
/// `layer_rope_freq_base` / `layer_rope_freq_scale` + the attn_factor
/// cancellation in `rope_tail_layer_inplace`.
fn layer_rope_params(
    cfg: &DeepseekV4Config,
    compress_ratio: u32,
) -> (f32, f32, f32, f32, f32, f32) {
    let compressed = compress_ratio != 0;
    let freq_base = if compressed {
        cfg.compress_rope_theta
    } else {
        cfg.rope_theta
    };
    let scale_factor = cfg.rope_scaling_factor;
    let freq_scale = if compressed && scale_factor > 1.0 {
        1.0 / scale_factor
    } else {
        1.0
    };
    let ext_factor = if compressed && scale_factor > 1.0 {
        1.0
    } else {
        0.0
    };
    // attn_factor: antirez pre-divides by (1+0.1*log(1/fs)) here so the
    // kernel's inner `mscale *= (1+0.1*log(1/fs))` cancels it back to 1.0
    // (see ds4.c:4769-4778). For dense (ext_factor=0) the kernel skips the
    // log multiplication, so attn_factor stays 1.0.
    let attn_factor = if ext_factor != 0.0 && freq_scale > 0.0 {
        1.0 / (1.0 + 0.1 * (1.0_f32 / freq_scale).ln())
    } else {
        1.0
    };
    let n_rot = cfg.qk_rope_head_dim as u32;
    let n_ctx_orig = cfg.rope_scaling_original_max_position_embeddings as u64;
    let beta_fast = cfg.rope_scaling_beta_fast as f32;
    let beta_slow = cfg.rope_scaling_beta_slow as f32;
    let (corr_low, corr_high) = if ext_factor != 0.0 {
        let lo = rope_yarn_corr_dim(n_rot, n_ctx_orig, beta_fast, freq_base)
            .floor()
            .max(0.0);
        let hi = rope_yarn_corr_dim(n_rot, n_ctx_orig, beta_slow, freq_base)
            .ceil()
            .min((n_rot - 1) as f32);
        (lo, hi)
    } else {
        (0.0, 0.0)
    };
    (
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        corr_low,
        corr_high,
    )
}

fn apply_tail_rope(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    position: u32,
    layer_idx: usize,
) -> Result<(), String> {
    // Position is pre-loaded into `state.pos_array_device` at decode_step
    // entry (single htod for all layers). Slice the qk_pos slot for this
    // layer. Also seed the legacy `state.pos_buf` field so other code
    // paths that still read it (inverse RoPE on attn_out, indexer) work
    // unchanged — they get the SAME slice. The per-layer memcpy_htod is
    // gone, lifting it out of any future HIP-graph captured region.
    let pos_slice = pos_slot(state, layer_idx, 0)?;
    state.pos_buf = Some(pos_slice);
    let pos_buf = state.pos_buf.as_ref().unwrap();
    let _ = position; // silence unused; precompute_positions already used it

    let q = state.q.as_ref().unwrap();
    let kv = state.kv.as_ref().unwrap();

    // DeepSeek V4 upstream (per antirez ds4 reference):
    //   compress_ratio == 0 (layers 0, 1, MTP): rope_theta = 10000, no YaRN
    //   compress_ratio  > 0 (layers 2..42):      compress_rope_theta = 160000,
    //                                            YaRN with scale_factor = 16
    let layer = weights.resolve_layer(layer_idx);
    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
        layer_rope_params(cfg, layer.compress_ratio);

    gpu.rope_tail_yarn_interleaved(
        q,
        kv,
        pos_buf,
        cfg.num_attention_heads as i32,
        cfg.num_key_value_heads as i32,
        cfg.head_dim as i32,
        cfg.qk_rope_head_dim as i32,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        corr_low,
        corr_high,
        /*inverse=*/ 0,
    )
    .map_err(|e| format!("rope_tail_yarn_interleaved: {e:?}"))?;

    Ok(())
}

/// Step 4 (attention block): Joint KV projection.
///
/// DeepSeek V4 has `n_kv_heads = 1`, `head_dim = 512`, so the entire KV
/// stream per token is one 512-dim vector. `wkv` shape on disk is
/// `[512, 4096]` — a standard small GEMV producing 512 outputs from
/// 4096 hidden inputs.
///
/// Tail-only RoPE applies to the last `qk_rope_head_dim = 64` dims.
/// The leading 448 dims are pass-through.
///
/// Caller assumes `state.tmp` is still the FWHT-rotated post-RMSNorm
/// input from `q_lora` (gemv_mq4g256_prerotated doesn't modify x).
fn kv_joint(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let wkv = layer
        .wkv
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wkv missing"))?;
    let kv_norm = layer
        .kv_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} kv_norm missing"))?;

    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    if state.kv.is_none() {
        state.kv = Some(
            gpu.alloc_tensor(&[kv_dim], DType::F32)
                .map_err(|e| format!("alloc kv: {e:?}"))?,
        );
    }
    let tmp = state.tmp.as_ref().unwrap();
    let tmp_plain = state
        .tmp_plain
        .as_ref()
        .ok_or_else(|| "kv_joint: tmp_plain missing (q_lora must run first)".to_string())?;
    let kv = state.kv.as_ref().unwrap();

    // wkv @ tmp → kv.  Dispatch on weight dtype (MQ4G256 / F32-from-F16).
    gemv_auto(gpu, wkv, tmp, tmp_plain, kv, kv_dim, cfg.hidden_size)?;

    // kv_norm RMSNorm in place (upstream DeepSeek V4: `kv = self.kv_norm(kv)`
    // after wkv, before apply_rotary_emb). Was missing — likely
    // contributed to the SWA attractor since Q is rmsnormed but K=V
    // had arbitrary magnitudes.
    gpu.rmsnorm_f32(kv, kv_norm, kv, cfg.rms_norm_eps)
        .map_err(|e| format!("kv_norm rmsnorm layer {layer_idx}: {e:?}"))?;

    Ok(())
}

/// Step 3 (attention block): Q via Q-LoRA + tail-only RoPE.
///
///   x = state.tmp (post-RMSNorm)  -- but actually we should re-do
///       RMSNorm here with the fused-rotate variant so x is in the
///       FWHT-rotated domain that MQ4 expects.
///
///   Algorithm:
///     1. fused_rmsnorm_rotate_mq(stream0, attn_norm, x_rot, hidden, eps)
///        → x_rot [hidden] in MQ-rotated domain
///     2. gemv_mq4g256_prerotated(wq_a, x_rot, q_lat, q_lora_rank, hidden)
///        → q_lat [q_lora_rank=1024]
///     3. rotate_x_mq(q_lat, q_lat_rot, q_lora_rank)
///        → q_lat_rot [q_lora_rank]
///     4. gemv_mq4g256_prerotated(wq_b, q_lat_rot, q, n_heads*head_dim, q_lora_rank)
///        → q [n_heads*head_dim = 32768]
///     5. rope_tail_halfsplit on q (only last qk_rope_head_dim=64 of each
///        head's 512 dims)
///
/// Reuses `state.tmp` as the rotated post-RMSNorm input. Reuses
/// `state.q_lat`, `state.q_lat_rot`, `state.q`.
fn q_lora(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    layer_idx: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let attn_norm = layer
        .attn_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_norm missing"))?;
    let q_norm = layer
        .q_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} q_norm missing"))?;
    let wq_a = layer
        .wq_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_a missing"))?;
    let wq_b = layer
        .wq_b
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_b missing"))?;
    let streams = state.residual_streams.as_ref().unwrap();

    // Allocate Q-LoRA state slots once.
    if state.q_lat.is_none() {
        state.q_lat = Some(
            gpu.alloc_tensor(&[cfg.q_lora_rank], DType::F32)
                .map_err(|e| format!("alloc q_lat: {e:?}"))?,
        );
    }
    if state.q_lat_rot.is_none() {
        state.q_lat_rot = Some(
            gpu.alloc_tensor(&[cfg.q_lora_rank], DType::F32)
                .map_err(|e| format!("alloc q_lat_rot: {e:?}"))?,
        );
    }
    if state.q.is_none() {
        // 2D shape so rmsnorm_f32 does per-head normalization.
        state.q = Some(
            gpu.alloc_tensor(&[cfg.num_attention_heads, cfg.head_dim], DType::F32)
                .map_err(|e| format!("alloc q: {e:?}"))?,
        );
    }
    if state.q_head_ones.is_none() {
        let ones = vec![1.0f32; cfg.head_dim];
        state.q_head_ones = Some(
            gpu.upload_f32(&ones, &[cfg.head_dim])
                .map_err(|e| format!("upload q_head_ones: {e:?}"))?,
        );
    }
    // Plain rmsnorm output for F16 non-expert GEMVs (antirez recipe).
    if state.tmp_plain.is_none() {
        state.tmp_plain = Some(
            gpu.alloc_tensor(&[cfg.hidden_size], DType::F32)
                .map_err(|e| format!("alloc tmp_plain: {e:?}"))?,
        );
    }

    let hc_x_in = state.hc_x_in.as_ref().unwrap();
    let tmp = state.tmp.as_ref().unwrap();
    let tmp_plain = state.tmp_plain.as_ref().unwrap();
    let q_lat = state.q_lat.as_ref().unwrap();
    let q_lat_rot = state.q_lat_rot.as_ref().unwrap();
    let q = state.q.as_ref().unwrap();
    let q_head_ones = state.q_head_ones.as_ref().unwrap();
    let _ = streams; // streams not used directly anymore; transform reads hc_x_in

    // Skip dead FWHT rotations when consuming weights are Q8/F16 (the
    // DeepSeek V4-q8 case for both wq_a and wq_b). The gemv_auto dispatch reads
    // x_plain on those paths; x_rotated is unused.
    let wq_a_needs_fwht = weight_needs_fwht(wq_a);
    let wq_b_needs_fwht = weight_needs_fwht(wq_b);

    // 1. RMSNorm (+ optional FWHT) hc_x_in → tmp / tmp_plain. When both
    //    outputs are needed (the common DeepSeek V4 case), use the fused variant
    //    that writes both in one launch.
    if wq_a_needs_fwht {
        gpu.fused_rmsnorm_rotate_mq_plain(
            hc_x_in,
            attn_norm,
            tmp,
            tmp_plain,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )
        .map_err(|e| format!("fused_rmsnorm_rotate_mq_plain layer {layer_idx}: {e:?}"))?;
    } else {
        // Plain only: wq_a is Q8/F16/F32 → tmp not consumed downstream,
        // but compressor + indexer still read tmp_plain so it's required.
        gpu.rmsnorm_f32(hc_x_in, attn_norm, tmp_plain, cfg.rms_norm_eps)
            .map_err(|e| format!("rmsnorm_f32 attn-side plain l{layer_idx}: {e:?}"))?;
    }

    // 2. wq_a @ tmp → q_lat. M = q_lora_rank, K = hidden.
    gemv_auto(
        gpu,
        wq_a,
        tmp,
        tmp_plain,
        q_lat,
        cfg.q_lora_rank,
        cfg.hidden_size,
    )?;

    // 2.5. Apply q_norm to the q-LoRA bottleneck (upstream DeepSeek V4:
    //     `q = self.q_norm(self.wq_a(x))`). RMSNorm with q_norm weight.
    //     In-place: read q_lat, write q_lat.
    gpu.rmsnorm_f32(q_lat, q_norm, q_lat, cfg.rms_norm_eps)
        .map_err(|e| format!("q_norm rmsnorm layer {layer_idx}: {e:?}"))?;

    // 3. Rotate q_lat for the second GEMV — only if wq_b is MQ4.
    if wq_b_needs_fwht {
        gpu.rotate_x_mq(q_lat, q_lat_rot, cfg.q_lora_rank)
            .map_err(|e| format!("rotate_x_mq q_lat layer {layer_idx}: {e:?}"))?;
    }

    // 4. wq_b @ q_lat_rot → q. M = n_heads * head_dim, K = q_lora_rank.
    //    Use q_lat (un-rotated) for F16 path; q_lat_rot for MQ4 path.
    let q_total = cfg.num_attention_heads * cfg.head_dim;
    gemv_auto(gpu, wq_b, q_lat_rot, q_lat, q, q_total, cfg.q_lora_rank)?;

    // 4.5. Per-head RMSNorm of Q (upstream DeepSeek V4:
    //     `q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)`).
    gpu.rmsnorm_f32(q, q_head_ones, q, cfg.rms_norm_eps)
        .map_err(|e| format!("q per-head rmsnorm layer {layer_idx}: {e:?}"))?;

    Ok(())
}

/// Step 1 of forward: embedding lookup + 4-stream residual init.
///
/// DeepSeek V4's HC pattern starts with `[embed, 0, 0, 0]` — stream 0 gets
/// the embedding, streams 1-3 zero-initialised. Subsequent layers'
/// HC mixes propagate signal across all four streams.
///
/// Allocates `state.residual_streams` and `state.embed_scratch`
/// lazily on first call.
/// Per-layer slot count in `pos_array_*`. Layout per layer:
///   [0] qk_pos              = position
///   [1] main_comp_rope_pos  = mid-of-window  (depends on ratio + COMP_ROPE_POS env)
///   [2] indexer_comp_rope_pos = start-of-window
/// Used by the HIP-graphs-friendly position-array path (default in
/// `decode_step` since 2026-05-21). Direct-dispatch path uses the same
/// array but doesn't strictly need the stable host source.
pub(crate) const POS_SLOTS_PER_LAYER: usize = 3;

/// Compute per-layer derived positions and update `state.pos_array_*`.
///
/// Single host-to-device copy of the entire `[(num_layers + 1) * 3]` i32
/// array, with `pos_array_host` as the stable source pointer (required so
/// captured graph nodes re-read valid values on replay).
///
/// Reads env vars HIPFIRE_DEEPSEEK4_COMP_ROPE_POS once into a cache (TODO:
/// migrate to OnceLock once we settle on a fixed default).
pub fn precompute_positions(
    cfg: &DeepseekV4Config,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    position: u32,
) -> Result<(), String> {
    let total_slots = (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER;

    // Lazy-alloc device buffer + stable host source (Box<[i32]>).
    if state.pos_array_device.is_none() {
        state.pos_array_device = Some(
            gpu.alloc_tensor(&[total_slots], DType::F32)
                .map_err(|e| format!("alloc pos_array_device: {e:?}"))?,
        );
    }
    if state.pos_array_host.is_none() {
        state.pos_array_host = Some(vec![0i32; total_slots].into_boxed_slice());
    }

    let pos_array_host = state.pos_array_host.as_mut().unwrap();
    fill_pos_array_host(cfg, pos_array_host, position);

    // ONE htod for the whole array. Source is the stable Box<[i32]> on
    // the heap, so captured graph nodes can re-read it on replay.
    let pos_array_device = state.pos_array_device.as_ref().unwrap();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            pos_array_host.as_ptr() as *const u8,
            pos_array_host.len() * 4,
        )
    };
    gpu.memcpy_htod_auto(&pos_array_device.buf, bytes)
        .map_err(|e| format!("htod pos_array: {e:?}"))?;

    // Also write SWA state (slot, n_valid) — same stable-host-source
    // pattern. The captured memcpy re-reads this on every graph_launch.
    precompute_attn_state(cfg, state, gpu)?;
    Ok(())
}

/// Lazy-alloc + populate `state.token_id_buf` (and stable host source
/// `state.token_id_host`) for the current step's token. The captured
/// htod node re-reads `token_id_host` on every graph replay, so the
/// HIP-graphs-safe `hash_router_normalize_f32_buf` kernel sees the
/// per-replay token_id.
pub(crate) fn precompute_token_id(
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    if state.token_id_buf.is_none() {
        state.token_id_buf = Some(
            gpu.alloc_tensor(&[1], DType::F32)
                .map_err(|e| format!("alloc token_id_buf: {e:?}"))?,
        );
    }
    if state.token_id_host.is_none() {
        state.token_id_host = Some(Box::new([0i32; 1]));
    }
    let host = state.token_id_host.as_mut().unwrap();
    host[0] = token_id as i32;
    let dev = state.token_id_buf.as_ref().unwrap();
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, 4) };
    gpu.memcpy_htod_auto(&dev.buf, bytes)
        .map_err(|e| format!("htod token_id: {e:?}"))?;
    Ok(())
}

/// Host-only update of `token_id_host[0]`. Used by the HIP-graphs
/// replay path — the captured memcpy node re-reads this byte on
/// graph_launch and propagates to `token_id_buf`.
pub(crate) fn update_token_id_host(state: &mut DeepseekV4State, token_id: u32) {
    let host = state.token_id_host.as_mut().expect(
        "update_token_id_host: token_id_host not initialised \
                 (call precompute_token_id first)",
    );
    host[0] = token_id as i32;
}

/// Per-batch twin of `precompute_positions`. Fills B contiguous stripes
/// of `(num_hidden_layers + 1) * POS_SLOTS_PER_LAYER` slots in
/// `pbs.pos_array_device_batch` — one stripe per batch row b at absolute
/// position `start_pos + b`. Single host-side build, single htod.
///
/// Stripe b layout matches the single-position `state.pos_array_device`:
/// `[layer_idx * 3 + slot]` where slot ∈ {0=qk_pos, 1=main_rope_pos,
/// 2=indexer_rope_pos}.
pub(crate) fn precompute_positions_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let slots_per_pos = (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER;
    let total_i32s = batch_size * slots_per_pos;

    let comp_rope_mode = std::env::var("HIPFIRE_DEEPSEEK4_COMP_ROPE_POS").ok();
    let comp_rope_mode = comp_rope_mode.as_deref();

    let mut host: Vec<i32> = vec![0i32; total_i32s];
    for b in 0..batch_size {
        let pos = (start_pos as usize) + b;
        let stripe = b * slots_per_pos;
        for layer_idx in 0..=cfg.num_hidden_layers {
            let ratio = if layer_idx < cfg.num_hidden_layers {
                cfg.compress_ratios[layer_idx] as usize
            } else {
                0
            };
            let base = stripe + layer_idx * POS_SLOTS_PER_LAYER;
            host[base] = pos as i32;
            if ratio > 0 {
                // Default MUST be "start" — `(pos/ratio)*ratio` — to match the
                // decode path (`fill_pos_array_host`) and the reference ds4
                // (comp_pos = start of the just-closed window). This previously
                // defaulted to "mid" (+ ratio/2) while decode defaults to
                // "start", so the compressed KV was BUILT here with a different
                // compressor-RoPE phase than it is READ with at decode → far-
                // context (compressed) recall lost the tail of the prompt.
                // Keep the named modes identical to `fill_pos_array_host`.
                let main_rope_pos: i32 = match comp_rope_mode {
                    Some("end") => pos as i32,
                    Some("mid") => (((pos / ratio) * ratio) + ratio / 2) as i32,
                    _ => ((pos / ratio) * ratio) as i32,
                };
                let indexer_rope_pos = ((pos / ratio) * ratio) as i32;
                host[base + 1] = main_rope_pos;
                host[base + 2] = indexer_rope_pos;
            } else {
                host[base + 1] = 0;
                host[base + 2] = 0;
            }
        }
    }

    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, total_i32s * 4) };
    gpu.memcpy_htod_auto(&pbs.pos_array_device_batch.buf, bytes)
        .map_err(|e| format!("htod pos_array_device_batch: {e:?}"))
}

/// Per-batch twin of `precompute_attn_state`. Fills B contiguous stripes
/// of 10 slots in `pbs.attn_state_buf_batch` — one stripe per batch row
/// b at absolute position `start_pos + b`. Slot layout matches
/// `fill_attn_state_host` (see line ~1389).
pub(crate) fn precompute_attn_state_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let slots_per_pos = 10;
    let total_i32s = batch_size * slots_per_pos;

    let win = cfg.sliding_window as i32;
    let topk = cfg.index_topk as i32;
    let max_compressed = env_cache::max_compress_pos() as i32;

    let mut host: Vec<i32> = vec![0i32; total_i32s];
    for b in 0..batch_size {
        let pos = (start_pos as i32) + b as i32;
        let stripe = b * slots_per_pos;

        let swa_slot = pos % win;
        let n_valid_swa = (pos + 1).min(win);
        let n_compressed_4 = (pos + 1) / 4;
        let n_compressed_128 = (pos + 1) / 128;
        let k_active_4 = topk.min(n_compressed_4);
        let k_active_128 = topk.min(n_compressed_128);
        let ring_slot_4 = 4 + (pos % 4);
        let commit_slot_4 = if (pos + 1) % 4 == 0 {
            let s = pos / 4;
            if s < max_compressed {
                s
            } else {
                -1
            }
        } else {
            -1
        };
        let ring_slot_128 = pos % 128;
        let commit_slot_128 = if (pos + 1) % 128 == 0 {
            let s = pos / 128;
            if s < max_compressed {
                s
            } else {
                -1
            }
        } else {
            -1
        };

        host[stripe] = swa_slot;
        host[stripe + 1] = n_valid_swa;
        host[stripe + 2] = n_compressed_4;
        host[stripe + 3] = n_compressed_128;
        host[stripe + 4] = k_active_4;
        host[stripe + 5] = k_active_128;
        host[stripe + 6] = ring_slot_4;
        host[stripe + 7] = commit_slot_4;
        host[stripe + 8] = ring_slot_128;
        host[stripe + 9] = commit_slot_128;
    }

    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, total_i32s * 4) };
    gpu.memcpy_htod_auto(&pbs.attn_state_buf_batch.buf, bytes)
        .map_err(|e| format!("htod attn_state_buf_batch: {e:?}"))
}

/// Slice the pos_array for a given layer's slot. Caller passes the slot
/// constant (0=qk_pos, 1=main_comp_rope, 2=indexer_comp_rope).
pub(crate) fn pos_slot(
    state: &DeepseekV4State,
    layer_idx: usize,
    slot: usize,
) -> Result<rdna_compute::GpuTensor, String> {
    let arr = state
        .pos_array_device
        .as_ref()
        .ok_or_else(|| "pos_array_device not initialised".to_string())?;
    let offset = layer_idx * POS_SLOTS_PER_LAYER + slot;
    Ok(arr.sub_offset(offset, 1))
}

fn init_residual_streams(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    token_id: u32,
) -> Result<(), String> {
    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or_else(|| "init_residual_streams: token_embd not uploaded".to_string())?;
    let hidden = cfg.hidden_size;
    let hc_mult = cfg.hc_mult;

    if state.embed_scratch.is_none() {
        state.embed_scratch = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc embed_scratch: {e:?}"))?,
        );
    }
    if state.residual_streams.is_none() {
        // Zero-init: alloc_tensor leaves memory uninitialized, but the
        // [embed, 0, 0, 0] init pattern relies on streams 1..hc_mult
        // being zero. `gpu.zeros` is the right primitive.
        let t = gpu
            .zeros(&[hc_mult, hidden], DType::F32)
            .map_err(|e| format!("alloc residual_streams: {e:?}"))?;
        state.residual_streams = Some(t);
    }
    if state.tmp.is_none() {
        state.tmp = Some(
            gpu.alloc_tensor(&[hidden], DType::F32)
                .map_err(|e| format!("alloc tmp: {e:?}"))?,
        );
    }

    // Dequant + lookup token row → embed_scratch [hidden].
    let embed_scratch = state.embed_scratch.as_ref().unwrap();
    gpu.embedding_lookup_q8(token_embd, embed_scratch, token_id, hidden)
        .map_err(|e| format!("embedding_lookup_q8: {e:?}"))?;

    // **HC init**: per antirez ds4 `hc_from_plain_embedding` (ds4.c:4358),
    // ALL `hc_mult` streams are initialised with a COPY of the embedding,
    // NOT `[embed, 0, 0, 0]` as our prior comment claimed. The "0 streams"
    // pattern would have forced HC pre/post/comb to propagate signal from
    // stream 0 across layers, producing wrong magnitudes throughout the
    // forward.
    let streams = state.residual_streams.as_ref().unwrap();
    let bytes_per_stream = hidden * 4; // F32 = 4 bytes
    for h in 0..hc_mult {
        let dst_view = streams.sub_offset(h * hidden, hidden);
        gpu.memcpy_dtod_auto(&dst_view.buf, &embed_scratch.buf, bytes_per_stream)
            .map_err(|e| format!("d2d copy stream {h}: {e:?}"))?;
    }

    Ok(())
}

/// Reusable per-call scratch for the batched-prefill driver.
///
/// **Phase B status (2026-05-18):** growing. Currently holds the
/// per-layer batched intermediates needed by `q_lora_batched`. Future
/// per-stage batched helpers (kv_joint_batched, attn_batched,
/// ffn_batched, hc_mix_batched) extend this struct as they land.
///
/// Sized to `max_batch` rows everywhere; tensors are reused across
/// per-chunk layer iterations.
pub struct PrefillBatchScratch {
    pub max_batch: usize,
    /// Embedding-lookup output `[max_batch, hidden]`. Source for the
    /// HC stream-broadcast init at chunk start.
    pub embed_batch: GpuTensor,
    /// HC residual streams `[max_batch, hc_mult, hidden]`. Lives across
    /// the full per-layer loop within a chunk.
    pub streams_batch: GpuTensor,
    /// Token-ids buffer feeding `embedding_lookup_q8_batched`.
    /// `[max_batch]` stored as F32 (same i32-in-F32-slots dtype-cosmetic
    /// pattern as qwen35's `pbs.tokens`).
    pub tokens: GpuTensor,
    /// FWHT-rotated attn_norm output `[max_batch, hidden]` feeding MQ4
    /// non-expert GEMMs.
    pub tmp_batch: GpuTensor,
    /// Plain attn_norm output `[max_batch, hidden]` feeding F32/Q8
    /// non-expert GEMMs.
    pub tmp_plain_batch: GpuTensor,
    /// Q-LoRA bottleneck `[max_batch, q_lora_rank]`. Reused: wq_a output
    /// → q_norm in place → fed to wq_b (after rotate into q_lat_rot_batch).
    pub q_lat_batch: GpuTensor,
    /// FWHT-rotated q_lat for the MQ4 wq_b path `[max_batch, q_lora_rank]`.
    pub q_lat_rot_batch: GpuTensor,
    /// Q output `[max_batch, n_heads, head_dim]`. wq_b output, then
    /// per-(batch, head) RMSNormed by `q_head_ones`.
    pub q_batch: GpuTensor,
    /// Per-head ones vector `[head_dim]` reused as the rmsnorm weight
    /// for the per-(batch, head) Q normalisation. Shared across batch.
    pub q_head_ones: GpuTensor,
    /// Joint KV `[max_batch, kv_dim]` where `kv_dim = n_kv_heads * head_dim`.
    /// wkv output, then kv_norm RMSNormed in place.
    pub kv_batch: GpuTensor,
    /// Per-batch absolute KV positions `[max_batch]` stored as F32 (the
    /// rope_tail_*_batched kernels read it as i32). Uploaded once per
    /// chunk: positions[b] = start_pos + b.
    pub positions: GpuTensor,
    /// HC control vector `[max_batch, 24]` — output of hc_compute_control
    /// _batched, in-place rescaled by hc_apply_alpha_batched, then split
    /// into pre/post/comb by hc_split_finalize_batched.
    pub hc_c_batch: GpuTensor,
    /// HC `pre` weights `[max_batch, hc_mult=4]`. Used by
    /// hc_input_map_4stream_batched and hc_mix_4stream_batched.
    pub hc_pre_batch: GpuTensor,
    /// HC `post` weights `[max_batch, hc_mult=4]`. Scale-multiplied
    /// sigmoid output. Feeds hc_mix_4stream_batched as the per-stream
    /// scale.
    pub hc_post_batch: GpuTensor,
    /// HC `comb` matrix `[max_batch, 4, 4]` — Sinkhorn-normalised to be
    /// doubly stochastic per batch row.
    pub hc_comb_batch: GpuTensor,
    /// HC transform input `[max_batch, hidden]` — output of mhc_pre's
    /// hc_input_map_4stream_batched. Feeds q_lora_batched / kv_joint
    /// _batched on the attention side, and the FFN gate/up on the FFN
    /// side.
    pub hc_x_in_batch: GpuTensor,
    /// Attention contribution `[max_batch, hidden]` produced by the
    /// attention block (Q · K → softmax → V → wo). Consumed by
    /// hc_attn_mix_batched as the `transform_out` argument.
    pub attn_out_batch: GpuTensor,
    /// FFN contribution `[max_batch, hidden]` produced by the routed
    /// MoE FFN. Consumed by hc_ffn_mix_batched as `transform_out`.
    pub ffn_out_batch: GpuTensor,
    /// Temporary `[max_batch, hc_mult, hidden]` for the hc_mix output
    /// before it's memcpy'd back into streams_batch. Mirrors the
    /// sequential path's reuse of `state.q` as the mix-output buffer.
    pub streams_out_batch: GpuTensor,
    /// Per-row visible SWA window `[max_batch, head_dim, swa_window]`
    /// produced by swa_visibility_stage_batched. DeepSeek V4 has K=V tied so
    /// one buffer feeds both the K and V args of the attention kernel.
    pub swa_staged_batch: GpuTensor,
    /// Per-row top-K K/V gather buffer `[max_batch, head_dim, topk_max]`
    /// produced by deepseek4_topk_kv_gather_batched (or the identity variant
    /// for ratio=128). Same K=V tied semantics.
    pub topk_staged_batch: GpuTensor,
    /// Per-row n_valid_swa array `[max_batch]` (i32-in-F32 slots).
    /// Tells deepseek4_attn_swa_topk_batched_f32 how many SWA entries are
    /// valid for each batch row.
    pub n_valid_swa_arr: GpuTensor,
    /// Per-row n_active_topk array `[max_batch]` (i32-in-F32 slots).
    pub n_active_topk_arr: GpuTensor,
    /// Raw attention output `[max_batch, n_heads, head_dim]`. Output of
    /// deepseek4_attn_swa_topk_batched_f32; consumed by inverse RoPE + the
    /// O-LoRA wo_a/wo_b projection chain.
    pub attn_out_raw_batch: GpuTensor,
    /// FWHT-rotated attn_out_raw `[max_batch, n_heads * head_dim]`.
    /// Input to per-group wo_a batched GEMV (MQ4 weight path).
    pub attn_out_raw_rot_batch: GpuTensor,
    /// wo_a output `[max_batch, n_groups, o_lora_rank]`.
    pub wo_a_out_batch: GpuTensor,
    /// FWHT-rotated wo_a output `[max_batch, n_groups * o_lora_rank]`.
    /// Input to wo_b batched GEMV (MQ4 weight path).
    pub wo_a_out_rot_batch: GpuTensor,
    // ── FFN-side scratch ──
    pub ffn_x_rot_batch: GpuTensor,        // [B, hidden]
    pub ffn_x_plain_batch: GpuTensor,      // [B, hidden]
    pub ffn_shared_gate_batch: GpuTensor,  // [B, IM]
    pub ffn_shared_up_batch: GpuTensor,    // [B, IM]
    pub ffn_shared_rot_batch: GpuTensor,   // [B, IM]
    pub moe_scores_batch: GpuTensor,       // [B, n_exp]
    pub moe_topk_indices_batch: GpuTensor, // [B, k_top]  i32-in-F32
    pub moe_topk_weights_batch: GpuTensor, // [B, k_top]
    pub moe_gate_batch: GpuTensor,         // [B, k_top, IM]
    pub moe_up_batch: GpuTensor,           // [B, k_top, IM]
    pub moe_rot_batch: GpuTensor,          // [B, k_top, IM]
    /// Per-(token, krank) expert outputs for the atomic-free down path.
    /// Gated by HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC=1 (or grouped path which
    /// uses its own scratch). Sized [B, k_top, hidden] f32 — at DeepSeek V4
    /// max_batch=512 hidden=4096 K_TOP=6 = 48 MB.
    pub moe_down_expert_outputs: GpuTensor,
    // ── Indexer chain scratch (Step-1 perf pass) ──
    pub idx_q_batch: GpuTensor,      // [B, idx_n_heads, idx_head_dim]
    pub idx_w_batch: GpuTensor,      // [B, idx_n_heads]
    pub idx_scores_batch: GpuTensor, // [B, max_compressed]
    pub idx_topk_indices_batch: GpuTensor, // [B, index_topk]  i32-in-F32
    // ── Compressor batched-GEMV scratch (Phase 2.5 perf pass) ──
    // Holds the wkv / wgate compressor outputs across all B positions
    // so the GEMVs can be batched out of the per-position loop. Main
    // and indexer compressors get separate buffers because the proj_dim
    // differs (main=2*head_dim=1024, idx=2*idx_head_dim=256 for DeepSeek V4).
    pub comp_main_kv_batch: GpuTensor,    // [B, 2*head_dim]
    pub comp_main_score_batch: GpuTensor, // [B, 2*head_dim]
    pub comp_idx_kv_batch: GpuTensor,     // [B, 2*idx_head_dim]
    pub comp_idx_score_batch: GpuTensor,  // [B, 2*idx_head_dim]
    // ── Scatter-by-expert MoE sort outputs ──
    // Single counting-sort produces these per layer; the grouped MoE
    // GEMVs then read each expert weight slab once with cache reuse.
    pub moe_sorted_b: GpuTensor,      // [B * K_TOP] i32
    pub moe_sorted_krank: GpuTensor,  // [B * K_TOP] i32
    pub moe_sorted_expert: GpuTensor, // [B * K_TOP] i32
    pub moe_expert_starts: GpuTensor, // [n_exp + 1] i32

    // ── SGLang-style scatter-grouped MoE pipeline (chunk_size ≥ 256) ──
    // Outputs of moe_scatter_fused_k8 feed gemm_mq2g256_lloyd_moe_grouped
    // _wmma_k2 and the unscatter/down-combine kernels. Sized for the
    // worst-case `m_total_max = max_batch * K_TOP + n_exp * BLOCK_M(=16)`
    // padded scatter layout.
    pub moe_expert_token_counts: GpuTensor, // [n_exp]      i32 (Raw)
    pub moe_expert_offsets: GpuTensor,      // [n_exp + 1]  i32 (Raw)
    pub moe_sorted_slot_index: GpuTensor,   // [m_total_max] i32 (Raw)
    pub moe_expert_tile_ids: GpuTensor,     // [m_total_max / 16] i32 (Raw)
    pub moe_inverse_perm: GpuTensor,        // [B * K_TOP]  i32 (Raw)
    /// Output of grouped gate_up GEMM. [m_total_max × 2*mi] f32.
    pub moe_y_gate_up_grouped: GpuTensor,
    /// Permuted, post-silu-mul intermediate. [m_total_max × mi] f32.
    /// Built by `moe_gate_up_unscatter_k8` then re-scattered by silu-mul,
    /// or written directly by a fused silu-mul kernel into grouped order.
    /// Input to the grouped down GEMM.
    pub moe_x_grouped: GpuTensor,
    /// Output of grouped down GEMM. [m_total_max × hidden] f32.
    /// Combined into the residual stream by `moe_down_combine_grouped_k8`.
    pub moe_y_down_grouped: GpuTensor,
    // ── F16 staging for WMMA compressor GEMMs ──
    // F32 attention-norm output gets converted once per layer into
    // these buffers, then the four compressor GEMMs (wkv/wgate ×
    // main/idx) consume F16 inputs directly. Sized at [max_batch,
    // hidden] like tmp_batch; 1/2 the per-element bytes of F32.
    pub tmp_batch_f16: GpuTensor, // [B, hidden] F16 (stored as Raw)
    pub tmp_plain_batch_f16: GpuTensor, // [B, hidden] F16 (stored as Raw)
    /// Generic F16 staging buffer for WMMA HFQ4 GEMMs. Sized at
    /// `max_batch * max_dim * 2 bytes` so any batched GEMM input can
    /// be converted F32→F16 in place before dispatch. max_dim is the
    /// largest K dim across all DeepSeek V4 batched GEMM call sites — wo_b's
    /// K = groups × o_lora_rank for DeepSeek V4 (= 8 × 1024 = 8192).
    pub wmma_x_scratch_f16: GpuTensor,
    /// Per-compress-event RoPE positions buffer for the Phase A
    /// batched compressor pipeline. Sized [max_batch] F32 (i32-in-F32)
    /// since at B=64 ratio=4 we have at most 16 events per layer and
    /// ratio=128 has at most 1; total fits well under max_batch slots.
    /// Separate from `pbs.positions` (which holds the chunk's
    /// [batch_size] absolute positions and is read by the indexer).
    pub comp_positions: GpuTensor,
    /// Per-batch-position pos_array (Option B per-batch state).
    /// `[max_batch * (num_hidden_layers + 1) * POS_SLOTS_PER_LAYER]` i32
    /// stored as F32 — each batch row b occupies a stripe of
    /// `(L+1) * 3` slots matching the single-position `state.pos_array_device`
    /// layout. Populated once per chunk by `precompute_positions_batched`,
    /// then sub-viewed into `state.pos_array_device` during the per-position
    /// compressor fallback loop so existing per-position kernels read the
    /// right batch row.
    pub pos_array_device_batch: GpuTensor,
    /// Per-batch-position attn_state buffer (Option B per-batch state).
    /// `[max_batch * ATTN_STATE_SLOTS=10]` i32 stored as F32. Same
    /// swap-and-sub-view pattern as pos_array_device_batch.
    pub attn_state_buf_batch: GpuTensor,
    /// Batched MTP next-token ids `[max_batch]` stored as F32 (i32-in-F32
    /// slot pattern). Per-position next-token id, fed to the batched
    /// embedding lookup at the start of `mtp_forward_batched`.
    pub mtp_tokens_batch: GpuTensor,
    /// Batched MTP embed output `[max_batch, hidden]`. embedding_lookup_q8
    /// _batched writes one row per batch position from `mtp_tokens_batch`.
    pub mtp_embed_batch: GpuTensor,
    /// Batched MTP e_norm output `[max_batch, hidden]`. `mtp_enorm`
    /// applied to `mtp_embed_batch` per batch row.
    pub mtp_e_norm_batch: GpuTensor,
    /// Batched MTP h_norm output `[max_batch, hc_mult, hidden]`. `mtp_hnorm`
    /// applied to the main forward's `streams_batch` per (batch, HC row).
    /// Consumed by the per-HC `mtp_h_proj` GEMV that writes the new
    /// `streams_batch` contents at the start of the MTP layer block.
    pub mtp_h_norm_batch: GpuTensor,
    /// Batched MTP x_e output `[max_batch, hidden]`. `mtp_e_proj @ e_norm`
    /// per batch row; broadcast-added to every HC row of the rebuilt
    /// streams_batch.
    pub mtp_x_e_batch: GpuTensor,
}

impl PrefillBatchScratch {
    /// Allocate scratch for prefill chunks of up to `max_batch` tokens.
    /// Sizes track the DeepSeek V4 config's hidden_size / q_lora_rank /
    /// num_attention_heads × head_dim.
    pub fn new(gpu: &mut Gpu, cfg: &DeepseekV4Config, max_batch: usize) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_rank = cfg.q_lora_rank;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let hc_mult = cfg.hc_mult;

        let alloc = |gpu: &mut Gpu, shape: &[usize], label: &str| -> Result<GpuTensor, String> {
            gpu.alloc_tensor(shape, DType::F32)
                .map_err(|e| format!("PrefillBatchScratch alloc {label}: {e:?}"))
        };
        let zeros = |gpu: &mut Gpu, shape: &[usize], label: &str| -> Result<GpuTensor, String> {
            gpu.zeros(shape, DType::F32)
                .map_err(|e| format!("PrefillBatchScratch zeros {label}: {e:?}"))
        };

        let ones_host = vec![1.0f32; head_dim];
        let q_head_ones = gpu
            .upload_f32(&ones_host, &[head_dim])
            .map_err(|e| format!("PrefillBatchScratch upload q_head_ones: {e:?}"))?;

        let kv_dim = cfg.num_key_value_heads * head_dim;

        Ok(Self {
            max_batch,
            embed_batch: alloc(gpu, &[max_batch, hidden], "embed_batch")?,
            streams_batch: zeros(gpu, &[max_batch, hc_mult, hidden], "streams_batch")?,
            tokens: alloc(gpu, &[max_batch], "tokens")?,
            tmp_batch: alloc(gpu, &[max_batch, hidden], "tmp_batch")?,
            tmp_plain_batch: alloc(gpu, &[max_batch, hidden], "tmp_plain_batch")?,
            q_lat_batch: alloc(gpu, &[max_batch, q_rank], "q_lat_batch")?,
            q_lat_rot_batch: alloc(gpu, &[max_batch, q_rank], "q_lat_rot_batch")?,
            q_batch: alloc(gpu, &[max_batch, n_heads, head_dim], "q_batch")?,
            q_head_ones,
            kv_batch: alloc(gpu, &[max_batch, kv_dim], "kv_batch")?,
            positions: alloc(gpu, &[max_batch], "positions")?,
            hc_c_batch: alloc(gpu, &[max_batch, 24], "hc_c_batch")?,
            hc_pre_batch: alloc(gpu, &[max_batch, hc_mult], "hc_pre_batch")?,
            hc_post_batch: alloc(gpu, &[max_batch, hc_mult], "hc_post_batch")?,
            hc_comb_batch: alloc(gpu, &[max_batch, hc_mult, hc_mult], "hc_comb_batch")?,
            hc_x_in_batch: alloc(gpu, &[max_batch, hidden], "hc_x_in_batch")?,
            attn_out_batch: alloc(gpu, &[max_batch, hidden], "attn_out_batch")?,
            ffn_out_batch: alloc(gpu, &[max_batch, hidden], "ffn_out_batch")?,
            streams_out_batch: alloc(gpu, &[max_batch, hc_mult, hidden], "streams_out_batch")?,
            swa_staged_batch: alloc(
                gpu,
                &[max_batch, head_dim, cfg.sliding_window],
                "swa_staged_batch",
            )?,
            topk_staged_batch: alloc(
                gpu,
                &[max_batch, head_dim, cfg.index_topk],
                "topk_staged_batch",
            )?,
            n_valid_swa_arr: alloc(gpu, &[max_batch], "n_valid_swa_arr")?,
            n_active_topk_arr: alloc(gpu, &[max_batch], "n_active_topk_arr")?,
            attn_out_raw_batch: alloc(gpu, &[max_batch, n_heads, head_dim], "attn_out_raw_batch")?,
            attn_out_raw_rot_batch: alloc(
                gpu,
                &[max_batch, n_heads * head_dim],
                "attn_out_raw_rot_batch",
            )?,
            wo_a_out_batch: alloc(
                gpu,
                &[max_batch, cfg.o_groups, cfg.o_lora_rank],
                "wo_a_out_batch",
            )?,
            wo_a_out_rot_batch: alloc(
                gpu,
                &[max_batch, cfg.o_groups * cfg.o_lora_rank],
                "wo_a_out_rot_batch",
            )?,
            ffn_x_rot_batch: alloc(gpu, &[max_batch, hidden], "ffn_x_rot_batch")?,
            ffn_x_plain_batch: alloc(gpu, &[max_batch, hidden], "ffn_x_plain_batch")?,
            ffn_shared_gate_batch: alloc(
                gpu,
                &[max_batch, cfg.moe_intermediate_size],
                "ffn_shared_gate_batch",
            )?,
            ffn_shared_up_batch: alloc(
                gpu,
                &[max_batch, cfg.moe_intermediate_size],
                "ffn_shared_up_batch",
            )?,
            ffn_shared_rot_batch: alloc(
                gpu,
                &[max_batch, cfg.moe_intermediate_size],
                "ffn_shared_rot_batch",
            )?,
            moe_scores_batch: alloc(gpu, &[max_batch, cfg.n_routed_experts], "moe_scores_batch")?,
            moe_topk_indices_batch: alloc(
                gpu,
                &[max_batch, cfg.num_experts_per_tok],
                "moe_topk_indices_batch",
            )?,
            moe_topk_weights_batch: alloc(
                gpu,
                &[max_batch, cfg.num_experts_per_tok],
                "moe_topk_weights_batch",
            )?,
            moe_gate_batch: alloc(
                gpu,
                &[
                    max_batch,
                    cfg.num_experts_per_tok,
                    cfg.moe_intermediate_size,
                ],
                "moe_gate_batch",
            )?,
            moe_up_batch: alloc(
                gpu,
                &[
                    max_batch,
                    cfg.num_experts_per_tok,
                    cfg.moe_intermediate_size,
                ],
                "moe_up_batch",
            )?,
            moe_rot_batch: alloc(
                gpu,
                &[
                    max_batch,
                    cfg.num_experts_per_tok,
                    cfg.moe_intermediate_size,
                ],
                "moe_rot_batch",
            )?,
            moe_down_expert_outputs: alloc(
                gpu,
                &[max_batch, cfg.num_experts_per_tok, hidden],
                "moe_down_expert_outputs",
            )?,
            // Indexer-chain scratch. max_compressed default 2048 unless overridden via env.
            idx_q_batch: alloc(
                gpu,
                &[max_batch, cfg.index_n_heads, cfg.index_head_dim],
                "idx_q_batch",
            )?,
            idx_w_batch: alloc(gpu, &[max_batch, cfg.index_n_heads], "idx_w_batch")?,
            idx_scores_batch: alloc(gpu, &[max_batch, 2048], "idx_scores_batch")?,
            idx_topk_indices_batch: alloc(
                gpu,
                &[max_batch, cfg.index_topk],
                "idx_topk_indices_batch",
            )?,
            // Compressor batched-GEMV scratch — main coff=2, idx coff=2.
            comp_main_kv_batch: alloc(gpu, &[max_batch, 2 * head_dim], "comp_main_kv_batch")?,
            comp_main_score_batch: alloc(gpu, &[max_batch, 2 * head_dim], "comp_main_score_batch")?,
            comp_idx_kv_batch: alloc(
                gpu,
                &[max_batch, 2 * cfg.index_head_dim],
                "comp_idx_kv_batch",
            )?,
            comp_idx_score_batch: alloc(
                gpu,
                &[max_batch, 2 * cfg.index_head_dim],
                "comp_idx_score_batch",
            )?,
            // Scatter-by-expert MoE sort scratch.
            moe_sorted_b: alloc(gpu, &[max_batch * cfg.num_experts_per_tok], "moe_sorted_b")?,
            moe_sorted_krank: alloc(
                gpu,
                &[max_batch * cfg.num_experts_per_tok],
                "moe_sorted_krank",
            )?,
            moe_sorted_expert: alloc(
                gpu,
                &[max_batch * cfg.num_experts_per_tok],
                "moe_sorted_expert",
            )?,
            moe_expert_starts: alloc(gpu, &[cfg.n_routed_experts + 1], "moe_expert_starts")?,
            // SGLang-style scatter-grouped MoE pipeline (chunk_size ≥ 256 only).
            // Sized for the worst-case `m_total_max = B*K_TOP + n_exp*BLOCK_M`
            // padded layout. All i32 buffers stored as Raw so byte-counts
            // match the kernels (which read int32 indices).
            moe_expert_token_counts: {
                let n = cfg.n_routed_experts;
                gpu.zeros(&[n * 4], DType::Raw)
                    .map_err(|e| format!("alloc moe_expert_token_counts: {e:?}"))?
            },
            moe_expert_offsets: {
                let n = cfg.n_routed_experts + 1;
                gpu.zeros(&[n * 4], DType::Raw)
                    .map_err(|e| format!("alloc moe_expert_offsets: {e:?}"))?
            },
            moe_sorted_slot_index: {
                let block_m = 16;
                let m_total_max =
                    max_batch * cfg.num_experts_per_tok + cfg.n_routed_experts * block_m;
                gpu.zeros(&[m_total_max * 4], DType::Raw)
                    .map_err(|e| format!("alloc moe_sorted_slot_index: {e:?}"))?
            },
            moe_expert_tile_ids: {
                let block_m = 16;
                let m_total_max =
                    max_batch * cfg.num_experts_per_tok + cfg.n_routed_experts * block_m;
                let n_tiles = m_total_max / block_m;
                gpu.zeros(&[n_tiles * 4], DType::Raw)
                    .map_err(|e| format!("alloc moe_expert_tile_ids: {e:?}"))?
            },
            moe_inverse_perm: {
                let n = max_batch * cfg.num_experts_per_tok;
                gpu.zeros(&[n * 4], DType::Raw)
                    .map_err(|e| format!("alloc moe_inverse_perm: {e:?}"))?
            },
            moe_y_gate_up_grouped: {
                let block_m = 16;
                let m_total_max =
                    max_batch * cfg.num_experts_per_tok + cfg.n_routed_experts * block_m;
                alloc(
                    gpu,
                    &[m_total_max, 2 * cfg.moe_intermediate_size],
                    "moe_y_gate_up_grouped",
                )?
            },
            moe_x_grouped: {
                let block_m = 16;
                let m_total_max =
                    max_batch * cfg.num_experts_per_tok + cfg.n_routed_experts * block_m;
                alloc(
                    gpu,
                    &[m_total_max, cfg.moe_intermediate_size],
                    "moe_x_grouped",
                )?
            },
            moe_y_down_grouped: {
                let block_m = 16;
                let m_total_max =
                    max_batch * cfg.num_experts_per_tok + cfg.n_routed_experts * block_m;
                alloc(gpu, &[m_total_max, hidden], "moe_y_down_grouped")?
            },
            // F16 staging buffers: 2 bytes per element. Allocate as Raw
            // with byte-count shape so DType::size() == 1 stays consistent.
            tmp_batch_f16: {
                let nbytes = max_batch * hidden * 2;
                let mut t = gpu
                    .zeros(&[nbytes], DType::Raw)
                    .map_err(|e| format!("PBS alloc tmp_batch_f16: {e:?}"))?;
                t.dtype = DType::F16;
                t.shape = vec![max_batch, hidden];
                t
            },
            tmp_plain_batch_f16: {
                let nbytes = max_batch * hidden * 2;
                let mut t = gpu
                    .zeros(&[nbytes], DType::Raw)
                    .map_err(|e| format!("PBS alloc tmp_plain_batch_f16: {e:?}"))?;
                t.dtype = DType::F16;
                t.shape = vec![max_batch, hidden];
                t
            },
            comp_positions: alloc(gpu, &[max_batch], "comp_positions")?,
            // Per-batch device-side state mirrors of state.pos_array_device
            // and state.attn_state_buf. Sized to cover B = max_batch rows.
            // POS_SLOTS_PER_LAYER = 3 per layer, ATTN_STATE_SLOTS = 10 total.
            pos_array_device_batch: alloc(
                gpu,
                &[max_batch * (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER],
                "pos_array_device_batch",
            )?,
            attn_state_buf_batch: alloc(gpu, &[max_batch * 10], "attn_state_buf_batch")?,
            mtp_tokens_batch: alloc(gpu, &[max_batch], "mtp_tokens_batch")?,
            mtp_embed_batch: alloc(gpu, &[max_batch, hidden], "mtp_embed_batch")?,
            mtp_e_norm_batch: alloc(gpu, &[max_batch, hidden], "mtp_e_norm_batch")?,
            mtp_h_norm_batch: alloc(gpu, &[max_batch, hc_mult, hidden], "mtp_h_norm_batch")?,
            mtp_x_e_batch: alloc(gpu, &[max_batch, hidden], "mtp_x_e_batch")?,
            wmma_x_scratch_f16: {
                // Cover the largest x-tensor size across all batched
                // WMMA call sites. wo_a's input is [B, G, per_group_in]
                // where per_group_in = (n_heads/n_groups) * head_dim —
                // can exceed `hidden` for DeepSeek V4 (G=8, per_group_in=4096
                // ⇒ G*per_group_in = 32768).
                let per_group_in = (n_heads / cfg.o_groups) * head_dim;
                let max_dim = cfg.o_groups * cfg.o_lora_rank;
                let max_dim = max_dim
                    .max(hidden)
                    .max(cfg.q_lora_rank)
                    .max(cfg.o_groups * per_group_in);
                let nbytes = max_batch * max_dim * 2;
                let mut t = gpu
                    .zeros(&[nbytes], DType::Raw)
                    .map_err(|e| format!("PBS alloc wmma_x_scratch_f16: {e:?}"))?;
                t.dtype = DType::F16;
                t.shape = vec![max_batch, max_dim];
                t
            },
        })
    }

    /// Release every GPU buffer this prefill-batch scratch owns back to
    /// the pool. Consumes self. Called from `unload_model` on idle
    /// eviction / explicit unload so the ~50 sizeable per-chunk buffers
    /// (embed_batch, streams_batch, swa_staged_batch, MoE grouped
    /// scratches, …) actually return their VRAM rather than leaking.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.embed_batch,
            self.streams_batch,
            self.tokens,
            self.tmp_batch,
            self.tmp_plain_batch,
            self.q_lat_batch,
            self.q_lat_rot_batch,
            self.q_batch,
            self.q_head_ones,
            self.kv_batch,
            self.positions,
            self.hc_c_batch,
            self.hc_pre_batch,
            self.hc_post_batch,
            self.hc_comb_batch,
            self.hc_x_in_batch,
            self.attn_out_batch,
            self.ffn_out_batch,
            self.streams_out_batch,
            self.swa_staged_batch,
            self.topk_staged_batch,
            self.n_valid_swa_arr,
            self.n_active_topk_arr,
            self.attn_out_raw_batch,
            self.attn_out_raw_rot_batch,
            self.wo_a_out_batch,
            self.wo_a_out_rot_batch,
            self.ffn_x_rot_batch,
            self.ffn_x_plain_batch,
            self.ffn_shared_gate_batch,
            self.ffn_shared_up_batch,
            self.ffn_shared_rot_batch,
            self.moe_scores_batch,
            self.moe_topk_indices_batch,
            self.moe_topk_weights_batch,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_down_expert_outputs,
            self.idx_q_batch,
            self.idx_w_batch,
            self.idx_scores_batch,
            self.idx_topk_indices_batch,
            self.comp_main_kv_batch,
            self.comp_main_score_batch,
            self.comp_idx_kv_batch,
            self.comp_idx_score_batch,
            self.moe_sorted_b,
            self.moe_sorted_krank,
            self.moe_sorted_expert,
            self.moe_expert_starts,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_expert_tile_ids,
            self.moe_inverse_perm,
            self.moe_y_gate_up_grouped,
            self.moe_x_grouped,
            self.moe_y_down_grouped,
            self.tmp_batch_f16,
            self.tmp_plain_batch_f16,
            self.wmma_x_scratch_f16,
            self.comp_positions,
            self.pos_array_device_batch,
            self.attn_state_buf_batch,
            self.mtp_tokens_batch,
            self.mtp_embed_batch,
            self.mtp_e_norm_batch,
            self.mtp_h_norm_batch,
            self.mtp_x_e_batch,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// Batched twin of `hc_attn_mix` for Phase B2 chunk forward.
///
/// X_{l+1}[b] = comb[b] · X_l[b] + post[b] · attn_out[b]
/// where comb, post are from the latest mhc_pre_batched(is_attn=true) call.
/// The mix output is written into pbs.streams_out_batch, then copied
/// back into pbs.streams_batch (mirrors the sequential pattern of
/// staging into state.q before the d2d memcpy).
#[allow(dead_code)]
fn hc_attn_mix_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<(), String> {
    gpu.hc_mix_4stream_batched(
        &pbs.streams_batch,
        &pbs.hc_comb_batch,
        &pbs.hc_post_batch,
        &pbs.attn_out_batch,
        &pbs.streams_out_batch,
        cfg.hidden_size as i32,
        batch_size as i32,
    )
    .map_err(|e| format!("hc_mix_4stream_batched (attn): {e:?}"))?;

    let bytes = batch_size * cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&pbs.streams_batch.buf, &pbs.streams_out_batch.buf, bytes)
        .map_err(|e| format!("d2d streams_out → streams: {e:?}"))?;
    Ok(())
}

/// Pure-SWA batched attention block (compress_ratio == 0 layers).
///
/// Stages:
///   1. Lazy-alloc state._attention[L].swa_k / swa_v rings (per layer)
///   2. swa_visibility_stage_batched: pre-chunk ring + within-chunk
///      kv_batch → pbs.swa_staged_batch [B, head_dim, swa_window]
///   3. Upload per-batch n_valid_swa_arr
///   4. deepseek4_attn_swa_batched (K=V tied: pass swa_staged for both args)
///      → pbs.attn_out_raw_batch
///   5. Inverse tail RoPE (per-layer YaRN params)
///   6. FWHT rotate attn_out_raw_batch → attn_out_raw_rot_batch
///   7. wo_per_group_batched_f32 → pbs.wo_a_out_batch (F32 wo_a only)
///   8. FWHT rotate wo_a_out_batch → wo_a_out_rot_batch
///   9. gemv_auto_batched_wmma(wo_b, ..., pbs.attn_out_batch, Some(&pbs.wmma_x_scratch_f16))
///   10. swa_ring_write_batched: advance ring with chunk's KVs
///
/// hc_attn_mix_batched is called by the chunk-forward caller after
/// this returns (mirrors the sequential ordering).
#[allow(dead_code)]
fn attention_block_batched_swa_only(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let attn_sink = layer
        .attn_sink
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_sink missing"))?;
    let wo_a = layer
        .wo_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_a missing"))?;
    let wo_b = layer
        .wo_b
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_b missing"))?;

    let n_kv = cfg.num_key_value_heads;
    let win = cfg.sliding_window;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let n_groups = cfg.o_groups;
    let o_lora_rank = cfg.o_lora_rank;
    let groups_o_lora = n_groups * o_lora_rank;

    // 1. Lazy-alloc the per-layer SWA ring (zero-init: pre-chunk
    //    visibility for early positions reads zero history correctly).
    {
        let attn = &mut state._attention[layer_idx];
        if attn.swa_k.is_none() {
            attn.swa_k = Some(
                gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("alloc swa_k l{layer_idx}: {e:?}"))?,
            );
        }
        if attn.swa_v.is_none() {
            attn.swa_v = Some(
                gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("alloc swa_v l{layer_idx}: {e:?}"))?,
            );
        }
    }
    let swa_k_ref = state._attention[layer_idx]
        .swa_k
        .as_ref()
        .unwrap()
        .buf
        .as_ptr();
    let swa_v_ref = state._attention[layer_idx]
        .swa_v
        .as_ref()
        .unwrap()
        .buf
        .as_ptr();
    let _ = (swa_k_ref, swa_v_ref); // borrow workaround handled below

    // 2. Stage per-batch SWA visibility window from pre-chunk ring +
    //    within-chunk kv_batch. DeepSeek V4 K=V tied so we only stage once and
    //    pass swa_staged_batch as both K and V args.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        gpu.swa_visibility_stage_batched(
            swa_k,
            &pbs.kv_batch,
            &pbs.swa_staged_batch,
            start_pos as i32,
            win as i32,
            head_dim as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("swa_visibility_stage_batched l{layer_idx}: {e:?}"))?;
    }
    if layer_idx == 0 {
        dump_buf(gpu, "06a_l0_swa_staged", &pbs.swa_staged_batch);
    }

    // 3. n_valid_swa_arr is uploaded once per chunk by
    //    `forward_prefill_batch_chunk`. Skip the per-layer htod.

    // 4. deepseek4_attn_swa_batched. o_groups passed through for ABI parity
    //    (unused inside the kernel).
    //
    // DEBUG: HIPFIRE_DEEPSEEK4_ATTN_PER_POS=1 substitutes a per-position loop
    // calling `deepseek4_attn_swa` (the sequential sibling). Used to isolate
    // whether deepseek4_attn_swa_batched-specific non-determinism is the cause,
    // vs a deeper issue shared with the per-position kernel.
    if std::env::var("HIPFIRE_DEEPSEEK4_ATTN_PER_POS").as_deref() == Ok("1") {
        let q_per = n_heads * head_dim;
        let kv_per = head_dim * win;
        let out_per = n_heads * head_dim;
        for b in 0..batch_size {
            let q_view = pbs.q_batch.sub_offset(b * q_per, q_per);
            let k_view = pbs.swa_staged_batch.sub_offset(b * kv_per, kv_per);
            let v_view = pbs.swa_staged_batch.sub_offset(b * kv_per, kv_per);
            let out_view = pbs.attn_out_raw_batch.sub_offset(b * out_per, out_per);
            let n_valid = ((start_pos as usize + b + 1).min(win)) as i32;
            gpu.deepseek4_attn_swa(
                &q_view,
                &k_view,
                &v_view,
                attn_sink,
                &out_view,
                n_heads as i32,
                head_dim as i32,
                n_groups as i32,
                n_valid,
                win as i32,
            )
            .map_err(|e| format!("deepseek4_attn_swa per-pos b={b} l{layer_idx}: {e:?}"))?;
        }
    } else {
        gpu.deepseek4_attn_swa_batched(
            &pbs.q_batch,
            &pbs.swa_staged_batch,
            &pbs.swa_staged_batch,
            attn_sink,
            &pbs.n_valid_swa_arr,
            &pbs.attn_out_raw_batch,
            n_heads as i32,
            head_dim as i32,
            n_groups as i32,
            win as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("deepseek4_attn_swa_batched l{layer_idx}: {e:?}"))?;
    }
    if layer_idx == 0 {
        dump_buf(gpu, "06b_l0_attn_swa_raw", &pbs.attn_out_raw_batch);
    }

    // DEBUG: same-process twin-call test (HIPFIRE_DEEPSEEK4_ATTN_TWIN=1).
    if layer_idx == 0 && std::env::var("HIPFIRE_DEEPSEEK4_ATTN_TWIN").as_deref() == Ok("1") {
        gpu.deepseek4_attn_swa_batched(
            &pbs.q_batch,
            &pbs.swa_staged_batch,
            &pbs.swa_staged_batch,
            attn_sink,
            &pbs.n_valid_swa_arr,
            &pbs.attn_out_raw_batch,
            n_heads as i32,
            head_dim as i32,
            n_groups as i32,
            win as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("deepseek4_attn_swa_batched twin l{layer_idx}: {e:?}"))?;
        dump_buf(gpu, "06b2_l0_attn_swa_raw_twin", &pbs.attn_out_raw_batch);
    }

    // DEBUG: in-kernel bisect (HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT=1).
    // Re-runs the kernel via the debug variant which also writes
    // max_score and sum_exp per (h, b) so we can compare across runs
    // and find which intermediate first diverges.
    if layer_idx == 0 && std::env::var("HIPFIRE_DEEPSEEK4_ATTN_DEBUG_BISECT").as_deref() == Ok("1") {
        // Lazy-alloc debug scratch on the GPU on first call.
        let n_h = n_heads;
        let debug_max = gpu
            .alloc_tensor(&[batch_size, n_h], rdna_compute::DType::F32)
            .map_err(|e| format!("alloc debug_max: {e:?}"))?;
        let debug_sumexp = gpu
            .alloc_tensor(&[batch_size, n_h], rdna_compute::DType::F32)
            .map_err(|e| format!("alloc debug_sumexp: {e:?}"))?;
        gpu.deepseek4_attn_swa_batched_debug(
            &pbs.q_batch,
            &pbs.swa_staged_batch,
            &pbs.swa_staged_batch,
            attn_sink,
            &pbs.n_valid_swa_arr,
            &pbs.attn_out_raw_batch,
            &debug_max,
            &debug_sumexp,
            n_heads as i32,
            head_dim as i32,
            n_groups as i32,
            win as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("deepseek4_attn_swa_batched_debug l{layer_idx}: {e:?}"))?;
        dump_buf(gpu, "06b_dbg_max", &debug_max);
        dump_buf(gpu, "06b_dbg_sumexp", &debug_sumexp);
        dump_buf(gpu, "06b_dbg_attn_out", &pbs.attn_out_raw_batch);
    }

    // 5. Inverse tail RoPE on attn_out_raw_batch.
    {
        let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
            layer_rope_params(cfg, layer.compress_ratio);
        // n_heads_k=0: K already written + tail-rope'd at kv_joint
        // time; only un-rotate Q-tail-equivalents in attn_out.
        gpu.rope_tail_yarn_interleaved_batched(
            &pbs.attn_out_raw_batch,
            &pbs.attn_out_raw_batch,
            &pbs.positions,
            n_heads as i32,
            0,
            head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            corr_low,
            corr_high,
            /*inverse=*/ 1,
            batch_size as i32,
        )
        .map_err(|e| format!("rope_tail_yarn_interleaved_batched (inv) l{layer_idx}: {e:?}"))?;
    }
    if layer_idx == 0 {
        dump_buf(gpu, "06c_l0_inv_rope_raw", &pbs.attn_out_raw_batch);
    }

    // 6. FWHT rotate attn_out_raw_batch → attn_out_raw_rot_batch.
    //    Skip if wo_a doesn't need FWHT input (Q8/F16/F32 weights).
    if weight_needs_fwht(wo_a) {
        gpu.rotate_x_mq_batched(
            &pbs.attn_out_raw_batch,
            &pbs.attn_out_raw_rot_batch,
            n_heads * head_dim,
            batch_size,
        )
        .map_err(|e| format!("rotate attn_out_raw_batch l{layer_idx}: {e:?}"))?;
    }
    if layer_idx == 0 {
        dump_buf(gpu, "06d_l0_attn_raw_rot", &pbs.attn_out_raw_rot_batch);
    }

    // 7. wo_a per-group batched.
    //    F32     → wo_per_group_batched_f32 (single launch).
    //    HFQ4G256→ wo_per_group_batched_hfq4g256 (single launch, MQ4 prerotated).
    //    Q8_0    → wo_per_group_batched_q8_0 (single launch, plain input).
    let per_group_in = (n_heads / n_groups) * head_dim;
    match wo_a.dtype {
        DType::F32 => {
            gpu.wo_per_group_batched_f32(
                wo_a,
                &pbs.attn_out_raw_batch,
                &pbs.wo_a_out_batch,
                n_groups as i32,
                o_lora_rank as i32,
                per_group_in as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("wo_per_group_batched_f32 l{layer_idx}: {e:?}"))?;
        }
        DType::Q8_0 => {
            // Q8_0 contract: plain (non-FWHT) input. attn_out_raw_batch
            // is [B, n_heads * head_dim] viewable as [B, G, per_group_in].
            // Multi-row variant if HIPFIRE_DEEPSEEK4_WO_MULTIROW=2 or 4.
            let mr: i32 = std::env::var("HIPFIRE_DEEPSEEK4_WO_MULTIROW")
                .ok().and_then(|v| v.parse().ok()).filter(|&r| r == 2 || r == 4)
                .unwrap_or(0);
            if mr == 0 {
                gpu.wo_per_group_batched_q8_0(
                    wo_a,
                    &pbs.attn_out_raw_batch,
                    &pbs.wo_a_out_batch,
                    n_groups as i32,
                    o_lora_rank as i32,
                    per_group_in as i32,
                    batch_size as i32,
                )
                .map_err(|e| format!("wo_per_group_batched_q8_0 l{layer_idx}: {e:?}"))?;
            } else {
                gpu.wo_per_group_batched_q8_0_multirow(
                    wo_a,
                    &pbs.attn_out_raw_batch,
                    &pbs.wo_a_out_batch,
                    n_groups as i32,
                    o_lora_rank as i32,
                    per_group_in as i32,
                    batch_size as i32,
                    mr,
                )
                .map_err(|e| format!("wo_per_group_batched_q8_0_multirow l{layer_idx}: {e:?}"))?;
            }
        }
        DType::Raw => {
            // MQ4G256 (HFQ4-packed weights, FWHT-rotated input).
            gpu.wo_per_group_batched_hfq4g256(
                wo_a,
                &pbs.attn_out_raw_rot_batch,
                &pbs.wo_a_out_batch,
                n_groups as i32,
                o_lora_rank as i32,
                per_group_in as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("wo_per_group_batched_hfq4g256 l{layer_idx}: {e:?}"))?;
        }
        other => {
            return Err(format!(
                "attention_block_batched_mixed l{layer_idx}: unsupported wo_a dtype {other:?}"
            ));
        }
    }

    // 8. FWHT rotate wo_a_out_batch → wo_a_out_rot_batch.
    //    Skip if wo_b doesn't need FWHT input.
    if weight_needs_fwht(wo_b) {
        gpu.rotate_x_mq_batched(
            &pbs.wo_a_out_batch,
            &pbs.wo_a_out_rot_batch,
            groups_o_lora,
            batch_size,
        )
        .map_err(|e| format!("rotate wo_a_out l{layer_idx}: {e:?}"))?;
    }

    if layer_idx == 0 {
        dump_buf(gpu, "06e_l0_wo_a_out", &pbs.wo_a_out_batch);
        dump_buf(gpu, "06f_l0_wo_a_out_rot", &pbs.wo_a_out_rot_batch);
    }

    // 9. wo_b GEMV batched: wo_a_out_rot_batch → attn_out_batch.
    //    Standard non-block-diagonal GEMV; gemv_auto_batched handles
    //    F32/Q8/MQ4 dispatch.
    gemv_auto_batched_wmma(
        gpu,
        wo_b,
        &pbs.wo_a_out_rot_batch,
        &pbs.wo_a_out_batch,
        &pbs.attn_out_batch,
        cfg.hidden_size,
        groups_o_lora,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // 10. Advance the SWA ring with this chunk's KVs for future steps.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch,
            swa_k,
            n_kv as i32,
            head_dim as i32,
            win as i32,
            start_pos as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("swa_ring_write_batched (k) l{layer_idx}: {e:?}"))?;
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch,
            swa_v,
            n_kv as i32,
            head_dim as i32,
            win as i32,
            start_pos as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("swa_ring_write_batched (v) l{layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Mixed-attention batched dispatch (compress_ratio > 0 layers).
///
/// DeepSeek V4's compressed layers attend jointly to (SWA window K/V) +
/// (top-K of compressed-K cache, gated by the indexer for ratio=4 or
/// the identity gather for ratio=128). The compressor + indexer
/// pipelines per-position are stateful (writes to kv_state ring,
/// conditional pool to main/indexer_kv_cache); we loop those
/// sequentially per batch position by temporarily swapping the
/// per-position state.* fields with sub_offset views into the
/// batched scratch buffers. The big-fish attention kernel still runs
/// in one batched launch.
///
/// Stages:
///   1. SWA visibility staging from pre-chunk ring + within-chunk kv_batch
///   2. For each batch position b:
///      a. Swap state.tmp / tmp_plain / q_lat / q_lat_rot to b's slice
///      b. compressor_forward(main, position=start_pos+b)
///      c. compressor_forward(indexer, position=start_pos+b) for ratio=4
///      d. indexer_forward → state._indexer[L].topk_idx_indices
///      e. Gather top-K K/V into pbs.topk_staged_batch[b] slot OR
///         identity-gather for ratio=128
///      f. Compute n_active_topk[b] = min(n_compressed, index_topk)
///   3. Upload n_valid_swa_arr + n_active_topk_arr
///   4. deepseek4_attn_swa_topk_batched_f32 (single launch over all batch rows)
///   5. Inverse RoPE batched
///   6. FWHT rotate attn_out_raw → attn_out_raw_rot
///   7. wo_per_group_batched_f32 (F32 wo_a only)
///   8. FWHT rotate wo_a_out → wo_a_out_rot
///   9. gemv_auto_batched_wmma(wo_b → attn_out_batch, Some(&pbs.wmma_x_scratch_f16))
///   10. swa_ring_write_batched
///
/// Errors out cleanly on non-F32 wo_a (Q8/MQ4 need separate per-group
/// batched kernels) or when the compressor/indexer state isn't
/// allocated.
#[allow(dead_code)]
fn attention_block_batched_mixed(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    start_pos: u32,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ratio = layer.compress_ratio as usize;
    assert!(
        ratio > 0,
        "attention_block_batched_mixed called on dense layer"
    );

    let attn_sink = layer
        .attn_sink
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_sink missing"))?;
    let wo_a = layer
        .wo_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_a missing"))?;
    let wo_b = layer
        .wo_b
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wo_b missing"))?;

    let n_kv = cfg.num_key_value_heads;
    let win = cfg.sliding_window;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;
    let n_groups = cfg.o_groups;
    let o_lora_rank = cfg.o_lora_rank;
    let groups_o_lora = n_groups * o_lora_rank;
    let topk_max = cfg.index_topk;
    let use_topk_direct = ratio == 4
        && std::env::var("HIPFIRE_DEEPSEEK4_ATTN_TOPK_DIRECT")
            .map(|s| s != "0")
            .unwrap_or(gpu.arch == "gfx1151" && batch_size >= 64);
    let mut topk_direct_n_compressed = 0usize;

    // Lazy-alloc SWA rings.
    {
        let attn = &mut state._attention[layer_idx];
        if attn.swa_k.is_none() {
            attn.swa_k = Some(
                gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("alloc swa_k l{layer_idx}: {e:?}"))?,
            );
        }
        if attn.swa_v.is_none() {
            attn.swa_v = Some(
                gpu.zeros(&[n_kv, head_dim, win], DType::F32)
                    .map_err(|e| format!("alloc swa_v l{layer_idx}: {e:?}"))?,
            );
        }
        if attn.gathered_k.is_none() {
            attn.gathered_k = Some(
                gpu.zeros(&[n_kv, head_dim, topk_max], DType::F32)
                    .map_err(|e| format!("alloc gathered_k l{layer_idx}: {e:?}"))?,
            );
        }
    }

    // 1. Stage per-batch SWA visibility window.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        gpu.swa_visibility_stage_batched(
            swa_k,
            &pbs.kv_batch,
            &pbs.swa_staged_batch,
            start_pos as i32,
            win as i32,
            head_dim as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("swa_visibility_stage_batched l{layer_idx}: {e:?}"))?;
    }

    // 2a. Compressor commits (sequential per batch — stateful ring writes
    //     and conditional pools to indexer/main_kv_cache). MUST run before
    //     the batched indexer chain so n_filled[b] reflects all relevant
    //     commits. We swap state.* fields to point at per-row sub-views.
    // n_valid_swa_arr is uploaded once per chunk by
    // `forward_prefill_batch_chunk` — same value for every layer in the
    // chunk (depends only on start_pos, batch_size, sliding_window).

    // Snapshot the per-token state fields so we can restore after the loop.
    let orig_tmp = state.tmp.take();
    let orig_tmp_plain = state.tmp_plain.take();
    let orig_q_lat = state.q_lat.take();
    let orig_q_lat_rot = state.q_lat_rot.take();

    let hidden = cfg.hidden_size;
    let q_rank = cfg.q_lora_rank;
    let mut loop_err: Option<String> = None;

    // 2a-pre. Batched compressor GEMVs for the whole chunk. Collapses
    // 2 × batch_size sequential gemv_auto calls into ONE batched GEMM
    // per (wkv|wgate) × (main|indexer). Wires through to
    // compressor_forward_prebatched in the per-position loop below.
    //
    // WMMA fast path: when all four compressor weights have F16-native
    // copies (`compressor_w{kv,gate}_f16` etc.), convert the F32 inputs
    // to F16 once and run gemm_f16_x_f16_wmma — measured 26× faster
    // than the F32 register-tiled path on DeepSeek V4 shapes (microbench).
    // Opt out via HIPFIRE_DEEPSEEK4_COMP_F16_WMMA=0.
    let comp_f16_wmma = std::env::var("HIPFIRE_DEEPSEEK4_COMP_F16_WMMA")
        .map(|s| s != "0")
        .unwrap_or(true);
    let main_coff = 2; // ratio=4 has overlap=true; ratio=128 has coff=1 → wastes half the buf.
    let main_proj_dim = main_coff * head_dim;
    let idx_coff = 2;
    let idx_proj_dim = idx_coff * cfg.index_head_dim;
    {
        let comp_wkv = layer
            .compressor_wkv
            .as_ref()
            .ok_or_else(|| format!("comp_wkv l{layer_idx}"))?;
        let comp_wgate = layer
            .compressor_wgate
            .as_ref()
            .ok_or_else(|| format!("comp_wgate l{layer_idx}"))?;
        let real_main_proj = if ratio == 4 { 2 * head_dim } else { head_dim };
        // WMMA route requires both main + idx (when ratio=4) F16 weights
        // and works on F16 inputs.
        let wkv_f16 = layer.compressor_wkv_f16.as_ref();
        let wgate_f16 = layer.compressor_wgate_f16.as_ref();
        let idx_wkv_f16 = layer.indexer_compressor_wkv_f16.as_ref();
        let idx_wgate_f16 = layer.indexer_compressor_wgate_f16.as_ref();
        let have_idx_f16 = ratio != 4 || (idx_wkv_f16.is_some() && idx_wgate_f16.is_some());
        let use_wmma = comp_f16_wmma && wkv_f16.is_some() && wgate_f16.is_some() && have_idx_f16;
        if use_wmma {
            // Stage F32 → F16 inputs once per layer.
            let n_inputs = (batch_size * hidden) as i64;
            gpu.deepseek4_convert_f32_to_f16(&pbs.tmp_batch, &pbs.tmp_batch_f16, n_inputs)
                .map_err(|e| format!("convert_f32_to_f16 tmp l{layer_idx}: {e:?}"))?;
            gpu.deepseek4_convert_f32_to_f16(&pbs.tmp_plain_batch, &pbs.tmp_plain_batch_f16, n_inputs)
                .map_err(|e| format!("convert_f32_to_f16 tmp_plain l{layer_idx}: {e:?}"))?;
            // DeepSeek V4 compressor uses FWHT-rotated input (tmp_batch) when the
            // weight is MQ4-style, and plain input (tmp_plain_batch) when
            // F16/F32. We're on the F16 path → tmp_plain_batch_f16.
            gpu.gemm_f16_x_f16_wmma(
                wkv_f16.unwrap(),
                &pbs.tmp_plain_batch_f16,
                &pbs.comp_main_kv_batch,
                real_main_proj,
                hidden,
                batch_size,
            )
            .map_err(|e| format!("gemm_f16_wmma comp_wkv l{layer_idx}: {e:?}"))?;
            gpu.gemm_f16_x_f16_wmma(
                wgate_f16.unwrap(),
                &pbs.tmp_plain_batch_f16,
                &pbs.comp_main_score_batch,
                real_main_proj,
                hidden,
                batch_size,
            )
            .map_err(|e| format!("gemm_f16_wmma comp_wgate l{layer_idx}: {e:?}"))?;
            if ratio == 4 {
                gpu.gemm_f16_x_f16_wmma(
                    idx_wkv_f16.unwrap(),
                    &pbs.tmp_plain_batch_f16,
                    &pbs.comp_idx_kv_batch,
                    idx_proj_dim,
                    hidden,
                    batch_size,
                )
                .map_err(|e| format!("gemm_f16_wmma idx_wkv l{layer_idx}: {e:?}"))?;
                gpu.gemm_f16_x_f16_wmma(
                    idx_wgate_f16.unwrap(),
                    &pbs.tmp_plain_batch_f16,
                    &pbs.comp_idx_score_batch,
                    idx_proj_dim,
                    hidden,
                    batch_size,
                )
                .map_err(|e| format!("gemm_f16_wmma idx_wgate l{layer_idx}: {e:?}"))?;
            }
        } else {
            gemv_auto_batched_wmma(
                gpu,
                comp_wkv,
                &pbs.tmp_batch,
                &pbs.tmp_plain_batch,
                &pbs.comp_main_kv_batch,
                real_main_proj,
                hidden,
                batch_size,
                Some(&pbs.wmma_x_scratch_f16),
            )?;
            gemv_auto_batched_wmma(
                gpu,
                comp_wgate,
                &pbs.tmp_batch,
                &pbs.tmp_plain_batch,
                &pbs.comp_main_score_batch,
                real_main_proj,
                hidden,
                batch_size,
                Some(&pbs.wmma_x_scratch_f16),
            )?;
            if ratio == 4 {
                let idx_wkv = layer
                    .indexer_compressor_wkv
                    .as_ref()
                    .ok_or_else(|| format!("idx_comp_wkv l{layer_idx}"))?;
                let idx_wgate = layer
                    .indexer_compressor_wgate
                    .as_ref()
                    .ok_or_else(|| format!("idx_comp_wgate l{layer_idx}"))?;
                gemv_auto_batched_wmma(
                    gpu,
                    idx_wkv,
                    &pbs.tmp_batch,
                    &pbs.tmp_plain_batch,
                    &pbs.comp_idx_kv_batch,
                    idx_proj_dim,
                    hidden,
                    batch_size,
                    Some(&pbs.wmma_x_scratch_f16),
                )?;
                gemv_auto_batched_wmma(
                    gpu,
                    idx_wgate,
                    &pbs.tmp_batch,
                    &pbs.tmp_plain_batch,
                    &pbs.comp_idx_score_batch,
                    idx_proj_dim,
                    hidden,
                    batch_size,
                    Some(&pbs.wmma_x_scratch_f16),
                )?;
            }
        }
    }
    // The pre-batched buffers are stored at stride main_proj_dim (=1024)
    // even when ratio=128 (proj_dim=512). For ratio=128 the second half
    // of each [B, 1024] slot is unused but still strided. That matches
    // the alloc but means the per-position offset uses the real proj_dim.
    let main_view_proj = if ratio == 4 { main_proj_dim } else { head_dim };

    // PHASE A: batched commit/compress for the whole chunk in one call
    // per (main, indexer) per layer. Replaces the per-batch loop when
    // start_pos % ratio == 0 (aligned chunk).
    let comp_fully_batched = (start_pos as usize).is_multiple_of(ratio);

    if comp_fully_batched {
        if let Err(e) = compressor_forward_batched(
            cfg, weights, state, pbs, gpu, layer_idx, start_pos, batch_size,
            /*is_indexer=*/ false,
        ) {
            loop_err = Some(format!(
                "compressor_forward_batched(main) l{layer_idx}: {e}"
            ));
        }
        if loop_err.is_none() && ratio == 4 {
            if let Err(e) = compressor_forward_batched(
                cfg, weights, state, pbs, gpu, layer_idx, start_pos, batch_size,
                /*is_indexer=*/ true,
            ) {
                loop_err = Some(format!("compressor_forward_batched(idx) l{layer_idx}: {e}"));
            }
        }
    } else {
        // Option B (2026-05-21): populate per-batch pos_array_device +
        // attn_state_buf in pbs ONCE for this chunk. The per-position
        // compressor kernels read indices from `state.pos_array_device`
        // and `state.attn_state_buf` — which only hold ONE position's
        // slots. To support the per-position fallback for ANY chunk
        // (including unaligned ones for ratio=128 layers), swap those
        // pointers to per-batch sub-views inside the loop.
        if let Err(e) = precompute_positions_batched(cfg, pbs, gpu, start_pos, batch_size) {
            loop_err = Some(format!("precompute_positions_batched l{layer_idx}: {e}"));
        }
        if loop_err.is_none() {
            if let Err(e) = precompute_attn_state_batched(cfg, pbs, gpu, start_pos, batch_size) {
                loop_err = Some(format!("precompute_attn_state_batched l{layer_idx}: {e}"));
            }
        }

        let slots_per_pos = (cfg.num_hidden_layers + 1) * POS_SLOTS_PER_LAYER;
        let attn_state_slots = 10;

        // Snapshot per-position state pointers so we can restore after
        // the loop. Decode-time (B=1) uses these; we transiently replace
        // them with per-batch sub-views.
        let orig_pos_array_device = state.pos_array_device.take();
        let orig_attn_state_buf = state.attn_state_buf.take();

        if loop_err.is_none() {
            for b in 0..batch_size {
                let pos = start_pos + b as u32;
                state.tmp = Some(pbs.tmp_batch.sub_offset(b * hidden, hidden));
                state.tmp_plain = Some(pbs.tmp_plain_batch.sub_offset(b * hidden, hidden));
                state.q_lat = Some(pbs.q_lat_batch.sub_offset(b * q_rank, q_rank));
                state.q_lat_rot = Some(pbs.q_lat_rot_batch.sub_offset(b * q_rank, q_rank));
                // Per-batch sub-views into the chunk-level device buffers.
                // Layout: stripe b starts at offset (b * stripe) for both.
                state.pos_array_device = Some(
                    pbs.pos_array_device_batch
                        .sub_offset(b * slots_per_pos, slots_per_pos),
                );
                state.attn_state_buf = Some(
                    pbs.attn_state_buf_batch
                        .sub_offset(b * attn_state_slots, attn_state_slots),
                );

                let _ = main_proj_dim;
                let cf_res = compressor_forward_prebatched(
                    cfg,
                    weights,
                    state,
                    gpu,
                    layer_idx,
                    pos,
                    /*is_indexer=*/ false,
                    &pbs.comp_main_kv_batch,
                    &pbs.comp_main_score_batch,
                    b,
                );
                if let Err(e) = cf_res {
                    loop_err = Some(format!("compressor_forward(main) b={b} l{layer_idx}: {e}"));
                    break;
                }
                if ratio == 4 {
                    let cf_res2 = compressor_forward_prebatched(
                        cfg,
                        weights,
                        state,
                        gpu,
                        layer_idx,
                        pos,
                        /*is_indexer=*/ true,
                        &pbs.comp_idx_kv_batch,
                        &pbs.comp_idx_score_batch,
                        b,
                    );
                    if let Err(e) = cf_res2 {
                        loop_err = Some(format!("compressor_forward(idx) b={b} l{layer_idx}: {e}"));
                        break;
                    }
                }
            }
        }

        // Restore decode-time per-position state pointers.
        state.pos_array_device = orig_pos_array_device;
        state.attn_state_buf = orig_attn_state_buf;
    }
    let _ = main_view_proj;

    // Restore per-token state fields before any potential early-return.
    state.tmp = orig_tmp;
    state.tmp_plain = orig_tmp_plain;
    state.q_lat = orig_q_lat;
    state.q_lat_rot = orig_q_lat_rot;
    if let Some(e) = loop_err {
        return Err(e);
    }

    // 2b. Batched indexer chain (ratio == 4 only) OR batched identity gather
    //     (ratio == 128). Replaces the per-batch indexer_forward + gather
    //     loop with one batched call per stage.
    let mut n_active_host: Vec<i32> = vec![0; batch_size];
    if ratio == 4 {
        let wq_b_idx = layer
            .indexer_wq_b
            .as_ref()
            .ok_or_else(|| format!("idx wq_b l{layer_idx}"))?;
        let weights_proj = layer
            .indexer_weights_proj
            .as_ref()
            .ok_or_else(|| format!("idx weights_proj l{layer_idx}"))?;
        let h_idx = cfg.index_n_heads;
        let d_idx = cfg.index_head_dim;

        // Per-batch n_filled = (start_pos+b+1)/ratio, clamped.
        // n_max across batch = max value, used as kernel's per-batch cap.
        let max_compressed: usize = std::env::var("HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048);
        let n_per_batch_host: Vec<i32> = (0..batch_size)
            .map(|b| (((start_pos as usize) + b + 1) / ratio).min(max_compressed) as i32)
            .collect();
        let n_max_chunk = *n_per_batch_host.iter().max().unwrap_or(&0) as usize;
        topk_direct_n_compressed = n_max_chunk;
        if n_max_chunk == 0 {
            // No commits yet — nothing to score/gather. n_active_topk stays 0.
        } else {
            // Upload n_per_batch via the existing n_active_topk_arr buffer
            // (repurposed temporarily — we'll overwrite it below with the
            // actual k_active values).
            let np_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(n_per_batch_host.as_ptr() as *const u8, batch_size * 4)
            };
            gpu.memcpy_htod_auto(&pbs.n_active_topk_arr.buf, np_bytes)
                .map_err(|e| format!("htod n_per_batch: {e:?}"))?;

            // wq_b_idx GEMV batched: q_lat_rot_batch → q_idx_batch.
            gemv_auto_batched_wmma(
                gpu,
                wq_b_idx,
                &pbs.q_lat_rot_batch,
                &pbs.q_lat_batch,
                &pbs.idx_q_batch,
                h_idx * d_idx,
                q_rank,
                batch_size,
                Some(&pbs.wmma_x_scratch_f16),
            )?;

            // Tail RoPE on q_idx_batch with compress_rope_theta.
            gpu.rope_tail_interleaved_batched(
                &pbs.idx_q_batch,
                &pbs.idx_q_batch,
                &pbs.positions,
                h_idx as i32,
                0,
                d_idx as i32,
                cfg.qk_rope_head_dim as i32,
                cfg.compress_rope_theta,
                batch_size as i32,
            )
            .map_err(|e| format!("rope_tail_batched idx l{layer_idx}: {e:?}"))?;

            // weights_proj GEMV batched: tmp_batch → idx_w_batch.
            gemv_auto_batched_wmma(
                gpu,
                weights_proj,
                &pbs.tmp_batch,
                &pbs.tmp_plain_batch,
                &pbs.idx_w_batch,
                h_idx,
                hidden,
                batch_size,
                Some(&pbs.wmma_x_scratch_f16),
            )?;

            // Batched scoring. Pass the SCORE BUFFER STRIDE (max_compressed,
            // = the allocated row stride of pbs.idx_scores_batch), not the
            // chunk's n_max_chunk. The kernel writes scores[b * stride + n];
            // slots with n >= n_per_batch[b] get -inf and slots with
            // n >= n_max_chunk read uninit K_cache data but also get -inf
            // (since n_per_batch[b] ≤ n_max_chunk ≤ n).
            let kv_cache = state._indexer[layer_idx]
                .indexer_kv_cache
                .as_ref()
                .ok_or_else(|| "indexer_kv_cache missing".to_string())?;
            // WMMA fast path: gated on gfx1100+ (RDNA3+) and the canonical
            // DeepSeek V4 indexer shape (H=64, D=128). 8-9% of prefill
            // PMC vs the F32 scalar baseline — Phase C1 of the prefill
            // catch-up plan. Opt out via HIPFIRE_DEEPSEEK4_INDEXER_WMMA=0.
            let use_indexer_wmma = h_idx == 64
                && d_idx == 128
                && (gpu.arch.starts_with("gfx11") || gpu.arch.starts_with("gfx12"))
                && std::env::var("HIPFIRE_DEEPSEEK4_INDEXER_WMMA")
                    .map(|s| s != "0")
                    .unwrap_or(true);
            if use_indexer_wmma {
                gpu.indexer_relu_score_wmma_batched_f32(
                    &pbs.idx_q_batch,
                    kv_cache,
                    &pbs.idx_w_batch,
                    &pbs.n_active_topk_arr,
                    &pbs.idx_scores_batch,
                    h_idx as i32,
                    d_idx as i32,
                    max_compressed as i32,
                    batch_size as i32,
                )
                .map_err(|e| format!("indexer_relu_score_wmma_batched l{layer_idx}: {e:?}"))?;
            } else {
                gpu.indexer_relu_score_batched_f32(
                    &pbs.idx_q_batch,
                    kv_cache,
                    &pbs.idx_w_batch,
                    &pbs.n_active_topk_arr, // reuse buffer: holds n_per_batch right now
                    &pbs.idx_scores_batch,
                    h_idx as i32,
                    d_idx as i32,
                    max_compressed as i32,
                    batch_size as i32,
                )
                .map_err(|e| format!("indexer_relu_score_batched l{layer_idx}: {e:?}"))?;
            }

            // Batched top-K. n_stride = max_compressed (storage),
            // n_iter = n_max_chunk (actual range with valid scores),
            // k_stride = topk_max (storage), k_fill = min(topk_max,
            // n_max_chunk). The bound matters a LOT — at low context
            // n_max_chunk ≈ 8 vs max_compressed = 2048, which is
            // ~100× iteration savings.
            let k_fill = topk_max.min(n_max_chunk);
            gpu.indexer_top_k_batched(
                &pbs.idx_scores_batch,
                &pbs.idx_topk_indices_batch,
                /*n_idx_heads=*/ 1,
                max_compressed as i32,
                n_max_chunk as i32,
                topk_max as i32,
                k_fill as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("indexer_top_k_batched l{layer_idx}: {e:?}"))?;

            // Batched gather: top-K K/V → pbs.topk_staged_batch. Pass
            // K=topk_max (storage stride); -1 indices write zeros.
            let main_kv_cache = state._indexer[layer_idx]
                .main_kv_cache
                .as_ref()
                .ok_or_else(|| "main_kv_cache missing".to_string())?;
            if !use_topk_direct {
                gpu.deepseek4_topk_kv_gather_batched_f32(
                    main_kv_cache,
                    &pbs.idx_topk_indices_batch,
                    &pbs.topk_staged_batch,
                    topk_max as i32,
                    head_dim as i32,
                    n_max_chunk as i32,
                    topk_max as i32,
                    0,
                    /*scale=*/ 1.0,
                    batch_size as i32,
                )
                .map_err(|e| format!("deepseek4_topk_kv_gather_batched l{layer_idx}: {e:?}"))?;
            }

            // n_active_topk[b] = min(topk_max, n_per_batch[b]) — top-K
            // returned -1 sentinels past n_per_batch[b], and gather wrote
            // zeros there. Cap attention's visible-slot count to the
            // actual valid range per batch row.
            for b in 0..batch_size {
                n_active_host[b] = topk_max.min(n_per_batch_host[b] as usize) as i32;
            }
        }
    } else {
        // ratio == 128: identity gather, no indexer. Per-batch n_compressed.
        let max_compressed: usize = std::env::var("HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048);
        let max_n_compressed = (((start_pos as usize) + batch_size) / ratio)
            .min(max_compressed)
            .min(topk_max);
        if max_n_compressed > 0 {
            let main_kv_cache = state._indexer[layer_idx]
                .main_kv_cache
                .as_ref()
                .ok_or_else(|| "main_kv_cache missing".to_string())?;
            gpu.deepseek4_topk_kv_gather_identity_batched_f32(
                main_kv_cache,
                &pbs.topk_staged_batch,
                max_n_compressed as i32,
                head_dim as i32,
                topk_max as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("deepseek4_topk_kv_gather_identity_batched l{layer_idx}: {e:?}"))?;
            for b in 0..batch_size {
                let n_b = (((start_pos as usize) + b + 1) / ratio)
                    .min(max_compressed)
                    .min(topk_max);
                n_active_host[b] = n_b as i32;
            }
        }
    }

    // 3. Upload per-batch n_active_topk_arr only — n_valid_swa_arr is
    //    populated once per chunk by the caller.
    let n_active_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(n_active_host.as_ptr() as *const u8, batch_size * 4) };
    gpu.memcpy_htod_auto(&pbs.n_active_topk_arr.buf, n_active_bytes)
        .map_err(|e| format!("htod n_active_topk_arr: {e:?}"))?;

    // 4. Batched joint-softmax attention over SWA + topK + sink.
    if use_topk_direct {
        let main_kv_cache = state._indexer[layer_idx]
            .main_kv_cache
            .as_ref()
            .ok_or_else(|| "main_kv_cache missing".to_string())?;
        // Head-batched f16-WMMA DSA attention (~4.4× the f32 kernel at prefill
        // batch); falls back to f32 if disabled, shapes don't tile, or the
        // score LDS would exceed 64 KB. max_n_total bounds the LDS (n_valid ≤ win).
        let use_dsa_wmma = std::env::var("HIPFIRE_DEEPSEEK4_DSA_WMMA").as_deref() != Ok("0")
            && gpu.arch_caps.has_wmma()
            && n_heads % 16 == 0
            && head_dim % 16 == 0;
        let max_n_total = win as i32 + n_active_host.iter().copied().max().unwrap_or(0);
        let mut done = false;
        if use_dsa_wmma {
            if gpu
                .deepseek4_attn_swa_topk_direct_wmma(
                    &pbs.q_batch,
                    &pbs.swa_staged_batch, // K=V tied
                    main_kv_cache,
                    &pbs.idx_topk_indices_batch,
                    attn_sink,
                    &pbs.n_valid_swa_arr,
                    &pbs.n_active_topk_arr,
                    &pbs.attn_out_raw_batch,
                    n_heads as i32,
                    head_dim as i32,
                    win as i32,
                    topk_max as i32,
                    topk_direct_n_compressed as i32,
                    batch_size as i32,
                    max_n_total,
                )
                .is_ok()
            {
                done = true;
            }
        }
        if !done {
            gpu.deepseek4_attn_swa_topk_direct_batched_f32(
                &pbs.q_batch,
                &pbs.swa_staged_batch,
                &pbs.swa_staged_batch, // K=V tied
                main_kv_cache,
                &pbs.idx_topk_indices_batch,
                attn_sink,
                &pbs.n_valid_swa_arr,
                &pbs.n_active_topk_arr,
                &pbs.attn_out_raw_batch,
                n_heads as i32,
                head_dim as i32,
                win as i32,
                topk_max as i32,
                topk_direct_n_compressed as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("deepseek4_attn_swa_topk_direct_batched l{layer_idx}: {e:?}"))?;
        }
    } else {
        // Head-batched f16-WMMA gathered DSA attention; f32 fallback on
        // disable / non-tiling shapes / LDS > 64 KB.
        let use_dsa_wmma = std::env::var("HIPFIRE_DEEPSEEK4_DSA_WMMA").as_deref() != Ok("0")
            && gpu.arch_caps.has_wmma()
            && n_heads % 16 == 0
            && head_dim % 16 == 0;
        let max_n_total = win as i32 + n_active_host.iter().copied().max().unwrap_or(0);
        let mut done = false;
        if use_dsa_wmma {
            if gpu
                .deepseek4_attn_swa_topk_batched_wmma(
                    &pbs.q_batch,
                    &pbs.swa_staged_batch,  // K=V tied
                    &pbs.topk_staged_batch, // K=V tied
                    attn_sink,
                    &pbs.n_valid_swa_arr,
                    &pbs.n_active_topk_arr,
                    &pbs.attn_out_raw_batch,
                    n_heads as i32,
                    head_dim as i32,
                    win as i32,
                    topk_max as i32,
                    batch_size as i32,
                    max_n_total,
                )
                .is_ok()
            {
                done = true;
            }
        }
        if !done {
            gpu.deepseek4_attn_swa_topk_batched_f32(
                &pbs.q_batch,
                &pbs.swa_staged_batch,
                &pbs.swa_staged_batch, // K=V tied
                &pbs.topk_staged_batch,
                &pbs.topk_staged_batch,
                attn_sink,
                &pbs.n_valid_swa_arr,
                &pbs.n_active_topk_arr,
                &pbs.attn_out_raw_batch,
                n_heads as i32,
                head_dim as i32,
                win as i32,
                topk_max as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("deepseek4_attn_swa_topk_batched l{layer_idx}: {e:?}"))?;
        }
    }

    // 5. Inverse RoPE.
    {
        let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
            layer_rope_params(cfg, layer.compress_ratio);
        gpu.rope_tail_yarn_interleaved_batched(
            &pbs.attn_out_raw_batch,
            &pbs.attn_out_raw_batch,
            &pbs.positions,
            n_heads as i32,
            0,
            head_dim as i32,
            cfg.qk_rope_head_dim as i32,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            corr_low,
            corr_high,
            /*inverse=*/ 1,
            batch_size as i32,
        )
        .map_err(|e| format!("rope_tail_yarn_inv_batched l{layer_idx}: {e:?}"))?;
    }

    // 6. FWHT rotate attn_out_raw_batch → attn_out_raw_rot_batch.
    gpu.rotate_x_mq_batched(
        &pbs.attn_out_raw_batch,
        &pbs.attn_out_raw_rot_batch,
        n_heads * head_dim,
        batch_size,
    )
    .map_err(|e| format!("rotate attn_out_raw l{layer_idx}: {e:?}"))?;

    // 7. wo_a per-group batched.
    //    F32     → wo_per_group_batched_f32 (single launch).
    //    HFQ4G256→ wo_per_group_batched_hfq4g256 (single launch).
    //    Q8_0    → wo_per_group_batched_q8_0 (single launch, plain input).
    let per_group_in = (n_heads / n_groups) * head_dim;
    match wo_a.dtype {
        DType::F32 => {
            gpu.wo_per_group_batched_f32(
                wo_a,
                &pbs.attn_out_raw_batch,
                &pbs.wo_a_out_batch,
                n_groups as i32,
                o_lora_rank as i32,
                per_group_in as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("wo_per_group_batched_f32 l{layer_idx}: {e:?}"))?;
        }
        DType::Q8_0 => {
            // Q8_0 contract: plain (non-FWHT) input. Same layout
            // assumption as the swa-only sibling.
            let mr: i32 = std::env::var("HIPFIRE_DEEPSEEK4_WO_MULTIROW")
                .ok().and_then(|v| v.parse().ok()).filter(|&r| r == 2 || r == 4)
                .unwrap_or(0);
            if mr == 0 {
                gpu.wo_per_group_batched_q8_0(
                    wo_a,
                    &pbs.attn_out_raw_batch,
                    &pbs.wo_a_out_batch,
                    n_groups as i32,
                    o_lora_rank as i32,
                    per_group_in as i32,
                    batch_size as i32,
                )
                .map_err(|e| format!("wo_per_group_batched_q8_0 l{layer_idx}: {e:?}"))?;
            } else {
                gpu.wo_per_group_batched_q8_0_multirow(
                    wo_a,
                    &pbs.attn_out_raw_batch,
                    &pbs.wo_a_out_batch,
                    n_groups as i32,
                    o_lora_rank as i32,
                    per_group_in as i32,
                    batch_size as i32,
                    mr,
                )
                .map_err(|e| format!("wo_per_group_batched_q8_0_multirow l{layer_idx}: {e:?}"))?;
            }
        }
        DType::Raw => {
            gpu.wo_per_group_batched_hfq4g256(
                wo_a,
                &pbs.attn_out_raw_rot_batch,
                &pbs.wo_a_out_batch,
                n_groups as i32,
                o_lora_rank as i32,
                per_group_in as i32,
                batch_size as i32,
            )
            .map_err(|e| format!("wo_per_group_batched_hfq4g256 l{layer_idx}: {e:?}"))?;
        }
        other => {
            return Err(format!(
                "attention_block_batched_swa_only l{layer_idx}: unsupported wo_a dtype {other:?}"
            ));
        }
    }

    // 8. FWHT rotate wo_a_out → wo_a_out_rot.
    gpu.rotate_x_mq_batched(
        &pbs.wo_a_out_batch,
        &pbs.wo_a_out_rot_batch,
        groups_o_lora,
        batch_size,
    )
    .map_err(|e| format!("rotate wo_a_out l{layer_idx}: {e:?}"))?;

    // 9. wo_b GEMV batched.
    gemv_auto_batched_wmma(
        gpu,
        wo_b,
        &pbs.wo_a_out_rot_batch,
        &pbs.wo_a_out_batch,
        &pbs.attn_out_batch,
        cfg.hidden_size,
        groups_o_lora,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // 10. Advance the SWA ring.
    {
        let swa_k = state._attention[layer_idx].swa_k.as_ref().unwrap();
        let swa_v = state._attention[layer_idx].swa_v.as_ref().unwrap();
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch,
            swa_k,
            n_kv as i32,
            head_dim as i32,
            win as i32,
            start_pos as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("swa_ring_write_batched (k) l{layer_idx}: {e:?}"))?;
        gpu.swa_ring_write_batched_f32(
            &pbs.kv_batch,
            swa_v,
            n_kv as i32,
            head_dim as i32,
            win as i32,
            start_pos as i32,
            batch_size as i32,
        )
        .map_err(|e| format!("swa_ring_write_batched (v) l{layer_idx}: {e:?}"))?;
    }

    Ok(())
}

/// Batched FFN: shared expert + routed-expert MoE, end-to-end.
///
/// Computes per-batch ffn_out_batch[b, :] = shared_expert(hc_x_in[b])
/// + (if score-routed) Σ_k topk_w[b,k] · routed_expert_{topk_idx[b,k]}(hc_x_in[b])
///
/// Stages:
///   1. fused_rmsnorm_rotate_mq_batched(hc_x_in → ffn_x_rot)
///   2. rmsnorm_batched(hc_x_in → ffn_x_plain)
///   3. gemv_auto_batched_wmma(shared_w1, → shared_gate, Some(&pbs.wmma_x_scratch_f16))
///   4. gemv_auto_batched_wmma(shared_w3, → shared_up, Some(&pbs.wmma_x_scratch_f16))
///   5. deepseek4_silu_mul_clamp_f32_batched(shared_gate, shared_up → shared_gate)
///   6. rotate_x_mq_batched(shared_gate → shared_rot)
///   7. gemv_auto_batched_wmma(shared_w2, shared_rot → ffn_out_batch, Some(&pbs.wmma_x_scratch_f16))
///   8. (score-routed only) gemv_auto_batched_wmma(gate.weight, ffn_x_rot → moe_scores, Some(&pbs.wmma_x_scratch_f16))
///   9. sqrt_softplus_f32 on moe_scores (operates on full [B*n_exp] numel)
///   10. deepseek4_moe_topk_bias_aware_batched_f32 → topk_indices, topk_weights
///   11. deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched → moe_gate, moe_up
///   12. deepseek4_silu_mul_clamp_f32_batched(B*k_top streams of MI) → moe_gate
///   13. rotate_x_mq_batched(B*k_top FWHT rotations) → moe_rot
///   14. deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_batched
///       (atomicAdds routed expert outputs into ffn_out_batch with scale)
///
/// Hash-routed layers (layer_idx < num_hash_layers) skip steps 8-14.
/// DeepSeek V4's hash routing uses static tid2eid lookup which is skipped at
/// quant time per the load_weights logic; falls back to shared-only.
#[allow(dead_code)]
fn ffn_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
    tokens: &[u32],
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let shared_w1 = layer.shared_w1.as_ref().unwrap();
    let shared_w2 = layer.shared_w2.as_ref().unwrap();
    let shared_w3 = layer.shared_w3.as_ref().unwrap();

    let hidden = cfg.hidden_size;
    let im = cfg.moe_intermediate_size;

    // Skip dead FWHT rotations on prefill (mirror decode-path FWHT skip).
    // Routed MoE consumes ffn_x_rot_batch (MQ2-Lloyd → needs FWHT), so keep
    // the gate/up rotation alive when MoE is on. Down rotation only feeds
    // shared_w2 — gate purely on shared_w2 dtype.
    let moe_will_run = env_cache::moe_on();
    let gate_up_need_fwht =
        moe_will_run || weight_needs_fwht(shared_w1) || weight_needs_fwht(shared_w3);
    let down_needs_fwht = weight_needs_fwht(shared_w2);

    // 1. RMSNorm (+ optional FWHT). Fused variant writes BOTH rot and
    //    plain outputs when both are needed (saves one launch per layer).
    if gate_up_need_fwht {
        gpu.fused_rmsnorm_rotate_mq_plain_batched(
            &pbs.hc_x_in_batch,
            ffn_norm,
            &pbs.ffn_x_rot_batch,
            &pbs.ffn_x_plain_batch,
            hidden,
            cfg.rms_norm_eps,
            batch_size,
        )
        .map_err(|e| format!("fused_rmsnorm_rotate_mq_plain_batched ffn l{layer_idx}: {e:?}"))?;
    } else {
        // Pure-plain (no MoE AND no MQ4 shared): only ffn_x_plain needed.
        gpu.rmsnorm_batched(
            &pbs.hc_x_in_batch,
            ffn_norm,
            &pbs.ffn_x_plain_batch,
            batch_size,
            hidden,
            cfg.rms_norm_eps,
        )
        .map_err(|e| format!("rmsnorm_batched ffn-side l{layer_idx}: {e:?}"))?;
    }

    // 2-3. Shared expert gate + up GEMVs.
    gemv_auto_batched_wmma(
        gpu,
        shared_w1,
        &pbs.ffn_x_rot_batch,
        &pbs.ffn_x_plain_batch,
        &pbs.ffn_shared_gate_batch,
        im,
        hidden,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;
    gemv_auto_batched_wmma(
        gpu,
        shared_w3,
        &pbs.ffn_x_rot_batch,
        &pbs.ffn_x_plain_batch,
        &pbs.ffn_shared_up_batch,
        im,
        hidden,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // 4. SwiGLU + clamp. The kernel batches `B` streams of length `n`.
    gpu.deepseek4_silu_mul_clamp_f32_batched(
        &pbs.ffn_shared_gate_batch,
        &pbs.ffn_shared_up_batch,
        &pbs.ffn_shared_gate_batch,
        im,
        batch_size,
        cfg.swiglu_limit,
    )
    .map_err(|e| format!("deepseek4_silu_mul_clamp_f32_batched shared l{layer_idx}: {e:?}"))?;

    // 5. FWHT rotate silu output — skip if shared_w2 doesn't need FWHT.
    if down_needs_fwht {
        gpu.rotate_x_mq_batched(
            &pbs.ffn_shared_gate_batch,
            &pbs.ffn_shared_rot_batch,
            im,
            batch_size,
        )
        .map_err(|e| format!("rotate_x_mq_batched shared silu l{layer_idx}: {e:?}"))?;
    }

    // 6. Shared down GEMV → ffn_out_batch.
    gemv_auto_batched_wmma(
        gpu,
        shared_w2,
        &pbs.ffn_shared_rot_batch,
        &pbs.ffn_shared_gate_batch,
        &pbs.ffn_out_batch,
        hidden,
        im,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // ── Routed-expert MoE ───────────────────────────────────────────
    let do_routed = std::env::var("HIPFIRE_DEEPSEEK4_MOE").ok().as_deref() != Some("0")
        && layer.expert_gate_up_blob.is_some()
        && layer.expert_w2_blob.is_some();
    if !do_routed {
        return Ok(());
    }
    // Layers 0..num_hash_layers use STATIC tid2eid routing per upstream DeepSeek V4.
    let hash_routing = layer_idx < cfg.num_hash_layers;
    if hash_routing && layer.tid2eid_host.is_empty() {
        return Ok(());
    }

    let gate_w = layer
        .gate_weight
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} gate.weight missing"))?;
    let gate_up_ptrs = layer.expert_gate_up_ptrs.as_ref().unwrap();
    let w2_ptrs = layer.expert_w2_ptrs.as_ref().unwrap();
    let n_exp = cfg.n_routed_experts;
    let k_top = cfg.num_experts_per_tok;
    let route_scale: f32 = std::env::var("HIPFIRE_DEEPSEEK4_ROUTE_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.2);

    // 8. Router GEMV: gate.weight @ ffn_x_rot_batch → moe_scores [B, n_exp].
    gemv_auto_batched_wmma(
        gpu,
        gate_w,
        &pbs.ffn_x_rot_batch,
        &pbs.ffn_x_plain_batch,
        &pbs.moe_scores_batch,
        n_exp,
        hidden,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // 9. sqrt_softplus over the full [B, n_exp] buffer.
    gpu.sqrt_softplus_f32(&pbs.moe_scores_batch)
        .map_err(|e| format!("sqrt_softplus_f32 moe scores l{layer_idx}: {e:?}"))?;

    // Routing + routed experts + combine now run through the centralized MoE
    // family (Ship 4.3 prefill). The router GEMV + sqrt_softplus (above) and the
    // shared expert stay model-owned; the family routes (hash or bias-aware),
    // runs the experts (grouped GEMM at B>=gate, else scalar K4), and
    // accumulates into ffn_out_batch (already holding the shared-expert output).
    let routing = if hash_routing {
        if tokens.len() < batch_size {
            return Err(format!(
                "ffn_batched l{layer_idx}: tokens len {} < batch_size {}",
                tokens.len(),
                batch_size,
            ));
        }
        let tid2eid_dev = layer.tid2eid_dev.as_ref().ok_or_else(|| {
            format!(
                "ffn_batched hash l{layer_idx}: tid2eid_dev missing (pre-FP4 \
                 quant skipped tid2eid; HFQ load_weights should still populate \
                 the device buffer)"
            )
        })?;
        hipfire_dispatch::families::moe::MoePrefillRouting::Hash {
            tid2eid: tid2eid_dev,
            tokens: &pbs.tokens,
        }
    } else {
        let gate_bias = layer
            .gate_bias
            .as_ref()
            .ok_or_else(|| format!("layer {layer_idx} gate.bias missing"))?;
        hipfire_dispatch::families::moe::MoePrefillRouting::BiasAware { gate_bias }
    };

    let moe_params = hipfire_dispatch::families::moe::MoeBiasAwarePrefillParams {
        hidden,
        mi: im,
        n_exp,
        k_top,
        batch_size,
        route_scale,
        swiglu_limit: cfg.swiglu_limit,
        layer_idx,
        routing,
        scores: &pbs.moe_scores_batch,
        topk_indices: &pbs.moe_topk_indices_batch,
        topk_weights: &pbs.moe_topk_weights_batch,
        expert_gate_up_ptrs: gate_up_ptrs,
        expert_down_ptrs: w2_ptrs,
        x_rot: &pbs.ffn_x_rot_batch,
        ffn_out: &pbs.ffn_out_batch,
        expert_token_counts: &pbs.moe_expert_token_counts,
        expert_offsets: &pbs.moe_expert_offsets,
        sorted_slot_index: &pbs.moe_sorted_slot_index,
        expert_tile_ids: &pbs.moe_expert_tile_ids,
        inverse_perm: &pbs.moe_inverse_perm,
        y_gate_up_grouped: &pbs.moe_y_gate_up_grouped,
        y_down_grouped: &pbs.moe_y_down_grouped,
        gate_batch: &pbs.moe_gate_batch,
        up_batch: &pbs.moe_up_batch,
        rot_batch: &pbs.moe_rot_batch,
        down_expert_outputs: &pbs.moe_down_expert_outputs,
    };
    hipfire_runtime::llama::moe_family()
        .run_bias_aware_prefill(gpu, &moe_params)
        .map_err(|e| format!("ffn_batched l{layer_idx} dispatch: {e}"))?;

    Ok(())
}

/// Batched-aware twin of `final_norm_and_head` — extracts the LAST
/// position's residual streams from pbs.streams_batch and runs the
/// existing per-position head pipeline against it.
///
/// Phase B2 chunk forward only needs logits at the last position
/// (matches qwen35::forward_prefill_batch's contract). All upstream
/// state.* scratch fields used by `final_norm_and_head` are sized for
/// one position and get reused unchanged.
///
/// Returns the logits at the last position. Caller is responsible for
/// any sampler integration.
#[allow(dead_code)]
pub fn final_norm_and_head_last_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<Vec<f32>, String> {
    if batch_size == 0 {
        return Err("final_norm_and_head_last_batched: empty batch".to_string());
    }
    // Snapshot the original residual_streams so we can restore it (the
    // sequential function reads/keeps state.residual_streams; we point
    // it temporarily at the last position's slice).
    let last_off = (batch_size - 1) * cfg.hc_mult * cfg.hidden_size;
    let last_len = cfg.hc_mult * cfg.hidden_size;
    let last_streams = pbs.streams_batch.sub_offset(last_off, last_len);

    let orig = state.residual_streams.take();
    state.residual_streams = Some(last_streams);

    let result = final_norm_and_head(cfg, weights, state, gpu);

    // Restore. Drop the temporary view (it shares the pbs buffer; the
    // underlying buffer is owned by pbs, so leaking the view is fine —
    // it's a thin GpuTensor wrapper, not a fresh allocation).
    state.residual_streams = orig;

    result?;
    let logits_tensor = state
        .logits
        .as_ref()
        .ok_or_else(|| "logits not allocated".to_string())?;
    gpu.download_f32(logits_tensor)
        .map_err(|e| format!("download logits: {e:?}"))
}

/// Run final_norm + head on EVERY position of the batched chunk.
///
/// `final_norm_and_head_last_batched` only produces logits for the last
/// position (the only position whose token is sampled in normal prefill).
/// Speculative-decode verification needs per-position logits so each
/// draft can be compared against the verifier's preferred token —
/// that's what this helper provides.
///
/// Cost: K invocations of the per-position final_norm_and_head pipeline
/// (head HC + RMSNorm + rotate + lm_head GEMV + d2h). The lm_head is
/// [vocab=129280, hidden=4096]; per-position it's well under 5 ms on
/// gfx1151, so K=8 takes <40 ms. Acceptable for spec-decode windows.
///
/// Returns Vec<Vec<f32>> of length `batch_size`, each inner Vec sized
/// `vocab_size`.
pub fn final_norm_and_head_all_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, String> {
    if batch_size == 0 {
        return Err("final_norm_and_head_all_batched: empty batch".to_string());
    }
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    // An MQ4 head needs a per-position FWHT rotation of the normed input that
    // we don't stage into the batch buffer — keep the scalar per-position loop
    // for that case. Q8/F16/F32 heads (this build's head is Q8) take the
    // batched path below.
    let head_needs_fwht = {
        let head = weights
            .head
            .as_ref()
            .ok_or_else(|| "head not uploaded".to_string())?;
        weight_needs_fwht(head)
    };
    // Opt-out: HIPFIRE_DEEPSEEK4_BATCH_HEAD=0 forces the legacy per-position
    // scalar loop — used for A/B measurement and as a safety fallback.
    let batch_head = std::env::var("HIPFIRE_DEEPSEEK4_BATCH_HEAD")
        .map(|s| s != "0")
        .unwrap_or(true);
    if head_needs_fwht || !batch_head {
        let orig = state.residual_streams.take();
        let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        let result: Result<(), String> = (|| {
            for i in 0..batch_size {
                let off = i * stream_len;
                let streams_i = pbs.streams_batch.sub_offset(off, stream_len);
                state.residual_streams = Some(streams_i);
                final_norm_and_head(cfg, weights, state, gpu)?;
                let logits_tensor = state
                    .logits
                    .as_ref()
                    .ok_or_else(|| "logits not allocated".to_string())?;
                let logits_host = gpu
                    .download_f32(logits_tensor)
                    .map_err(|e| format!("download logits @pos {i}: {e:?}"))?;
                all_logits.push(logits_host);
            }
            Ok(())
        })();
        state.residual_streams = orig;
        result?;
        return Ok(all_logits);
    }

    // ── Batched lm_head path ──────────────────────────────────────────────
    // The lm_head weight is `[vocab, hidden]` (~565 MB Q8) and its GEMV is
    // pure weight-bandwidth-bound (~2.4 ms on gfx1151). The scalar loop above
    // re-reads that whole weight once PER verify position. Here we run only
    // the cheap per-position prologue (head-HC + RMSNorm, ~22 µs) into a
    // `[K, hidden]` buffer, then issue ONE batched GEMV that reads the weight
    // a single time for all K positions. Buffers are cached on `state` so the
    // hot spec-decode loop allocates them once.
    if state
        .head_norm_batch
        .as_ref()
        .map(|t| t.numel() != batch_size * hidden)
        .unwrap_or(true)
    {
        state.head_norm_batch = Some(
            gpu.alloc_tensor(&[batch_size, hidden], DType::F32)
                .map_err(|e| format!("alloc head_norm_batch: {e:?}"))?,
        );
    }
    if state
        .head_x_f16
        .as_ref()
        .map(|t| t.numel() != batch_size * hidden)
        .unwrap_or(true)
    {
        state.head_x_f16 = Some(
            gpu.alloc_tensor(&[batch_size * hidden], DType::F16)
                .map_err(|e| format!("alloc head_x_f16: {e:?}"))?,
        );
    }
    if state
        .head_logits_batch
        .as_ref()
        .map(|t| t.numel() != batch_size * vocab)
        .unwrap_or(true)
    {
        state.head_logits_batch = Some(
            gpu.alloc_tensor(&[batch_size, vocab], DType::F32)
                .map_err(|e| format!("alloc head_logits_batch: {e:?}"))?,
        );
    }

    let orig = state.residual_streams.take();
    let result: Result<(), String> = (|| {
        for i in 0..batch_size {
            let off = i * stream_len;
            let streams_i = pbs.streams_batch.sub_offset(off, stream_len);
            state.residual_streams = Some(streams_i);
            final_norm_compute(cfg, weights, state, gpu)?;
            // Stage this position's plain normed activation into row i of the
            // `[K, hidden]` batched GEMV input.
            let fn_i = state
                .final_norm
                .as_ref()
                .ok_or_else(|| "final_norm not allocated".to_string())?;
            let dst = state.head_norm_batch.as_ref().unwrap();
            let dst_row = dst.sub_offset(i * hidden, hidden);
            gpu.memcpy_dtod_auto(&dst_row.buf, &fn_i.buf, hidden * 4)
                .map_err(|e| format!("stage final_norm → batch @pos {i}: {e:?}"))?;
        }
        Ok(())
    })();
    state.residual_streams = orig;
    result?;

    // ONE batched lm_head GEMV over all K positions (weight read once). The
    // head is Q8/F16/F32 here so `x_rotated_batch` is ignored; pass the plain
    // batch for both. `Some(x_f16)` selects the proven WMMA route.
    let head = weights
        .head
        .as_ref()
        .ok_or_else(|| "head not uploaded".to_string())?;
    let norm_batch = state.head_norm_batch.as_ref().unwrap();
    let logits_batch = state.head_logits_batch.as_ref().unwrap();
    let x_f16 = state.head_x_f16.as_ref().unwrap();
    gemv_auto_batched_wmma(
        gpu,
        head,
        norm_batch,
        norm_batch,
        logits_batch,
        vocab,
        hidden,
        batch_size,
        Some(x_f16),
    )?;

    // Download the `[K, vocab]` logits block once, split per position.
    let flat = gpu
        .download_f32(logits_batch)
        .map_err(|e| format!("download batched logits: {e:?}"))?;
    let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        all_logits.push(flat[i * vocab..(i + 1) * vocab].to_vec());
    }
    Ok(all_logits)
}

/// Batched twin of `hc_ffn_mix`. Same shape as `hc_attn_mix_batched`
/// but mixes the FFN-side post/comb (produced by the second
/// mhc_pre_batched call with is_attn=false) and the FFN's transform
/// output `pbs.ffn_out_batch`.
#[allow(dead_code)]
fn hc_ffn_mix_batched(
    cfg: &DeepseekV4Config,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    batch_size: usize,
) -> Result<(), String> {
    gpu.hc_mix_4stream_batched(
        &pbs.streams_batch,
        &pbs.hc_comb_batch,
        &pbs.hc_post_batch,
        &pbs.ffn_out_batch,
        &pbs.streams_out_batch,
        cfg.hidden_size as i32,
        batch_size as i32,
    )
    .map_err(|e| format!("hc_mix_4stream_batched (ffn): {e:?}"))?;

    let bytes = batch_size * cfg.hc_mult * cfg.hidden_size * 4;
    gpu.memcpy_dtod_auto(&pbs.streams_batch.buf, &pbs.streams_out_batch.buf, bytes)
        .map_err(|e| format!("d2d streams_out → streams: {e:?}"))?;
    Ok(())
}

/// Batched twin of `mhc_pre` for Phase B2 chunk forward.
///
/// Per batch position b, after this returns:
///   pbs.hc_pre_batch[b, :]  = sigmoid(c[b, 0..4])
///   pbs.hc_post_batch[b, :] = post_scale * sigmoid(c[b, 4..8])
///   pbs.hc_comb_batch[b, :, :] = Sinkhorn(c[b, 8..24])
///   pbs.hc_x_in_batch[b, :] = sum_h hc_pre_batch[b, h] · streams[b, h, :]
///
/// where c is the post-α-rescale control vector. The split into separate
/// pre/post/comb buffers avoids strided sigmoid_f32 calls on the [B, 24]
/// layout (per-row segments are not memory-contiguous).
///
/// `is_attn` selects attn-side vs FFN-side W_fn / base / scale.
/// `HIPFIRE_DEEPSEEK4_POST_SCALE` env override (default 1.5) is honoured.
#[allow(dead_code)]
fn mhc_pre_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    is_attn: bool,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let (hc_fn, hc_base, hc_scale) = if is_attn {
        (
            layer.hc_attn_fn.as_ref().unwrap(),
            layer.hc_attn_base.as_ref().unwrap(),
            layer.hc_attn_scale.as_ref().unwrap(),
        )
    } else {
        (
            layer.hc_ffn_fn.as_ref().unwrap(),
            layer.hc_ffn_base.as_ref().unwrap(),
            layer.hc_ffn_scale.as_ref().unwrap(),
        )
    };

    let n_ctrl = 24usize;
    let x_dim = cfg.hidden_size * cfg.hc_mult;
    let post_scale: f32 = std::env::var("HIPFIRE_DEEPSEEK4_POST_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5);

    // 1. c = streams · W_fn · rsqrt(mean) + base. Per-batch.
    gpu.hc_compute_control_batched(
        &pbs.streams_batch,
        hc_fn,
        hc_base,
        &pbs.hc_c_batch,
        n_ctrl as i32,
        x_dim as i32,
        batch_size as i32,
    )
    .map_err(|e| format!("hc_compute_control_batched l{layer_idx}: {e:?}"))?;

    // 2. α-rescale c in place per batch.
    gpu.hc_apply_alpha_batched(&pbs.hc_c_batch, hc_scale, hc_base, batch_size as i32)
        .map_err(|e| format!("hc_apply_alpha_batched l{layer_idx}: {e:?}"))?;

    // 3. Split c[B, 24] → contiguous pre[B, 4] / post[B, 4] / comb[B, 16]
    //    with sigmoid on pre, post_scale·sigmoid on post.
    gpu.hc_split_finalize_batched(
        &pbs.hc_c_batch,
        &pbs.hc_pre_batch,
        &pbs.hc_post_batch,
        &pbs.hc_comb_batch,
        post_scale,
        batch_size as i32,
    )
    .map_err(|e| format!("hc_split_finalize_batched l{layer_idx}: {e:?}"))?;

    // 4. Sinkhorn-normalize comb[B, 4, 4] in place per batch.
    gpu.hc_sinkhorn_4x4_batched(
        &pbs.hc_comb_batch,
        cfg.hc_eps,
        cfg.hc_sinkhorn_iters as i32,
        batch_size as i32,
    )
    .map_err(|e| format!("hc_sinkhorn_4x4_batched l{layer_idx}: {e:?}"))?;

    // 5. Input mapping: hc_x_in[b, d] = sum_h pre[b, h] · streams[b, h, d].
    gpu.hc_input_map_4stream_batched(
        &pbs.hc_pre_batch,
        &pbs.streams_batch,
        &pbs.hc_x_in_batch,
        cfg.hidden_size as i32,
        batch_size as i32,
    )
    .map_err(|e| format!("hc_input_map_4stream_batched l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// Batched twin of `apply_tail_rope` for Phase B2 chunk forward.
///
/// Per batch position b: applies DeepSeek V4's tail-only RoPE on the last
/// `qk_rope_head_dim` dims of each head in pbs.q_batch and pbs.kv_batch.
/// Reads positions[b] from `pbs.positions` (caller responsible for
/// pre-uploading `start_pos + b` per batch row at chunk start).
///
/// Per-layer YaRN parameters resolved via `layer_rope_params` exactly as
/// in the sequential path.
#[allow(dead_code)]
fn apply_tail_rope_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let (freq_base, freq_scale, ext_factor, attn_factor, corr_low, corr_high) =
        layer_rope_params(cfg, layer.compress_ratio);

    gpu.rope_tail_yarn_interleaved_batched(
        &pbs.q_batch,
        &pbs.kv_batch,
        &pbs.positions,
        cfg.num_attention_heads as i32,
        cfg.num_key_value_heads as i32,
        cfg.head_dim as i32,
        cfg.qk_rope_head_dim as i32,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        corr_low,
        corr_high,
        /*inverse=*/ 0,
        batch_size as i32,
    )
    .map_err(|e| format!("rope_tail_yarn_interleaved_batched l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// Batched twin of `kv_joint` for Phase B2 chunk forward.
///
/// Per batch position b:
///   kv[b] = wkv @ {tmp[b] or tmp_plain[b]}   (gemv_auto_batched)
///   kv[b] = RMSNorm(kv[b], kv_norm)          (in-place)
///
/// Reuses pbs.tmp_batch / pbs.tmp_plain_batch produced by q_lora_batched
/// in the same layer iteration. Writes pbs.kv_batch.
#[allow(dead_code)]
fn kv_joint_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let wkv = layer
        .wkv
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wkv missing"))?;
    let kv_norm = layer
        .kv_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} kv_norm missing"))?;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    // wkv @ tmp → kv.
    gemv_auto_batched_wmma(
        gpu,
        wkv,
        &pbs.tmp_batch,
        &pbs.tmp_plain_batch,
        &pbs.kv_batch,
        kv_dim,
        cfg.hidden_size,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // kv_norm RMSNorm in-place: batch x [kv_dim].
    gpu.rmsnorm_batched(
        &pbs.kv_batch,
        kv_norm,
        &pbs.kv_batch,
        batch_size,
        kv_dim,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("kv_norm rmsnorm_batched l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// Batched twin of `q_lora` for Phase B2 chunk forward.
///
/// Per batch position b:
///   tmp[b] = FWHT(RMSNorm(hc_x_in[b], attn_norm))
///   tmp_plain[b] = RMSNorm(hc_x_in[b], attn_norm)
///   q_lat[b] = wq_a @ {tmp[b] or tmp_plain[b]}  (gemv_auto_batched)
///   q_lat[b] = RMSNorm(q_lat[b], q_norm)        (in-place per row)
///   q_lat_rot[b] = FWHT(q_lat[b])
///   q[b] = wq_b @ {q_lat_rot[b] or q_lat[b]}    (gemv_auto_batched)
///   q[b, head] = RMSNorm(q[b, head], q_head_ones) for each head  (per-head)
///
/// All seven steps stay in lockstep across the B positions by riding the
/// existing `*_batched` kernels. The per-head Q normalisation at the end
/// flattens `[B, n_heads, head_dim]` into `B * n_heads` rows of head_dim
/// elements before calling `rmsnorm_batched`.
#[allow(dead_code, clippy::too_many_arguments)]
fn q_lora_batched(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    pbs: &PrefillBatchScratch,
    hc_x_in_batch: &GpuTensor, // [B, hidden]
    gpu: &mut Gpu,
    layer_idx: usize,
    batch_size: usize,
) -> Result<(), String> {
    let layer = weights.resolve_layer(layer_idx);
    let attn_norm = layer
        .attn_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} attn_norm missing"))?;
    let q_norm = layer
        .q_norm
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} q_norm missing"))?;
    let wq_a = layer
        .wq_a
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_a missing"))?;
    let wq_b = layer
        .wq_b
        .as_ref()
        .ok_or_else(|| format!("layer {layer_idx} wq_b missing"))?;

    let hidden = cfg.hidden_size;
    let q_rank = cfg.q_lora_rank;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim;

    // Skip FWHT rotations when weights don't need them. Also covers the
    // compressor (consumes tmp_batch as well, but only for MQ4 wkv/wgate
    // which are F16 in deepseek4-q8-mtp → no FWHT). Indexer compressor wkv/wgate
    // are F16 too. So we can skip both rotations when wq_a/wq_b are
    // Q8/F16 AND there's no MQ4 compressor (which there isn't on deepseek4-q8-mtp).
    let wq_a_needs_fwht = weight_needs_fwht(wq_a);
    let wq_b_needs_fwht = weight_needs_fwht(wq_b);

    // 1. RMSNorm (+ optional FWHT) batched. Fused variant writes BOTH
    //    rot and plain outputs in one launch when both are needed
    //    (common DeepSeek V4 case — compressor + indexer always read tmp_plain).
    if wq_a_needs_fwht {
        gpu.fused_rmsnorm_rotate_mq_plain_batched(
            hc_x_in_batch,
            attn_norm,
            &pbs.tmp_batch,
            &pbs.tmp_plain_batch,
            hidden,
            cfg.rms_norm_eps,
            batch_size,
        )
        .map_err(|e| format!("fused_rmsnorm_rotate_mq_plain_batched l{layer_idx}: {e:?}"))?;
    } else {
        // Plain only (wq_a Q8/F16/F32 doesn't need FWHT).
        gpu.rmsnorm_batched(
            hc_x_in_batch,
            attn_norm,
            &pbs.tmp_plain_batch,
            batch_size,
            hidden,
            cfg.rms_norm_eps,
        )
        .map_err(|e| format!("rmsnorm_batched attn-side plain l{layer_idx}: {e:?}"))?;
    }

    // 2. wq_a GEMV batched: tmp* → q_lat_batch. M = q_lora_rank, K = hidden.
    gemv_auto_batched_wmma(
        gpu,
        wq_a,
        &pbs.tmp_batch,
        &pbs.tmp_plain_batch,
        &pbs.q_lat_batch,
        q_rank,
        hidden,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // 3. q_norm RMSNorm batched (in-place): batch x [q_lora_rank].
    gpu.rmsnorm_batched(
        &pbs.q_lat_batch,
        q_norm,
        &pbs.q_lat_batch,
        batch_size,
        q_rank,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("q_norm rmsnorm_batched l{layer_idx}: {e:?}"))?;

    // 4. FWHT rotate q_lat → q_lat_rot for the MQ4 wq_b path — skip if not MQ4.
    if wq_b_needs_fwht {
        gpu.rotate_x_mq_batched(&pbs.q_lat_batch, &pbs.q_lat_rot_batch, q_rank, batch_size)
            .map_err(|e| format!("rotate_x_mq_batched q_lat l{layer_idx}: {e:?}"))?;
    }

    // 5. wq_b GEMV batched: q_lat_rot* → q_batch. M = n_heads*head_dim, K = q_lora_rank.
    let q_total = n_heads * head_dim;
    gemv_auto_batched_wmma(
        gpu,
        wq_b,
        &pbs.q_lat_rot_batch,
        &pbs.q_lat_batch,
        &pbs.q_batch,
        q_total,
        q_rank,
        batch_size,
        Some(&pbs.wmma_x_scratch_f16),
    )?;

    // 6. Per-(batch, head) RMSNorm of Q using q_head_ones as weight.
    //    [B, n_heads, head_dim] viewed as [B*n_heads, head_dim].
    gpu.rmsnorm_batched(
        &pbs.q_batch,
        &pbs.q_head_ones,
        &pbs.q_batch,
        batch_size * n_heads,
        head_dim,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("q per-head rmsnorm_batched l{layer_idx}: {e:?}"))?;

    Ok(())
}

/// Batched-prefill entry point for DeepSeek V4.
///
/// Processes the `tokens` slice starting at absolute KV position
/// `start_pos`. Returns the logits at the LAST position only (matches
/// the qwen35 forward_prefill_batch contract).
///
/// **Phase B status (2026-05-18):** scaffold. The body falls back to a
/// per-token `decode_step` loop — byte-identical to the existing
/// sequential prefill semantics. Phase B2 will replace the loop body
/// with a `forward_prefill_batch_chunk` call that processes `max_batch`
/// positions at once using the Phase A batched kernels (A1: SWA-topK,
/// A2: SWA, A3: indexer top-K, A5: HC mix).
///
/// The entry-point shape is finalised now so callers (eval harnesses,
/// daemon, eventual prefill API) can wire against the stable signature
/// while the inner batched body grows behind it. Production callers
/// should use `forward_prefill_batch_chunked` for actual batched prefill.
pub fn forward_prefill_batch(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    _scratch: &mut PrefillBatchScratch,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("forward_prefill_batch: empty tokens slice".to_string());
    }
    // Per-token fallback until forward_prefill_batch_chunk is end-to-end.
    let mut last_logits = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        last_logits = decode_step(cfg, weights, state, gpu, tok, start_pos + i as u32)?;
    }
    Ok(last_logits)
}

/// Single-chunk batched forward pass — Phase B2 work in progress.
///
/// Processes a chunk of `tokens.len()` ≤ `pbs.max_batch` positions
/// starting at `start_pos` through one batched forward. Mirrors
/// `decode_step` but with each per-layer stage swapped for its batched
/// twin. Returns the logits at the LAST position only.
///
/// Currently a partial wiring — runs through the stages that have
/// shipped batched bodies (embedding, HC stream init, q_lora,
/// kv_joint, tail RoPE) then errors out at the first unbatched stage
/// (the indexer + mixed attention dispatch). Each subsequent commit
/// replaces one error path with a real batched body until the chunk
/// runs end-to-end.
///
/// **Stages and their status (2026-05-18):**
///   ✓ token-ids upload → pbs.tokens
///   ✓ positions upload → pbs.positions
///   ✓ batched embedding lookup → pbs.embed_batch
///   ✓ HC streams broadcast init → pbs.streams_batch
///   ✓ per-layer q_lora_batched (Phase B2)
///   ✓ per-layer kv_joint_batched (Phase B2)
///   ✓ per-layer apply_tail_rope_batched (Phase B2)
///   ☐ per-layer mhc_pre_batched
///   ☐ per-layer compressor (loop sequential per A4 deferral)
///   ☐ per-layer indexer_forward_batched
///   ☐ per-layer mixed attention (wire deepseek4_attn_swa_topk_batched)
///   ☐ per-layer wo projection (gemv_auto_batched, two-stage O-LoRA)
///   ☐ per-layer hc_attn_mix_batched
///   ☐ per-layer ffn_routed_batched + hc_ffn_mix_batched
///   ☐ final_norm + lm_head (last position only)
///
/// Until all stages are wired this function returns an error from the
/// first unimplemented stage; callers should keep dispatching through
/// `forward_prefill_batch`'s per-token fallback for now.
#[allow(dead_code)]
pub fn forward_prefill_batch_chunk(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    tokens: &[u32],
    start_pos: u32,
) -> Result<(), String> {
    let n = tokens.len();
    if n == 0 {
        return Err("forward_prefill_batch_chunk: empty tokens".to_string());
    }
    if n > pbs.max_batch {
        return Err(format!(
            "forward_prefill_batch_chunk: chunk size {n} > max_batch {}",
            pbs.max_batch
        ));
    }

    // Phase C: ensure we have an active stream so all the small h2d
    // uploads in this chunk forward go async-on-stream via
    // `memcpy_htod_auto`. Subsequent kernels submitted to the same
    // stream order naturally — no host blocking on each tiny upload.
    if gpu.active_stream.is_none() {
        let new_stream = gpu
            .hip
            .stream_create()
            .map_err(|e| format!("stream_create for async htod: {e:?}"))?;
        gpu.active_stream = Some(new_stream);
    }

    // 1. Upload token ids and absolute positions for this chunk.
    let token_ids_host: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(token_ids_host.as_ptr() as *const u8, n * 4) };
    gpu.memcpy_htod_auto(&pbs.tokens.buf, token_bytes)
        .map_err(|e| format!("htod tokens: {e:?}"))?;

    let positions_host: Vec<i32> = (0..n).map(|i| (start_pos as i32) + i as i32).collect();
    let positions_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(positions_host.as_ptr() as *const u8, n * 4) };
    gpu.memcpy_htod_auto(&pbs.positions.buf, positions_bytes)
        .map_err(|e| format!("htod positions: {e:?}"))?;

    // 1.5. Hoist `n_valid_swa_arr` upload to once-per-chunk. Both
    // `attention_block_batched_swa_only` and `attention_block_batched_mixed`
    // used to upload this identical buffer per-layer (43× per chunk on DeepSeek V4),
    // each upload synchronising the active stream. The value depends only
    // on (start_pos, batch_size, sliding_window) — chunk-invariant across
    // all layers. The per-layer uploads are now skipped (the device buffer
    // is already populated when `attention_block_*` runs).
    let win = cfg.sliding_window;
    let n_valid_host: Vec<i32> = (0..n)
        .map(|b| ((start_pos as usize + b + 1).min(win)) as i32)
        .collect();
    let n_valid_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(n_valid_host.as_ptr() as *const u8, n * 4) };
    gpu.memcpy_htod_auto(&pbs.n_valid_swa_arr.buf, n_valid_bytes)
        .map_err(|e| format!("htod n_valid_swa_arr (chunk-level): {e:?}"))?;

    // 2. Batched embedding lookup → pbs.embed_batch [n, hidden].
    let token_embd = weights
        .token_embd
        .as_ref()
        .ok_or_else(|| "forward_prefill_batch_chunk: token_embd not uploaded".to_string())?;
    gpu.embedding_lookup_q8_batched(
        token_embd,
        &pbs.embed_batch,
        &pbs.tokens,
        n,
        cfg.hidden_size,
    )
    .map_err(|e| format!("embedding_lookup_q8_batched: {e:?}"))?;
    dump_buf(gpu, "01_embed", &pbs.embed_batch);

    // 3. Broadcast embed → all 4 HC residual streams [n, hc_mult, hidden].
    gpu.hc_streams_init_from_embed_batched(
        &pbs.embed_batch,
        &pbs.streams_batch,
        cfg.hidden_size as i32,
        cfg.hc_mult as i32,
        n as i32,
    )
    .map_err(|e| format!("hc_streams_init_from_embed_batched: {e:?}"))?;
    dump_buf(gpu, "02_hc_streams_init", &pbs.streams_batch);

    // 4. Per-layer loop. Stages that DO run:
    //   ✓ mhc_pre_batched(is_attn=true)  → pbs.{hc_pre,hc_post,hc_comb,hc_x_in}_batch
    //   ✓ q_lora_batched   (consumes hc_x_in_batch) → pbs.q_batch
    //   ✓ kv_joint_batched (consumes tmp/tmp_plain) → pbs.kv_batch
    //   ✓ apply_tail_rope_batched         (in-place on q_batch & kv_batch)
    //
    // Then we hit the attention stage which still needs per-batch SWA
    // staging + indexer top-K gather + wo_a/wo_b O-LoRA projection.
    for layer_idx in 0..cfg.num_hidden_layers {
        // Attention-side HC pre + per-stream input mapping.
        mhc_pre_batched(cfg, weights, pbs, gpu, layer_idx, /*is_attn=*/ true, n)?;
        if layer_idx == 0 {
            dump_buf(gpu, "03_l0_mhc_pre_attn_hc_x_in", &pbs.hc_x_in_batch);
        }

        // Q-LoRA: pbs.hc_x_in_batch → tmp/tmp_plain → q_lat → q_batch.
        q_lora_batched(cfg, weights, pbs, &pbs.hc_x_in_batch, gpu, layer_idx, n)?;
        if layer_idx == 0 {
            dump_buf(gpu, "04_l0_q_lora_q_batch", &pbs.q_batch);
        }

        // Joint KV projection: tmp/tmp_plain → kv_batch.
        kv_joint_batched(cfg, weights, pbs, gpu, layer_idx, n)?;
        if layer_idx == 0 {
            dump_buf(gpu, "05_l0_kv_joint_kv_batch", &pbs.kv_batch);
        }

        // Tail-only RoPE on q_batch and kv_batch in-place.
        apply_tail_rope_batched(cfg, weights, pbs, gpu, layer_idx, n)?;
        if layer_idx == 0 {
            dump_buf(gpu, "06_l0_tail_rope_q_batch", &pbs.q_batch);
            dump_buf(gpu, "06_l0_tail_rope_kv_batch", &pbs.kv_batch);
        }

        // ── Attention block: pure-SWA for compress_ratio==0, mixed
        //    (SWA + indexer/identity topk) for compress_ratio>0.
        let layer = weights.resolve_layer(layer_idx);
        if layer.compress_ratio == 0 {
            attention_block_batched_swa_only(
                cfg, weights, state, pbs, gpu, layer_idx, start_pos, n,
            )?;
        } else {
            attention_block_batched_mixed(cfg, weights, state, pbs, gpu, layer_idx, start_pos, n)?;
        }
        if layer_idx == 0 {
            dump_buf(gpu, "07_l0_attn_out_batch", &pbs.attn_out_batch);
        }

        // hc_attn_mix: integrate attn_out_batch into streams_batch.
        hc_attn_mix_batched(cfg, pbs, gpu, n)?;
        if layer_idx == 0 {
            dump_buf(gpu, "08_l0_hc_attn_mix_streams", &pbs.streams_batch);
        }

        // FFN side: mhc_pre(is_attn=false) → ffn_batched (shared + routed)
        // → hc_ffn_mix_batched.
        mhc_pre_batched(
            cfg, weights, pbs, gpu, layer_idx, /*is_attn=*/ false, n,
        )?;
        if layer_idx == 0 {
            dump_buf(gpu, "09_l0_mhc_pre_ffn_hc_x_in", &pbs.hc_x_in_batch);
        }
        ffn_batched(cfg, weights, pbs, gpu, layer_idx, n, tokens)?;
        if layer_idx == 0 {
            dump_buf(gpu, "10_l0_ffn_out", &pbs.ffn_out_batch);
        }
        hc_ffn_mix_batched(cfg, pbs, gpu, n)?;
        if layer_idx <= 3 {
            dump_buf(
                gpu,
                &format!("11_l{layer_idx}_end_streams"),
                &pbs.streams_batch,
            );
        }
    }

    Ok(())
}

/// Top-level batched-prefill driver — chunks the prompt by max_batch
/// and dispatches each chunk through `forward_prefill_batch_chunk`.
///
/// Returns logits at the LAST position only (matches the qwen35
/// contract). Falls back to per-token decode_step if any chunk fails
/// (typically because a layer's compress_ratio path isn't yet wired —
/// pure-SWA-only for now, mixed-attention layers error out).
///
/// **Phase B2 status (2026-05-18):** the chunk-forward path handles
/// pure-SWA layers (compress_ratio == 0) end-to-end including the
/// MoE FFN; mixed-attention layers (compress_ratio > 0) still bail
/// at the indexer chain. Until mixed is wired, this function falls
/// back to per-token decode_step for any chunk that contains a
/// mixed-attention layer (i.e. all DeepSeek V4 prompts except the trivial
/// case where all 43 layers are dense, which doesn't exist).
#[allow(dead_code)]
pub fn forward_prefill_batch_chunked(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    pbs: &PrefillBatchScratch,
) -> Result<Vec<f32>, String> {
    if tokens.is_empty() {
        return Err("forward_prefill_batch_chunked: empty tokens".to_string());
    }

    // Strict batched-only path. Any chunk failure surfaces immediately —
    // we do NOT silently fall back to per-token decode_step. The original
    // fallback masked a real correctness bug in chunk 2+ (per-batch state
    // not initialised; see Option B fix, memory entry
    // `feedback_deepseek4_chunked_silent_fallback_bug`). Keeping the fallback
    // hides any future regression in the same place.
    let mut pos_cursor = start_pos as usize;
    let mut remaining = tokens;
    while !remaining.is_empty() {
        let take = remaining.len().min(pbs.max_batch);
        let chunk = &remaining[..take];
        forward_prefill_batch_chunk(cfg, weights, state, gpu, pbs, chunk, pos_cursor as u32)?;
        if take == remaining.len() {
            return final_norm_and_head_last_batched(cfg, weights, state, pbs, gpu, take);
        }
        pos_cursor += take;
        remaining = &remaining[take..];
    }
    Err("forward_prefill_batch_chunked: chunk loop completed without producing logits".to_string())
}

/// Manual-chunk prefill with per-position MTP fill interleaved.
///
/// Mirrors the deepseek4_mtp_smoke "batched main + per-position MTP" path.
/// Used by the spec-decode entry points (deepseek4_chat / daemon) so the MTP
/// layer's SWA cache is populated during prefill — without this the
/// first spec-decode draft step sees an empty MTP attention history
/// and accept rate collapses.
///
/// Returns logits at the LAST position (the prediction for the first
/// generated token). Side-effect: leaves `state.mtp_last_hidden`
/// populated and `state.n_tokens` advanced to `start_pos + prompt.len()`.
///
/// Temporarily sets `HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD=1` around the MTP pass
/// so `mtp_forward_batched` short-circuits the lm_head + logits
/// download — that's per-MTP-position waste during prefill fill (we
/// only need the MTP attention SWA state to be primed, not the
/// per-position MTP logits). Restored after each chunk.
pub fn prefill_with_mtp_fill(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &PrefillBatchScratch,
    prompt_tokens: &[u32],
    start_pos: u32,
) -> Result<Vec<f32>, String> {
    let stream_len = cfg.hc_mult * cfg.hidden_size;
    if state.mtp_last_hidden.is_none() {
        state.mtp_last_hidden = Some(
            gpu.alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], rdna_compute::DType::F32)
                .map_err(|e| format!("alloc mtp_last_hidden (spec prefill): {e:?}"))?,
        );
    }
    // compressor_forward_prebatched reads pos_array_device via pos_slot()
    // for any compressed layer. Init to start_pos.
    precompute_positions(cfg, state, gpu, start_pos)?;

    let mut last_logits: Vec<f32> = vec![];
    let mut pos_cursor: usize = 0;
    while pos_cursor < prompt_tokens.len() {
        let chunk_size = (prompt_tokens.len() - pos_cursor).min(pbs.max_batch);
        let chunk = &prompt_tokens[pos_cursor..pos_cursor + chunk_size];
        let abs_chunk_start = start_pos as usize + pos_cursor;
        let is_last_chunk = pos_cursor + chunk_size == prompt_tokens.len();

        // 1. Batched main forward over this chunk's positions. After this,
        //    pbs.streams_batch holds [chunk_size, hc_mult, hidden] residuals
        //    — these are the per-position h_n inputs the MTP layer needs.
        forward_prefill_batch_chunk(cfg, weights, state, gpu, pbs, chunk, abs_chunk_start as u32)?;

        // 2. Capture the last position's stream for the head on the last
        //    chunk, BEFORE mtp_forward_batched overwrites streams_batch.
        let last_stream_pre_mtp: Option<rdna_compute::GpuTensor> = if is_last_chunk {
            let off = (chunk_size - 1) * stream_len;
            let src = pbs.streams_batch.sub_offset(off, stream_len);
            let mut snap = gpu
                .alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], rdna_compute::DType::F32)
                .map_err(|e| format!("alloc head_input_snap: {e:?}"))?;
            gpu.memcpy_dtod_auto(&snap.buf, &src.buf, stream_len * 4)
                .map_err(|e| format!("d2d streams[last] → head_input_snap: {e:?}"))?;
            snap.shape = vec![cfg.hc_mult, cfg.hidden_size];
            Some(snap)
        } else {
            None
        };

        // 3. Batched MTP fill — single pass through the MTP layer for all
        //    mtp_end_b positions in this chunk. Skip the global last
        //    position (next-token unknown, that's what we're about to
        //    generate).
        std::env::set_var("HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD", "1");
        let mtp_end_b = if is_last_chunk {
            chunk_size.saturating_sub(1)
        } else {
            chunk_size
        };
        if mtp_end_b > 0 {
            let h_n_streams = pbs.streams_batch.sub_offset(0, mtp_end_b * stream_len);
            let next_tokens: Vec<u32> = (0..mtp_end_b)
                .map(|b| prompt_tokens[pos_cursor + b + 1])
                .collect();
            mtp_forward_batched(
                cfg,
                weights,
                state,
                gpu,
                pbs,
                &h_n_streams,
                &next_tokens,
                abs_chunk_start as u32,
                mtp_end_b,
            )?;
        }
        std::env::remove_var("HIPFIRE_DEEPSEEK4_MTP_SKIP_HEAD");

        pos_cursor += chunk_size;
        state.n_tokens = (abs_chunk_start + chunk_size) as u64;

        // 4. Last chunk: run final_norm_and_head from the snapshot we
        //    captured pre-MTP (streams_batch is now MTP outputs). Write
        //    the snapshot back into the last slot so the existing
        //    final_norm_and_head_last_batched can read it.
        if is_last_chunk {
            if let Some(snap) = last_stream_pre_mtp {
                let off = (chunk_size - 1) * stream_len;
                let dst = pbs.streams_batch.sub_offset(off, stream_len);
                gpu.memcpy_dtod_auto(&dst.buf, &snap.buf, stream_len * 4)
                    .map_err(|e| format!("d2d restore streams[last] for head: {e:?}"))?;
            }
            last_logits =
                final_norm_and_head_last_batched(cfg, weights, state, pbs, gpu, chunk_size)?;
        }
    }
    Ok(last_logits)
}

/// CPU reference implementation of bias-aware top-k: picks the `k` highest
/// `scores[i] + bias[i]` entries, then weights them by their UNBIASED
/// scores (per DeepSeek V4 router semantics — bias only steers selection).
/// Production routing goes through the GPU kernel
/// `deepseek4_moe_topk_bias_aware_f32`; this is kept as a tested reference.
#[cfg(test)]
fn bias_aware_topk_weights(
    scores: &[f32],
    bias: &[f32],
    k: usize,
) -> Option<(Vec<u32>, Vec<f32>)> {
    let n = scores.len();
    if k == 0 || n == 0 {
        return None;
    }
    let mut biased: Vec<f32> = (0..n)
        .map(|i| scores[i] + bias.get(i).copied().unwrap_or(0.0))
        .collect();
    let k = k.min(n);
    let mut indices: Vec<u32> = Vec::with_capacity(k);
    for _ in 0..k {
        let (best_i, _) = biased
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        indices.push(best_i as u32);
        biased[best_i] = f32::NEG_INFINITY;
    }
    let mut wts: Vec<f32> = indices.iter().map(|&i| scores[i as usize]).collect();
    let w_sum: f32 = wts.iter().sum();
    if w_sum <= 0.0 {
        return None;
    }
    for w in wts.iter_mut() {
        *w /= w_sum;
    }
    Some((indices, wts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_aware_topk_picks_biased_indices() {
        // Bias steers selection. scores=[1,1,1,1,1,1], bias=[0,0,0,3,2,0]
        // → biased=[1,1,1,4,3,1] → top-2 = [3, 4].
        let scores = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 3.0, 2.0, 0.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert_eq!(idx, vec![3, 4]);
        // Weights come from UNBIASED scores (both 1.0), normalized.
        assert!((wts[0] - 0.5).abs() < 1e-6);
        assert!((wts[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_weights_use_unbiased_scores() {
        // scores=[5, 1, 1], bias=[0, 10, 10] → biased=[5, 11, 11].
        // Top-2 by biased = [1, 2]. Weights from unbiased = [1, 1] → [0.5, 0.5].
        let scores = vec![5.0, 1.0, 1.0];
        let bias = vec![0.0, 10.0, 10.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert!(idx == vec![1, 2] || idx == vec![2, 1]);
        assert!((wts[0] - 0.5).abs() < 1e-6);
        assert!((wts[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_falls_back_zero_bias() {
        // No bias → pure top-K from scores.
        let scores = vec![0.1, 0.9, 0.5, 0.7];
        let bias: Vec<f32> = vec![];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 2).unwrap();
        assert_eq!(idx, vec![1, 3]);
        let s = 0.9 + 0.7;
        assert!((wts[0] - 0.9 / s).abs() < 1e-6);
        assert!((wts[1] - 0.7 / s).abs() < 1e-6);
    }

    #[test]
    fn bias_aware_topk_returns_none_on_zero_sum() {
        // All scores zero → no positive weight sum.
        let scores = vec![0.0, 0.0, 0.0];
        let bias = vec![5.0, 0.0, 0.0]; // bias picks idx 0 but score is 0
        assert!(bias_aware_topk_weights(&scores, &bias, 1).is_none());
    }

    #[test]
    fn bias_aware_topk_handles_k_geq_n() {
        // k=4 but only n=2 scores — caller's job to set k correctly,
        // but we silently clamp rather than panic.
        let scores = vec![1.0, 2.0];
        let bias = vec![0.0, 0.0];
        let (idx, wts) = bias_aware_topk_weights(&scores, &bias, 4).unwrap();
        assert_eq!(idx.len(), 2);
        assert!(wts.iter().sum::<f32>() > 0.99 && wts.iter().sum::<f32>() < 1.01);
    }

    #[test]
    fn gather_normalized_weights_basic() {
        let scores = vec![0.0, 2.0, 0.0, 1.0, 0.0];
        let idx = vec![1u32, 3];
        let wts = gather_normalized_weights(&scores, &idx).unwrap();
        // scores at idx = [2, 1] → normalized [2/3, 1/3]
        assert!((wts[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((wts[1] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn gather_normalized_weights_zero_sum_returns_none() {
        let scores = vec![0.0; 8];
        let idx = vec![0u32, 1, 2];
        assert!(gather_normalized_weights(&scores, &idx).is_none());
    }

    #[test]
    fn gather_normalized_weights_out_of_range_idx_is_zero() {
        // Hash table can in theory point past scores; we treat OOR as 0
        // (better than panicking — tid2eid is supposed to be in range).
        let scores = vec![1.0, 2.0, 3.0];
        let idx = vec![1u32, 999];
        let wts = gather_normalized_weights(&scores, &idx).unwrap();
        // sum = 2 + 0 = 2 → normalized [1.0, 0.0]
        assert!((wts[0] - 1.0).abs() < 1e-6);
        assert!((wts[1] - 0.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod ship6_lower_tests {
    use super::*;
    use superop::SuperOpKind::{Attend, Moe};

    // #397 Ship 6 — deepseek4 is one variant (every layer Attn+MoE; per-layer
    // conditionals live inside the handlers).
    #[test]
    fn ds4_program_is_attend_then_moe() {
        let kinds: Vec<_> = ds4_lower_program().iter().map(|o| o.kind).collect();
        assert_eq!(kinds, vec![Attend, Moe]);
    }
}
