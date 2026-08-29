// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kate, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 model: hybrid sliding-window + full attention, dense FFN (SwiGLU + gelu_pytorch_tanh).
//!
//! Architectural features vs. Qwen3.5:
//!   • Sliding-window attention on 5 of every 6 layers (window=1024).
//!   • Full attention layers use head_dim=512 (global_head_dim) with
//!     attention_k_eq_v: V is the pre-k_norm output of k_proj (no v_proj).
//!   • Partial proportional RoPE on full layers (first 64 of 512 dims rotate,
//!     rope_theta=1e6; sliding uses default RoPE with theta=10000).
//!   • Sandwich RMSNorm: input + post-attn + pre-FFN + post-FFN per layer,
//!     plus a learned per-layer `layer_scalar [1]` at layer end.
//!   • Attention scale = 1.0 (not 1/√d); Q/K norms absorb scaling.
//!   • Final logit softcap: `tanh(logits/30) * 30` before sampling.
//!   • MLP: SwiGLU with `gelu_pytorch_tanh` activation.
//!   • Tied LM head (embed_tokens.weight aliased).
//!   • Embed scale: sqrt(hidden_size) multiplied onto every embedding row lookup.

use hip_bridge::HipResult;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::gemm::GemmParams;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::families::moe::{
    launch_moe_gelu_experts, MoeGeluExpertsRef, MoeRouterBackend,
};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_runtime::gpu_cleanup::{GpuCleanupFailure, RetainedGpuTensor, RetryableOwner};
use hipfire_runtime::hfq::{load_awq_scale, HfqFile};
use hipfire_runtime::llama::{self, f16_to_f32, weight_gemv, EmbeddingFormat, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

#[cfg(feature = "lowered-fault-inject")]
static LIVE_OWNER_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Live bytes registered by lowered construction owners.
///
/// This counter is compiled as a no-op outside the fault-injection feature so
/// production loads do not pay for test accounting. A nonzero value after a
/// failed construction means a real owner escaped its transaction.
pub fn live_owner_bytes() -> usize {
    #[cfg(feature = "lowered-fault-inject")]
    {
        return LIVE_OWNER_BYTES.load(std::sync::atomic::Ordering::SeqCst);
    }
    #[cfg(not(feature = "lowered-fault-inject"))]
    0
}

#[inline]
pub fn register_live_owner_bytes(bytes: usize) {
    #[cfg(feature = "lowered-fault-inject")]
    LIVE_OWNER_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::SeqCst);
    #[cfg(not(feature = "lowered-fault-inject"))]
    let _ = bytes;
}

#[inline]
pub fn unregister_live_owner_bytes(bytes: usize) {
    #[cfg(feature = "lowered-fault-inject")]
    {
        let previous = LIVE_OWNER_BYTES.fetch_sub(bytes, std::sync::atomic::Ordering::SeqCst);
        debug_assert!(
            previous >= bytes,
            "lowered owner accounting underflow: previous={previous}, bytes={bytes}"
        );
    }
    #[cfg(not(feature = "lowered-fault-inject"))]
    let _ = bytes;
}

/// #397 Ship 5.2: route a single PLAIN-batched prefill GEMM through
/// [`GemmFamily::run_key`] against an explicit dispatcher-entry key.
///
/// Byte-identical to the old `weight_gemm()` call for every (dtype × arch).
/// Only HFQ4G256, HFQ4G128 have batched kernels; other dtypes fall back to
/// the repeated-GEMV loop (same as `weight_gemm`'s fallback).
#[inline]
/// Env gate for WMMA prefill. Default OFF because WMMA F16 input quantization
/// loses ~3 mantissa bits vs F32 scalar, so WMMA results are NOT byte-identical
/// to the scalar GEMV path. Set HIPFIRE_WMMA_PREFILL=1 to opt in.
pub fn wmma_prefill_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("HIPFIRE_WMMA_PREFILL").map_or(false, |v| v == "1"))
}

/// Env gate for batched prefill (v2). Independent from WMMA.
/// Set HIPFIRE_BATCHED_PREFILL=1 to use batched projections + per-token attention.
pub fn batched_prefill_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("HIPFIRE_BATCHED_PREFILL").map_or(false, |v| v == "1"))
}

/// Result of rolling back a working Gemma request.
///
/// `RestoredCommitted` is valid only for append-only work that did not damage
/// any row belonging to the committed sliding window. `Invalidated` means the
/// caller must perform the total cold reset before the next request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmaRollback {
    RestoredCommitted,
    Invalidated,
}
/// Allocation boundaries in the lowered bundle constructor.
///
/// Keep this ordered list in construction order. The carrier stages each
/// completed owner at one of these boundaries and rolls them back in reverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gemma4ConstructionStage {
    Weights,
    Scratch,
    SlidingKv,
    FullKv,
    Session,
}

/// Return the deterministic lowered-construction fault matrix in construction
/// order. This is also the contract used by focused ownership tests.
pub const fn construction_fault_stages() -> &'static [Gemma4ConstructionStage] {
    &[
        Gemma4ConstructionStage::Weights,
        Gemma4ConstructionStage::Scratch,
        Gemma4ConstructionStage::SlidingKv,
        Gemma4ConstructionStage::FullKv,
        Gemma4ConstructionStage::Session,
    ]
}

impl Gemma4ConstructionStage {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::Scratch => "scratch",
            Self::SlidingKv => "sliding_kv",
            Self::FullKv => "full_kv",
            Self::Session => "session",
        }
    }
}

/// Test-only deterministic failure after a completed construction stage.
///
/// The hook is compiled out of production builds. Keeping the stage check at
/// the carrier boundary means a fault exercises exactly the same rollback path
/// as a real later-stage error.
pub fn fail_after_construction_stage(stage: Gemma4ConstructionStage) -> HipResult<()> {
    #[cfg(feature = "lowered-fault-inject")]
    if std::env::var("HIPFIRE_GEMMA4_FAIL_STAGE").ok().as_deref() == Some(stage.label()) {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("injected Gemma 4 lowered failure after {}", stage.label()),
        ));
    }
    #[cfg(not(feature = "lowered-fault-inject"))]
    let _ = stage;
    Ok(())
}
/// One developer-only lifecycle allocation observation.
///
/// The values are deliberately actual owner/pool capacities rather than
/// logical tensor estimates. No observation is emitted unless
/// `HIPFIRE_GEMMA4_ALLOC_TELEMETRY=1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gemma4AllocationTelemetry {
    pub phase: &'static str,
    pub cycle: u64,
    pub owner_bytes: usize,
    pub live_owner_bytes: usize,
    pub pool_bytes: usize,
    pub free_device_bytes: Option<usize>,
    pub graph_resident: bool,
    pub graph_blob_count: usize,
    pub module_count: usize,
    pub freed_owner_labels: Vec<String>,
}

fn allocation_telemetry_enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Return whether developer-only Gemma lifecycle telemetry is enabled.
pub fn allocation_telemetry_enabled() -> bool {
    allocation_telemetry_enabled_value(
        std::env::var("HIPFIRE_GEMMA4_ALLOC_TELEMETRY")
            .ok()
            .as_deref(),
    )
}

/// Optional operator-provided lifecycle cycle identifier.
pub fn allocation_telemetry_cycle() -> u64 {
    std::env::var("HIPFIRE_GEMMA4_ALLOC_CYCLE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

impl Gemma4AllocationTelemetry {
    pub fn from_gpu(
        phase: &'static str,
        cycle: u64,
        owner_bytes: usize,
        gpu: &Gpu,
        freed_owner_labels: Vec<String>,
    ) -> Self {
        Self {
            phase,
            cycle,
            owner_bytes,
            live_owner_bytes: live_owner_bytes(),
            pool_bytes: gpu.pool_cached_bytes(),
            free_device_bytes: gpu.hip.get_vram_info().ok().map(|(free, _)| free),
            graph_resident: gpu.graphs.graph_exec.is_some()
                || gpu.graphs.captured_graph.is_some()
                || !gpu.graphs.verify.cache.is_empty()
                || !gpu.graphs.replay.cache.is_empty(),
            graph_blob_count: gpu.graph_blob_count(),
            module_count: gpu.loaded_module_count(),
            freed_owner_labels,
        }
    }

    pub fn format_line(&self) -> String {
        let freed = if self.freed_owner_labels.is_empty() {
            "-".to_string()
        } else {
            self.freed_owner_labels.join(",")
        };
        format!(
            "[gemma4 alloc] phase={} cycle={} owner_bytes={} live_owner_bytes={} pool_bytes={} free_device_bytes={} graph_resident={} graph_blob_count={} module_count={} freed_owner_labels={}",
            self.phase,
            self.cycle,
            self.owner_bytes,
            self.live_owner_bytes,
            self.pool_bytes,
            self.free_device_bytes
                .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
            self.graph_resident,
            self.graph_blob_count,
            self.module_count,
            freed,
        )
    }

    pub fn emit_from_gpu(
        phase: &'static str,
        cycle: u64,
        owner_bytes: usize,
        gpu: &Gpu,
        freed_owner_labels: Vec<String>,
    ) {
        if allocation_telemetry_enabled() {
            Self::from_gpu(phase, cycle, owner_bytes, gpu, freed_owner_labels).emit();
        }
    }

    pub fn emit(&self) {
        if allocation_telemetry_enabled() {
            eprintln!("{}", self.format_line());
        }
    }
}

/// Authoritative runtime cursor for the lowered Gemma bundle.
///
/// Both the sliding and full KV families advance against this one logical
/// position. The host prefix cache owns committed/request transaction metadata;
/// this type owns only the live runtime cursor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gemma4Cursor {
    position: usize,
}

/// Apply the exact-prefix rollback policy to an architecture-owned cursor.
///
/// This free helper lets the eager `Gemma4State` adapter and the lowered
/// cursor share one overwrite/identity decision without sharing host history.
pub fn rollback_gemma_cursor(
    position: &mut usize,
    committed_cursor: usize,
    identity_matches: bool,
    overwrite_boundary: Option<usize>,
) -> GemmaRollback {
    let overwrote_committed =
        overwrite_boundary.is_some_and(|boundary| boundary < committed_cursor);
    if identity_matches && *position >= committed_cursor && !overwrote_committed {
        *position = committed_cursor;
        GemmaRollback::RestoredCommitted
    } else {
        *position = 0;
        GemmaRollback::Invalidated
    }
}

impl Gemma4Cursor {
    #[inline]
    pub const fn new(position: usize) -> Self {
        Self { position }
    }

    #[inline]
    pub const fn position(self) -> usize {
        self.position
    }

    /// Current number of tokens materialized in both lowered KV families.
    #[inline]
    pub const fn materialized_cursor(self) -> usize {
        self.position
    }

    /// Set the shared lowered runtime cursor after a committed forward.
    #[inline]
    pub fn set_materialized_cursor(&mut self, cursor: usize) {
        self.position = cursor;
    }

    #[inline]
    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    #[inline]
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Roll back the live cursor after a working request.
    ///
    /// `overwrite_boundary` is the earliest absolute row overwritten by that
    /// request. A boundary at or after `committed_cursor` touches only working
    /// rows and is therefore safe; a boundary below it destroys committed
    /// sliding-window data. Identity mismatches and cursor regressions fail
    /// closed to the cold state.
    pub fn rollback_working_request(
        &mut self,
        committed_cursor: usize,
        identity_matches: bool,
        overwrite_boundary: Option<usize>,
    ) -> GemmaRollback {
        rollback_gemma_cursor(
            &mut self.position,
            committed_cursor,
            identity_matches,
            overwrite_boundary,
        )
    }
}

/// Reset the two lowered KV families' logical offset state together.
///
/// KV bytes are reusable after a cursor reset; the offsets are the metadata
/// that must not survive a cold session reset.
#[inline]
pub fn reset_kv_offsets(
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
) {
    kv_sliding.compact_offset = 0;
    kv_full.compact_offset = 0;
}

/// Batched GEMM for prefill projections.
///
/// Scalar path: routes through `GemmFamily::run_key` with the appropriate
/// kernel key for the weight dtype.
///
/// WMMA path (opt-in via `HIPFIRE_WMMA_PREFILL=1`): routes through
/// `GemmFamily::run` which resolves WMMA-eligible dtypes to their WMMA
/// kernel variant. The WMMA path converts F32 input to F16 automatically
/// (Bug 1 fix: `gemm_hfq4g256_wmma` now calls `ensure_fp16_x`).
///
/// MQ4G256 weights (Bug 2 fix): `GemmFamily::resolve` now maps
/// `DType::MQ4G256` → `GemmHfq4G256Wmma` / `GemmHfq4G256`, sharing the
/// same kernel binary since the layout is identical (136 B/group).
///
/// For dtypes that have no GEMM kernel, falls back to per-token GEMV.
fn run_prefill_gemm(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
    bf16_scratch: Option<&GpuTensor>,
) -> HipResult<()> {
    // MQ4G256 weights are FWHT-rotated at quantize time. The per-token GEMV
    // path (weight_gemv) FWHT-rotates x before running the prerotated kernel;
    // the batched GEMM kernels (scalar gemm_hfq4g256 AND its WMMA siblings)
    // decode the same 136 B/group layout but do NOT rotate internally, so the
    // input must be pre-rotated here -- exactly like the eager
    // forward.rs::proj_gemm_batched (EAGLE verify) does. Skipping this fed raw
    // activations to the GEMM and produced garbage logits from token 0 on
    // every arch (root-caused on gfx1201 2026-06-10; gfx11 batched-prefill
    // validation predated the *.union.mq4 files, which is why it never fired
    // there).
    let x_rot_holder = if matches!(w.gpu_dtype, DType::MQ4G256) {
        let xr = gpu.alloc_tensor(&[batch_size, w.k], DType::F32)?;
        llama::rotate_x_mq_batched_for(gpu, w, x, &xr, w.k, batch_size)?;
        Some(xr)
    } else {
        None
    };
    let x_gemm: &GpuTensor = x_rot_holder.as_ref().unwrap_or(x);

    let result = run_prefill_gemm_inner(gpu, w, x, x_gemm, y, batch_size, bf16_scratch);
    if let Some(t) = x_rot_holder {
        gpu.free_tensor(t)?;
    }
    result
}

/// Inner body of `run_prefill_gemm`: `x_gemm` is the (possibly FWHT-rotated)
/// GEMM input; `x_raw` is the original activation, used only by the per-token
/// GEMV fallback + verify hook (both rotate internally via ).
fn run_prefill_gemm_inner(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_raw: &GpuTensor,
    x_gemm: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
    bf16_scratch: Option<&GpuTensor>,
) -> HipResult<()> {
    // BF16 MFMA path — calibration override, DEFAULT path (no WMMA gate).
    // Stages F32 activation to BF16 via persistent scratch, then dispatches
    // GemmBf16Mfma. The wrapper refuses F32 x, so this must be BF16.
    if w.gpu_dtype == DType::BF16 {
        let nelems = batch_size * w.k;
        if x_gemm.dtype == DType::BF16 {
            // Hoisted: caller already staged to BF16 (q/k/v or gate+up reuse).
            // F32 source was captured at the hoist site before staging, so the
            // run_key tap (which skips BF16 x) is bypassed correctly there.
            let w_ref = WeightRef {
                buf: &w.buf,
                dtype: w.gpu_dtype,
                m: w.m,
                k: w.k,
                row_stride: w.k,
                rotation: None,
                awq_scale: None,
            };
            let ctx = DispatchCtx::new(gpu);
            let params = GemmParams {
                w: &w_ref,
                x: x_gemm,
                y,
                batch_size,
            };
            let family = llama::gemm_family();
            family
                .run_key(
                    hipfire_dispatch::types::KernelKey::GemmBf16Mfma,
                    &ctx,
                    gpu,
                    &params,
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            return Ok(());
        }
        // Non-hoisted BF16 path: capture the true F32 activation before
        // staging to BF16. The GemmFamily::run_key tap deliberately skips
        // BF16 x (would overrun and record rounded values), so this explicit
        // F32 capture restores coverage. Zero-cost when unarmed.
        gpu.maybe_capture_activation(&w.buf, x_gemm, batch_size, w.k);
        if let Some(scratch) = bf16_scratch {
            let bf16_view = scratch.sub_offset(0, nelems);
            gpu.convert_f32_to_bf16(x_gemm, &bf16_view, nelems)?;
            let w_ref = WeightRef {
                buf: &w.buf,
                dtype: w.gpu_dtype,
                m: w.m,
                k: w.k,
                row_stride: w.k,
                rotation: None,
                awq_scale: None,
            };
            let ctx = DispatchCtx::new(gpu);
            let params = GemmParams {
                w: &w_ref,
                x: &bf16_view,
                y,
                batch_size,
            };
            let family = llama::gemm_family();
            family
                .run_key(
                    hipfire_dispatch::types::KernelKey::GemmBf16Mfma,
                    &ctx,
                    gpu,
                    &params,
                )
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
            return Ok(());
        }
        // No persistent scratch — fallback per-call alloc (not hoisted, but functional).
        let tmp = gpu.alloc_tensor(&[nelems], DType::BF16)?;
        gpu.convert_f32_to_bf16(x_gemm, &tmp, nelems)?;
        let w_ref = WeightRef {
            buf: &w.buf,
            dtype: w.gpu_dtype,
            m: w.m,
            k: w.k,
            row_stride: w.k,
            rotation: None,
            awq_scale: None,
        };
        let ctx = DispatchCtx::new(gpu);
        let params = GemmParams {
            w: &w_ref,
            x: &tmp,
            y,
            batch_size,
        };
        let family = llama::gemm_family();
        let res = family
            .run_key(
                hipfire_dispatch::types::KernelKey::GemmBf16Mfma,
                &ctx,
                gpu,
                &params,
            )
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()));
        gpu.free_tensor(tmp)?;
        return res;
    }
    let ctx = DispatchCtx::new(gpu);
    let w_ref = WeightRef {
        buf: &w.buf,
        dtype: w.gpu_dtype,
        m: w.m,
        k: w.k,
        row_stride: w.k,
        rotation: None,
        awq_scale: None,
    };
    let params = GemmParams {
        w: &w_ref,
        x: x_gemm,
        y,
        batch_size,
    };
    let family = llama::gemm_family();

    if wmma_prefill_enabled() {
        // WMMA path: use GemmFamily::run which resolves the best kernel
        // for each dtype (WMMA where available, scalar fallback otherwise).
        // The F32->F16 conversion is handled inside the kernel methods.
        //
        // HARD ERROR on failure -- no silent fallthrough. The old code fell
        // through to the scalar key path when  errored (e.g. missing
        // WMMA kernel for this dtype x arch), which silently produced
        // special-token garbage from token 0 on gfx1201 (RDNA4) where the
        // gfx11  WMMA kernels do not exist. Refuse-dont-degrade: the
        // operator opted into WMMA explicitly, so a missing kernel is an
        // error, not a degrade. Unset HIPFIRE_WMMA_PREFILL (or use the
        // scalar HIPFIRE_BATCHED_PREFILL=1 path) on archs without coverage.
        return family.run(&ctx, gpu, &params).map_err(|e| {
            hip_bridge::HipError::new(0, &format!(
                "gemma4 WMMA prefill: GemmFamily::run failed for dtype {:?} on arch {}                  (HIPFIRE_WMMA_PREFILL=1 requires a WMMA kernel for every projection                  dtype -- unset it to use the scalar batched path): {e}",
                w.gpu_dtype, gpu.arch,
            ))
        });
    }

    // Scalar path (default) or fallback for unsupported WMMA dtypes.
    let key = match w.gpu_dtype {
        DType::HFQ4G256 => hipfire_dispatch::types::KernelKey::GemmHfq4G256,
        DType::HFQ4G128 => hipfire_dispatch::types::KernelKey::GemmHfq4G128,
        // Same kernel as HFQ4G256 (layout-identical); input pre-rotated above.
        DType::MQ4G256 => hipfire_dispatch::types::KernelKey::GemmHfq4G256,
        DType::Q8_0 => hipfire_dispatch::types::KernelKey::GemmQ8_0BatchedChunked,
        // No batched GEMM kernel for this dtype -- fall back to repeated GEMV
        // on the RAW input (weight_gemv applies any needed rotation itself).
        // This matches the old  fallback path.
        _ => {
            let x_tok = gpu.alloc_tensor(&[w.k], DType::F32)?;
            let y_tok = gpu.alloc_tensor(&[w.m], DType::F32)?;
            for b in 0..batch_size {
                gpu.hip
                    .memcpy_dtod_at(&x_tok.buf, 0, &x_raw.buf, b * w.k * 4, w.k * 4)?;
                weight_gemv(gpu, w, &x_tok, &y_tok)?;
                gpu.hip
                    .memcpy_dtod_at(&y.buf, b * w.m * 4, &y_tok.buf, 0, w.m * 4)?;
            }
            gpu.free_tensor(x_tok)?;
            gpu.free_tensor(y_tok)?;
            return Ok(());
        }
    };
    family
        .run_key(key, &ctx, gpu, &params)
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    // Debug parity hook: re-run token 0 through the per-token GEMV path
    // (raw input; weight_gemv rotates internally) and diff against the GEMM.
    if std::env::var("HIPFIRE_GEMMA4_GEMM_VERIFY").ok().as_deref() == Some("1") {
        let x_tok = gpu.alloc_tensor(&[w.k], DType::F32)?;
        let y_tok = gpu.alloc_tensor(&[w.m], DType::F32)?;
        gpu.hip
            .memcpy_dtod_at(&x_tok.buf, 0, &x_raw.buf, 0, w.k * 4)?;
        weight_gemv(gpu, w, &x_tok, &y_tok)?;
        let yv = gpu.download_f32(&y_tok)?;
        let yg = gpu.download_f32(y)?;
        let mut worst = 0f32;
        let mut wi = 0usize;
        for i in 0..w.m {
            let d = (yv[i] - yg[i]).abs();
            if d > worst {
                worst = d;
                wi = i;
            }
        }
        eprintln!("[gemm-verify] dtype={:?} m={} k={} b={} key={:?} worst={:.5} at {} gemv={:.4} gemm={:.4} head_gemv={:?} head_gemm={:?}",
            w.gpu_dtype, w.m, w.k, batch_size, key, worst, wi, yv[wi], yg[wi], &yv[..2], &yg[..2]);
        gpu.free_tensor(x_tok)?;
        gpu.free_tensor(y_tok)?;
    }
    Ok(())
}

/// Env-gated dump helper for v1-vs-v2 root-cause work.
/// Set HIPFIRE_GEMMA4_DUMP=1 to enable. Prints first 4 floats + sum + nan/inf count.
#[allow(dead_code)]
fn dbg_dump(gpu: &mut Gpu, label: &str, t: &GpuTensor, take: usize) {
    if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() != Some("1") {
        return;
    }
    let data = match gpu.download_f32(t) {
        Ok(d) => d,
        Err(_) => return,
    };
    let take = take.min(data.len());
    let head: Vec<f32> = data[..take.min(4)].iter().copied().collect();
    let sum: f64 = data[..take].iter().map(|&v| v as f64).sum();
    let nans = data[..take].iter().filter(|&&v| v.is_nan()).count();
    let infs = data[..take].iter().filter(|&&v| v.is_infinite()).count();
    eprintln!(
        "[dump] {label:42} sum={sum:>+14.4e} n={take:>6} head={head:?} nan={nans} inf={infs}"
    );
}
/// Optional last-position diagnostic capture for the lowered MoE path.
///
/// Set `HIPFIRE_GEMMA4_LAYER_DUMP` to a JSON output path and
/// `HIPFIRE_GEMMA4_LAYER_DUMP_POS` to the absolute token position to capture.
/// `HIPFIRE_GEMMA4_LAYER_DUMP_BOUNDARIES=1` additionally captures the
/// attention, dense-FFN, router, and MoE branch boundaries.  The path is
/// deliberately separate from `HIPFIRE_GEMMA4_DUMP`: the latter is an older
/// stderr probe used by the hand implementation.  No D2H work occurs unless
/// the JSON path and position are both configured.
#[derive(Debug, PartialEq, Eq)]
struct Gemma4LayerDumpConfig {
    path: String,
    position: usize,
    boundaries: bool,
}

fn parse_layer_dump_config(
    path: Option<String>,
    position: Option<&str>,
    boundaries: Option<&str>,
) -> Option<Gemma4LayerDumpConfig> {
    Some(Gemma4LayerDumpConfig {
        path: path?,
        position: position?.parse().ok()?,
        boundaries: boundaries == Some("1"),
    })
}

fn layer_dump_config() -> Option<&'static Gemma4LayerDumpConfig> {
    static CONFIG: std::sync::OnceLock<Option<Gemma4LayerDumpConfig>> = std::sync::OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let position = std::env::var("HIPFIRE_GEMMA4_LAYER_DUMP_POS").ok();
            let boundaries = std::env::var("HIPFIRE_GEMMA4_LAYER_DUMP_BOUNDARIES").ok();
            parse_layer_dump_config(
                std::env::var("HIPFIRE_GEMMA4_LAYER_DUMP").ok(),
                position.as_deref(),
                boundaries.as_deref(),
            )
        })
        .as_ref()
}

struct Gemma4LayerDump {
    path: String,
    position: usize,
    boundaries: bool,
    captured: serde_json::Map<String, serde_json::Value>,
}

impl Gemma4LayerDump {
    fn for_position(position: usize) -> Option<Self> {
        let config = layer_dump_config()?;
        (config.position == position).then(|| Self {
            path: config.path.clone(),
            position,
            boundaries: config.boundaries,
            captured: serde_json::Map::new(),
        })
    }

    fn capture(&mut self, gpu: &mut Gpu, label: impl Into<String>, tensor: &GpuTensor) {
        if let Some(stats) = gemma4_tensor_stats(gpu, tensor) {
            self.captured.insert(label.into(), stats);
        }
    }

    fn capture_boundary(
        &mut self,
        gpu: &mut Gpu,
        layer_idx: usize,
        label: &str,
        tensor: &GpuTensor,
    ) {
        self.capture(gpu, format!("L{layer_idx}_{label}"), tensor);
    }

    fn write(self, gpu: &mut Gpu, logits: &GpuTensor) {
        let logit_argmax = gpu.download_f32(logits).ok().and_then(|values| {
            values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index as u32)
        });
        let mut output = serde_json::Map::new();
        output.insert(
            "position".to_string(),
            serde_json::Value::from(self.position as u64),
        );
        output.insert(
            "captured".to_string(),
            serde_json::Value::Object(self.captured),
        );
        if let Some(argmax) = logit_argmax {
            output.insert("logit_argmax".to_string(), serde_json::Value::from(argmax));
        }
        if let Err(error) = std::fs::write(
            &self.path,
            serde_json::to_vec_pretty(&serde_json::Value::Object(output))
                .unwrap_or_else(|_| b"{}".to_vec()),
        ) {
            eprintln!("[gemma4 layer dump] failed to write {}: {error}", self.path);
        }
    }
}

fn rounded_json(value: f64, decimals: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::Value::from((value * decimals).round() / decimals)
    } else {
        serde_json::Value::Null
    }
}

fn gemma4_tensor_stats(gpu: &mut Gpu, tensor: &GpuTensor) -> Option<serde_json::Value> {
    let data = gpu.download_f32(tensor).ok()?;
    if data.is_empty() {
        return None;
    }
    let first8 = data
        .iter()
        .take(8)
        .map(|&value| rounded_json(value as f64, 100_000.0))
        .collect::<Vec<_>>();
    let sum = data.iter().map(|&value| value as f64).sum::<f64>();
    let norm = data
        .iter()
        .map(|&value| (value as f64) * (value as f64))
        .sum::<f64>()
        .sqrt();
    let min = data
        .iter()
        .map(|&value| value as f64)
        .fold(f64::INFINITY, f64::min);
    let max = data
        .iter()
        .map(|&value| value as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    Some(serde_json::json!({
        "first8": first8,
        "sum": rounded_json(sum, 10_000.0),
        "norm": rounded_json(norm, 10_000.0),
        "min": rounded_json(min, 10_000.0),
        "max": rounded_json(max, 10_000.0),
    }))
}

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Sliding-window causal attention (window=1024 on 31B).
    Sliding,
    /// Full causal attention (global).
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeType {
    /// Standard RoPE: all head_dim positions rotate.
    Default,
    /// Proportional RoPE (Gemma 4 full layers): only the first
    /// `partial_rotary_factor × head_dim` positions rotate; rest are NoPE.
    Proportional,
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    // Common
    pub dim: usize,        // hidden_size, e.g. 5376 on 31B
    pub n_layers: usize,   // 60 on 31B
    pub vocab_size: usize, // 262144 on Gemma 4
    pub norm_eps: f32,     // 1e-6
    pub bos_token: u32,    // 2
    pub eos_token: u32,    // 1
    pub pad_token: u32,    // 0

    // Attention heads (same count for sliding + full)
    pub n_heads: usize, // 32 on 31B

    // Sliding-window attention
    pub sliding_head_dim: usize,   // 256 on 31B
    pub sliding_n_kv_heads: usize, // 16 on 31B
    pub sliding_rope_theta: f32,   // 10000.0
    pub sliding_window: usize,     // 1024

    // Full attention (global)
    pub full_head_dim: usize,            // 512 on 31B (= global_head_dim)
    pub full_n_kv_heads: usize,          // 4 on 31B
    pub full_rope_theta: f32,            // 1_000_000.0
    pub full_rope_type: RopeType,        // Proportional on 31B
    pub full_partial_rotary_factor: f32, // 0.25
    pub attention_k_eq_v: bool,          // true on 31B — V = pre-k_norm output

    // FFN (SwiGLU, gelu_pytorch_tanh)
    pub hidden_dim: usize, // intermediate_size = 21504 on 31B

    // MoE (26B-A4B). enable_moe_block=true → every layer carries a parallel
    // MoE branch whose output sums with the standard SwiGLU output before
    // the post_feedforward_layernorm. Zero on dense models (31B).
    pub enable_moe_block: bool,       // true on 26B-A4B
    pub moe_intermediate_size: usize, // 704 on 26B-A4B (per-expert FFN hidden)
    pub num_experts: usize,           // 128 on 26B-A4B
    pub top_k_experts: usize,         // 8 on 26B-A4B (kernel hardcoded to 8)

    // Output
    pub final_logit_softcapping: f32, // 30.0 — tanh(x/30)*30
    pub tie_word_embeddings: bool,    // true — lm_head aliases embed_tokens
    pub embed_scale: f32,             // sqrt(dim), applied at embed lookup

    // Per-layer dispatch (len == n_layers)
    pub layer_types: Vec<LayerType>,

    // Vision integration (present even on text-only 31B since config ships it)
    pub has_vision: bool,
    pub image_token_id: u32, // 258880
    pub boi_token_id: u32,   // 255999
    pub eoi_token_id: u32,   // 258882
    pub audio_token_id: u32, // 258881 (reserved, unused on dense 31B)
    pub video_token_id: u32, // 258884 (reserved)
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Gemma4Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let bos_token = tc.get("bos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
    let eos_token = tc.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let pad_token = tc.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;

    // Sliding attention params
    let sliding_head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);
    let sliding_n_kv_heads = tc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let sliding_window = tc
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;

    // Full attention params (may differ from sliding)
    let full_head_dim = tc
        .get("global_head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(sliding_head_dim);
    let full_n_kv_heads = tc
        .get("num_global_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(sliding_n_kv_heads as u64) as usize;
    let attention_k_eq_v = tc
        .get("attention_k_eq_v")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // rope_parameters is a dict with "sliding_attention" and "full_attention" sub-dicts
    // per the Gemma 4 config schema. Parse both independently.
    let rope_params = tc.get("rope_parameters");
    let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
    let full_rope = rope_params.and_then(|r| r.get("full_attention"));

    let sliding_rope_theta = sliding_rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0) as f32;
    let full_rope_theta = full_rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;
    let full_rope_type = match full_rope
        .and_then(|r| r.get("rope_type"))
        .and_then(|v| v.as_str())
    {
        Some("proportional") => RopeType::Proportional,
        _ => RopeType::Default,
    };
    let full_partial_rotary_factor = full_rope
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let hidden_dim = tc.get("intermediate_size")?.as_u64()? as usize;

    // MoE config (26B-A4B). Absent / false on dense models (31B).
    let enable_moe_block = tc
        .get("enable_moe_block")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let moe_intermediate_size = tc
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let top_k_experts = tc
        .get("top_k_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let final_logit_softcapping = tc
        .get("final_logit_softcapping")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let tie_word_embeddings = tc
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .or_else(|| config.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    let embed_scale = (dim as f32).sqrt();

    let layer_types: Vec<LayerType> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| match v.as_str().unwrap_or("sliding_attention") {
                    "full_attention" => LayerType::Full,
                    _ => LayerType::Sliding,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![LayerType::Sliding; n_layers]);

    // Multimodal token IDs (top-level in config, not under text_config)
    let has_vision = config
        .get("vision_config")
        .map(|v| !v.is_null())
        .unwrap_or(false);
    let image_token_id = config
        .get("image_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258880) as u32;
    let boi_token_id = config
        .get("boi_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(255999) as u32;
    let eoi_token_id = config
        .get("eoi_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258882) as u32;
    let audio_token_id = config
        .get("audio_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258881) as u32;
    let video_token_id = config
        .get("video_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258884) as u32;

    Some(Gemma4Config {
        dim,
        n_layers,
        vocab_size,
        norm_eps,
        bos_token,
        eos_token,
        pad_token,
        n_heads,
        sliding_head_dim,
        sliding_n_kv_heads,
        sliding_rope_theta,
        sliding_window,
        full_head_dim,
        full_n_kv_heads,
        full_rope_theta,
        full_rope_type,
        full_partial_rotary_factor,
        attention_k_eq_v,
        hidden_dim,
        enable_moe_block,
        moe_intermediate_size,
        num_experts,
        top_k_experts,
        final_logit_softcapping,
        tie_word_embeddings,
        embed_scale,
        layer_types,
        has_vision,
        image_token_id,
        boi_token_id,
        eoi_token_id,
        audio_token_id,
        video_token_id,
    })
}

// ─── Weights ────────────────────────────────────────────────────────────

/// Per-layer weights for a SLIDING layer (head_dim=256, 16 KV heads, full RoPE).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,            // [dim]
    pub post_attention_layernorm: GpuTensor,   // [dim]
    pub pre_feedforward_layernorm: GpuTensor,  // [dim]
    pub post_feedforward_layernorm: GpuTensor, // [dim]
    pub layer_scalar: GpuTensor,               // [1]
    /// Host-side mirror of layer_scalar. Populated at load time so decode can
    /// call `gpu.scale_f32(x, layer_scalar_host)` without a D2H round-trip.
    pub layer_scalar_host: f32,

    // Attention (sliding — head_dim=256)
    pub q_proj: WeightTensor, // [n_heads * 256, dim]
    pub k_proj: WeightTensor, // [16 * 256, dim]
    pub v_proj: WeightTensor, // [16 * 256, dim]
    pub o_proj: WeightTensor, // [dim, n_heads * 256]
    pub q_norm: GpuTensor,    // [256]
    pub k_norm: GpuTensor,    // [256]

    // MLP (SwiGLU)
    pub gate_proj: WeightTensor, // [hidden_dim, dim]
    pub up_proj: WeightTensor,   // [hidden_dim, dim]
    pub down_proj: WeightTensor, // [dim, hidden_dim]

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-layer weights for a FULL layer (head_dim=512, 4 KV heads, K=V shared).
///
/// Note: no `v_proj` — V is the pre-k_norm output of k_proj, renormed by
/// weight-less `v_norm`. No `v_norm` tensor either (no_scale — the `with_scale=False`
/// RMSNorm applies only the divide, no learned gain). We reuse the existing
/// rmsnorm kernel with a ones-filled `v_norm_ones` buffer (shared across
/// full-attn layers) to preserve the no-scale semantics.
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar: GpuTensor,
    /// Host-side mirror of layer_scalar. See SlidingLayerWeights for rationale.
    pub layer_scalar_host: f32,

    // Attention (full — head_dim=512, K=V)
    pub q_proj: WeightTensor, // [n_heads * 512, dim]
    pub k_proj: WeightTensor, // [4 * 512, dim]
    // no v_proj — V = pre-k_norm output of k_proj
    pub o_proj: WeightTensor, // [dim, n_heads * 512]
    pub q_norm: GpuTensor,    // [512]
    pub k_norm: GpuTensor,    // [512]
    // no v_norm weight — v_norm is no-scale (divide only)

    // MLP (SwiGLU, same shape as sliding)
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-expert FFN weights for a single MoE expert. 128 of these per layer
/// on 26B-A4B; views into the per-layer pool allocation (so `free_gpu`
/// doesn't free these — the pool owns the bytes).
pub struct MoeExpertWeights {
    /// `[2 * moe_intermediate, dim]` — gate + up fused. Rows [0, mi) are
    /// gate; rows [mi, 2*mi) are up. Quantized as MQ4G256 / MG4G256 when
    /// dim is 256-aligned (it is on 26B-A4B: dim=2816).
    pub gate_up_proj: WeightTensor,
    /// `[dim, moe_intermediate]` — projects per-expert FFN hidden back to dim.
    /// On 26B-A4B, mi=704 isn't 256-aligned so this drops to Q8_0 via the
    /// quantizer fallback chain.
    pub down_proj: WeightTensor,
}

/// MoE branch weights for a Gemma 4 MoE layer (26B-A4B). Present on every
/// layer when `config.enable_moe_block` is set. The branch adds a parallel
/// FFN computation alongside the standard SwiGLU; outputs are summed via
/// sandwich norms then a final post_feedforward_layernorm closes the layer.
pub struct MoeLayerExtras {
    /// `[n_experts, dim]` — projects router input to expert logits.
    pub router_proj: WeightTensor,
    /// `[dim]` — multiplicative scale on router input (`router.scale` in HF).
    pub router_scale: GpuTensor,
    /// `[n_experts]` — per-expert post-`down_proj` scale (`router.per_expert_scale`).
    pub per_expert_scale: GpuTensor,
    /// Host mirror of `per_expert_scale` for fast top-K weight composition.
    pub per_expert_scale_host: Vec<f32>,
    /// `[dim]` — RMSNorm applied to attn_out before the MoE branch.
    pub pre_feedforward_layernorm_2: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_mlp (standard SwiGLU output) BEFORE summing.
    pub post_feedforward_layernorm_1: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_moe (MoE branch output) BEFORE summing.
    pub post_feedforward_layernorm_2: GpuTensor,
    /// Pool allocation for all gate_up tensors. Per-expert WeightTensors
    /// alias into this; `free_gpu` frees the pool, not each WeightTensor.
    pub experts_gate_up_pool: GpuTensor,
    /// Pool allocation for all down tensors. Same aliasing.
    pub experts_down_pool: GpuTensor,
    /// Byte stride between adjacent experts in the corresponding raw pool.
    pub gate_up_bytes: usize,
    pub down_bytes: usize,
    /// Per-expert views into the pools above.
    pub experts: Vec<MoeExpertWeights>,
    /// `[n_exp]` u64 device pointers — one per expert's gate_up weight
    /// base. Built once at load by reading each expert's pool sub-view
    /// pointer. The indexed MoE kernels read
    /// `expert_ptrs[topk_indices[krank]]` to locate the active expert
    /// weight WITHOUT a D2H sync.
    pub experts_gate_up_ptrs: GpuTensor,
    /// `[n_exp]` u64 device pointers for each expert's down weight base.
    pub experts_down_ptrs: GpuTensor,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

pub(crate) fn tensor_owner_bytes(tensor: &GpuTensor) -> usize {
    if tensor.buf.is_borrowed() {
        0
    } else {
        tensor.buf.size()
    }
}

pub(crate) fn device_buffer_owner_bytes(buffer: &hip_bridge::DeviceBuffer) -> usize {
    if buffer.is_borrowed() {
        0
    } else {
        buffer.size()
    }
}

pub(crate) fn weight_owner_bytes(weight: &WeightTensor) -> usize {
    let mut bytes = tensor_owner_bytes(&weight.buf);
    if let Some(awq_scale) = weight.awq_scale.as_ref() {
        bytes += tensor_owner_bytes(awq_scale);
    }
    if let Some(paro) = weight.paro.as_ref() {
        if !paro.is_alias {
            bytes += tensor_owner_bytes(&paro.pairs);
            bytes += tensor_owner_bytes(&paro.theta);
            bytes += tensor_owner_bytes(&paro.channel_scales);
        }
    }
    bytes
}

impl LayerWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            LayerWeights::Sliding(s) => {
                let SlidingLayerWeights {
                    input_layernorm,
                    post_attention_layernorm,
                    pre_feedforward_layernorm,
                    post_feedforward_layernorm,
                    layer_scalar,
                    layer_scalar_host: _,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    gate_proj,
                    up_proj,
                    down_proj,
                    moe,
                } = s;
                down_proj.free_all(gpu);
                up_proj.free_all(gpu);
                gate_proj.free_all(gpu);
                let _ = gpu.free_tensor(k_norm);
                let _ = gpu.free_tensor(q_norm);
                o_proj.free_all(gpu);
                v_proj.free_all(gpu);
                k_proj.free_all(gpu);
                q_proj.free_all(gpu);
                let _ = gpu.free_tensor(post_feedforward_layernorm);
                let _ = gpu.free_tensor(pre_feedforward_layernorm);
                let _ = gpu.free_tensor(post_attention_layernorm);
                let _ = gpu.free_tensor(input_layernorm);
                if let Some(moe) = moe {
                    Gemma4Weights::free_moe(gpu, moe);
                }
                let _ = gpu.free_tensor(layer_scalar);
            }
            LayerWeights::Full(f) => {
                let FullLayerWeights {
                    input_layernorm,
                    post_attention_layernorm,
                    pre_feedforward_layernorm,
                    post_feedforward_layernorm,
                    layer_scalar,
                    layer_scalar_host: _,
                    q_proj,
                    k_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    gate_proj,
                    up_proj,
                    down_proj,
                    moe,
                } = f;
                down_proj.free_all(gpu);
                up_proj.free_all(gpu);
                gate_proj.free_all(gpu);
                let _ = gpu.free_tensor(k_norm);
                let _ = gpu.free_tensor(q_norm);
                o_proj.free_all(gpu);
                k_proj.free_all(gpu);
                q_proj.free_all(gpu);
                let _ = gpu.free_tensor(post_feedforward_layernorm);
                let _ = gpu.free_tensor(pre_feedforward_layernorm);
                let _ = gpu.free_tensor(post_attention_layernorm);
                let _ = gpu.free_tensor(input_layernorm);
                if let Some(moe) = moe {
                    Gemma4Weights::free_moe(gpu, moe);
                }
                let _ = gpu.free_tensor(layer_scalar);
            }
        }
    }

    fn owner_bytes(&self) -> usize {
        match self {
            LayerWeights::Sliding(s) => {
                tensor_owner_bytes(&s.input_layernorm)
                    + tensor_owner_bytes(&s.post_attention_layernorm)
                    + tensor_owner_bytes(&s.pre_feedforward_layernorm)
                    + tensor_owner_bytes(&s.post_feedforward_layernorm)
                    + tensor_owner_bytes(&s.layer_scalar)
                    + weight_owner_bytes(&s.q_proj)
                    + weight_owner_bytes(&s.k_proj)
                    + weight_owner_bytes(&s.v_proj)
                    + weight_owner_bytes(&s.o_proj)
                    + tensor_owner_bytes(&s.q_norm)
                    + tensor_owner_bytes(&s.k_norm)
                    + weight_owner_bytes(&s.gate_proj)
                    + weight_owner_bytes(&s.up_proj)
                    + weight_owner_bytes(&s.down_proj)
                    + s.moe.as_ref().map_or(0, MoeLayerExtras::owner_bytes)
            }
            LayerWeights::Full(f) => {
                tensor_owner_bytes(&f.input_layernorm)
                    + tensor_owner_bytes(&f.post_attention_layernorm)
                    + tensor_owner_bytes(&f.pre_feedforward_layernorm)
                    + tensor_owner_bytes(&f.post_feedforward_layernorm)
                    + tensor_owner_bytes(&f.layer_scalar)
                    + weight_owner_bytes(&f.q_proj)
                    + weight_owner_bytes(&f.k_proj)
                    + weight_owner_bytes(&f.o_proj)
                    + tensor_owner_bytes(&f.q_norm)
                    + tensor_owner_bytes(&f.k_norm)
                    + weight_owner_bytes(&f.gate_proj)
                    + weight_owner_bytes(&f.up_proj)
                    + weight_owner_bytes(&f.down_proj)
                    + f.moe.as_ref().map_or(0, MoeLayerExtras::owner_bytes)
            }
        }
    }
}

impl MoeLayerExtras {
    fn owner_bytes(&self) -> usize {
        weight_owner_bytes(&self.router_proj)
            + tensor_owner_bytes(&self.router_scale)
            + tensor_owner_bytes(&self.per_expert_scale)
            + tensor_owner_bytes(&self.pre_feedforward_layernorm_2)
            + tensor_owner_bytes(&self.post_feedforward_layernorm_1)
            + tensor_owner_bytes(&self.post_feedforward_layernorm_2)
            + tensor_owner_bytes(&self.experts_gate_up_pool)
            + tensor_owner_bytes(&self.experts_down_pool)
            + tensor_owner_bytes(&self.experts_gate_up_ptrs)
            + tensor_owner_bytes(&self.experts_down_ptrs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LoweredOwnerId(usize);

enum LoweredOwner {
    Empty,
    Tensor(GpuTensor),
    Weight(WeightTensor),
    Buffer(DeviceBuffer),
    Moe(MoeLayerExtras),
    Layer(LayerWeights),
}

fn lowered_owner_bytes(owner: &LoweredOwner) -> usize {
    match owner {
        LoweredOwner::Empty => 0,
        LoweredOwner::Tensor(tensor) => tensor_owner_bytes(tensor),
        LoweredOwner::Weight(weight) => weight_owner_bytes(weight),
        LoweredOwner::Buffer(buffer) => buffer.size(),
        LoweredOwner::Moe(moe) => moe.owner_bytes(),
        LoweredOwner::Layer(layer) => layer.owner_bytes(),
    }
}

struct LoweredOwnerTransaction<'a> {
    gpu: &'a mut Gpu,
    owners: Vec<LoweredOwner>,
    live_owner_bytes: usize,
}

impl<'a> LoweredOwnerTransaction<'a> {
    fn new(gpu: &'a mut Gpu) -> Self {
        Self {
            gpu,
            owners: Vec::new(),
            live_owner_bytes: 0,
        }
    }

    fn gpu_mut(&mut self) -> &mut Gpu {
        self.gpu
    }

    fn push_new(&mut self, owner: LoweredOwner) -> LoweredOwnerId {
        let bytes = lowered_owner_bytes(&owner);
        register_live_owner_bytes(bytes);
        self.live_owner_bytes += bytes;
        let id = LoweredOwnerId(self.owners.len());
        self.owners.push(owner);
        id
    }

    fn push_transferred(&mut self, owner: LoweredOwner) -> LoweredOwnerId {
        let bytes = lowered_owner_bytes(&owner);
        self.live_owner_bytes += bytes;
        let id = LoweredOwnerId(self.owners.len());
        self.owners.push(owner);
        id
    }

    fn push_tensor(&mut self, tensor: GpuTensor) -> LoweredOwnerId {
        self.push_new(LoweredOwner::Tensor(tensor))
    }

    fn push_weight(&mut self, weight: WeightTensor) -> LoweredOwnerId {
        self.push_new(LoweredOwner::Weight(weight))
    }

    fn push_buffer(&mut self, buffer: DeviceBuffer) -> LoweredOwnerId {
        self.push_new(LoweredOwner::Buffer(buffer))
    }

    fn push_moe(&mut self, moe: MoeLayerExtras) -> LoweredOwnerId {
        self.push_transferred(LoweredOwner::Moe(moe))
    }

    fn push_layer(&mut self, layer: LayerWeights) -> LoweredOwnerId {
        self.push_transferred(LoweredOwner::Layer(layer))
    }

    fn tensor_ref(&self, id: LoweredOwnerId) -> &GpuTensor {
        match self.owners.get(id.0) {
            Some(LoweredOwner::Tensor(tensor)) => tensor,
            _ => panic!("lowered owner is not a GPU tensor"),
        }
    }

    fn take_owner(&mut self, id: LoweredOwnerId) -> LoweredOwner {
        let owner = std::mem::replace(&mut self.owners[id.0], LoweredOwner::Empty);
        let bytes = lowered_owner_bytes(&owner);
        debug_assert!(
            self.live_owner_bytes >= bytes,
            "lowered owner transaction accounting underflow on take"
        );
        self.live_owner_bytes -= bytes;
        owner
    }

    fn take_tensor(&mut self, id: LoweredOwnerId) -> GpuTensor {
        match self.take_owner(id) {
            LoweredOwner::Tensor(tensor) => tensor,
            _ => panic!("lowered owner is not a GPU tensor"),
        }
    }

    fn take_weight(&mut self, id: LoweredOwnerId) -> WeightTensor {
        match self.take_owner(id) {
            LoweredOwner::Weight(weight) => weight,
            _ => panic!("lowered owner is not a weight"),
        }
    }

    fn take_buffer(&mut self, id: LoweredOwnerId) -> DeviceBuffer {
        match self.take_owner(id) {
            LoweredOwner::Buffer(buffer) => buffer,
            _ => panic!("lowered owner is not a device buffer"),
        }
    }

    fn take_moe(&mut self, id: LoweredOwnerId) -> MoeLayerExtras {
        match self.take_owner(id) {
            LoweredOwner::Moe(moe) => moe,
            _ => panic!("lowered owner is not MoE extras"),
        }
    }

    fn take_layer(&mut self, id: LoweredOwnerId) -> LayerWeights {
        match self.take_owner(id) {
            LoweredOwner::Layer(layer) => layer,
            _ => panic!("lowered owner is not a layer"),
        }
    }

    fn commit(mut self) {
        debug_assert_eq!(
            self.live_owner_bytes, 0,
            "lowered owner transaction committed with live local bytes"
        );
        debug_assert!(self
            .owners
            .iter()
            .all(|owner| matches!(owner, LoweredOwner::Empty)));
        self.owners.clear();
    }
}

fn free_lowered_owner(gpu: &mut Gpu, owner: LoweredOwner) {
    match owner {
        LoweredOwner::Empty => {}
        LoweredOwner::Tensor(tensor) => {
            let _ = gpu.free_tensor(tensor);
        }
        LoweredOwner::Weight(weight) => weight.free_all(gpu),
        LoweredOwner::Buffer(buffer) => {
            let _ = gpu.hip.free(buffer);
        }
        LoweredOwner::Moe(moe) => Gemma4Weights::free_moe(gpu, moe),
        LoweredOwner::Layer(layer) => layer.free_gpu(gpu),
    }
}

impl Drop for LoweredOwnerTransaction<'_> {
    fn drop(&mut self) {
        while let Some(owner) = self.owners.pop() {
            let bytes = lowered_owner_bytes(&owner);
            debug_assert!(
                self.live_owner_bytes >= bytes,
                "lowered owner transaction accounting underflow on drop"
            );
            self.live_owner_bytes -= bytes;
            free_lowered_owner(self.gpu, owner);
            unregister_live_owner_bytes(bytes);
        }
        debug_assert_eq!(
            self.live_owner_bytes, 0,
            "lowered owner transaction dropped with live local bytes"
        );
    }
}

pub struct Gemma4Weights {
    /// Token embedding [vocab_size, dim], Q8F16 to keep the 262144×5376 table manageable.
    /// Aliased as lm_head when tie_word_embeddings is true.
    pub embed_tokens: GpuTensor,
    /// Embed/LM-head format tag for dispatch.
    pub embd_format: EmbeddingFormat,
    /// LM-head projection (shares bytes with embed_tokens when tied).
    pub lm_head: WeightTensor,
    /// Model-final RMSNorm scale [dim].
    pub final_norm: GpuTensor,
    /// Per-layer weights indexed by layer ordinal.
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    /// Return all owning GPU buffers to the pool. The tied `lm_head` is a
    /// borrowed alias of `embed_tokens` and is intentionally not freed.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let owner_bytes = self.owner_bytes();
        let Gemma4Weights {
            embed_tokens,
            embd_format: _,
            lm_head: _,
            final_norm,
            layers,
        } = self;
        for layer in layers.into_iter().rev() {
            layer.free_gpu(gpu);
        }
        let _ = gpu.free_tensor(final_norm);
        let _ = gpu.free_tensor(embed_tokens);
        unregister_live_owner_bytes(owner_bytes);
    }

    /// Sum the actual capacities of all owning descriptors. Borrowed aliases
    /// (the tied LM head and pool-backed expert views) contribute zero.
    pub fn owner_bytes(&self) -> usize {
        tensor_owner_bytes(&self.embed_tokens)
            + tensor_owner_bytes(&self.final_norm)
            + self
                .layers
                .iter()
                .map(LayerWeights::owner_bytes)
                .sum::<usize>()
    }

    fn free_moe(gpu: &mut Gpu, moe: MoeLayerExtras) {
        let MoeLayerExtras {
            router_proj,
            router_scale,
            per_expert_scale,
            per_expert_scale_host: _,
            pre_feedforward_layernorm_2,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            experts_gate_up_pool,
            experts_down_pool,
            gate_up_bytes: _,
            down_bytes: _,
            experts: _,
            experts_gate_up_ptrs,
            experts_down_ptrs,
        } = moe;
        // Reverse construction order: pointer tables, pools, norms, scales,
        // and the router weight. Expert descriptors are borrowed pool views.
        let _ = gpu.free_tensor(experts_down_ptrs);
        let _ = gpu.free_tensor(experts_gate_up_ptrs);
        let _ = gpu.free_tensor(experts_down_pool);
        let _ = gpu.free_tensor(experts_gate_up_pool);
        let _ = gpu.free_tensor(post_feedforward_layernorm_2);
        let _ = gpu.free_tensor(post_feedforward_layernorm_1);
        let _ = gpu.free_tensor(pre_feedforward_layernorm_2);
        let _ = gpu.free_tensor(per_expert_scale);
        let _ = gpu.free_tensor(router_scale);
        router_proj.free_all(gpu);
    }
}
/// Sum actual bytes held by a lowered KV cache's owning tensors.
pub fn kv_owner_bytes(kv: &llama::KvCache) -> usize {
    kv.k_gpu
        .iter()
        .chain(kv.v_gpu.iter())
        .chain(kv.k_scales.iter())
        .chain(kv.v_scales.iter())
        .map(tensor_owner_bytes)
        .chain(kv.givens_cos.iter().map(tensor_owner_bytes))
        .chain(kv.givens_sin.iter().map(tensor_owner_bytes))
        .sum()
}

/// Convert a checked KV teardown's failed tensors into the common retained
/// cleanup container. Successful frees have already been consumed.
pub fn kv_cleanup_failure_from_remaining(remaining: Vec<(String, GpuTensor)>) -> GpuCleanupFailure {
    let mut failure = GpuCleanupFailure::empty();
    for (label, tensor) in remaining {
        failure.add_retained(RetainedGpuTensor {
            label,
            tensor,
            last_error: "kv free_checked failed".to_string(),
        });
    }
    failure
}

/// Sum actual bytes still held by a retained KV cleanup failure.
pub fn kv_cleanup_failure_bytes(failure: &GpuCleanupFailure) -> usize {
    failure
        .failed_tensors
        .iter()
        .map(|retained| tensor_owner_bytes(&retained.tensor))
        .sum()
}

#[derive(Debug)]
struct TrackedKvCleanup {
    failure: Option<GpuCleanupFailure>,
    live_bytes: usize,
}

impl RetryableOwner for TrackedKvCleanup {
    fn retry_boxed(mut self: Box<Self>, gpu: &mut Gpu) -> Result<(), Box<dyn RetryableOwner>> {
        let failure = self
            .failure
            .take()
            .expect("tracked KV cleanup retry called without retained failure");
        let before = self.live_bytes;
        match failure.retry(gpu) {
            Ok(()) => {
                unregister_live_owner_bytes(before);
                Ok(())
            }
            Err(remaining) => {
                let after = kv_cleanup_failure_bytes(&remaining);
                unregister_live_owner_bytes(before.saturating_sub(after));
                self.live_bytes = after;
                self.failure = Some(remaining);
                Err(self)
            }
        }
    }

    fn num_failed(&self) -> usize {
        self.failure
            .as_ref()
            .map_or(0, GpuCleanupFailure::num_failed)
    }

    fn error_summaries(&self) -> Vec<String> {
        self.failure
            .as_ref()
            .map_or_else(Vec::new, GpuCleanupFailure::error_summaries)
    }
}

/// Wrap a failed KV free with live-owner accounting so a later retained
/// cleanup retry decrements bytes only for allocations actually released.
pub fn tracked_kv_cleanup_failure(
    failure: GpuCleanupFailure,
    live_bytes: usize,
) -> GpuCleanupFailure {
    let mut tracked = GpuCleanupFailure::empty();
    tracked.add_other(Box::new(TrackedKvCleanup {
        failure: Some(failure),
        live_bytes,
    }));
    tracked
}

// ─── Loading helpers ───────────────────────────────────────────────────

/// Decode a shape-[n] F16 or F32 tensor from HFQ into an F32 host Vec.
fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> HipResult<Vec<f32>> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| hip_bridge::HipError::new(0, &format!("tensor not found: {name}")))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if n != expected_n {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("shape mismatch for {name}: expected {expected_n}, got {n}"),
        ));
    }
    if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1")
        && data.len() <= 4
        && name.contains("layer_scalar")
    {
        eprintln!(
            "[gemma4] load_f32_vec({name}): qt={}, shape={:?}, raw_bytes={:02x?}",
            info.quant_type,
            info.shape,
            &data[..data.len().min(4)]
        );
    }
    let f32_data = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        qt => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("expected F16/F32 for {name}, got qt={qt}"),
            ))
        }
    };
    Ok(f32_data)
}

/// Load a Gemma 4 RMSNorm weight — `x * weight` form, NO +1 shift.
///
/// Distinct from qwen35::load_norm_weight which shifts by +1 for HF Gemma
/// 2/3-style `x * (1 + weight)`. Gemma 4 uses plain `x * weight` with weights
/// initialized to 1.0 (see modeling_gemma4.py::Gemma4RMSNorm line 157).
fn load_gemma4_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize) -> HipResult<GpuTensor> {
    let f32_data = load_f32_vec(hfq, name, dim)?;
    gpu.upload_f32(&f32_data, &[dim])
}

/// Load a 256-element head-dim Q/K RMSNorm weight. Same semantics as
/// `load_gemma4_norm` but scoped to the attention head_dim (256 on sliding,
/// 512 on full).
fn load_gemma4_head_norm(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    head_dim: usize,
) -> HipResult<GpuTensor> {
    load_gemma4_norm(hfq, gpu, name, head_dim)
}

/// Load the per-layer `layer_scalar` — shape-[1] BF16/F16 tensor — returning
/// both a GPU-resident [1]-tensor (for potential batched use) and its host-side
/// f32 value (used by the decode path to call `scale_f32(x, cpu_scalar)`).
fn load_layer_scalar(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<(GpuTensor, f32)> {
    let data = load_f32_vec(hfq, name, 1)?;
    let host_val = data[0];
    let gpu_tensor = gpu.upload_f32(&data, &[1])?;
    Ok((gpu_tensor, host_val))
}

/// Load a quantized projection weight. Mirrors qwen35::load_weight_tensor_raw
/// but uses the Gemma 4 tensor-name convention (`model.language_model.<name>`).
fn load_gemma4_weight(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| hip_bridge::HipError::new(0, &format!("tensor not found: {name}")))?;
    let dtype = match info.quant_type {
        1 => {
            // F16 → upload as f32
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                awq_scale: None,
                paro: None,
            });
        }
        16 => {
            if rdna_compute::calib_force_bf16() {
                // Native BF16 teacher — keep raw 2-byte payload as BF16 for MFMA.
                // Otherwise the batched GEMM would land on the scalar F32 kernel.
                let buf = gpu.upload_raw(data, &[m, k])?;
                return Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::BF16,
                    m,
                    k,
                    row_stride: 0,
                    awq_scale: None,
                    paro: None,
                });
            }
            // Default: BF16 → widen to F32 (shift, not f16 decode)
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                awq_scale: None,
                paro: None,
            });
        }
        2 => {
            // F32 raw (oracle / --format f32 passthrough .hfq) — upload as-is.
            let buf = gpu.upload_raw(data, &[m, k])?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                awq_scale: None,
                paro: None,
            });
        }
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        10 => DType::HFQ2G128,
        11 => DType::HFQ3G256,
        12 => DType::HFQ3G128,
        13 => DType::MQ4G256,
        14 => DType::MQ8G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        // MG4-G256 — Magnum-Gemma 4-bit. Same binary layout as MQ4G256 (136 B/group),
        // differs only in calibration policy at quant time. Alias to MQ4G256 so the
        // existing GEMV path handles it without a kernel change. ID was 19 on
        // origin/gemma4 pre-rebase; reassigned to 30 because master shipped
        // MQ2G256Lloyd at 19.
        30 => DType::MQ4G256,
        qt => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("unsupported quant_type {qt} for {name}"),
            ))
        }
    };
    let buf = gpu.upload_raw(data, &[data.len()])?;
    let mut wt = WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        awq_scale: None,
        paro: None,
    };
    if wt.gpu_dtype.supports_awq_sidecar() {
        // `buf` is uploaded but not yet in the caller's staging set. Free it
        // before propagating the sidecar error so outer staging can reclaim
        // earlier allocations without leaking this weight.
        wt.awq_scale = match load_awq_scale(hfq, gpu, name, k) {
            Ok(scale) => scale,
            Err(error) => {
                wt.free_all(gpu);
                return Err(error);
            }
        };
    }
    Ok(wt)
}

fn gemma_moe_pool_dtype(quant_type: u8) -> Option<DType> {
    Some(match quant_type {
        1 => DType::F16,
        2 => DType::F32,
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        10 => DType::HFQ2G128,
        11 => DType::HFQ3G256,
        12 => DType::HFQ3G128,
        // MQ4G256 (13) and MG4G256 (30) share dispatch.
        13 | 30 => DType::MQ4G256,
        14 => DType::MQ8G256,
        15 => DType::MQ6G256,
        16 => DType::BF16,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        _ => return None,
    })
}

/// Load all experts of one MoE projection into a single owning pool.
fn load_moe_pool(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    n_exp: usize,
    base: &str,
) -> HipResult<(GpuTensor, DType, usize)> {
    // First pass: read first expert to learn quant_type + bytes-per-expert.
    let first_name = format!("{p}.experts.0.{base}.weight");
    let (first_info, first_data) = hfq.tensor_data(&first_name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {first_name}"))
    })?;
    let bytes_per_expert = first_data.len();
    let dtype = gemma_moe_pool_dtype(first_info.quant_type).ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "unsupported MoE expert quant_type {} for {first_name}",
                first_info.quant_type
            ),
        )
    })?;
    // Concat all experts' bytes into one CPU buffer, upload once.
    let mut concat = Vec::with_capacity(bytes_per_expert * n_exp);
    concat.extend_from_slice(first_data);
    for x in 1..n_exp {
        let name = format!("{p}.experts.{x}.{base}.weight");
        let (info, data) = hfq.tensor_data(&name).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {name}"))
        })?;
        if data.len() != bytes_per_expert {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MoE expert {name} byte size mismatch ({} vs {bytes_per_expert})",
                    data.len()
                ),
            ));
        }
        if info.quant_type != first_info.quant_type {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MoE expert {name} quant_type mismatch ({} vs {})",
                    info.quant_type, first_info.quant_type
                ),
            ));
        }
        concat.extend_from_slice(data);
    }
    let pool = gpu.upload_raw(&concat, &[concat.len()])?;
    Ok((pool, dtype, bytes_per_expert))
}

/// Load the MoE branch weights for a single Gemma 4 MoE layer (26B-A4B).
/// Builds expert `WeightTensor`s as non-owning views into two transaction-owned
/// pools. The local owner transaction frees every partial stage on error.
fn load_moe_layer_extras(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    config: &Gemma4Config,
) -> HipResult<MoeLayerExtras> {
    let n_exp = config.num_experts;
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let mut txn = LoweredOwnerTransaction::new(gpu);

    let router_proj = {
        let weight = load_gemma4_weight(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.router.proj.weight"),
            n_exp,
            dim,
        )?;
        txn.push_weight(weight)
    };
    // NOTE: `router.scale` and `per_expert_scale` ship WITHOUT the `.weight`
    // suffix in HF's 26B-A4B safetensors, so the loader uses bare paths.
    let router_scale = {
        let tensor = load_gemma4_norm(hfq, txn.gpu_mut(), &format!("{p}.router.scale"), dim)?;
        txn.push_tensor(tensor)
    };
    let per_expert_scale_host = load_f32_vec(hfq, &format!("{p}.router.per_expert_scale"), n_exp)?;
    let per_expert_scale = {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                per_expert_scale_host.as_ptr() as *const u8,
                per_expert_scale_host.len() * 4,
            )
        };
        let tensor = txn.gpu_mut().upload_raw(bytes, &[n_exp])?;
        txn.push_tensor(tensor)
    };
    let pre_feedforward_layernorm_2 = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.pre_feedforward_layernorm_2.weight"),
            dim,
        )?;
        txn.push_tensor(tensor)
    };
    let post_feedforward_layernorm_1 = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.post_feedforward_layernorm_1.weight"),
            dim,
        )?;
        txn.push_tensor(tensor)
    };
    let post_feedforward_layernorm_2 = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.post_feedforward_layernorm_2.weight"),
            dim,
        )?;
        txn.push_tensor(tensor)
    };

    let (gate_up_pool, gate_up_dtype, gate_up_bytes) = {
        let (pool, dtype, bytes) = load_moe_pool(hfq, txn.gpu_mut(), p, n_exp, "gate_up_proj")?;
        (txn.push_tensor(pool), dtype, bytes)
    };
    let (down_pool, down_dtype, down_bytes) = {
        let (pool, dtype, bytes) = load_moe_pool(hfq, txn.gpu_mut(), p, n_exp, "down_proj")?;
        (txn.push_tensor(pool), dtype, bytes)
    };

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gu_view = txn
            .tensor_ref(gate_up_pool)
            .sub_offset(x * gate_up_bytes, gate_up_bytes);
        let dn_view = txn
            .tensor_ref(down_pool)
            .sub_offset(x * down_bytes, down_bytes);
        experts.push(MoeExpertWeights {
            gate_up_proj: WeightTensor {
                buf: gu_view,
                gpu_dtype: gate_up_dtype,
                m: 2 * mi,
                k: dim,
                row_stride: 0,
                awq_scale: None,
                paro: None,
            },
            down_proj: WeightTensor {
                buf: dn_view,
                gpu_dtype: down_dtype,
                m: dim,
                k: mi,
                row_stride: 0,
                awq_scale: None,
                paro: None,
            },
        });
    }

    // Build [n_exp] device tensors of u64 weight-base pointers. Each u64 is
    // represented by two f32 slots, matching the indexed-MoE kernel ABI.
    let gate_up_ptr_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| (e.gate_up_proj.buf.buf.as_ptr() as u64).to_ne_bytes())
        .collect();
    let down_ptr_bytes: Vec<u8> = experts
        .iter()
        .flat_map(|e| (e.down_proj.buf.buf.as_ptr() as u64).to_ne_bytes())
        .collect();
    let experts_gate_up_ptrs = {
        let tensor = txn.gpu_mut().upload_raw(&gate_up_ptr_bytes, &[n_exp * 2])?;
        txn.push_tensor(tensor)
    };
    let experts_down_ptrs = {
        let tensor = txn.gpu_mut().upload_raw(&down_ptr_bytes, &[n_exp * 2])?;
        txn.push_tensor(tensor)
    };

    let result = MoeLayerExtras {
        router_proj: txn.take_weight(router_proj),
        router_scale: txn.take_tensor(router_scale),
        per_expert_scale: txn.take_tensor(per_expert_scale),
        per_expert_scale_host,
        pre_feedforward_layernorm_2: txn.take_tensor(pre_feedforward_layernorm_2),
        post_feedforward_layernorm_1: txn.take_tensor(post_feedforward_layernorm_1),
        post_feedforward_layernorm_2: txn.take_tensor(post_feedforward_layernorm_2),
        experts_gate_up_pool: txn.take_tensor(gate_up_pool),
        experts_down_pool: txn.take_tensor(down_pool),
        gate_up_bytes,
        down_bytes,
        experts,
        experts_gate_up_ptrs: txn.take_tensor(experts_gate_up_ptrs),
        experts_down_ptrs: txn.take_tensor(experts_down_ptrs),
    };
    txn.commit();
    Ok(result)
}

fn load_layer_weights(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    layer_idx: usize,
    config: &Gemma4Config,
) -> HipResult<LayerWeights> {
    let mut txn = LoweredOwnerTransaction::new(gpu);
    let (layer_scalar, layer_scalar_host) =
        load_layer_scalar(hfq, txn.gpu_mut(), &format!("{p}.layer_scalar"))?;

    let layer_scalar = txn.push_tensor(layer_scalar);
    let moe = if config.enable_moe_block {
        let loaded = load_moe_layer_extras(hfq, txn.gpu_mut(), p, config)?;
        Some(txn.push_moe(loaded))
    } else {
        None
    };

    let input_layernorm = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.input_layernorm.weight"),
            config.dim,
        )?;
        txn.push_tensor(tensor)
    };
    let post_attention_layernorm = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.post_attention_layernorm.weight"),
            config.dim,
        )?;
        txn.push_tensor(tensor)
    };
    let pre_feedforward_layernorm = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.pre_feedforward_layernorm.weight"),
            config.dim,
        )?;
        txn.push_tensor(tensor)
    };
    let post_feedforward_layernorm = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            &format!("{p}.post_feedforward_layernorm.weight"),
            config.dim,
        )?;
        txn.push_tensor(tensor)
    };

    let layer = match config.layer_types[layer_idx] {
        LayerType::Sliding => {
            let hd = config.sliding_head_dim;
            let kv_dim = config.sliding_n_kv_heads * hd;
            let q_dim = config.n_heads * hd;
            let q_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let k_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let v_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let o_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.o_proj.weight"),
                    config.dim,
                    q_dim,
                )?;
                txn.push_weight(weight)
            };
            let q_norm = {
                let tensor = load_gemma4_head_norm(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.q_norm.weight"),
                    hd,
                )?;
                txn.push_tensor(tensor)
            };
            let k_norm = {
                let tensor = load_gemma4_head_norm(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.k_norm.weight"),
                    hd,
                )?;
                txn.push_tensor(tensor)
            };
            let gate_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let up_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let down_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )?;
                txn.push_weight(weight)
            };
            LayerWeights::Sliding(SlidingLayerWeights {
                input_layernorm: txn.take_tensor(input_layernorm),
                post_attention_layernorm: txn.take_tensor(post_attention_layernorm),
                pre_feedforward_layernorm: txn.take_tensor(pre_feedforward_layernorm),
                post_feedforward_layernorm: txn.take_tensor(post_feedforward_layernorm),
                layer_scalar: txn.take_tensor(layer_scalar),
                layer_scalar_host,
                q_proj: txn.take_weight(q_proj),
                k_proj: txn.take_weight(k_proj),
                v_proj: txn.take_weight(v_proj),
                o_proj: txn.take_weight(o_proj),
                q_norm: txn.take_tensor(q_norm),
                k_norm: txn.take_tensor(k_norm),
                gate_proj: txn.take_weight(gate_proj),
                up_proj: txn.take_weight(up_proj),
                down_proj: txn.take_weight(down_proj),
                moe: moe.map(|id| txn.take_moe(id)),
            })
        }
        LayerType::Full => {
            let hd = config.full_head_dim;
            let kv_dim = config.full_n_kv_heads * hd;
            let q_dim = config.n_heads * hd;
            let q_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let k_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let o_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.o_proj.weight"),
                    config.dim,
                    q_dim,
                )?;
                txn.push_weight(weight)
            };
            let q_norm = {
                let tensor = load_gemma4_head_norm(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.q_norm.weight"),
                    hd,
                )?;
                txn.push_tensor(tensor)
            };
            let k_norm = {
                let tensor = load_gemma4_head_norm(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.self_attn.k_norm.weight"),
                    hd,
                )?;
                txn.push_tensor(tensor)
            };
            let gate_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let up_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?;
                txn.push_weight(weight)
            };
            let down_proj = {
                let weight = load_gemma4_weight(
                    hfq,
                    txn.gpu_mut(),
                    &format!("{p}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )?;
                txn.push_weight(weight)
            };
            // Full layers deliberately have no v_proj: V is the pre-k_norm
            // output of k_proj, renormalized without a learned v weight.
            LayerWeights::Full(FullLayerWeights {
                input_layernorm: txn.take_tensor(input_layernorm),
                post_attention_layernorm: txn.take_tensor(post_attention_layernorm),
                pre_feedforward_layernorm: txn.take_tensor(pre_feedforward_layernorm),
                post_feedforward_layernorm: txn.take_tensor(post_feedforward_layernorm),
                layer_scalar: txn.take_tensor(layer_scalar),
                layer_scalar_host,
                q_proj: txn.take_weight(q_proj),
                k_proj: txn.take_weight(k_proj),
                o_proj: txn.take_weight(o_proj),
                q_norm: txn.take_tensor(q_norm),
                k_norm: txn.take_tensor(k_norm),
                gate_proj: txn.take_weight(gate_proj),
                up_proj: txn.take_weight(up_proj),
                down_proj: txn.take_weight(down_proj),
                moe: moe.map(|id| txn.take_moe(id)),
            })
        }
    };
    txn.commit();
    Ok(layer)
}

/// Load Gemma 4 text model weights from an HFQ file.
///
/// Every uploaded owner remains in a constructor-local transaction until the
/// complete `Gemma4Weights` value is ready. Tied LM-head and MoE expert views
/// are borrowed aliases and are never registered as independent owners.
pub fn load_weights(
    hfq: &mut HfqFile,
    config: &Gemma4Config,
    gpu: &mut Gpu,
) -> HipResult<Gemma4Weights> {
    let mut txn = LoweredOwnerTransaction::new(gpu);
    eprintln!("gemma4: loading embed_tokens...");
    let embed_name = "model.language_model.embed_tokens.weight";
    let (embed_info, embed_data) = hfq
        .tensor_data(embed_name)
        .ok_or_else(|| hip_bridge::HipError::new(0, "embed_tokens not found in HFQ"))?;
    let (embed_tokens, embd_format) = match embed_info.quant_type {
        3 => {
            eprintln!("  (Q8_0 / Q8F16, {} MB)", embed_data.len() / 1_000_000);
            let tensor = txn.gpu_mut().upload_raw(embed_data, &[embed_data.len()])?;
            (txn.push_tensor(tensor), EmbeddingFormat::Q8_0)
        }
        6 => {
            eprintln!("  (HFQ4-G256, {} MB)", embed_data.len() / 1_000_000);
            let tensor = txn.gpu_mut().upload_raw(embed_data, &[embed_data.len()])?;
            (txn.push_tensor(tensor), EmbeddingFormat::HFQ4G256)
        }
        7 => {
            eprintln!("  (HFQ4-G128, {} MB)", embed_data.len() / 1_000_000);
            let tensor = txn.gpu_mut().upload_raw(embed_data, &[embed_data.len()])?;
            (txn.push_tensor(tensor), EmbeddingFormat::HFQ4G128)
        }
        1 => {
            eprintln!("  (F16 → F32)");
            let f32_data: Vec<f32> = embed_data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let tensor = txn
                .gpu_mut()
                .upload_f32(&f32_data, &[config.vocab_size, config.dim])?;
            (txn.push_tensor(tensor), EmbeddingFormat::F32)
        }
        16 => {
            eprintln!("  (BF16 → F32)");
            let f32_data: Vec<f32> = embed_data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let tensor = txn
                .gpu_mut()
                .upload_f32(&f32_data, &[config.vocab_size, config.dim])?;
            (txn.push_tensor(tensor), EmbeddingFormat::F32)
        }
        2 => {
            // F32 raw (oracle / --format f32 passthrough .hfq).
            eprintln!("  (F32 raw, {} MB)", embed_data.len() / 1_000_000);
            let f32_data: Vec<f32> = embed_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let tensor = txn
                .gpu_mut()
                .upload_f32(&f32_data, &[config.vocab_size, config.dim])?;
            (txn.push_tensor(tensor), EmbeddingFormat::F32)
        }
        qt => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("unsupported embed quant_type {qt}"),
            ))
        }
    };

    // Tied LM head: a non-owning alias of the transaction-owned embedding.
    let lm_head = {
        let embed = txn.tensor_ref(embed_tokens);
        let alias_buf = unsafe { embed.buf.alias() };
        let dtype = match embd_format {
            EmbeddingFormat::Q8_0 => DType::Q8_0,
            EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
            EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
            EmbeddingFormat::F32 => DType::F32,
            EmbeddingFormat::Q4K => DType::Q4K,
        };
        let alias_tensor = GpuTensor {
            buf: alias_buf,
            shape: embed.shape.clone(),
            dtype,
        };
        WeightTensor {
            buf: alias_tensor,
            gpu_dtype: dtype,
            m: config.vocab_size,
            k: config.dim,
            row_stride: 0,
            awq_scale: None,
            paro: None,
        }
    };

    eprintln!("gemma4: loading final norm...");
    let final_norm = {
        let tensor = load_gemma4_norm(
            hfq,
            txn.gpu_mut(),
            "model.language_model.norm.weight",
            config.dim,
        )?;
        txn.push_tensor(tensor)
    };

    eprintln!("gemma4: loading {} layers...", config.n_layers);
    let mut layer_ids = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let p = format!("model.language_model.layers.{i}");
        let layer = load_layer_weights(hfq, txn.gpu_mut(), &p, i, config)?;
        if i == 0 {
            let scalar = match &layer {
                LayerWeights::Sliding(s) => s.layer_scalar_host,
                LayerWeights::Full(f) => f.layer_scalar_host,
            };
            eprintln!(
                "[gemma4] L{} {} layer_scalar = {}",
                i,
                match &layer {
                    LayerWeights::Sliding(_) => "sliding",
                    LayerWeights::Full(_) => "full",
                },
                scalar
            );
        } else if i <= 6 && matches!(layer, LayerWeights::Full(_)) {
            let scalar = match &layer {
                LayerWeights::Full(f) => f.layer_scalar_host,
                LayerWeights::Sliding(_) => unreachable!(),
            };
            eprintln!("[gemma4] L{i} full layer_scalar = {scalar}");
        }
        layer_ids.push(txn.push_layer(layer));
    }
    eprintln!("gemma4: loaded all {} layers", config.n_layers);

    let layers = layer_ids.into_iter().map(|id| txn.take_layer(id)).collect();
    let weights = Gemma4Weights {
        embed_tokens: txn.take_tensor(embed_tokens),
        embd_format,
        lm_head,
        final_norm: txn.take_tensor(final_norm),
        layers,
    };
    txn.commit();
    Ok(weights)
}

/// One-time init for the scratch buffers that must hold a constant value
/// across forward passes (notably the ones-filled `v_norm_ones_full`).
/// Call once after `Gemma4Scratch::new` before the first forward pass.
pub fn init_scratch_constants(
    gpu: &mut Gpu,
    scratch: &Gemma4Scratch,
    full_head_dim: usize,
) -> HipResult<()> {
    let ones: Vec<f32> = vec![1.0; full_head_dim];
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
    gpu.hip.memcpy_htod(&scratch.v_norm_ones_full.buf, bytes)?;
    Ok(())
}

// ─── Scratch ────────────────────────────────────────────────────────────

use hip_bridge::DeviceBuffer;

/// Per-decode scratch, sized once at model-load time against the MAX of
/// sliding and full attention dimensions so a single buffer works across
/// layer types. 31B target shapes: sliding Q=[32*256]=8192, full Q=[32*512]=16384
/// → size Q at 16384. Sliding KV=[16*256]=4096, full KV=[4*512]=2048 → size at 4096.
pub struct Gemma4Scratch {
    pub x: GpuTensor,        // [dim] — hidden state
    pub residual: GpuTensor, // [dim] — saved for sandwich residual
    pub tmp: GpuTensor,      // [dim] — norm output scratch

    /// Position buffer (single i32 on device, updated per decode step).
    pub pos_buf: DeviceBuffer,

    // Attention scratch — sized for max(sliding, full)
    pub q: GpuTensor, // [max(n_heads*head_dim_sliding, n_heads*head_dim_full)]
    pub k: GpuTensor, // [max(n_kv_heads*head_dim for each layer type)]
    pub v: GpuTensor, // [same as k]
    pub attn_out: GpuTensor, // [same as q]

    // MLP scratch
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // Output
    pub logits: GpuTensor,     // [vocab_size]
    pub sample_buf: GpuTensor, // [2] — (token_id, new_rng_state) for GPU sampling
    pub repeat_buf: GpuTensor, // [1024] — rolling window for repeat penalty

    // Flash attention tile partials. Sized for the LARGER of the two
    // cache shapes: full-attn uses head_dim=512, max_tiles=max_seq/128.
    // Sliding uses head_dim=256, max_tiles=sliding_window/128 (much smaller).
    pub flash_partials: GpuTensor,

    // No-scale v_norm ones buffer (full-attn layers compute v_norm without
    // a learned weight — we pass this ones-filled tensor to the existing
    // rmsnorm kernel to get no-scale RMS semantics).
    pub v_norm_ones_full: GpuTensor, // [full_head_dim]

    // ── MoE scratch (26B-A4B only). Zero-sized on dense models. ─────────
    pub moe_cur_mlp: GpuTensor,   // [dim] — rmsnorm(ffn_out, post_norm_1)
    pub moe_pre2: GpuTensor,      // [dim] — rmsnorm(attn_out, pre_norm_2)
    pub moe_router_in: GpuTensor, // [dim] — router input (post-rmsnorm + scale)
    pub moe_router_logits: GpuTensor, // [n_experts]
    pub moe_topk_indices: GpuTensor, // [top_k_experts] — i32 packed in f32 slots
    pub moe_topk_weights: GpuTensor, // [top_k_experts]
    pub moe_cur_moe: GpuTensor,   // [dim] — accumulator across top-K experts
    pub moe_expert_gate_up: GpuTensor, // [2 * moe_intermediate_size]
    pub moe_expert_hidden: GpuTensor, // [moe_intermediate_size] — gelu(gate) * up
    pub moe_expert_out: GpuTensor, // [dim] — single expert's down_proj output

    // ── Indexed MoE batched scratch (k_top=8 hardcoded by kernel). ──────
    // These back the device-side fused path that replaces the 8-iteration
    // per-expert CPU loop. Only allocated when the MoE branch is enabled.
    /// `[dim]` — moe_pre2 after one FWHT pass (MQ4 gate_up expects pre-rotated x).
    pub moe_pre2_rot: GpuTensor,
    /// `[k_top × 2 × mi]` — fused gate+up output, one row per top-K rank.
    pub moe_expert_gate_batch: GpuTensor, // [k_top × mi]
    pub moe_expert_up_batch: GpuTensor, // [k_top × mi]

    // ── Prefill-batch (N tokens at a time) scratch ─────────────────────
    // Sized for max_prefill_batch tokens. Used by forward_prefill_chunk to
    // amortize MoE launch overhead across N tokens via the batched-indexed
    // kernels. None on models with no MoE (n_experts==0).
    pub max_prefill_batch: usize,
    /// `[N × dim]` — per-token attention output after o_proj (before residual).
    pub pb_attn_out: GpuTensor,
    /// `[N × dim]` — per-token dense FFN output (gemma4 26B-A4B-it computes
    /// this in parallel to MoE; both feed the sandwich norm).
    pub pb_ffn_out: GpuTensor,
    /// `[N × dim]` — pre-norm-2(attn_out) for each token.
    pub pb_moe_pre2: GpuTensor,
    /// `[N × dim]` — FWHT-rotated pre2 (MQ4 gate_up expects rotated x).
    pub pb_moe_pre2_rot: GpuTensor,
    /// `[N × dim]` — router input (rmsnorm(attn_out, router_scale) / sqrt(dim)).
    pub pb_moe_router_in: GpuTensor,
    /// `[N × n_experts]` — router logits batched.
    pub pb_moe_router_logits: GpuTensor,
    /// `[N × k_top]` i32 — top-K expert indices per token.
    pub pb_moe_topk_indices: GpuTensor,
    /// `[N × k_top]` — renormalized top-K weights per token.
    pub pb_moe_topk_weights: GpuTensor,
    /// `[n_exp + 1]` i32 — prefix-sum bucket offsets for routing-bucketed MoE GEMM.
    pub pb_moe_expert_offsets: GpuTensor,
    /// `[N × k_top]` i32 — packed (token_idx × k_top + krank), sorted by expert.
    pub pb_moe_expert_token_list: GpuTensor,
    /// `[N × k_top × mi]` — gate output across tokens × experts.
    pub pb_moe_gate_batch: GpuTensor,
    /// `[N × k_top × mi]` — up output across tokens × experts.
    pub pb_moe_up_batch: GpuTensor,
    /// `[N × k_top × mi]` — gelu_tanh(gate) * up.
    pub pb_moe_hidden_batch: GpuTensor,
    /// `[N × dim]` — accumulator for MoE branch output per token.
    pub pb_moe_cur_moe: GpuTensor,
    /// `[N × dim]` — post_norm_1(ffn_out) per token (dense branch).
    pub pb_moe_cur_mlp: GpuTensor,
    /// `[N × dim]` — residual stream per token across the prefill batch.
    pub pb_residual: GpuTensor,
    /// `[N × dim]` — post-rmsnorm input for batched projections.
    pub pb_tmp: GpuTensor,
    /// `[N × max_q_dim]` — batched Q projection output (sized for full layer = n_heads * full_head_dim).
    pub pb_q: GpuTensor,
    /// `[N × max_q_dim]` — batched attention output, kept separate from `pb_q`
    /// so batched flash attention never reads and writes the same buffer.
    pub pb_attn_q: GpuTensor,
    /// `[N × n_heads × max_tiles_full × (2+full_head_dim)]` — batch-sized flash
    /// partials for batched masked prefill attention. The single-query
    /// `flash_partials` overflows with batch_size>1.
    pub pb_flash_partials: GpuTensor,
    /// `[N × max_kv_dim]` — batched K projection output.
    pub pb_k: GpuTensor,
    /// `[N × max_kv_dim]` — batched V projection output.
    pub pb_v: GpuTensor,
    /// `[N × hidden_dim]` — batched gate proj output.
    pub pb_gate: GpuTensor,
    /// `[N × hidden_dim]` — batched up proj output.
    pub pb_up: GpuTensor,
    /// `[N × hidden_dim]` — batched gelu(gate)*up.
    pub pb_ffn_hidden: GpuTensor,
    /// `[N]` — i32 position buffer for batched RoPE.
    pub pb_positions: GpuTensor,
    /// `[MAX_BATCH × maxK]` BF16 — persistent staging for BF16 MFMA prefill.
    /// Sized once as `MAX_PREFILL_BATCH * max(dim, hidden_dim)` BF16 elements
    /// (~5.5 MB on 31B). Used by `run_prefill_gemm_inner`'s BF16 arm via
    /// `gpu.convert_f32_to_bf16(x, &pb_bf16_view, batch*k)`. Reused across
    /// q/k/v and gate+up groups (hoisted once per shared x).
    pub pb_bf16: GpuTensor,
    /// `[k_top × mi]` — gelu_tanh(gate)*up batched over k_top experts.
    pub moe_expert_hidden_batch: GpuTensor,
}

impl Gemma4Scratch {
    pub fn new(gpu: &mut Gpu, config: &Gemma4Config, _max_prefill: usize) -> HipResult<Self> {
        let mut txn = LoweredOwnerTransaction::new(gpu);
        macro_rules! alloc {
            ($name:ident, $shape:expr, $dtype:expr) => {
                let tensor = txn.gpu_mut().zeros($shape, $dtype)?;
                let $name = txn.push_tensor(tensor);
            };
        }

        let dim = config.dim;
        let q_dim =
            (config.n_heads * config.sliding_head_dim).max(config.n_heads * config.full_head_dim);
        let kv_dim = (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim);

        alloc!(x, &[dim], DType::F32);
        alloc!(residual, &[dim], DType::F32);
        alloc!(tmp, &[dim], DType::F32);
        let pos_buf = {
            let buffer = txn.gpu_mut().hip.malloc(4)?;
            txn.push_buffer(buffer)
        };
        alloc!(q, &[q_dim], DType::F32);
        alloc!(k, &[kv_dim], DType::F32);
        alloc!(v, &[kv_dim], DType::F32);
        alloc!(attn_out, &[q_dim], DType::F32);
        alloc!(gate_ffn, &[config.hidden_dim], DType::F32);
        alloc!(up_ffn, &[config.hidden_dim], DType::F32);
        alloc!(ffn_hidden, &[config.hidden_dim], DType::F32);
        alloc!(ffn_out, &[dim], DType::F32);
        alloc!(logits, &[config.vocab_size], DType::F32);
        alloc!(sample_buf, &[2], DType::F32);
        alloc!(repeat_buf, &[1024], DType::F32);

        // Flash partials are sized for full attention, which is the larger of
        // the two layer shapes. Keep the existing environment range.
        const FALLBACK_KV_SEQ: usize = 32768;
        const TILE_SIZE: usize = 128;
        let max_kv_seq: usize = std::env::var("HIPFIRE_KV_SEQ")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 128 && n <= 524_288)
            .unwrap_or(FALLBACK_KV_SEQ);
        let max_tiles_full = (max_kv_seq + TILE_SIZE - 1) / TILE_SIZE;
        let flash_partials_sz = config.n_heads * max_tiles_full * (2 + config.full_head_dim);
        alloc!(flash_partials, &[flash_partials_sz], DType::F32);

        let v_norm_max = config.sliding_head_dim.max(config.full_head_dim);
        alloc!(v_norm_ones_full, &[v_norm_max], DType::F32);

        // MoE scratch is intentionally allocated for dense models too; the
        // buffers are small and this keeps the forward path branch-free.
        let n_exp = config.num_experts.max(1);
        let mi = config.moe_intermediate_size.max(1);
        let k_top = config.top_k_experts.max(1);
        alloc!(moe_cur_mlp, &[dim], DType::F32);
        alloc!(moe_pre2, &[dim], DType::F32);
        alloc!(moe_router_in, &[dim], DType::F32);
        alloc!(moe_router_logits, &[n_exp], DType::F32);
        alloc!(moe_topk_indices, &[k_top], DType::F32);
        alloc!(moe_topk_weights, &[k_top], DType::F32);
        alloc!(moe_cur_moe, &[dim], DType::F32);
        alloc!(moe_expert_gate_up, &[2 * mi], DType::F32);
        alloc!(moe_expert_hidden, &[mi], DType::F32);
        alloc!(moe_expert_out, &[dim], DType::F32);
        alloc!(moe_pre2_rot, &[dim], DType::F32);
        alloc!(moe_expert_gate_batch, &[k_top * mi], DType::F32);
        alloc!(moe_expert_up_batch, &[k_top * mi], DType::F32);
        alloc!(moe_expert_hidden_batch, &[k_top * mi], DType::F32);

        const MAX_PREFILL_BATCH: usize = 128;
        alloc!(pb_attn_out, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(pb_ffn_out, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(pb_moe_pre2, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(pb_moe_pre2_rot, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(pb_moe_router_in, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(
            pb_moe_router_logits,
            &[MAX_PREFILL_BATCH, n_exp],
            DType::F32
        );
        alloc!(pb_moe_topk_indices, &[MAX_PREFILL_BATCH, k_top], DType::F32);
        alloc!(pb_moe_topk_weights, &[MAX_PREFILL_BATCH, k_top], DType::F32);
        alloc!(pb_moe_expert_offsets, &[n_exp + 1], DType::F32);
        alloc!(
            pb_moe_expert_token_list,
            &[MAX_PREFILL_BATCH, k_top],
            DType::F32
        );
        alloc!(
            pb_moe_gate_batch,
            &[MAX_PREFILL_BATCH, k_top * mi],
            DType::F32
        );
        alloc!(
            pb_moe_up_batch,
            &[MAX_PREFILL_BATCH, k_top * mi],
            DType::F32
        );
        alloc!(
            pb_moe_hidden_batch,
            &[MAX_PREFILL_BATCH, k_top * mi],
            DType::F32
        );
        alloc!(pb_moe_cur_moe, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(pb_moe_cur_mlp, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(pb_residual, &[MAX_PREFILL_BATCH, dim], DType::F32);
        alloc!(pb_tmp, &[MAX_PREFILL_BATCH, dim], DType::F32);
        let q_dim_max = config.n_heads * config.sliding_head_dim.max(config.full_head_dim);
        let kv_dim_max = (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim);
        alloc!(pb_q, &[MAX_PREFILL_BATCH, q_dim_max], DType::F32);
        alloc!(pb_attn_q, &[MAX_PREFILL_BATCH, q_dim_max], DType::F32);
        alloc!(
            pb_flash_partials,
            &[MAX_PREFILL_BATCH * flash_partials_sz],
            DType::F32
        );
        alloc!(pb_k, &[MAX_PREFILL_BATCH, kv_dim_max], DType::F32);
        alloc!(pb_v, &[MAX_PREFILL_BATCH, kv_dim_max], DType::F32);
        alloc!(pb_gate, &[MAX_PREFILL_BATCH, config.hidden_dim], DType::F32);
        alloc!(pb_up, &[MAX_PREFILL_BATCH, config.hidden_dim], DType::F32);
        alloc!(
            pb_ffn_hidden,
            &[MAX_PREFILL_BATCH, config.hidden_dim],
            DType::F32
        );
        alloc!(pb_positions, &[MAX_PREFILL_BATCH], DType::F32);
        let max_k_bf16 = config.dim.max(config.hidden_dim);
        alloc!(pb_bf16, &[MAX_PREFILL_BATCH * max_k_bf16], DType::BF16);

        let scratch = Gemma4Scratch {
            x: txn.take_tensor(x),
            residual: txn.take_tensor(residual),
            tmp: txn.take_tensor(tmp),
            pos_buf: txn.take_buffer(pos_buf),
            q: txn.take_tensor(q),
            k: txn.take_tensor(k),
            v: txn.take_tensor(v),
            attn_out: txn.take_tensor(attn_out),
            gate_ffn: txn.take_tensor(gate_ffn),
            up_ffn: txn.take_tensor(up_ffn),
            ffn_hidden: txn.take_tensor(ffn_hidden),
            ffn_out: txn.take_tensor(ffn_out),
            logits: txn.take_tensor(logits),
            sample_buf: txn.take_tensor(sample_buf),
            repeat_buf: txn.take_tensor(repeat_buf),
            flash_partials: txn.take_tensor(flash_partials),
            v_norm_ones_full: txn.take_tensor(v_norm_ones_full),
            moe_cur_mlp: txn.take_tensor(moe_cur_mlp),
            moe_pre2: txn.take_tensor(moe_pre2),
            moe_router_in: txn.take_tensor(moe_router_in),
            moe_router_logits: txn.take_tensor(moe_router_logits),
            moe_topk_indices: txn.take_tensor(moe_topk_indices),
            moe_topk_weights: txn.take_tensor(moe_topk_weights),
            moe_cur_moe: txn.take_tensor(moe_cur_moe),
            moe_expert_gate_up: txn.take_tensor(moe_expert_gate_up),
            moe_expert_hidden: txn.take_tensor(moe_expert_hidden),
            moe_expert_out: txn.take_tensor(moe_expert_out),
            moe_pre2_rot: txn.take_tensor(moe_pre2_rot),
            moe_expert_gate_batch: txn.take_tensor(moe_expert_gate_batch),
            moe_expert_up_batch: txn.take_tensor(moe_expert_up_batch),
            moe_expert_hidden_batch: txn.take_tensor(moe_expert_hidden_batch),
            max_prefill_batch: MAX_PREFILL_BATCH,
            pb_attn_out: txn.take_tensor(pb_attn_out),
            pb_ffn_out: txn.take_tensor(pb_ffn_out),
            pb_moe_pre2: txn.take_tensor(pb_moe_pre2),
            pb_moe_pre2_rot: txn.take_tensor(pb_moe_pre2_rot),
            pb_moe_router_in: txn.take_tensor(pb_moe_router_in),
            pb_moe_router_logits: txn.take_tensor(pb_moe_router_logits),
            pb_moe_topk_indices: txn.take_tensor(pb_moe_topk_indices),
            pb_moe_topk_weights: txn.take_tensor(pb_moe_topk_weights),
            pb_moe_expert_offsets: txn.take_tensor(pb_moe_expert_offsets),
            pb_moe_expert_token_list: txn.take_tensor(pb_moe_expert_token_list),
            pb_moe_gate_batch: txn.take_tensor(pb_moe_gate_batch),
            pb_moe_up_batch: txn.take_tensor(pb_moe_up_batch),
            pb_moe_hidden_batch: txn.take_tensor(pb_moe_hidden_batch),
            pb_moe_cur_moe: txn.take_tensor(pb_moe_cur_moe),
            pb_moe_cur_mlp: txn.take_tensor(pb_moe_cur_mlp),
            pb_residual: txn.take_tensor(pb_residual),
            pb_tmp: txn.take_tensor(pb_tmp),
            pb_q: txn.take_tensor(pb_q),
            pb_attn_q: txn.take_tensor(pb_attn_q),
            pb_flash_partials: txn.take_tensor(pb_flash_partials),
            pb_k: txn.take_tensor(pb_k),
            pb_v: txn.take_tensor(pb_v),
            pb_gate: txn.take_tensor(pb_gate),
            pb_up: txn.take_tensor(pb_up),
            pb_ffn_hidden: txn.take_tensor(pb_ffn_hidden),
            pb_positions: txn.take_tensor(pb_positions),
            pb_bf16: txn.take_tensor(pb_bf16),
        };
        txn.commit();
        Ok(scratch)
    }

    /// Release every GPU allocation owned by this scratch. Mirrors the
    /// Qwen35Scratch / LlamaScratch pattern so `unload_model` in the daemon
    /// can reclaim VRAM on idle eviction.
    /// Sum actual capacities of all scratch owners, including the raw device
    /// position buffer. This excludes no scratch allocation and performs no
    /// logical-shape estimation.
    pub fn owner_bytes(&self) -> usize {
        [
            &self.x,
            &self.residual,
            &self.tmp,
            &self.q,
            &self.k,
            &self.v,
            &self.attn_out,
            &self.gate_ffn,
            &self.up_ffn,
            &self.ffn_hidden,
            &self.ffn_out,
            &self.logits,
            &self.sample_buf,
            &self.repeat_buf,
            &self.flash_partials,
            &self.v_norm_ones_full,
            &self.moe_cur_mlp,
            &self.moe_pre2,
            &self.moe_router_in,
            &self.moe_router_logits,
            &self.moe_topk_indices,
            &self.moe_topk_weights,
            &self.moe_cur_moe,
            &self.moe_expert_gate_up,
            &self.moe_expert_hidden,
            &self.moe_expert_out,
            &self.moe_pre2_rot,
            &self.moe_expert_gate_batch,
            &self.moe_expert_up_batch,
            &self.moe_expert_hidden_batch,
            &self.pb_attn_out,
            &self.pb_ffn_out,
            &self.pb_moe_pre2,
            &self.pb_moe_pre2_rot,
            &self.pb_moe_router_in,
            &self.pb_moe_router_logits,
            &self.pb_moe_topk_indices,
            &self.pb_moe_topk_weights,
            &self.pb_moe_expert_offsets,
            &self.pb_moe_expert_token_list,
            &self.pb_moe_gate_batch,
            &self.pb_moe_up_batch,
            &self.pb_moe_hidden_batch,
            &self.pb_moe_cur_moe,
            &self.pb_moe_cur_mlp,
            &self.pb_residual,
            &self.pb_tmp,
            &self.pb_q,
            &self.pb_attn_q,
            &self.pb_flash_partials,
            &self.pb_k,
            &self.pb_v,
            &self.pb_gate,
            &self.pb_up,
            &self.pb_ffn_hidden,
            &self.pb_positions,
            &self.pb_bf16,
        ]
        .into_iter()
        .map(|tensor| tensor_owner_bytes(tensor))
        .sum::<usize>()
            + device_buffer_owner_bytes(&self.pos_buf)
    }

    /// Release every GPU allocation owned by this scratch. The position
    /// buffer is a raw HIP allocation and must be freed explicitly; it has no
    /// Drop implementation.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let owner_bytes = self.owner_bytes();
        let Gemma4Scratch {
            x,
            residual,
            tmp,
            pos_buf,
            q,
            k,
            v,
            attn_out,
            gate_ffn,
            up_ffn,
            ffn_hidden,
            ffn_out,
            logits,
            sample_buf,
            repeat_buf,
            flash_partials,
            v_norm_ones_full,
            moe_cur_mlp,
            moe_pre2,
            moe_router_in,
            moe_router_logits,
            moe_topk_indices,
            moe_topk_weights,
            moe_cur_moe,
            moe_expert_gate_up,
            moe_expert_hidden,
            moe_expert_out,
            moe_pre2_rot,
            moe_expert_gate_batch,
            moe_expert_up_batch,
            moe_expert_hidden_batch,
            max_prefill_batch: _,
            pb_attn_out,
            pb_ffn_out,
            pb_moe_pre2,
            pb_moe_pre2_rot,
            pb_moe_router_in,
            pb_moe_router_logits,
            pb_moe_topk_indices,
            pb_moe_topk_weights,
            pb_moe_expert_offsets,
            pb_moe_expert_token_list,
            pb_moe_gate_batch,
            pb_moe_up_batch,
            pb_moe_hidden_batch,
            pb_moe_cur_moe,
            pb_moe_cur_mlp,
            pb_residual,
            pb_tmp,
            pb_q,
            pb_attn_q,
            pb_flash_partials,
            pb_k,
            pb_v,
            pb_gate,
            pb_up,
            pb_ffn_hidden,
            pb_positions,
            pb_bf16,
        } = self;

        // Reverse construction order, with the raw position buffer released
        // at its construction boundary between `tmp` and `q`.
        for tensor in [
            pb_bf16,
            pb_positions,
            pb_ffn_hidden,
            pb_up,
            pb_gate,
            pb_v,
            pb_k,
            pb_flash_partials,
            pb_attn_q,
            pb_q,
            pb_tmp,
            pb_residual,
            pb_moe_cur_mlp,
            pb_moe_cur_moe,
            pb_moe_hidden_batch,
            pb_moe_up_batch,
            pb_moe_gate_batch,
            pb_moe_expert_token_list,
            pb_moe_expert_offsets,
            pb_moe_topk_weights,
            pb_moe_topk_indices,
            pb_moe_router_logits,
            pb_moe_router_in,
            pb_moe_pre2_rot,
            pb_moe_pre2,
            pb_ffn_out,
            pb_attn_out,
            moe_expert_hidden_batch,
            moe_expert_up_batch,
            moe_expert_gate_batch,
            moe_pre2_rot,
            moe_expert_out,
            moe_expert_hidden,
            moe_expert_gate_up,
            moe_cur_moe,
            moe_topk_weights,
            moe_topk_indices,
            moe_router_logits,
            moe_router_in,
            moe_pre2,
            moe_cur_mlp,
            v_norm_ones_full,
            flash_partials,
            repeat_buf,
            sample_buf,
            logits,
            ffn_out,
            ffn_hidden,
            up_ffn,
            gate_ffn,
            attn_out,
            v,
            k,
            q,
        ] {
            let _ = gpu.free_tensor(tensor);
        }
        let _ = gpu.hip.free(pos_buf);
        let _ = gpu.free_tensor(tmp);
        let _ = gpu.free_tensor(residual);
        let _ = gpu.free_tensor(x);
        unregister_live_owner_bytes(owner_bytes);
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────

/// Apply the Gemma 4 MoE parallel branch (26B-A4B). Called from each layer
/// AFTER `down_proj` produces `scratch.ffn_out`, REPLACING the standalone
/// `post_feedforward_layernorm` call. On exit, `scratch.tmp` holds the
/// combined `post_norm(cur_mlp + cur_moe)`, ready for `x = residual + tmp`.
///
/// Legacy serialized path only (8 experts × 5 launches = 40 launches/layer).
/// The fused indexed-GEMV path (`gemv_hfq4g256_moe_gate_up_k8_indexed`) and
/// fused-down path from origin/gemma4 are NOT yet ported — they require a
/// `rotate_x_mq` + `mq_signs` plumbing the modular crate doesn't have yet.
/// Both produce mathematically identical output; the legacy path is the
/// safety/reference baseline.
///
/// HF reference (modeling_gemma4.py Gemma4MoeBlock + Gemma4MoeMLP):
///   cur_mlp = post_feedforward_layernorm_1(ffn_out)        # standard SwiGLU out, normed
///   pre2    = pre_feedforward_layernorm_2(attn_out)        # MoE branch input
///   router_in    = rmsnorm(attn_out, router_scale) / sqrt(dim)
///   router_logits = router_proj @ router_in
///   topk_idx, topk_w = softmax_topk_renorm(router_logits, k=8)
///   cur_moe = sum_k [ topk_w[k] * per_expert_scale[i_k] *
///                     down_proj_{i_k}( gelu_tanh(gate) * up
///                                      where (gate, up) = split(gate_up_proj_{i_k} @ pre2) ) ]
///   cur_moe = post_feedforward_layernorm_2(cur_moe)
///   tmp     = post_feedforward_layernorm(cur_mlp + cur_moe)
///
/// `attn_out` parameter is the layer's post-attention residual stream
/// (= `scratch.residual` at the call site, since the caller stored
/// `residual = x` after the attention sandwich).
fn apply_moe_branch(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    moe: &MoeLayerExtras,
    post_ffn_norm: &GpuTensor,
    attn_out: &GpuTensor,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let n_exp = config.num_experts;
    let k_top = config.top_k_experts;
    if k_top != 8 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("MoE top_k_experts={k_top} unsupported (kernel hardcoded to 8)"),
        ));
    }

    // 1) cur_mlp = post_feedforward_layernorm_1(ffn_out)
    gpu.rmsnorm_f32(
        &scratch.ffn_out,
        &moe.post_feedforward_layernorm_1,
        &scratch.moe_cur_mlp,
        config.norm_eps,
    )?;

    // 2) pre2 = pre_feedforward_layernorm_2(attn_out)
    gpu.rmsnorm_f32(
        attn_out,
        &moe.pre_feedforward_layernorm_2,
        &scratch.moe_pre2,
        config.norm_eps,
    )?;

    // Dump moe_pre2 on first call
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static PRE2_DUMP: AtomicUsize = AtomicUsize::new(0);
        let pc = PRE2_DUMP.fetch_add(1, Ordering::Relaxed);
        if pc == 0 {
            if let Ok(data) = gpu.download_f32(&scratch.moe_pre2) {
                let sum: f64 = data.iter().map(|&v| v as f64).sum();
                eprintln!(
                    "[moe pre2] first4={:?} sum={sum:.4}",
                    &data[..4.min(data.len())]
                );
            }
        }
    }
    // 3) Router input: rmsnorm(attn_out, router_scale) / sqrt(dim).
    //    Equivalent to ref `rms_norm(x) * router_scale / sqrt(dim)` since
    //    rmsnorm_f32(x, w) = w * x / sqrt(mean(x²) + eps) — elementwise commutative.
    gpu.rmsnorm_f32(
        attn_out,
        &moe.router_scale,
        &scratch.moe_router_in,
        config.norm_eps,
    )?;
    gpu.scale_f32(&scratch.moe_router_in, 1.0 / (dim as f32).sqrt())?;

    // 4) Router GEMV → logits [n_exp]
    weight_gemv(
        gpu,
        &moe.router_proj,
        &scratch.moe_router_in,
        &scratch.moe_router_logits,
    )?;

    // 5) Top-K softmax + renorm on device. Kernel hardcoded to k_top=8.
    gpu.moe_softmax_topk_renorm_k8(
        &scratch.moe_router_logits,
        &scratch.moe_topk_indices,
        &scratch.moe_topk_weights,
        n_exp,
        true,
    )?;

    // 6) Dispatch top-K experts through the typed dispatch-owned backend.
    // The pointer tables are used by indexed kernels when available; the
    // pooled raw storage and byte strides retain the generic fallback.
    let experts = MoeGeluExpertsRef {
        gate_up_pool: &moe.experts_gate_up_pool,
        down_pool: &moe.experts_down_pool,
        gate_up_ptrs: &moe.experts_gate_up_ptrs,
        down_ptrs: &moe.experts_down_ptrs,
        gate_up_dtype: moe.experts[0].gate_up_proj.gpu_dtype,
        down_dtype: moe.experts[0].down_proj.gpu_dtype,
        gate_up_bytes: moe.gate_up_bytes,
        down_bytes: moe.down_bytes,
        n_experts: n_exp,
    };
    launch_moe_gelu_experts(
        gpu,
        &DispatchCtx::new(gpu),
        &experts,
        &scratch.moe_pre2,
        &scratch.moe_pre2_rot,
        &scratch.moe_topk_indices,
        &scratch.moe_topk_weights,
        &moe.per_expert_scale,
        &moe.per_expert_scale_host,
        &scratch.moe_expert_gate_batch,
        &scratch.moe_expert_up_batch,
        &scratch.moe_expert_hidden_batch,
        &scratch.moe_cur_moe,
        dim,
        mi,
        k_top,
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;

    // 9) cur_moe = post_feedforward_layernorm_2(cur_moe) — in-place
    // Dump MoE branch intermediates on first call
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static MOE_END: AtomicUsize = AtomicUsize::new(0);
        let mc = MOE_END.fetch_add(1, Ordering::Relaxed);
        if mc == 0 {
            if let Ok(mlp_data) = gpu.download_f32(&scratch.moe_cur_mlp) {
                let sum: f64 = mlp_data.iter().map(|&v| v as f64).sum();
                eprintln!(
                    "[moe branch] cur_mlp first4={:?} sum={sum:.4}",
                    &mlp_data[..4.min(mlp_data.len())]
                );
            }
            if let Ok(moe_data) = gpu.download_f32(&scratch.moe_cur_moe) {
                let sum: f64 = moe_data.iter().map(|&v| v as f64).sum();
                eprintln!(
                    "[moe branch] cur_moe (before norm2) first4={:?} sum={sum:.4}",
                    &moe_data[..4.min(moe_data.len())]
                );
            }
        }
    }
    gpu.rmsnorm_f32(
        &scratch.moe_cur_moe,
        &moe.post_feedforward_layernorm_2,
        &scratch.moe_cur_moe,
        config.norm_eps,
    )?;

    // 10) combined = cur_mlp + cur_moe → scratch.tmp
    gpu.add_f32(&scratch.moe_cur_mlp, &scratch.moe_cur_moe, &scratch.tmp)?;

    // 11) tmp = post_feedforward_layernorm(combined)
    gpu.rmsnorm_f32(&scratch.tmp, post_ffn_norm, &scratch.tmp, config.norm_eps)?;

    Ok(())
}

/// Batched MoE branch — processes N tokens at once. Mirrors `apply_moe_branch`
/// but every per-token tensor becomes a `[N × *]` row-major batch and every
/// kernel call uses the `_batched` variant.
///
/// Inputs (live in `scratch.pb_*`):
///   - pb_attn_out[N × dim]: post-attention output (input to pre_norm_2 and router)
///   - pb_ffn_out [N × dim]: dense FFN output  (input to post_norm_1)
///
/// Output:
///   - pb_moe_pre2[N × dim] holds the post-FFN-layernorm(cur_mlp + cur_moe) —
///     i.e. what the per-token path writes to `scratch.tmp`. Caller adds it
///     to the per-token residual + applies the layer scalar.
///
/// Only the indexed-fast path (MQ4G256 gate_up + HFQ4G128/Q8_0 down) is wired.
/// Falls back to per-token calls when the quant mix doesn't match (slow but
/// correct).
fn apply_moe_branch_batched(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    moe: &MoeLayerExtras,
    post_ffn_norm: &GpuTensor,
    n_batch: usize,
) -> HipResult<()> {
    let dim = config.dim;
    let mi = config.moe_intermediate_size;
    let n_exp = config.num_experts;
    let k_top = config.top_k_experts;
    if k_top != 8 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("MoE top_k_experts={k_top} unsupported (kernel hardcoded to 8)"),
        ));
    }
    debug_assert!(
        n_batch <= scratch.max_prefill_batch,
        "n_batch={n_batch} > MAX_PREFILL_BATCH={}",
        scratch.max_prefill_batch
    );

    let first = &moe.experts[0];
    let _gate_dtype = first.gate_up_proj.gpu_dtype;
    let _down_dtype = first.down_proj.gpu_dtype;
    // TODO(Phase 4): fused batched MoE kernels not yet ported.
    // Using per-token expert loop (correct but slow for prefill).

    // 1) cur_mlp_batch = post_feedforward_layernorm_1(pb_ffn_out)
    gpu.rmsnorm_batched(
        &scratch.pb_ffn_out,
        &moe.post_feedforward_layernorm_1,
        &scratch.pb_moe_cur_mlp,
        n_batch,
        dim,
        config.norm_eps,
    )?;

    // 2) pre2_batch = pre_feedforward_layernorm_2(pb_attn_out)
    gpu.rmsnorm_batched(
        &scratch.pb_attn_out,
        &moe.pre_feedforward_layernorm_2,
        &scratch.pb_moe_pre2,
        n_batch,
        dim,
        config.norm_eps,
    )?;

    // 3) router_in = rmsnorm(attn_out, router_scale) / sqrt(dim)
    gpu.rmsnorm_batched(
        &scratch.pb_attn_out,
        &moe.router_scale,
        &scratch.pb_moe_router_in,
        n_batch,
        dim,
        config.norm_eps,
    )?;
    gpu.scale_f32(&scratch.pb_moe_router_in, 1.0 / (dim as f32).sqrt())?;
    // scale_f32 operates on the full tensor — for [N × dim] this scales
    // every element, which is what we want (each row is divided by sqrt(dim)).

    // 4) Router GEMM → logits [N × n_exp]
    run_prefill_gemm(
        gpu,
        &moe.router_proj,
        &scratch.pb_moe_router_in,
        &scratch.pb_moe_router_logits,
        n_batch,
        Some(&scratch.pb_bf16),
    )?;

    // 5) Batched top-K softmax + renorm.
    gpu.moe_softmax_topk_renorm_k8_batched(
        &scratch.pb_moe_router_logits,
        &scratch.pb_moe_topk_indices,
        &scratch.pb_moe_topk_weights,
        n_exp,
        true,
        n_batch,
    )?;

    // 5b) Phase B (opt-in): build per-expert buckets so the bucketed GEMV
    // can reuse the weight tile across all tokens routed to the same expert.
    // Measured -5.7% prefill regression on Gemma 4 26B-A4B-it / gfx1201 at
    // N=128 batch (132 → 124 tok/s): the bucketed kernel pre-loads the
    // weight tile into ~48 VGPRs per lane (sc/zp/pk × 16 groups) which
    // cuts occupancy below the indexed_batched kernel that streams weights
    // in the inner loop, and the serialized loop over bucket tokens loses
    // more parallelism than is recovered from launch-overhead savings
    // (180k blocks vs 1.4M). Default OFF; kept behind opt-in for future
    // tuning (smaller MAX_GROUPS, LDS staging, fewer launch_bounds waves).
    let use_bucketed = std::env::var("HIPFIRE_MOE_BUCKETED")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    if use_bucketed {
        gpu.moe_bucket_build(
            &scratch.pb_moe_topk_indices,
            &scratch.pb_moe_expert_offsets,
            &scratch.pb_moe_expert_token_list,
            n_batch,
            k_top,
            n_exp,
        )?;
    }

    // 6-13) Per-token expert loop (Phase 4 fallback until fused kernels ported).
    // For each token: extract per-token topk indices/weights,
    // run 8 expert GEMVs, accumulate into pb_moe_cur_moe.
    let dim_bytes = dim * 4;
    let topk_idx_host = gpu.download_f32(&scratch.pb_moe_topk_indices)?;
    let topk_wt_host = gpu.download_f32(&scratch.pb_moe_topk_weights)?;
    let topk_indices_batch: Vec<Vec<usize>> = (0..n_batch)
        .map(|b| {
            unsafe {
                std::slice::from_raw_parts(
                    topk_idx_host.as_ptr().add(b * k_top) as *const i32,
                    k_top,
                )
            }
            .iter()
            .map(|&i| i as usize)
            .collect()
        })
        .collect();
    let topk_weights_batch: Vec<Vec<f32>> = (0..n_batch)
        .map(|b| topk_wt_host[b * k_top..(b + 1) * k_top].to_vec())
        .collect();

    // Zero cur_moe_batch accumulator.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memset_async(&scratch.pb_moe_cur_moe.buf, 0, n_batch * dim_bytes, s)?;
    } else {
        gpu.hip
            .memset(&scratch.pb_moe_cur_moe.buf, 0, n_batch * dim_bytes)?;
    }

    for b in 0..n_batch {
        for ki in 0..k_top {
            let e = topk_indices_batch[b][ki];
            let weight = topk_weights_batch[b][ki] * moe.per_expert_scale_host[e];
            let expert = &moe.experts[e];

            // Copy this token's pre2 row into scratch.moe_pre2
            if let Some(s) = gpu.active_stream.as_ref() {
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.moe_pre2.buf,
                    0,
                    &scratch.pb_moe_pre2.buf,
                    b * dim_bytes,
                    dim_bytes,
                    s,
                )?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.moe_pre2.buf,
                    0,
                    &scratch.pb_moe_pre2.buf,
                    b * dim_bytes,
                    dim_bytes,
                )?;
            }

            // gate_up = expert.gate_up_proj @ pre2
            weight_gemv(
                gpu,
                &expert.gate_up_proj,
                &scratch.moe_pre2,
                &scratch.moe_expert_gate_up,
            )?;
            let gate = scratch.moe_expert_gate_up.sub_offset(0, mi);
            let up = scratch.moe_expert_gate_up.sub_offset(mi, mi);
            // hidden = gelu_tanh(gate) * up
            gpu.gelu_tanh_f32(&gate, &scratch.moe_expert_hidden, mi)?;
            gpu.mul_f32(&scratch.moe_expert_hidden, &up, &scratch.moe_expert_hidden)?;
            // expert_out = expert.down_proj @ hidden
            weight_gemv(
                gpu,
                &expert.down_proj,
                &scratch.moe_expert_hidden,
                &scratch.moe_expert_out,
            )?;
            // scaled_add into the correct row of pb_moe_cur_moe
            if let Some(s) = gpu.active_stream.as_ref() {
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.tmp.buf,
                    0,
                    &scratch.pb_moe_cur_moe.buf,
                    b * dim_bytes,
                    dim_bytes,
                    s,
                )?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.tmp.buf,
                    0,
                    &scratch.pb_moe_cur_moe.buf,
                    b * dim_bytes,
                    dim_bytes,
                )?;
            }
            gpu.scaled_add_inplace_cpu_scalar_f32(&scratch.tmp, &scratch.moe_expert_out, weight)?;
            if let Some(s) = gpu.active_stream.as_ref() {
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.pb_moe_cur_moe.buf,
                    b * dim_bytes,
                    &scratch.tmp.buf,
                    0,
                    dim_bytes,
                    s,
                )?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.pb_moe_cur_moe.buf,
                    b * dim_bytes,
                    &scratch.tmp.buf,
                    0,
                    dim_bytes,
                )?;
            }
        }
    }

    // 11) post_feedforward_layernorm_2(cur_moe) in-place batched.
    gpu.rmsnorm_batched(
        &scratch.pb_moe_cur_moe,
        &moe.post_feedforward_layernorm_2,
        &scratch.pb_moe_cur_moe,
        n_batch,
        dim,
        config.norm_eps,
    )?;

    // 12) combined = cur_mlp + cur_moe → reuse pb_moe_pre2 as the combined buffer.
    gpu.add_f32(
        &scratch.pb_moe_cur_mlp,
        &scratch.pb_moe_cur_moe,
        &scratch.pb_moe_pre2,
    )?;

    // 13) post_feedforward_layernorm(combined) → batched, in-place.
    gpu.rmsnorm_batched(
        &scratch.pb_moe_pre2,
        post_ffn_norm,
        &scratch.pb_moe_pre2,
        n_batch,
        dim,
        config.norm_eps,
    )?;

    Ok(())
}

/// Single-token decode. Phase 3 implementation.
///
/// Precondition: `scratch.sliding_cos/sin` + `scratch.full_cos/sin` +
/// `scratch.v_norm_ones_full` must be populated by the loader before the
/// first forward call (one-time init).
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    token: u32,
    pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;
    let mut layer_dump = Gemma4LayerDump::for_position(pos);

    // 1) Embedding lookup + sqrt(dim) scale.
    // ALWAYS direct — the captured graph can't bake in `token` (varies per
    // call). The embed lookup fills scratch.x; the rest of the forward
    // (which is graph-capturable now that the MoE branch has no D2H syncs)
    // reads from scratch.x. Same split Qwen35 uses for its captured path.
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, token, dim)?
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, token, dim)?
        }
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, token, dim)?
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, token, dim)?
        }
        _ => {
            return Err(hip_bridge::HipError::new(
                0,
                "unsupported Gemma 4 embed format",
            ))
        }
    }
    gpu.scale_f32(&scratch.x, config.embed_scale)?;
    if let Some(dump) = layer_dump.as_mut() {
        dump.capture(gpu, "embedding_last", &scratch.x);
    }

    // hipGraph capture/replay policy.
    //   - DEFAULT-OFF for Gemma 4 (until cross-arch / long-context validation).
    //   - Fixed 2026-05-19 (evening): the earlier diagnosis ("kv_len = pos + 1
    //     is a scalar arg baked at capture") was wrong — attention_flash_*_window
    //     kernels actually compute seq_len = pos_buf[0] + 1 at runtime
    //     (`attention_flash_asym3_tile.hip:47`). The real bugs were three
    //     elementwise kernels (`scale_f32`, `mul_f32`, `add_f32` in
    //     `crates/rdna-compute/src/dispatch.rs`) that used direct `launch_kernel`
    //     instead of `launch_maybe_blob`. `mul_f32` and `add_f32` additionally
    //     ran on the default stream (`None`) instead of `stream_ref()`, so
    //     during graph capture they were NOT recorded at all — every replay
    //     skipped the FFN's `ffn_hidden = gelu(gate) * up` multiply, feeding
    //     wrong tensors into down_proj → token attractor on greedy decode.
    //     `scale_f32` was on the capture stream but using raw `kernelParams`
    //     (stack pointers that dangle by replay under ROCm 7.x loader).
    //     All three converted to `launch_maybe_blob` in the same commit as
    //     this comment. HIPFIRE_GEMMA4_GRAPH=1 requests the graph path.
    //   - Compact offset != 0 (TriAttention eviction) still breaks capture
    //     for the same reason as Qwen35 — bail to direct in that case.
    static GRAPH_OVERRIDE_ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let graph_override = *GRAPH_OVERRIDE_ENV.get_or_init(|| {
        match std::env::var("HIPFIRE_GEMMA4_GRAPH").ok().as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        }
    });
    let use_graph = graph_override.unwrap_or(false)
        && layer_dump.is_none()
        && kv_sliding.compact_offset == 0
        && kv_full.compact_offset == 0;

    if use_graph && gpu.graphs.graph_exec.is_some() {
        // ── Replay path. Update pos_buf via stream_write_value32 (graph-
        //    replay-safe, no host→device copy). The captured graph reads
        //    pos_buf at kernel-launch time, so a fresh `pos` propagates
        //    without recapture. ──
        let stream = gpu.active_stream.as_ref().unwrap();
        gpu.hip
            .stream_write_value32(stream, &scratch.pos_buf, pos as u32, 0)?;
        gpu.graphs
            .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())?;
    } else if use_graph && gpu.graphs.graph_exec.is_none() {
        let pos_i32 = pos as i32;
        if !false
        /* graph-capture-not-wired */
        {
            // ── Warmup: direct dispatch so any JIT kernel compiles or lazy
            //    scratch allocations happen OUTSIDE a capture region.
            //    Capturing on the first call hits "hipMalloc not permitted
            //    under stream capture". Same pattern as Qwen35. ──
            /* graph-capture-not-wired */
            gpu.hip
                .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            forward_scratch_inner(
                gpu,
                weights,
                config,
                pos,
                kv_sliding,
                kv_full,
                scratch,
                layer_dump.as_mut(),
            )?;
        } else {
            // ── First post-warmup call: capture the forward into a graph. ──
            if gpu.active_stream.is_none() {
                gpu.active_stream = Some(gpu.hip.stream_create()?);
            }
            gpu.hip
                .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            gpu.graphs.begin_graph_capture(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
            )?;
            forward_scratch_inner(
                gpu,
                weights,
                config,
                pos,
                kv_sliding,
                kv_full,
                scratch,
                layer_dump.as_mut(),
            )?;
            gpu.graphs.end_graph_capture(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
            )?;
            // hipStreamCaptureModeGlobal RECORDS — kernels don't execute
            // during capture. Replay once so THIS pos's forward actually
            // runs (KV write, logits update).
            gpu.graphs.graph_launch(
                &gpu.hip,
                gpu.device_id,
                gpu.active_stream.as_ref().unwrap(),
            )?;
            eprintln!(
                "[gemma4 hipGraph] captured {} blobs, instantiated",
                gpu.graphs.capture_blobs.len()
            );
        }
    } else {
        // ── Direct path (no graph) ──
        let pos_i32 = pos as i32;
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
        forward_scratch_inner(
            gpu,
            weights,
            config,
            pos,
            kv_sliding,
            kv_full,
            scratch,
            layer_dump.as_mut(),
        )?;
    }
    if let Some(dump) = layer_dump {
        dump.write(gpu, &scratch.logits);
    }
    Ok(())
}

/// Single sliding-window attention layer.
///
/// Order matches HF modeling_gemma4.py::Gemma4TextDecoderLayer +
/// Gemma4TextAttention (sliding branch):
///   residual = x
///   x = input_layernorm(x)              — RMSNorm (sandwich pre-attn)
///   q = q_proj(x); q = q_norm(q)        — RMSNorm over head_dim=256
///   k = k_proj(x); k = k_norm(k)
///   v = v_proj(x); v = v_norm(v)         — no_scale RMSNorm (ones buffer)
///   RoPE(q, k) with rotate_half, theta=10000, full head_dim=256
///   write K, V to KV cache at position `pos`
///   attn = flash_attention(q, K, V, window_size=1024, scale=1.0 effective)
///   x = o_proj(attn)
///   x = post_attention_layernorm(x)     — RMSNorm (sandwich post-attn)
///   x = residual + x
///   residual = x
///   x = pre_feedforward_layernorm(x)    — RMSNorm (sandwich pre-FFN)
///   gate = gate_proj(x); up = up_proj(x)
///   ffn = gelu_pytorch_tanh(gate) * up  — SwiGLU
///   x = down_proj(ffn)
///   x = post_feedforward_layernorm(x)   — RMSNorm (sandwich post-FFN)
///   x = residual + x
///   x = x * layer_scalar                — learned per-layer scalar
///
/// Gemma 4 attention uses `scaling=1.0` in HF (see modeling_gemma4.py line 1143).
/// Our flash kernels bake in `scale = 1/sqrt(head_dim)`; we compensate by
/// pre-scaling Q by sqrt(head_dim) so the effective scale is 1.0.
// Sliding-window kernel correctness: the 4 attention_flash_*_window dispatch
// sites here use the kernels modified by commit b608c5f (sliding-window arg
// threaded through 7 attention_flash_*.hip files). Static review of the diff
// is clean — per-position out_of_window predicate writes -1e30 to scores;
// early-tile-skip writes {-1e30, 0, zeros} so reduce sees no contribution;
// all 32 lanes participate in __shfl_xor; window_size=0 = byte-identical
// for Qwen 3.5/3.6. Runtime NRMSE validation: verify_against_torch.rs Phase 2
// (D2.5) — per-layer attention NRMSE against the bf16 PyTorch reference dump.
fn sliding_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    sliding_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, false)
}

/// As `sliding_layer_decode` but with `stop_before_moe=true`, the function
/// returns immediately after the dense FFN computes `scratch.ffn_out`,
/// leaving `scratch.residual` holding the post-attention x. Used by
/// `forward_prefill_chunk` to interleave per-token attention with a
/// batched MoE call across all N tokens.
fn sliding_layer_attn_ffn_only(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    sliding_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, true)
}

fn sliding_layer_decode_impl(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &SlidingLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
    stop_before_moe: bool,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.sliding_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.sliding_n_kv_heads;
    let dim_bytes = dim * 4;

    // residual = x
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }
    let _dump_on = std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1")
        && (pos == 0 || pos == 1)
        && kv_layer_idx == 0;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 input scratch.x", &scratch.x, dim);
    }
    let ctx = DispatchCtx::new(gpu);

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(
        &scratch.x,
        &lw.input_layernorm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after input_norm", &scratch.tmp, dim);
    }

    // Q/K/V projections: q[n_heads*head_dim], k/v[n_kv*head_dim].
    let wr_q = lw.q_proj.dispatch_ref();
    let wr_k = lw.k_proj.dispatch_ref();
    execute_steps(
        gpu,
        &ctx,
        &[
            Step::Gemv {
                w: &wr_q,
                input: GemvInput::Raw(&scratch.tmp),
                out: &scratch.q,
            },
            Step::Gemv {
                w: &wr_k,
                input: GemvInput::Raw(&scratch.tmp),
                out: &scratch.k,
            },
        ],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    weight_gemv(gpu, &lw.v_proj, &scratch.tmp, &scratch.v)?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after q_proj", &scratch.q, n_heads * head_dim);
        dbg_dump(gpu, "[v1] L0 after k_proj", &scratch.k, n_kv * head_dim);
        dbg_dump(gpu, "[v1] L0 after v_proj", &scratch.v, n_kv * head_dim);
    }

    // q_norm + k_norm + no-scale v_norm across head_dim (in-place).
    // v_norm matches HF Gemma 4 `value_states = v_norm(v_proj(x))` (no_scale=True
    // RMSNorm — divide-only, ones buffer as weight). Same pattern as full_layer_decode.
    // Omitting v_norm compounds attention-output bias across 48 sliding layers and
    // produces single-token garbage end-to-end while still passing Phase 2 kernel
    // NRMSE (which tests q_norm + k_norm but not v_norm — see d2.5-results.md).
    gpu.rmsnorm_batched(
        &scratch.q,
        &lw.q_norm,
        &scratch.q,
        n_heads,
        head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &scratch.k,
        &lw.k_norm,
        &scratch.k,
        n_kv,
        head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &scratch.v,
        &scratch.v_norm_ones_full,
        &scratch.v,
        n_kv,
        head_dim,
        config.norm_eps,
    )?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after q_norm", &scratch.q, n_heads * head_dim);
        dbg_dump(gpu, "[v1] L0 after k_norm", &scratch.k, n_kv * head_dim);
        dbg_dump(gpu, "[v1] L0 after v_norm", &scratch.v, n_kv * head_dim);
    }

    // Pre-scale Q by sqrt(head_dim) so the flash-attn kernel's internal
    // 1/sqrt(head_dim) cancels, leaving the effective Gemma 4 scale of 1.0.
    // Only the first n_heads*head_dim elements of scratch.q are live.
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after scale_q", &scratch.q, n_heads * head_dim);
    }

    // Full rotate_half RoPE, theta=10000, head_dim=256 (all dims rotate).
    gpu.rope_f32(
        &scratch.q,
        &scratch.k,
        &scratch.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        config.sliding_rope_theta,
    )?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after rope_q", &scratch.q, n_heads * head_dim);
        dbg_dump(gpu, "[v1] L0 after rope_k", &scratch.k, n_kv * head_dim);
        // Dump raw Q/K/V after RoPE for score analysis
        let q_data = gpu.download_f32(&scratch.q).unwrap_or_default();
        let k_data = gpu.download_f32(&scratch.k).unwrap_or_default();
        let v_data = gpu.download_f32(&scratch.v).unwrap_or_default();
        let _ = std::fs::write("/tmp/gemma4_q_rope.bin", unsafe {
            std::slice::from_raw_parts(q_data.as_ptr() as *const u8, q_data.len() * 4)
        });
        let _ = std::fs::write("/tmp/gemma4_k_rope.bin", unsafe {
            std::slice::from_raw_parts(k_data.as_ptr() as *const u8, k_data.len() * 4)
        });
        let _ = std::fs::write("/tmp/gemma4_v.bin", unsafe {
            std::slice::from_raw_parts(v_data.as_ptr() as *const u8, v_data.len() * 4)
        });
        eprintln!(
            "[v1] Dumped post-RoPE Q({}), K({}), V({})",
            q_data.len(),
            k_data.len(),
            v_data.len()
        );
    }

    // KV cache write + flash attention via dispatch framework (Step::Attend).
    // flash_mode=2 (forced) because sliding layers always need flash for window masking.
    let sliding_cap = kv_cache.physical_cap as u32;
    {
        let tier_inputs = KvTierInputs {
            quant_asym4: kv_cache.quant_asym4,
            quant_asym3: kv_cache.quant_asym3,
            quant_asym2: kv_cache.quant_asym2,
            quant_q8: kv_cache.quant_q8,
            quant_fwht: kv_cache.quant_fwht,
            quant_hfq4: false,
            quant_q4: false,
            quant_int8: false,
            quant_hfq8: false,
            f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
            v_mode_bits: kv_cache.v_mode_bits(),
            pos,
            flash_mode: 2, // forced flash — sliding layers need window masking
            capture_mode: gpu.graphs.capture_mode,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            q8_windowed: false,
            window: config.sliding_window as i32,
        };
        let plan = KvTierPlan::derive(tier_inputs)
            .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
        let io = AttnParams {
            q: &scratch.q,
            k: &scratch.k,
            v: &scratch.v,
            k_cache: &kv_cache.k_gpu[kv_layer_idx],
            v_cache: &kv_cache.v_gpu[kv_layer_idx],
            k_scales: None,
            v_scales: None,
            pos_buf: &scratch.pos_buf,
            pos,
            positions: None,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            physical_cap: kv_cache.max_seq,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&scratch.flash_partials),
            givens_cos: kv_cache.givens_cos.as_ref(),
            givens_sin: kv_cache.givens_sin.as_ref(),
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &scratch.attn_out,
        };
        execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }
    // Dump attention output regardless of KV cache branch.
    if _dump_on {
        dbg_dump(
            gpu,
            "[v1] L0 after attention",
            &scratch.attn_out,
            n_heads * head_dim,
        );
        let data = gpu.download_f32(&scratch.attn_out).unwrap_or_default();
        let gqa_ratio = n_heads / n_kv;
        for h in 0..n_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            let h_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
            eprintln!(
                "[v1]   head {:2} (kv={}): sum={:+10.4}  first2=[{:.4}, {:.4}]",
                h,
                h / gqa_ratio,
                h_sum,
                data[start],
                data[start + 1]
            );
        }
    }

    // o_proj → tmp (reuse tmp, overwriting input_layernorm output).
    let wr_o = lw.o_proj.dispatch_ref();
    execute_steps(
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &wr_o,
            input: GemvInput::Raw(&scratch.attn_out),
            out: &scratch.tmp,
        }],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after o_proj", &scratch.tmp, dim);
    }

    // Sandwich post-attn norm (in-place on tmp).
    gpu.rmsnorm_f32(
        &scratch.tmp,
        &lw.post_attention_layernorm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after post_attn_norm", &scratch.tmp, dim);
    }

    // x = residual + tmp. (Reset x first since earlier ops mutated it.)
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after attn_residual", &scratch.x, dim);
    }

    // residual = x (for the FFN residual stream).
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }

    // Pre-FFN norm.
    gpu.rmsnorm_f32(
        &scratch.x,
        &lw.pre_feedforward_layernorm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after pre_ffn_norm", &scratch.tmp, dim);
    }

    // SwiGLU(gelu_pytorch_tanh): gate_proj, up_proj, gelu_tanh(gate) * up → down_proj.
    let wr_gate = lw.gate_proj.dispatch_ref();
    let wr_up = lw.up_proj.dispatch_ref();
    execute_steps(
        gpu,
        &ctx,
        &[
            Step::Gemv {
                w: &wr_gate,
                input: GemvInput::Raw(&scratch.tmp),
                out: &scratch.gate_ffn,
            },
            Step::Gemv {
                w: &wr_up,
                input: GemvInput::Raw(&scratch.tmp),
                out: &scratch.up_ffn,
            },
        ],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if _dump_on {
        dbg_dump(
            gpu,
            "[v1] L0 after gate_proj",
            &scratch.gate_ffn,
            config.hidden_dim,
        );
        dbg_dump(
            gpu,
            "[v1] L0 after up_proj",
            &scratch.up_ffn,
            config.hidden_dim,
        );
    }
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    if _dump_on {
        dbg_dump(
            gpu,
            "[v1] L0 after gelu*up",
            &scratch.ffn_hidden,
            config.hidden_dim,
        );
    }
    let wr_down = lw.down_proj.dispatch_ref();
    execute_steps(
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &wr_down,
            input: GemvInput::Raw(&scratch.ffn_hidden),
            out: &scratch.ffn_out,
        }],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if _dump_on {
        dbg_dump(gpu, "[v1] L0 after down_proj", &scratch.ffn_out, dim);
    }

    // Batched prefill hand-off point: caller wants to batch the MoE branch
    // across N tokens. Return now with scratch.ffn_out + scratch.residual
    // populated; caller assembles the batched MoE call externally.
    if stop_before_moe {
        return Ok(());
    }

    // Sandwich post-FFN norm. On MoE layers (26B-A4B) this is folded into
    // apply_moe_branch (which adds the parallel MoE branch + sandwich norms
    // 1 and 2 before this outer norm); on dense layers we just call the
    // standalone post_feedforward_layernorm.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(
            gpu,
            config,
            scratch,
            moe,
            &lw.post_feedforward_layernorm,
            &scratch.residual,
        )?,
        _ => gpu.rmsnorm_f32(
            &scratch.ffn_out,
            &lw.post_feedforward_layernorm,
            &scratch.tmp,
            config.norm_eps,
        )?,
    }

    // x = residual + tmp (again, reset x from saved residual).
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
}

/// Single full (global) attention layer with K=V weight sharing.
///
/// Key differences from sliding:
///   • head_dim = 512 (global_head_dim), 4 KV heads (vs sliding's 256 / 16).
///   • V is the *pre*-k_norm output of k_proj — CRITICAL ordering (line 1214
///     of modeling_gemma4.py). In Python:
///         key_states = k_proj(x)
///         value_states = v_proj(x) if v_proj else key_states   # bound BEFORE norm
///         key_states   = k_norm(key_states)                    # rebind, value_states holds pre-norm
///         value_states = v_norm(value_states)
///     Our translation: write k_proj output into `scratch.k`, memcpy into
///     `scratch.v`, then apply k_norm in-place on scratch.k.
///   • v_norm is `no_scale=true` RMSNorm — divide only, no learned gain.
///     We call the existing `rmsnorm_batched` with the ones-filled
///     `scratch.v_norm_ones_full` as the weight vector.
///   • RoPE is partial_rotary_factor=0.25 proportional:
///     pairs (i, i+head_dim/2) for i in [0, 64) rotate with theta=1e6;
///     pairs [64, 256) are NoPE (identity). See `rope_partial_halved_f32`.
///   • No sliding window (window_size=0 = full causal).
///   • Attention scale = 1.0 (same as sliding — Gemma 4 sets
///     `self.scaling = 1.0`; we compensate by pre-scaling Q by sqrt(head_dim)).
fn full_layer_decode(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    full_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, false)
}

fn full_layer_attn_ffn_only(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    full_layer_decode_impl(gpu, config, lw, pos, kv_cache, kv_layer_idx, scratch, true)
}

fn full_layer_decode_impl(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    lw: &FullLayerWeights,
    pos: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    kv_layer_idx: usize,
    scratch: &Gemma4Scratch,
    stop_before_moe: bool,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.full_head_dim;
    let n_heads = config.n_heads;
    let n_kv = config.full_n_kv_heads;
    let dim_bytes = dim * 4;
    let ctx = DispatchCtx::new(gpu);
    let kv_bytes = n_kv * head_dim * 4;

    // residual = x
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }

    // tmp = input_layernorm(x)
    gpu.rmsnorm_f32(
        &scratch.x,
        &lw.input_layernorm,
        &scratch.tmp,
        config.norm_eps,
    )?;

    let _fdump = std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1")
        && pos == 1
        && kv_layer_idx == 0;
    if _fdump {
        dbg_dump(gpu, "[FL] L5 input scratch.x", &scratch.x, dim);
    }
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_input_norm", &scratch.tmp, dim);
    }

    // Q + K projections. V is derived from K's pre-norm output below.
    weight_gemv(gpu, &lw.q_proj, &scratch.tmp, &scratch.q)?;
    weight_gemv(gpu, &lw.k_proj, &scratch.tmp, &scratch.k)?;
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_q_proj", &scratch.q, n_heads * head_dim);
    }
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_k_proj", &scratch.k, n_kv * head_dim);
    }

    // CRITICAL: capture pre-k_norm bytes as V before applying k_norm.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.v.buf, 0, &scratch.k.buf, 0, kv_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.v.buf, &scratch.k.buf, kv_bytes)?;
    }

    // q_norm, k_norm, and no-scale v_norm (all head_dim = 512).
    gpu.rmsnorm_batched(
        &scratch.q,
        &lw.q_norm,
        &scratch.q,
        n_heads,
        head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &scratch.k,
        &lw.k_norm,
        &scratch.k,
        n_kv,
        head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &scratch.v,
        &scratch.v_norm_ones_full,
        &scratch.v,
        n_kv,
        head_dim,
        config.norm_eps,
    )?;
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_q_norm", &scratch.q, n_heads * head_dim);
    }
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_k_norm", &scratch.k, n_kv * head_dim);
    }
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_v_norm", &scratch.v, n_kv * head_dim);
    }

    // Pre-scale Q by sqrt(head_dim=512) so the flash kernel's 1/sqrt(head_dim)
    // cancels (Gemma 4 attention scaling is 1.0).
    gpu.scale_f32(&scratch.q, (head_dim as f32).sqrt())?;

    // Proportional RoPE: rotate_half of the first 64 pairs of every 512-dim head.
    let n_rot_pairs = ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
    gpu.rope_partial_halved_f32(
        &scratch.q,
        &scratch.k,
        &scratch.pos_buf,
        n_heads,
        n_kv,
        head_dim,
        n_rot_pairs,
        config.full_rope_theta,
    )?;
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_rope_q", &scratch.q, n_heads * head_dim);
    }
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_rope_k", &scratch.k, n_kv * head_dim);
    }

    // KV cache write + attention via dispatch framework (Step::Attend).
    // Full-attention layers: window_size=0 (full causal), cache_capacity=0.
    {
        let tier_inputs = KvTierInputs {
            quant_asym4: kv_cache.quant_asym4,
            quant_asym3: kv_cache.quant_asym3,
            quant_asym2: kv_cache.quant_asym2,
            quant_q8: kv_cache.quant_q8,
            quant_fwht: kv_cache.quant_fwht,
            quant_hfq4: false,
            quant_q4: false,
            quant_int8: false,
            quant_hfq8: false,
            f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
            v_mode_bits: kv_cache.v_mode_bits(),
            pos,
            flash_mode: 2,
            capture_mode: gpu.graphs.capture_mode,
            batch_size: 1,
            is_tree: false,
            is_boundary: false,
            q8_windowed: false,
            window: 0,
        };
        let plan = KvTierPlan::derive(tier_inputs)
            .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
        let io = AttnParams {
            q: &scratch.q,
            k: &scratch.k,
            v: &scratch.v,
            k_cache: &kv_cache.k_gpu[kv_layer_idx],
            v_cache: &kv_cache.v_gpu[kv_layer_idx],
            k_scales: None,
            v_scales: None,
            pos_buf: &scratch.pos_buf,
            pos,
            positions: None,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            physical_cap: kv_cache.max_seq,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(&scratch.flash_partials),
            givens_cos: kv_cache.givens_cos.as_ref(),
            givens_sin: kv_cache.givens_sin.as_ref(),
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output: &scratch.attn_out,
        };
        execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }

    // o_proj → tmp.
    weight_gemv(gpu, &lw.o_proj, &scratch.attn_out, &scratch.tmp)?;
    if _fdump {
        dbg_dump(
            gpu,
            "[FL] L5 post_attn_out",
            &scratch.attn_out,
            n_heads * head_dim,
        );
    }
    if _fdump {
        dbg_dump(gpu, "[FL] L5 post_o_proj", &scratch.tmp, dim);
    }

    // Sandwich post-attn norm.
    gpu.rmsnorm_f32(
        &scratch.tmp,
        &lw.post_attention_layernorm,
        &scratch.tmp,
        config.norm_eps,
    )?;

    // x = residual + tmp.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Save new residual.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.residual.buf, 0, &scratch.x.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.residual.buf, &scratch.x.buf, dim_bytes)?;
    }

    // Pre-FFN norm.
    gpu.rmsnorm_f32(
        &scratch.x,
        &lw.pre_feedforward_layernorm,
        &scratch.tmp,
        config.norm_eps,
    )?;

    // SwiGLU with gelu_pytorch_tanh activation.
    weight_gemv(gpu, &lw.gate_proj, &scratch.tmp, &scratch.gate_ffn)?;
    weight_gemv(gpu, &lw.up_proj, &scratch.tmp, &scratch.up_ffn)?;
    gpu.gelu_tanh_f32(&scratch.gate_ffn, &scratch.ffn_hidden, config.hidden_dim)?;
    gpu.mul_f32(&scratch.ffn_hidden, &scratch.up_ffn, &scratch.ffn_hidden)?;
    weight_gemv(gpu, &lw.down_proj, &scratch.ffn_hidden, &scratch.ffn_out)?;

    // Batched prefill hand-off (see sliding_layer_decode_impl for rationale).
    if stop_before_moe {
        return Ok(());
    }

    // Sandwich post-FFN norm. Same MoE dispatch as sliding_layer_decode.
    let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
    match (lw.moe.as_ref(), moe_bypass) {
        (Some(moe), false) => apply_moe_branch(
            gpu,
            config,
            scratch,
            moe,
            &lw.post_feedforward_layernorm,
            &scratch.residual,
        )?,
        _ => gpu.rmsnorm_f32(
            &scratch.ffn_out,
            &lw.post_feedforward_layernorm,
            &scratch.tmp,
            config.norm_eps,
        )?,
    }

    // x = residual + tmp.
    if let Some(s) = gpu.active_stream.as_ref() {
        gpu.hip
            .memcpy_dtod_async_at(&scratch.x.buf, 0, &scratch.residual.buf, 0, dim_bytes, s)?;
    } else {
        gpu.hip
            .memcpy_dtod(&scratch.x.buf, &scratch.residual.buf, dim_bytes)?;
    }
    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;

    // Learned per-layer scalar multiplier.
    gpu.scale_f32(&scratch.x, lw.layer_scalar_host)?;

    Ok(())
}

/// Token-batched prefill. Processes up to `scratch.max_prefill_batch` tokens
/// at once, amortizing MoE-branch launch overhead across the batch via the
/// batched-indexed kernels (router GEMM, top-K, gate_up, gelu*mul, down).
///
/// Per-token operations (attention, dense projections, RoPE, KV writes)
/// stay per-token in V1 — no batched flash-prefill kernel for asym3
/// sliding window. The win comes from collapsing N per-token MoE calls
/// (~10 launches each) into one batched MoE call per layer.
///
/// On exit: `kv_sliding` / `kv_full` are updated for every token. The KV
/// cache is the source of truth for downstream sampling; this function
/// does NOT compute logits (the caller should follow with a per-token
/// `forward_scratch` for the LAST token if it needs logits).
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    // v2 — batched dense projections + batched MoE. The +55% prefill win
    // from 521161f8 is back: the regressing bug was in `gemm_hfq4g128`'s
    // partial-trailing-group handling (used floor instead of ceil for
    // groups_per_row, silently dropped 64 input dims at K=2112 dense
    // down_proj → garbage). Fixed in kernels/src/gemm_hfq4g128.hip.
    forward_prefill_batch_v2(
        gpu, weights, config, tokens, start_pos, kv_sliding, kv_full, scratch,
    )
}

fn forward_prefill_batch_v1(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let n_batch = tokens.len();
    if n_batch == 0 {
        return Ok(());
    }
    if n_batch > scratch.max_prefill_batch {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch: n_batch={n_batch} > max_prefill_batch={}; \
             caller must chunk",
                scratch.max_prefill_batch
            ),
        ));
    }
    let dim = config.dim;
    let dim_bytes = dim * 4;

    // ── Step 1: per-token embed + scale into pb_residual[i]. ─────────────
    for (i, &tok) in tokens.iter().enumerate() {
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "unsupported Gemma 4 embed format",
                ))
            }
        }
        gpu.scale_f32(&scratch.x, config.embed_scale)?;
        // pb_residual[i] = scratch.x
        let stream = gpu.active_stream.as_ref();
        if let Some(s) = stream {
            gpu.hip.memcpy_dtod_async_at(
                &scratch.pb_residual.buf,
                i * dim_bytes,
                &scratch.x.buf,
                0,
                dim_bytes,
                s,
            )?;
        } else {
            gpu.hip.memcpy_dtod_at(
                &scratch.pb_residual.buf,
                i * dim_bytes,
                &scratch.x.buf,
                0,
                dim_bytes,
            )?;
        }
    }

    // ── Step 2: layer loop. Each layer: per-token attn+FFN, then batched MoE. ──
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        // Stage A — per-token attn + FFN. Fills pb_attn_out[i] = post-attn x,
        // pb_ffn_out[i] = dense FFN out. Also writes the per-token KV at
        // position start_pos+i.
        let stream_present = gpu.active_stream.is_some();
        for i in 0..n_batch {
            let pos = start_pos + i;
            let pos_i32 = pos as i32;
            gpu.hip
                .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
            // Load this token's residual into scratch.x.
            if stream_present {
                let s = gpu.active_stream.as_ref().unwrap();
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.x.buf,
                    0,
                    &scratch.pb_residual.buf,
                    i * dim_bytes,
                    dim_bytes,
                    s,
                )?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.x.buf,
                    0,
                    &scratch.pb_residual.buf,
                    i * dim_bytes,
                    dim_bytes,
                )?;
            }
            match (layer_type, &weights.layers[layer_idx]) {
                (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                    sliding_layer_attn_ffn_only(
                        gpu,
                        config,
                        lw,
                        pos,
                        kv_sliding,
                        sliding_kv_idx,
                        scratch,
                    )?;
                }
                (LayerType::Full, LayerWeights::Full(lw)) => {
                    full_layer_attn_ffn_only(gpu, config, lw, pos, kv_full, full_kv_idx, scratch)?;
                }
                _ => {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!("Gemma 4 layer {} type/weights mismatch", layer_idx),
                    ))
                }
            }
            // Copy outputs into batch slots:
            //   pb_attn_out[i] = scratch.residual (post-attn x; input to MoE pre2/router_in)
            //   pb_ffn_out [i] = scratch.ffn_out  (dense FFN output)
            //   pb_residual[i] = scratch.residual (running residual)
            if stream_present {
                let s = gpu.active_stream.as_ref().unwrap();
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.pb_attn_out.buf,
                    i * dim_bytes,
                    &scratch.residual.buf,
                    0,
                    dim_bytes,
                    s,
                )?;
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.pb_ffn_out.buf,
                    i * dim_bytes,
                    &scratch.ffn_out.buf,
                    0,
                    dim_bytes,
                    s,
                )?;
                gpu.hip.memcpy_dtod_async_at(
                    &scratch.pb_residual.buf,
                    i * dim_bytes,
                    &scratch.residual.buf,
                    0,
                    dim_bytes,
                    s,
                )?;
            } else {
                gpu.hip.memcpy_dtod_at(
                    &scratch.pb_attn_out.buf,
                    i * dim_bytes,
                    &scratch.residual.buf,
                    0,
                    dim_bytes,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &scratch.pb_ffn_out.buf,
                    i * dim_bytes,
                    &scratch.ffn_out.buf,
                    0,
                    dim_bytes,
                )?;
                gpu.hip.memcpy_dtod_at(
                    &scratch.pb_residual.buf,
                    i * dim_bytes,
                    &scratch.residual.buf,
                    0,
                    dim_bytes,
                )?;
            }
        }
        match layer_type {
            LayerType::Sliding => sliding_kv_idx += 1,
            LayerType::Full => full_kv_idx += 1,
        }

        // Stage B — batched MoE branch (or dense post-FFN on layers without MoE).
        let layer_scalar = match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => lw.layer_scalar_host,
            (LayerType::Full, LayerWeights::Full(lw)) => lw.layer_scalar_host,
            _ => unreachable!(),
        };
        let (moe_opt, post_ffn_norm) = match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                (lw.moe.as_ref(), &lw.post_feedforward_layernorm)
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                (lw.moe.as_ref(), &lw.post_feedforward_layernorm)
            }
            _ => unreachable!(),
        };
        match moe_opt {
            Some(moe) => {
                apply_moe_branch_batched(gpu, config, scratch, moe, post_ffn_norm, n_batch)?;
                // Stage C — batched residual add + layer scale.
                // pb_residual[0..N×dim] += pb_moe_pre2[0..N×dim]
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_moe_pre2)?;
                gpu.scale_f32(&scratch.pb_residual, layer_scalar)?;
            }
            None => {
                // Dense path — fall back to per-token finalization to keep
                // semantics identical to single-token forward.
                for i in 0..n_batch {
                    if stream_present {
                        let s = gpu.active_stream.as_ref().unwrap();
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.x.buf,
                            0,
                            &scratch.pb_residual.buf,
                            i * dim_bytes,
                            dim_bytes,
                            s,
                        )?;
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.residual.buf,
                            0,
                            &scratch.pb_residual.buf,
                            i * dim_bytes,
                            dim_bytes,
                            s,
                        )?;
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.ffn_out.buf,
                            0,
                            &scratch.pb_ffn_out.buf,
                            i * dim_bytes,
                            dim_bytes,
                            s,
                        )?;
                    }
                    gpu.rmsnorm_f32(
                        &scratch.ffn_out,
                        post_ffn_norm,
                        &scratch.tmp,
                        config.norm_eps,
                    )?;
                    if stream_present {
                        let s = gpu.active_stream.as_ref().unwrap();
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.x.buf,
                            0,
                            &scratch.residual.buf,
                            0,
                            dim_bytes,
                            s,
                        )?;
                    }
                    gpu.add_inplace_f32(&scratch.x, &scratch.tmp)?;
                    gpu.scale_f32(&scratch.x, layer_scalar)?;
                    if stream_present {
                        let s = gpu.active_stream.as_ref().unwrap();
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.pb_residual.buf,
                            i * dim_bytes,
                            &scratch.x.buf,
                            0,
                            dim_bytes,
                            s,
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// V2 batched prefill — batches both the dense projections AND the MoE
/// branch across N tokens. Per-token RoPE and attention stay sequential
/// because the flash kernels haven't been ported to prefill-batched layout.
///
/// Fixed 2026-05-19 by the gemm_hfq4g128 partial-trailing-group fix
/// (kernels/src/gemm_hfq4g128.hip). The v2 dispatch path is byte-identical
/// to v1 through attention; the divergence was the missing 64-element
/// partial group in the dense down_proj GEMM at K=2112.
fn forward_prefill_batch_v2(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let n_batch = tokens.len();
    if n_batch == 0 {
        return Ok(());
    }
    if n_batch > scratch.max_prefill_batch {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "forward_prefill_batch: n_batch={n_batch} > max_prefill_batch={}",
                scratch.max_prefill_batch
            ),
        ));
    }
    let dim = config.dim;
    let dim_bytes = dim * 4;

    // Upload positions array [start_pos, start_pos+1, ..., start_pos+N-1] for batched RoPE.
    let pos_array: Vec<i32> = (0..n_batch).map(|i| (start_pos + i) as i32).collect();
    let pos_bytes: Vec<u8> = pos_array.iter().flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&scratch.pb_positions.buf, &pos_bytes)?;

    // Step 1: per-token embed lookup into pb_residual.
    for (i, &tok) in tokens.iter().enumerate() {
        match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            EmbeddingFormat::F32 => {
                gpu.embedding_lookup(&weights.embed_tokens, &scratch.x, tok, dim)?
            }
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "unsupported Gemma 4 embed format",
                ))
            }
        }
        gpu.scale_f32(&scratch.x, config.embed_scale)?;
        if let Some(_s) = gpu.active_stream.as_ref() {
            gpu.hip.memcpy_dtod_async_at(
                &scratch.pb_residual.buf,
                i * dim_bytes,
                &scratch.x.buf,
                0,
                dim_bytes,
                _s,
            )?;
        } else {
            gpu.hip.memcpy_dtod_at(
                &scratch.pb_residual.buf,
                i * dim_bytes,
                &scratch.x.buf,
                0,
                dim_bytes,
            )?;
        }
    }

    // Step 2: layer loop.
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;
    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        // Invalidate the pointer-keyed F16 conversion cache between layers.
        // The same activation buffer (pb_tmp) is reused with different contents
        // each layer; without invalidation, ensure_fp16_x would skip the
        // F32→F16 conversion and serve stale data from the previous layer.
        gpu.invalidate_fp16_cache();
        let (layer_scalar, post_ffn_norm_ref, moe_opt) =
            match (layer_type, &weights.layers[layer_idx]) {
                (LayerType::Sliding, LayerWeights::Sliding(lw)) => (
                    lw.layer_scalar_host,
                    &lw.post_feedforward_layernorm,
                    lw.moe.as_ref(),
                ),
                (LayerType::Full, LayerWeights::Full(lw)) => (
                    lw.layer_scalar_host,
                    &lw.post_feedforward_layernorm,
                    lw.moe.as_ref(),
                ),
                _ => {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!("layer {layer_idx} type/weights mismatch"),
                    ))
                }
            };

        // ── 2A: batched pre-attn rmsnorm + Q/K/V projections + Q/K/V norms + RoPE. ──
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                let head_dim = config.sliding_head_dim;
                let n_heads = config.n_heads;
                let n_kv = config.sliding_n_kv_heads;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv * head_dim;
                let kv_dim_bytes = kv_dim * 4;
                let q_dim_bytes = q_dim * 4;

                let _dump_on = layer_idx == 0 && start_pos == 0;
                if _dump_on {
                    dbg_dump(
                        gpu,
                        "[v2] L0 input pb_residual[0]",
                        &scratch.pb_residual,
                        dim,
                    );
                }
                gpu.rmsnorm_batched(
                    &scratch.pb_residual,
                    &lw.input_layernorm,
                    &scratch.pb_tmp,
                    n_batch,
                    dim,
                    config.norm_eps,
                )?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after input_norm", &scratch.pb_tmp, dim);
                }
                // BF16 hoist: one F32->BF16 for the shared pb_tmp -> q/k/v group.
                // Capture the shared F32 source (pb_tmp) once per constituent
                // weight before staging — same src, per-weight k. The inner
                // run_prefill_gemm_inner skips capture for BF16 x (hoisted),
                // so this hoist-site capture restores coverage. Zero-cost when unarmed.
                if lw.q_proj.gpu_dtype == DType::BF16 {
                    let nelems = n_batch * dim;
                    let bf16_x = scratch.pb_bf16.sub_offset(0, nelems);
                    gpu.maybe_capture_activation(
                        &lw.q_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.q_proj.k,
                    );
                    gpu.maybe_capture_activation(
                        &lw.k_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.k_proj.k,
                    );
                    gpu.maybe_capture_activation(
                        &lw.v_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.v_proj.k,
                    );
                    gpu.convert_f32_to_bf16(&scratch.pb_tmp, &bf16_x, nelems)?;
                    run_prefill_gemm(
                        gpu,
                        &lw.q_proj,
                        &bf16_x,
                        &scratch.pb_q,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.k_proj,
                        &bf16_x,
                        &scratch.pb_k,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.v_proj,
                        &bf16_x,
                        &scratch.pb_v,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                } else {
                    run_prefill_gemm(
                        gpu,
                        &lw.q_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_q,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.k_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_k,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.v_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_v,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                }
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after q_proj", &scratch.pb_q, q_dim);
                    dbg_dump(gpu, "[v2] L0 after k_proj", &scratch.pb_k, kv_dim);
                    dbg_dump(gpu, "[v2] L0 after v_proj", &scratch.pb_v, kv_dim);
                }
                gpu.rmsnorm_batched(
                    &scratch.pb_q,
                    &lw.q_norm,
                    &scratch.pb_q,
                    n_batch * n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &scratch.pb_k,
                    &lw.k_norm,
                    &scratch.pb_k,
                    n_batch * n_kv,
                    head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &scratch.pb_v,
                    &scratch.v_norm_ones_full,
                    &scratch.pb_v,
                    n_batch * n_kv,
                    head_dim,
                    config.norm_eps,
                )?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after q_norm", &scratch.pb_q, q_dim);
                    dbg_dump(gpu, "[v2] L0 after k_norm", &scratch.pb_k, kv_dim);
                    dbg_dump(gpu, "[v2] L0 after v_norm", &scratch.pb_v, kv_dim);
                }
                gpu.scale_f32(&scratch.pb_q, (head_dim as f32).sqrt())?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after scale_q", &scratch.pb_q, q_dim);
                }
                gpu.rope_batched_f32(
                    &scratch.pb_q,
                    &scratch.pb_k,
                    &scratch.pb_positions,
                    n_heads,
                    n_kv,
                    head_dim,
                    config.sliding_rope_theta,
                    n_batch,
                )?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after rope_q", &scratch.pb_q, q_dim);
                    dbg_dump(gpu, "[v2] L0 after rope_k", &scratch.pb_k, kv_dim);
                }

                // Per-token KV write + attention (batched projections, per-token
                // attention). The batched q8 attention path has a bug that
                // corrupts KV when cache_capacity > 0 (ring-buffer mode), so
                // we run attention per-token through the proven decode-path
                // Step::Attend with batch_size=1. This is only 1.7% of GPU
                // time per the rocprof profile, so the overhead is minimal.
                let sliding_cap = config.sliding_window as u32;
                for i in 0..n_batch {
                    let pos = start_pos + i;
                    if let Some(stream) = gpu.active_stream.as_ref() {
                        gpu.hip
                            .stream_write_value32(stream, &scratch.pos_buf, pos as u32, 0)?;
                    } else {
                        gpu.hip
                            .memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.q.buf,
                            0,
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            q_dim_bytes,
                            _s,
                        )?;
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.k.buf,
                            0,
                            &scratch.pb_k.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                            _s,
                        )?;
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.v.buf,
                            0,
                            &scratch.pb_v.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                            _s,
                        )?;
                    } else {
                        gpu.hip.memcpy_dtod_at(
                            &scratch.q.buf,
                            0,
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            q_dim_bytes,
                        )?;
                        gpu.hip.memcpy_dtod_at(
                            &scratch.k.buf,
                            0,
                            &scratch.pb_k.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                        )?;
                        gpu.hip.memcpy_dtod_at(
                            &scratch.v.buf,
                            0,
                            &scratch.pb_v.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                        )?;
                    }
                    let tier_inputs = KvTierInputs {
                        quant_asym4: kv_sliding.quant_asym4,
                        quant_asym3: kv_sliding.quant_asym3,
                        quant_asym2: kv_sliding.quant_asym2,
                        quant_q8: kv_sliding.quant_q8,
                        quant_fwht: kv_sliding.quant_fwht,
                        quant_hfq4: false,
                        quant_q4: false,
                        quant_int8: false,
                        quant_hfq8: false,
                        f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
                        v_mode_bits: kv_sliding.v_mode_bits(),
                        pos,
                        flash_mode: 2,
                        capture_mode: gpu.graphs.capture_mode,
                        batch_size: 1,
                        is_tree: false,
                        is_boundary: false,
                        q8_windowed: false,
                        window: sliding_cap as i32,
                    };
                    let plan = KvTierPlan::derive(tier_inputs)
                        .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
                    let io = AttnParams {
                        q: &scratch.q,
                        k: &scratch.k,
                        v: &scratch.v,
                        k_cache: &kv_sliding.k_gpu[sliding_kv_idx],
                        v_cache: &kv_sliding.v_gpu[sliding_kv_idx],
                        k_scales: None,
                        v_scales: None,
                        pos_buf: &scratch.pos_buf,
                        pos,
                        positions: None,
                        n_heads,
                        n_kv_heads: n_kv,
                        head_dim,
                        physical_cap: kv_sliding.max_seq,
                        batch_size: 1,
                        max_ctx_len: 0,
                        flash_partials: Some(&scratch.flash_partials),
                        givens_cos: kv_sliding.givens_cos.as_ref(),
                        givens_sin: kv_sliding.givens_sin.as_ref(),
                        tree_bias: None,
                        block_start: 0,
                        block_cols: 0,
                        output_gate: None,
                        output: &scratch.attn_out,
                    };
                    let ctx = DispatchCtx::new(gpu);
                    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                    // Copy attention output back to batched buffer.
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            &scratch.attn_out.buf,
                            0,
                            q_dim_bytes,
                            _s,
                        )?;
                    } else {
                        gpu.hip.memcpy_dtod_at(
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            &scratch.attn_out.buf,
                            0,
                            q_dim_bytes,
                        )?;
                    }
                }
                sliding_kv_idx += 1;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after attention (pb_q)", &scratch.pb_q, q_dim);
                }

                run_prefill_gemm(
                    gpu,
                    &lw.o_proj,
                    &scratch.pb_q,
                    &scratch.pb_attn_out,
                    n_batch,
                    Some(&scratch.pb_bf16),
                )?;
                if _dump_on {
                    dbg_dump(gpu, "[v2] L0 after o_proj", &scratch.pb_attn_out, dim);
                }
                gpu.rmsnorm_batched(
                    &scratch.pb_attn_out,
                    &lw.post_attention_layernorm,
                    &scratch.pb_attn_out,
                    n_batch,
                    dim,
                    config.norm_eps,
                )?;
                if _dump_on {
                    dbg_dump(
                        gpu,
                        "[v2] L0 after post_attn_norm",
                        &scratch.pb_attn_out,
                        dim,
                    );
                }
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_attn_out)?;
                if _dump_on {
                    dbg_dump(
                        gpu,
                        "[v2] L0 after attn_residual",
                        &scratch.pb_residual,
                        dim,
                    );
                }
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                let head_dim = config.full_head_dim;
                let n_heads = config.n_heads;
                let n_kv = config.full_n_kv_heads;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv * head_dim;
                let kv_dim_bytes = kv_dim * 4;
                let q_dim_bytes = q_dim * 4;

                gpu.rmsnorm_batched(
                    &scratch.pb_residual,
                    &lw.input_layernorm,
                    &scratch.pb_tmp,
                    n_batch,
                    dim,
                    config.norm_eps,
                )?;
                // BF16 hoist for shared pb_tmp -> q/k (full). Capture shared F32
                // pb_tmp once per weight before staging. Gate+up hoisted separately below.
                if lw.q_proj.gpu_dtype == DType::BF16 {
                    let nelems = n_batch * dim;
                    let bf16_x = scratch.pb_bf16.sub_offset(0, nelems);
                    gpu.maybe_capture_activation(
                        &lw.q_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.q_proj.k,
                    );
                    gpu.maybe_capture_activation(
                        &lw.k_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.k_proj.k,
                    );
                    gpu.convert_f32_to_bf16(&scratch.pb_tmp, &bf16_x, nelems)?;
                    run_prefill_gemm(
                        gpu,
                        &lw.q_proj,
                        &bf16_x,
                        &scratch.pb_q,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.k_proj,
                        &bf16_x,
                        &scratch.pb_k,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                } else {
                    run_prefill_gemm(
                        gpu,
                        &lw.q_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_q,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.k_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_k,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                }
                if let Some(s) = gpu.active_stream.as_ref() {
                    gpu.hip.memcpy_dtod_async_at(
                        &scratch.pb_v.buf,
                        0,
                        &scratch.pb_k.buf,
                        0,
                        n_batch * kv_dim_bytes,
                        s,
                    )?;
                } else {
                    gpu.hip.memcpy_dtod_at(
                        &scratch.pb_v.buf,
                        0,
                        &scratch.pb_k.buf,
                        0,
                        n_batch * kv_dim_bytes,
                    )?;
                }
                gpu.rmsnorm_batched(
                    &scratch.pb_q,
                    &lw.q_norm,
                    &scratch.pb_q,
                    n_batch * n_heads,
                    head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &scratch.pb_k,
                    &lw.k_norm,
                    &scratch.pb_k,
                    n_batch * n_kv,
                    head_dim,
                    config.norm_eps,
                )?;
                gpu.rmsnorm_batched(
                    &scratch.pb_v,
                    &scratch.v_norm_ones_full,
                    &scratch.pb_v,
                    n_batch * n_kv,
                    head_dim,
                    config.norm_eps,
                )?;
                gpu.scale_f32(&scratch.pb_q, (head_dim as f32).sqrt())?;
                let n_rot_pairs =
                    ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize;
                // Per-token partial-halved RoPE (no batched variant yet for this shape).
                for i in 0..n_batch {
                    let pos = start_pos + i;
                    if let Some(stream) = gpu.active_stream.as_ref() {
                        gpu.hip
                            .stream_write_value32(stream, &scratch.pos_buf, pos as u32, 0)?;
                    } else {
                        let pos_i32 = pos as i32;
                        gpu.hip
                            .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.q.buf,
                            0,
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            q_dim_bytes,
                            _s,
                        )?;
                    } else {
                        gpu.hip.memcpy_dtod_at(
                            &scratch.q.buf,
                            0,
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            q_dim_bytes,
                        )?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.k.buf,
                            0,
                            &scratch.pb_k.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                            _s,
                        )?;
                    } else {
                        gpu.hip.memcpy_dtod_at(
                            &scratch.k.buf,
                            0,
                            &scratch.pb_k.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                        )?;
                    }
                    gpu.rope_partial_halved_f32(
                        &scratch.q,
                        &scratch.k,
                        &scratch.pos_buf,
                        n_heads,
                        n_kv,
                        head_dim,
                        n_rot_pairs,
                        config.full_rope_theta,
                    )?;
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            &scratch.q.buf,
                            0,
                            q_dim_bytes,
                            _s,
                        )?;
                    } else {
                        gpu.hip.memcpy_dtod_at(
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            &scratch.q.buf,
                            0,
                            q_dim_bytes,
                        )?;
                    }
                    if let Some(_s) = gpu.active_stream.as_ref() {
                        gpu.hip.memcpy_dtod_async_at(
                            &scratch.pb_k.buf,
                            i * kv_dim_bytes,
                            &scratch.k.buf,
                            0,
                            kv_dim_bytes,
                            _s,
                        )?;
                    } else {
                        gpu.hip.memcpy_dtod_at(
                            &scratch.pb_k.buf,
                            i * kv_dim_bytes,
                            &scratch.k.buf,
                            0,
                            kv_dim_bytes,
                        )?;
                    }
                }

                // Batched full-layer attention (hd=512): all N tokens in ONE
                // masked flash call instead of a per-token loop. Full layers are
                // non-ring (cache_capacity=0) and non-windowed (window_size=0);
                // the asym3 batched-masked kernel derives each query's causal
                // bound from pb_positions, so this reproduces the per-token
                // result with 1 launch/layer instead of N. (Sliding layers stay
                // per-token until the q8 batched ring kernel lands — A.2.)
                // Proportional RoPE was already applied to pb_q/pb_k above.
                let _ = (q_dim_bytes, kv_dim_bytes); // (no per-token staging now)
                let tier_inputs = KvTierInputs {
                    quant_asym4: kv_full.quant_asym4,
                    quant_asym3: kv_full.quant_asym3,
                    quant_asym2: kv_full.quant_asym2,
                    quant_q8: kv_full.quant_q8,
                    quant_fwht: kv_full.quant_fwht,
                    quant_hfq4: false,
                    quant_q4: false,
                    quant_int8: false,
                    quant_hfq8: false,
                    f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
                    v_mode_bits: kv_full.v_mode_bits(),
                    pos: start_pos + n_batch - 1,
                    flash_mode: 2,
                    capture_mode: gpu.graphs.capture_mode,
                    batch_size: n_batch,
                    is_tree: false,
                    is_boundary: false,
                    q8_windowed: false,
                    window: 0,
                };
                let plan = KvTierPlan::derive(tier_inputs)
                    .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
                let io = AttnParams {
                    q: &scratch.pb_q,
                    k: &scratch.pb_k,
                    v: &scratch.pb_v, // k_eq_v model: V = pre-k_norm k_proj out, v-normed (NOT k-normed/roped pb_k)
                    k_cache: &kv_full.k_gpu[full_kv_idx],
                    v_cache: &kv_full.v_gpu[full_kv_idx],
                    k_scales: None,
                    v_scales: None,
                    pos_buf: &scratch.pos_buf,
                    pos: start_pos + n_batch - 1,
                    positions: Some(&scratch.pb_positions),
                    n_heads,
                    n_kv_heads: n_kv,
                    head_dim,
                    physical_cap: kv_full.max_seq,
                    batch_size: n_batch,
                    max_ctx_len: start_pos + n_batch,
                    flash_partials: Some(&scratch.pb_flash_partials),
                    givens_cos: kv_full.givens_cos.as_ref(),
                    givens_sin: kv_full.givens_sin.as_ref(),
                    tree_bias: None,
                    block_start: 0,
                    block_cols: 0,
                    output_gate: None,
                    output: &scratch.pb_attn_q,
                };
                let ctx = DispatchCtx::new(gpu);
                execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                if std::env::var("HIPFIRE_GEMMA4_ATTN_VERIFY").ok().as_deref() == Some("1") {
                    let batched_out = gpu.download_f32(&scratch.pb_attn_q)?;
                    for i in 0..n_batch {
                        let pos = start_pos + i;
                        gpu.hip
                            .memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
                        gpu.hip.memcpy_dtod_at(
                            &scratch.q.buf,
                            0,
                            &scratch.pb_q.buf,
                            i * q_dim_bytes,
                            q_dim_bytes,
                        )?;
                        gpu.hip.memcpy_dtod_at(
                            &scratch.k.buf,
                            0,
                            &scratch.pb_k.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                        )?;
                        gpu.hip.memcpy_dtod_at(
                            &scratch.v.buf,
                            0,
                            &scratch.pb_v.buf,
                            i * kv_dim_bytes,
                            kv_dim_bytes,
                        )?;
                        let ti = KvTierInputs {
                            quant_asym4: kv_full.quant_asym4,
                            quant_asym3: kv_full.quant_asym3,
                            quant_asym2: kv_full.quant_asym2,
                            quant_q8: kv_full.quant_q8,
                            quant_fwht: kv_full.quant_fwht,
                            quant_hfq4: false,
                            quant_q4: false,
                            quant_int8: false,
                            quant_hfq8: false,
                            f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
                            v_mode_bits: kv_full.v_mode_bits(),
                            pos,
                            flash_mode: 2,
                            capture_mode: gpu.graphs.capture_mode,
                            batch_size: 1,
                            is_tree: false,
                            is_boundary: false,
                            q8_windowed: false,
                            window: 0,
                        };
                        let p1 = KvTierPlan::derive(ti)
                            .map_err(|e| hip_bridge::HipError::new(0, &format!("{:?}", e)))?;
                        let io1 = AttnParams {
                            q: &scratch.q,
                            k: &scratch.k,
                            v: &scratch.v,
                            k_cache: &kv_full.k_gpu[full_kv_idx],
                            v_cache: &kv_full.v_gpu[full_kv_idx],
                            k_scales: None,
                            v_scales: None,
                            pos_buf: &scratch.pos_buf,
                            pos,
                            positions: None,
                            n_heads,
                            n_kv_heads: n_kv,
                            head_dim,
                            physical_cap: kv_full.max_seq,
                            batch_size: 1,
                            max_ctx_len: 0,
                            flash_partials: Some(&scratch.flash_partials),
                            givens_cos: kv_full.givens_cos.as_ref(),
                            givens_sin: kv_full.givens_sin.as_ref(),
                            tree_bias: None,
                            block_start: 0,
                            block_cols: 0,
                            output_gate: None,
                            output: &scratch.attn_out,
                        };
                        let c1 = DispatchCtx::new(gpu);
                        execute_steps(gpu, &c1, &[Step::Attend { plan: p1, io: io1 }])
                            .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                        let single = gpu.download_f32(&scratch.attn_out)?;
                        let row = &batched_out[i * q_dim..(i + 1) * q_dim];
                        let mut worst = 0f32;
                        let mut wi = 0usize;
                        for j in 0..q_dim {
                            let d = (single[j] - row[j]).abs();
                            if d > worst {
                                worst = d;
                                wi = j;
                            }
                        }
                        eprintln!("[attn-verify] L{layer_idx} start={start_pos} tok={i} pos={pos} worst={worst:.5} at {wi} single={:.4} batched={:.4}",
                            single[wi], row[wi]);
                    }
                }
                full_kv_idx += 1;

                run_prefill_gemm(
                    gpu,
                    &lw.o_proj,
                    &scratch.pb_attn_q,
                    &scratch.pb_attn_out,
                    n_batch,
                    Some(&scratch.pb_bf16),
                )?;
                gpu.rmsnorm_batched(
                    &scratch.pb_attn_out,
                    &lw.post_attention_layernorm,
                    &scratch.pb_attn_out,
                    n_batch,
                    dim,
                    config.norm_eps,
                )?;
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_attn_out)?;
            }
            _ => unreachable!(),
        }

        // Snapshot post-attn residual for MoE input.
        if let Some(s) = gpu.active_stream.as_ref() {
            gpu.hip.memcpy_dtod_async_at(
                &scratch.pb_attn_out.buf,
                0,
                &scratch.pb_residual.buf,
                0,
                n_batch * dim_bytes,
                s,
            )?;
        } else {
            gpu.hip.memcpy_dtod_at(
                &scratch.pb_attn_out.buf,
                0,
                &scratch.pb_residual.buf,
                0,
                n_batch * dim_bytes,
            )?;
        }

        // Batched pre-FFN rmsnorm + dense FFN.
        let _dump_ffn = layer_idx == 0 && start_pos == 0;
        match (layer_type, &weights.layers[layer_idx]) {
            (LayerType::Sliding, LayerWeights::Sliding(lw)) => {
                gpu.rmsnorm_batched(
                    &scratch.pb_residual,
                    &lw.pre_feedforward_layernorm,
                    &scratch.pb_tmp,
                    n_batch,
                    dim,
                    config.norm_eps,
                )?;
                if _dump_ffn {
                    dbg_dump(gpu, "[v2] L0 after pre_ffn_norm", &scratch.pb_tmp, dim);
                }
                // BF16 hoist for shared pb_tmp -> gate+up. Capture shared F32
                // pb_tmp once per weight before staging.
                if lw.gate_proj.gpu_dtype == DType::BF16 {
                    let nelems = n_batch * dim;
                    let bf16_x = scratch.pb_bf16.sub_offset(0, nelems);
                    gpu.maybe_capture_activation(
                        &lw.gate_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.gate_proj.k,
                    );
                    gpu.maybe_capture_activation(
                        &lw.up_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.up_proj.k,
                    );
                    gpu.convert_f32_to_bf16(&scratch.pb_tmp, &bf16_x, nelems)?;
                    run_prefill_gemm(
                        gpu,
                        &lw.gate_proj,
                        &bf16_x,
                        &scratch.pb_gate,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.up_proj,
                        &bf16_x,
                        &scratch.pb_up,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                } else {
                    run_prefill_gemm(
                        gpu,
                        &lw.gate_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_gate,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.up_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_up,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                }
                if _dump_ffn {
                    dbg_dump(
                        gpu,
                        "[v2] L0 after gate_proj",
                        &scratch.pb_gate,
                        config.hidden_dim,
                    );
                    dbg_dump(
                        gpu,
                        "[v2] L0 after up_proj",
                        &scratch.pb_up,
                        config.hidden_dim,
                    );
                }
                gpu.gelu_tanh_f32(
                    &scratch.pb_gate,
                    &scratch.pb_ffn_hidden,
                    n_batch * config.hidden_dim,
                )?;
                gpu.mul_f32(
                    &scratch.pb_ffn_hidden,
                    &scratch.pb_up,
                    &scratch.pb_ffn_hidden,
                )?;
                if _dump_ffn {
                    dbg_dump(
                        gpu,
                        "[v2] L0 after gelu*up",
                        &scratch.pb_ffn_hidden,
                        config.hidden_dim,
                    );
                }
                run_prefill_gemm(
                    gpu,
                    &lw.down_proj,
                    &scratch.pb_ffn_hidden,
                    &scratch.pb_ffn_out,
                    n_batch,
                    Some(&scratch.pb_bf16),
                )?;
                if _dump_ffn {
                    dbg_dump(gpu, "[v2] L0 after down_proj", &scratch.pb_ffn_out, dim);
                }
            }
            (LayerType::Full, LayerWeights::Full(lw)) => {
                gpu.rmsnorm_batched(
                    &scratch.pb_residual,
                    &lw.pre_feedforward_layernorm,
                    &scratch.pb_tmp,
                    n_batch,
                    dim,
                    config.norm_eps,
                )?;
                // BF16 hoist for shared pb_tmp -> gate+up (full). Capture shared
                // F32 pb_tmp once per weight before staging.
                if lw.gate_proj.gpu_dtype == DType::BF16 {
                    let nelems = n_batch * dim;
                    let bf16_x = scratch.pb_bf16.sub_offset(0, nelems);
                    gpu.maybe_capture_activation(
                        &lw.gate_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.gate_proj.k,
                    );
                    gpu.maybe_capture_activation(
                        &lw.up_proj.buf,
                        &scratch.pb_tmp,
                        n_batch,
                        lw.up_proj.k,
                    );
                    gpu.convert_f32_to_bf16(&scratch.pb_tmp, &bf16_x, nelems)?;
                    run_prefill_gemm(
                        gpu,
                        &lw.gate_proj,
                        &bf16_x,
                        &scratch.pb_gate,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.up_proj,
                        &bf16_x,
                        &scratch.pb_up,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                } else {
                    run_prefill_gemm(
                        gpu,
                        &lw.gate_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_gate,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                    run_prefill_gemm(
                        gpu,
                        &lw.up_proj,
                        &scratch.pb_tmp,
                        &scratch.pb_up,
                        n_batch,
                        Some(&scratch.pb_bf16),
                    )?;
                }
                gpu.gelu_tanh_f32(
                    &scratch.pb_gate,
                    &scratch.pb_ffn_hidden,
                    n_batch * config.hidden_dim,
                )?;
                gpu.mul_f32(
                    &scratch.pb_ffn_hidden,
                    &scratch.pb_up,
                    &scratch.pb_ffn_hidden,
                )?;
                run_prefill_gemm(
                    gpu,
                    &lw.down_proj,
                    &scratch.pb_ffn_hidden,
                    &scratch.pb_ffn_out,
                    n_batch,
                    Some(&scratch.pb_bf16),
                )?;
            }
            _ => unreachable!(),
        }

        // MoE branch (or dense fallback). HIPFIRE_MOE_BYPASS=1 forces dense
        // path even on MoE layers (parity with v1 — used to isolate whether
        // a regression lives in apply_moe_branch_batched vs the dense path).
        let moe_bypass = std::env::var("HIPFIRE_MOE_BYPASS").ok().as_deref() == Some("1");
        match (moe_opt, moe_bypass) {
            (Some(moe), false) => {
                apply_moe_branch_batched(gpu, config, scratch, moe, post_ffn_norm_ref, n_batch)?;
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_moe_pre2)?;
                gpu.scale_f32(&scratch.pb_residual, layer_scalar)?;
            }
            _ => {
                gpu.rmsnorm_batched(
                    &scratch.pb_ffn_out,
                    post_ffn_norm_ref,
                    &scratch.pb_ffn_out,
                    n_batch,
                    dim,
                    config.norm_eps,
                )?;
                gpu.add_inplace_f32(&scratch.pb_residual, &scratch.pb_ffn_out)?;
                gpu.scale_f32(&scratch.pb_residual, layer_scalar)?;
            }
        }
        if std::env::var("HIPFIRE_GEMMA4_DUMP").ok().as_deref() == Some("1") {
            let data = gpu.download_f32(&scratch.pb_residual).unwrap_or_default();
            let last = &data[(n_batch - 1) * dim..n_batch * dim];
            let sum: f64 = last.iter().map(|&v| v as f64).sum();
            let min = last.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = last.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            eprintln!("[v2 diag] L{layer_idx} {:?} last-tok hidden: first4={:?} sum={sum:.4e} min={min:.4} max={max:.4}",
                layer_type, &last[..4.min(last.len())]);
        }
    }

    // ── Final logits: extract last token, run final_norm + lm_head + softcap ──
    // The decode loop samples from scratch.logits, which must be filled
    // with the last token's logits. Copy last position from pb_residual
    // to scratch.tmp, run final_norm, lm_head, and softcap.
    let last_offset = (n_batch - 1) * dim * 4;
    if let Some(_s) = gpu.active_stream.as_ref() {
        gpu.hip.memcpy_dtod_async_at(
            &scratch.tmp.buf,
            0,
            &scratch.pb_residual.buf,
            last_offset,
            dim * 4,
            _s,
        )?;
    } else {
        gpu.hip.memcpy_dtod_at(
            &scratch.tmp.buf,
            0,
            &scratch.pb_residual.buf,
            last_offset,
            dim * 4,
        )?;
    }
    gpu.rmsnorm_f32(
        &scratch.tmp,
        &weights.final_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    let ctx = DispatchCtx::new(gpu);
    let wr_lm = weights.lm_head.dispatch_ref();
    execute_steps(
        gpu,
        &ctx,
        &[Step::Gemv {
            w: &wr_lm,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.logits,
        }],
    )
    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    if config.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(
            &scratch.logits,
            config.vocab_size,
            config.final_logit_softcapping,
        )?;
    }

    Ok(())
}

// ── Variant enum ──────────────────────────────────────────────────────────

/// The four Gemma 4 decoder-layer shapes. Derived from the layer type
/// (Sliding vs Full) and the presence of MoE extras. Pure → unit-testable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gemma4Variant {
    SlidingDense,
    SlidingMoe,
    FullDense,
    FullMoe,
}

fn gemma4_variant_of(layer_type: LayerType, layer: &LayerWeights) -> Gemma4Variant {
    let has_moe = match layer {
        LayerWeights::Sliding(lw) => lw.moe.is_some(),
        LayerWeights::Full(lw) => lw.moe.is_some(),
    };
    match (layer_type, has_moe) {
        (LayerType::Sliding, false) => Gemma4Variant::SlidingDense,
        (LayerType::Sliding, true) => Gemma4Variant::SlidingMoe,
        (LayerType::Full, false) => Gemma4Variant::FullDense,
        (LayerType::Full, true) => Gemma4Variant::FullMoe,
    }
}

/// Architecture-local metadata for the typed Gemma4 layer lowering.
///
/// These descriptors are deliberately separate from dispatch's `Step` IR:
/// they pin the semantic order without becoming an opcode or a runtime value.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gemma4Op {
    NormInput,
    ProjQ,
    ProjK,
    ProjV,
    CopyKToV,
    NormQ,
    NormK,
    NormV,
    ScaleQ,
    RopeFull,
    RopePartial,
    Attend,
    ProjO,
    NormPostAttn,
    ResidualAddAttn,
    SaveResidual,
    NormPreFfn,
    ProjGate,
    ProjUp,
    GeluTanhMul,
    ProjDown,
    NormPostFfn,
    NormPostFfn1,
    NormPreFfn2,
    NormRouter,
    ScaleRouter,
    ProjRouter,
    MoeSoftmaxTopK,
    MoeExperts,
    NormPostFfn2,
    ResidualAddMoe,
    NormOuterFfn,
    RestoreResidual,
    ResidualAddFfn,
    ScaleLayer,
}

#[cfg(test)]
static SLIDING_DENSE_OPS: [Gemma4Op; 23] = [
    Gemma4Op::NormInput,
    Gemma4Op::ProjQ,
    Gemma4Op::ProjK,
    Gemma4Op::ProjV,
    Gemma4Op::NormQ,
    Gemma4Op::NormK,
    Gemma4Op::NormV,
    Gemma4Op::ScaleQ,
    Gemma4Op::RopeFull,
    Gemma4Op::Attend,
    Gemma4Op::ProjO,
    Gemma4Op::NormPostAttn,
    Gemma4Op::ResidualAddAttn,
    Gemma4Op::SaveResidual,
    Gemma4Op::NormPreFfn,
    Gemma4Op::ProjGate,
    Gemma4Op::ProjUp,
    Gemma4Op::GeluTanhMul,
    Gemma4Op::ProjDown,
    Gemma4Op::NormPostFfn,
    Gemma4Op::RestoreResidual,
    Gemma4Op::ResidualAddFfn,
    Gemma4Op::ScaleLayer,
];

#[cfg(test)]
static FULL_DENSE_OPS: [Gemma4Op; 23] = [
    Gemma4Op::NormInput,
    Gemma4Op::ProjQ,
    Gemma4Op::ProjK,
    Gemma4Op::CopyKToV,
    Gemma4Op::NormQ,
    Gemma4Op::NormK,
    Gemma4Op::NormV,
    Gemma4Op::ScaleQ,
    Gemma4Op::RopePartial,
    Gemma4Op::Attend,
    Gemma4Op::ProjO,
    Gemma4Op::NormPostAttn,
    Gemma4Op::ResidualAddAttn,
    Gemma4Op::SaveResidual,
    Gemma4Op::NormPreFfn,
    Gemma4Op::ProjGate,
    Gemma4Op::ProjUp,
    Gemma4Op::GeluTanhMul,
    Gemma4Op::ProjDown,
    Gemma4Op::NormPostFfn,
    Gemma4Op::RestoreResidual,
    Gemma4Op::ResidualAddFfn,
    Gemma4Op::ScaleLayer,
];

#[cfg(test)]
static SLIDING_MOE_OPS: [Gemma4Op; 32] = [
    Gemma4Op::NormInput,
    Gemma4Op::ProjQ,
    Gemma4Op::ProjK,
    Gemma4Op::ProjV,
    Gemma4Op::NormQ,
    Gemma4Op::NormK,
    Gemma4Op::NormV,
    Gemma4Op::ScaleQ,
    Gemma4Op::RopeFull,
    Gemma4Op::Attend,
    Gemma4Op::ProjO,
    Gemma4Op::NormPostAttn,
    Gemma4Op::ResidualAddAttn,
    Gemma4Op::SaveResidual,
    Gemma4Op::NormPreFfn,
    Gemma4Op::ProjGate,
    Gemma4Op::ProjUp,
    Gemma4Op::GeluTanhMul,
    Gemma4Op::ProjDown,
    Gemma4Op::NormPostFfn1,
    Gemma4Op::NormPreFfn2,
    Gemma4Op::NormRouter,
    Gemma4Op::ScaleRouter,
    Gemma4Op::ProjRouter,
    Gemma4Op::MoeSoftmaxTopK,
    Gemma4Op::MoeExperts,
    Gemma4Op::NormPostFfn2,
    Gemma4Op::ResidualAddMoe,
    Gemma4Op::NormOuterFfn,
    Gemma4Op::RestoreResidual,
    Gemma4Op::ResidualAddFfn,
    Gemma4Op::ScaleLayer,
];

#[cfg(test)]
static FULL_MOE_OPS: [Gemma4Op; 32] = [
    Gemma4Op::NormInput,
    Gemma4Op::ProjQ,
    Gemma4Op::ProjK,
    Gemma4Op::CopyKToV,
    Gemma4Op::NormQ,
    Gemma4Op::NormK,
    Gemma4Op::NormV,
    Gemma4Op::ScaleQ,
    Gemma4Op::RopePartial,
    Gemma4Op::Attend,
    Gemma4Op::ProjO,
    Gemma4Op::NormPostAttn,
    Gemma4Op::ResidualAddAttn,
    Gemma4Op::SaveResidual,
    Gemma4Op::NormPreFfn,
    Gemma4Op::ProjGate,
    Gemma4Op::ProjUp,
    Gemma4Op::GeluTanhMul,
    Gemma4Op::ProjDown,
    Gemma4Op::NormPostFfn1,
    Gemma4Op::NormPreFfn2,
    Gemma4Op::NormRouter,
    Gemma4Op::ScaleRouter,
    Gemma4Op::ProjRouter,
    Gemma4Op::MoeSoftmaxTopK,
    Gemma4Op::MoeExperts,
    Gemma4Op::NormPostFfn2,
    Gemma4Op::ResidualAddMoe,
    Gemma4Op::NormOuterFfn,
    Gemma4Op::RestoreResidual,
    Gemma4Op::ResidualAddFfn,
    Gemma4Op::ScaleLayer,
];

#[cfg(test)]
fn gemma4_op_sequence(variant: Gemma4Variant) -> &'static [Gemma4Op] {
    match variant {
        Gemma4Variant::SlidingDense => &SLIDING_DENSE_OPS,
        Gemma4Variant::SlidingMoe => &SLIDING_MOE_OPS,
        Gemma4Variant::FullDense => &FULL_DENSE_OPS,
        Gemma4Variant::FullMoe => &FULL_MOE_OPS,
    }
}

// ── Borrowed Step lowering ───────────────────────────────────────────────

fn execute_bound_gemma4_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    layer_idx: usize,
    steps: &[Step<'_>],
) -> HipResult<()> {
    execute_steps(gpu, ctx, steps).map_err(|error| {
        hip_bridge::HipError::new(0, &format!("Gemma4 layer {layer_idx}: {error}"))
    })
}

fn execute_bound_gemma4_steps_dumped(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    layer_idx: usize,
    steps: &[Step<'_>],
    boundaries: &[(usize, &str, &GpuTensor)],
    dump: &mut Gemma4LayerDump,
) -> HipResult<()> {
    let mut start = 0usize;
    for &(end, label, tensor) in boundaries {
        debug_assert!(start <= end && end <= steps.len());
        execute_bound_gemma4_steps(gpu, ctx, layer_idx, &steps[start..end])?;
        dump.capture_boundary(gpu, layer_idx, label, tensor);
        start = end;
    }
    execute_bound_gemma4_steps(gpu, ctx, layer_idx, &steps[start..])
}

fn validate_gemma4_layer(
    layer_idx: usize,
    layer_type: LayerType,
    layer: &LayerWeights,
    config: &Gemma4Config,
) -> HipResult<()> {
    let actual = match layer {
        LayerWeights::Sliding(_) => LayerType::Sliding,
        LayerWeights::Full(_) => LayerType::Full,
    };
    if actual != layer_type {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "Gemma4 layer {layer_idx}: configured {layer_type:?} but weights are {actual:?}"
            ),
        ));
    }
    if matches!(
        gemma4_variant_of(layer_type, layer),
        Gemma4Variant::SlidingMoe | Gemma4Variant::FullMoe
    ) && config.top_k_experts != 8
    {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "Gemma4 layer {layer_idx}: top_k_experts={} unsupported (expected 8)",
                config.top_k_experts
            ),
        ));
    }
    Ok(())
}

fn gemma4_attention_params<'a>(
    gpu: &Gpu,
    config: &Gemma4Config,
    pos: usize,
    kv: &'a mut llama::KvCache,
    kv_layer_idx: usize,
    scratch: &'a Gemma4Scratch,
    head_dim: usize,
    n_kv_heads: usize,
    window: usize,
) -> HipResult<(KvTierPlan, AttnParams<'a>)> {
    let tier_inputs = KvTierInputs {
        quant_asym4: kv.quant_asym4,
        quant_asym3: kv.quant_asym3,
        quant_asym2: kv.quant_asym2,
        quant_q8: kv.quant_q8,
        quant_fwht: kv.quant_fwht,
        quant_hfq4: false,
        quant_q4: false,
        quant_int8: false,
        quant_hfq8: false,
        f32_policy: hipfire_dispatch::families::kv_tier::F32AttnPolicy::Simple,
        v_mode_bits: kv.v_mode_bits(),
        pos,
        flash_mode: 2,
        capture_mode: gpu.graphs.capture_mode,
        batch_size: 1,
        is_tree: false,
        is_boundary: false,
        q8_windowed: false,
        window: window as i32,
    };
    let plan = KvTierPlan::derive(tier_inputs)
        .map_err(|error| hip_bridge::HipError::new(0, &format!("{error:?}")))?;
    let io = AttnParams {
        q: &scratch.q,
        k: &scratch.k,
        v: &scratch.v,
        k_cache: &kv.k_gpu[kv_layer_idx],
        v_cache: &kv.v_gpu[kv_layer_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &scratch.pos_buf,
        pos,
        positions: None,
        n_heads: config.n_heads,
        n_kv_heads,
        head_dim,
        physical_cap: kv.max_seq,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(&scratch.flash_partials),
        givens_cos: kv.givens_cos.as_ref(),
        givens_sin: kv.givens_sin.as_ref(),
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: None,
        output: &scratch.attn_out,
    };
    Ok((plan, io))
}

fn execute_gemma4_layer(
    gpu: &mut Gpu,
    config: &Gemma4Config,
    layer_type: LayerType,
    layer: &LayerWeights,
    scratch: &Gemma4Scratch,
    pos: usize,
    kv_sliding: &mut llama::KvCache,
    kv_full: &mut llama::KvCache,
    sliding_kv_idx: usize,
    full_kv_idx: usize,
    layer_idx: usize,
    layer_dump: Option<&mut Gemma4LayerDump>,
) -> HipResult<()> {
    validate_gemma4_layer(layer_idx, layer_type, layer, config)?;
    let ctx = DispatchCtx::new(gpu);
    match (gemma4_variant_of(layer_type, layer), layer) {
        (Gemma4Variant::SlidingDense, LayerWeights::Sliding(weights)) => {
            execute_sliding_dense_steps(
                gpu,
                &ctx,
                layer_idx,
                weights,
                config,
                scratch,
                pos,
                kv_sliding,
                sliding_kv_idx,
                layer_dump,
            )
        }
        (Gemma4Variant::SlidingMoe, LayerWeights::Sliding(weights)) => execute_sliding_moe_steps(
            gpu,
            &ctx,
            layer_idx,
            weights,
            config,
            scratch,
            pos,
            kv_sliding,
            sliding_kv_idx,
            layer_dump,
        ),
        (Gemma4Variant::FullDense, LayerWeights::Full(weights)) => execute_full_dense_steps(
            gpu,
            &ctx,
            layer_idx,
            weights,
            config,
            scratch,
            pos,
            kv_full,
            full_kv_idx,
            layer_dump,
        ),
        (Gemma4Variant::FullMoe, LayerWeights::Full(weights)) => execute_full_moe_steps(
            gpu,
            &ctx,
            layer_idx,
            weights,
            config,
            scratch,
            pos,
            kv_full,
            full_kv_idx,
            layer_dump,
        ),
        _ => unreachable!("validate_gemma4_layer rejected the mismatch"),
    }
}

fn execute_sliding_dense_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    layer_idx: usize,
    weights: &SlidingLayerWeights,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    kv_layer_idx: usize,
    layer_dump: Option<&mut Gemma4LayerDump>,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.sliding_head_dim;
    let n_kv_heads = config.sliding_n_kv_heads;
    let (plan, io) = gemma4_attention_params(
        gpu,
        config,
        pos,
        kv_cache,
        kv_layer_idx,
        scratch,
        head_dim,
        n_kv_heads,
        config.sliding_window,
    )?;
    let wr_q = weights.q_proj.dispatch_ref();
    let wr_k = weights.k_proj.dispatch_ref();
    let wr_v = weights.v_proj.dispatch_ref();
    let wr_o = weights.o_proj.dispatch_ref();
    let wr_gate = weights.gate_proj.dispatch_ref();
    let wr_up = weights.up_proj.dispatch_ref();
    let wr_down = weights.down_proj.dispatch_ref();
    let steps: [Step<'_>; 23] = [
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.input_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_q,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.q,
        },
        Step::Gemv {
            w: &wr_k,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.k,
        },
        Step::Gemv {
            w: &wr_v,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.v,
        },
        Step::QkNorm {
            x: &scratch.q,
            weight: &weights.q_norm,
            n_groups: config.n_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.k,
            weight: &weights.k_norm,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.v,
            weight: &scratch.v_norm_ones_full,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::Scale {
            x: &scratch.q,
            scale: (head_dim as f32).sqrt(),
        },
        Step::Rope {
            q: &scratch.q,
            k: &scratch.k,
            pos_buf: &scratch.pos_buf,
            n_heads: config.n_heads,
            n_kv_heads,
            head_dim,
            theta: config.sliding_rope_theta,
        },
        Step::Attend { plan, io },
        Step::Gemv {
            w: &wr_o,
            input: GemvInput::Raw(&scratch.attn_out),
            out: &scratch.tmp,
        },
        Step::RmsNorm {
            x: &scratch.tmp,
            weight: &weights.post_attention_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Copy {
            src: &scratch.x,
            dst: &scratch.residual,
            bytes: dim * 4,
        },
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.pre_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_gate,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.gate_ffn,
        },
        Step::Gemv {
            w: &wr_up,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.up_ffn,
        },
        Step::GeluTanhMul {
            gate: &scratch.gate_ffn,
            up: &scratch.up_ffn,
            out: &scratch.ffn_hidden,
            n: config.hidden_dim,
        },
        Step::Gemv {
            w: &wr_down,
            input: GemvInput::Raw(&scratch.ffn_hidden),
            out: &scratch.ffn_out,
        },
        Step::RmsNorm {
            x: &scratch.ffn_out,
            weight: &weights.post_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Copy {
            src: &scratch.residual,
            dst: &scratch.x,
            bytes: dim * 4,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Scale {
            x: &scratch.x,
            scale: weights.layer_scalar_host,
        },
    ];
    if let Some(dump) = layer_dump {
        if dump.boundaries {
            return execute_bound_gemma4_steps_dumped(
                gpu,
                ctx,
                layer_idx,
                &steps,
                &[
                    (11, "attention_output", &scratch.tmp),
                    (12, "attention_norm", &scratch.tmp),
                    (13, "attention_residual", &scratch.x),
                    (15, "pre_ffn_norm", &scratch.tmp),
                    (19, "dense_ffn", &scratch.ffn_out),
                    (20, "dense_ffn_norm", &scratch.tmp),
                ],
                dump,
            );
        }
    }
    execute_bound_gemma4_steps(gpu, ctx, layer_idx, &steps)
}

fn execute_full_dense_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    layer_idx: usize,
    weights: &FullLayerWeights,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    kv_layer_idx: usize,
    layer_dump: Option<&mut Gemma4LayerDump>,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.full_head_dim;
    let n_kv_heads = config.full_n_kv_heads;
    let (plan, io) = gemma4_attention_params(
        gpu,
        config,
        pos,
        kv_cache,
        kv_layer_idx,
        scratch,
        head_dim,
        n_kv_heads,
        0,
    )?;
    let wr_q = weights.q_proj.dispatch_ref();
    let wr_k = weights.k_proj.dispatch_ref();
    let wr_o = weights.o_proj.dispatch_ref();
    let wr_gate = weights.gate_proj.dispatch_ref();
    let wr_up = weights.up_proj.dispatch_ref();
    let wr_down = weights.down_proj.dispatch_ref();
    let steps: [Step<'_>; 23] = [
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.input_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_q,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.q,
        },
        Step::Gemv {
            w: &wr_k,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.k,
        },
        Step::Copy {
            src: &scratch.k,
            dst: &scratch.v,
            bytes: n_kv_heads * head_dim * 4,
        },
        Step::QkNorm {
            x: &scratch.q,
            weight: &weights.q_norm,
            n_groups: config.n_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.k,
            weight: &weights.k_norm,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.v,
            weight: &scratch.v_norm_ones_full,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::Scale {
            x: &scratch.q,
            scale: (head_dim as f32).sqrt(),
        },
        Step::RopePartial {
            q: &scratch.q,
            k: &scratch.k,
            pos_buf: &scratch.pos_buf,
            n_heads: config.n_heads,
            n_kv_heads,
            head_dim,
            n_rot_pairs: ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize,
            theta: config.full_rope_theta,
        },
        Step::Attend { plan, io },
        Step::Gemv {
            w: &wr_o,
            input: GemvInput::Raw(&scratch.attn_out),
            out: &scratch.tmp,
        },
        Step::RmsNorm {
            x: &scratch.tmp,
            weight: &weights.post_attention_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Copy {
            src: &scratch.x,
            dst: &scratch.residual,
            bytes: dim * 4,
        },
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.pre_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_gate,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.gate_ffn,
        },
        Step::Gemv {
            w: &wr_up,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.up_ffn,
        },
        Step::GeluTanhMul {
            gate: &scratch.gate_ffn,
            up: &scratch.up_ffn,
            out: &scratch.ffn_hidden,
            n: config.hidden_dim,
        },
        Step::Gemv {
            w: &wr_down,
            input: GemvInput::Raw(&scratch.ffn_hidden),
            out: &scratch.ffn_out,
        },
        Step::RmsNorm {
            x: &scratch.ffn_out,
            weight: &weights.post_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Copy {
            src: &scratch.residual,
            dst: &scratch.x,
            bytes: dim * 4,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Scale {
            x: &scratch.x,
            scale: weights.layer_scalar_host,
        },
    ];
    if let Some(dump) = layer_dump {
        if dump.boundaries {
            return execute_bound_gemma4_steps_dumped(
                gpu,
                ctx,
                layer_idx,
                &steps,
                &[
                    (11, "attention_output", &scratch.tmp),
                    (12, "attention_norm", &scratch.tmp),
                    (13, "attention_residual", &scratch.x),
                    (15, "pre_ffn_norm", &scratch.tmp),
                    (19, "dense_ffn", &scratch.ffn_out),
                    (20, "dense_ffn_norm", &scratch.tmp),
                ],
                dump,
            );
        }
    }
    execute_bound_gemma4_steps(gpu, ctx, layer_idx, &steps)
}

fn execute_sliding_moe_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    layer_idx: usize,
    weights: &SlidingLayerWeights,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    kv_layer_idx: usize,
    layer_dump: Option<&mut Gemma4LayerDump>,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.sliding_head_dim;
    let n_kv_heads = config.sliding_n_kv_heads;
    let moe = weights.moe.as_ref().expect("validated MoE layer");
    let (plan, io) = gemma4_attention_params(
        gpu,
        config,
        pos,
        kv_cache,
        kv_layer_idx,
        scratch,
        head_dim,
        n_kv_heads,
        config.sliding_window,
    )?;
    let wr_q = weights.q_proj.dispatch_ref();
    let wr_k = weights.k_proj.dispatch_ref();
    let wr_v = weights.v_proj.dispatch_ref();
    let wr_o = weights.o_proj.dispatch_ref();
    let wr_gate = weights.gate_proj.dispatch_ref();
    let wr_up = weights.up_proj.dispatch_ref();
    let wr_down = weights.down_proj.dispatch_ref();
    let wr_router = moe.router_proj.dispatch_ref();
    let experts = MoeGeluExpertsRef {
        gate_up_pool: &moe.experts_gate_up_pool,
        down_pool: &moe.experts_down_pool,
        gate_up_ptrs: &moe.experts_gate_up_ptrs,
        down_ptrs: &moe.experts_down_ptrs,
        gate_up_dtype: moe.experts[0].gate_up_proj.gpu_dtype,
        down_dtype: moe.experts[0].down_proj.gpu_dtype,
        gate_up_bytes: moe.gate_up_bytes,
        down_bytes: moe.down_bytes,
        n_experts: config.num_experts,
    };
    let steps: [Step<'_>; 32] = [
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.input_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_q,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.q,
        },
        Step::Gemv {
            w: &wr_k,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.k,
        },
        Step::Gemv {
            w: &wr_v,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.v,
        },
        Step::QkNorm {
            x: &scratch.q,
            weight: &weights.q_norm,
            n_groups: config.n_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.k,
            weight: &weights.k_norm,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.v,
            weight: &scratch.v_norm_ones_full,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::Scale {
            x: &scratch.q,
            scale: (head_dim as f32).sqrt(),
        },
        Step::Rope {
            q: &scratch.q,
            k: &scratch.k,
            pos_buf: &scratch.pos_buf,
            n_heads: config.n_heads,
            n_kv_heads,
            head_dim,
            theta: config.sliding_rope_theta,
        },
        Step::Attend { plan, io },
        Step::Gemv {
            w: &wr_o,
            input: GemvInput::Raw(&scratch.attn_out),
            out: &scratch.tmp,
        },
        Step::RmsNorm {
            x: &scratch.tmp,
            weight: &weights.post_attention_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Copy {
            src: &scratch.x,
            dst: &scratch.residual,
            bytes: dim * 4,
        },
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.pre_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_gate,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.gate_ffn,
        },
        Step::Gemv {
            w: &wr_up,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.up_ffn,
        },
        Step::GeluTanhMul {
            gate: &scratch.gate_ffn,
            up: &scratch.up_ffn,
            out: &scratch.ffn_hidden,
            n: config.hidden_dim,
        },
        Step::Gemv {
            w: &wr_down,
            input: GemvInput::Raw(&scratch.ffn_hidden),
            out: &scratch.ffn_out,
        },
        Step::RmsNorm {
            x: &scratch.ffn_out,
            weight: &moe.post_feedforward_layernorm_1,
            out: &scratch.moe_cur_mlp,
            eps: config.norm_eps,
        },
        Step::RmsNorm {
            x: &scratch.residual,
            weight: &moe.pre_feedforward_layernorm_2,
            out: &scratch.moe_pre2,
            eps: config.norm_eps,
        },
        Step::RmsNorm {
            x: &scratch.residual,
            weight: &moe.router_scale,
            out: &scratch.moe_router_in,
            eps: config.norm_eps,
        },
        Step::Scale {
            x: &scratch.moe_router_in,
            scale: 1.0 / (dim as f32).sqrt(),
        },
        Step::Gemv {
            w: &wr_router,
            input: GemvInput::Raw(&scratch.moe_router_in),
            out: &scratch.moe_router_logits,
        },
        Step::MoeSoftmaxTopK {
            logits: &scratch.moe_router_logits,
            topk_indices: &scratch.moe_topk_indices,
            topk_weights: &scratch.moe_topk_weights,
            n_exp: config.num_experts,
            norm_topk_prob: true,
            backend: MoeRouterBackend::FusedSoftmaxTopK,
        },
        Step::MoeGeluExperts {
            experts,
            input: &scratch.moe_pre2,
            input_rot: &scratch.moe_pre2_rot,
            topk_indices: &scratch.moe_topk_indices,
            topk_weights: &scratch.moe_topk_weights,
            expert_scales: &moe.per_expert_scale,
            expert_scales_host: &moe.per_expert_scale_host,
            gate: &scratch.moe_expert_gate_batch,
            up: &scratch.moe_expert_up_batch,
            hidden: &scratch.moe_expert_hidden_batch,
            out: &scratch.moe_cur_moe,
            hidden_dim: dim,
            expert_dim: config.moe_intermediate_size,
            k_top: config.top_k_experts,
        },
        Step::RmsNorm {
            x: &scratch.moe_cur_moe,
            weight: &moe.post_feedforward_layernorm_2,
            out: &scratch.moe_cur_moe,
            eps: config.norm_eps,
        },
        Step::ResidualAdd {
            x: &scratch.moe_cur_mlp,
            y: &scratch.moe_cur_moe,
            dim,
        },
        Step::RmsNorm {
            x: &scratch.moe_cur_mlp,
            weight: &weights.post_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Copy {
            src: &scratch.residual,
            dst: &scratch.x,
            bytes: dim * 4,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Scale {
            x: &scratch.x,
            scale: weights.layer_scalar_host,
        },
    ];
    if let Some(dump) = layer_dump {
        if dump.boundaries {
            return execute_bound_gemma4_steps_dumped(
                gpu,
                ctx,
                layer_idx,
                &steps,
                &[
                    (11, "attention_output", &scratch.tmp),
                    (12, "attention_norm", &scratch.tmp),
                    (13, "attention_residual", &scratch.x),
                    (15, "pre_ffn_norm", &scratch.tmp),
                    (19, "dense_ffn", &scratch.ffn_out),
                    (20, "dense_branch_norm", &scratch.moe_cur_mlp),
                    (21, "moe_pre2", &scratch.moe_pre2),
                    (24, "router_logits", &scratch.moe_router_logits),
                    (26, "moe_branch", &scratch.moe_cur_moe),
                    (27, "moe_branch_norm", &scratch.moe_cur_moe),
                    (28, "moe_combined", &scratch.moe_cur_mlp),
                    (29, "outer_norm", &scratch.tmp),
                ],
                dump,
            );
        }
    }
    execute_bound_gemma4_steps(gpu, ctx, layer_idx, &steps)
}

fn execute_full_moe_steps(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    layer_idx: usize,
    weights: &FullLayerWeights,
    config: &Gemma4Config,
    scratch: &Gemma4Scratch,
    pos: usize,
    kv_cache: &mut llama::KvCache,
    kv_layer_idx: usize,
    layer_dump: Option<&mut Gemma4LayerDump>,
) -> HipResult<()> {
    let dim = config.dim;
    let head_dim = config.full_head_dim;
    let n_kv_heads = config.full_n_kv_heads;
    let moe = weights.moe.as_ref().expect("validated MoE layer");
    let (plan, io) = gemma4_attention_params(
        gpu,
        config,
        pos,
        kv_cache,
        kv_layer_idx,
        scratch,
        head_dim,
        n_kv_heads,
        0,
    )?;
    let wr_q = weights.q_proj.dispatch_ref();
    let wr_k = weights.k_proj.dispatch_ref();
    let wr_o = weights.o_proj.dispatch_ref();
    let wr_gate = weights.gate_proj.dispatch_ref();
    let wr_up = weights.up_proj.dispatch_ref();
    let wr_down = weights.down_proj.dispatch_ref();
    let wr_router = moe.router_proj.dispatch_ref();
    let experts = MoeGeluExpertsRef {
        gate_up_pool: &moe.experts_gate_up_pool,
        down_pool: &moe.experts_down_pool,
        gate_up_ptrs: &moe.experts_gate_up_ptrs,
        down_ptrs: &moe.experts_down_ptrs,
        gate_up_dtype: moe.experts[0].gate_up_proj.gpu_dtype,
        down_dtype: moe.experts[0].down_proj.gpu_dtype,
        gate_up_bytes: moe.gate_up_bytes,
        down_bytes: moe.down_bytes,
        n_experts: config.num_experts,
    };
    let steps: [Step<'_>; 32] = [
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.input_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_q,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.q,
        },
        Step::Gemv {
            w: &wr_k,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.k,
        },
        Step::Copy {
            src: &scratch.k,
            dst: &scratch.v,
            bytes: n_kv_heads * head_dim * 4,
        },
        Step::QkNorm {
            x: &scratch.q,
            weight: &weights.q_norm,
            n_groups: config.n_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.k,
            weight: &weights.k_norm,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::QkNorm {
            x: &scratch.v,
            weight: &scratch.v_norm_ones_full,
            n_groups: n_kv_heads,
            head_dim,
            eps: config.norm_eps,
        },
        Step::Scale {
            x: &scratch.q,
            scale: (head_dim as f32).sqrt(),
        },
        Step::RopePartial {
            q: &scratch.q,
            k: &scratch.k,
            pos_buf: &scratch.pos_buf,
            n_heads: config.n_heads,
            n_kv_heads,
            head_dim,
            n_rot_pairs: ((head_dim as f32) * config.full_partial_rotary_factor * 0.5) as usize,
            theta: config.full_rope_theta,
        },
        Step::Attend { plan, io },
        Step::Gemv {
            w: &wr_o,
            input: GemvInput::Raw(&scratch.attn_out),
            out: &scratch.tmp,
        },
        Step::RmsNorm {
            x: &scratch.tmp,
            weight: &weights.post_attention_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Copy {
            src: &scratch.x,
            dst: &scratch.residual,
            bytes: dim * 4,
        },
        Step::RmsNorm {
            x: &scratch.x,
            weight: &weights.pre_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Gemv {
            w: &wr_gate,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.gate_ffn,
        },
        Step::Gemv {
            w: &wr_up,
            input: GemvInput::Raw(&scratch.tmp),
            out: &scratch.up_ffn,
        },
        Step::GeluTanhMul {
            gate: &scratch.gate_ffn,
            up: &scratch.up_ffn,
            out: &scratch.ffn_hidden,
            n: config.hidden_dim,
        },
        Step::Gemv {
            w: &wr_down,
            input: GemvInput::Raw(&scratch.ffn_hidden),
            out: &scratch.ffn_out,
        },
        Step::RmsNorm {
            x: &scratch.ffn_out,
            weight: &moe.post_feedforward_layernorm_1,
            out: &scratch.moe_cur_mlp,
            eps: config.norm_eps,
        },
        Step::RmsNorm {
            x: &scratch.residual,
            weight: &moe.pre_feedforward_layernorm_2,
            out: &scratch.moe_pre2,
            eps: config.norm_eps,
        },
        Step::RmsNorm {
            x: &scratch.residual,
            weight: &moe.router_scale,
            out: &scratch.moe_router_in,
            eps: config.norm_eps,
        },
        Step::Scale {
            x: &scratch.moe_router_in,
            scale: 1.0 / (dim as f32).sqrt(),
        },
        Step::Gemv {
            w: &wr_router,
            input: GemvInput::Raw(&scratch.moe_router_in),
            out: &scratch.moe_router_logits,
        },
        Step::MoeSoftmaxTopK {
            logits: &scratch.moe_router_logits,
            topk_indices: &scratch.moe_topk_indices,
            topk_weights: &scratch.moe_topk_weights,
            n_exp: config.num_experts,
            norm_topk_prob: true,
            backend: MoeRouterBackend::FusedSoftmaxTopK,
        },
        Step::MoeGeluExperts {
            experts,
            input: &scratch.moe_pre2,
            input_rot: &scratch.moe_pre2_rot,
            topk_indices: &scratch.moe_topk_indices,
            topk_weights: &scratch.moe_topk_weights,
            expert_scales: &moe.per_expert_scale,
            expert_scales_host: &moe.per_expert_scale_host,
            gate: &scratch.moe_expert_gate_batch,
            up: &scratch.moe_expert_up_batch,
            hidden: &scratch.moe_expert_hidden_batch,
            out: &scratch.moe_cur_moe,
            hidden_dim: dim,
            expert_dim: config.moe_intermediate_size,
            k_top: config.top_k_experts,
        },
        Step::RmsNorm {
            x: &scratch.moe_cur_moe,
            weight: &moe.post_feedforward_layernorm_2,
            out: &scratch.moe_cur_moe,
            eps: config.norm_eps,
        },
        Step::ResidualAdd {
            x: &scratch.moe_cur_mlp,
            y: &scratch.moe_cur_moe,
            dim,
        },
        Step::RmsNorm {
            x: &scratch.moe_cur_mlp,
            weight: &weights.post_feedforward_layernorm,
            out: &scratch.tmp,
            eps: config.norm_eps,
        },
        Step::Copy {
            src: &scratch.residual,
            dst: &scratch.x,
            bytes: dim * 4,
        },
        Step::ResidualAdd {
            x: &scratch.x,
            y: &scratch.tmp,
            dim,
        },
        Step::Scale {
            x: &scratch.x,
            scale: weights.layer_scalar_host,
        },
    ];
    if let Some(dump) = layer_dump {
        if dump.boundaries {
            return execute_bound_gemma4_steps_dumped(
                gpu,
                ctx,
                layer_idx,
                &steps,
                &[
                    (11, "attention_output", &scratch.tmp),
                    (12, "attention_norm", &scratch.tmp),
                    (13, "attention_residual", &scratch.x),
                    (15, "pre_ffn_norm", &scratch.tmp),
                    (19, "dense_ffn", &scratch.ffn_out),
                    (20, "dense_branch_norm", &scratch.moe_cur_mlp),
                    (21, "moe_pre2", &scratch.moe_pre2),
                    (24, "router_logits", &scratch.moe_router_logits),
                    (26, "moe_branch", &scratch.moe_cur_moe),
                    (27, "moe_branch_norm", &scratch.moe_cur_moe),
                    (28, "moe_combined", &scratch.moe_cur_mlp),
                    (29, "outer_norm", &scratch.tmp),
                ],
                dump,
            );
        }
    }
    execute_bound_gemma4_steps(gpu, ctx, layer_idx, &steps)
}

/// Typed-Step decode body used by `forward_scratch`.
fn forward_scratch_inner(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    pos: usize,
    kv_sliding: &mut llama::KvCache,
    kv_full: &mut llama::KvCache,
    scratch: &Gemma4Scratch,
    mut layer_dump: Option<&mut Gemma4LayerDump>,
) -> HipResult<()> {
    let mut sliding_kv_idx = 0usize;
    let mut full_kv_idx = 0usize;

    for (layer_idx, layer_type) in config.layer_types.iter().copied().enumerate() {
        execute_gemma4_layer(
            gpu,
            config,
            layer_type,
            &weights.layers[layer_idx],
            scratch,
            pos,
            kv_sliding,
            kv_full,
            sliding_kv_idx,
            full_kv_idx,
            layer_idx,
            layer_dump.as_mut().map(|dump| &mut **dump),
        )?;
        if let Some(dump) = layer_dump.as_mut() {
            dump.capture(gpu, format!("L{layer_idx}_layer_out"), &scratch.x);
        }
        match layer_type {
            LayerType::Sliding => sliding_kv_idx += 1,
            LayerType::Full => full_kv_idx += 1,
        }
    }

    // Output stage: final norm + lm_head + softcap (same as hand path).
    gpu.rmsnorm_f32(
        &scratch.x,
        &weights.final_norm,
        &scratch.tmp,
        config.norm_eps,
    )?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = weights.lm_head.dispatch_ref();
        execute_steps(
            gpu,
            &ctx,
            &[Step::Gemv {
                w: &wr,
                input: GemvInput::Raw(&scratch.tmp),
                out: &scratch.logits,
            }],
        )
        .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
    }
    if config.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(
            &scratch.logits,
            config.vocab_size,
            config.final_logit_softcapping,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gemma_cache_rollback_append_without_window_overwrite_restores_committed() {
        let mut cursor = Gemma4Cursor::new(3);
        cursor.set_position(7);

        assert_eq!(
            cursor.rollback_working_request(3, true, None),
            GemmaRollback::RestoredCommitted
        );
        assert_eq!(cursor.position(), 3);
    }

    #[test]
    fn gemma_cache_rollback_overwrite_of_committed_sliding_rows_invalidates() {
        let mut cursor = Gemma4Cursor::new(3);
        cursor.set_position(9);

        assert_eq!(
            cursor.rollback_working_request(3, true, Some(2)),
            GemmaRollback::Invalidated
        );
        assert_eq!(cursor.position(), 0);
    }

    #[test]
    fn gemma_cache_rollback_identity_mismatch_invalidates() {
        let mut cursor = Gemma4Cursor::new(3);
        cursor.set_position(7);

        assert_eq!(
            cursor.rollback_working_request(3, false, None),
            GemmaRollback::Invalidated
        );
        assert_eq!(cursor.position(), 0);
    }

    #[test]
    fn gemma_cache_rollback_overwrite_after_request_start_is_safe() {
        let mut cursor = Gemma4Cursor::new(3);
        cursor.set_position(7);

        assert_eq!(
            cursor.rollback_working_request(3, true, Some(3)),
            GemmaRollback::RestoredCommitted
        );
        assert_eq!(cursor.position(), 3);
    }

    #[test]
    fn failed_kv_teardown_is_retained_with_exact_owner_labels() {
        let failure = kv_cleanup_failure_from_remaining(vec![(
            "kv_full.k_gpu[0]".to_string(),
            GpuTensor::null_for_test(),
        )]);
        assert_eq!(failure.num_failed(), 1);
        assert_eq!(
            failure.error_summaries(),
            vec!["kv_full.k_gpu[0]: kv free_checked failed"]
        );
        let tracked = tracked_kv_cleanup_failure(failure, 0);
        assert_eq!(tracked.num_failed(), 1);
        assert!(tracked.error_summaries()[0].contains("kv_full.k_gpu[0]"));
    }

    use super::{gemma4_op_sequence, Gemma4Op, Gemma4Variant};

    #[test]
    fn sliding_dense_step_order_is_total() {
        assert_eq!(
            gemma4_op_sequence(Gemma4Variant::SlidingDense),
            &[
                Gemma4Op::NormInput,
                Gemma4Op::ProjQ,
                Gemma4Op::ProjK,
                Gemma4Op::ProjV,
                Gemma4Op::NormQ,
                Gemma4Op::NormK,
                Gemma4Op::NormV,
                Gemma4Op::ScaleQ,
                Gemma4Op::RopeFull,
                Gemma4Op::Attend,
                Gemma4Op::ProjO,
                Gemma4Op::NormPostAttn,
                Gemma4Op::ResidualAddAttn,
                Gemma4Op::SaveResidual,
                Gemma4Op::NormPreFfn,
                Gemma4Op::ProjGate,
                Gemma4Op::ProjUp,
                Gemma4Op::GeluTanhMul,
                Gemma4Op::ProjDown,
                Gemma4Op::NormPostFfn,
                Gemma4Op::RestoreResidual,
                Gemma4Op::ResidualAddFfn,
                Gemma4Op::ScaleLayer,
            ]
        );
    }

    #[test]
    fn full_attention_copies_k_before_normalization_and_uses_partial_rope() {
        let ops = gemma4_op_sequence(Gemma4Variant::FullDense);
        let copy = ops.iter().position(|op| *op == Gemma4Op::CopyKToV).unwrap();
        let norm_k = ops.iter().position(|op| *op == Gemma4Op::NormK).unwrap();
        let norm_v = ops.iter().position(|op| *op == Gemma4Op::NormV).unwrap();
        assert!(copy < norm_k && copy < norm_v);
        assert!(ops.contains(&Gemma4Op::RopePartial));
        assert!(!ops.contains(&Gemma4Op::ProjV));
    }

    #[test]
    fn moe_replaces_dense_post_ffn_norm_once() {
        for variant in [Gemma4Variant::SlidingMoe, Gemma4Variant::FullMoe] {
            let ops = gemma4_op_sequence(variant);
            assert_eq!(
                ops.iter().filter(|op| **op == Gemma4Op::MoeExperts).count(),
                1
            );
            assert!(!ops.contains(&Gemma4Op::NormPostFfn));
            assert_eq!(
                ops.iter()
                    .filter(|op| **op == Gemma4Op::NormOuterFfn)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn lowered_allocation_telemetry_formats_size_aware_fields() {
        let telemetry = Gemma4AllocationTelemetry {
            phase: "unload",
            cycle: 4,
            owner_bytes: 1280,
            live_owner_bytes: 1280,
            pool_bytes: 2048,
            free_device_bytes: Some(4096),
            graph_resident: true,
            graph_blob_count: 7,
            module_count: 3,
            freed_owner_labels: vec!["weights".into(), "scratch".into()],
        };
        let line = telemetry.format_line();
        assert!(line.contains("phase=unload"));
        assert!(line.contains("cycle=4"));
        assert!(line.contains("owner_bytes=1280"));
        assert!(line.contains("live_owner_bytes=1280"));
        assert!(line.contains("pool_bytes=2048"));
        assert!(line.contains("free_device_bytes=4096"));
        assert!(line.contains("graph_resident=true"));
        assert!(line.contains("module_count=3"));
        assert!(line.contains("freed_owner_labels=weights,scratch"));
    }

    #[test]
    fn lowered_allocation_telemetry_is_disabled_without_opt_in() {
        assert!(!allocation_telemetry_enabled_value(None));
        assert!(!allocation_telemetry_enabled_value(Some("0")));
        assert!(allocation_telemetry_enabled_value(Some("1")));
    }

    #[test]
    fn layer_dump_config_is_silent_without_path_and_position() {
        assert!(parse_layer_dump_config(None, None, None).is_none());
        assert!(parse_layer_dump_config(Some("/tmp/gemma.json".into()), None, None).is_none());
        assert!(parse_layer_dump_config(
            Some("/tmp/gemma.json".into()),
            Some("not-a-position"),
            Some("1")
        )
        .is_none());
    }

    #[test]
    fn layer_dump_config_parses_position_and_boundary_gate() {
        let config = parse_layer_dump_config(Some("/tmp/gemma.json".into()), Some("4"), Some("1"))
            .expect("valid dump config");
        assert_eq!(config.path, "/tmp/gemma.json");
        assert_eq!(config.position, 4);
        assert!(config.boundaries);

        let config = parse_layer_dump_config(Some("/tmp/gemma.json".into()), Some("0"), Some("0"))
            .expect("valid dump config");
        assert!(!config.boundaries);
    }
    #[test]
    #[ignore = "requires an AMD GPU"]
    fn lowered_teardown_frees_pos_sidecars_aliases_and_moe_pools() {
        use rdna_compute::{DType, Gpu, GpuTensor};

        static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: no GPU");
            return;
        };
        let vram_free = |gpu: &Gpu| gpu.hip.get_vram_info().expect("hipMemGetInfo").0;
        let tensor = |gpu: &mut Gpu| gpu.zeros(&[1], DType::F32).expect("tiny tensor");
        let weight = |gpu: &mut Gpu| WeightTensor {
            buf: tensor(gpu),
            gpu_dtype: DType::F32,
            m: 1,
            k: 1,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };

        let config = Gemma4Config {
            dim: 4,
            n_layers: 1,
            vocab_size: 8,
            norm_eps: 1e-6,
            bos_token: 2,
            eos_token: 1,
            pad_token: 0,
            n_heads: 1,
            sliding_head_dim: 32,
            sliding_n_kv_heads: 1,
            sliding_rope_theta: 10_000.0,
            sliding_window: 128,
            full_head_dim: 32,
            full_n_kv_heads: 1,
            full_rope_theta: 1_000_000.0,
            full_rope_type: RopeType::Proportional,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            hidden_dim: 8,
            enable_moe_block: false,
            moe_intermediate_size: 4,
            num_experts: 0,
            top_k_experts: 0,
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            embed_scale: 2.0,
            layer_types: vec![LayerType::Sliding],
            has_vision: false,
            image_token_id: 0,
            boi_token_id: 0,
            eoi_token_id: 0,
            audio_token_id: 0,
            video_token_id: 0,
        };
        let warmup = Gemma4Scratch::new(&mut gpu, &config, 1).expect("scratch warmup");
        warmup.free_gpu(&mut gpu);
        gpu.drain_pool();
        let baseline = vram_free(&gpu);

        let scratch = Gemma4Scratch::new(&mut gpu, &config, 1).expect("tiny scratch");
        assert_eq!(scratch.pos_buf.size(), 4);
        assert!(scratch.owner_bytes() > scratch.pos_buf.size());
        scratch.free_gpu(&mut gpu);
        gpu.drain_pool();
        assert!(
            baseline.abs_diff(vram_free(&gpu)) < 64 * 1024 * 1024,
            "scratch teardown did not reclaim VRAM"
        );

        let embed_tokens = tensor(&mut gpu);
        let lm_head = {
            let alias = unsafe { embed_tokens.buf.alias() };
            WeightTensor {
                buf: GpuTensor {
                    buf: alias,
                    shape: embed_tokens.shape.clone(),
                    dtype: DType::F32,
                },
                gpu_dtype: DType::F32,
                m: 1,
                k: 1,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        };
        assert!(lm_head.buf.buf.is_borrowed());

        let gate_pool = gpu.upload_raw(&[0u8; 8], &[8]).expect("gate pool");
        let down_pool = gpu.upload_raw(&[0u8; 8], &[8]).expect("down pool");
        let expert_gate = WeightTensor {
            buf: gate_pool.sub_offset(0, 4),
            gpu_dtype: DType::Raw,
            m: 1,
            k: 1,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        let expert_down = WeightTensor {
            buf: down_pool.sub_offset(0, 4),
            gpu_dtype: DType::Raw,
            m: 1,
            k: 1,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        };
        assert!(expert_gate.buf.buf.is_borrowed());
        assert!(expert_down.buf.buf.is_borrowed());
        let moe = MoeLayerExtras {
            router_proj: weight(&mut gpu),
            router_scale: tensor(&mut gpu),
            per_expert_scale: tensor(&mut gpu),
            per_expert_scale_host: vec![1.0],
            pre_feedforward_layernorm_2: tensor(&mut gpu),
            post_feedforward_layernorm_1: tensor(&mut gpu),
            post_feedforward_layernorm_2: tensor(&mut gpu),
            experts_gate_up_pool: gate_pool,
            experts_down_pool: down_pool,
            gate_up_bytes: 4,
            down_bytes: 4,
            experts: vec![MoeExpertWeights {
                gate_up_proj: expert_gate,
                down_proj: expert_down,
            }],
            experts_gate_up_ptrs: tensor(&mut gpu),
            experts_down_ptrs: tensor(&mut gpu),
        };
        let mut q_proj = weight(&mut gpu);
        q_proj.awq_scale = Some(tensor(&mut gpu));
        let weights = Gemma4Weights {
            embed_tokens,
            embd_format: EmbeddingFormat::F32,
            lm_head,
            final_norm: tensor(&mut gpu),
            layers: vec![LayerWeights::Sliding(SlidingLayerWeights {
                input_layernorm: tensor(&mut gpu),
                post_attention_layernorm: tensor(&mut gpu),
                pre_feedforward_layernorm: tensor(&mut gpu),
                post_feedforward_layernorm: tensor(&mut gpu),
                layer_scalar: tensor(&mut gpu),
                layer_scalar_host: 1.0,
                q_proj,
                k_proj: weight(&mut gpu),
                v_proj: weight(&mut gpu),
                o_proj: weight(&mut gpu),
                q_norm: tensor(&mut gpu),
                k_norm: tensor(&mut gpu),
                gate_proj: weight(&mut gpu),
                up_proj: weight(&mut gpu),
                down_proj: weight(&mut gpu),
                moe: Some(moe),
            })],
        };
        assert!(weights.owner_bytes() > 0);
        register_live_owner_bytes(weights.owner_bytes());
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();
        assert!(
            baseline.abs_diff(vram_free(&gpu)) < 64 * 1024 * 1024,
            "weight teardown did not reclaim sidecars and MoE pools"
        );
    }
    #[test]
    fn moe_pool_dtype_accepts_passthrough_quant_types() {
        assert_eq!(gemma_moe_pool_dtype(1), Some(DType::F16));
        assert_eq!(gemma_moe_pool_dtype(2), Some(DType::F32));
        assert_eq!(gemma_moe_pool_dtype(16), Some(DType::BF16));
        assert_eq!(gemma_moe_pool_dtype(13), Some(DType::MQ4G256));
        assert_eq!(gemma_moe_pool_dtype(255), None);
    }
}
