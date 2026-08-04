//! Arch-agnostic DSpark spec-decode core. The drafter body (deepseek4 MoE/MLA
//! chain, qwen3 dense transformer) is the only arch-specific seam — see
//! [`DsparkBody`]. Everything else (main_proj ingest, markov head, confidence
//! head, window orchestration) lives here.
use crate::spec::{
    accept_greedy_prefix, MtpDrafter, MtpSpeculator, MtpWindow, SpecGrammar, SpecTarget, Speculator,
};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Per-window phase profiler (HIPFIRE_DSPARK_PROFILE=1) ─────────────────────
// Gated behind an env var; zero overhead when disabled.
#[derive(Default)]
struct DsparkProfiler {
    enabled: bool,
    kernel_profile_position: Option<usize>,
    windows: u64,
    bootstrap_ms: f64,
    draft_ms: f64,
    heads_ms: f64,
    verify_ms: f64,
    rest_ms: f64,
}

impl DsparkProfiler {
    fn new() -> Self {
        let enabled = hipfire_config::developer_var("HIPFIRE_DSPARK_PROFILE")
            .map(|s| s == "1")
            .unwrap_or(false);
        let kernel_profile_position =
            hipfire_config::developer_var("HIPFIRE_DSPARK_KERNEL_PROFILE_POSITION")
                .ok()
                .and_then(|value| value.parse::<usize>().ok());
        Self {
            enabled,
            kernel_profile_position,
            ..Default::default()
        }
    }

    /// Sync GPU and start a phase timer. Returns `Instant::now()` (always, even
    /// when disabled, to avoid branching at the call site — the cost is one
    /// `Instant::now()` per phase when profiling is off, which is nanoseconds).
    fn sync_start(&self, gpu: &mut Gpu) -> std::time::Instant {
        if self.enabled {
            let _ = gpu.hip.device_synchronize();
        }
        std::time::Instant::now()
    }

    /// Sync GPU, accumulate elapsed ms into the given bucket.
    fn sync_end(&mut self, gpu: &mut Gpu, t: std::time::Instant, bucket: u8) {
        if !self.enabled {
            return;
        }
        let _ = gpu.hip.device_synchronize();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        match bucket {
            0 => self.bootstrap_ms += ms,
            1 => self.draft_ms += ms,
            2 => self.heads_ms += ms,
            3 => self.verify_ms += ms,
            _ => self.rest_ms += ms,
        }
    }

    fn end_window(&mut self) {
        if self.enabled {
            self.windows += 1;
        }
    }

    fn print_summary(&self) {
        if !self.enabled || self.windows == 0 {
            return;
        }
        let total =
            self.bootstrap_ms + self.draft_ms + self.heads_ms + self.verify_ms + self.rest_ms;
        let pct = |x: f64| if total > 0.0 { x / total * 100.0 } else { 0.0 };
        let mean = |x: f64| x / self.windows as f64;
        eprintln!("=== HIPFIRE_DSPARK_PROFILE ({} windows) ===", self.windows);
        eprintln!(
            "  bootstrap (initial-ctx seed capture):  {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.bootstrap_ms,
            pct(self.bootstrap_ms),
            mean(self.bootstrap_ms)
        );
        eprintln!(
            "  draft_block:                          {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.draft_ms,
            pct(self.draft_ms),
            mean(self.draft_ms)
        );
        eprintln!(
            "  run_heads:                            {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.heads_ms,
            pct(self.heads_ms),
            mean(self.heads_ms)
        );
        eprintln!(
            "  verify_block:                         {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.verify_ms,
            pct(self.verify_ms),
            mean(self.verify_ms)
        );
        eprintln!(
            "  rest (accept+commit+etc):             {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.rest_ms,
            pct(self.rest_ms),
            mean(self.rest_ms)
        );
        eprintln!(
            "  total window time: {:.2} ms  mean={:.2}ms/window",
            total,
            mean(total)
        );
    }
}

#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub block_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub noise_token_id: u32,
    /// false ⇒ DFlash heads-off path (no confidence truncation).
    pub enable_confidence: bool,
    /// true ⇒ confidence head reads `normed[i]` (once-normed hidden, after
    /// `rmsnorm(x_head, stage_norm)`), matching the qwen3 reference
    /// `predict_confidence_step` which feeds `output_hidden = self.norm(hidden)`.
    /// false ⇒ confidence head reads raw `x_head[i]` (pre-norm), preserving
    /// deepseek4's original behavior (byte-identical baseline from task-5).
    /// Default: false (deepseek4-preserving).
    pub confidence_uses_normed: bool,
    /// RMSNorm epsilon used in `main_proj_ingest` and `run_heads`.
    /// Both DeepSeek V4 and Qwen3 use 1e-6; set per-arch at construction time
    /// so callers can plumb the model's actual config value.
    pub rms_norm_eps: f32,
    /// Reduced draft-vocabulary size. 0 ⇒ no reduction (drafter shares the
    /// target vocab). When >0 the drafter's lm_head + markov_w2 emit
    /// `draft_vocab_size` logits and [`DsparkWeights::d2t`] maps each draft id
    /// back to a target token id (EAGLE-3 compressed vocab, e.g. qwen35 ORNITH).
    pub draft_vocab_size: usize,
    /// Fraction of `head_dim` that RoPE rotates in the drafter body. 1.0 ⇒ full
    /// rotary (qwen3-8B); Qwen3.5 uses 0.25 (partial-interleaved, 64/256 dims).
    pub partial_rotary_factor: f32,
    /// RoPE theta (base) for the drafter body. qwen3-8B = 1e6, Qwen3.5 = 1e7.
    pub rope_theta: f32,
}

impl DsparkConfig {
    /// Parse from the HFQ `metadata_json` string (the outer
    /// `{"architecture":.., "config":{..}}` envelope). Returns `None`
    /// when `config` or `dspark_block_size` is absent — i.e. the file
    /// is not a DSpark sidecar.
    ///
    /// Reads: `dspark_block_size`, `dspark_target_layer_ids`,
    /// `dspark_markov_rank`, `dspark_noise_token_id`,
    /// `dspark_enable_confidence` (defaults to `true` when absent, matching
    /// the deepseek4 behaviour where confidence is always enabled).
    /// `dspark_confidence_uses_normed` (defaults to `false` — callers that
    /// need the once-normed input, e.g. qwen3, must set this explicitly after
    /// parsing or emit it in the sidecar metadata).
    /// `norm_eps` (defaults to `1e-6` — compatible with both DeepSeek V4 and Qwen3).
    pub fn from_metadata_json(metadata_json: &str) -> Option<Self> {
        let wrapper: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
        let cfg = wrapper.get("config")?;
        let block_size = cfg.get("dspark_block_size")?.as_u64()? as usize;
        if block_size == 0 {
            return None;
        }
        let target_layer_ids = cfg
            .get("dspark_target_layer_ids")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect::<Vec<_>>();
        let markov_rank = cfg.get("dspark_markov_rank")?.as_u64()? as usize;
        let noise_token_id = cfg.get("dspark_noise_token_id")?.as_u64()? as u32;
        let enable_confidence = cfg
            .get("dspark_enable_confidence")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let confidence_uses_normed = cfg
            .get("dspark_confidence_uses_normed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let rms_norm_eps = cfg
            .get("norm_eps")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1e-6f32);
        let draft_vocab_size = cfg
            .get("dspark_draft_vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let partial_rotary_factor = cfg
            .get("dspark_partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0);
        let rope_theta = cfg
            .get("dspark_rope_theta")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1_000_000.0);
        Some(Self {
            block_size,
            target_layer_ids,
            markov_rank,
            noise_token_id,
            enable_confidence,
            confidence_uses_normed,
            rms_norm_eps,
            draft_vocab_size,
            partial_rotary_factor,
            rope_theta,
        })
    }
}

pub struct DsparkWeights {
    pub cfg: DsparkConfig,
    pub main_proj: Option<GpuTensor>, // [dim, target_layer_ids.len()*dim]
    pub main_norm: Option<GpuTensor>,
    pub markov_w1: Option<GpuTensor>, // [vocab, rank]; None when markov_rank==0
    pub markov_w2: Option<GpuTensor>, // [vocab, rank]
    pub confidence_proj: Option<GpuTensor>, // [1, dim+rank]; None when !enable_confidence
    pub confidence_bias: Option<GpuTensor>, // [1]; qwen3 has bias, deepseek4 None
    /// Reduced-vocab draft→target token map (host). `Some` ⇒ [`run_heads`]
    /// remaps each argmaxed draft id through this before it enters the markov
    /// chain / verify. `len() == cfg.draft_vocab_size`. `None` ⇒ shared vocab.
    pub d2t: Option<Vec<u32>>,
}

/// The arch-specific seam: draft one window's block given the assembled
/// multi-slot context, returning the per-slot post-final-norm hidden (`x_head`).
///
/// `main_hidden` is `[ctx_len * n_targets * dim]` flat (row-major): the raw
/// target extract-layer hidden states for the `ctx_len` context slots.
/// `ctx_positions` has length `ctx_len` and carries the absolute RoPE position
/// of each context slot.  `ctx_len = ctx_positions.len()` and must be ≥ 1.
///
/// For `ctx_len = 1` the behaviour is identical to the original single-slot
/// contract (backward-compatible with Stage 1).  The deepseek4 body always
/// calls with `ctx_len = 1` (Stage 3 will change this); the qwen3 body uses
/// the full multi-slot `ctx_len` once Stage 2 wires it.
pub trait DsparkBody {
    fn draft_block(
        &mut self,
        gpu: &mut Gpu,
        weights: &DsparkWeights,
        main_hidden: &GpuTensor, // [ctx_len * n_targets * dim] flat
        ctx_positions: &[usize], // absolute RoPE positions; len = ctx_len
        seed: u32,
        position: usize,
        block: usize,
        x_head_out: &GpuTensor, // [block, dim] out
    ) -> Result<(), String>;
    fn block_size(&self) -> usize;
    fn free(self: Box<Self>, gpu: &mut Gpu);

    /// Clear body-local conversation/draft residuals so a cold retry cannot
    /// bleed prior-window rings (DeepSeek `dspark_swa_k`, etc.).
    ///
    /// Default: no-op for bodies without persistent draft-local GPU state.
    /// [`DsparkDrafter::mtp_reset`] always invokes this after clearing its own
    /// multi-slot context cache.
    fn reset_for_retry(&mut self, _gpu: &mut Gpu) {}
}

/// Per-window draft output: `block_size` token ids + per-slot confidence.
pub struct DraftResult {
    pub tokens: Vec<u32>,
    /// Per-slot draft logits `[block * vocab]`. Currently always EMPTY: the
    /// markov bias-add + argmax run on-GPU and no caller consumes the draft
    /// logits (the verify forward recomputes the trunk head). Populate this
    /// only if a future consumer needs them — it costs a `[block, vocab]` d2h.
    pub logits: Vec<f32>,
    pub confidence: Vec<f32>,
}

// ── private helpers (ported from hipfire-arch-deepseek4::forward) ────────────

/// Returns true when the weight's dtype requires an FWHT rotation of the
/// input before dispatch.
fn weight_needs_fwht(weight: &GpuTensor) -> bool {
    hipfire_dispatch::types::dtype_needs_rotation(weight.dtype)
}

/// Single-token embedding lookup into `out` (`[dim]` F32), dispatching on
/// the embedding-table dtype. Ported verbatim from deepseek4::forward
/// `dspark_embed_one`.
fn dspark_embed_one(
    gpu: &mut Gpu,
    table: &GpuTensor,
    out: &GpuTensor,
    token_id: u32,
    dim: usize,
) -> Result<(), String> {
    match table.dtype {
        DType::Q8_0 => gpu
            .embedding_lookup_q8(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed q8: {e:?}")),
        DType::F32 => gpu
            .embedding_lookup(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed f32: {e:?}")),
        DType::Raw => gpu
            .embedding_lookup_hfq4g256(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed hfq4g256: {e:?}")),
        DType::F16 => {
            // No single-row F16 lookup kernel — extract the row to host,
            // convert F16→F32, upload into `out`. Cheap (dim ≤ 4096).
            let mut row_bytes = vec![0u8; dim * 2];
            let off = (token_id as usize) * dim * 2;
            gpu.hip
                .memcpy_dtoh_at(&mut row_bytes, &table.buf, off)
                .map_err(|e| format!("dspark embed f16 dtoh: {e:?}"))?;
            let row_f32: Vec<f32> = (0..dim)
                .map(|i| {
                    let h = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                    crate::llama::f16_to_f32(h)
                })
                .collect();
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(row_f32.as_ptr() as *const u8, dim * 4) };
            gpu.hip
                .memcpy_htod(&out.buf, bytes)
                .map_err(|e| format!("dspark embed f16 htod: {e:?}"))
        }
        other => Err(format!("dspark embed: unsupported table dtype {other:?}")),
    }
}

/// True when `table`'s dtype has a device-token-indexed batched embedding
/// lookup — i.e. the markov chain can stay GPU-resident (token id read straight
/// from a device buffer, no per-slot D2H+H2D). Covers the dtypes DSpark markov
/// heads actually ship as (F16, Q8_0, and Raw==HFQ4-G256); anything else falls
/// back to the host-token path in `dspark_embed_one`.
fn markov_w1_device_embeddable(table: &GpuTensor) -> bool {
    matches!(table.dtype, DType::F16 | DType::Q8_0 | DType::Raw)
}

/// Single-token embedding lookup where the token id is read from a device
/// buffer `token_slot` (`[1]` i32) rather than a host `u32`, keeping the markov
/// sampling loop free of per-slot host round-trips. Byte-identical to
/// [`dspark_embed_one`]: the batched kernels dequantize/widen the same way, and
/// the F16 path's `(float)_Float16` widening matches host `f16_to_f32` exactly.
/// Only the dtypes accepted by [`markov_w1_device_embeddable`] are supported.
fn dspark_embed_one_device(
    gpu: &mut Gpu,
    table: &GpuTensor,
    out: &GpuTensor,
    token_slot: &GpuTensor,
    dim: usize,
) -> Result<(), String> {
    match table.dtype {
        DType::F16 => gpu
            .embedding_lookup_f16_batched(table, out, token_slot, 1, dim)
            .map_err(|e| format!("dspark embed f16 device: {e:?}")),
        DType::Q8_0 => gpu
            .embedding_lookup_q8_batched(table, out, token_slot, 1, dim)
            .map_err(|e| format!("dspark embed q8 device: {e:?}")),
        DType::Raw => gpu
            .embedding_lookup_hfq4g256_batched(table, out, token_slot, 1, dim)
            .map_err(|e| format!("dspark embed hfq4g256 device: {e:?}")),
        other => Err(format!(
            "dspark embed device: unsupported table dtype {other:?}"
        )),
    }
}

/// 1-row GEMV dispatched by weight dtype. `x_rotated` is the FWHT-rotated
/// activation (used for Raw/MQ4 weights); `x_plain` is used for all others.
/// Ported from deepseek4::forward `gemv_auto`.
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

    let gemv = crate::llama::gemv_family();
    let ctx = DispatchCtx::new(gpu);
    let x = if weight_needs_fwht(weight) {
        x_rotated
    } else {
        x_plain
    };
    let wr = WeightRef {
        buf: weight,
        dtype: weight.dtype,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    gemv.run_auto(&ctx, gpu, &wr, x, y)
        .map_err(|e| format!("gemv dispatch: {e}"))
}

/// Batched GEMV/GEMM dispatched by weight dtype with optional WMMA path.
/// `x_rotated_batch` is pre-FWHT; `x_plain_batch` is plain F32 activations.
/// `x_f16_scratch` is a `[batch_size * k]` F16 scratch for WMMA kernels.
/// Ported from deepseek4::forward `gemv_auto_batched_wmma`.
#[allow(clippy::too_many_arguments)]
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
        // MQ2R's trunk head/dense tensors are qt=35 MFP4G32E8SOA (the loader
        // enforces it at hipfire-arch-deepseek4 arch.rs:708-710). Without this
        // arm they fell through to the `_` catch-all below and were decoded as
        // HFQ4-G256 — an incompatible byte layout, so the draft lm_head produced
        // garbage logits and EVERY speculative token lost the exact u32 token-id
        // comparison in `accept_greedy_prefix` (spec.rs:149-178). Measured
        // accept=0.000 (0 of 87 proposed) on MQ2R vs 0.515 on MQ2-Lloyd with the
        // SAME sidecar bytes. This is a recipe-driven bug, not arch-driven: the
        // trunk selector `hipfire-arch-deepseek4 forward.rs:1338-1374` has four
        // E8 arms; this trimmed copy never received them.
        //
        // Mirrors the trunk's generic E8 fallback: try the batched rocBLAS path,
        // else per-row GEMV. The arch-specialised WMMA/B2/B4 tile variants are
        // deliberately NOT replicated here — draft windows are tiny (B<=6), so
        // the tiled paths carry no benefit and every extra arm is another place
        // for the two copies to drift.
        DType::MFP4G32E8SOA => {
            if gpu
                .rocblas_gemm_mfp4e8_soa_prefill_auto(weight, x_rotated_batch, y, m, k, batch_size)
                .map_err(|e| format!("rocBLAS MFP4-E8-SoA draft prefill: {e:?}"))?
            {
                return Ok(());
            }
            // E8-SoA consumes FWHT-rotated activations (gemv.rs:4089).
            for b in 0..batch_size {
                let x_rot = x_rotated_batch.sub_offset(b * k, k);
                let y_row = y.sub_offset(b * m, m);
                gpu.gemv_mfp4g32_e8_soa(weight, &x_rot, &y_row, m, k)
                    .map_err(|e| format!("gemv MFP4-E8-SoA draft row {b}: {e:?}"))?;
            }
            Ok(())
        }
        DType::F32 => gpu
            .gemm_f32_register_tiled(weight, x_plain_batch, y, m, k, batch_size)
            .map_err(|e| format!("gemm_f32_register_tiled: {e:?}")),
        DType::Q8_0 => {
            let wmma_on = hipfire_config::developer_var("HIPFIRE_DSPARK_Q8_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if wmma_on && gpu.arch_caps.is_rdna4() {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    let opt_out =
                        hipfire_config::developer_var("HIPFIRE_DSPARK_Q8_4W").as_deref() == Ok("0");
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
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    let opt_out_4w =
                        hipfire_config::developer_var("HIPFIRE_DSPARK_Q8_4W").as_deref() == Ok("0");
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
        // F16 lm-head: arch-dispatch to avoid polluting the shared ensure_fp16_x
        // cache (which is keyed on the F32 source pointer and is used by trunk
        // GEMMs in the verify pass — cache poisoning by run_heads causes +18ms
        // verify regression per window).
        //
        // gfx12 (RDNA4): gemm_f16_wmma_mb8 takes F32 x directly.
        // RDNA3 with WMMA (gfx11xx, gfx1151, …): K2-pipelined mw16 residual
        //   using the caller-owned x_f16_scratch (bypasses ensure_fp16_x entirely).
        // Non-WMMA: gemm_f16_x_f16_wmma on the external scratch (baseline path).
        DType::F16 => {
            if gpu.arch_caps.has_wmma_w32_gfx12() {
                return gpu
                    .gemm_f16_wmma_mb8(weight, x_plain_batch, y, m, k, batch_size)
                    .map_err(|e| format!("gemm_f16_wmma_mb8 (gfx12 f16): {e:?}"));
            }
            if let Some(scratch) = x_f16_scratch {
                let n = (batch_size * k) as i64;
                gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                    .map_err(|e| format!("convert_f32_to_f16 (F16 weight): {e:?}"))?;
                if gpu.arch_caps.has_wmma_w32() && k % 32 == 0 {
                    // K2-pipelined path: uses caller-owned F16 scratch, skips
                    // ensure_fp16_x to preserve the trunk's fp16_x cache state.
                    gpu.gemm_mw16_residual_wmma_f16x(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_mw16_residual_wmma_f16x: {e:?}"))
                } else if gpu.arch_caps.is_gfx942() {
                    // CDNA3 has no wave32 WMMA — `gemm_f16_x_f16_wmma` cannot
                    // compile for gfx942. Same math via the MFMA port.
                    gpu.gemm_f16_x_f16_mfma_gfx942(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_f16_x_f16_mfma_gfx942: {e:?}"))
                } else {
                    gpu.gemm_f16_x_f16_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_f16_x_f16_wmma: {e:?}"))
                }
            } else {
                Err("F16 weight requires WMMA path with x_f16_scratch".to_string())
            }
        }
        _ => {
            // Gate on `has_wmma()` (false on CDNA3): `gemm_hfq4g256_wmma` is
            // wave32-WMMA and fails to COMPILE for gfx942, so without this
            // the DSpark path dies at JIT instead of using `gemm_hfq4g256`.
            let wmma_on = gpu.arch_caps.has_wmma()
                && hipfire_config::developer_var("HIPFIRE_DSPARK_HFQ4_WMMA")
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

// ── public API ───────────────────────────────────────────────────────────────

/// GEMV `main_proj` over `main_hidden` then `rmsnorm_f32` with `main_norm`
/// in place, writing the result into `out` (`[dim]` F32).
///
/// The concat width is `weights.cfg.target_layer_ids.len() * dim`
/// (generalised from deepseek4's hard-coded `3*hidden`), so 5 target layers
/// work for qwen3 as well as 3 for deepseek4.
///
/// The FWHT-rotation guard is preserved faithfully: when `main_proj` is an
/// MQ4 weight (`weight_needs_fwht` is true), `main_hidden` is rotated into a
/// temporary buffer before the GEMV.
///
/// `main_norm` is `Option` in [`DsparkWeights`]; an error is returned if
/// either `main_proj` or `main_norm` is `None`.
pub fn main_proj_ingest(
    gpu: &mut Gpu,
    weights: &DsparkWeights,
    main_hidden: &GpuTensor,
    out: &GpuTensor,
) -> Result<(), String> {
    let main_proj = weights
        .main_proj
        .as_ref()
        .ok_or("main_proj_ingest: main_proj missing")?;
    let main_norm = weights
        .main_norm
        .as_ref()
        .ok_or("main_proj_ingest: main_norm missing")?;

    let n_targets = weights.cfg.target_layer_ids.len();
    // out is [dim]; infer dim from its length.
    let dim = out.shape.last().copied().unwrap_or(0);
    if dim == 0 {
        return Err("main_proj_ingest: out has zero dim".to_string());
    }
    let concat_w = n_targets * dim;

    if weight_needs_fwht(main_proj) {
        let rot = gpu
            .alloc_tensor(&[concat_w], DType::F32)
            .map_err(|e| format!("main_proj_ingest alloc rot: {e:?}"))?;
        gpu.rotate_x_mq(main_hidden, &rot, concat_w)
            .map_err(|e| format!("main_proj_ingest rotate main_hidden: {e:?}"))?;
        gemv_auto(gpu, main_proj, &rot, main_hidden, out, dim, concat_w)?;
        let _ = gpu.free_tensor(rot);
    } else {
        gemv_auto(gpu, main_proj, main_hidden, main_hidden, out, dim, concat_w)?;
    }

    // main_norm RMSNorm in place.
    gpu.rmsnorm_f32(out, main_norm, out, weights.cfg.rms_norm_eps)
        .map_err(|e| format!("main_proj_ingest main_norm: {e:?}"))?;

    Ok(())
}

/// Multi-slot variant of [`main_proj_ingest`]: applies `hidden_norm(fc(·))` to
/// `ctx_len` independent rows, writing `[ctx_len, dim]` F32 into `out`.
///
/// `main_hidden` is `[ctx_len * concat_w]` where `concat_w = n_targets * dim`.
/// `out` is `[ctx_len * dim]` (shape `[ctx_len * dim]` — 1D flat storage).
/// `ctx_len=1` produces the same result as the single-slot [`main_proj_ingest`].
///
/// `dim` must equal the hidden dimension (e.g. 4096 for Qwen3-8B drafter).
/// It cannot be inferred from `out.shape` because `out` is stored as a flat
/// `[ctx_len * dim]` 1D tensor.
///
/// This matches `modeling.py:373`
/// `target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))`
/// where `target_hidden_states` is `[B, ctx_len, n_targets*hidden]`.
pub fn main_proj_ingest_batched(
    gpu: &mut Gpu,
    weights: &DsparkWeights,
    main_hidden: &GpuTensor, // [ctx_len * concat_w]
    out: &GpuTensor,         // [ctx_len * dim]
    ctx_len: usize,
    dim: usize,
) -> Result<(), String> {
    if ctx_len == 0 {
        return Err("main_proj_ingest_batched: ctx_len=0".to_string());
    }
    if dim == 0 {
        return Err("main_proj_ingest_batched: dim=0".to_string());
    }
    let main_proj = weights
        .main_proj
        .as_ref()
        .ok_or("main_proj_ingest_batched: main_proj missing")?;
    let main_norm = weights
        .main_norm
        .as_ref()
        .ok_or("main_proj_ingest_batched: main_norm missing")?;

    let n_targets = weights.cfg.target_layer_ids.len();
    let concat_w = n_targets * dim;
    let needs_fwht = weight_needs_fwht(main_proj);

    // FWHT scratch (only needed when main_proj is MQ4/Raw).
    let rot_buf = if needs_fwht {
        Some(
            gpu.alloc_tensor(&[concat_w], DType::F32)
                .map_err(|e| format!("main_proj_ingest_batched alloc rot: {e:?}"))?,
        )
    } else {
        None
    };

    // Allocate a per-row output scratch [dim] F32. We use a full independent
    // allocation per GEMV call to avoid sub-offset output tensors: some GPU
    // GEMV implementations (gemm_f16_batched_lmhead → mw16 WMMA path on gfx1151)
    // pre-zero the output buffer and may behave incorrectly when `y.buf` is a
    // sub-offset (non-base-pointer) view. After each GEMV we d2d copy the row
    // into the correct position of `out`.
    let row_scratch = gpu
        .alloc_tensor(&[dim], DType::F32)
        .map_err(|e| format!("main_proj_ingest_batched alloc row_scratch: {e:?}"))?;

    for j in 0..ctx_len {
        let h_row = main_hidden.sub_offset(j * concat_w, concat_w);
        if let Some(ref rot) = rot_buf {
            gpu.rotate_x_mq(&h_row, rot, concat_w)
                .map_err(|e| format!("main_proj_ingest_batched rotate row {j}: {e:?}"))?;
            gemv_auto(gpu, main_proj, rot, &h_row, &row_scratch, dim, concat_w)?;
        } else {
            gemv_auto(gpu, main_proj, &h_row, &h_row, &row_scratch, dim, concat_w)?;
        }
        // Copy row_scratch → out[j*dim .. (j+1)*dim] using offset-aware D2D.
        gpu.memcpy_dtod_at_auto(&out.buf, j * dim * 4, &row_scratch.buf, 0, dim * 4)
            .map_err(|e| format!("main_proj_ingest_batched d2d row {j}: {e:?}"))?;
    }

    let _ = gpu.free_tensor(row_scratch);
    if let Some(rot) = rot_buf {
        let _ = gpu.free_tensor(rot);
    }

    // Apply main_norm RMSNorm to all ctx_len rows in one batched call.
    // rmsnorm_batched treats out as [ctx_len * dim] → [ctx_len] rows of [dim].
    gpu.rmsnorm_batched(out, main_norm, out, ctx_len, dim, weights.cfg.rms_norm_eps)
        .map_err(|e| format!("main_proj_ingest_batched rmsnorm: {e:?}"))?;

    Ok(())
}

/// Build the noise token id block for one DSpark window:
/// `[seed, noise_token_id, noise_token_id, ...]` of length `cfg.block_size`.
pub fn noise_block_ids(cfg: &DsparkConfig, seed: u32) -> Vec<u32> {
    let mut ids = vec![cfg.noise_token_id; cfg.block_size];
    ids[0] = seed;
    ids
}

/// Run the DSpark head pipeline over a completed `x_head` block:
/// 1. lm-head GEMV: `rmsnorm(x_head, stage_norm)` → optional FWHT → batched
///    `lm_head` GEMV → `logits[block, vocab]` (resident on GPU).
/// 2. Sequential markov sampling loop: for each slot `i`, look up
///    `markov_w1[prev_token]`, (optionally) compute the on-GPU confidence
///    logit for slot `i`, then add `markov_w2 @ emb` bias to `logits[i]` and
///    argmax — all on GPU. Sequential dependency forces one argmax per slot.
/// 3. Confidence head download: `block` scalars transferred from
///    `conf_batch`; sigmoid applied on host. When `!cfg.enable_confidence`
///    the confidence vector is filled with `f32::INFINITY` (no-op for
///    downstream truncation).
///
/// `x_head` is `[block, hidden]` F32 produced by [`DsparkBody::draft_block`].
/// `stage_norm` is the per-stage final RMSNorm weight applied to `x_head`
/// before the lm-head GEMV. Callers supply the arch-specific norm: deepseek4
/// passes `mtp_final_norm`; qwen3 passes its drafter `norm` tensor.
/// `lm_head` is `[vocab, hidden]`.
/// `prev_token` is the last committed token before this draft window.
pub fn run_heads(
    gpu: &mut Gpu,
    weights: &DsparkWeights,
    stage_norm: &GpuTensor,
    lm_head: &GpuTensor,
    x_head: &GpuTensor, // [block, hidden]
    prev_token: u32,
    block: usize,
    vocab: usize,
) -> Result<DraftResult, String> {
    let cfg = &weights.cfg;

    // Guard: confidence head requires confidence_proj to be present.
    if cfg.enable_confidence && weights.confidence_proj.is_none() {
        return Err("run_heads: enable_confidence=true but confidence_proj missing".to_string());
    }

    let markov_rank = cfg.markov_rank;

    let markov_w1 = weights
        .markov_w1
        .as_ref()
        .ok_or("run_heads: markov_w1 missing")?;
    let markov_w2 = weights
        .markov_w2
        .as_ref()
        .ok_or("run_heads: markov_w2 missing")?;

    // Infer `hidden` from x_head shape.
    let hidden = x_head.shape.last().copied().unwrap_or(0);
    if hidden == 0 {
        return Err("run_heads: x_head has zero hidden dim".to_string());
    }

    // ── lm-head GEMV: rmsnorm(x_head, stage_norm) → [optional FWHT] → batched GEMV ──
    //
    // `stage_norm` is the per-stage final norm supplied by the caller
    // (deepseek4: `mtp_final_norm`; qwen3: drafter `norm`).
    let normed = gpu
        .alloc_tensor(&[block, hidden], DType::F32)
        .map_err(|e| format!("run_heads alloc normed: {e:?}"))?;
    let rms_norm_eps = cfg.rms_norm_eps;
    gpu.rmsnorm_batched(x_head, stage_norm, &normed, block, hidden, rms_norm_eps)
        .map_err(|e| format!("run_heads final rmsnorm: {e:?}"))?;

    let normed_rot = if weight_needs_fwht(lm_head) {
        let r = gpu
            .alloc_tensor(&[block, hidden], DType::F32)
            .map_err(|e| format!("run_heads alloc normed_rot: {e:?}"))?;
        gpu.rotate_x_mq_batched(&normed, &r, hidden, block)
            .map_err(|e| format!("run_heads rotate head input: {e:?}"))?;
        Some(r)
    } else {
        None
    };
    let logits_dev = gpu
        .alloc_tensor(&[block, vocab], DType::F32)
        .map_err(|e| format!("run_heads alloc logits: {e:?}"))?;
    let x_f16 = gpu
        .alloc_tensor(&[block * hidden], DType::F16)
        .map_err(|e| format!("run_heads alloc x_f16: {e:?}"))?;
    gemv_auto_batched_wmma(
        gpu,
        lm_head,
        normed_rot.as_ref().unwrap_or(&normed),
        &normed,
        &logits_dev,
        vocab,
        hidden,
        block,
        Some(&x_f16),
    )?;
    if let Some(r) = normed_rot {
        let _ = gpu.free_tensor(r);
    }
    let _ = gpu.free_tensor(x_f16);
    // `normed` is kept alive until after the loop: the confidence head
    // reads normed[i] per slot (modeling.py uses once-normed hidden for confidence).
    // `logits_dev` stays resident: the markov loop adds each slot's bias
    // and argmaxes ON-GPU. Both freed after the loop.

    // ── Sequential markov in-block sampling (greedy) ─────────────────────
    // out_ids[0] = prev_token; out_ids[i+1] = argmax(logits[i] + markov_bias).
    let mut out_ids = vec![prev_token; block + 1];

    // Device-resident token chain: when markov_w1 supports a device-token-indexed
    // lookup, keep the whole sequential loop on the GPU — the previous slot's
    // argmax lands in `chain[i+1]` (via `argmax_token_chain_f32`) and the next
    // embed reads `chain[i]` straight from device memory. This removes the
    // per-slot embed D2H+H2D and per-slot argmax D2H (each ~free on UMA, but a
    // full pipeline drain on a discrete-VRAM GPU); the block's token ids come
    // back in a single D2H after the loop. `chain` is a 4-byte-per-elem F32-typed
    // buffer holding i32 ids (same convention as qwen35's `mtp_token_chain`);
    // `argmax_scratch` is the throwaway index sink `argmax_token_chain_f32` also
    // writes. `None` ⇒ host-token fallback (byte-identical).
    // Reduced-vocab (d2t Some) forces the host path: each argmax is a DRAFT id
    // that must be d2t-remapped to a TARGET id before it can index the
    // full-target-vocab markov chain — the on-GPU chain kernel can't do that gather.
    let chain_bufs: Option<(GpuTensor, GpuTensor)> =
        if markov_w1_device_embeddable(markov_w1) && weights.d2t.is_none() {
            let chain = gpu
                .alloc_tensor(&[block + 1], DType::F32)
                .map_err(|e| format!("run_heads alloc token chain: {e:?}"))?;
            let seed_i32 = prev_token as i32;
            let seed_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(&seed_i32 as *const i32 as *const u8, 4) };
            gpu.hip
                .memcpy_htod(&chain.buf, seed_bytes)
                .map_err(|e| format!("run_heads htod token chain seed: {e:?}"))?;
            let argmax_scratch = gpu
                .alloc_tensor(&[1], DType::F32)
                .map_err(|e| format!("run_heads alloc argmax scratch: {e:?}"))?;
            Some((chain, argmax_scratch))
        } else {
            None
        };

    // Confidence head buffers (ON GPU per slot inside the loop):
    // `conf_batch[block]` holds the per-slot confidence logit.
    // `concat_dev` stages `[x_head[i] ++ markov_embed[i]]` for the 1-row
    // `confidence_proj` gemv. Downloaded once after the loop (block floats).
    let proj_in = hidden + markov_rank;
    let conf_batch = gpu
        .alloc_tensor(&[block], DType::F32)
        .map_err(|e| format!("run_heads alloc conf_batch: {e:?}"))?;
    let concat_dev = gpu
        .alloc_tensor(&[proj_in], DType::F32)
        .map_err(|e| format!("run_heads alloc conf concat: {e:?}"))?;
    // Reusable device scratch for the markov embedding.
    let emb_dev = gpu
        .alloc_tensor(&[markov_rank], DType::F32)
        .map_err(|e| format!("run_heads alloc markov emb: {e:?}"))?;
    let bias_dev = gpu
        .alloc_tensor(&[vocab], DType::F32)
        .map_err(|e| format!("run_heads alloc markov bias: {e:?}"))?;
    let emb_rot = if weight_needs_fwht(markov_w2) {
        Some(
            gpu.alloc_tensor(&[markov_rank], DType::F32)
                .map_err(|e| format!("run_heads alloc markov emb rot: {e:?}"))?,
        )
    } else {
        None
    };
    for i in 0..block {
        // markov_w1 lookup → emb_dev [markov_rank] (unrotated). Device chain reads
        // the previous slot's token id straight from GPU (`chain[i]`); the host
        // fallback uses the argmax'd host id (`out_ids[i]`).
        if let Some((chain, _)) = chain_bufs.as_ref() {
            let token_slot = chain.sub_offset(i, 1);
            dspark_embed_one_device(gpu, markov_w1, &emb_dev, &token_slot, markov_rank)?;
        } else {
            dspark_embed_one(gpu, markov_w1, &emb_dev, out_ids[i], markov_rank)?;
        }

        // Confidence slot i ON GPU: stage [hidden_i ++ markov_embed[i]] then
        // a 1-row `confidence_proj` gemv → conf_batch[i]. Uses the UNROTATED
        // emb_dev (matches the reference which dotted the raw markov embed).
        // `hidden_i` is arch-specific: qwen3 uses `normed[i]` (once-normed,
        // matching modeling.py's predict_confidence_step input = self.norm(hidden));
        // deepseek4 uses raw `x_head[i]` (pre-norm, byte-identical to task-5 baseline).
        if cfg.enable_confidence {
            if let Some(confidence_proj) = weights.confidence_proj.as_ref() {
                let xh_i = if cfg.confidence_uses_normed {
                    normed.sub_offset(i * hidden, hidden)
                } else {
                    x_head.sub_offset(i * hidden, hidden)
                };
                let c_hidden = concat_dev.sub_offset(0, hidden);
                let c_markov = concat_dev.sub_offset(hidden, markov_rank);
                gpu.memcpy_dtod_auto(&c_hidden.buf, &xh_i.buf, hidden * 4)
                    .map_err(|e| format!("run_heads conf stage x_head {i}: {e:?}"))?;
                gpu.memcpy_dtod_auto(&c_markov.buf, &emb_dev.buf, markov_rank * 4)
                    .map_err(|e| format!("run_heads conf stage emb {i}: {e:?}"))?;
                let conf_i = conf_batch.sub_offset(i, 1);
                gemv_auto(
                    gpu,
                    confidence_proj,
                    &concat_dev,
                    &concat_dev,
                    &conf_i,
                    1,
                    proj_in,
                )?;
                // Add optional bias (qwen3 has a [1] bias; deepseek4 has None).
                if let Some(bias) = weights.confidence_bias.as_ref() {
                    gpu.add_inplace_f32(&conf_i, bias)
                        .map_err(|e| format!("run_heads conf bias add {i}: {e:?}"))?;
                }
            }
        }

        // bias = markov_w2 @ emb  ([vocab, markov_rank] · [markov_rank]).
        let x_for_w2 = if let Some(r) = emb_rot.as_ref() {
            gpu.rotate_x_mq(&emb_dev, r, markov_rank)
                .map_err(|e| format!("run_heads rotate markov emb {i}: {e:?}"))?;
            r
        } else {
            &emb_dev
        };
        gemv_auto(
            gpu,
            markov_w2,
            x_for_w2,
            &emb_dev,
            &bias_dev,
            vocab,
            markov_rank,
        )?;
        // logits[i] += bias, then argmax — both ON-GPU.
        let row = logits_dev.sub_offset(i * vocab, vocab);
        gpu.add_inplace_f32(&row, &bias_dev)
            .map_err(|e| format!("run_heads markov bias add {i}: {e:?}"))?;
        if let Some((chain, argmax_scratch)) = chain_bufs.as_ref() {
            // argmax → chain[i+1] (device), feeding the next slot's embed without
            // a host round-trip. `argmax_token_chain_f32`'s reduction is identical
            // to `argmax_f32` (strict `>`, lowest-index tie-break).
            gpu.argmax_token_chain_f32(&row, argmax_scratch, chain, None, vocab, i + 1)
                .map_err(|e| format!("run_heads markov argmax chain {i}: {e:?}"))?;
        } else {
            let draft_id = gpu
                .argmax_f32(&row, vocab)
                .map_err(|e| format!("run_heads markov argmax {i}: {e:?}"))?;
            // Reduced-vocab: argmax is a DRAFT id over `vocab` (=draft_vocab_size)
            // logits; map it to a target token id so it feeds the full-vocab markov
            // chain (out_ids[i] indexes markov_w1 over target vocab) and verify.
            // Shared vocab (d2t None) passes through unchanged.
            out_ids[i + 1] = match weights.d2t.as_ref() {
                Some(d2t) => d2t.get(draft_id as usize).copied().unwrap_or(draft_id),
                None => draft_id,
            };
        }
    }
    // Device chain: pull the whole block's token ids back in one D2H (vs one per
    // slot in the host path).
    if let Some((chain, _)) = chain_bufs.as_ref() {
        let mut ids_i32 = vec![0i32; block + 1];
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(ids_i32.as_mut_ptr() as *mut u8, (block + 1) * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &chain.buf)
            .map_err(|e| format!("run_heads d2h token chain: {e:?}"))?;
        for i in 1..=block {
            out_ids[i] = ids_i32[i] as u32;
        }
    }
    let _ = gpu.free_tensor(emb_dev);
    let _ = gpu.free_tensor(bias_dev);
    let _ = gpu.free_tensor(logits_dev);
    let _ = gpu.free_tensor(normed);
    if let Some(r) = emb_rot {
        let _ = gpu.free_tensor(r);
    }
    if let Some((chain, argmax_scratch)) = chain_bufs {
        let _ = gpu.free_tensor(chain);
        let _ = gpu.free_tensor(argmax_scratch);
    }

    // ── Confidence download ───────────────────────────────────────────────
    // When confidence is disabled, return +inf so downstream truncation
    // is a no-op. When enabled, download the `block` raw confidence LOGITS
    // (pre-sigmoid). The caller (`DsparkDrafter::mtp_step`) applies
    // `sigmoid(c)` itself when comparing against `conf_threshold`, matching
    // the `Deepseek4DsparkDrafter` convention: confidence stores raw logits,
    // sigmoid is applied at the truncation site.
    let confidence = if cfg.enable_confidence && weights.confidence_proj.is_some() {
        let mut raw = vec![0.0f32; block];
        {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, block * 4) };
            gpu.hip
                .memcpy_dtoh(bytes, &conf_batch.buf)
                .map_err(|e| format!("run_heads d2h confidence: {e:?}"))?;
        }
        raw
    } else {
        vec![f32::INFINITY; block]
    };
    let _ = gpu.free_tensor(conf_batch);
    let _ = gpu.free_tensor(concat_dev);

    Ok(DraftResult {
        tokens: out_ids[1..=block].to_vec(),
        logits: Vec::new(),
        confidence,
    })
}

// ── Generic DsparkDrafter ─────────────────────────────────────────────────────
//
// Drives any [`DsparkBody`] through the [`MtpDrafter`] interface. The arch-
// specific body (deepseek4 MoE chain, qwen3 dense transformer) is injected at
// build time; the target is reached only through [`SpecTarget`] trait methods
// (notably [`SpecTarget::capture_seed_main_hidden`] for bootstrap + hidden
// capture, and the generic verify primitives). No `Deepseek4*` type appears here.

/// Upload a host F32 slice to a freshly-allocated GPU tensor.
fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(&[v.len()], DType::F32)
        .map_err(|e| format!("DsparkDrafter: alloc main_hidden: {e:?}"))?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(|e| format!("DsparkDrafter: htod main_hidden: {e:?}"))?;
    Ok(t)
}

/// Generic DSpark drafter: window orchestration over any [`DsparkBody`].
///
/// ## Context bookkeeping (Stage 2+)
///
/// After each verify+accept window, the multi-slot context for the NEXT window
/// is the verify-forward's accepted-prefix hidden at the target's extract layers
/// (positions `P ..= P + accepted`, i.e. `accepted+1` slots). This matches the
/// Python reference `_update`: `context.target_hidden_states =
/// verified_target_hidden[:, :accepted_draft_tokens+1, :]`.
///
/// `ctx_hidden_raw` holds these raw (pre-`main_proj_ingest`) hidden floats as
/// `[ctx_len * n_targets * dim]` flat F32 on the host.  `ctx_positions` holds
/// the corresponding absolute positions.
///
/// The INITIAL window (no previous verify) uses a single-slot bootstrap via
/// `capture_seed_main_hidden` (matching `_init_context`), giving `ctx_len=1`.
/// After the first window the loop uses the verify hidden directly and skips
/// `capture_seed_main_hidden` entirely.
pub struct DsparkDrafter {
    body: Box<dyn DsparkBody>,
    weights: DsparkWeights,
    /// Per-stage final norm fed to [`run_heads`].
    stage_norm: GpuTensor,
    /// lm-head weight fed to [`run_heads`].
    lm_head: GpuTensor,
    conf_threshold: f32,
    block: usize,
    ctx_capacity: usize,
    /// Whether the target supports distribution-preserving sampled verify
    /// ([`SpecTarget::verify_block_sampled_capture_gpu`]). Set per-arch at build
    /// time (llama/qwen3: true; deepseek4: false until its sampled head lands).
    /// Drives `requires_greedy()`/`supports_temp_verify()`: when false, temp>0
    /// requests never reach this drafter (the daemon routes them to AR).
    supports_temp: bool,
    /// Request sampling params, stashed by `set_sampling` before each `mtp_step`.
    /// `temp <= 1e-6` selects the greedy (argmax) verify; else the sampled verify.
    temp: f32,
    top_p: f32,
    top_k: usize,
    /// CACTUS acceptance-boost δ (0 = lossless). Applied only in the sampled
    /// (temp>0) verify via `verify_block_sampled_capture_gpu`.
    cactus: f32,
    /// RNG for the sampled verify's categorical draw. Fixed seed per drafter
    /// (mirrors the DFlash speculator) so a warm run is reproducible.
    rng_state: u64,
    /// GPU tensor holding the multi-slot context main_hidden for `ctx_positions`.
    /// `[ctx_len * n_targets * dim]` F32 (flat).  `None` ⇒ must bootstrap on
    /// next `mtp_step` (initial window or after reset/prefill).
    main_hidden_dev: Option<GpuTensor>,
    /// Absolute positions whose main_hidden is cached in `main_hidden_dev`.
    /// Empty ⇒ cache invalid (same condition as `main_hidden_dev.is_none()`).
    ctx_positions: Vec<usize>,
    /// Per-window phase profiler; active only when `HIPFIRE_DSPARK_PROFILE=1`.
    profiler: DsparkProfiler,
    /// Adaptive block-size controller. `None` when `HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0`
    /// (opt-out; fixed block == pre-change behaviour).
    block_controller: Option<crate::dspark_block_controller::BlockController>,
}

impl MtpDrafter for DsparkDrafter {
    fn name(&self) -> &'static str {
        "dspark"
    }

    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
    ) -> Result<u32, String> {
        if !cache_hit {
            target.reset_recurrent(gpu)?;
        }
        // Invalidate multi-slot context: mtp_step re-bootstraps for the first seed.
        // As in the deepseek4 drafter, we do NOT warm the DSpark stage rings
        // during prefill (measured LOSS on code prompts — see dspark_speculator.rs).
        //
        // DELIBERATE DEVIATION from Deepseek4DsparkDrafter::mtp_prefill, which
        // clears main_hidden_pos only on `!cache_hit`. We clear it
        // unconditionally because the generic body may not preserve a valid
        // cached hidden across a cache-hit prefill. This is behaviourally
        // identical today; the unconditional clear is the safe default.
        if let Some(dev) = self.main_hidden_dev.take() {
            let _ = gpu.free_tensor(dev);
        }
        self.ctx_positions.clear();
        if let Some(c) = self.block_controller.as_mut() {
            c.reset();
        }

        if fill_tokens.is_empty() {
            return Err("DsparkDrafter::mtp_prefill: fill_tokens is empty".into());
        }

        // Run the full prefill through spec_advance (reset=false; recurrent was
        // already reset above on cache_miss). Returns argmax at the last position,
        // which is the seed for the first decode window.
        let abort = &|| false;
        match target.spec_advance(gpu, fill_tokens, start_pos, false, abort, None)? {
            crate::spec::SpecAdvance::Ready { last_argmax, .. } => Ok(last_argmax),
            crate::spec::SpecAdvance::Aborted => {
                Err("DsparkDrafter::mtp_prefill: spec_advance aborted".into())
            }
        }
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        // DSpark drafts the whole block at once. In-step grammar is not yet
        // wired for the generic drafter (no arch-specific grammar type to
        // downcast to) — ignore silently; post-hoc emission-layer grammar applies.

        let layers = self.weights.cfg.target_layer_ids.clone();
        let n_targets = layers.len();
        // Adaptive path may shrink below the caller's k, but MUST NOT exceed it:
        // per-call k from MtpSpeculator is the remaining max_emit budget.
        // fixed path (opt-out) keeps the old behaviour exactly.
        let effective_k = self
            .block_controller
            .as_ref()
            .map(|c| c.block().min(k))
            .unwrap_or(k);
        // k==0 is the true one-token path (verify seed only, no drafts).
        let block = if k == 0 {
            0
        } else {
            self.weights.cfg.block_size.min(effective_k).max(1)
        };
        let vocab = self.lm_head.shape[0];
        let hidden = {
            // Infer hidden from stage_norm shape (it's [hidden]).
            self.stage_norm.shape[0]
        };

        // ── 1. Bootstrap or reuse ctx from previous verify ──────────────────
        // INITIAL window (ctx_positions is empty): materialise a 1-slot bootstrap
        // via `capture_seed_main_hidden` — matches the reference `_init_context`.
        // STEADY-STATE: ctx_hidden_dev/ctx_positions were populated by the
        // previous window's verify; skip `capture_seed_main_hidden` entirely.
        let t_bootstrap = self.profiler.sync_start(gpu);
        if self.ctx_positions.is_empty() {
            // Initial window: bootstrap with a 1-token capture-armed forward.
            let hidden_host = target.capture_seed_main_hidden(gpu, seed, position, &layers)?;
            if hipfire_config::developer_var("HIPFIRE_DSPARK_DEBUG").as_deref() == Ok("1") {
                let n = hidden_host.len();
                let mean = hidden_host.iter().sum::<f32>() / n.max(1) as f32;
                let rms = (hidden_host.iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32).sqrt();
                let nz = hidden_host.iter().filter(|&&x| x != 0.0).count();
                eprintln!(
                    "[dspark] capture_seed hidden len={n} nonzero={nz} mean={mean:.4} rms={rms:.4} first4={:?}",
                    &hidden_host[..n.min(4)]
                );
            }
            let dev = upload_f32(gpu, &hidden_host)?;
            if let Some(old) = self.main_hidden_dev.take() {
                let _ = gpu.free_tensor(old);
            }
            self.main_hidden_dev = Some(dev);
            self.ctx_positions = vec![position];
        }
        // Steady-state: main_hidden_dev and ctx_positions are already set from
        // the previous window's accepted-prefix capture.
        self.profiler.sync_end(gpu, t_bootstrap, 0);

        // Start the FULL-window cost timer (adaptive path only). It spans draft +
        // heads + verify — the whole per-window GPU cost, not just verify. Both
        // run_heads (tokens/confidence) and verify (target_pick) force a D2H, so the
        // elapsed at the verify D2H is real wall time with no added sync. Timing the
        // whole window (vs verify only) is what lets the argmax see that a larger
        // block barely raises the per-window wall time on cheap-verify arches.
        let _t_win = self
            .block_controller
            .is_some()
            .then(std::time::Instant::now);

        let ctx_positions = self.ctx_positions.clone();
        let main_hidden_real = self
            .main_hidden_dev
            .as_ref()
            .ok_or("DsparkDrafter: main_hidden_dev missing")?;
        // DEBUG: HIPFIRE_DSPARK_ZERO_CTX=1 zeros the context main_hidden to test
        // whether the drafter attends to it at all (identical drafts ⇒ ctx unused).
        let _zero_ctx_holder;
        let main_hidden =
            if hipfire_config::developer_var("HIPFIRE_DSPARK_ZERO_CTX").as_deref() == Ok("1") {
                let n: usize = main_hidden_real.shape.iter().product();
                _zero_ctx_holder = upload_f32(gpu, &vec![0.0f32; n])?;
                &_zero_ctx_holder
            } else {
                main_hidden_real
            };

        // ── 2–3. Draft the block (skipped on k==0 one-token path) ──────────
        // k==0: verify [seed] only and emit the single bonus — no draft_block /
        // run_heads, and no conf-threshold that would force keep≥1.
        let (mut drafts, draft_confidence): (Vec<u32>, Vec<f32>) = if block == 0 {
            (Vec::new(), Vec::new())
        } else {
            let x_head_out = gpu
                .alloc_tensor(&[block, hidden], DType::F32)
                .map_err(|e| format!("DsparkDrafter: alloc x_head: {e:?}"))?;
            let t_draft = self.profiler.sync_start(gpu);
            self.body.draft_block(
                gpu,
                &self.weights,
                main_hidden,
                &ctx_positions,
                seed,
                position,
                block,
                &x_head_out,
            )?;
            self.profiler.sync_end(gpu, t_draft, 1);

            let t_heads = self.profiler.sync_start(gpu);
            let draft = run_heads(
                gpu,
                &self.weights,
                &self.stage_norm,
                &self.lm_head,
                &x_head_out,
                seed,
                block,
                vocab,
            )?;
            self.profiler.sync_end(gpu, t_heads, 2);
            let _ = gpu.free_tensor(x_head_out);

            let mut drafts: Vec<u32> = draft.tokens.into_iter().take(block).collect();
            // Confidence-threshold truncation: walk slots, truncate at first
            // slot whose sigmoid(confidence) < conf_threshold; always keep ≥1.
            let conf_threshold = self.conf_threshold;
            let confident_len = {
                let mut l = drafts.len();
                for (i, &c) in draft.confidence.iter().enumerate().take(drafts.len()) {
                    let survival = 1.0f32 / (1.0 + (-c).exp());
                    if survival < conf_threshold {
                        l = i;
                        break;
                    }
                }
                l.max(1)
            };
            drafts.truncate(confident_len);
            (drafts, draft.confidence)
        };
        let n_proposed = drafts.len();

        // ── 4. Verify: target forward over [seed, draft0..draft_{n-1}] ───────
        // Capture the target's hidden states at extract layers for ALL verified
        // positions.  After acceptance, positions 0..=accepted (accepted+1 slots)
        // become the NEXT window's multi-slot context (reference `_update`).
        // The hidden_out format: (verify_tokens.len()) × (n_targets × hidden)
        // row-major — one (n_targets × hidden) row per consumed position.
        let verify_tokens: Vec<u32> = std::iter::once(seed)
            .chain(drafts.iter().copied())
            .collect();
        let n_verify = verify_tokens.len(); // seed + n_proposed drafts

        let mut scratch = target.new_spec_scratch(gpu, n_verify)?;
        // Capture the accepted-prefix hidden GPU-resident: verify writes the
        // per-position extract-layer hidden straight into `capture_buf`, and the
        // context update below slices the accepted prefix out of it GPU→GPU — so
        // the reuse never round-trips through host memory (a D2H+H2D per window,
        // ~free on UMA but a real cost on a discrete-VRAM GPU). `captured` is
        // false when the target's batched capture path can't run for this block
        // (llama with n_verify < 4); the update then re-bootstraps, matching the
        // old host path's empty-capture branch. deepseek4 captures every size.
        let expected_hidden_per_pos = n_targets * hidden;
        let capture_buf = gpu
            .alloc_tensor(&[n_verify * expected_hidden_per_pos], DType::F32)
            .map_err(|e| format!("DsparkDrafter: alloc capture_buf: {e:?}"))?;
        let kernel_profile_this_window = self.profiler.kernel_profile_position == Some(position);
        if kernel_profile_this_window {
            rdna_compute::profile::start();
        }
        let t_verify = self.profiler.sync_start(gpu);
        // temp<=eps → greedy argmax verify (byte-identical to the pre-temp path);
        // temp>0 → distribution-preserving sampled verify (target samples the
        // token, drafts accepted iff they match). Both capture hidden GPU-resident.
        let (target_pick, captured) = if self.temp <= 1e-6 {
            target.verify_block_capture_gpu(
                gpu,
                &verify_tokens,
                position,
                scratch.as_mut(),
                &capture_buf,
            )?
        } else {
            target.verify_block_sampled_capture_gpu(
                gpu,
                &verify_tokens,
                position,
                scratch.as_mut(),
                self.temp,
                self.top_p,
                self.top_k,
                self.cactus,
                &mut self.rng_state,
                &capture_buf,
            )?
        };
        // Close the full-window timer here: the verify's D2H (target_pick) has just
        // forced GPU completion of draft + heads + verify, so this is real wall time.
        let t_window_ms = _t_win.map(|t| t.elapsed().as_secs_f32() * 1000.0);
        self.profiler.sync_end(gpu, t_verify, 3);
        if kernel_profile_this_window {
            let entries = rdna_compute::profile::stop().unwrap_or_default();
            let mut by_kernel: std::collections::HashMap<&str, (f64, usize, usize)> =
                std::collections::HashMap::new();
            for entry in &entries {
                let aggregate = by_kernel.entry(entry.kernel).or_insert((0.0, 0, 0));
                aggregate.0 += entry.time_us;
                aggregate.1 += 1;
                aggregate.2 += entry.bytes;
            }
            let mut kernels: Vec<_> = by_kernel.into_iter().collect();
            kernels.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            eprintln!(
                "=== DSPARK VERIFY KERNEL PROFILE position={position} n_verify={n_verify} ==="
            );
            for (kernel, (time_us, calls, bytes)) in kernels {
                let gbps = if time_us > 0.0 {
                    bytes as f64 / 1.0e3 / time_us
                } else {
                    0.0
                };
                eprintln!(
                    "kernel={kernel} calls={calls} total_us={time_us:.3} bytes={bytes} gbps={gbps:.3}"
                );
            }
        }

        // ── 5. Greedy accept ─────────────────────────────────────────────────
        let t_rest = self.profiler.sync_start(gpu);
        if hipfire_config::developer_var("HIPFIRE_DSPARK_DEBUG").as_deref() == Ok("1") {
            eprintln!(
                "[dspark] pos={position} seed={seed} n_prop={n_proposed} drafts={:?} target_pick={:?} conf={:?}",
                &drafts,
                &target_pick[..target_pick.len().min(drafts.len() + 1)],
                &draft_confidence[..draft_confidence.len().min(4)]
            );
        }
        let acc = accept_greedy_prefix(&drafts, &target_pick, Some(eos));
        let committed = acc.committed;
        let n_accepted = acc.accepted;

        // ── 6. Commit the accepted prefix into the target's state ────────────
        // verify_block advanced the target by verify_tokens.len() slots. We
        // committed only n_accepted drafts + the bonus. Rewind to the true commit
        // length via commit_prefix (no-op for stateless/full-accept; replays for
        // recurrent arches).
        let accept_len = n_accepted; // accepted drafts; bonus is at accept_len
        target.commit_prefix(gpu, &verify_tokens, accept_len, position, scratch.as_mut())?;
        scratch.free(gpu);

        // ── 7. Update multi-slot context for the NEXT window ─────────────────
        // Reference `_update`: next ctx = verify hidden at accepted prefix
        // (positions 0..=accepted, i.e. accepted+1 slots).
        // Each slot contributes n_targets * hidden floats (row = concat of
        // extract-layer hiddens at that position).
        // Cap ctx_len to block+1 (the scratch `max_ctx_len` in the qwen3 body).
        // block==0 (k==0 one-token path) still needs at least one ctx slot.
        let new_ctx_len_raw = accept_len + 1; // seed + accepted drafts
        let new_ctx_len = new_ctx_len_raw.min(block.max(1) + 1);
        if captured {
            // Slice the accepted prefix (new_ctx_len slots) out of the GPU capture
            // buffer into a fresh drafter-owned buffer — all on-device, no D2H+H2D.
            // We take the LAST new_ctx_len slots of the accepted prefix when the
            // full ctx_len_raw > block+1 (cap); in practice block+1 is usually
            // ≥ accepted+1, so start_slot is 0.
            let start_slot = new_ctx_len_raw - new_ctx_len; // 0 in normal case
            let n_floats = new_ctx_len * expected_hidden_per_pos;
            let src_offset = start_slot * expected_hidden_per_pos;
            let dev = gpu
                .alloc_tensor(&[n_floats], DType::F32)
                .map_err(|e| format!("DsparkDrafter: alloc ctx hidden: {e:?}"))?;
            gpu.memcpy_dtod_at_auto(&dev.buf, 0, &capture_buf.buf, src_offset * 4, n_floats * 4)
                .map_err(|e| format!("DsparkDrafter: d2d ctx hidden: {e:?}"))?;
            if let Some(old) = self.main_hidden_dev.take() {
                let _ = gpu.free_tensor(old);
            }
            self.main_hidden_dev = Some(dev);
            // ctx positions: position + start_slot .. position + start_slot + new_ctx_len
            self.ctx_positions = (start_slot..start_slot + new_ctx_len)
                .map(|s| position + s)
                .collect();
        } else {
            // Hidden capture not available (n_verify < 4 on llama): clear the
            // context so the NEXT window bootstraps via `capture_seed_main_hidden`
            // with the bonus token as seed.  That bootstrap is KV-correct (the
            // commit_prefix leaves KV at the committed prefix; the next
            // `mtp_step` call receives seed=bonus, position=committed_end, so
            // `capture_seed_main_hidden` advances the KV by exactly 1 for the
            // bonus, which is the right thing).  This is the Stage 1 behaviour
            // and is correct — just forfeits the multi-slot ctx for one window.
            if let Some(old) = self.main_hidden_dev.take() {
                let _ = gpu.free_tensor(old);
            }
            self.ctx_positions.clear();
        }
        let _ = gpu.free_tensor(capture_buf);
        self.profiler.sync_end(gpu, t_rest, 4);
        if let Some(c) = self.block_controller.as_mut() {
            if let Some(tw) = t_window_ms {
                c.observe_timing(tw, n_verify);
            }
            c.observe(accept_len, n_proposed);
        }
        self.profiler.end_window();

        Ok(MtpWindow {
            committed,
            accepted: n_accepted,
            drafts_generated: n_proposed,
        })
    }

    fn mtp_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        // Forced tokens advance the trunk via plain spec_advance; multi-slot
        // DSpark context is invalidated so the next mtp_step re-bootstraps
        // from the new seed (same posture as mtp_prefill on any cache path).
        if tokens.is_empty() {
            return Ok(true);
        }
        if let Some(dev) = self.main_hidden_dev.take() {
            let _ = gpu.free_tensor(dev);
        }
        self.ctx_positions.clear();
        match target.spec_advance(gpu, tokens, start_pos, false, abort, None)? {
            crate::spec::SpecAdvance::Ready { .. } => Ok(true),
            crate::spec::SpecAdvance::Aborted => Ok(true),
        }
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        // Clear multi-slot context so the next prefill re-bootstraps.
        if let Some(dev) = self.main_hidden_dev.take() {
            let _ = gpu.free_tensor(dev);
        }
        self.ctx_positions.clear();
        // Body-owned draft rings / scratch (e.g. DeepSeek dspark_swa_k) must
        // not survive cold reset — otherwise retry sees prior-window KV.
        self.body.reset_for_retry(gpu);
        Ok(())
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        self.profiler.print_summary();
        if let Some(dev) = self.main_hidden_dev {
            let _ = gpu.free_tensor(dev);
        }
        self.body.free(gpu);
    }

    fn k(&self) -> usize {
        self.block
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        // Greedy-only unless the target supports sampled verify. When false, the
        // daemon's `spec_can_sample` admits temp>0 through the CHAIN-style route
        // (→ sampled verify in mtp_step, honoring temp+top_p+top_k).
        !self.supports_temp
    }

    // `supports_temp_verify()` stays the trait default (false): in the daemon it
    // flags a ddtree-SWOR *temperature-only* verify. DSpark is NOT that — its
    // sampled verify honors temp+top_p+top_k (fused `sample_top_p_pf`), i.e. it's
    // the "chain" kind. Sampling capability is signaled by `requires_greedy()`
    // ==false; returning true here would mislabel DSpark as SWOR and drop the
    // nucleus controls.

    fn set_sampling(&mut self, temp: f32, top_p: f32, top_k: usize, cactus_delta: f32) {
        self.temp = temp;
        self.top_p = top_p;
        self.top_k = top_k;
        self.cactus = cactus_delta;
    }
}

/// Build the generic DSpark speculator wrapping any [`DsparkBody`]. The
/// `stage_norm` is the per-stage final RMSNorm weight (deepseek4:
/// `mtp_final_norm`; qwen3: drafter `norm`); `lm_head` is `[vocab, hidden]`.
/// `supports_temp` enables distribution-preserving sampled verify at temp>0
/// (requires the target to implement `verify_block_sampled_capture_gpu`).
#[allow(clippy::too_many_arguments)]
pub fn build_dspark_speculator(
    body: Box<dyn DsparkBody>,
    weights: DsparkWeights,
    stage_norm: GpuTensor,
    lm_head: GpuTensor,
    block: usize,
    ctx_capacity: usize,
    conf_threshold: f32,
    supports_temp: bool,
    max_cost_ratio: f32,
) -> Box<dyn Speculator> {
    let block = block.clamp(1, 8);
    // Default-on; HIPFIRE_DSPARK_ADAPTIVE_BLOCK=0 opts out (fixed block == today).
    let adaptive = hipfire_config::developer_var("HIPFIRE_DSPARK_ADAPTIVE_BLOCK")
        .ok()
        .as_deref()
        != Some("0");
    let block_controller = adaptive.then(|| {
        // Start at a MID block so the hill-climb can grow (high-accept content) or
        // shrink (low-accept content) from a neutral point. min=1, max=cfg.block_size.
        // p*=0.18 prior (a later task measures it live).
        let start_block = 2.min(block).max(1);
        crate::dspark_block_controller::BlockController::new(
            start_block,
            1,
            block,
            0.18,
            max_cost_ratio,
        )
    });
    Box::new(MtpSpeculator::new(DsparkDrafter {
        body,
        weights,
        stage_norm,
        lm_head,
        conf_threshold,
        supports_temp,
        temp: 0.0,
        top_p: 1.0,
        top_k: 0,
        cactus: 0.0,
        rng_state: 0x13579BDF,
        block,
        ctx_capacity,
        main_hidden_dev: None,
        ctx_positions: Vec::new(),
        profiler: DsparkProfiler::new(),
        block_controller,
    }))
}
