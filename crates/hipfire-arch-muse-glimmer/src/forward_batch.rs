// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Continuous-batch decode forward for Muse Glimmer.
//!
//! Lane-major absolute Q8 KV, masked writes, windowed independent attention,
//! current Glimmer layer math/order, final scale/softcap.
//! No DFlash, no graph, no fusions beyond separate batched primitives.

use crate::batch::batch_weight_formats_supported;
use crate::batch::GlimmerDecodeBatchState;
use crate::config::{GlimmerConfig, GlimmerLayerType};
use crate::glimmer::GlimmerWeights;
use hipfire_runtime::llama::{weight_gemv, EmbeddingFormat, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

fn proj_gemm_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    bf16_scratch: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(&w.buf, x, y, w.m, w.k, b)
            .map_err(|e| format!("glimmer batch {label} (q8): {e:?}")),
        DType::MQ4G256 | DType::HFQ4G256 => {
            hipfire_runtime::llama::rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer batch {label} rotate: {e:?}"))?;
            gpu.gemm_hfq4g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("glimmer batch {label} (mq4): {e:?}"))
        }
        DType::MQ6G256 => {
            hipfire_runtime::llama::rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, b)
                .map_err(|e| format!("glimmer batch {label} rotate: {e:?}"))?;
            gpu.gemm_mq6g256_batched_lmhead(&w.buf, x_rot, y, w.m, w.k, b)
                .map_err(|e| format!("glimmer batch {label} (mq6): {e:?}"))
        }
        DType::F32 => {
            // BF16 teacher widened to F32 — routed through GemmF32Batched (Always,
            // gfx942). No new kernel; run_key validates arch + captures for cal.
            use hipfire_dispatch::context::DispatchCtx;
            use hipfire_dispatch::families::gemm::GemmParams;
            use hipfire_dispatch::families::gemv::WeightRef;
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
                x,
                y,
                batch_size: b,
            };
            hipfire_runtime::llama::gemm_family()
                .run_key(
                    hipfire_dispatch::types::KernelKey::GemmF32Batched,
                    &ctx,
                    gpu,
                    &params,
                )
                .map_err(|e| format!("glimmer batch {label} (f32 dispatch): {e:?}"))?;
            Ok(())
        }
        DType::BF16 => {
            // Calibration capture: F32 source before BF16 staging. Zero-cost when
            // unarmed (active_capture is None). Do NOT capture the BF16 tensor.
            gpu.maybe_capture_activation(&w.buf, x, b, w.k);
            // Calibration BF16 — stage F32 activation to BF16 scratch once per
            // projection (hoisted at caller for shared x), then MFMA.
            let x_bf16 = bf16_scratch.sub_offset(0, b * w.k);
            gpu.convert_f32_to_bf16(x, &x_bf16, b * w.k)
                .map_err(|e| format!("glimmer batch {label} f32->bf16: {e:?}"))?;
            use hipfire_dispatch::context::DispatchCtx;
            use hipfire_dispatch::families::gemm::GemmParams;
            use hipfire_dispatch::families::gemv::WeightRef;
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
                x: &x_bf16,
                y,
                batch_size: b,
            };
            hipfire_runtime::llama::gemm_family()
                .run_key(
                    hipfire_dispatch::types::KernelKey::GemmBf16Mfma,
                    &ctx,
                    gpu,
                    &params,
                )
                .map_err(|e| format!("glimmer batch {label} (bf16 dispatch): {e:?}"))?;
            Ok(())
        }
        _ => {
            for i in 0..b {
                let x_row = x.sub_offset(i * w.k, w.k);
                let y_row = y.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)
                    .map_err(|e| format!("glimmer batch {label} row {i}: {e}"))?;
            }
            Ok(())
        }
    }
}

/// BF16 dispatch helper for hoisted shared-x cases: assumes x_bf16 already
/// staged (caller did convert_f32_to_bf16 once). Used for q/k/v/attn_gate and
/// gate/up where one F32 x feeds multiple GEMMs.
fn dispatch_bf16_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_bf16: &GpuTensor,
    y: &GpuTensor,
    b: usize,
    label: &str,
) -> Result<(), String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemm::GemmParams;
    use hipfire_dispatch::families::gemv::WeightRef;
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
        x: x_bf16,
        y,
        batch_size: b,
    };
    hipfire_runtime::llama::gemm_family()
        .run_key(
            hipfire_dispatch::types::KernelKey::GemmBf16Mfma,
            &ctx,
            gpu,
            &params,
        )
        .map_err(|e| format!("glimmer batch {label} (bf16 hoisted): {e:?}"))?;
    Ok(())
}

/// Public entry: stages host tokens/positions into device batch buffers then
/// calls the prepared core. Preserves physical lanes/holes; inactive rows'
/// scratch may be touched but KV and attention are masked.
pub fn forward_decode_batch_glimmer(
    gpu: &mut Gpu,
    weights: &GlimmerWeights,
    cfg: &GlimmerConfig,
    tokens: &[u32],
    positions: &[usize],
    active_mask: u64,
    state: &mut GlimmerDecodeBatchState,
) -> Result<(), String> {
    if tokens.len() != positions.len() {
        return Err(format!(
            "glimmer batch: tokens len {} != positions len {}",
            tokens.len(),
            positions.len()
        ));
    }
    let batch_size = tokens.len();
    if batch_size == 0 {
        return Ok(());
    }
    if batch_size > state.max_batch {
        return Err(format!(
            "glimmer batch: batch_size {batch_size} > max_batch {}",
            state.max_batch
        ));
    }
    // active_mask must not have bits beyond batch_size
    if batch_size < 64 {
        let allowed = if batch_size == 64 {
            u64::MAX
        } else {
            (1u64 << batch_size) - 1
        };
        if active_mask & !allowed != 0 {
            return Err(format!(
                "glimmer batch: active_mask {active_mask:#x} has bits beyond batch_size {batch_size}"
            ));
        }
    }
    if active_mask == 0 {
        return Ok(());
    }
    // positions must fit lane_capacity for active lanes
    for (i, &pos) in positions.iter().enumerate() {
        if (active_mask >> i) & 1 == 0 {
            continue;
        }
        if pos >= state.lane_capacity {
            return Err(format!(
                "glimmer batch: active lane {i} pos {pos} >= lane_capacity {}",
                state.lane_capacity
            ));
        }
    }
    batch_weight_formats_supported(weights)?;
    gpu.scratch.fp16_x_source_ptr = std::ptr::null_mut();
    gpu.scratch.fp8_x_source_ptr = std::ptr::null_mut();
    let tok_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let tok_bytes: Vec<u8> = tok_i32.iter().flat_map(|v| v.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&state.tokens.buf, &tok_bytes)
        .map_err(|e| format!("glimmer batch htod tokens: {e:?}"))?;
    let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let pos_bytes: Vec<u8> = pos_i32.iter().flat_map(|v| v.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&state.positions.buf, &pos_bytes)
        .map_err(|e| format!("glimmer batch htod positions: {e:?}"))?;

    forward_decode_batch_prepared_glimmer(gpu, weights, cfg, batch_size, active_mask, state, true)
}

/// Prepared core: device buffers already staged, `batch_size` and `active_mask`
/// authoritative. `compute_logits` controls final norm/lm_head (true for decode).
pub(crate) fn forward_decode_batch_prepared_glimmer(
    gpu: &mut Gpu,
    weights: &GlimmerWeights,
    cfg: &GlimmerConfig,
    batch_size: usize,
    active_mask: u64,
    state: &mut GlimmerDecodeBatchState,
    compute_logits: bool,
) -> Result<(), String> {
    if batch_size == 0 || active_mask == 0 {
        return Ok(());
    }
    if batch_size > state.max_batch {
        return Err(format!(
            "glimmer batch prepared: batch {batch_size} > max_batch {}",
            state.max_batch
        ));
    }

    // VMM mapping: maximum active physical offset = max(lane*cap + pos +1)
    // Needed before any KV write.
    let mut max_required: usize = 0;
    // Need host positions to compute max; read back from staged device or use
    // caller-provided? We can reuse the host positions by downloading the staged
    // positions buffer (small D2H) or require caller to pass host positions.
    // Instead compute from device positions via download (batch<=64, tiny).
    {
        let mut pos_host = vec![0i32; batch_size];
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(pos_host.as_mut_ptr() as *mut u8, batch_size * 4)
        };
        gpu.hip
            .memcpy_dtoh(bytes, &state.positions.buf)
            .map_err(|e| format!("glimmer batch dtoh positions for VMM: {e:?}"))?;
        for lane in 0..batch_size {
            if (active_mask >> lane) & 1 == 0 {
                continue;
            }
            let pos = pos_host[lane] as usize;
            let req = lane * state.lane_capacity + pos + 1;
            if req > max_required {
                max_required = req;
            }
        }
    }
    if max_required > 0 {
        state
            .kv_sliding
            .ensure_mapped_capacity(gpu, max_required)
            .map_err(|e| format!("glimmer batch kv_sliding ensure_mapped: {e:?}"))?;
        state
            .kv_full
            .ensure_mapped_capacity(gpu, max_required)
            .map_err(|e| format!("glimmer batch kv_full ensure_mapped: {e:?}"))?;
    }

    let dim = cfg.dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let hidden_dim = cfg.hidden_dim;
    let n_heads = cfg.n_heads;
    let n_kv = cfg.n_kv_heads;
    let hd = cfg.head_dim;
    let rms_eps = cfg.rms_norm_eps;
    let post_eps = cfg.post_norm_eps;
    let b = batch_size;

    // batch views (first b rows) — dense physical-lane layout: row i == lane i
    let x = state.x_batch.sub_offset(0, b * dim);
    let residual = state.residual_batch.sub_offset(0, b * dim);
    let tmp = state.tmp_batch.sub_offset(0, b * dim);
    let x_rot = state.x_rot_batch.sub_offset(0, b * dim);
    let q = state.q_batch.sub_offset(0, b * q_dim);
    let k = state.k_batch.sub_offset(0, b * kv_dim);
    let v = state.v_batch.sub_offset(0, b * kv_dim);
    let attn_gate = state.attn_gate_batch.sub_offset(0, b * q_dim);
    let attn_out = state.attn_out_batch.sub_offset(0, b * q_dim);
    let o_out = state.o_out_batch.sub_offset(0, b * dim);
    let gate = state.gate_batch.sub_offset(0, b * hidden_dim);
    let up = state.up_batch.sub_offset(0, b * hidden_dim);
    let ffn_hidden = state.ffn_hidden_batch.sub_offset(0, b * hidden_dim);
    let ffn_out = state.ffn_out_batch.sub_offset(0, b * dim);
    let down_rot = state.down_rot_batch.sub_offset(0, b * hidden_dim);
    let final_hidden_view = state.final_hidden.sub_offset(0, b * dim);
    let logits_view = state.logits.sub_offset(0, b * cfg.vocab_size);
    let pos_view = state.positions.sub_offset(0, b);
    let tok_view = state.tokens.sub_offset(0, b);

    // embedding: batched lookup + scale-less norm into x
    // F32 has no batched lookup kernel — per-token gather loop (B ≤64, negligible
    // vs projections). Same approach sibling lfm2moe uses for F32 teachers.
    {
        let ok = match weights.embd_format {
            EmbeddingFormat::HFQ4G256 => {
                gpu.embedding_lookup_hfq4g256_batched(&weights.embed_tokens, &x, &tok_view, b, dim)
                    .map_err(|e| format!("glimmer batch embed hfq4g256 batched: {e:?}"))?;
                true
            }
            EmbeddingFormat::HFQ4G128 => {
                gpu.embedding_lookup_hfq4g128_batched(&weights.embed_tokens, &x, &tok_view, b, dim)
                    .map_err(|e| format!("glimmer batch embed hfq4g128 batched: {e:?}"))?;
                true
            }
            EmbeddingFormat::Q8_0 => {
                gpu.embedding_lookup_q8_batched(&weights.embed_tokens, &x, &tok_view, b, dim)
                    .map_err(|e| format!("glimmer batch embed q8 batched: {e:?}"))?;
                true
            }
            EmbeddingFormat::F32 => {
                // No batched F32 lookup kernel exists — loop per-token over
                // gpu.embedding_lookup into batch rows (B ≤64, gather cost
                // negligible beside projections). Tokens are staged on device
                // in tok_view; download scalar per row.
                for i in 0..b {
                    let tok = {
                        let mut one = [0u8; 4];
                        let src_ptr = unsafe { (tok_view.buf.as_ptr() as *const u8).add(i * 4) };
                        let tmp_buf = unsafe {
                            hip_bridge::DeviceBuffer::from_raw(src_ptr as *mut std::ffi::c_void, 4)
                        };
                        gpu.hip.memcpy_dtoh(&mut one, &tmp_buf).map_err(|e| {
                            format!("glimmer batch embed f32 fallback dtoh token: {e:?}")
                        })?;
                        std::mem::forget(tmp_buf);
                        u32::from_ne_bytes(one)
                    };
                    let x_row = x.sub_offset(i * dim, dim);
                    gpu.embedding_lookup(&weights.embed_tokens, &x_row, tok, dim)
                        .map_err(|e| format!("glimmer batch embed f32 row {i}: {e:?}"))?;
                }
                true
            }
            EmbeddingFormat::Q4K => false,
        };
        if !ok {
            return Err("glimmer batch: unsupported embedding format Q4K".to_string());
        }
        gpu.rmsnorm_batched(&x, &state.embed_norm_ones, &x, b, dim, rms_eps)
            .map_err(|e| format!("glimmer batch embed_norm batched: {e:?}"))?;
    }

    for layer_idx in 0..cfg.n_layers {
        let slot = state.kv_slot_for_layer[layer_idx];
        let lw = &weights.layers[layer_idx];
        let has_rope = cfg.has_rope(layer_idx);
        let window = cfg.window_for(layer_idx) as usize;

        // residual = x
        gpu.hip
            .memcpy_dtod(&residual.buf, &x.buf, b * dim * 4)
            .map_err(|e| format!("glimmer batch L{layer_idx} save residual: {e:?}"))?;

        // input layernorm -> tmp
        gpu.rmsnorm_batched(&x, &lw.input_layernorm, &tmp, b, dim, rms_eps)
            .map_err(|e| format!("glimmer batch L{layer_idx} input rmsnorm: {e:?}"))?;

        // q/k/v/gate projections: hoist BF16 staging where one x feeds 4 GEMMs.
        // For BF16 teacher, tmp (F32, B*dim) is converted ONCE to BF16 scratch
        // (persistent, sized to max_batch*hidden_dim) and reused for all four.
        // Non-BF16 (Q8/MQ4/F32) keep their existing kernels byte-identically.
        let bf16_scratch = &state.bf16_scratch;
        let q_is_bf16 = lw.q_proj.gpu_dtype == DType::BF16;
        let k_is_bf16 = lw.k_proj.gpu_dtype == DType::BF16;
        let v_is_bf16 = lw.v_proj.gpu_dtype == DType::BF16;
        let gate_is_bf16 = lw.attn_gate_proj.gpu_dtype == DType::BF16;
        let any_qkv_bf16 = q_is_bf16 || k_is_bf16 || v_is_bf16 || gate_is_bf16;
        if any_qkv_bf16 {
            // Calibration capture: shared F32 `tmp` (input layernorm output) before
            // BF16 staging. One capture per BF16 constituent against same tmp.
            if q_is_bf16 {
                gpu.maybe_capture_activation(&lw.q_proj.buf, &tmp, b, lw.q_proj.k);
            }
            if k_is_bf16 {
                gpu.maybe_capture_activation(&lw.k_proj.buf, &tmp, b, lw.k_proj.k);
            }
            if v_is_bf16 {
                gpu.maybe_capture_activation(&lw.v_proj.buf, &tmp, b, lw.v_proj.k);
            }
            if gate_is_bf16 {
                gpu.maybe_capture_activation(&lw.attn_gate_proj.buf, &tmp, b, lw.attn_gate_proj.k);
            }
            // Stage once, reuse for all BF16 projections in this group.
            // BF16 scratch is sized to hidden_dim, but we only need dim prefix.
            let bf16_tmp = bf16_scratch.sub_offset(0, b * dim);
            gpu.convert_f32_to_bf16(&tmp, &bf16_tmp, b * dim)
                .map_err(|e| format!("glimmer batch L{layer_idx} f32->bf16 qkv: {e:?}"))?;
            if q_is_bf16 {
                dispatch_bf16_batched(gpu, &lw.q_proj, &bf16_tmp, &q, b, "q_proj")?;
            } else {
                proj_gemm_batched(gpu, &lw.q_proj, &tmp, &q, &x_rot, bf16_scratch, b, "q_proj")?;
            }
            if k_is_bf16 {
                dispatch_bf16_batched(gpu, &lw.k_proj, &bf16_tmp, &k, b, "k_proj")?;
            } else {
                proj_gemm_batched(gpu, &lw.k_proj, &tmp, &k, &x_rot, bf16_scratch, b, "k_proj")?;
            }
            if v_is_bf16 {
                dispatch_bf16_batched(gpu, &lw.v_proj, &bf16_tmp, &v, b, "v_proj")?;
            } else {
                proj_gemm_batched(gpu, &lw.v_proj, &tmp, &v, &x_rot, bf16_scratch, b, "v_proj")?;
            }
            if gate_is_bf16 {
                dispatch_bf16_batched(
                    gpu,
                    &lw.attn_gate_proj,
                    &bf16_tmp,
                    &attn_gate,
                    b,
                    "attn_gate",
                )?;
            } else {
                proj_gemm_batched(
                    gpu,
                    &lw.attn_gate_proj,
                    &tmp,
                    &attn_gate,
                    &x_rot,
                    bf16_scratch,
                    b,
                    "attn_gate",
                )?;
            }
        } else {
            proj_gemm_batched(gpu, &lw.q_proj, &tmp, &q, &x_rot, bf16_scratch, b, "q_proj")?;
            proj_gemm_batched(gpu, &lw.k_proj, &tmp, &k, &x_rot, bf16_scratch, b, "k_proj")?;
            proj_gemm_batched(gpu, &lw.v_proj, &tmp, &v, &x_rot, bf16_scratch, b, "v_proj")?;
            proj_gemm_batched(
                gpu,
                &lw.attn_gate_proj,
                &tmp,
                &attn_gate,
                &x_rot,
                bf16_scratch,
                b,
                "attn_gate",
            )?;
        }
        gpu.rmsnorm_batched(&k, &state.qk_norm_ones, &k, b * n_kv, hd, rms_eps)
            .map_err(|e| format!("glimmer batch L{layer_idx} k_norm: {e:?}"))?;
        gpu.scale_f32(&q, cfg.qk_scale_factor)
            .map_err(|e| format!("glimmer batch L{layer_idx} q scale: {e:?}"))?;

        if has_rope {
            let theta = cfg.rope_theta_for(layer_idx);
            gpu.rope_batched_f32(&q, &k, &pos_view, n_heads, n_kv, hd, theta, b)
                .map_err(|e| format!("glimmer batch L{layer_idx} rope batched: {e:?}"))?;
        }

        // masked KV writes
        match cfg.layer_types[layer_idx] {
            GlimmerLayerType::Sliding => {
                let k_cache = &state.kv_sliding.k_gpu[slot];
                let v_cache = &state.kv_sliding.v_gpu[slot];
                gpu.kv_cache_write_q8_0_independent_masked(
                    k_cache,
                    &k,
                    &pos_view,
                    n_kv,
                    hd,
                    b,
                    state.lane_capacity,
                    active_mask,
                )
                .map_err(|e| format!("glimmer batch L{layer_idx} kv k masked: {e:?}"))?;
                gpu.kv_cache_write_q8_0_independent_masked(
                    v_cache,
                    &v,
                    &pos_view,
                    n_kv,
                    hd,
                    b,
                    state.lane_capacity,
                    active_mask,
                )
                .map_err(|e| format!("glimmer batch L{layer_idx} kv v masked: {e:?}"))?;
                // windowed independent attention
                gpu.attention_q8_0_kv_independent_masked_windowed(
                    &q,
                    k_cache,
                    v_cache,
                    &attn_out,
                    &pos_view,
                    n_heads,
                    n_kv,
                    hd,
                    state.lane_capacity,
                    state.lane_capacity,
                    b,
                    active_mask,
                    window,
                )
                .map_err(|e| format!("glimmer batch L{layer_idx} attn sliding: {e:?}"))?;
            }
            GlimmerLayerType::Full => {
                let k_cache = &state.kv_full.k_gpu[slot];
                let v_cache = &state.kv_full.v_gpu[slot];
                gpu.kv_cache_write_q8_0_independent_masked(
                    k_cache,
                    &k,
                    &pos_view,
                    n_kv,
                    hd,
                    b,
                    state.lane_capacity,
                    active_mask,
                )
                .map_err(|e| format!("glimmer batch L{layer_idx} kv k masked: {e:?}"))?;
                gpu.kv_cache_write_q8_0_independent_masked(
                    v_cache,
                    &v,
                    &pos_view,
                    n_kv,
                    hd,
                    b,
                    state.lane_capacity,
                    active_mask,
                )
                .map_err(|e| format!("glimmer batch L{layer_idx} kv v masked: {e:?}"))?;
                gpu.attention_q8_0_kv_independent_masked_windowed(
                    &q,
                    k_cache,
                    v_cache,
                    &attn_out,
                    &pos_view,
                    n_heads,
                    n_kv,
                    hd,
                    state.lane_capacity,
                    state.lane_capacity,
                    b,
                    active_mask,
                    0,
                )
                .map_err(|e| format!("glimmer batch L{layer_idx} attn full: {e:?}"))?;
            }
        }

        // gated attention
        gpu.sigmoid_mul_f32(&attn_out, &attn_gate)
            .map_err(|e| format!("glimmer batch L{layer_idx} sigmoid_mul: {e:?}"))?;

        // o_proj
        proj_gemm_batched(
            gpu,
            &lw.o_proj,
            &attn_out,
            &o_out,
            &x_rot,
            bf16_scratch,
            b,
            "o_proj",
        )?;

        // sandwich post-attention norm + residual: x = residual + norm(o_out)
        gpu.rmsnorm_batched(
            &o_out,
            &lw.post_attention_layernorm,
            &o_out,
            b,
            dim,
            post_eps,
        )
        .map_err(|e| format!("glimmer batch L{layer_idx} post_attn rmsnorm: {e:?}"))?;
        // x currently holds pre-attn value, residual holds same; we want x = residual + o_out
        // reset x to residual then add
        gpu.hip
            .memcpy_dtod(&x.buf, &residual.buf, b * dim * 4)
            .map_err(|e| format!("glimmer batch L{layer_idx} reset x attn: {e:?}"))?;
        gpu.add_inplace_f32(&x, &o_out)
            .map_err(|e| format!("glimmer batch L{layer_idx} attn residual add: {e:?}"))?;

        // residual = x for FFN
        gpu.hip
            .memcpy_dtod(&residual.buf, &x.buf, b * dim * 4)
            .map_err(|e| format!("glimmer batch L{layer_idx} save ffn residual: {e:?}"))?;

        // pre_ffn norm
        gpu.rmsnorm_batched(&x, &lw.pre_feedforward_layernorm, &tmp, b, dim, rms_eps)
            .map_err(|e| format!("glimmer batch L{layer_idx} pre_ffn rmsnorm: {e:?}"))?;

        // gate/up share same tmp (pre_ffn norm output) — hoist BF16 staging once.
        let gate_is_bf16 = lw.gate_proj.gpu_dtype == DType::BF16;
        let up_is_bf16 = lw.up_proj.gpu_dtype == DType::BF16;
        let any_gate_up_bf16 = gate_is_bf16 || up_is_bf16;
        if any_gate_up_bf16 {
            // Calibration capture: shared F32 `tmp` (pre_ffn norm output) before BF16 staging.
            if gate_is_bf16 {
                gpu.maybe_capture_activation(&lw.gate_proj.buf, &tmp, b, lw.gate_proj.k);
            }
            if up_is_bf16 {
                gpu.maybe_capture_activation(&lw.up_proj.buf, &tmp, b, lw.up_proj.k);
            }
            let bf16_tmp2 = bf16_scratch.sub_offset(0, b * dim);
            gpu.convert_f32_to_bf16(&tmp, &bf16_tmp2, b * dim)
                .map_err(|e| format!("glimmer batch L{layer_idx} f32->bf16 gate_up: {e:?}"))?;
            if gate_is_bf16 {
                dispatch_bf16_batched(gpu, &lw.gate_proj, &bf16_tmp2, &gate, b, "gate_proj")?;
            } else {
                proj_gemm_batched(
                    gpu,
                    &lw.gate_proj,
                    &tmp,
                    &gate,
                    &x_rot,
                    bf16_scratch,
                    b,
                    "gate_proj",
                )?;
            }
            if up_is_bf16 {
                dispatch_bf16_batched(gpu, &lw.up_proj, &bf16_tmp2, &up, b, "up_proj")?;
            } else {
                proj_gemm_batched(
                    gpu,
                    &lw.up_proj,
                    &tmp,
                    &up,
                    &x_rot,
                    bf16_scratch,
                    b,
                    "up_proj",
                )?;
            }
        } else {
            proj_gemm_batched(
                gpu,
                &lw.gate_proj,
                &tmp,
                &gate,
                &x_rot,
                bf16_scratch,
                b,
                "gate_proj",
            )?;
            proj_gemm_batched(
                gpu,
                &lw.up_proj,
                &tmp,
                &up,
                &x_rot,
                bf16_scratch,
                b,
                "up_proj",
            )?;
        }

        gpu.silu_mul_f32(&gate, &up, &ffn_hidden)
            .map_err(|e| format!("glimmer batch L{layer_idx} silu_mul: {e:?}"))?;

        proj_gemm_batched(
            gpu,
            &lw.down_proj,
            &ffn_hidden,
            &ffn_out,
            &down_rot,
            bf16_scratch,
            b,
            "down_proj",
        )?;
        gpu.rmsnorm_batched(
            &ffn_out,
            &lw.post_feedforward_layernorm,
            &ffn_out,
            b,
            dim,
            post_eps,
        )
        .map_err(|e| format!("glimmer batch L{layer_idx} post_ffn rmsnorm: {e:?}"))?;
        gpu.hip
            .memcpy_dtod(&x.buf, &residual.buf, b * dim * 4)
            .map_err(|e| format!("glimmer batch L{layer_idx} reset x ffn: {e:?}"))?;
        gpu.add_inplace_f32(&x, &ffn_out)
            .map_err(|e| format!("glimmer batch L{layer_idx} ffn residual add: {e:?}"))?;
    }

    if !compute_logits {
        return Ok(());
    }

    // final norm -> final_hidden
    gpu.rmsnorm_batched(
        &x,
        &weights.final_norm,
        &final_hidden_view,
        b,
        dim,
        cfg.rms_norm_eps,
    )
    .map_err(|e| format!("glimmer batch final rmsnorm: {e:?}"))?;

    // lm_head -> logits (batched)
    // Q8_0: batched chunked (WMMA). F32: generic batched GEMM (GemmF32Batched,
    // Always on gfx942). BF16: MFMA (GemmBf16Mfma on gfx942) via convert + dispatch.
    // No dedicated F32/BF16 batched-lmhead kernel exists — route through generic.
    match weights.lm_head.gpu_dtype {
        DType::Q8_0 => {
            gpu.gemm_q8_0_batched_chunked(
                &weights.lm_head.buf,
                &final_hidden_view,
                &logits_view,
                cfg.vocab_size,
                dim,
                b,
            )
            .map_err(|e| format!("glimmer batch lm_head q8 batched: {e:?}"))?;
        }
        DType::F32 => {
            use hipfire_dispatch::context::DispatchCtx;
            use hipfire_dispatch::families::gemm::GemmParams;
            use hipfire_dispatch::families::gemv::WeightRef;
            let ctx = DispatchCtx::new(gpu);
            let w_ref = WeightRef {
                buf: &weights.lm_head.buf,
                dtype: weights.lm_head.gpu_dtype,
                m: cfg.vocab_size,
                k: dim,
                row_stride: dim,
                rotation: None,
                awq_scale: None,
            };
            let params = GemmParams {
                w: &w_ref,
                x: &final_hidden_view,
                y: &logits_view,
                batch_size: b,
            };
            hipfire_runtime::llama::gemm_family()
                .run_key(
                    hipfire_dispatch::types::KernelKey::GemmF32Batched,
                    &ctx,
                    gpu,
                    &params,
                )
                .map_err(|e| format!("glimmer batch lm_head f32 dispatch: {e:?}"))?;
        }
        DType::BF16 => {
            // Calibration capture: F32 `final_hidden_view` before BF16 staging. Single projection.
            gpu.maybe_capture_activation(&weights.lm_head.buf, &final_hidden_view, b, dim);
            // BF16 lm_head: stage F32 hidden to BF16 scratch (persistent, sized to hidden_dim)
            // then MFMA. Reuses same bf16_scratch as projections (max hidden_dim).
            let bf16_hidden = state.bf16_scratch.sub_offset(0, b * dim);
            gpu.convert_f32_to_bf16(&final_hidden_view, &bf16_hidden, b * dim)
                .map_err(|e| format!("glimmer batch lm_head bf16 convert: {e:?}"))?;
            use hipfire_dispatch::context::DispatchCtx;
            use hipfire_dispatch::families::gemm::GemmParams;
            use hipfire_dispatch::families::gemv::WeightRef;
            let ctx = DispatchCtx::new(gpu);
            let w_ref = WeightRef {
                buf: &weights.lm_head.buf,
                dtype: weights.lm_head.gpu_dtype,
                m: cfg.vocab_size,
                k: dim,
                row_stride: dim,
                rotation: None,
                awq_scale: None,
            };
            let params = GemmParams {
                w: &w_ref,
                x: &bf16_hidden,
                y: &logits_view,
                batch_size: b,
            };
            hipfire_runtime::llama::gemm_family()
                .run_key(
                    hipfire_dispatch::types::KernelKey::GemmBf16Mfma,
                    &ctx,
                    gpu,
                    &params,
                )
                .map_err(|e| format!("glimmer batch lm_head bf16 dispatch: {e:?}"))?;
        }
        _ => {
            for i in 0..b {
                let h_row = final_hidden_view.sub_offset(i * dim, dim);
                let l_row = logits_view.sub_offset(i * cfg.vocab_size, cfg.vocab_size);
                weight_gemv(gpu, &weights.lm_head, &h_row, &l_row)
                    .map_err(|e| format!("glimmer batch lm_head row {i}: {e}"))?;
            }
        }
    }
    if cfg.output_multiplier != 1.0 {
        gpu.scale_f32(&logits_view, cfg.output_multiplier)
            .map_err(|e| format!("glimmer batch output_multiplier: {e:?}"))?;
    }
    // Per-lane isolation: scaling is per-element on the dense b*vocab view, so holes stay inert
    if cfg.final_logit_softcapping > 0.0 {
        gpu.logit_softcap_f32(
            &logits_view,
            b * cfg.vocab_size,
            cfg.final_logit_softcapping,
        )
        .map_err(|e| format!("glimmer batch softcap: {e:?}"))?;
    }
    Ok(())
}
