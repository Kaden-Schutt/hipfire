// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 single-GPU decode layer loop (`forward_scratch_layers`): the
//! hand-written per-layer decode arms (dense + MoE, DeltaNet + FullAttn).
//! Dispatches to the lowered super-op path when `HIPFIRE_FORWARD_LOWERED`.

use super::*;

pub(crate) fn forward_scratch_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &mut kv::KvCache,
    dn_state: &mut DeltaNetState,
    s: &Qwen35Scratch,
    hidden_rb: Option<&mut HiddenStateRingBuffer>,
    needs_last_token_logits: bool,
    mut gdn_tape_capture: Option<(&mut crate::speculative::GdnTape, usize)>,
) -> HipResult<()> {
    ensure_qwen35_forward_capability(config)?;
    let _dim = config.dim;
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let _qkv_dim = k_dim * 2 + v_dim;
    // #397 Ship 6 — forward-as-pipeline. When HIPFIRE_FORWARD_LOWERED=1, route
    // single-GPU decode through the lowered super-op executor. Skipped when a
    // hidden-state ring buffer or GDN tape capture is active (spec-decode
    // capture engages only the hand path for now). Default off → the hand arms
    // below run unchanged.
    // RoughQuant corrections are wired into THIS hand path, but the hand path is
    // currently broken (bf16 self-KLD 13.89 vs lowered 0.000 — see
    // docs/roughquant/phase3-real-format-scope.md). Until it is resurrected OR the
    // correction is wired into the lowered super-op executor, route rq models to
    // the hand path ONLY under the opt-in HIPFIRE_RQ_HAND=1 (experiments); by
    // default rq models use the working (uncorrected) lowered path so they stay
    // coherent. The correction stack stays as a proven, dormant foundation.
    let rq_hand_optin = !weights.rq_corrections.is_empty()
        && std::env::var("HIPFIRE_RQ_HAND").as_deref() == Ok("1");
    if forward_lowered_enabled()
        && hidden_rb.is_none()
        && gdn_tape_capture.is_none()
        && !rq_hand_optin
    {
        return forward_scratch_layers_lowered(
            gpu,
            weights,
            config,
            pos,
            kv_cache,
            dn_state,
            s,
            needs_last_token_logits,
        );
    }

    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    let ctx = DispatchCtx::new(gpu);

    let mut delta_layer_idx = 0usize;
    let mut _kv_layer_idx = 0usize;

    for layer_idx in 0..config.n_layers {
        match (&weights.layers[layer_idx], config.layer_types[layer_idx]) {
            (LayerWeights::DeltaNet(layer), LayerType::LinearAttention) => {
                // ── DeltaNet QKVZA via pipeline ──
                qkvza_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wqkv,
                    &layer.wz,
                    &layer.w_beta,
                    &layer.w_alpha,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                )?;
                let x_rot: Option<&GpuTensor> = Some(&s.x_rot);
                // Lever 1 — Fused rmsnorm + PARO per-group rotation for wqkv.
                let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                    && layer.wqkv.gpu_dtype == DType::ParoQ4G128
                    && layer.wqkv.k % 128 == 0
                    && layer.wqkv.m % 8 == 0
                {
                    fused_rmsnorm_rotate_for_paro(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?
                } else {
                    None
                };
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 LA attn_norm", &s.tmp)?;
                }
                // Cross-arch fast path: one fused 4-way projection kernel
                // (wqkv + wz + w_beta + w_alpha) in a single launch. Works
                // for BOTH MQ4 (weights FWHT-rotated, input x_rot FWHT-rotated)
                // and HF4 (weights not rotated, input is plain rmsnormed x).
                // The kernel math is the same — it's a gemv_hfq4g256 inner
                // loop; MQ4 and HF4 just live in different "rotated spaces"
                // and the caller hands the matching x. Inner loop is unified
                // across all RDNA generations after the 5302926 4-accumulator
                // port to gemv_hfq4g256.hip.
                let dt = layer.wqkv.gpu_dtype;
                let la4_same_dtype = layer.wz.gpu_dtype == dt
                    && layer.w_beta.gpu_dtype == dt
                    && layer.w_alpha.gpu_dtype == dt;
                let fused_la4_mq4 =
                    la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                let fused_la4_f16 = la4_same_dtype && dt == DType::F16;
                let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                let fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                let fused_la4_paro4t = la4_same_dtype
                    && dt == DType::ParoQ4G128
                    && x_rot_paro.is_none()
                    && std::env::var_os("HIPFIRE_PARO_LA4_FUSED").is_some();
                let fused_la2_paro4t = dt == DType::ParoQ4G128
                    && layer.wz.gpu_dtype == DType::ParoQ4G128
                    && x_rot_paro.is_none()
                    && std::env::var_os("HIPFIRE_PARO_LA2_FUSED").is_some();
                // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
                let fused_la4_hfq6 = la4_same_dtype
                    && (dt == DType::MQ6G256 || dt == DType::HFQ6G256)
                    && gpu.arch_caps.gemv_dp4a_enabled();
                if fused_la4_f16 {
                    gpu.fused_qkvza_f16_xf32(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &s.tmp,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la4_mq4 {
                    // MQ4: x_rot is Some(rotated x); HF4: x_rot is None and
                    // s.tmp holds the plain rmsnormed x from the fallback path.
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkvza_hfq4g256(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la4_lloyd_mq3 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkvza_mq3g256_lloyd(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la4_lloyd_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkvza_mq4g256_lloyd(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la4_hfq6 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkvza_hfq6g256_dp4a(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la4_paro4t {
                    gpu.fused_qkvza_paro4g128t(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &s.tmp,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        &s.x_rot,
                        &s.ffn_hidden,
                        &s.ffn_out,
                        &s.o,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la2_paro4t {
                    gpu.fused_qkvza_paro4g128t(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.wqkv.buf,
                        &layer.wqkv.buf,
                        &s.tmp,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        &s.x_rot,
                        &s.ffn_hidden,
                        &s.ffn_out,
                        &s.o,
                        layer.wqkv.m,
                        layer.wz.m,
                        0,
                        0,
                        layer.wqkv.k,
                    )?;
                    weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                    weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                } else {
                    if let Some(xr_first) = x_rot_paro {
                        gpu.gemv_paro4g128t_prerotated(
                            &layer.wqkv.buf,
                            xr_first,
                            &s.dn_qkv,
                            layer.wqkv.m,
                            layer.wqkv.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                    }
                    weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                    weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                    weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                }
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 LA wqkv", &s.dn_qkv)?;
                    trace_finite_if_enabled(gpu, "layer 0 LA wz", &s.dn_z)?;
                    trace_finite_if_enabled(gpu, "layer 0 LA w_beta", &s.dn_beta)?;
                    trace_finite_if_enabled(gpu, "layer 0 LA w_alpha", &s.dn_alpha)?;
                }
                // RoughQuant residual-reader corrections (DeltaNet in_proj_*).
                rq_apply_readers(
                    gpu,
                    weights,
                    layer_idx,
                    &layer.attn_norm,
                    &s.x,
                    config.norm_eps,
                    config.dim,
                    &[
                        (RqProj::Wqkv, &s.dn_qkv),
                        (RqProj::Wz, &s.dn_z),
                        (RqProj::Walpha, &s.dn_alpha),
                        (RqProj::Wbeta, &s.dn_beta),
                    ],
                )?;
                if layer_idx == 0
                    && dflash_serial_qkvza_self_compare_enabled()
                    && gdn_tape_capture.is_some()
                    && fused_la4_mq4
                {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    let probe_qkv = gpu.alloc_tensor(&[layer.wqkv.m], DType::F32)?;
                    let probe_z = gpu.alloc_tensor(&[layer.wz.m], DType::F32)?;
                    let probe_beta = gpu.alloc_tensor(&[layer.w_beta.m], DType::F32)?;
                    let probe_alpha = gpu.alloc_tensor(&[layer.w_alpha.m], DType::F32)?;
                    gpu.fused_qkvza_hfq4g256(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        eff_x,
                        &probe_qkv,
                        &probe_z,
                        &probe_beta,
                        &probe_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                    let probe_qkv_host = gpu.download_f32(&probe_qkv)?;
                    let serial_qkv_host = gpu.download_f32(&s.dn_qkv)?;
                    log_dflash_serial_qkvza_self_diff(
                        "qkv",
                        layer_idx,
                        pos,
                        &probe_qkv_host,
                        &serial_qkv_host,
                    );
                    let probe_z_host = gpu.download_f32(&probe_z)?;
                    let serial_z_host = gpu.download_f32(&s.dn_z)?;
                    log_dflash_serial_qkvza_self_diff(
                        "z",
                        layer_idx,
                        pos,
                        &probe_z_host,
                        &serial_z_host,
                    );
                    let probe_beta_host = gpu.download_f32(&probe_beta)?;
                    let serial_beta_host = gpu.download_f32(&s.dn_beta)?;
                    log_dflash_serial_qkvza_self_diff(
                        "beta_raw",
                        layer_idx,
                        pos,
                        &probe_beta_host,
                        &serial_beta_host,
                    );
                    let probe_alpha_host = gpu.download_f32(&probe_alpha)?;
                    let serial_alpha_host = gpu.download_f32(&s.dn_alpha)?;
                    log_dflash_serial_qkvza_self_diff(
                        "alpha_raw",
                        layer_idx,
                        pos,
                        &probe_alpha_host,
                        &serial_alpha_host,
                    );
                    gpu.free_tensor(probe_qkv)?;
                    gpu.free_tensor(probe_z)?;
                    gpu.free_tensor(probe_beta)?;
                    gpu.free_tensor(probe_alpha)?;
                }
                // gfx1151 fuses the independent sigmoid/alpha gate with the
                // single-token conv1d+SiLU+split below, shaving one dispatch
                // per LA decode layer. Other arches preserve the old pair.
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let x_in_for_tape = x_rot_paro.or(x_rot).unwrap_or(&s.tmp);
                    let x_in_row_bytes = tape.x_in_dim * 4;
                    let alpha_beta_row_bytes = tape.n_v_heads * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.x_in_bufs[delta_layer_idx].buf,
                        *tape_row * x_in_row_bytes,
                        &x_in_for_tape.buf,
                        0,
                        x_in_row_bytes,
                    )?;
                    if layer_idx == 0 && dflash_serial_tape_x_in_compare_enabled() {
                        gpu.hip.device_synchronize()?;
                        let captured_x_in = gpu.alloc_tensor(&[tape.x_in_dim], DType::F32)?;
                        gpu.memcpy_dtod_at_auto(
                            &captured_x_in.buf,
                            0,
                            &tape.x_in_bufs[delta_layer_idx].buf,
                            *tape_row * x_in_row_bytes,
                            x_in_row_bytes,
                        )?;
                        gpu.hip.device_synchronize()?;
                        let source_host = gpu.download_f32(x_in_for_tape)?;
                        let captured_host = gpu.download_f32(&captured_x_in)?;
                        log_dflash_serial_tape_x_in_diff(
                            layer_idx,
                            pos,
                            *tape_row,
                            &source_host,
                            &captured_host,
                        );
                        gpu.free_tensor(captured_x_in)?;
                    }
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_raw_bufs[delta_layer_idx].buf,
                        *tape_row * alpha_beta_row_bytes,
                        &s.dn_alpha.buf,
                        0,
                        alpha_beta_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_raw_bufs[delta_layer_idx].buf,
                        *tape_row * alpha_beta_row_bytes,
                        &s.dn_beta.buf,
                        0,
                        alpha_beta_row_bytes,
                    )?;
                }
                gpu.fused_sigmoid_alpha_gate_f32(
                    &s.dn_beta,
                    &s.dn_alpha,
                    &layer.dt_bias,
                    &layer.a_log,
                    n_v_heads,
                )?;

                gpu.conv1d_silu_split_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    &s.dn_v,
                    &s.dn_qkv,
                    &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    k_dim,
                    v_dim,
                )?;
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let qkv_row_bytes = tape.qkv_dim * 4;
                    let alpha_beta_row_bytes = tape.n_v_heads * 4;
                    let q_raw_row_bytes = tape.k_dim * 4;
                    let v_row_bytes = tape.v_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.qkv_bufs[delta_layer_idx].buf,
                        *tape_row * qkv_row_bytes,
                        &s.dn_qkv.buf,
                        0,
                        qkv_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.alpha_bufs[delta_layer_idx].buf,
                        *tape_row * alpha_beta_row_bytes,
                        &s.dn_alpha.buf,
                        0,
                        alpha_beta_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.beta_bufs[delta_layer_idx].buf,
                        *tape_row * alpha_beta_row_bytes,
                        &s.dn_beta.buf,
                        0,
                        alpha_beta_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_raw_bufs[delta_layer_idx].buf,
                        *tape_row * q_raw_row_bytes,
                        &s.dn_q_raw.buf,
                        0,
                        q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_raw_bufs[delta_layer_idx].buf,
                        *tape_row * q_raw_row_bytes,
                        &s.dn_k_raw.buf,
                        0,
                        q_raw_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.v_bufs[delta_layer_idx].buf,
                        *tape_row * v_row_bytes,
                        &s.dn_v.buf,
                        0,
                        v_row_bytes,
                    )?;
                }
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 LA beta", &s.dn_beta)?;
                    trace_finite_if_enabled(gpu, "layer 0 LA alpha", &s.dn_alpha)?;
                    trace_finite_if_enabled(gpu, "layer 0 LA conv q", &s.dn_q_raw)?;
                    trace_finite_if_enabled(gpu, "layer 0 LA conv k", &s.dn_k_raw)?;
                    trace_finite_if_enabled(gpu, "layer 0 LA conv v", &s.dn_v)?;
                }

                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;

                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let q_row_bytes = tape.v_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.q_bufs[delta_layer_idx].buf,
                        *tape_row * q_row_bytes,
                        &s.dn_q.buf,
                        0,
                        q_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.k_bufs[delta_layer_idx].buf,
                        *tape_row * q_row_bytes,
                        &s.dn_k.buf,
                        0,
                        q_row_bytes,
                    )?;
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                }

                gpu.gated_norm_f32(
                    &s.dn_attn_out,
                    &s.dn_z,
                    &layer.norm_weight,
                    &s.dn_normed,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                )?;
                {
                    let wr = layer.wo.dispatch_ref();
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input: GemvInput::Raw(&s.dn_normed),
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let v_row_bytes = tape.v_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_out_bufs[delta_layer_idx].buf,
                        *tape_row * v_row_bytes,
                        &s.dn_attn_out.buf,
                        0,
                        v_row_bytes,
                    )?;
                }

                // Phase-A norm-recovery capture: the FFN-norm INPUT (what gate_up
                // normalizes). The trainer recomputes the bf16 FFN output from this
                // (so no FFN-output capture is needed — avoids the attention-residual
                // contamination that the residual-difference targets had).
                dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "premlp");
                // ── FFN ──
                gate_up_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.w_gate,
                    &layer.w_up,
                    &layer.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                )?;
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 LA gated norm", &s.dn_normed)?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let v_row_bytes = tape.v_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.normed_bufs[delta_layer_idx].buf,
                        *tape_row * v_row_bytes,
                        &s.dn_normed.buf,
                        0,
                        v_row_bytes,
                    )?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_residual_in_bufs[delta_layer_idx].buf,
                        *tape_row * hidden_row_bytes,
                        &s.x.buf,
                        0,
                        hidden_row_bytes,
                    )?;
                }
                // Fused wo GEMV + residual add: s.x += layer.wo * s.dn_normed
                weight_gemv_residual(gpu, &layer.wo, &s.dn_normed, &s.x)?;
                // RoughQuant residual-writer correction (DeltaNet out_proj rows).
                if let Some(c) = weights.rq_corrections.get(&(layer_idx as u32, RqProj::Wo)) {
                    rq_apply_writer(gpu, c, &s.dn_normed, &s.x)?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let v_row_bytes = tape.v_dim * 4;
                    let wo_input = if matches!(
                        layer.wo.gpu_dtype,
                        DType::MQ4G256
                            | DType::MQ6G256
                            | DType::MQ3G256
                            | DType::MQ3G256Lloyd
                            | DType::MFP4G32
                    ) {
                        gpu.mq_x_rot.as_ref().unwrap()
                    } else {
                        &s.dn_normed
                    };
                    gpu.memcpy_dtod_at_auto(
                        &tape.wo_input_bufs[delta_layer_idx].buf,
                        *tape_row * v_row_bytes,
                        &wo_input.buf,
                        0,
                        v_row_bytes,
                    )?;
                }
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 LA wo residual", &s.x)?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.attn_residual_bufs[delta_layer_idx].buf,
                        *tape_row * hidden_row_bytes,
                        &s.x.buf,
                        0,
                        hidden_row_bytes,
                    )?;
                }

                hipfire_runtime::weights::weight_gemv_swiglu_residual(
                    gpu,
                    &layer.w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;
                // Lever 1 — Fused rmsnorm + PARO per-group rotation for w_gate.
                let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                    && layer.w_gate.gpu_dtype == DType::ParoQ4G128
                    && layer.w_gate.k % 128 == 0
                    && layer.w_gate.m % 8 == 0
                {
                    fused_rmsnorm_rotate_for_paro(
                        gpu,
                        &layer.w_gate,
                        &s.x,
                        &layer.ffn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?
                } else {
                    None
                };
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 FFN norm", &s.tmp)?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    let ffn_input = x_rot_paro.or(x_rot).unwrap_or(&s.tmp);
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_input_bufs[delta_layer_idx].buf,
                        *tape_row * hidden_row_bytes,
                        &ffn_input.buf,
                        0,
                        hidden_row_bytes,
                    )?;
                }
                // Cross-arch fast path: fused gate+up in one launch. Works
                // for both MQ4 (x_rot Some) and HF4 (x_rot None → s.tmp).
                let dt_g = layer.w_gate.gpu_dtype;
                let same_dtype = layer.w_up.gpu_dtype == dt_g;
                let fused_gu_mq4 =
                    same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                let fused_gu_f16 = same_dtype && dt_g == DType::F16;
                let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                let fused_gu_paro4t = same_dtype
                    && dt_g == DType::ParoQ4G128
                    && layer.w_gate.m == layer.w_up.m
                    && layer.w_gate.k == layer.w_up.k
                    && x_rot_paro.is_none()
                    && std::env::var("HIPFIRE_PARO_GATE_UP_FUSED")
                        .map(|v| v != "0")
                        .unwrap_or(true);
                // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
                let fused_gu_hfq6 = same_dtype
                    && (dt_g == DType::MQ6G256 || dt_g == DType::HFQ6G256)
                    && gpu.arch_caps.gemv_dp4a_enabled();
                if fused_gu_f16 {
                    gpu.fused_gate_up_f16_xf32(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &s.tmp,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_hfq4g256(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_lloyd_mq3 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_mq3g256_lloyd(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_lloyd_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_mq4g256_lloyd(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_hfq6 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_hfq6g256_dp4a(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_paro4t {
                    gpu.fused_gate_up_paro4g128t(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &s.tmp,
                        &s.gate_ffn,
                        &s.up,
                        &s.x_rot,
                        layer.w_gate.m,
                        layer.w_gate.k,
                    )?;
                } else {
                    if let Some(xr_first) = x_rot_paro {
                        gpu.gemv_paro4g128t_prerotated(
                            &layer.w_gate.buf,
                            xr_first,
                            &s.gate_ffn,
                            layer.w_gate.m,
                            layer.w_gate.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
                    }
                    weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                }
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 FFN gate", &s.gate_ffn)?;
                    trace_finite_if_enabled(gpu, "layer 0 FFN up", &s.up)?;
                }
                // RoughQuant residual-reader corrections (DeltaNet mlp gate/up).
                // Applied before SwiGLU so the corrected gate/up feed w_down.
                rq_apply_readers(
                    gpu,
                    weights,
                    layer_idx,
                    &layer.ffn_norm,
                    &s.x,
                    config.norm_eps,
                    config.dim,
                    &[(RqProj::Wgate, &s.gate_ffn), (RqProj::Wup, &s.up)],
                )?;
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let ffn_row_bytes = tape.ffn_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_gate_bufs[delta_layer_idx].buf,
                        *tape_row * ffn_row_bytes,
                        &s.gate_ffn.buf,
                        0,
                        ffn_row_bytes,
                    )?;
                    gpu.memcpy_dtod_at_auto(
                        &tape.ffn_up_bufs[delta_layer_idx].buf,
                        *tape_row * ffn_row_bytes,
                        &s.up.buf,
                        0,
                        ffn_row_bytes,
                    )?;
                }
                // Fused SwiGLU + w_down residual GEMV:
                //   MQ4: fused_silu_rotate(gate,up) + gemv_residual(w_down, rotated, x)
                //   HF4: silu_mul + weight_gemv_residual (unchanged)
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.w_down_residual_in_bufs[delta_layer_idx].buf,
                        *tape_row * hidden_row_bytes,
                        &s.x.buf,
                        0,
                        hidden_row_bytes,
                    )?;
                }
                weight_gemv_swiglu_residual_bf16_probe(
                    gpu,
                    layer_idx,
                    &layer.w_down,
                    &layer.bf16_down_shadow,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;
                // RoughQuant residual-writer correction (DeltaNet mlp.down_proj).
                // w_down's logical input is silu(gate)*up in ORIGINAL frame (the
                // fused kernel rotates internally), so recompute it explicitly.
                if let Some(c) = weights
                    .rq_corrections
                    .get(&(layer_idx as u32, RqProj::Wdown))
                {
                    let inp = gpu.zeros_owned(&[layer.w_down.k], DType::F32)?;
                    gpu.silu_mul_f32(&s.gate_ffn, &s.up, &inp)?;
                    rq_apply_writer(gpu, c, &inp, &s.x)?;
                    drop(inp); // enqueue this layer's RAII scratch, then drain.
                    gpu.reclaim_pending();
                }
                if layer_idx == 0 {
                    trace_finite_if_enabled(gpu, "layer 0 FFN residual", &s.x)?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let ffn_row_bytes = tape.ffn_dim * 4;
                    let w_down_input = if matches!(
                        layer.w_down.gpu_dtype,
                        DType::MQ4G256
                            | DType::MQ6G256
                            | DType::MQ3G256
                            | DType::MQ3G256Lloyd
                            | DType::MFP4G32
                    ) {
                        gpu.mq_x_rot.as_ref().unwrap()
                    } else {
                        &s.ffn_hidden
                    };
                    gpu.memcpy_dtod_at_auto(
                        &tape.w_down_input_bufs[delta_layer_idx].buf,
                        *tape_row * ffn_row_bytes,
                        &w_down_input.buf,
                        0,
                        ffn_row_bytes,
                    )?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    let hidden_row_bytes = tape.x_in_dim * 4;
                    gpu.memcpy_dtod_at_auto(
                        &tape.layer_out_bufs[delta_layer_idx].buf,
                        *tape_row * hidden_row_bytes,
                        &s.x.buf,
                        0,
                        hidden_row_bytes,
                    )?;
                }

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                trace_finite_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} LinearAttention residual"),
                    &s.x,
                )?;
                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttn(layer), LayerType::FullAttention) => {
                // Fused rmsnorm + FWHT rotation for wq/wk/wv (all share input).
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu,
                    &layer.wq,
                    &s.x,
                    &layer.attn_norm,
                    &s.tmp,
                    &s.x_rot,
                    config.norm_eps,
                )?;
                // Lever 1 — Fused rmsnorm + PARO per-group rotation for wq.
                let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                    && layer.wq.gpu_dtype == DType::ParoQ4G128
                    && layer.wq.k % 128 == 0
                    && layer.wq.m % 8 == 0
                {
                    fused_rmsnorm_rotate_for_paro(
                        gpu,
                        &layer.wq,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?
                } else {
                    None
                };
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_input_bufs[delta_layer_idx].buf,
                            *tape_row * hidden_row_bytes,
                            &s.x.buf,
                            0,
                            hidden_row_bytes,
                        )?;
                    }
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let x_for_tape = x_rot_paro.or(x_rot).unwrap_or(&s.tmp);
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_x_bufs[delta_layer_idx].buf,
                            *tape_row * hidden_row_bytes,
                            &x_for_tape.buf,
                            0,
                            hidden_row_bytes,
                        )?;
                    }
                }
                // Cross-arch fast path: fused 3-way projection for wq+wk+wv.
                // Works for MQ4 and HF4 — same kernel math as the LA 4-way.
                let dt = layer.wq.gpu_dtype;
                let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                let fused_fa3_mq4 = config.attn_output_gate
                    && fa3_same_dtype
                    && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                let fused_fa3_f16 = config.attn_output_gate && fa3_same_dtype && dt == DType::F16;
                let fused_fa3_lloyd_mq3 =
                    config.attn_output_gate && fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                let fused_fa3_lloyd_mq4 =
                    config.attn_output_gate && fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                let fused_fa3_paro4t = config.attn_output_gate
                    && fa3_same_dtype
                    && dt == DType::ParoQ4G128
                    && x_rot_paro.is_none()
                    && std::env::var("HIPFIRE_PARO_FA3_FUSED")
                        .map(|v| v != "0")
                        .unwrap_or(true);
                // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
                let fused_fa3_hfq6 = config.attn_output_gate
                    && fa3_same_dtype
                    && (dt == DType::MQ6G256 || dt == DType::HFQ6G256)
                    && gpu.arch_caps.gemv_dp4a_enabled();
                if fused_fa3_f16 {
                    gpu.fused_qkvza_f16_xf32(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &layer.wq.buf,
                        &s.tmp,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        &s.o,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        0,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_hfq4g256(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_lloyd_mq3 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_mq3g256_lloyd(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_lloyd_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_mq4g256_lloyd(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_hfq6 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_hfq6g256_dp4a(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_paro4t {
                    gpu.fused_qkvza_paro4g128t(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &layer.wq.buf,
                        &s.tmp,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        &s.o,
                        &s.x_rot,
                        &s.ffn_hidden,
                        &s.ffn_out,
                        &s.o,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        0,
                        layer.wq.k,
                    )?;
                } else {
                    if let Some(xr_first) = x_rot_paro {
                        gpu.gemv_paro4g128t_prerotated(
                            &layer.wq.buf,
                            xr_first,
                            &s.fa_q_full,
                            layer.wq.m,
                            layer.wq.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
                    }
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe split q projection done"),
                    )?;
                    weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe split k projection done"),
                    )?;
                    weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe split v projection done"),
                    )?;
                }
                // RoughQuant residual-reader corrections (FullAttn q/k/v_proj),
                // applied to the raw projection outputs before q_norm/materialize.
                rq_apply_readers(
                    gpu,
                    weights,
                    layer_idx,
                    &layer.attn_norm,
                    &s.x,
                    config.norm_eps,
                    config.dim,
                    &[
                        (RqProj::Wq, &s.fa_q_full),
                        (RqProj::Wk, &s.fa_k),
                        (RqProj::Wv, &s.fa_v),
                    ],
                )?;
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_full_row_bytes = tape.fa_q_full_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_full_bufs[delta_layer_idx].buf,
                            *tape_row * q_full_row_bytes,
                            &s.fa_q_full.buf,
                            0,
                            q_full_row_bytes,
                        )?;
                    }
                }

                qwen35_materialize_fa_q(gpu, config, &s.fa_q_full, &s.fa_q, &s.fa_gate, 1)?;

                gpu.rmsnorm_batched(
                    &s.fa_q,
                    &layer.q_norm,
                    &s.fa_q,
                    config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_norm_bufs[delta_layer_idx].buf,
                            *tape_row * q_row_bytes,
                            &s.fa_q.buf,
                            0,
                            q_row_bytes,
                        )?;
                    }
                }

                let kv_dim = config.n_kv_heads * config.head_dim;
                gpu.rmsnorm_batched(
                    &s.fa_k,
                    &layer.k_norm,
                    &s.fa_k,
                    config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                qkv_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wq,
                    &layer.wk,
                    &layer.wv,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                )?;

                gpu.deinterleave_f32(
                    &s.fa_q_full,
                    &s.fa_q,
                    &s.fa_gate,
                    config.n_heads,
                    config.head_dim,
                )?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                let npu_hnr_ok = if hipfire_runtime::triattn::tap_enabled() {
                    false
                } else {
                    try_npu_headnorm_rope(
                        gpu,
                        layer_idx,
                        &s.fa_q,
                        &s.fa_k,
                        &layer.q_norm,
                        &layer.k_norm,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                        pos,
                    )?
                };
                if !npu_hnr_ok {
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    if hipfire_runtime::triattn::tap_enabled() {
                        triattn_tap(gpu, layer_idx, s, config)?;
                    }
                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        let kv_row_bytes = tape.fa_kv_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_q_bufs[delta_layer_idx].buf,
                            *tape_row * q_row_bytes,
                            &s.fa_q.buf,
                            0,
                            q_row_bytes,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_k_bufs[delta_layer_idx].buf,
                            *tape_row * kv_row_bytes,
                            &s.fa_k.buf,
                            0,
                            kv_row_bytes,
                        )?;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_v_bufs[delta_layer_idx].buf,
                            *tape_row * kv_row_bytes,
                            &s.fa_v.buf,
                            0,
                            kv_row_bytes,
                        )?;
                    }
                }
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht4_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            0,
                        )?;
                        gpu.attention_flash_fwht4(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym4_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.attention_flash_asym4(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht3_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            0,
                        )?;
                        gpu.attention_flash_fwht3(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym3_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.attention_flash_asym3(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht2_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            0,
                        )?;
                        gpu.attention_flash_fwht2(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym2_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.attention_flash_asym2(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_q8 {
                    trace_stage_if_enabled(&format!(
                        "layer {layer_idx} FullAttnMoe q8 kv write begin"
                    ));
                    gpu.kv_cache_write_q8_0(
                        &kv_cache.k_gpu[layer_idx],
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    gpu.kv_cache_write_q8_0(
                        &kv_cache.v_gpu[layer_idx],
                        &s.fa_v,
                        &s.pos_buf,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    // Flash dispatch (Q8 path):
                    //   - capture_mode (hipGraph): always flash — position-independent grid.
                    //   - flash_mode=2 (always): force flash at any ctx.
                    //   - flash_mode=1 (auto, default): flash at ctx >= 2048.
                    //   - flash_mode=0 (never): non-flash until sanity cap (>15K ctx).
                    //   - >15K: always flash (non-flash VRAM blowup).
                    let use_flash = gpu.capture_mode
                        || s.flash_mode == 2
                        || (s.flash_mode == 1 && pos + 1 >= 2048)
                        || pos + 1 > 15000;
                    if use_flash {
                        gpu.attention_flash_q8_0(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    } else {
                        gpu.attention_q8_0_kv(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                        )?;
                    }
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &s.fa_q,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out,
                        &s.pos_buf,
                        pos + 1,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                    )?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let q_row_bytes = tape.fa_q_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_attn_raw_bufs[delta_layer_idx].buf,
                            *tape_row * q_row_bytes,
                            &s.fa_attn_out.buf,
                            0,
                            q_row_bytes,
                        )?;
                    }
                }

                if config.attn_output_gate {
                    let npu_gate_ok = try_npu_attn_gate(
                        gpu,
                        layer_idx,
                        &s.fa_attn_out,
                        &s.fa_gate,
                        config.n_heads,
                        config.head_dim,
                    )?;
                    if !npu_gate_ok {
                        qwen35_apply_fa_gate(gpu, config, &s.fa_attn_out, &s.fa_gate)?;
                    }
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_attn_out_bufs[delta_layer_idx].buf,
                            *tape_row * hidden_row_bytes,
                            &s.fa_attn_out.buf,
                            0,
                            hidden_row_bytes,
                        )?;
                    }
                }
                // Fused wo GEMV + residual add: s.x += layer.wo * s.fa_attn_out
                qwen35_attention_wo_residual(
                    gpu,
                    config,
                    layer_idx,
                    &layer.wo,
                    &s.fa_attn_out,
                    &s.x,
                    &s.o,
                )?;
                // RoughQuant residual-writer correction (FullAttn o_proj rows).
                if let Some(c) = weights.rq_corrections.get(&(layer_idx as u32, RqProj::Wo)) {
                    rq_apply_writer(gpu, c, &s.fa_attn_out, &s.x)?;
                }
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_wo_residual_bufs[delta_layer_idx].buf,
                            *tape_row * hidden_row_bytes,
                            &s.x.buf,
                            0,
                            hidden_row_bytes,
                        )?;
                    }
                }

                // Phase-A block-local recovery capture: pre-MLP residual (x_mid)
                // for full-attention layers (post-attn residual feeding ffn_norm).
                dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "premlp");
                // FFN: fused rmsnorm + rotate for w_gate/w_up.
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu,
                    &layer.w_gate,
                    &s.x,
                    &layer.ffn_norm,
                    &s.tmp,
                    &s.x_rot,
                    config.norm_eps,
                )?;
                // Lever 1 — Fused rmsnorm + PARO per-group rotation for w_gate.
                let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                    && layer.w_gate.gpu_dtype == DType::ParoQ4G128
                    && layer.w_gate.k % 128 == 0
                    && layer.w_gate.m % 8 == 0
                {
                    fused_rmsnorm_rotate_for_paro(
                        gpu,
                        &layer.w_gate,
                        &s.x,
                        &layer.ffn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?
                } else {
                    None
                };
                // Cross-arch fast path: fused gate+up in one launch. Works
                // for both MQ4 (x_rot Some) and HF4 (x_rot None → s.tmp).
                let dt_g = layer.w_gate.gpu_dtype;
                let same_dtype = layer.w_up.gpu_dtype == dt_g;
                let fused_gu_mq4 =
                    same_dtype && (dt_g == DType::MQ4G256 || dt_g == DType::HFQ4G256);
                let fused_gu_f16 = same_dtype && dt_g == DType::F16;
                let fused_gu_lloyd_mq3 = same_dtype && dt_g == DType::MQ3G256Lloyd;
                let fused_gu_lloyd_mq4 = same_dtype && dt_g == DType::MQ4G256Lloyd;
                let fused_gu_paro4t = same_dtype
                    && dt_g == DType::ParoQ4G128
                    && layer.w_gate.m == layer.w_up.m
                    && layer.w_gate.k == layer.w_up.k
                    && x_rot_paro.is_none()
                    && std::env::var("HIPFIRE_PARO_GATE_UP_FUSED")
                        .map(|v| v != "0")
                        .unwrap_or(true);
                // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
                let fused_gu_hfq6 = same_dtype
                    && (dt_g == DType::MQ6G256 || dt_g == DType::HFQ6G256)
                    && gpu.arch_caps.gemv_dp4a_enabled();
                if fused_gu_f16 {
                    gpu.fused_gate_up_f16_xf32(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &s.tmp,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_hfq4g256(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_lloyd_mq3 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_mq3g256_lloyd(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_lloyd_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_mq4g256_lloyd(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_hfq6 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_gate_up_hfq6g256_dp4a(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        eff_x,
                        &s.gate_ffn,
                        &s.up,
                        layer.w_gate.m,
                        layer.w_up.m,
                        layer.w_gate.k,
                    )?;
                } else if fused_gu_paro4t {
                    gpu.fused_gate_up_paro4g128t(
                        &layer.w_gate.buf,
                        &layer.w_up.buf,
                        &s.tmp,
                        &s.gate_ffn,
                        &s.up,
                        &s.x_rot,
                        layer.w_gate.m,
                        layer.w_gate.k,
                    )?;
                } else {
                    if let Some(xr_first) = x_rot_paro {
                        gpu.gemv_paro4g128t_prerotated(
                            &layer.w_gate.buf,
                            xr_first,
                            &s.gate_ffn,
                            layer.w_gate.m,
                            layer.w_gate.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.w_gate, &s.tmp, x_rot, &s.gate_ffn)?;
                    }
                    weight_gemv_prerotated(gpu, &layer.w_up, &s.tmp, x_rot, &s.up)?;
                }
                // RoughQuant residual-reader corrections (FullAttn mlp gate/up).
                rq_apply_readers(
                    gpu,
                    weights,
                    layer_idx,
                    &layer.ffn_norm,
                    &s.x,
                    config.norm_eps,
                    config.dim,
                    &[(RqProj::Wgate, &s.gate_ffn), (RqProj::Wup, &s.up)],
                )?;
                // Fused SwiGLU + w_down residual GEMV:
                //   MQ4: fused_silu_rotate(gate,up) + gemv_residual(w_down, rotated, x)
                //   HF4: silu_mul + weight_gemv_residual (unchanged)
                weight_gemv_swiglu_residual_bf16_probe(
                    gpu,
                    layer_idx,
                    &layer.w_down,
                    &layer.bf16_down_shadow,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;
                // RoughQuant residual-writer correction (FullAttn mlp.down_proj).
                if let Some(c) = weights
                    .rq_corrections
                    .get(&(layer_idx as u32, RqProj::Wdown))
                {
                    let inp = gpu.zeros_owned(&[layer.w_down.k], DType::F32)?;
                    gpu.silu_mul_f32(&s.gate_ffn, &s.up, &inp)?;
                    rq_apply_writer(gpu, c, &inp, &s.x)?;
                    drop(inp); // enqueue this layer's RAII scratch, then drain.
                    gpu.reclaim_pending();
                }
                kv_cache_attention_dispatch(&ctx, gpu, kv_cache, s, config, layer_idx, pos)?;

                gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
                {
                    let wr = layer.wo.dispatch_ref();
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input: GemvInput::Raw(&s.fa_attn_out),
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── FFN ──
                gate_up_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.w_gate,
                    &layer.w_up,
                    &layer.ffn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.gate_ffn,
                    &s.up,
                    config.norm_eps,
                )?;

                hipfire_runtime::weights::weight_gemv_swiglu_residual(
                    gpu,
                    &layer.w_down,
                    &s.gate_ffn,
                    &s.up,
                    &s.ffn_hidden,
                    &s.x,
                )?;
                if let Some((tape, tape_row)) = gdn_tape_capture.as_mut() {
                    if delta_layer_idx < tape.fa_bridge_valid.len()
                        && tape.fa_bridge_valid[delta_layer_idx]
                    {
                        let hidden_row_bytes = tape.x_in_dim * 4;
                        gpu.memcpy_dtod_at_auto(
                            &tape.fa_bridge_layer_out_bufs[delta_layer_idx].buf,
                            *tape_row * hidden_row_bytes,
                            &s.x.buf,
                            0,
                            hidden_row_bytes,
                        )?;
                    }
                }

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                trace_finite_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttention residual"),
                    &s.x,
                )?;
                _kv_layer_idx += 1;
            }

            (LayerWeights::DeltaNetMoe(layer), LayerType::LinearAttention) => {
                // ── DeltaNetMoe QKVZA via pipeline ──
                qkvza_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wqkv,
                    &layer.wz,
                    &layer.w_beta,
                    &layer.w_alpha,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.dn_qkv,
                    &s.dn_z,
                    &s.dn_beta,
                    &s.dn_alpha,
                    config.norm_eps,
                )?;
                let x_rot: Option<&GpuTensor> = Some(&s.x_rot);
                // Lever 1 — Fused rmsnorm + PARO per-group rotation for wqkv.
                let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                    && layer.wqkv.gpu_dtype == DType::ParoQ4G128
                    && layer.wqkv.k % 128 == 0
                    && layer.wqkv.m % 8 == 0
                {
                    fused_rmsnorm_rotate_for_paro(
                        gpu,
                        &layer.wqkv,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?
                } else {
                    None
                };
                let dt = layer.wqkv.gpu_dtype;
                let la4_same_dtype = layer.wz.gpu_dtype == dt
                    && layer.w_beta.gpu_dtype == dt
                    && layer.w_alpha.gpu_dtype == dt;
                let fused_la4_mq4 =
                    la4_same_dtype && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                let fused_la4_lloyd_mq3 = la4_same_dtype && dt == DType::MQ3G256Lloyd;
                let _fused_la4_lloyd_mq4 = la4_same_dtype && dt == DType::MQ4G256Lloyd;
                let fused_la4_paro4t = la4_same_dtype
                    && dt == DType::ParoQ4G128
                    && x_rot_paro.is_none()
                    && std::env::var_os("HIPFIRE_PARO_LA4_FUSED").is_some();
                let fused_la2_paro4t = dt == DType::ParoQ4G128
                    && layer.wz.gpu_dtype == DType::ParoQ4G128
                    && x_rot_paro.is_none()
                    && std::env::var_os("HIPFIRE_PARO_LA2_FUSED").is_some();
                if fused_la4_mq4 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkvza_hfq4g256(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la4_lloyd_mq3 {
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkvza_mq3g256_lloyd(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        eff_x,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la4_paro4t {
                    gpu.fused_qkvza_paro4g128t(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.w_beta.buf,
                        &layer.w_alpha.buf,
                        &s.tmp,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        &s.x_rot,
                        &s.ffn_hidden,
                        &s.ffn_out,
                        &s.o,
                        layer.wqkv.m,
                        layer.wz.m,
                        layer.w_beta.m,
                        layer.w_alpha.m,
                        layer.wqkv.k,
                    )?;
                } else if fused_la2_paro4t {
                    gpu.fused_qkvza_paro4g128t(
                        &layer.wqkv.buf,
                        &layer.wz.buf,
                        &layer.wqkv.buf,
                        &layer.wqkv.buf,
                        &s.tmp,
                        &s.dn_qkv,
                        &s.dn_z,
                        &s.dn_beta,
                        &s.dn_alpha,
                        &s.x_rot,
                        &s.ffn_hidden,
                        &s.ffn_out,
                        &s.o,
                        layer.wqkv.m,
                        layer.wz.m,
                        0,
                        0,
                        layer.wqkv.k,
                    )?;
                    weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                    weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                } else {
                    if let Some(xr_first) = x_rot_paro {
                        gpu.gemv_paro4g128t_prerotated(
                            &layer.wqkv.buf,
                            xr_first,
                            &s.dn_qkv,
                            layer.wqkv.m,
                            layer.wqkv.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wqkv, &s.tmp, x_rot, &s.dn_qkv)?;
                    }
                    weight_gemv_prerotated(gpu, &layer.wz, &s.tmp, x_rot, &s.dn_z)?;
                    weight_gemv_prerotated(gpu, &layer.w_beta, &s.tmp, x_rot, &s.dn_beta)?;
                    weight_gemv_prerotated(gpu, &layer.w_alpha, &s.tmp, x_rot, &s.dn_alpha)?;
                }
                // Find GDN call location by dumping after common operations
                gpu.fused_sigmoid_alpha_gate_conv1d_silu_split_f32(
                    &s.dn_beta,
                    &s.dn_alpha,
                    &layer.dt_bias,
                    &layer.a_log,
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    &s.dn_v,
                    &s.dn_qkv,
                    &layer.conv_weight,
                    &dn_state.conv_states[delta_layer_idx],
                    n_v_heads,
                    k_dim,
                    v_dim,
                )?;
                gpu.fused_qk_l2_norm_scale_f32(
                    &s.dn_q_raw,
                    &s.dn_k_raw,
                    config.linear_num_key_heads,
                    hd,
                    1.0 / (hd as f32).sqrt(),
                    config.norm_eps,
                )?;
                if config.linear_num_key_heads < n_v_heads {
                    let ratio = n_v_heads / config.linear_num_key_heads;
                    gpu.repeat_interleave_qk_f32(
                        &s.dn_q_raw,
                        &s.dn_k_raw,
                        &s.dn_q,
                        &s.dn_k,
                        config.linear_num_key_heads,
                        ratio,
                        hd,
                    )?;
                } else {
                    gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
                    gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
                }

                // DIAG: dump GDN inputs (per-token)
                if layer_idx == 0 {
                    let qk_dim = n_v_heads * config.linear_key_head_dim;
                    dump_hidden_localize(gpu, &s.dn_q, 1, pos, qk_dim, 0, "q_p");
                    dump_hidden_localize(gpu, &s.dn_k, 1, pos, qk_dim, 0, "k_p");
                    dump_hidden_localize(gpu, &s.dn_v, 1, pos, v_dim, 0, "v_p");
                    dump_hidden_localize(gpu, &s.dn_alpha, 1, pos, n_v_heads, 0, "alpha_p");
                    dump_hidden_localize(gpu, &s.dn_beta, 1, pos, n_v_heads, 0, "beta_p");
                }

                match dn_state.quant {
                    StateQuant::FP32 => gpu.gated_delta_net_f32(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q8 => gpu.gated_delta_net_q8(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                    StateQuant::Q4 => gpu.gated_delta_net_q4(
                        &s.dn_q,
                        &s.dn_k,
                        &s.dn_v,
                        &s.dn_alpha,
                        &s.dn_beta,
                        &dn_state.s_matrices[delta_layer_idx],
                        &dn_state.s_scales[delta_layer_idx],
                        &s.dn_attn_out,
                        1,
                        n_v_heads,
                        config.linear_value_head_dim,
                    )?,
                }
                // DIAG: dump GDN attention output (per-token)
                if layer_idx == 0 {
                    dump_hidden_localize(
                        gpu,
                        &s.dn_attn_out,
                        1,
                        pos,
                        n_v_heads * config.linear_value_head_dim,
                        0,
                        "gdn_p",
                    );
                }

                gpu.gated_norm_f32(
                    &s.dn_attn_out,
                    &s.dn_z,
                    &layer.norm_weight,
                    &s.dn_normed,
                    n_v_heads,
                    config.linear_value_head_dim,
                    config.norm_eps,
                )?;
                {
                    let wr = layer.wo.dispatch_ref();
                    execute_steps(
                        gpu,
                        &ctx,
                        &[Step::GemvResidual {
                            w: &wr,
                            input: GemvInput::Raw(&s.dn_normed),
                            residual: &s.x,
                            out: &s.x,
                        }],
                    )
                    .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
                }

                // ── MoE FFN ──
                // Fuse rmsnorm + FWHT-rotate when all MoE weights are MQ4:
                // one `fused_rmsnorm_rotate_mq` kernel writes FWHT(rmsnorm(s.x))
                // directly into `s.moe_x_rot`, replacing the separate
                // `rmsnorm_f32` + internal `rotate_x_mq` pair. When the
                // prerotated flag is set, `moe_ffn_decode_impl` consumes
                // s.x_rot_local only — `x_norm` becomes a dummy on that path.
                if ffn_all_mq4_for_moe(&layer.ffn) {
                    gpu.fused_rmsnorm_rotate_mq(
                        &s.x,
                        &layer.ffn_norm,
                        s.moe_x_rot.as_ref().expect("MoE scratch"),
                        config.dim,
                        config.norm_eps,
                    )?;
                    moe_ffn_decode_with_scratch_prerotated(
                        gpu,
                        weights.pager.as_ref(),
                        &layer.ffn,
                        &s.x,
                        &s.x,
                        config,
                        s,
                        layer_idx,
                    )?;
                } else if ffn_routed_mq2_lloyd_plain_prerotate_for_moe(&layer.ffn) {
                    gpu.fused_rmsnorm_rotate_mq_plain(
                        &s.x,
                        &layer.ffn_norm,
                        s.moe_x_rot.as_ref().expect("MoE scratch"),
                        &s.tmp,
                        config.dim,
                        config.norm_eps,
                    )?;
                    moe_ffn_decode_with_scratch_prerotated(
                        gpu,
                        weights.pager.as_ref(),
                        &layer.ffn,
                        &s.tmp,
                        &s.x,
                        config,
                        s,
                        layer_idx,
                    )?;
                } else {
                    gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                    moe_ffn_decode_with_scratch(
                        gpu,
                        weights.pager.as_ref(),
                        &layer.ffn,
                        &s.tmp,
                        &s.x,
                        config,
                        s,
                        layer_idx,
                    )?;
                }
                // DIAG: dump MoE router logits (per-token)
                if layer_idx == 0 {
                    if let Some(ref rl) = s.moe_router_logits {
                        dump_hidden_localize(gpu, rl, 1, pos, config.num_experts, 0, "router_p");
                    }
                }

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }

                delta_layer_idx += 1;
            }

            (LayerWeights::FullAttnMoe(layer), LayerType::FullAttention) => {
                trace_stage_if_enabled(&format!("layer {layer_idx} FullAttnMoe enter"));
                let x_rot = fused_rmsnorm_rotate_for_mq(
                    gpu,
                    &layer.wq,
                    &s.x,
                    &layer.attn_norm,
                    &s.tmp,
                    &s.x_rot,
                    config.norm_eps,
                )?;
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe attn norm done"),
                )?;
                // Lever 1 — Fused rmsnorm + PARO per-group rotation for wq.
                let x_rot_paro: Option<&GpuTensor> = if x_rot.is_none()
                    && layer.wq.gpu_dtype == DType::ParoQ4G128
                    && layer.wq.k % 128 == 0
                    && layer.wq.m % 8 == 0
                {
                    fused_rmsnorm_rotate_for_paro(
                        gpu,
                        &layer.wq,
                        &s.x,
                        &layer.attn_norm,
                        &s.tmp,
                        &s.x_rot,
                        config.norm_eps,
                    )?
                } else {
                    None
                };
                let dt = layer.wq.gpu_dtype;
                let fa3_same_dtype = layer.wk.gpu_dtype == dt && layer.wv.gpu_dtype == dt;
                let fused_fa3_mq4 = config.attn_output_gate
                    && fa3_same_dtype
                    && (dt == DType::MQ4G256 || dt == DType::HFQ4G256);
                let fused_fa3_lloyd_mq3 =
                    config.attn_output_gate && fa3_same_dtype && dt == DType::MQ3G256Lloyd;
                let fused_fa3_lloyd_mq4 =
                    config.attn_output_gate && fa3_same_dtype && dt == DType::MQ4G256Lloyd;
                let fused_fa3_paro4t = config.attn_output_gate
                    && fa3_same_dtype
                    && dt == DType::ParoQ4G128
                    && x_rot_paro.is_none()
                    && std::env::var("HIPFIRE_PARO_FA3_FUSED")
                        .map(|v| v != "0")
                        .unwrap_or(true);
                // Phase A.1c (gfx906): fused dp4a path for HFQ6/MQ6 weights.
                let fused_fa3_hfq6 = config.attn_output_gate
                    && fa3_same_dtype
                    && (dt == DType::MQ6G256 || dt == DType::HFQ6G256)
                    && gpu.arch_caps.gemv_dp4a_enabled();
                if fused_fa3_mq4 {
                    trace_stage_if_enabled(&format!("layer {layer_idx} FullAttnMoe fused qkv mq4"));
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_hfq4g256(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_lloyd_mq3 {
                    trace_stage_if_enabled(&format!(
                        "layer {layer_idx} FullAttnMoe fused qkv lloyd mq3"
                    ));
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_mq3g256_lloyd(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_lloyd_mq4 {
                    trace_stage_if_enabled(&format!(
                        "layer {layer_idx} FullAttnMoe fused qkv lloyd mq4"
                    ));
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_mq4g256_lloyd(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_hfq6 {
                    trace_stage_if_enabled(&format!(
                        "layer {layer_idx} FullAttnMoe fused qkv hfq6"
                    ));
                    let eff_x = match x_rot {
                        Some(xr) => xr,
                        None => &s.tmp,
                    };
                    gpu.fused_qkv_hfq6g256_dp4a(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        eff_x,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        layer.wq.k,
                    )?;
                } else if fused_fa3_paro4t {
                    trace_stage_if_enabled(&format!(
                        "layer {layer_idx} FullAttnMoe fused qkv paro"
                    ));
                    gpu.fused_qkvza_paro4g128t(
                        &layer.wq.buf,
                        &layer.wk.buf,
                        &layer.wv.buf,
                        &layer.wq.buf,
                        &s.tmp,
                        &s.fa_q_full,
                        &s.fa_k,
                        &s.fa_v,
                        &s.o,
                        &s.x_rot,
                        &s.ffn_hidden,
                        &s.ffn_out,
                        &s.o,
                        layer.wq.m,
                        layer.wk.m,
                        layer.wv.m,
                        0,
                        layer.wq.k,
                    )?;
                } else {
                    trace_stage_if_enabled(&format!("layer {layer_idx} FullAttnMoe split qkv"));
                    if let Some(xr_first) = x_rot_paro {
                        gpu.gemv_paro4g128t_prerotated(
                            &layer.wq.buf,
                            xr_first,
                            &s.fa_q_full,
                            layer.wq.m,
                            layer.wq.k,
                        )?;
                    } else {
                        weight_gemv_prerotated(gpu, &layer.wq, &s.tmp, x_rot, &s.fa_q_full)?;
                    }
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe split q projection done"),
                    )?;
                    weight_gemv_prerotated(gpu, &layer.wk, &s.tmp, x_rot, &s.fa_k)?;
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe split k projection done"),
                    )?;
                    weight_gemv_prerotated(gpu, &layer.wv, &s.tmp, x_rot, &s.fa_v)?;
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe split v projection done"),
                    )?;
                }
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe qkv projection done"),
                )?;

                qwen35_materialize_fa_q(gpu, config, &s.fa_q_full, &s.fa_q, &s.fa_gate, 1)?;
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe q materialized"),
                )?;
                gpu.rmsnorm_batched(
                    &s.fa_q,
                    &layer.q_norm,
                    &s.fa_q,
                    config.n_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                qkv_via_execute_steps(
                    gpu,
                    &ctx,
                    &layer.wq,
                    &layer.wk,
                    &layer.wv,
                    &layer.attn_norm,
                    &s.x,
                    &s.tmp,
                    &s.x_rot,
                    &s.fa_q_full,
                    &s.fa_k,
                    &s.fa_v,
                    config.norm_eps,
                )?;
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe q norm done"),
                )?;

                let kv_dim = config.n_kv_heads * config.head_dim;
                gpu.rmsnorm_batched(
                    &s.fa_k,
                    &layer.k_norm,
                    &s.fa_k,
                    config.n_kv_heads,
                    config.head_dim,
                    config.norm_eps,
                )?;
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe k norm done"),
                )?;
                gpu.deinterleave_f32(
                    &s.fa_q_full,
                    &s.fa_q,
                    &s.fa_gate,
                    config.n_heads,
                    config.head_dim,
                )?;
                let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
                let npu_hnr_ok = if hipfire_runtime::triattn::tap_enabled() {
                    false
                } else {
                    try_npu_headnorm_rope(
                        gpu,
                        layer_idx,
                        &s.fa_q,
                        &s.fa_k,
                        &layer.q_norm,
                        &layer.k_norm,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                        pos,
                    )?
                };
                if !npu_hnr_ok {
                    gpu.rmsnorm_batched(
                        &s.fa_q,
                        &layer.q_norm,
                        &s.fa_q,
                        config.n_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    gpu.rmsnorm_batched(
                        &s.fa_k,
                        &layer.k_norm,
                        &s.fa_k,
                        config.n_kv_heads,
                        config.head_dim,
                        config.norm_eps,
                    )?;
                    if hipfire_runtime::triattn::tap_enabled() {
                        triattn_tap(gpu, layer_idx, s, config)?;
                    }
                    if kv_cache.compact_offset > 0 {
                        let abs = (pos + kv_cache.compact_offset) as i32;
                        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
                    }
                    gpu.rope_partial_interleaved_f32(
                        &s.fa_q,
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        n_rot,
                        config.rope_theta,
                    )?;
                }
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe rope done"),
                )?;
                if kv_cache.compact_offset > 0 {
                    let phys = pos as i32;
                    gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
                }

                if kv_cache.quant_asym4 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht4_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            0,
                        )?;
                        gpu.attention_flash_fwht4(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym4_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.attention_flash_asym4(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_asym3 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht3_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            0,
                        )?;
                        gpu.attention_flash_fwht3(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym3_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.attention_flash_asym3(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_asym2 {
                    let ct = kv_cache.givens_cos.as_ref().unwrap();
                    let st = kv_cache.givens_sin.as_ref().unwrap();
                    if kv_cache.quant_fwht {
                        gpu.kv_cache_write_fwht2_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                            0,
                        )?;
                        gpu.attention_flash_fwht2(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                            0,
                        )?;
                    } else {
                        gpu.kv_cache_write_asym2_fused(
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_k,
                            &s.fa_v,
                            &s.pos_buf,
                            ct,
                            st,
                            config.n_kv_heads,
                            config.head_dim,
                        )?;
                        gpu.attention_flash_asym2(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            ct,
                            st,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    }
                } else if kv_cache.quant_q8 {
                    gpu.kv_cache_write_q8_0(
                        &kv_cache.k_gpu[layer_idx],
                        &s.fa_k,
                        &s.pos_buf,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    gpu.kv_cache_write_q8_0(
                        &kv_cache.v_gpu[layer_idx],
                        &s.fa_v,
                        &s.pos_buf,
                        config.n_kv_heads,
                        config.head_dim,
                    )?;
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe q8 kv write done"),
                    )?;
                    let use_flash = gpu.capture_mode
                        || s.flash_mode == 2
                        || (s.flash_mode == 1 && pos + 1 >= 2048)
                        || pos + 1 > 15000;
                    if use_flash {
                        trace_stage_if_enabled(&format!(
                            "layer {layer_idx} FullAttnMoe q8 flash attention begin"
                        ));
                        gpu.attention_flash_q8_0(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                            &s.flash_partials,
                        )?;
                    } else {
                        trace_stage_if_enabled(&format!(
                            "layer {layer_idx} FullAttnMoe q8 attention begin"
                        ));
                        gpu.attention_q8_0_kv(
                            &s.fa_q,
                            &kv_cache.k_gpu[layer_idx],
                            &kv_cache.v_gpu[layer_idx],
                            &s.fa_attn_out,
                            &s.pos_buf,
                            pos + 1,
                            config.n_heads,
                            config.n_kv_heads,
                            config.head_dim,
                            kv_cache.physical_cap,
                        )?;
                    }
                    trace_stage_sync_if_enabled(
                        gpu,
                        &format!("layer {layer_idx} FullAttnMoe q8 attention done"),
                    )?;
                } else {
                    gpu.kv_cache_write(&kv_cache.k_gpu[layer_idx], &s.fa_k, &s.pos_buf, kv_dim)?;
                    gpu.kv_cache_write(&kv_cache.v_gpu[layer_idx], &s.fa_v, &s.pos_buf, kv_dim)?;
                    gpu.attention_f32(
                        &s.fa_q,
                        &kv_cache.k_gpu[layer_idx],
                        &kv_cache.v_gpu[layer_idx],
                        &s.fa_attn_out,
                        &s.pos_buf,
                        pos + 1,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        kv_cache.physical_cap,
                    )?;
                }

                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe attn gate begin"),
                )?;
                qwen35_apply_fa_gate(gpu, config, &s.fa_attn_out, &s.fa_gate)?;
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe attn gate done"),
                )?;
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe wo residual begin"),
                )?;
                qwen35_attention_wo_residual(
                    gpu,
                    config,
                    layer_idx,
                    &layer.wo,
                    &s.fa_attn_out,
                    &s.x,
                    &s.o,
                )?;
                trace_stage_sync_if_enabled(
                    gpu,
                    &format!("layer {layer_idx} FullAttnMoe wo residual done"),
                )?;

                // ── MoE FFN ──
                // Fuse rmsnorm + FWHT-rotate when all MoE weights are MQ4:
                // one `fused_rmsnorm_rotate_mq` kernel writes FWHT(rmsnorm(s.x))
                // directly into `s.moe_x_rot`, replacing the separate
                // `rmsnorm_f32` + internal `rotate_x_mq` pair. When the
                // prerotated flag is set, `moe_ffn_decode_impl` consumes
                // s.x_rot_local only — `x_norm` becomes a dummy on that path.
                if ffn_all_mq4_for_moe(&layer.ffn) {
                    gpu.fused_rmsnorm_rotate_mq(
                        &s.x,
                        &layer.ffn_norm,
                        s.moe_x_rot.as_ref().expect("MoE scratch"),
                        config.dim,
                        config.norm_eps,
                    )?;
                    moe_ffn_decode_with_scratch_prerotated(
                        gpu,
                        weights.pager.as_ref(),
                        &layer.ffn,
                        &s.x,
                        &s.x,
                        config,
                        s,
                        layer_idx,
                    )?;
                } else if ffn_routed_mq2_lloyd_plain_prerotate_for_moe(&layer.ffn) {
                    gpu.fused_rmsnorm_rotate_mq_plain(
                        &s.x,
                        &layer.ffn_norm,
                        s.moe_x_rot.as_ref().expect("MoE scratch"),
                        &s.tmp,
                        config.dim,
                        config.norm_eps,
                    )?;
                    moe_ffn_decode_with_scratch_prerotated(
                        gpu,
                        weights.pager.as_ref(),
                        &layer.ffn,
                        &s.tmp,
                        &s.x,
                        config,
                        s,
                        layer_idx,
                    )?;
                } else {
                    gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
                    moe_ffn_decode_with_scratch(
                        gpu,
                        weights.pager.as_ref(),
                        &layer.ffn,
                        &s.tmp,
                        &s.x,
                        config,
                        s,
                        layer_idx,
                    )?;
                }

                if let Some(ref rb) = hidden_rb {
                    if let Some(slot) = rb.extract_slot(layer_idx) {
                        rb.write_at_head(gpu, slot, &s.x)?;
                    }
                }
                _kv_layer_idx += 1;
            }

            // Mismatched layer weight / type combinations are unreachable
            // (the loader guarantees alignment).
            _ => unreachable!(),
        }
        dump_hidden_localize(gpu, &s.x, 1, pos, config.dim, layer_idx, "pertoken");
    }

    // Final norm into scratch.tmp; optionally emit logits into scratch.logits.
    gpu.rmsnorm_f32(&s.x, &weights.output_norm, &s.tmp, config.norm_eps)?;
    if needs_last_token_logits {
        if weights.output.gpu_dtype == DType::Oq4G256 {
            // Opus W4A4 lm_head: the dispatch pipeline has no Oq4 GEMV entry
            // (it needs runtime int4 activation quant), so route through the
            // dedicated weight_gemv Oq4 arm (rotate → quantize_act_oq4 → grouped
            // iu4 GEMM). Same treatment the layer projections get.
            weight_gemv(gpu, &weights.output, &s.tmp, &s.logits)?;
        } else {
            let ctx = DispatchCtx::new(gpu);
            let wr = weights.output.dispatch_ref();
            let step = Step::Gemv {
                w: &wr,
                input: GemvInput::Raw(&s.tmp),
                out: &s.logits,
            };
            execute_steps(gpu, &ctx, &[step])
                .map_err(|e| hip_bridge::HipError::new(0, &e.to_string()))?;
        }
    }

    Ok(())
}
