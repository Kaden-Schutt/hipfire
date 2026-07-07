// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5 weight loading: HFQ / safetensors / ParoQuant paths, GPU slab
//! packing, RQ residual corrections, NPU (xdna1) rope/attn-gate helpers, and
//! calibration-artifact capture. Off the decode/prefill hot path.

use super::*;

// ─── Weight loading ─────────────────────────────────────────────────────

fn qwen35_tensor_name_candidates(name: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(4);
    let mut push = |s: String| {
        if !out.iter().any(|x| x == &s) {
            out.push(s);
        }
    };

    if name == "lm_head.weight" {
        push(name.to_string());
        push("model.language_model.lm_head.weight".to_string());
        push("model.lm_head.weight".to_string());
        return out;
    }

    if name.starts_with("model.") {
        push(name.to_string());
    } else {
        push(format!("model.language_model.{name}"));
        push(format!("model.{name}"));
        push(name.to_string());
    }
    out
}

fn qwen35_tensor_data_vec<'a>(
    hfq: &'a HfqFile,
    name: &str,
) -> Option<(&'a HfqTensorInfo, Vec<u8>)> {
    for candidate in qwen35_tensor_name_candidates(name) {
        if let Some(found) = hfq.tensor_data_vec(&candidate) {
            return Some(found);
        }
    }
    None
}

fn qwen35_tensor_data<'a>(hfq: &'a HfqFile, name: &str) -> Option<(&'a HfqTensorInfo, &'a [u8])> {
    for candidate in qwen35_tensor_name_candidates(name) {
        if let Some(found) = hfq.tensor_data(&candidate) {
            return Some(found);
        }
    }
    None
}

fn load_bf16_down_shadow_for(
    hfq: &HfqFile,
    name: &str,
    layer_idx: usize,
    m: usize,
    k: usize,
) -> HipResult<Option<Bf16DownShadow>> {
    if !ffn_bf16::enabled() || !ffn_bf16::layer_selected(layer_idx) {
        return Ok(None);
    }
    // Xdna1 mode only needs the hidden_size dimension, not the actual F32/BF16 weight
    // data — the down GEMV runs on GPU with the original (quantized) tensor. Skip the
    // shadow decode for xdna1 so MQ4 and other quantized models load without error.
    if ffn_bf16::config().mode == FfnBf16Mode::Xdna1 {
        return Ok(None);
    }
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));
    ffn_bf16::decode_w_down_shadow(&data, info.quant_type, m, k)
        .map(Some)
        .ok_or_else(|| {
            HipError::new(
                0,
                &format!(
                    "HIPFIRE_QWEN35_FFN_BF16 requires F32-oracle HFQ w_down (qt=2 F32 or qt=16 BF16) for layer {layer_idx}, tensor {name}; got qt={} shape={:?} bytes={}",
                    info.quant_type,
                    info.shape,
                    data.len()
                ),
            )
        })
}

fn validate_ffn_bf16_hfq_load(config: &Qwen35Config) -> HipResult<()> {
    if !ffn_bf16::enabled() {
        return Ok(());
    }
    if let ffn_bf16::LayerSelect::One(layer_idx) = ffn_bf16::config().layer {
        if layer_idx >= config.n_layers {
            return Err(HipError::new(
                0,
                &format!(
                    "HIPFIRE_QWEN35_FFN_BF16_LAYER={layer_idx} is out of range for {} layers",
                    config.n_layers
                ),
            ));
        }
    }
    if config.num_experts > 0 {
        return Err(HipError::new(
            0,
            "HIPFIRE_QWEN35_FFN_BF16 is dense-FFN only; MoE/A3B layers are out of scope for this probe",
        ));
    }
    Ok(())
}

fn reject_ffn_bf16_non_hfq_load(source: &str) -> HipResult<()> {
    if ffn_bf16::enabled() {
        return Err(HipError::new(
            0,
            &format!(
                "HIPFIRE_QWEN35_FFN_BF16 requires F32-oracle HFQ load path; {source} is unsupported"
            ),
        ));
    }
    Ok(())
}

fn ffn_bf16_selected_shadow(
    layer_idx: usize,
    shadow: &Option<Bf16DownShadow>,
) -> HipResult<Option<&Bf16DownShadow>> {
    if !ffn_bf16::enabled() || !ffn_bf16::layer_selected(layer_idx) {
        return Ok(None);
    }
    shadow.as_ref().map(Some).ok_or_else(|| {
        HipError::new(
            0,
            &format!(
                "HIPFIRE_QWEN35_FFN_BF16 selected dense layer {layer_idx}, but no BF16 w_down shadow was loaded; requires F32-oracle HFQ"
            ),
        )
    })
}

fn qwen35_rocm_device_identity(gpu: &Gpu) -> hipfire_rocm::RocmDeviceIdentity {
    hipfire_rocm::rocm_device_identity(gpu.device_id, gpu.arch.clone(), gpu.integrated)
}

pub(crate) fn weight_gemv_swiglu_residual_bf16_probe(
    gpu: &mut Gpu,
    layer_idx: usize,
    w_down: &WeightTensor,
    shadow: &Option<Bf16DownShadow>,
    gate: &GpuTensor,
    up: &GpuTensor,
    ffn_hidden: &GpuTensor,
    x: &GpuTensor,
) -> HipResult<()> {
    // Xdna1 mode does not require a BF16 shadow — only w_down.k (hidden_size)
    // is needed to size the xclbin handle, and the down GEMV uses the original
    // (quantized) w_down on GPU. Skip the shadow guard for this mode so MQ4
    // and other quantized models can use the NPU SwiGLU activation path.
    if ffn_bf16::enabled()
        && ffn_bf16::layer_selected(layer_idx)
        && ffn_bf16::config().mode == FfnBf16Mode::Xdna1
    {
        let invocation = ffn_bf16::xdna1_dense_ffn_module_invocation_from_shape(
            layer_idx,
            w_down.m,
            w_down.k,
            &ffn_bf16::config().xdna1_artifacts,
        );
        return weight_gemv_swiglu_residual_xdna1(
            gpu,
            layer_idx,
            w_down.k,
            gate,
            up,
            ffn_hidden,
            w_down,
            x,
            &invocation,
        );
    }

    let Some(shadow) = ffn_bf16_selected_shadow(layer_idx, shadow)? else {
        let invocation = ffn_bf16::dense_ffn_module_invocation_from_shape(
            layer_idx,
            w_down.m,
            w_down.k,
            ffn_bf16::DenseFfnBackendPreference::GpuProduction,
            false,
        );
        let result = weight_gemv_swiglu_residual(gpu, w_down, gate, up, ffn_hidden, x);
        if result.is_ok() && ffn_bf16::config().trace {
            let output = ffn_bf16::dense_ffn_module_output(&invocation, None);
            let evidence_json = ffn_bf16::dense_ffn_module_output_json(&output);
            let rocm_output = hipfire_rocm::rocm_dense_ffn_module_output(
                &invocation,
                qwen35_rocm_device_identity(gpu),
                "weight_gemv_swiglu_residual",
                None,
            );
            let rocm_evidence_json = hipfire_rocm::rocm_module_output_json(&rocm_output);
            eprintln!(
                "[qwen35 ffn module] module={} preferred_backend={} selected_backend={} oracle_backend={} fallback_reason={} mutates_residual={} evidence_json={} rocm_evidence_json={}",
                output.evidence.module_id,
                invocation.contract.preferred_backend.as_str(),
                output.evidence.selected_backend.as_str(),
                output.evidence.oracle_backend.as_str(),
                output.evidence.fallback_reason.unwrap_or("none"),
                output.mutates_residual,
                evidence_json,
                rocm_evidence_json,
            );
        }
        return result;
    };

    match ffn_bf16::config().mode {
        FfnBf16Mode::Off => weight_gemv_swiglu_residual(gpu, w_down, gate, up, ffn_hidden, x),
        FfnBf16Mode::Compare | FfnBf16Mode::Cpu | FfnBf16Mode::Xdna1 => {
            let mode = ffn_bf16::config().mode;
            let preferred_backend = ffn_bf16::dense_ffn_backend_preference_for_mode(mode)
                .expect("enabled BF16 mode has a backend preference");
            let invocation = if mode == FfnBf16Mode::Xdna1 {
                ffn_bf16::xdna1_dense_ffn_module_invocation_from_shape(
                    layer_idx,
                    shadow.m,
                    shadow.k,
                    &ffn_bf16::config().xdna1_artifacts,
                )
            } else {
                ffn_bf16::dense_ffn_module_invocation(layer_idx, shadow, preferred_backend, false)
            };
            if mode == FfnBf16Mode::Xdna1 {
                return weight_gemv_swiglu_residual_xdna1(
                    gpu,
                    layer_idx,
                    shadow.k,
                    gate,
                    up,
                    ffn_hidden,
                    w_down,
                    x,
                    &invocation,
                );
            }
            let t0 = std::time::Instant::now();
            let residual_pre = gpu.download_f32(x)?;
            let gate_cpu = gpu.download_f32(gate)?;
            let up_cpu = gpu.download_f32(up)?;
            let download_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let cpu_t0 = std::time::Instant::now();
            let cpu_out = ffn_bf16::swiglu_down_bf16_cpu(&gate_cpu, &up_cpu, &residual_pre, shadow);
            let cpu_ms = cpu_t0.elapsed().as_secs_f64() * 1000.0;

            match ffn_bf16::config().mode {
                FfnBf16Mode::Compare => {
                    weight_gemv_swiglu_residual(gpu, w_down, gate, up, ffn_hidden, x)?;
                    let gpu_out = gpu.download_f32(x)?;
                    let stats = ffn_bf16::diff_stats(&gpu_out, &cpu_out);
                    let output = ffn_bf16::dense_ffn_module_output(&invocation, Some(stats));
                    let evidence_json = ffn_bf16::dense_ffn_module_output_json(&output);
                    let rocm_output = hipfire_rocm::rocm_dense_ffn_module_output(
                        &invocation,
                        qwen35_rocm_device_identity(gpu),
                        "weight_gemv_swiglu_residual",
                        Some(stats),
                    );
                    let rocm_evidence_json = hipfire_rocm::rocm_module_output_json(&rocm_output);
                    eprintln!(
                        "[qwen35 ffn bf16] module={} preferred_backend={} selected_backend={} oracle_backend={} fallback_reason={} n={} max_abs={:.6e} mean_abs={:.6e} rms={:.6e} nan={} inf={} evidence_json={} rocm_evidence_json={}",
                        output.evidence.module_id,
                        invocation.contract.preferred_backend.as_str(),
                        output.evidence.selected_backend.as_str(),
                        output.evidence.oracle_backend.as_str(),
                        output.evidence.fallback_reason.unwrap_or("none"),
                        stats.n,
                        stats.max_abs,
                        stats.mean_abs,
                        stats.rms,
                        stats.n_nan,
                        stats.n_inf,
                        evidence_json,
                        rocm_evidence_json,
                    );
                    if ffn_bf16::config().trace {
                        eprintln!(
                            "[qwen35 ffn bf16] layer={layer_idx} timings download_ms={download_ms:.3} cpu_ms={cpu_ms:.3}"
                        );
                    }
                    Ok(())
                }
                FfnBf16Mode::Cpu => {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(cpu_out.as_ptr().cast::<u8>(), cpu_out.len() * 4)
                    };
                    gpu.hip.memcpy_htod(&x.buf, bytes)?;
                    let output = ffn_bf16::dense_ffn_module_output(&invocation, None);
                    if ffn_bf16::config().trace {
                        let evidence_json = ffn_bf16::dense_ffn_module_output_json(&output);
                        eprintln!(
                            "[qwen35 ffn bf16] module={} preferred_backend={} selected_backend={} oracle_backend={} fallback_reason={} download_ms={download_ms:.3} cpu_ms={cpu_ms:.3} evidence_json={}",
                            output.evidence.module_id,
                            invocation.contract.preferred_backend.as_str(),
                            output.evidence.selected_backend.as_str(),
                            output.evidence.oracle_backend.as_str(),
                            output.evidence.fallback_reason.unwrap_or("none"),
                            evidence_json,
                        );
                    }
                    Ok(())
                }
                FfnBf16Mode::Off => unreachable!(),
                FfnBf16Mode::Xdna1 => unreachable!(),
            }
        }
    }
}

/// NPU SwiGLU dispatch for `FfnBf16Mode::Xdna1`.
///
/// Hybrid path: NPU handles the elementwise SwiGLU activation
/// (`silu(gate) * up → ffn_hidden`), then falls back to the GPU for the
/// w_down matmul (`x += w_down @ ffn_hidden`).  When the NPU paths are not
/// configured or the handle can't be created the function falls back to the
/// full GPU path so callers always get a valid result.
#[allow(clippy::too_many_arguments)]
fn weight_gemv_swiglu_residual_xdna1(
    gpu: &mut Gpu,
    layer_idx: usize,
    hidden_size: usize,
    gate: &GpuTensor,
    up: &GpuTensor,
    ffn_hidden: &GpuTensor,
    w_down: &WeightTensor,
    x: &GpuTensor,
    invocation: &ffn_bf16::DenseFfnModuleInvocation,
) -> HipResult<()> {
    let cfg = ffn_bf16::config();
    let artifacts = &cfg.xdna1_artifacts;
    let (xclbin, instr) = match (artifacts.xclbin.as_deref(), artifacts.instr.as_deref()) {
        (Some(xc), Some(ins)) => (xc, ins),
        _ => {
            if cfg.trace {
                eprintln!(
                    "[qwen35 xdna1] layer={layer_idx} HIPFIRE_QWEN35_XDNA1_XCLBIN / \
                     HIPFIRE_QWEN35_XDNA1_INSTR not set — falling back to GPU"
                );
            }
            return weight_gemv_swiglu_residual(gpu, w_down, gate, up, ffn_hidden, x);
        }
    };
    // All layers with the same hidden_size share one XRT handle — XRT contexts
    // are limited, and there is no per-layer state in the SwiGLU kernel.
    let handle = xdna1_ffi::swiglu_handle_for(hidden_size, hidden_size, xclbin, instr);
    let handle = match handle {
        Some(h) => h,
        None => {
            if cfg.trace {
                eprintln!(
                    "[qwen35 xdna1] layer={layer_idx} handle unavailable — falling back to GPU"
                );
            }
            return weight_gemv_swiglu_residual(gpu, w_down, gate, up, ffn_hidden, x);
        }
    };

    let t0 = std::time::Instant::now();
    let gate_f32 = gpu.download_f32(gate)?;
    let up_f32 = gpu.download_f32(up)?;
    let download_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // f32 → bf16 conversion for NPU inputs
    let gate_bf16: Vec<u16> = gate_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let up_bf16: Vec<u16> = up_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let mut out_bf16: Vec<u16> = vec![0u16; hidden_size];

    let npu_t0 = std::time::Instant::now();
    let ok = unsafe { xdna1_ffi::swiglu_run(handle, &gate_bf16, &up_bf16, &mut out_bf16) };
    let npu_ms = npu_t0.elapsed().as_secs_f64() * 1000.0;

    if !ok {
        if cfg.trace {
            eprintln!("[qwen35 xdna1] layer={layer_idx} run_handle failed — falling back to GPU");
        }
        return weight_gemv_swiglu_residual(gpu, w_down, gate, up, ffn_hidden, x);
    }

    // bf16 → f32 then upload to GPU ffn_hidden for the w_down matmul
    let out_f32: Vec<f32> = out_bf16
        .iter()
        .map(|&b| ffn_bf16::bf16_bits_to_f32(b))
        .collect();
    let upload_t0 = std::time::Instant::now();
    let bytes =
        unsafe { std::slice::from_raw_parts(out_f32.as_ptr().cast::<u8>(), out_f32.len() * 4) };
    gpu.hip.memcpy_htod(&ffn_hidden.buf, bytes)?;

    // Down matmul on GPU: x += w_down @ ffn_hidden
    weight_gemv_residual(gpu, w_down, ffn_hidden, x)?;
    let upload_ms = upload_t0.elapsed().as_secs_f64() * 1000.0;

    if cfg.trace {
        let output = ffn_bf16::dense_ffn_module_output(invocation, None);
        let evidence_json = ffn_bf16::dense_ffn_module_output_json(&output);
        eprintln!(
            "[qwen35 xdna1] layer={layer_idx} selected_backend={} \
             download_ms={download_ms:.3} npu_ms={npu_ms:.3} upload_ms={upload_ms:.3} \
             evidence_json={}",
            output.evidence.selected_backend.as_str(),
            evidence_json,
        );
    }
    Ok(())
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ─── NPU rope / attn-gate helpers ────────────────────────────────────────────

/// Return `(xclbin_path, instr_path)` for a named kernel+shape if both files
/// exist under `$HIPFIRE_NPU_DIR` (default: `target/npu/`).
///
/// The naming convention is `qwen35-{kernel}-{shape}.xclbin` and
/// `qwen35-{kernel}-{shape}-instr.bin`.
fn npu_xclbin_for(kernel: &str, shape: &str) -> Option<(String, String)> {
    let dir = std::env::var("HIPFIRE_NPU_DIR").unwrap_or_else(|_| "target/npu".to_string());
    let xclbin = format!("{dir}/qwen35-{kernel}-{shape}.xclbin");
    let instr = format!("{dir}/qwen35-{kernel}-{shape}-instr.bin");
    if std::path::Path::new(&xclbin).exists() && std::path::Path::new(&instr).exists() {
        Some((xclbin, instr))
    } else {
        None
    }
}

/// Compute BF16 cos/sin buffer for one token position in half-split layout:
/// `[cos(pos*θ_0), ..., cos(pos*θ_{h-1}), sin(pos*θ_0), ..., sin(pos*θ_{h-1})]`
/// where `h = n_rot/2` and `θ_d = rope_theta^(-2d/n_rot)`.
///
/// Total length = n_rot BF16 values.
fn rope_cs_halfsplit_bf16(pos: usize, n_rot: usize, rope_theta: f32) -> Vec<u16> {
    let half = n_rot / 2;
    let pos_f = pos as f32;
    let mut cs = Vec::with_capacity(n_rot);
    for d in 0..half {
        let theta = (rope_theta as f64).powf(-2.0 * d as f64 / n_rot as f64) as f32;
        cs.push(ffn_bf16::f32_to_bf16_bits_rne((pos_f * theta).cos()));
    }
    for d in 0..half {
        let theta = (rope_theta as f64).powf(-2.0 * d as f64 / n_rot as f64) as f32;
        cs.push(ffn_bf16::f32_to_bf16_bits_rne((pos_f * theta).sin()));
    }
    cs
}

/// Attempt NPU rope_q + rope_k dispatch.  Returns `true` on success; `false`
/// means the caller should use the GPU rope fallback.
///
/// The NPU rope kernel uses the half-split layout (pairs at `d` and `d+n_rot/2`)
/// which matches the xclbin produced by `build_qwen35_rope.py`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn try_npu_rope(
    gpu: &mut Gpu,
    layer_idx: usize,
    fa_q: &GpuTensor,
    fa_k: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    rope_theta: f32,
    pos: usize,
) -> HipResult<bool> {
    let q_shape = format!("{n_heads}h{head_dim}d");
    let k_shape = format!("{n_kv_heads}h{head_dim}d");
    let q_paths = match npu_xclbin_for("rope-q", &q_shape) {
        Some(p) => p,
        None => return Ok(false),
    };
    let k_paths = match npu_xclbin_for("rope-k", &k_shape) {
        Some(p) => p,
        None => return Ok(false),
    };
    let q_n = n_heads * head_dim;
    let k_n = n_kv_heads * head_dim;
    let hq = xdna1_ffi::rope_q_handle_for(layer_idx, q_n, &q_paths.0, &q_paths.1);
    let hk = xdna1_ffi::rope_k_handle_for(layer_idx, k_n, &k_paths.0, &k_paths.1);
    let (hq, hk) = match (hq, hk) {
        (Some(q), Some(k)) => (q, k),
        _ => return Ok(false),
    };
    let cs = rope_cs_halfsplit_bf16(pos, n_rot, rope_theta);
    let q_f32 = gpu.download_f32(fa_q)?;
    let k_f32 = gpu.download_f32(fa_k)?;
    let q_bf16: Vec<u16> = q_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let k_bf16: Vec<u16> = k_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let mut q_out = vec![0u16; q_n];
    let mut k_out = vec![0u16; k_n];
    let q_ok = unsafe { xdna1_ffi::rope_q_run(hq, &q_bf16, &cs, &mut q_out) };
    let k_ok = unsafe { xdna1_ffi::rope_k_run(hk, &k_bf16, &cs, &mut k_out) };
    if !q_ok || !k_ok {
        return Ok(false);
    }
    let q_f32_out: Vec<f32> = q_out.iter().map(|&b| bf16_to_f32(b)).collect();
    let k_f32_out: Vec<f32> = k_out.iter().map(|&b| bf16_to_f32(b)).collect();
    let q_bytes =
        unsafe { std::slice::from_raw_parts(q_f32_out.as_ptr().cast::<u8>(), q_f32_out.len() * 4) };
    let k_bytes =
        unsafe { std::slice::from_raw_parts(k_f32_out.as_ptr().cast::<u8>(), k_f32_out.len() * 4) };
    gpu.hip.memcpy_htod(&fa_q.buf, q_bytes)?;
    gpu.hip.memcpy_htod(&fa_k.buf, k_bytes)?;
    eprintln!("[xdna1] layer={layer_idx} npu rope q+k ok");
    Ok(true)
}

/// Attempt NPU fused headnorm+rope dispatch for Q and K.
/// Returns `true` on success; `false` means caller should use GPU headnorm + rope fallback.
///
/// Replaces separate rmsnorm_batched(Q) + rmsnorm_batched(K) + rope calls in one pass.
/// Downloads Q/K and norm weights from GPU (F32), converts to BF16, dispatches the fused
/// NPU kernel, and uploads the normalized+rotated F32 results back to Q/K buffers.
///
/// Callers must skip this function when `triattn::tap_enabled()` because triattn tap
/// requires access to Q/K between headnorm and rope — the fused kernel can't interleave.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_npu_headnorm_rope(
    gpu: &mut Gpu,
    layer_idx: usize,
    fa_q: &GpuTensor,
    fa_k: &GpuTensor,
    q_norm: &GpuTensor,
    k_norm: &GpuTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    rope_theta: f32,
    pos: usize,
) -> HipResult<bool> {
    let q_shape = format!("{n_heads}h{head_dim}d");
    let k_shape = format!("{n_kv_heads}h{head_dim}d");
    let q_paths = match npu_xclbin_for("headnorm-rope-q", &q_shape) {
        Some(p) => p,
        None => return Ok(false),
    };
    let k_paths = match npu_xclbin_for("headnorm-rope-k", &k_shape) {
        Some(p) => p,
        None => return Ok(false),
    };
    let q_n = n_heads * head_dim;
    let k_n = n_kv_heads * head_dim;
    let hq = xdna1_ffi::headnorm_rope_q_handle_for(layer_idx, q_n, &q_paths.0, &q_paths.1);
    let hk = xdna1_ffi::headnorm_rope_k_handle_for(layer_idx, k_n, &k_paths.0, &k_paths.1);
    let (hq, hk) = match (hq, hk) {
        (Some(q), Some(k)) => (q, k),
        _ => return Ok(false),
    };
    let cs = rope_cs_halfsplit_bf16(pos, n_rot, rope_theta);
    let q_f32 = gpu.download_f32(fa_q)?;
    let k_f32 = gpu.download_f32(fa_k)?;
    let qw_f32 = gpu.download_f32(q_norm)?;
    let kw_f32 = gpu.download_f32(k_norm)?;
    let q_bf16: Vec<u16> = q_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let k_bf16: Vec<u16> = k_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let qw_bf16: Vec<u16> = qw_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let kw_bf16: Vec<u16> = kw_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let mut q_out = vec![0u16; q_n];
    let mut k_out = vec![0u16; k_n];
    let q_ok = unsafe { xdna1_ffi::headnorm_rope_q_run(hq, &q_bf16, &qw_bf16, &cs, &mut q_out) };
    let k_ok = unsafe { xdna1_ffi::headnorm_rope_k_run(hk, &k_bf16, &kw_bf16, &cs, &mut k_out) };
    if !q_ok || !k_ok {
        return Ok(false);
    }
    let q_f32_out: Vec<f32> = q_out.iter().map(|&b| bf16_to_f32(b)).collect();
    let k_f32_out: Vec<f32> = k_out.iter().map(|&b| bf16_to_f32(b)).collect();
    let q_bytes =
        unsafe { std::slice::from_raw_parts(q_f32_out.as_ptr().cast::<u8>(), q_f32_out.len() * 4) };
    let k_bytes =
        unsafe { std::slice::from_raw_parts(k_f32_out.as_ptr().cast::<u8>(), k_f32_out.len() * 4) };
    gpu.hip.memcpy_htod(&fa_q.buf, q_bytes)?;
    gpu.hip.memcpy_htod(&fa_k.buf, k_bytes)?;
    eprintln!("[xdna1] layer={layer_idx} npu headnorm_rope q+k ok");
    Ok(true)
}

/// Attempt NPU attn_gate dispatch (`sigmoid(gate) * attn_out → attn_out`).
/// Returns `true` on success; `false` means the caller should use GPU fallback.
pub(crate) fn try_npu_attn_gate(
    gpu: &mut Gpu,
    layer_idx: usize,
    attn_out: &GpuTensor,
    gate: &GpuTensor,
    n_heads: usize,
    head_dim: usize,
) -> HipResult<bool> {
    let shape = format!("{n_heads}h{head_dim}d");
    let paths = match npu_xclbin_for("attn-gate", &shape) {
        Some(p) => p,
        None => return Ok(false),
    };
    let q_dim = n_heads * head_dim;
    let h = xdna1_ffi::attn_gate_handle_for(layer_idx, q_dim, &paths.0, &paths.1);
    let h = match h {
        Some(h) => h,
        None => return Ok(false),
    };
    let out_f32 = gpu.download_f32(attn_out)?;
    let gate_f32 = gpu.download_f32(gate)?;
    let out_bf16: Vec<u16> = out_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let gate_bf16: Vec<u16> = gate_f32
        .iter()
        .map(|&v| ffn_bf16::f32_to_bf16_bits_rne(v))
        .collect();
    let mut result_bf16 = vec![0u16; q_dim];
    let ok = unsafe { xdna1_ffi::attn_gate_run(h, &gate_bf16, &out_bf16, &mut result_bf16) };
    if !ok {
        return Ok(false);
    }
    let result_f32: Vec<f32> = result_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
    let bytes = unsafe {
        std::slice::from_raw_parts(result_f32.as_ptr().cast::<u8>(), result_f32.len() * 4)
    };
    gpu.hip.memcpy_htod(&attn_out.buf, bytes)?;
    eprintln!("[xdna1] layer={layer_idx} npu attn_gate ok");
    Ok(true)
}

fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

pub(crate) fn bf16_bytes_to_f16_bytes(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(2)
        .flat_map(|c| {
            let v = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            f32_to_f16(v).to_le_bytes()
        })
        .collect()
}

fn hfq_plain_tensor_as_f32(info: &HfqTensorInfo, data: &[u8], name: &str) -> Vec<f32> {
    match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => bf16_bytes_to_f32(data),
        _ => panic!(
            "expected F16/F32/BF16 for {name}, got qt={}",
            info.quant_type
        ),
    }
}

/// Load norm weight for Qwen3.5: stored as offset from 1.0 (output = x * (1 + weight))
///
/// TODO(transformer-extraction): cross-arch duplicate. The Qwen2 variant
/// in `hipfire-arch-qwen2::qwen2::load_norm_weight_raw` is the same
/// shape minus the `+= 1.0` offset (Qwen2 uses standard RMSNorm) and
/// without the `model.language_model.` name prefix (Qwen2 stores norms
/// flat). Pull both into `hipfire_runtime::transformer::norm` during the
/// Transformer-extraction PR with the offset and prefix as parameters.
fn load_norm_weight(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));

    let mut f32_data = hfq_plain_tensor_as_f32(info, &data, name);
    // Qwen3.5 RMSNorm: output = x * rsqrt(var+eps) * (1 + weight)
    for v in &mut f32_data {
        *v += 1.0;
    }
    gpu.upload_f32(&f32_data, shape)
}

/// Load norm weight without the +1.0 offset — for standard RMSNorm tensors
/// (e.g., the final `model.language_model.norm.weight` stored as raw scale,
/// mean ~1.6 on Qwen3.5-MoE A3B). Applying +1.0 would over-amplify by ~60%.
#[allow(dead_code)]
fn load_norm_weight_raw(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));
    let f32_data = hfq_plain_tensor_as_f32(info, &data, name);
    gpu.upload_f32(&f32_data, shape)
}

/// Load a qt=16 matrix weight. BF16-capable RDNA arches keep raw BF16 by
/// default; older arches host-convert to same-size F16 so a single BF16-
/// preserving HFQ artifact remains portable. `HIPFIRE_BF16_WEIGHTS` accepts
/// `native`, `f16`, or `f32` as explicit overrides.
fn load_bf16_matrix_weight(gpu: &Gpu, data: &[u8], m: usize, k: usize) -> HipResult<WeightTensor> {
    match resolve_bf16_weight_load_mode(bf16_weight_load_mode_from_env(), &gpu.arch) {
        Bf16WeightLoadMode::Native => {
            let mut buf = gpu.upload_raw(data, &[data.len()])?;
            buf.dtype = DType::BF16;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::BF16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        Bf16WeightLoadMode::F16 | Bf16WeightLoadMode::Auto => {
            let f16_data = bf16_bytes_to_f16_bytes(data);
            let buf = gpu.upload_raw(&f16_data, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F16,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        Bf16WeightLoadMode::F32 => {
            let f32_data = bf16_bytes_to_f32(data);
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
    }
}

// OQ4 arch-packing is the SINGLE source of truth in `hipfire_runtime::oq4_arch`
// (re-exported through `hipfire_runtime::hfq`): every qt=34 loader and the
// offline optimize tool call it. Re-exported here — including the canonical (34)
// / arch-packed (37) quant-type codes and the `oq4_arch_load` decision helper —
// to preserve the historical `hipfire_arch_qwen35::qwen35::{...}` paths without
// keeping a second copy. Canonical (34) repacks at load; arch-packed (37) is the
// combined layout already and uploads verbatim. The quant-type code IS the
// layout version: a future layout change takes a NEW code, so a stale artifact
// refuses via the loader's catch-all rather than reading as garbage.
pub use hipfire_runtime::hfq::{
    oq4_arch_combined_len, oq4_arch_load, oq4_pack_arch_combined, OQ4_ARCH_PACKED_QT,
    OQ4_CANONICAL_QT,
};

/// TODO(transformer-extraction): cross-arch duplicate. The Qwen2 variant
/// in `hipfire-arch-qwen2::qwen2::load_weight_tensor` inlines a subset
/// of this match (only HFQ4G256, HFQ4G128, F16 — the formats Qwen2 HFQ
/// files actually use). Pull this full quant-type matcher into
/// `hipfire_runtime::transformer::weights` so every arch crate shares
/// one implementation. Will also resolve the AWQ-sidecar attachment
/// hand-off cleanly.
fn load_weight_tensor_raw(
    gpu: &Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    match quant_type {
        6 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        7 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        8 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        11 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        12 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFQ3G128,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        13 => {
            // MQ4-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        14 => {
            // MQ8-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        15 => {
            // MQ6-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ6G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        17 => {
            // MQ3-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        31 => {
            // QTIP-3 G256 (trellis-coded 3-bit, 100 B/group)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Qtip3G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        18 => {
            // MQ2-G256
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        19 => {
            // MQ2-G256-Lloyd — 2-bit + 4-entry fp16 codebook (72 bytes/group)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ2G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        20 => {
            // MQ3-G256-Lloyd — 3-bit + 8-entry fp16 codebook (112 bytes/group)
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ3G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        30 => {
            // MQ4-G256-Lloyd — 4-bit + 16-entry fp16 codebook (160 bytes/group)
            // Renumbered from qt 21 → 30 in mq4-lloyd merge to avoid HFP4G32=21 collision.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256Lloyd,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        21 => {
            // HFP4G32 — E2M1 + UE8M0 g32 + FP16 row scale. See docs/quant-formats/hfp4.md.
            // K%256 — kernel constraint (gemv_hfp4g32 in dispatch.rs); refuse here so a
            // stale or externally-quantized file fails at load instead of panicking on
            // first dispatch.
            assert!(
                k.is_multiple_of(256),
                "HFP4G32 v1 lm_head has K={k} but kernel requires K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::HFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        24 => {
            // MFP4G32 — HFP4G32 + offline FWHT. Drop-in MQ4 replacement; same byte
            // layout as qtype 21 with format_flags=0x05 stamped in the per-row hdr.
            assert!(
                k.is_multiple_of(256),
                "MFP4G32 lm_head has K={k} but kernel + FWHT both require K%256==0"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::MFP4G32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        3 => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        2 => {
            let buf = gpu.upload_raw(data, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        1 => match f16_lm_head_mode_from_env() {
            F16LmHeadMode::Native => {
                // qt=1 is F16. Keep raw F16 on GPU (previously decompressed
                // host-side to F32). Native F16 storage halves the lm_head
                // bandwidth and lets the dispatch path hit the WMMA-backed
                // `gemm_f16_batched_lmhead` kernel on gfx11. Set
                // HIPFIRE_LM_HEAD_F16=f32 to force the legacy F32 expansion.
                let buf = gpu.upload_raw(data, &[data.len()])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F16,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
            F16LmHeadMode::F32 => {
                let f32_data: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[m, k])?;
                Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m,
                    k,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                })
            }
        },
        16 => load_bf16_matrix_weight(gpu, data, m, k),
        36 => {
            // OQ+ compact magnitude-tiered W4A8 (quant_type id 36, ~4 b/w). On-disk
            // block = [f16 scale][128 int4 nibbles][N_out × (u8 idx, i8 val)] =
            // 130 + 2·N_out B/256-group (codec `quantize_oqplus_compact`). N_out is
            // DERIVED from the byte length (no extra metadata): expand the int4 bulk
            // nibbles to int8, then overlay the sparse int8 outliers at their
            // in-group indices → the Oq8 kernel buffer [int8 M*K | f32 scales M*ng].
            // Tagged Oq8G256 → the single iu8 W8A8 forward, unchanged. AWQ sidecar
            // (when present) is applied to x by the wrapper (Oq8G256 is allow-listed).
            const GROUP: usize = 256;
            assert_eq!(k % GROUP, 0, "OQ+C requires K % 256 == 0 (got K={k})");
            let ng = k / GROUP;
            let n_groups = m * ng;
            assert!(n_groups > 0 && !data.is_empty(), "OQ+C empty tensor");
            assert_eq!(
                data.len() % n_groups,
                0,
                "OQ+C byte length {} not divisible by n_groups {n_groups}",
                data.len()
            );
            let block_bytes = data.len() / n_groups;
            assert!(
                block_bytes >= 132 && (block_bytes - 130).is_multiple_of(2),
                "OQ+C block_bytes {block_bytes} invalid (expected 130 + 2·N_out)"
            );
            let n_out = (block_bytes - 130) / 2;
            let sext4 = |nib: u8| -> i8 {
                let v = (nib & 0xf) as i8;
                if v > 7 {
                    v - 16
                } else {
                    v
                }
            };
            let weight_bytes = m * k; // one int8 per weight after expand
            let scales_bytes = m * ng * 4;
            let mut combined = vec![0u8; weight_bytes + scales_bytes];
            for r in 0..m {
                for g in 0..ng {
                    let src = (r * ng + g) * block_bytes;
                    let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
                    let dst = r * k + g * GROUP;
                    // Expand the 128 int4 nibble bytes → 256 int8 (bulk).
                    for i in 0..128 {
                        let byte = data[src + 2 + i];
                        combined[dst + 2 * i] = sext4(byte & 0xf) as u8;
                        combined[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
                    }
                    // Overlay the sparse int8 outliers: (u8 idx, i8 val) × N_out.
                    let tbl = src + 130;
                    for s in 0..n_out {
                        let idx = data[tbl + 2 * s] as usize;
                        let val = data[tbl + 2 * s + 1];
                        combined[dst + idx] = val;
                    }
                    let so = weight_bytes + (r * ng + g) * 4;
                    combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
                }
            }
            let buf = gpu.upload_raw(&combined, &[combined.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Oq8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        33 => {
            // OQ+ / Opus Plus W4A8 (quant_type id 33). On-disk bytes are IDENTICAL
            // to Oq4 (qt=34): [f16 scale][128 signed nibbles] per 256-group, from
            // codec `quantize_oq4g256` (incl. its LDLQ/AWQ calibration). The format
            // difference is the runtime contract: nibble-EXPAND the int4 weights to
            // int8 here and tag the tensor Oq8G256 so it dispatches the W8A8 iu8
            // grouped-WMMA path with int8 ACTIVATIONS. Weight VALUES stay 4-bit (16
            // levels in [-8,7]); activations gain int8 precision (W4A8). Layout
            // becomes the Oq8 kernel buffer [int8 M*K | f32 scales M*ng] — the exact
            // upcast the `quantize_oq4g256` doc names. AWQ smooth, when present, is
            // applied to x by the wrapper via the awq_scale sidecar (unchanged).
            const GROUP: usize = 256;
            // Single-sourced from hipfire-quant-format (WP-3.3): Oq4G256 on-disk
            // block = 2 (f16 scale) + 128 nibbles = 130.
            const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq4G256
                .block_bytes()
                .unwrap();
            assert_eq!(k % GROUP, 0, "OQPLUS requires K % 256 == 0 (got K={k})");
            let ng = k / GROUP;
            let weight_bytes = m * k; // one int8 per weight after expand
            let scales_bytes = m * ng * 4;
            let expect = m * ng * BLOCK;
            assert_eq!(
                data.len(),
                expect,
                "OQPLUS weight byte length {} != M*ng*130 = {expect} (M={m} K={k})",
                data.len()
            );
            let sext4 = |nib: u8| -> i8 {
                let v = (nib & 0xf) as i8;
                if v > 7 {
                    v - 16
                } else {
                    v
                }
            };
            let mut combined = vec![0u8; weight_bytes + scales_bytes];
            for r in 0..m {
                for g in 0..ng {
                    let src = (r * ng + g) * BLOCK;
                    let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
                    let dst = r * k + g * GROUP;
                    for i in 0..128 {
                        let byte = data[src + 2 + i];
                        combined[dst + 2 * i] = sext4(byte & 0xf) as u8;
                        combined[dst + 2 * i + 1] = sext4(byte >> 4) as u8;
                    }
                    let so = weight_bytes + (r * ng + g) * 4;
                    combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
                }
            }
            let buf = gpu.upload_raw(&combined, &[combined.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Oq8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        OQ4_CANONICAL_QT | OQ4_ARCH_PACKED_QT => {
            // Opus Quant W4A4 (OQ4G256). Canonical (34) on-disk `[f16 scale][128
            // nibbles]`/256-group repacks to the arch combined device layout — packed
            // nibbles [M,K/2] + per-group f32 scales [M,K/256] + interleaved decode
            // records — so the forward derives the weight-scale pointer via
            // `GpuTensor::sub_offset(M*K/2, ..)` and feeds `gemm_oq4_grouped_wmma`.
            // Arch-packed (37, `hipfire optimize` output) IS that layout already and
            // uploads verbatim (zero-copy) after a length check. Prefill (MMQ/f16)
            // reads the split region (sub_offset 0); decode GEMVs read the interleaved
            // region. Activations quantize to int4 at runtime (`quantize_act_oq4`);
            // weights are FWHT-rotated offline so the forward FWHT-rotates x to match
            // (shared mq_rotate_x path). AWQ smooth, when present, is applied to x by
            // the wrapper via the awq_scale sidecar.
            let (bytes, gpu_dtype) = oq4_arch_load(quant_type, data, m, k)
                .expect("oq4_arch_load resolves the OQ4 canonical/arch-packed codes");
            let buf = gpu.upload_raw(&bytes, &[bytes.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        35 => {
            // Opus Quant W8A8 (OQ8G256, quant_type id 35). On-disk: [f16 scale]
            // [256 int8] per 256-group, row-contiguous (codec `quantize_oq8g256`).
            // Repack to the kernel layout in ONE buffer — int8 weights [M,K]
            // followed by per-group f32 scales [M,K/256] — so the forward derives
            // the weight-scale pointer via `sub_offset(M*K, ..)` and feeds
            // `gemm_oq8_grouped_wmma`. Activations are int8-quantized at runtime
            // (`quantize_act_oq8`); weights are FWHT-rotated offline so the forward
            // FWHT-rotates x to match.
            const GROUP: usize = 256;
            // Single-sourced from hipfire-quant-format (WP-3.3): Oq8G256 on-disk
            // block = 2 (f16 scale) + 256 int8 = 258.
            const BLOCK: usize = hipfire_runtime::quant::QuantType::Oq8G256
                .block_bytes()
                .unwrap();
            assert_eq!(k % GROUP, 0, "OQ8G256 requires K % 256 == 0 (got K={k})");
            let ng = k / GROUP;
            let weight_bytes = m * k;
            let scales_bytes = m * ng * 4;
            let expect = m * ng * BLOCK;
            assert_eq!(
                data.len(),
                expect,
                "OQ8G256 weight byte length {} != M*ng*258 = {expect} (M={m} K={k})",
                data.len()
            );
            let mut combined = vec![0u8; weight_bytes + scales_bytes];
            for r in 0..m {
                for g in 0..ng {
                    let src = (r * ng + g) * BLOCK;
                    let scale = f16_to_f32(u16::from_le_bytes([data[src], data[src + 1]]));
                    let dst = r * k + g * GROUP;
                    combined[dst..dst + 256].copy_from_slice(&data[src + 2..src + BLOCK]);
                    let so = weight_bytes + (r * ng + g) * 4;
                    combined[so..so + 4].copy_from_slice(&scale.to_le_bytes());
                }
            }
            let buf = gpu.upload_raw(&combined, &[combined.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::Oq8G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            })
        }
        // Honest refusal (Layer 3): an unrecognized quant_type is a capability
        // gap, not a crash. Refusing at LOAD — the earliest point — means the
        // forward never runs an unsupported weight, so this can't resurface as a
        // panic deep in a fused/GEMV dispatch. Classifiable via is_unsupported().
        _ => Err(HipError::unsupported(&format!(
            "qwen35 weight: unsupported quant_type {quant_type}"
        ))),
    }
}

fn alias_raw_tensor(slab: &GpuTensor, byte_offset: usize, len: usize) -> GpuTensor {
    let ptr = unsafe { (slab.buf.as_ptr() as *mut u8).add(byte_offset) as *mut std::ffi::c_void };
    GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, len) },
        shape: vec![len],
        dtype: DType::Raw,
    }
}

fn load_weight_tensor_from_slabs(
    slabs: Option<&SlabTensorIndex>,
    name: &str,
    m: usize,
    k: usize,
) -> Option<(String, WeightTensor)> {
    let idx = slabs?;
    let (entry_name, entry) = qwen35_tensor_name_candidates(name)
        .into_iter()
        .find_map(|candidate| idx.entries.get_key_value(&candidate))?;
    let dtype = slab_dtype_for_quant(entry.quant_type, k)?;
    if matches!(dtype, DType::HFP4G32 | DType::MFP4G32) {
        assert!(
            k.is_multiple_of(256),
            "{entry_name} has K={k} but kernel requires K%256==0"
        );
    }
    let slab = &idx.storage.slabs[entry.slab_idx];
    Some((
        entry_name.clone(),
        WeightTensor {
            buf: alias_raw_tensor(slab, entry.rel, entry.len),
            gpu_dtype: dtype,
            m,
            k,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        },
    ))
}

fn load_gpu_tensor_from_slabs(
    slabs: Option<&SlabTensorIndex>,
    name: &str,
) -> Option<(u8, GpuTensor)> {
    let idx = slabs?;
    let entry = qwen35_tensor_name_candidates(name)
        .into_iter()
        .find_map(|candidate| idx.entries.get(&candidate))?;
    let slab = &idx.storage.slabs[entry.slab_idx];
    Some((
        entry.quant_type,
        alias_raw_tensor(slab, entry.rel, entry.len),
    ))
}

/// Phase A Stage A — AWQ sidecar loader for the Qwen3.5 forward path.
///
/// The .hfq quantizer emits `<weight>.awq_scale.weight` (1D F16, length K)
/// alongside MQ4G256 weights that were AWQ pre-scaled. The dispatcher in
/// `fused_rmsnorm_rotate_for_mq` / `fused_rmsnorm_rotate_mq_batched_for`
/// looks at `WeightTensor.awq_scale.is_some()` to pick the AWQ-aware
/// kernel variant. WITHOUT this loader populating the field, every MQ4
/// weight ends up with `awq_scale: None`, the dispatcher falls through
/// to the non-AWQ kernel, and the math `(W·s) · (x/s) = W·x` breaks
/// because the runtime never divides by `s` — observed KLD blowup
/// 0.6721 → 13.4893 on 0.8B Qwen3.5 before this landed.
///
/// Lookup pattern matches `hipfire_runtime::hfq::load_awq_scale`:
/// strip trailing `.weight`, append `.awq_scale.weight`. Try both the
/// `model.language_model.`-prefixed name and the bare name (the qwen35
/// crate uses prefixed names; older sidecars or tests may use either).
fn load_awq_scale_for(hfq: &HfqFile, gpu: &Gpu, name: &str, k: usize) -> Option<GpuTensor> {
    let sidecar_name = match name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{name}.awq_scale.weight"),
    };
    let (sc_info, sc_data) = hfq.tensor_data_pread(&sidecar_name)?;
    // Must be 1D F16, length K. quant_type 1 = F16.
    if sc_info.quant_type != 1 {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} has quant_type={} (expected 1=F16); skipping",
            sc_info.quant_type
        );
        return None;
    }
    if sc_info.shape.len() != 1 || sc_info.shape[0] as usize != k {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} shape mismatch ({:?} vs expected [{}]); skipping",
            sc_info.shape, k
        );
        return None;
    }
    // F16 → F32 on host so the kernel takes a plain `const float*`.
    let f32_data: Vec<f32> = sc_data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_bytes.len()]).ok()
}

/// TODO(transformer-extraction): cross-arch duplicate of
/// `hipfire-arch-qwen2::qwen2::load_weight_tensor` — same name-lookup +
/// pread + AWQ-sidecar pattern, but qwen35 uses the
/// `model.language_model.` prefix (its HFQ files put text weights under
/// the VL-friendly nested name) where qwen2 uses flat `model.{...}`.
/// Pull into `hipfire_runtime::transformer::weights` with the prefix
/// as a parameter during consolidation.
fn load_weight_tensor(
    hfq: &HfqFile,
    gpu: &Gpu,
    slabs: Option<&SlabTensorIndex>,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    if let Some((matched_name, mut wt)) = load_weight_tensor_from_slabs(slabs, name, m, k) {
        if wt.gpu_dtype.supports_awq_sidecar() {
            wt.awq_scale = load_awq_scale_for(hfq, gpu, &matched_name, k)
                .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
        }
        return Ok(wt);
    }
    // Use pread path to avoid page cache buildup on unified-memory APUs.
    #[cfg(unix)]
    {
        let mut wt: Option<WeightTensor> = None;
        let mut matched: Option<String> = None;
        for candidate in qwen35_tensor_name_candidates(name) {
            if let Some((info, buf)) = hfq.tensor_data_pread(&candidate) {
                let qt = info.quant_type;
                wt = Some(load_weight_tensor_raw(gpu, qt, &buf, m, k)?);
                matched = Some(candidate);
                break;
            }
        }
        let mut wt = wt.unwrap_or_else(|| panic!("tensor not found: {name}"));
        // Phase A Stage A — populate awq_scale when the dtype is on
        // the AWQ allow-list (centralized at `DType::supports_awq_sidecar`).
        // The pread call invalidates the prior pread_buf borrow, but
        // the weight bytes have already been uploaded to GPU (owned by
        // `wt.buf`) so the borrow no longer matters.
        if wt.gpu_dtype.supports_awq_sidecar() {
            if let Some(matched_name) = matched.as_deref() {
                wt.awq_scale = load_awq_scale_for(hfq, gpu, matched_name, k)
                    .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
            } else {
                wt.awq_scale = load_awq_scale_for(hfq, gpu, name, k);
            }
        }
        Ok(wt)
    }
    #[cfg(not(unix))]
    {
        let (info, data, matched_name) = {
            let mut found = None;
            for candidate in qwen35_tensor_name_candidates(name) {
                if let Some((info, data)) = hfq.tensor_data(&candidate) {
                    found = Some((info, data, candidate));
                    break;
                }
            }
            found.unwrap_or_else(|| panic!("tensor not found: {name}"))
        };
        let mut wt = load_weight_tensor_raw(gpu, info.quant_type, data, m, k)?;
        if wt.gpu_dtype.supports_awq_sidecar() {
            wt.awq_scale = load_awq_scale_for(hfq, gpu, &matched_name, k)
                .or_else(|| load_awq_scale_for(hfq, gpu, name, k));
        }
        Ok(wt)
    }
}

// ─── ParoQuant AWQ → HFQ4G128 repack ────────────────────────────────────────

/// Repack AWQ-format INT4 weights into HFQ4G128 inline layout.
///
/// AWQ layout (3 separate tensors):
///   qweight: I32 [in_dim, out_dim/8] — 8 nibbles per I32
///   qzeros:  I32 [in_dim/group_size, out_dim/8] — 8 zero-point nibbles per I32
///   scales:  F16 [in_dim/group_size, out_dim] — per-group scales
///
/// HFQ4G128 layout (per output row, one contiguous buffer):
///   For each group of 128 input elements:
///     [f32 scale (4B)][f32 zero (4B)][64B packed nibbles] = 72 bytes
///
/// Returns: Vec<u8> in HFQ4G128 format, ready for gpu.upload_raw.
///
/// SYNC: must match `repack_awq_to_hfq4g128` in
/// `crates/hipfire-runtime/src/hfq.rs`. Duplicated to avoid a cross-crate
/// dependency cycle (hipfire-arch-qwen35 -> hipfire-runtime); keep the two
/// bodies byte-identical when editing.
fn repack_awq_to_hfq4g128(
    qweight: &[u8],    // I32 raw bytes
    qzeros: &[u8],     // I32 raw bytes
    scales: &[u8],     // F16 raw bytes
    out_dim: usize,    // M (output features)
    in_dim: usize,     // K (input features)
    group_size: usize, // 128
) -> Vec<u8> {
    let groups_per_row = in_dim / group_size;
    let bytes_per_row = groups_per_row * 72;
    let mut out = vec![0u8; out_dim * bytes_per_row];

    // Parse qweight as &[u32] (LE)
    debug_assert_eq!(
        qweight.as_ptr() as usize % 4,
        0,
        "AWQ qweight not 4-byte aligned"
    );
    let qw: &[u32] =
        unsafe { std::slice::from_raw_parts(qweight.as_ptr() as *const u32, qweight.len() / 4) };
    // qweight shape: [in_dim, out_dim/8] → row-major
    let qw_cols = out_dim / 8;

    // Parse qzeros as &[u32]
    debug_assert_eq!(
        qzeros.as_ptr() as usize % 4,
        0,
        "AWQ qzeros not 4-byte aligned"
    );
    let qz: &[u32] =
        unsafe { std::slice::from_raw_parts(qzeros.as_ptr() as *const u32, qzeros.len() / 4) };
    // qzeros shape: [in_dim/group_size, out_dim/8]
    let qz_cols = out_dim / 8;

    // Parse scales as &[u16] (F16)
    debug_assert_eq!(
        scales.as_ptr() as usize % 2,
        0,
        "AWQ scales not 2-byte aligned"
    );
    let sc: &[u16] =
        unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u16, scales.len() / 2) };
    // scales shape: [in_dim/group_size, out_dim]

    // AWQ nibble reorder: ParoQuant packs with _AWQ_REORDER=(0,2,4,6,1,3,5,7).
    // To extract element m, use the inverse permutation:
    const AWQ_DEQUANT: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

    for m in 0..out_dim {
        for g in 0..groups_per_row {
            let row_off = m * bytes_per_row + g * 72;

            let scale_f16 = sc[g * out_dim + m];
            let scale_f32 = f16_to_f32(scale_f16);

            let zero_i32 = qz[g * qz_cols + m / 8];
            let zero_nibble = ((zero_i32 >> (AWQ_DEQUANT[m % 8] * 4)) & 0xF) as f32;
            let zero_f32 = -scale_f32 * zero_nibble;

            out[row_off..row_off + 4].copy_from_slice(&scale_f32.to_le_bytes());
            out[row_off + 4..row_off + 8].copy_from_slice(&zero_f32.to_le_bytes());

            let nibble_shift = AWQ_DEQUANT[m % 8] * 4;
            let qw_col = m / 8;
            for i in 0..64 {
                let in_idx0 = g * group_size + i * 2;
                let in_idx1 = in_idx0 + 1;

                let nib0 = ((qw[in_idx0 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;
                let nib1 = ((qw[in_idx1 * qw_cols + qw_col] >> nibble_shift) & 0xF) as u8;

                // HFQ4G128: lo nibble = even element, hi nibble = odd element
                out[row_off + 8 + i] = nib0 | (nib1 << 4);
            }
        }
    }

    out
}

/// Load a ParoQuant-quantized weight from a SafetensorsSource.
/// Repacks AWQ INT4 → HFQ4G128 and uploads rotation metadata.
fn load_paroquant_weight(
    source: &dyn ModelSource,
    gpu: &Gpu,
    tensor_prefix: &str, // e.g. "model.language_model.layers.0.mlp.gate_proj"
    out_dim: usize,      // M
    in_dim: usize,       // K
    group_size: u32,
    krot: u8,
) -> HipResult<WeightTensor> {
    let qw_name = format!("{tensor_prefix}.qweight");
    let qz_name = format!("{tensor_prefix}.qzeros");
    let sc_name = format!("{tensor_prefix}.scales");
    let pairs_name = format!("{tensor_prefix}.pairs");
    let theta_name = format!("{tensor_prefix}.theta");
    let cs_name = format!("{tensor_prefix}.channel_scales");

    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {sc_name}")))?;

    // Repack AWQ → HFQ4G128
    let hfq_data = repack_awq_to_hfq4g128(
        qw_data,
        qz_data,
        sc_data,
        out_dim,
        in_dim,
        group_size as usize,
    );
    let buf = gpu.upload_raw(&hfq_data, &[hfq_data.len()])?;

    // Load rotation metadata
    let (_, pairs_data) = source
        .tensor_data(&pairs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {pairs_name}")))?;
    let (_, theta_data) = source
        .tensor_data(&theta_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {theta_name}")))?;
    let (_, cs_data) = source
        .tensor_data(&cs_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant tensor not found: {cs_name}")))?;

    let pairs = gpu.upload_raw(pairs_data, &[pairs_data.len()])?;
    let theta = gpu.upload_raw(theta_data, &[theta_data.len()])?;
    let channel_scales = gpu.upload_raw(cs_data, &[cs_data.len()])?;

    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::ParoQ4G128,
        m: out_dim,
        k: in_dim,
        row_stride: 0,
        paro: Some(ParoRotation {
            pairs,
            theta,
            channel_scales,
            krot: krot as u32,
            group_size,
            is_alias: false,
        }),
        awq_scale: None,
    })
}

/// Load an FP16 weight and encode it into MQ4G128 byte layout at load time.
/// Used by `paro_load_wt` for LinearAttention `in_proj_a` / `in_proj_b` weights
/// (alpha/beta) when the PARO checkpoint doesn't include them in the calibrated
/// set AND the per-arch/env gating chose the MQ4G128 path.
///
/// At decode time, the weight routes through `gemv_mq4g128_prerotated` which
/// applies FWHT-128 to the activation (via `rotate_x_mq_128_for`) before the
/// inner GEMV. Encoder applies FWHT-128 to weight with the same sign tables,
/// so the two FWHTs orthogonally cancel.
fn load_fp16_then_encode_mq4g128(
    source: &dyn ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (_, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    debug_assert_eq!(
        data.len(),
        2 * m * k,
        "load_fp16_then_encode_mq4g128: tensor {name} byte len {} != 2*m*k {}",
        data.len(),
        2 * m * k
    );
    let fp16: &[u16] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2) };
    let encoded = crate::paro_la_gates_codec::encode_mq4g128_from_fp16(fp16, m, k);
    let buf = gpu.upload_raw(&encoded.bytes, &[encoded.bytes.len()])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::MQ4G128,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

/// Load an FP16 weight tensor from safetensors (for excluded/unquantized layers).
fn load_fp16_weight_from_source(
    source: &dyn ModelSource,
    gpu: &Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (_, data) = source
        .tensor_data(name)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {name}")))?;
    let f32_data: Vec<f32> = data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4) };
    let buf = gpu.upload_raw(bytes, &[m, k])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::F32,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

// ─── ParoQuant MoE expert loading (Option A — per-expert qweight, shared sidecars) ──

/// Repack a single per-expert AWQ projection (gate, up, or down) into HFQ4G128
/// byte rows. Returns the row-major byte buffer (size `out_dim * groups_per_row * 72`).
///
/// Caller is responsible for uploading the buffer to GPU (or concatenating with
/// another projection's rows before upload — gate||up fusion path).
fn paro_repack_moe_projection(
    source: &dyn ModelSource,
    full_prefix: &str, // e.g. "model.language_model.layers.0.mlp.experts.5.gate_proj"
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> HipResult<Vec<u8>> {
    let qw_name = format!("{full_prefix}.qweight");
    let qz_name = format!("{full_prefix}.qzeros");
    let sc_name = format!("{full_prefix}.scales");
    let (_, qw_data) = source
        .tensor_data(&qw_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {qw_name}")))?;
    let (_, qz_data) = source
        .tensor_data(&qz_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {qz_name}")))?;
    let (_, sc_data) = source
        .tensor_data(&sc_name)
        .ok_or_else(|| HipError::new(0, &format!("ParoQuant MoE tensor not found: {sc_name}")))?;
    Ok(repack_awq_to_hfq4g128(
        qw_data, qz_data, sc_data, out_dim, in_dim, group_size,
    ))
}

/// Upload the per-layer shared PARO rotation sidecars (one tuple for gate||up,
/// one for down). All 256 experts will reference these via non-owning
/// `ParoRotation` aliases.
///
/// Shisa-ai's PARO checkpoint stores these at:
///   `model.language_model.layers.{L}.mlp.experts.{gate_up,down}_weight_{pairs,theta,channel_scales}`
pub(crate) fn paro_load_moe_shared_sidecars(
    source: &dyn ModelSource,
    gpu: &Gpu,
    p: &str, // e.g. "layers.0"
) -> HipResult<MoeParoSidecars> {
    let mp = paro_text_prefix(source)?;
    let base = format!("{mp}.{p}.mlp.experts");
    let load = |name: &str| -> HipResult<GpuTensor> {
        let full = format!("{base}.{name}");
        let (_, data) = source.tensor_data(&full).ok_or_else(|| {
            HipError::new(
                0,
                &format!("ParoQuant MoE shared sidecar not found: {full}"),
            )
        })?;
        gpu.upload_raw(data, &[data.len()])
    };
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant: quant_config required"))?;
    Ok(MoeParoSidecars {
        gate_up_pairs: load("gate_up_weight_pairs")?,
        gate_up_theta: load("gate_up_weight_theta")?,
        gate_up_channel_scales: load("gate_up_weight_channel_scales")?,
        down_pairs: load("down_weight_pairs")?,
        down_theta: load("down_weight_theta")?,
        down_channel_scales: load("down_weight_channel_scales")?,
        krot: qc.krot as u32,
        group_size: qc.group_size,
    })
}

/// Build a non-owning `ParoRotation` whose tensor fields alias `src`'s
/// underlying GPU memory. The returned rotation must NOT outlive `src`;
/// callers store the owning `MoeParoSidecars` in `MoeFfnWeights.paro_shared`
/// to guarantee that.
fn alias_paro_rotation(
    pairs_src: &GpuTensor,
    theta_src: &GpuTensor,
    cs_src: &GpuTensor,
    krot: u32,
    group_size: u32,
) -> ParoRotation {
    let alias = |t: &GpuTensor| -> GpuTensor {
        GpuTensor {
            buf: unsafe { t.buf.alias() },
            shape: t.shape.clone(),
            dtype: t.dtype,
        }
    };
    ParoRotation {
        pairs: alias(pairs_src),
        theta: alias(theta_src),
        channel_scales: alias(cs_src),
        krot,
        group_size,
        is_alias: true,
    }
}

/// Load the full ParoQuant MoE FFN block for one layer:
///   - dense FP16 router (`mlp.gate.weight [n_exp, hidden]`)
///   - dense FP16 shared-expert scalar gate (`mlp.shared_expert_gate.weight [1, hidden]`)
///   - shared expert (three per-projection PARO tensors: gate, up, down)
///   - 256 routed experts, each with a fused gate||up HFQ4G128 buffer + a down
///     HFQ4G128 buffer, all referencing layer-shared PARO sidecars
fn paro_load_moe_ffn(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    p: &str, // e.g. "layers.0"
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;
    let smi = config.shared_expert_intermediate_size;
    let dim = config.dim;
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant MoE requires quant_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;

    let mp = paro_text_prefix(source)?;

    // ── Router (FP16 dense in shisa-ai's PARO checkpoint) ──
    // mlp.gate.weight is NOT PARO-quantized — only the expert FFN matmuls are.
    let router = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.gate.weight"),
        n_exp,
        dim,
    )?;

    // Scalar gate on the shared-expert add — also FP16 dense.
    let shared_expert_gate = load_fp16_weight_from_source(
        source,
        gpu,
        &format!("{mp}.{p}.mlp.shared_expert_gate.weight"),
        1,
        dim,
    )?;

    // ── Shared expert (its own per-projection PARO sidecars, no sharing) ──
    let shared_expert = SharedExpertWeights {
        gate: paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.gate_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        up: paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.up_proj"),
            smi,
            dim,
            gs,
            kr,
        )?,
        down: paro_load_wt(
            source,
            gpu,
            &format!("{p}.mlp.shared_expert.down_proj"),
            dim,
            smi,
            gs,
            kr,
        )?,
    };

    // ── Routed experts ──
    // shisa-ai stores per-expert qweight/qzeros/scales but ONE shared
    // pairs/theta/channel_scales tuple per projection-group (gate||up vs down)
    // for ALL experts in the layer. Upload sidecars once, alias into each
    // expert's WeightTensor.paro.
    let shared = paro_load_moe_shared_sidecars(source, gpu, p)?;

    let groups_per_row_hidden = dim / (gs as usize); // 2048/128 = 16
    let bytes_per_row_hidden = groups_per_row_hidden * 72; // 1152
    let groups_per_row_mi = mi / (gs as usize); // 512/128 = 4
    let bytes_per_row_mi = groups_per_row_mi * 72; // 288

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        // Per-expert prefixes (full dot-path is constructed inside the helper).
        let gate_prefix = format!("{mp}.{p}.mlp.experts.{x}.gate_proj");
        let up_prefix = format!("{mp}.{p}.mlp.experts.{x}.up_proj");
        let down_prefix = format!("{mp}.{p}.mlp.experts.{x}.down_proj");

        // Fuse gate || up at HFQ4G128 row level: each row is independent
        // (`bytes_per_row` bytes, no cross-row state), so concat works.
        // Final shape: [2*mi, dim], rows [0..mi] = gate, rows [mi..2*mi] = up.
        let gate_bytes = paro_repack_moe_projection(source, &gate_prefix, mi, dim, gs as usize)?;
        let up_bytes = paro_repack_moe_projection(source, &up_prefix, mi, dim, gs as usize)?;
        debug_assert_eq!(gate_bytes.len(), mi * bytes_per_row_hidden);
        debug_assert_eq!(up_bytes.len(), mi * bytes_per_row_hidden);
        let mut gate_up_bytes = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        gate_up_bytes.extend_from_slice(&gate_bytes);
        gate_up_bytes.extend_from_slice(&up_bytes);
        let gate_up_buf = gpu.upload_raw(&gate_up_bytes, &[gate_up_bytes.len()])?;

        let down_bytes = paro_repack_moe_projection(source, &down_prefix, dim, mi, gs as usize)?;
        debug_assert_eq!(down_bytes.len(), dim * bytes_per_row_mi);
        let down_buf = gpu.upload_raw(&down_bytes, &[down_bytes.len()])?;

        let gate_up = WeightTensor {
            buf: gate_up_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: 2 * mi,
            k: dim,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.gate_up_pairs,
                &shared.gate_up_theta,
                &shared.gate_up_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        let down = WeightTensor {
            buf: down_buf,
            gpu_dtype: DType::ParoQ4G128,
            m: dim,
            k: mi,
            row_stride: 0,
            paro: Some(alias_paro_rotation(
                &shared.down_pairs,
                &shared.down_theta,
                &shared.down_channel_scales,
                shared.krot,
                shared.group_size,
            )),
            awq_scale: None,
        };
        experts.push(ExpertWeights { gate_up, down });
    }

    // ── Device-side expert pointer tables (mirrors load_moe_ffn) ──
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts {
        gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(e.down.buf.buf.as_ptr() as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;
    let expert_gate_up_dtype = experts.first().map(|e| e.gate_up.gpu_dtype);
    let expert_down_dtype = experts.first().map(|e| e.down.gpu_dtype);

    Ok(MoeFfnWeights {
        router,
        experts,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        layer_idx,
        expert_shape: None,
        expert_gate_up_dtype,
        expert_down_dtype,
        paro_shared: Some(shared),
    })
}

// ─── Standard HFQ loading ───────────────────────────────────────────────────

/// Load a tensor as F32 on GPU, handling any quant type by dequanting on CPU.
fn load_any_as_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let (info, data) =
        qwen35_tensor_data_vec(hfq, name).unwrap_or_else(|| panic!("tensor not found: {name}"));

    let f32_data: Vec<f32> = match info.quant_type {
        1 | 2 | 16 => hfq_plain_tensor_as_f32(info, &data, name),
        3 => hipfire_runtime::quant::dequant_q8f16(&data, n),
        14 => {
            // MQ8-G256: [f16 scale][int8 × 256] = 258 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 258;
            let n_groups = data.len() / bytes_per_group;
            let signs1 = hipfire_runtime::kv::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::kv::KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale_bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
                let scale = hipfire_runtime::quant::f16_to_f32(scale_bits);
                let start = out.len();
                for i in 0..256 {
                    let q = data[off + 2 + i] as i8;
                    out.push(scale * q as f32);
                }
                // Inverse FWHT to recover original values
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let inv_s = 0.0625;
                for i in 0..256 {
                    group[i] *= inv_s * signs1[i];
                }
            }
            out
        }
        6 | 7 | 13 | 15 => {
            // HFQ4-G256 or G128 or MQ4-G256 or MQ6-G256 — CPU dequant
            // MQ4/MQ6 store rotated weights. For small tensors loaded here,
            // we dequant then inverse-rotate to recover the original values.
            let is_6bit = info.quant_type == 15;
            let group_size: usize =
                if info.quant_type == 6 || info.quant_type == 13 || info.quant_type == 15 {
                    256
                } else {
                    128
                };
            let bytes_per_group = if is_6bit { 200 } else { 8 + group_size / 2 };
            let n_groups = data.len() / bytes_per_group;
            let is_mq = info.quant_type == 13 || info.quant_type == 15;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let (signs1, signs2) = if is_mq {
                (
                    Some(hipfire_runtime::kv::KvCache::gen_fwht_signs(42, 256)),
                    Some(hipfire_runtime::kv::KvCache::gen_fwht_signs(1042, 256)),
                )
            } else {
                (None, None)
            };
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_6bit {
                    for i in (0..group_size).step_by(4) {
                        let bo = off + 8 + (i / 4) * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        out.push(scale * ((b0 & 0x3F) as f32) + zero);
                        out.push(scale * ((((b0 >> 6) | (b1 << 2)) & 0x3F) as f32) + zero);
                        out.push(scale * ((((b1 >> 4) | (b2 << 4)) & 0x3F) as f32) + zero);
                        out.push(scale * (((b2 >> 2) & 0x3F) as f32) + zero);
                    }
                } else {
                    for i in 0..group_size {
                        let byte_idx = i / 2;
                        let byte_val = data[off + 8 + byte_idx];
                        let nibble = if i % 2 == 0 {
                            byte_val & 0xF
                        } else {
                            byte_val >> 4
                        };
                        out.push(scale * nibble as f32 + zero);
                    }
                }
                // Inverse FWHT for MQ4/MQ6: recover original weight values
                if is_mq && group_size == 256 {
                    let s1 = signs1.as_ref().unwrap();
                    let s2 = signs2.as_ref().unwrap();
                    let group = &mut out[start..start + 256];
                    // Inverse FWHT: signs2 → butterfly → scale → signs1
                    for i in 0..256 {
                        group[i] *= s2[i];
                    }
                    let mut stride = 1;
                    while stride < 256 {
                        let mut j = 0;
                        while j < 256 {
                            for k in 0..stride {
                                let a = group[j + k];
                                let b = group[j + k + stride];
                                group[j + k] = a + b;
                                group[j + k + stride] = a - b;
                            }
                            j += stride * 2;
                        }
                        stride <<= 1;
                    }
                    let scale_inv = 0.0625; // 1/sqrt(256)
                    for i in 0..256 {
                        group[i] *= scale_inv * s1[i];
                    }
                }
            }
            out
        }
        8 => {
            // HFQ6-G256 — CPU dequant: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 200; // 8 + 192
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                // 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
                for i in (0..group_size).step_by(4) {
                    let byte_off = 8 + (i / 4) * 3;
                    let b0 = data[off + byte_off] as u32;
                    let b1 = data[off + byte_off + 1] as u32;
                    let b2 = data[off + byte_off + 2] as u32;
                    let q0 = (b0 & 0x3F) as f32;
                    let q1 = (((b0 >> 6) | (b1 << 2)) & 0x3F) as f32;
                    let q2 = (((b1 >> 4) | (b2 << 4)) & 0x3F) as f32;
                    let q3 = ((b2 >> 2) & 0x3F) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                }
            }
            out
        }
        11 => {
            // HFQ3-G256: [f32 scale][f32 zero][96B packed 3-bit] = 104 bytes per 256 weights
            let group_size: usize = 256;
            let bytes_per_group: usize = 104;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                // 8 values per 3 bytes (matching kernel unpack)
                for chunk in 0..32 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        12 => {
            // HFQ3-G128: [f32 scale][f32 zero][48B packed 3-bit] = 56 bytes per 128 weights
            let group_size: usize = 128;
            let bytes_per_group: usize = 56;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                for chunk in 0..16 {
                    let bo = off + 8 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as f32;
                    let q1 = ((b0 >> 3) & 7) as f32;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                    let q3 = ((b1 >> 1) & 7) as f32;
                    let q4 = ((b1 >> 4) & 7) as f32;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                    let q6 = ((b2 >> 2) & 7) as f32;
                    let q7 = ((b2 >> 5) & 7) as f32;
                    out.push(scale * q0 + zero);
                    out.push(scale * q1 + zero);
                    out.push(scale * q2 + zero);
                    out.push(scale * q3 + zero);
                    out.push(scale * q4 + zero);
                    out.push(scale * q5 + zero);
                    out.push(scale * q6 + zero);
                    out.push(scale * q7 + zero);
                }
            }
            out
        }
        20 => {
            // MQ3-G256-Lloyd (qt 20, 112 B/group): 8 fp16 codebook entries + 3-bit
            // indices (cross-byte, 32 chunks × 3 bytes × 8 weights). Decode is
            // direct lookup `cb[idx]` then inverse FWHT for CPU consumers.
            let group_size: usize = 256;
            let bytes_per_group: usize = 112;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::kv::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::kv::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 8];
                for k in 0..8 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::quant::f16_to_f32(bits);
                }
                let start = out.len();
                for chunk in 0..32 {
                    let bo = off + 16 + chunk * 3;
                    let b0 = data[bo] as u32;
                    let b1 = data[bo + 1] as u32;
                    let b2 = data[bo + 2] as u32;
                    let q0 = (b0 & 7) as usize;
                    let q1 = ((b0 >> 3) & 7) as usize;
                    let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as usize;
                    let q3 = ((b1 >> 1) & 7) as usize;
                    let q4 = ((b1 >> 4) & 7) as usize;
                    let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as usize;
                    let q6 = ((b2 >> 2) & 7) as usize;
                    let q7 = ((b2 >> 5) & 7) as usize;
                    out.push(cb[q0]);
                    out.push(cb[q1]);
                    out.push(cb[q2]);
                    out.push(cb[q3]);
                    out.push(cb[q4]);
                    out.push(cb[q5]);
                    out.push(cb[q6]);
                    out.push(cb[q7]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        19 => {
            // MQ2-G256-Lloyd (qt 19, 72 B/group): 4 fp16 codebook entries + 2-bit indices.
            // Decode is direct lookup `cb[idx]`, then inverse FWHT to recover original
            // pre-rotation values for CPU consumers (DeltaNet conv1d).
            let group_size: usize = 256;
            let bytes_per_group: usize = 72;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::kv::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::kv::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 4];
                for k in 0..4 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::quant::f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..64 {
                    let byte_val = data[off + 8 + i] as usize;
                    out.push(cb[byte_val & 3]);
                    out.push(cb[(byte_val >> 2) & 3]);
                    out.push(cb[(byte_val >> 4) & 3]);
                    out.push(cb[(byte_val >> 6) & 3]);
                }
                // Inverse FWHT to recover pre-rotation weights — same butterfly as the
                // MQ3/MQ2 arm below.
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        30 => {
            // MQ4-G256-Lloyd (qt 30, 160 B/group): 16 fp16 codebook entries (bytes [0..32))
            // + 4-bit packed indices (bytes [32..160), low nibble = idx[2i], high = idx[2i+1]).
            // Decode is direct lookup `cb[idx]` then inverse FWHT for CPU consumers.
            // Renumbered from qt 21 → 30 to avoid HFP4G32=21 collision.
            let group_size: usize = 256;
            let bytes_per_group: usize = 160;
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::kv::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::kv::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let mut cb = [0.0f32; 16];
                for k in 0..16 {
                    let bits = u16::from_le_bytes([data[off + 2 * k], data[off + 2 * k + 1]]);
                    cb[k] = hipfire_runtime::quant::f16_to_f32(bits);
                }
                let start = out.len();
                for i in 0..128 {
                    let byte_val = data[off + 32 + i] as usize;
                    out.push(cb[byte_val & 0xF]);
                    out.push(cb[(byte_val >> 4) & 0xF]);
                }
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625;
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        17 | 18 => {
            // MQ3-G256 (qt 17, 104 B/group, 3-bit) or MQ2-G256 (qt 18, 72 B/group, 2-bit).
            // Both store FWHT-rotated weights — dequant then inverse-rotate to recover
            // original values for CPU consumers (e.g., DeltaNet conv1d).
            let is_mq3 = info.quant_type == 17;
            let group_size: usize = 256;
            let bytes_per_group: usize = if is_mq3 { 104 } else { 72 };
            let n_groups = data.len() / bytes_per_group;
            let mut out = Vec::with_capacity(n_groups * group_size);
            let signs1 = hipfire_runtime::kv::KvCache::gen_fwht_signs(42, 256);
            let signs2 = hipfire_runtime::kv::KvCache::gen_fwht_signs(1042, 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale =
                    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
                let zero = f32::from_le_bytes([
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]);
                let start = out.len();
                if is_mq3 {
                    // 8 values per 3 bytes (matches gemv_hfq3g256.hip unpack).
                    for chunk in 0..32 {
                        let bo = off + 8 + chunk * 3;
                        let b0 = data[bo] as u32;
                        let b1 = data[bo + 1] as u32;
                        let b2 = data[bo + 2] as u32;
                        let q0 = (b0 & 7) as f32;
                        let q1 = ((b0 >> 3) & 7) as f32;
                        let q2 = (((b0 >> 6) | (b1 << 2)) & 7) as f32;
                        let q3 = ((b1 >> 1) & 7) as f32;
                        let q4 = ((b1 >> 4) & 7) as f32;
                        let q5 = (((b1 >> 7) | (b2 << 1)) & 7) as f32;
                        let q6 = ((b2 >> 2) & 7) as f32;
                        let q7 = ((b2 >> 5) & 7) as f32;
                        out.push(scale * q0 + zero);
                        out.push(scale * q1 + zero);
                        out.push(scale * q2 + zero);
                        out.push(scale * q3 + zero);
                        out.push(scale * q4 + zero);
                        out.push(scale * q5 + zero);
                        out.push(scale * q6 + zero);
                        out.push(scale * q7 + zero);
                    }
                } else {
                    // MQ2: 4 values per byte (matches gemv_hfq2g256.hip unpack).
                    for i in 0..64 {
                        let byte_val = data[off + 8 + i] as u32;
                        out.push(scale * ((byte_val & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 2) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 4) & 3) as f32) + zero);
                        out.push(scale * (((byte_val >> 6) & 3) as f32) + zero);
                    }
                }
                // Inverse FWHT: recover original (pre-rotation) weight values.
                let group = &mut out[start..start + 256];
                for i in 0..256 {
                    group[i] *= signs2[i];
                }
                let mut stride = 1;
                while stride < 256 {
                    let mut j = 0;
                    while j < 256 {
                        for k in 0..stride {
                            let a = group[j + k];
                            let b = group[j + k + stride];
                            group[j + k] = a + b;
                            group[j + k + stride] = a - b;
                        }
                        j += stride * 2;
                    }
                    stride <<= 1;
                }
                let scale_inv = 0.0625; // 1/sqrt(256)
                for i in 0..256 {
                    group[i] *= scale_inv * signs1[i];
                }
            }
            out
        }
        // Honest refusal (Layer 3): unrecognized quant_type → typed capability
        // gap instead of a load-time panic. See load_weight_tensor_raw.
        _ => {
            return Err(HipError::unsupported(&format!(
                "qwen35 {name}: unsupported quant_type {}",
                info.quant_type
            )))
        }
    };
    gpu.upload_f32(&f32_data[..n], &[n])
}

/// Alias for load_any_as_f32.
fn load_raw_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    load_any_as_f32(hfq, gpu, name, n)
}

// TODO(transformer-extraction): the overall `load_weights` orchestration
// here (drop_mmap → embedding+tied-lm_head → norm → per-layer loop) is
// the model the Qwen2 loader at
// `hipfire-arch-qwen2::qwen2::load_weights` follows. The tied-embedding
// re-upload pattern (re-reading `embed_tokens.weight` to construct a
// second GpuTensor for the lm_head) is duplicated in both crates
// because GpuTensor is not Clone. Consolidation PR should either add
// `GpuTensor::shallow_clone()` or switch to `Arc<GpuTensor>` so tied
// embeddings stop costing 2× the embedding VRAM.

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn load_throughput_gibs(bytes: usize, seconds: f64) -> f64 {
    gib(bytes) / seconds.max(f64::MIN_POSITIVE)
}

struct SlabTensorEntry {
    slab_idx: usize,
    rel: usize,
    len: usize,
    quant_type: u8,
}

struct SlabTensorIndex {
    entries: HashMap<String, SlabTensorEntry>,
    storage: ModelGpuStorage,
}

struct SlabPlanBank {
    offset: usize,
    len: usize,
    tensor_indices: Vec<usize>,
}

fn gpu_slab_load_enabled(gpu: &Gpu) -> bool {
    match std::env::var("HIPFIRE_GPU_SLAB_LOAD").ok().as_deref() {
        Some("0" | "false" | "off" | "none") => false,
        Some("1" | "true" | "on" | "direct" | "slab") => true,
        Some("auto" | "uma") | None => gpu.integrated,
        Some(other) => {
            eprintln!("  warning: unknown HIPFIRE_GPU_SLAB_LOAD={other:?}; using UMA auto-detect");
            gpu.integrated
        }
    }
}

fn gpu_slab_bank_size() -> usize {
    std::env::var("HIPFIRE_GPU_SLAB_MIB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512)
        .max(1)
        * 1024
        * 1024
}

fn slab_dtype_for_quant(qt: u8, k: usize) -> Option<DType> {
    // Canonical map lives in hipfire_runtime::quant (shared across all arch
    // loaders). Thin alias retained for the slab-planning call sites.
    hipfire_runtime::quant::dtype_for_quant_type(qt, k)
}

fn build_slab_banks(hfq: &HfqFile, bank_size: usize) -> Vec<SlabPlanBank> {
    let mut banks = Vec::new();
    let mut cur: Option<SlabPlanBank> = None;
    let skip_routed_experts = matches!(
        std::env::var("HIPFIRE_QWEN35_PAGED_EXPERTS")
            .ok()
            .as_deref(),
        Some("1" | "true" | "on" | "yes")
    );
    for (idx, info) in hfq.tensors().iter().enumerate() {
        if skip_routed_experts && is_qwen35_routed_expert_tensor(&info.name) {
            continue;
        }
        if slab_dtype_for_quant(info.quant_type, 256).is_none() {
            continue;
        }
        let start = info.data_offset;
        let end = info.data_offset + info.data_size;
        match cur.as_mut() {
            Some(bank) => {
                let next_len = end - bank.offset;
                if next_len <= bank_size || bank.tensor_indices.is_empty() {
                    bank.len = next_len;
                    bank.tensor_indices.push(idx);
                } else {
                    banks.push(cur.take().unwrap());
                    cur = Some(SlabPlanBank {
                        offset: start,
                        len: info.data_size,
                        tensor_indices: vec![idx],
                    });
                }
            }
            None => {
                cur = Some(SlabPlanBank {
                    offset: start,
                    len: info.data_size,
                    tensor_indices: vec![idx],
                });
            }
        }
    }
    if let Some(bank) = cur {
        banks.push(bank);
    }
    banks
}

fn is_qwen35_routed_expert_tensor(name: &str) -> bool {
    name.contains(".mlp.experts.")
        && (name.contains(".gate_up_proj.weight")
            || name.contains(".gate_proj.weight")
            || name.contains(".up_proj.weight")
            || name.contains(".down_proj.weight"))
}

fn qwen35_layer_payload_bytes(hfq: &HfqFile, layer_prefix: &str, paged_experts: bool) -> usize {
    let prefixes = qwen35_tensor_name_candidates(layer_prefix)
        .into_iter()
        .map(|p| format!("{p}."))
        .collect::<Vec<_>>();
    hfq.tensors()
        .iter()
        .filter(|t| prefixes.iter().any(|p| t.name.starts_with(p)))
        .filter(|t| !paged_experts || !is_qwen35_routed_expert_tensor(&t.name))
        .map(|t| t.data_size)
        .sum()
}

fn load_gpu_slabs(hfq: &HfqFile, gpu: &mut Gpu) -> HipResult<SlabTensorIndex> {
    #[cfg(not(unix))]
    {
        let _ = (hfq, gpu);
        return Err(hip_bridge::HipError::new(
            0,
            "GPU slab loader requires unix O_DIRECT",
        ));
    }
    #[cfg(unix)]
    {
        gpu.bind_thread()?;
        let bank_size = gpu_slab_bank_size();
        let banks = build_slab_banks(hfq, bank_size);
        let file_len = std::fs::metadata(hfq.path())
            .map_err(|e| {
                hip_bridge::HipError::new(0, &format!("stat {}: {e}", hfq.path().display()))
            })?
            .len() as usize;
        let mut entries = HashMap::new();
        let mut slabs = Vec::with_capacity(banks.len());
        let mut total_bytes = 0usize;
        let t_alloc = std::time::Instant::now();
        for bank in &banks {
            let buf = gpu.hip.malloc(bank.len)?;
            slabs.push(GpuTensor {
                buf,
                shape: vec![bank.len],
                dtype: DType::Raw,
            });
            total_bytes += bank.len;
        }
        let alloc_s = t_alloc.elapsed().as_secs_f64();

        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(hfq.path())
            .map_err(|e| {
                hip_bridge::HipError::new(
                    0,
                    &format!("open O_DIRECT {}: {e}", hfq.path().display()),
                )
            })?;
        let max_direct_len = banks
            .iter()
            .map(|b| {
                let start = align_down(b.offset, GPU_SLAB_ALIGN);
                // NB: do NOT clamp the aligned end back to file_len — O_DIRECT
                // requires a block-aligned length, and read_direct_allow_eof
                // absorbs the short read when the last block extends past EOF.
                let end = align_up((b.offset + b.len).min(file_len), GPU_SLAB_ALIGN);
                end - start
            })
            .max()
            .unwrap_or(GPU_SLAB_ALIGN);
        let mut staging =
            AlignedLoadBuffer::new(max_direct_len.max(GPU_SLAB_ALIGN), GPU_SLAB_ALIGN).map_err(
                |e| hip_bridge::HipError::new(0, &format!("posix_memalign staging: {e}")),
            )?;

        let mut read_s = 0.0;
        let mut copy_s = 0.0;
        let t_load = std::time::Instant::now();
        for (bank_idx, bank) in banks.iter().enumerate() {
            let aligned_start = align_down(bank.offset, GPU_SLAB_ALIGN);
            // Aligned length must stay a GPU_SLAB_ALIGN multiple for O_DIRECT;
            // clamping the end to the (unaligned) file_len is what produced the
            // EINVAL "Invalid argument" on files whose size isn't 4K-aligned.
            // The tail block reads short at EOF via read_direct_allow_eof.
            let aligned_end = align_up((bank.offset + bank.len).min(file_len), GPU_SLAB_ALIGN);
            let aligned_len = aligned_end - aligned_start;
            let rel = bank.offset - aligned_start;
            let t_read = std::time::Instant::now();
            let got = read_direct_allow_eof(
                &file,
                staging.as_mut_slice(aligned_len),
                aligned_start as u64,
            )
            .map_err(|e| {
                hip_bridge::HipError::new(
                    0,
                    &format!(
                        "O_DIRECT read offset={} len={} aligned_offset={} aligned_len={}: {e}",
                        bank.offset, bank.len, aligned_start, aligned_len
                    ),
                )
            })?;
            if got < rel + bank.len {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "short O_DIRECT read offset={} len={} got={} need={}",
                        bank.offset,
                        bank.len,
                        got,
                        rel + bank.len
                    ),
                ));
            }
            read_s += t_read.elapsed().as_secs_f64();
            let t_copy = std::time::Instant::now();
            gpu.hip.memcpy_htod(
                &slabs[bank_idx].buf,
                &staging.as_slice(got)[rel..rel + bank.len],
            )?;
            copy_s += t_copy.elapsed().as_secs_f64();

            for &tensor_idx in &bank.tensor_indices {
                let info = &hfq.tensors()[tensor_idx];
                entries.insert(
                    info.name.clone(),
                    SlabTensorEntry {
                        slab_idx: bank_idx,
                        rel: info.data_offset - bank.offset,
                        len: info.data_size,
                        quant_type: info.quant_type,
                    },
                );
            }
        }
        let load_s = t_load.elapsed().as_secs_f64();
        eprintln!(
            "  GPU slab preload: banks={} tensors={} payload={:.2} GiB prealloc={:.2}s load={:.2}s read={:.2}s copy={:.2}s total_bw={:.2} GiB/s load_bw={:.2} GiB/s",
            banks.len(),
            entries.len(),
            gib(total_bytes),
            alloc_s,
            load_s,
            read_s,
            copy_s,
            load_throughput_gibs(total_bytes, alloc_s + load_s),
            load_throughput_gibs(total_bytes, load_s),
        );
        Ok(SlabTensorIndex {
            entries,
            storage: ModelGpuStorage::new(slabs, total_bytes),
        })
    }
}

fn align_down(v: usize, align: usize) -> usize {
    v & !(align - 1)
}

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}

#[cfg(unix)]
fn read_direct_allow_eof(
    file: &std::fs::File,
    dst: &mut [u8],
    offset: u64,
) -> std::io::Result<usize> {
    let mut done = 0usize;
    while done < dst.len() {
        let remaining = dst.len() - done;
        let n = file.read_at(&mut dst[done..], offset + done as u64)?;
        if n == 0 {
            break;
        }
        done += n;
        if n < remaining {
            break;
        }
    }
    Ok(done)
}

struct AlignedLoadBuffer {
    ptr: *mut u8,
    len: usize,
}

impl AlignedLoadBuffer {
    fn new(len: usize, align: usize) -> std::io::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut ptr, align, len.max(1)) };
        if rc != 0 {
            return Err(std::io::Error::from_raw_os_error(rc));
        }
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }

    fn as_slice(&self, len: usize) -> &[u8] {
        assert!(len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }
}

impl Drop for AlignedLoadBuffer {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr.cast()) };
    }
}

/// Map a `<base>.weight` tensor name to its `(layer_idx, RqProj)` key, or None if
/// it is not a residual projection roughquant protects.
fn rq_parse_proj(name: &str) -> Option<(u32, RqProj)> {
    let layers_pos = name.find("layers.")? + "layers.".len();
    let rest = &name[layers_pos..];
    let dot = rest.find('.')?;
    let layer_idx: u32 = rest[..dot].parse().ok()?;
    let proj = if name.contains("linear_attn.in_proj_qkv") {
        RqProj::Wqkv
    } else if name.contains("linear_attn.in_proj_z") {
        RqProj::Wz
    } else if name.contains("linear_attn.in_proj_a") {
        RqProj::Walpha
    } else if name.contains("linear_attn.in_proj_b") {
        RqProj::Wbeta
    } else if name.contains("self_attn.q_proj") {
        RqProj::Wq
    } else if name.contains("self_attn.k_proj") {
        RqProj::Wk
    } else if name.contains("self_attn.v_proj") {
        RqProj::Wv
    } else if name.contains("self_attn.o_proj") || name.contains("linear_attn.out_proj") {
        RqProj::Wo
    } else if name.contains("mlp.gate_proj") {
        RqProj::Wgate
    } else if name.contains("mlp.up_proj") {
        RqProj::Wup
    } else if name.contains("mlp.down_proj") {
        RqProj::Wdown
    } else {
        return None;
    };
    Some((layer_idx, proj))
}

#[inline]
fn rq_bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Load the roughquant real-format corrections from `metadata["roughquant_sidecar"]`
/// + the `<name>.rqcorr` bf16 tensors into a `(layer_idx, RqProj) -> RqCorr` map.
/// Empty when the model carries no sidecar (backward-compatible).
fn load_rq_corrections(
    hfq: &HfqFile,
    gpu: &mut Gpu,
) -> HipResult<std::collections::HashMap<(u32, RqProj), RqCorr>> {
    use std::collections::HashMap;
    let mut map: HashMap<(u32, RqProj), RqCorr> = HashMap::new();
    let meta: serde_json::Value = match serde_json::from_str(&hfq.metadata_json) {
        Ok(v) => v,
        Err(_) => return Ok(map),
    };
    let sidecar = match meta.get("roughquant_sidecar") {
        Some(s) => s,
        None => return Ok(map),
    };
    let tensors = match sidecar.get("tensors").and_then(|t| t.as_object()) {
        Some(t) => t,
        None => return Ok(map),
    };
    let (mut n_r, mut n_w) = (0usize, 0usize);
    for (name, entry) in tensors {
        let Some((layer_idx, proj)) = rq_parse_proj(name) else {
            continue;
        };
        let role = entry.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let corr_name = entry
            .get("corr")
            .and_then(|c| c.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("{name}.rqcorr"));
        let channels: Vec<i32> = entry
            .get("channels")
            .and_then(|c| c.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as i32))
                    .collect()
            })
            .unwrap_or_default();
        if channels.is_empty() {
            continue;
        }
        let Some((info, data)) = hfq.tensor_data_pread(&corr_name) else {
            eprintln!("  roughquant: WARN sidecar tensor {corr_name} missing; skipping");
            continue;
        };
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
        if shape.len() != 2 {
            continue;
        }
        let f32_vals: Vec<f32> = data
            .chunks_exact(2)
            .map(|c| rq_bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let idx_bytes: Vec<u8> = channels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let idx_tensor = gpu.upload_raw(&idx_bytes, &[channels.len()])?;
        if role == "reader" {
            // corr is [m × n_idx]; pad columns to a power-of-2 `np` (gemv_f32 tree
            // reduction needs a power-of-2 block when k<256).
            let m = shape[0];
            let n_idx = shape[1];
            let np = n_idx.next_power_of_two();
            let mut padded = vec![0.0f32; m * np];
            for r in 0..m {
                padded[r * np..r * np + n_idx]
                    .copy_from_slice(&f32_vals[r * n_idx..r * n_idx + n_idx]);
            }
            let corr = gpu.upload_f32(&padded, &[m, np])?;
            map.insert(
                (layer_idx, proj),
                RqCorr::Reader {
                    corr,
                    idx: idx_tensor,
                    m,
                    n_idx,
                    np,
                },
            );
            n_r += 1;
        } else {
            // writer: corr is [n_s × k]; gemv over k (≥256) needs no padding.
            let n_s = shape[0];
            let k = shape[1];
            let corr = gpu.upload_f32(&f32_vals, &[n_s, k])?;
            map.insert(
                (layer_idx, proj),
                RqCorr::Writer {
                    corr,
                    idx: idx_tensor,
                    n_s,
                    k,
                },
            );
            n_w += 1;
        }
    }
    if !map.is_empty() {
        eprintln!(
            "  roughquant: loaded {n_r} reader + {n_w} writer corrections \
             (DORMANT by default — hand path is broken; opt in with HIPFIRE_RQ_HAND=1 \
             for experiments; lowered super-op wiring is the verdict path. See \
             docs/roughquant/phase3-real-format-scope.md)"
        );
    }
    Ok(map)
}

/// Apply a residual-reader correction in place: `out += corr · gather(src_normed, S)`.
/// `src_normed` is the ORIGINAL-frame rmsnormed hidden (the projection's logical
/// input). Allocates small scratch per call (eval/decode path; graph disabled).
fn rq_apply_reader(
    gpu: &mut Gpu,
    corr: &RqCorr,
    src_normed: &GpuTensor,
    out: &GpuTensor,
) -> HipResult<()> {
    if let RqCorr::Reader {
        corr,
        idx,
        m,
        n_idx,
        np,
    } = corr
    {
        let xs = gpu.zeros_owned(&[*np], DType::F32)?;
        gpu.rq_gather_f32(src_normed, idx, &xs, *n_idx, *np)?;
        let tmp = gpu.zeros_owned(&[*m], DType::F32)?;
        gpu.gemv_f32(corr, &xs, &tmp)?;
        gpu.add_inplace_f32(out, &tmp)?;
    }
    // `xs`/`tmp` (RAII `OwnedTensor`) dropped at the block end; drain the pool.
    gpu.reclaim_pending();
    Ok(())
}

/// Apply all residual-reader corrections for a projection group sharing one
/// normed input. Computes the original-frame `rmsnorm(x)` gather-source ONCE
/// (only when at least one site has a correction), then `out += corr · src[S]`
/// per site. No-op for models without sidecars.
pub(crate) fn rq_apply_readers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    layer_idx: usize,
    norm_weight: &GpuTensor,
    x: &GpuTensor,
    eps: f32,
    dim: usize,
    sites: &[(RqProj, &GpuTensor)],
) -> HipResult<()> {
    let li = layer_idx as u32;
    if !sites
        .iter()
        .any(|(p, _)| weights.rq_corrections.contains_key(&(li, *p)))
    {
        return Ok(());
    }
    let src = gpu.zeros_owned(&[dim], DType::F32)?;
    gpu.rmsnorm_f32(x, norm_weight, &src, eps)?;
    for (p, out) in sites {
        if let Some(c) = weights.rq_corrections.get(&(li, *p)) {
            rq_apply_reader(gpu, c, &src, out)?;
        }
    }
    drop(src); // enqueue the RAII `OwnedTensor` before draining the pool below.
    gpu.reclaim_pending();
    Ok(())
}

/// Apply a residual-writer correction in place: `c = corr · input;
/// out_resid[S[j]] += c[j]`. `input` is the writer's full-width input activation.
pub(crate) fn rq_apply_writer(
    gpu: &mut Gpu,
    corr: &RqCorr,
    input: &GpuTensor,
    out_resid: &GpuTensor,
) -> HipResult<()> {
    if let RqCorr::Writer {
        corr,
        idx,
        n_s,
        k: _,
    } = corr
    {
        let c = gpu.zeros_owned(&[*n_s], DType::F32)?;
        gpu.gemv_f32(corr, input, &c)?;
        gpu.rq_scatter_add_f32(out_resid, idx, &c, *n_s)?;
    }
    // `c` (RAII `OwnedTensor`) dropped at the block end; drain the pool.
    gpu.reclaim_pending();
    Ok(())
}

/// Build the calibration capture map: device-buffer-pointer → canonical .hfq
/// tensor name (matching the `.calib.hfq` Hessian/imatrix key convention,
/// `model.language_model.layers.{i}.{role}`, no `.weight` suffix), for every
/// residual-linear weight. The forward's gemv arms resolve their weight buffer
/// pointer through this map to attribute captured activations. No loader change —
/// walks the typed `Qwen35Weights`. Set `gpu.capture_names` to this before arming
/// `gpu.active_capture`.
/// Map the MoE FFN weight buffers → canonical names. The router + shared
/// expert are dense (full Hessian); the routed experts are named only when
/// resident (`experts` populated) and the names contain `.experts.` so the
/// collector keeps them imatrix-only (full per-expert Hessians don't fit).
/// Names mirror the checkpoint tensor names so the quantizer can match
/// imatrix → weight.
fn put_moe_ffn(put: &mut impl FnMut(&WeightTensor, String), p: &str, ffn: &MoeFfnWeights) {
    put(&ffn.router, format!("{p}.mlp.gate"));
    put(
        &ffn.shared_expert.gate,
        format!("{p}.mlp.shared_expert.gate_proj"),
    );
    put(
        &ffn.shared_expert.up,
        format!("{p}.mlp.shared_expert.up_proj"),
    );
    put(
        &ffn.shared_expert.down,
        format!("{p}.mlp.shared_expert.down_proj"),
    );
    for (x, e) in ffn.experts.iter().enumerate() {
        put(&e.gate_up, format!("{p}.mlp.experts.{x}.gate_up_proj"));
        put(&e.down, format!("{p}.mlp.experts.{x}.down_proj"));
    }
}

pub fn build_capture_names(weights: &Qwen35Weights) -> std::collections::HashMap<usize, String> {
    let mut m = std::collections::HashMap::new();
    let mut put = |wt: &WeightTensor, name: String| {
        m.insert(wt.buf.buf.as_ptr() as usize, name);
    };
    for (i, layer) in weights.layers.iter().enumerate() {
        let p = format!("model.language_model.layers.{i}");
        match layer {
            LayerWeights::DeltaNet(l) => {
                put(&l.wqkv, format!("{p}.linear_attn.in_proj_qkv"));
                put(&l.wz, format!("{p}.linear_attn.in_proj_z"));
                put(&l.w_alpha, format!("{p}.linear_attn.in_proj_a"));
                put(&l.w_beta, format!("{p}.linear_attn.in_proj_b"));
                put(&l.wo, format!("{p}.linear_attn.out_proj"));
                put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                put(&l.w_up, format!("{p}.mlp.up_proj"));
                put(&l.w_down, format!("{p}.mlp.down_proj"));
            }
            LayerWeights::FullAttn(l) => {
                put(&l.wq, format!("{p}.self_attn.q_proj"));
                put(&l.wk, format!("{p}.self_attn.k_proj"));
                put(&l.wv, format!("{p}.self_attn.v_proj"));
                put(&l.wo, format!("{p}.self_attn.o_proj"));
                put(&l.w_gate, format!("{p}.mlp.gate_proj"));
                put(&l.w_up, format!("{p}.mlp.up_proj"));
                put(&l.w_down, format!("{p}.mlp.down_proj"));
            }
            // MoE variants: the dense attention projections + router + shared
            // expert flow through the same gemm chokepoint as the dense layers,
            // so they get full Hessians. The routed experts go through the
            // indexed-MoE GEMV kernels and are imatrix-only (full per-expert
            // Hessians don't fit — see CalibCollector). Routed-expert names are
            // only emitted when the experts are resident (`experts` populated);
            // in paged mode the buffers are owned by the WeightPager and the
            // ptrs are patched per-token, so capture-by-buf-ptr can't key them.
            LayerWeights::DeltaNetMoe(l) => {
                put(&l.wqkv, format!("{p}.linear_attn.in_proj_qkv"));
                put(&l.wz, format!("{p}.linear_attn.in_proj_z"));
                put(&l.w_alpha, format!("{p}.linear_attn.in_proj_a"));
                put(&l.w_beta, format!("{p}.linear_attn.in_proj_b"));
                put(&l.wo, format!("{p}.linear_attn.out_proj"));
                put_moe_ffn(&mut put, &p, &l.ffn);
            }
            LayerWeights::FullAttnMoe(l) => {
                put(&l.wq, format!("{p}.self_attn.q_proj"));
                put(&l.wk, format!("{p}.self_attn.k_proj"));
                put(&l.wv, format!("{p}.self_attn.v_proj"));
                put(&l.wo, format!("{p}.self_attn.o_proj"));
                put_moe_ffn(&mut put, &p, &l.ffn);
            }
        }
    }
    m
}

/// Options for [`collect_calibration_artifacts`].
pub struct CalibOpts {
    /// Capture the lm-head top-K logits + logZ per position (KLDREF reference).
    pub kldref: bool,
    pub kldref_topk: usize,
}

impl Default for CalibOpts {
    fn default() -> Self {
        Self {
            kldref: false,
            kldref_topk: 64,
        }
    }
}

/// Summary of a calibration pass after the `.calib.hfq` has been streamed to
/// disk (re-exported from the shared driver). The tensors themselves are NOT
/// returned — they are written one at a time (see
/// [`CalibCollector::write_streaming`]) so a 9B's ~32 GB of Hessians never sits
/// in host RAM at once.
pub use hipfire_runtime::calibration::CalibSummary;

/// FullAttention layer indices, in order. (LinearAttention/DeltaNet layers are
/// SSM and do not populate the KV cache's `k_gpu`, so PFlash scoring — and the
/// drafter teacher — only ever use FullAttention layers.)
pub fn full_attention_layers(config: &Qwen35Config) -> Vec<usize> {
    config
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(i, _)| i)
        .collect()
}

/// PFlash drafter TEACHER capture (teacher/student split — see
/// docs/plans/2026-06-19-training-via-daemon-forward.md). Forward `tokens`
/// through the resident Qwen3.5 target (FP32 KV + FP32 DeltaNet state) and return
/// per-block cosine-K scores `score(b) = cos(block_mean_K, last_K)` at each
/// requested FullAttention layer — the exact signal `pflash_drafter_train`
/// distils into a tiny custom drafter, but from the REAL target instead of a
/// loadable stand-in. Reuses the per-token `forward_scratch` path (like
/// `collect_calibration_artifacts`) + the fp32 `pflash_score_f32` kernel.
pub fn capture_pflash_block_scores(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    block_size: usize,
    layers: &[usize],
) -> HipResult<Vec<Vec<f32>>> {
    let n_tok = tokens.len();
    assert!(n_tok > 0, "capture_pflash_block_scores: empty tokens");
    assert!(
        block_size > 0,
        "capture_pflash_block_scores: block_size must be > 0"
    );
    for &l in layers {
        assert!(
            config.layer_types.get(l) == Some(&LayerType::FullAttention),
            "capture_pflash_block_scores: layer {l} is not FullAttention (no K in cache)"
        );
    }

    let mut kv = kv::KvCache::new_gpu(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        n_tok + 16,
    )?;
    let mut dn = DeltaNetState::new(gpu, config)?;
    let scratch = Qwen35Scratch::new(gpu, config, 64)?;
    for (pos, &tok) in tokens.iter().enumerate() {
        forward_scratch(gpu, weights, config, tok, pos, &mut kv, &mut dn, &scratch)?;
    }

    let kv_dim = config.n_kv_heads * config.head_dim;
    let n_blocks = n_tok.div_ceil(block_size);
    let mut out = Vec::with_capacity(layers.len());
    for &layer in layers {
        let scores = gpu.alloc_owned(&[n_blocks], DType::F32)?;
        gpu.pflash_score_f32_fwd(
            &kv.k_gpu[layer],
            &scores,
            n_tok,
            kv_dim,
            block_size,
            n_blocks,
            n_tok - 1,
        )?;
        let row = gpu.download_f32(&scores)?;
        drop(scores); // enqueue this iteration's RAII scratch, then drain.
        gpu.reclaim_pending();
        out.push(row);
    }
    Ok(out)
}

/// Write the target's token embedding as an fp32 sidecar for the PFlash drafter,
/// which shares it read-only (teacher/student split). Format: magic `QEMB`, u32
/// vocab, u32 dim (little-endian), then vocab*dim little-endian f32 rows. Only F32
/// embeddings are supported (the bf16/unquantized teacher path uploads token_embd
/// as F32); quantized embeddings would need a per-format dequant first.
pub fn dump_embed_fp32(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    path: &std::path::Path,
) -> HipResult<(usize, usize)> {
    use std::io::Write;
    if !matches!(weights.embd_format, EmbeddingFormat::F32) {
        return Err(hipfire_rdna::HipError {
            code: u32::MAX,
            message: format!(
                "dump_embed_fp32: embedding is {:?}, only F32 supported (load the bf16/unquantized target)",
                weights.embd_format
            ),
        });
    }
    let (vocab, dim) = (config.vocab_size, config.dim);
    let data = gpu.download_f32(&weights.token_embd)?;
    assert_eq!(
        data.len(),
        vocab * dim,
        "dump_embed_fp32: embed size mismatch"
    );
    let io_err = |e: std::io::Error| hipfire_rdna::HipError {
        code: u32::MAX,
        message: format!("dump_embed_fp32 io: {e}"),
    };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io_err)?);
    f.write_all(b"QEMB").map_err(io_err)?;
    f.write_all(&(vocab as u32).to_le_bytes()).map_err(io_err)?;
    f.write_all(&(dim as u32).to_le_bytes()).map_err(io_err)?;
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes).map_err(io_err)?;
    f.flush().map_err(io_err)?;
    Ok((vocab, dim))
}

/// Run the resident model over `chunk` (per-token decode, fresh KV + DeltaNet
/// state) and invoke `at_scored(j, full_logits, actual_next)` for each scored
/// position `j` in `[scoring_start, n_ctx-1)`. The single forward path that both
/// reference build and candidate scoring funnel through.
fn forward_chunk_scored(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    chunk: &[u32],
    scoring_start: usize,
    mut at_scored: impl FnMut(usize, &[f32], usize),
) -> HipResult<()> {
    let n = chunk.len();
    let mut kv = kv::KvCache::new_gpu(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        n + 16,
    )?;
    let mut dn = DeltaNetState::new(gpu, config)?;
    let scratch = Qwen35Scratch::new(gpu, config, 64)?;
    for pos in 0..n.saturating_sub(1) {
        forward_scratch(
            gpu, weights, config, chunk[pos], pos, &mut kv, &mut dn, &scratch,
        )?;
        if pos >= scoring_start {
            let lg = gpu.download_f32(&scratch.logits)?;
            at_scored(pos - scoring_start, &lg, chunk[pos + 1] as usize);
        }
    }
    Ok(())
}

/// Adapter making a resident Qwen3.5 (the loose `weights`+`config` slots, which
/// don't implement [`hipfire_runtime::arch::SimpleAr`]) KLD-scorable through the
/// generic [`hipfire_runtime::kld_eval`] driver — reusing the per-token
/// `forward_chunk_scored` path above. Every other arch gets the blanket impl.
pub struct Qwen35KldForward<'a> {
    pub weights: &'a Qwen35Weights,
    pub config: &'a Qwen35Config,
}

impl hipfire_runtime::kld_eval::ChunkScoredForward for Qwen35KldForward<'_> {
    fn forward_chunk_scored(
        &mut self,
        gpu: &mut Gpu,
        chunk: &[u32],
        scoring_start: usize,
        at_scored: &mut dyn FnMut(usize, &[f32], usize),
    ) -> Result<(), String> {
        forward_chunk_scored(
            gpu,
            self.weights,
            self.config,
            chunk,
            scoring_start,
            |j, lg, next| at_scored(j, lg, next),
        )
        .map_err(|e| format!("qwen35 kld forward: {e:?}"))
    }

    fn kld_vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

/// Thin `&weights`/`&config` adapter so the daemon's loose Qwen3.5 resident slots
/// satisfy the calibration seam ([`hipfire_runtime::calibration::CalibratableBackend`])
/// without a bundled backend type — the calibration analogue of
/// [`Qwen35KldForward`].
pub struct Qwen35CalibBackend<'a> {
    pub weights: &'a Qwen35Weights,
    pub config: &'a Qwen35Config,
}

impl hipfire_runtime::calibration::CalibratableBackend for Qwen35CalibBackend<'_> {
    fn collect_calibration(
        &self,
        gpu: &mut Gpu,
        _tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
        tokens: &[u32],
        kldref: bool,
        output: &std::path::Path,
        provenance: &[(&str, serde_json::Value)],
    ) -> Result<CalibSummary, String> {
        let opts = CalibOpts {
            kldref,
            kldref_topk: 64,
        };
        collect_calibration_artifacts(
            gpu,
            self.weights,
            self.config,
            tokens,
            &opts,
            output,
            provenance,
        )
        .map_err(|e| e.to_string())
    }
}

/// Single-load calibration driver: arm the [`CalibCollector`] on the resident
/// weights, run the engine forward over `tokens` (capturing per-tensor Hessian +
/// imatrix, the MoE router histogram for MoE models, and optionally KLDREF), and
/// assemble the HFQ artifact tensors + metadata. Reused by the `collect_artifacts`
/// CLI and (Phase 5) the daemon `Collect` op. Uses f32 KV + FP32 DeltaNet state
/// for faithful (lossless) activations. Restores `gpu.active_capture`/`capture_names`
/// to empty on return.
pub fn collect_calibration_artifacts(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    tokens: &[u32],
    opts: &CalibOpts,
    output: &std::path::Path,
    provenance: &[(&str, serde_json::Value)],
) -> HipResult<CalibSummary> {
    use hipfire_runtime::calibration::{collect, logsumexp, topk_logits, CalibForward};
    use hipfire_runtime::hfq::HfqMemTensor;

    let is_moe = config.num_experts > 0;
    // Routed MoE experts are imatrix-only: a full per-expert Hessian
    // (num_experts × n_layers × [K,K]) does not fit (~196 GB for A3B), but the
    // imatrix is ~100 MB and is the importance signal quant needs. Dense
    // projections (attention + router + shared expert) keep full Hessians. The
    // shared driver owns the collector + streaming; the closure runs the engine
    // forward and returns the qwen-specific extras (router histogram + KLDREF).
    collect(
        gpu,
        0,
        build_capture_names(weights),
        vec![".experts.".to_string()],
        output,
        provenance,
        |gpu| {
            if is_moe {
                reset_moe_router_histogram(config.num_experts, config.num_experts_per_tok);
            }
            let n_tok = tokens.len();
            let mut kv = kv::KvCache::new_gpu(
                gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                n_tok + 16,
            )
            .map_err(|e| format!("qwen35 calib kv: {e}"))?;
            let mut dn =
                DeltaNetState::new(gpu, config).map_err(|e| format!("qwen35 calib dn: {e}"))?;
            let scratch = Qwen35Scratch::new(gpu, config, 64)
                .map_err(|e| format!("qwen35 calib scratch: {e}"))?;

            let mut kldref: Vec<(f32, Vec<(u32, f32)>)> = Vec::new();
            for (pos, &tok) in tokens.iter().enumerate() {
                forward_scratch(gpu, weights, config, tok, pos, &mut kv, &mut dn, &scratch)
                    .map_err(|e| format!("qwen35 calib forward: {e}"))?;
                if opts.kldref {
                    let lg = gpu
                        .download_f32(&scratch.logits)
                        .map_err(|e| format!("qwen35 calib logits: {e}"))?;
                    kldref.push((logsumexp(&lg), topk_logits(&lg, opts.kldref_topk)));
                }
            }

            let f32_bytes = |v: &[f32]| -> Vec<u8> {
                let mut b = Vec::with_capacity(v.len() * 4);
                for &x in v {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                b
            };

            let mut extra_meta: Vec<(String, serde_json::Value)> = Vec::new();
            let mut artifacts = vec![serde_json::json!("hessian"), serde_json::json!("imatrix")];

            // MoE router histogram (summed co-occurrence = scheduler-affinity signal).
            if is_moe {
                if let Some(h) = take_moe_router_histogram() {
                    let mut cooc: std::collections::HashMap<u64, u64> =
                        std::collections::HashMap::new();
                    for l in &h.per_layer {
                        for (&k, &v) in &l.cooccurrence {
                            *cooc.entry(k).or_insert(0) += v;
                        }
                    }
                    let mut pairs: Vec<(u64, u64)> = cooc.into_iter().collect();
                    pairs.sort_by_key(|pair| std::cmp::Reverse(pair.1));
                    pairs.truncate(64);
                    let ne = h.num_experts as u64;
                    let cooc_json: Vec<serde_json::Value> = pairs
                        .iter()
                        .map(|(key, cnt)| serde_json::json!([key / ne, key % ne, cnt]))
                        .collect();
                    extra_meta.push((
                        "moe_router_histogram".to_string(),
                        serde_json::json!({
                            "num_experts": h.num_experts,
                            "k_top": h.k_top,
                            "routed_tokens": h.routed_tokens,
                            "routed_slots": h.routed_slots,
                            "top1_histogram": h.top1_histogram,
                            "topk_histogram": h.topk_histogram,
                            "per_layer_topk": h.per_layer.iter().map(|l| serde_json::json!(l.topk_histogram)).collect::<Vec<_>>(),
                            "top_cooccurrence": cooc_json,
                        }),
                    ));
                    artifacts.push(serde_json::json!("moe_router_histogram"));
                }
            }

            // KLDREF tensors — small, already in host RAM; passed as `extra` to the
            // streaming writer (the big Hessians stream straight from GPU).
            let mut extra_tensors: Vec<HfqMemTensor> = Vec::new();
            if !kldref.is_empty() {
                let np = kldref.len();
                let kk = kldref[0].1.len();
                let (mut idx_v, mut lg_v, mut lz_v) = (Vec::new(), Vec::new(), Vec::new());
                for (logz, tk) in &kldref {
                    lz_v.push(*logz);
                    for j in 0..kk {
                        let (i, l) = tk.get(j).copied().unwrap_or((0, f32::NEG_INFINITY));
                        idx_v.push(i as f32);
                        lg_v.push(l);
                    }
                }
                for (nm, shape, data) in [
                    ("lm_head.kldref_idx", vec![np as u32, kk as u32], idx_v),
                    ("lm_head.kldref_logit", vec![np as u32, kk as u32], lg_v),
                    ("lm_head.kldref_logz", vec![np as u32], lz_v),
                ] {
                    extra_tensors.push(HfqMemTensor {
                        name: nm.to_string(),
                        quant_type: 2,
                        shape,
                        group_size: 0,
                        data: f32_bytes(&data),
                    });
                }
                extra_meta.push((
                    "kldref".to_string(),
                    serde_json::json!({ "n_positions": np, "top_k": kk }),
                ));
                artifacts.push(serde_json::json!("kldref"));
            }
            extra_meta.push(("artifacts".to_string(), serde_json::Value::Array(artifacts)));

            Ok(CalibForward {
                extra_tensors,
                extra_meta,
            })
        },
    )
    .map_err(|e| HipError::new(0, &e))
}

pub fn load_weights(
    hfq: &mut HfqFile,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> HipResult<Qwen35Weights> {
    validate_ffn_bf16_hfq_load(config)?;
    let load_t0 = std::time::Instant::now();
    let file_payload_bytes: usize = hfq.tensors().iter().map(|t| t.data_size).sum();
    let mut loaded_bytes = 0usize;
    eprintln!(
        "  loading weights: {} tensors, {:.2} GiB HFQ payload",
        hfq.tensors().len(),
        gib(file_payload_bytes),
    );
    // Drop the mmap on unix to avoid double-buffering on UMA systems.
    // All tensor data reads go through pread + fadvise_dontneed, which
    // doesn't require the mmap. On discrete-GPU systems this is harmless
    // (pread is slightly slower than mmap but avoids page cache buildup).
    #[cfg(unix)]
    hfq.drop_mmap();

    let slab_index = if gpu_slab_load_enabled(gpu) {
        if std::env::var("HIPFIRE_GPU_SLAB_LOAD").ok().is_none() {
            eprintln!("  GPU slab load: auto-enabled for integrated/UMA GPU");
        }
        Some(load_gpu_slabs(hfq, gpu)?)
    } else {
        None
    };
    let slabs = slab_index.as_ref();
    if let Some(idx) = slabs {
        loaded_bytes = loaded_bytes.saturating_add(idx.storage.bytes);
    }

    eprintln!("  loading token_embd...");
    if config.is_vl_text {
        eprintln!(
            "  qwen3.5-vl text wrapper: mrope_interleaved={} mrope_section={:?}",
            config.mrope_interleaved, config.mrope_section
        );
    }
    let embd_qt = qwen35_tensor_name_candidates("embed_tokens.weight")
        .into_iter()
        .find_map(|candidate| {
            hfq.tensors()
                .iter()
                .find(|t| t.name == candidate)
                .map(|t| t.quant_type)
        })
        .expect("embed_tokens not found");
    let (token_embd, embd_fmt) =
        if let Some((qt, tensor)) = load_gpu_tensor_from_slabs(slabs, "embed_tokens.weight") {
            match qt {
                6 => (tensor, EmbeddingFormat::HFQ4G256),
                7 => (tensor, EmbeddingFormat::HFQ4G128),
                3 => (tensor, EmbeddingFormat::Q8_0),
                _ => {
                    let (embd_meta, embd_data) = qwen35_tensor_data_vec(hfq, "embed_tokens.weight")
                        .expect("embed_tokens not found");
                    loaded_bytes += embd_data.len();
                    let f32_data =
                        hfq_plain_tensor_as_f32(embd_meta, &embd_data, "embed_tokens.weight");
                    (
                        gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?,
                        EmbeddingFormat::F32,
                    )
                }
            }
        } else if embd_qt == 6 {
            let (_, embd_data) =
                qwen35_tensor_data_vec(hfq, "embed_tokens.weight").expect("embed_tokens not found");
            loaded_bytes += embd_data.len();
            eprintln!("    (HFQ4-G256 raw, {} MB)", embd_data.len() / 1_000_000);
            (
                gpu.upload_raw(&embd_data, &[embd_data.len()])?,
                EmbeddingFormat::HFQ4G256,
            )
        } else if embd_qt == 7 {
            let (_, embd_data) =
                qwen35_tensor_data_vec(hfq, "embed_tokens.weight").expect("embed_tokens not found");
            loaded_bytes += embd_data.len();
            eprintln!("    (HFQ4-G128 raw, {} MB)", embd_data.len() / 1_000_000);
            (
                gpu.upload_raw(&embd_data, &[embd_data.len()])?,
                EmbeddingFormat::HFQ4G128,
            )
        } else if embd_qt == 3 {
            let (_, embd_data) =
                qwen35_tensor_data_vec(hfq, "embed_tokens.weight").expect("embed_tokens not found");
            loaded_bytes += embd_data.len();
            eprintln!("    (Q8_0 raw, {} MB)", embd_data.len() / 1_000_000);
            (
                gpu.upload_raw(&embd_data, &[embd_data.len()])?,
                EmbeddingFormat::Q8_0,
            )
        } else {
            let (embd_meta, embd_data) =
                qwen35_tensor_data_vec(hfq, "embed_tokens.weight").expect("embed_tokens not found");
            loaded_bytes += embd_data.len();
            let f32_data = hfq_plain_tensor_as_f32(embd_meta, &embd_data, "embed_tokens.weight");
            (
                gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?,
                EmbeddingFormat::F32,
            )
        };

    eprintln!("  loading output_norm...");
    // GemmaRMSNorm storage convention is uniform across the Qwen3.5+ family:
    // safetensors store raw `w` (init from zero, can train to any magnitude),
    // engines apply `(1 + w)` at runtime. Hipfire's `load_norm_weight` bakes
    // `+= 1.0` at load time so the kernel can stay plain `x * w * rms` —
    // mathematically equivalent to vLLM's runtime `weight + 1.0` and
    // llama.cpp's GGUF-conversion-time bake. See
    // docs/plans/qwen35-moe-rmsnorm-fix.md for the concrete arithmetic trace.
    //
    // The earlier `if config.num_experts > 0` fork (commit 1e01c0b) skipped
    // the `+= 1.0` bake on MoE final norms to silence a `<think>` infinite-
    // spiral on Qwen3.6-A3B reasoning prompts. That under-scaled the MoE
    // final norm by ~38% (e.g. on 3.6-A3B: stored mean +1.63 → effective
    // scale 1.63 instead of the correct 2.63 = 1 + 1.63 that vLLM/llama.cpp
    // produce). It was a magnitude mask, not a fix — the spiral's real root
    // cause was the daemon's `repeat_penalty` default of 1.3 over a 128-token
    // window penalizing legitimately repeated chain-of-thought formatting
    // tokens, which fell off the model's well-trained reasoning path into a
    // self-doubt / number-hallucination attractor (fixed in commit 9b4ab74a:
    // default repeat_penalty 1.3 → 1.0). Bench A/B on Qwen3.6-35B-A3B MQ4
    // confirms the spiral is dissolved with the new default; the prior
    // `HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1` env-var escape hatch was removed
    // together with this fork.
    let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?;

    // Try separate lm_head first (untied embeddings, e.g. 9B), fall back to tied embed_tokens.
    let mut output = if let Some((matched_name, mut wt)) =
        load_weight_tensor_from_slabs(slabs, "lm_head.weight", config.vocab_size, config.dim)
    {
        eprintln!(
            "  loading output (separate lm_head, slab-backed qt={:?})...",
            wt.gpu_dtype
        );
        if wt.gpu_dtype.supports_awq_sidecar() {
            wt.awq_scale = load_awq_scale_for(hfq, gpu, &matched_name, config.dim)
                .or_else(|| load_awq_scale_for(hfq, gpu, "lm_head.weight", config.dim));
        }
        wt
    } else if let Some((lm_info, lm_data)) = qwen35_tensor_data_vec(hfq, "lm_head.weight") {
        eprintln!(
            "  loading output (separate lm_head, qt={})...",
            lm_info.quant_type
        );
        loaded_bytes += lm_data.len();
        load_weight_tensor_raw(
            gpu,
            lm_info.quant_type,
            &lm_data,
            config.vocab_size,
            config.dim,
        )?
    } else {
        eprintln!("  loading output (tied embeddings, qt={})...", embd_qt);
        if let Some((matched_name, mut wt)) = load_weight_tensor_from_slabs(
            slabs,
            "embed_tokens.weight",
            config.vocab_size,
            config.dim,
        ) {
            if wt.gpu_dtype.supports_awq_sidecar() {
                wt.awq_scale = load_awq_scale_for(hfq, gpu, &matched_name, config.dim)
                    .or_else(|| load_awq_scale_for(hfq, gpu, "embed_tokens.weight", config.dim));
            }
            wt
        } else {
            let (tied_info, tied_data) =
                qwen35_tensor_data_vec(hfq, "embed_tokens.weight").unwrap();
            loaded_bytes += tied_data.len();
            if embd_qt == 6 || embd_qt == 7 || embd_qt == 8 {
                let buf = gpu.upload_raw(&tied_data, &[tied_data.len()])?;
                let dtype = match embd_qt {
                    6 => DType::HFQ4G256,
                    7 => DType::HFQ4G128,
                    8 => DType::HFQ6G256,
                    _ => unreachable!(),
                };
                WeightTensor {
                    buf,
                    gpu_dtype: dtype,
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                }
            } else if embd_qt == 13 {
                let buf = gpu.upload_raw(&tied_data, &[tied_data.len()])?;
                WeightTensor {
                    buf,
                    gpu_dtype: DType::MQ4G256,
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                }
            } else if embd_qt == 14 {
                let buf = gpu.upload_raw(&tied_data, &[tied_data.len()])?;
                WeightTensor {
                    buf,
                    gpu_dtype: DType::MQ8G256,
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                }
            } else if embd_qt == 3 {
                let buf = gpu.upload_raw(&tied_data, &[tied_data.len()])?;
                WeightTensor {
                    buf,
                    gpu_dtype: DType::Q8_0,
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                }
            } else if embd_qt == 16 {
                load_bf16_matrix_weight(gpu, &tied_data, config.vocab_size, config.dim)?
            } else {
                let f32_data =
                    hfq_plain_tensor_as_f32(tied_info, &tied_data, "embed_tokens.weight");
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
                };
                let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
                WeightTensor {
                    buf,
                    gpu_dtype: DType::F32,
                    m: config.vocab_size,
                    k: config.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                }
            }
        }
    };
    // AWQ sidecar attachment for lm_head / tied embed_tokens. Safe now
    // that both decode (`weight_gemv` → `rotate_x_mq_for`) AND spec-
    // decode verify (`speculative.rs::rotate_x_mq_batched_for`) apply
    // the `x /= s` inverse when `output.awq_scale.is_some()`. Pre-fix,
    // attaching a sidecar here would have driven the 0.67 → 13.5 KLD
    // corruption documented at `docs/plans/awq_fix_claude.md` because
    // the spec-verify path used the non-AWQ `rotate_x_mq_batched`.
    // Try each plausible tensor name; `load_awq_scale_for` returns
    // None when no sidecar exists, so this is a no-op for current
    // pre-CUDA-pipeline files.
    if output.gpu_dtype.supports_awq_sidecar() {
        if output.awq_scale.is_none() {
            output.awq_scale = load_awq_scale_for(hfq, gpu, "lm_head.weight", config.dim)
                .or_else(|| {
                    load_awq_scale_for(hfq, gpu, "model.language_model.lm_head.weight", config.dim)
                })
                .or_else(|| {
                    load_awq_scale_for(
                        hfq,
                        gpu,
                        "model.language_model.embed_tokens.weight",
                        config.dim,
                    )
                });
        }
        eprintln!(
            "  lm_head AWQ sidecar: {}",
            if output.awq_scale.is_some() {
                "attached"
            } else {
                "absent (no-op)"
            }
        );
    }

    let is_moe = config.num_experts > 0;
    let mut pager = if config.paged_experts && is_moe {
        if hfq.modules().is_empty() {
            return Err(HipError::new(
                0,
                "HIPFIRE_QWEN35_PAGED_EXPERTS=1 requires an HFQM v2 module table; run qwen35_hfq_modules repack first",
            ));
        }
        let mut pager = hipfire_runtime::weight_pager::WeightPager::with_env_transport(
            hfq.path(),
            hipfire_runtime::weight_pager::PagerConfig {
                vram_budget_bytes: config.vram_budget_bytes,
                trace: matches!(
                    std::env::var("HIPFIRE_QWEN35_EXPERT_CACHE_TRACE")
                        .ok()
                        .as_deref(),
                    Some("1" | "true" | "on" | "yes")
                ),
            },
        )
        .map_err(|e| HipError::new(0, &format!("open expert module pager: {e}")))?;
        let registered = pager
            .register_expert_modules(
                hfq.modules()
                    .iter()
                    .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
                    .cloned(),
            )
            .map_err(|e| HipError::new(0, &format!("register expert modules: {e}")))?;
        eprintln!(
            "  paged experts enabled: registered {} routed expert modules, cache_budget={:.2} GiB",
            registered,
            gib(config.vram_budget_bytes as usize)
        );
        Some(pager)
    } else {
        None
    };
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!(
            "  loading layer {i}/{} ({:?}{})...",
            config.n_layers,
            config.layer_types[i],
            if is_moe { " + MoE" } else { "" }
        );
        hipfire_runtime::load_progress::report(i as u32 + 1, config.n_layers as u32, "weights");
        let p = format!("layers.{i}");
        // Track page range for this layer so we can MADV_DONTNEED after upload.
        let layer_page_start = hfq.layer_data_range(&p);

        match (config.layer_types[i], is_moe) {
            (LayerType::LinearAttention, false) => {
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                    + config.linear_num_value_heads * config.linear_value_head_dim;
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;

                layers.push(LayerWeights::DeltaNet(DeltaNetLayerWeights {
                    attn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wqkv: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_qkv.weight"),
                        qkv_dim,
                        config.dim,
                    )?,
                    wz: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_z.weight"),
                        d_inner,
                        config.dim,
                    )?,
                    w_alpha: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_a.weight"),
                        config.linear_num_value_heads,
                        config.dim,
                    )?,
                    w_beta: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_b.weight"),
                        config.linear_num_value_heads,
                        config.dim,
                    )?,
                    a_log: load_raw_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.A_log"),
                        config.linear_num_value_heads,
                    )?,
                    dt_bias: load_raw_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.dt_bias"),
                        config.linear_num_value_heads,
                    )?,
                    conv_weight: load_any_as_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.conv1d.weight"),
                        qkv_dim * config.conv_kernel_dim,
                    )?, // flatten [channels, 1, kernel] → [channels * kernel]
                    norm_weight: load_any_as_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.norm.weight"),
                        config.linear_value_head_dim,
                    )?,
                    wo: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.out_proj.weight"),
                        config.dim,
                        d_inner,
                    )?,
                    ffn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    w_gate: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.mlp.gate_proj.weight"),
                        config.hidden_dim,
                        config.dim,
                    )?,
                    w_up: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.mlp.up_proj.weight"),
                        config.hidden_dim,
                        config.dim,
                    )?,
                    w_down: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.mlp.down_proj.weight"),
                        config.dim,
                        config.hidden_dim,
                    )?,
                    bf16_down_shadow: load_bf16_down_shadow_for(
                        hfq,
                        &format!("{p}.mlp.down_proj.weight"),
                        i,
                        config.dim,
                        config.hidden_dim,
                    )?,
                }));
            }
            (LayerType::FullAttention, false) => {
                let q_out_dim = qwen35_fa_q_out_dim(config);
                let kv_dim = config.n_kv_heads * config.head_dim;

                layers.push(LayerWeights::FullAttn(FullAttnLayerWeights {
                    attn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wq: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.q_proj.weight"),
                        q_out_dim,
                        config.dim,
                    )?,
                    wk: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.k_proj.weight"),
                        kv_dim,
                        config.dim,
                    )?,
                    wv: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.v_proj.weight"),
                        kv_dim,
                        config.dim,
                    )?,
                    wo: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.o_proj.weight"),
                        config.dim,
                        config.n_heads * config.head_dim,
                    )?,
                    q_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.q_norm.weight"),
                        &[config.head_dim],
                    )?,
                    k_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.k_norm.weight"),
                        &[config.head_dim],
                    )?,
                    ffn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    w_gate: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.mlp.gate_proj.weight"),
                        config.hidden_dim,
                        config.dim,
                    )?,
                    w_up: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.mlp.up_proj.weight"),
                        config.hidden_dim,
                        config.dim,
                    )?,
                    w_down: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.mlp.down_proj.weight"),
                        config.dim,
                        config.hidden_dim,
                    )?,
                    bf16_down_shadow: load_bf16_down_shadow_for(
                        hfq,
                        &format!("{p}.mlp.down_proj.weight"),
                        i,
                        config.dim,
                        config.hidden_dim,
                    )?,
                }));
            }
            (LayerType::LinearAttention, true) => {
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                    + config.linear_num_value_heads * config.linear_value_head_dim;
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;

                layers.push(LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                    attn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wqkv: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_qkv.weight"),
                        qkv_dim,
                        config.dim,
                    )?,
                    wz: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_z.weight"),
                        d_inner,
                        config.dim,
                    )?,
                    w_alpha: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_a.weight"),
                        config.linear_num_value_heads,
                        config.dim,
                    )?,
                    w_beta: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.in_proj_b.weight"),
                        config.linear_num_value_heads,
                        config.dim,
                    )?,
                    a_log: load_raw_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.A_log"),
                        config.linear_num_value_heads,
                    )?,
                    dt_bias: load_raw_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.dt_bias"),
                        config.linear_num_value_heads,
                    )?,
                    conv_weight: load_any_as_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.conv1d.weight"),
                        qkv_dim * config.conv_kernel_dim,
                    )?,
                    norm_weight: load_any_as_f32(
                        hfq,
                        gpu,
                        &format!("{p}.linear_attn.norm.weight"),
                        config.linear_value_head_dim,
                    )?,
                    wo: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.linear_attn.out_proj.weight"),
                        config.dim,
                        d_inner,
                    )?,
                    ffn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    ffn: if config.paged_experts {
                        load_moe_ffn_paged(hfq, gpu, slabs, &p, config, i as u16)?
                    } else {
                        load_moe_ffn(hfq, gpu, slabs, &p, config, i as u16)?
                    },
                }));
            }
            (LayerType::FullAttention, true) => {
                let q_out_dim = qwen35_fa_q_out_dim(config);
                let kv_dim = config.n_kv_heads * config.head_dim;

                layers.push(LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                    attn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wq: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.q_proj.weight"),
                        q_out_dim,
                        config.dim,
                    )?,
                    wk: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.k_proj.weight"),
                        kv_dim,
                        config.dim,
                    )?,
                    wv: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.v_proj.weight"),
                        kv_dim,
                        config.dim,
                    )?,
                    wo: load_weight_tensor(
                        hfq,
                        gpu,
                        slabs,
                        &format!("{p}.self_attn.o_proj.weight"),
                        config.dim,
                        config.n_heads * config.head_dim,
                    )?,
                    q_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.q_norm.weight"),
                        &[config.head_dim],
                    )?,
                    k_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.k_norm.weight"),
                        &[config.head_dim],
                    )?,
                    ffn_norm: load_norm_weight(
                        hfq,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    ffn: if config.paged_experts {
                        load_moe_ffn_paged(hfq, gpu, slabs, &p, config, i as u16)?
                    } else {
                        load_moe_ffn(hfq, gpu, slabs, &p, config, i as u16)?
                    },
                }));
            }
        }
        // Drop mmap page cache for this layer (supplements pread-based loading).
        if slabs.is_none() {
            loaded_bytes = loaded_bytes.saturating_add(qwen35_layer_payload_bytes(
                hfq,
                &p,
                config.paged_experts,
            ));
            if let Some((start, end)) = layer_page_start {
                hfq.drop_pages_range(start, end - start);
            }
        }
        let elapsed = load_t0.elapsed().as_secs_f64();
        eprintln!(
            "  load progress: layer {}/{} elapsed={:.2}s throughput={:.2} GiB/s ({:.2}/{:.2} GiB)",
            i + 1,
            config.n_layers,
            elapsed,
            load_throughput_gibs(loaded_bytes, elapsed),
            gib(loaded_bytes),
            gib(file_payload_bytes),
        );
    }

    let elapsed = load_t0.elapsed().as_secs_f64();
    eprintln!(
        "  weights loaded: elapsed={:.2}s throughput={:.2} GiB/s payload={:.2} GiB streamed={:.2} GiB",
        elapsed,
        load_throughput_gibs(loaded_bytes, elapsed),
        gib(file_payload_bytes),
        gib(loaded_bytes),
    );

    let slab_storage = slab_index.map(|idx| idx.storage);
    let rq_corrections = load_rq_corrections(hfq, gpu)?;
    Ok(Qwen35Weights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
        slab_storage,
        rq_corrections,
        pager: pager.take().map(RefCell::new),
    })
}

// ─── ParoQuant safetensors loading ──────────────────────────────────────────

/// Resolve the text-tower prefix this PARO checkpoint uses.
///   - `"model.language_model"` for Qwen3.5 / 3.6 (multimodal layout — even
///     the text-only A3B inherits the prefix from the multimodal config).
///   - `"model"` for Qwen3 v1 / pure-text-LLM PARO checkpoints (e.g.
///     z-lab/Qwen3-0.6B-PARO).
/// Probed via `embed_tokens.weight` which exists in both layouts. Returns an
/// `Err` if neither form is present — caller is exercising a non-Qwen3 family.
fn paro_text_prefix(source: &dyn ModelSource) -> HipResult<&'static str> {
    if source
        .tensor_info("model.language_model.embed_tokens.weight")
        .is_some()
    {
        Ok("model.language_model")
    } else if source.tensor_info("model.embed_tokens.weight").is_some() {
        Ok("model")
    } else {
        Err(HipError::new(0, "ParoQuant: embed_tokens.weight not found under either model.language_model. or model. layout"))
    }
}

pub(crate) fn paro_load_wt(
    source: &dyn ModelSource,
    gpu: &Gpu,
    prefix: &str,
    m: usize,
    k: usize,
    gs: u32,
    kr: u8,
) -> HipResult<WeightTensor> {
    let mp = paro_text_prefix(source)?;
    let fp = format!("{mp}.{prefix}");
    if source.tensor_info(&format!("{fp}.qweight")).is_some() {
        return load_paroquant_weight(source, gpu, &fp, m, k, gs, kr);
    }
    if crate::paro_la_gates_codec::should_quantize_la_gate(prefix, gpu.arch.as_str()) {
        return load_fp16_then_encode_mq4g128(source, gpu, &format!("{fp}.weight"), m, k);
    }
    load_fp16_weight_from_source(source, gpu, &format!("{fp}.weight"), m, k)
}

fn paro_load_norm(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> HipResult<GpuTensor> {
    let mp = paro_text_prefix(source)?;
    let full = format!("{mp}.{name}");
    let (info, data) = source
        .tensor_data(&full)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {full}")))?;
    let mut v: Vec<f32> = if info.dtype == "F16" {
        data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    for x in &mut v {
        *x += 1.0;
    }
    gpu.upload_f32(&v, shape)
}

fn paro_load_f32(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> HipResult<GpuTensor> {
    let mp = paro_text_prefix(source)?;
    let full = format!("{mp}.{name}");
    let (info, data) = source
        .tensor_data(&full)
        .ok_or_else(|| HipError::new(0, &format!("PARO tensor not found: {full}")))?;
    let v: Vec<f32> = if info.dtype == "F16" {
        data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    gpu.upload_f32(&v, &[n])
}

pub fn load_weights_paroquant(
    source: &dyn ModelSource,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> HipResult<Qwen35Weights> {
    reject_ffn_bf16_non_hfq_load("ParoQuant/safetensors load path")?;
    let qc = source
        .quant_config()
        .ok_or_else(|| HipError::new(0, "ParoQuant model must have quantization_config"))?;
    let gs = qc.group_size;
    let kr = qc.krot;

    let mp = paro_text_prefix(source)?;
    eprintln!("  loading token_embd (ParoQuant)...");
    let embd_name = format!("{mp}.embed_tokens.weight");
    let (_, embd_data) = source.tensor_data(&embd_name).ok_or_else(|| {
        HipError::new(
            0,
            &format!("PARO tensor not found: embed_tokens not found at {embd_name}"),
        )
    })?;
    let f32_embd: Vec<f32> = embd_data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let token_embd = gpu.upload_f32(&f32_embd, &[config.vocab_size, config.dim])?;
    let embd_fmt = EmbeddingFormat::F32;

    eprintln!("  loading output_norm...");
    let output_norm = paro_load_norm(source, gpu, "norm.weight", &[config.dim])?;

    // Prefer separate lm_head when checkpoint provides one (tie_word_embeddings:false);
    // fall back to embed_tokens for tied checkpoints. shisa-ai/Qwen3.6-35B-A3B-PARO
    // ships a distinct lm_head.weight; tying would project logits against the wrong
    // matrix and produce coherent-but-semantically-wrong output (decoded as token 118401
    // "出错" on the smoke prompt before this fix).
    let lm_head_name = String::from("lm_head.weight");
    let (output_src_name, output_tied) = if source.tensor_data(&lm_head_name).is_some() {
        (lm_head_name, false)
    } else {
        (embd_name, true)
    };
    eprintln!(
        "  loading output ({})...",
        if output_tied {
            "tied embeddings"
        } else {
            "separate lm_head"
        }
    );
    let output = {
        let (_, td) = source.tensor_data(&output_src_name).ok_or_else(|| {
            HipError::new(
                0,
                &format!("PARO tensor not found: output projection tensor {output_src_name}"),
            )
        })?;
        let f: Vec<f32> = td
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) };
        let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
        WeightTensor {
            buf,
            gpu_dtype: DType::F32,
            m: config.vocab_size,
            k: config.dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }
    };

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        eprintln!(
            "  loading layer {i}/{} ({:?}, ParoQuant)...",
            config.n_layers, config.layer_types[i]
        );
        hipfire_runtime::load_progress::report(i as u32 + 1, config.n_layers as u32, "weights");
        let p = format!("layers.{i}");
        let is_moe = config.num_experts > 0;

        match (config.layer_types[i], is_moe) {
            (LayerType::LinearAttention, false) => {
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                    + config.linear_num_value_heads * config.linear_value_head_dim;
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
                layers.push(LayerWeights::DeltaNet(DeltaNetLayerWeights {
                    attn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wqkv: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_qkv"),
                        qkv_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wz: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_z"),
                        d_inner,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    w_alpha: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_a"),
                        config.linear_num_value_heads,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    w_beta: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_b"),
                        config.linear_num_value_heads,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    a_log: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.A_log"),
                        config.linear_num_value_heads,
                    )?,
                    dt_bias: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.dt_bias"),
                        config.linear_num_value_heads,
                    )?,
                    conv_weight: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.conv1d.weight"),
                        qkv_dim * config.conv_kernel_dim,
                    )?,
                    norm_weight: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.norm.weight"),
                        config.linear_value_head_dim,
                    )?,
                    wo: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.out_proj"),
                        config.dim,
                        d_inner,
                        gs,
                        kr,
                    )?,
                    ffn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    w_gate: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.mlp.gate_proj"),
                        config.hidden_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    w_up: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.mlp.up_proj"),
                        config.hidden_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    w_down: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.mlp.down_proj"),
                        config.dim,
                        config.hidden_dim,
                        gs,
                        kr,
                    )?,
                    bf16_down_shadow: None,
                }));
            }
            (LayerType::FullAttention, false) => {
                let q_out_dim = qwen35_fa_q_out_dim(config);
                let kv_dim = config.n_kv_heads * config.head_dim;
                layers.push(LayerWeights::FullAttn(FullAttnLayerWeights {
                    attn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wq: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.q_proj"),
                        q_out_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wk: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.k_proj"),
                        kv_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wv: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.v_proj"),
                        kv_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wo: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.o_proj"),
                        config.dim,
                        config.n_heads * config.head_dim,
                        gs,
                        kr,
                    )?,
                    q_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.self_attn.q_norm.weight"),
                        &[config.head_dim],
                    )?,
                    k_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.self_attn.k_norm.weight"),
                        &[config.head_dim],
                    )?,
                    ffn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    w_gate: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.mlp.gate_proj"),
                        config.hidden_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    w_up: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.mlp.up_proj"),
                        config.hidden_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    w_down: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.mlp.down_proj"),
                        config.dim,
                        config.hidden_dim,
                        gs,
                        kr,
                    )?,
                    bf16_down_shadow: None,
                }));
            }
            (LayerType::LinearAttention, true) => {
                let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                    + config.linear_num_value_heads * config.linear_value_head_dim;
                let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
                layers.push(LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                    attn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wqkv: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_qkv"),
                        qkv_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wz: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_z"),
                        d_inner,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    // in_proj_a / in_proj_b are dense FP16 in PARO checkpoints
                    // (paro_load_wt auto-falls-back to FP16 when no `.qweight` sibling exists).
                    w_alpha: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_a"),
                        config.linear_num_value_heads,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    w_beta: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.in_proj_b"),
                        config.linear_num_value_heads,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    a_log: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.A_log"),
                        config.linear_num_value_heads,
                    )?,
                    dt_bias: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.dt_bias"),
                        config.linear_num_value_heads,
                    )?,
                    conv_weight: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.conv1d.weight"),
                        qkv_dim * config.conv_kernel_dim,
                    )?,
                    norm_weight: paro_load_f32(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.norm.weight"),
                        config.linear_value_head_dim,
                    )?,
                    wo: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.linear_attn.out_proj"),
                        config.dim,
                        d_inner,
                        gs,
                        kr,
                    )?,
                    ffn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    ffn: paro_load_moe_ffn(source, gpu, &p, config, i as u16)?,
                }));
            }
            (LayerType::FullAttention, true) => {
                let q_out_dim = qwen35_fa_q_out_dim(config);
                let kv_dim = config.n_kv_heads * config.head_dim;
                layers.push(LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                    attn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.input_layernorm.weight"),
                        &[config.dim],
                    )?,
                    wq: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.q_proj"),
                        q_out_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wk: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.k_proj"),
                        kv_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wv: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.v_proj"),
                        kv_dim,
                        config.dim,
                        gs,
                        kr,
                    )?,
                    wo: paro_load_wt(
                        source,
                        gpu,
                        &format!("{p}.self_attn.o_proj"),
                        config.dim,
                        config.n_heads * config.head_dim,
                        gs,
                        kr,
                    )?,
                    q_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.self_attn.q_norm.weight"),
                        &[config.head_dim],
                    )?,
                    k_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.self_attn.k_norm.weight"),
                        &[config.head_dim],
                    )?,
                    ffn_norm: paro_load_norm(
                        source,
                        gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                        &[config.dim],
                    )?,
                    ffn: paro_load_moe_ffn(source, gpu, &p, config, i as u16)?,
                }));
            }
        }
    }

    Ok(Qwen35Weights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
        slab_storage: None,
        rq_corrections: std::collections::HashMap::new(),
        pager: None,
    })
}

/// Multi-GPU weight loader. Variant 2 placement: `token_embd` on `gpus.devices[0]`,
/// `output_norm + output` on `gpus.devices[gpus.output_device]`, each layer on
/// `gpus.devices[gpus.device_for_layer(i)]`. The single-GPU `load_weights` path is
/// not consumed by this — keeping it byte-exact for the pp=1 daemon.
///
/// `pager` is always `None` on this path: paged-experts (MAD-93) is not wired
/// for pp>1 yet — would need per-band drain semantics in `WeightPager::free_all`.
pub fn load_weights_multi(
    hfq: &HfqFile,
    config: &Qwen35Config,
    gpus: &mut Gpus,
) -> HipResult<Qwen35Weights> {
    validate_ffn_bf16_hfq_load(config)?;
    let (token_embd, embd_fmt) = load_token_embd_into(hfq, config, &mut gpus.devices[0])?;
    let out_dev = gpus.output_device;
    let (output_norm, output) = load_output_into(hfq, config, &mut gpus.devices[out_dev])?;
    let is_moe = config.num_experts > 0;
    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let dev_idx = gpus.device_for_layer(i);
        eprintln!(
            "  loading layer {i}/{} on dev {dev_idx} ({:?}{})...",
            config.n_layers,
            config.layer_types[i],
            if is_moe { " + MoE" } else { "" },
        );
        hipfire_runtime::load_progress::report(i as u32 + 1, config.n_layers as u32, "weights");
        let p = format!("layers.{i}");
        let layer_page_start = hfq.layer_data_range(&p);
        layers.push(load_layer_into(
            hfq,
            config,
            i,
            &p,
            &mut gpus.devices[dev_idx],
            None,
        )?);
        if let Some((start, end)) = layer_page_start {
            hfq.drop_pages_range(start, end - start);
        }
    }
    Ok(Qwen35Weights {
        token_embd,
        embd_format: embd_fmt,
        output_norm,
        output,
        layers,
        slab_storage: None,
        rq_corrections: std::collections::HashMap::new(),
        pager: None,
    })
}

fn load_token_embd_into(
    hfq: &HfqFile,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    eprintln!("  loading token_embd...");
    if config.is_vl_text {
        eprintln!(
            "  qwen3.5-vl text wrapper: mrope_interleaved={} mrope_section={:?}",
            config.mrope_interleaved, config.mrope_section
        );
    }
    let embd_info = qwen35_tensor_data(hfq, "embed_tokens.weight").expect("embed_tokens not found");
    Ok(if embd_info.0.quant_type == 6 {
        eprintln!("    (HFQ4-G256 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::HFQ4G256,
        )
    } else if embd_info.0.quant_type == 7 {
        eprintln!("    (HFQ4-G128 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::HFQ4G128,
        )
    } else if embd_info.0.quant_type == 3 {
        eprintln!("    (Q8_0 raw, {} MB)", embd_info.1.len() / 1_000_000);
        (
            gpu.upload_raw(embd_info.1, &[embd_info.1.len()])?,
            EmbeddingFormat::Q8_0,
        )
    } else {
        let f32_data = hfq_plain_tensor_as_f32(embd_info.0, embd_info.1, "embed_tokens.weight");
        (
            gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?,
            EmbeddingFormat::F32,
        )
    })
}

fn load_output_into(
    hfq: &HfqFile,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> HipResult<(GpuTensor, WeightTensor)> {
    eprintln!("  loading output_norm...");
    // See the matching block in the main load path for the rationale —
    // GemmaRMSNorm `+= 1.0` bake applies uniformly for dense and MoE.
    let output_norm = load_norm_weight(hfq, gpu, "norm.weight", &[config.dim])?;

    let lm_head_info = qwen35_tensor_data(hfq, "lm_head.weight");
    let mut output = if let Some((lm_info, lm_data)) = lm_head_info {
        eprintln!(
            "  loading output (separate lm_head, qt={})...",
            lm_info.quant_type
        );
        load_weight_tensor_raw(
            gpu,
            lm_info.quant_type,
            lm_data,
            config.vocab_size,
            config.dim,
        )?
    } else {
        let embd_info =
            qwen35_tensor_data(hfq, "embed_tokens.weight").expect("embed_tokens not found");
        eprintln!(
            "  loading output (tied embeddings, qt={})...",
            embd_info.0.quant_type
        );
        let embd_data = embd_info.1;
        if embd_info.0.quant_type == 6 || embd_info.0.quant_type == 7 || embd_info.0.quant_type == 8
        {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            let dtype = match embd_info.0.quant_type {
                6 => DType::HFQ4G256,
                7 => DType::HFQ4G128,
                8 => DType::HFQ6G256,
                _ => unreachable!(),
            };
            WeightTensor {
                buf,
                gpu_dtype: dtype,
                m: config.vocab_size,
                k: config.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        } else if embd_info.0.quant_type == 13 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m: config.vocab_size,
                k: config.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        } else if embd_info.0.quant_type == 14 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            WeightTensor {
                buf,
                gpu_dtype: DType::MQ8G256,
                m: config.vocab_size,
                k: config.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        } else if embd_info.0.quant_type == 3 {
            let buf = gpu.upload_raw(embd_data, &[embd_data.len()])?;
            WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m: config.vocab_size,
                k: config.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        } else if embd_info.0.quant_type == 16 {
            load_bf16_matrix_weight(gpu, embd_data, config.vocab_size, config.dim)?
        } else {
            let f32_data = hfq_plain_tensor_as_f32(embd_info.0, embd_data, "embed_tokens.weight");
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[config.vocab_size, config.dim])?;
            WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m: config.vocab_size,
                k: config.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            }
        }
    };
    // AWQ sidecar attachment — sister of the `load_weights` block.
    // Safe because both `weight_gemv` (decode) and `speculative.rs`
    // (spec-verify) route through AWQ-aware rotations on
    // `output.awq_scale.is_some()`. No-op on current files.
    if output.gpu_dtype.supports_awq_sidecar() {
        output.awq_scale = load_awq_scale_for(hfq, gpu, "lm_head.weight", config.dim)
            .or_else(|| {
                load_awq_scale_for(hfq, gpu, "model.language_model.lm_head.weight", config.dim)
            })
            .or_else(|| {
                load_awq_scale_for(
                    hfq,
                    gpu,
                    "model.language_model.embed_tokens.weight",
                    config.dim,
                )
            });
        eprintln!(
            "  lm_head AWQ sidecar: {}",
            if output.awq_scale.is_some() {
                "attached"
            } else {
                "absent (no-op)"
            }
        );
    }
    Ok((output_norm, output))
}

/// Build one layer's `LayerWeights` on `gpu`. Extracted for `load_weights_multi`
/// so the multi-GPU loader can route each layer to its band-owning device
/// without duplicating the tensor-name table. Master's `load_weights` keeps
/// its inline body — does not consume this helper.
fn load_layer_into(
    hfq: &HfqFile,
    config: &Qwen35Config,
    layer_idx: usize,
    p: &str,
    gpu: &mut Gpu,
    slabs: Option<&SlabTensorIndex>,
) -> HipResult<LayerWeights> {
    let is_moe = config.num_experts > 0;
    Ok(match (config.layer_types[layer_idx], is_moe) {
        (LayerType::LinearAttention, false) => {
            let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim;
            let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
            LayerWeights::DeltaNet(DeltaNetLayerWeights {
                attn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    &[config.dim],
                )?,
                wqkv: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_qkv.weight"),
                    qkv_dim,
                    config.dim,
                )?,
                wz: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_z.weight"),
                    d_inner,
                    config.dim,
                )?,
                w_alpha: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_a.weight"),
                    config.linear_num_value_heads,
                    config.dim,
                )?,
                w_beta: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_b.weight"),
                    config.linear_num_value_heads,
                    config.dim,
                )?,
                a_log: load_raw_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.A_log"),
                    config.linear_num_value_heads,
                )?,
                dt_bias: load_raw_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.dt_bias"),
                    config.linear_num_value_heads,
                )?,
                conv_weight: load_any_as_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.conv1d.weight"),
                    qkv_dim * config.conv_kernel_dim,
                )?,
                norm_weight: load_any_as_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.norm.weight"),
                    config.linear_value_head_dim,
                )?,
                wo: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.out_proj.weight"),
                    config.dim,
                    d_inner,
                )?,
                ffn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[config.dim],
                )?,
                w_gate: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?,
                w_up: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?,
                w_down: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )?,
                bf16_down_shadow: load_bf16_down_shadow_for(
                    hfq,
                    &format!("{p}.mlp.down_proj.weight"),
                    layer_idx,
                    config.dim,
                    config.hidden_dim,
                )?,
            })
        }
        (LayerType::FullAttention, false) => {
            let q_out_dim = qwen35_fa_q_out_dim(config);
            let kv_dim = config.n_kv_heads * config.head_dim;
            LayerWeights::FullAttn(FullAttnLayerWeights {
                attn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    &[config.dim],
                )?,
                wq: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_out_dim,
                    config.dim,
                )?,
                wk: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.dim,
                )?,
                wv: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    config.dim,
                )?,
                wo: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.o_proj.weight"),
                    config.dim,
                    config.n_heads * config.head_dim,
                )?,
                q_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[config.head_dim],
                )?,
                k_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[config.head_dim],
                )?,
                ffn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[config.dim],
                )?,
                w_gate: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?,
                w_up: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )?,
                w_down: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )?,
                bf16_down_shadow: load_bf16_down_shadow_for(
                    hfq,
                    &format!("{p}.mlp.down_proj.weight"),
                    layer_idx,
                    config.dim,
                    config.hidden_dim,
                )?,
            })
        }
        (LayerType::LinearAttention, true) => {
            let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
                + config.linear_num_value_heads * config.linear_value_head_dim;
            let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
            LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                attn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    &[config.dim],
                )?,
                wqkv: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_qkv.weight"),
                    qkv_dim,
                    config.dim,
                )?,
                wz: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_z.weight"),
                    d_inner,
                    config.dim,
                )?,
                w_alpha: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_a.weight"),
                    config.linear_num_value_heads,
                    config.dim,
                )?,
                w_beta: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.in_proj_b.weight"),
                    config.linear_num_value_heads,
                    config.dim,
                )?,
                a_log: load_raw_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.A_log"),
                    config.linear_num_value_heads,
                )?,
                dt_bias: load_raw_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.dt_bias"),
                    config.linear_num_value_heads,
                )?,
                conv_weight: load_any_as_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.conv1d.weight"),
                    qkv_dim * config.conv_kernel_dim,
                )?,
                norm_weight: load_any_as_f32(
                    hfq,
                    gpu,
                    &format!("{p}.linear_attn.norm.weight"),
                    config.linear_value_head_dim,
                )?,
                wo: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.linear_attn.out_proj.weight"),
                    config.dim,
                    d_inner,
                )?,
                ffn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[config.dim],
                )?,
                ffn: load_moe_ffn(hfq, gpu, slabs, p, config, layer_idx as u16)?,
            })
        }
        (LayerType::FullAttention, true) => {
            let q_out_dim = qwen35_fa_q_out_dim(config);
            let kv_dim = config.n_kv_heads * config.head_dim;
            LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                attn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.input_layernorm.weight"),
                    &[config.dim],
                )?,
                wq: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_out_dim,
                    config.dim,
                )?,
                wk: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.dim,
                )?,
                wv: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    config.dim,
                )?,
                wo: load_weight_tensor(
                    hfq,
                    gpu,
                    slabs,
                    &format!("{p}.self_attn.o_proj.weight"),
                    config.dim,
                    config.n_heads * config.head_dim,
                )?,
                q_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[config.head_dim],
                )?,
                k_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[config.head_dim],
                )?,
                ffn_norm: load_norm_weight(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[config.dim],
                )?,
                ffn: load_moe_ffn(hfq, gpu, slabs, p, config, layer_idx as u16)?,
            })
        }
    })
}

/// Load one layer's full MoE FFN block: router, all routed experts, shared expert,
/// and the per-layer scalar shared-expert gate. Tensor naming follows what the
/// quantizer emits for qwen3_5_moe (commit 4860575): the 3D stacked-expert source
/// tensors get split per-expert into `mlp.experts.{X}.{base}.weight`.
fn dummy_moe_weight_tensor(gpu: &mut Gpu) -> HipResult<WeightTensor> {
    Ok(WeightTensor {
        buf: gpu.upload_f32(&[0.0], &[1])?,
        gpu_dtype: DType::F32,
        m: 0,
        k: 0,
        row_stride: 0,
        paro: None,
        awq_scale: None,
    })
}

fn load_shared_moe_weights(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    slabs: Option<&SlabTensorIndex>,
    p: &str,
    config: &Qwen35Config,
) -> HipResult<(SharedExpertWeights, WeightTensor)> {
    if !config.has_shared_expert {
        return Ok((
            SharedExpertWeights {
                gate: dummy_moe_weight_tensor(gpu)?,
                up: dummy_moe_weight_tensor(gpu)?,
                down: dummy_moe_weight_tensor(gpu)?,
            },
            dummy_moe_weight_tensor(gpu)?,
        ));
    }

    let smi = config.shared_expert_intermediate_size;
    let shared_expert = SharedExpertWeights {
        gate: load_weight_tensor(
            hfq,
            gpu,
            slabs,
            &format!("{p}.mlp.shared_expert.gate_proj.weight"),
            smi,
            config.dim,
        )?,
        up: load_weight_tensor(
            hfq,
            gpu,
            slabs,
            &format!("{p}.mlp.shared_expert.up_proj.weight"),
            smi,
            config.dim,
        )?,
        down: load_weight_tensor(
            hfq,
            gpu,
            slabs,
            &format!("{p}.mlp.shared_expert.down_proj.weight"),
            config.dim,
            smi,
        )?,
    };
    let shared_expert_gate = load_weight_tensor(
        hfq,
        gpu,
        slabs,
        &format!("{p}.mlp.shared_expert_gate.weight"),
        1,
        config.dim,
    )?;
    Ok((shared_expert, shared_expert_gate))
}

fn load_moe_ffn(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    slabs: Option<&SlabTensorIndex>,
    p: &str,
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;

    // Router: hidden_size → num_experts. Precision-sensitive but small.
    let router = load_weight_tensor(
        hfq,
        gpu,
        slabs,
        &format!("{p}.mlp.gate.weight"),
        n_exp,
        config.dim,
    )?;

    let (shared_expert, shared_expert_gate) = load_shared_moe_weights(hfq, gpu, slabs, p, config)?;

    // Routed experts — quantizer wrote per-expert tensors named
    // `{p}.mlp.experts.{X}.gate_up_proj.weight` (shape [2*moe_intermediate, hidden_size])
    // and `{p}.mlp.experts.{X}.down_proj.weight` (shape [hidden_size, moe_intermediate]).
    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gate_up = load_weight_tensor(
            hfq,
            gpu,
            slabs,
            &format!("{p}.mlp.experts.{x}.gate_up_proj.weight"),
            2 * mi,
            config.dim,
        )?;
        let down = load_weight_tensor(
            hfq,
            gpu,
            slabs,
            &format!("{p}.mlp.experts.{x}.down_proj.weight"),
            config.dim,
            mi,
        )?;
        experts.push(ExpertWeights { gate_up, down });
    }

    // Build the device-side pointer tables consumed by the indexed MoE
    // GEMV kernels. Each slot is an `unsigned long long` (the device
    // address of an expert's `gate_up.buf` / `down.buf`). Stored as an
    // F32 tensor of length 2 * num_experts because each pointer occupies
    // 8 bytes = 2 F32 slots; the kernel reads them via a u64 cast.
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(n_exp);
    for e in &experts {
        gu_ptrs.push(e.gate_up.buf.buf.as_ptr() as u64);
        dn_ptrs.push(e.down.buf.buf.as_ptr() as u64);
    }
    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &gu_bytes)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &dn_bytes)?;
    let expert_gate_up_dtype = experts.first().map(|e| e.gate_up.gpu_dtype);
    let expert_down_dtype = experts.first().map(|e| e.down.gpu_dtype);

    Ok(MoeFfnWeights {
        router,
        experts,
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        // MAD-93 v0.1: non-paged loader path. Layer identity for pager-keyed
        // future work, expert_shape None (callers read shapes off `experts`
        // directly when paged_experts==false).
        layer_idx,
        expert_shape: None,
        expert_gate_up_dtype,
        expert_down_dtype,
        paro_shared: None,
    })
}

fn load_moe_ffn_paged(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    slabs: Option<&SlabTensorIndex>,
    p: &str,
    config: &Qwen35Config,
    layer_idx: u16,
) -> HipResult<MoeFfnWeights> {
    let n_exp = config.num_experts;
    let mi = config.moe_intermediate_size;

    let router = load_weight_tensor(
        hfq,
        gpu,
        slabs,
        &format!("{p}.mlp.gate.weight"),
        n_exp,
        config.dim,
    )?;

    let (shared_expert, shared_expert_gate) = load_shared_moe_weights(hfq, gpu, slabs, p, config)?;

    let gate_up_name = format!("{p}.mlp.experts.0.gate_up_proj.weight");
    let down_name = format!("{p}.mlp.experts.0.down_proj.weight");
    let gate_up = hfq.find_tensor_info(&gate_up_name).ok_or_else(|| {
        HipError::new(
            0,
            &format!("paged MoE missing routed expert tensor {gate_up_name}"),
        )
    })?;
    let down = hfq.find_tensor_info(&down_name).ok_or_else(|| {
        HipError::new(
            0,
            &format!("paged MoE missing routed expert tensor {down_name}"),
        )
    })?;
    if gate_up.shape.as_slice() != [((2 * mi) as u32), config.dim as u32] {
        eprintln!(
            "  warning: paged expert gate_up shape {:?} differs from expected [{}, {}]",
            gate_up.shape,
            2 * mi,
            config.dim
        );
    }
    if down.shape.as_slice() != [config.dim as u32, mi as u32] {
        eprintln!(
            "  warning: paged expert down shape {:?} differs from expected [{}, {}]",
            down.shape, config.dim, mi
        );
    }

    let zero_ptrs = vec![0u8; n_exp * 8];
    let expert_gate_up_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    let expert_down_ptrs = gpu.alloc_tensor(&[2 * n_exp], DType::F32)?;
    gpu.hip.memcpy_htod(&expert_gate_up_ptrs.buf, &zero_ptrs)?;
    gpu.hip.memcpy_htod(&expert_down_ptrs.buf, &zero_ptrs)?;

    Ok(MoeFfnWeights {
        router,
        experts: Vec::new(),
        shared_expert,
        shared_expert_gate,
        expert_gate_up_ptrs,
        expert_down_ptrs,
        layer_idx,
        expert_shape: Some(hipfire_runtime::weight_pager::ExpertShape {
            gate_up_m: 2 * mi,
            gate_up_k: config.dim,
            down_m: config.dim,
            down_k: mi,
        }),
        expert_gate_up_dtype: slab_dtype_for_quant(gate_up.quant_type, config.dim),
        expert_down_dtype: slab_dtype_for_quant(down.quant_type, mi),
        paro_shared: None,
    })
}
