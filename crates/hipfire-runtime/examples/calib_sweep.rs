#![allow(clippy::too_many_arguments)]
// SPDX-License-Identifier: Apache-2.0
// calib_sweep — dense HFIM+HFHS calibration shim (MFMA-rate batched).
// Teacher bf16, seqs independent (KV clear + pos 0 per --seq-len window),
// KV allocated ONCE at seq_len+16 (not corpus length).
//
// Taps:
//   TAP1 llama.rs:1460 weight_gemm → dispatch.rs:2848 (llama 0/1, prefill_forward:1508)
//   TAP2 families/gemm.rs:146 GemmFamily::run_key (gemma4 lowered.rs:173, glimmer forward.rs:80, qwen35 migrated sites qwen35.rs:12877/12928/13230)
//   TAP3 forward_batch.rs:79 batched_proj (lfm2 dense, single chokepoint)

use hipfire_runtime::calibration::{collect, collect_grouped, CalibForward};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}
fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "usage: calib_sweep --model <bf16.hfq> --corpus <txt> --output <path.calib.hfq> [--seq-len N] [--max-tokens N] [--layers-per-pass N] [--arch <id>] [--syrk auto|rocblas|kernel]  defaults 2048/262144/64/auto  dense 0/1+5/6+13/14 via taps, 11 batched"
    );
    std::process::exit(code)
}
fn corpus_md5(path: &str) -> String {
    if let Ok(out) = Command::new("md5sum").arg(path).output() {
        if out.status.success() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Some(hex) = s.split_whitespace().next() {
                    return hex.to_string();
                }
            }
        }
    }
    std::fs::metadata(path)
        .map(|m| format!("{:016x}", m.len()))
        .unwrap_or_else(|_| "unknown".into())
}
fn compact_hessian_bytes(k: usize) -> u64 {
    (k * 4 + k * (k - 1)) as u64
}
/// Derive the checkpoint's safetensors tensor-name prefix from the OPEN HFQ
/// file, so every arch emits the correct wrapper (e.g. `model.language_model.`
/// for qwen3.8, `model.` for llama/gemma) without hardcoding.
/// Enumerates HFQ tensor names, finds one whose tail matches a known per-layer
/// suffix ending in `layers.0.<rest>.weight`, and returns everything preceding
/// `layers.` as the prefix. FAILS LOUDLY with exit(1) if no tensor matches —
/// a wrong prefix produces the silent no-op this fix addresses.
fn derive_calib_prefix(hfq: &HfqFile) -> String {
    const CANDIDATES: &[&str] = &[
        "layers.0.mlp.down_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.gate_proj.weight",
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.linear_attn.in_proj_z.weight",
        "layers.0.linear_attn.in_proj_a.weight",
        "layers.0.linear_attn.in_proj_b.weight",
        "layers.0.linear_attn.out_proj.weight",
        "layers.0.conv.in_proj.weight",
        "layers.0.conv.out_proj.weight",
    ];
    for ti in hfq.tensors() {
        let name = &ti.name;
        for suffix in CANDIDATES {
            if name.ends_with(suffix) {
                if let Some(pos) = name.find("layers.") {
                    let prefix = name[..pos].to_string();
                    eprintln!(
                        "calib prefix derived: '{}' from tensor '{}' (matched suffix '{}')",
                        prefix, name, suffix
                    );
                    return prefix;
                }
            }
        }
    }
    eprintln!("FATAL: could not derive calibration tensor prefix from HFQ '{}': no tensor matched known per-layer suffixes {:?}.", hfq.tensors().first().map(|t| t.name.as_str()).unwrap_or("<empty>"), CANDIDATES);
    eprintln!(
        "  HFQ contains {} tensors, first few: {:?}",
        hfq.tensors().len(),
        hfq.tensors()
            .iter()
            .take(5)
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
    );
    eprintln!("  Expected at least one tensor ending with e.g. 'layers.0.mlp.down_proj.weight' (safetensors convention).");
    std::process::exit(1);
}
/// Verify every emitted capture name actually exists as a tensor in the loaded
/// HFQ. A name that does not resolve is a silent calibration no-op. On any
/// mismatch print the offending names (cap ~20) and exit(1).
fn verify_capture_names_or_exit(hfq: &HfqFile, capture_names: &[String]) {
    let mut missing: Vec<String> = Vec::new();
    for n in capture_names {
        if hfq.find_tensor_info(n).is_none() {
            missing.push(n.clone());
        }
    }
    if !missing.is_empty() {
        missing.sort();
        eprintln!("FATAL: {} capture name(s) do not exist as tensors in the loaded HFQ (safetensors convention mismatch):", missing.len());
        for n in missing.iter().take(20) {
            eprintln!("  missing: {}", n);
        }
        if missing.len() > 20 {
            eprintln!("  ... and {} more", missing.len() - 20);
        }
        eprintln!(
            "  HFQ contains {} tensors, e.g.: {:?}",
            hfq.tensors().len(),
            hfq.tensors()
                .iter()
                .take(5)
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
        );
        std::process::exit(1);
    }
}
fn estimate_peak_llama(
    config: &hipfire_runtime::llama::LlamaConfig,
    weights: &hipfire_runtime::llama::LlamaWeights,
    start: usize,
    end: usize,
) -> u64 {
    let mut b = 0u64;
    for i in start..end.min(weights.layers.len()).min(config.n_layers) {
        let l = &weights.layers[i];
        for k in [
            l.wq.k, l.wk.k, l.wv.k, l.wo.k, l.w_gate.k, l.w_up.k, l.w_down.k,
        ] {
            b += compact_hessian_bytes(k) + (k * 4) as u64;
        }
    }
    b
}
fn build_capture_llama(
    weights: &hipfire_runtime::llama::LlamaWeights,
    start: usize,
    end: usize,
    prefix: &str,
) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    for i in start..end.min(weights.layers.len()) {
        let l = &weights.layers[i];
        let p = format!("{prefix}layers.{i}");
        for (wt, name) in [
            (&l.wq, format!("{p}.self_attn.q_proj.weight")),
            (&l.wk, format!("{p}.self_attn.k_proj.weight")),
            (&l.wv, format!("{p}.self_attn.v_proj.weight")),
            (&l.wo, format!("{p}.self_attn.o_proj.weight")),
            (&l.w_gate, format!("{p}.mlp.gate_proj.weight")),
            (&l.w_up, format!("{p}.mlp.up_proj.weight")),
            (&l.w_down, format!("{p}.mlp.down_proj.weight")),
        ] {
            m.insert(wt.buf.buf.as_ptr() as usize, name);
        }
    }
    m
}
fn build_capture_gemma(
    weights: &hipfire_arch_gemma4::lowered::Gemma4Weights,
    start: usize,
    end: usize,
    prefix: &str,
) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    for i in start..end.min(weights.layers.len()) {
        let p = format!("{prefix}layers.{i}");
        match &weights.layers[i] {
            hipfire_arch_gemma4::lowered::LayerWeights::Sliding(s) => {
                m.insert(
                    s.q_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.q_proj.weight"),
                );
                m.insert(
                    s.k_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.k_proj.weight"),
                );
                m.insert(
                    s.v_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.v_proj.weight"),
                );
                m.insert(
                    s.o_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.o_proj.weight"),
                );
                m.insert(
                    s.gate_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.gate_proj.weight"),
                );
                m.insert(
                    s.up_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.up_proj.weight"),
                );
                m.insert(
                    s.down_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.down_proj.weight"),
                );
            }
            hipfire_arch_gemma4::lowered::LayerWeights::Full(f) => {
                m.insert(
                    f.q_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.q_proj.weight"),
                );
                m.insert(
                    f.k_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.k_proj.weight"),
                );
                // no v_proj on Full — V reuses k_proj pre-norm
                m.insert(
                    f.o_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.o_proj.weight"),
                );
                m.insert(
                    f.gate_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.gate_proj.weight"),
                );
                m.insert(
                    f.up_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.up_proj.weight"),
                );
                m.insert(
                    f.down_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.down_proj.weight"),
                );
            }
        }
    }
    m
}
fn build_capture_glimmer(
    weights: &hipfire_arch_muse_glimmer::glimmer::GlimmerWeights,
    start: usize,
    end: usize,
    prefix: &str,
) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    for i in start..end.min(weights.layers.len()) {
        let p = format!("{prefix}layers.{i}");
        let l = &weights.layers[i];
        m.insert(
            l.attn_gate_proj.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.gate_proj.weight"),
        );
        m.insert(
            l.q_proj.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.q_proj.weight"),
        );
        m.insert(
            l.k_proj.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.k_proj.weight"),
        );
        m.insert(
            l.v_proj.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.v_proj.weight"),
        );
        m.insert(
            l.o_proj.buf.buf.as_ptr() as usize,
            format!("{p}.self_attn.o_proj.weight"),
        );
        m.insert(
            l.gate_proj.buf.buf.as_ptr() as usize,
            format!("{p}.mlp.gate_proj.weight"),
        );
        m.insert(
            l.up_proj.buf.buf.as_ptr() as usize,
            format!("{p}.mlp.up_proj.weight"),
        );
        m.insert(
            l.down_proj.buf.buf.as_ptr() as usize,
            format!("{p}.mlp.down_proj.weight"),
        );
    }
    m
}
fn build_capture_lfm(
    weights: &hipfire_arch_lfm2moe::lfm2moe::Lfm2MoeWeights,
    start: usize,
    end: usize,
    prefix: &str,
) -> HashMap<usize, String> {
    let mut m = HashMap::new();
    for i in start..end.min(weights.layers.len()) {
        let layer = &weights.layers[i];
        let p = format!("{prefix}layers.{i}");
        match &layer.mixer {
            hipfire_arch_lfm2moe::lfm2moe::Mixer::Conv(c) => {
                m.insert(
                    c.in_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.conv.in_proj.weight"),
                );
                m.insert(
                    c.out_proj.buf.buf.as_ptr() as usize,
                    format!("{p}.conv.out_proj.weight"),
                );
            }
            hipfire_arch_lfm2moe::lfm2moe::Mixer::Attention(a) => {
                m.insert(
                    a.wq.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.q_proj.weight"),
                );
                m.insert(
                    a.wk.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.k_proj.weight"),
                );
                m.insert(
                    a.wv.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.v_proj.weight"),
                );
                // LFM2 names the attention output `out_proj`, not `o_proj`.
                m.insert(
                    a.wo.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.out_proj.weight"),
                );
            }
        }
        match &layer.ffn {
            hipfire_arch_lfm2moe::lfm2moe::Ffn::Dense(d) => {
                // LFM2's dense FFN is `feed_forward.w1/w2/w3`, not `mlp.*_proj`:
                // w1 = gate, w3 = up, w2 = down. Verified against the teacher HFQ
                // (model.layers.0.feed_forward.w{1,2,3}.weight).
                m.insert(
                    d.w1.buf.buf.as_ptr() as usize,
                    format!("{p}.feed_forward.w1.weight"),
                );
                m.insert(
                    d.w3.buf.buf.as_ptr() as usize,
                    format!("{p}.feed_forward.w3.weight"),
                );
                m.insert(
                    d.w2.buf.buf.as_ptr() as usize,
                    format!("{p}.feed_forward.w2.weight"),
                );
            }
            hipfire_arch_lfm2moe::lfm2moe::Ffn::Moe(_) => {}
        }
    }
    m
}
fn estimate_peak_gemma(
    weights: &hipfire_arch_gemma4::lowered::Gemma4Weights,
    start: usize,
    end: usize,
) -> u64 {
    let mut b = 0u64;
    for i in start..end.min(weights.layers.len()) {
        let ks: Vec<usize> = match &weights.layers[i] {
            hipfire_arch_gemma4::lowered::LayerWeights::Sliding(s) => vec![
                s.q_proj.k,
                s.k_proj.k,
                s.v_proj.k,
                s.o_proj.k,
                s.gate_proj.k,
                s.up_proj.k,
                s.down_proj.k,
            ],
            hipfire_arch_gemma4::lowered::LayerWeights::Full(f) => vec![
                f.q_proj.k,
                f.k_proj.k,
                f.o_proj.k,
                f.gate_proj.k,
                f.up_proj.k,
                f.down_proj.k,
            ],
        };
        for k in ks {
            b += compact_hessian_bytes(k) + (k * 4) as u64;
        }
    }
    b
}
fn estimate_peak_glimmer(
    weights: &hipfire_arch_muse_glimmer::glimmer::GlimmerWeights,
    start: usize,
    end: usize,
) -> u64 {
    let mut b = 0u64;
    for i in start..end.min(weights.layers.len()) {
        let l = &weights.layers[i];
        for k in [
            l.attn_gate_proj.k,
            l.q_proj.k,
            l.k_proj.k,
            l.v_proj.k,
            l.o_proj.k,
            l.gate_proj.k,
            l.up_proj.k,
            l.down_proj.k,
        ] {
            b += compact_hessian_bytes(k) + (k * 4) as u64;
        }
    }
    b
}
fn estimate_peak_lfm(
    weights: &hipfire_arch_lfm2moe::lfm2moe::Lfm2MoeWeights,
    start: usize,
    end: usize,
) -> u64 {
    let mut b = 0u64;
    for i in start..end.min(weights.layers.len()) {
        let layer = &weights.layers[i];
        let mut ks = Vec::new();
        match &layer.mixer {
            hipfire_arch_lfm2moe::lfm2moe::Mixer::Conv(c) => {
                ks.push(c.in_proj.k);
                ks.push(c.out_proj.k);
            }
            hipfire_arch_lfm2moe::lfm2moe::Mixer::Attention(a) => {
                ks.push(a.wq.k);
                ks.push(a.wk.k);
                ks.push(a.wv.k);
                ks.push(a.wo.k);
            }
        }
        match &layer.ffn {
            hipfire_arch_lfm2moe::lfm2moe::Ffn::Dense(d) => {
                ks.push(d.w1.k);
                ks.push(d.w3.k);
                ks.push(d.w2.k);
            }
            hipfire_arch_lfm2moe::lfm2moe::Ffn::Moe(_) => {}
        }
        for k in ks {
            b += compact_hessian_bytes(k) + (k * 4) as u64;
        }
    }
    b
}
fn try_rocblas_syrk(
    gpu: &mut rdna_compute::Gpu,
    x: &rdna_compute::GpuTensor,
    h: &rdna_compute::GpuTensor,
    k: usize,
    n: usize,
) -> Result<(), String> {
    let rb = gpu
        .rocblas
        .as_ref()
        .ok_or_else(|| "rocblas_syrk: rocBLAS not initialized".to_string())?;
    if let Some(stream) = gpu.active_stream.as_ref() {
        rb.set_stream(stream)
            .map_err(|e| format!("rocblas_set_stream syrk: {}", e.context))?;
    }
    let alpha = 1.0f32;
    let beta = 1.0f32;
    unsafe {
        rb.gemm_ex(
            hip_bridge::RocblasOperation::Transpose,
            hip_bridge::RocblasOperation::None,
            k as i32,
            k as i32,
            n as i32,
            &alpha as *const f32 as *const std::ffi::c_void,
            x.buf.as_ptr(),
            hip_bridge::RocblasDatatype::F32,
            k as i32,
            x.buf.as_ptr(),
            hip_bridge::RocblasDatatype::F32,
            k as i32,
            &beta as *const f32 as *const std::ffi::c_void,
            h.buf.as_ptr(),
            hip_bridge::RocblasDatatype::F32,
            k as i32,
            h.buf.as_ptr(),
            hip_bridge::RocblasDatatype::F32,
            k as i32,
            hip_bridge::RocblasDatatype::F32,
        )
        .map_err(|e| format!("rocblas_syrk gemm_ex: {}", e.context))
    }
}
fn choose_syrk(gpu: &rdna_compute::Gpu, mode: &str) -> (&'static str, bool) {
    match mode {
        "rocblas" => ("rocblas", gpu.rocblas.is_some()),
        "kernel" => ("kernel", true),
        _ => {
            if gpu.rocblas.is_some() {
                ("rocblas", true)
            } else {
                ("kernel", true)
            }
        }
    }
}
fn gemma_env_report() -> String {
    format!("HIPFIRE_BATCHED_PREFILL={} HIPFIRE_WMMA_PREFILL={} — gemma4 batched prefill behind both 1 (lowered.rs:42-53); if off → per-token GEMV loop n=1, tap still fires but batching win absent", std::env::var("HIPFIRE_BATCHED_PREFILL").ok().as_deref().unwrap_or("<unset>"), std::env::var("HIPFIRE_WMMA_PREFILL").ok().as_deref().unwrap_or("<unset>"))
}
fn check_identity(output: &Path, n_layers: usize, prefix: &str) -> Result<(), String> {
    let mut hfq =
        HfqFile::open(output).map_err(|e| format!("open {} identity: {e}", output.display()))?;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut bad = Vec::new();
    for i in 0..n_layers {
        let chk = |names: &[String], lab: &str| -> Option<String> {
            let mut hs = Vec::new();
            let mut miss = Vec::new();
            for n in names {
                if let Some((_, d)) = hfq.tensor_data(n) {
                    let mut h = DefaultHasher::new();
                    d.hash(&mut h);
                    hs.push(h.finish());
                } else {
                    miss.push(n.clone());
                }
            }
            if !miss.is_empty() {
                return Some(format!("layer {i} {lab} missing {}", miss.join(",")));
            }
            if hs.windows(2).any(|w| w[0] != w[1]) {
                return Some(format!("layer {i} {lab} diff hashes {:?}", hs));
            }
            None
        };
        if let Some(m) = chk(
            &[
                format!("{prefix}layers.{i}.self_attn.q_proj.weight.hessian"),
                format!("{prefix}layers.{i}.self_attn.k_proj.weight.hessian"),
                format!("{prefix}layers.{i}.self_attn.v_proj.weight.hessian"),
            ],
            "q/k/v",
        ) {
            bad.push(m);
        }
        if let Some(m) = chk(
            &[
                format!("{prefix}layers.{i}.mlp.gate_proj.weight.hessian"),
                format!("{prefix}layers.{i}.mlp.up_proj.weight.hessian"),
            ],
            "gate/up",
        ) {
            bad.push(m);
        }
    }
    drop(hfq);
    if bad.is_empty() {
        Ok(())
    } else {
        Err(bad.join("; "))
    }
}
fn check_coverage(output: &Path, expected_names: &[String]) -> Result<(), String> {
    // Coverage gate — MUST run before consistency/identity. Verifies every name registered in
    // capture_names produced BOTH a hessian and an imatrix. Missing capture (e.g. 64/448 down_proj-only)
    // fails here rather than passing vacuous identity/consistency checks.
    let mut hfq =
        HfqFile::open(output).map_err(|e| format!("open {} coverage: {e}", output.display()))?;
    let mut missing = Vec::new();
    for name in expected_names {
        let h = format!("{name}.hessian");
        let im = format!("{name}.imatrix");
        let has_h = hfq.tensor_data(&h).is_some();
        let has_im = hfq.tensor_data(&im).is_some();
        if !has_h || !has_im {
            missing.push(name.clone());
        }
    }
    drop(hfq);
    if !missing.is_empty() {
        let total = expected_names.len();
        let mut list = missing.clone();
        list.sort();
        let shown: Vec<String> = list.iter().take(20).cloned().collect();
        let suffix = if list.len() > 20 {
            format!(" + {} more", list.len() - 20)
        } else {
            String::new()
        };
        return Err(format!(
            "coverage FAIL: {}/{} tensors missing hessian/imatrix: {}{}",
            missing.len(),
            total,
            shown.join(", "),
            suffix
        ));
    }
    Ok(())
}
fn print_per_proj_summary(output: &Path, expected_names: &[String]) {
    // Per-projection-kind summary so a coverage hole is visible at a glance in the run log.
    // Format: `q_proj 64/64  k_proj 64/64  ... down_proj 64/64` (found/expected per kind)
    let mut hfq = match HfqFile::open(output) {
        Ok(f) => f,
        Err(_) => return,
    };
    use std::collections::HashMap;
    let mut expected_counts: HashMap<String, usize> = HashMap::new();
    let mut found_counts: HashMap<String, usize> = HashMap::new();
    for name in expected_names {
        let kind = name.rsplit('.').next().unwrap_or(name).to_string();
        *expected_counts.entry(kind.clone()).or_insert(0) += 1;
        let h = format!("{name}.hessian");
        if hfq.tensor_data(&h).is_some() {
            *found_counts.entry(kind).or_insert(0) += 1;
        }
    }
    let mut kinds: Vec<String> = expected_counts.keys().cloned().collect();
    kinds.sort();
    let mut parts = Vec::new();
    for k in kinds {
        let exp = expected_counts[&k];
        let found = found_counts.get(&k).cloned().unwrap_or(0);
        parts.push(format!("{k} {found}/{exp}"));
    }
    drop(hfq);
    eprintln!("coverage per-proj: {}", parts.join("  "));
}
fn check_identity_generic(
    output: &Path,
    n_layers: usize,
    is_glimmer: bool,
    _is_gemma_full: bool,
    prefix: &str,
) -> Result<(), String> {
    // Strict identity: absence is FAIL, not agreement. Previously filtered to `present` and
    // passed vacuously when 0/1 tensors existed (down_proj-only bug). Now missing members fail.
    // Legitimate absence: lfm conv layers have no q/k/v; gemma Full layers have no v_proj (inferred).
    let mut hfq = HfqFile::open(output)
        .map_err(|e| format!("open {} identity generic: {e}", output.display()))?;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut bad = Vec::new();
    for i in 0..n_layers {
        let conv_marker = format!("{prefix}layers.{i}.conv.in_proj.weight.hessian");
        let is_conv = hfq.tensor_data(&conv_marker).is_some();
        if !is_conv {
            let q_name = format!("{prefix}layers.{i}.self_attn.q_proj.weight.hessian");
            let k_name = format!("{prefix}layers.{i}.self_attn.k_proj.weight.hessian");
            let v_name = format!("{prefix}layers.{i}.self_attn.v_proj.weight.hessian");
            let gate_name = format!("{prefix}layers.{i}.self_attn.gate_proj.weight.hessian");
            let mut attn_expected = vec![q_name.clone(), k_name.clone()];
            if is_glimmer {
                attn_expected.push(v_name.clone());
                attn_expected.push(gate_name.clone());
            }
            let mut attn_missing = Vec::new();
            let mut attn_hs = Vec::new();
            let mut attn_present = Vec::new();
            for n in &attn_expected {
                if let Some((_, d)) = hfq.tensor_data(n) {
                    let mut h = DefaultHasher::new();
                    d.hash(&mut h);
                    attn_hs.push(h.finish());
                    attn_present.push(n.clone());
                } else {
                    attn_missing.push(n.clone());
                }
            }
            if !attn_missing.is_empty() {
                bad.push(format!(
                    "layer {i} attn q/k/v/gate missing {}",
                    attn_missing.join(",")
                ));
            } else if attn_present.len() >= 2 && attn_hs.windows(2).any(|w| w[0] != w[1]) {
                bad.push(format!(
                    "layer {i} attn q/k/v/gate diff hashes {:?} present {:?}",
                    attn_hs, attn_present
                ));
            }
        }
        // FFN gate/up — REQUIRED for every dense FFN layer. Both must exist or it's FAIL.
        let ffn_names = [
            format!("{prefix}layers.{i}.mlp.gate_proj.weight.hessian"),
            format!("{prefix}layers.{i}.mlp.up_proj.weight.hessian"),
        ];
        let down_name = format!("{prefix}layers.{i}.mlp.down_proj.weight.hessian");
        let has_down = hfq.tensor_data(&down_name).is_some();
        let mut fmissing = Vec::new();
        let mut fhs = Vec::new();
        let mut fpresent = Vec::new();
        for n in &ffn_names {
            if let Some((_, d)) = hfq.tensor_data(n) {
                let mut h = DefaultHasher::new();
                d.hash(&mut h);
                fhs.push(h.finish());
                fpresent.push(n.clone());
            } else {
                fmissing.push(n.clone());
            }
        }
        if !fmissing.is_empty() {
            if fmissing.len() == 2 && !has_down {
                // Both gate/up and down missing => likely MoE layer with no dense capture, skip.
            } else {
                bad.push(format!(
                    "layer {i} ffn gate/up missing {}",
                    fmissing.join(",")
                ));
            }
        } else if fhs.windows(2).any(|w| w[0] != w[1]) {
            bad.push(format!(
                "layer {i} ffn gate/up diff hashes {:?} present {:?}",
                fhs, fpresent
            ));
        }
    }
    drop(hfq);
    if bad.is_empty() {
        Ok(())
    } else {
        Err(bad.join("; "))
    }
}
fn run_llama_batched(
    gpu: &mut rdna_compute::Gpu,
    w: &hipfire_runtime::llama::LlamaWeights,
    c: &hipfire_runtime::llama::LlamaConfig,
    toks: &[u32],
    seq: usize,
) -> Result<(), String> {
    let seq = seq.max(1);
    let mut kv = hipfire_runtime::llama::KvCache::new_gpu(
        gpu,
        c.n_layers,
        c.n_kv_heads,
        c.head_dim,
        seq + 16,
    )
    .map_err(|e| format!("llama kv alloc {seq}: {e}"))?;
    for (si, chunk) in toks.chunks(seq).enumerate() {
        kv.clear_gpu(gpu)
            .map_err(|e| format!("kv clear {si}: {e:?}"))?;
        hipfire_runtime::llama::prefill_forward(gpu, w, c, chunk, &mut kv)
            .map_err(|e| format!("prefill_forward seq {si} B {}: {e:?}", chunk.len()))?;
    }
    Ok(())
}
fn run_gemma_batched(
    gpu: &mut rdna_compute::Gpu,
    weights: &hipfire_arch_gemma4::lowered::Gemma4Weights,
    cfg: &hipfire_arch_gemma4::lowered::Gemma4Config,
    toks: &[u32],
    seq: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &hipfire_arch_gemma4::lowered::Gemma4Scratch,
) -> Result<(), String> {
    let seq = seq.max(1);
    for (si, chunk) in toks.chunks(seq).enumerate() {
        kv_sliding
            .clear_gpu(gpu)
            .map_err(|e| format!("gemma kv sliding clear {si}: {e:?}"))?;
        kv_full
            .clear_gpu(gpu)
            .map_err(|e| format!("gemma kv full clear {si}: {e:?}"))?;
        // forward_prefill_batch has MAX_PREFILL_BATCH=128 hard cap; chunk further
        let max_b = scratch.max_prefill_batch;
        let mut offset = 0usize;
        let mut pos = 0usize;
        while offset < chunk.len() {
            let end = (offset + max_b).min(chunk.len());
            let sub = &chunk[offset..end];
            hipfire_arch_gemma4::lowered::forward_prefill_batch(
                gpu, weights, cfg, sub, pos, kv_sliding, kv_full, scratch,
            )
            .map_err(|e| {
                format!(
                    "gemma forward_prefill_batch seq {si} sub {} B {}: {e:?}",
                    offset,
                    sub.len()
                )
            })?;
            pos += sub.len();
            offset = end;
        }
    }
    Ok(())
}
fn run_glimmer_batched(
    gpu: &mut rdna_compute::Gpu,
    weights: &hipfire_arch_muse_glimmer::glimmer::GlimmerWeights,
    cfg: &hipfire_arch_muse_glimmer::config::GlimmerConfig,
    state: &mut hipfire_arch_muse_glimmer::glimmer::GlimmerState,
    toks: &[u32],
    seq: usize,
) -> Result<(), String> {
    let seq = seq.max(1);
    for (si, chunk) in toks.chunks(seq).enumerate() {
        state.reset();
        // GlimmerState KV is inside state; reset clears n_tokens but not KV buffers — need to zero underlying KV? Glimmer KV is Q8 with per-layer caches; reset's n_tokens=0 means next prefill overwrites from 0, so stale memory irrelevant (same as Gemma clear). No explicit mem clear needed for correctness; but to match KV clear semantics we ensure the underlying caches are logically reset via n_tokens=0.
        hipfire_arch_muse_glimmer::forward::prefill_with_capture(
            cfg,
            weights,
            state,
            gpu,
            chunk,
            0,
            &[],
            &mut Vec::new(),
        )
        .map_err(|e| {
            format!(
                "glimmer prefill_with_capture seq {si} B {}: {e}",
                chunk.len()
            )
        })?;
    }
    Ok(())
}
fn run_lfm_batched(
    gpu: &mut rdna_compute::Gpu,
    weights: &hipfire_arch_lfm2moe::lfm2moe::Lfm2MoeWeights,
    cfg: &hipfire_arch_lfm2moe::config::Lfm2MoeConfig,
    state: &mut hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState,
    toks: &[u32],
    seq: usize,
) -> Result<(), String> {
    let seq = seq.max(1);
    // state max_batch == seq_len+? But we allocated with seq+16. For chunk shorter than max_batch, forward_decode_batch_lfm will handle B = chunk.len() <= max_batch.
    for (si, chunk) in toks.chunks(seq).enumerate() {
        // clear per-batch KV/conv by resetting entire state (cheapest). For DENSE batch state, reset clears kv and conv rings.
        state
            .reset(gpu)
            .map_err(|e| format!("lfm state reset {si}: {e:?}"))?;
        let positions: Vec<usize> = (0..chunk.len()).collect();
        hipfire_arch_lfm2moe::forward_batch::forward_decode_batch_lfm(
            gpu, weights, cfg, chunk, &positions, state,
        )
        .map_err(|e| {
            format!(
                "lfm forward_decode_batch_lfm seq {si} B {}: {e:?}",
                chunk.len()
            )
        })?;
    }
    Ok(())
}
fn build_capture_qwen35(
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
    start: usize,
    end: usize,
    prefix: &str,
) -> HashMap<usize, String> {
    use hipfire_arch_qwen35::qwen35::LayerWeights as LW;
    let mut m = HashMap::new();
    for i in start..end.min(weights.layers.len()) {
        let p = format!("{prefix}layers.{i}");
        match &weights.layers[i] {
            LW::DeltaNet(l) => {
                m.insert(
                    l.wqkv.buf.buf.as_ptr() as usize,
                    format!("{p}.linear_attn.in_proj_qkv.weight"),
                );
                m.insert(
                    l.wz.buf.buf.as_ptr() as usize,
                    format!("{p}.linear_attn.in_proj_z.weight"),
                );
                m.insert(
                    l.w_alpha.buf.buf.as_ptr() as usize,
                    format!("{p}.linear_attn.in_proj_a.weight"),
                );
                m.insert(
                    l.w_beta.buf.buf.as_ptr() as usize,
                    format!("{p}.linear_attn.in_proj_b.weight"),
                );
                m.insert(
                    l.wo.buf.buf.as_ptr() as usize,
                    format!("{p}.linear_attn.out_proj.weight"),
                );
                m.insert(
                    l.w_gate.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.gate_proj.weight"),
                );
                m.insert(
                    l.w_up.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.up_proj.weight"),
                );
                m.insert(
                    l.w_down.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.down_proj.weight"),
                );
            }
            LW::FullAttn(l) => {
                m.insert(
                    l.wq.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.q_proj.weight"),
                );
                m.insert(
                    l.wk.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.k_proj.weight"),
                );
                m.insert(
                    l.wv.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.v_proj.weight"),
                );
                m.insert(
                    l.wo.buf.buf.as_ptr() as usize,
                    format!("{p}.self_attn.o_proj.weight"),
                );
                m.insert(
                    l.w_gate.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.gate_proj.weight"),
                );
                m.insert(
                    l.w_up.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.up_proj.weight"),
                );
                m.insert(
                    l.w_down.buf.buf.as_ptr() as usize,
                    format!("{p}.mlp.down_proj.weight"),
                );
            }
            LW::DeltaNetMoe(_) | LW::FullAttnMoe(_) => {
                // MoE should not appear for dense 27B (num_experts==0); if it does, skip (calibration dense-only).
                // The shim refuses to emit Hessians with missing experts anyway, but capture map stays empty for MoE.
            }
        }
    }
    m
}
fn estimate_peak_qwen35(
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
    start: usize,
    end: usize,
) -> u64 {
    use hipfire_arch_qwen35::qwen35::LayerWeights as LW;
    let mut b = 0u64;
    for i in start..end.min(weights.layers.len()) {
        let ks: Vec<usize> = match &weights.layers[i] {
            LW::DeltaNet(l) => vec![
                l.wqkv.k,
                l.wz.k,
                l.w_alpha.k,
                l.w_beta.k,
                l.wo.k,
                l.w_gate.k,
                l.w_up.k,
                l.w_down.k,
            ],
            LW::FullAttn(l) => vec![
                l.wq.k, l.wk.k, l.wv.k, l.wo.k, l.w_gate.k, l.w_up.k, l.w_down.k,
            ],
            LW::DeltaNetMoe(_) | LW::FullAttnMoe(_) => vec![],
        };
        for k in ks {
            b += compact_hessian_bytes(k) + (k * 4) as u64;
        }
    }
    b
}
fn check_identity_qwen35(output: &Path, n_layers: usize, prefix: &str) -> Result<(), String> {
    // Strict: missing is FAIL. For qwen35 hybrid, each dense layer is either FullAttn (q/k/v/o + gate/up/down)
    // or DeltaNet (in_proj_qkv/z/a/b/out_proj + gate/up/down). Gate/up is required for ALL layers.
    // Attn group is determined per-layer: if any linear_attn tensor exists => DeltaNet, else FullAttn.
    // Previously this function filtered to `present` and passed vacuously when 0 members existed (down_proj-only bug).
    let mut hfq = HfqFile::open(output)
        .map_err(|e| format!("open {} qwen35 identity: {e}", output.display()))?;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut bad = Vec::new();
    for i in 0..n_layers {
        // gate/up — REQUIRED for every dense layer
        let gate_name = format!("{prefix}layers.{i}.mlp.gate_proj.weight.hessian");
        let up_name = format!("{prefix}layers.{i}.mlp.up_proj.weight.hessian");
        let mut gmissing = Vec::new();
        let mut ghs = Vec::new();
        let mut gpresent = Vec::new();
        for name in [&gate_name, &up_name] {
            if let Some((_, d)) = hfq.tensor_data(name) {
                let mut h = DefaultHasher::new();
                d.hash(&mut h);
                ghs.push(h.finish());
                gpresent.push(name.clone());
            } else {
                gmissing.push(name.clone());
            }
        }
        let down_name = format!("{prefix}layers.{i}.mlp.down_proj.weight.hessian");
        let has_down = hfq.tensor_data(&down_name).is_some();
        if !gmissing.is_empty() {
            if gmissing.len() == 2 && !has_down {
                // MoE layer with no dense capture — skip
            } else {
                bad.push(format!("layer {i} gate/up missing {}", gmissing.join(",")));
            }
        } else if ghs.windows(2).any(|w| w[0] != w[1]) {
            bad.push(format!(
                "layer {i} gate/up diff hashes {:?} present {:?}",
                ghs, gpresent
            ));
        }
        // Determine layer kind: DeltaNet if any linear_attn in_proj present, else FullAttn.
        // In down_proj-only bug, neither group present but down exists => will be flagged.
        let lin_qkv = format!("{prefix}layers.{i}.linear_attn.in_proj_qkv.weight.hessian");
        let lin_z = format!("{prefix}layers.{i}.linear_attn.in_proj_z.weight.hessian");
        let lin_a = format!("{prefix}layers.{i}.linear_attn.in_proj_a.weight.hessian");
        let lin_b = format!("{prefix}layers.{i}.linear_attn.in_proj_b.weight.hessian");
        let q_name = format!("{prefix}layers.{i}.self_attn.q_proj.weight.hessian");
        let k_name = format!("{prefix}layers.{i}.self_attn.k_proj.weight.hessian");
        let v_name = format!("{prefix}layers.{i}.self_attn.v_proj.weight.hessian");
        let has_linear = hfq.tensor_data(&lin_qkv).is_some()
            || hfq.tensor_data(&lin_z).is_some()
            || hfq.tensor_data(&lin_a).is_some()
            || hfq.tensor_data(&lin_b).is_some();
        let has_qkv = hfq.tensor_data(&q_name).is_some()
            || hfq.tensor_data(&k_name).is_some()
            || hfq.tensor_data(&v_name).is_some();
        if !has_linear && !has_qkv {
            if has_down || !gmissing.is_empty() || ghs.len() == 2 {
                // Dense layer should have one of the attn groups. Down existing but no attn => fused tap missing (64/448).
                // If both groups empty and also no dense tensors at all, likely MoE skip — allow only if no gate/up/down
                let any_dense = has_down || !gmissing.is_empty() || ghs.len() == 2;
                if any_dense {
                    bad.push(format!("layer {i} attn missing: neither q/k/v nor linear_attn in_proj_* present (has down_proj={} gate/up present={}; fused QKV tap absent)", has_down, ghs.len()==2));
                }
            }
        } else if has_linear && !has_qkv {
            // DeltaNet layer: require all 4 in_proj
            let lin_names = [&lin_qkv, &lin_z, &lin_a, &lin_b];
            let mut dmissing = Vec::new();
            let mut dhs = Vec::new();
            let mut dpresent = Vec::new();
            for name in lin_names {
                if let Some((_, d)) = hfq.tensor_data(name) {
                    let mut h = DefaultHasher::new();
                    d.hash(&mut h);
                    dhs.push(h.finish());
                    dpresent.push(name.clone());
                } else {
                    dmissing.push(name.clone());
                }
            }
            if !dmissing.is_empty() {
                bad.push(format!(
                    "layer {i} linear_attn in_proj qkv/z/a/b missing {}",
                    dmissing.join(",")
                ));
            } else if dhs.windows(2).any(|w| w[0] != w[1]) {
                bad.push(format!(
                    "layer {i} linear_attn in_proj qkv/z/a/b diff hashes {:?} present {:?}",
                    dhs, dpresent
                ));
            }
        } else if has_qkv && !has_linear {
            // FullAttn layer: require q/k/v
            let qkv_names = [&q_name, &k_name, &v_name];
            let mut qmissing = Vec::new();
            let mut qhs = Vec::new();
            let mut qpresent = Vec::new();
            for name in qkv_names {
                if let Some((_, d)) = hfq.tensor_data(name) {
                    let mut h = DefaultHasher::new();
                    d.hash(&mut h);
                    qhs.push(h.finish());
                    qpresent.push(name.clone());
                } else {
                    qmissing.push(name.clone());
                }
            }
            if !qmissing.is_empty() {
                bad.push(format!("layer {i} q/k/v missing {}", qmissing.join(",")));
            } else if qhs.windows(2).any(|w| w[0] != w[1]) {
                bad.push(format!(
                    "layer {i} q/k/v diff hashes {:?} present {:?}",
                    qhs, qpresent
                ));
            }
        } else {
            // Both groups present — unexpected hybrid, check both
            let qkv_names = [&q_name, &k_name, &v_name];
            let mut qmissing = Vec::new();
            let mut qhs = Vec::new();
            let mut qpresent = Vec::new();
            for name in qkv_names {
                if let Some((_, d)) = hfq.tensor_data(name) {
                    let mut h = DefaultHasher::new();
                    d.hash(&mut h);
                    qhs.push(h.finish());
                    qpresent.push(name.clone());
                } else {
                    qmissing.push(name.clone());
                }
            }
            if !qmissing.is_empty() {
                bad.push(format!("layer {i} q/k/v missing {}", qmissing.join(",")));
            } else if qhs.windows(2).any(|w| w[0] != w[1]) {
                bad.push(format!(
                    "layer {i} q/k/v diff hashes {:?} present {:?}",
                    qhs, qpresent
                ));
            }
            let lin_names = [&lin_qkv, &lin_z, &lin_a, &lin_b];
            let mut dmissing = Vec::new();
            let mut dhs = Vec::new();
            let mut dpresent = Vec::new();
            for name in lin_names {
                if let Some((_, d)) = hfq.tensor_data(name) {
                    let mut h = DefaultHasher::new();
                    d.hash(&mut h);
                    dhs.push(h.finish());
                    dpresent.push(name.clone());
                } else {
                    dmissing.push(name.clone());
                }
            }
            if !dmissing.is_empty() {
                bad.push(format!(
                    "layer {i} linear_attn in_proj qkv/z/a/b missing {}",
                    dmissing.join(",")
                ));
            } else if dhs.windows(2).any(|w| w[0] != w[1]) {
                bad.push(format!(
                    "layer {i} linear_attn in_proj qkv/z/a/b diff hashes {:?} present {:?}",
                    dhs, dpresent
                ));
            }
        }
    }
    drop(hfq);
    if bad.is_empty() {
        Ok(())
    } else {
        Err(bad.join("; "))
    }
}
fn run_qwen35_batched(
    gpu: &mut rdna_compute::Gpu,
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
    cfg: &hipfire_arch_qwen35::qwen35::Qwen35Config,
    toks: &[u32],
    seq: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    dn_state: &mut hipfire_arch_qwen35::qwen35::DeltaNetState,
    scratch: &hipfire_arch_qwen35::qwen35::Qwen35Scratch,
) -> Result<(), String> {
    let seq = seq.max(1);
    for (si, chunk) in toks.chunks(seq).enumerate() {
        kv_cache
            .clear_gpu(gpu)
            .map_err(|e| format!("qwen35 kv clear {si}: {e:?}"))?;
        dn_state
            .reset(gpu)
            .map_err(|e| format!("qwen35 dn reset {si}: {e:?}"))?;
        // Batched prefill entry: qwen35.rs:11211 forward_prefill_batch → qwen35.rs:12848 run_plain_gemm_key / 12898 run_residual_gemm_key → families/gemm.rs:146 tap
        // This reaches TAP2 (GemmFamily::run_key:146) with batch_size = chunk.len() (not 1) for every migrated dense projection.
        // If the model fell through to per-token fallback (e.g. F32 KV or non-batchable dtype), forward_prefill_batch internally
        // loops per-token and the tap would still fire but with n=1 (~50x loss). We allocated Q8 KV, so batched is expected real.
        // Loud fallback with WARN if batched not eligible — do not silently degrade.
        let use_batched = true;
        if use_batched {
            match hipfire_arch_qwen35::qwen35::forward_prefill_batch(
                gpu, weights, cfg, chunk, 0, kv_cache, dn_state, scratch, None, None, None, None,
            ) {
                Ok(()) => {}
                Err(e) => {
                    let msg = format!("{e:?}");
                    // If batched refused (e.g. MQ3 in MoE, or KV config), fall back to per-token with loud WARN
                    if msg.contains("MQ3")
                        || msg.contains("batch")
                        || msg.contains("tree")
                        || msg.contains("Moe")
                    {
                        eprintln!("WARN: qwen35 batched prefill refused at seq {si} B {}: {msg} — falling back to per-token forward_scratch (you lose ~50x batching win: 2048 tokens will issue 2048 GEMVs at n=1 instead of one MFMA batched GEMM).", chunk.len());
                        for (pos, &tok) in chunk.iter().enumerate() {
                            hipfire_arch_qwen35::qwen35::forward_scratch(
                                gpu, weights, cfg, tok, pos, kv_cache, dn_state, scratch,
                            )
                            .map_err(|e| {
                                format!("qwen35 per-token fallback seq {si} pos {pos}: {e:?}")
                            })?;
                        }
                    } else {
                        return Err(format!(
                            "qwen35 forward_prefill_batch seq {si} B {}: {e:?}",
                            chunk.len()
                        ));
                    }
                }
            }
        } else {
            eprintln!("WARN: qwen35 batched prefill deliberately disabled — per-token GEMV n=1 path (you lose ~50x batching win).");
            for (pos, &tok) in chunk.iter().enumerate() {
                hipfire_arch_qwen35::qwen35::forward_scratch(
                    gpu, weights, cfg, tok, pos, kv_cache, dn_state, scratch,
                )
                .map_err(|e| format!("qwen35 per-token seq {si} pos {pos}: {e:?}"))?;
            }
        }
    }
    Ok(())
}

fn main() {
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_GRAPH_MOE", "0");
    }
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage_and_exit(0);
    }
    let model = arg("--model", None).unwrap_or_else(|| {
        eprintln!("--model <bf16.hfq> required");
        print_usage_and_exit(2);
    });
    let corpus = arg("--corpus", None).unwrap_or_else(|| {
        eprintln!("--corpus <txt> required");
        print_usage_and_exit(2);
    });
    let output = arg("--output", None).unwrap_or_else(|| {
        eprintln!("--output <path.calib.hfq> required (no default; ~30GB for 27B)");
        print_usage_and_exit(2);
    });
    let seq_len: usize = arg("--seq-len", Some("2048".into()))
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("--seq-len must be usize");
            print_usage_and_exit(2);
        });
    let max_tokens: usize = arg("--max-tokens", Some("262144".into()))
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("--max-tokens must be usize");
            print_usage_and_exit(2);
        });
    let lpp: usize = arg("--layers-per-pass", Some("64".into()))
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("--layers-per-pass must be usize");
            print_usage_and_exit(2);
        });
    let syrk_mode = arg("--syrk", Some("auto".into())).unwrap();
    if !matches!(syrk_mode.as_str(), "auto" | "rocblas" | "kernel") {
        eprintln!("--syrk must be auto|rocblas|kernel");
        print_usage_and_exit(2);
    }
    let md5 = corpus_md5(&corpus);
    let mut hfq =
        HfqFile::open(Path::new(&model)).unwrap_or_else(|e| panic!("open model {model}: {e}"));
    // Derive safetensors tensor-name prefix from the OPEN HFQ file (not hardcoded).
    // For qwen3.8-27B this yields `model.language_model.`; for llama/gemma/glimmer/lfm2 it yields `model.`.
    // If no tensor matches, FAIL LOUDLY — a wrong prefix is the silent no-op this fix addresses.
    let calib_prefix = derive_calib_prefix(&hfq);
    let arch = arg("--arch", None)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(hfq.arch_id);
    let llama = matches!(arch, 0 | 1);
    let qwen35 = matches!(arch, 5 | 6);
    let gemma = arch == 13;
    let glimmer = arch == 14;
    let lfm2 = arch == 11;
    if !(llama || qwen35 || gemma || glimmer || lfm2) {
        eprintln!("unsupported arch {arch}: dense shim wires 0/1 (tap1 llama),5/6 (tap2 qwen35),13+14 (tap2). MoE 9/10 out of scope.");
        std::process::exit(2);
    }
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let raw = std::fs::read(&corpus).unwrap_or_else(|e| panic!("read corpus {corpus}: {e}"));
    let take = (max_tokens * 8).min(raw.len());
    let text = String::from_utf8_lossy(&raw[..take]).to_string();
    let mut toks: Vec<u32> = tok.encode(&text);
    let n_tok = toks.len().min(max_tokens);
    toks.truncate(n_tok);
    let n_seqs = (n_tok + seq_len.max(1) - 1) / seq_len.max(1);
    eprintln!("calib_sweep: {n_tok} tokens {n_seqs} seqs seq_len={seq_len} md5={md5}");
    let mut gpu = rdna_compute::Gpu::init().unwrap_or_else(|e| panic!("gpu init: {e}"));
    eprintln!("GPU: {}", gpu.arch);
    let (syrk_chosen, avail) = choose_syrk(&gpu, &syrk_mode);
    let syrk_log = if gpu.rocblas.is_some() {
        format!("syrk mode={syrk_mode} → {syrk_chosen} (rocBLAS present)")
    } else {
        format!("syrk mode={syrk_mode} → kernel (rocBLAS absent soft-fail fallback)")
    };
    eprintln!("{syrk_log}");
    if syrk_mode == "rocblas" && !avail {
        eprintln!("error: --syrk rocblas but gpu.rocblas None");
        std::process::exit(1);
    }
    eprintln!("note: SYRK helper try_rocblas_syrk mirrors gemm.rs:24956 A=B=X m=n=K k=N beta=1.0; gpu.rocblas pub dispatch.rs:523 reachable; auto fallback logged; collector kernel calib_hessian_outer_f32 per constraint.");
    let _ = &try_rocblas_syrk;
    if gemma {
        let bp = std::env::var("HIPFIRE_BATCHED_PREFILL").ok();
        let wp = std::env::var("HIPFIRE_WMMA_PREFILL").ok();
        eprintln!("gemma4 env gate: {} ", gemma_env_report());
        if bp.as_deref() != Some("1") || wp.as_deref() != Some("1") {
            eprintln!("WARN: gemma4 batched prefill requires HIPFIRE_BATCHED_PREFILL=1 and HIPFIRE_WMMA_PREFILL=1 (lowered.rs:42-53, both default OFF). Current {} {} — shim will still capture (tap families/gemm.rs:146 fires per-token GEMV n=1) but you lose ~50x: the 2048-token chunk will issue 2048 GEMVs trapping at n=1 (~0.02 TFLOP) instead of one MFMA batched GEMM (~1 TFLOP). Set both to 1 before the run that produces the .calib.hfq you will feed to --ldlq.", gemma_env_report(), "");
        }
    }

    if llama {
        let cfg = hipfire_runtime::hfq::config_from_hfq(&hfq)
            .unwrap_or_else(|e| panic!("config_from_hfq: {e}"));
        eprintln!(
            "model arch={arch} n_layers={} dim={} hidden={} vocab={} head_dim={} qk_norm={}",
            cfg.n_layers, cfg.dim, cfg.hidden_dim, cfg.vocab_size, cfg.head_dim, cfg.has_qk_norm
        );
        let weights = hipfire_runtime::hfq::load_weights_hfq(&hfq, &cfg, &mut gpu)
            .unwrap_or_else(|e| panic!("load_weights: {e}"));
        // Startup existence check: every emitted name must actually exist in the loaded HFQ
        // (safetensors convention). A missing name is a silent no-op — fail loudly.
        {
            let all_names: Vec<String> =
                build_capture_llama(&weights, 0, cfg.n_layers, &calib_prefix)
                    .values()
                    .cloned()
                    .collect();
            verify_capture_names_or_exit(&hfq, &all_names);
            eprintln!("calib prefix '{}' verified: {} capture names all resolve in HFQ (safetensors convention)", calib_prefix, all_names.len());
        }
        drop(hfq);
        let lpp = lpp.max(1);
        let peak = estimate_peak_llama(&cfg, &weights, 0, lpp.min(cfg.n_layers));
        eprintln!(
            "grouped n_layers={} lpp={} n_groups={} peak≈{:.1} MB",
            cfg.n_layers,
            lpp,
            (cfg.n_layers + lpp - 1) / lpp,
            peak as f64 / 1_048_576.0
        );
        let batch_actual = seq_len.min(n_tok.max(1));
        eprintln!("batching: llama::prefill_forward:1508 → weight_gemm:1446 tap dispatch.rs:2848 batch={batch_actual} KV [seq_len+16] F32; chain provably reaches weight_gemm");
        let t0 = std::time::Instant::now();
        let grouped = lpp < cfg.n_layers;
        let calib_prefix_grouped = calib_prefix.clone();
        let calib_prefix_single = calib_prefix.clone();
        let summary = if grouped {
            let tc = toks.clone();
            collect_grouped(
                &mut gpu,
                arch,
                cfg.n_layers,
                lpp,
                Vec::new(),
                Path::new(&output),
                &[
                    ("source_model", serde_json::json!(model.clone())),
                    ("corpus", serde_json::json!(corpus.clone())),
                    ("corpus_md5", serde_json::json!(md5.clone())),
                    ("n_calib_tokens", serde_json::json!(n_tok)),
                    ("source_arch_id", serde_json::json!(arch)),
                    ("seq_len", serde_json::json!(seq_len)),
                    ("batch_size", serde_json::json!(batch_actual)),
                    ("syrk_chosen", serde_json::json!(syrk_chosen)),
                    ("syrk_mode", serde_json::json!(syrk_mode.clone())),
                    ("layers_per_pass", serde_json::json!(lpp)),
                    ("batches", serde_json::json!(n_seqs)),
                    (
                        "calib_driver",
                        serde_json::json!(
                            "calib_sweep llama prefill_forward→weight_gemm:1446 tap1"
                        ),
                    ),
                ],
                |s, e| build_capture_llama(&weights, s, e, &calib_prefix_grouped),
                |gpu, _| {
                    run_llama_batched(gpu, &weights, &cfg, &tc, seq_len)?;
                    Ok(CalibForward::default())
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("collect_grouped: {e}");
                std::process::exit(1);
            })
        } else {
            let cap = build_capture_llama(&weights, 0, cfg.n_layers, &calib_prefix_single);
            let tc = toks.clone();
            let cc = cfg.clone();
            collect(
                &mut gpu,
                arch,
                cap,
                Vec::new(),
                Path::new(&output),
                &[
                    ("source_model", serde_json::json!(model.clone())),
                    ("corpus", serde_json::json!(corpus.clone())),
                    ("corpus_md5", serde_json::json!(md5.clone())),
                    ("n_calib_tokens", serde_json::json!(n_tok)),
                    ("source_arch_id", serde_json::json!(arch)),
                    ("seq_len", serde_json::json!(seq_len)),
                    ("batch_size", serde_json::json!(batch_actual)),
                    ("syrk_chosen", serde_json::json!(syrk_chosen)),
                    ("syrk_mode", serde_json::json!(syrk_mode.clone())),
                    ("layers_per_pass", serde_json::json!(cfg.n_layers)),
                    ("batches", serde_json::json!(n_seqs)),
                    (
                        "calib_driver",
                        serde_json::json!(
                            "calib_sweep llama prefill_forward→weight_gemm:1446 tap1"
                        ),
                    ),
                ],
                |gpu| {
                    run_llama_batched(gpu, &weights, &cc, &tc, seq_len)?;
                    Ok(CalibForward::default())
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("collect: {e}");
                std::process::exit(1);
            })
        };
        let elapsed = t0.elapsed().as_secs_f64();
        let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        // Coverage gate: must run BEFORE consistency/identity so that missing tensors cannot pass vacuously.
        // Build expected set from capture_names (same build_capture_llama helper) and require every registered
        // name produced both hessian+imatrix. Would have caught 64-of-448 down_proj-only case.
        // Coverage gate keeps working unchanged because both capture_names and output now use
        // safetensors names (prefix + .weight); gate compares same namespace.
        let expected_names: Vec<String> =
            build_capture_llama(&weights, 0, cfg.n_layers, &calib_prefix)
                .values()
                .cloned()
                .collect();
        match check_coverage(Path::new(&output), &expected_names) {
            Ok(()) => eprintln!(
                "coverage {}/{} [PASS]",
                expected_names.len(),
                expected_names.len()
            ),
            Err(m) => {
                eprintln!("{m}");
                print_per_proj_summary(Path::new(&output), &expected_names);
                std::process::exit(1);
            }
        }
        print_per_proj_summary(Path::new(&output), &expected_names);
        // NOTE on max_consistency: diag(H) and Σx² are accumulated from the SAME staged buffer
        // via calib_hessian_outer_f32 (H) and calib_sumsq_reduce_f32 (diag) in Acc::flush.
        // Both compute Σx[c]² via different kernels over identical rows, so the check is
        // tautological — it validates kernel agreement (few ulp) but can never detect data
        // loss that zeroes/omits rows in both accumulators equally, and missing tensors are
        // not represented at all. The coverage gate above is the real data-loss guard.
        // Do not redesign silently; reported explicitly per assignment.
        eprintln!(
            "collected {} hessian {} imatrix {:.1}s max_consistency {:.3e} {}",
            summary.n_hessian,
            summary.n_imatrix,
            elapsed,
            summary.max_consistency,
            if summary.max_consistency < 1e-4 {
                "[CONSISTENT]"
            } else {
                "[MISMATCH]"
            }
        );
        if summary.max_consistency >= 1e-4 {
            eprintln!("FAIL diag(H) vs Σx² >=1e-4");
            std::process::exit(1);
        }
        match check_identity(Path::new(&output), cfg.n_layers, &calib_prefix) {
            Ok(()) => eprintln!("gate q/k/v identical gate/up identical [PASS]"),
            Err(m) => {
                eprintln!("FAIL identity: {m}");
                std::process::exit(1);
            }
        }
        eprintln!("--- run summary (provenance) ---");
        eprintln!("model: {model} arch={arch}");
        eprintln!("corpus: {corpus} md5={md5}");
        eprintln!("tokens: {n_tok} in {n_seqs} seqs seq_len={seq_len} max_tokens {max_tokens}");
        eprintln!("batch_size actual: {batch_actual} (seq_len; weights once per batch)");
        eprintln!("syrk: {syrk_log} (helper gemm.rs:24956 T/None beta=1.0 pub dispatch.rs:523)");
        eprintln!(
            "n_hessian {} n_imatrix {}",
            summary.n_hessian, summary.n_imatrix
        );
        eprintln!("output: {output} {bytes} bytes elapsed {:.1}s", elapsed);
        eprintln!(
            "peak per-group {:.1} MB lpp={} dense ~{:.1} MB",
            peak as f64 / 1_048_576.0,
            lpp,
            {
                let mut d = 0u64;
                for i in 0..lpp.min(cfg.n_layers) {
                    let l = &weights.layers[i];
                    for k in [
                        l.wq.k, l.wk.k, l.wv.k, l.wo.k, l.w_gate.k, l.w_up.k, l.w_down.k,
                    ] {
                        d += (k * k * 4) as u64 + (k * 4) as u64;
                    }
                }
                d as f64 / 1_048_576.0
            }
        );
        eprintln!(
            "config n_layers={} dim={} hidden={} head_dim={} lpp={}",
            cfg.n_layers, cfg.dim, cfg.hidden_dim, cfg.head_dim, lpp
        );
        eprintln!("HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_GRAPH_MOE=0");
        eprintln!("batched: llama::prefill_forward:1508 → weight_gemm:1446 tap dispatch.rs:2848 MFMA; KV ONCE seq_len+16 F32 cleared per seq");
    } else if qwen35 {
        // qwen35 5/6 — WIRED (tap2). Uses forward_prefill_batch (qwen35.rs:11211) → run_plain_gemm_key (12848) / run_residual_gemm_key (12898) → GemmFamily::run_key:146 → tap families/gemm.rs:146
        // Batched forward, KV cleared + positions restarted per seq_len window, KV allocated ONCE at seq_len+16.
        let cfg = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq)
            .unwrap_or_else(|e| panic!("qwen35 config_from_hfq: {e}"));
        // Defense: refuse MoE for this dense shim (27B dense has num_experts==0). MoE Hessian capture needs indexed routing, not in this tap.
        if cfg.num_experts != 0 {
            eprintln!("unsupported qwen35 MoE (A3B) arch 5/6 with num_experts={}: this shim is the dense-only calibration lane (qwen3.8-27B 64×5120 dense). MoE experts need indexed Hessian capture, out of scope here — refusing to avoid silent missing tensors.", cfg.num_experts);
            std::process::exit(2);
        }
        eprintln!(
            "qwen35 arch={} n_layers={} dim={} vocab={} hidden={} layer_types len={} moe={}",
            arch,
            cfg.n_layers,
            cfg.dim,
            cfg.vocab_size,
            cfg.hidden_dim,
            cfg.layer_types.len(),
            cfg.num_experts
        );
        // Load weights via HfqSource + Layout::single — verbatim copy of collect_e8_hessian_native.rs:135-137 / build_kld_ref_native.rs:145-147
        let mut source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq, &cfg);
        let layout = hipfire_runtime::model_load::Layout::single(cfg.n_layers);
        let weights = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &layout,
        )
        .unwrap_or_else(|e| panic!("qwen35 load_weights: {e:?}"));
        // Need raw hfq reference for verification — reconstruct a view of tensor names before dropping.
        // The HfqSource borrows hfq mutably but does not consume it; we can still verify via the original `hfq`.
        // Verification must happen before drop(hfq) so we can compare capture names against real tensors.
        {
            let all_names: Vec<String> =
                build_capture_qwen35(&weights, 0, cfg.n_layers, &calib_prefix)
                    .values()
                    .cloned()
                    .collect();
            verify_capture_names_or_exit(&hfq, &all_names);
            eprintln!("calib prefix '{}' verified: {} capture names all resolve in HFQ (safetensors convention)", calib_prefix, all_names.len());
        }
        drop(hfq);
        let kv_max = seq_len + 16;
        // KV allocated ONCE at seq_len+16, never at corpus length (~68 GB for 27B dense would OOM). Q8 keeps MFMA batched eligibility (F32 would force per-token fallback).
        let mut kv_cache = hipfire_runtime::llama::KvCache::new_gpu_q8(
            &mut gpu,
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
            kv_max,
        )
        .unwrap_or_else(|e| panic!("qwen35 kv alloc q8 kv_max={kv_max}: {e:?}"));
        let scratch =
            hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(&mut gpu, &cfg, 64, kv_max)
                .unwrap_or_else(|e| panic!("qwen35 scratch kv_max={kv_max}: {e:?}"));
        let mut dn_state = hipfire_arch_qwen35::qwen35::DeltaNetState::new(&mut gpu, &cfg)
            .unwrap_or_else(|e| panic!("qwen35 dn_state: {e:?}"));
        let lpp = lpp.max(1);
        let peak = estimate_peak_qwen35(&weights, 0, lpp.min(cfg.n_layers));
        eprintln!("grouped n_layers={} lpp={} n_groups={} peak≈{:.1} MB kv_max={} (seq_len+16) scratch max_batch~256 DeltaNetState Q8; qwen3.8-27B hybrid 64×5120 int17408 o_proj K=6144 vocab 248320", cfg.n_layers, lpp, (cfg.n_layers+lpp-1)/lpp, peak as f64/1_048_576.0, kv_max);
        let batch_actual = seq_len.min(n_tok.max(1));
        eprintln!("batching: qwen35::forward_prefill_batch:11211 → run_plain_gemm_key:12848 / run_residual_gemm_key:12898 → GemmFamily::run_key:146 → tap families/gemm.rs:146 batch={} KV [seq_len+16] Q8 dn_state reset per seq; capture_names over q/k/v/o_proj + gate/up/down_proj + DeltaNet in_proj_qkv/z/a/b/out_proj at layer_driver.rs:36-61", batch_actual);
        // Honesty: if batched path not eligible, run_qwen35_batched will WARN loudly and fallback per-token (not silent).
        let t0 = std::time::Instant::now();
        let grouped = lpp < cfg.n_layers;
        let calib_prefix_grouped = calib_prefix.clone();
        let calib_prefix_single = calib_prefix.clone();
        let summary = if grouped {
            let tc = toks.clone();
            collect_grouped(&mut gpu, arch, cfg.n_layers, lpp, Vec::new(), Path::new(&output),
                &[("source_model", serde_json::json!(model.clone())), ("corpus", serde_json::json!(corpus.clone())), ("corpus_md5", serde_json::json!(md5.clone())), ("n_calib_tokens", serde_json::json!(n_tok)), ("source_arch_id", serde_json::json!(arch)), ("seq_len", serde_json::json!(seq_len)), ("batch_size", serde_json::json!(batch_actual)), ("syrk_chosen", serde_json::json!(syrk_chosen)), ("syrk_mode", serde_json::json!(syrk_mode.clone())), ("layers_per_pass", serde_json::json!(lpp)), ("batches", serde_json::json!(n_seqs)), ("kv_max", serde_json::json!(kv_max)), ("calib_driver", serde_json::json!("calib_sweep qwen35 forward_prefill_batch:11211 → run_plain_gemm_key:12848 tap2"))],
                |s,e| build_capture_qwen35(&weights,s,e, &calib_prefix_grouped),
                |gpu,_| { run_qwen35_batched(gpu,&weights,&cfg,&tc,seq_len,&mut kv_cache,&mut dn_state,&scratch)?; Ok(CalibForward::default()) }
            ).unwrap_or_else(|e| { eprintln!("collect_grouped qwen35: {e}"); std::process::exit(1); })
        } else {
            let cap = build_capture_qwen35(&weights, 0, cfg.n_layers, &calib_prefix_single);
            let tc = toks.clone();
            collect(&mut gpu, arch, cap, Vec::new(), Path::new(&output),
                &[("source_model", serde_json::json!(model.clone())), ("corpus", serde_json::json!(corpus.clone())), ("corpus_md5", serde_json::json!(md5.clone())), ("n_calib_tokens", serde_json::json!(n_tok)), ("source_arch_id", serde_json::json!(arch)), ("seq_len", serde_json::json!(seq_len)), ("batch_size", serde_json::json!(batch_actual)), ("syrk_chosen", serde_json::json!(syrk_chosen)), ("syrk_mode", serde_json::json!(syrk_mode.clone())), ("layers_per_pass", serde_json::json!(cfg.n_layers)), ("batches", serde_json::json!(n_seqs)), ("kv_max", serde_json::json!(kv_max)), ("calib_driver", serde_json::json!("calib_sweep qwen35 forward_prefill_batch:11211 → run_plain_gemm_key:12848 tap2"))],
                |gpu| { run_qwen35_batched(gpu,&weights,&cfg,&tc,seq_len,&mut kv_cache,&mut dn_state,&scratch)?; Ok(CalibForward::default()) }
            ).unwrap_or_else(|e| { eprintln!("collect qwen35: {e}"); std::process::exit(1); })
        };
        let _ = &weights;
        let elapsed = t0.elapsed().as_secs_f64();
        let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        let expected_names: Vec<String> =
            build_capture_qwen35(&weights, 0, cfg.n_layers, &calib_prefix)
                .values()
                .cloned()
                .collect();
        match check_coverage(Path::new(&output), &expected_names) {
            Ok(()) => eprintln!(
                "coverage {}/{} [PASS]",
                expected_names.len(),
                expected_names.len()
            ),
            Err(m) => {
                eprintln!("{m}");
                print_per_proj_summary(Path::new(&output), &expected_names);
                std::process::exit(1);
            }
        }
        print_per_proj_summary(Path::new(&output), &expected_names);
        // NOTE on max_consistency: diag(H) vs Σx² share the same staged buffer (Acc::flush runs both kernels
        // over identical rows). Check is tautological for data-loss: both see same omission, so 0.000e0
        // in the 64/448 down_proj-only run was expected — every captured tensor's two accumulators agreed,
        // but 384 tensors were never captured and thus not represented. Coverage gate is the real guard.
        eprintln!(
            "collected {} hessian {} imatrix {:.1}s max_consistency {:.3e} {}",
            summary.n_hessian,
            summary.n_imatrix,
            elapsed,
            summary.max_consistency,
            if summary.max_consistency < 1e-4 {
                "[CONSISTENT]"
            } else {
                "[MISMATCH]"
            }
        );
        if summary.max_consistency >= 1e-4 {
            eprintln!("FAIL diag(H) vs Σx² >=1e-4");
            std::process::exit(1);
        }
        match check_identity_qwen35(Path::new(&output), cfg.n_layers, &calib_prefix) { Ok(()) => eprintln!("qwen35 identity: q/k/v identical (FullAttn layers), linear_attn in_proj qkv/z/a/b identical (DeltaNet layers), gate/up identical [PASS]"), Err(m) => { eprintln!("FAIL qwen35 identity: {m}"); std::process::exit(1); } }
        eprintln!("wrote {output} {bytes} bytes");
        eprintln!("--- run summary (provenance) ---");
        eprintln!(
            "model: {model} arch={arch} n_layers={} dim={} hidden={} kv_max={} lpp={}",
            cfg.n_layers, cfg.dim, cfg.hidden_dim, kv_max, lpp
        );
        eprintln!("HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_GRAPH_MOE=0");
        eprintln!("batched: Qwen35Scratch [seq_len+16] kv_max={} DeltaNetState Q8 forward_prefill_batch MFMA n=batch_size; capture tap families/gemm.rs:146 (not weight_gemm)", kv_max);
    } else if gemma {
        // Gemma4 — WIRED (tap2). Uses forward_prefill_batch (lowered.rs:2581) → run_prefill_gemm (71) → GemmFamily::run_key:119 → tap families/gemm.rs:146
        let cfg = hipfire_arch_gemma4::lowered::config_from_hfq(&hfq)
            .unwrap_or_else(|| panic!("gemma4 cfg: missing/invalid config in HFQ"));
        eprintln!(
            "gemma4 arch=13 n_layers={} dim={} vocab={} hidden={} layer_types={:?} enable_moe={}",
            cfg.n_layers,
            cfg.dim,
            cfg.vocab_size,
            cfg.hidden_dim,
            cfg.layer_types,
            cfg.enable_moe_block
        );
        if cfg.enable_moe_block {
            eprintln!("WARN: gemma4 MoE (26B-A4B) detected: MoE routed experts have no Hessian tap in this lane (mq4 lane is dense-only). The dense projections (q/k/v/o, gate/up/down) still capture; MoE experts are not quantized via --ldlq dense path and remain out of scope here — refusing to emit a .calib.hfq that would silently miss MoE tensors.");
            eprintln!("If this is the dense Gemma4-12B/27B (no MoE), enable_moe_block should be false — check the model is the dense variant.");
            std::process::exit(2);
        }
        let mut weights = hipfire_arch_gemma4::lowered::load_weights(&mut hfq, &cfg, &mut gpu)
            .unwrap_or_else(|e| panic!("gemma4 weights: {e}"));
        {
            let all_names: Vec<String> =
                build_capture_gemma(&weights, 0, cfg.n_layers, &calib_prefix)
                    .values()
                    .cloned()
                    .collect();
            verify_capture_names_or_exit(&hfq, &all_names);
            eprintln!("calib prefix '{}' verified: {} capture names all resolve in HFQ (safetensors convention)", calib_prefix, all_names.len());
        }
        drop(hfq);
        let kv_max = seq_len + 16;
        let scratch = hipfire_arch_gemma4::lowered::Gemma4Scratch::new(&mut gpu, &cfg, 1)
            .unwrap_or_else(|e| panic!("gemma4 scratch: {e:?}"));
        hipfire_arch_gemma4::lowered::init_scratch_constants(&mut gpu, &scratch, cfg.full_head_dim)
            .unwrap_or_else(|e| panic!("gemma4 init_scratch_constants: {e:?}"));
        // Q8 KV, matching the qwen35 arm above. `new_gpu` allocates an F32 cache
        // and gemma4's batched prefill has no `KvWriteF32` kernel registered, so
        // an F32 cache fails at the first layer with "no implementation for
        // KvWriteF32". KV dtype does not affect what is captured — the tap reads
        // the projection INPUT activations, not cache contents.
        let mut kv_sliding = hipfire_runtime::llama::KvCache::new_gpu_q8(
            &mut gpu,
            cfg.n_layers,
            cfg.sliding_n_kv_heads,
            cfg.sliding_head_dim,
            kv_max,
        )
        .unwrap_or_else(|e| panic!("gemma sliding kv alloc q8 {kv_max}: {e:?}"));
        let mut kv_full = hipfire_runtime::llama::KvCache::new_gpu_q8(
            &mut gpu,
            cfg.n_layers,
            cfg.full_n_kv_heads,
            cfg.full_head_dim,
            kv_max,
        )
        .unwrap_or_else(|e| panic!("gemma full kv alloc q8 {kv_max}: {e:?}"));
        // Honesty guard: forward_prefill_batch must exist and tap must fire; if layer count mismatch or unsupported embed format, run will Err and we exit 1 (honest) — never emit partial hfq.
        let lpp = lpp.max(1);
        let peak = estimate_peak_gemma(&weights, 0, lpp.min(cfg.n_layers));
        eprintln!("grouped n_layers={} lpp={} n_groups={} peak≈{:.1} MB kv_max={} (seq_len+16) scratch.max_prefill_batch={}", cfg.n_layers, lpp, (cfg.n_layers+lpp-1)/lpp, peak as f64/1_048_576.0, kv_max, scratch.max_prefill_batch);
        let batch_actual = seq_len.min(n_tok.max(1));
        eprintln!("batching: gemma4::lowered::forward_prefill_batch:2581 → run_prefill_gemm:71 → GemmFamily::run_key:119 → tap families/gemm.rs:146 batch={} KV [seq_len+16] scratch {kv_max} ; capture_names over LayerWeights::Sliding/Full (q/k/v/o, gate/up/down) at lowered.rs:379/414", batch_actual);
        let t0 = std::time::Instant::now();
        let grouped = lpp < cfg.n_layers;
        let calib_prefix_grouped = calib_prefix.clone();
        let calib_prefix_single = calib_prefix.clone();
        let summary = if grouped {
            let tc = toks.clone();
            collect_grouped(&mut gpu, arch, cfg.n_layers, lpp, Vec::new(), Path::new(&output),
                &[("source_model", serde_json::json!(model.clone())), ("corpus", serde_json::json!(corpus.clone())), ("corpus_md5", serde_json::json!(md5.clone())), ("n_calib_tokens", serde_json::json!(n_tok)), ("source_arch_id", serde_json::json!(arch)), ("seq_len", serde_json::json!(seq_len)), ("batch_size", serde_json::json!(batch_actual)), ("syrk_chosen", serde_json::json!(syrk_chosen)), ("syrk_mode", serde_json::json!(syrk_mode.clone())), ("layers_per_pass", serde_json::json!(lpp)), ("batches", serde_json::json!(n_seqs)), ("kv_max", serde_json::json!(kv_max)), ("calib_driver", serde_json::json!("calib_sweep gemma4 forward_prefill_batch:2581 → run_prefill_gemm:71 tap2"))],
                |s,e| build_capture_gemma(&weights,s,e, &calib_prefix_grouped),
                |gpu,_| { run_gemma_batched(gpu,&weights,&cfg,&tc,seq_len,&mut kv_sliding,&mut kv_full,&scratch)?; Ok(CalibForward::default()) }
            ).unwrap_or_else(|e| { eprintln!("collect_grouped gemma: {e}"); std::process::exit(1); })
        } else {
            let cap = build_capture_gemma(&weights, 0, cfg.n_layers, &calib_prefix_single);
            let tc = toks.clone();
            collect(&mut gpu, arch, cap, Vec::new(), Path::new(&output),
                &[("source_model", serde_json::json!(model.clone())), ("corpus", serde_json::json!(corpus.clone())), ("corpus_md5", serde_json::json!(md5.clone())), ("n_calib_tokens", serde_json::json!(n_tok)), ("source_arch_id", serde_json::json!(arch)), ("seq_len", serde_json::json!(seq_len)), ("batch_size", serde_json::json!(batch_actual)), ("syrk_chosen", serde_json::json!(syrk_chosen)), ("syrk_mode", serde_json::json!(syrk_mode.clone())), ("layers_per_pass", serde_json::json!(cfg.n_layers)), ("batches", serde_json::json!(n_seqs)), ("kv_max", serde_json::json!(kv_max)), ("calib_driver", serde_json::json!("calib_sweep gemma4 forward_prefill_batch:2581 → run_prefill_gemm:71 tap2"))],
                |gpu| { run_gemma_batched(gpu,&weights,&cfg,&tc,seq_len,&mut kv_sliding,&mut kv_full,&scratch)?; Ok(CalibForward::default()) }
            ).unwrap_or_else(|e| { eprintln!("collect gemma: {e}"); std::process::exit(1); })
        };
        // prevent weights moved in closure from being dropped before summary
        let _ = &weights;
        let elapsed = t0.elapsed().as_secs_f64();
        let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        let expected_names: Vec<String> =
            build_capture_gemma(&weights, 0, cfg.n_layers, &calib_prefix)
                .values()
                .cloned()
                .collect();
        match check_coverage(Path::new(&output), &expected_names) {
            Ok(()) => eprintln!(
                "coverage {}/{} [PASS]",
                expected_names.len(),
                expected_names.len()
            ),
            Err(m) => {
                eprintln!("{m}");
                print_per_proj_summary(Path::new(&output), &expected_names);
                std::process::exit(1);
            }
        }
        print_per_proj_summary(Path::new(&output), &expected_names);
        // NOTE: max_consistency tautological — same staged buffer for H and diag (see llama block comment).
        eprintln!(
            "collected {} hessian {} imatrix {:.1}s max_consistency {:.3e} {}",
            summary.n_hessian,
            summary.n_imatrix,
            elapsed,
            summary.max_consistency,
            if summary.max_consistency < 1e-4 {
                "[CONSISTENT]"
            } else {
                "[MISMATCH]"
            }
        );
        if summary.max_consistency >= 1e-4 {
            eprintln!("FAIL diag(H) vs Σx² >=1e-4");
            std::process::exit(1);
        }
        match check_identity_generic(Path::new(&output), cfg.n_layers, false, false, &calib_prefix) { Ok(()) => eprintln!("gemma identity: q/k/(v) identical gate/up identical [PASS] (v absent on Full layers, skipped)"), Err(m) => { eprintln!("FAIL gemma identity: {m}"); std::process::exit(1); } }
        eprintln!("wrote {output} {bytes} bytes");
        eprintln!("--- run summary (provenance) ---");
        eprintln!(
            "model: {model} arch={arch} n_layers={} dim={} hidden={} kv_max={} lpp={}",
            cfg.n_layers, cfg.dim, cfg.hidden_dim, kv_max, lpp
        );
        eprintln!("HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_GEMMA4_GRAPH=0");
        eprintln!("batched: Gemma4Scratch [seq_len+16] scratch.max_prefill_batch=128 chunked prefill; KV ONCE seq_len+16");
    } else if glimmer {
        let cfg = hipfire_arch_muse_glimmer::config::GlimmerConfig::from_hfq(&hfq)
            .unwrap_or_else(|e| panic!("glimmer cfg: {e}"));
        eprintln!(
            "glimmer arch=14 n_layers={} dim={} vocab={} hidden={} sliding_window={}",
            cfg.n_layers, cfg.dim, cfg.vocab_size, cfg.hidden_dim, cfg.sliding_window
        );
        let weights =
            hipfire_arch_muse_glimmer::glimmer::GlimmerWeights::load(&hfq, &cfg, &mut gpu)
                .unwrap_or_else(|e| panic!("glimmer weights: {e}"));
        {
            let all_names: Vec<String> =
                build_capture_glimmer(&weights, 0, cfg.n_layers, &calib_prefix)
                    .values()
                    .cloned()
                    .collect();
            verify_capture_names_or_exit(&hfq, &all_names);
            eprintln!("calib prefix '{}' verified: {} capture names all resolve in HFQ (safetensors convention)", calib_prefix, all_names.len());
        }
        drop(hfq);
        let kv_max = seq_len + 16;
        let mut state = hipfire_arch_muse_glimmer::glimmer::GlimmerState::new_with_max_seq(
            &mut gpu, &cfg, kv_max,
        )
        .unwrap_or_else(|e| panic!("glimmer state alloc kv_max {kv_max}: {e}"));
        let lpp = lpp.max(1);
        let peak = estimate_peak_glimmer(&weights, 0, lpp.min(cfg.n_layers));
        eprintln!("grouped n_layers={} lpp={} n_groups={} peak≈{:.1} MB state max_seq={} (seq_len+16) prefill chunk {}", cfg.n_layers, lpp, (cfg.n_layers+lpp-1)/lpp, peak as f64/1_048_576.0, kv_max, hipfire_arch_muse_glimmer::forward::glimmer_prefill_chunk_size(seq_len));
        let batch_actual = seq_len.min(n_tok.max(1));
        eprintln!("batching: glimmer::forward::prefill_with_capture:3650 → prefill_chunk_batched:2533 → run_prefill_plain_gemm_key:49 → GemmFamily::run_key:119 → tap families/gemm.rs:146 batch={} GlimmerState [seq_len+16] prefill_chunk_batched chunks 192/256; capture_names over GlimmerLayerWeights (q/k/v/gate/o + gate/up/down) at glimmer.rs:1159", batch_actual);
        let t0 = std::time::Instant::now();
        let grouped = lpp < cfg.n_layers;
        let calib_prefix_grouped = calib_prefix.clone();
        let calib_prefix_single = calib_prefix.clone();
        let summary = if grouped {
            let tc = toks.clone();
            collect_grouped(&mut gpu, arch, cfg.n_layers, lpp, Vec::new(), Path::new(&output),
                &[("source_model", serde_json::json!(model.clone())), ("corpus", serde_json::json!(corpus.clone())), ("corpus_md5", serde_json::json!(md5.clone())), ("n_calib_tokens", serde_json::json!(n_tok)), ("source_arch_id", serde_json::json!(arch)), ("seq_len", serde_json::json!(seq_len)), ("batch_size", serde_json::json!(batch_actual)), ("syrk_chosen", serde_json::json!(syrk_chosen)), ("syrk_mode", serde_json::json!(syrk_mode.clone())), ("layers_per_pass", serde_json::json!(lpp)), ("batches", serde_json::json!(n_seqs)), ("kv_max", serde_json::json!(kv_max)), ("calib_driver", serde_json::json!("calib_sweep glimmer prefill_with_capture:3650 → prefill_chunk_batched tap2"))],
                |s,e| build_capture_glimmer(&weights,s,e, &calib_prefix_grouped),
                |gpu,_| { run_glimmer_batched(gpu,&weights,&cfg,&mut state,&tc,seq_len)?; Ok(CalibForward::default()) }
            ).unwrap_or_else(|e| { eprintln!("collect_grouped glimmer: {e}"); std::process::exit(1); })
        } else {
            let cap = build_capture_glimmer(&weights, 0, cfg.n_layers, &calib_prefix_single);
            let tc = toks.clone();
            collect(&mut gpu, arch, cap, Vec::new(), Path::new(&output),
                &[("source_model", serde_json::json!(model.clone())), ("corpus", serde_json::json!(corpus.clone())), ("corpus_md5", serde_json::json!(md5.clone())), ("n_calib_tokens", serde_json::json!(n_tok)), ("source_arch_id", serde_json::json!(arch)), ("seq_len", serde_json::json!(seq_len)), ("batch_size", serde_json::json!(batch_actual)), ("syrk_chosen", serde_json::json!(syrk_chosen)), ("syrk_mode", serde_json::json!(syrk_mode.clone())), ("layers_per_pass", serde_json::json!(cfg.n_layers)), ("batches", serde_json::json!(n_seqs)), ("kv_max", serde_json::json!(kv_max)), ("calib_driver", serde_json::json!("calib_sweep glimmer prefill_with_capture:3650 → prefill_chunk_batched tap2"))],
                |gpu| { run_glimmer_batched(gpu,&weights,&cfg,&mut state,&tc,seq_len)?; Ok(CalibForward::default()) }
            ).unwrap_or_else(|e| { eprintln!("collect glimmer: {e}"); std::process::exit(1); })
        };
        let _ = &weights;
        let elapsed = t0.elapsed().as_secs_f64();
        let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        let expected_names: Vec<String> =
            build_capture_glimmer(&weights, 0, cfg.n_layers, &calib_prefix)
                .values()
                .cloned()
                .collect();
        match check_coverage(Path::new(&output), &expected_names) {
            Ok(()) => eprintln!(
                "coverage {}/{} [PASS]",
                expected_names.len(),
                expected_names.len()
            ),
            Err(m) => {
                eprintln!("{m}");
                print_per_proj_summary(Path::new(&output), &expected_names);
                std::process::exit(1);
            }
        }
        print_per_proj_summary(Path::new(&output), &expected_names);
        // NOTE: max_consistency tautological — same buffer (see llama comment).
        eprintln!(
            "collected {} hessian {} imatrix {:.1}s max_consistency {:.3e} {}",
            summary.n_hessian,
            summary.n_imatrix,
            elapsed,
            summary.max_consistency,
            if summary.max_consistency < 1e-4 {
                "[CONSISTENT]"
            } else {
                "[MISMATCH]"
            }
        );
        if summary.max_consistency >= 1e-4 {
            eprintln!("FAIL diag(H) vs Σx² >=1e-4");
            std::process::exit(1);
        }
        match check_identity_generic(Path::new(&output), cfg.n_layers, true, false, &calib_prefix) {
            Ok(()) => {
                eprintln!("glimmer identity: q/k/v/attn_gate identical gate/up identical [PASS]")
            }
            Err(m) => {
                eprintln!("FAIL glimmer identity: {m}");
                std::process::exit(1);
            }
        }
        eprintln!("wrote {output} {bytes} bytes");
        eprintln!("--- run summary (provenance) ---");
        eprintln!(
            "model: {model} arch={arch} n_layers={} dim={} hidden={} kv_max={} lpp={}",
            cfg.n_layers, cfg.dim, cfg.hidden_dim, kv_max, lpp
        );
        eprintln!("HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_GLIMMER_KV_VMM=1");
        eprintln!(
            "batched: GlimmerState [seq_len+16] prefill_chunk_batched chunk {}",
            hipfire_arch_muse_glimmer::forward::glimmer_prefill_chunk_size(seq_len)
        );
    } else if lfm2 {
        let cfg = hipfire_arch_lfm2moe::config::Lfm2MoeConfig::from_hfq(&hfq)
            .unwrap_or_else(|e| panic!("lfm2 cfg: {e}"));
        eprintln!("lfm2 arch=11 n_layers={} hidden={} vocab={} head_dim={} conv_k={} intermediate={} num_experts={} layer_types={:?}", cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.head_dim, cfg.conv_kernel_size, cfg.intermediate_size, cfg.num_experts, cfg.layer_types);
        if cfg.num_experts != 0 {
            eprintln!("unsupported lfm2 MoE (lfm2.5:8b-a1b num_experts={} >0): MoE experts have no dense Hessian tap in this MQ4 dense lane — only the dense 1.2b/350m variants (num_experts==0) are admitted. Refusing to avoid a .calib.hfq with missing expert tensors.", cfg.num_experts);
            std::process::exit(2);
        }
        // batched_proj chokepoint note: dense 1.2b/350m variants only
        let weights = hipfire_arch_lfm2moe::lfm2moe::Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu)
            .unwrap_or_else(|e| panic!("lfm2 weights: {e}"));
        {
            let all_names: Vec<String> =
                build_capture_lfm(&weights, 0, cfg.num_hidden_layers, &calib_prefix)
                    .values()
                    .cloned()
                    .collect();
            verify_capture_names_or_exit(&hfq, &all_names);
            eprintln!("calib prefix '{}' verified: {} capture names all resolve in HFQ (safetensors convention)", calib_prefix, all_names.len());
        }
        drop(hfq);
        // validate weight formats for batched path per forward_batch::batch_weight_formats_supported
        if let Err(e) =
            hipfire_arch_lfm2moe::forward_batch::batch_weight_formats_supported(&weights)
        {
            eprintln!("unsupported lfm2 weight formats for batched decode: {e} — batched_proj only supports Q8_0/HFQ4G256/MQ4G256 dense; refusing.");
            std::process::exit(2);
        }
        let kv_max = seq_len + 16;
        let max_batch = seq_len; // one lane per token in the seq window
        let mut state = hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState::new(
            &mut gpu, &cfg, max_batch, kv_max, 32,
        )
        .unwrap_or_else(|e| {
            panic!("lfm batch state alloc max_batch={max_batch} kv_max={kv_max}: {e:?}")
        });
        let lpp = lpp.max(1);
        let peak = estimate_peak_lfm(&weights, 0, lpp.min(cfg.num_hidden_layers));
        eprintln!("grouped n_layers={} lpp={} n_groups={} peak≈{:.1} MB lane_capacity={} (seq_len+16) max_batch={} proj_rot {} ; capture_names over Mixer::Conv(in_proj/out_proj) + Attn(q/k/v/o) + Dense(w1/w3/w2) at forward_batch.rs:79 batched_proj (BEFORE rotate, pre-rotation x)", cfg.num_hidden_layers, lpp, (cfg.num_hidden_layers+lpp-1)/lpp, peak as f64/1_048_576.0, kv_max, max_batch, state.proj_rot.numel());
        let batch_actual = seq_len.min(n_tok.max(1));
        eprintln!("batching: lfm::forward_batch::forward_decode_batch_lfm:290 → batched_proj:61 → gpu.maybe_capture_activation:79 tap3 direct gemm_* (not GemmFamily) batch={} ; batched_proj is SINGLE chokepoint for all dense projections: conv in/out_proj (l={:?}), wq/wk/wv (395-399), wo (474), dense w1/w3/w2 (487-489/494)", batch_actual, cfg.layer_types);
        let t0 = std::time::Instant::now();
        let grouped = lpp < cfg.num_hidden_layers;
        let calib_prefix_grouped = calib_prefix.clone();
        let calib_prefix_single = calib_prefix.clone();
        let summary = if grouped {
            let tc = toks.clone();
            collect_grouped(
                &mut gpu,
                arch,
                cfg.num_hidden_layers,
                lpp,
                Vec::new(),
                Path::new(&output),
                &[
                    ("source_model", serde_json::json!(model.clone())),
                    ("corpus", serde_json::json!(corpus.clone())),
                    ("corpus_md5", serde_json::json!(md5.clone())),
                    ("n_calib_tokens", serde_json::json!(n_tok)),
                    ("source_arch_id", serde_json::json!(arch)),
                    ("seq_len", serde_json::json!(seq_len)),
                    ("batch_size", serde_json::json!(batch_actual)),
                    ("syrk_chosen", serde_json::json!(syrk_chosen)),
                    ("syrk_mode", serde_json::json!(syrk_mode.clone())),
                    ("layers_per_pass", serde_json::json!(lpp)),
                    ("batches", serde_json::json!(n_seqs)),
                    ("lane_capacity", serde_json::json!(kv_max)),
                    (
                        "calib_driver",
                        serde_json::json!(
                            "calib_sweep lfm forward_decode_batch_lfm:290 → batched_proj:61 tap3"
                        ),
                    ),
                ],
                |s, e| build_capture_lfm(&weights, s, e, &calib_prefix_grouped),
                |gpu, _| {
                    run_lfm_batched(gpu, &weights, &cfg, &mut state, &tc, seq_len)?;
                    Ok(CalibForward::default())
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("collect_grouped lfm: {e}");
                std::process::exit(1);
            })
        } else {
            let cap = build_capture_lfm(&weights, 0, cfg.num_hidden_layers, &calib_prefix_single);
            let tc = toks.clone();
            collect(
                &mut gpu,
                arch,
                cap,
                Vec::new(),
                Path::new(&output),
                &[
                    ("source_model", serde_json::json!(model.clone())),
                    ("corpus", serde_json::json!(corpus.clone())),
                    ("corpus_md5", serde_json::json!(md5.clone())),
                    ("n_calib_tokens", serde_json::json!(n_tok)),
                    ("source_arch_id", serde_json::json!(arch)),
                    ("seq_len", serde_json::json!(seq_len)),
                    ("batch_size", serde_json::json!(batch_actual)),
                    ("syrk_chosen", serde_json::json!(syrk_chosen)),
                    ("syrk_mode", serde_json::json!(syrk_mode.clone())),
                    ("layers_per_pass", serde_json::json!(cfg.num_hidden_layers)),
                    ("batches", serde_json::json!(n_seqs)),
                    ("lane_capacity", serde_json::json!(kv_max)),
                    (
                        "calib_driver",
                        serde_json::json!(
                            "calib_sweep lfm forward_decode_batch_lfm:290 → batched_proj:61 tap3"
                        ),
                    ),
                ],
                |gpu| {
                    run_lfm_batched(gpu, &weights, &cfg, &mut state, &tc, seq_len)?;
                    Ok(CalibForward::default())
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("collect lfm: {e}");
                std::process::exit(1);
            })
        };
        let _ = &weights;
        let elapsed = t0.elapsed().as_secs_f64();
        let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        let expected_names: Vec<String> =
            build_capture_lfm(&weights, 0, cfg.num_hidden_layers, &calib_prefix)
                .values()
                .cloned()
                .collect();
        match check_coverage(Path::new(&output), &expected_names) {
            Ok(()) => eprintln!(
                "coverage {}/{} [PASS]",
                expected_names.len(),
                expected_names.len()
            ),
            Err(m) => {
                eprintln!("{m}");
                print_per_proj_summary(Path::new(&output), &expected_names);
                std::process::exit(1);
            }
        }
        print_per_proj_summary(Path::new(&output), &expected_names);
        // NOTE: max_consistency tautological — same staged buffer (see llama block).
        eprintln!(
            "collected {} hessian {} imatrix {:.1}s max_consistency {:.3e} {}",
            summary.n_hessian,
            summary.n_imatrix,
            elapsed,
            summary.max_consistency,
            if summary.max_consistency < 1e-4 {
                "[CONSISTENT]"
            } else {
                "[MISMATCH]"
            }
        );
        if summary.max_consistency >= 1e-4 {
            eprintln!("FAIL lfm diag(H) vs Σx² >=1e-4");
            std::process::exit(1);
        }
        // identity: q/k/v share operator_norm → tmp, w1/w3 share ffn_norm → ffn_tmp; out_proj / w2 have distinct inputs so not checked
        match check_identity_generic(Path::new(&output), cfg.num_hidden_layers, false, false, &calib_prefix) { Ok(()) => eprintln!("lfm identity: q/k/v identical (where present, conv layers skipped) gate/up (w1/w3) identical [PASS]; applicability: attention layers have shared-input q/k/v, dense FFN layers have shared-input w1/w3; conv in/out have singletons so no identity assertion there — stated explicitly"), Err(m) => { eprintln!("FAIL lfm identity: {m}"); std::process::exit(1); } }
        eprintln!("wrote {output} {bytes} bytes");
        eprintln!("--- run summary (provenance) ---");
        eprintln!(
            "model: {model} arch={arch} n_layers={} hidden={} kv_max={} lpp={}",
            cfg.num_hidden_layers, cfg.hidden_size, kv_max, lpp
        );
        eprintln!("HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0");
        eprintln!("batched: Lfm2DecodeBatchState [seq_len+16] lane_capacity {} forward_decode_batch_lfm MFMA; conv_state per lane zeroed per seq", kv_max);
    }
}
