// SPDX-License-Identifier: Apache-2.0
// hipfire — Tier-1 native single-load artifact collector (Phase 3 harness).
//
//! Loads a bf16 `.hfq`, arms a unified `ActivationCapture` collector (per-tensor
//! GPTQ Hessian Σxxᵀ + imatrix diag Σx²), runs the engine forward over the
//! calibration corpus (single model load), drains, and writes the HFHS Hessian
//! sidecar. Verifies internal consistency: diag(Σxxᵀ) must equal Σx² (the two
//! reduction kernels agree on the SAME captured activations).
//!
//! This is the in-process core that the daemon `Collect` op (Phase 5) will host.
//!
//! Run:
//!   cargo run --release -p hipfire-runtime --example collect_artifacts -- \
//!     --model ~/.hipfire/models/qwen3.5-0.8b-bf16.hfq \
//!     --corpus benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
//!     --output /tmp/qwen3.5-0.8b-native.hessian.bin --max-tokens 512

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
use hipfire_runtime::llama::KvCache;
use rdna_compute::{ActivationCapture, DType, Gpu, GpuTensor};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

struct Acc {
    diag: GpuTensor, // [K]   Σx²  (imatrix)
    h: GpuTensor,    // [K,K] Σxxᵀ (Hessian)
    k: usize,
    n_tokens: u64,
}

#[derive(Default)]
struct UnifiedCollector {
    accs: Mutex<HashMap<String, Acc>>,
}

impl ActivationCapture for UnifiedCollector {
    fn capture(&self, gpu: &mut Gpu, tensor_name: &str, input: &GpuTensor, n: usize, k: usize) {
        // Use the gemm's actual n/k — `input` is a shared scratch buffer whose
        // shape (max(dim,hidden)) does NOT reflect the linear's input width.
        let mut accs = self.accs.lock().unwrap();
        if !accs.contains_key(tensor_name) {
            let diag = gpu.zeros(&[k], DType::F32).unwrap();
            let h = gpu.zeros(&[k, k], DType::F32).unwrap();
            accs.insert(
                tensor_name.to_string(),
                Acc {
                    diag,
                    h,
                    k,
                    n_tokens: 0,
                },
            );
        }
        let acc = accs.get_mut(tensor_name).unwrap();
        gpu.calib_sumsq_reduce_f32(input, &acc.diag, n, k).unwrap();
        gpu.calib_hessian_outer_f32(input, &acc.h, n, k).unwrap();
        acc.n_tokens += n as u64;
    }
}

fn arg(flag: &str, default: Option<String>) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == flag)
        .and_then(|i| a.get(i + 1).cloned())
        .or(default)
}

fn main() {
    let model = arg("--model", None).expect("--model required");
    let corpus = arg("--corpus", None).expect("--corpus required");
    let output = arg("--output", Some("/tmp/native.hessian.bin".into())).unwrap();
    let max_tokens: usize = arg("--max-tokens", Some("512".into()))
        .unwrap()
        .parse()
        .unwrap();

    let mut hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&model)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tok");

    // Tokenize a prefix of the corpus (single calibration sequence for the harness).
    let raw = std::fs::read(&corpus).expect("read corpus");
    let take = (max_tokens * 8).min(raw.len());
    let text = String::from_utf8_lossy(&raw[..take]).to_string();
    let all: Vec<u32> = tokenizer.encode(&text);
    let n_tok = all.len().min(max_tokens);
    eprintln!("calibrating on {n_tok} tokens");

    let mut gpu = Gpu::init().expect("gpu");
    eprintln!("GPU: {}", gpu.arch);
    let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu).expect("load_weights");

    // Arm calibration: ptr→name map + the collector. f32 KV + FP32 DeltaNet state
    // for faithful (lossless) activations.
    let names = qwen35::build_capture_names(&weights);
    eprintln!("capture targets: {}", names.len());
    // Diagnostic: which gpu_dtype do the linears load as → which gemv kernel.
    if let Some(qwen35::LayerWeights::DeltaNet(l0)) = weights.layers.first() {
        eprintln!(
            "DIAG layer0: wqkv={:?} wo={:?} w_down={:?}",
            l0.wqkv.gpu_dtype, l0.wo.gpu_dtype, l0.w_down.gpu_dtype
        );
    }
    let collector = std::sync::Arc::new(UnifiedCollector::default());
    gpu.capture_names = names;
    gpu.active_capture = Some(collector.clone());

    let mut kv = KvCache::new_gpu(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        n_tok + 16,
    )
    .unwrap();
    let mut dn = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 64).unwrap();

    let t0 = std::time::Instant::now();
    for (pos, &tok) in all.iter().take(n_tok).enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, tok, pos, &mut kv, &mut dn, &scratch,
        )
        .expect("forward");
    }
    eprintln!(
        "forward over {n_tok} tokens: {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // Disarm before draining (avoid capturing the drain's own ops, if any).
    gpu.active_capture = None;

    // Drain + verify diag(H) == Σx² (the two kernels agree on real activations).
    let accs = collector.accs.lock().unwrap();
    let mut names_sorted: Vec<&String> = accs.keys().collect();
    names_sorted.sort();
    let mut max_consistency = 0.0f32;
    let mut records: Vec<(String, usize, Vec<f32>)> = Vec::new();
    for name in &names_sorted {
        let acc = &accs[*name];
        let diag = gpu.download_f32(&acc.diag).unwrap();
        let h = gpu.download_f32(&acc.h).unwrap();
        // diag(H)[c] vs Σx²[c]
        let mut md = 0.0f32;
        for c in 0..acc.k {
            md = md.max((h[c * acc.k + c] - diag[c]).abs() / diag[c].abs().max(1.0));
        }
        max_consistency = max_consistency.max(md);
        // Finalize H / n_tokens for the HFHS payload.
        let inv = 1.0 / acc.n_tokens as f32;
        let h_final: Vec<f32> = h.iter().map(|v| v * inv).collect();
        records.push(((*name).clone(), acc.k, h_final));
    }
    eprintln!(
        "drained {} tensors; max diag(H)-vs-Σx² rel-err = {max_consistency:.3e} {}",
        records.len(),
        if max_consistency < 1e-4 {
            "[CONSISTENT]"
        } else {
            "[MISMATCH]"
        }
    );

    // Write HFHS v1 (matches scripts/collect_hessian.py / hessian_io.rs reader).
    let mut f = std::io::BufWriter::new(std::fs::File::create(&output).expect("create out"));
    f.write_all(b"HFHS").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap(); // version
    f.write_all(&(records.len() as u64).to_le_bytes()).unwrap(); // n_tensors
    f.write_all(&0u64.to_le_bytes()).unwrap(); // reserved
    for (name, k, h_final) in &records {
        let nb = name.as_bytes();
        f.write_all(&(nb.len() as u32).to_le_bytes()).unwrap();
        f.write_all(nb).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap(); // expert_idx
        f.write_all(&(*k as u32).to_le_bytes()).unwrap(); // K
        f.write_all(&1u32.to_le_bytes()).unwrap(); // dtype_flag = F32
        let bytes =
            unsafe { std::slice::from_raw_parts(h_final.as_ptr() as *const u8, h_final.len() * 4) };
        f.write_all(bytes).unwrap();
    }
    f.flush().unwrap();
    eprintln!("wrote HFHS: {output}");
    if max_consistency >= 1e-4 {
        std::process::exit(1);
    }
}
