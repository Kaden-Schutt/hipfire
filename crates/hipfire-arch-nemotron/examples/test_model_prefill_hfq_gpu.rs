// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! End-to-end equivalence for quantized HFQ Nemotron prefill:
//! `NemotronModel::prefill_batched` vs the per-token `forward_gpu` decode loop.
//! This exercises Q8, HFQ4G128, and MQ4G256 `LinearWeight::gemm_seq` arms on the
//! real Nano-4B HFQ artifact. Skips cleanly when the local checkpoint/HFQ is
//! absent.
//!
//!   hipfire lock acquire test_model_prefill_hfq_gpu --watch-pid $$
//!   cargo run -p hipfire-arch-nemotron --example test_model_prefill_hfq_gpu

use hipfire_arch_nemotron::model::NemotronModel;
use hipfire_arch_nemotron::NemotronHConfig;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const DEFAULT_DIR: &str = "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f";
const DEFAULT_HFQ: &str = "/tmp/nano4b-mq4.hfq";
const TOKENS: [u32; 5] = [1784, 8961, 1307, 5498, 1395];

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..v.len() {
        if v[i] > v[bi] {
            bi = i;
        }
    }
    bi
}

fn load_cfg(dir: &Path) -> NemotronHConfig {
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    NemotronHConfig::from_json(&cfg_json).unwrap()
}

fn main() {
    let dir =
        PathBuf::from(std::env::var("NANO4B_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string()));
    let hfq_path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| DEFAULT_HFQ.to_string()),
    );
    if !dir.join("config.json").exists() {
        eprintln!("SKIP: checkpoint config not found at {}", dir.display());
        return;
    }
    if !hfq_path.exists() {
        eprintln!("SKIP: hfq not found at {}", hfq_path.display());
        return;
    }

    let cfg = load_cfg(&dir);
    let max_seq = (TOKENS.len() + 4).max(16);
    let hfq = HfqFile::open(Path::new(&hfq_path)).unwrap();
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    eprintln!("loading hfq for batched prefill...");
    let mut model_pf = NemotronModel::from_hfq(&mut gpu, &hfq, cfg.clone(), max_seq).unwrap();
    assert!(
        model_pf.can_batched_prefill(),
        "quant HFQ model should allow batched prefill"
    );
    model_pf.prefill_batched(&mut gpu, &TOKENS).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let logits_pf = gpu.download_f32(model_pf.logits_tensor()).unwrap();
    model_pf.free(&mut gpu);

    eprintln!("loading hfq for per-token decode...");
    let mut model_dec = NemotronModel::from_hfq(&mut gpu, &hfq, cfg, max_seq).unwrap();
    for (pos, &tok) in TOKENS.iter().enumerate() {
        model_dec.forward_gpu(&mut gpu, tok, pos).unwrap();
    }
    gpu.hip.device_synchronize().unwrap();
    let logits_dec = gpu.download_f32(model_dec.logits_tensor()).unwrap();
    model_dec.free(&mut gpu);

    let max_d = logits_pf
        .iter()
        .zip(&logits_dec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ap = argmax(&logits_pf);
    let ad = argmax(&logits_dec);
    eprintln!("tokens={TOKENS:?}  max|delta logit|={max_d:.3e}  argmax pf={ap} dec={ad}");

    if max_d < 1e-3 && ap == ad {
        println!(
            "PASS: hfq prefill_batched last-pos logits == decode loop (max|delta|={max_d:.2e})"
        );
    } else {
        println!("FAIL: hfq prefill_batched diverges from decode (max|delta|={max_d:.2e})");
        std::process::exit(1);
    }
}
