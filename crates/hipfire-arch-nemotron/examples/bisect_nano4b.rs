// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! HF-reference numeric bisect (FU2), hipfire side: load Nano-4B, run the same
//! fixed token sequence as `benchmarks/nemotron/dump_hf_reference.py`, and dump
//! the last-position residual-stream hidden after the embedding + each block
//! (43 vectors) plus final logits to a raw-f32 file. `compare_bisect.py` then
//! diffs it against the HF dump to find the first divergent layer.
//!
//!   hipfire lock acquire bisect --watch-pid $$
//!   NANO4B_DIR=<snap> cargo run --release -p hipfire-arch-nemotron \
//!       --example bisect_nano4b -- /tmp/nemo_hipfire.bin
//!
//! Dump layout (little-endian): [u32 n_caps][u32 hidden][u32 vocab] then
//! n_caps*hidden f32 (caps) then vocab f32 (logits).

use hipfire_arch_nemotron::loader::load_nemotron_weights;
use hipfire_arch_nemotron::model::NemotronModel;
use hipfire_arch_nemotron::NemotronHConfig;
use hipfire_model::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use rdna_compute::Gpu;
use std::io::Write;
use std::path::PathBuf;

const DEFAULT_DIR: &str = "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f";
// "The capital of France is" per the HF dump (must match exactly).
const TOKENS: [u32; 5] = [1784, 8961, 1307, 5498, 1395];

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/nemo_hipfire.bin".to_string());
    let dir =
        PathBuf::from(std::env::var("NANO4B_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string()));
    if !dir.join("config.json").exists() {
        eprintln!("SKIP: checkpoint not found at {}", dir.display());
        return;
    }
    let cfg_str = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg_json: serde_json::Value = serde_json::from_str(&cfg_str).unwrap();
    let cfg = NemotronHConfig::from_json(&cfg_json).unwrap();
    let src = SafetensorsSource::open(&dir).unwrap();
    assert_eq!(src.arch_id(), 14);
    eprintln!("loading weights...");
    let weights = load_nemotron_weights(&src, &cfg).unwrap();

    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    let mut model = NemotronModel::new(&mut gpu, cfg.clone(), &weights, 64).unwrap();

    // Capture position: 0 isolates block math (fresh state); env CAP_POS=last
    // builds state over the whole prompt and captures the final position.
    let cap_last = std::env::var("CAP_POS").ok().as_deref() == Some("last");
    let (caps, logits) = if cap_last {
        for (pos, &t) in TOKENS.iter().enumerate().take(TOKENS.len() - 1) {
            model.forward_gpu(&mut gpu, t, pos).unwrap();
        }
        let last = TOKENS.len() - 1;
        model.forward_capture(&mut gpu, TOKENS[last], last).unwrap()
    } else {
        model.forward_capture(&mut gpu, TOKENS[0], 0).unwrap()
    };
    eprintln!(
        "captured at position {}",
        if cap_last { TOKENS.len() - 1 } else { 0 }
    );

    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    eprintln!(
        "captured {} hidden vectors (hidden={hidden}), logits={}",
        caps.len(),
        logits.len()
    );

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    f.write_all(&(caps.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&(hidden as u32).to_le_bytes()).unwrap();
    f.write_all(&(vocab as u32).to_le_bytes()).unwrap();
    for c in &caps {
        for &v in c {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    for &v in &logits {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    let top5 = {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        idx[..5].to_vec()
    };
    eprintln!("wrote {out}; final top5: {top5:?}");
    println!("PASS: bisect dump written to {out}");
}
