// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Load the real Nemotron-3 Nano 30B-A3B MQ4/HFQ artifact and run a short
//! decode-only smoke. The 30B checkpoint contains MoE blocks, so batched prefill
//! is intentionally unavailable until FU6 grows an expert-sorted prefill path.
//!
//!   hipfire lock acquire test_load_nano30b_hfq --watch-pid $$
//!   NANO30B_DIR=<snap> cargo run -p hipfire-arch-nemotron \
//!       --example test_load_nano30b_hfq -- /path/to/nemotron-3-nano-30b-a3b-mq4.hfq

use hipfire_arch_nemotron::model::NemotronModel;
use hipfire_arch_nemotron::{BlockKind, NemotronHConfig};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};

const DEFAULT_DIR: &str = "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848";
const DEFAULT_HFQ: &str = "/home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-mq4.hfq";
const DEFAULT_TOKENS: [u32; 2] = [1784, 8961];

fn tokens() -> Vec<u32> {
    match std::env::var("NEMO_TOKENS") {
        Ok(s) => s.split(',').map(|x| x.trim().parse().unwrap()).collect(),
        Err(_) => DEFAULT_TOKENS.to_vec(),
    }
}

fn load_cfg(dir: &Path) -> NemotronHConfig {
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    NemotronHConfig::from_json(&cfg_json).unwrap()
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..v.len() {
        if v[i] > v[bi] {
            bi = i;
        }
    }
    bi
}

fn main() {
    let dir =
        PathBuf::from(std::env::var("NANO30B_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string()));
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
    assert!(
        cfg.blocks.iter().any(|b| *b == BlockKind::Moe),
        "30B config should include MoE blocks"
    );

    let toks = tokens();
    let max_seq = (toks.len() + 4).max(16);
    let hfq = HfqFile::open(Path::new(&hfq_path)).unwrap();
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    eprintln!("loading hfq {}...", hfq_path.display());

    let mut model = NemotronModel::from_hfq(&mut gpu, &hfq, cfg, max_seq).unwrap();
    assert!(
        !model.can_batched_prefill(),
        "MoE model should use decode-loop prefill until MoE batched prefill lands"
    );

    let mut final_argmax = 0usize;
    for (pos, &tok) in toks.iter().enumerate() {
        let logits = model.forward(&mut gpu, tok, pos).unwrap();
        if logits.iter().any(|x| !x.is_finite()) {
            eprintln!("FAIL: non-finite logits at pos {pos} token {tok}");
            std::process::exit(1);
        }
        final_argmax = argmax(&logits);
        eprintln!("pos {pos} tok {tok}: argmax={final_argmax}");
    }
    model.free(&mut gpu);

    println!(
        "PASS: Nemotron 30B HFQ loaded and decoded {} token(s), final argmax={final_argmax}",
        toks.len()
    );
}
