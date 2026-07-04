// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! HFQ-side numeric bisect for `nemotron_h`.
//!
//! Loads a quantized/source HFQ artifact, runs the same token sequence as
//! `benchmarks/nemotron/dump_hf_reference.py`, and dumps the residual stream
//! after embeddings plus each block, followed by final logits. The binary format
//! matches `bisect_nano4b.rs`, so `benchmarks/nemotron/compare_bisect.py` can
//! compare this dump directly against the HF/Lyra `.npz` reference.
//!
//!   hipfire lock acquire bisect_hfq --watch-pid $$
//!   NEMO_TOKENS=10,25708,... CAP_POS=last \
//!     cargo run -p hipfire-arch-nemotron --example bisect_hfq -- \
//!       /path/to/model.hfq /tmp/nemotron_hfq.bin

use hipfire_arch_nemotron::model::NemotronModel;
use hipfire_arch_nemotron::NemotronHConfig;
use hipfire_runtime::hfq::HfqFile;
use hipfire_rdna::Gpu;
use std::io::Write;
use std::path::Path;

const DEFAULT_HFQ: &str = "/home/sadara/.hipfire/models/nemotron-3-nano-30b-a3b-mq4.hfq";
const DEFAULT_OUT: &str = "/tmp/nemotron_hfq.bin";
const TOKENS: [u32; 5] = [1784, 8961, 1307, 5498, 1395];

fn tokens() -> Vec<u32> {
    match std::env::var("NEMO_TOKENS") {
        Ok(s) => s.split(',').map(|x| x.trim().parse().unwrap()).collect(),
        Err(_) => TOKENS.to_vec(),
    }
}

fn main() {
    let hfq_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_HFQ.to_string());
    let out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_OUT.to_string());

    let hfq = HfqFile::open(Path::new(&hfq_path)).unwrap();
    assert_eq!(hfq.arch_id, 14, "bisect_hfq expects a nemotron_h HFQ");
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).unwrap();
    let cfg_json = meta
        .get("config")
        .expect("nemotron HFQ metadata must contain config");
    let cfg = NemotronHConfig::from_json(cfg_json).unwrap();

    let toks = tokens();
    let max_seq = (toks.len() + 4).max(16);
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    eprintln!("loading hfq {hfq_path}...");
    let mut model = NemotronModel::from_hfq(&mut gpu, &hfq, cfg.clone(), max_seq).unwrap();

    let cap_last = std::env::var("CAP_POS").ok().as_deref() == Some("last");
    let (caps, logits) = if cap_last {
        for (pos, &t) in toks.iter().enumerate().take(toks.len() - 1) {
            model.forward_gpu(&mut gpu, t, pos).unwrap();
        }
        let last = toks.len() - 1;
        model.forward_capture(&mut gpu, toks[last], last).unwrap()
    } else {
        model.forward_capture(&mut gpu, toks[0], 0).unwrap()
    };
    model.free(&mut gpu);

    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    eprintln!(
        "captured at position {}",
        if cap_last { toks.len() - 1 } else { 0 }
    );
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
    println!("PASS: HFQ bisect dump written to {out}");
}
