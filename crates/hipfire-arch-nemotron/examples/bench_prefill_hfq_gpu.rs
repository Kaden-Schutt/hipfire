// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Fresh-process prefill benchmark for a quantized Nemotron-H HFQ artifact.
//!
//! Measures `NemotronModel::prefill_batched` against the per-token
//! `forward_gpu` decode loop for the same synthetic token sequence. This is a
//! performance harness only; use `test_model_prefill_hfq_gpu` for the numerical
//! equivalence gate.
//!
//!   hipfire lock acquire bench_nemotron_prefill --watch-pid $$
//!   NANO4B_DIR=/path/to/NVIDIA-Nemotron-3-Nano-4B-BF16 \
//!     cargo run --release -p hipfire-arch-nemotron \
//!       --example bench_prefill_hfq_gpu -- /tmp/nano4b-mq4-protected.hfq \
//!       --seq 128 --warmup 2 --iters 5

use hipfire_arch_nemotron::model::NemotronModel;
use hipfire_arch_nemotron::NemotronHConfig;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_DIR: &str = "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f";
const DEFAULT_HFQ: &str = "/tmp/nano4b-mq4-protected.hfq";

struct Args {
    hfq_path: PathBuf,
    seq: usize,
    warmup: usize,
    iters: usize,
}

fn parse_args() -> Args {
    let mut hfq_path = PathBuf::from(DEFAULT_HFQ);
    let mut seq = 128usize;
    let mut warmup = 2usize;
    let mut iters = 5usize;

    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seq" => {
                seq = args
                    .next()
                    .expect("--seq value")
                    .parse()
                    .expect("--seq must be an integer");
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .expect("--warmup value")
                    .parse()
                    .expect("--warmup must be an integer");
            }
            "--iters" => {
                iters = args
                    .next()
                    .expect("--iters value")
                    .parse()
                    .expect("--iters must be an integer");
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: bench_prefill_hfq_gpu [model.hfq] [--seq N] [--warmup N] [--iters N]"
                );
                std::process::exit(0);
            }
            other if other.starts_with('-') => panic!("unknown flag: {other}"),
            path => hfq_path = PathBuf::from(path),
        }
    }

    assert!(seq > 0, "--seq must be > 0");
    assert!(iters > 0, "--iters must be > 0");
    Args {
        hfq_path,
        seq,
        warmup,
        iters,
    }
}

fn load_cfg(dir: &Path) -> NemotronHConfig {
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    NemotronHConfig::from_json(&cfg_json).unwrap()
}

fn synthetic_tokens(seq: usize, vocab_size: usize) -> Vec<u32> {
    let span = vocab_size.saturating_sub(2048).max(1);
    (0..seq)
        .map(|i| 1024 + ((i * 37 + 17) % span) as u32)
        .collect()
}

fn summarize(label: &str, seq: usize, samples_ms: &[f64]) {
    let min_ms = samples_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = samples_ms.iter().copied().fold(0.0f64, f64::max);
    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let tok_s = seq as f64 / (mean_ms / 1000.0);
    eprintln!(
        "{label:>11}: min={min_ms:.2}ms mean={mean_ms:.2}ms max={max_ms:.2}ms tok/s={tok_s:.1}"
    );
}

fn main() {
    let args = parse_args();
    let dir =
        PathBuf::from(std::env::var("NANO4B_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string()));
    if !dir.join("config.json").exists() {
        eprintln!("SKIP: checkpoint config not found at {}", dir.display());
        return;
    }
    if !args.hfq_path.exists() {
        eprintln!("SKIP: hfq not found at {}", args.hfq_path.display());
        return;
    }

    let cfg = load_cfg(&dir);
    let max_seq = args.seq.max(16);
    let tokens = synthetic_tokens(args.seq, cfg.vocab_size);
    let hfq = HfqFile::open(Path::new(&args.hfq_path)).unwrap();
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);
    eprintln!(
        "model={} seq={} warmup={} iters={}",
        args.hfq_path.display(),
        args.seq,
        args.warmup,
        args.iters
    );

    eprintln!("loading hfq model...");
    let mut model = NemotronModel::from_hfq(&mut gpu, &hfq, cfg, max_seq).unwrap();
    assert!(
        model.can_batched_prefill(),
        "quant HFQ model should allow batched prefill"
    );

    for _ in 0..args.warmup {
        model.reset(&mut gpu).unwrap();
        model.prefill_batched(&mut gpu, &tokens).unwrap();
        gpu.hip.device_synchronize().unwrap();
    }

    let mut batched_ms = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        model.reset(&mut gpu).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        model.prefill_batched(&mut gpu, &tokens).unwrap();
        gpu.hip.device_synchronize().unwrap();
        batched_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    for _ in 0..args.warmup {
        model.reset(&mut gpu).unwrap();
        for (pos, &tok) in tokens.iter().enumerate() {
            model.forward_gpu(&mut gpu, tok, pos).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
    }

    let mut decode_ms = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        model.reset(&mut gpu).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for (pos, &tok) in tokens.iter().enumerate() {
            model.forward_gpu(&mut gpu, tok, pos).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        decode_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let batched_mean = batched_ms.iter().sum::<f64>() / batched_ms.len() as f64;
    let decode_mean = decode_ms.iter().sum::<f64>() / decode_ms.len() as f64;
    eprintln!("===== nemotron_h HFQ prefill benchmark =====");
    summarize("batched", args.seq, &batched_ms);
    summarize("decode-loop", args.seq, &decode_ms);
    eprintln!("speedup={:.2}x", decode_mean / batched_mean);

    model.free(&mut gpu);
}
