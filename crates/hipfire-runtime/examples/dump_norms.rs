//! Dump RMSNorm weight distributions from a `.hfq` file.
//!
//! Inspects every 1D norm-like tensor (1D shape, name contains "norm")
//! and prints mean / std / min / max of its f32 values. Useful for
//! debugging RMSNorm convention questions: HF transformers'
//! `nn.Parameter(torch.zeros(dim))` init produces deviation-from-zero
//! weights (mean ~0); a tensor that drifts from zero during training
//! (e.g. final norm under loss pressure) shows up as mean ~1+.
//!
//! Background: `docs/plans/qwen35-moe-rmsnorm-fix.md` documents how
//! this diagnostic distinguishes the two cases for Qwen3.5/3.6's
//! GemmaRMSNorm. Mirrors `/tmp/inspect_norms.py` for use without a
//! Python environment.
//!
//! Usage:
//!   cargo run --release --example dump_norms -- /local/hipfire/qwen3.6-27b.mq4
//!   cargo run --release --example dump_norms -- /local/hipfire/qwen3.6-27b.mq4 norm.weight
//!
//! Optional second arg filters tensor names by substring.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use std::path::Path;

fn stats(values: &[f32]) -> (f32, f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let lo = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, var.sqrt(), lo, hi)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: dump_norms <path.hfq> [name_substring]");
        std::process::exit(2);
    });
    let filter = args.next();

    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    println!("arch_id={} n_tensors={}", hfq.arch_id, hfq.tensors().len());

    let norm_like: Vec<_> = hfq
        .tensors()
        .iter()
        .filter(|t| {
            let name = t.name.as_str();
            let is_norm = name.contains("norm")
                && t.shape.len() == 1
                && (t.quant_type == 1 || t.quant_type == 2);
            let matches_filter = filter.as_deref().map_or(true, |f| name.contains(f));
            is_norm && matches_filter
        })
        .collect();

    println!("\n{} norm-like 1D F16/F32 tensors:", norm_like.len());
    println!(
        "{:<70} {:>3} {:<14} {:>10} {:>10} {:>9} {:>9}",
        "name", "qt", "shape", "mean", "std", "min", "max"
    );
    println!("{}", "-".repeat(140));

    for t in &norm_like {
        // tensor_data_vec returns owned bytes via pread on Unix. f16_to_f32
        // is exported from llama.rs so we don't need to reimplement it here.
        let (info, data) = match hfq.tensor_data_vec(&t.name) {
            Some(p) => p,
            None => {
                eprintln!("could not read {}", t.name);
                continue;
            }
        };
        let values: Vec<f32> = match info.quant_type {
            1 => data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            2 => data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            qt => {
                eprintln!("skipping {} (qt={qt} not f16/f32)", t.name);
                continue;
            }
        };
        let (mean, std, lo, hi) = stats(&values);
        let shape = format!("{:?}", info.shape);
        println!(
            "{:<70} {:>3} {:<14} {:>+10.4} {:>10.4} {:>+9.3} {:>+9.3}",
            info.name, info.quant_type, shape, mean, std, lo, hi
        );
    }
}
