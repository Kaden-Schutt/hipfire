//! Verify the .hfq parser against a real qtip3 artifact (Phase 3 Path A, CPU).
//! Parses the container, summarizes formats, and decodes `model.norm.weight`
//! (BF16) — RMSNorm weights should be ~1.0, confirming the index/offset parsing
//! is correct before the norm-patch is wired into recovery.
//!
//! Run: cargo run -p hipfire-train --release --example hfq_inspect -- <file.hfq>

use hipfire_train::hfq_patch::{bf16_bits_to_f32, is_norm, parse_hfq};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/hfq-export/supra-50m-qtip3.hfq".to_string());
    let bytes = std::fs::read(&path)?;
    let (entries, _meta) = parse_hfq(&bytes)?;

    // quant_type histogram
    let mut by_qt: BTreeMap<u8, usize> = BTreeMap::new();
    let mut n_norm = 0;
    for e in &entries {
        *by_qt.entry(e.quant_type).or_default() += 1;
        if is_norm(&e.name) {
            n_norm += 1;
        }
    }
    println!("{}: {} tensors", path, entries.len());
    println!("quant_type histogram (count by QuantType byte): {by_qt:?}");
    println!("norm tensors (recovery-tunable): {n_norm}");

    // Decode model.norm.weight (BF16) and sanity-check.
    let e = entries
        .iter()
        .find(|e| e.name == "model.norm.weight")
        .ok_or("no model.norm.weight")?;
    println!(
        "\nmodel.norm.weight: qt={} shape={:?} off={} size={}",
        e.quant_type, e.shape, e.data_offset, e.data_size
    );
    let vals: Vec<f32> = bytes[e.data_offset..e.data_offset + e.data_size]
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let (mn, mx) = vals
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let finite = vals.iter().filter(|v| v.is_finite()).count();
    println!(
        "  {} values, mean={mean:.4}, min={mn:.4}, max={mx:.4}, finite={finite}",
        vals.len()
    );
    println!("  first 6: {:?}", &vals[..6.min(vals.len())]);

    // The decisive check: the .hfq norm must decode to real, smooth weight data
    // (finite, bounded). Supra's final norm is ~5.15 (verified byte-identical to
    // the source model.norm.weight), so don't assume ~1.0.
    let spread = mx - mn;
    if finite == vals.len()
        && mean.abs() > 0.05
        && mean.abs() < 100.0
        && spread > 0.0
        && spread < 100.0
    {
        println!("\nPASS — norm decodes to real weight data (offset parsing correct).");
        Ok(())
    } else {
        Err(format!(
            "norm values implausible (mean {mean}, spread {spread}) — offset parsing likely wrong"
        )
        .into())
    }
}
