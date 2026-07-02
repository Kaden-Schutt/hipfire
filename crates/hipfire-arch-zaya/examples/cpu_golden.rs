// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validate the ZAYA1 CPU reference forward against the golden dump.
//!
//! Loads host f32 weights from the native converted checkpoint, runs
//! `forward_cpu` for the golden prompt's input_ids, and reports per-block cosine
//! vs `golden/raw/*.bin` (exported from `golden/zaya_golden.npz`).
//!
//! Run:
//!   cargo run --release -p hipfire-arch-zaya --example cpu_golden -- \
//!     /home/sadara/zaya1-8b-native /home/sadara/zaya1-8b-native/golden/raw

use hipfire_arch_zaya::cpu::forward_cpu;
use hipfire_arch_zaya::weights::ZayaWeights;
use hipfire_arch_zaya::ZayaConfig;
use hipfire_model::ModelSource;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::path::Path;

fn read_bin(path: &Path) -> (Vec<usize>, Vec<u8>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let ndim = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut off = 4;
    for _ in 0..ndim {
        shape.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize);
        off += 4;
    }
    (shape, bytes[off..].to_vec())
}

fn read_f32(dir: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data) = read_bin(&dir.join(format!("{name}.bin")));
    let v = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, v)
}

fn read_i32(dir: &Path, name: &str) -> Vec<i32> {
    let (_, data) = read_bin(&dir.join(format!("{name}.bin")));
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "/home/sadara/zaya1-8b-native".to_string());
    let golden_dir = args
        .next()
        .unwrap_or_else(|| format!("{model_dir}/golden/raw"));
    let golden = Path::new(&golden_dir);

    eprintln!("opening {model_dir}");
    let src = SafetensorsSource::open(Path::new(&model_dir)).expect("open safetensors");
    let meta: serde_json::Value =
        serde_json::from_str(src.metadata_json()).expect("parse metadata json");
    let cfg_json = meta.get("config").unwrap_or(&meta);
    let cfg = ZayaConfig::from_json(cfg_json).expect("zaya config");
    eprintln!(
        "config: blocks={} hidden={} experts={} vocab={}",
        cfg.num_blocks, cfg.hidden_size, cfg.moe.num_experts, cfg.vocab_size
    );

    eprintln!("loading host weights (f32)...");
    let w = ZayaWeights::load_host(&src, &cfg).expect("load weights");
    eprintln!("loaded.");

    let ids: Vec<u32> = read_i32(golden, "input_ids")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    eprintln!("input_ids = {ids:?}");

    let trace = forward_cpu(&w, &cfg, &ids);

    let (_, g_embed) = read_f32(golden, "embed_scaled");
    println!(
        "embed_scaled : cos={:.6} maxdiff={:.4e}",
        cosine(&trace.embed_scaled, &g_embed),
        max_abs_diff(&trace.embed_scaled, &g_embed)
    );

    let mut worst = (f32::INFINITY, 0usize);
    for l in 0..cfg.num_blocks {
        let (_, gb) = read_f32(golden, &format!("block_{l}"));
        let cos = cosine(&trace.block[l], &gb);
        if cos < worst.0 {
            worst = (cos, l);
        }
        // router top-1 agreement
        let gidx = read_i32(golden, &format!("router_idx_{l}"));
        let agree = trace.router_idx[l]
            .iter()
            .zip(&gidx)
            .filter(|(a, b)| **a as i32 == **b)
            .count();
        println!(
            "block_{l:<2} : cos={:.6} maxdiff={:.3e}  router top1 {}/{} match",
            cos,
            max_abs_diff(&trace.block[l], &gb),
            agree,
            gidx.len()
        );
    }

    let (_, g_fn) = read_f32(golden, "final_norm");
    println!("final_norm   : cos={:.6}", cosine(&trace.final_norm, &g_fn));
    let (lshape, g_log) = read_f32(golden, "logits");
    println!("logits       : cos={:.6}", cosine(&trace.logits, &g_log));

    // next-token argmax agreement at last position.
    let vocab = lshape[lshape.len() - 1];
    let last = trace.seq - 1;
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 {
                    (i, v)
                } else {
                    b
                }
            })
            .0
    };
    let mine = argmax(&trace.logits[last * vocab..(last + 1) * vocab]);
    let theirs = argmax(&g_log[last * vocab..(last + 1) * vocab]);
    println!(
        "next-token argmax: mine={mine} golden={theirs} {}",
        if mine == theirs { "MATCH" } else { "MISMATCH" }
    );

    println!("\nworst block cosine: block_{} = {:.6}", worst.1, worst.0);
    if worst.0 >= 0.999 {
        println!("PASS: all blocks >= 0.999");
    } else {
        println!("FAIL: worst block below 0.999");
    }
}
