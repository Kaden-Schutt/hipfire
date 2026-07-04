#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Validate the ZAYA1 GPU prefill forward against the golden dump.
//!
//! Loads the bf16 HFQ, uploads f32 weights to the GPU, runs `gpu_forward_prefill`
//! for the golden prompt's input_ids, and reports per-block cosine vs the fp32
//! golden raw bins.
//!
//! Run:
//!   cargo run --release -p hipfire-arch-zaya --example gpu_golden -- \
//!     /home/sadara/zaya1-8b-native.bf16.hfq /home/sadara/zaya1-8b-native/golden/raw_fp32

use hipfire_arch_zaya::gpu::{gpu_forward_prefill, ZayaGpuWeights};
use hipfire_arch_zaya::ZayaConfig;
use hipfire_runtime::hfq::HfqFile;
use hipfire_rdna::Gpu;
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
fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let (_, data) = read_bin(&dir.join(format!("{name}.bin")));
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn read_i32(dir: &Path, name: &str) -> Vec<i32> {
    let (_, data) = read_bin(&dir.join(format!("{name}.bin")));
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let hfq_path = args
        .next()
        .unwrap_or_else(|| "/home/sadara/zaya1-8b-native.bf16.hfq".to_string());
    let golden_dir = args
        .next()
        .unwrap_or_else(|| "/home/sadara/zaya1-8b-native/golden/raw_fp32".to_string());
    let golden = Path::new(&golden_dir);

    eprintln!("opening hfq {hfq_path}");
    let hfq = HfqFile::open(Path::new(&hfq_path)).expect("open hfq");
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).expect("metadata");
    let cfg_json = meta.get("config").unwrap_or(&meta);
    let cfg = ZayaConfig::from_json(cfg_json).expect("zaya config");
    eprintln!(
        "config: blocks={} hidden={} experts={}",
        cfg.num_blocks, cfg.hidden_size, cfg.moe.num_experts
    );

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("loading weights to GPU (f32)...");
    let w = ZayaGpuWeights::load(&hfq, &mut gpu, &cfg).expect("load gpu weights");
    eprintln!("loaded.");

    let ids: Vec<u32> = read_i32(golden, "input_ids")
        .into_iter()
        .map(|x| x as u32)
        .collect();
    eprintln!("input_ids = {ids:?}");
    let trace = gpu_forward_prefill(&mut gpu, &w, &cfg, &ids).expect("gpu forward");

    let ge = read_f32(golden, "embed_scaled");
    println!("embed_scaled : cos={:.6}", cosine(&trace.embed_scaled, &ge));
    let mut worst = (f32::INFINITY, 0usize);
    for l in 0..cfg.num_blocks {
        let gb = read_f32(golden, &format!("block_{l}"));
        let c = cosine(&trace.block[l], &gb);
        if c < worst.0 {
            worst = (c, l);
        }
        println!("block_{l:<2} : cos={c:.6}");
    }
    println!(
        "final_norm   : cos={:.6}",
        cosine(&trace.final_norm, &read_f32(golden, "final_norm"))
    );
    let glog = read_f32(golden, "logits");
    println!("logits       : cos={:.6}", cosine(&trace.logits, &glog));
    let vocab = cfg.vocab_size;
    let last = trace.seq - 1;
    let am = |r: &[f32]| {
        r.iter()
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
    println!(
        "next-token argmax: mine={} golden={} {}",
        am(&trace.logits[last * vocab..(last + 1) * vocab]),
        am(&glog[last * vocab..(last + 1) * vocab]),
        if am(&trace.logits[last * vocab..(last + 1) * vocab])
            == am(&glog[last * vocab..(last + 1) * vocab])
        {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "\nworst block cosine: block_{} = {:.6}  {}",
        worst.1,
        worst.0,
        if worst.0 >= 0.999 { "PASS" } else { "FAIL" }
    );
}
