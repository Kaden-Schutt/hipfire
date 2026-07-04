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

//! FU4 isolation probe: compare a SINGLE quantized gemv against the f32 gemv of
//! the same weight, on one fixed random input — to tell whether the HFQ4G128
//! path (MLP up_proj), the MQ4G256 path (MLP down_proj), or both are the source
//! of the hfq-vs-f32 divergence. Uses MLP weights (no out_proj residual rescale).

use hipfire_arch_nemotron::loader::load_linear_hfq;
use hipfire_model::ModelSource;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_rdna::{DType, Gpu};
use std::path::{Path, PathBuf};

const DEFAULT_DIR: &str = "/srv/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f";
const DEFAULT_HFQ: &str = "/tmp/nano4b-mq4.hfq";

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn probe(gpu: &mut Gpu, src: &SafetensorsSource, hfq: &HfqFile, name: &str, m: usize, k: usize) {
    // f32 reference weight
    let (info, bytes) = src.tensor_data(name).unwrap();
    assert_eq!(info.dtype, "BF16");
    let wf = bf16_to_f32(bytes);
    let wf_gpu = gpu.upload_f32(&wf, &[m, k]).unwrap();

    // quantized weight (the actual hfq path)
    let wq = load_linear_hfq(hfq, gpu, name, m, k).unwrap();

    // fixed pseudo-random input x[k]
    let mut s = 0x1234_5678u32;
    let x: Vec<f32> = (0..k)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
        })
        .collect();
    let xg = gpu.upload_f32(&x, &[k]).unwrap();

    let out_f = gpu.zeros(&[m], DType::F32).unwrap();
    let out_q = gpu.zeros(&[m], DType::F32).unwrap();
    gpu.gemv_f32(&wf_gpu, &xg, &out_f).unwrap();
    wq.gemv(gpu, &xg, &out_q).unwrap();
    gpu.hip.device_synchronize().unwrap();

    let of = gpu.download_f32(&out_f).unwrap();
    let oq = gpu.download_f32(&out_q).unwrap();
    let c = cos(&of, &oq);
    let rel: f64 = {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..m {
            num += (of[i] - oq[i]).abs() as f64;
            den += of[i].abs() as f64;
        }
        num / den.max(1e-9)
    };
    eprintln!(
        "  {name}\n      [{m}x{k}] cos={c:.6}  mean_rel_err={rel:.4}  {}",
        if c > 0.99 { "OK" } else { "*** BROKEN ***" }
    );
}

fn main() {
    let dir =
        PathBuf::from(std::env::var("NANO4B_DIR").unwrap_or_else(|_| DEFAULT_DIR.to_string()));
    let hfq_path = PathBuf::from(DEFAULT_HFQ);
    if !dir.join("config.json").exists() || !hfq_path.exists() {
        eprintln!("SKIP: inputs missing");
        return;
    }
    let src = SafetensorsSource::open(&dir).unwrap();
    let hfq = HfqFile::open(Path::new(&hfq_path)).unwrap();
    let mut gpu = Gpu::init().unwrap();
    eprintln!("GPU: {}", gpu.arch);

    // MLP layer 1: up_proj = HFQ4G128 (k=3136), down_proj = MQ4G256 (k=12544).
    eprintln!("HFQ4G128 path:");
    probe(
        &mut gpu,
        &src,
        &hfq,
        "backbone.layers.1.mixer.up_proj.weight",
        12544,
        3136,
    );
    eprintln!("MQ4G256 path:");
    probe(
        &mut gpu,
        &src,
        &hfq,
        "backbone.layers.1.mixer.down_proj.weight",
        3136,
        12544,
    );
    println!("PASS: probe complete");
}
