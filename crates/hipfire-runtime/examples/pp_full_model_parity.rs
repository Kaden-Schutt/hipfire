// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PB-P-C PC-1 oracle: a whole real llama banded pipeline-parallel through
//! `PpModel` == single-device `forward_scratch`, **bit-exact (max|Δ|=0)**. PP is
//! exact (a plain F32 residual byte-copy across the stage seam, no collective /
//! reorder), so the bar is 0.0, not a tolerance.
//!
//! Reference (single-device) and PP (`PpModel`, banded `forward_scratch_band` +
//! `boundary_copy`) run the SAME token at pos 0; the residual crosses the seam
//! byte-exact. Emulated Pp-2 (`HIPFIRE_EMULATE_GPUS=2`, all stages alias device 0)
//! → proves the BANDING logic; per-device residency + cross-device transport need
//! real HW.
//!
//! Run: HIP_VISIBLE_DEVICES=0 HIPFIRE_DETERMINISTIC=1 \
//!   cargo run -p hipfire-runtime --release --example pp_full_model_parity [model.mq4] [token] [pp]

use hipfire_hardware::{DeviceMesh, DimKind};
use hipfire_loader::model_parallel::PipelineImpl;
use hipfire_loader::ModelParallel;
use hipfire_runtime::llama::{self, ForwardScratch, KvCache, LlamaConfig};

const MAX_SEQ: usize = 512;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));
    let token: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9707); // "Hello"
    let pp: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);

    std::env::set_var("HIPFIRE_EMULATE_GPUS", &pp.to_string());

    // ── Reference: single-device forward_scratch (pos 0). Scoped so its Gpu +
    // weights drop before PpModel brings up the emulated Gpus. ──
    let logits_ref: Vec<f32> = {
        let hfq = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(model_path))
            .expect("open model");
        let config: LlamaConfig = hipfire_runtime::hfq::config_from_hfq(&hfq).expect("config");
        let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");
        gpu.bind_thread().unwrap();
        let weights =
            hipfire_runtime::hfq::load_weights_hfq(&hfq, &config, &mut gpu).expect("load_weights");
        let mut kv = KvCache::new_gpu_q8(
            &mut gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            MAX_SEQ,
        )
        .unwrap();
        let scratch = ForwardScratch::new_with_max_seq(&mut gpu, &config, MAX_SEQ).unwrap();
        llama::forward_scratch(
            &mut gpu, &weights, &config, token, 0, &mut kv, &scratch, 0.0, 1.0, 0, 0, 1.0,
        )
        .expect("ref forward");
        gpu.download_f32(&scratch.logits).unwrap()
    };
    let argmax_ref = llama::argmax(&logits_ref);

    // ── PP path: PpModel banded forward across `pp` stages. ──
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, pp)]);
    let loaded = match hipfire_loader::load_model_pp(model_path, MAX_SEQ, &mesh, Default::default())
    {
        Ok(m) => m,
        Err(e) => {
            println!("pp_full_model_parity: SKIPPED (load_model_pp: {e})");
            return;
        }
    };
        let mut model = loaded
        .pp_model
        .expect("expected dense PP model (pp_model carrier)");
    model.forward_token(token, 0).expect("pp forward");
    let logits_pp = model.logits().expect("pp logits");
    let argmax_pp = llama::argmax(&logits_pp);

    let diff = logits_pp
        .iter()
        .zip(&logits_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mag = logits_ref.iter().map(|v| v.abs()).fold(0.0, f32::max);
    println!(
        "[pp-full-model] Pp-{pp} banded forward vs single-GPU forward_scratch: \
         argmax_pp={argmax_pp} argmax_ref={argmax_ref} max|Δ|={diff:.3e} (ref max|logit|={mag:.3e})"
    );
    assert!(
        logits_pp.iter().all(|x| x.is_finite()),
        "PP produced non-finite logits"
    );
    assert_eq!(
        argmax_pp, argmax_ref,
        "PP argmax {argmax_pp} != single-GPU {argmax_ref}"
    );
    assert!(
        diff == 0.0,
        "PP is exact — expected max|Δ|=0 but got {diff:.3e}"
    );
    println!(
        "pp_full_model_parity: Pp-{pp} banded forward == single-GPU forward_scratch \
         (max|Δ|=0, argmax {argmax_pp}) — PC-1 validated"
    );
}
