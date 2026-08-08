// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU validation: dense TP / PP load->unload cycles must RECLAIM VRAM.
//!
//! Guards the `TpModel::free` / `PpModel::free` fix. Before it, `unload_model`'s
//! `drop(tp)` / `drop(pp)` freed nothing — none of `GpuTensor` /
//! `hip_bridge::DeviceBuffer` / `GpuPool` has a freeing `Drop`, and `Gpu::drop`
//! only re-binds — so every load/unload cycle leaked the whole model. The
//! no-panic `pp_unload_reload` example cannot catch this (a leak does not
//! panic); only measuring free VRAM does.
//!
//! Method: after a warmup load/unload (so one-time JIT/sign-table/cache allocs
//! are already resident and don't read as "leak"), run N cycles measuring free
//! VRAM (`hipMemGetInfo`) after each unload. The after-unload free must not fall
//! monotonically by ~model size; a real leak drops by model-size × (N-1).
//!
//! Emulated 2-rank on a single GPU. Run under the GPU lock:
//!   HIPFIRE_EMULATE_GPUS=2 cargo run -p hipfire-runtime --release \
//!     --example mesh_unload_vram -- <tp|pp> [model.mq4]

use hipfire_hardware::{DeviceMesh, DimKind};

const MAX_SEQ: usize = 512;
const CYCLES: usize = 4;
/// Allocator/fixed-cache noise budget. A real leak is model-sized (100s of MB),
/// far above this; genuine reclaim leaves only allocator fragmentation.
const TOL_MB: f64 = 64.0;

fn free_mb(gpu: &rdna_compute::Gpu) -> f64 {
    let (free, _total) = gpu.hip.get_vram_info().expect("get_vram_info");
    free as f64 / 1e6
}

fn load(axis: &str, model: &str, mesh: &DeviceMesh) -> Result<hipfire_loader::LoadedModel, String> {
    match axis {
        "pp" => hipfire_loader::load_model_pp(
            model,
            MAX_SEQ,
            mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ),
        _ => hipfire_loader::load_model_tp(
            model,
            MAX_SEQ,
            mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        ),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let axis = args.get(1).map(String::as_str).unwrap_or("tp");
    let model = args.get(2).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));
    let mesh = match axis {
        "pp" => DeviceMesh::rect(&[(DimKind::Pp, 2)]),
        "tp" => DeviceMesh::rect(&[(DimKind::Tp, 2)]),
        other => {
            eprintln!("axis must be `tp` or `pp`, got `{other}`");
            std::process::exit(2);
        }
    };

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");

    // Warmup cycle: pay all one-time fixed costs (JIT kernels, sign tables) so
    // they are not misread as a per-cycle leak below.
    match load(axis, model, &mesh) {
        Ok(m) => {
            let _ = hipfire_loader::unload_model(m, &mut gpu);
        }
        Err(e) => {
            eprintln!("axis={axis} model={model}\nwarmup load failed: {e}");
            std::process::exit(2);
        }
    }

    let baseline = free_mb(&gpu);
    println!("axis={axis} model={model}");
    println!("baseline free after warmup unload: {baseline:.0} MB");

    let mut after_unload = Vec::with_capacity(CYCLES);
    for cycle in 0..CYCLES {
        let before = free_mb(&gpu);
        let m =
            load(axis, model, &mesh).unwrap_or_else(|e| panic!("cycle {cycle}: load failed: {e}"));
        let loaded = free_mb(&gpu);
        hipfire_loader::unload_model(m, &mut gpu);
        let unloaded = free_mb(&gpu);
        println!(
            "cycle {cycle}: before={before:.0} loaded={loaded:.0} (model≈{:.0} MB) \
             unloaded={unloaded:.0} (reclaimed={:.0} MB)",
            before - loaded,
            unloaded - loaded
        );
        after_unload.push(unloaded);
    }

    // Leak test: after-unload free must not trend down by model size.
    let drift = after_unload[0] - after_unload[CYCLES - 1];
    println!(
        "after-unload free: first={:.0} last={:.0} drift={drift:.0} MB (tol {TOL_MB})",
        after_unload[0],
        after_unload[CYCLES - 1]
    );
    if drift > TOL_MB {
        println!(
            "FAIL: VRAM leaked ~{:.0} MB/cycle across {CYCLES} cycles — unload not reclaiming",
            drift / (CYCLES - 1) as f64
        );
        std::process::exit(1);
    }
    println!(
        "PASS: {axis} load/unload reclaims VRAM (drift {drift:.0} MB ≤ {TOL_MB} over {CYCLES} cycles)"
    );
}
