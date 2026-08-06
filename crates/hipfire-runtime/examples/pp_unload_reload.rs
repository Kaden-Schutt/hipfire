// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Regression: dense pipeline-parallel LOAD -> UNLOAD -> RELOAD must not panic.
//!
//! Guards the dense-PP unload bug: before the fix, `unload_model`'s
//! `pp_dense` arm did not early-return, so a dense-PP unload fell through into
//! the qwen35-PP `if m.pp > 1` arm and panicked.
//! See `.agent-memory/notes/dense-pp-unload-panic-pp-gpus-expect.md`.
//!
//! This exercises the LOADER path (`hipfire_loader::load_model_pp` +
//! `unload_model`) — the path the daemon uses. The `PpModel` parity examples
//! call `PpModel::load` directly and never reach `unload_model`, so they cannot
//! catch this.
//!
//! Emulated Pp-2 (single gfx1151). Exit 0 + `PASS` = no panic across two
//! load/unload cycles; the process aborts on the bug.
//!
//! Run: HIPFIRE_EMULATE_GPUS=2 \
//!   cargo run -p hipfire-runtime --release --example pp_unload_reload [model.mq4]

use hipfire_hardware::{DeviceMesh, DimKind};

const MAX_SEQ: usize = 512;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or(concat!(
        env!("HOME"),
        "/.hipfire/models/qwen3-0.6b-llama.mq4"
    ));

    std::env::set_var("HIPFIRE_EMULATE_GPUS", "2");
    let mesh = DeviceMesh::rect(&[(DimKind::Pp, 2)]);

    let mut gpu = rdna_compute::Gpu::init().expect("Gpu::init");

    // Two full cycles: the reload proves the first unload freed cleanly. Pre-fix,
    // the FIRST unload already panicked at `pp_gpus.expect`.
    for cycle in 0..2 {
        let m = hipfire_loader::load_model_pp(
            model_path,
            MAX_SEQ,
            &mesh,
            hipfire_runtime::loader_api::SpecLoadCfg::default(),
        )
        .unwrap_or_else(|e| panic!("cycle {cycle}: load_model_pp failed: {e}"));
        assert!(
            m.pp_model.is_some(),
            "cycle {cycle}: dense PP model did not load its pp_model carrier"
        );
        // Pre-fix: panics here at unload_model falling into the qwen35-PP arm.
        hipfire_loader::unload_model(m, &mut gpu);
        println!("cycle {cycle}: load -> unload OK");
    }

    println!("PASS: dense-PP load->unload->reload->unload, no panic");
}
