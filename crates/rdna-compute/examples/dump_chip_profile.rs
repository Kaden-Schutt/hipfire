// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Dump the `ArchCaps` atom/molecule/capability set + tuning knobs as JSON.
//!
//! Two modes:
//!   `--arch <gfx>`  — NO GPU required. Builds `ArchCaps` for the named arch
//!                     from env-derived `FeatureFlags` and prints its
//!                     `dump_json()`. This is the mode consumed by CI /
//!                     no-GPU chip-profile tooling.
//!   (no args)       — requires a live GPU. Calls `Gpu::init()` and dumps
//!                     the detected device's live `arch_caps`.
//!
//! Usage:
//!   cargo run -p rdna-compute --example dump_chip_profile -- --arch gfx1201
//!   cargo run -p rdna-compute --example dump_chip_profile

use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::{FeatureFlags, Gpu};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arch_arg = args
        .iter()
        .position(|a| a == "--arch")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let json = match arch_arg {
        Some(arch) => {
            // NO-GPU path: build ArchCaps purely from the arch string + env.
            let flags = std::sync::Arc::new(FeatureFlags::from_env(&arch));
            let caps = ArchCaps::new(&arch, flags);
            caps.dump_json()
        }
        None => {
            // Live-GPU path.
            let gpu = Gpu::init().unwrap_or_else(|e| {
                eprintln!("Gpu::init() failed: {e}. Pass --arch <gfx> for a no-GPU dump.");
                std::process::exit(1);
            });
            gpu.arch_caps.dump_json()
        }
    };

    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}
