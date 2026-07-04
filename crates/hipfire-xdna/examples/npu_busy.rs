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
//
//! A poor-man's `amdgpu_top` for the XDNA NPU: prints per-column utilization,
//! power, clocks, and TOPS/task usage from the amdxdna driver.
//!
//! Usage:
//!   npu_busy                 # one-shot snapshot
//!   npu_busy --watch         # refresh every 500ms until Ctrl-C
//!   npu_busy --watch 200     # refresh every 200ms

use std::time::Duration;

use hipfire_xdna::XdnaDevice;

fn snapshot(dev: &XdnaDevice) {
    print!("\x1b[H\x1b[2J"); // home + clear (harmless in one-shot)
    println!("XDNA NPU @ {}", dev.path());

    match dev.resource_info() {
        Ok(r) => println!(
            "  TOPS {}/{}   tasks {}/{}   clk_max {} MHz",
            r.npu_tops_curr, r.npu_tops_max, r.npu_task_curr, r.npu_task_max, r.npu_clk_max
        ),
        Err(e) => println!("  resource_info: {e}"),
    }
    match dev.clocks() {
        Ok(c) => println!("  clocks: MP-NPU {} MHz   H {} MHz", c.mp_npu_mhz, c.h_mhz),
        Err(e) => println!("  clocks: {e}"),
    }
    match dev.sensors() {
        Ok(s) => {
            match s.power_mw {
                Some(p) => println!("  power: {p} mW"),
                None => println!("  power: n/a"),
            }
            println!(
                "  util:  mean {:.0}%   columns {:?}",
                s.mean_utilization_pct(),
                s.column_utilization_pct
            );
        }
        Err(e) => println!("  sensors: {e}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let watch = args.iter().any(|a| a == "--watch");
    let interval_ms: u64 = args
        .iter()
        .skip_while(|a| *a != "--watch")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let dev = match XdnaDevice::open_default() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("cannot open NPU: {e}");
            std::process::exit(1);
        }
    };

    if watch {
        loop {
            snapshot(&dev);
            std::thread::sleep(Duration::from_millis(interval_ms));
        }
    } else {
        snapshot(&dev);
    }
}
