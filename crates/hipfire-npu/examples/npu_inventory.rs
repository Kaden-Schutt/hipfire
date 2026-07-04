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
//! Print the NPU accelerator inventory produced by the live config+hardware
//! seam (`xdna_inventory_devices_from_env` → `hipfire-xdna` probe). This is the
//! same device list the daemon folds into its accelerator inventory.

use hipfire_model::{accelerator_inventory_json, AcceleratorInventory};
use hipfire_npu::{xdna_inventory_devices_from_env, XdnaHardwareProbe};

fn main() {
    let probe = XdnaHardwareProbe::detect();
    eprintln!(
        "hardware probe: present={} ordinal={:?} detail={:?}",
        probe.present, probe.ordinal, probe.detail
    );

    let inventory =
        AcceleratorInventory::from_devices("npu_env", xdna_inventory_devices_from_env());
    let json = accelerator_inventory_json(&inventory);
    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}
