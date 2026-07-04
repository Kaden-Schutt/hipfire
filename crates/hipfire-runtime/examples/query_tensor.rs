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
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Look up a single tensor in an `.hfq` file by exact name and print its
//! quant_type + shape. Complement to `compare_hfq` (which diffs two files)
//! and `dump_norms` (which dumps 1D norm tensor distributions).
//!
//! Usage: `cargo run --release --example query_tensor -- <path.hfq> <tensor_name>`
//!
//! Example:
//!   query_tensor /local/hipfire/qwen3.6-35b-a3b-mfp4.hfq \
//!       model.language_model.layers.0.mlp.experts.0.down_proj.weight
//!   → model.language_model.layers.0.mlp.experts.0.down_proj.weight
//!       qt=24 shape=[2048, 768]
//!
//! quant_type IDs are documented in `docs/quant-formats/hfp4.md` and
//! `docs/QUANTIZATION.md` (e.g., 13=MQ4G256, 21=HFP4G32, 24=MFP4G32).

use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: query_tensor <path.hfq> <tensor_name>");
    let target = args
        .next()
        .expect("usage: query_tensor <path.hfq> <tensor_name>");
    let hfq = HfqFile::open(Path::new(&path)).expect("open .hfq");
    for t in hfq.tensors() {
        if t.name == target {
            println!("{}\n  qt={} shape={:?}", t.name, t.quant_type, t.shape);
            return;
        }
    }
    println!("not found");
    std::process::exit(1);
}
