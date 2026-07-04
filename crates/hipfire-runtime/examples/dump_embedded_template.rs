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
//! Dump the chat_template embedded in an .hfq model (the production fallback
//! used when no HIPFIRE_CHAT_TEMPLATE_FILE / per-model override is set).
//! Usage: dump_embedded_template <model.hfq> [out.jinja]
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("usage: <model.hfq> [out.jinja]");
    let hfq = HfqFile::open(Path::new(model)).expect("open model");
    match hfq.chat_template() {
        Some(t) => {
            eprintln!("EMBEDDED chat_template present: {} bytes", t.len());
            if let Some(out) = args.get(2) {
                std::fs::write(out, t.as_bytes()).expect("write");
                eprintln!("wrote {out}");
            } else {
                print!("{t}");
            }
        }
        None => eprintln!("NO embedded chat_template (daemon would use ChatScaffold/Plain)"),
    }
}
