// SPDX-License-Identifier: Apache-2.0
//! Per-family model adapters for the hipfire-steer driver.
//!
//! The generic driver, scoring, and the `ModelHarness` trait live in
//! `hipfire-steer`. This crate only provides the family-specific adapters that
//! implement that trait (so the driver never depends on an arch crate, avoiding
//! the gemma3 → hipfire-steer cycle), plus a dispatcher keyed on the HFQ
//! `arch_id`.
//!
//! Adding a family: drop in `src/<family>.rs` implementing `ModelHarness`, add a
//! match arm below, and wire `maybe_steer_block` into that arch's forward.

pub mod gemma3;

use std::path::Path;

use hipfire_runtime::hfq::HfqFile;
use hipfire_steer::driver::ModelHarness;
use rdna_compute::Gpu;

/// Load the right family harness for an HFQ, dispatching on its `arch_id`.
pub fn build_harness(
    gpu: Gpu,
    hfq_path: &Path,
    max_seq: usize,
    max_new_tokens: usize,
) -> Result<Box<dyn ModelHarness>, String> {
    let hfq = HfqFile::open(hfq_path).map_err(|e| format!("open hfq {hfq_path:?}: {e}"))?;
    match hfq.arch_id {
        // 12 = Gemma3ForCausalLM (text-only); 13 = Gemma3ForConditionalGeneration
        // (multimodal wrapper). Same gemma3 text decoder — the text forward
        // ignores the vision tensors, and our steer hook lives there.
        12 | 13 => Ok(Box::new(gemma3::Gemma3Harness::load(
            gpu,
            hfq,
            max_seq,
            max_new_tokens,
        )?)),
        other => Err(format!(
            "hipfire-steer-harness: no harness for arch_id {other} (gemma3 = 12|13 wired)"
        )),
    }
}
