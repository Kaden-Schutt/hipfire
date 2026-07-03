// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Library surface for the hipfire quantizer.
//!
//! Historically this crate was binary-only (`main.rs` owned every module). The
//! pure quantization codecs, the LDLQ/GPTQ calibration machinery, and the
//! Hessian sidecar I/O are useful to other crates — notably `hipfire-diffusion`,
//! which reuses the oq4/oq8 packers and decoders for activation-calibrated
//! diffusion weight quantization. Those modules now live here and the
//! `hipfire-quantize` binary (`main.rs`) consumes this same library via
//! `use hipfire_quantize::…`.
//!
//! Crate-root helpers (`cpu_fwht_256`, `gen_fwht_signs`, `f16_to_f32`,
//! `f32_to_f16`) are re-exported from `hipfire-primitives` so the in-crate
//! `crate::{…}` references inside the modules below keep resolving unchanged.

use std::sync::OnceLock;

pub use hipfire_primitives::conv::{f16_to_f32, f32_to_f16};
pub use hipfire_primitives::fwht::{cpu_fwht_256, gen_fwht_signs};
pub use hipfire_kvquant::{kv_compact, kvarn};

pub mod codecs;
pub mod hfq_out;
pub mod gptq;
pub mod hessian_io;
#[allow(dead_code)]
pub mod hfhs_diag;
#[allow(dead_code)]
pub mod ldlq;
// QTIP encoder core: some helpers are not yet wired into the dispatch.
#[allow(dead_code)]
pub mod qtip;
pub mod roughquant;
pub mod fixture;

// Process-global toggle for the `mqN+` clip-search codec variant. Lives in the
// library so the codecs (which read it via `crate::mq_clipsearch_enabled`) and
// the binary (which arms it from a CLI flag via `set_mq_clipsearch`) share one
// source of truth.
static MQ_CLIPSEARCH: OnceLock<bool> = OnceLock::new();

/// Whether the `mqN+` clip-search variant is active for MQ codecs.
pub fn mq_clipsearch_enabled() -> bool {
    MQ_CLIPSEARCH.get().copied().unwrap_or(false)
}

/// Arm the `mqN+` clip-search variant (idempotent; first set wins).
pub fn set_mq_clipsearch(enabled: bool) {
    let _ = MQ_CLIPSEARCH.set(enabled);
}
