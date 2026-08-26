//! Shared model-format helpers used by the quantizer binaries.

pub mod float16;
pub mod gptq;
pub mod hessian_io;
pub mod hfhs_diag;
pub mod hfqm;
pub mod safetensors_file;

use std::sync::OnceLock;

static MQ_CLIPSEARCH: OnceLock<bool> = OnceLock::new();

/// Whether the `mqN+` clip-search variant is active for MQ codecs.
pub fn mq_clipsearch_enabled() -> bool {
    MQ_CLIPSEARCH.get().copied().unwrap_or(false)
}

/// Arm the `mqN+` clip-search variant (idempotent; first set wins).
pub fn set_mq_clipsearch(enabled: bool) {
    let _ = MQ_CLIPSEARCH.set(enabled);
}
