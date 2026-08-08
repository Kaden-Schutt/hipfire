//! `multi_gpu` moved to the leaf crate `hipfire-hardware` (Phase 0 of the
//! device-mesh work) so `hipfire-dispatch` can depend on the collective /
//! `Gpus` layer without a dispatch→runtime cycle. Re-exported here so every
//! existing `hipfire_runtime::multi_gpu::…` path keeps working unchanged.
pub use hipfire_hardware::*;
