//! The EP executor moved to `hipfire-dispatch` (Phase 0b of the device-mesh
//! work) so the unified `run_layer_program` can live in one crate alongside
//! the super-op IR + `ForwardBindings`. Re-exported here so existing
//! `hipfire_runtime::ep::…` callers keep working unchanged.
pub use hipfire_dispatch::ep::*;
