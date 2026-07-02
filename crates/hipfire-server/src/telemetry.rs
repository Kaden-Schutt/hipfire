//! Host GPU telemetry for the `/admin` dashboard.
//!
//! The collector moved to the HIP-free, portable `hipfire-sysinfo` crate so the
//! TUI can reuse it without pulling in the server. This module re-exports the
//! reader to keep `crate::telemetry::read_gpu_telemetry` stable for existing
//! callers; new code should prefer [`hipfire_sysinfo::snapshot`], which also
//! carries host system memory.

pub use hipfire_sysinfo::read_gpu_telemetry;
