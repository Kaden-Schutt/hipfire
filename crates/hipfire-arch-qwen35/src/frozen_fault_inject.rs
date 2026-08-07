// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Env-var-driven fault injection for the Qwen35 Frozen construction path
//! (feature `frozen-fault-inject`).
//!
//! The stage injector (`HIPFIRE_FROZEN_FAIL_STAGE`) makes the next
//! construction attempt fail at a named stage AFTER that stage's GPU
//! allocations complete but BEFORE publishing, so the transactional
//! rollback paths are exercised end-to-end. Free-failure injection
//! (`HIPFIRE_FROZEN_FAIL_FREE`) lives in `rdna_compute`'s
//! `free_tensor_checked` (see `rdna_compute::frozen_fault_inject`), the
//! single choke point every checked-free family routes through.

/// The construction stage at which the next load attempt fails, or `None`.
///
/// Production builds (feature off) compile this to a constant `None` — no
/// environment variable is read, so no env value can alter serving.
///
/// Reads the LIVE process environment (`std::env`), NOT the hipfire-config
/// process snapshot: tests set the stage with `EnvGuard` after startup, and
/// `hipfire_config::developer_var` (a one-time `OnceLock` snapshot) would
/// never see it.
pub(crate) fn fail_stage() -> Option<&'static str> {
    #[cfg(feature = "frozen-fault-inject")]
    {
        match std::env::var("HIPFIRE_FROZEN_FAIL_STAGE") {
            Ok(v) => match v.as_str() {
                "common_fulfill" => Some("common_fulfill"),
                "common_assembly" => Some("common_assembly"),
                "moe_build" => Some("moe_build"),
                "kv_construct" => Some("kv_construct"),
                "dn_construct" => Some("dn_construct"),
                "scratch_construct" => Some("scratch_construct"),
                "mtp_upload" => Some("mtp_upload"),
                _ => None,
            },
            Err(_) => None,
        }
    }
    #[cfg(not(feature = "frozen-fault-inject"))]
    {
        None
    }
}
