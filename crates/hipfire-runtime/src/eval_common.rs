// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Compatibility facade for eval integrity helpers.
//!
//! Evidence-owned eval provenance and integrity checks live in
//! `hipfire-evidence`. Runtime keeps this module as a stable import path for
//! existing examples until eval execution moves out of runtime.

pub use hipfire_evidence::{verify_llama_commit, verify_ref_sha256, verify_slice_md5};
