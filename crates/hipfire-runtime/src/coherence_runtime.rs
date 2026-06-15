// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compatibility facade for coherence runtime orchestration.
//!
//! Coherence detector policy, report serialization, and daemon-backed runner
//! orchestration live in `hipfire-coherence`. Runtime keeps this module as a
//! stable import path for `coherence_probe` and eval harness callers while
//! modularization continues.

pub use hipfire_coherence::*;
