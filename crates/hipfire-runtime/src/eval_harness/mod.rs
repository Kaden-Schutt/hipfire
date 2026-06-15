// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compatibility facade for the eval harness.
//!
//! The harness implementation and `hipfire-eval` runner entrypoint live in
//! `hipfire-eval`. Runtime keeps this module as a stable import path while
//! downstream callers migrate.

pub use hipfire_eval::*;
