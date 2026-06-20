// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-serving-core` — shared model-serving orchestration.
//!
//! This crate holds the load + generate + session/memory/event plumbing that
//! was historically buried inside the `hipfire-daemon` binary. Extracting it
//! lets both the daemon (a thin JSONL protocol shell) and the `hipfire-eval`
//! harness drive the same serving paths — the daemon via stdin/stdout, eval
//! in-process — without forking the orchestration logic.
//!
//! Migration is incremental (plan: `docs/plans/2026-06-20-serving-core-eval-
//! lock-extraction.md`, workstream A). Modules move out of the daemon bin
//! bottom-up by dependency; the daemon re-points its `use` paths here as each
//! lands. No behavior change per move.

pub mod dummy;
pub mod events;
pub mod generate_vl;
pub mod load;
pub mod memory;
pub mod model;
pub mod output_filter;
pub mod qwen35_decode;
pub mod qwen35_prefill;
pub mod session;
