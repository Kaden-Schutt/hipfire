// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Kernel Atlas: typed schema + JSONL writer + analysis helpers for the
//! hipfire bench corpus.
//!
//! See `docs/methodology/kernel-atlas-architecture.md` for the
//! three-layer split:
//!
//! 1. **Collection (this crate)** — bench/inference binaries emit typed
//!    [`AtlasRow`] values via `--emit-atlas <path>`. No stdout-scraping.
//! 2. **Analysis** — `scripts/kernel_atlas.py` (on the HIPa branch)
//!    handles ranking and pandas-style iteration.
//! 3. **Advisor** — future autotuner / advisor model consumes the corpus.
//!
//! The legacy stdout parsers in [`parse`] remain as a migration bridge
//! for captures from binaries not yet wired with `--emit-atlas`.

pub mod eval;
pub mod parse;
pub mod profile_report;
pub mod render;
pub mod schema;
pub mod suggest;
pub mod task;

pub use profile_report::{AtlasProfileReport, AtlasRocprofKernel};
pub use schema::{
    load_row, load_rows, truncate_jsonl, value_object, AtlasRow, ATLAS_SCHEMA,
};
