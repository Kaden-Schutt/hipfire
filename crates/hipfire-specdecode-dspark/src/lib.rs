// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dspark — arch-agnostic DSpark speculative-decode seam + drafter core.
//!
//! This crate hosts the transparent speculative-decode seam ([`spec`]) and the
//! arch-agnostic DSpark drafter core ([`dspark_core`]) that were previously
//! carried in `hipfire-runtime`. The verifier boundary ([`spec::SpecTarget`]),
//! the drafter interface ([`spec::Speculator`] / [`spec::MtpDrafter`]), and the
//! generic DSpark drafter ([`dspark_core::DsparkDrafter`]) live here so an
//! arch crate implements only [`spec::SpecTarget`] + a [`dspark_core::DsparkBody`]
//! to gain DSpark spec-decode without pulling the whole runtime.
//!
//! Note: [`spec::SpecTarget`] here is DSpark's own verifier trait; it is a
//! distinct, intentionally-separate design from `hipfire_specdecode`'s
//! `SpecDecodeTarget` — the two do not share a boundary.

pub mod dspark_block_controller;
pub mod dspark_core;
pub mod spec;

pub use dspark_core::{
    build_dspark_speculator, main_proj_ingest, main_proj_ingest_batched, noise_block_ids,
    run_heads, DraftResult, DsparkBody, DsparkConfig, DsparkDrafter, DsparkWeights,
};
pub use spec::{
    accept_greedy_prefix, EvictRetain, GreedyAccept, MtpDrafter, MtpSpeculator, MtpWindow,
    PrefillOutcome, SpecAdvance, SpecGrammar, SpecScratch, SpecStep, SpecTarget, Speculator,
};
