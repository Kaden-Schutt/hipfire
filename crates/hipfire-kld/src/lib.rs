// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-kld — the pure, GPU-independent KLD scoring core.
//!
//! This crate is the single source of truth for quant-quality KLD evaluation
//! math and formats. It exists so that **reference build** and **candidate
//! scoring** share byte-identical reduction/scoring/config code and therefore
//! cannot drift — the failure mode (`docs/plans/eval-tooling-refactor.md`) where
//! two standalone binaries computed the "same" thing differently and produced a
//! spurious self-inconsistency.
//!
//! Nothing here touches the GPU. Logits come in as `&[f32]` (downloaded by the
//! caller); everything reduces/accumulates in `f64`. The crate is exercised by
//! CPU unit tests in `no-gpu-ci`.
//!
//! Modules:
//! - [`math`]   — `log_z`, [`top_k_log_softmax`](math::top_k_log_softmax) (ref
//!   reduction), [`score_position`](math::score_position) (candidate KLD + NLL).
//! - [`config`] — [`KldConfig`](config::KldConfig): the one env contract shared
//!   by ref-build and score.
//! - [`refblock`] — [`RefBlock`](refblock::RefBlock) reference-distribution
//!   block + canonical (de)serialize.
//! - [`hfkseq`] — per-chunk result file (`HFKSEQ`) read/write.
//! - [`meta`]   — [`RefMeta`](meta::RefMeta) self-describing header +
//!   [`compat`](meta::compat) guard against cross-version/config/arch scoring.
//! - [`codec`]  — per-blob payload codecs (bit-packed ids; reserved fp16/zstd).

pub mod archive;
pub mod codec;
pub mod config;
pub mod hfkseq;
pub mod math;
pub mod meta;
pub mod refblock;

pub use archive::RefArchive;
pub use codec::BlobCodec;
pub use config::KldConfig;
pub use hfkseq::ChunkResult;
pub use math::{log_z, score_position, top_k_log_softmax, PositionScore, TopKReduction};
pub use meta::{compat, CompatReport, Mismatch, ProducerInfo, RefMeta, RunEnv, Severity};
pub use refblock::RefBlock;
