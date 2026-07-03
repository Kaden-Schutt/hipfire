// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-runtime: GGUF model loading and LLaMA inference on RDNA GPUs.
//!
//! This crate is arch-agnostic. Architecture implementations live in
//! sibling crates (`hipfire-arch-qwen35`, `hipfire-arch-qwen35-vl`,
//! future `hipfire-arch-llama`, etc.) and depend on this crate for
//! shared infrastructure: HFQ/GGUF file readers, the LLaMA-style
//! scratch / KV / sampler primitives, tokenizer, prompt framing, eos
//! filter, loop guard, eviction (TriAttn, CASK), spec-decode primitives
//! (DFlash, DDTree), demand paging (cpu_router, weight_pager), and the
//! [`arch::Architecture`] trait.

pub mod arch;
pub mod bf16_loader;
pub mod calibration;
#[cfg(feature = "deltanet")]
pub mod cask;
pub mod config;
#[cfg(feature = "deltanet")]
pub mod cpu_router;
#[cfg(feature = "deltanet")]
pub mod ddtree;
#[cfg(feature = "deltanet")]
pub mod dflash;
pub mod dispatch;
pub mod env_docs;
pub mod eos_filter;
pub mod ep;
pub mod hfq;
pub mod hfq_modules;
pub mod host_profile;
pub mod kld_eval;
pub mod kv;
pub mod kv_hier;
pub mod llama;
pub mod logging;
pub mod loop_guard;
pub mod model_source;
pub mod mtp_mirror;
pub mod multi_gpu;
pub mod quant;
pub mod safetensors_source;
pub mod sampler;
pub mod sequence_state;
pub mod speed_bench;
pub mod tokenizer;
pub mod tool_call;
pub mod tp_shard;
pub mod transformer;
#[cfg(feature = "deltanet")]
pub mod triattn;
#[cfg(feature = "deltanet")]
pub mod weight_pager;
pub mod weights;
