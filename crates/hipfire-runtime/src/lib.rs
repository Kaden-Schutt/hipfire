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

pub mod admission;
pub mod arch;
pub mod arch_spec;
pub mod augmentor;
pub mod bf16_loader;
pub mod cache_plan;
#[cfg(feature = "deltanet")]
pub mod cask;
pub mod config;
#[cfg(feature = "deltanet")]
pub mod cpu_router;
#[cfg(feature = "deltanet")]
pub mod ddtree;
#[cfg(feature = "deltanet")]
pub mod dflash;
pub mod dflash_generic;
pub mod dspark_block_controller;
pub mod dspark_core;
pub mod ep;
pub mod eval_common;
pub mod gguf;
pub mod hfq;
pub mod kv_adaptive;
pub mod kv_backend;
pub mod kv_mode;
pub mod llama;
pub mod llama_spec;
pub mod loader_api;
pub mod loop_guard;
pub mod model_load;
pub mod model_source;
pub mod multi_gpu;
pub mod paro;
pub mod prefix;
pub mod reset_core;
pub mod safetensors_source;
pub mod sampler;
pub mod spec;

pub mod spec_ngram;
pub mod tp_shard;
#[cfg(feature = "deltanet")]
pub mod triattn;
#[cfg(feature = "deltanet")]
pub mod weight_pager;

pub mod emit_text;
pub mod eos_filter;
pub mod prompt_frame;
pub mod semantic;
pub mod session_table;
pub mod tokenizer;

pub mod tool_call;
pub mod weight_backend;

pub use crate::arch::{maybe_screen_mmq, screen_weight_tensor, MmqScreenable};
