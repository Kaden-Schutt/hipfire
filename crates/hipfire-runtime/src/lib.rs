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
#[cfg(feature = "deltanet")]
pub mod cask;
pub mod coherence_runtime;
pub mod config;
#[cfg(feature = "deltanet")]
pub mod cpu_router;
#[cfg(feature = "deltanet")]
pub mod ddtree;
#[cfg(feature = "deltanet")]
pub mod dflash;
pub mod env_docs;
pub mod eos_filter;
pub mod ep;
pub mod eval_harness;
pub mod gguf;
pub mod hfq;
pub mod hfq_modules;
pub mod host_profile;
pub mod llama;
pub mod logging;
pub mod loop_guard;
pub mod model_source;
pub mod mtp_mirror;
pub mod multi_gpu;
pub mod safetensors_source;
pub mod sampler;
pub mod speed_bench;
pub mod tokenizer;
pub mod tool_call;
pub mod tp_shard;
#[cfg(feature = "deltanet")]
pub mod triattn;
#[cfg(feature = "deltanet")]
pub mod weight_pager;
pub mod prompt_frame {
    pub use hipfire_prompt::*;

    impl hipfire_prompt::PromptTokenizer for crate::tokenizer::Tokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            crate::tokenizer::Tokenizer::encode(self, text)
        }

        fn special_token_id(&self, content: &str) -> Option<u32> {
            crate::tokenizer::Tokenizer::special_token_id(self, content)
        }

        fn special_tokens(&self) -> &[(String, u32)] {
            crate::tokenizer::Tokenizer::special_tokens(self)
        }

        fn bos_token_text(&self) -> String {
            let bytes = self.decode_bytes(&[self.bos_id]);
            String::from_utf8_lossy(&bytes).to_string()
        }
    }
}
