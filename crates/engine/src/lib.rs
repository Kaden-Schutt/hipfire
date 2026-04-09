//! engine: GGUF model loading and LLaMA inference on RDNA GPUs.

pub mod gguf;
pub mod hfq;
pub mod llama;
#[cfg(feature = "deltanet")]
pub mod qwen35;
#[cfg(feature = "deltanet")]
pub mod qwen35_vl;
#[cfg(feature = "deltanet")]
pub mod speculative;
pub mod image;
pub mod tokenizer;
