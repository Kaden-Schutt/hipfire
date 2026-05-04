//! engine: GGUF model loading and LLaMA inference on RDNA GPUs.

pub mod gguf;
pub mod hfq;
pub mod hfq4v4;
pub mod llama;
#[cfg(feature = "deltanet")]
pub mod qwen35;
#[cfg(feature = "deltanet")]
pub mod qwen35_vl;
#[cfg(feature = "deltanet")]
pub mod speculative;
#[cfg(feature = "deltanet")]
pub mod dflash;
#[cfg(feature = "deltanet")]
pub mod ddtree;
#[cfg(feature = "deltanet")]
pub mod triattn;
#[cfg(feature = "deltanet")]
pub mod cask;
#[cfg(feature = "deltanet")]
pub mod cpu_router;
#[cfg(feature = "deltanet")]
pub mod weight_pager;
pub mod image;
pub mod tokenizer;
pub mod pflash;
