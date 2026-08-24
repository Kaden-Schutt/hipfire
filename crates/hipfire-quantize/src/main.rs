// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-quantize: Quantize raw FP16/BF16/FP32 model weights to Q4_F16 format.
//!
//! Usage: hipfire-quantize --input <model_dir-or-gguf> --output <output.hfq> [--format mq4]
//!
//! Reads safetensors files from a HuggingFace model directory OR a single
//! `.gguf` file and produces a `.hfq` (HipFire Quantized) file with
//! RDNA-native quantized weights.

mod e8;
mod e8_gptq;
mod gguf_input;
mod reap_overlay;
mod cli;
mod dequant;
mod quant_q4;
mod quant_fwht;
mod quant_hfp4;
mod quant_e8;
mod quant_mq;
mod hfq;
mod model_filter;
mod calibration;
mod pipeline_gguf;
mod pipeline_deepseek;
mod pipeline;
#[cfg(test)]
mod diagnostics;

use clap::Parser;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub(crate) use calibration::{AWQ_ALPHA, IMATRIX};

fn main() {
    pipeline::run();
}
