// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

use std::path::Path;
use hipfire_runtime::hfq::HfqFile;
use hipfire_arch_gemma4::lowered::{self as gemma4, Gemma4Scratch};
use hipfire_runtime::llama;
use rdna_compute::Gpu;

fn main() {
    let path = Path::new("/local/models/google/gemma-4-12B-it.hfq");
    let mut gpu = Gpu::init().expect("failed to init GPU");
    let mut hfq = HfqFile::open(path).expect("failed to open HFQ");

    let config = gemma4::config_from_hfq(&hfq).expect("failed config");
    let weights = gemma4::load_weights(&mut hfq, &config, &mut gpu).expect("failed weights");
    let scratch = Gemma4Scratch::new(&mut gpu, &config, 1).expect("failed scratch");
    gemma4::init_scratch_constants(&mut gpu, &scratch, config.full_head_dim).expect("failed init scratch");
    
    // Allocate caches
    let mut kv_sliding = llama::KvCache::new_gpu_asym3(
        &mut gpu, config.n_layers, config.sliding_n_kv_heads,
        config.sliding_head_dim, config.sliding_window,
    ).expect("failed sliding cache");
    
    let mut kv_full = llama::KvCache::new_gpu_asym3(
        &mut gpu, config.n_layers, config.full_n_kv_heads,
        config.full_head_dim, 2048,
    ).expect("failed full cache");
    
    // Set DUMP env var to 1 programmatically
    std::env::set_var("HIPFIRE_GEMMA4_DUMP", "1");
    
    println!("Running Layer 0 forward with BOS (2)...");
    gemma4::forward_scratch(
        &mut gpu, &weights, &config, 2, 0, &mut kv_sliding, &mut kv_full, &scratch,
    ).expect("BOS forward failed");
    
    println!("Running Layer 0 forward with 'Hello' (9259)...");
    gemma4::forward_scratch(
        &mut gpu, &weights, &config, 9259, 1, &mut kv_sliding, &mut kv_full, &scratch,
    ).expect("Hello forward failed");
}
