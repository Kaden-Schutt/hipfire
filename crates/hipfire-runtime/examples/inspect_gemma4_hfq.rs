// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

use std::path::Path;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;

fn main() {
    let path = Path::new("/local/models/google/gemma-4-12B-it.hfq");
    match HfqFile::open(path) {
        Ok(hfq) => {
            println!("Loaded HFQ model. arch_id = {}", hfq.arch_id);
            let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).unwrap();
            println!("Metadata tie_word_embeddings: {:?}", meta.get("config").and_then(|c| c.get("tie_word_embeddings")));
            println!("Metadata text_config tie_word_embeddings: {:?}", meta.get("config").and_then(|c| c.get("text_config")).and_then(|tc| tc.get("tie_word_embeddings")));
            let has_lm_head = hfq.find_tensor_info("model.language_model.lm_head.weight").is_some() 
                || hfq.find_tensor_info("lm_head.weight").is_some()
                || hfq.find_tensor_info("model.lm_head.weight").is_some();
            println!("Has explicit lm_head in HFQ: {}", has_lm_head);
            if let Some((info, data)) = hfq.tensor_data("model.language_model.norm.weight") {
                println!("model.language_model.norm.weight quant_type: {}", info.quant_type);
                let f32_data: Vec<f32> = data.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                println!("model.language_model.norm.weight (first 10): {:?}", &f32_data[..10.min(f32_data.len())]);
            }
            if let Some((info, data)) = hfq.tensor_data("model.language_model.layers.0.input_layernorm.weight") {
                println!("layer 0 input_layernorm.weight quant_type: {}", info.quant_type);
                let f32_data: Vec<f32> = data.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                println!("layer 0 input_layernorm.weight (first 10): {:?}", &f32_data[..10.min(f32_data.len())]);
            }
        }
        Err(e) => {
            println!("Failed to open HFQ: {:?}", e);
        }
    }
}
