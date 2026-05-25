// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! QA mirror for Qwen3.5 HFQ loading and config validation.

use hipfire_runtime::gguf::GgufFile;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::any::Any;
use std::path::Path;
use std::process::ExitCode;

const SKIP_EXIT: u8 = 10;
const QWEN_GGUF_FALLBACK: &str = "/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("QWEN35_TEST_MODEL").ok());

    let path = match path {
        Some(path) => path,
        None => {
            eprintln!("Qwen35 load QA SKIP: pass <model.hfq> or set QWEN35_TEST_MODEL");
            return ExitCode::from(SKIP_EXIT);
        }
    };

    match run(&path) {
        Ok(msg) => {
            eprintln!("Qwen35 load QA PASS: {msg}");
            ExitCode::SUCCESS
        }
        Err(Outcome::Skip(msg)) => {
            eprintln!("Qwen35 load QA SKIP: {msg}");
            ExitCode::from(SKIP_EXIT)
        }
        Err(Outcome::Fail(msg)) => {
            eprintln!("Qwen35 load QA FAIL: {msg}");
            ExitCode::from(1)
        }
    }
}

enum Outcome {
    Skip(String),
    Fail(String),
}

fn run(path: &str) -> Result<String, Outcome> {
    let mut hfq = HfqFile::open(Path::new(path))
        .map_err(|e| Outcome::Fail(format!("failed to open HFQ: {e}")))?;
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .map_err(|e| Outcome::Fail(format!("bad metadata JSON: {e}")))?;
    let config = meta
        .get("config")
        .ok_or_else(|| Outcome::Fail("no config in metadata".to_string()))?;
    let text_config = config.get("text_config").unwrap_or(config);

    let model_type = text_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let layers = text_config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let heads = text_config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let kv_heads = text_config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let vocab = text_config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if layers == 0 || heads == 0 || kv_heads == 0 || vocab == 0 {
        return Err(Outcome::Fail(format!(
            "invalid config counts: layers={layers} heads={heads} kv_heads={kv_heads} vocab={vocab}"
        )));
    }

    if !is_qwen35_candidate(model_type, &hfq) {
        return Err(Outcome::Skip(format!(
            "model_type={model_type} is not a Qwen3.5 text HFQ layout"
        )));
    }

    let (tokenizer, tokenizer_source) = load_tokenizer(&hfq)?;

    #[cfg(feature = "deltanet")]
    {
        let q35_config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq)
            .ok_or_else(|| Outcome::Fail("failed to parse qwen35 config".to_string()))?;
        let linear_layers = q35_config
            .layer_types
            .iter()
            .filter(|t| matches!(t, hipfire_arch_qwen35::qwen35::LayerType::LinearAttention))
            .count();
        let full_layers = q35_config.n_layers.saturating_sub(linear_layers);
        let mut gpu = rdna_compute::Gpu::init()
            .map_err(|e| Outcome::Skip(format!("GPU init unavailable: {e}")))?;
        let weights = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hipfire_arch_qwen35::qwen35::load_weights(&mut hfq, &q35_config, &mut gpu)
        }))
        .map_err(|panic| Outcome::Fail(format!("weight load panicked: {}", panic_message(panic))))?
        .map_err(|e| Outcome::Fail(format!("weight load failed: {e}")))?;
        if weights.layers.len() != q35_config.n_layers {
            return Err(Outcome::Fail(format!(
                "loaded {} layers but config says {}",
                weights.layers.len(),
                q35_config.n_layers
            )));
        }

        return Ok(format!(
            "model_type={model_type} layers={} ({} linear + {} full) heads={} kv_heads={} vocab={} tokenizer={} source={}",
            q35_config.n_layers,
            linear_layers,
            full_layers,
            q35_config.n_heads,
            q35_config.n_kv_heads,
            q35_config.vocab_size,
            tokenizer.vocab_size(),
            tokenizer_source,
        ));
    }

    #[allow(unreachable_code)]
    Ok(format!(
        "model_type={model_type} layers={layers} heads={heads} kv_heads={kv_heads} vocab={vocab} tokenizer={} source={}",
        tokenizer.vocab_size(),
        tokenizer_source,
    ))
}

fn is_qwen35_candidate(model_type: &str, hfq: &HfqFile) -> bool {
    model_type.starts_with("qwen3_5")
        || model_type == "qwen3.5"
        || hfq.tensor_data("model.language_model.embed_tokens.weight").is_some()
}

fn load_tokenizer(hfq: &HfqFile) -> Result<(Tokenizer, String), Outcome> {
    if let Ok(tokenizer) = Tokenizer::from_hfq_metadata(&hfq.metadata_json) {
        return Ok((tokenizer, "hfq-metadata".to_string()));
    }

    let fallback = Path::new(QWEN_GGUF_FALLBACK);
    if !fallback.exists() {
        return Err(Outcome::Skip(format!(
            "tokenizer metadata missing and fallback GGUF not found at {}",
            fallback.display()
        )));
    }

    let gguf = GgufFile::open(fallback)
        .map_err(|e| Outcome::Skip(format!("failed to open fallback GGUF tokenizer: {e}")))?;
    let tokenizer = Tokenizer::from_gguf(&gguf)
        .map_err(|e| Outcome::Skip(format!("failed to parse fallback GGUF tokenizer: {e}")))?;
    Ok((tokenizer, format!("gguf:{}", fallback.display())))
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
