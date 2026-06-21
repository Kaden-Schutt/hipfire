use std::io::Write;

use clap::Args;
use hipfire_config::HipfireConfig;
use hipfire_daemon_adapter::{find_daemon_bin_or_error, DaemonEngine};
use hipfire_generate::{GenerateTextRequest, GenerationSamplingPolicy};
use hipfire_model::ModelLoadParams;
use uuid::Uuid;

use crate::model::find_model;

#[derive(Debug, Args)]
pub struct ChatArgs {
    /// Model name, alias, or path. Falls back to the `default_model` config
    /// value when omitted.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Prompt text
    pub prompt: String,
    /// Max tokens to generate
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature
    #[arg(long)]
    pub temperature: Option<f64>,
}

fn load_params_from_config(config: &HipfireConfig) -> ModelLoadParams {
    ModelLoadParams::from_hipfire_config(config)
}

fn generate_request_from_prompt(
    id: String,
    prompt: &str,
    sampling: GenerationSamplingPolicy,
    worker_key_id: Option<String>,
) -> GenerateTextRequest {
    GenerateTextRequest::from_prompt(id, prompt, sampling).with_worker_key_id(worker_key_id)
}

pub async fn run(args: ChatArgs, config: HipfireConfig) -> anyhow::Result<()> {
    let model = args
        .model
        .as_deref()
        .or(config.default_model.as_deref())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model specified and no `default_model` configured; \
                 pass --model <name> or set `default_model` in {}",
                hipfire_config::config_path().display()
            )
        })?;
    let model_path =
        find_model(model).ok_or_else(|| anyhow::anyhow!("model not found: {model}"))?;

    let bin = find_daemon_bin_or_error()?;

    eprintln!("Loading {}…", model_path.display());
    let mut engine = DaemonEngine::spawn(&bin).await?;

    engine
        .load(
            &model_path.to_string_lossy(),
            load_params_from_config(&config),
        )
        .await?;

    let gen_req = generate_request_from_prompt(
        Uuid::new_v4().to_string(),
        &args.prompt,
        GenerationSamplingPolicy::from_defaults(
            config.temperature,
            config.top_p,
            config.repeat_penalty,
            config.max_tokens,
            args.temperature,
            None,
            None,
            args.max_tokens,
        ),
        engine.worker_key_id.clone(),
    );

    let done = engine
        .generate_streaming(gen_req, |token| {
            print!("{token}");
            let _ = std::io::stdout().flush();
        })
        .await?;

    println!();
    eprintln!(
        "\n[{} tokens, {:.2} tok/s]",
        done.tokens,
        done.decode_tok_s.unwrap_or(0.0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_params_from_config_preserves_cli_load_policy() {
        let config = HipfireConfig {
            max_seq: 8192,
            kv_cache: "asym3".to_string(),
            flash_mode: "auto".to_string(),
            dflash_mode: "off".to_string(),
            cask_sidecar: Some("/models/qwen3.5-27b.triattn.hfq".to_string()),
            ..Default::default()
        };

        let params = load_params_from_config(&config);

        assert_eq!(params.max_seq, 8192);
        assert_eq!(params.kv_cache.as_deref(), Some("asym3"));
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode.as_deref(), Some("off"));
        assert_eq!(
            params.cask_sidecar.as_deref(),
            Some("/models/qwen3.5-27b.triattn.hfq")
        );
    }

    #[test]
    fn load_params_from_config_omits_auto_and_empty_sidecar() {
        let config = HipfireConfig {
            max_seq: 4096,
            kv_cache: "auto".to_string(),
            flash_mode: "auto".to_string(),
            dflash_mode: "auto".to_string(),
            cask_sidecar: Some(String::new()),
            ..Default::default()
        };

        let params = load_params_from_config(&config);

        assert_eq!(params.max_seq, 4096);
        assert_eq!(params.kv_cache, None);
        assert_eq!(params.flash_mode, None);
        assert_eq!(params.dflash_mode, None);
        assert_eq!(params.cask_sidecar, None);
    }

    #[test]
    fn generate_request_from_prompt_preserves_structured_boundary() {
        let req = generate_request_from_prompt(
            "req-1".to_string(),
            "hello",
            GenerationSamplingPolicy::greedy(8),
            Some("worker-a".to_string()),
        );

        assert_eq!(req.prompt, "hello");
        assert!(!req.prompt.contains("<|im_start|>"));
        assert_eq!(req.worker_key_id.as_deref(), Some("worker-a"));
        let messages = req.messages.as_ref().expect("structured messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(serde_json::to_value(&messages[0]).unwrap()["role"], "user");
        assert_eq!(messages[0].content, "hello");
    }
}
