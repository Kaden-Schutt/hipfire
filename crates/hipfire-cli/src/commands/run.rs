use std::io::Write;

use clap::Args;
use hipfire_daemon_adapter::{find_daemon_bin, DaemonEngine};
use hipfire_daemon_protocol::{GenerateRequest, GenerationSamplingPolicy, LoadParams};
use hipfire_server::HipfireConfig;
use uuid::Uuid;

use crate::model::find_model;

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Model name, alias, or path
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Max tokens to generate
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// Sampling temperature
    #[arg(long)]
    pub temperature: Option<f64>,
}

fn load_params_from_config(config: &HipfireConfig) -> LoadParams {
    LoadParams::from_common_config_values(
        config.max_seq,
        &config.kv_cache,
        &config.flash_mode,
        &config.dflash_mode,
        config.cask_sidecar.as_deref(),
    )
}

pub async fn run(args: RunArgs, config: HipfireConfig) -> anyhow::Result<()> {
    let model_path = find_model(&args.model)
        .ok_or_else(|| anyhow::anyhow!("model not found: {}", args.model))?;

    let bin = find_daemon_bin().ok_or_else(|| {
        anyhow::anyhow!(
            "daemon binary not found; build with: cargo build -p hipfire-daemon --bin hipfire-daemon"
        )
    })?;

    eprintln!("Loading {}…", model_path.display());
    let mut engine = DaemonEngine::spawn(&bin).await?;

    engine
        .load(
            &model_path.to_string_lossy(),
            load_params_from_config(&config),
        )
        .await?;

    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );

    let gen_req = GenerateRequest {
        id: Uuid::new_v4().to_string(),
        prompt,
        messages: None,
        sampling: GenerationSamplingPolicy {
            temperature: args.temperature.unwrap_or(config.temperature),
            max_tokens: args.max_tokens.unwrap_or(config.max_tokens),
            top_p: Some(config.top_p),
            repeat_penalty: Some(config.repeat_penalty),
        },
        worker_key_id: engine.worker_key_id.clone(),
        tools: None,
        system: None,
        thinking: None,
        max_think_tokens: None,
        request_id: None,
    };

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
}
