use std::io::Write;

use clap::Args;
use hipfire_server::{
    daemon::{
        engine::{find_daemon_bin, DaemonEngine},
        protocol::{GenerateRequest, LoadParams},
    },
    model::discovery::find_model,
    HipfireConfig,
};
use uuid::Uuid;

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

    let params = LoadParams {
        max_seq: config.max_seq,
        ..Default::default()
    };
    engine.load(&model_path.to_string_lossy(), params).await?;

    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );

    let gen_req = GenerateRequest {
        id: Uuid::new_v4().to_string(),
        prompt,
        temperature: args.temperature.unwrap_or(config.temperature),
        max_tokens: args.max_tokens.unwrap_or(config.max_tokens),
        top_p: Some(config.top_p),
        repeat_penalty: Some(config.repeat_penalty),
        worker_key_id: engine.worker_key_id.clone(),
        tools: None,
        system: None,
        thinking: None,
        max_think_tokens: None,
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
