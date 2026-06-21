mod commands;
mod model;

use clap::{Parser, Subcommand};
use hipfire_config::load_config_bundle;

#[derive(Debug, Parser)]
#[command(
    name = "hipfire",
    version = hipfire_build_info::VERSION,
    about = "hipfire LLM inference CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the hipfire HTTP server (OpenAI-compatible)
    Serve(commands::serve::ServeArgs),
    /// Load a model and generate a response (one-shot)
    Chat(commands::chat::ChatArgs),
    /// List locally available models
    #[command(alias = "models")]
    List,
    /// Run the quant admission/model evaluation harness
    Eval(commands::forward::EvalArgs),
    /// Measure host, GPU-copy, and model storage bandwidth
    HostProfile(commands::forward::HostProfileArgs),
    /// Collect Tier-1 calibration artifacts (Hessian/imatrix/router-histogram) in one model load
    CollectArtifacts(commands::forward::CollectArtifactsArgs),
    /// GPU mutex for multi-agent coordination (acquire/release/status)
    GpuLock(commands::gpu_lock::GpuLockArgs),
    /// Query the running hipfire admin API for scripts and agents
    #[command(alias = "op")]
    Admin(commands::admin::AdminArgs),
    /// Regenerate the committed CLI docs (docs/cli.md + man pages) from this
    /// clap definition. Hidden: a maintenance command, not part of the
    /// user-facing surface; run via `cargo run -p hipfire-cli -- gen-docs`.
    #[command(hide = true)]
    GenDocs(commands::gen_docs::GenDocsArgs),
    /// Render the shared config schema. Hidden: maintenance command for docs
    /// and operator UI schema artifacts.
    #[command(hide = true)]
    GenConfigSchema(commands::gen_config_schema::GenConfigSchemaArgs),
    /// Regenerate the committed env-var docs (docs/env-vars.md +
    /// crates/hipfire-runtime/src/env_docs.rs) by scanning the source tree.
    /// Hidden: a maintenance command; run via
    /// `cargo run -p hipfire-cli -- gen-env-docs`.
    #[command(hide = true)]
    GenEnvDocs(commands::gen_env_docs::GenEnvDocsArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hipfire=info,hipfire_server=info,warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let loaded_config = load_config_bundle();
    let config = loaded_config.config.clone();

    match cli.command {
        Command::Serve(args) => commands::serve::run(args, loaded_config).await,
        Command::Chat(args) => commands::chat::run(args, config).await,
        Command::List => {
            commands::list::run();
            Ok(())
        }
        Command::Eval(args) => commands::forward::run_eval(args),
        Command::HostProfile(args) => commands::forward::run_host_profile(args),
        Command::CollectArtifacts(args) => commands::forward::run_collect_artifacts(args),
        Command::GpuLock(args) => commands::gpu_lock::run(args),
        Command::Admin(args) => commands::admin::run(args, config).await,
        Command::GenDocs(args) => commands::gen_docs::run(args),
        Command::GenConfigSchema(args) => commands::gen_config_schema::run(args),
        Command::GenEnvDocs(args) => commands::gen_env_docs::run(args),
    }
}
