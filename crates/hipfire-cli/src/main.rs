mod commands;
mod model;

use clap::{Parser, Subcommand};
use hipfire_server::load_config;

#[derive(Debug, Parser)]
#[command(name = "hipfire", version, about = "hipfire LLM inference CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the hipfire HTTP server (OpenAI-compatible)
    Serve(commands::serve::ServeArgs),
    /// Load a model and generate a response (one-shot)
    Run(commands::run::RunArgs),
    /// List locally available models
    #[command(alias = "models")]
    List,
    /// Run the quant admission/model evaluation harness
    Eval(commands::forward::EvalArgs),
    /// Measure host, GPU-copy, and model storage bandwidth
    HostProfile(commands::forward::HostProfileArgs),
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
    let config = load_config();

    match cli.command {
        Command::Serve(args) => commands::serve::run(args, config).await,
        Command::Run(args) => commands::run::run(args, config).await,
        Command::List => {
            commands::list::run();
            Ok(())
        }
        Command::Eval(args) => commands::forward::run_eval(args),
        Command::HostProfile(args) => commands::forward::run_host_profile(args),
    }
}
