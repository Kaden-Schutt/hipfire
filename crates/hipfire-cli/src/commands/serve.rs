use clap::Args;
use hipfire_server::HipfireConfig;

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Override bind host
    #[arg(long)]
    pub host: Option<String>,
    /// Override bind port
    #[arg(long, short)]
    pub port: Option<u16>,
    /// Pre-load a model on startup
    #[arg(long, short)]
    pub model: Option<String>,
}

pub async fn run(args: ServeArgs, mut config: HipfireConfig) -> anyhow::Result<()> {
    if let Some(h) = args.host {
        config.host = h;
    }
    if let Some(p) = args.port {
        config.port = p;
    }
    if let Some(m) = args.model {
        config.default_model = Some(m);
    }
    hipfire_server::serve(config).await
}
