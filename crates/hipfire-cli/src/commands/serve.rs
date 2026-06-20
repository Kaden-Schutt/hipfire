use clap::Args;
use hipfire_config::{ConfigLayer, ConfigLayerKind, LoadedConfig};
use serde_json::json;

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

pub async fn run(args: ServeArgs, config: LoadedConfig) -> anyhow::Result<()> {
    let mut cli_layer = ConfigLayer::new(ConfigLayerKind::Cli);
    if let Some(h) = args.host {
        cli_layer.values.insert("host".to_string(), json!(h));
    }
    if let Some(p) = args.port {
        cli_layer.values.insert("port".to_string(), json!(p));
    }
    if let Some(m) = args.model {
        cli_layer
            .values
            .insert("default_model".to_string(), json!(m));
    }
    hipfire_server::serve_loaded(config.with_additional_layer(cli_layer)).await
}
