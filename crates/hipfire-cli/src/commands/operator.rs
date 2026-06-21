use clap::{Args, Subcommand};
use hipfire_config::HipfireConfig;
use serde_json::{json, Value};

#[derive(Debug, Args)]
pub struct OperatorArgs {
    /// Override operator API host. Defaults to config host, with 0.0.0.0 mapped to 127.0.0.1.
    #[arg(long, global = true)]
    pub host: Option<String>,
    /// Override operator API port. Defaults to config port.
    #[arg(long, global = true)]
    pub port: Option<u16>,
    #[command(subcommand)]
    pub command: OperatorCommand,
}

#[derive(Debug, Subcommand)]
pub enum OperatorCommand {
    /// Combined status snapshot for scripts and agents
    Status,
    /// Raw /health payload
    Health,
    /// Local model registry from the operator API
    Models,
    /// Resolved runtime config
    Config {
        /// Resolve config for a specific model tag
        #[arg(long)]
        model: Option<String>,
    },
    /// Training run summaries or one run detail
    Training {
        /// Optional run ID
        id: Option<String>,
        /// Return full events for the run ID
        #[arg(long)]
        events: bool,
    },
    /// Filesystem, binary, kernel-cache, lock, and log diagnostics
    Diagnostics,
    /// Tail known hipfire logs
    Logs {
        /// Number of lines per log file
        #[arg(long, default_value_t = 120)]
        lines: usize,
    },
    /// GET an arbitrary operator/server path, e.g. /operator/training/runs
    Get {
        /// Absolute or relative server path
        path: String,
    },
}

pub async fn run(args: OperatorArgs, config: HipfireConfig) -> anyhow::Result<()> {
    let client = OperatorClient::new(args.host, args.port, &config);
    let value = match args.command {
        OperatorCommand::Status => {
            let health = client.get("/health").await?;
            let diagnostics = client.get("/operator/diagnostics").await?;
            let models = client.get("/operator/models/registry").await?;
            let training = client.get("/operator/training/runs").await?;
            json!({
                "base_url": client.base_url,
                "health": health,
                "diagnostics": diagnostics,
                "models": models,
                "training": training,
            })
        }
        OperatorCommand::Health => client.get("/health").await?,
        OperatorCommand::Models => client.get("/operator/models/registry").await?,
        OperatorCommand::Config { model } => {
            let path = match model {
                Some(model) => format!("/operator/config/resolved?model={}", url_encode(&model)),
                None => "/operator/config/resolved".to_string(),
            };
            client.get(&path).await?
        }
        OperatorCommand::Training { id, events } => match (id, events) {
            (Some(id), true) => {
                client
                    .get(&format!(
                        "/operator/training/runs/{}/events",
                        url_encode_path_segment(&id)
                    ))
                    .await?
            }
            (Some(id), false) => {
                client
                    .get(&format!(
                        "/operator/training/runs/{}",
                        url_encode_path_segment(&id)
                    ))
                    .await?
            }
            (None, _) => client.get("/operator/training/runs").await?,
        },
        OperatorCommand::Diagnostics => client.get("/operator/diagnostics").await?,
        OperatorCommand::Logs { lines } => {
            client
                .get(&format!("/operator/logs?lines={}", lines.clamp(1, 1000)))
                .await?
        }
        OperatorCommand::Get { path } => client.get(&normalize_path(&path)).await?,
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

struct OperatorClient {
    base_url: String,
    http: reqwest::Client,
}

impl OperatorClient {
    fn new(host: Option<String>, port: Option<u16>, config: &HipfireConfig) -> Self {
        let host = host.unwrap_or_else(|| probe_host_for(&config.host));
        let port = port.unwrap_or(config.port);
        Self {
            base_url: base_url_for(&host, port),
            http: reqwest::Client::new(),
        }
    }

    async fn get(&self, path: &str) -> anyhow::Result<Value> {
        let url = format!("{}{}", self.base_url, normalize_path(path));
        let response = self.http.get(&url).send().await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            anyhow::bail!("GET {url} failed with {status}: {text}");
        }
        serde_json::from_str(&text)
            .map_err(|err| anyhow::anyhow!("GET {url}: JSON parse error: {err}; body: {text}"))
    }
}

fn probe_host_for(host: &str) -> String {
    match host {
        "0.0.0.0" | "" => "127.0.0.1".into(),
        "::" => "::1".into(),
        other => other.to_string(),
    }
}

fn normalize_path(path: &str) -> String {
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    }
}

fn base_url_for(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("http://[{host}]:{port}")
    } else {
        format!("http://{host}:{port}")
    }
}

fn url_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

fn url_encode_path_segment(value: &str) -> String {
    url_encode(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_bind_all_to_loopback_for_client() {
        assert_eq!(probe_host_for("0.0.0.0"), "127.0.0.1");
        assert_eq!(probe_host_for("::"), "::1");
        assert_eq!(probe_host_for("192.168.1.2"), "192.168.1.2");
    }

    #[test]
    fn builds_ipv4_and_ipv6_base_urls() {
        assert_eq!(base_url_for("127.0.0.1", 11435), "http://127.0.0.1:11435");
        assert_eq!(base_url_for("::1", 11435), "http://[::1]:11435");
    }

    #[test]
    fn normalizes_operator_paths() {
        assert_eq!(normalize_path("health"), "/health");
        assert_eq!(normalize_path("/operator/logs"), "/operator/logs");
    }

    #[test]
    fn encodes_operator_query_values_and_path_segments() {
        assert_eq!(url_encode("qwen3.5:9b"), "qwen3.5%3A9b");
        assert_eq!(url_encode_path_segment("run/a b"), "run%2Fa%20b");
    }
}
