use clap::{Args, Subcommand};
use hipfire_config::HipfireConfig;
use serde_json::{json, Value};

#[derive(Debug, Args)]
pub struct AdminArgs {
    /// Override admin API host. Defaults to config host, with 0.0.0.0 mapped to 127.0.0.1.
    #[arg(long, global = true)]
    pub host: Option<String>,
    /// Override admin API port. Defaults to config port.
    #[arg(long, global = true)]
    pub port: Option<u16>,
    #[command(subcommand)]
    pub command: AdminCommand,
}

#[derive(Debug, Subcommand)]
pub enum AdminCommand {
    /// Combined status snapshot for scripts and agents
    Status,
    /// Send one non-streaming chat request through /v1/chat/completions
    Chat {
        /// Model tag/path. Defaults to server config when omitted.
        #[arg(long)]
        model: Option<String>,
        /// Optional system message
        #[arg(long)]
        system: Option<String>,
        /// Max tokens to generate
        #[arg(long)]
        max_tokens: Option<u32>,
        /// Sampling temperature
        #[arg(long)]
        temperature: Option<f64>,
        /// Nucleus sampling top-p
        #[arg(long)]
        top_p: Option<f64>,
        /// Print only the assistant message text
        #[arg(long)]
        text: bool,
        /// User prompt text
        #[arg(required = true, trailing_var_arg = true)]
        prompt: Vec<String>,
    },
    /// Raw /health payload
    Health,
    /// Local model registry from the admin API
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
    /// GET an arbitrary admin/server path, e.g. /admin/training/runs
    Get {
        /// Absolute or relative server path
        path: String,
    },
}

pub async fn run(args: AdminArgs, config: HipfireConfig) -> anyhow::Result<()> {
    let client = AdminClient::new(args.host, args.port, &config);
    let value = match args.command {
        AdminCommand::Status => {
            let health = client.get("/health").await?;
            let diagnostics = client.get("/admin/diagnostics").await?;
            let models = client.get("/admin/models/registry").await?;
            let training = client.get("/admin/training/runs").await?;
            json!({
                "base_url": client.base_url,
                "health": health,
                "diagnostics": diagnostics,
                "models": models,
                "training": training,
            })
        }
        AdminCommand::Chat {
            model,
            system,
            max_tokens,
            temperature,
            top_p,
            text,
            prompt,
        } => {
            let mut messages = Vec::new();
            if let Some(system) = system {
                messages.push(json!({"role": "system", "content": system}));
            }
            messages.push(json!({"role": "user", "content": prompt.join(" ")}));

            let mut body = json!({
                "stream": false,
                "messages": messages,
            });
            if let Some(model) = model {
                body["model"] = json!(model);
            }
            if let Some(max_tokens) = max_tokens {
                body["max_tokens"] = json!(max_tokens);
            }
            if let Some(temperature) = temperature {
                body["temperature"] = json!(temperature);
            }
            if let Some(top_p) = top_p {
                body["top_p"] = json!(top_p);
            }

            let value = client.post_json("/v1/chat/completions", &body).await?;
            if text {
                println!("{}", assistant_text(&value).unwrap_or_default());
                return Ok(());
            }
            value
        }
        AdminCommand::Health => client.get("/health").await?,
        AdminCommand::Models => client.get("/admin/models/registry").await?,
        AdminCommand::Config { model } => {
            let path = match model {
                Some(model) => format!("/admin/config/resolved?model={}", url_encode(&model)),
                None => "/admin/config/resolved".to_string(),
            };
            client.get(&path).await?
        }
        AdminCommand::Training { id, events } => match (id, events) {
            (Some(id), true) => {
                client
                    .get(&format!(
                        "/admin/training/runs/{}/events",
                        url_encode_path_segment(&id)
                    ))
                    .await?
            }
            (Some(id), false) => {
                client
                    .get(&format!(
                        "/admin/training/runs/{}",
                        url_encode_path_segment(&id)
                    ))
                    .await?
            }
            (None, _) => client.get("/admin/training/runs").await?,
        },
        AdminCommand::Diagnostics => client.get("/admin/diagnostics").await?,
        AdminCommand::Logs { lines } => {
            client
                .get(&format!("/admin/logs?lines={}", lines.clamp(1, 1000)))
                .await?
        }
        AdminCommand::Get { path } => client.get(&normalize_path(&path)).await?,
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

struct AdminClient {
    base_url: String,
    http: reqwest::Client,
}

impl AdminClient {
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

    async fn post_json(&self, path: &str, body: &Value) -> anyhow::Result<Value> {
        let url = format!("{}{}", self.base_url, normalize_path(path));
        let response = self.http.post(&url).json(body).send().await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            anyhow::bail!("POST {url} failed with {status}: {text}");
        }
        serde_json::from_str(&text)
            .map_err(|err| anyhow::anyhow!("POST {url}: JSON parse error: {err}; body: {text}"))
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

fn assistant_text(value: &Value) -> Option<&str> {
    value
        .get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()
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
    fn normalizes_admin_paths() {
        assert_eq!(normalize_path("health"), "/health");
        assert_eq!(normalize_path("/admin/logs"), "/admin/logs");
    }

    #[test]
    fn encodes_admin_query_values_and_path_segments() {
        assert_eq!(url_encode("qwen3.5:9b"), "qwen3.5%3A9b");
        assert_eq!(url_encode_path_segment("run/a b"), "run%2Fa%20b");
    }

    #[test]
    fn extracts_assistant_text_from_chat_completion() {
        let payload = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "pong"
                }
            }]
        });

        assert_eq!(assistant_text(&payload), Some("pong"));
    }
}
