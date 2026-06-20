// SPDX-License-Identifier: Apache-2.0
// hipfire - see LICENSE and NOTICE in the project root.

//! `hipfire gen-config-schema` (hidden) - render the shared config schema as
//! machine-readable JSON/TOML or human-readable Markdown.

use std::path::Path;

use clap::{Args, ValueEnum};
use hipfire_config::{
    config_schema, ConfigField, ConfigMutability, ConfigScope, ConfigType, Requirement,
    RestartImpact,
};

#[derive(Debug, Args)]
pub struct GenConfigSchemaArgs {
    /// Output format.
    #[arg(long, value_enum, default_value_t = ConfigSchemaFormat::Markdown)]
    pub format: ConfigSchemaFormat,
    /// Write output to this path instead of stdout.
    #[arg(long)]
    pub output: Option<String>,
    /// Verify the output path already matches the rendered schema.
    #[arg(long)]
    pub check: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ConfigSchemaFormat {
    Json,
    Toml,
    Markdown,
}

pub fn run(args: GenConfigSchemaArgs) -> anyhow::Result<()> {
    let fields = sorted_fields();
    let rendered = match args.format {
        ConfigSchemaFormat::Json => serde_json::to_string_pretty(&fields)? + "\n",
        ConfigSchemaFormat::Toml => render_toml(&fields),
        ConfigSchemaFormat::Markdown => render_markdown(&fields),
    };

    if let Some(output) = args.output {
        let path = Path::new(&output);
        if args.check {
            let matches = std::fs::read_to_string(path)
                .map(|got| got == rendered)
                .unwrap_or(false);
            if !matches {
                anyhow::bail!(
                    "config schema is stale: {}\nregenerate with `cargo run -p hipfire-cli -- gen-config-schema --format {:?} --output {}`",
                    path.display(),
                    args.format,
                    path.display(),
                );
            }
            eprintln!("gen-config-schema: {} is up to date", path.display());
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, rendered)?;
        eprintln!("gen-config-schema: wrote {}", path.display());
        return Ok(());
    }

    if args.check {
        anyhow::bail!("--check requires --output");
    }
    print!("{rendered}");
    Ok(())
}

fn sorted_fields() -> Vec<&'static ConfigField> {
    let mut fields = config_schema().iter().collect::<Vec<_>>();
    fields.sort_by_key(|field| field.key);
    fields
}

fn render_markdown(fields: &[&ConfigField]) -> String {
    let mut out = String::from("# hipfire config schema\n\n");
    out.push_str(
        "| Key | Type | Required | Default | Scopes | Mutability | Impact | Description |\n",
    );
    out.push_str(
        "|-----|------|----------|---------|--------|------------|--------|-------------|\n",
    );
    for field in fields {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | {} | {} | `{}` | `{}` | {} |\n",
            field.key,
            type_label(field.ty),
            requirement_label(field.requirement),
            field
                .default
                .map(|v| format!("`{}`", escape_md(v)))
                .unwrap_or_else(|| "-".to_string()),
            field
                .scopes
                .iter()
                .map(|scope| format!("`{}`", scope_label(*scope)))
                .collect::<Vec<_>>()
                .join(", "),
            mutability_label(field.mutability),
            restart_impact_label(field.restart_impact),
            escape_md(field.description),
        ));
    }
    out
}

fn render_toml(fields: &[&ConfigField]) -> String {
    let mut out = String::new();
    for field in fields {
        out.push_str("[[field]]\n");
        out.push_str(&format!("key = \"{}\"\n", toml_escape(field.key)));
        out.push_str(&format!(
            "type = \"{}\"\n",
            toml_escape(&type_label(field.ty))
        ));
        out.push_str(&format!(
            "requirement = \"{}\"\n",
            toml_escape(&requirement_label(field.requirement))
        ));
        if let Some(default) = field.default {
            out.push_str(&format!("default = \"{}\"\n", toml_escape(default)));
        }
        out.push_str("scopes = [");
        out.push_str(
            &field
                .scopes
                .iter()
                .map(|scope| format!("\"{}\"", scope_label(*scope)))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str("]\n");
        out.push_str(&format!(
            "mutability = \"{}\"\n",
            mutability_label(field.mutability)
        ));
        out.push_str(&format!("owner = \"{}\"\n", toml_escape(field.owner)));
        out.push_str(&format!(
            "description = \"{}\"\n",
            toml_escape(field.description)
        ));
        if let Some(validation) = field.validation {
            out.push_str(&format!("validation = \"{}\"\n", toml_escape(validation)));
        }
        out.push_str(&format!("secret = {}\n", field.secret));
        out.push_str(&format!(
            "restart_impact = \"{}\"\n",
            restart_impact_label(field.restart_impact)
        ));
        if !field.env.is_empty() {
            out.push_str("env = [");
            out.push_str(
                &field
                    .env
                    .iter()
                    .map(|env| format!("\"{}\"", toml_escape(env)))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            out.push_str("]\n");
        }
        out.push('\n');
    }
    out
}

fn type_label(ty: ConfigType) -> String {
    match ty {
        ConfigType::Bool => "bool".to_string(),
        ConfigType::U8 => "u8".to_string(),
        ConfigType::U16 => "u16".to_string(),
        ConfigType::U32 => "u32".to_string(),
        ConfigType::I32 => "i32".to_string(),
        ConfigType::F64 => "f64".to_string(),
        ConfigType::String => "string".to_string(),
        ConfigType::Path => "path".to_string(),
        ConfigType::Enum { values } => format!("enum({})", values.join("|")),
        ConfigType::Json => "json".to_string(),
    }
}

fn requirement_label(requirement: Requirement) -> String {
    match requirement {
        Requirement::Optional => "optional".to_string(),
        Requirement::Required => "required".to_string(),
        Requirement::RequiredWhen(condition) => format!("required when `{condition}`"),
    }
}

fn scope_label(scope: ConfigScope) -> &'static str {
    match scope {
        ConfigScope::Global => "global",
        ConfigScope::Host => "host",
        ConfigScope::Node => "node",
        ConfigScope::Pool => "pool",
        ConfigScope::Model => "model",
        ConfigScope::Runtime => "runtime",
        ConfigScope::Eval => "eval",
        ConfigScope::Training => "training",
        ConfigScope::Request => "request",
    }
}

fn mutability_label(mutability: ConfigMutability) -> &'static str {
    match mutability {
        ConfigMutability::Static => "static",
        ConfigMutability::LoadTime => "load_time",
        ConfigMutability::RuntimeReloadable => "runtime_reloadable",
        ConfigMutability::RequestOnly => "request_only",
    }
}

fn restart_impact_label(impact: RestartImpact) -> &'static str {
    match impact {
        RestartImpact::None => "none",
        RestartImpact::ReloadModel => "reload_model",
        RestartImpact::RestartDaemon => "restart_daemon",
        RestartImpact::RestartService => "restart_service",
        RestartImpact::ReconnectClients => "reconnect_clients",
    }
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|")
}

fn toml_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
