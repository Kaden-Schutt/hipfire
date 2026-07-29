// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native hipfire control plane.
//!
//! This binary owns hipfire's operator surface and never shells out to a
//! JavaScript or TypeScript runtime.

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use hipfire_client::{
    complete_openai_chat, probe_host, service_ready, service_url, stream_openai_chat, Engine,
};
use hipfire_config::{
    apply_config_profile, canonical_config_key, create_config_profile, developer_env_for_key,
    field, fields, is_developer_key, load_catalog, load_env_layer, load_global, resolve,
    write_catalog_toml, write_global_toml, CatalogFormat, ConfigFormat, ConfigLayer, ConfigPaths,
    ConfigSource, NamedLayer, ValueRule, CONFIG_SCHEMA_VERSION,
};
use hipfire_registry::{
    load as load_registry, LoadedRegistry, ModelEntry, RegistryPaths, RegistrySource, RegistryV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{mpsc, Arc, Condvar, Mutex},
    thread,
    time::{Duration, Instant},
};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

const MODEL_SUFFIXES: &[&str] = &[
    ".hf4",
    ".hf6",
    ".hfq",
    ".mq2",
    ".mq2lloyd",
    ".mq3",
    ".mq3p",
    ".mq4",
    ".mq4p",
    ".mq4r",
    ".mq5",
    ".mq6",
    ".mfp4",
    ".q8",
];
const BUILD_COMMIT: &str = env!("HIPFIRE_BUILD_COMMIT");
const BUILD_REF: &str = env!("HIPFIRE_BUILD_REF");
const BUILD_DIRTY: &str = env!("HIPFIRE_BUILD_DIRTY");
const BUILD_TARGET: &str = env!("HIPFIRE_BUILD_TARGET");

#[derive(Parser, Debug)]
#[command(
    name = "hipfire",
    version = env!("HIPFIRE_BUILD_VERSION"),
    about = "LLM inference for AMD GPUs",
    long_about = "Native Rust control plane for hipfire. Configuration, registry, model lifecycle, serving, chat, and diagnostics are implemented without a JavaScript runtime."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Read and edit typed TOML configuration.
    Config(ConfigArgs),
    /// Inspect or refresh the signed-model-registry migration surface.
    Registry(RegistryArgs),
    /// List local models and optionally the remote registry.
    List(ListArgs),
    /// Download and verify a registered model and its sidecars.
    Pull(PullArgs),
    /// Remove a local model and registered sidecars.
    #[command(alias = "remove")]
    Rm(RmArgs),
    /// Launch the Rust terminal UI.
    Tui(TuiArgs),
    /// Report local GPU/runtime/model/control-plane readiness.
    Diag(OutputArgs),
    /// Show the native service process and loaded model.
    Ps(OutputArgs),
    /// Benchmark a model through the native daemon protocol.
    Bench(BenchArgs),
    /// Report compiled kernel inventory for the detected architecture.
    Profile(ProfileArgs),
    /// Print build, source-checkout, and installed-daemon identity.
    Version(OutputArgs),
    /// Update to a branch, tag, or commit and rebuild the native control plane.
    Update(UpdateArgs),
    /// Quantize a Hugging Face or local model with the Rust quantizer.
    Quantize(QuantizeArgs),
    /// Generate a TriAttention calibration sidecar.
    SidecarGen(SidecarArgs),
    /// Generate text through a fresh native daemon process.
    Run(RunArgs),
    /// Start an interactive conversation through the native HTTP service.
    Chat(ChatArgs),
    /// Start the native OpenAI-compatible HTTP service.
    Serve(ServeArgs),
    /// Stop a detached native serve process.
    Stop(StopArgs),
    /// Stop and start the native HTTP service.
    Restart(ServeArgs),
}

#[derive(Args, Debug)]
#[command(subcommand_precedence_over_arg = true)]
struct ConfigArgs {
    /// Optional model tag, alias, filename, or local catalog identity.
    #[arg(value_name = "MODEL")]
    model: Option<String>,
    #[command(subcommand)]
    action: Option<ConfigAction>,
}

#[derive(Subcommand, Debug)]
enum ConfigAction {
    /// Print every effective key, its source, and override state.
    List(OutputArgs),
    /// Print one effective value.
    Get {
        key: String,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// Persist one global user override to config.toml.
    Set { key: String, value: String },
    /// Remove one override, or all overrides when no key is supplied.
    Reset { key: Option<String> },
    /// Explain a key's type, scope, default, effective value, and provenance.
    Explain {
        key: String,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// Print the authoritative typed configuration schema.
    Schema(OutputArgs),
    /// Convert legacy config.json to sparse config.toml without deleting JSON.
    Migrate,
    /// Select or create named configuration profiles.
    Profile {
        #[command(subcommand)]
        action: Option<ConfigProfileAction>,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigProfileAction {
    /// Replace the global sparse config with a built-in or custom profile.
    Set {
        /// Built-in (`default`, `dev`, `hip`, `redline`) or custom profile name.
        name: String,
    },
    /// Snapshot the current global sparse config as a new custom profile.
    Create {
        /// New custom profile name (not a built-in).
        name: String,
    },
}

#[derive(Args, Debug, Clone, Copy)]
struct OutputArgs {
    /// Emit machine-readable JSON.
    #[arg(short, long)]
    json: bool,
}

#[derive(Args, Debug, Default)]
struct UpdateArgs {
    /// Branch, tag, or commit to install. A leading '@' is optional.
    #[arg(
        value_name = "REF",
        conflicts_with_all = ["branch", "tag", "commit"]
    )]
    reference: Option<String>,
    /// Install the tip of a named remote branch.
    #[arg(long, value_name = "NAME", conflicts_with_all = ["tag", "commit"])]
    branch: Option<String>,
    /// Install a named git tag in detached/pinned mode.
    #[arg(long, value_name = "TAG", conflicts_with = "commit")]
    tag: Option<String>,
    /// Install an exact git commit in detached/pinned mode.
    #[arg(long, value_name = "SHA")]
    commit: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RevisionKind {
    Auto,
    Branch,
    Tag,
    Commit,
}

impl RevisionKind {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "ref",
            Self::Branch => "branch",
            Self::Tag => "tag",
            Self::Commit => "commit",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RevisionSelector {
    value: String,
    kind: RevisionKind,
}

#[derive(Debug)]
struct ResolvedRevision {
    selector: RevisionSelector,
    commit: String,
    tracking_ref: Option<String>,
}

#[derive(Args, Debug)]
struct RegistryArgs {
    #[command(subcommand)]
    action: RegistryAction,
}

#[derive(Subcommand, Debug)]
enum RegistryAction {
    /// Show registry source, revision, cache path, and warnings.
    Status(OutputArgs),
    /// List registered model tags.
    List(OutputArgs),
    /// Show one canonical registry entry.
    Show {
        tag: String,
        #[command(flatten)]
        output: OutputArgs,
    },
    /// Refresh the dynamic registry cache.
    Update(OutputArgs),
    /// Validate the bundled registry or an explicit v1 JSON file.
    Verify {
        path: Option<PathBuf>,
        #[command(flatten)]
        output: OutputArgs,
    },
}

#[derive(Args, Debug)]
struct ListArgs {
    /// Include registry models that are not downloaded.
    #[arg(short, long)]
    remote: bool,
    /// Emit machine-readable JSON.
    #[arg(short, long)]
    json: bool,
}

#[derive(Args, Debug)]
struct PullArgs {
    model: String,
    /// Replace an existing target after downloading and verifying a new copy.
    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug)]
struct RmArgs {
    model: String,
    /// Skip the interactive confirmation.
    #[arg(short, long)]
    yes: bool,
}

#[derive(Args, Debug, Default)]
struct TuiArgs {
    /// Arguments forwarded to hipfire-tui, such as --check.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    arguments: Vec<String>,
}

#[derive(Args, Debug)]
struct RunArgs {
    /// Registry tag, local alias, filename, or model path.
    model: String,
    /// Prompt words. Quote the prompt to preserve exact whitespace.
    #[arg(num_args = 0..)]
    prompt: Vec<String>,
    #[arg(short = 't', long)]
    /// Sampling temperature in 0..=2.
    temp: Option<f64>,
    #[arg(long)]
    /// Nucleus probability in (0, 1].
    top_p: Option<f64>,
    #[arg(long)]
    /// Multiplicative repetition penalty.
    repeat_penalty: Option<f64>,
    #[arg(short = 'n', long)]
    /// Maximum generated tokens.
    max_tokens: Option<u64>,
    #[arg(long)]
    /// One-shot KV format override for this model load.
    kv_mode: Option<String>,
    #[arg(long, value_parser = ["contiguous", "vmm"])]
    /// One-shot KV storage backend override for this model load.
    kv_backend: Option<String>,
    /// Select one speculative mechanism: off, auto, ngram, dflash, mtp, or dspark.
    #[arg(long = "spec", alias = "speculation")]
    speculation: Option<String>,
    /// Explicit DFlash draft model.
    #[arg(long, alias = "md")]
    model_draft: Option<PathBuf>,
    /// Override the active MTP/n-gram draft window.
    #[arg(long, alias = "draft")]
    draft_max: Option<u64>,
    /// DSpark confidence cutoff.
    #[arg(long)]
    dspark_conf_threshold: Option<f64>,
    #[arg(long)]
    /// Override the resolved system prompt.
    system: Option<String>,
    #[arg(long)]
    /// Local image path for a vision-language model.
    image: Option<PathBuf>,
    #[arg(short = 'j', long)]
    /// Emit one JSON result object.
    json: bool,
    #[arg(long)]
    /// Buffer visible output instead of streaming it.
    no_stream: bool,
}

#[derive(Args, Debug)]
struct ChatArgs {
    /// Model tag, alias, filename, or local catalog identity.
    model: Option<String>,
    #[arg(short = 't', long)]
    temp: Option<f64>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(short = 'n', long)]
    max_tokens: Option<u64>,
    #[arg(long)]
    system: Option<String>,
    /// Accepted for compatibility; native chat does not emit ANSI colors.
    #[arg(long)]
    no_color: bool,
}

#[derive(Args, Debug)]
struct BenchArgs {
    model: String,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(short = 'j', long)]
    json: bool,
    /// Compare the five RDNA2 kernel variants in isolated daemon processes.
    #[arg(long)]
    exp: bool,
    /// Run deterministic synthetic prefill/decode rows.
    #[arg(long)]
    matrix: bool,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "128,512,2048,4096,8192,20000"
    )]
    pp: Vec<usize>,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "128,512,2048,4096,8192,20000"
    )]
    ctx: Vec<usize>,
    #[arg(long, default_value_t = 128)]
    tg: usize,
    #[arg(long)]
    sustained_tg: Option<usize>,
    #[arg(long, value_delimiter = ',', default_value = "128,8192")]
    sustained_ctx: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    warmups: usize,
    #[arg(long)]
    kv_mode: Option<String>,
    #[arg(long, value_parser = ["contiguous", "vmm"])]
    kv_backend: Option<String>,
    #[arg(long)]
    redline: bool,
    /// Prompt words for the standard benchmark.
    #[arg(num_args = 0..)]
    prompt: Vec<String>,
}

#[derive(Args, Debug)]
struct ProfileArgs {
    model: Option<String>,
    #[arg(long)]
    kernel: Option<String>,
    #[arg(short = 'j', long)]
    json: bool,
}

#[derive(Args, Debug)]
struct QuantizeArgs {
    /// Hugging Face model ID, local safetensors directory, or GGUF file.
    input: String,
    #[arg(long = "format")]
    /// Repeatable output format: mq4, mq6, q8, q8f16, hf4, or hf6.
    formats: Vec<String>,
    #[arg(long)]
    /// Produce both MQ4 and MQ6.
    both: bool,
    #[arg(short = 'o', long)]
    /// Exact output path; valid with one format only.
    output: Option<PathBuf>,
    #[arg(long)]
    /// Output directory for one or more formats.
    output_dir: Option<PathBuf>,
    #[arg(long)]
    /// Override the output filename stem.
    stem: Option<String>,
    #[arg(long)]
    /// Upload completed artifacts to owner/repo on Hugging Face.
    upload: Option<String>,
    #[arg(long)]
    /// Create the Hugging Face model repository if needed.
    create_repo: bool,
    #[arg(long)]
    /// Copy completed artifacts into ~/.hipfire/models.
    install: bool,
    #[arg(long)]
    /// Register a local model alias in models.toml.
    register: Option<String>,
}

#[derive(Args, Debug)]
struct SidecarArgs {
    model: String,
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long, default_value_t = 4000)]
    max_tokens: usize,
    #[arg(long, default_value_t = 256)]
    chunk_len: usize,
    #[arg(long, conflicts_with = "cpu_calib")]
    gpu_calib: bool,
    #[arg(long)]
    cpu_calib: bool,
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
    #[arg(long)]
    skip_validation: bool,
}

#[derive(Args, Debug, Clone)]
struct ServeArgs {
    /// Optional model, host, host:port, and/or port in legacy-compatible order.
    #[arg(value_name = "MODEL_HOST_OR_PORT", num_args = 0..=3)]
    positionals: Vec<String>,
    /// Model tag/path to pre-warm for this process.
    #[arg(long)]
    model: Option<String>,
    /// Run in the background and log to ~/.hipfire/serve.log.
    #[arg(short = 'd', long, alias = "background")]
    detach: bool,
    /// Do not load the configured default model before accepting requests.
    #[arg(long)]
    no_prewarm: bool,
    /// KV cache mode for models loaded by this service.
    #[arg(long)]
    kv_mode: Option<String>,
    /// KV storage backend for models loaded by this service.
    #[arg(long, value_parser = ["contiguous", "vmm"])]
    kv_backend: Option<String>,
    /// Idle model-unload timeout in seconds; zero disables eviction.
    #[arg(long, value_parser = clap::value_parser!(u64).range(0..=86400))]
    idle_timeout: Option<u64>,
    /// Expert-parallel degree.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=64))]
    tp: Option<u64>,
    /// Internal marker used by the detached child.
    #[arg(long, hide = true)]
    foreground_child: bool,
}

#[derive(Args, Debug, Clone, Copy)]
struct StopArgs {
    /// Port to free when --force or --all is used.
    port: Option<u16>,
    /// Reap orphan daemon processes and free the configured port.
    #[arg(long)]
    force: bool,
    /// Also reap native quantizer processes.
    #[arg(long)]
    all: bool,
}

#[derive(Clone, Debug)]
struct Paths {
    root: PathBuf,
    models: PathBuf,
    config: ConfigPaths,
    registry: RegistryPaths,
}

impl Paths {
    fn discover() -> Self {
        let config = ConfigPaths::discover();
        let root = config.root.clone();
        Self {
            models: config.models.clone(),
            registry: RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("hipfire: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse_from(env::args_os().map(|argument| {
        if argument == "-md" {
            OsString::from("--model-draft")
        } else {
            argument
        }
    }));
    let paths = Paths::discover();
    match cli.command {
        None => launch_tui(&paths, &[]),
        Some(Commands::Tui(args)) => launch_tui(&paths, &args.arguments),
        Some(Commands::Config(args)) => config_command(&paths, args),
        Some(Commands::Registry(args)) => registry_command(&paths, args),
        Some(Commands::List(args)) => list_command(&paths, args),
        Some(Commands::Pull(args)) => pull_command(&paths, args),
        Some(Commands::Rm(args)) => rm_command(&paths, args),
        Some(Commands::Diag(output)) => diag_command(&paths, output),
        Some(Commands::Ps(output)) => ps_command(&paths, output),
        Some(Commands::Bench(args)) => bench_command(&paths, args),
        Some(Commands::Profile(args)) => profile_command(&paths, args),
        Some(Commands::Version(output)) => version_command(&paths, output),
        Some(Commands::Update(args)) => update_command(&paths, args),
        Some(Commands::Quantize(args)) => quantize_command(&paths, args),
        Some(Commands::SidecarGen(args)) => sidecar_command(&paths, args),
        Some(Commands::Run(args)) => run_command(&paths, args),
        Some(Commands::Chat(args)) => chat_command(&paths, args),
        Some(Commands::Serve(args)) => serve_command(&paths, args),
        Some(Commands::Stop(args)) => stop_command(&paths, args),
        Some(Commands::Restart(args)) => {
            let port = args.positionals.iter().find_map(|value| {
                value
                    .parse::<u16>()
                    .ok()
                    .or_else(|| parse_host_port(value).ok().flatten().map(|(_, port)| port))
            });
            let _ = stop_command(
                &paths,
                StopArgs {
                    port,
                    force: true,
                    all: false,
                },
            );
            serve_command(&paths, args)
        }
    }
}

fn config_command(paths: &Paths, args: ConfigArgs) -> Result<()> {
    if let Some(model) = args.model {
        return model_config_command(paths, &model, args.action);
    }
    let Some(action) = args.action else {
        return launch_tui(paths, &[]);
    };
    match action {
        ConfigAction::List(output) => {
            let (loaded, resolved) = resolved_global(paths, true)?;
            if output.json {
                let mut values = fields()
                    .iter()
                    .map(|field| {
                        let resolved = resolved.get(field.key).expect("schema key resolved");
                        (
                            field.key.to_owned(),
                            serde_json::json!({
                                "legacy_key": field.legacy_key,
                                "value": resolved.value,
                                "default": format_default(field),
                                "source": resolved.source,
                                "overridden": loaded.layer.get(field.key).is_some(),
                            }),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>();
                for (key, item) in resolved
                    .values
                    .iter()
                    .filter(|(key, _)| is_developer_key(key))
                {
                    values.insert(
                        key.clone(),
                        serde_json::json!({
                            "legacy_key": null,
                            "value": item.value,
                            "default": null,
                            "source": item.source,
                            "overridden": loaded.layer.get(key).is_some(),
                        }),
                    );
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "scope": "global",
                        "path": loaded.path,
                        "format": format!("{:?}", loaded.format).to_lowercase(),
                        "values": values,
                        "warnings": loaded.warnings,
                    }))?
                );
            } else {
                println!("Global configuration: {}", loaded.path.display());
                if loaded.format == ConfigFormat::LegacyJson {
                    println!("  legacy JSON is active; the next write will create config.toml");
                }
                println!();
                for schema in fields() {
                    let item = resolved.get(schema.key).expect("schema key resolved");
                    let marker = if loaded.layer.get(schema.key).is_some() {
                        "override"
                    } else {
                        "inherited"
                    };
                    println!(
                        "  {:<43} {:<16} {:<9} {}",
                        schema.key,
                        item.value,
                        marker,
                        source_label(&item.source)
                    );
                }
                for (key, item) in resolved
                    .values
                    .iter()
                    .filter(|(key, _)| is_developer_key(key))
                {
                    let marker = if loaded.layer.get(key).is_some() {
                        "override"
                    } else {
                        "inherited"
                    };
                    println!(
                        "  {:<43} {:<16} {:<9} {}",
                        key,
                        item.value,
                        marker,
                        source_label(&item.source)
                    );
                }
                for warning in loaded.warnings {
                    eprintln!("warning: {warning}");
                }
            }
            Ok(())
        }
        ConfigAction::Get { key, output } => {
            let (_, resolved) = resolved_global(paths, true)?;
            let canonical = canonical_config_key(&key)
                .ok_or_else(|| anyhow!("unknown configuration key '{key}'"))?;
            let schema = field(&canonical);
            let value = resolved
                .get(&canonical)
                .ok_or_else(|| anyhow!("configuration key '{canonical}' is not set"))?;
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "key": canonical,
                        "legacy_key": schema.map(|schema| schema.legacy_key),
                        "value": value.value,
                        "source": value.source,
                    }))?
                );
            } else {
                println!("{}", value.value);
            }
            Ok(())
        }
        ConfigAction::Set { key, value } => {
            let mut loaded = load_global(&paths.config)?;
            loaded.layer.set_cli(&key, &value)?;
            write_global_toml(&paths.config, &loaded.layer)?;
            let canonical = canonical_config_key(&key).expect("set_cli accepted key");
            let value = loaded.layer.get(&canonical).expect("set value");
            println!("{canonical} = {value}");
            if loaded.format == ConfigFormat::LegacyJson {
                println!(
                    "migrated active configuration to {}; preserved {} as a rollback copy",
                    paths.config.config_toml.display(),
                    paths.config.config_json.display()
                );
            }
            Ok(())
        }
        ConfigAction::Reset { key } => {
            let mut loaded = load_global(&paths.config)?;
            if let Some(key) = key {
                let canonical = canonical_config_key(&key)
                    .ok_or_else(|| anyhow!("unknown configuration key '{key}'"))?;
                let existed = loaded.layer.remove(&canonical)?.is_some();
                write_global_toml(&paths.config, &loaded.layer)?;
                if existed {
                    println!("{canonical} override removed");
                } else {
                    println!("{canonical} was already inherited");
                }
            } else {
                write_global_toml(&paths.config, &ConfigLayer::default())?;
                println!("all global overrides removed");
            }
            Ok(())
        }
        ConfigAction::Explain { key, output } => {
            let (loaded, resolved) = resolved_global(paths, true)?;
            let canonical = canonical_config_key(&key)
                .ok_or_else(|| anyhow!("unknown configuration key '{key}'"))?;
            let value = resolved
                .get(&canonical)
                .ok_or_else(|| anyhow!("configuration key '{canonical}' is not set"))?;
            if is_developer_key(&canonical) {
                let env_compat =
                    developer_env_for_key(&canonical).expect("validated developer key");
                if output.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "key": canonical,
                            "legacy_key": null,
                            "value": value.value,
                            "source": value.source,
                            "shadowed": value.shadowed,
                            "default": null,
                            "category": "diagnostic",
                            "scope": "process",
                            "registry_allowed": false,
                            "experimental": true,
                            "env_compat": env_compat,
                            "help": "Experimental process-scoped override. Prefer a typed field when one exists.",
                            "config_path": loaded.path,
                        }))?
                    );
                } else {
                    println!("{canonical}");
                    println!("  value:       {}", value.value);
                    println!("  source:      {}", source_label(&value.source));
                    println!("  default:     unset");
                    println!("  category:    Diagnostic");
                    println!("  scope:       Process");
                    println!("  registry:    false");
                    println!("  legacy env:  {env_compat}");
                    println!(
                        "  about:       Experimental process-scoped override. Prefer a typed field when one exists."
                    );
                    if !value.shadowed.is_empty() {
                        println!("  shadowed:");
                        for candidate in value.shadowed.iter().rev() {
                            println!(
                                "    {:<16} {}",
                                candidate.value,
                                source_label(&candidate.source)
                            );
                        }
                    }
                }
                return Ok(());
            }
            let schema = field(&canonical).expect("stable configuration key");
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "key": schema.key,
                        "legacy_key": schema.legacy_key,
                        "value": value.value,
                        "source": value.source,
                        "shadowed": value.shadowed,
                        "default": format_default(schema),
                        "category": schema.category,
                        "scope": schema.scope,
                        "registry_allowed": schema.registry_allowed,
                        "experimental": schema.experimental,
                        "env_compat": schema.env_compat,
                        "help": schema.help,
                        "config_path": loaded.path,
                    }))?
                );
            } else {
                println!("{}", schema.key);
                println!("  value:       {}", value.value);
                println!("  source:      {}", source_label(&value.source));
                println!("  default:     {}", format_default(schema));
                println!("  category:    {:?}", schema.category);
                println!("  scope:       {:?}", schema.scope);
                println!("  registry:    {}", schema.registry_allowed);
                if let Some(env) = schema.env_compat {
                    println!("  legacy env:  {env}");
                }
                println!("  about:       {}", schema.help);
                if !value.shadowed.is_empty() {
                    println!("  shadowed:");
                    for candidate in value.shadowed.iter().rev() {
                        println!(
                            "    {:<16} {}",
                            candidate.value,
                            source_label(&candidate.source)
                        );
                    }
                }
            }
            Ok(())
        }
        ConfigAction::Schema(output) => {
            let schema = fields()
                .iter()
                .map(|field| {
                    serde_json::json!({
                        "key": field.key,
                        "legacy_key": field.legacy_key,
                        "category": field.category,
                        "scope": field.scope,
                        "default": config_default_value(field),
                        "rule": config_rule_json(field.rule),
                        "registry_allowed": field.registry_allowed,
                        "experimental": field.experimental,
                        "env_compat": field.env_compat,
                        "help": field.help,
                    })
                })
                .collect::<Vec<_>>();
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "schema_version": CONFIG_SCHEMA_VERSION,
                        "fields": schema,
                        "developer_namespace": {
                            "prefix": "developer.",
                            "scope": "process",
                            "registry_allowed": false,
                            "experimental": true,
                            "value_types": ["boolean", "integer", "number", "string"],
                            "legacy_mapping": "HIPFIRE_FOO -> developer.foo"
                        },
                    }))?
                );
            } else {
                println!("Configuration schema v{CONFIG_SCHEMA_VERSION}");
                for field in fields() {
                    println!(
                        "  {:<48} {:<18} {:<12} {}",
                        field.key,
                        format_default(field),
                        config_rule_label(field.rule),
                        field.help
                    );
                }
                println!(
                    "  {:<48} {:<18} {:<12} Experimental process-scoped overrides (HIPFIRE_FOO -> developer.foo).",
                    "developer.<name>", "unset", "scalar"
                );
            }
            Ok(())
        }
        ConfigAction::Migrate => {
            let loaded = load_global(&paths.config)?;
            match loaded.format {
                ConfigFormat::Toml => {
                    println!("already using {}", paths.config.config_toml.display());
                }
                ConfigFormat::LegacyJson => {
                    write_global_toml(&paths.config, &loaded.layer)?;
                    println!(
                        "wrote {}; preserved {} unchanged",
                        paths.config.config_toml.display(),
                        paths.config.config_json.display()
                    );
                }
                ConfigFormat::Empty => {
                    write_global_toml(&paths.config, &ConfigLayer::default())?;
                    println!("wrote {}", paths.config.config_toml.display());
                }
            }
            let catalog = load_catalog(&paths.config)?;
            match catalog.format {
                CatalogFormat::Toml => {
                    println!("already using {}", paths.config.models_toml.display());
                }
                CatalogFormat::LegacyJson => {
                    write_catalog_toml(&paths.config, &catalog.catalog)?;
                    println!(
                        "wrote {}; preserved {} and {} unchanged",
                        paths.config.models_toml.display(),
                        paths.config.models_json.display(),
                        paths.config.legacy_per_model_json.display()
                    );
                    for warning in catalog.warnings {
                        eprintln!("warning: {warning}");
                    }
                }
                CatalogFormat::Empty => {
                    write_catalog_toml(&paths.config, &catalog.catalog)?;
                    println!("wrote {}", paths.config.models_toml.display());
                }
            }
            Ok(())
        }
        ConfigAction::Profile { action } => config_profile_command(paths, action),
    }
}

fn config_profile_command(paths: &Paths, action: Option<ConfigProfileAction>) -> Result<()> {
    let Some(action) = action else {
        return launch_tui(paths, &["--config-profile-wizard".to_owned()]);
    };
    match action {
        ConfigProfileAction::Set { name } => {
            let mut loaded = load_global(&paths.config)?;
            apply_config_profile(&mut loaded.layer, &paths.config, &name)?;
            write_global_toml(&paths.config, &loaded.layer)?;
            println!("applied configuration profile '{name}'");
            if loaded.format == ConfigFormat::LegacyJson {
                println!(
                    "migrated active configuration to {}; preserved {} as a rollback copy",
                    paths.config.config_toml.display(),
                    paths.config.config_json.display()
                );
            }
            Ok(())
        }
        ConfigProfileAction::Create { name } => {
            let loaded = load_global(&paths.config)?;
            let path = create_config_profile(&paths.config, &name, &loaded.layer)?;
            println!(
                "created configuration profile '{name}' at {}",
                path.display()
            );
            Ok(())
        }
    }
}

fn model_config_command(
    paths: &Paths,
    model_name: &str,
    action: Option<ConfigAction>,
) -> Result<()> {
    let registry = load_registry(&paths.registry).registry;
    let (tag, entry) = registry
        .model(model_name)
        .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry)))
        .unwrap_or((None, None));
    let action = action.unwrap_or(ConfigAction::List(OutputArgs { json: false }));
    if matches!(
        action,
        ConfigAction::Migrate | ConfigAction::Schema(_) | ConfigAction::Profile { .. }
    ) {
        bail!("config migrate/schema/profile are global; omit the model argument");
    }

    match action {
        ConfigAction::List(output) => {
            let catalog = load_catalog(&paths.config)?;
            let record = catalog
                .catalog
                .model(model_name)
                .or_else(|| tag.as_deref().and_then(|tag| catalog.catalog.model(tag)));
            let overrides = record
                .map(|(_, model)| &model.overrides)
                .cloned()
                .unwrap_or_default();
            let resolved = resolved_for_model(paths, model_name, tag.as_deref(), entry)?;
            if output.json {
                let values = fields()
                    .iter()
                    .map(|schema| {
                        let item = resolved.get(schema.key).expect("schema key resolved");
                        (
                            schema.key.to_owned(),
                            serde_json::json!({
                                "legacy_key": schema.legacy_key,
                                "value": item.value,
                                "source": item.source,
                                "overridden": overrides.get(schema.key).is_some(),
                            }),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "scope": "model",
                        "model": model_name,
                        "canonical_tag": tag,
                        "path": catalog.path,
                        "format": format!("{:?}", catalog.format).to_lowercase(),
                        "values": values,
                        "warnings": catalog.warnings,
                    }))?
                );
            } else {
                println!("Model configuration: {model_name}");
                println!(
                    "Catalog: {} ({:?})\n",
                    catalog.path.display(),
                    catalog.format
                );
                for schema in fields() {
                    let item = resolved.get(schema.key).expect("schema key resolved");
                    let marker = if overrides.get(schema.key).is_some() {
                        "override"
                    } else {
                        "inherited"
                    };
                    println!(
                        "  {:<43} {:<16} {:<9} {}",
                        schema.key,
                        item.value,
                        marker,
                        source_label(&item.source)
                    );
                }
                for warning in catalog.warnings {
                    eprintln!("warning: {warning}");
                }
            }
            Ok(())
        }
        ConfigAction::Get { key, output } => {
            if is_developer_key(&key) {
                bail!("developer configuration is global process policy; omit the model argument");
            }
            let resolved = resolved_for_model(paths, model_name, tag.as_deref(), entry)?;
            let schema = field(&key).ok_or_else(|| anyhow!("unknown configuration key '{key}'"))?;
            let value = resolved.get(schema.key).expect("schema key resolved");
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "model": model_name,
                        "key": schema.key,
                        "legacy_key": schema.legacy_key,
                        "value": value.value,
                        "source": value.source,
                    }))?
                );
            } else {
                println!("{}", value.value);
            }
            Ok(())
        }
        ConfigAction::Set { key, value } => {
            if is_developer_key(&key) {
                bail!("developer configuration is global process policy; omit the model argument");
            }
            let mut loaded = load_catalog(&paths.config)?;
            let id = loaded
                .catalog
                .model_id(model_name)
                .map(str::to_owned)
                .unwrap_or_else(|| tag.clone().unwrap_or_else(|| model_name.to_owned()));
            let local_path = find_model_path(paths, &registry, model_name);
            let saved = {
                let record = loaded.catalog.models.entry(id.clone()).or_default();
                if record.path.is_none() {
                    record.path = local_path;
                }
                if record.registry_tag.is_none() {
                    record.registry_tag = tag.clone();
                }
                record.overrides.set_cli(&key, &value)?;
                let schema = field(&key).expect("set_cli accepted key");
                record.overrides.get(schema.key).unwrap().clone()
            };
            write_catalog_toml(&paths.config, &loaded.catalog)?;
            let schema = field(&key).expect("set_cli accepted key");
            println!("{id} {} = {saved}", schema.key);
            if loaded.format == CatalogFormat::LegacyJson {
                println!(
                    "migrated model catalog to {}; preserved legacy JSON as rollback copies",
                    paths.config.models_toml.display()
                );
            }
            Ok(())
        }
        ConfigAction::Reset { key } => {
            if key.as_deref().is_some_and(is_developer_key) {
                bail!("developer configuration is global process policy; omit the model argument");
            }
            let mut loaded = load_catalog(&paths.config)?;
            let Some(id) = loaded.catalog.model_id(model_name).map(str::to_owned) else {
                println!("{model_name} has no per-model overrides");
                return Ok(());
            };
            let record = loaded
                .catalog
                .models
                .get_mut(&id)
                .expect("resolved model id");
            if let Some(key) = key {
                let schema =
                    field(&key).ok_or_else(|| anyhow!("unknown configuration key '{key}'"))?;
                let existed = record.overrides.remove(schema.key)?.is_some();
                if existed {
                    println!("{id} {} override removed", schema.key);
                } else {
                    println!("{id} {} was already inherited", schema.key);
                }
            } else {
                record.overrides = ConfigLayer::default();
                println!("all {id} overrides removed");
            }
            write_catalog_toml(&paths.config, &loaded.catalog)?;
            Ok(())
        }
        ConfigAction::Explain { key, output } => {
            if is_developer_key(&key) {
                bail!("developer configuration is global process policy; omit the model argument");
            }
            let resolved = resolved_for_model(paths, model_name, tag.as_deref(), entry)?;
            let schema = field(&key).ok_or_else(|| anyhow!("unknown configuration key '{key}'"))?;
            let value = resolved.get(schema.key).expect("schema key resolved");
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "model": model_name,
                        "key": schema.key,
                        "value": value.value,
                        "source": value.source,
                        "shadowed": value.shadowed,
                        "scope": schema.scope,
                        "help": schema.help,
                    }))?
                );
            } else {
                println!("{}", schema.key);
                println!("  model:       {model_name}");
                println!("  value:       {}", value.value);
                println!("  source:      {}", source_label(&value.source));
                println!("  scope:       {:?}", schema.scope);
                println!("  about:       {}", schema.help);
                if !value.shadowed.is_empty() {
                    println!("  shadowed:");
                    for candidate in value.shadowed.iter().rev() {
                        println!(
                            "    {:<16} {}",
                            candidate.value,
                            source_label(&candidate.source)
                        );
                    }
                }
            }
            Ok(())
        }
        ConfigAction::Migrate | ConfigAction::Schema(_) | ConfigAction::Profile { .. } => {
            unreachable!()
        }
    }
}

fn resolved_global(
    paths: &Paths,
    include_env: bool,
) -> Result<(hipfire_config::LoadedConfig, hipfire_config::ResolvedConfig)> {
    let loaded = load_global(&paths.config)?;
    let mut layers = vec![NamedLayer {
        source: ConfigSource::GlobalUser {
            path: loaded.path.clone(),
        },
        layer: loaded.layer.clone(),
    }];
    if include_env {
        let env_layer = load_env_layer()?;
        if !env_layer.values.is_empty() {
            // Field-level env names remain available from schema metadata. A
            // single layer source makes effective output concise while explain
            // still names the compatibility surface.
            layers.push(NamedLayer {
                source: ConfigSource::LegacyEnv {
                    name: "HIPFIRE_*".into(),
                },
                layer: env_layer,
            });
        }
    }
    Ok((loaded, resolve(layers)?))
}

fn registry_command(paths: &Paths, args: RegistryArgs) -> Result<()> {
    match args.action {
        RegistryAction::Verify { path, output } => {
            let registry = if let Some(path) = path.as_deref() {
                let raw = fs::read_to_string(path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                RegistryV1::parse(&raw, path.display().to_string())?
            } else {
                hipfire_registry::bundled()?
            };
            if output.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "valid": true,
                        "schema_version": registry.schema_version,
                        "generated_at": registry.generated_at,
                        "models": registry.models.len(),
                        "aliases": registry.aliases.len(),
                    }))?
                );
            } else {
                println!(
                    "registry valid: schema v{}, {} models, {} aliases, generated {}",
                    registry.schema_version,
                    registry.models.len(),
                    registry.aliases.len(),
                    registry.generated_at
                );
            }
            Ok(())
        }
        action => {
            let loaded = load_registry(&paths.registry);
            match action {
                RegistryAction::Status(output) | RegistryAction::Update(output) => {
                    if output.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&registry_status_json(paths, &loaded))?
                        );
                    } else {
                        println!("source:       {}", registry_source(loaded.source));
                        println!("schema:       v{}", loaded.registry.schema_version);
                        println!("generated:    {}", loaded.registry.generated_at);
                        println!("models:       {}", loaded.registry.models.len());
                        println!("aliases:      {}", loaded.registry.aliases.len());
                        println!("cache:        {}", paths.registry.cache.display());
                        for warning in &loaded.warnings {
                            eprintln!("warning: {warning}");
                        }
                    }
                    Ok(())
                }
                RegistryAction::List(output) => print_registry_list(&loaded, output.json),
                RegistryAction::Show { tag, output } => {
                    let (canonical, entry) = loaded
                        .registry
                        .model(&tag)
                        .ok_or_else(|| anyhow!("unknown model '{tag}'"))?;
                    if output.json {
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&serde_json::json!({
                                "tag": canonical,
                                "entry": entry,
                                "registry_source": registry_source(loaded.source),
                            }))?
                        );
                    } else {
                        println!("tag:          {canonical}");
                        println!("repo:         {}", entry.repo);
                        println!("file:         {}", entry.file);
                        println!("size:         {:.3} GB", entry.size_gb);
                        println!("minimum VRAM: {:.3} GB", entry.min_vram_gb);
                        if let Some(hash) = &entry.sha256 {
                            println!("sha256:       {hash}");
                        }
                        if let Some(quant) = &entry.quant {
                            println!("quant:        {quant}");
                        }
                        println!("about:        {}", entry.desc);
                    }
                    Ok(())
                }
                RegistryAction::Verify { .. } => unreachable!(),
            }
        }
    }
}

fn registry_status_json(paths: &Paths, loaded: &LoadedRegistry) -> serde_json::Value {
    serde_json::json!({
        "source": registry_source(loaded.source),
        "schema_version": loaded.registry.schema_version,
        "generated_at": loaded.registry.generated_at,
        "models": loaded.registry.models.len(),
        "aliases": loaded.registry.aliases.len(),
        "cache_path": paths.registry.cache,
        "warnings": loaded.warnings,
    })
}

fn print_registry_list(loaded: &LoadedRegistry, json: bool) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "source": registry_source(loaded.source),
                "models": loaded.registry.models,
                "aliases": loaded.registry.aliases,
            }))?
        );
    } else {
        for (tag, model) in &loaded.registry.models {
            println!("  {:<32} {:>7.2} GB  {}", tag, model.size_gb, model.desc);
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct LocalModel {
    name: String,
    path: PathBuf,
    size_bytes: u64,
    registry_tag: Option<String>,
}

fn list_command(paths: &Paths, args: ListArgs) -> Result<()> {
    let loaded = load_registry(&paths.registry);
    let local = list_local_models(paths, &loaded.registry)?;
    let local_files = local
        .iter()
        .map(|model| model.name.as_str())
        .collect::<BTreeSet<_>>();
    if args.json {
        let registry = loaded
            .registry
            .models
            .iter()
            .map(|(tag, entry)| {
                serde_json::json!({
                    "tag": tag,
                    "name": entry.file,
                    "size_bytes": entry.size_bytes.unwrap_or_else(|| (entry.size_gb * 1e9).round() as u64),
                    "quant": entry.quant,
                    "downloaded": local_files.contains(entry.file.as_str()),
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "models": local,
                "registry": registry,
                "registry_source": registry_source(loaded.source),
            }))?
        );
        return Ok(());
    }
    if local.is_empty() {
        println!("No local models. Pull one:\n  hipfire pull qwen3.6:35b-a3b-mq4r");
    } else {
        println!("Local models:\n");
        for model in &local {
            let tag = model
                .registry_tag
                .as_deref()
                .map(|tag| format!(" ({tag})"))
                .unwrap_or_default();
            println!(
                "  {:<42} {:>7.2} GB{}",
                model.name,
                model.size_bytes as f64 / 1e9,
                tag
            );
        }
    }
    if args.remote || local.is_empty() {
        println!("\nAvailable models:\n");
        for (tag, entry) in &loaded.registry.models {
            let status = if local_files.contains(entry.file.as_str()) {
                " [downloaded]"
            } else {
                ""
            };
            println!(
                "  {:<32} {:>7.2} GB  {}{}",
                tag, entry.size_gb, entry.desc, status
            );
        }
    }
    Ok(())
}

fn list_local_models(paths: &Paths, registry: &RegistryV1) -> Result<Vec<LocalModel>> {
    let mut candidates = local_model_paths(paths)?;
    if let Ok(catalog) = load_catalog(&paths.config) {
        candidates.extend(
            catalog
                .catalog
                .models
                .values()
                .filter_map(|model| model.path.clone())
                .filter(|path| path.is_file()),
        );
    }
    let mut seen = BTreeSet::new();
    let mut models = Vec::new();
    for path in candidates {
        let canonical = fs::canonicalize(&path).unwrap_or(path);
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let metadata = fs::metadata(&canonical)?;
        let name = canonical
            .file_name()
            .and_then(|file| file.to_str())
            .unwrap_or_default()
            .to_owned();
        if !is_model_file(&name) {
            continue;
        }
        let registry_tag = registry
            .models
            .iter()
            .find_map(|(tag, model)| (model.file == name).then(|| tag.clone()));
        models.push(LocalModel {
            name,
            path: canonical,
            size_bytes: metadata.len(),
            registry_tag,
        });
    }
    models.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(models)
}

fn local_model_paths(paths: &Paths) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(&paths.models) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).context("failed to list model directory"),
    };
    let mut models = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path
                .file_name()
                .and_then(|file| file.to_str())
                .is_some_and(is_model_file)
            {
                models.push(path);
            }
            continue;
        }
        if !path.is_dir() {
            continue;
        }
        let Ok(children) = fs::read_dir(path) else {
            continue;
        };
        models.extend(children.flatten().map(|entry| entry.path()).filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|file| file.to_str())
                    .is_some_and(is_model_file)
        }));
    }
    Ok(models)
}

fn pull_command(paths: &Paths, args: PullArgs) -> Result<()> {
    let loaded = load_registry(&paths.registry);
    let (tag, entry) = loaded
        .registry
        .model(&args.model)
        .ok_or_else(|| anyhow!("unknown model '{}'", args.model))?;
    if entry.repo.is_empty() {
        bail!(
            "cannot pull {tag}: registry entry is local-only; place {} in {}",
            entry.file,
            paths.models.display()
        );
    }
    fs::create_dir_all(&paths.models)
        .with_context(|| format!("failed to create {}", paths.models.display()))?;
    let destination = paths.models.join(&entry.file);
    if destination.exists() && !args.force {
        eprintln!("Already downloaded: {}", destination.display());
    } else {
        let url = artifact_url(entry, &entry.file);
        eprintln!("Pulling {tag} ({:.2} GB)...", entry.size_gb);
        download_verified(
            &url,
            &destination,
            entry.sha256.as_deref(),
            entry.size_bytes,
            false,
        )?;
    }
    for (label, sidecar) in [
        ("TriAttention", entry.triattn.as_ref()),
        ("MTP", entry.mtp.as_ref()),
        ("DSpark", entry.dspark.as_ref()),
    ] {
        let Some(sidecar) = sidecar else {
            continue;
        };
        let destination = paths.models.join(&sidecar.file);
        if destination.exists() {
            eprintln!("  {label} sidecar already present: {}", sidecar.file);
            continue;
        }
        eprintln!("  Fetching {label} sidecar: {}", sidecar.file);
        let url = artifact_url(entry, &sidecar.file);
        if let Err(error) = download_verified(
            &url,
            &destination,
            sidecar.sha256.as_deref(),
            sidecar.size_bytes,
            true,
        ) {
            eprintln!("  warning: {label} sidecar unavailable: {error:#}");
        }
    }
    println!("{}", paths.models.join(&entry.file).display());
    Ok(())
}

fn artifact_url(entry: &ModelEntry, file: &str) -> String {
    let base = env::var("HIPFIRE_HF_BASE")
        .or_else(|_| env::var("HF_ENDPOINT"))
        .unwrap_or_else(|_| "https://huggingface.co".into());
    format!(
        "{}/{}/resolve/main/{}",
        base.trim_end_matches('/'),
        entry.repo,
        file
    )
}

fn download_verified(
    url: &str,
    destination: &Path,
    expected_sha256: Option<&str>,
    expected_size: Option<u64>,
    quiet: bool,
) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(24 * 60 * 60)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut request = agent.get(url);
    if let Some(token) = env::var_os("HF_TOKEN").or_else(|| env::var_os("HUGGING_FACE_HUB_TOKEN")) {
        request = request.header(
            "Authorization",
            &format!("Bearer {}", token.to_string_lossy()),
        );
    }
    let mut response = request
        .call()
        .map_err(|error| anyhow!("download request failed: {error}"))?;
    if !response.status().is_success() {
        bail!("download returned HTTP {} for {url}", response.status());
    }
    let announced = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let temporary = destination.with_extension(format!("part.{}", std::process::id()));
    let mut output = fs::File::create(&temporary)
        .with_context(|| format!("failed to create {}", temporary.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut downloaded = 0_u64;
    let started = Instant::now();
    let mut last_report = Instant::now();
    let result = (|| -> Result<()> {
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            output.write_all(&buffer[..count])?;
            hasher.update(&buffer[..count]);
            downloaded += count as u64;
            if !quiet && last_report.elapsed() >= Duration::from_millis(500) {
                report_progress(downloaded, announced.or(expected_size), started.elapsed());
                last_report = Instant::now();
            }
        }
        output.sync_all()?;
        if !quiet {
            report_progress(downloaded, announced.or(expected_size), started.elapsed());
            eprintln!();
        }
        if let Some(expected) = expected_size {
            if downloaded != expected {
                bail!("size mismatch: expected {expected} bytes, received {downloaded}");
            }
        }
        let actual = format!("{:x}", hasher.finalize());
        if let Some(expected) = expected_sha256 {
            if !actual.eq_ignore_ascii_case(expected) {
                bail!("SHA-256 mismatch: expected {expected}, received {actual}");
            }
        }
        fs::rename(&temporary, destination).with_context(|| {
            format!(
                "failed to install {} as {}",
                temporary.display(),
                destination.display()
            )
        })?;
        eprintln!(
            "  Saved: {} ({:.3} GB, sha256 {})",
            destination.display(),
            downloaded as f64 / 1e9,
            actual
        );
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn report_progress(downloaded: u64, total: Option<u64>, elapsed: Duration) {
    let rate = downloaded as f64 / elapsed.as_secs_f64().max(0.001);
    if let Some(total) = total.filter(|total| *total > 0) {
        let percent = downloaded as f64 / total as f64 * 100.0;
        let remaining = total.saturating_sub(downloaded) as f64 / rate.max(1.0);
        eprint!(
            "\r  {:>6.2}%  {:.2}/{:.2} GB  {:.1} MB/s  ETA {:.0}s",
            percent,
            downloaded as f64 / 1e9,
            total as f64 / 1e9,
            rate / 1e6,
            remaining
        );
    } else {
        eprint!(
            "\r  {:.2} GB  {:.1} MB/s",
            downloaded as f64 / 1e9,
            rate / 1e6
        );
    }
    let _ = std::io::stderr().flush();
}

fn rm_command(paths: &Paths, args: RmArgs) -> Result<()> {
    let loaded = load_registry(&paths.registry);
    let resolved = loaded.registry.model(&args.model);
    let path = find_model_path(paths, &loaded.registry, &args.model)
        .unwrap_or_else(|| paths.models.join(&args.model));
    if !path.is_file() {
        bail!("model not found: {}", path.display());
    }
    let mut targets = BTreeSet::from([path.clone()]);
    if let Some((_, entry)) = resolved {
        targets.extend(
            [&entry.triattn, &entry.mtp, &entry.dspark]
                .into_iter()
                .flatten()
                .map(|sidecar| paths.models.join(&sidecar.file))
                .filter(|path| path.is_file()),
        );
    }
    if let (Some(parent), Some(file)) = (
        path.parent(),
        path.file_name().and_then(|file| file.to_str()),
    ) {
        let stem = file.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(file);
        if let Ok(entries) = fs::read_dir(parent) {
            targets.extend(
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|candidate| {
                        let Some(name) = candidate.file_name().and_then(|name| name.to_str())
                        else {
                            return false;
                        };
                        candidate.is_file()
                            && name != file
                            && ((name.starts_with(&format!("{stem}.triattn"))
                                && name.ends_with(".bin"))
                                || (name.starts_with(stem)
                                    && (name.ends_with(".mtp")
                                        || name.contains("-mtp.")
                                        || name.contains("-dspark."))))
                    }),
            );
        }
    }
    if !args.yes {
        eprint!("Remove {} file(s)? [y/N] ", targets.len());
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            println!("cancelled");
            return Ok(());
        }
    }
    for target in targets {
        fs::remove_file(&target)
            .with_context(|| format!("failed to remove {}", target.display()))?;
        println!("removed {}", target.display());
    }
    Ok(())
}

fn run_command(paths: &Paths, args: RunArgs) -> Result<()> {
    let loaded_registry = load_registry(&paths.registry);
    let registry = &loaded_registry.registry;
    let (canonical, entry) = registry
        .model(&args.model)
        .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry)))
        .unwrap_or((None, None));
    let mut model_path = find_model_path(paths, registry, &args.model);
    if model_path.is_none() {
        if let Some(entry) = entry {
            eprintln!(
                "Model not found locally. Pulling {}...",
                canonical.as_deref().unwrap_or(&args.model)
            );
            pull_command(
                paths,
                PullArgs {
                    model: args.model.clone(),
                    force: false,
                },
            )?;
            model_path = Some(paths.models.join(&entry.file));
        }
    }
    let model_path = model_path.ok_or_else(|| anyhow!("model not found: {}", args.model))?;
    if let Some(image) = &args.image {
        if !image.is_file() {
            bail!("image not found: {}", image.display());
        }
    }
    if let Some(draft) = &args.model_draft {
        if !draft.is_file() {
            bail!("DFlash draft not found: {}", draft.display());
        }
    }
    if args
        .dspark_conf_threshold
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        bail!("--dspark-conf-threshold must be between 0 and 1");
    }

    let resolved = resolved_for_model(paths, &args.model, canonical.as_deref(), entry)?;
    let configured_max_tokens = config_u64(&resolved, "generation.max_tokens")?;
    let max_tokens = args.max_tokens.unwrap_or(configured_max_tokens);
    if max_tokens == 0 || max_tokens > 131_072 {
        bail!("--max-tokens must be between 1 and 131072");
    }
    let temperature = request_f64(&resolved, "generation.temperature", args.temp)?;
    let top_p = request_f64(&resolved, "generation.top_p", args.top_p)?;
    let top_k = request_u64(&resolved, "generation.top_k", None)?;
    let min_p = request_f64(&resolved, "generation.min_p", None)?;
    let presence_penalty = request_f64(&resolved, "generation.presence_penalty", None)?;
    let repeat_penalty = request_f64(&resolved, "generation.repeat_penalty", args.repeat_penalty)?;
    let system_prompt = request_string(&resolved, "prompt.system", args.system.clone())?
        .filter(|value| !value.is_empty());
    if temperature.is_some_and(|value| !(0.0..=2.0).contains(&value)) {
        bail!("--temp must be between 0 and 2");
    }
    if top_p.is_some_and(|value| !(0.0 < value && value <= 1.0)) {
        bail!("--top-p must be in (0, 1]");
    }
    if repeat_penalty.is_some_and(|value| !(1.0..=3.0).contains(&value)) {
        bail!("--repeat-penalty must be between 1 and 3");
    }

    let prompt = if args.prompt.is_empty() {
        if args.image.is_some() {
            "Describe this image.".to_owned()
        } else {
            "Hello".to_owned()
        }
    } else {
        args.prompt.join(" ")
    };
    let host = config_string(&resolved, "serve.host")?;
    let port = config_u64(&resolved, "serve.port")? as u16;
    let force_local = process_truthy("HIPFIRE_LOCAL")
        || args.image.is_some()
        || args.kv_mode.is_some()
        || args.kv_backend.is_some()
        || args.speculation.is_some()
        || args.model_draft.is_some()
        || args.draft_max.is_some()
        || args.dspark_conf_threshold.is_some();
    if !force_local && service_ready(&host, port, Duration::from_millis(150)) {
        return run_via_http(
            &host,
            port,
            &args.model,
            &prompt,
            system_prompt.as_deref(),
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            repeat_penalty,
            max_tokens,
            args.json,
            args.no_stream,
        );
    }

    let daemon = find_daemon(paths).ok_or_else(|| {
        anyhow!(
            "daemon binary not found; build `cargo build --release --features deltanet -p hipfire-runtime --example daemon`"
        )
    })?;
    let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    let mut engine = Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config)?;
    engine.ping()?;
    let mut params = load_params(
        &resolved,
        entry,
        &model_path,
        max_tokens,
        args.kv_mode.as_deref(),
        args.kv_backend.as_deref(),
    )?;
    let selector = args
        .speculation
        .clone()
        .unwrap_or(config_string(&resolved, "speculation.mode")?);
    apply_speculation_selector(&mut params, &selector)?;
    // Final effective selector wins: re-project inherited draft only when DFlash
    // remains enabled (config-off + `run --spec dflash` must still carry draft).
    project_dflash_draft(&mut params);
    if let Some(draft) = &args.model_draft {
        params["draft"] = serde_json::json!(draft.display().to_string());
        if args.speculation.is_none() {
            apply_speculation_selector(&mut params, "dflash")?;
        }
    }
    if let Some(window) = args.draft_max {
        if !(1..=32).contains(&window) {
            bail!("--draft-max must be between 1 and 32");
        }
        match args.speculation.as_deref().unwrap_or("auto") {
            "ngram" => params["ngram_k"] = serde_json::json!(window),
            "mtp" => params["mtp_k"] = serde_json::json!(window),
            _ => {
                params["mtp_k"] = serde_json::json!(window);
                params["ngram_k"] = serde_json::json!(window);
            }
        }
    }
    if let Some(value) = args.dspark_conf_threshold {
        params["dspark_conf_threshold"] = serde_json::json!(value);
    }
    let loaded = engine.load(&model_path, params)?;
    if !args.json {
        eprintln!(
            "[{}] {}d {}L {} vocab",
            loaded
                .get("arch")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            loaded
                .get("dim")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            loaded
                .get("layers")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            loaded
                .get("vocab")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        );
    }

    let mut request = serde_json::json!({
        "type": "generate",
        "id": "run",
        "prompt": prompt,
        "max_tokens": max_tokens,
    });
    insert_optional_f64(&mut request, "temperature", temperature);
    insert_optional_f64(&mut request, "top_p", top_p);
    insert_optional_u64(&mut request, "top_k", top_k);
    insert_optional_f64(&mut request, "min_p", min_p);
    insert_optional_f64(&mut request, "presence_penalty", presence_penalty);
    insert_optional_f64(&mut request, "repeat_penalty", repeat_penalty);
    if let Some(system) = system_prompt {
        request["system"] = serde_json::Value::String(system);
    }
    if let Some(image) = args.image {
        request["image"] = serde_json::Value::String(image.display().to_string());
    }
    apply_reasoning_request(&resolved, &mut request)?;

    let mut content = String::new();
    let stream = !args.no_stream && !args.json;
    let done = engine.generate(&request, |event| {
        if event.get("type").and_then(serde_json::Value::as_str) == Some("token") {
            if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                content.push_str(text);
                if stream {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
            }
        }
        Ok(())
    })?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "content": content,
                "tokens": done.get("tokens").and_then(serde_json::Value::as_u64),
                "tok_s": done.get("tok_s").and_then(serde_json::Value::as_f64),
                "finish_reason": done.get("finish_reason"),
            }))?
        );
    } else if args.no_stream {
        println!("{content}");
    } else {
        println!();
    }
    let _ = engine.unload();
    Ok(())
}

fn process_truthy(name: &str) -> bool {
    hipfire_config::process_value(name).is_some_and(|value| {
        !matches!(
            value.to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn run_via_http(
    host: &str,
    port: u16,
    model: &str,
    prompt: &str,
    system: Option<&str>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    min_p: Option<f64>,
    presence_penalty: Option<f64>,
    repeat_penalty: Option<f64>,
    max_tokens: u64,
    json: bool,
    no_stream: bool,
) -> Result<()> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": prompt }));
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    });
    insert_optional_f64(&mut body, "temperature", temperature);
    insert_optional_f64(&mut body, "top_p", top_p);
    insert_optional_u64(&mut body, "top_k", top_k);
    insert_optional_f64(&mut body, "min_p", min_p);
    insert_optional_f64(&mut body, "presence_penalty", presence_penalty);
    insert_optional_f64(&mut body, "repeat_penalty", repeat_penalty);
    let timeout = Duration::from_secs(60 * 60);
    if json || no_stream {
        let response = complete_openai_chat(host, port, body, timeout)?;
        let content = response
            .pointer("/choices/0/message/content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "content": content,
                    "tokens": response.pointer("/usage/completion_tokens"),
                    "tok_s": response.pointer("/hipfire/tok_s"),
                    "finish_reason": response.pointer("/choices/0/finish_reason"),
                }))?
            );
        } else {
            println!("{content}");
        }
        return Ok(());
    }

    stream_openai_chat(
        host,
        port,
        body,
        timeout,
        |text| {
            print!("{text}");
            std::io::stdout().flush()?;
            Ok(())
        },
        || false,
    )?;
    println!();
    Ok(())
}

fn chat_command(paths: &Paths, args: ChatArgs) -> Result<()> {
    let (_, resolved) = resolved_global(paths, true)?;
    let host = config_string(&resolved, "serve.host")?;
    let port = config_u64(&resolved, "serve.port")? as u16;
    let model = args
        .model
        .unwrap_or(config_string(&resolved, "serve.default_model")?);
    let max_tokens = args
        .max_tokens
        .unwrap_or(config_u64(&resolved, "generation.max_tokens")?);
    if max_tokens == 0 || max_tokens > 131_072 {
        bail!("--max-tokens must be between 1 and 131072");
    }
    if let Some(value) = args.temp {
        if !(0.0..=2.0).contains(&value) {
            bail!("--temp must be between 0 and 2");
        }
    }
    if let Some(value) = args.top_p {
        if !(0.0 < value && value <= 1.0) {
            bail!("--top-p must be in (0, 1]");
        }
    }

    if !service_ready(&host, port, Duration::from_millis(150)) {
        let serve_args = ServeArgs {
            positionals: vec![host.clone(), port.to_string()],
            model: None,
            detach: true,
            no_prewarm: true,
            kv_mode: None,
            kv_backend: None,
            idle_timeout: None,
            tp: None,
            foreground_child: false,
        };
        detach_serve(paths, &serve_args, &host, port)?;
    }
    let client_host = probe_host(&host);
    eprintln!("Interactive chat with {model}. Commands: /clear, /exit");
    let mut messages = Vec::new();
    if let Some(system) = args.system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    let stdin = std::io::stdin();
    loop {
        eprint!("you> ");
        std::io::stderr().flush()?;
        let mut input = String::new();
        if stdin.read_line(&mut input)? == 0 {
            println!();
            break;
        }
        let input = input.trim_end_matches(['\r', '\n']);
        match input.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/clear" => {
                messages.retain(|message| {
                    message.get("role").and_then(serde_json::Value::as_str) == Some("system")
                });
                eprintln!("conversation cleared");
                continue;
            }
            _ => {}
        }
        messages.push(serde_json::json!({ "role": "user", "content": input }));
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        });
        if let Some(value) = args.temp {
            body["temperature"] = serde_json::json!(value);
        }
        if let Some(value) = args.top_p {
            body["top_p"] = serde_json::json!(value);
        }
        print!("assistant> ");
        std::io::stdout().flush()?;
        let mut assistant = String::new();
        let result = stream_openai_chat(
            client_host,
            port,
            body,
            Duration::from_secs(60 * 60),
            |text| {
                assistant.push_str(text);
                print!("{text}");
                std::io::stdout().flush()?;
                Ok(())
            },
            || false,
        );
        println!();
        if let Err(error) = result {
            messages.pop();
            return Err(error.into());
        }
        messages.push(serde_json::json!({ "role": "assistant", "content": assistant }));
    }
    let _ = args.no_color;
    Ok(())
}

#[derive(Debug)]
struct ServeMeta {
    current_model: Option<String>,
    loading_model: Option<String>,
    instance_token: String,
    requests_served: u64,
    recent_tok_s: Option<f64>,
    started: Instant,
    last_activity: Instant,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct ServePidRecord {
    pid: u32,
    #[serde(default)]
    start_time: Option<u64>,
    #[serde(default)]
    port: Option<u16>,
    #[serde(default)]
    token: Option<String>,
    #[serde(skip)]
    legacy: bool,
}

struct ServeRuntime {
    engine: Engine,
    paths: Paths,
    registry: RegistryV1,
    current_path: Option<PathBuf>,
    current_max_seq: u64,
    cache_capable: bool,
    kv_override: Option<String>,
    kv_backend_override: Option<String>,
    tp: Option<u64>,
}

struct ServeShared {
    runtime: Mutex<ServeRuntime>,
    meta: Mutex<ServeMeta>,
    max_request_bytes: u64,
    admission: Arc<Admission>,
    idle_timeout: Duration,
}

#[derive(Debug)]
struct Completion {
    id: String,
    created: u64,
    model: String,
    content: String,
    reasoning_content: String,
    preserve_thinking: bool,
    tool_calls: Vec<serde_json::Value>,
    done: serde_json::Value,
}

#[derive(Debug, PartialEq, Eq)]
enum ThinkFragment {
    Content(String),
    Reasoning(String),
}

#[derive(Debug, Default)]
struct ThinkChannelRouter {
    in_think: bool,
    pending: String,
    strip_answer_newlines: bool,
    semantic_split: bool,
    semantic_pending: String,
    semantic_reasoning: Option<bool>,
}

impl ThinkChannelRouter {
    fn set_started_in_think(&mut self, started: bool) {
        self.in_think = started;
    }

    fn push(&mut self, text: &str) -> Vec<ThinkFragment> {
        if self.semantic_split {
            return self.push_semantic(text, false);
        }
        self.pending.push_str(text);
        self.drain(false)
    }

    fn push_semantic(&mut self, text: &str, reasoning: bool) -> Vec<ThinkFragment> {
        let mut out = if self.pending.is_empty() {
            Vec::new()
        } else {
            self.drain(true)
        };
        self.semantic_split = true;
        if self.semantic_reasoning != Some(reasoning) {
            out.extend(self.drain_semantic(true));
            self.semantic_reasoning = Some(reasoning);
        }
        self.semantic_pending.push_str(text);
        out.extend(self.drain_semantic(false));
        out
    }

    fn finish(&mut self) -> Vec<ThinkFragment> {
        let mut out = self.drain(true);
        out.extend(self.drain_semantic(true));
        out
    }

    fn drain(&mut self, flush: bool) -> Vec<ThinkFragment> {
        const OPEN: &str = "<think>";
        const CLOSE: &str = "</think>";
        let mut out = Vec::new();
        loop {
            if let Some((index, marker)) = next_control_marker(&self.pending) {
                let before = self.pending[..index].to_owned();
                self.emit(before, &mut out);
                self.pending.drain(..index + marker.len());
                match marker {
                    OPEN => self.in_think = true,
                    CLOSE => {
                        self.in_think = false;
                        self.strip_answer_newlines = true;
                    }
                    _ => {}
                }
                continue;
            }

            let held = if flush {
                0
            } else {
                longest_control_prefix_suffix(&self.pending)
            };
            let emit_len = self.pending.len().saturating_sub(held);
            if emit_len > 0 {
                let text = self.pending[..emit_len].to_owned();
                self.pending.drain(..emit_len);
                self.emit(text, &mut out);
            }
            break;
        }
        out
    }

    fn drain_semantic(&mut self, flush: bool) -> Vec<ThinkFragment> {
        let mut out = Vec::new();
        loop {
            if let Some((index, marker)) = next_control_marker(&self.semantic_pending) {
                let before = self.semantic_pending[..index].to_owned();
                self.emit_semantic(before, &mut out);
                self.semantic_pending.drain(..index + marker.len());
                continue;
            }
            let held = if flush {
                0
            } else {
                longest_control_prefix_suffix(&self.semantic_pending)
            };
            let emit_len = self.semantic_pending.len().saturating_sub(held);
            if emit_len > 0 {
                let text = self.semantic_pending[..emit_len].to_owned();
                self.semantic_pending.drain(..emit_len);
                self.emit_semantic(text, &mut out);
            }
            break;
        }
        out
    }

    fn emit(&mut self, mut text: String, out: &mut Vec<ThinkFragment>) {
        if !self.in_think && self.strip_answer_newlines {
            let trimmed = text.trim_start_matches(['\r', '\n']);
            if trimmed.is_empty() {
                return;
            }
            text = trimmed.to_owned();
            self.strip_answer_newlines = false;
        }
        if text.is_empty() {
            return;
        }
        if self.in_think {
            out.push(ThinkFragment::Reasoning(text));
        } else {
            out.push(ThinkFragment::Content(text));
        }
    }

    fn emit_semantic(&self, text: String, out: &mut Vec<ThinkFragment>) {
        if text.is_empty() {
            return;
        }
        if self.semantic_reasoning == Some(true) {
            out.push(ThinkFragment::Reasoning(text));
        } else {
            out.push(ThinkFragment::Content(text));
        }
    }
}

const OUTPUT_CONTROL_MARKERS: &[&str] = &[
    "<think>",
    "</think>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
];

fn next_control_marker(text: &str) -> Option<(usize, &'static str)> {
    OUTPUT_CONTROL_MARKERS
        .iter()
        .filter_map(|marker| text.find(marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
}

fn longest_control_prefix_suffix(text: &str) -> usize {
    OUTPUT_CONTROL_MARKERS
        .iter()
        .map(|marker| {
            let max = text.len().min(marker.len().saturating_sub(1));
            (1..=max)
                .rev()
                .find(|&len| text.ends_with(&marker[..len]))
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
}

#[derive(Debug, Default)]
struct AdmissionState {
    busy: bool,
    queued: usize,
}

#[derive(Debug)]
struct Admission {
    state: Mutex<AdmissionState>,
    available: Condvar,
    max_queue: usize,
    timeout: Duration,
}

#[derive(Debug)]
struct AdmissionError {
    message: String,
    retry_after_seconds: u64,
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AdmissionError {}

#[derive(Debug)]
struct AdmissionGuard {
    admission: Arc<Admission>,
}

impl Admission {
    fn new(max_queue: usize, timeout: Duration) -> Self {
        Self {
            state: Mutex::new(AdmissionState::default()),
            available: Condvar::new(),
            max_queue,
            timeout,
        }
    }

    fn acquire(self: &Arc<Self>) -> std::result::Result<AdmissionGuard, AdmissionError> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if !state.busy {
            state.busy = true;
            return Ok(AdmissionGuard {
                admission: Arc::clone(self),
            });
        }
        if self.max_queue != 0 && state.queued >= self.max_queue {
            return Err(AdmissionError {
                message: format!(
                    "serve queue full (depth {}/{})",
                    state.queued, self.max_queue
                ),
                retry_after_seconds: self.retry_after_seconds(),
            });
        }
        state.queued = state.queued.saturating_add(1);
        let started = Instant::now();
        loop {
            if self.timeout.is_zero() {
                state = self
                    .available
                    .wait(state)
                    .unwrap_or_else(|error| error.into_inner());
            } else {
                let remaining = self.timeout.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    state.queued = state.queued.saturating_sub(1);
                    return Err(AdmissionError {
                        message: format!(
                            "serve queue wait exceeded {}ms",
                            self.timeout.as_millis()
                        ),
                        retry_after_seconds: self.retry_after_seconds(),
                    });
                }
                let (next, wait) = self
                    .available
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(|error| error.into_inner());
                state = next;
                if wait.timed_out() && state.busy {
                    state.queued = state.queued.saturating_sub(1);
                    return Err(AdmissionError {
                        message: format!(
                            "serve queue wait exceeded {}ms",
                            self.timeout.as_millis()
                        ),
                        retry_after_seconds: self.retry_after_seconds(),
                    });
                }
            }
            if !state.busy {
                state.queued = state.queued.saturating_sub(1);
                state.busy = true;
                return Ok(AdmissionGuard {
                    admission: Arc::clone(self),
                });
            }
        }
    }

    fn inflight(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        usize::from(state.busy) + state.queued
    }

    fn retry_after_seconds(&self) -> u64 {
        if self.timeout.is_zero() {
            1
        } else {
            self.timeout.as_secs().max(1)
        }
    }
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        let mut state = self
            .admission
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.busy = false;
        self.admission.available.notify_one();
    }
}

fn serve_command(paths: &Paths, mut args: ServeArgs) -> Result<()> {
    let (_, resolved) = resolved_global(paths, true)?;
    let default_host = config_string(&resolved, "serve.host")?;
    let default_port = config_u64(&resolved, "serve.port")? as u16;
    let (host, port, positional_model) =
        resolve_serve_positionals(paths, &args.positionals, &default_host, default_port)?;
    if let Some(positional_model) = positional_model {
        if args
            .model
            .as_ref()
            .is_some_and(|model| model != &positional_model)
        {
            bail!("serve model specified more than once");
        }
        args.model = Some(positional_model);
    }
    if args.detach && !args.foreground_child {
        return detach_serve(paths, &args, &host, port);
    }
    serve_foreground(paths, &args, &host, port, resolved)
}

fn resolve_serve_positionals(
    paths: &Paths,
    values: &[String],
    default_host: &str,
    default_port: u16,
) -> Result<(String, u16, Option<String>)> {
    let registry = load_registry(&paths.registry).registry;
    let mut host = None;
    let mut port = None;
    let mut model = None;
    for value in values {
        if let Ok(value_port) = value.parse::<u16>() {
            if port.replace(value_port).is_some() {
                bail!("serve port specified more than once");
            }
            continue;
        }
        if let Some((value_host, value_port)) = parse_host_port(value)? {
            if host.replace(value_host).is_some() || port.replace(value_port).is_some() {
                bail!("serve bind specified more than once");
            }
            continue;
        }
        let is_model =
            registry.model(value).is_some() || find_model_path(paths, &registry, value).is_some();
        if is_model && model.is_none() {
            model = Some(value.clone());
        } else if host.replace(value.clone()).is_some() {
            bail!("serve host specified more than once");
        }
    }
    Ok((
        host.unwrap_or_else(|| default_host.to_owned()),
        port.unwrap_or(default_port),
        model,
    ))
}

fn parse_host_port(value: &str) -> Result<Option<(String, u16)>> {
    if let Some(stripped) = value.strip_prefix('[') {
        if let Some((host, port)) = stripped.split_once("]:") {
            return Ok(Some((
                host.to_owned(),
                port.parse().context("invalid serve port")?,
            )));
        }
    }
    if value.matches(':').count() == 1 {
        if let Some((host, port)) = value.rsplit_once(':') {
            if let Ok(port) = port.parse::<u16>() {
                return Ok(Some((host.to_owned(), port)));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
fn parse_bind(
    address: Option<&str>,
    port: Option<u16>,
    default_host: &str,
    default_port: u16,
) -> Result<(String, u16)> {
    let Some(address) = address else {
        return Ok((default_host.to_owned(), port.unwrap_or(default_port)));
    };
    if let Ok(port_only) = address.parse::<u16>() {
        return Ok((default_host.to_owned(), port_only));
    }
    if let Some(stripped) = address.strip_prefix('[') {
        if let Some((host, port_text)) = stripped.split_once("]:") {
            return Ok((
                host.to_owned(),
                port_text.parse().context("invalid serve port")?,
            ));
        }
    }
    if address.matches(':').count() == 1 {
        if let Some((host, port_text)) = address.rsplit_once(':') {
            if let Ok(parsed) = port_text.parse::<u16>() {
                return Ok((host.to_owned(), parsed));
            }
        }
    }
    Ok((address.to_owned(), port.unwrap_or(default_port)))
}

fn detach_serve(paths: &Paths, args: &ServeArgs, host: &str, port: u16) -> Result<()> {
    fs::create_dir_all(&paths.root)?;
    let log_path = paths.root.join("serve.log");
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let executable = env::current_exe().context("failed to resolve native hipfire binary")?;
    let mut command = Command::new(executable);
    command
        .arg("serve")
        .arg(host)
        .arg(port.to_string())
        .arg("--foreground-child")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    if args.no_prewarm {
        command.arg("--no-prewarm");
    }
    if let Some(model) = &args.model {
        command.arg("--model").arg(model);
    }
    if let Some(mode) = &args.kv_mode {
        command.arg("--kv-mode").arg(mode);
    }
    if let Some(backend) = &args.kv_backend {
        command.arg("--kv-backend").arg(backend);
    }
    if let Some(seconds) = args.idle_timeout {
        command.arg("--idle-timeout").arg(seconds.to_string());
    }
    if let Some(tp) = args.tp {
        command.arg("--tp").arg(tp.to_string());
    }
    let mut child = command.spawn().context("failed to detach native serve")?;
    let probe_host = match host {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        other => other,
    };
    for _ in 0..600 {
        if let Some(status) = child.try_wait()? {
            bail!(
                "native serve exited before readiness ({status}); see {}",
                log_path.display()
            );
        }
        if health_ready(probe_host, port) {
            println!(
                "hipfire serve running at http://{}:{} (PID {}, log {})",
                host,
                port,
                child.id(),
                log_path.display()
            );
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    bail!(
        "native serve did not become ready within 60s; PID {}, see {}",
        child.id(),
        log_path.display()
    )
}

fn health_ready(host: &str, port: u16) -> bool {
    let url = if host.contains(':') {
        format!("http://[{host}]:{port}/health")
    } else {
        format!("http://{host}:{port}/health")
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(100)))
        .http_status_as_error(false)
        .build()
        .into();
    agent
        .get(&url)
        .call()
        .is_ok_and(|response| response.status().is_success())
}

fn serve_foreground(
    paths: &Paths,
    args: &ServeArgs,
    host: &str,
    port: u16,
    global: hipfire_config::ResolvedConfig,
) -> Result<()> {
    let daemon = find_daemon(paths).ok_or_else(|| anyhow!("daemon binary not found"))?;
    let registry = load_registry(&paths.registry).registry;
    let process_config = hipfire_config::ProcessConfig::from_resolved(&global)?;
    let mut engine = Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config)?;
    engine.ping()?;
    let max_request_bytes = config_u64(&global, "serve.max_request_bytes")?;
    let max_queue = config_u64(&global, "serve.max_queue")? as usize;
    let queue_timeout = Duration::from_millis(config_u64(&global, "serve.queue_timeout_ms")?);
    let idle_timeout = Duration::from_secs(
        args.idle_timeout
            .unwrap_or(config_u64(&global, "serve.idle_timeout_seconds")?),
    );
    let default_model = args
        .model
        .clone()
        .unwrap_or(config_string(&global, "serve.default_model")?);
    let instance_token = serve_instance_token();
    let shared = Arc::new(ServeShared {
        runtime: Mutex::new(ServeRuntime {
            engine,
            paths: paths.clone(),
            registry: registry.clone(),
            current_path: None,
            current_max_seq: 0,
            cache_capable: false,
            kv_override: args.kv_mode.clone(),
            kv_backend_override: args.kv_backend.clone(),
            tp: args.tp,
        }),
        meta: Mutex::new(ServeMeta {
            current_model: None,
            loading_model: None,
            instance_token: instance_token.clone(),
            requests_served: 0,
            recent_tok_s: None,
            started: Instant::now(),
            last_activity: Instant::now(),
        }),
        max_request_bytes,
        admission: Arc::new(Admission::new(max_queue, queue_timeout)),
        idle_timeout,
    });

    let bind = format_bind(host, port);
    let server = Server::http(&bind).map_err(|error| anyhow!("failed to bind {bind}: {error}"))?;
    fs::create_dir_all(&paths.root)?;
    let pid_path = paths.root.join("serve.pid");
    let pid_record = ServePidRecord {
        pid: std::process::id(),
        start_time: proc_start_time(std::process::id()),
        port: Some(port),
        token: Some(instance_token),
        legacy: false,
    };
    fs::write(
        &pid_path,
        format!("{}\n", serde_json::to_string(&pid_record)?),
    )?;
    let cleanup = pid_path.clone();
    ctrlc::set_handler(move || {
        let _ = fs::remove_file(&cleanup);
        std::process::exit(0);
    })
    .context("failed to install serve signal handler")?;
    eprintln!("[hipfire] native serve listening on http://{bind}");
    if !args.no_prewarm {
        let shared = Arc::clone(&shared);
        thread::spawn(move || {
            shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .loading_model = Some(default_model.clone());
            let result = shared
                .runtime
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .ensure_model(&default_model, &shared.meta, None);
            shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .loading_model = None;
            match result {
                Ok(_) => eprintln!("[hipfire] pre-warmed {default_model}"),
                Err(error) => eprintln!("[hipfire] pre-warm failed: {error:#}; serving lazily"),
            }
        });
    }
    if !shared.idle_timeout.is_zero() {
        let shared = Arc::clone(&shared);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs(1));
            if shared.admission.inflight() != 0 {
                continue;
            }
            let expired = {
                let meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                meta.current_model.is_some() && meta.last_activity.elapsed() >= shared.idle_timeout
            };
            if !expired {
                continue;
            }
            let unloaded = {
                let mut runtime = shared
                    .runtime
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                if runtime.current_path.is_some() {
                    let result = runtime.engine.unload();
                    if result.is_ok() {
                        runtime.current_path = None;
                        runtime.current_max_seq = 0;
                        runtime.cache_capable = false;
                    }
                    result
                } else {
                    Ok(())
                }
            };
            if unloaded.is_ok() {
                let mut meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                meta.current_model = None;
                meta.loading_model = None;
                meta.last_activity = Instant::now();
                eprintln!("[hipfire] unloaded idle model");
            }
        });
    }
    for request in server.incoming_requests() {
        let shared = Arc::clone(&shared);
        thread::spawn(move || {
            if let Err(error) = handle_http(request, shared) {
                eprintln!("[hipfire] HTTP request failed: {error:#}");
            }
        });
    }
    let _ = fs::remove_file(pid_path);
    Ok(())
}

fn format_bind(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

fn handle_http(mut request: Request, shared: Arc<ServeShared>) -> Result<()> {
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or(request.url())
        .to_owned();
    match (request.method(), path.as_str()) {
        (&Method::Get, "/health") => {
            let meta = shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            request.respond(json_response(
                serde_json::json!({
                    "status": "ok",
                    "model": meta.current_model,
                    "loading_model": meta.loading_model,
                    "pid": std::process::id(),
                    "token": meta.instance_token,
                    "native": true,
                }),
                200,
            ))?;
        }
        (&Method::Get, "/stats") => {
            let meta = shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            request.respond(json_response(
                serde_json::json!({
                    "model": meta.current_model,
                    "uptime_sec": meta.started.elapsed().as_secs(),
                    "queue_depth": shared.admission.inflight(),
                    "requests_served": meta.requests_served,
                    "recent_tok_s": meta.recent_tok_s,
                }),
                200,
            ))?;
        }
        (&Method::Get, "/v1/models") => {
            let runtime = shared
                .runtime
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let local = list_local_models(&runtime.paths, &runtime.registry)?;
            request.respond(json_response(
                serde_json::json!({
                    "object": "list",
                    "data": local.into_iter().map(|model| serde_json::json!({
                        "id": model.registry_tag.unwrap_or(model.name),
                        "object": "model",
                        "owned_by": "hipfire",
                    })).collect::<Vec<_>>()
                }),
                200,
            ))?;
        }
        (&Method::Options, _) => {
            request.respond(
                Response::empty(204)
                    .with_header(header("Access-Control-Allow-Origin", "*"))
                    .with_header(header(
                        "Access-Control-Allow-Headers",
                        "Content-Type, Authorization",
                    ))
                    .with_header(header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")),
            )?;
        }
        (&Method::Post, "/v1/chat/completions") => {
            let body = match read_request_json(&mut request, shared.max_request_bytes) {
                Ok(body) => body,
                Err(error) => {
                    let message = error.to_string();
                    let status = if message.contains("exceeds") {
                        413
                    } else {
                        400
                    };
                    request.respond(openai_error(&message, status))?;
                    return Ok(());
                }
            };
            let guard = match shared.admission.acquire() {
                Ok(guard) => guard,
                Err(error) => {
                    request.respond(admission_error_response(&error))?;
                    return Ok(());
                }
            };
            if body.get("stream").and_then(serde_json::Value::as_bool) == Some(true) {
                respond_streaming(request, shared, body, guard)?;
            } else {
                let completion = complete_request(&shared, &body, guard, None, |_| Ok(()));
                match completion {
                    Ok(completion) => {
                        request.respond(json_response(completion_json(&completion), 200))?
                    }
                    Err(error) => {
                        let message = error.to_string();
                        request.respond(openai_error(&message, request_error_status(&message)))?
                    }
                };
            }
        }
        _ => request.respond(openai_error("not found", 404))?,
    }
    Ok(())
}

fn request_error_status(message: &str) -> u16 {
    let lower = message.to_ascii_lowercase();
    if lower.contains("model not found") {
        404
    } else if lower.contains("kv budget")
        || lower.contains("max_tokens")
        || lower.contains("invalid")
        || lower.contains("required")
    {
        400
    } else {
        500
    }
}

fn read_request_json(request: &mut Request, max_bytes: u64) -> Result<serde_json::Value> {
    if request
        .headers()
        .iter()
        .find(|header| header.field.equiv("Content-Length"))
        .and_then(|header| header.value.as_str().parse::<u64>().ok())
        .is_some_and(|length| length > max_bytes)
    {
        bail!("request body exceeds {max_bytes} bytes");
    }
    let mut bytes = Vec::new();
    request
        .as_reader()
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        bail!("request body exceeds {max_bytes} bytes");
    }
    serde_json::from_slice(&bytes).context("request body is not valid JSON")
}

fn respond_streaming(
    request: Request,
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
) -> Result<()> {
    let (sender, receiver) = mpsc::channel::<Vec<u8>>();
    thread::spawn(move || {
        let id = request_id();
        let created = unix_timestamp();
        let include_usage = body
            .pointer("/stream_options/include_usage")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        let model = body
            .get("model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let first = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
        });
        let _ = sender.send(sse_data(&first));
        let result = complete_request(
            &shared,
            &body,
            guard,
            Some((id.clone(), created)),
            |event| {
                let delta = match event.get("type").and_then(serde_json::Value::as_str) {
                    Some("token") => event
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .map(|text| serde_json::json!({ "content": text })),
                    Some("reasoning") => event
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .map(|text| serde_json::json!({ "reasoning_content": text })),
                    Some("tool_calls") => event.get("calls").map(openai_tool_call_delta),
                    _ => None,
                };
                if let Some(delta) = delta {
                    let chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
                    });
                    sender.send(sse_data(&chunk)).ok();
                }
                Ok(())
            },
        );
        match result {
            Ok(completion) => {
                let finish_reason = if completion.tool_calls.is_empty() {
                    completion
                        .done
                        .get("finish_reason")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("stop")
                } else {
                    "tool_calls"
                };
                let mut final_chunk = serde_json::json!({
                    "id": completion.id,
                    "object": "chat.completion.chunk",
                    "created": completion.created,
                    "model": completion.model,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
                    "timings": completion_timings(&completion),
                });
                if include_usage {
                    final_chunk["usage"] = completion_usage(&completion);
                }
                let _ = sender.send(sse_data(&final_chunk));
            }
            Err(error) => {
                let _ = sender.send(sse_data(&serde_json::json!({
                    "error": { "message": error.to_string(), "type": "server_error" }
                })));
            }
        }
        let _ = sender.send(b"data: [DONE]\n\n".to_vec());
    });
    request.respond(Response::new(
        StatusCode(200),
        vec![
            header("Content-Type", "text/event-stream"),
            header("Cache-Control", "no-cache"),
            header("Connection", "keep-alive"),
            header("Access-Control-Allow-Origin", "*"),
        ],
        ChannelReader::new(receiver),
        None,
        None,
    ))?;
    Ok(())
}

fn complete_request(
    shared: &ServeShared,
    body: &serde_json::Value,
    _guard: AdmissionGuard,
    request_identity: Option<(String, u64)>,
    mut event_callback: impl FnMut(&serde_json::Value) -> Result<()>,
) -> Result<Completion> {
    let model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("model is required"))?
        .to_owned();
    let image_base64 = request_image_base64(body.get("messages"))?;
    let mut runtime = shared
        .runtime
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let resolved = runtime.ensure_model(&model, &shared.meta, None)?;
    if !runtime.cache_capable {
        runtime.engine.reset()?;
    }
    let max_tokens = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(config_u64(&resolved, "generation.max_tokens")?);
    let required_max_seq = max_tokens.saturating_add(1024);
    if runtime.current_max_seq < required_max_seq {
        runtime.ensure_model(&model, &shared.meta, Some(required_max_seq))?;
    }
    let mut normalized_messages = normalize_openai_messages(body.get("messages"));
    let default_system = request_string(&resolved, "prompt.system", None)?;
    inject_default_system_message(&mut normalized_messages, default_system.as_deref());
    let mut generate = serde_json::json!({
        "type": "generate",
        "id": request_id(),
        "prompt": last_user_prompt(&normalized_messages).unwrap_or_else(|| "Hello".into()),
        "messages": normalized_messages,
        "max_tokens": max_tokens,
    });
    if let Some(image) = image_base64 {
        generate["image_base64"] = serde_json::Value::String(image);
    }
    for (key, config_key) in [
        ("temperature", "generation.temperature"),
        ("top_p", "generation.top_p"),
        ("repeat_penalty", "generation.repeat_penalty"),
    ] {
        let explicit = body.get(key).and_then(serde_json::Value::as_f64);
        insert_optional_f64(
            &mut generate,
            key,
            request_f64(&resolved, config_key, explicit)?,
        );
    }
    for name in [
        "tools",
        "tool_choice",
        "frequency_penalty",
        "stop",
        "reasoning_effort",
    ] {
        if let Some(value) = body.get(name) {
            generate[name] = value.clone();
        }
    }
    if let Some(value) = body.get("top_k") {
        generate["top_k"] = value.clone();
    } else {
        insert_optional_u64(
            &mut generate,
            "top_k",
            request_u64(&resolved, "generation.top_k", None)?,
        );
    }
    for (key, config_key) in [
        ("min_p", "generation.min_p"),
        ("presence_penalty", "generation.presence_penalty"),
    ] {
        if let Some(value) = body.get(key) {
            generate[key] = value.clone();
        } else {
            insert_optional_f64(
                &mut generate,
                key,
                request_f64(&resolved, config_key, None)?,
            );
        }
    }
    apply_http_reasoning_request(body, &resolved, &mut generate)?;
    let (id, created) = request_identity.unwrap_or_else(|| (request_id(), unix_timestamp()));
    generate["id"] = serde_json::Value::String(id.clone());
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut tool_calls = Vec::new();
    let mut think_router = ThinkChannelRouter::default();
    let done = runtime.engine.generate(&generate, |event| {
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("gen_start") => {
                if let Some(started) = event
                    .get("started_in_think")
                    .and_then(serde_json::Value::as_bool)
                {
                    think_router.set_started_in_think(started);
                }
                return Ok(());
            }
            Some("token") => {
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    let fragments = if event.get("reasoning").and_then(serde_json::Value::as_bool)
                        == Some(true)
                    {
                        think_router.push_semantic(text, true)
                    } else {
                        think_router.push(text)
                    };
                    forward_think_fragments(
                        fragments,
                        &mut content,
                        &mut reasoning_content,
                        &mut event_callback,
                    )
                    .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;
                }
                return Ok(());
            }
            Some("reasoning") => {
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    forward_think_fragments(
                        think_router.push_semantic(text, true),
                        &mut content,
                        &mut reasoning_content,
                        &mut event_callback,
                    )
                    .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;
                }
                return Ok(());
            }
            Some("tool_calls") => {
                if let Some(calls) = event.get("calls").and_then(serde_json::Value::as_array) {
                    tool_calls.extend(calls.iter().cloned());
                }
            }
            Some("done") => {
                forward_think_fragments(
                    think_router.finish(),
                    &mut content,
                    &mut reasoning_content,
                    &mut event_callback,
                )
                .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;
            }
            _ => {}
        }
        event_callback(event)
            .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;
        Ok(())
    })?;
    let mut meta = shared
        .meta
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    meta.requests_served = meta.requests_served.saturating_add(1);
    meta.recent_tok_s = done.get("tok_s").and_then(serde_json::Value::as_f64);
    meta.last_activity = Instant::now();
    Ok(Completion {
        id,
        created,
        model,
        content,
        reasoning_content,
        preserve_thinking: body
            .pointer("/chat_template_kwargs/preserve_thinking")
            .and_then(serde_json::Value::as_bool)
            == Some(true),
        tool_calls,
        done,
    })
}

fn forward_think_fragments(
    fragments: Vec<ThinkFragment>,
    content: &mut String,
    reasoning_content: &mut String,
    event_callback: &mut impl FnMut(&serde_json::Value) -> Result<()>,
) -> Result<()> {
    for fragment in fragments {
        let logical = match fragment {
            ThinkFragment::Content(text) => {
                content.push_str(&text);
                serde_json::json!({ "type": "token", "text": text })
            }
            ThinkFragment::Reasoning(text) => {
                reasoning_content.push_str(&text);
                serde_json::json!({ "type": "reasoning", "text": text })
            }
        };
        event_callback(&logical)?;
    }
    Ok(())
}

impl ServeRuntime {
    fn ensure_model(
        &mut self,
        model: &str,
        meta: &Mutex<ServeMeta>,
        minimum_max_seq: Option<u64>,
    ) -> Result<hipfire_config::ResolvedConfig> {
        let (tag, entry) = self
            .registry
            .model(model)
            .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry)))
            .unwrap_or((None, None));
        let mut path = find_model_path(&self.paths, &self.registry, model);
        if path.is_none() && entry.is_some() {
            pull_command(
                &self.paths,
                PullArgs {
                    model: model.to_owned(),
                    force: false,
                },
            )?;
            path = entry.map(|entry| self.paths.models.join(&entry.file));
        }
        let path = path.ok_or_else(|| anyhow!("model not found locally: {model}"))?;
        let resolved = resolved_for_model(&self.paths, model, tag.as_deref(), entry)?;
        let must_reload = self.current_path.as_ref() != Some(&path)
            || minimum_max_seq.is_some_and(|minimum| self.current_max_seq < minimum);
        if must_reload {
            let max_tokens = minimum_max_seq
                .map(|minimum| minimum.saturating_sub(1024))
                .unwrap_or(config_u64(&resolved, "generation.max_tokens")?);
            let mut params = load_params(
                &resolved,
                entry,
                &path,
                max_tokens,
                self.kv_override.as_deref(),
                self.kv_backend_override.as_deref(),
            )?;
            if let Some(tp) = self.tp {
                params["tp"] = serde_json::json!(tp);
            }
            let loaded_max_seq = params["max_seq"].as_u64().unwrap_or(0);
            if minimum_max_seq.is_some() {
                eprintln!("[hipfire] bumping load max_seq to {loaded_max_seq} for request budget");
            }
            let loaded = self.engine.load(&path, params)?;
            self.cache_capable = loaded
                .get("cache_capable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.current_path = Some(path);
            self.current_max_seq = loaded_max_seq;
            meta.lock()
                .unwrap_or_else(|error| error.into_inner())
                .current_model = Some(tag.unwrap_or_else(|| model.to_owned()));
        }
        Ok(resolved)
    }
}

fn openai_content_text(content: Option<&serde_json::Value>) -> String {
    match content {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter(|part| part.get("type").and_then(serde_json::Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
            .collect(),
        Some(other) => other.to_string(),
    }
}

fn request_image_base64(messages: Option<&serde_json::Value>) -> Result<Option<String>> {
    let Some(messages) = messages.and_then(serde_json::Value::as_array) else {
        return Ok(None);
    };
    let mut image = None;
    for message in messages {
        let Some(parts) = message.get("content").and_then(serde_json::Value::as_array) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(serde_json::Value::as_str) != Some("image_url") {
                continue;
            }
            let url = part
                .pointer("/image_url/url")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("image_url content part requires image_url.url"))?;
            let payload = ["data:image/png;base64,", "data:image/jpeg;base64,"]
                .into_iter()
                .find_map(|prefix| url.strip_prefix(prefix))
                .ok_or_else(|| {
                    if url.starts_with("data:") {
                        anyhow!("only base64 PNG and JPEG image_url data URIs are supported")
                    } else {
                        anyhow!("remote image_url values are unsupported; send a base64 data URI")
                    }
                })?;
            if payload.is_empty() {
                bail!("image_url data URI has an empty base64 payload");
            }
            if image.replace(payload.to_owned()).is_some() {
                bail!("at most one image_url is supported per request");
            }
        }
    }
    Ok(image)
}

fn strip_inline_thinking(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut visible = String::new();
    let mut remaining = text;
    while let Some(open) = remaining.find(OPEN) {
        visible.push_str(&remaining[..open]);
        let after_open = &remaining[open + OPEN.len()..];
        let Some(close) = after_open.find(CLOSE) else {
            return visible;
        };
        remaining = after_open[close + CLOSE.len()..].trim_start();
    }
    visible.push_str(remaining);
    visible
}

fn inline_thinking(text: &str) -> Option<String> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let after_open = text.split_once(OPEN)?.1;
    let reasoning = after_open.split_once(CLOSE)?.0.trim();
    (!reasoning.is_empty()).then(|| reasoning.to_owned())
}

fn normalize_openai_tool_call(call: &serde_json::Value) -> serde_json::Value {
    let function = call.get("function").unwrap_or(call);
    let name = function
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let arguments = match function.get("arguments") {
        Some(serde_json::Value::String(raw)) => {
            serde_json::from_str(raw).unwrap_or_else(|_| serde_json::json!({ "_raw": raw }))
        }
        Some(value) => value.clone(),
        None => serde_json::json!({}),
    };
    serde_json::json!({ "name": name, "arguments": arguments })
}

fn normalize_openai_messages(messages: Option<&serde_json::Value>) -> serde_json::Value {
    let Some(messages) = messages.and_then(serde_json::Value::as_array) else {
        return serde_json::json!([]);
    };
    let normalized = messages
        .iter()
        .filter_map(|message| {
            let role = match message.get("role").and_then(serde_json::Value::as_str)? {
                "developer" => "system",
                "toolResult" | "tool_result" => "tool",
                role @ ("system" | "user" | "assistant" | "tool") => role,
                _ => return None,
            };
            let raw_content = openai_content_text(message.get("content"));
            let mut entry = serde_json::json!({
                "role": role,
                "content": if role == "assistant" {
                    strip_inline_thinking(&raw_content)
                } else {
                    raw_content.clone()
                },
            });
            if role == "assistant" {
                let reasoning = message
                    .get("reasoning")
                    .and_then(serde_json::Value::as_str)
                    .filter(|text| !text.is_empty())
                    .or_else(|| {
                        message
                            .get("reasoning_content")
                            .and_then(serde_json::Value::as_str)
                            .filter(|text| !text.is_empty())
                    })
                    .map(str::to_owned)
                    .or_else(|| inline_thinking(&raw_content));
                if let Some(reasoning) = reasoning {
                    entry["tool_plan"] = serde_json::Value::String(reasoning);
                }
                if let Some(calls) = message
                    .get("tool_calls")
                    .and_then(serde_json::Value::as_array)
                    .filter(|calls| !calls.is_empty())
                {
                    entry["tool_calls"] = serde_json::Value::Array(
                        calls.iter().map(normalize_openai_tool_call).collect(),
                    );
                }
            } else if role == "tool" {
                if let Some(tool_call_id) = message
                    .get("tool_call_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|id| !id.is_empty())
                {
                    entry["tool_call_id"] = serde_json::Value::String(tool_call_id.to_owned());
                }
            }
            Some(entry)
        })
        .collect();
    serde_json::Value::Array(normalized)
}

fn inject_default_system_message(messages: &mut serde_json::Value, system: Option<&str>) {
    let Some(system) = system.filter(|value| !value.is_empty()) else {
        return;
    };
    let Some(messages) = messages.as_array_mut() else {
        return;
    };
    if messages
        .iter()
        .any(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("system"))
    {
        return;
    }
    messages.insert(
        0,
        serde_json::json!({ "role": "system", "content": system }),
    );
}

fn last_user_prompt(messages: &serde_json::Value) -> Option<String> {
    messages
        .as_array()?
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

fn completion_json(completion: &Completion) -> serde_json::Value {
    let tool_calls = openai_tool_calls(&completion.tool_calls);
    let finish_reason = if tool_calls.is_empty() {
        completion
            .done
            .get("finish_reason")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("stop")
    } else {
        "tool_calls"
    };
    let visible_content =
        if completion.preserve_thinking && !completion.reasoning_content.is_empty() {
            format!(
                "<think>{}</think>\n{}",
                completion.reasoning_content, completion.content
            )
        } else {
            completion.content.clone()
        };
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": visible_content,
    });
    if !completion.preserve_thinking && !completion.reasoning_content.is_empty() {
        message["reasoning_content"] =
            serde_json::Value::String(completion.reasoning_content.clone());
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = serde_json::Value::Array(tool_calls);
    }
    serde_json::json!({
        "id": completion.id,
        "object": "chat.completion",
        "created": completion.created,
        "model": completion.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": completion_usage(completion),
        "timings": completion_timings(completion),
        "hipfire": {
            "tok_s": completion.done.get("tok_s"),
            "prefill_tok_s": completion.done.get("prefill_tok_s"),
            "decode_tok_s": completion.done.get("decode_tok_s"),
        }
    })
}

fn completion_usage(completion: &Completion) -> serde_json::Value {
    let cached_tokens = completion
        .done
        .get("cached_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let prompt_tokens = completion
        .done
        .get("prompt_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| {
            completion
                .done
                .get("prefill_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                .saturating_add(cached_tokens)
        });
    let completion_tokens = completion
        .done
        .get("tokens")
        .or_else(|| completion.done.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": { "cached_tokens": cached_tokens },
    })
}

fn completion_timings(completion: &Completion) -> serde_json::Value {
    let done = &completion.done;
    serde_json::json!({
        "ttft_ms": done.get("ttft_ms"),
        "prefill_ms": done.get("prefill_ms"),
        "prefill_tok_s": done.get("prefill_tok_s"),
        "decode_tok_s": done.get("decode_tok_s").or_else(|| done.get("tok_s")),
        "tau": done.get("tau"),
        "cycles": done.get("cycles"),
        "dflash": done.get("dflash"),
        "mtp": done.get("mtp"),
    })
}

fn openai_tool_calls(calls: &[serde_json::Value]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .enumerate()
        .filter_map(|(index, call)| {
            let name = call.get("name").and_then(serde_json::Value::as_str)?;
            let arguments = call
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            Some(serde_json::json!({
                "id": format!("call_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".into()),
                }
            }))
        })
        .collect()
}

fn openai_tool_call_delta(calls: &serde_json::Value) -> serde_json::Value {
    let calls = calls.as_array().cloned().unwrap_or_default();
    serde_json::json!({
        "tool_calls": openai_tool_calls(&calls)
            .into_iter()
            .enumerate()
            .map(|(index, mut call)| {
                call["index"] = serde_json::json!(index);
                call
            })
            .collect::<Vec<_>>()
    })
}

fn request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "chatcmpl-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

fn sse_data(value: &serde_json::Value) -> Vec<u8> {
    format!("data: {}\n\n", value).into_bytes()
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("static HTTP header")
}

fn json_response(value: serde_json::Value, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    let bytes = serde_json::to_vec(&value).expect("JSON value serializes");
    Response::new(
        StatusCode(status),
        vec![
            header("Content-Type", "application/json"),
            header("Access-Control-Allow-Origin", "*"),
        ],
        std::io::Cursor::new(bytes.clone()),
        Some(bytes.len()),
        None,
    )
}

fn openai_error(message: &str, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    let error_type = if (400..500).contains(&status) {
        "invalid_request_error"
    } else {
        "server_error"
    };
    json_response(
        serde_json::json!({
            "error": { "message": message, "type": error_type }
        }),
        status,
    )
}

fn admission_error_response(error: &AdmissionError) -> Response<std::io::Cursor<Vec<u8>>> {
    openai_error(&error.message, 503).with_header(header(
        "Retry-After",
        &error.retry_after_seconds.to_string(),
    ))
}

struct ChannelReader {
    receiver: mpsc::Receiver<Vec<u8>>,
    current: std::io::Cursor<Vec<u8>>,
}

impl ChannelReader {
    fn new(receiver: mpsc::Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            current: std::io::Cursor::new(Vec::new()),
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let read = self.current.read(output)?;
            if read > 0 {
                return Ok(read);
            }
            match self.receiver.recv() {
                Ok(bytes) => self.current = std::io::Cursor::new(bytes),
                Err(_) => return Ok(0),
            }
        }
    }
}

fn serve_instance_token() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut digest = Sha256::new();
    digest.update(std::process::id().to_le_bytes());
    digest.update(now.to_le_bytes());
    format!("{:x}", digest.finalize())
}

fn proc_start_time(pid: u32) -> Option<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = stat.rsplit_once(") ")?.1;
    after_comm.split_whitespace().nth(19)?.parse().ok()
}

fn pid_owns_listen_port(pid: u32, port: u16) -> Option<bool> {
    let mut listen_inodes = BTreeSet::new();
    let port_hex = format!("{port:04X}");
    let mut read_any = false;
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(raw) = fs::read_to_string(table) else {
            continue;
        };
        read_any = true;
        for line in raw.lines().skip(1) {
            let columns = line.split_whitespace().collect::<Vec<_>>();
            if columns.len() < 10 || columns[3] != "0A" {
                continue;
            }
            let Some((_, local_port)) = columns[1].rsplit_once(':') else {
                continue;
            };
            if local_port.eq_ignore_ascii_case(&port_hex) {
                listen_inodes.insert(columns[9].to_owned());
            }
        }
    }
    if !read_any {
        return None;
    }
    if listen_inodes.is_empty() {
        return Some(false);
    }
    let entries = fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
    for entry in entries.flatten() {
        let Ok(target) = fs::read_link(entry.path()) else {
            continue;
        };
        let target = target.to_string_lossy();
        if let Some(inode) = target
            .strip_prefix("socket:[")
            .and_then(|value| value.strip_suffix(']'))
        {
            if listen_inodes.contains(inode) {
                return Some(true);
            }
        }
    }
    Some(false)
}

fn validate_serve_pid(record: &ServePidRecord, host: &str, fallback_port: u16) -> Result<()> {
    let proc_dir = PathBuf::from(format!("/proc/{}", record.pid));
    if !proc_dir.is_dir() {
        bail!("tracked serve PID {} is no longer alive", record.pid);
    }
    let cmdline = fs::read(proc_dir.join("cmdline")).unwrap_or_default();
    let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
    if !cmdline.contains("hipfire") || !cmdline.contains("serve") {
        bail!("PID {} is not a hipfire serve process", record.pid);
    }
    if let Some(expected) = record.start_time {
        if proc_start_time(record.pid) != Some(expected) {
            bail!("PID {} was reused after serve.pid was written", record.pid);
        }
    }

    let port = record.port.unwrap_or(fallback_port);
    let owns_port = pid_owns_listen_port(record.pid, port);
    if owns_port == Some(false) {
        bail!(
            "PID {} does not own the tracked serve port {port}",
            record.pid
        );
    }
    let health_matches = record.token.as_deref().is_some_and(|expected| {
        http_get_json(host, port, "/health").is_some_and(|health| {
            health.get("pid").and_then(serde_json::Value::as_u64) == Some(record.pid as u64)
                && health.get("token").and_then(serde_json::Value::as_str) == Some(expected)
        })
    });
    if owns_port == Some(true) || health_matches || record.legacy && owns_port.is_none() {
        Ok(())
    } else {
        bail!(
            "could not prove ownership of PID {} with port or health token",
            record.pid
        )
    }
}

fn stop_command(paths: &Paths, args: StopArgs) -> Result<()> {
    let pid_path = paths.root.join("serve.pid");
    match fs::read_to_string(&pid_path) {
        Ok(raw) => {
            let record = parse_pid_record(&raw)
                .ok_or_else(|| anyhow!("invalid serve.pid; refusing to signal"))?;
            let resolved = resolved_global(paths, true)
                .ok()
                .map(|(_, resolved)| resolved);
            let host = resolved
                .as_ref()
                .and_then(|resolved| config_string(resolved, "serve.host").ok())
                .unwrap_or_else(|| "127.0.0.1".into());
            let fallback_port = args
                .port
                .or_else(|| {
                    resolved.as_ref().and_then(|resolved| {
                        config_u64(resolved, "serve.port")
                            .ok()
                            .and_then(|port| u16::try_from(port).ok())
                    })
                })
                .unwrap_or(11435);
            if let Err(error) = validate_serve_pid(&record, probe_host(&host), fallback_port) {
                fs::remove_file(&pid_path)?;
                if !args.force {
                    bail!("{error}; removed stale pidfile without signaling");
                }
                eprintln!(
                    "warning: {error}; refusing direct PID signal and continuing forced reap"
                );
            } else {
                let status = Command::new("kill")
                    .arg("-TERM")
                    .arg(record.pid.to_string())
                    .status()
                    .context("failed to invoke kill")?;
                if !status.success() {
                    bail!("failed to stop native serve PID {}", record.pid);
                }
                for _ in 0..50 {
                    if !Path::new(&format!("/proc/{}", record.pid)).exists() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                let _ = fs::remove_file(&pid_path);
                println!("hipfire serve stopped (PID {})", record.pid);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            println!("hipfire serve is not running");
        }
        Err(error) => return Err(error).context("failed to read serve.pid"),
    }
    if args.force || args.all {
        let (_, resolved) = resolved_global(paths, true)?;
        let port = args
            .port
            .unwrap_or(config_u64(&resolved, "serve.port")? as u16);
        let _ = Command::new("pkill").args(["-x", "daemon"]).status();
        if args.all {
            let _ = Command::new("pkill")
                .args(["-f", "target/release/hipfire-quantize"])
                .status();
        }
        let _ = Command::new("fuser")
            .args(["-k", &format!("{port}/tcp")])
            .status();
        println!("reaped orphan daemon processes and freed port {port}");
    }
    Ok(())
}

fn parse_pid_record(raw: &str) -> Option<ServePidRecord> {
    if let Ok(pid) = raw.trim().parse() {
        return Some(ServePidRecord {
            pid,
            start_time: None,
            port: None,
            token: None,
            legacy: true,
        });
    }
    let mut record = serde_json::from_str::<ServePidRecord>(raw).ok()?;
    record.legacy = record.start_time.is_none() && record.port.is_none() && record.token.is_none();
    Some(record)
}

fn resolved_for_model(
    paths: &Paths,
    model_name: &str,
    tag: Option<&str>,
    entry: Option<&ModelEntry>,
) -> Result<hipfire_config::ResolvedConfig> {
    let loaded = load_global(&paths.config)?;
    let mut layers = Vec::new();
    if let (Some(tag), Some(settings)) = (
        tag,
        entry.and_then(|entry| entry.recommended_settings.as_ref()),
    ) {
        layers.push(NamedLayer {
            source: ConfigSource::RegistryModel {
                tag: tag.to_owned(),
                revision: "v1".into(),
            },
            layer: settings
                .config_layer()
                .map_err(|error| anyhow!("invalid registry recommendations: {error}"))?,
        });
    }
    layers.push(NamedLayer {
        source: ConfigSource::GlobalUser { path: loaded.path },
        layer: loaded.layer,
    });
    let catalog = load_catalog(&paths.config)?;
    let model_override = catalog
        .catalog
        .model(model_name)
        .or_else(|| tag.and_then(|tag| catalog.catalog.model(tag)))
        .or_else(|| entry.and_then(|entry| catalog.catalog.model(entry.file.as_str())));
    if let Some((model_id, model)) = model_override {
        if !model.overrides.values.is_empty() {
            layers.push(NamedLayer {
                source: ConfigSource::ModelUser {
                    model: model_id.to_owned(),
                    path: catalog.path,
                },
                layer: model.overrides.clone(),
            });
        }
    }
    let env_layer = load_env_layer()?;
    if !env_layer.values.is_empty() {
        layers.push(NamedLayer {
            source: ConfigSource::LegacyEnv {
                name: "HIPFIRE_*".into(),
            },
            layer: env_layer,
        });
    }
    Ok(resolve(layers)?)
}

fn find_model_path(paths: &Paths, registry: &RegistryV1, model: &str) -> Option<PathBuf> {
    let direct = PathBuf::from(model);
    if direct.is_file() {
        return fs::canonicalize(direct).ok();
    }
    if let Ok(catalog) = load_catalog(&paths.config) {
        if let Some((_, record)) = catalog.catalog.model(model) {
            if let Some(path) = record.path.as_ref().filter(|path| path.is_file()) {
                return fs::canonicalize(path).ok().or_else(|| Some(path.clone()));
            }
        }
    }
    if let Some((_, entry)) = registry.model(model) {
        let path = paths.models.join(&entry.file);
        if path.is_file() {
            return Some(path);
        }
    }
    let path = paths.models.join(model);
    if path.is_file() {
        return Some(path);
    }
    let search = model.replace(':', "-").to_ascii_lowercase();
    let explicit_quant = MODEL_SUFFIXES.iter().any(|suffix| search.ends_with(suffix));
    let mut candidates = local_model_paths(paths)
        .ok()?
        .into_iter()
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|file| file.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            name == search || name.contains(&search)
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|path| {
        let name = path
            .file_name()
            .and_then(|file| file.to_str())
            .unwrap_or_default();
        if explicit_quant || name.ends_with(".mq4r") {
            0
        } else if name.ends_with(".mq4") {
            1
        } else if name.ends_with(".hf4") || name.ends_with(".hfq") {
            2
        } else {
            3
        }
    });
    candidates.into_iter().next()
}

fn load_params(
    resolved: &hipfire_config::ResolvedConfig,
    entry: Option<&ModelEntry>,
    model_path: &Path,
    max_tokens: u64,
    kv_override: Option<&str>,
    kv_backend_override: Option<&str>,
) -> Result<serde_json::Value> {
    let configured_max_seq = config_u64(resolved, "memory.max_seq")?;
    let max_seq = configured_max_seq.max(max_tokens.saturating_add(1024));
    let configured_kv = config_string(resolved, "memory.kv_cache")?;
    let kv_mode = kv_override
        .map(str::to_owned)
        .or_else(|| (configured_kv != "auto").then_some(configured_kv))
        .or_else(|| entry.and_then(|entry| entry.default_kv_mode.clone()))
        .unwrap_or_else(|| "q8".into());
    // Validate a one-shot override through the shared schema.
    field("memory.kv_cache")
        .expect("schema field")
        .parse_cli(&kv_mode)?;
    let kv_backend = kv_backend_override
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "contiguous".into())
        .to_ascii_lowercase();
    if !matches!(kv_backend.as_str(), "contiguous" | "vmm") {
        bail!("--kv-backend must be contiguous or vmm");
    }
    let mut cask_sidecar = config_string(resolved, "memory.cask.sidecar")?;
    if cask_sidecar.is_empty() && config_bool(resolved, "memory.cask.auto_attach")? {
        if let Some(sidecar) = entry.and_then(|entry| entry.triattn.as_ref()) {
            let candidate = model_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&sidecar.file);
            if candidate.is_file() {
                cask_sidecar = candidate.display().to_string();
            }
        }
    }
    let mut params = serde_json::json!({
        "max_seq": max_seq,
        "kv_mode": kv_mode,
        "kv_backend": kv_backend,
        "kv_adaptive": config_string(resolved, "memory.kv_adaptive")?,
        "dflash_mode": config_string(resolved, "speculation.dflash")?,
        "dflash_adaptive_b": config_bool(resolved, "speculation.dflash_adaptive_b")?,
        "mtp_mode": config_string(resolved, "speculation.mtp")?,
        "mtp_k": config_u64(resolved, "speculation.mtp_k")?,
        "ngram_draft": matches!(config_string(resolved, "speculation.ngram")?.as_str(), "on" | "auto"),
        "ngram_k": config_u64(resolved, "speculation.ngram_k")?,
        "ngram_min_count": config_u64(resolved, "speculation.ngram_min_count")?,
        "ddtree_budget": config_u64(resolved, "speculation.ddtree_budget")?,
        "ddtree_topk": config_u64(resolved, "speculation.ddtree_topk")?,
        "cask_sidecar": cask_sidecar,
        "cask": config_bool(resolved, "memory.cask.enabled")?,
        "cask_budget": config_u64(resolved, "memory.cask.budget")?,
        "cask_beta": config_u64(resolved, "memory.cask.beta")?,
        "cask_core_frac": config_f64(resolved, "memory.cask.core_fraction")?,
        "cask_fold_m": config_u64(resolved, "memory.cask.fold")?,
        "prefill_compression": config_string(resolved, "speculation.prefill.mode")?,
        "prefill_threshold": config_u64(resolved, "speculation.prefill.threshold")?,
        "prefill_keep_ratio": config_f64(resolved, "speculation.prefill.keep_ratio")?,
        "prefill_alpha": config_f64(resolved, "speculation.prefill.alpha")?,
        "prefill_min_keep": config_u64(resolved, "speculation.prefill.min_keep")?,
        "prefill_sink": config_u64(resolved, "speculation.prefill.sink")?,
        "prefill_recent": config_u64(resolved, "speculation.prefill.recent")?,
        "prefill_block": config_u64(resolved, "speculation.prefill.block")?,
        "prefill_drafter": config_string(resolved, "speculation.prefill.drafter")?,
        "prefill_drafter_device": config_i64(resolved, "speculation.prefill.drafter_device")?,
        "prefill_sparse_threshold": config_u64(resolved, "speculation.prefill.sparse_threshold")?,
        "speculation": config_string(resolved, "speculation.mode")?,
    });
    let selector = config_string(resolved, "speculation.mode")?;
    apply_speculation_selector(&mut params, &selector)?;
    project_dflash_draft(&mut params);
    Ok(params)
}

/// Project inherited `HIPFIRE_DFLASH_DRAFT` after the effective speculation selector.
///
/// Call only once final `dflash_mode` is known. Config-off must not carry a draft;
/// a later CLI selector (e.g. `run --spec dflash`) can opt back in here.
fn project_dflash_draft(params: &mut serde_json::Value) {
    if params["dflash_mode"].as_str() == Some("off") {
        if let Some(obj) = params.as_object_mut() {
            obj.remove("draft");
        }
        return;
    }
    if let Ok(draft) = env::var("HIPFIRE_DFLASH_DRAFT") {
        if !draft.is_empty() {
            params["draft"] = serde_json::json!(draft);
        }
    }
}

fn apply_speculation_selector(params: &mut serde_json::Value, selector: &str) -> Result<()> {
    match selector {
        "off" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "dflash" => {
            params["dflash_mode"] = serde_json::json!("on");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "mtp" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("on");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "ngram" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(true);
            params["dspark_mode"] = serde_json::json!("off");
        }
        "dspark" => {
            params["dflash_mode"] = serde_json::json!("off");
            params["mtp_mode"] = serde_json::json!("off");
            params["ngram_draft"] = serde_json::json!(false);
            params["dspark_mode"] = serde_json::json!("on");
        }
        "auto" => {
            params["dspark_mode"] = serde_json::json!("auto");
        }
        other => bail!("unknown speculation selector '{other}'"),
    }
    Ok(())
}

fn apply_reasoning_request(
    resolved: &hipfire_config::ResolvedConfig,
    request: &mut serde_json::Value,
) -> Result<()> {
    if config_string(resolved, "reasoning.mode")? == "off" {
        request["max_think_tokens"] = serde_json::json!(1);
        request["assistant_prefix"] = serde_json::json!("closed_think");
        return Ok(());
    }
    let explicit = resolved
        .get("reasoning.max_tokens")
        .map(|value| &value.value)
        .filter(|value| !matches!(value, hipfire_config::ConfigValue::Null));
    let max_think = if let Some(value) = explicit {
        match value {
            hipfire_config::ConfigValue::Integer(value) => *value as u64,
            _ => bail!("reasoning.max_tokens resolved to a non-integer"),
        }
    } else {
        match config_string(resolved, "reasoning.budget")?.as_str() {
            // 1 = the engine's "no thinking" sentinel (daemon: `enable_thinking:
            // max_think_tokens != 1`), matching what the OpenAI
            // enable_thinking=false / reasoning_effort="none" paths send. Pair it
            // with the closed-think assistant prefix so the turn starts in answer
            // mode instead of relying on the template alone.
            "off" => {
                request["max_think_tokens"] = serde_json::json!(1);
                request["assistant_prefix"] = serde_json::json!("closed_think");
                request["reasoning_effort"] = serde_json::json!("none");
                return Ok(());
            }
            "low" => 512,
            "med" => 2048,
            "high" => 8192,
            "xhigh" => 24576,
            "max" => 32768,
            "uncapped" => 0,
            value => bail!("unknown reasoning budget {value}"),
        }
    };
    request["max_think_tokens"] = serde_json::json!(max_think);
    Ok(())
}

fn apply_http_reasoning_request(
    body: &serde_json::Value,
    resolved: &hipfire_config::ResolvedConfig,
    request: &mut serde_json::Value,
) -> Result<()> {
    let thinking_disabled = body
        .pointer("/chat_template_kwargs/enable_thinking")
        .and_then(serde_json::Value::as_bool)
        == Some(false);
    let effort = body
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            body.pointer("/reasoning/effort")
                .and_then(serde_json::Value::as_str)
        });
    if thinking_disabled || effort == Some("none") {
        request["max_think_tokens"] = serde_json::json!(1);
        request["assistant_prefix"] = serde_json::json!("closed_think");
        request["reasoning_effort"] = serde_json::json!("none");
        return Ok(());
    }
    if let Some(effort) = effort {
        let max_think = match effort {
            "minimal" => 64,
            "low" => 256,
            "medium" | "med" => 1024,
            "high" => 4096,
            "xhigh" | "max" | "uncapped" => 0,
            other => bail!("unknown reasoning effort '{other}'"),
        };
        request["max_think_tokens"] = serde_json::json!(max_think);
        request["reasoning_effort"] = serde_json::json!(effort);
        return Ok(());
    }
    apply_reasoning_request(resolved, request)
}

fn config_value<'a>(
    resolved: &'a hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<&'a hipfire_config::ConfigValue> {
    resolved
        .get(key)
        .map(|value| &value.value)
        .ok_or_else(|| anyhow!("missing resolved configuration key {key}"))
}

fn config_string(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<String> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::String(value) => Ok(value.clone()),
        value => bail!("{key} resolved as {}, expected string", value.kind()),
    }
}

fn config_bool(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<bool> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Bool(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected bool", value.kind()),
    }
}

fn config_i64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<i64> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Integer(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected integer", value.kind()),
    }
}

fn config_u64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<u64> {
    let value = config_i64(resolved, key)?;
    u64::try_from(value).map_err(|_| anyhow!("{key} cannot be negative"))
}

fn config_f64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<f64> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Integer(value) => Ok(*value as f64),
        hipfire_config::ConfigValue::Float(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected number", value.kind()),
    }
}

fn launch_tui(paths: &Paths, arguments: &[String]) -> Result<()> {
    let executable = env::var_os("HIPFIRE_TUI_BIN")
        .map(PathBuf::from)
        .or_else(|| {
            let installed = paths.root.join("bin/hipfire-tui");
            installed.is_file().then_some(installed)
        })
        .or_else(|| {
            let workspace =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/release/hipfire-tui");
            workspace.is_file().then_some(workspace)
        })
        .or_else(|| find_on_path("hipfire-tui"))
        .ok_or_else(|| {
            anyhow!(
                "hipfire-tui is not installed; build it with `cargo build --release -p hipfire-tui`"
            )
        })?;
    let status = Command::new(&executable)
        .args(arguments)
        .status()
        .with_context(|| format!("failed to launch {}", executable.display()))?;
    if status.success() {
        Ok(())
    } else {
        bail!("hipfire-tui exited with {status}")
    }
}

#[derive(Debug, Serialize)]
struct ProcessRecord {
    pid: u32,
    rss_mb: u64,
    command: String,
}

fn scan_auxiliary_processes() -> (Vec<ProcessRecord>, Vec<ProcessRecord>) {
    let mut quantize = Vec::new();
    let mut uploads = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return (quantize, uploads);
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == std::process::id() {
            continue;
        }
        let Ok(raw) = fs::read(entry.path().join("cmdline")) else {
            continue;
        };
        let command = String::from_utf8_lossy(&raw)
            .replace('\0', " ")
            .trim()
            .to_owned();
        if command.is_empty() {
            continue;
        }
        let rss_mb = fs::read_to_string(entry.path().join("status"))
            .ok()
            .and_then(|status| {
                status.lines().find_map(|line| {
                    line.strip_prefix("VmRSS:")?
                        .split_whitespace()
                        .next()?
                        .parse::<u64>()
                        .ok()
                })
            })
            .unwrap_or(0)
            / 1024;
        let record = ProcessRecord {
            pid,
            rss_mb,
            command,
        };
        if record.command.contains("hf upload") {
            uploads.push(record);
        } else if record.command.contains("hipfire-quantize")
            || record.command.contains("hipfire quantize")
        {
            quantize.push(record);
        }
    }
    (quantize, uploads)
}

fn ps_command(paths: &Paths, output: OutputArgs) -> Result<()> {
    let (_, resolved) = resolved_global(paths, true)?;
    let host = config_string(&resolved, "serve.host")?;
    let port = config_u64(&resolved, "serve.port")? as u16;
    let pid_path = paths.root.join("serve.pid");
    let pid_record = fs::read_to_string(&pid_path)
        .ok()
        .and_then(|raw| parse_pid_record(&raw));
    let pid = pid_record.as_ref().map(|record| record.pid);
    let alive = pid.is_some_and(|pid| Path::new(&format!("/proc/{pid}")).exists());
    let health = http_get_json(&host, port, "/health");
    let stats = http_get_json(&host, port, "/stats");
    let (quantize, uploads) = scan_auxiliary_processes();
    let report = serde_json::json!({
        "running": health.is_some(),
        "pid": pid,
        "pid_record": pid_record,
        "pid_alive": alive,
        "endpoint": service_url(&host, port, ""),
        "health": health,
        "stats": stats,
        "quantize": quantize,
        "uploads": uploads,
    });
    if output.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report["running"].as_bool() == Some(true) {
        println!(
            "hipfire serve is online at {}",
            report["endpoint"].as_str().unwrap()
        );
        println!(
            "  pid:       {}",
            pid.map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".into())
        );
        println!(
            "  model:     {}",
            report
                .pointer("/health/model")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("idle")
        );
        println!(
            "  requests:  {}",
            report
                .pointer("/stats/requests_served")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
        );
        if let Some(tok_s) = report
            .pointer("/stats/recent_tok_s")
            .and_then(serde_json::Value::as_f64)
        {
            println!("  recent:    {tok_s:.2} tok/s");
        }
    } else if alive {
        println!(
            "hipfire serve PID {} is alive but HTTP is not ready",
            pid.unwrap()
        );
    } else {
        println!("hipfire serve is not running");
    }
    for (label, records) in [("quantize", &quantize), ("HF upload", &uploads)] {
        for process in records {
            println!(
                "{label}: PID {}  {} MB  {}",
                process.pid, process.rss_mb, process.command
            );
        }
    }
    Ok(())
}

fn http_get_json(host: &str, port: u16, path: &str) -> Option<serde_json::Value> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(1)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut response = agent.get(&service_url(host, port, path)).call().ok()?;
    if !response.status().is_success() {
        return None;
    }
    let text = response.body_mut().read_to_string().ok()?;
    serde_json::from_str(&text).ok()
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SampleStats {
    median: f64,
    mean: f64,
    min: f64,
    max: f64,
    stdev: f64,
}

fn sample_stats(values: &[f64]) -> Option<SampleStats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    Some(SampleStats {
        median,
        mean,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        stdev: variance.sqrt(),
    })
}

fn bench_command(paths: &Paths, args: BenchArgs) -> Result<()> {
    if args.runs == 0 {
        bail!("--runs must be positive");
    }
    if args.exp && (args.matrix || args.redline) {
        bail!("--exp cannot be combined with matrix or Redline options");
    }
    if args.exp && args.json {
        bail!("--json is not supported with --exp");
    }
    for (name, values) in [
        ("--pp", &args.pp),
        ("--ctx", &args.ctx),
        ("--sustained-ctx", &args.sustained_ctx),
    ] {
        if values.is_empty() || values.contains(&0) {
            bail!("{name} values must be positive");
        }
    }
    if args.tg == 0 || args.sustained_tg == Some(0) {
        bail!("decode lengths must be positive");
    }
    if let Some(mode) = args.kv_mode.as_deref() {
        if !matches!(mode, "q8" | "fwht2" | "fwht3" | "fwht4") {
            bail!("--kv-mode must be q8, fwht2, fwht3, or fwht4");
        }
    }

    if args.exp {
        return bench_experimental(paths, &args);
    }
    let (mut engine, loaded, pre_diag, post_diag) = open_bench_engine(paths, &args, None)?;
    let prompt = if args.prompt.is_empty() {
        "Explain the theory of general relativity in simple terms.".to_owned()
    } else {
        args.prompt.join(" ")
    };
    eprintln!("hipfire bench");
    eprintln!("  model:  {}", args.model);
    eprintln!(
        "  arch:   {}",
        loaded
            .get("arch")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
    );
    eprintln!(
        "  gpu:    {}",
        post_diag
            .get("arch")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
    );
    eprintln!("  runs:   {}", args.runs);
    if args.matrix || args.redline {
        bench_matrix(&mut engine, &args, &loaded, &post_diag)
    } else {
        let _ = bench_generate(&mut engine, "Hello", 16)?;
        let mut decode = Vec::new();
        let mut prefill = Vec::new();
        let mut wall = Vec::new();
        let mut ttft = Vec::new();
        for _ in 0..args.runs {
            let done = bench_generate(&mut engine, &prompt, 128)?;
            if let Some(value) = done.get("decode_tok_s").and_then(serde_json::Value::as_f64) {
                decode.push(value);
            }
            if let Some(value) = done
                .get("prefill_tok_s")
                .and_then(serde_json::Value::as_f64)
            {
                prefill.push(value);
            }
            if let Some(value) = done.get("tok_s").and_then(serde_json::Value::as_f64) {
                wall.push(value);
            }
            if let Some(value) = done.get("ttft_ms").and_then(serde_json::Value::as_f64) {
                ttft.push(value);
            }
            eprint!(".");
            std::io::stderr().flush()?;
        }
        eprintln!();
        let report = serde_json::json!({
            "protocol": "native-generate-v1",
            "model": args.model,
            "loaded": loaded,
            "gpu": post_diag,
            "vram_free_before_mb": pre_diag.get("vram_free_mb"),
            "decode_tok_s": sample_stats(&decode),
            "prefill_tok_s": sample_stats(&prefill),
            "wall_tok_s": sample_stats(&wall),
            "ttft_ms": sample_stats(&ttft),
            "samples": { "decode": decode, "prefill": prefill, "wall": wall, "ttft_ms": ttft },
        });
        if args.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_sample_row("decode", sample_stats(&decode));
            print_sample_row("prefill", sample_stats(&prefill));
            print_sample_row("wall", sample_stats(&wall));
            print_sample_row("ttft ms", sample_stats(&ttft));
        }
        Ok(())
    }
}

fn open_bench_engine(
    paths: &Paths,
    args: &BenchArgs,
    rdna2_variant: Option<u8>,
) -> Result<(
    Engine,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
)> {
    let registry = load_registry(&paths.registry).registry;
    let (tag, entry) = registry
        .model(&args.model)
        .map(|(tag, entry)| (Some(tag.to_owned()), Some(entry.clone())))
        .unwrap_or((None, None));
    let mut path = find_model_path(paths, &registry, &args.model);
    if path.is_none() && entry.is_some() {
        pull_command(
            paths,
            PullArgs {
                model: args.model.clone(),
                force: false,
            },
        )?;
        path = entry.as_ref().map(|entry| paths.models.join(&entry.file));
    }
    let path = path.ok_or_else(|| anyhow!("model not found: {}", args.model))?;
    let resolved = resolved_for_model(paths, &args.model, tag.as_deref(), entry.as_ref())?;
    let daemon = find_daemon(paths).ok_or_else(|| anyhow!("daemon binary not found"))?;
    let environment = BTreeMap::new();
    let mut process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    if args.redline {
        process_config.values.set_cli("replay.backend", "redline")?;
        process_config.values.set_cli("replay.transport", "pm4")?;
        process_config
            .values
            .set_cli("experimental.graph.ar", "true")?;
        process_config
            .values
            .set_cli("experimental.graph.forward", "true")?;
    }
    if let Some(variant) = rdna2_variant {
        process_config
            .values
            .set_cli("diagnostic.kernel.rdna2_variant", &variant.to_string())?;
    }
    let mut engine = Engine::spawn_configured(daemon, &environment, &process_config)?;
    engine.ping()?;
    let pre_diag = engine.request(&serde_json::json!({ "type": "diag" }))?;
    let longest_prefill = args.pp.iter().copied().max().unwrap_or(0) as u64;
    let longest_decode = args
        .ctx
        .iter()
        .chain(args.sustained_ctx.iter())
        .copied()
        .max()
        .unwrap_or(0) as u64
        + args.sustained_tg.unwrap_or(args.tg) as u64;
    let max_tokens = config_u64(&resolved, "generation.max_tokens")?;
    let mut params = load_params(
        &resolved,
        entry.as_ref(),
        &path,
        max_tokens,
        args.kv_mode.as_deref(),
        args.kv_backend.as_deref(),
    )?;
    if args.matrix || args.redline {
        let requested = longest_prefill.max(longest_decode).saturating_add(32);
        let configured = params["max_seq"].as_u64().unwrap_or(0);
        params["max_seq"] = serde_json::json!(configured.max(requested));
    }
    let loaded = engine.load(&path, params)?;
    let post_diag = engine.request(&serde_json::json!({ "type": "diag" }))?;
    Ok((engine, loaded, pre_diag, post_diag))
}

fn bench_generate(engine: &mut Engine, prompt: &str, max_tokens: u64) -> Result<serde_json::Value> {
    Ok(engine.generate(
        &serde_json::json!({
            "type": "generate",
            "id": request_id(),
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.1,
            "max_tokens": max_tokens,
        }),
        |_| Ok(()),
    )?)
}

fn bench_probe(
    engine: &mut Engine,
    message: serde_json::Value,
    expected: &str,
) -> Result<serde_json::Value> {
    let response = engine.request(&message)?;
    match response.get("type").and_then(serde_json::Value::as_str) {
        Some(actual) if actual == expected => Ok(response),
        Some("error") => bail!(
            "{}",
            response
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("benchmark probe failed")
        ),
        other => bail!(
            "expected {expected}, received {}",
            other.unwrap_or("missing type")
        ),
    }
}

fn bench_matrix(
    engine: &mut Engine,
    args: &BenchArgs,
    loaded: &serde_json::Value,
    diag: &serde_json::Value,
) -> Result<()> {
    for size in &args.pp {
        let _ = bench_probe(
            engine,
            serde_json::json!({ "type": "bench_prefill", "tokens": size }),
            "prefill_result",
        )?;
    }
    let warm_context = args.ctx[0];
    for _ in 0..args.warmups {
        let _ = bench_probe(
            engine,
            serde_json::json!({ "type": "bench_decode", "context_tokens": warm_context, "iterations": args.tg }),
            "decode_result",
        )?;
    }
    let mut pp_rows = Vec::new();
    for size in &args.pp {
        let mut samples = Vec::new();
        for _ in 0..args.runs {
            let result = bench_probe(
                engine,
                serde_json::json!({ "type": "bench_prefill", "tokens": size }),
                "prefill_result",
            )?;
            samples.push(
                result
                    .get("tok_s")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
            );
        }
        eprintln!(
            "  pp{size}: {:.2} tok/s median",
            sample_stats(&samples).unwrap().median
        );
        pp_rows.push(serde_json::json!({ "tokens": size, "stats": sample_stats(&samples), "samples": samples }));
    }
    let mut decode_rows = Vec::new();
    for context in &args.ctx {
        let _ = bench_probe(
            engine,
            serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": args.tg }),
            "decode_result",
        )?;
        let mut samples = Vec::new();
        for _ in 0..args.runs {
            let result = bench_probe(
                engine,
                serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": args.tg }),
                "decode_result",
            )?;
            samples.push(
                result
                    .get("tok_s")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
            );
        }
        eprintln!(
            "  tg{}@{}: {:.2} tok/s median",
            args.tg,
            context,
            sample_stats(&samples).unwrap().median
        );
        decode_rows.push(serde_json::json!({ "context": context, "tokens": args.tg, "stats": sample_stats(&samples), "samples": samples }));
    }
    let mut sustained_rows = Vec::new();
    if let Some(tg) = args.sustained_tg {
        for context in &args.sustained_ctx {
            let _ = bench_probe(
                engine,
                serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": tg }),
                "decode_result",
            )?;
            let mut samples = Vec::new();
            for _ in 0..args.runs {
                let result = bench_probe(
                    engine,
                    serde_json::json!({ "type": "bench_decode", "context_tokens": context, "iterations": tg }),
                    "decode_result",
                )?;
                samples.push(
                    result
                        .get("tok_s")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0),
                );
            }
            eprintln!(
                "  tg{tg}@{context}: {:.2} tok/s median",
                sample_stats(&samples).unwrap().median
            );
            sustained_rows.push(serde_json::json!({ "context": context, "tokens": tg, "stats": sample_stats(&samples), "samples": samples }));
        }
    }
    let report = serde_json::json!({
        "protocol": "synthetic-pp-tg-matrix-v1",
        "model": args.model,
        "loaded": loaded,
        "gpu": diag,
        "redline_pm4": args.redline,
        "kv_mode": args.kv_mode,
        "runs": args.runs,
        "prefill": pp_rows,
        "decode": decode_rows,
        "sustained": sustained_rows,
    });
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}

fn bench_experimental(paths: &Paths, args: &BenchArgs) -> Result<()> {
    let mut rows = Vec::new();
    for variant in 1..=5 {
        let (mut engine, _, _, diag) = open_bench_engine(paths, args, Some(variant))?;
        let arch = diag
            .get("arch")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        if !matches!(arch, "gfx1030" | "gfx1031") {
            bail!("--exp requires RDNA2 (gfx1030/gfx1031), detected {arch}");
        }
        let _ = bench_generate(&mut engine, "Hello", 16)?;
        let prompt = if args.prompt.is_empty() {
            "Explain the theory of general relativity in simple terms.".to_owned()
        } else {
            args.prompt.join(" ")
        };
        let mut samples = Vec::new();
        for _ in 0..args.runs {
            let done = bench_generate(&mut engine, &prompt, 128)?;
            if let Some(value) = done.get("decode_tok_s").and_then(serde_json::Value::as_f64) {
                samples.push(value);
            }
        }
        let stats = sample_stats(&samples)
            .ok_or_else(|| anyhow!("variant {variant} produced no measurements"))?;
        println!(
            "v{variant}: median {:.2}, mean {:.2}, range {:.2}-{:.2} tok/s",
            stats.median, stats.mean, stats.min, stats.max
        );
        rows.push((variant, stats));
    }
    if let Some((variant, stats)) = rows.iter().max_by(|a, b| a.1.median.total_cmp(&b.1.median)) {
        println!("best: v{variant} at {:.2} tok/s median", stats.median);
    }
    Ok(())
}

fn print_sample_row(label: &str, stats: Option<SampleStats>) {
    if let Some(stats) = stats {
        println!(
            "  {label:<10} median {:>9.2}  mean {:>9.2}  range {:>9.2}-{:>9.2}  sd {:>7.2}",
            stats.median, stats.mean, stats.min, stats.max, stats.stdev
        );
    }
}

fn profile_command(paths: &Paths, args: ProfileArgs) -> Result<()> {
    let mut engine = if let Some(model) = args.model.as_deref() {
        eprintln!("loading {model} once so its kernels are present in the inventory...");
        let bench = BenchArgs {
            model: model.to_owned(),
            runs: 1,
            json: false,
            exp: false,
            matrix: false,
            pp: vec![128],
            ctx: vec![128],
            tg: 1,
            sustained_tg: None,
            sustained_ctx: vec![128],
            warmups: 1,
            kv_mode: None,
            kv_backend: None,
            redline: false,
            prompt: Vec::new(),
        };
        let (mut engine, _, _, _) = open_bench_engine(paths, &bench, None)?;
        let _ = bench_generate(&mut engine, "Hello", 1)?;
        engine
    } else {
        let (_, resolved) = resolved_global(paths, true)?;
        let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
        let daemon = find_daemon(paths).ok_or_else(|| anyhow!("daemon binary not found"))?;
        let mut engine = Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config)?;
        engine.ping()?;
        engine
    };
    let mut report = engine.request(&serde_json::json!({ "type": "profile" }))?;
    if report.get("type").and_then(serde_json::Value::as_str) != Some("profile") {
        bail!(
            "daemon profile failed: {}",
            report
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unexpected response")
        );
    }
    if let Some(filter) = args.kernel.as_deref() {
        let filtered = report
            .get("kernels")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter(|kernel| {
                kernel
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|name| name.contains(filter))
            })
            .cloned()
            .collect::<Vec<_>>();
        report["kernels"] = serde_json::Value::Array(filtered);
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let gpu = &report["gpu"];
        println!(
            "GPU: {} ({})",
            gpu.get("arch")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            gpu.get("generation")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
        );
        println!(
            "{} CUs | peak BW {:.0} GB/s | boost {} MHz | ridge {:.1} FLOP/byte",
            gpu.get("cu_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            gpu.get("peak_bw_gbs")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
            gpu.get("boost_clock_mhz")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
            gpu.get("ridge_point")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0),
        );
        let kernels = report["kernels"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        println!("\nKernel report ({} kernels):", kernels.len());
        println!(
            "  {:<38} {:>5} {:>5} {:>8} {:>10}  limiter",
            "kernel", "VGPR", "SGPR", "LDS", "occupancy"
        );
        for kernel in kernels {
            println!(
                "  {:<38} {:>5} {:>5} {:>8} {:>9.1}%  {}",
                kernel
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
                kernel
                    .get("vgprs")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                kernel
                    .get("sgprs")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                kernel
                    .get("lds_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                kernel
                    .pointer("/occupancy/pct")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
                kernel
                    .pointer("/occupancy/limiter")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
            );
        }
        println!("\nFor phase-aware ISA fit evidence, run hipfire-atlas.");
    }
    Ok(())
}

fn version_command(paths: &Paths, output: OutputArgs) -> Result<()> {
    let installed = paths.root.join("src");
    let (source_kind, source) = if installed.join("Cargo.toml").is_file() {
        ("managed", installed)
    } else {
        (
            "build checkout",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        )
    };
    let source = fs::canonicalize(&source).unwrap_or(source);
    let source_commit = git_output(&source, &["rev-parse", "--verify", "HEAD"]).ok();
    let source_ref = git_output(&source, &["describe", "--tags", "--exact-match", "HEAD"])
        .ok()
        .or_else(|| git_output(&source, &["symbolic-ref", "--short", "HEAD"]).ok());
    let source_dirty = git_output(&source, &["status", "--porcelain"])
        .ok()
        .map(|status| !status.is_empty());
    let source_matches_build = source_commit
        .as_deref()
        .filter(|_| BUILD_COMMIT != "unknown")
        .map(|commit| commit == BUILD_COMMIT);
    let daemon = ["daemon", "daemon.exe"]
        .into_iter()
        .map(|name| paths.root.join("bin").join(name))
        .find(|path| path.is_file());
    let daemon_sha256 = daemon
        .as_deref()
        .map(sha256_path)
        .transpose()
        .context("failed to hash installed daemon")?;
    let value = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "build": {
            "commit": BUILD_COMMIT,
            "ref": BUILD_REF,
            "dirty": BUILD_DIRTY == "true",
            "target": BUILD_TARGET,
        },
        "source": {
            "kind": source_kind,
            "path": source,
            "commit": source_commit,
            "ref": source_ref,
            "dirty": source_dirty,
            "matches_build": source_matches_build,
        },
        "daemon": {
            "path": daemon,
            "sha256": daemon_sha256,
        },
        "config_schema_version": CONFIG_SCHEMA_VERSION,
    });
    if output.json {
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    println!("hipfire {}", env!("CARGO_PKG_VERSION"));
    println!("  build commit: {BUILD_COMMIT}");
    println!(
        "  build ref:    {BUILD_REF}{}",
        if BUILD_DIRTY == "true" {
            " (dirty)"
        } else {
            ""
        }
    );
    println!("  build target: {BUILD_TARGET}");
    println!("  source:       {source_kind} {}", source.display());
    println!(
        "  source ref:   {}",
        source_ref.as_deref().unwrap_or("unknown")
    );
    println!(
        "  source commit: {}",
        source_commit
            .as_deref()
            .map(str::to_owned)
            .unwrap_or_else(|| "unknown".into())
    );
    println!(
        "  source state: {}",
        match source_dirty {
            Some(true) => "dirty",
            Some(false) => "clean",
            None => "unknown",
        }
    );
    println!(
        "  source/build: {}",
        match source_matches_build {
            Some(true) => "match",
            Some(false) => "MISMATCH",
            None => "unknown",
        }
    );
    if let (Some(path), Some(digest)) = (daemon, daemon_sha256) {
        println!("  daemon:       {}", path.display());
        println!("  daemon sha256: {digest}");
    } else {
        println!("  daemon:       not installed");
    }
    Ok(())
}

fn update_command(paths: &Paths, args: UpdateArgs) -> Result<()> {
    if !cfg!(target_os = "linux") {
        bail!("hipfire update is Linux-only; re-run the platform installer with a revision selector on this OS");
    }
    let requested = parse_revision_selector(&args)?;
    let installed = paths.root.join("src");
    let managed = installed.join("Cargo.toml").is_file();
    let repo = if managed {
        installed
    } else {
        if requested.is_some() {
            bail!(
                "revision switching is limited to managed installs under {}; \
                 run install.sh --ref <ref> once to create one",
                paths.root.join("src").display()
            );
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    };
    let repo = fs::canonicalize(&repo).unwrap_or(repo);
    let current_branch = git_output(&repo, &["symbolic-ref", "--short", "HEAD"]).ok();
    if !managed && current_branch.as_deref() != Some("master") {
        bail!(
            "this binary was built from an unmanaged '{}' checkout; \
             update it with git or install a managed copy",
            current_branch.as_deref().unwrap_or("detached")
        );
    }
    let selector = requested
        .or_else(|| {
            current_branch.as_ref().map(|branch| RevisionSelector {
                value: branch.clone(),
                kind: RevisionKind::Branch,
            })
        })
        .ok_or_else(|| {
            anyhow!(
                "this installation is pinned at a detached commit; \
                 choose a target such as `hipfire update @master`"
            )
        })?;

    eprintln!(
        "fetching {} '{}' from origin...",
        selector.kind.label(),
        selector.value
    );
    let resolved = fetch_revision(&repo, selector)?;
    let previous = git_output(&repo, &["rev-parse", "--verify", "HEAD"])?;
    let short = previous.get(..12).unwrap_or(&previous);
    let backup_ref = format!(
        "refs/hipfire/backups/pre-update-{}-{short}",
        unix_timestamp()
    );
    run_checked(
        Command::new("git")
            .current_dir(&repo)
            .args(["update-ref", &backup_ref, &previous]),
        "git update-ref backup",
    )?;
    eprintln!("previous source retained at {backup_ref}");

    let dirty = !git_output(&repo, &["status", "--porcelain"])?.is_empty();
    if dirty {
        let message = format!("hipfire-update-{}", unix_timestamp());
        eprintln!("local modifications detected; stashing as {message}");
        run_checked(
            Command::new("git").current_dir(&repo).args([
                "stash",
                "push",
                "--include-untracked",
                "-m",
                &message,
            ]),
            "git stash",
        )?;
        eprintln!("recover later with: git -C {} stash pop", repo.display());
    }
    checkout_revision(&repo, &resolved)?;

    let installer = repo.join("scripts/install.sh");
    if !installer.is_file() {
        bail!("updated checkout has no {}", installer.display());
    }
    run_checked(
        Command::new("bash")
            .arg(installer)
            .current_dir(&repo)
            .env("HIPFIRE_FORCE_REBUILD", "1"),
        "native installer",
    )?;
    println!(
        "hipfire updated to {} '{}' ({})",
        resolved.selector.kind.label(),
        resolved.selector.value,
        resolved.commit
    );
    println!("verify with: hipfire version");
    Ok(())
}

fn parse_revision_selector(args: &UpdateArgs) -> Result<Option<RevisionSelector>> {
    let candidates = [
        args.reference
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Auto)),
        args.branch
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Branch)),
        args.tag
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Tag)),
        args.commit
            .as_ref()
            .map(|value| (value.as_str(), RevisionKind::Commit)),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if candidates.len() > 1 {
        bail!("choose only one update ref, --branch, --tag, or --commit");
    }
    let Some((raw, mut kind)) = candidates.first().copied() else {
        return Ok(None);
    };
    let mut value = raw.trim().trim_start_matches('@');
    if let Some(branch) = value.strip_prefix("refs/heads/") {
        value = branch;
        kind = RevisionKind::Branch;
    } else if let Some(tag) = value.strip_prefix("refs/tags/") {
        value = tag;
        kind = RevisionKind::Tag;
    } else if let Some(branch) = value.strip_prefix("origin/") {
        value = branch;
        if kind == RevisionKind::Auto {
            kind = RevisionKind::Branch;
        }
    }
    validate_revision(value, kind)?;
    Ok(Some(RevisionSelector {
        value: value.to_owned(),
        kind,
    }))
}

fn validate_revision(value: &str, kind: RevisionKind) -> Result<()> {
    let invalid = value.is_empty()
        || value.starts_with(['-', '.', '/'])
        || value.ends_with(['.', '/'])
        || value.contains("..")
        || value.contains("@{")
        || value.contains("//")
        || value.chars().any(|character| {
            character.is_whitespace()
                || character.is_control()
                || matches!(character, '\\' | ':' | '?' | '*' | '[' | '^' | '~')
        });
    if invalid {
        bail!("unsafe or invalid git revision {value:?}");
    }
    if kind == RevisionKind::Commit
        && (!(7..=40).contains(&value.len())
            || !value.chars().all(|character| character.is_ascii_hexdigit()))
    {
        bail!("--commit requires a 7-40 character hexadecimal git commit");
    }
    Ok(())
}

fn fetch_revision(repo: &Path, mut selector: RevisionSelector) -> Result<ResolvedRevision> {
    if selector.kind == RevisionKind::Auto {
        selector.kind = if remote_ref_exists(repo, &format!("refs/heads/{}", selector.value))? {
            RevisionKind::Branch
        } else if remote_ref_exists(repo, &format!("refs/tags/{}", selector.value))? {
            RevisionKind::Tag
        } else {
            RevisionKind::Commit
        };
    }

    match selector.kind {
        RevisionKind::Branch => {
            let remote = format!("refs/heads/{}", selector.value);
            if !remote_ref_exists(repo, &remote)? {
                bail!("origin has no branch '{}'", selector.value);
            }
            let tracking = format!("refs/remotes/origin/{}", selector.value);
            let refspec = format!("+{remote}:{tracking}");
            run_checked(
                Command::new("git")
                    .current_dir(repo)
                    .args(["fetch", "--depth", "1", "origin", &refspec]),
                "git fetch branch",
            )?;
            let commit = git_output(repo, &["rev-parse", "--verify", &tracking])?;
            Ok(ResolvedRevision {
                selector,
                commit,
                tracking_ref: Some(tracking),
            })
        }
        RevisionKind::Tag => {
            let remote = format!("refs/tags/{}", selector.value);
            if !remote_ref_exists(repo, &remote)? {
                bail!("origin has no tag '{}'", selector.value);
            }
            run_checked(
                Command::new("git")
                    .current_dir(repo)
                    .args(["fetch", "--depth", "1", "origin", &remote]),
                "git fetch tag",
            )?;
            let commit = git_output(repo, &["rev-parse", "--verify", "FETCH_HEAD^{commit}"])?;
            Ok(ResolvedRevision {
                selector,
                commit,
                tracking_ref: None,
            })
        }
        RevisionKind::Commit => {
            run_checked(
                Command::new("git").current_dir(repo).args([
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    &selector.value,
                ]),
                "git fetch commit",
            )?;
            let commit = git_output(repo, &["rev-parse", "--verify", "FETCH_HEAD^{commit}"])?;
            Ok(ResolvedRevision {
                selector,
                commit,
                tracking_ref: None,
            })
        }
        RevisionKind::Auto => unreachable!("auto revisions are resolved before fetch"),
    }
}

fn checkout_revision(repo: &Path, resolved: &ResolvedRevision) -> Result<()> {
    if let Some(tracking) = &resolved.tracking_ref {
        run_checked(
            Command::new("git").current_dir(repo).args([
                "checkout",
                "-B",
                &resolved.selector.value,
                tracking,
            ]),
            "git checkout branch",
        )
    } else {
        run_checked(
            Command::new("git")
                .current_dir(repo)
                .args(["checkout", "--detach", &resolved.commit]),
            "git checkout pinned revision",
        )
    }
}

fn remote_ref_exists(repo: &Path, reference: &str) -> Result<bool> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(["ls-remote", "--exit-code", "origin", reference])
        .output()
        .with_context(|| format!("failed to query origin for {reference}"))?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(2) => Ok(false),
        _ => bail!(
            "git ls-remote failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ),
    }
}

fn sha256_path(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn git_output(repo: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(args)
        .output()
        .with_context(|| format!("failed to run git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn run_checked(command: &mut Command, label: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed to start {label}"))?;
    if status.success() {
        Ok(())
    } else {
        bail!("{label} failed with {status}")
    }
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn quantize_command(paths: &Paths, mut args: QuantizeArgs) -> Result<()> {
    let quantizer = find_workspace_binary(paths, "hipfire-quantize").ok_or_else(|| {
        anyhow!(
            "hipfire-quantize is not installed; build `cargo build --release -p hipfire-quantize`"
        )
    })?;
    if args.both {
        args.formats.extend(["mq4".into(), "mq6".into()]);
    }
    let input_path = PathBuf::from(&args.input);
    let is_gguf = input_path.is_file()
        && input_path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("gguf"));
    if args.formats.is_empty() {
        args.formats
            .push(if is_gguf { "hf4".into() } else { "mq4".into() });
    }
    for format in &mut args.formats {
        *format = match format.as_str() {
            "hfq4" | "hfq4g256" => "hf4".into(),
            "hfq6" | "hfq6g256" => "hf6".into(),
            _ => format.clone(),
        };
    }
    let mut seen = BTreeSet::new();
    args.formats.retain(|format| seen.insert(format.clone()));
    let valid = ["mq4", "mq6", "q8", "q8f16", "hf4", "hf6"];
    for format in &args.formats {
        if !valid.contains(&format.as_str()) {
            bail!(
                "unsupported format {format}; supported: {}",
                valid.join(", ")
            );
        }
        if is_gguf && !matches!(format.as_str(), "hf4" | "hf6" | "mq4" | "mq6") {
            bail!("GGUF input supports hf4, hf6, mq4, or mq6");
        }
    }
    if args.output.is_some() && args.formats.len() != 1 {
        bail!("--output requires exactly one format; use --output-dir for multiple formats");
    }
    if let Some(repo) = args.upload.as_deref() {
        if repo.split('/').count() != 2 {
            bail!("--upload requires owner/repo");
        }
    }
    let input = if input_path.exists() {
        fs::canonicalize(input_path)?.display().to_string()
    } else {
        args.input.clone()
    };
    let stem = args.stem.unwrap_or_else(|| {
        Path::new(&args.input)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(&args.input)
            .trim_end_matches(".gguf")
            .to_owned()
    });
    let output_dir = args.output_dir.unwrap_or(env::current_dir()?);
    fs::create_dir_all(&output_dir)?;
    let mut produced = Vec::new();
    for format in &args.formats {
        let output = args
            .output
            .clone()
            .unwrap_or_else(|| output_dir.join(format!("{stem}.{format}")));
        eprintln!("quantizing {input} -> {} ({format})", output.display());
        run_checked(
            Command::new(&quantizer)
                .arg("--input")
                .arg(&input)
                .arg("--output")
                .arg(&output)
                .arg("--format")
                .arg(format),
            "hipfire-quantize",
        )?;
        if !output.is_file() {
            bail!(
                "quantizer reported success but {} was not created",
                output.display()
            );
        }
        produced.push((format.clone(), fs::canonicalize(&output).unwrap_or(output)));
    }
    if args.install {
        fs::create_dir_all(&paths.models)?;
        for (_, output) in &mut produced {
            let file = output
                .file_name()
                .ok_or_else(|| anyhow!("invalid output path {}", output.display()))?;
            let destination = paths.models.join(file);
            if fs::canonicalize(&destination).ok().as_ref()
                != fs::canonicalize(&*output).ok().as_ref()
            {
                fs::copy(&*output, &destination)?;
            }
            *output = fs::canonicalize(&destination).unwrap_or(destination);
            eprintln!("installed {}", output.display());
        }
    }
    if let Some(repo) = args.upload.as_deref() {
        if args.create_repo {
            run_checked(
                Command::new("hf").args(["repos", "create", repo, "--type", "model", "--exist-ok"]),
                "hf repos create",
            )?;
        }
        for (_, output) in &produced {
            let file = output
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| anyhow!("invalid output filename"))?;
            run_checked(
                Command::new("hf")
                    .arg("upload")
                    .arg(repo)
                    .arg(output)
                    .arg(file),
                "hf upload",
            )?;
        }
    }
    if let Some(alias) = args.register {
        let (_, primary) = produced
            .iter()
            .find(|(format, _)| format == "mq4")
            .or_else(|| produced.first())
            .ok_or_else(|| anyhow!("no quantized artifact produced"))?;
        let mut loaded = load_catalog(&paths.config)?;
        let id = primary
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| anyhow!("invalid output filename"))?
            .to_owned();
        loaded.catalog.models.insert(
            id.clone(),
            hipfire_config::LocalModelConfig {
                path: Some(primary.clone()),
                registry_tag: None,
                overrides: ConfigLayer::default(),
            },
        );
        loaded.catalog.aliases.insert(alias.clone(), id);
        write_catalog_toml(&paths.config, &loaded.catalog)?;
        eprintln!("registered {alias} -> {}", primary.display());
    }
    Ok(())
}

fn sidecar_command(paths: &Paths, args: SidecarArgs) -> Result<()> {
    if !(1..=1_000_000).contains(&args.max_tokens) {
        bail!("--max-tokens must be between 1 and 1000000");
    }
    if !(1..=16_384).contains(&args.chunk_len) {
        bail!("--chunk-len must be between 1 and 16384");
    }
    if let Some(corpus) = args.corpus.as_ref().filter(|path| !path.is_file()) {
        bail!("corpus not found: {}", corpus.display());
    }
    let registry = load_registry(&paths.registry).registry;
    let model = find_model_path(paths, &registry, &args.model)
        .ok_or_else(|| anyhow!("model not found: {}", args.model))?;
    let output = args
        .output
        .unwrap_or_else(|| PathBuf::from(format!("{}.triattn.bin", model.display())));
    let binary = find_workspace_example(paths, "triattn_validate").ok_or_else(|| anyhow!(
        "triattn_validate is not installed; build `cargo build --release --features deltanet -p hipfire-runtime --example triattn_validate`"
    ))?;
    let mut command = Command::new(binary);
    command
        .arg(&model)
        .arg("--sidecar")
        .arg(&output)
        .arg("--max-tokens")
        .arg(args.max_tokens.to_string())
        .arg("--chunk-len")
        .arg(args.chunk_len.to_string());
    if let Some(corpus) = args.corpus {
        command.arg("--corpus").arg(corpus);
    }
    if args.cpu_calib {
        command.arg("--cpu-calib");
    }
    if args.skip_validation {
        command.arg("--val-prompt").arg("");
    }
    let _ = args.gpu_calib;
    run_checked(&mut command, "triattn_validate")?;
    if !output.is_file() {
        bail!(
            "sidecar generator reported success but {} was not created",
            output.display()
        );
    }
    println!("{}", output.display());
    Ok(())
}

fn find_workspace_binary(paths: &Paths, name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
    [
        paths.root.join("bin").join(&exe),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release")
            .join(&exe),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .or_else(|| find_on_path(&exe))
}

fn find_workspace_example(paths: &Paths, name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
    [
        paths.root.join("bin").join(&exe),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release/examples")
            .join(&exe),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .or_else(|| find_on_path(&exe))
}

fn diag_command(paths: &Paths, output: OutputArgs) -> Result<()> {
    let loaded_registry = load_registry(&paths.registry);
    let models = list_local_models(paths, &loaded_registry.registry)?;
    let loaded_config = load_global(&paths.config)?;
    let platform = format!("{}-{}", env::consts::OS, env::consts::ARCH);
    let kfd = Path::new("/dev/kfd").exists();
    let amdgpu_loaded = Path::new("/sys/module/amdgpu").exists();
    let gpu_arches = detect_gpu_arches();
    let gpus = detect_amd_drm_cards();
    let hipcc = command_version("hipcc", "--version");
    let daemon_path = find_daemon(paths);
    let daemon = daemon_path.as_ref().map(|path| path.display().to_string());
    let live_gpu = daemon_path.as_ref().and_then(|daemon| {
        let (_, resolved) = resolved_global(paths, true).ok()?;
        let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved).ok()?;
        let mut engine =
            Engine::spawn_configured(daemon, &BTreeMap::new(), &process_config).ok()?;
        engine.ping().ok()?;
        engine.request(&serde_json::json!({ "type": "diag" })).ok()
    });
    let gpu = gpu_arches
        .first()
        .map(|arch| serde_json::json!({ "arch": arch }))
        .unwrap_or_else(
            || serde_json::json!({ "error": "no gfx target detected in KFD topology" }),
        );
    let config_overrides = loaded_config
        .layer
        .values
        .iter()
        .map(|(key, value)| (key.clone(), serde_json::json!(value)))
        .collect::<serde_json::Map<_, _>>();
    let report = serde_json::json!({
        "registry": registry_source(loaded_registry.source),
        "platform": platform,
        "hardware_probe": if cfg!(target_os = "linux") { "linux" } else { "limited" },
        "gpus": gpus,
        "dri_nodes": list_dri_nodes(),
        "kfd": kfd,
        "amdgpu_loaded": amdgpu_loaded,
        "rocm": { "hipcc": hipcc },
        "daemon": daemon,
        "live_gpu": live_gpu,
        "models": models,
        "gpu": gpu,
        "config_path": loaded_config.path,
        "config_format": format!("{:?}", loaded_config.format).to_lowercase(),
        "config_overrides": config_overrides,
        "warnings": loaded_registry.warnings,
    });
    if output.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("hipfire diagnostics\n");
        println!(
            "registry:      {}",
            report["registry"].as_str().unwrap_or("unknown")
        );
        println!("platform:      {platform}");
        println!(
            "amdgpu:       {}",
            if amdgpu_loaded {
                "loaded"
            } else {
                "not loaded"
            }
        );
        println!("/dev/kfd:      {}", if kfd { "present" } else { "missing" });
        println!(
            "GPU targets:   {}",
            if gpu_arches.is_empty() {
                "none".into()
            } else {
                gpu_arches.join(", ")
            }
        );
        println!("local models:  {}", models.len());
        println!(
            "config:        {} ({:?})",
            loaded_config.path.display(),
            loaded_config.format
        );
        println!("daemon:        {}", daemon.as_deref().unwrap_or("missing"));
        if let Some(live) = report.get("live_gpu").filter(|value| !value.is_null()) {
            println!(
                "HIP GPU:       {} (HIP {})",
                live.get("arch")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
                live.get("hip_version")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
            );
            println!(
                "VRAM:          {} MB free / {} MB total",
                live.get("vram_free_mb")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                live.get("vram_total_mb")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
            );
            if matches!(
                live.get("arch").and_then(serde_json::Value::as_str),
                Some("gfx1150" | "gfx1151" | "gfx1152")
            ) && live
                .get("hip_version")
                .and_then(serde_json::Value::as_str)
                .and_then(parse_major_minor)
                .is_some_and(|version| version < (7, 2))
            {
                println!("WARNING: RDNA 3.5 requires ROCm/HIP 7.2 or newer.");
            }
        } else if daemon.is_some() {
            println!("HIP probe:     failed (run the daemon directly for detailed startup errors)");
        }
    }
    Ok(())
}

fn parse_major_minor(value: &str) -> Option<(u64, u64)> {
    let mut parts = value.split('.');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

fn detect_gpu_arches() -> Vec<String> {
    let root = Path::new("/sys/class/kfd/kfd/topology/nodes");
    let mut arches = Vec::new();
    let Ok(nodes) = fs::read_dir(root) else {
        return arches;
    };
    for node in nodes.flatten() {
        let Ok(properties) = fs::read_to_string(node.path().join("properties")) else {
            continue;
        };
        let Some(version) = properties.lines().find_map(|line| {
            line.split_whitespace()
                .collect::<Vec<_>>()
                .as_slice()
                .strip_prefix(&["gfx_target_version"])
                .and_then(|rest| rest.first())
                .and_then(|value| value.parse::<u32>().ok())
        }) else {
            continue;
        };
        if let Some(arch) = gfx_version_to_arch(version) {
            if !arches.iter().any(|candidate| candidate == arch) {
                arches.push(arch.to_owned());
            }
        }
    }
    arches
}

fn gfx_version_to_arch(version: u32) -> Option<&'static str> {
    match version {
        90006 => Some("gfx906"),
        90008 => Some("gfx908"),
        90010 => Some("gfx90a"),
        90400..=90402 => Some("gfx94x"),
        100100 => Some("gfx1010"),
        100300 | 100302 => Some("gfx1030"),
        110000..=110002 => Some("gfx1100"),
        110500 => Some("gfx1150"),
        110501 => Some("gfx1151"),
        120000 => Some("gfx1200"),
        120001 => Some("gfx1201"),
        _ => None,
    }
}

fn detect_amd_drm_cards() -> Vec<String> {
    let mut cards = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return cards;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || !name[4..].bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let vendor = fs::read_to_string(entry.path().join("device/vendor")).unwrap_or_default();
        if vendor.trim() == "0x1002" {
            cards.push(name);
        }
    }
    cards.sort();
    cards
}

fn list_dri_nodes() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/dev/dri") else {
        return Vec::new();
    };
    let mut nodes = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    nodes.sort();
    nodes
}

fn command_version(command: &str, argument: &str) -> Option<String> {
    Command::new(command)
        .arg(argument)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .unwrap_or_default()
                .trim()
                .to_owned()
        })
        .filter(|line| !line.is_empty())
}

fn find_daemon(paths: &Paths) -> Option<PathBuf> {
    if let Some(path) = env::var_os("HIPFIRE_DAEMON_BIN").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    [
        paths.root.join("bin/daemon"),
        workspace.join("release/examples/daemon"),
        workspace.join("debug/examples/daemon"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn request_f64(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
    explicit: Option<f64>,
) -> Result<Option<f64>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    request_config_value(resolved, key)?
        .map(|value| config_value_f64(value, key))
        .transpose()
}

fn request_u64(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
    explicit: Option<u64>,
) -> Result<Option<u64>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    request_config_value(resolved, key)?
        .map(|value| config_value_u64(value, key))
        .transpose()
}

fn request_string(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
    explicit: Option<String>,
) -> Result<Option<String>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    request_config_value(resolved, key)?
        .map(|value| match value {
            hipfire_config::ConfigValue::String(value) => Ok(value.clone()),
            value => bail!(
                "configuration key '{key}' resolved as {}, expected string",
                value.kind()
            ),
        })
        .transpose()
}

fn request_config_value<'a>(
    resolved: &'a hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<Option<&'a hipfire_config::ConfigValue>> {
    let value = resolved
        .get(key)
        .ok_or_else(|| anyhow!("configuration key '{key}' is not resolved"))?;
    match &value.source {
        ConfigSource::BuiltIn => Ok(None),
        ConfigSource::GlobalUser { .. } => Ok(value
            .shadowed
            .iter()
            .rev()
            .find(|candidate| {
                matches!(
                    candidate.source,
                    ConfigSource::RegistryModel { .. } | ConfigSource::RegistryTarget { .. }
                )
            })
            .map(|candidate| &candidate.value)),
        _ => Ok(Some(&value.value)),
    }
}

fn config_value_f64(value: &hipfire_config::ConfigValue, key: &str) -> Result<f64> {
    match value {
        hipfire_config::ConfigValue::Float(value) => Ok(*value),
        hipfire_config::ConfigValue::Integer(value) => Ok(*value as f64),
        _ => bail!("configuration key '{key}' did not resolve to a number"),
    }
}

fn config_value_u64(value: &hipfire_config::ConfigValue, key: &str) -> Result<u64> {
    match value {
        hipfire_config::ConfigValue::Integer(value) => u64::try_from(*value)
            .map_err(|_| anyhow!("configuration key '{key}' cannot be negative")),
        value => bail!(
            "configuration key '{key}' resolved as {}, expected integer",
            value.kind()
        ),
    }
}

fn insert_optional_f64(target: &mut serde_json::Value, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        target[key] = serde_json::json!(value);
    }
}

fn insert_optional_u64(target: &mut serde_json::Value, key: &str, value: Option<u64>) {
    if let Some(value) = value {
        target[key] = serde_json::json!(value);
    }
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|directory| directory.join(name))
            .find(|candidate| candidate.is_file())
    })
}

fn is_model_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    MODEL_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix))
}

fn source_label(source: &ConfigSource) -> String {
    match source {
        ConfigSource::BuiltIn => "built-in".into(),
        ConfigSource::RegistryModel { tag, revision } => {
            format!("registry model {tag}@{revision}")
        }
        ConfigSource::RegistryTarget {
            tag,
            arch,
            revision,
        } => format!("registry target {tag}/{arch}@{revision}"),
        ConfigSource::GlobalUser { path } => format!("global user ({})", path.display()),
        ConfigSource::ModelUser { model, path } => {
            format!("model user {model} ({})", path.display())
        }
        ConfigSource::LegacyEnv { name } => format!("legacy env {name}"),
        ConfigSource::OneShot { argument } => format!("one-shot {argument}"),
    }
}

fn config_rule_json(rule: ValueRule) -> serde_json::Value {
    match rule {
        ValueRule::Bool => serde_json::json!({ "type": "boolean" }),
        ValueRule::Integer { min, max } => {
            serde_json::json!({ "type": "integer", "minimum": min, "maximum": max })
        }
        ValueRule::Float {
            min,
            max,
            min_inclusive,
        } => serde_json::json!({
            "type": "number",
            "minimum": min,
            "maximum": max,
            "minimum_inclusive": min_inclusive,
        }),
        ValueRule::String => serde_json::json!({ "type": "string" }),
        ValueRule::NonEmptyString => {
            serde_json::json!({ "type": "string", "min_length": 1 })
        }
        ValueRule::Host => serde_json::json!({ "type": "string", "format": "host" }),
        ValueRule::PathOrEmpty => {
            serde_json::json!({ "type": "string", "format": "existing-path-or-empty" })
        }
        ValueRule::Enum(values) => {
            serde_json::json!({ "type": "string", "enum": values })
        }
        ValueRule::AutoBool => serde_json::json!({
            "type": ["boolean", "string"],
            "enum": [true, false, "auto"],
        }),
        ValueRule::NullableString => {
            serde_json::json!({ "type": ["string", "null"] })
        }
        ValueRule::NullableEnum(values) => serde_json::json!({
            "type": ["string", "null"],
            "enum": values,
            "nullable": true,
        }),
        ValueRule::NullableInteger { min, max } => serde_json::json!({
            "type": ["integer", "null"],
            "minimum": min,
            "maximum": max,
        }),
        ValueRule::NullableFloat { min, max } => serde_json::json!({
            "type": ["number", "null"],
            "minimum": min,
            "maximum": max,
        }),
        ValueRule::KvAdaptive => serde_json::json!({
            "type": "string",
            "format": "kv-adaptive-policy",
        }),
    }
}

fn config_rule_label(rule: ValueRule) -> &'static str {
    match rule {
        ValueRule::Bool => "bool",
        ValueRule::Integer { .. } => "integer",
        ValueRule::Float { .. } => "number",
        ValueRule::String => "string",
        ValueRule::NonEmptyString => "nonempty-string",
        ValueRule::Host => "host",
        ValueRule::PathOrEmpty => "path-or-empty",
        ValueRule::Enum(_) => "enum",
        ValueRule::AutoBool => "auto-bool",
        ValueRule::NullableString => "string|null",
        ValueRule::NullableEnum(_) => "enum|null",
        ValueRule::NullableInteger { .. } => "integer|null",
        ValueRule::NullableFloat { .. } => "number|null",
        ValueRule::KvAdaptive => "kv-adaptive",
    }
}

fn config_default_value(schema: &hipfire_config::ConfigField) -> hipfire_config::ConfigValue {
    // Resolve one empty layer set so the config crate remains the only place
    // that turns the private DefaultValue representation into a public value.
    resolve(Vec::<NamedLayer>::new())
        .expect("built-in schema validates")
        .get(schema.key)
        .expect("schema key resolved")
        .value
        .clone()
}

fn format_default(schema: &hipfire_config::ConfigField) -> String {
    config_default_value(schema).to_string()
}

fn registry_source(source: RegistrySource) -> &'static str {
    match source {
        RegistrySource::Cache => "cache",
        RegistrySource::Network => "network",
        RegistrySource::StaleCache => "stale-cache",
        RegistrySource::Bundled => "bundled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::CONFIG_PROFILE_NAMES;

    fn test_paths(label: &str) -> Paths {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "hipfire-cli-{label}-{}-{nonce}",
            std::process::id()
        ));
        let config = ConfigPaths::under(&root);
        Paths {
            models: config.models.clone(),
            registry: RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        }
    }

    #[test]
    fn model_suffix_filter_covers_current_formats() {
        assert!(is_model_file("qwen3.6-35b-a3b.mq4r"));
        assert!(is_model_file("deepseek.mq2lloyd"));
        assert!(is_model_file("draft.hfq"));
        assert!(!is_model_file("model.triattn.bin"));
        assert!(!is_model_file("README.md"));
    }

    #[test]
    fn nested_model_discovery_matches_native_registry_layout() {
        let paths = test_paths("nested-models");
        let nested = paths.models.join("community").join("example-model.mq4r");
        fs::create_dir_all(nested.parent().unwrap()).unwrap();
        fs::write(&nested, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        assert_eq!(
            find_model_path(&paths, &registry, "example-model"),
            Some(fs::canonicalize(&nested).unwrap())
        );
        assert!(list_local_models(&paths, &registry)
            .unwrap()
            .iter()
            .any(|model| model.path == fs::canonicalize(&nested).unwrap()));
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn cask_triattn_and_pflash_remain_opt_in_at_load() {
        let paths = test_paths("experimental-defaults");
        fs::create_dir_all(&paths.models).unwrap();
        let registry = hipfire_registry::bundled().unwrap();
        let entry = registry
            .models
            .values()
            .find(|entry| entry.triattn.is_some())
            .expect("bundled registry should retain a TriAttention sidecar");
        let model_path = paths.models.join(&entry.file);
        fs::write(&model_path, b"model").unwrap();
        let triattn = entry.triattn.as_ref().unwrap();
        let sidecar_path = paths.models.join(&triattn.file);
        fs::write(&sidecar_path, b"sidecar").unwrap();

        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let params = load_params(&defaults, Some(entry), &model_path, 64, None, None).unwrap();
        assert_eq!(params["cask"], false);
        assert_eq!(params["cask_sidecar"], "");
        assert_eq!(params["prefill_compression"], "off");

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("memory.cask.auto_attach", "true").unwrap();
        let enabled = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "memory.cask.auto_attach=true".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let params = load_params(&enabled, Some(entry), &model_path, 64, None, None).unwrap();
        assert_eq!(params["cask"], false);
        assert_eq!(params["cask_sidecar"], sidecar_path.display().to_string());
        assert_eq!(params["prefill_compression"], "off");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn load_params_forwards_explicit_vmm_backend() {
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params =
            load_params(&defaults, None, &model_path, 64, Some("q8"), Some("vmm")).unwrap();
        assert_eq!(params["kv_backend"], "vmm");
    }

    #[test]
    fn load_params_forwards_dflash_draft_from_environment() {
        struct EnvRestore(Option<std::ffi::OsString>);

        impl Drop for EnvRestore {
            fn drop(&mut self) {
                match &self.0 {
                    Some(value) => env::set_var("HIPFIRE_DFLASH_DRAFT", value),
                    None => env::remove_var("HIPFIRE_DFLASH_DRAFT"),
                }
            }
        }

        let _restore = EnvRestore(env::var_os("HIPFIRE_DFLASH_DRAFT"));
        let draft = "/tmp/qwen35-9b-dflash-mq4.hfq";
        env::set_var("HIPFIRE_DFLASH_DRAFT", draft);

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("speculation.mode", "dflash").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "speculation.mode=dflash".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");

        let params = load_params(&resolved, None, &model_path, 64, Some("q8"), None).unwrap();
        assert_eq!(params["draft"], draft);
    }

    #[test]
    fn run_spec_dflash_projects_inherited_draft_after_config_off() {
        // Reviewer case: resolved config leaves DFlash off, but an inherited
        // HIPFIRE_DFLASH_DRAFT is present and `run --spec dflash` re-enables
        // DFlash after load_params. Draft must land on the final load params.
        struct EnvRestore(Option<std::ffi::OsString>);

        impl Drop for EnvRestore {
            fn drop(&mut self) {
                match &self.0 {
                    Some(value) => env::set_var("HIPFIRE_DFLASH_DRAFT", value),
                    None => env::remove_var("HIPFIRE_DFLASH_DRAFT"),
                }
            }
        }

        let _restore = EnvRestore(env::var_os("HIPFIRE_DFLASH_DRAFT"));
        let draft = "/tmp/qwen35-9b-dflash-mq4.hfq";
        env::set_var("HIPFIRE_DFLASH_DRAFT", draft);

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("speculation.mode", "off").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "speculation.mode=off".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");

        // load_params alone must not carry the draft while config mode is off.
        let mut params = load_params(&resolved, None, &model_path, 64, Some("q8"), None).unwrap();
        assert_eq!(params["dflash_mode"], "off");
        assert!(
            params.get("draft").is_none(),
            "config-off load_params must not project HIPFIRE_DFLASH_DRAFT"
        );

        // Final run-path selector: CLI `--spec dflash` then project inherited draft.
        apply_speculation_selector(&mut params, "dflash").unwrap();
        project_dflash_draft(&mut params);
        assert_eq!(params["dflash_mode"], "on");
        assert_eq!(params["draft"], draft);

        // Final off must clear any previously projected draft.
        apply_speculation_selector(&mut params, "off").unwrap();
        project_dflash_draft(&mut params);
        assert_eq!(params["dflash_mode"], "off");
        assert!(
            params.get("draft").is_none(),
            "final off must drop projected HIPFIRE_DFLASH_DRAFT"
        );
    }

    #[test]
    fn completion_timings_preserves_speculator_identity() {
        let completion = |done| Completion {
            id: "req-test".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done,
        };

        let dflash = completion_timings(&completion(serde_json::json!({
            "dflash": true,
            "tau": 3.5,
            "cycles": 4,
        })));
        assert_eq!(dflash["dflash"], true);
        assert!(dflash["mtp"].is_null());

        let mtp = completion_timings(&completion(serde_json::json!({
            "mtp": true,
            "tau": 2.0,
            "cycles": 6,
        })));
        assert!(mtp["dflash"].is_null());
        assert_eq!(mtp["mtp"], true);
    }

    #[test]
    fn artifact_urls_honor_endpoint_precedence() {
        struct EnvRestore(&'static str, Option<std::ffi::OsString>);

        impl Drop for EnvRestore {
            fn drop(&mut self) {
                match &self.1 {
                    Some(value) => env::set_var(self.0, value),
                    None => env::remove_var(self.0),
                }
            }
        }

        let _hf_base = EnvRestore("HIPFIRE_HF_BASE", env::var_os("HIPFIRE_HF_BASE"));
        let _hf_endpoint = EnvRestore("HF_ENDPOINT", env::var_os("HF_ENDPOINT"));
        let registry = hipfire_registry::bundled().unwrap();
        let (_, entry) = registry.model("qwen3.6:35b-a3b-mq4r").unwrap();
        let suffix = "schuttdev/hipfire-qwen3.6-35b-a3b/resolve/main/qwen3.6-35b-a3b.mq4r";

        env::remove_var("HIPFIRE_HF_BASE");
        env::remove_var("HF_ENDPOINT");
        assert_eq!(
            artifact_url(entry, &entry.file),
            format!("https://huggingface.co/{suffix}")
        );

        env::set_var("HF_ENDPOINT", "https://hf-mirror.example/");
        assert_eq!(
            artifact_url(entry, &entry.file),
            format!("https://hf-mirror.example/{suffix}")
        );

        env::set_var("HIPFIRE_HF_BASE", "https://hipfire-mirror.example///");
        assert_eq!(
            artifact_url(entry, &entry.file),
            format!("https://hipfire-mirror.example/{suffix}")
        );
    }

    #[test]
    fn native_help_exposes_migrated_command_families() {
        use clap::CommandFactory;
        let command = Cli::command();
        let names = command
            .get_subcommands()
            .map(|command| command.get_name())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("config"));
        assert!(names.contains("registry"));
        assert!(names.contains("pull"));
        assert!(names.contains("run"));
        assert!(names.contains("chat"));
        assert!(names.contains("serve"));
        assert!(names.contains("stop"));
        assert!(names.contains("restart"));
        assert!(names.contains("bench"));
        assert!(names.contains("profile"));
        assert!(names.contains("version"));
        assert!(names.contains("update"));
        assert!(names.contains("quantize"));
        assert!(names.contains("sidecar-gen"));
    }

    #[test]
    fn build_version_includes_commit_and_ref_identity() {
        use clap::error::ErrorKind;

        let error = Cli::try_parse_from(["hipfire", "--version"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayVersion);
        let rendered = error.to_string();
        assert!(rendered.contains(env!("CARGO_PKG_VERSION")));
        assert!(rendered.contains(BUILD_COMMIT.get(..12).unwrap_or(BUILD_COMMIT)));
        assert!(rendered.contains(BUILD_REF));
    }

    #[test]
    fn update_accepts_branch_tag_commit_and_at_shorthand() {
        let cases = [
            (
                UpdateArgs {
                    reference: Some("@beta".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "beta".into(),
                    kind: RevisionKind::Auto,
                },
            ),
            (
                UpdateArgs {
                    reference: Some("@origin/beta".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "beta".into(),
                    kind: RevisionKind::Branch,
                },
            ),
            (
                UpdateArgs {
                    tag: Some("v0.3.0".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "v0.3.0".into(),
                    kind: RevisionKind::Tag,
                },
            ),
            (
                UpdateArgs {
                    commit: Some("0123456789abcdef".into()),
                    ..UpdateArgs::default()
                },
                RevisionSelector {
                    value: "0123456789abcdef".into(),
                    kind: RevisionKind::Commit,
                },
            ),
        ];
        for (args, expected) in cases {
            assert_eq!(parse_revision_selector(&args).unwrap(), Some(expected));
        }

        let cli = Cli::try_parse_from(["hipfire", "update", "@beta"]).unwrap();
        let Some(Commands::Update(args)) = cli.command else {
            panic!("expected update command");
        };
        assert_eq!(args.reference.as_deref(), Some("@beta"));
    }

    #[test]
    fn update_rejects_unsafe_or_ambiguous_revisions() {
        for value in ["../beta", "-beta", "beta^{tree}", "beta branch"] {
            let args = UpdateArgs {
                reference: Some(value.into()),
                ..UpdateArgs::default()
            };
            assert!(parse_revision_selector(&args).is_err(), "{value}");
        }
        let short_commit = UpdateArgs {
            commit: Some("123".into()),
            ..UpdateArgs::default()
        };
        assert!(parse_revision_selector(&short_commit).is_err());
        let ambiguous = UpdateArgs {
            branch: Some("beta".into()),
            tag: Some("v0.3.0".into()),
            ..UpdateArgs::default()
        };
        assert!(parse_revision_selector(&ambiguous).is_err());
    }

    #[test]
    fn update_fetches_and_checks_out_branch_from_local_origin() {
        fn git(repo: &Path, args: &[&str]) {
            let status = Command::new("git")
                .current_dir(repo)
                .args(args)
                .status()
                .unwrap();
            assert!(status.success(), "git {}", args.join(" "));
        }

        let root = env::temp_dir().join(format!(
            "hipfire-update-ref-test-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        let origin = root.join("origin.git");
        let seed = root.join("seed");
        let installed = root.join("installed");
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "--bare", origin.to_str().unwrap()]);
        fs::create_dir_all(&seed).unwrap();
        git(&seed, &["init"]);
        git(&seed, &["config", "user.name", "hipfire test"]);
        git(
            &seed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        fs::write(seed.join("channel"), "master\n").unwrap();
        git(&seed, &["add", "channel"]);
        git(&seed, &["commit", "-m", "master"]);
        git(&seed, &["branch", "-M", "master"]);
        git(
            &seed,
            &["remote", "add", "origin", origin.to_str().unwrap()],
        );
        git(&seed, &["push", "-u", "origin", "master"]);
        git(&seed, &["checkout", "-b", "beta"]);
        fs::write(seed.join("channel"), "beta\n").unwrap();
        git(&seed, &["commit", "-am", "beta"]);
        git(&seed, &["push", "-u", "origin", "beta"]);
        git(
            &root,
            &[
                "clone",
                "--branch",
                "master",
                origin.to_str().unwrap(),
                installed.to_str().unwrap(),
            ],
        );

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Auto,
            },
        )
        .unwrap();
        assert_eq!(resolved.selector.kind, RevisionKind::Branch);
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );
        assert_eq!(
            fs::read_to_string(installed.join("channel")).unwrap(),
            "beta\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bind_and_pid_compatibility_parsers_cover_legacy_shapes() {
        assert_eq!(
            parse_bind(Some("127.0.0.1:12000"), None, "0.0.0.0", 11435).unwrap(),
            ("127.0.0.1".into(), 12000)
        );
        assert_eq!(
            parse_bind(Some("[::1]:12001"), None, "0.0.0.0", 11435).unwrap(),
            ("::1".into(), 12001)
        );
        let legacy = parse_pid_record("42\n").unwrap();
        assert_eq!(legacy.pid, 42);
        assert!(legacy.legacy);
        let json = parse_pid_record(r#"{"pid":43,"token":"old"}"#).unwrap();
        assert_eq!(json.pid, 43);
        assert_eq!(json.token.as_deref(), Some("old"));
        assert!(!json.legacy);
    }

    #[test]
    fn run_options_after_prompt_and_tui_passthrough_parse() {
        let cli =
            Cli::try_parse_from(["hipfire", "run", "qwen:test", "hello", "--max-tokens", "7"])
                .unwrap();
        let Some(Commands::Run(args)) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(args.prompt, ["hello"]);
        assert_eq!(args.max_tokens, Some(7));

        let cli = Cli::try_parse_from(["hipfire", "tui", "--check"]).unwrap();
        let Some(Commands::Tui(args)) = cli.command else {
            panic!("expected tui command");
        };
        assert_eq!(args.arguments, ["--check"]);
    }

    #[test]
    fn last_user_prompt_handles_text_parts() {
        let body = serde_json::json!({
            "messages": [
                { "role": "assistant", "content": "old" },
                { "role": "user", "content": [
                    { "type": "text", "text": "one" },
                    { "type": "text", "text": "two" }
                ] }
            ]
        });
        let messages = normalize_openai_messages(body.get("messages"));
        assert_eq!(last_user_prompt(&messages).as_deref(), Some("onetwo"));
    }

    #[test]
    fn openai_images_forward_one_base64_payload_and_reject_unsafe_shapes() {
        let messages = serde_json::json!([{
            "role": "user",
            "content": [
                { "type": "text", "text": "describe" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,YWJj" } }
            ]
        }]);
        assert_eq!(
            request_image_base64(Some(&messages)).unwrap().as_deref(),
            Some("YWJj")
        );
        let remote = serde_json::json!([{
            "role": "user",
            "content": [{ "type": "image_url", "image_url": { "url": "https://example/image.png" } }]
        }]);
        assert!(request_image_base64(Some(&remote))
            .unwrap_err()
            .to_string()
            .contains("remote"));
    }

    #[test]
    fn openai_messages_normalize_roles_content_and_tool_history() {
        let body = serde_json::json!({
            "messages": [
                { "role": "developer", "content": "system policy" },
                { "role": "user", "content": [
                    { "type": "text", "text": "first" },
                    { "type": "image_url", "image_url": { "url": "ignored" } },
                    { "type": "text", "text": " second" }
                ] },
                {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "tool reasoning",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"README.md\"}"
                        }
                    }]
                },
                { "role": "toolResult", "tool_call_id": "call_1", "content": "done" },
                { "role": "unsupported", "content": "drop me" }
            ]
        });
        let normalized = normalize_openai_messages(body.get("messages"));
        assert_eq!(normalized.as_array().unwrap().len(), 4);
        assert_eq!(normalized[0]["role"], "system");
        assert_eq!(normalized[1]["content"], "first second");
        assert_eq!(normalized[2]["content"], "");
        assert_eq!(normalized[2]["tool_plan"], "tool reasoning");
        assert_eq!(normalized[2]["tool_calls"][0]["name"], "read_file");
        assert_eq!(
            normalized[2]["tool_calls"][0]["arguments"],
            serde_json::json!({ "path": "README.md" })
        );
        assert_eq!(normalized[3]["role"], "tool");
        assert_eq!(normalized[3]["tool_call_id"], "call_1");
    }

    #[test]
    fn registry_system_prompt_is_injected_only_when_client_omits_one() {
        let mut messages = normalize_openai_messages(Some(&serde_json::json!([
            { "role": "user", "content": "hello" }
        ])));
        inject_default_system_message(&mut messages, Some("registry identity"));
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "registry identity");

        let mut messages = normalize_openai_messages(Some(&serde_json::json!([
            { "role": "developer", "content": "client policy" },
            { "role": "user", "content": "hello" }
        ])));
        inject_default_system_message(&mut messages, Some("registry identity"));
        assert_eq!(messages.as_array().unwrap().len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "client policy");
    }

    #[test]
    fn openai_assistant_history_strips_thinking_and_preserves_fallback_arguments() {
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "<think>private plan</think>\n\nvisible answer",
                "tool_calls": [{
                    "function": { "name": "broken", "arguments": "not-json" }
                }]
            }]
        });
        let normalized = normalize_openai_messages(body.get("messages"));
        assert_eq!(normalized[0]["content"], "visible answer");
        assert_eq!(normalized[0]["tool_plan"], "private plan");
        assert_eq!(
            normalized[0]["tool_calls"][0]["arguments"],
            serde_json::json!({ "_raw": "not-json" })
        );
    }

    #[test]
    fn jinja_started_think_routes_reasoning_then_visible_answer() {
        let mut router = ThinkChannelRouter::default();
        router.set_started_in_think(true);
        assert_eq!(
            router.push("reasoning body"),
            vec![ThinkFragment::Reasoning("reasoning body".into())]
        );
        assert!(router.push("</thi").is_empty());
        assert_eq!(
            router.push("nk>\n\nvisible answer"),
            vec![ThinkFragment::Content("visible answer".into())]
        );
        assert!(router.finish().is_empty());
    }

    #[test]
    fn plain_jinja_tail_keeps_output_in_content() {
        let mut router = ThinkChannelRouter::default();
        router.set_started_in_think(false);
        assert_eq!(
            router.push("direct answer"),
            vec![ThinkFragment::Content("direct answer".into())]
        );
    }

    #[test]
    fn model_literal_think_frames_route_consistently() {
        for family in ["qwen", "lfm", "minimax"] {
            let mut router = ThinkChannelRouter::default();
            router.set_started_in_think(true);
            let mut fragments = router.push(&format!("{family} reasoning</thi"));
            fragments.extend(router.push("nk>\n\nvisible<|im_"));
            fragments.extend(router.push("end|>"));
            fragments.extend(router.finish());
            assert_eq!(
                fragments,
                vec![
                    ThinkFragment::Reasoning(format!("{family} reasoning")),
                    ThinkFragment::Content("visible".into()),
                ],
                "{family}"
            );
        }
    }

    #[test]
    fn daemon_semantic_channels_override_literal_think_state() {
        for family in ["deepseek", "cohere"] {
            let mut router = ThinkChannelRouter::default();
            router.set_started_in_think(true);
            let mut fragments = router.push_semantic(&format!("{family} reason<|im_"), true);
            fragments.extend(router.push_semantic("end|>", true));
            fragments.extend(router.push("visible answer"));
            fragments.extend(router.finish());
            assert_eq!(
                fragments,
                vec![
                    ThinkFragment::Reasoning(format!("{family} reason")),
                    ThinkFragment::Content("visible answer".into()),
                ],
                "{family}"
            );
        }
    }

    #[test]
    fn output_router_removes_orphan_close_and_split_terminators() {
        let mut router = ThinkChannelRouter::default();
        let mut fragments = router.push("</thi");
        fragments.extend(router.push("nk>\n\nanswer<|endof"));
        fragments.extend(router.push("text|>tail"));
        fragments.extend(router.finish());
        assert_eq!(
            fragments,
            vec![
                ThinkFragment::Content("answer".into()),
                ThinkFragment::Content("tail".into()),
            ]
        );
    }

    #[test]
    fn positional_model_config_scope_parses_without_stealing_global_actions() {
        let global = Cli::try_parse_from(["hipfire", "config", "list", "--json"]).unwrap();
        let Some(Commands::Config(global)) = global.command else {
            panic!("expected config command")
        };
        assert!(global.model.is_none());
        assert!(matches!(global.action, Some(ConfigAction::List(_))));

        let model =
            Cli::try_parse_from(["hipfire", "config", "qwen:test", "get", "memory.kv_cache"])
                .unwrap();
        let Some(Commands::Config(model)) = model.command else {
            panic!("expected config command")
        };
        assert_eq!(model.model.as_deref(), Some("qwen:test"));
        assert!(matches!(model.action, Some(ConfigAction::Get { .. })));

        let schema = Cli::try_parse_from(["hipfire", "config", "schema", "--json"]).unwrap();
        let Some(Commands::Config(schema)) = schema.command else {
            panic!("expected config command")
        };
        assert!(schema.model.is_none());
        assert!(matches!(
            schema.action,
            Some(ConfigAction::Schema(OutputArgs { json: true }))
        ));
    }

    #[test]
    fn config_profile_set_and_create_parse_as_dedicated_actions() {
        let set = Cli::try_parse_from(["hipfire", "config", "profile", "set", "dev"]).unwrap();
        let Some(Commands::Config(args)) = set.command else {
            panic!("expected config command")
        };
        assert!(args.model.is_none());
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile {
                action: Some(ConfigProfileAction::Set { ref name })
            }) if name == "dev"
        ));

        let create =
            Cli::try_parse_from(["hipfire", "config", "profile", "create", "lab"]).unwrap();
        let Some(Commands::Config(args)) = create.command else {
            panic!("expected config command")
        };
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile {
                action: Some(ConfigProfileAction::Create { ref name })
            }) if name == "lab"
        ));

        let bare = Cli::try_parse_from(["hipfire", "config", "profile"]).unwrap();
        let Some(Commands::Config(args)) = bare.command else {
            panic!("expected config command")
        };
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile { action: None })
        ));
    }

    #[test]
    fn config_profile_helpers_replace_layer_and_are_global_only() {
        assert_eq!(CONFIG_PROFILE_NAMES, &["default", "dev", "hip", "redline"]);
        let root = env::temp_dir().join(format!("hipfire-cli-profile-{}", std::process::id()));
        let config_paths = ConfigPaths::under(&root);
        let mut layer = ConfigLayer::default();
        layer
            .set(
                "generation.temperature",
                hipfire_config::ConfigValue::Float(0.5),
            )
            .unwrap();
        apply_config_profile(&mut layer, &config_paths, "redline").unwrap();
        assert!(layer.get("generation.temperature").is_none());
        assert_eq!(
            layer.get("replay.backend"),
            Some(&hipfire_config::ConfigValue::String("redline".into()))
        );

        let model = Cli::try_parse_from([
            "hipfire",
            "config",
            "qwen:test",
            "profile",
            "set",
            "default",
        ])
        .unwrap();
        let Some(Commands::Config(args)) = model.command else {
            panic!("expected config command")
        };
        assert_eq!(args.model.as_deref(), Some("qwen:test"));
        assert!(matches!(
            args.action,
            Some(ConfigAction::Profile {
                action: Some(ConfigProfileAction::Set { .. })
            })
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn schema_json_preserves_default_types_and_validation_rules() {
        let bool_field = field("hardware.allow_mixed_arch").unwrap();
        assert_eq!(
            config_default_value(bool_field),
            hipfire_config::ConfigValue::Bool(false)
        );
        assert_eq!(config_rule_json(bool_field.rule)["type"], "boolean");

        let variant_field = field("diagnostic.kernel.rdna2_variant").unwrap();
        assert_eq!(
            config_default_value(variant_field),
            hipfire_config::ConfigValue::Null
        );
        assert_eq!(config_rule_json(variant_field.rule)["minimum"], 1);
        assert_eq!(config_rule_json(variant_field.rule)["maximum"], 5);
    }

    #[test]
    fn serve_accepts_legacy_positionals_and_native_overrides() {
        let parsed = Cli::try_parse_from([
            "hipfire",
            "serve",
            "qwen3.6:35b-a3b-mq4r",
            "127.0.0.1",
            "11520",
            "--kv-mode",
            "q8",
            "--kv-backend",
            "vmm",
            "--idle-timeout",
            "0",
            "--tp",
            "2",
        ])
        .unwrap();
        let Some(Commands::Serve(args)) = parsed.command else {
            panic!("expected serve command")
        };
        assert_eq!(args.positionals.len(), 3);
        assert_eq!(args.kv_mode.as_deref(), Some("q8"));
        assert_eq!(args.kv_backend.as_deref(), Some("vmm"));
        assert_eq!(args.idle_timeout, Some(0));
        assert_eq!(args.tp, Some(2));
    }

    #[test]
    fn request_sampling_omits_builtins_but_recovers_shadowed_registry_values() {
        let builtins = resolve(Vec::<NamedLayer>::new()).unwrap();
        assert_eq!(
            request_f64(&builtins, "generation.temperature", None).unwrap(),
            None
        );

        let mut registry = ConfigLayer::default();
        registry.set_cli("generation.temperature", "1.0").unwrap();
        registry.set_cli("generation.top_k", "40").unwrap();
        registry.set_cli("generation.min_p", "0.05").unwrap();
        registry
            .set_cli("generation.presence_penalty", "1.5")
            .unwrap();
        registry
            .set_cli("prompt.system", "registry identity")
            .unwrap();
        let mut global = ConfigLayer::default();
        global.set_cli("generation.temperature", "0.7").unwrap();
        global.set_cli("generation.top_k", "10").unwrap();
        global.set_cli("generation.min_p", "0.1").unwrap();
        global
            .set_cli("generation.presence_penalty", "0.5")
            .unwrap();
        global.set_cli("prompt.system", "global identity").unwrap();
        let resolved = resolve([
            NamedLayer {
                source: ConfigSource::RegistryModel {
                    tag: "qwen:test".into(),
                    revision: "v1".into(),
                },
                layer: registry,
            },
            NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: PathBuf::from("config.toml"),
                },
                layer: global,
            },
        ])
        .unwrap();
        assert_eq!(
            request_f64(&resolved, "generation.temperature", None).unwrap(),
            Some(1.0)
        );
        assert_eq!(
            request_f64(&resolved, "generation.temperature", Some(0.25)).unwrap(),
            Some(0.25)
        );
        assert_eq!(
            request_u64(&resolved, "generation.top_k", None).unwrap(),
            Some(40)
        );
        assert_eq!(
            request_f64(&resolved, "generation.min_p", None).unwrap(),
            Some(0.05)
        );
        assert_eq!(
            request_f64(&resolved, "generation.presence_penalty", None).unwrap(),
            Some(1.5)
        );
        assert_eq!(
            request_string(&resolved, "prompt.system", None).unwrap(),
            Some("registry identity".into())
        );
        assert_eq!(
            request_string(&resolved, "prompt.system", Some("explicit".into())).unwrap(),
            Some("explicit".into())
        );
    }

    #[test]
    fn process_config_projects_only_explicit_arch_sensitive_config() {
        const NAME: &str = "HIPFIRE_FP16";
        let builtins = resolve(Vec::<NamedLayer>::new()).unwrap();
        let process = hipfire_config::ProcessConfig::from_resolved(&builtins).unwrap();
        assert_eq!(process.legacy_value(NAME), None);

        let mut global = ConfigLayer::default();
        global.set_cli("kernel.fp16", "false").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(process.legacy_value(NAME).as_deref(), Some("0"));
    }

    #[test]
    fn process_config_projects_typed_scalar_and_variant_config() {
        const NAMES: &[&str] = &[
            "HIPFIRE_DEVICES",
            "HIPFIRE_UNIFORM_VRAM_TOLERANCE_GB",
            "HIPFIRE_GEMV_ROWS",
            "HIPFIRE_LM_HEAD_F16",
        ];
        let mut global = ConfigLayer::default();
        global.set_cli("hardware.devices", "2,3").unwrap();
        global
            .set_cli("hardware.uniform_vram_tolerance_gb", "1.5")
            .unwrap();
        global.set_cli("diagnostic.kernel.gemv_rows", "4").unwrap();
        global.set_cli("kernel.lm_head_f16", "f32").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: PathBuf::from("config.toml"),
            },
            layer: global,
        }])
        .unwrap();
        let process = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();
        assert_eq!(process.legacy_value(NAMES[0]).as_deref(), Some("2,3"));
        assert_eq!(process.legacy_value(NAMES[1]).as_deref(), Some("1.5"));
        assert_eq!(process.legacy_value(NAMES[2]).as_deref(), Some("4"));
        assert_eq!(process.legacy_value(NAMES[3]).as_deref(), Some("f32"));
    }

    #[test]
    fn http_reasoning_and_completion_metadata_match_native_contract() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut request = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "high" }),
            &resolved,
            &mut request,
        )
        .unwrap();
        assert_eq!(request["max_think_tokens"], 4096);

        let mut disabled = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": false }
            }),
            &resolved,
            &mut disabled,
        )
        .unwrap();
        assert_eq!(disabled["reasoning_effort"], "none");

        let completion = Completion {
            id: "chatcmpl_test".into(),
            created: 7,
            model: "qwen:test".into(),
            content: "answer".into(),
            reasoning_content: "reason".into(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "prompt_tokens": 12,
                "tokens": 7,
                "cached_tokens": 4,
                "ttft_ms": 8.5,
                "decode_tok_s": 115.0,
                "finish_reason": "stop"
            }),
        };
        let json = completion_json(&completion);
        assert_eq!(json["usage"]["total_tokens"], 19);
        assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(json["timings"]["decode_tok_s"], 115.0);
        assert_eq!(json["created"], 7);

        let qwen_cached = Completion {
            id: "chatcmpl_qwen_cached".into(),
            created: 8,
            model: "qwen:test".into(),
            content: "answer".into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "prefill_tokens": 8,
                "cached_tokens": 12,
                "tokens": 7
            }),
        };
        let qwen_json = completion_json(&qwen_cached);
        assert_eq!(qwen_json["usage"]["prompt_tokens"], 20);
        assert_eq!(qwen_json["usage"]["total_tokens"], 27);
        assert_eq!(
            qwen_json["usage"]["prompt_tokens_details"]["cached_tokens"],
            12
        );

        let preserved = Completion {
            preserve_thinking: true,
            reasoning_content: "private chain".into(),
            ..qwen_cached
        };
        let preserved_json = completion_json(&preserved);
        assert_eq!(
            preserved_json["choices"][0]["message"]["content"],
            "<think>private chain</think>\nanswer"
        );
        assert!(preserved_json["choices"][0]["message"]
            .get("reasoning_content")
            .is_none());
    }

    #[test]
    fn admission_queue_is_bounded_and_times_out() {
        let admission = Arc::new(Admission::new(1, Duration::from_millis(200)));
        let holder = admission.acquire().unwrap();
        let queued_admission = Arc::clone(&admission);
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let guard = queued_admission.acquire().unwrap();
            sender.send(()).unwrap();
            drop(guard);
        });
        for _ in 0..100 {
            if admission.inflight() == 2 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let saturated = admission.acquire().unwrap_err();
        assert!(saturated.message.contains("queue full"));
        drop(holder);
        receiver.recv_timeout(Duration::from_secs(1)).unwrap();
        waiter.join().unwrap();

        let admission = Arc::new(Admission::new(1, Duration::from_millis(5)));
        let _holder = admission.acquire().unwrap();
        let timeout = admission.acquire().unwrap_err();
        assert!(timeout.message.contains("wait exceeded"));
        assert_eq!(admission.inflight(), 1);
    }

    #[test]
    fn daemon_tool_calls_map_to_openai_shape() {
        let calls = vec![serde_json::json!({
            "name": "read_file",
            "arguments": { "path": "README.md" }
        })];
        let mapped = openai_tool_calls(&calls);
        assert_eq!(mapped[0]["type"], "function");
        assert_eq!(mapped[0]["function"]["name"], "read_file");
        assert_eq!(
            mapped[0]["function"]["arguments"],
            serde_json::json!(r#"{"path":"README.md"}"#)
        );
    }
}
