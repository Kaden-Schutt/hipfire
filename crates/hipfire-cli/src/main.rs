// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native hipfire control plane.
//!
//! This binary owns hipfire's operator surface and never shells out to a
//! JavaScript or TypeScript runtime.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use hipfire_client::{
    Engine, OpenAiSseEvent, complete_openai_chat, probe_host, service_ready, service_url,
    stream_openai_chat,
};
use hipfire_config::{
    CONFIG_SCHEMA_VERSION, CatalogFormat, ConfigFormat, ConfigLayer, ConfigPaths, ConfigSource,
    NamedLayer, ValueRule, apply_config_profile, canonical_config_key, create_config_profile,
    developer_env_for_key, field, fields, is_developer_key, load_catalog, load_env_layer,
    load_global, resolve, write_catalog_toml, write_global_toml,
};
use hipfire_registry::{
    LoadedRegistry, ModelEntry, RegistryPaths, RegistrySource, RegistryV1, load as load_registry,
};
use hipfire_runtime::prompt_frame::ToolCall;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

mod bench_concurrency;
mod setup;
use setup::setup_command;

const MODEL_SUFFIXES: &[&str] = &[
    ".hf4",
    ".hf6",
    ".hfq",
    ".mq2",
    ".mq2lloyd",
    ".mq2r",
    ".mq2rxt",
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
    /// Install or repair this machine's hipfire runtime.
    Setup(SetupArgs),
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

#[derive(Args, Debug, Default)]
struct SetupArgs {
    /// Source checkout to build from (set by scripts/install.sh).
    #[arg(long, value_name = "PATH")]
    source: PathBuf,
    #[arg(long, value_name = "PATH")]
    rocm_root: Option<PathBuf>,
    #[arg(long, value_name = "ARCH")]
    gpu_arch: Option<String>,
    /// auto (default) leaves replay.backend=auto so .mq4r models select Redline.
    #[arg(long, value_parser = ["auto", "hip", "redline"])]
    profile: Option<String>,
    #[arg(long, short = 'y', visible_alias = "non-interactive")]
    yes: bool,
    /// Requested revision ref forwarded by scripts/install.sh for install.json.
    #[arg(
        long = "ref",
        value_name = "REF",
        hide = true,
        conflicts_with_all = ["branch", "tag", "commit"]
    )]
    reference: Option<String>,
    /// Requested branch forwarded by scripts/install.sh for install.json.
    #[arg(
        long,
        value_name = "NAME",
        hide = true,
        conflicts_with_all = ["tag", "commit"]
    )]
    branch: Option<String>,
    /// Requested tag forwarded by scripts/install.sh for install.json.
    #[arg(long, value_name = "TAG", hide = true, conflicts_with = "commit")]
    tag: Option<String>,
    /// Requested commit forwarded by scripts/install.sh for install.json.
    #[arg(long, value_name = "SHA", hide = true)]
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

#[derive(Args, Debug, Clone)]
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
    /// Generated tokens per standard-bench measurement run.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
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
    /// Speculation mode to benchmark (off, dflash, mtp, ngram, dspark, or auto).
    #[arg(long = "spec")]
    speculation: Option<String>,
    /// Let the model think during the benchmark. Off by default: a reasoning
    /// model cannot close its `<think>` span inside the benchmark's token
    /// budget, and the daemon fails such a turn closed as a validation error.
    /// Pair this with `--max-tokens` large enough for the span to close.
    #[arg(long = "reasoning-on")]
    reasoning_on: bool,
    /// Sweep concurrent stream counts, e.g. `1,2,3,4`. Absent leaves bench
    /// on its single-stream path, unchanged.
    #[arg(long)]
    concurrency: Option<String>,
    /// Which concurrent backend to drive: slots, batch, or both.
    #[arg(long, value_parser = ["slots", "batch", "both"], default_value = "both")]
    backend: String,
    /// Which workload arm to run: stateless, multiturn, or both.
    #[arg(long, value_parser = ["stateless", "multiturn", "both"], default_value = "both")]
    workload: String,
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
    /// Maximum concurrent eligible batched lanes; 1 preserves sequential behavior.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=256))]
    continuous_batch_size: Option<u64>,
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
        Some(Commands::Setup(args)) => setup_command(&paths, args),
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
    if max_tokens == 0 || max_tokens > 393_216 {
        bail!("--max-tokens must be between 1 and 393216");
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
    project_dflash_draft(&mut params, developer_dflash_draft(&resolved));
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
        "attempt_id": next_attempt_id(),
        "prompt": prompt,
        "max_tokens": max_tokens,
        // `Engine::generate` rejects a request without `attempt_id`
        // (hipfire-client lib.rs:557 -> "generate request missing attempt_id"),
        // and `hipfire run` never set one, so EVERY `hipfire run` failed with a
        // daemon protocol error. `run` is a one-shot, non-retrying caller, so a
        // literal 1 is correct — same as `bench_generate_request` (main.rs:6407).
        // The retrying serve path threads a real counter instead (main.rs:4234).
        "attempt_id": 1,
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
        |event| {
            match event {
                OpenAiSseEvent::Reasoning { text } | OpenAiSseEvent::Content { text } => {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                OpenAiSseEvent::Role { .. }
                | OpenAiSseEvent::ToolCall { .. }
                | OpenAiSseEvent::Finish { .. }
                | OpenAiSseEvent::Usage { .. }
                | OpenAiSseEvent::Done => {}
            }
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
    if max_tokens == 0 || max_tokens > 393_216 {
        bail!("--max-tokens must be between 1 and 393216");
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
            continuous_batch_size: None,
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
            |event| {
                match event {
                    OpenAiSseEvent::Reasoning { text } | OpenAiSseEvent::Content { text } => {
                        assistant.push_str(&text);
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                    OpenAiSseEvent::Role { .. }
                    | OpenAiSseEvent::ToolCall { .. }
                    | OpenAiSseEvent::Finish { .. }
                    | OpenAiSseEvent::Usage { .. }
                    | OpenAiSseEvent::Done => {}
                }
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
    retries_attempted: u64,
    retries_succeeded: u64,
    recent_tok_s: Option<f64>,
    started: Instant,
    last_activity: Instant,
}

fn finish_prewarm(meta: &mut ServeMeta, succeeded: bool) {
    meta.loading_model = None;
    if succeeded {
        meta.last_activity = Instant::now();
    }
}

fn idle_model_expired(meta: &ServeMeta, idle_timeout: Duration) -> bool {
    meta.loading_model.is_none()
        && meta.current_model.is_some()
        && meta.last_activity.elapsed() >= idle_timeout
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
    current_arch: Option<String>,
    continuous_batch_capable: bool,
    current_max_seq: u64,
    cache_capable: bool,
    kv_override: Option<String>,
    kv_backend_override: Option<String>,
    tp: Option<u64>,
    continuous_batch_size: u64,
}

struct ServeShared {
    runtime: Mutex<ServeRuntime>,
    meta: Mutex<ServeMeta>,
    max_request_bytes: u64,
    admission: Arc<Admission>,
    idle_timeout: Duration,
    retry_enabled: bool,
    retry_backoff: Duration,
    /// Test seam: when set, invoked instead of `thread::sleep` during retry backoff.
    backoff_hook: Mutex<Option<Arc<dyn Fn(Duration) + Send + Sync>>>,
    /// Concurrent multi-slot backend (`serve.multi_slot`). Deliberately NOT
    /// inside `runtime`: that mutex is exactly what serialises requests today,
    /// so an engine behind it would serve one caller at a time and change
    /// nothing.
    slot_engine: Option<Arc<SlotBackend>>,
}

/// The multi-slot backend plus what the HTTP layer needs to talk to it: a
/// tokenizer to render chat messages into ids and to turn generated ids back
/// into text for the SSE deltas.
struct SlotBackend {
    engine: hipfire_arch_qwen35::serve_engine::SlotEngine,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
}

#[derive(Debug)]
struct Completion {
    id: String,
    created: u64,
    model: String,
    content: String,
    reasoning_content: String,
    preserve_thinking: bool,
    tool_calls: Vec<ToolCall>,
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
    /// Batch-eligible requests currently admitted. The single-daemon backend
    /// caps this at one; the multi-slot engine admits `capacity` at once.
    eligible: usize,
    /// A batch-ineligible request holds the backend exclusively.
    ineligible_busy: bool,
    queued: usize,
    batch_model: Option<String>,
}

#[derive(Debug)]
struct Admission {
    state: Mutex<AdmissionState>,
    available: Condvar,
    max_queue: usize,
    timeout: Duration,
    /// How many batch-eligible requests may be in flight at once. One for the
    /// single-daemon backend -- this gate is what protects it -- and the slot
    /// count when the multi-slot engine is active, which has its own admission
    /// behind it.
    capacity: usize,
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
    is_eligible: bool,
    model: Option<String>,
}

impl Admission {
    fn new(max_queue: usize, timeout: Duration) -> Self {
        Self::new_with_capacity(max_queue, timeout, 1)
    }

    fn new_with_capacity(max_queue: usize, timeout: Duration, capacity: usize) -> Self {
        Self {
            state: Mutex::new(AdmissionState::default()),
            available: Condvar::new(),
            max_queue,
            timeout,
            capacity: capacity.max(1),
        }
    }

    fn is_model_compatible(current: &Option<String>, requested: Option<&str>) -> bool {
        match (current, requested) {
            (None, _) => true,
            (Some(cur), Some(req)) => cur == req,
            (Some(_), None) => true,
        }
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn acquire(self: &Arc<Self>) -> std::result::Result<AdmissionGuard, AdmissionError> {
        self.acquire_for(false, None)
    }

    fn acquire_for(
        self: &Arc<Self>,
        is_eligible: bool,
        model: Option<&str>,
    ) -> std::result::Result<AdmissionGuard, AdmissionError> {
        let model_owned = model.map(|s| s.to_owned());
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        // Fast path when no queued waiters and resource is available.
        if is_eligible {
            if !state.ineligible_busy
                && state.eligible < self.capacity
                && state.queued == 0
                && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
            {
                state.eligible += 1;
                if state.batch_model.is_none() {
                    state.batch_model = model_owned.clone();
                }
                return Ok(AdmissionGuard {
                    admission: Arc::clone(self),
                    is_eligible: true,
                    model: model_owned,
                });
            }
        } else if state.eligible == 0 && !state.ineligible_busy && state.queued == 0 {
            state.ineligible_busy = true;
            return Ok(AdmissionGuard {
                admission: Arc::clone(self),
                is_eligible: false,
                model: None,
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
                if wait.timed_out() {
                    let can_acquire = if is_eligible {
                        !state.ineligible_busy
                            && state.eligible < self.capacity
                            && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
                    } else {
                        state.eligible == 0 && !state.ineligible_busy
                    };
                    if !can_acquire {
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
            }
            let can_acquire = if is_eligible {
                !state.ineligible_busy
                    && state.eligible < self.capacity
                    && Self::is_model_compatible(&state.batch_model, model_owned.as_deref())
            } else {
                state.eligible == 0 && !state.ineligible_busy
            };
            if can_acquire {
                state.queued = state.queued.saturating_sub(1);
                if is_eligible {
                    state.eligible += 1;
                    if state.batch_model.is_none() {
                        state.batch_model = model_owned.clone();
                    }
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: true,
                        model: model_owned.clone(),
                    });
                } else {
                    state.ineligible_busy = true;
                    return Ok(AdmissionGuard {
                        admission: Arc::clone(self),
                        is_eligible: false,
                        model: None,
                    });
                }
            }
        }
    }

    fn inflight(&self) -> usize {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.eligible + usize::from(state.ineligible_busy) + state.queued
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
        if self.is_eligible {
            state.eligible = state.eligible.saturating_sub(1);
            if state.eligible == 0 {
                state.batch_model = None;
            }
        } else {
            state.ineligible_busy = false;
        }
        self.admission.available.notify_all();
    }
}

/// Conservative batch eligibility for independent continuous-batch decode.
///
/// Eligible only for Qwen (arch 5/6) or dense LFM2 (`lfm` arch identity),
/// stateless text with no tools/images/stops/spec/adaptive/prefix behavior.
/// TP policy: Qwen admits ordinary tp=1 or pure expert-parallel tp=4 when the
/// daemon advertises batch capability; dense LFM remains tp=1 only. All other
/// tp/arch combinations fall back to sequential. Check is intentionally strict
/// and synchronous; model arch is taken from `current_arch` when available,
/// otherwise inferred from the requested model name containing `qwen` or `lfm`.
///
/// Message-shape gate matches daemon admission: absent/empty `messages` are
/// eligible; otherwise only exactly one `user` message with plain string
/// content (no tool_calls, no multipart/array content).
fn is_batch_eligible_request(
    body: &serde_json::Value,
    tp: Option<u64>,
    current_arch: Option<&str>,
    daemon_batch_capable: bool,
) -> bool {
    if !daemon_batch_capable {
        return false;
    }
    // Qwen or LFM2 only. Prefer runtime arch when known.
    let (is_qwen, is_lfm) = if let Some(arch) = current_arch {
        let arch_l = arch.to_ascii_lowercase();
        (arch_l.contains("qwen"), arch_l.contains("lfm"))
    } else if let Some(model) = body.get("model").and_then(|v| v.as_str()) {
        let model_l = model.to_ascii_lowercase();
        (model_l.contains("qwen"), model_l.contains("lfm"))
    } else {
        (false, false)
    };
    if !is_qwen && !is_lfm {
        return false;
    }
    // TP policy: Qwen tp=1 ordinary or tp=4 pure EP; dense LFM tp=1 only.
    let tp_degree = tp.unwrap_or(1);
    let tp_ok = if is_qwen {
        tp_degree == 1 || tp_degree == 4
    } else {
        tp_degree == 1
    };
    if !tp_ok {
        return false;
    }
    // Stateless text: no tools, no images, no stops, no spec, no adaptive, no prefix.
    if body
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        return false;
    }
    if body.get("tool_choice").is_some() {
        return false;
    }
    // Images via explicit image_base64.
    if body.get("image_base64").is_some() {
        return false;
    }
    // Message history must match daemon single-user plain-string shape.
    if !batch_messages_are_single_user(body) {
        return false;
    }
    if body.get("stop").is_some() {
        return false;
    }
    // Speculation / adaptive / prefix behavior disqualifies.
    for key in [
        "speculation",
        "dflash_mode",
        "mtp_mode",
        "ngram_draft",
        "prefill_sparse_threshold",
        "kv_adaptive",
        "prefix",
    ] {
        if body.get(key).is_some() {
            return false;
        }
    }
    true
}

/// HTTP/OpenAI `messages` are batch-eligible only when absent/empty (prompt
/// path) or exactly one user turn with plain string content. Multi-turn,
/// system/assistant/tool roles, tool_call payloads, and multipart/image
/// content stay on the sequential route — same contract as the daemon.
fn batch_messages_are_single_user(body: &serde_json::Value) -> bool {
    let Some(messages) = body.get("messages") else {
        return true;
    };
    let Some(arr) = messages.as_array() else {
        return false;
    };
    if arr.is_empty() {
        return true;
    }
    if arr.len() != 1 {
        return false;
    }
    let m0 = &arr[0];
    if m0.get("role").and_then(|v| v.as_str()) != Some("user") {
        return false;
    }
    // Tool-call payloads on the sole message force sequential (batch v1 has
    // no tools path). Empty / missing tool_calls is fine.
    if let Some(tc) = m0.get("tool_calls") {
        if tc.as_array().is_some_and(|a| !a.is_empty()) || tc.is_object() {
            return false;
        }
    }
    // Content must be a plain string (reject multipart/array/image parts).
    match m0.get("content") {
        Some(serde_json::Value::String(_)) => true,
        // Missing content is not a plain user string turn.
        None => false,
        // Arrays (multipart/text+image), objects, numbers, bool, null.
        Some(_) => false,
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
    if let Some(batch) = args.continuous_batch_size {
        command
            .arg("--continuous-batch-size")
            .arg(batch.to_string());
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
    let continuous_batch_size = args
        .continuous_batch_size
        .unwrap_or(config_u64(&global, "serve.continuous_batch_size")?);
    if continuous_batch_size == 0 || continuous_batch_size > 256 {
        bail!("--continuous-batch-size must be between 1 and 256");
    }
    let retry_enabled = config_bool(&global, "serve.retry_enabled")?;
    let retry_backoff = Duration::from_millis(config_u64(&global, "serve.retry_backoff_ms")?);
    let idle_timeout = Duration::from_secs(
        args.idle_timeout
            .unwrap_or(config_u64(&global, "serve.idle_timeout_seconds")?),
    );
    let default_model = args
        .model
        .clone()
        .unwrap_or(config_string(&global, "serve.default_model")?);
    let instance_token = serve_instance_token();
    // Opt-in concurrent backend. Built before ServeShared so a failure here is
    // a clean startup error rather than a half-configured server.
    let slot_engine = match config_bool(&global, "serve.multi_slot") {
        Ok(true) => {
            let model_path = find_model_path(paths, &registry, &default_model)
                .ok_or_else(|| anyhow!("multi_slot: cannot resolve model {default_model}"))?;
            let n_slots = config_u64(&global, "serve.multi_slot_slots").unwrap_or(4) as usize;
            let cap_tokens = config_u64(&global, "serve.multi_slot_ctx").unwrap_or(8192) as usize;
            let mut hfq = hipfire_runtime::hfq::HfqFile::open(&model_path)
                .with_context(|| format!("multi_slot: open {}", model_path.display()))?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                    .map_err(|e| anyhow!("multi_slot: tokenizer: {e}"))?;
            drop(hfq);
            let engine = hipfire_arch_qwen35::serve_engine::SlotEngine::spawn(
                hipfire_arch_qwen35::serve_engine::EngineConfig {
                    model_path: model_path.clone(),
                    n_slots,
                    cap_tokens,
                    host_budget_bytes: 16 * 1024 * 1024 * 1024,
                    swap_dir: std::env::temp_dir().join("hipfire-serve-swap"),
                },
            )
            .map_err(|e| anyhow!("multi_slot: {e}"))?;
            eprintln!(
                "serve: multi-slot backend up ({} slots, {} ctx) - requests run concurrently",
                n_slots, cap_tokens
            );
            Some(Arc::new(SlotBackend { engine, tokenizer }))
        }
        _ => None,
    };
    let slot_concurrency = if slot_engine.is_some() {
        config_u64(&global, "serve.multi_slot_slots").unwrap_or(4) as usize
    } else {
        1
    };
    let shared = Arc::new(ServeShared {
        slot_engine,
        runtime: Mutex::new(ServeRuntime {
            engine,
            paths: paths.clone(),
            registry: registry.clone(),
            current_path: None,
            current_arch: None,
            continuous_batch_capable: false,
            current_max_seq: 0,
            cache_capable: false,
            kv_override: args.kv_mode.clone(),
            kv_backend_override: args.kv_backend.clone(),
            tp: args.tp,
            continuous_batch_size,
        }),
        meta: Mutex::new(ServeMeta {
            current_model: None,
            loading_model: None,
            instance_token: instance_token.clone(),
            requests_served: 0,
            retries_attempted: 0,
            retries_succeeded: 0,
            recent_tok_s: None,
            started: Instant::now(),
            last_activity: Instant::now(),
        }),
        max_request_bytes,
        // The HTTP gate must admit as many requests as the live backend can
        // actually run, or they queue here and never reach it -- the exact
        // serialisation both batching paths exist to remove. Two independent
        // mechanisms feed this: the multi-slot engine's slot count (1 when it
        // is off) and the single-daemon continuous batch width. Take the
        // larger; each has its own admission behind this gate.
        admission: Arc::new(Admission::new_with_capacity(
            max_queue,
            queue_timeout,
            slot_concurrency.max(continuous_batch_size as usize),
        )),
        idle_timeout,
        retry_enabled,
        retry_backoff,
        backoff_hook: Mutex::new(None),
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
            {
                let mut meta = shared
                    .meta
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                finish_prewarm(&mut meta, result.is_ok());
            }
            match result {
                Ok(_) => eprintln!("[hipfire] pre-warmed {default_model}"),
                Err(error) => eprintln!("[hipfire] pre-warm failed: {error:#}; serving lazily"),
            }
        });
    }
    if !shared.idle_timeout.is_zero() {
        let shared = Arc::clone(&shared);
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(1));
                if shared.admission.inflight() != 0 {
                    continue;
                }
                let expired = {
                    let meta = shared
                        .meta
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    idle_model_expired(&meta, shared.idle_timeout)
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
                            runtime.current_arch = None;
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
                    "retries_attempted": meta.retries_attempted,
                    "retries_succeeded": meta.retries_succeeded,
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
            // Class-aware admission: eligible requests share capacity up to
            // continuous_batch_size, ineligible are exclusive single-flight.
            let (is_eligible, model_for_lease) = {
                let runtime = shared
                    .runtime
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let tp = runtime.tp;
                let arch = runtime.current_arch.clone();
                let batch_capable = runtime.continuous_batch_capable;
                drop(runtime);
                let eligible = is_batch_eligible_request(&body, tp, arch.as_deref(), batch_capable);
                let model = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());
                (eligible, model)
            };
            let guard = if is_eligible {
                match shared
                    .admission
                    .acquire_for(true, model_for_lease.as_deref())
                {
                    Ok(g) => g,
                    Err(e) => {
                        request.respond(admission_error_response(&e))?;
                        return Ok(());
                    }
                }
            } else {
                match shared.admission.acquire() {
                    Ok(g) => g,
                    Err(e) => {
                        request.respond(admission_error_response(&e))?;
                        return Ok(());
                    }
                }
            };
            // Tools require a lossless endpoint adapter before any generation.
            if let Err(error) = gate_chat_completions_tools(&body) {
                request.respond(openai_error(&error.to_string(), 400))?;
                return Ok(());
            }
            if body.get("stream").and_then(serde_json::Value::as_bool) == Some(true) {
                respond_streaming(request, shared, body, guard)?;
            } else {
                respond_nonstreaming(request, shared, body, guard)?;
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
        || lower.contains("endpoint adapter")
        || lower.contains("lossy")
        || lower.contains("malformed canonical tool call")
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
    let (sender, receiver) = mpsc::channel::<ResponseChunk>();
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
        let _ = sender.send(ResponseChunk::plain(sse_data(&first)));
        let result = complete_request(
            &shared,
            &body,
            guard,
            Some((id.clone(), created)),
            |event| forward_sse_stream_event(&sender, &id, created, &model, event),
            |completion| {
                // Full terminal representation before Engine can commit.
                deliver_sse_terminal_ack(&sender, completion, include_usage)
            },
        );
        finish_sse_stream(sender, result);
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

/// Non-stream OpenAI completion: stage the full JSON body before commit, then
/// wait for worker commit+done before EOF. Pre-terminal failures keep error status.
fn respond_nonstreaming(
    request: Request,
    shared: Arc<ServeShared>,
    body: serde_json::Value,
    guard: AdmissionGuard,
) -> Result<()> {
    let (sender, receiver) = mpsc::channel::<ResponseChunk>();
    let (status_tx, status_rx) = mpsc::channel::<Result<(), String>>();
    thread::spawn(move || {
        let result = complete_request(
            &shared,
            &body,
            guard,
            None,
            |_event| Ok(()),
            |completion| {
                let bytes = serde_json::to_vec(&completion_json(completion)).map_err(|err| {
                    hipfire_client::ClientError::Protocol(format!(
                        "completion json serialize failed: {err}"
                    ))
                })?;
                if bytes.is_empty() {
                    return Err(hipfire_client::ClientError::Protocol(
                        "nonstream terminal body must be non-empty".into(),
                    ));
                }
                let (ack_tx, ack_rx) = mpsc::channel();
                sender
                    .send(ResponseChunk {
                        bytes,
                        ack: Some(ack_tx),
                        fail: false,
                    })
                    .map_err(|_| hipfire_client::ClientError::Cancelled)?;
                // Signal handler that terminal bytes are staged (success headers).
                let _ = status_tx.send(Ok(()));
                match ack_rx.recv() {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
                }
            },
        );
        match result {
            Ok(_completion) => {
                // Terminal already delivered+acked; close body with no post-commit bytes.
                drop(sender);
            }
            Err(error) => {
                let cancelled = error
                    .downcast_ref::<hipfire_client::ClientError>()
                    .is_some_and(|err| matches!(err, hipfire_client::ClientError::Cancelled));
                if cancelled {
                    // Drop without framing — unclean only if bytes already went out.
                    drop(sender);
                    return;
                }
                // If terminal was never staged, report error status to the handler.
                let message = error.to_string();
                if status_tx.send(Err(message)).is_err() {
                    // Handler already started success body — force unclean close.
                    drop(sender);
                }
            }
        }
    });

    match status_rx.recv() {
        Ok(Ok(())) => {
            // Terminal body staged — success headers, reader owns JSON + waits for EOF.
            request.respond(Response::new(
                StatusCode(200),
                vec![
                    header("Content-Type", "application/json"),
                    header("Access-Control-Allow-Origin", "*"),
                ],
                ChannelReader::new(receiver),
                None,
                None,
            ))?;
        }
        Ok(Err(message)) => {
            request.respond(openai_error(&message, request_error_status(&message)))?;
        }
        Err(_) => {
            // Worker died before status — treat as internal failure.
            request.respond(openai_error("generation worker disconnected", 500))?;
        }
    }
    Ok(())
}

/// Convert a daemon v2 structured tool-call JSON object into canonical [`ToolCall`].
fn tool_call_from_canonical_value(value: &serde_json::Value) -> Result<ToolCall, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "tool call must be a JSON object".to_owned())?;
    let name = obj
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "tool call missing non-empty name".to_owned())?
        .to_owned();
    let arguments = obj
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Ok(ToolCall { name, arguments })
}

/// Convert a retained legacy completion-boundary tool-call JSON value into
/// canonical [`ToolCall`] without marker parsing.
fn tool_call_from_legacy_value(value: &serde_json::Value) -> Result<ToolCall, String> {
    // Legacy wire already used `{name, arguments}` objects (same shape as v2).
    // Keep an explicit boundary so legacy retention never reintroduces text scans.
    tool_call_from_canonical_value(value)
}

/// Endpoint adapter kinds known to the serve HTTP surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointAdapterKind {
    OpenAiChatCompletions,
}

/// Capability status of an endpoint adapter for non-empty tools requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointAdapterStatus {
    /// Adapter is present and preserves canonical tool-call semantics losslessly.
    AvailableLossless,
    /// No adapter is registered for this endpoint.
    Unavailable,
    /// Adapter exists but would drop or rewrite tool-call semantics.
    Lossy,
}

/// Pre-generation denial when tools are requested without a safe adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EndpointAdapterError {
    Unavailable { endpoint: &'static str },
    Lossy { endpoint: &'static str },
}

impl std::fmt::Display for EndpointAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable { endpoint } => {
                write!(f, "endpoint adapter unavailable for tools on {endpoint}")
            }
            Self::Lossy { endpoint } => {
                write!(f, "endpoint adapter lossy for tools on {endpoint}")
            }
        }
    }
}

impl std::error::Error for EndpointAdapterError {}

/// Typed registry of HTTP endpoint adapters and their tool-call capability.
struct EndpointAdapterRegistry;

impl EndpointAdapterRegistry {
    fn status(kind: EndpointAdapterKind) -> EndpointAdapterStatus {
        match kind {
            // OpenAI chat completions lowering is present and lossless for ToolCall.
            EndpointAdapterKind::OpenAiChatCompletions => EndpointAdapterStatus::AvailableLossless,
        }
    }
}

fn endpoint_adapter_status(kind: EndpointAdapterKind) -> EndpointAdapterStatus {
    EndpointAdapterRegistry::status(kind)
}

/// Gate `/v1/chat/completions` when the request carries a non-empty `tools` array.
/// Tool-free requests are unchanged. Adapter availability never overrides producer safety.
fn gate_chat_completions_tools(body: &serde_json::Value) -> Result<(), EndpointAdapterError> {
    let has_tools = body
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if !has_tools {
        return Ok(());
    }
    match endpoint_adapter_status(EndpointAdapterKind::OpenAiChatCompletions) {
        EndpointAdapterStatus::AvailableLossless => Ok(()),
        EndpointAdapterStatus::Unavailable => Err(EndpointAdapterError::Unavailable {
            endpoint: "/v1/chat/completions",
        }),
        EndpointAdapterStatus::Lossy => Err(EndpointAdapterError::Lossy {
            endpoint: "/v1/chat/completions",
        }),
    }
}

/// Errors from request+attempt correlated semantic event folding.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SemanticFoldError {
    /// Fold was used before `begin_attempt` established required ids.
    NoActiveAttempt,
    /// Event carried a different attempt id than the fold's active attempt.
    StaleAttempt { current: u64, got: u64 },
    /// Active attempt requires attempt_id on every subsequent event.
    MissingAttemptId { current: u64 },
    /// attempt_id was present but not a JSON number (u64 / non-neg i64).
    MalformedAttemptId { current: u64 },
    /// Event carried a different request id than the fold's active request.
    StaleRequestId { current: String, got: String },
    /// Active request requires nonempty string `id` on every subsequent event.
    MissingRequestId { current: String },
    /// `id` was present but empty or not a string.
    MalformedRequestId { current: String },
    /// Canonical tool-call payload failed structured conversion.
    MalformedToolCall { detail: String },
}

impl std::fmt::Display for SemanticFoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActiveAttempt => {
                write!(f, "semantic fold requires begin_attempt before events")
            }
            Self::StaleAttempt { current, got } => {
                write!(f, "stale attempt event: current={current} got={got}")
            }
            Self::MissingAttemptId { current } => {
                write!(f, "missing attempt_id on event for attempt {current}")
            }
            Self::MalformedAttemptId { current } => {
                write!(f, "malformed attempt_id on event for attempt {current}")
            }
            Self::StaleRequestId { current, got } => {
                write!(f, "stale request id: current={current} got={got}")
            }
            Self::MissingRequestId { current } => {
                write!(f, "missing request id on event for request {current}")
            }
            Self::MalformedRequestId { current } => {
                write!(f, "malformed request id on event for request {current}")
            }
            Self::MalformedToolCall { detail } => {
                write!(f, "malformed canonical tool call: {detail}")
            }
        }
    }
}

impl std::error::Error for SemanticFoldError {}

/// Attempt-local pure fold over daemon **contract v2** semantic JSON events.
///
/// Accumulates clean content/reasoning verbatim (no marker scanning), buffers
/// structured tool calls until a tool-safe done, preserves the daemon
/// finish_reason, and rejects stale/missing/malformed attempt correlation from
/// the first event. Never invokes [`ThinkChannelRouter`].
///
/// Activate only when `gen_start.contract_version == 2`. Legacy (non-v2)
/// MiniMax/Cohere raw-think streams stay on the explicit ThinkChannelRouter
/// path outside this type.
#[derive(Debug, Default)]
struct SemanticEventFold {
    content: String,
    reasoning_content: String,
    buffered_tool_calls: Vec<ToolCall>,
    current_request_id: Option<String>,
    current_attempt_id: Option<u64>,
    done: Option<serde_json::Value>,
}

impl SemanticEventFold {
    fn new() -> Self {
        Self::default()
    }

    /// Start (or restart) a correlated request+attempt, clearing attempt-local state.
    /// Must be called with the allocated wire ids before the first event.
    fn begin_attempt(&mut self, request_id: impl Into<String>, attempt_id: u64) {
        self.current_request_id = Some(request_id.into());
        self.current_attempt_id = Some(attempt_id);
        self.content.clear();
        self.reasoning_content.clear();
        self.buffered_tool_calls.clear();
        self.done = None;
    }

    fn current_request_id(&self) -> Option<&str> {
        self.current_request_id.as_deref()
    }

    fn current_attempt_id(&self) -> Option<u64> {
        self.current_attempt_id
    }

    fn content(&self) -> &str {
        &self.content
    }

    fn reasoning_content(&self) -> &str {
        &self.reasoning_content
    }

    fn buffered_tool_calls(&self) -> &[ToolCall] {
        &self.buffered_tool_calls
    }

    fn done(&self) -> Option<&serde_json::Value> {
        self.done.as_ref()
    }

    /// Executable calls after a tool-safe terminal; empty otherwise.
    /// Prefer canonical `calls` embedded on the staged/final done payload
    /// (authoritative). Fall back to mid-stream buffered events only when the
    /// terminal omits `calls` (legacy producers).
    fn executable_tool_calls(&self) -> &[ToolCall] {
        match self
            .done
            .as_ref()
            .and_then(|done| done.get("finish_reason"))
            .and_then(serde_json::Value::as_str)
        {
            // Only the daemon's tool_calls terminal may release calls.
            Some("tool_calls") => &self.buffered_tool_calls,
            _ => &[],
        }
    }

    /// Parse canonical `calls` from a staged/final done envelope.
    /// When `finish_reason=tool_calls`, missing/non-array/malformed fails closed.
    /// Other finish reasons ignore `calls` (leave buffer untouched for withhold).
    fn absorb_terminal_calls(&mut self, done: &serde_json::Value) -> Result<(), SemanticFoldError> {
        let finish = done
            .get("finish_reason")
            .and_then(serde_json::Value::as_str);
        if finish != Some("tool_calls") {
            return Ok(());
        }
        let Some(calls_val) = done.get("calls") else {
            return Err(SemanticFoldError::MalformedToolCall {
                detail: "tool_calls terminal requires `calls` array on staged done".to_owned(),
            });
        };
        let Some(calls) = calls_val.as_array() else {
            return Err(SemanticFoldError::MalformedToolCall {
                detail: "tool_calls terminal `calls` must be a JSON array".to_owned(),
            });
        };
        // Authoritative staged payload — replace any previously buffered calls
        // so we never duplicate mid-stream + terminal arrays.
        let mut parsed = Vec::with_capacity(calls.len());
        for call in calls {
            let tc = tool_call_from_canonical_value(call)
                .map_err(|detail| SemanticFoldError::MalformedToolCall { detail })?;
            parsed.push(tc);
        }
        self.buffered_tool_calls = parsed;
        Ok(())
    }

    /// Parse attempt_id: JSON numbers only (u64 or non-neg i64). Distinguishes
    /// missing vs malformed (string / null / negative / object).
    fn parse_event_attempt_id(event: &serde_json::Value) -> Result<Option<u64>, SemanticFoldError> {
        match event.get("attempt_id") {
            None => Ok(None),
            Some(value) => {
                if let Some(n) = value.as_u64() {
                    return Ok(Some(n));
                }
                if let Some(n) = value.as_i64() {
                    if n >= 0 {
                        return Ok(Some(n as u64));
                    }
                }
                // Present but not a usable number — caller maps to Malformed.
                Err(SemanticFoldError::MalformedAttemptId {
                    current: 0, // placeholder; check_correlation overwrites with active id
                })
            }
        }
    }

    /// Parse request `id`: nonempty JSON string only. Distinguishes missing vs
    /// malformed (empty string / non-string).
    fn parse_event_request_id(
        event: &serde_json::Value,
    ) -> Result<Option<String>, SemanticFoldError> {
        match event.get("id") {
            None => Ok(None),
            Some(value) => match value.as_str() {
                Some(s) if !s.is_empty() => Ok(Some(s.to_owned())),
                Some(_) | None => Err(SemanticFoldError::MalformedRequestId {
                    current: String::new(),
                }),
            },
        }
    }

    fn check_correlation(&self, event: &serde_json::Value) -> Result<(), SemanticFoldError> {
        let current_attempt = self
            .current_attempt_id
            .ok_or(SemanticFoldError::NoActiveAttempt)?;
        let current_request = self
            .current_request_id
            .as_deref()
            .ok_or(SemanticFoldError::NoActiveAttempt)?;

        match Self::parse_event_request_id(event) {
            Ok(None) => {
                return Err(SemanticFoldError::MissingRequestId {
                    current: current_request.to_owned(),
                });
            }
            Ok(Some(got)) if got != current_request => {
                return Err(SemanticFoldError::StaleRequestId {
                    current: current_request.to_owned(),
                    got,
                });
            }
            Ok(Some(_)) => {}
            Err(SemanticFoldError::MalformedRequestId { .. }) => {
                return Err(SemanticFoldError::MalformedRequestId {
                    current: current_request.to_owned(),
                });
            }
            Err(other) => return Err(other),
        }

        match Self::parse_event_attempt_id(event) {
            Ok(None) => Err(SemanticFoldError::MissingAttemptId {
                current: current_attempt,
            }),
            Ok(Some(got)) if got != current_attempt => Err(SemanticFoldError::StaleAttempt {
                current: current_attempt,
                got,
            }),
            Ok(Some(_)) => Ok(()),
            Err(SemanticFoldError::MalformedAttemptId { .. }) => {
                Err(SemanticFoldError::MalformedAttemptId {
                    current: current_attempt,
                })
            }
            Err(other) => Err(other),
        }
    }

    /// Fold one daemon v2 semantic event. Returns logical events the caller may
    /// forward (token/reasoning fragments, done, …). Structured `tool_calls`
    /// are buffered and never returned for mid-stream forwarding.
    ///
    /// Token and reasoning text are appended **verbatim** — no marker scan.
    fn push(
        &mut self,
        event: &serde_json::Value,
    ) -> Result<Vec<serde_json::Value>, SemanticFoldError> {
        self.check_correlation(event)?;
        let mut forward = Vec::new();
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("gen_start") => {
                // v2 channels are typed; started_in_think is ignored (no marker router).
            }
            Some("token") => {
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    // Daemon may tag reasoning on the token envelope; still verbatim.
                    if event.get("reasoning").and_then(serde_json::Value::as_bool) == Some(true) {
                        self.reasoning_content.push_str(text);
                        forward.push(serde_json::json!({ "type": "reasoning", "text": text }));
                    } else {
                        self.content.push_str(text);
                        forward.push(serde_json::json!({ "type": "token", "text": text }));
                    }
                }
            }
            Some("reasoning") => {
                if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                    self.reasoning_content.push_str(text);
                    forward.push(serde_json::json!({ "type": "reasoning", "text": text }));
                }
            }
            Some("tool_calls") => {
                let calls = event
                    .get("calls")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| SemanticFoldError::MalformedToolCall {
                        detail: "tool_calls event requires `calls` array".to_owned(),
                    })?;
                for call in calls {
                    let tc = tool_call_from_canonical_value(call)
                        .map_err(|detail| SemanticFoldError::MalformedToolCall { detail })?;
                    self.buffered_tool_calls.push(tc);
                }
                // Intentionally not forwarded — release only after tool-safe done.
            }
            Some("done") => {
                // Staged commit_ready is folded as type=done; absorb canonical
                // calls from the terminal payload before latching done.
                self.absorb_terminal_calls(event)?;
                self.done = Some(event.clone());
                forward.push(event.clone());
            }
            _ => {
                // Pass through unknown/control events (committed, error envelopes, …).
                forward.push(event.clone());
            }
        }
        Ok(forward)
    }
}

/// Allocate a fresh numeric generation attempt id (never 0 on success paths).
fn next_attempt_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Latched daemon event-contract for one `complete_request` stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamContract {
    /// Non-v2 / missing contract_version — ThinkChannelRouter path.
    Legacy,
    /// `gen_start.contract_version == 2` — SemanticEventFold path.
    V2,
}

/// Fail-closed stream framing / contract-selection errors for `complete_request`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamContractError {
    /// Stream opened with a non-`gen_start` event (no unchecked legacy default).
    PreStartEvent { event_type: String },
    /// More than one `gen_start` in a single attempt stream.
    SecondGenStart,
    /// `gen_start` lacked `attempt_id`.
    MissingAttemptId { expected: u64 },
    /// `gen_start.attempt_id` was present but not a usable number.
    MalformedAttemptId { expected: u64 },
    /// `gen_start.attempt_id` did not match the allocated wire id.
    StaleAttempt { expected: u64, got: u64 },
    /// `gen_start` lacked nonempty request `id`.
    MissingRequestId { expected: String },
    /// `gen_start.id` was present but empty or not a string.
    MalformedRequestId { expected: String },
    /// `gen_start.id` did not match the allocated wire request id.
    StaleRequestId { expected: String, got: String },
    /// Canonical tool-call payload failed structured conversion.
    MalformedToolCall { detail: String },
}

impl std::fmt::Display for StreamContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreStartEvent { event_type } => {
                write!(
                    f,
                    "stream must begin with gen_start; got {event_type} before contract latch"
                )
            }
            Self::SecondGenStart => {
                write!(f, "duplicate gen_start after contract already latched")
            }
            Self::MissingAttemptId { expected } => {
                write!(
                    f,
                    "gen_start missing attempt_id (expected {expected}); contract not latched"
                )
            }
            Self::MalformedAttemptId { expected } => {
                write!(
                    f,
                    "gen_start malformed attempt_id (expected {expected}); contract not latched"
                )
            }
            Self::StaleAttempt { expected, got } => {
                write!(
                    f,
                    "gen_start stale attempt_id: expected={expected} got={got}; contract not latched"
                )
            }
            Self::MissingRequestId { expected } => {
                write!(
                    f,
                    "gen_start missing request id (expected {expected}); contract not latched"
                )
            }
            Self::MalformedRequestId { expected } => {
                write!(
                    f,
                    "gen_start malformed request id (expected {expected}); contract not latched"
                )
            }
            Self::StaleRequestId { expected, got } => {
                write!(
                    f,
                    "gen_start stale request id: expected={expected} got={got}; contract not latched"
                )
            }
            Self::MalformedToolCall { detail } => {
                write!(f, "malformed canonical tool call: {detail}")
            }
        }
    }
}

impl std::error::Error for StreamContractError {}

/// One-shot contract latch for a generate stream.
///
/// Rules:
/// - first event must be exactly one `gen_start` with the expected numeric `attempt_id`
/// - correlation is validated **before** reading/latching `contract_version`
/// - legacy or v2 is latched once; a second `gen_start` is always rejected
/// - pre-start events never default to unchecked legacy
/// - after v2 is latched, nothing may switch the stream to legacy
#[derive(Debug)]
struct StreamContractGate {
    expected_request_id: String,
    expected_attempt_id: u64,
    latched: Option<StreamContract>,
}

impl StreamContractGate {
    fn new(expected_request_id: impl Into<String>, expected_attempt_id: u64) -> Self {
        Self {
            expected_request_id: expected_request_id.into(),
            expected_attempt_id,
            latched: None,
        }
    }

    fn contract(&self) -> Option<StreamContract> {
        self.latched
    }

    fn is_v2(&self) -> bool {
        self.latched == Some(StreamContract::V2)
    }

    /// Observe the next daemon event for framing/contract selection only.
    ///
    /// Returns the latched contract after a successful observe. Does not fold
    /// payloads — callers route to `SemanticEventFold` or legacy separately.
    fn observe(
        &mut self,
        event: &serde_json::Value,
    ) -> Result<StreamContract, StreamContractError> {
        let event_type = event
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<missing>");

        if let Some(contract) = self.latched {
            if event_type == "gen_start" {
                // Never re-latch / never allow a stale or missing-id start to
                // downgrade v2 → legacy (or flip legacy → v2).
                return Err(StreamContractError::SecondGenStart);
            }
            return Ok(contract);
        }

        // Unlatched: first event must be gen_start. No pre-start legacy default.
        if event_type != "gen_start" {
            return Err(StreamContractError::PreStartEvent {
                event_type: event_type.to_owned(),
            });
        }

        // Correlation BEFORE contract_version read/latch (exact id + attempt).
        let expected_request = self.expected_request_id.as_str();
        match SemanticEventFold::parse_event_request_id(event) {
            Ok(None) => {
                return Err(StreamContractError::MissingRequestId {
                    expected: expected_request.to_owned(),
                });
            }
            Ok(Some(got)) if got != expected_request => {
                return Err(StreamContractError::StaleRequestId {
                    expected: expected_request.to_owned(),
                    got,
                });
            }
            Ok(Some(_)) => {}
            Err(SemanticFoldError::MalformedRequestId { .. }) => {
                return Err(StreamContractError::MalformedRequestId {
                    expected: expected_request.to_owned(),
                });
            }
            Err(_) => {
                return Err(StreamContractError::MalformedRequestId {
                    expected: expected_request.to_owned(),
                });
            }
        }

        let expected = self.expected_attempt_id;
        match SemanticEventFold::parse_event_attempt_id(event) {
            Ok(None) => {
                return Err(StreamContractError::MissingAttemptId { expected });
            }
            Ok(Some(got)) if got != expected => {
                return Err(StreamContractError::StaleAttempt { expected, got });
            }
            Ok(Some(_)) => {}
            Err(SemanticFoldError::MalformedAttemptId { .. }) => {
                return Err(StreamContractError::MalformedAttemptId { expected });
            }
            Err(_) => {
                return Err(StreamContractError::MalformedAttemptId { expected });
            }
        }

        let contract = if event
            .get("contract_version")
            .and_then(serde_json::Value::as_u64)
            == Some(2)
        {
            StreamContract::V2
        } else {
            StreamContract::Legacy
        };
        self.latched = Some(contract);
        Ok(contract)
    }
}

/// Retry-disabling observations for one generation attempt.
///
/// Every event about to hit the client callback passes through [`Self::observe`]
/// (v2 fold logicals and legacy router fragments alike), so latching is
/// route-agnostic. `visible` matches exactly the wire-visible delta set of
/// [`openai_stream_delta_for_event`] (token/reasoning).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct AttemptLatches {
    visible: bool,
    commit_ready_seen: bool,
}

impl AttemptLatches {
    fn observe(&mut self, event: &serde_json::Value) {
        match event.get("type").and_then(serde_json::Value::as_str) {
            Some("token") | Some("reasoning") => self.visible = true,
            Some("commit_ready") => self.commit_ready_seen = true,
            _ => {}
        }
    }
}

/// Whether the failed attempt may be retried once server-side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetryDecision {
    Retry,
    Fail,
}

/// Single enforced retry-eligibility decision for the serve retry driver.
///
/// Retry iff ALL hold: gate enabled; first attempt; no visible token/reasoning
/// observed; commit_ready handshake never entered; daemon attested retry-reset
/// eligibility; the error is a typed daemon error of class `transient` with
/// `retryable=true` and an `attempt_id` matching the failed attempt. Every
/// other failure — malformed/validation/context/schema/tool/cancel classes,
/// callback/cancellation, I/O, EOF, invalid JSON, protocol/framing errors,
/// untyped legacy errors — fails closed with no retry.
fn decide_retry(
    error: &anyhow::Error,
    attempt_id: u64,
    latches: &AttemptLatches,
    eligible: bool,
    enabled: bool,
    attempt_index: u32,
) -> RetryDecision {
    if !enabled || attempt_index != 1 || latches.visible || latches.commit_ready_seen || !eligible {
        return RetryDecision::Fail;
    }
    let Some(hipfire_client::ClientError::Daemon(typed)) =
        error.downcast_ref::<hipfire_client::ClientError>()
    else {
        return RetryDecision::Fail;
    };
    if typed.class != hipfire_client::error_class::TRANSIENT
        || !typed.retryable
        || typed.attempt_id != attempt_id
    {
        return RetryDecision::Fail;
    }
    RetryDecision::Retry
}

/// Pure dual-route fold over a generate event sequence (test + `complete_request` core).
///
/// Applies [`StreamContractGate`] framing, then either [`SemanticEventFold`] (v2)
/// or legacy ThinkChannelRouter accumulation. Used by focused stream-contract
/// tests so production framing rules are exercised without a live Engine.
#[cfg(test)]
#[derive(Debug)]
struct FoldedStream {
    contract: StreamContract,
    content: String,
    reasoning_content: String,
    tool_calls: Vec<ToolCall>,
    done: Option<serde_json::Value>,
    #[allow(dead_code)]
    forwarded: Vec<serde_json::Value>,
}

#[cfg(test)]
fn fold_complete_request_stream(
    expected_request_id: &str,
    expected_attempt_id: u64,
    events: &[serde_json::Value],
) -> Result<FoldedStream, StreamContractError> {
    let mut fold = SemanticEventFold::new();
    fold.begin_attempt(expected_request_id, expected_attempt_id);
    let mut gate = StreamContractGate::new(expected_request_id, expected_attempt_id);
    let mut legacy_router = ThinkChannelRouter::default();
    let mut legacy_content = String::new();
    let mut legacy_reasoning = String::new();
    let mut legacy_tool_calls: Vec<ToolCall> = Vec::new();
    let mut legacy_done: Option<serde_json::Value> = None;
    let mut forwarded = Vec::new();

    for event in events {
        let contract = gate.observe(event)?;
        match contract {
            StreamContract::V2 => {
                // Map fold correlation errors onto stream framing errors for the
                // shared test surface (gate already validated gen_start).
                let logicals = fold.push(event).map_err(|err| match err {
                    SemanticFoldError::MissingAttemptId { current } => {
                        StreamContractError::MissingAttemptId { expected: current }
                    }
                    SemanticFoldError::MalformedAttemptId { current } => {
                        StreamContractError::MalformedAttemptId { expected: current }
                    }
                    SemanticFoldError::StaleAttempt { current, got } => {
                        StreamContractError::StaleAttempt {
                            expected: current,
                            got,
                        }
                    }
                    SemanticFoldError::MissingRequestId { current } => {
                        StreamContractError::MissingRequestId { expected: current }
                    }
                    SemanticFoldError::MalformedRequestId { current } => {
                        StreamContractError::MalformedRequestId { expected: current }
                    }
                    SemanticFoldError::StaleRequestId { current, got } => {
                        StreamContractError::StaleRequestId {
                            expected: current,
                            got,
                        }
                    }
                    SemanticFoldError::NoActiveAttempt => StreamContractError::PreStartEvent {
                        event_type: "no_active_attempt".into(),
                    },
                    SemanticFoldError::MalformedToolCall { detail } => {
                        StreamContractError::MalformedToolCall { detail }
                    }
                })?;
                for logical in logicals {
                    let ty = logical.get("type").and_then(serde_json::Value::as_str);
                    if ty == Some("done") || ty == Some("gen_start") {
                        continue;
                    }
                    forwarded.push(logical);
                }
            }
            StreamContract::Legacy => match event.get("type").and_then(serde_json::Value::as_str) {
                Some("gen_start") => {
                    if let Some(started) = event
                        .get("started_in_think")
                        .and_then(serde_json::Value::as_bool)
                    {
                        legacy_router.set_started_in_think(started);
                    }
                }
                Some("token") => {
                    if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                        let fragments =
                            if event.get("reasoning").and_then(serde_json::Value::as_bool)
                                == Some(true)
                            {
                                legacy_router.push_semantic(text, true)
                            } else {
                                legacy_router.push(text)
                            };
                        for fragment in fragments {
                            match fragment {
                                ThinkFragment::Content(t) => {
                                    legacy_content.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "token",
                                        "text": t
                                    }));
                                }
                                ThinkFragment::Reasoning(t) => {
                                    legacy_reasoning.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "reasoning",
                                        "text": t
                                    }));
                                }
                            }
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                        for fragment in legacy_router.push_semantic(text, true) {
                            match fragment {
                                ThinkFragment::Content(t) => {
                                    legacy_content.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "token",
                                        "text": t
                                    }));
                                }
                                ThinkFragment::Reasoning(t) => {
                                    legacy_reasoning.push_str(&t);
                                    forwarded.push(serde_json::json!({
                                        "type": "reasoning",
                                        "text": t
                                    }));
                                }
                            }
                        }
                    }
                }
                Some("tool_calls") => {
                    if let Some(calls) = event.get("calls").and_then(serde_json::Value::as_array) {
                        for call in calls {
                            let tc = tool_call_from_legacy_value(call).map_err(|detail| {
                                StreamContractError::MalformedToolCall { detail }
                            })?;
                            legacy_tool_calls.push(tc);
                        }
                    }
                }
                Some("done") => {
                    for fragment in legacy_router.finish() {
                        match fragment {
                            ThinkFragment::Content(t) => {
                                legacy_content.push_str(&t);
                                forwarded.push(serde_json::json!({
                                    "type": "token",
                                    "text": t
                                }));
                            }
                            ThinkFragment::Reasoning(t) => {
                                legacy_reasoning.push_str(&t);
                                forwarded.push(serde_json::json!({
                                    "type": "reasoning",
                                    "text": t
                                }));
                            }
                        }
                    }
                    legacy_done = Some(event.clone());
                }
                _ => {
                    forwarded.push(event.clone());
                }
            },
        }
    }

    let contract = gate
        .contract()
        .ok_or_else(|| StreamContractError::PreStartEvent {
            event_type: "<empty stream>".into(),
        })?;

    match contract {
        StreamContract::V2 => {
            let finish = fold
                .done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(serde_json::Value::as_str);
            let tool_calls = if finish == Some("tool_calls") {
                fold.executable_tool_calls().to_vec()
            } else {
                Vec::new()
            };
            Ok(FoldedStream {
                contract,
                content: fold.content().to_owned(),
                reasoning_content: fold.reasoning_content().to_owned(),
                tool_calls,
                done: fold.done().cloned(),
                forwarded,
            })
        }
        StreamContract::Legacy => {
            let finish = legacy_done
                .as_ref()
                .and_then(|d| d.get("finish_reason"))
                .and_then(serde_json::Value::as_str);
            let tool_calls = if finish == Some("tool_calls") {
                legacy_tool_calls
            } else {
                Vec::new()
            };
            Ok(FoldedStream {
                contract,
                content: legacy_content,
                reasoning_content: legacy_reasoning,
                tool_calls,
                done: legacy_done,
                forwarded,
            })
        }
    }
}

/// One correlated generation attempt under the shared serve runtime lock.
///
/// `identity` is the public completion identity (stable across retries);
/// `attempt_id` is the freshly allocated wire attempt id for this attempt.
/// `force_reset` (retry attempts) cold-resets before generate; a failed forced
/// reset poisons cached model state so the next request full-reloads.
/// `latches` records retry-disabling observations for the driver.
fn complete_request_attempt(
    shared: &ServeShared,
    body: &serde_json::Value,
    guard: AdmissionGuard,
    identity: &(String, u64),
    attempt_id: u64,
    force_reset: bool,
    latches: &std::cell::RefCell<AttemptLatches>,
    event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    // Latch retry-disabling observations on every event bound for the client.
    let mut event_callback = |event: &serde_json::Value| {
        latches.borrow_mut().observe(event);
        event_callback(event)
    };
    let model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("model is required"))?
        .to_owned();
    let image_base64 = request_image_base64(body.get("messages"))?;
    // Acquire runtime, ensure model, and build the generate request while
    // holding the lock. Clone the engine handle before dropping the lock so
    // concurrent eligible requests can share the multiplexed transport.
    let (generate, resolved, engine_clone) = {
        let mut runtime = shared
            .runtime
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Attempt id is allocated by the retry driver before any cold reset /
        // generate so reset ack, generate request, and the semantic fold share one
        // wire id.
        let resolved = runtime.ensure_model(&model, &shared.meta, None)?;
        if force_reset || (!runtime.cache_capable && !runtime.continuous_batch_capable) {
            if let Err(error) = runtime.engine.reset(attempt_id) {
                if force_reset {
                    // Rollback could not be attested: model state is unknown, so
                    // the next request must full-reload rather than trust it.
                    runtime.current_path = None;
                    runtime.current_arch = None;
                    runtime.continuous_batch_capable = false;
                    runtime.current_max_seq = 0;
                    runtime.cache_capable = false;
                }
                return Err(error.into());
            }
        }
        let max_tokens = body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(config_u64(&resolved, "generation.max_tokens")?);
        if max_tokens == 0 || max_tokens > 393_216 {
            bail!("max_tokens must be between 1 and 393216");
        }
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
            "attempt_id": attempt_id,
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
        let deepseek4_effort_contract = runtime.current_arch.as_deref() == Some("deepseek4");
        apply_http_reasoning_request(body, &resolved, &mut generate, deepseek4_effort_contract)?;
        let (id, created) = identity.clone();
        generate["id"] = serde_json::Value::String(id.clone());
        generate["attempt_id"] = serde_json::json!(attempt_id);
        if guard.is_eligible {
            generate["serve_continuous_batch"] = serde_json::Value::Bool(true);
        }
        let engine_clone = runtime.engine.clone();
        (generate, resolved, engine_clone)
    };
    let (id, created) = identity.clone();
    // Dual route gated by StreamContractGate:
    // - first event must be gen_start with matching request id + attempt_id
    // - contract_version is read only after correlation succeeds
    // - legacy/v2 latched once; second gen_start and pre-start events rejected
    // - v2 cannot be downgraded by a later stale/missing-id gen_start
    // - commit_ready is staged done: folded once via terminal_callback before commit
    let mut fold = SemanticEventFold::new();
    fold.begin_attempt(&id, attempt_id);
    let mut contract_gate = StreamContractGate::new(id.clone(), attempt_id);
    let mut legacy_router = ThinkChannelRouter::default();
    let mut legacy_content = String::new();
    let mut legacy_reasoning = String::new();
    let mut legacy_tool_calls: Vec<ToolCall> = Vec::new();
    let mut legacy_done: Option<serde_json::Value> = None;
    let mut terminal_delivered = false;
    let preserve_thinking = body
        .pointer("/chat_template_kwargs/preserve_thinking")
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    let model_for_fold = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();

    let gen_result = engine_clone.generate(&generate, |event| {
        let event_type = event.get("type").and_then(serde_json::Value::as_str);

        // Staged terminal: commit_ready carries done fields with type != done.
        // Fold/validate a type=done clone and deliver HTTP terminal before Ok.
        if event_type == Some("commit_ready") {
            latches.borrow_mut().commit_ready_seen = true;
            if terminal_delivered {
                return Err(hipfire_client::ClientError::Protocol(
                    "duplicate commit_ready".into(),
                ));
            }
            // Gate correlation on the raw envelope first (same id+attempt rules).
            let contract = contract_gate
                .observe(event)
                .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;

            let mut staged = event.clone();
            if let Some(obj) = staged.as_object_mut() {
                obj.insert("type".into(), serde_json::Value::String("done".into()));
            } else {
                return Err(hipfire_client::ClientError::Protocol(
                    "commit_ready must be a JSON object".into(),
                ));
            }

            let preview = match contract {
                StreamContract::V2 => {
                    let forward = fold.push(&staged).map_err(|error| {
                        hipfire_client::ClientError::Protocol(error.to_string())
                    })?;
                    // Staged done is held on the fold; do not forward mid-stream.
                    let _ = forward;
                    Completion {
                        id: id.clone(),
                        created,
                        model: model.clone(),
                        content: fold.content().to_owned(),
                        reasoning_content: fold.reasoning_content().to_owned(),
                        preserve_thinking,
                        tool_calls: fold.executable_tool_calls().to_vec(),
                        done: fold.done().cloned().unwrap_or(staged),
                    }
                }
                StreamContract::Legacy => {
                    forward_think_fragments(
                        legacy_router.finish(),
                        &mut legacy_content,
                        &mut legacy_reasoning,
                        &mut event_callback,
                    )?;
                    legacy_done = Some(staged.clone());
                    let finish = staged
                        .get("finish_reason")
                        .and_then(serde_json::Value::as_str);
                    let tool_calls = if finish == Some("tool_calls") {
                        legacy_tool_calls.clone()
                    } else {
                        Vec::new()
                    };
                    Completion {
                        id: id.clone(),
                        created,
                        model: model.clone(),
                        content: legacy_content.clone(),
                        reasoning_content: legacy_reasoning.clone(),
                        preserve_thinking,
                        tool_calls,
                        done: staged,
                    }
                }
            };

            terminal_callback(&preview)?;
            terminal_delivered = true;
            return Ok(());
        }

        let contract = contract_gate
            .observe(event)
            .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;

        match contract {
            StreamContract::V2 => {
                let forward = fold
                    .push(event)
                    .map_err(|error| hipfire_client::ClientError::Protocol(error.to_string()))?;
                for logical in forward {
                    // gen_start is consumed for latching; done is held on the fold.
                    // Post-commit done is not callback-visible from Engine, but
                    // still ignore if seen.
                    let ty = logical.get("type").and_then(serde_json::Value::as_str);
                    if ty == Some("done") || ty == Some("gen_start") {
                        continue;
                    }
                    event_callback(&logical)?;
                }
            }
            StreamContract::Legacy => {
                match event_type {
                    Some("gen_start") => {
                        if let Some(started) = event
                            .get("started_in_think")
                            .and_then(serde_json::Value::as_bool)
                        {
                            legacy_router.set_started_in_think(started);
                        }
                    }
                    Some("token") => {
                        if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                            let fragments =
                                if event.get("reasoning").and_then(serde_json::Value::as_bool)
                                    == Some(true)
                                {
                                    legacy_router.push_semantic(text, true)
                                } else {
                                    legacy_router.push(text)
                                };
                            forward_think_fragments(
                                fragments,
                                &mut legacy_content,
                                &mut legacy_reasoning,
                                &mut event_callback,
                            )?;
                        }
                    }
                    Some("reasoning") => {
                        if let Some(text) = event.get("text").and_then(serde_json::Value::as_str) {
                            let fragments = legacy_router.push_semantic(text, true);
                            forward_think_fragments(
                                fragments,
                                &mut legacy_content,
                                &mut legacy_reasoning,
                                &mut event_callback,
                            )?;
                        }
                    }
                    Some("tool_calls") => {
                        if let Some(calls) =
                            event.get("calls").and_then(serde_json::Value::as_array)
                        {
                            for call in calls {
                                let tc = tool_call_from_legacy_value(call).map_err(|detail| {
                                    hipfire_client::ClientError::Protocol(format!(
                                        "malformed canonical tool call: {detail}"
                                    ))
                                })?;
                                legacy_tool_calls.push(tc);
                            }
                        }
                    }
                    Some("done") => {
                        // Prefer staged commit_ready terminal; keep post-commit done
                        // only as payload fill if staging was skipped (legacy path).
                        if !terminal_delivered {
                            forward_think_fragments(
                                legacy_router.finish(),
                                &mut legacy_content,
                                &mut legacy_reasoning,
                                &mut event_callback,
                            )?;
                            legacy_done = Some(event.clone());
                        }
                    }
                    _ => {
                        event_callback(event)?;
                    }
                }
            }
        }
        Ok(())
    });
    let done = gen_result?;
    let mut meta = shared
        .meta
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    meta.requests_served = meta.requests_served.saturating_add(1);
    meta.recent_tok_s = done.get("tok_s").and_then(serde_json::Value::as_f64);
    meta.last_activity = Instant::now();

    if contract_gate.is_v2() {
        let done = fold.done().cloned().unwrap_or(done);
        return Ok(Completion {
            id,
            created,
            model,
            content: fold.content().to_owned(),
            reasoning_content: fold.reasoning_content().to_owned(),
            preserve_thinking,
            tool_calls: fold.executable_tool_calls().to_vec(),
            done,
        });
    }

    let done = legacy_done.unwrap_or(done);
    let finish = done
        .get("finish_reason")
        .and_then(serde_json::Value::as_str);
    let tool_calls = if finish == Some("tool_calls") {
        legacy_tool_calls
    } else {
        Vec::new()
    };
    Ok(Completion {
        id,
        created,
        model,
        content: legacy_content,
        reasoning_content: legacy_reasoning,
        preserve_thinking,
        tool_calls,
        done,
    })
}

/// Server-owned one-retry driver over [`complete_request_attempt`].
///
/// Disabled unless `serve.retry_enabled`; at most one retry; typed transient
/// daemon failures only; only before any visible token/reasoning delta or the
/// commit_ready terminal handshake; only after the daemon attested retry-reset
/// eligibility. The retry attempt performs a forced cold reset whose validated
/// ack is the synchronized matching rollback attestation, under the same
/// runtime lock acquisition as the re-generate. Backoff sleeps with neither
/// the runtime mutex nor an admission guard held (the failed attempt's guard
/// dropped with it); admission is re-acquired after the backoff, and a
/// re-acquire failure surfaces the original error. The public completion id is
/// allocated once and reused; attempt ids are distinct and monotonic.
/// Serve one chat completion through the multi-slot engine.
///
/// Renders the chat messages with the model's frame, submits, and turns the
/// engine's `Event`s into the same `{"type":"token","text":...}` events the
/// SSE writer already consumes -- so streaming, terminal acks and the HTTP
/// layer are all unchanged.
fn complete_request_slots(
    backend: &SlotBackend,
    body: &serde_json::Value,
    identity: &(String, u64),
    event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    use hipfire_runtime::emit_text::{ThinkOutputRouter, ThinkRouteEvent};
    use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame, Role};
    use hipfire_runtime::serve::{Event, SubmitRequest};

    let model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let max_tokens = body
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(512) as usize;

    // Messages -> the shape ChatFrame actually wants. This is not a free
    // mapping: `build_multi_turn` PANICS on a System or Tool role inside the
    // history, because system text belongs in `ChatFrame.system` and the final
    // user turn in `ChatFrame.user`. History carries only the prior
    // User/Assistant exchange.
    let mut system: Option<String> = None;
    let mut turns: Vec<(Role, String)> = Vec::new();
    let Some(msgs) = body.get("messages").and_then(serde_json::Value::as_array) else {
        bail!("messages is required");
    };
    for m in msgs {
        let text = m
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        match m.get("role").and_then(serde_json::Value::as_str) {
            Some("system") => {
                // Multiple system messages concatenate, matching how the
                // single-session path folds them.
                match system.as_mut() {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(&text);
                    }
                    None => system = Some(text),
                }
            }
            Some("assistant") => turns.push((Role::Assistant, text)),
            // A tool result reads as user-side context to the model.
            Some("tool") => turns.push((Role::User, text)),
            _ => turns.push((Role::User, text)),
        }
    }
    // The final user turn is the prompt; everything before it is history.
    let last_user = match turns.iter().rposition(|(r, _)| *r == Role::User) {
        Some(i) => turns.remove(i).1,
        None => bail!("messages must contain at least one user message"),
    };
    let history: Vec<(Role, &str)> = turns.iter().map(|(r, t)| (*r, t.as_str())).collect();
    // A thinking model must be handed an OPEN <think> block, as the daemon's
    // spec_assistant_prefix does. With Plain, the model has to emit `<think>`
    // itself and instead loops on `</think>` and re-opens user turns.
    let prefix = if backend.tokenizer.special_token_id("<think>").is_some() {
        AssistantPrefix::OpenThink
    } else {
        AssistantPrefix::Plain
    };
    let frame = ChatFrame {
        tokenizer: &backend.tokenizer,
        system: system.as_deref(),
        user: &last_user,
        assistant_prefix: prefix,
        raw: false,
    };
    let prompt_tokens = frame.build_multi_turn(&history);

    // Conversation identity is the USER turns. The assistant side is whatever
    // we generated, and the client's echo of it may differ (reasoning split to
    // its own channel, whitespace, an edited message), so it cannot be part of
    // the key.
    fn turn_hash(s: &str) -> u64 {
        let mut h = 0xcbf29ce484222325_u64;
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
    let mut convo: Vec<u64> = turns
        .iter()
        .filter(|(r, _)| *r == Role::User)
        .map(|(_, t)| turn_hash(t))
        .collect();
    convo.push(turn_hash(&last_user));

    // Tokens that continue the previous assistant turn into this user turn.
    // The engine appends these to the session's exact stored tokens instead of
    // re-rendering the history, which is what makes the result a strict
    // extension of the KV. Only meaningful from turn 2 onwards.
    let continuation = if convo.len() >= 2 {
        hipfire_runtime::prompt_frame::continuation_suffix(&backend.tokenizer, &last_user, prefix)
    } else {
        Vec::new()
    };

    let (tx, rx) = mpsc::channel::<Event>();
    backend
        .engine
        .submit(SubmitRequest {
            session: None,
            prompt_tokens,
            convo,
            continuation,
            max_tokens,
            reply: tx,
        })
        .map_err(|e| anyhow!("multi_slot submit: {e}"))?;

    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut finish = "stop";

    // Split <think> spans out of the visible answer. Without this the whole
    // reasoning preamble lands in `content` -- a 4B answer arrives as ~2 KB of
    // "Thinking Process: 1. **Analyze the Request**..." with reasoning_content
    // empty. `started_in_think` is true because the prompt itself opened the
    // block (AssistantPrefix::OpenThink above), so generation begins inside it
    // and there is no opening marker in the generated stream to detect.
    let mut think = ThinkOutputRouter::new(matches!(prefix, AssistantPrefix::OpenThink));
    let mut routed: Vec<ThinkRouteEvent> = Vec::new();

    let mut drain_routed = |routed: &mut Vec<ThinkRouteEvent>,
                            content: &mut String,
                            reasoning_content: &mut String,
                            cb: &mut dyn FnMut(
        &serde_json::Value,
    ) -> Result<(), hipfire_client::ClientError>| {
        for ev in routed.drain(..) {
            match ev {
                ThinkRouteEvent::Content(text) => {
                    content.push_str(&text);
                    let _ = cb(&serde_json::json!({ "type": "token", "text": text }));
                }
                ThinkRouteEvent::Reasoning(text) => {
                    reasoning_content.push_str(&text);
                    let _ = cb(&serde_json::json!({ "type": "reasoning", "text": text }));
                }
            }
        }
    };

    while let Ok(ev) = rx.recv() {
        match ev {
            Event::Accepted { .. } => {}
            Event::Token { id } => {
                let text = backend.tokenizer.decode(&[id]);
                if !text.is_empty() {
                    think.push_into(&text, &mut routed);
                    drain_routed(
                        &mut routed,
                        &mut content,
                        &mut reasoning_content,
                        event_callback,
                    );
                }
            }
            Event::Done { reason } => {
                finish = match reason {
                    hipfire_runtime::serve::DoneReason::MaxTokens => "length",
                    _ => "stop",
                };
                break;
            }
            Event::Rejected { reason } => {
                // Saturation is a real, expected answer here, not a crash: all
                // slots busy means the caller should retry, not that anything
                // failed.
                bail!("multi_slot rejected: {reason}");
            }
        }
    }

    // Flush any trailing partial marker as ordinary text in its channel.
    think.finish_into(&mut routed);
    drain_routed(
        &mut routed,
        &mut content,
        &mut reasoning_content,
        event_callback,
    );

    let completion = Completion {
        id: identity.0.clone(),
        created: identity.1,
        model,
        content,
        reasoning_content,
        preserve_thinking: false,
        tool_calls: Vec::new(),
        done: serde_json::json!({ "finish_reason": finish }),
    };
    // The terminal callback is what stages the response body and signals the
    // HTTP handler that the request succeeded. Skipping it leaves the handler
    // waiting on a status that never arrives, which surfaces to the client as
    // "generation worker disconnected".
    terminal_callback(&completion).map_err(|e| anyhow!("terminal callback: {e}"))?;
    Ok(completion)
}

fn complete_request(
    shared: &ServeShared,
    body: &serde_json::Value,
    guard: AdmissionGuard,
    request_identity: Option<(String, u64)>,
    mut event_callback: impl FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    mut terminal_callback: impl FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    let identity = request_identity.unwrap_or_else(|| (request_id(), unix_timestamp()));

    // Concurrent path. Takes no `shared.runtime` lock, which is the entire
    // point: that mutex is what makes concurrent callers queue today. The
    // retry/attempt machinery below is for the daemon Engine's cold-reset
    // semantics and has no analogue here -- the engine either admits a request
    // or rejects it with a reason.
    if let Some(backend) = shared.slot_engine.clone() {
        return complete_request_slots(
            &backend,
            body,
            &identity,
            &mut event_callback,
            &mut terminal_callback,
        );
    }

    let mut attempt_index = 1u32;
    let mut guard = guard;
    loop {
        let attempt_id = next_attempt_id();
        let latches = std::cell::RefCell::new(AttemptLatches::default());
        let outcome = complete_request_attempt(
            shared,
            body,
            guard,
            &identity,
            attempt_id,
            attempt_index > 1,
            &latches,
            &mut event_callback,
            &mut terminal_callback,
        );
        let latches = latches.into_inner();
        match outcome {
            Ok(completion) => {
                if attempt_index > 1 {
                    let mut meta = shared
                        .meta
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    meta.retries_succeeded = meta.retries_succeeded.saturating_add(1);
                }
                return Ok(completion);
            }
            Err(error) => {
                let eligible = {
                    let runtime = shared
                        .runtime
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    runtime.engine.last_retry_reset_eligible() == Some(true)
                };
                if decide_retry(
                    &error,
                    attempt_id,
                    &latches,
                    eligible,
                    shared.retry_enabled,
                    attempt_index,
                ) == RetryDecision::Fail
                {
                    return Err(error);
                }
                {
                    let mut meta = shared
                        .meta
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    meta.retries_attempted = meta.retries_attempted.saturating_add(1);
                }
                eprintln!(
                    "[hipfire] {}: typed transient daemon failure on attempt {attempt_index}; \
                     rolling back and retrying once",
                    identity.0
                );
                let hook = shared
                    .backoff_hook
                    .lock()
                    .ok()
                    .and_then(|guard| guard.clone());
                if let Some(hook) = hook {
                    hook(shared.retry_backoff);
                } else {
                    std::thread::sleep(shared.retry_backoff);
                }
                guard = match shared.admission.acquire() {
                    Ok(guard) => guard,
                    Err(_) => {
                        return Err(error.context("retry aborted: admission re-acquire failed"));
                    }
                };
                attempt_index = attempt_index.saturating_add(1);
            }
        }
    }
}

fn forward_think_fragments(
    fragments: Vec<ThinkFragment>,
    content: &mut String,
    reasoning_content: &mut String,
    event_callback: &mut impl FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
) -> Result<(), hipfire_client::ClientError> {
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

fn should_prewarm_qwen_mq4r_decode(
    path: &Path,
    loaded: &serde_json::Value,
    tp: Option<u64>,
) -> bool {
    let qwen = loaded
        .get("arch")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|arch| arch.starts_with("qwen3_5"));
    let mq4r = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mq4r"));
    qwen && mq4r && tp.unwrap_or(1) == 1
}

fn prewarm_qwen_mq4r_decode(engine: &mut Engine) -> Result<()> {
    let response = engine.request(&serde_json::json!({
        "type": "bench_decode",
        "context_tokens": 64,
        "iterations": 32,
    }))?;
    match response.get("type").and_then(serde_json::Value::as_str) {
        Some("decode_result") => {
            let tok_s = response
                .get("tok_s")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            eprintln!("[hipfire] pre-warmed Qwen MQ4R decode route ({tok_s:.1} tok/s)");
            Ok(())
        }
        Some("error") => bail!(
            "Qwen MQ4R decode pre-warm failed: {}",
            response
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("daemon returned an unspecified error")
        ),
        other => bail!(
            "Qwen MQ4R decode pre-warm expected decode_result, received {}",
            other.unwrap_or("missing type")
        ),
    }
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
            params["continuous_batch_size"] = serde_json::json!(self.continuous_batch_size);
            let loaded_max_seq = params["max_seq"].as_u64().unwrap_or(0);
            if minimum_max_seq.is_some() {
                eprintln!("[hipfire] bumping load max_seq to {loaded_max_seq} for request budget");
            }
            let loaded = self.engine.load(&path, params)?;
            if should_prewarm_qwen_mq4r_decode(&path, &loaded, self.tp) {
                prewarm_qwen_mq4r_decode(&mut self.engine)?;
            }
            self.cache_capable = loaded
                .get("cache_capable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.current_path = Some(path);
            self.current_arch = loaded
                .get("arch")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            self.continuous_batch_capable = loaded
                .get("continuous_batch_capable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
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

fn openai_finish_reason(done: &serde_json::Value) -> String {
    // Only an explicit raw daemon finish_reason string is authoritative.
    // Never synthesize "tool_calls" from buffered/leaked calls when the
    // terminal is missing, null, non-string, or any other unsafe value.
    match done
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        Some(reason) => reason.to_owned(),
        // Missing/null/non-string → fail closed to stop (not tool_calls).
        None => "stop".into(),
    }
}

fn completion_json(completion: &Completion) -> serde_json::Value {
    let finish_reason = openai_finish_reason(&completion.done);
    // Structured calls only for a tool-safe terminal; never on length/error/cancel.
    let tool_calls = if finish_reason == "tool_calls" {
        openai_tool_calls(&completion.tool_calls)
    } else {
        Vec::new()
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
    // Pure tool turns: OpenAI content is JSON null (not "").
    let content_value = if visible_content.is_empty() && !tool_calls.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(visible_content)
    };
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content_value,
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
        "hipfire": completion_hipfire(completion),
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
        "latency_ms": done.get("latency_ms"),
        "tau": done.get("tau"),
        "cycles": done.get("cycles"),
        "dflash": done.get("dflash"),
        "mtp": done.get("mtp"),
        "mtp_ngram": done.get("mtp_ngram"),
        "ngram_mod_windows": done.get("ngram_mod_windows"),
        "ngram_mod_drafts": done.get("ngram_mod_drafts"),
        "ngram_mod_accepted": done.get("ngram_mod_accepted"),
        "ngram_mod_accept_rate": done.get("ngram_mod_accept_rate"),
        "mtp_windows": done.get("mtp_windows"),
        "ar_windows": done.get("ar_windows"),
        "mtp_retired": done.get("mtp_retired"),
        "mtp_window_timings": done.get("mtp_window_timings"),
    })
}

/// Shared hipfire evidence projection for normal and streaming terminals.
fn completion_hipfire(completion: &Completion) -> serde_json::Value {
    let done = &completion.done;
    serde_json::json!({
        "tok_s": done.get("tok_s"),
        "prefill_tok_s": done.get("prefill_tok_s"),
        "decode_tok_s": done.get("decode_tok_s"),
        "execution_mode": done.get("execution_mode"),
        "continuous_batch": done.get("continuous_batch"),
    })
}

/// One OpenAI-lowered tool call from the shared canonical adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenAiToolCallAdapterResult {
    pub index: usize,
    pub id: String,
    pub name: String,
    /// JSON-text arguments (OpenAI wire requires a string, not an object).
    pub arguments: String,
}

/// Build the single shared OpenAI adapter result vector for a completion.
/// Deterministic response-scoped ids `call_{index}`; no filtering/dropping.
fn openai_tool_call_adapter_results(calls: &[ToolCall]) -> Vec<OpenAiToolCallAdapterResult> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| OpenAiToolCallAdapterResult {
            index,
            id: format!("call_{index}"),
            name: call.name.clone(),
            arguments: serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into()),
        })
        .collect()
}

/// Lower shared adapter results into OpenAI `message.tool_calls` objects.
fn openai_tool_calls_from_adapter(
    adapted: &[OpenAiToolCallAdapterResult],
) -> Vec<serde_json::Value> {
    adapted
        .iter()
        .map(|call| {
            serde_json::json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                }
            })
        })
        .collect()
}

/// Canonical → OpenAI non-stream tool_calls array (shared adapter path).
fn openai_tool_calls(calls: &[ToolCall]) -> Vec<serde_json::Value> {
    openai_tool_calls_from_adapter(&openai_tool_call_adapter_results(calls))
}

/// Map a folded callback event to an OpenAI stream delta.
/// Only clean content/reasoning are forwarded mid-stream; structured tool
/// calls release only via [`openai_stream_terminal_chunks`].
fn openai_stream_delta_for_event(event: &serde_json::Value) -> Option<serde_json::Value> {
    match event.get("type").and_then(serde_json::Value::as_str) {
        Some("token") => event
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(|text| serde_json::json!({ "content": text })),
        Some("reasoning") => event
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(|text| serde_json::json!({ "reasoning_content": text })),
        // tool_calls are released only after a tool-safe terminal verdict.
        Some("tool_calls") => None,
        _ => None,
    }
}

/// Lower shared adapter results into an OpenAI stream `delta` tool_calls object.
fn openai_tool_call_delta_from_adapter(
    adapted: &[OpenAiToolCallAdapterResult],
) -> serde_json::Value {
    serde_json::json!({
        "tool_calls": adapted
            .iter()
            .map(|call| {
                serde_json::json!({
                    "index": call.index,
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                })
            })
            .collect::<Vec<_>>()
    })
}

/// Post-completion SSE chunks: optional tool_calls release, terminal choice,
/// then optional separate `choices: []` usage chunk (never on the terminal).
fn openai_stream_terminal_chunks(
    completion: &Completion,
    include_usage: bool,
) -> Vec<serde_json::Value> {
    let finish_reason = openai_finish_reason(&completion.done);
    let mut chunks = Vec::new();

    // Release structured calls only on a tool-safe terminal.
    // Same adapter vector as non-stream `openai_tool_calls`.
    if finish_reason == "tool_calls" && !completion.tool_calls.is_empty() {
        let adapted = openai_tool_call_adapter_results(&completion.tool_calls);
        let delta = openai_tool_call_delta_from_adapter(&adapted);
        chunks.push(serde_json::json!({
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": null
            }],
        }));
    }

    chunks.push(serde_json::json!({
        "id": completion.id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
        "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
        "timings": completion_timings(completion),
        "hipfire": completion_hipfire(completion),
    }));

    if include_usage {
        chunks.push(serde_json::json!({
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [],
            "usage": completion_usage(completion),
        }));
    }

    chunks
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

/// Forward one logical generate event onto the OpenAI SSE channel.
///
/// Delta-bearing events serialize to plain (no-ack) SSE bytes. No-delta mid-stream
/// events (e.g. withheld tool_calls) are silent — terminal ack handles pure-tool
/// delivery. A dropped receiver maps to [`hipfire_client::ClientError::Cancelled`].
fn forward_sse_stream_event(
    sender: &mpsc::Sender<ResponseChunk>,
    id: &str,
    created: u64,
    model: &str,
    event: &serde_json::Value,
) -> Result<(), hipfire_client::ClientError> {
    if let Some(delta) = openai_stream_delta_for_event(event) {
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
        });
        sender
            .send(ResponseChunk::plain(sse_data(&chunk)))
            .map_err(|_| hipfire_client::ClientError::Cancelled)
    } else {
        // Mid-stream no-delta: do not queue empty probes. Terminal path acks.
        let _ = sender;
        Ok(())
    }
}

/// Serialize terminal tool_calls (if safe), finish, optional usage, and `[DONE]`
/// into one non-empty acknowledged chunk. Waits for ChannelReader progress ack.
fn deliver_sse_terminal_ack(
    sender: &mpsc::Sender<ResponseChunk>,
    completion: &Completion,
    include_usage: bool,
) -> Result<(), hipfire_client::ClientError> {
    let mut bytes = Vec::new();
    for chunk in openai_stream_terminal_chunks(completion, include_usage) {
        bytes.extend_from_slice(&sse_data(&chunk));
    }
    bytes.extend_from_slice(b"data: [DONE]\n\n");
    if bytes.is_empty() {
        return Err(hipfire_client::ClientError::Protocol(
            "stream terminal payload must be non-empty".into(),
        ));
    }
    let (ack_tx, ack_rx) = mpsc::channel();
    sender
        .send(ResponseChunk {
            bytes,
            ack: Some(ack_tx),
            fail: false,
        })
        .map_err(|_| hipfire_client::ClientError::Cancelled)?;
    match ack_rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) | Err(_) => Err(hipfire_client::ClientError::Cancelled),
    }
}

/// Close an OpenAI SSE body after `complete_request`.
///
/// Success: terminal already delivered+acked at commit_ready — emit no post-commit
/// bytes. Cancelled: no server_error/`[DONE]`. Post-terminal engine errors force
/// an unclean reader failure rather than appending a success/error frame.
fn finish_sse_stream(sender: mpsc::Sender<ResponseChunk>, result: Result<Completion>) {
    match result {
        Ok(_completion) => {
            // Terminal representation already went out before commit.
            drop(sender);
        }
        Err(error) => {
            let cancelled = error
                .downcast_ref::<hipfire_client::ClientError>()
                .is_some_and(|err| matches!(err, hipfire_client::ClientError::Cancelled));
            if cancelled {
                drop(sender);
                return;
            }
            eprintln!("[hipfire] streaming completion failed: {error:#}");
            // Unclean failure: poison the reader instead of framing success/error.
            let _ = sender.send(ResponseChunk {
                bytes: Vec::new(),
                ack: None,
                fail: false,
            });
            // Marker for reader: empty+no-ack is ignored; use fail signal via drop
            // after a special poison is not needed — ChannelReader fails when the
            // optional fail flag is set. Prefer ResponseChunk::fail.
            let _ = sender.send(ResponseChunk::fail());
            drop(sender);
        }
    }
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

/// One HTTP response body record. Optional `ack` is signaled only after the
/// reader fully drains `bytes` and the *next* `read` begins (proving writer
/// progress). Queue insertion alone never acknowledges. Drop before that next
/// read disconnects the waiter as Cancelled.
#[derive(Debug)]
struct ResponseChunk {
    bytes: Vec<u8>,
    ack: Option<mpsc::Sender<Result<(), ()>>>,
    /// When set, the next read fails uncleanly (post-terminal engine error).
    fail: bool,
}

impl ResponseChunk {
    fn plain(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            ack: None,
            fail: false,
        }
    }

    fn fail() -> Self {
        Self {
            bytes: Vec::new(),
            ack: None,
            fail: true,
        }
    }
}

struct ChannelReader {
    receiver: mpsc::Receiver<ResponseChunk>,
    current: std::io::Cursor<Vec<u8>>,
    /// Ack to fire on the *next* read after the current chunk is fully drained.
    pending_ack: Option<mpsc::Sender<Result<(), ()>>>,
    failed: bool,
}

impl ChannelReader {
    fn new(receiver: mpsc::Receiver<ResponseChunk>) -> Self {
        Self {
            receiver,
            current: std::io::Cursor::new(Vec::new()),
            pending_ack: None,
            failed: false,
        }
    }

    fn fire_pending_ack(&mut self) {
        if let Some(ack) = self.pending_ack.take() {
            let _ = ack.send(Ok(()));
        }
    }
}

impl Drop for ChannelReader {
    fn drop(&mut self) {
        // Drop before the next-read ack → waiter sees disconnect/Cancelled.
        if let Some(ack) = self.pending_ack.take() {
            let _ = ack.send(Err(()));
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.failed {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "response body failed after terminal delivery",
            ));
        }
        // Next-read after full drain acknowledges the prior chunk. Partial
        // reads must keep draining without firing the pending ack.
        if self.current.position() == self.current.get_ref().len() as u64 {
            self.fire_pending_ack();
        }

        loop {
            let read = self.current.read(output)?;
            if read > 0 {
                return Ok(read);
            }
            // Current buffer exhausted. Do not ack yet — ack waits for *next* read.
            match self.receiver.recv() {
                Ok(chunk) if chunk.fail => {
                    self.failed = true;
                    if let Some(ack) = chunk.ack {
                        let _ = ack.send(Err(()));
                    }
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "response body failed after terminal delivery",
                    ));
                }
                // Empty non-fail chunks are ignored (no ack on empty).
                Ok(chunk) if chunk.bytes.is_empty() => {
                    if let Some(ack) = chunk.ack {
                        // Empty acknowledged chunk is invalid — disconnect waiter.
                        let _ = ack.send(Err(()));
                    }
                    continue;
                }
                Ok(chunk) => {
                    // If a previous chunk still had a pending ack (shouldn't with
                    // single outstanding), fire it only on this next read entry —
                    // already fired at top. Stage this chunk's ack for the read
                    // *after* it is fully drained.
                    self.current = std::io::Cursor::new(chunk.bytes);
                    self.pending_ack = chunk.ack;
                }
                Err(_) => {
                    // Channel closed: any pending ack is a disconnect.
                    if let Some(ack) = self.pending_ack.take() {
                        let _ = ack.send(Err(()));
                    }
                    return Ok(0);
                }
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
        "deepseek4_compute_placement": config_string(
            resolved,
            "hardware.deepseek4_compute_placement",
        )?,
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
        "continuous_batch_size": config_u64(resolved, "serve.continuous_batch_size")?,
    });
    if let Some(experts_per_token) =
        config_optional_u64(resolved, "model.deepseek4_experts_per_token")?
    {
        params["deepseek4_experts_per_token"] = serde_json::json!(experts_per_token);
    }
    let selector = config_string(resolved, "speculation.mode")?;
    apply_speculation_selector(&mut params, &selector)?;
    project_dflash_draft(&mut params, developer_dflash_draft(resolved));
    Ok(params)
}

/// Project snapshotted `developer.dflash_draft` after the effective speculation selector.
///
/// Call only once final `dflash_mode` is known. Config-off must not carry a draft;
/// a later CLI selector (e.g. `run --spec dflash`) can opt back in here.
fn project_dflash_draft(params: &mut serde_json::Value, draft: Option<&str>) {
    if params["dflash_mode"].as_str() == Some("off") {
        if let Some(obj) = params.as_object_mut() {
            obj.remove("draft");
        }
        return;
    }
    if let Some(draft) = draft {
        if !draft.is_empty() {
            params["draft"] = serde_json::json!(draft);
        }
    }
}

/// Optional draft path from resolved `developer.dflash_draft` (legacy HIPFIRE_DFLASH_DRAFT).
fn developer_dflash_draft(resolved: &hipfire_config::ResolvedConfig) -> Option<&str> {
    match resolved
        .get("developer.dflash_draft")
        .map(|item| &item.value)
    {
        Some(hipfire_config::ConfigValue::String(value)) => Some(value.as_str()),
        _ => None,
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
    match config_string(resolved, "reasoning.effort")?.as_str() {
        "auto" => {}
        "none" => {
            request["max_think_tokens"] = serde_json::json!(1);
            request["assistant_prefix"] = serde_json::json!("closed_think");
            request["reasoning_effort"] = serde_json::json!("none");
        }
        effort @ ("low" | "high" | "max") => {
            request["reasoning_effort"] = serde_json::json!(effort);
        }
        effort => bail!("unknown reasoning effort '{effort}'"),
    }
    Ok(())
}

fn apply_http_reasoning_request(
    body: &serde_json::Value,
    resolved: &hipfire_config::ResolvedConfig,
    request: &mut serde_json::Value,
    deepseek4_effort_contract: bool,
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
    // Thinking is ON from here down (config default, or explicit
    // enable_thinking=true / reasoning effort >= minimal). OPEN the <think>
    // block so the model actually reasons instead of emitting an empty
    // <think></think> and answering directly.
    //
    // Only the thinking-OFF cases above used to set assistant_prefix; the ON
    // path set none, so the daemon fell back to plain framing and generic
    // OpenAI clients -- which never send assistant_prefix -- silently got
    // no-think behaviour. On Qwen3.6 that reasons in plain prose and derails
    // (LiveBench reasoning ~0).
    //
    // This was fixed in the TypeScript CLI in June 2026 (98c65020) and lost
    // when the CLI was rewritten in Rust: that commit edits cli/index.ts,
    // which no longer exists, so the fix cannot be cherry-picked -- only
    // re-applied here.
    //
    // Safe for non-thinking models: the daemon's prompt frame falls back to
    // Plain when the tokenizer has no `<think>` special token.
    request["assistant_prefix"] = serde_json::json!("open_think");
    if let Some(effort) = effort {
        if !deepseek4_effort_contract {
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
        let normalized = match effort {
            "minimal" | "medium" | "med" | "low" => "low",
            "high" => "high",
            "xhigh" | "max" | "uncapped" => "max",
            other => bail!("unknown reasoning effort '{other}'"),
        };
        let explicit_cap = body
            .get("max_think_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if explicit_cap > 393_216 {
            bail!("max_think_tokens must be between 0 and 393216");
        }
        // Effort selects the parent model's prompt semantics. It never invents
        // a hipfire token cap; absent an explicit cap, 0 means uncapped.
        request["max_think_tokens"] = serde_json::json!(explicit_cap);
        request["reasoning_effort"] = serde_json::json!(normalized);
        return Ok(());
    }
    apply_reasoning_request(resolved, request)?;
    if deepseek4_effort_contract
        && request
            .get("reasoning_effort")
            .and_then(serde_json::Value::as_str)
            != Some("none")
    {
        if let Some(explicit_cap) = body
            .get("max_think_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            if explicit_cap > 393_216 {
                bail!("max_think_tokens must be between 0 and 393216");
            }
            request["max_think_tokens"] = serde_json::json!(explicit_cap);
        }
    }
    Ok(())
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

fn config_optional_u64(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<Option<u64>> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Null => Ok(None),
        hipfire_config::ConfigValue::Integer(value) => u64::try_from(*value)
            .map(Some)
            .map_err(|_| anyhow!("{key} cannot be negative")),
        value => bail!(
            "{key} resolved as {}, expected integer or null",
            value.kind()
        ),
    }
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
    // --exp runs a fixed 128-token protocol across five RDNA2 variants and
    // ignores --max-tokens, so it can never give a think span room to close.
    // Reject the combination rather than silently drop the flag and abort
    // later on an open-think terminal.
    if args.exp && args.reasoning_on {
        bail!("--reasoning-on is not supported with --exp (its token budget is fixed at 128)");
    }
    if let Some(spec) = args.concurrency.clone() {
        return bench_concurrency_command(paths, &args, &spec);
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
        // Validate against the canonical `memory.kv_cache` schema instead of a
        // local subset. The old hardcoded list accepted only q8/fwht{2,3,4} and
        // so rejected `f32`/`f16` — the only KV formats DeepSeek V4 implements,
        // and precisely what the loader tells you to pass when it falls back
        // ("Pass --kv f32 for the golden configuration"). That made the advised
        // configuration unreachable through `bench`.
        let field = hipfire_config::field("memory.kv_cache")
            .ok_or_else(|| anyhow!("missing memory.kv_cache configuration field"))?;
        field
            .validate(&hipfire_config::ConfigValue::String(mode.to_owned()))
            .map_err(|err| anyhow!("--kv-mode {mode}: {err}"))?;
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
    eprintln!("  max_tokens: {}", args.max_tokens);
    if args.matrix || args.redline {
        bench_matrix(&mut engine, &args, &loaded, &post_diag)
    } else {
        // The warmup exists to populate kernel caches and its output is
        // discarded, so it stays in answer mode even under --reasoning-on: a
        // 16-token budget cannot close a think span, and letting the warmup
        // think would abort the run before a single measured sample.
        let _ = bench_generate(&mut engine, "Hello", 16)?;
        let mut decode = Vec::new();
        let mut prefill = Vec::new();
        let mut wall = Vec::new();
        let mut ttft = Vec::new();
        for _ in 0..args.runs {
            let done = bench_generate_with_reasoning(
                &mut engine,
                &prompt,
                args.max_tokens as u64,
                args.reasoning_on,
            )?;
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
            "max_tokens": args.max_tokens,
            "runs": args.runs,
            "batch": 1,
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

/// Concurrency sweep across both concurrent backends. Only reached when
/// `--concurrency` is given; the single-stream path above is untouched.
fn bench_concurrency_command(paths: &Paths, args: &BenchArgs, spec: &str) -> Result<()> {
    use crate::bench_concurrency::{
        parse_concurrency, render_table, sweep_backend, BackendSel, ConcurrencyBackend,
        DaemonDriver, Point, SlotDriver, WorkloadSel,
    };

    let points = parse_concurrency(spec)?;
    let max_k = *points.iter().max().expect("non-empty");
    let backend_sel = match args.backend.as_str() {
        "slots" => BackendSel::Slots,
        "batch" => BackendSel::Batch,
        _ => BackendSel::Both,
    };
    let arms: Vec<WorkloadSel> = match args.workload.as_str() {
        "stateless" => vec![WorkloadSel::Stateless],
        "multiturn" => vec![WorkloadSel::Multiturn],
        _ => vec![WorkloadSel::Stateless, WorkloadSel::Multiturn],
    };

    eprintln!("hipfire bench — concurrency sweep");
    eprintln!("  model:       {}", args.model);
    eprintln!("  concurrency: {points:?}");
    eprintln!("  runs/point:  {}", args.runs);
    eprintln!("  max_tokens:  {}", args.max_tokens);

    let mut out: Vec<Point> = Vec::new();

    // The two backends run STRICTLY SEQUENTIALLY, and the scope below is what
    // enforces it. Each holds a full copy of the model's weights -- 18.7 GB for
    // a 35B-A3B mq4r -- so overlapping them doubles resident footprint. On a
    // box with no swap that is not "slower", it is an OOM kill. `SlotEngine`'s
    // Drop closes its channel and joins the worker thread that owns the Gpu,
    // weights and KV arenas, so leaving this scope is what actually frees the
    // first model before the daemon loads the second.
    if matches!(backend_sel, BackendSel::Slots | BackendSel::Both) {
        let registry = load_registry(&paths.registry).registry;
        let model_path = find_model_path(paths, &registry, &args.model)
            .ok_or_else(|| anyhow!("model not found: {}", args.model))?;
        // 2048-token slots, not the serve default of 8192: the sweep's prompts
        // are one short turn and --max-tokens is small, so a larger arena buys
        // nothing and multiplies per-slot KV by four.
        match SlotDriver::start(&model_path, max_k, 2048) {
            Ok(mut d) => {
                eprintln!("  slots backend up ({max_k} slots)");
                let r = sweep_backend(
                    &mut d as &mut dyn ConcurrencyBackend,
                    &arms,
                    &points,
                    args.runs,
                    args.max_tokens as u64,
                    &mut out,
                );
                // Free the weights before the batch backend loads its own copy,
                // even on the error path.
                drop(d);
                r?;
            }
            // A backend that cannot run this model is a RESULT, not a crash.
            Err(e) => eprintln!("  slots backend unavailable: {e}"),
        }
    }

    if matches!(backend_sel, BackendSel::Batch | BackendSel::Both) {
        preflight_headroom_for_model(paths, &args.model)?;
        let mut batch_args = args.clone();
        batch_args.concurrency = None;
        let (engine, loaded, _, _) = open_bench_engine_batched(paths, &batch_args, max_k)?;
        let capable = loaded
            .get("continuous_batch_capable")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        match DaemonDriver::start(engine, max_k, capable) {
            Ok(mut d) => {
                eprintln!("  batch backend up (continuous_batch_size={max_k})");
                sweep_backend(
                    &mut d as &mut dyn ConcurrencyBackend,
                    &arms,
                    &points,
                    args.runs,
                    args.max_tokens as u64,
                    &mut out,
                )?;
            }
            Err(e) => eprintln!("  batch backend unavailable: {e}"),
        }
    }

    println!("{}", render_table(&out));
    Ok(())
}

/// Refuse to load a model that will not fit in available host memory.
///
/// The GPU allocates from system RAM on this class of box and there is no
/// swap, so an overcommit is an OOM kill of the whole machine rather than a
/// slow run. Checked between the two backends because that is precisely where
/// a leaked first model would show up: if the slots engine did not actually
/// release its weights, `MemAvailable` is still depressed here and this stops
/// the sweep instead of taking the box down.
fn preflight_headroom_for_model(paths: &Paths, model: &str) -> Result<()> {
    let registry = load_registry(&paths.registry).registry;
    let Some(path) = find_model_path(paths, &registry, model) else {
        return Ok(());
    };
    let Ok(meta) = std::fs::metadata(&path) else {
        return Ok(());
    };
    let need = meta.len();
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return Ok(());
    };
    let avail_kb = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let avail = avail_kb * 1024;
    // 20% headroom over the raw weight bytes for KV arenas and scratch.
    let want = need + need / 5;
    if avail < want {
        bail!(
            "refusing to load {}: needs ~{:.1} GB with headroom, only {:.1} GB available. \
             A previous backend may not have released its weights.",
            path.display(),
            want as f64 / 1e9,
            avail as f64 / 1e9
        );
    }
    Ok(())
}

/// `open_bench_engine`, but loading with `continuous_batch_size` set so the
/// daemon allocates batch lanes and advertises `continuous_batch_capable`.
/// The value is fixed per load, which is why the sweep holds it at max.
fn open_bench_engine_batched(
    paths: &Paths,
    args: &BenchArgs,
    batch_size: usize,
) -> Result<(
    Engine,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
)> {
    std::env::set_var("HIPFIRE_BENCH_CONTINUOUS_BATCH", batch_size.to_string());
    let r = open_bench_engine(paths, args, None);
    std::env::remove_var("HIPFIRE_BENCH_CONTINUOUS_BATCH");
    r
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
    if let Some(selector) = args.speculation.as_deref() {
        apply_speculation_selector(&mut params, selector)?;
    }
    if args.matrix || args.redline {
        let requested = longest_prefill.max(longest_decode).saturating_add(32);
        let configured = params["max_seq"].as_u64().unwrap_or(0);
        params["max_seq"] = serde_json::json!(configured.max(requested));
    }
    if let Ok(n) = std::env::var("HIPFIRE_BENCH_CONTINUOUS_BATCH") {
        if let Ok(n) = n.parse::<u64>() {
            params["continuous_batch_size"] = serde_json::json!(n);
        }
    }
    let loaded = engine.load(&path, params)?;
    let post_diag = engine.request(&serde_json::json!({ "type": "diag" }))?;
    Ok((engine, loaded, pre_diag, post_diag))
}

/// The standard benchmark generate: greedy, fixed budget, and **answer mode**.
///
/// Answer mode is the default rather than an opt-in because a benchmark that
/// lets the model think cannot complete. A reasoning model (any Qwen3.6 SKU,
/// for one) opens `<think>` within its first tokens and has no chance of
/// closing it inside the benchmark's budget — 16 tokens for the warmup, 128
/// for a measured run. The daemon ranks an unclosed think span at finish above
/// the length cap in both terminal classifiers (`QwenArTerminalCause::resolve`
/// and `qwen_dflash_wire_terminal`), so it reports the truncation as a
/// non-retryable validation error rather than `finish_reason=length`. The
/// benchmark then aborts on the warmup generate, before recording a sample.
///
/// Benchmarks measure tokens per second and never read the text, so asking for
/// answer mode costs nothing and removes the dependency on the model finishing
/// a thought inside an arbitrary budget. `--reasoning-on` restores the
/// thinking turn for anyone who wants to measure that path — with a budget
/// large enough to close the span.
fn bench_generate_request(prompt: &str, max_tokens: u64) -> serde_json::Value {
    bench_generate_request_reasoning(prompt, max_tokens, false)
}

fn bench_generate_request_reasoning(
    prompt: &str,
    max_tokens: u64,
    reasoning_on: bool,
) -> serde_json::Value {
    let mut request = serde_json::json!({
        "type": "generate",
        "id": request_id(),
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens,
        "attempt_id": 1,
    });
    if !reasoning_on {
        request["max_think_tokens"] = serde_json::json!(1);
        request["assistant_prefix"] = serde_json::json!("closed_think");
        request["reasoning_effort"] = serde_json::json!("none");
    }
    request
}

fn bench_generate(engine: &mut Engine, prompt: &str, max_tokens: u64) -> Result<serde_json::Value> {
    Ok(engine.generate(&bench_generate_request(prompt, max_tokens), |_| Ok(()))?)
}

fn bench_generate_with_reasoning(
    engine: &mut Engine,
    prompt: &str,
    max_tokens: u64,
    reasoning_on: bool,
) -> Result<serde_json::Value> {
    let request = bench_generate_request_reasoning(prompt, max_tokens, reasoning_on);
    Ok(engine.generate(&request, |_| Ok(()))?)
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
            max_tokens: 128,
            sustained_tg: None,
            sustained_ctx: vec![128],
            warmups: 1,
            kv_mode: None,
            kv_backend: None,
            redline: false,
            speculation: None,
            reasoning_on: false,
            concurrency: None,
            backend: "both".to_owned(),
            workload: "both".to_owned(),
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

/// Cooperative cancel flag for `hipfire update`. SIGINT/SIGTERM set this;
/// handlers never call `process::exit` so the armed rollback guard can run.
static UPDATE_INTERRUPT: AtomicBool = AtomicBool::new(false);

fn install_update_interrupt_handler() {
    // `termination` enables SIGTERM alongside SIGINT. Ignore AlreadyExists so a
    // pre-installed process handler does not abort update.
    let _ = ctrlc::set_handler(|| {
        UPDATE_INTERRUPT.store(true, Ordering::SeqCst);
    });
}

fn update_interrupted() -> bool {
    UPDATE_INTERRUPT.load(Ordering::SeqCst)
}

fn ensure_update_not_interrupted() -> Result<()> {
    if update_interrupted() {
        bail!("update interrupted");
    }
    Ok(())
}

fn update_command(paths: &Paths, args: UpdateArgs) -> Result<()> {
    if !cfg!(target_os = "linux") {
        bail!(
            "hipfire update is Linux-only; re-run the platform installer with a revision selector on this OS"
        );
    }
    // Install before any fetch/mutation so SIGINT cannot race past the guard.
    install_update_interrupt_handler();
    UPDATE_INTERRUPT.store(false, Ordering::SeqCst);

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

    ensure_update_not_interrupted()?;
    eprintln!(
        "fetching {} '{}' from origin...",
        selector.kind.label(),
        selector.value
    );
    let resolved = fetch_revision(&repo, selector)?;
    ensure_update_not_interrupted()?;
    let previous_head = git_output(&repo, &["rev-parse", "--verify", "HEAD"])?;
    let previous_branch = git_output(&repo, &["symbolic-ref", "--short", "HEAD"]).ok();
    let short = previous_head.get(..12).unwrap_or(&previous_head);
    let backup_ref = format!(
        "refs/hipfire/backups/pre-update-{}-{short}",
        unix_timestamp()
    );
    run_checked(
        Command::new("git")
            .current_dir(&repo)
            .args(["update-ref", &backup_ref, &previous_head]),
        "git update-ref backup",
    )?;
    eprintln!("previous source retained at {backup_ref}");

    let mut checkpoint = UpdateCheckpoint {
        head: previous_head,
        branch: previous_branch,
        stash_sha: None,
    };

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
        checkpoint.stash_sha = Some(git_output(&repo, &["rev-parse", "stash@{0}"])?);
        eprintln!("recover later with: git -C {} stash pop", repo.display());
    }

    // Armed after clean/stashed checkpoint and before checkout. Drop/error
    // rolls back unless explicitly committed after installer exit 0.
    let mut guard = UpdateRollbackGuard::arm(repo.clone(), checkpoint);
    ensure_update_not_interrupted()?;

    if let Err(err) = checkout_revision(&repo, &resolved) {
        return Err(guard.fail(err));
    }
    ensure_update_not_interrupted()?;

    match run_update_installer(&repo, paths, &resolved) {
        Ok(()) => {
            guard.commit();
            println!(
                "hipfire updated to {} '{}' ({})",
                resolved.selector.kind.label(),
                resolved.selector.value,
                resolved.commit
            );
            println!("verify with: hipfire version");
            Ok(())
        }
        Err(err) => Err(guard.fail(err)),
    }
}

#[derive(Debug, Clone)]
struct UpdateCheckpoint {
    head: String,
    branch: Option<String>,
    stash_sha: Option<String>,
}

/// Restores pre-update checkout/stash unless [`Self::commit`] is called after
/// a successful installer handoff. Drop and explicit fail both roll back.
struct UpdateRollbackGuard {
    repo: PathBuf,
    checkpoint: UpdateCheckpoint,
    armed: bool,
}

impl UpdateRollbackGuard {
    fn arm(repo: PathBuf, checkpoint: UpdateCheckpoint) -> Self {
        Self {
            repo,
            checkpoint,
            armed: true,
        }
    }

    fn commit(&mut self) {
        self.armed = false;
    }

    #[cfg(test)]
    fn is_armed(&self) -> bool {
        self.armed
    }

    fn fail(mut self, err: anyhow::Error) -> anyhow::Error {
        if let Err(restore_err) = self.rollback() {
            eprintln!(
                "WARNING: failed to restore pre-update checkout after failure: {restore_err}"
            );
            return err.context(format!("pre-update restore also failed: {restore_err}"));
        }
        err
    }

    fn rollback(&mut self) -> Result<()> {
        if !self.armed {
            return Ok(());
        }
        self.armed = false;
        restore_update_checkpoint(&self.repo, &self.checkpoint)
    }
}

impl Drop for UpdateRollbackGuard {
    fn drop(&mut self) {
        if let Err(err) = self.rollback() {
            eprintln!("WARNING: failed to restore pre-update checkout on drop: {err}");
        }
    }
}

fn run_update_installer(repo: &Path, paths: &Paths, resolved: &ResolvedRevision) -> Result<()> {
    let installer = repo.join("scripts/install.sh");
    if !installer.is_file() {
        bail!("updated checkout has no {}", installer.display());
    }
    let recorded = recorded_install_metadata(&paths.root);
    let mut installer_cmd = Command::new("bash");
    installer_cmd
        .arg(&installer)
        .current_dir(repo)
        .env("HIPFIRE_FORCE_REBUILD", "1");
    #[cfg(unix)]
    {
        // Own process group so SIGTERM/KILL can reach the whole installer tree.
        installer_cmd.process_group(0);
    }
    for arg in installer_handoff_args(
        &resolved.selector,
        recorded.rocm_root.as_deref(),
        recorded.gpu_arch.as_deref(),
    ) {
        installer_cmd.arg(arg);
    }
    run_update_installer_child(installer_cmd)
}

/// Spawn the installer, poll-wait, and on interrupt TERM then KILL the group.
fn run_update_installer_child(mut installer_cmd: Command) -> Result<()> {
    let mut child = installer_cmd
        .spawn()
        .context("failed to start native installer")?;
    let status = wait_update_installer_child(&mut child)?;
    if update_interrupted() {
        bail!("update interrupted");
    }
    if status.success() {
        Ok(())
    } else {
        bail!("native installer failed with {status}")
    }
}

fn wait_update_installer_child(child: &mut Child) -> Result<std::process::ExitStatus> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                if update_interrupted() {
                    terminate_update_installer_group(child);
                    return child
                        .wait()
                        .context("failed to reap interrupted native installer");
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(err) => {
                return Err(err).context("failed to wait for native installer");
            }
        }
    }
}

/// TERM the installer process group, then bounded KILL fallback; always wait/reap.
fn terminate_update_installer_group(child: &mut Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        if pid > 0 {
            // Negative pid targets the process group created via process_group(0).
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_secs(2);
            while Instant::now() < deadline {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) => thread::sleep(Duration::from_millis(50)),
                    Err(_) => break,
                }
            }
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
}

/// Args forwarded to scripts/install.sh during noninteractive update handoff.
fn installer_handoff_args(
    selector: &RevisionSelector,
    rocm_root: Option<&Path>,
    gpu_arch: Option<&str>,
) -> Vec<String> {
    let mut args = vec!["--yes".to_owned()];
    match selector.kind {
        RevisionKind::Auto => {
            args.push("--ref".to_owned());
            args.push(selector.value.clone());
        }
        RevisionKind::Branch => {
            args.push("--branch".to_owned());
            args.push(selector.value.clone());
        }
        RevisionKind::Tag => {
            args.push("--tag".to_owned());
            args.push(selector.value.clone());
        }
        RevisionKind::Commit => {
            args.push("--commit".to_owned());
            args.push(selector.value.clone());
        }
    }
    if let Some(root) = rocm_root {
        args.push("--rocm-root".to_owned());
        args.push(root.to_string_lossy().into_owned());
    }
    if let Some(arch) = gpu_arch.map(str::trim).filter(|arch| !arch.is_empty()) {
        args.push("--gpu-arch".to_owned());
        args.push(arch.to_owned());
    }
    args
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct RecordedInstallMetadata {
    rocm_root: Option<PathBuf>,
    gpu_arch: Option<String>,
}

fn recorded_install_metadata(install_home: &Path) -> RecordedInstallMetadata {
    let text = match fs::read_to_string(install_home.join("install.json")) {
        Ok(text) => text,
        Err(_) => return RecordedInstallMetadata::default(),
    };
    let value: serde_json::Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(_) => return RecordedInstallMetadata::default(),
    };
    let rocm_root = value
        .get("rocm_root")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|root| !root.is_empty())
        .map(PathBuf::from);
    let gpu_arch = value
        .get("gpu_arch")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|arch| !arch.is_empty())
        .map(str::to_owned);
    RecordedInstallMetadata {
        rocm_root,
        gpu_arch,
    }
}

fn restore_update_checkpoint(repo: &Path, checkpoint: &UpdateCheckpoint) -> Result<()> {
    // Failed-target Cargo/source mutations are wiped only after the original
    // user work is already in the update stash, so restore cannot be blocked.
    run_checked(
        Command::new("git")
            .current_dir(repo)
            .args(["reset", "--hard"]),
        "git reset failed target",
    )?;
    run_checked(
        Command::new("git").current_dir(repo).args(["clean", "-fd"]),
        "git clean failed target",
    )?;
    if let Some(branch) = checkpoint.branch.as_deref() {
        run_checked(
            Command::new("git").current_dir(repo).args([
                "checkout",
                "-B",
                branch,
                &checkpoint.head,
            ]),
            "git restore previous branch",
        )?;
    } else {
        run_checked(
            Command::new("git")
                .current_dir(repo)
                .args(["checkout", "--detach", &checkpoint.head]),
            "git restore previous commit",
        )?;
    }
    if let Some(stash_sha) = checkpoint.stash_sha.as_deref() {
        reapply_update_stash(repo, stash_sha)?;
    }
    Ok(())
}

fn reapply_update_stash(repo: &Path, stash_sha: &str) -> Result<()> {
    // Preserve staged index state from the original dirty tree.
    if let Err(err) = run_checked(
        Command::new("git")
            .current_dir(repo)
            .args(["stash", "apply", "--index", stash_sha]),
        "git stash apply --index",
    ) {
        bail!(
            "failed to restore pre-update stash {stash_sha} (kept for recovery: \
             git -C {} stash apply --index {stash_sha}): {err}",
            repo.display()
        );
    }
    // Drop only after successful apply so a conflicted restore keeps the stash.
    if let Ok(list) = git_output(repo, &["stash", "list", "--format=%gd %H"]) {
        for line in list.lines() {
            let mut parts = line.split_whitespace();
            let Some(gd) = parts.next() else {
                continue;
            };
            let Some(hash) = parts.next() else {
                continue;
            };
            if hash == stash_sha || stash_sha.starts_with(hash) || hash.starts_with(stash_sha) {
                let _ = run_checked(
                    Command::new("git")
                        .current_dir(repo)
                        .args(["stash", "drop", gd]),
                    "git stash drop",
                );
                break;
            }
        }
    }
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
                    .args(["fetch", "origin", &refspec]),
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
        refuse_unpushed_branch_commits(repo, &resolved.selector.value, tracking)?;
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

/// Refuse to reset a local branch that still has commits not present on the
/// resolved remote-tracking tip. Channel switches onto a different branch are
/// unaffected when that target branch is not ahead.
fn refuse_unpushed_branch_commits(repo: &Path, branch: &str, tracking: &str) -> Result<()> {
    let local_ref = format!("refs/heads/{branch}");
    let local_tip = match git_output(repo, &["rev-parse", "--verify", &local_ref]) {
        Ok(tip) => tip,
        Err(_) => return Ok(()),
    };
    let remote_tip = git_output(repo, &["rev-parse", "--verify", tracking])?;
    if local_tip == remote_tip {
        return Ok(());
    }
    // Behind (or equal ancestry): remote contains local tip → safe to fast-forward reset.
    if is_ancestor(repo, &local_tip, &remote_tip)? {
        return Ok(());
    }
    let ahead = git_output(
        repo,
        &["rev-list", "--count", &format!("{tracking}..{local_ref}")],
    )
    .ok()
    .and_then(|count| count.parse::<u64>().ok())
    .unwrap_or(1);
    bail!(
        "refusing to update branch '{branch}': {ahead} local commit(s) ahead of {tracking}; \
         push or move them before updating"
    );
}

fn is_ancestor(repo: &Path, maybe_ancestor: &str, commit: &str) -> Result<bool> {
    let status = Command::new("git")
        .current_dir(repo)
        .args(["merge-base", "--is-ancestor", maybe_ancestor, commit])
        .status()
        .with_context(|| format!("failed to compare git ancestry {maybe_ancestor} vs {commit}"))?;
    Ok(status.success())
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
    // Per-root component inventory. A working `hipcc` says nothing about the
    // HIP headers or runtime — they are separate packages — so reporting only
    // the hipcc version made a half-installed ROCm look healthy here while
    // every kernel compile and dlopen failed elsewhere.
    let rocm_roots = hipfire_config::rocm::roots()
        .iter()
        .filter(|root| root.is_dir())
        .map(|root| {
            let missing = hipfire_config::rocm::missing_components(root);
            serde_json::json!({
                "path": root.display().to_string(),
                "device_compiler": hipfire_config::rocm::DEVICE_COMPILERS
                    .iter()
                    .find(|name| root.join("bin").join(name).is_file()),
                "hip_headers": hipfire_config::rocm::is_complete_root(root),
                "hip_runtime": hipfire_config::rocm::runtime_library(root)
                    .map(|p| p.display().to_string()),
                "missing": missing.iter().map(|m| m.what).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let rocm_selected = hipfire_config::rocm::root().map(|p| p.display().to_string());
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
        "rocm": { "hipcc": hipcc, "selected_root": rocm_selected, "roots": rocm_roots },
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
            "ROCm root:     {}",
            rocm_selected.as_deref().unwrap_or("none found")
        );
        // Only actionable for a root that HAS a compiler: one without is a shim
        // directory (the /opt/rocm of a split-tree install), so its "missing"
        // components are expected rather than a problem to fix.
        let mut incomplete_toolchain = false;
        for root in &rocm_roots {
            let s = |k: &str| root[k].as_str().map(str::to_owned);
            println!(
                "  {}\n    compiler: {}   headers: {}   runtime: {}",
                root["path"].as_str().unwrap_or("?"),
                s("device_compiler").unwrap_or_else(|| "MISSING".into()),
                if root["hip_headers"].as_bool().unwrap_or(false) {
                    "yes"
                } else {
                    "MISSING"
                },
                s("hip_runtime").unwrap_or_else(|| "MISSING".into()),
            );
            let missing = root["missing"].as_array().map(Vec::len).unwrap_or(0);
            incomplete_toolchain |= missing > 0 && s("device_compiler").is_some();
        }
        if incomplete_toolchain {
            println!("  a ROCm root above has a compiler but no HIP runtime/headers:");
            for line in hipfire_config::rocm::install_guidance() {
                println!("    {line}");
            }
        }
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
        ValueRule::Deepseek4Placement => serde_json::json!({
            "type": "string",
            "format": "deepseek4-compute-placement",
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
        ValueRule::Deepseek4Placement => "deepseek4-placement",
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

    fn idle_test_meta() -> ServeMeta {
        ServeMeta {
            current_model: Some("model.hfq".to_owned()),
            loading_model: Some("model.hfq".to_owned()),
            instance_token: "test".to_owned(),
            requests_served: 0,
            retries_attempted: 0,
            retries_succeeded: 0,
            recent_tok_s: None,
            started: Instant::now(),
            last_activity: Instant::now() - Duration::from_secs(600),
        }
    }

    #[test]
    fn idle_timeout_does_not_evict_a_loading_model() {
        let meta = idle_test_meta();
        assert!(!idle_model_expired(&meta, Duration::from_secs(300)));
    }

    #[test]
    fn successful_prewarm_starts_a_fresh_idle_window() {
        let mut meta = idle_test_meta();
        finish_prewarm(&mut meta, true);
        assert!(meta.loading_model.is_none());
        assert!(!idle_model_expired(&meta, Duration::from_secs(300)));
        assert!(meta.last_activity.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn model_suffix_filter_covers_current_formats() {
        assert!(is_model_file("qwen3.6-35b-a3b.mq4r"));
        assert!(is_model_file("deepseek.mq2lloyd"));
        assert!(is_model_file("deepseek-v4-flash-0731.mq2r"));
        assert!(is_model_file("deepseek-v4-flash-0731.mq2rxt"));
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
        assert!(
            list_local_models(&paths, &registry)
                .unwrap()
                .iter()
                .any(|model| model.path == fs::canonicalize(&nested).unwrap())
        );
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
    fn load_params_only_forwards_explicit_deepseek4_expert_fanout() {
        let model_path = PathBuf::from("/tmp/test-model.mq2r");
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let params = load_params(&defaults, None, &model_path, 64, Some("q8"), None).unwrap();
        assert_eq!(params["deepseek4_compute_placement"], "single");
        assert!(params.get("deepseek4_experts_per_token").is_none());

        let mut explicit = ConfigLayer::default();
        explicit
            .set_cli("model.deepseek4_experts_per_token", "4")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "model.deepseek4_experts_per_token=4".into(),
            },
            layer: explicit,
        }])
        .unwrap();
        let params = load_params(&resolved, None, &model_path, 64, Some("q8"), None).unwrap();
        assert_eq!(params["deepseek4_experts_per_token"], 4);
    }

    #[test]
    fn load_params_forwards_typed_deepseek4_compute_placement() {
        let raw = "dense-expert-split(dense=arch:gfx1100,experts=arch:gfx1151)";
        let mut explicit = ConfigLayer::default();
        explicit
            .set_cli("hardware.deepseek4_compute_placement", raw)
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: format!("hardware.deepseek4_compute_placement={raw}"),
            },
            layer: explicit,
        }])
        .unwrap();
        let params = load_params(
            &resolved,
            None,
            Path::new("/tmp/test-model.mq2r"),
            64,
            Some("q8"),
            None,
        )
        .unwrap();
        assert_eq!(params["deepseek4_compute_placement"], raw);
    }

    #[test]
    fn load_params_forwards_dflash_draft_from_environment() {
        let draft = "/tmp/qwen35-9b-dflash-mq4.hfq";

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("speculation.mode", "dflash").unwrap();
        explicit.set_cli("developer.dflash_draft", draft).unwrap();
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
        // developer.dflash_draft is present and `run --spec dflash` re-enables
        // DFlash after load_params. Draft must land on the final load params.
        let draft = "/tmp/qwen35-9b-dflash-mq4.hfq";

        let mut explicit = ConfigLayer::default();
        explicit.set_cli("speculation.mode", "off").unwrap();
        explicit.set_cli("developer.dflash_draft", draft).unwrap();
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
            "config-off load_params must not project developer.dflash_draft"
        );

        // Final run-path selector: CLI `--spec dflash` then project inherited draft.
        apply_speculation_selector(&mut params, "dflash").unwrap();
        project_dflash_draft(&mut params, developer_dflash_draft(&resolved));
        assert_eq!(params["dflash_mode"], "on");
        assert_eq!(params["draft"], draft);

        // Final off must clear any previously projected draft.
        apply_speculation_selector(&mut params, "off").unwrap();
        project_dflash_draft(&mut params, developer_dflash_draft(&resolved));
        assert_eq!(params["dflash_mode"], "off");
        assert!(
            params.get("draft").is_none(),
            "final off must drop projected developer.dflash_draft"
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

        let mtp_window_timings = serde_json::json!([{
            "kind": "mtp",
            "wall_us": 1234,
            "draft_lookup_us": 12,
            "launch_us": 34,
            "h2d_us": 56,
            "d2h_us": 78,
            "d2d_us": 90,
            "memset_us": 11,
            "stream_sync_us": 22,
            "event_sync_us": 33,
            "device_sync_us": 44,
            "graph_launch_us": 55,
        }]);
        let mtp = completion_timings(&completion(serde_json::json!({
            "mtp": true,
            "tau": 2.0,
            "cycles": 6,
            "mtp_window_timings": mtp_window_timings,
        })));
        assert!(mtp["dflash"].is_null());
        assert_eq!(mtp["mtp"], true);
        assert_eq!(mtp["mtp_window_timings"], mtp_window_timings);
        assert_eq!(mtp["mtp_window_timings"][0]["kind"], "mtp");
        assert_eq!(mtp["mtp_window_timings"][0]["wall_us"], 1234);
    }

    #[test]
    fn completion_timings_projects_latency_ms() {
        let completion = Completion {
            id: "req-lat".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "ttft_ms": 12.5,
                "prefill_ms": 10.0,
                "prefill_tok_s": 100.0,
                "decode_tok_s": 40.0,
                "latency_ms": 250.5,
                "tok_s": 20.0,
            }),
        };
        let timings = completion_timings(&completion);
        assert_eq!(timings["latency_ms"], 250.5);
        assert_eq!(timings["ttft_ms"], 12.5);
        // Absent latency stays null-like (serde_json::Value::Null via .get).
        let no_lat = completion_timings(&Completion {
            id: "req-lat".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({"ttft_ms": 1.0}),
        });
        assert!(no_lat["latency_ms"].is_null());
    }

    #[test]
    fn completion_hipfire_projects_batch_route_evidence() {
        let batch = serde_json::json!({
            "executed": true,
            "slots": 4,
            "lane": 1,
            "lane_capacity": 4096,
            "max_active_lanes": 2,
            "refill": "continuous",
        });
        let completion = Completion {
            id: "req-batch".into(),
            created: 0,
            model: "test-model".into(),
            content: "hi".into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "tok_s": 18.5,
                "prefill_tok_s": 200.0,
                "decode_tok_s": 22.0,
                "latency_ms": 300.0,
                "execution_mode": "continuous_batch_independent",
                "continuous_batch": batch,
                "finish_reason": "stop",
                "tokens": 3,
                "prefill_tokens": 8,
            }),
        };
        let hip = completion_hipfire(&completion);
        assert_eq!(hip["execution_mode"], "continuous_batch_independent");
        assert_eq!(hip["continuous_batch"]["max_active_lanes"], 2);
        assert_eq!(hip["continuous_batch"]["slots"], 4);
        assert_eq!(hip["continuous_batch"]["lane"], 1);
        assert_eq!(hip["tok_s"], 18.5);

        let json = completion_json(&completion);
        assert_eq!(json["timings"]["latency_ms"], 300.0);
        assert_eq!(
            json["hipfire"]["execution_mode"],
            "continuous_batch_independent"
        );
        assert_eq!(json["hipfire"]["continuous_batch"]["refill"], "continuous");

        let chunks = openai_stream_terminal_chunks(&completion, false);
        let terminal = chunks.last().unwrap();
        assert_eq!(terminal["timings"]["latency_ms"], 300.0);
        assert_eq!(
            terminal["hipfire"]["execution_mode"],
            "continuous_batch_independent"
        );
        assert_eq!(
            terminal["hipfire"]["continuous_batch"]["max_active_lanes"],
            2
        );

        // Sequential done without route evidence stays null (unchanged fields).
        let sequential = Completion {
            id: "req-seq".into(),
            created: 0,
            model: "test-model".into(),
            content: String::new(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: Vec::new(),
            done: serde_json::json!({
                "tok_s": 10.0,
                "prefill_tok_s": 50.0,
                "decode_tok_s": 12.0,
                "finish_reason": "stop",
                "tokens": 2,
            }),
        };
        let seq_hip = completion_hipfire(&sequential);
        assert!(seq_hip["execution_mode"].is_null());
        assert!(seq_hip["continuous_batch"].is_null());
        assert_eq!(seq_hip["tok_s"], 10.0);
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
    fn update_refuses_branch_with_unpushed_commits() {
        fn git(repo: &Path, args: &[&str]) {
            let status = Command::new("git")
                .current_dir(repo)
                .args(args)
                .status()
                .unwrap();
            assert!(status.success(), "git {}", args.join(" "));
        }

        let root = env::temp_dir().join(format!(
            "hipfire-update-ahead-test-{}-{}",
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
        git(&installed, &["config", "user.name", "hipfire test"]);
        git(
            &installed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        fs::write(installed.join("local_only.txt"), "keep-me\n").unwrap();
        git(&installed, &["add", "local_only.txt"]);
        git(&installed, &["commit", "-m", "local-only"]);
        let local_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "master".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        let err = checkout_revision(&installed, &resolved)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("ahead") && err.contains("master"),
            "unexpected error: {err}"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            local_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("local_only.txt")).unwrap(),
            "keep-me\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    fn update_signal_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn git_test(repo: &Path, args: &[&str]) {
        let status = Command::new("git")
            .current_dir(repo)
            .args(args)
            .status()
            .unwrap();
        assert!(status.success(), "git {}", args.join(" "));
    }

    fn init_update_fixture(label: &str) -> (PathBuf, PathBuf) {
        let root = env::temp_dir().join(format!(
            "hipfire-update-{label}-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        let origin = root.join("origin.git");
        let seed = root.join("seed");
        let installed = root.join("installed");
        fs::create_dir_all(&root).unwrap();
        git_test(&root, &["init", "--bare", origin.to_str().unwrap()]);
        fs::create_dir_all(&seed).unwrap();
        git_test(&seed, &["init"]);
        git_test(&seed, &["config", "user.name", "hipfire test"]);
        git_test(
            &seed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        fs::write(seed.join("channel"), "master\n").unwrap();
        git_test(&seed, &["add", "channel"]);
        git_test(&seed, &["commit", "-m", "master"]);
        git_test(&seed, &["branch", "-M", "master"]);
        git_test(
            &seed,
            &["remote", "add", "origin", origin.to_str().unwrap()],
        );
        git_test(&seed, &["push", "-u", "origin", "master"]);
        git_test(&seed, &["checkout", "-b", "beta"]);
        fs::write(seed.join("channel"), "beta\n").unwrap();
        git_test(&seed, &["commit", "-am", "beta"]);
        git_test(&seed, &["push", "-u", "origin", "beta"]);
        git_test(
            &root,
            &[
                "clone",
                "--branch",
                "master",
                origin.to_str().unwrap(),
                installed.to_str().unwrap(),
            ],
        );
        git_test(&installed, &["config", "user.name", "hipfire test"]);
        git_test(
            &installed,
            &["config", "user.email", "hipfire-test@example.invalid"],
        );
        (root, installed)
    }

    #[test]
    fn update_handoff_forwards_recorded_rocm_root_and_gpu_arch() {
        let home = env::temp_dir().join(format!(
            "hipfire-update-rocm-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("install.json"),
            r#"{"commit":"abc","ref":"master","rocm_root":"/opt/rocm/core-7.14","gpu_arch":"gfx1201","profile":"auto","installed_at":1}"#,
        )
        .unwrap();
        let recorded = recorded_install_metadata(&home);
        assert_eq!(
            recorded.rocm_root.as_deref(),
            Some(Path::new("/opt/rocm/core-7.14"))
        );
        assert_eq!(recorded.gpu_arch.as_deref(), Some("gfx1201"));
        let args = installer_handoff_args(
            &RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
            recorded.rocm_root.as_deref(),
            recorded.gpu_arch.as_deref(),
        );
        assert_eq!(
            args,
            vec![
                "--yes".to_owned(),
                "--branch".to_owned(),
                "beta".to_owned(),
                "--rocm-root".to_owned(),
                "/opt/rocm/core-7.14".to_owned(),
                "--gpu-arch".to_owned(),
                "gfx1201".to_owned(),
            ]
        );

        fs::write(
            home.join("install.json"),
            r#"{"rocm_root":"  ","gpu_arch":"  "}"#,
        )
        .unwrap();
        let empty = recorded_install_metadata(&home);
        assert!(empty.rocm_root.is_none());
        assert!(empty.gpu_arch.is_none());
        let bare = installer_handoff_args(
            &RevisionSelector {
                value: "deadbeef".into(),
                kind: RevisionKind::Commit,
            },
            None,
            None,
        );
        assert_eq!(
            bare,
            vec![
                "--yes".to_owned(),
                "--commit".to_owned(),
                "deadbeef".to_owned(),
            ]
        );

        // Selector remains before optional install metadata; --yes stays first.
        let arch_only = installer_handoff_args(
            &RevisionSelector {
                value: "master".into(),
                kind: RevisionKind::Auto,
            },
            None,
            Some("gfx1100"),
        );
        assert_eq!(
            arch_only,
            vec![
                "--yes".to_owned(),
                "--ref".to_owned(),
                "master".to_owned(),
                "--gpu-arch".to_owned(),
                "gfx1100".to_owned(),
            ]
        );
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn update_restores_staged_unstaged_and_untracked_after_failed_handoff() {
        let (root, installed) = init_update_fixture("index-restore");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();

        // Tracked file with staged + unstaged split, plus untracked work.
        fs::write(installed.join("channel"), "staged-base\n").unwrap();
        git_test(&installed, &["add", "channel"]);
        fs::write(installed.join("channel"), "staged-base\nunstaged-tail\n").unwrap();
        fs::write(installed.join("scratch.txt"), "untracked-user\n").unwrap();

        run_checked(
            Command::new("git").current_dir(&installed).args([
                "stash",
                "push",
                "--include-untracked",
                "-m",
                "hipfire-update-index-test",
            ]),
            "git stash",
        )
        .unwrap();
        let stash_sha = git_output(&installed, &["rev-parse", "stash@{0}"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: Some(stash_sha),
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );

        // Installer dirties the failed target with tracked + untracked junk.
        fs::write(installed.join("channel"), "installer-mutated\n").unwrap();
        fs::write(installed.join("installer-junk.txt"), "leftover\n").unwrap();

        restore_update_checkpoint(&installed, &checkpoint).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("channel")).unwrap(),
            "staged-base\nunstaged-tail\n"
        );
        assert_eq!(
            fs::read_to_string(installed.join("scratch.txt")).unwrap(),
            "untracked-user\n"
        );
        // Index holds the staged half; worktree holds the full dirty file.
        let cached = git_output(&installed, &["show", ":channel"]).unwrap();
        assert_eq!(cached, "staged-base");
        assert!(!installed.join("installer-junk.txt").exists());
        // Successful --index apply drops the update stash.
        let stash_list = git_output(&installed, &["stash", "list"]).unwrap_or_default();
        assert!(
            stash_list.is_empty(),
            "update stash should be dropped after successful apply: {stash_list}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_installer_mutations_cannot_block_checkout_restore() {
        let (root, installed) = init_update_fixture("dirty-target");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: None,
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();

        // Simulate cargo/installer tracked + untracked mutations on the target.
        fs::write(installed.join("channel"), "lockfile-like-mutation\n").unwrap();
        fs::write(installed.join("target-artifact.bin"), "blob\n").unwrap();

        restore_update_checkpoint(&installed, &checkpoint).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("channel")).unwrap(),
            "master\n"
        );
        assert!(!installed.join("target-artifact.bin").exists());
        let porcelain = git_output(&installed, &["status", "--porcelain"]).unwrap_or_default();
        assert!(
            porcelain.is_empty(),
            "restored tree should be clean: {porcelain}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_rollback_guard_stays_armed_until_commit() {
        let (root, installed) = init_update_fixture("guard-arm");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: None,
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );

        {
            let mut guard = UpdateRollbackGuard::arm(installed.clone(), checkpoint.clone());
            assert!(guard.is_armed());
            // Drop without commit must restore master.
        }
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );

        // Success path: commit disarms so drop leaves the new revision alone.
        checkout_revision(&installed, &resolved).unwrap();
        {
            let mut guard = UpdateRollbackGuard::arm(installed.clone(), checkpoint);
            assert!(guard.is_armed());
            guard.commit();
            assert!(!guard.is_armed());
        }
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_interrupted_child_is_reaped_while_checkpoint_stays_armed() {
        let _lock = update_signal_test_lock();
        UPDATE_INTERRUPT.store(false, Ordering::SeqCst);

        let (root, installed) = init_update_fixture("interrupt-reap");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: None,
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();

        let mut guard = UpdateRollbackGuard::arm(installed.clone(), checkpoint);
        assert!(guard.is_armed());

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg("trap 'exit 0' TERM; while true; do sleep 0.05; done")
            .current_dir(&installed);
        #[cfg(unix)]
        {
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().unwrap();
        let child_pid = child.id();

        // Arm interrupt after spawn so the wait loop takes the TERM path.
        UPDATE_INTERRUPT.store(true, Ordering::SeqCst);
        let status = wait_update_installer_child(&mut child).unwrap();
        assert!(
            !status.success() || update_interrupted(),
            "interrupted wait should surface cancel state"
        );

        // Child must be reaped (no zombie); try_wait Ok(Some) or Err after wait.
        match child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => panic!("installer child {child_pid} was not reaped"),
            Err(_) => {}
        }

        // Guard remains armed until explicit fail/drop performs rollback.
        assert!(guard.is_armed());
        let err = guard.fail(anyhow!("update interrupted"));
        assert!(
            err.to_string().contains("update interrupted"),
            "unexpected error: {err}"
        );
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );

        UPDATE_INTERRUPT.store(false, Ordering::SeqCst);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_restores_checkout_and_stash_after_failed_handoff() {
        let (root, installed) = init_update_fixture("restore-basic");
        let previous_head = git_output(&installed, &["rev-parse", "HEAD"]).unwrap();
        fs::write(installed.join("dirty.txt"), "user-edit\n").unwrap();
        run_checked(
            Command::new("git").current_dir(&installed).args([
                "stash",
                "push",
                "--include-untracked",
                "-m",
                "hipfire-update-test",
            ]),
            "git stash",
        )
        .unwrap();
        let stash_sha = git_output(&installed, &["rev-parse", "stash@{0}"]).unwrap();
        let checkpoint = UpdateCheckpoint {
            head: previous_head.clone(),
            branch: Some("master".into()),
            stash_sha: Some(stash_sha),
        };

        let resolved = fetch_revision(
            &installed,
            RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
        )
        .unwrap();
        checkout_revision(&installed, &resolved).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "beta"
        );

        // Simulate installer handoff failure recovery.
        restore_update_checkpoint(&installed, &checkpoint).unwrap();
        assert_eq!(
            git_output(&installed, &["symbolic-ref", "--short", "HEAD"]).unwrap(),
            "master"
        );
        assert_eq!(
            git_output(&installed, &["rev-parse", "HEAD"]).unwrap(),
            previous_head
        );
        assert_eq!(
            fs::read_to_string(installed.join("dirty.txt")).unwrap(),
            "user-edit\n"
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
        assert!(
            request_image_base64(Some(&remote))
                .unwrap_err()
                .to_string()
                .contains("remote")
        );
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
            false,
        )
        .unwrap();
        assert_eq!(request["reasoning_effort"], "high");
        assert_eq!(request["max_think_tokens"], 4096);

        let mut deepseek_uncapped = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "max" }),
            &resolved,
            &mut deepseek_uncapped,
            true,
        )
        .unwrap();
        assert_eq!(deepseek_uncapped["reasoning_effort"], "max");
        assert_eq!(deepseek_uncapped["max_think_tokens"], 0);

        let mut deepseek_explicitly_capped = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "max",
                "max_think_tokens": 1234
            }),
            &resolved,
            &mut deepseek_explicitly_capped,
            true,
        )
        .unwrap();
        assert_eq!(deepseek_explicitly_capped["reasoning_effort"], "max");
        assert_eq!(deepseek_explicitly_capped["max_think_tokens"], 1234);

        let mut disabled = serde_json::json!({});
        apply_http_reasoning_request(
            &serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": false }
            }),
            &resolved,
            &mut disabled,
            false,
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
                "finish_reason": "stop",
                "mtp_ngram": 3,
                "ngram_mod_windows": 5,
                "ngram_mod_drafts": 12,
                "ngram_mod_accepted": 9,
                "ngram_mod_accept_rate": 0.75,
                "mtp_windows": 4,
                "ar_windows": 2,
                "mtp_retired": true
            }),
        };
        let json = completion_json(&completion);
        assert_eq!(json["usage"]["total_tokens"], 19);
        assert_eq!(json["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(json["timings"]["decode_tok_s"], 115.0);
        assert_eq!(json["timings"]["mtp_ngram"], 3);
        assert_eq!(json["timings"]["ngram_mod_windows"], 5);
        assert_eq!(json["timings"]["ngram_mod_drafts"], 12);
        assert_eq!(json["timings"]["ngram_mod_accepted"], 9);
        assert_eq!(json["timings"]["ngram_mod_accept_rate"], 0.75);
        assert_eq!(json["timings"]["mtp_windows"], 4);
        assert_eq!(json["timings"]["ar_windows"], 2);
        assert_eq!(json["timings"]["mtp_retired"], true);
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
        assert!(
            preserved_json["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );
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
    fn admission_eligible_concurrent_up_to_capacity() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        let g2 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        assert_eq!(admission.inflight(), 2);
        // Third eligible same model should queue, then timeout quickly.
        let admission2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || admission2.acquire_for(true, Some("qwen3.5:7b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 3); // 2 held + 1 queued
        drop(g1);
        let g3 = handle.join().unwrap();
        assert_eq!(admission.inflight(), 2);
        drop(g2);
        drop(g3);
        assert_eq!(admission.inflight(), 0);
    }

    #[test]
    fn admission_ineligible_is_exclusive() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        // Ineligible must wait for eligible to finish, even though capacity not full.
        let admission2 = Arc::clone(&admission);
        let handle = thread::spawn(move || admission2.acquire().unwrap());
        thread::sleep(Duration::from_millis(50));
        assert!(admission.inflight() == 2); // 1 eligible + 1 queued ineligible
        drop(g1);
        let g_inelig = handle.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        // While ineligible holds, eligible must wait.
        let admission3 = Arc::clone(&admission);
        let handle2 =
            thread::spawn(move || admission3.acquire_for(true, Some("qwen3.5:7b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 2);
        drop(g_inelig);
        let g2 = handle2.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        drop(g2);
    }

    #[test]
    fn admission_model_lease_prevents_cross_model_batch() {
        let admission = Arc::new(Admission::new_with_capacity(8, Duration::from_secs(1), 2));
        let g1 = admission.acquire_for(true, Some("qwen3.5:7b")).unwrap();
        // Different model cannot share batch lanes, must wait for exclusive.
        let admission2 = Arc::clone(&admission);
        let handle =
            thread::spawn(move || admission2.acquire_for(true, Some("qwen3.5:14b")).unwrap());
        thread::sleep(Duration::from_millis(50));
        assert_eq!(admission.inflight(), 2); // 1 held + 1 queued due to model mismatch
        drop(g1);
        let g2 = handle.join().unwrap();
        assert_eq!(admission.inflight(), 1);
        drop(g2);
    }

    #[test]
    fn batch_eligibility_conservative_checks() {
        // Eligible: tp=1 qwen with one plain user string.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Eligible: absent messages (prompt path) for Qwen.
        let body = serde_json::json!({"model":"qwen3.5:7b","prompt":"hi"});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Eligible: empty messages array for Qwen.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Tools disqualify.
        let body = serde_json::json!({"model":"qwen3.5:7b","tools":[{"type":"function","function":{"name":"x"}}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Image / multipart content disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:"}}]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Multipart text-only array content also disqualifies (must be plain string).
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // system+user history disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":"hi"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // user+assistant multi-turn disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"hello"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Tool-call content on the sole message disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b","messages":[{
            "role":"user",
            "content":"hi",
            "tool_calls":[{"id":"c0","type":"function","function":{"name":"x","arguments":"{}"}}]
        }]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            true
        ));
        // Qwen tp=4 pure EP is eligible when daemon admits batch.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(4),
            Some("qwen35"),
            true
        ));
        // Qwen tp=2 (and any non-1/non-4) disqualifies.
        let body = serde_json::json!({"model":"qwen3.5:7b"});
        assert!(!is_batch_eligible_request(
            &body,
            Some(2),
            Some("qwen35"),
            true
        ));
        // Non-qwen disqualifies.
        let body = serde_json::json!({"model":"deepseek4:671b"});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("deepseek4"),
            true
        ));
        // Daemon load says batch incapable: HTTP admission must not invent it.
        let body =
            serde_json::json!({"model":"qwen3.5:7b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("qwen35"),
            false
        ));
        // Eligible: tp=1 LFM2 dense with one plain user string when daemon admits batch.
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // Eligible: absent messages for LFM.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","prompt":"hi"});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // Eligible: empty messages for LFM.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[]});
        assert!(is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects system+user the same way.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":"hi"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects user+assistant.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":"hello"}
        ]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects tool-call content.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[{
            "role":"user",
            "content":"hi",
            "tool_calls":[{"id":"c0","type":"function","function":{"name":"x","arguments":"{}"}}]
        }]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM rejects image/multipart content.
        let body = serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":[
            {"type":"text","text":"describe"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}}
        ]}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            true
        ));
        // LFM tp=4 disqualifies (dense remains tp=1 only).
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(4),
            Some("lfm2"),
            true
        ));
        // Daemon load says batch incapable: LFM HTTP admission must not invent it.
        let body =
            serde_json::json!({"model":"lfm2.5:1.2b","messages":[{"role":"user","content":"hi"}]});
        assert!(!is_batch_eligible_request(
            &body,
            Some(1),
            Some("lfm2"),
            false
        ));
    }

    #[test]
    fn batch_messages_shape_matches_daemon_contract() {
        // Absent / empty → eligible shape.
        assert!(batch_messages_are_single_user(&serde_json::json!({})));
        assert!(batch_messages_are_single_user(
            &serde_json::json!({"messages":[]})
        ));
        // Exactly one plain user string → eligible.
        assert!(batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":"hi"}]
        })));
        // system+user, user+assistant → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[
                {"role":"system","content":"sys"},
                {"role":"user","content":"hi"}
            ]
        })));
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"yo"}
            ]
        })));
        // Non-user sole role → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"system","content":"sys"}]
        })));
        // tool_calls payload → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{
                "role":"user",
                "content":"hi",
                "tool_calls":[{"id":"c0"}]
            }]
        })));
        // Multipart / image array content → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]
        })));
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:"}}
            ]}]
        })));
        // Non-array messages → reject.
        assert!(!batch_messages_are_single_user(&serde_json::json!({
            "messages":"not-an-array"
        })));
    }

    #[test]
    fn daemon_tool_calls_map_to_openai_shape() {
        let calls = vec![
            ToolCall {
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "README.md" }),
            },
            ToolCall {
                name: "write_file".into(),
                arguments: serde_json::json!({ "path": "out.txt", "text": "hi" }),
            },
        ];
        let mapped = openai_tool_calls(&calls);
        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[0]["id"], "call_0");
        assert_eq!(mapped[1]["id"], "call_1");
        assert_eq!(mapped[0]["type"], "function");
        assert_eq!(mapped[0]["function"]["name"], "read_file");
        assert_eq!(
            mapped[0]["function"]["arguments"],
            serde_json::json!(r#"{"path":"README.md"}"#)
        );
        assert_eq!(mapped[1]["function"]["name"], "write_file");
    }

    fn sample_completion(
        content: &str,
        tool_calls: Vec<ToolCall>,
        finish_reason: &str,
    ) -> Completion {
        Completion {
            id: "chatcmpl_test".into(),
            created: 42,
            model: "qwen:test".into(),
            content: content.into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls,
            done: serde_json::json!({
                "finish_reason": finish_reason,
                "prompt_tokens": 3,
                "tokens": 5,
                "cached_tokens": 1,
                "tok_s": 10.0,
            }),
        }
    }

    fn sample_tc(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments,
        }
    }

    #[test]
    fn completion_json_pure_tool_turn_uses_null_content() {
        let completion = sample_completion(
            "",
            vec![sample_tc(
                "read_file",
                serde_json::json!({ "path": "a.rs" }),
            )],
            "tool_calls",
        );
        let json = completion_json(&completion);
        assert!(json["choices"][0]["message"]["content"].is_null());
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
    }

    #[test]
    fn completion_json_preserves_daemon_length_without_calls() {
        // Fold withholds calls on length; serializer must not invent tool_calls.
        let completion = sample_completion("", Vec::new(), "length");
        let json = completion_json(&completion);
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert!(json["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(json["choices"][0]["message"]["content"], "");
    }

    #[test]
    fn completion_json_never_overrides_length_error_cancel_when_calls_present() {
        // Defense in depth: even if tool_calls leaked onto Completion, daemon
        // finish_reason wins and must not be rewritten to tool_calls; calls stay off wire.
        for reason in ["length", "error", "cancelled", "aborted"] {
            let completion = sample_completion(
                "",
                vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))],
                reason,
            );
            let json = completion_json(&completion);
            assert_eq!(
                json["choices"][0]["finish_reason"], reason,
                "must preserve daemon finish_reason={reason}"
            );
            assert!(
                json["choices"][0]["message"].get("tool_calls").is_none(),
                "{reason} must not expose message.tool_calls"
            );
            // empty content + withheld calls → empty string, not null pure-tool
            assert_eq!(json["choices"][0]["message"]["content"], "");
        }
    }

    #[test]
    fn completion_json_stop_text_has_string_content_no_tool_calls() {
        let completion = sample_completion("hello world", Vec::new(), "stop");
        let json = completion_json(&completion);
        assert_eq!(json["choices"][0]["message"]["content"], "hello world");
        assert!(json["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn openai_stream_delta_forwards_only_clean_content_reasoning() {
        assert_eq!(
            openai_stream_delta_for_event(&serde_json::json!({
                "type": "token",
                "text": "hi"
            })),
            Some(serde_json::json!({ "content": "hi" }))
        );
        assert_eq!(
            openai_stream_delta_for_event(&serde_json::json!({
                "type": "reasoning",
                "text": "plan"
            })),
            Some(serde_json::json!({ "reasoning_content": "plan" }))
        );
        // Mid-stream tool_calls must never become an SSE delta.
        assert!(
            openai_stream_delta_for_event(&serde_json::json!({
                "type": "tool_calls",
                "calls": [{ "name": "read_file", "arguments": {} }]
            }))
            .is_none()
        );
        assert!(
            openai_stream_delta_for_event(&serde_json::json!({
                "type": "done",
                "finish_reason": "stop"
            }))
            .is_none()
        );
    }

    #[test]
    fn openai_stream_tool_safe_terminal_releases_calls_then_usage_then_done_shape() {
        let completion = sample_completion(
            "",
            vec![
                sample_tc("read_file", serde_json::json!({ "path": "a.rs" })),
                sample_tc("write_file", serde_json::json!({ "path": "b.rs" })),
            ],
            "tool_calls",
        );
        let chunks = openai_stream_terminal_chunks(&completion, true);
        assert_eq!(chunks.len(), 3, "tool delta + terminal + usage");

        // 1) tool_calls release with stable response-scoped ids/indices
        let tool_delta = &chunks[0]["choices"][0]["delta"]["tool_calls"];
        assert_eq!(tool_delta.as_array().map(|a| a.len()), Some(2));
        assert_eq!(tool_delta[0]["id"], "call_0");
        assert_eq!(tool_delta[0]["index"], 0);
        assert_eq!(tool_delta[1]["id"], "call_1");
        assert_eq!(tool_delta[1]["index"], 1);
        assert!(chunks[0]["choices"][0]["finish_reason"].is_null());
        assert!(chunks[0].get("usage").is_none());

        // 2) terminal choice with empty delta
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[1]["choices"][0]["delta"], serde_json::json!({}));
        assert!(
            chunks[1].get("usage").is_none(),
            "usage must not ride terminal"
        );

        // 3) separate choices:[] usage chunk
        assert_eq!(chunks[2]["choices"], serde_json::json!([]));
        assert_eq!(chunks[2]["usage"]["prompt_tokens"], 3);
        assert_eq!(chunks[2]["usage"]["completion_tokens"], 5);

        // Parity with non-stream ids/arguments
        let nonstream = completion_json(&completion);
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"],
            tool_delta[0]["id"]
        );
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["function"],
            tool_delta[0]["function"]
        );
        assert!(nonstream["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn openai_stream_length_terminal_exposes_no_call_deltas() {
        let completion = sample_completion(
            "partial",
            // Even if present, non-tool-safe finish must not release.
            vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))],
            "length",
        );
        let chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], "length");
        assert!(chunks[0]["choices"][0]["delta"].get("tool_calls").is_none());
        assert!(chunks[0].get("usage").is_none());
    }

    #[test]
    fn openai_stream_include_usage_false_skips_usage_chunk() {
        let completion = sample_completion("ok", Vec::new(), "stop");
        let chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], "stop");
        assert!(chunks[0].get("usage").is_none());
    }

    #[test]
    fn openai_stream_and_nonstream_paired_transcript_tool_safe() {
        // Paired transcript: fold-shaped Completion → both serializers agree.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-7", 7);
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [
                { "name": "read_file", "arguments": { "path": "a" } },
                { "name": "write_file", "arguments": { "path": "b" } }
            ],
            "id": "req-7",
            "attempt_id": 7
        }))
        .unwrap();
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 4,
            "id": "req-7",
            "attempt_id": 7,
            "calls": [
                { "name": "read_file", "arguments": { "path": "a" } },
                { "name": "write_file", "arguments": { "path": "b" } }
            ]
        }))
        .unwrap();

        let completion = Completion {
            id: "chatcmpl_pair".into(),
            created: 99,
            model: "m".into(),
            content: fold.content().to_owned(),
            reasoning_content: fold.reasoning_content().to_owned(),
            preserve_thinking: false,
            tool_calls: fold.executable_tool_calls().to_vec(),
            done: fold.done().cloned().unwrap(),
        };

        let nonstream = completion_json(&completion);
        assert!(nonstream["choices"][0]["message"]["content"].is_null());
        assert_eq!(nonstream["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][1]["id"],
            "call_1"
        );

        // Mid-stream fold forward never includes tool_calls.
        assert!(
            openai_stream_delta_for_event(&serde_json::json!({
                "type": "tool_calls",
                "calls": fold.executable_tool_calls()
            }))
            .is_none()
        );

        let stream_chunks = openai_stream_terminal_chunks(&completion, true);
        assert_eq!(
            stream_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"]
        );
        assert_eq!(
            stream_chunks[0]["choices"][0]["delta"]["tool_calls"][1]["function"],
            nonstream["choices"][0]["message"]["tool_calls"][1]["function"]
        );
        assert_eq!(
            stream_chunks[1]["choices"][0]["finish_reason"],
            "tool_calls"
        );
        assert_eq!(stream_chunks[2]["choices"], serde_json::json!([]));
        assert!(stream_chunks[2].get("usage").is_some());
    }

    #[test]
    fn openai_stream_and_nonstream_paired_transcript_length_no_calls() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-8", 8);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "partial",
            "id": "req-8",
            "attempt_id": 8
        }))
        .unwrap();
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [{ "name": "read_file", "arguments": { "path": "x" } }],
            "id": "req-8",
            "attempt_id": 8
        }))
        .unwrap();
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "length",
            "id": "req-8",
            "attempt_id": 8
        }))
        .unwrap();

        let completion = Completion {
            id: "chatcmpl_len".into(),
            created: 1,
            model: "m".into(),
            content: fold.content().to_owned(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls: fold.executable_tool_calls().to_vec(),
            done: fold.done().cloned().unwrap(),
        };
        assert!(completion.tool_calls.is_empty());

        let nonstream = completion_json(&completion);
        assert_eq!(nonstream["choices"][0]["finish_reason"], "length");
        assert!(
            nonstream["choices"][0]["message"]
                .get("tool_calls")
                .is_none()
        );
        assert_eq!(nonstream["choices"][0]["message"]["content"], "partial");

        let stream_chunks = openai_stream_terminal_chunks(&completion, true);
        assert_eq!(stream_chunks.len(), 2); // terminal + usage, no tool release
        assert_eq!(stream_chunks[0]["choices"][0]["finish_reason"], "length");
        assert!(
            stream_chunks[0]["choices"][0]["delta"]
                .get("tool_calls")
                .is_none()
        );
        assert_eq!(stream_chunks[1]["choices"], serde_json::json!([]));
    }

    /// Build a Completion whose done envelope has a non-string/missing finish_reason.
    fn sample_completion_with_done(
        content: &str,
        tool_calls: Vec<ToolCall>,
        done: serde_json::Value,
    ) -> Completion {
        Completion {
            id: "chatcmpl_test".into(),
            created: 42,
            model: "qwen:test".into(),
            content: content.into(),
            reasoning_content: String::new(),
            preserve_thinking: false,
            tool_calls,
            done,
        }
    }

    #[test]
    fn openai_stream_and_nonstream_paired_missing_finish_suppresses_leaked_calls() {
        // Missing finish_reason must never synthesize tool_calls from buffered/leaked calls.
        let leaked = vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))];
        let completion = sample_completion_with_done(
            "",
            leaked,
            serde_json::json!({
                "prompt_tokens": 3,
                "tokens": 5,
                "cached_tokens": 1,
                "tok_s": 10.0,
            }),
        );

        let nonstream = completion_json(&completion);
        assert_ne!(
            nonstream["choices"][0]["finish_reason"], "tool_calls",
            "missing finish_reason must not become tool_calls"
        );
        assert!(
            nonstream["choices"][0]["message"]
                .get("tool_calls")
                .is_none(),
            "missing finish_reason must suppress structured calls"
        );

        let stream_chunks = openai_stream_terminal_chunks(&completion, false);
        assert!(
            stream_chunks.iter().all(|c| c["choices"][0]
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .is_none()),
            "stream must not release tool deltas without explicit tool_calls terminal"
        );
        assert_ne!(
            stream_chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    fn openai_stream_and_nonstream_paired_null_finish_suppresses_leaked_calls() {
        // Null finish_reason is not an explicit tool_calls terminal.
        let leaked = vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))];
        let completion = sample_completion_with_done(
            "",
            leaked,
            serde_json::json!({
                "finish_reason": null,
                "prompt_tokens": 3,
                "tokens": 5,
                "cached_tokens": 1,
                "tok_s": 10.0,
            }),
        );

        let nonstream = completion_json(&completion);
        assert_ne!(
            nonstream["choices"][0]["finish_reason"], "tool_calls",
            "null finish_reason must not become tool_calls"
        );
        assert!(
            nonstream["choices"][0]["message"]
                .get("tool_calls")
                .is_none(),
            "null finish_reason must suppress structured calls"
        );

        let stream_chunks = openai_stream_terminal_chunks(&completion, false);
        assert!(
            stream_chunks.iter().all(|c| c["choices"][0]
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .is_none()),
            "stream must not release tool deltas on null finish_reason"
        );
        assert_ne!(
            stream_chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    fn openai_stream_and_nonstream_paired_explicit_tool_calls_releases_calls() {
        // Only an explicit raw daemon finish_reason of tool_calls may expose calls.
        let calls = vec![sample_tc(
            "read_file",
            serde_json::json!({ "path": "a.rs" }),
        )];
        let completion = sample_completion("", calls, "tool_calls");

        let nonstream = completion_json(&completion);
        assert_eq!(nonstream["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            nonstream["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert!(nonstream["choices"][0]["message"]["content"].is_null());

        let stream_chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(stream_chunks.len(), 2, "tool delta + terminal");
        assert_eq!(
            stream_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_0"
        );
        assert_eq!(
            stream_chunks[1]["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    fn sample_tool_call(name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "arguments": { "path": "README.md" }
        })
    }

    #[test]
    fn semantic_fold_accumulates_content_and_reasoning_without_marker_parse() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-1", 1);
        // Classifier-authorized prose may quote protocol lexemes; fold must not
        // invent tool calls or strip them — only daemon tool_calls count.
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "use <tool_call> as documentation",
                "id": "req-1",
                "attempt_id": 1
            }))
            .expect("token");
        assert_eq!(
            forwarded,
            vec![serde_json::json!({
                "type": "token",
                "text": "use <tool_call> as documentation"
            })]
        );
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "reasoning",
                "text": "plan step",
                "id": "req-1",
                "attempt_id": 1
            }))
            .expect("reasoning");
        assert_eq!(
            forwarded,
            vec![serde_json::json!({ "type": "reasoning", "text": "plan step" })]
        );
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "stop",
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("done");
        assert_eq!(fold.content(), "use <tool_call> as documentation");
        assert_eq!(fold.reasoning_content(), "plan step");
        assert!(fold.executable_tool_calls().is_empty());
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("stop")
        );
    }

    #[test]
    fn semantic_fold_keeps_think_and_im_end_markers_verbatim_including_splits() {
        // Critical: v2 fold must never invoke ThinkChannelRouter / marker scan.
        // Recognized control literals must survive whole and across chunk boundaries.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-42", 42);
        let pieces = ["<thi", "nk>plan</thi", "nk>\nanswer<|im_", "end|>tail"];
        let mut forwarded_text = String::new();
        for piece in pieces {
            let forwarded = fold
                .push(&serde_json::json!({
                    "type": "token",
                    "text": piece,
                    "id": "req-42",
                    "attempt_id": 42
                }))
                .expect("chunk");
            assert_eq!(forwarded.len(), 1);
            assert_eq!(forwarded[0]["type"], "token");
            assert_eq!(forwarded[0]["text"], piece);
            forwarded_text.push_str(piece);
        }
        let expected = "<think>plan</think>\nanswer<|im_end|>tail";
        assert_eq!(forwarded_text, expected);
        assert_eq!(fold.content(), expected);

        // Reasoning channel is also verbatim (no strip of </think> / <|im_end|>).
        fold.begin_attempt("req-43", 43);
        fold.push(&serde_json::json!({
            "type": "reasoning",
            "text": "r<think>x</think><|im_end|>",
            "id": "req-43",
            "attempt_id": 43
        }))
        .expect("reasoning markers");
        assert_eq!(fold.reasoning_content(), "r<think>x</think><|im_end|>");
        assert!(fold.content().is_empty());
    }

    #[test]
    fn semantic_fold_buffers_tool_calls_until_tool_safe_done() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-2", 2);
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "calling",
                "id": "req-2",
                "attempt_id": 2
            }))
            .expect("token");
        assert_eq!(forwarded.len(), 1);
        let forwarded = fold
            .push(&serde_json::json!({
                "type": "tool_calls",
                "calls": [sample_tool_call("read_file")],
                "id": "req-2",
                "attempt_id": 2
            }))
            .expect("tool_calls");
        // Mid-stream: nothing forwarded; calls stay buffered and non-executable.
        assert!(forwarded.is_empty());
        assert_eq!(fold.buffered_tool_calls().len(), 1);
        assert!(fold.executable_tool_calls().is_empty());

        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "tool_calls",
            "tok_s": 12.5,
            "id": "req-2",
            "attempt_id": 2,
            "calls": [sample_tool_call("read_file")]
        }))
        .expect("done");
        assert_eq!(fold.executable_tool_calls().len(), 1);
        assert_eq!(fold.executable_tool_calls()[0].name, "read_file");
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("tool_calls")
        );
        // Daemon finish_reason preserved verbatim (no fold-side rewrite).
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("tok_s"))
                .and_then(|v| v.as_f64()),
            Some(12.5)
        );
    }

    #[test]
    fn semantic_fold_length_terminal_exposes_no_executable_calls() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-3", 3);
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [sample_tool_call("write_file"), sample_tool_call("read_file")],
            "id": "req-3",
            "attempt_id": 3
        }))
        .expect("buffered");
        assert_eq!(fold.buffered_tool_calls().len(), 2);
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "length",
            "id": "req-3",
            "attempt_id": 3
        }))
        .expect("done");
        assert!(
            fold.executable_tool_calls().is_empty(),
            "length must never release buffered calls"
        );
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("length"),
            "daemon finish_reason must not be rewritten to tool_calls"
        );
    }

    #[test]
    fn semantic_fold_error_and_abort_terminals_expose_no_calls() {
        for reason in ["error", "aborted", "cancelled"] {
            let mut fold = SemanticEventFold::new();
            fold.begin_attempt("req-4", 4);
            fold.push(&serde_json::json!({
                "type": "tool_calls",
                "calls": [sample_tool_call("read_file")],
                "id": "req-4",
                "attempt_id": 4
            }))
            .expect("buffered");
            fold.push(&serde_json::json!({
                "type": "done",
                "finish_reason": reason,
                "id": "req-4",
                "attempt_id": 4
            }))
            .expect("done");
            assert!(
                fold.executable_tool_calls().is_empty(),
                "{reason} must not expose executable calls"
            );
            assert_eq!(
                fold.done()
                    .and_then(|d| d.get("finish_reason"))
                    .and_then(|v| v.as_str()),
                Some(reason)
            );
        }
    }

    #[test]
    fn semantic_fold_stop_with_empty_buffer_stays_empty() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-5", 5);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "hello",
            "id": "req-5",
            "attempt_id": 5
        }))
        .expect("token");
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "stop",
            "id": "req-5",
            "attempt_id": 5
        }))
        .expect("done");
        assert!(fold.executable_tool_calls().is_empty());
        assert_eq!(fold.content(), "hello");
    }

    #[test]
    fn semantic_fold_rejects_stale_attempt_events() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-10", 10);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "a1",
            "id": "req-10",
            "attempt_id": 10
        }))
        .expect("current");
        let err = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "stale",
                "id": "req-10",
                "attempt_id": 9
            }))
            .expect_err("stale attempt");
        assert_eq!(
            err,
            SemanticFoldError::StaleAttempt {
                current: 10,
                got: 9
            }
        );
        // Current attempt state must remain intact after rejection.
        assert_eq!(fold.content(), "a1");
        assert_eq!(fold.current_attempt_id(), Some(10));
    }

    #[test]
    fn semantic_fold_rejects_missing_malformed_from_first_event() {
        // Critical: after begin_attempt, every event including the first must
        // carry a matching numeric attempt_id. No lazy correlation / uncorrelated
        // stream acceptance.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-11", 11);

        let missing = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "no id",
                "id": "req-11"
            }))
            .expect_err("missing attempt_id");
        assert_eq!(missing, SemanticFoldError::MissingAttemptId { current: 11 });
        assert!(fold.content().is_empty());

        let malformed_string = fold
            .push(&serde_json::json!({
                "type": "token",
                "text": "bad",
                "id": "req-11",
                "attempt_id": "11"
            }))
            .expect_err("string attempt_id");
        assert_eq!(
            malformed_string,
            SemanticFoldError::MalformedAttemptId { current: 11 }
        );

        let malformed_null = fold
            .push(&serde_json::json!({
                "type": "gen_start",
                "id": "req-11",
                "attempt_id": null
            }))
            .expect_err("null attempt_id");
        assert_eq!(
            malformed_null,
            SemanticFoldError::MalformedAttemptId { current: 11 }
        );

        // Without begin_attempt, push fails closed (no uncorrelated stream).
        let mut cold = SemanticEventFold::new();
        let err = cold
            .push(&serde_json::json!({
                "type": "token",
                "text": "uncorrelated",
                "id": "req-1",
                "attempt_id": 1
            }))
            .expect_err("no active attempt");
        assert_eq!(err, SemanticFoldError::NoActiveAttempt);
    }

    #[test]
    fn semantic_fold_begin_attempt_clears_attempt_local_state() {
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-1", 1);
        fold.push(&serde_json::json!({
            "type": "token",
            "text": "first",
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("token");
        fold.push(&serde_json::json!({
            "type": "reasoning",
            "text": "think1",
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("reasoning");
        fold.push(&serde_json::json!({
            "type": "tool_calls",
            "calls": [sample_tool_call("read_file")],
            "id": "req-1",
            "attempt_id": 1
        }))
        .expect("calls");
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "tool_calls",
            "id": "req-1",
            "attempt_id": 1,
            "calls": [sample_tool_call("read_file")]
        }))
        .expect("done");
        assert!(!fold.content().is_empty());
        assert!(!fold.executable_tool_calls().is_empty());

        fold.begin_attempt("req-2", 2);
        assert_eq!(fold.current_attempt_id(), Some(2));
        assert!(fold.content().is_empty());
        assert!(fold.reasoning_content().is_empty());
        assert!(fold.buffered_tool_calls().is_empty());
        assert!(fold.executable_tool_calls().is_empty());
        assert!(fold.done().is_none());

        fold.push(&serde_json::json!({
            "type": "token",
            "text": "second",
            "id": "req-2",
            "attempt_id": 2
        }))
        .expect("retry token");
        fold.push(&serde_json::json!({
            "type": "done",
            "finish_reason": "stop",
            "id": "req-2",
            "attempt_id": 2
        }))
        .expect("retry done");
        assert_eq!(fold.content(), "second");
        assert!(fold.executable_tool_calls().is_empty());
        assert_eq!(
            fold.done()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("stop")
        );
    }

    #[test]
    fn producer_to_fold_contract_v2_verbatim_legacy_outside() {
        // Important: gen_start.contract_version == 2 selects SemanticEventFold
        // (verbatim text). Legacy non-tool raw-think stays on ThinkChannelRouter
        // outside the fold — proved here without inventing a second fold path.

        // --- v2 producer path (fold) ---
        let mut v2 = SemanticEventFold::new();
        v2.begin_attempt("req-100", 100);
        v2.push(&serde_json::json!({
            "type": "gen_start",
            "contract_version": 2,
            "started_in_think": true,
            "id": "req-100",
            "attempt_id": 100
        }))
        .expect("v2 gen_start");
        // started_in_think must not open a think channel inside the fold.
        let text = "visible <think>not-routed</think> <|im_end|>";
        let fwd = v2
            .push(&serde_json::json!({
                "type": "token",
                "text": text,
                "id": "req-100",
                "attempt_id": 100
            }))
            .expect("v2 token");
        assert_eq!(
            fwd,
            vec![serde_json::json!({ "type": "token", "text": text })]
        );
        assert_eq!(v2.content(), text);
        assert!(v2.reasoning_content().is_empty());

        // --- legacy producer path (ThinkChannelRouter only; outside fold) ---
        let mut router = ThinkChannelRouter::default();
        router.set_started_in_think(true);
        let mut fragments = router.push("legacy reason</thi");
        fragments.extend(router.push("nk>\n\nlegacy answer"));
        fragments.extend(router.finish());
        assert_eq!(
            fragments,
            vec![
                ThinkFragment::Reasoning("legacy reason".into()),
                ThinkFragment::Content("legacy answer".into()),
            ],
            "legacy MiniMax/Cohere-style markers still route outside SemanticEventFold"
        );

        // Fold must not be the home of that routing: pushing the same bytes
        // through a correlated fold keeps them as content, not reasoning split.
        let mut not_legacy = SemanticEventFold::new();
        not_legacy.begin_attempt("req-101", 101);
        not_legacy
            .push(&serde_json::json!({
                "type": "token",
                "text": "legacy reason</think>\n\nlegacy answer",
                "id": "req-101",
                "attempt_id": 101
            }))
            .expect("fold token");
        assert_eq!(
            not_legacy.content(),
            "legacy reason</think>\n\nlegacy answer"
        );
        assert!(not_legacy.reasoning_content().is_empty());
    }

    #[test]
    fn next_attempt_id_is_nonzero_and_monotonic() {
        let a = next_attempt_id();
        let b = next_attempt_id();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
        assert!(b > a);
    }

    #[test]
    fn task15_attempt_latches_truth_table() {
        let cases: &[(&str, bool, bool)] = &[
            ("token", true, false),
            ("reasoning", true, false),
            ("commit_ready", false, true),
            ("tool_calls", false, false),
            ("gen_start", false, false),
            ("done", false, false),
            ("error", false, false),
        ];
        for (ty, want_visible, want_commit) in cases {
            let mut latches = AttemptLatches::default();
            latches.observe(&serde_json::json!({ "type": ty }));
            assert_eq!(latches.visible, *want_visible, "visible for {ty}");
            assert_eq!(
                latches.commit_ready_seen, *want_commit,
                "commit_ready_seen for {ty}"
            );
        }
    }

    fn task15_daemon_err(class: &str, retryable: bool, attempt_id: u64) -> anyhow::Error {
        anyhow::Error::new(hipfire_client::ClientError::Daemon(
            hipfire_client::TypedDaemonError {
                message: format!("t15 {class}"),
                class: class.to_owned(),
                retryable,
                rolled_back: false,
                attempt_id,
                id: Some("req-t15".into()),
            },
        ))
    }

    #[test]
    fn task15_decide_retry_classifier_truth_table() {
        let aid = 42u64;
        let clean = AttemptLatches::default();
        let mut visible = AttemptLatches::default();
        visible.visible = true;
        let mut committed = AttemptLatches::default();
        committed.commit_ready_seen = true;

        let ok_err = task15_daemon_err(hipfire_client::error_class::TRANSIENT, true, aid);
        assert_eq!(
            decide_retry(&ok_err, aid, &clean, true, true, 1),
            RetryDecision::Retry
        );

        let denials: &[(&str, RetryDecision)] = &[
            (
                "gate_off",
                decide_retry(&ok_err, aid, &clean, true, false, 1),
            ),
            (
                "attempt_2",
                decide_retry(&ok_err, aid, &clean, true, true, 2),
            ),
            (
                "visible",
                decide_retry(&ok_err, aid, &visible, true, true, 1),
            ),
            (
                "commit_ready",
                decide_retry(&ok_err, aid, &committed, true, true, 1),
            ),
            (
                "ineligible",
                decide_retry(&ok_err, aid, &clean, false, true, 1),
            ),
            (
                "attempt_mismatch",
                decide_retry(&ok_err, aid + 1, &clean, true, true, 1),
            ),
            (
                "not_retryable",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::TRANSIENT, false, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_validation",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::VALIDATION, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_malformed",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::MALFORMED, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_internal",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::INTERNAL, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "class_cancel",
                decide_retry(
                    &task15_daemon_err(hipfire_client::error_class::CANCEL, true, aid),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
            (
                "non_daemon",
                decide_retry(&anyhow::anyhow!("plain error"), aid, &clean, true, true, 1),
            ),
            (
                "protocol",
                decide_retry(
                    &anyhow::Error::new(hipfire_client::ClientError::Protocol("x".into())),
                    aid,
                    &clean,
                    true,
                    true,
                    1,
                ),
            ),
        ];
        for (name, decision) in denials {
            assert_eq!(*decision, RetryDecision::Fail, "expected Fail for {name}");
        }
    }

    #[test]
    fn task15_serve_retry_config_defaults_off() {
        let resolved = resolve(Vec::<NamedLayer>::new()).expect("resolve empty layers");
        let enabled = config_bool(&resolved, "serve.retry_enabled").expect("retry_enabled");
        let backoff = config_u64(&resolved, "serve.retry_backoff_ms").expect("retry_backoff_ms");
        assert!(!enabled, "serve.retry_enabled must default false");
        assert_eq!(backoff, 50);
    }

    // --- StreamContractGate / complete_request framing (fix round 2) ---

    #[test]
    fn complete_request_fold_rejects_pre_start_token() {
        // Critical: no unchecked legacy default before gen_start.
        let err = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "token",
                "text": "too early",
                "id": "req-7",
                "attempt_id": 7
            })],
        )
        .expect_err("pre-start token must fail closed");
        assert_eq!(
            err,
            StreamContractError::PreStartEvent {
                event_type: "token".into()
            }
        );
    }

    #[test]
    fn complete_request_fold_rejects_missing_id_gen_start() {
        // Correlation before contract_version latch — missing id never selects legacy/v2.
        let err = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2
            })],
        )
        .expect_err("missing request id on gen_start");
        assert_eq!(
            err,
            StreamContractError::MissingRequestId {
                expected: "req-7".into()
            }
        );

        let err_missing_attempt = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7"
            })],
        )
        .expect_err("missing attempt_id on gen_start");
        assert_eq!(
            err_missing_attempt,
            StreamContractError::MissingAttemptId { expected: 7 }
        );

        let err_malformed = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": "7"
            })],
        )
        .expect_err("string attempt_id on gen_start");
        assert_eq!(
            err_malformed,
            StreamContractError::MalformedAttemptId { expected: 7 }
        );

        let err_stale_first = fold_complete_request_stream(
            "req-7",
            7,
            &[serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": 99
            })],
        )
        .expect_err("stale attempt_id on first gen_start");
        assert_eq!(
            err_stale_first,
            StreamContractError::StaleAttempt {
                expected: 7,
                got: 99
            }
        );
    }

    #[test]
    fn complete_request_fold_rejects_stale_second_start_downgrade() {
        // Critical: after v2 is latched, a later stale/missing-id gen_start must
        // NOT downgrade the stream to legacy or re-latch contract_version.
        let events = [
            serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": 7
            }),
            serde_json::json!({
                "type": "token",
                "text": "kept",
                "id": "req-7",
                "attempt_id": 7
            }),
            // Stale second start — previously could flip contract_v2 = Some(false).
            serde_json::json!({
                "type": "gen_start",
                "id": "req-1",
                "attempt_id": 1
            }),
            serde_json::json!({
                "type": "token",
                "text": "would-be-legacy",
                "id": "req-7",
                "attempt_id": 7
            }),
        ];
        let err = fold_complete_request_stream("req-7", 7, &events)
            .expect_err("second gen_start must reject without downgrade");
        assert_eq!(err, StreamContractError::SecondGenStart);

        // Gate unit: v2 stays latched even if observe is retried after error.
        let mut gate = StreamContractGate::new("req-7", 7);
        assert_eq!(gate.observe(&events[0]).expect("first"), StreamContract::V2);
        assert_eq!(gate.observe(&events[1]).expect("token"), StreamContract::V2);
        assert_eq!(
            gate.observe(&events[2]).expect_err("second start"),
            StreamContractError::SecondGenStart
        );
        assert_eq!(gate.contract(), Some(StreamContract::V2));
        assert!(gate.is_v2());
        // Subsequent non-start events would still be v2 if caller continued —
        // production aborts the generate callback on SecondGenStart instead.
        assert_eq!(
            gate.observe(&events[3]).expect("still v2"),
            StreamContract::V2
        );

        // Missing-id second start is also SecondGenStart (never re-reads version).
        let mut gate2 = StreamContractGate::new("req-7", 7);
        gate2
            .observe(&serde_json::json!({
                "type": "gen_start",
                "contract_version": 2,
                "id": "req-7",
                "attempt_id": 7
            }))
            .unwrap();
        assert_eq!(
            gate2
                .observe(&serde_json::json!({
                    "type": "gen_start",
                    "contract_version": 1
                }))
                .expect_err("missing-id second start"),
            StreamContractError::SecondGenStart
        );
        assert_eq!(gate2.contract(), Some(StreamContract::V2));
    }

    #[test]
    fn complete_request_fold_valid_legacy_and_v2_starts() {
        // Valid v2 start → SemanticEventFold verbatim path.
        let v2 = fold_complete_request_stream(
            "req-42",
            42,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "contract_version": 2,
                    "started_in_think": true,
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "token",
                    "text": "hi <think>raw</think>",
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "stop",
                    "id": "req-42",
                    "attempt_id": 42
                }),
            ],
        )
        .expect("valid v2 stream");
        assert_eq!(v2.contract, StreamContract::V2);
        assert_eq!(v2.content, "hi <think>raw</think>");
        assert!(v2.reasoning_content.is_empty());
        assert!(v2.tool_calls.is_empty());
        assert_eq!(
            v2.done
                .as_ref()
                .and_then(|d| d.get("finish_reason"))
                .and_then(|v| v.as_str()),
            Some("stop")
        );

        // Valid legacy start (missing/other contract_version) → ThinkChannelRouter.
        let legacy = fold_complete_request_stream(
            "req-42",
            42,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "started_in_think": true,
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "token",
                    "text": "plan</think>\n\nanswer",
                    "id": "req-42",
                    "attempt_id": 42
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "stop",
                    "id": "req-42",
                    "attempt_id": 42
                }),
            ],
        )
        .expect("valid legacy stream");
        assert_eq!(legacy.contract, StreamContract::Legacy);
        assert_eq!(legacy.reasoning_content, "plan");
        assert_eq!(legacy.content, "answer");

        // Explicit non-2 contract_version is also legacy.
        let legacy_v1 = fold_complete_request_stream(
            "req-3",
            3,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "contract_version": 1,
                    "id": "req-3",
                    "attempt_id": 3
                }),
                serde_json::json!({
                    "type": "token",
                    "text": "plain",
                    "id": "req-3",
                    "attempt_id": 3
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "stop",
                    "id": "req-3",
                    "attempt_id": 3
                }),
            ],
        )
        .expect("contract_version 1 is legacy");
        assert_eq!(legacy_v1.contract, StreamContract::Legacy);
        assert_eq!(legacy_v1.content, "plain");
    }

    // ── Task 6: canonical OpenAI tool-call adapter + endpoint registry ──

    #[test]
    fn openai_adapter_preserves_names_and_nested_arguments() {
        let calls = vec![
            sample_tc(
                "search",
                serde_json::json!({
                    "query": "hipfire",
                    "filters": { "lang": ["rust", "c"], "limit": 3 },
                    "opts": { "nested": { "deep": true } }
                }),
            ),
            sample_tc("ping", serde_json::json!({})),
        ];
        let adapted = openai_tool_call_adapter_results(&calls);
        assert_eq!(adapted.len(), 2);
        assert_eq!(adapted[0].name, "search");
        assert_eq!(adapted[1].name, "ping");

        let args0: serde_json::Value =
            serde_json::from_str(&adapted[0].arguments).expect("args json");
        assert_eq!(args0["filters"]["lang"][1], "c");
        assert_eq!(args0["opts"]["nested"]["deep"], true);
        assert_eq!(adapted[1].arguments, "{}");

        let lowered = openai_tool_calls(&calls);
        assert_eq!(lowered[0]["function"]["name"], "search");
        assert_eq!(
            lowered[0]["function"]["arguments"].as_str().unwrap(),
            adapted[0].arguments
        );
    }

    #[test]
    fn openai_adapter_deterministic_stable_ids_and_indices() {
        let calls = vec![
            sample_tc("a", serde_json::json!({"n": 1})),
            sample_tc("b", serde_json::json!({"n": 2})),
            sample_tc("c", serde_json::json!({"n": 3})),
        ];
        let first = openai_tool_call_adapter_results(&calls);
        let second = openai_tool_call_adapter_results(&calls);
        assert_eq!(first, second, "adapter result must be deterministic");
        for (i, row) in first.iter().enumerate() {
            assert_eq!(row.index, i);
            assert_eq!(row.id, format!("call_{i}"));
            assert_eq!(row.name, calls[i].name);
        }

        let stream_delta = openai_tool_call_delta_from_adapter(&first);
        let nonstream = openai_tool_calls_from_adapter(&first);
        for i in 0..3 {
            assert_eq!(stream_delta["tool_calls"][i]["index"], i);
            assert_eq!(stream_delta["tool_calls"][i]["id"], format!("call_{i}"));
            assert_eq!(nonstream[i]["id"], format!("call_{i}"));
            assert!(
                nonstream[i].get("index").is_none(),
                "non-stream message.tool_calls must not carry stream index"
            );
        }
    }

    #[test]
    fn openai_stream_and_nonstream_share_one_adapter_result() {
        let calls = vec![
            sample_tc(
                "read_file",
                serde_json::json!({ "path": "a.rs", "meta": { "k": "v" } }),
            ),
            sample_tc("write_file", serde_json::json!({ "path": "b.rs" })),
        ];
        let adapted = openai_tool_call_adapter_results(&calls);
        let completion = sample_completion("", calls.clone(), "tool_calls");

        let nonstream = completion_json(&completion);
        let stream = openai_stream_terminal_chunks(&completion, false);
        let ns_calls = nonstream["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("nonstream tool_calls");
        let st_calls = stream[0]["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .expect("stream tool_calls");

        assert_eq!(ns_calls.len(), adapted.len());
        assert_eq!(st_calls.len(), adapted.len());
        for (i, row) in adapted.iter().enumerate() {
            assert_eq!(ns_calls[i]["id"], row.id);
            assert_eq!(ns_calls[i]["function"]["name"], row.name);
            assert_eq!(ns_calls[i]["function"]["arguments"], row.arguments);
            assert_eq!(st_calls[i]["id"], row.id);
            assert_eq!(st_calls[i]["index"], row.index);
            assert_eq!(st_calls[i]["function"]["name"], row.name);
            assert_eq!(st_calls[i]["function"]["arguments"], row.arguments);
            assert_eq!(ns_calls[i]["function"], st_calls[i]["function"]);
            assert_eq!(ns_calls[i]["id"], st_calls[i]["id"]);
        }
    }

    #[test]
    fn openai_pure_tool_turn_content_is_null_not_empty_string() {
        let completion = sample_completion(
            "",
            vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))],
            "tool_calls",
        );
        let json = completion_json(&completion);
        assert!(json["choices"][0]["message"]["content"].is_null());
        assert!(json["choices"][0]["message"]["tool_calls"].is_array());
    }

    #[test]
    fn openai_mixed_prose_and_calls_retains_prose_content() {
        let completion = sample_completion(
            "I'll look that up.",
            vec![sample_tc("search", serde_json::json!({ "q": "docs" }))],
            "tool_calls",
        );
        let json = completion_json(&completion);
        assert_eq!(
            json["choices"][0]["message"]["content"],
            "I'll look that up."
        );
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "search"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");

        let chunks = openai_stream_terminal_chunks(&completion, false);
        assert_eq!(
            chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "search"
        );
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn openai_length_error_cancel_malformed_never_release_calls() {
        let leaked = vec![sample_tc("read_file", serde_json::json!({ "path": "x" }))];
        for reason in [
            "length",
            "error",
            "cancelled",
            "aborted",
            "malformed_protocol",
        ] {
            let completion = sample_completion("partial", leaked.clone(), reason);
            let nonstream = completion_json(&completion);
            assert_eq!(
                nonstream["choices"][0]["finish_reason"], reason,
                "{reason}: finish_reason must stay authoritative"
            );
            assert!(
                nonstream["choices"][0]["message"]
                    .get("tool_calls")
                    .is_none(),
                "{reason}: must not release message.tool_calls"
            );
            assert_eq!(nonstream["choices"][0]["message"]["content"], "partial");

            let stream = openai_stream_terminal_chunks(&completion, false);
            assert!(
                stream.iter().all(|c| {
                    c["choices"]
                        .as_array()
                        .and_then(|choices| choices.first())
                        .and_then(|ch| ch.get("delta"))
                        .and_then(|d| d.get("tool_calls"))
                        .is_none()
                }),
                "{reason}: stream must not emit tool_call deltas"
            );
            assert_eq!(stream[0]["choices"][0]["finish_reason"], reason);
        }
    }

    #[test]
    fn malformed_daemon_call_fails_at_canonical_boundary() {
        let err = tool_call_from_canonical_value(&serde_json::json!("not-an-object"))
            .expect_err("non-object");
        assert!(
            err.contains("JSON object") || err.contains("object"),
            "detail={err}"
        );

        let err = tool_call_from_canonical_value(&serde_json::json!({
            "arguments": { "path": "x" }
        }))
        .expect_err("missing name");
        assert!(err.contains("name"), "detail={err}");

        let err = tool_call_from_canonical_value(&serde_json::json!({
            "name": "   ",
            "arguments": {}
        }))
        .expect_err("empty name");
        assert!(err.contains("name"), "detail={err}");

        let legacy_err = tool_call_from_legacy_value(&serde_json::json!(null)).expect_err("null");
        assert!(!legacy_err.is_empty());

        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-11", 11);
        let fold_err = fold
            .push(&serde_json::json!({
                "type": "tool_calls",
                "calls": [{ "arguments": { "path": "x" } }],
                "id": "req-11",
                "attempt_id": 11
            }))
            .expect_err("malformed call must fail fold push");
        match fold_err {
            SemanticFoldError::MalformedToolCall { detail } => {
                assert!(detail.contains("name"), "detail={detail}");
            }
            other => panic!("expected MalformedToolCall, got {other:?}"),
        }
        assert!(
            fold.buffered_tool_calls().is_empty(),
            "malformed must not buffer a call"
        );
        assert!(fold.executable_tool_calls().is_empty());
    }

    #[test]
    fn semantic_fold_missing_calls_fails_closed_before_tool_terminal() {
        // Canonical v2 tool_calls without `calls` must not succeed the fold or
        // lower to finish_reason=tool_calls via the production stream boundary.
        let mut fold = SemanticEventFold::new();
        fold.begin_attempt("req-21", 21);
        let err = fold
            .push(&serde_json::json!({
                "type": "tool_calls",
                "id": "req-21",
                "attempt_id": 21
            }))
            .expect_err("missing calls must fail fold push");
        match err {
            SemanticFoldError::MalformedToolCall { detail } => {
                assert!(
                    detail.contains("calls") && detail.contains("array"),
                    "detail={detail}"
                );
            }
            other => panic!("expected MalformedToolCall, got {other:?}"),
        }
        assert!(fold.buffered_tool_calls().is_empty());
        assert!(fold.executable_tool_calls().is_empty());
        assert!(fold.done().is_none());

        let stream_err = fold_complete_request_stream(
            "req-21",
            21,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "id": "req-21",
                    "attempt_id": 21,
                    "contract_version": 2
                }),
                serde_json::json!({
                    "type": "tool_calls",
                    "id": "req-21",
                    "attempt_id": 21
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "tool_calls",
                    "id": "req-21",
                    "attempt_id": 21
                }),
            ],
        )
        .expect_err("missing calls must fail complete_request fold boundary");
        match stream_err {
            StreamContractError::MalformedToolCall { detail } => {
                assert!(
                    detail.contains("calls") && detail.contains("array"),
                    "detail={detail}"
                );
            }
            other => panic!("expected StreamContractError::MalformedToolCall, got {other:?}"),
        }
    }

    #[test]
    fn semantic_fold_non_array_calls_fails_closed_before_tool_terminal() {
        // null / object / string `calls` are not arrays — fail closed before any
        // successful tool_calls terminal lowering can be produced.
        for calls in [
            serde_json::Value::Null,
            serde_json::json!({ "name": "read_file" }),
            serde_json::json!("read_file"),
        ] {
            let mut fold = SemanticEventFold::new();
            fold.begin_attempt("req-22", 22);
            let err = fold
                .push(&serde_json::json!({
                    "type": "tool_calls",
                    "calls": calls,
                    "id": "req-22",
                    "attempt_id": 22
                }))
                .expect_err("non-array calls must fail fold push");
            match err {
                SemanticFoldError::MalformedToolCall { detail } => {
                    assert!(
                        detail.contains("calls") && detail.contains("array"),
                        "detail={detail}"
                    );
                }
                other => panic!("expected MalformedToolCall, got {other:?}"),
            }
            assert!(fold.buffered_tool_calls().is_empty());
            assert!(fold.executable_tool_calls().is_empty());
            assert!(fold.done().is_none());
        }

        let stream_err = fold_complete_request_stream(
            "req-22",
            22,
            &[
                serde_json::json!({
                    "type": "gen_start",
                    "id": "req-22",
                    "attempt_id": 22,
                    "contract_version": 2
                }),
                serde_json::json!({
                    "type": "tool_calls",
                    "calls": null,
                    "id": "req-22",
                    "attempt_id": 22
                }),
                serde_json::json!({
                    "type": "done",
                    "finish_reason": "tool_calls",
                    "id": "req-22",
                    "attempt_id": 22
                }),
            ],
        )
        .expect_err("null calls must fail complete_request fold boundary");
        match stream_err {
            StreamContractError::MalformedToolCall { detail } => {
                assert!(
                    detail.contains("calls") && detail.contains("array"),
                    "detail={detail}"
                );
            }
            other => panic!("expected StreamContractError::MalformedToolCall, got {other:?}"),
        }
    }

    #[test]
    fn missing_or_lossy_endpoint_adapter_rejects_before_mutation() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mutations = AtomicUsize::new(0);
        let mut fire_if_allowed = |body: &serde_json::Value| -> Result<(), EndpointAdapterError> {
            gate_chat_completions_tools(body)?;
            mutations.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };

        let with_tools = serde_json::json!({
            "model": "m",
            "tools": [{ "type": "function", "function": { "name": "x" } }],
            "messages": []
        });

        assert_eq!(
            endpoint_adapter_status(EndpointAdapterKind::OpenAiChatCompletions),
            EndpointAdapterStatus::AvailableLossless
        );
        fire_if_allowed(&with_tools).expect("lossless adapter allows tools");
        assert_eq!(mutations.load(Ordering::SeqCst), 1);

        let before = mutations.load(Ordering::SeqCst);
        let deny_unavailable = |body: &serde_json::Value| -> Result<(), EndpointAdapterError> {
            let _ = body;
            Err(EndpointAdapterError::Unavailable {
                endpoint: "/v1/chat/completions",
            })
        };
        let deny_lossy = |body: &serde_json::Value| -> Result<(), EndpointAdapterError> {
            let _ = body;
            Err(EndpointAdapterError::Lossy {
                endpoint: "/v1/chat/completions",
            })
        };

        let run_gated =
            |gate: &dyn Fn(&serde_json::Value) -> Result<(), EndpointAdapterError>| match gate(
                &with_tools,
            ) {
                Ok(()) => {
                    mutations.fetch_add(1, Ordering::SeqCst);
                }
                Err(_) => {}
            };

        run_gated(&deny_unavailable);
        assert_eq!(
            mutations.load(Ordering::SeqCst),
            before,
            "unavailable adapter must not fire mutation counter"
        );
        let msg = EndpointAdapterError::Unavailable {
            endpoint: "/v1/chat/completions",
        }
        .to_string();
        assert!(msg.contains("unavailable"), "{msg}");

        run_gated(&deny_lossy);
        assert_eq!(
            mutations.load(Ordering::SeqCst),
            before,
            "lossy adapter must not fire mutation counter"
        );
        let msg = EndpointAdapterError::Lossy {
            endpoint: "/v1/chat/completions",
        }
        .to_string();
        assert!(msg.contains("lossy"), "{msg}");

        assert!(gate_chat_completions_tools(&with_tools).is_ok());
    }

    #[test]
    fn tools_absent_bypasses_adapter_capability_gate() {
        let mutations = std::sync::atomic::AtomicUsize::new(0);
        let bodies = [
            serde_json::json!({ "model": "m", "messages": [] }),
            serde_json::json!({ "model": "m", "tools": [], "messages": [] }),
            serde_json::json!({ "model": "m", "tools": null, "messages": [] }),
        ];
        for body in &bodies {
            gate_chat_completions_tools(body).expect("tool-free must bypass adapter capability");
            mutations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        assert_eq!(
            mutations.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "absent/empty tools must not block the mutation/dispatch path"
        );
    }

    #[test]
    fn endpoint_adapter_registry_covers_all_declared_kinds() {
        const ALL_KINDS: &[EndpointAdapterKind] = &[EndpointAdapterKind::OpenAiChatCompletions];

        for kind in ALL_KINDS {
            let status = endpoint_adapter_status(*kind);
            match kind {
                EndpointAdapterKind::OpenAiChatCompletions => {
                    assert_eq!(status, EndpointAdapterStatus::AvailableLossless);
                }
            }
            assert_eq!(status, EndpointAdapterRegistry::status(*kind));
        }

        assert!(
            ALL_KINDS
                .iter()
                .any(|k| endpoint_adapter_status(*k) == EndpointAdapterStatus::AvailableLossless),
            "registry must declare at least one AvailableLossless adapter"
        );
    }

    #[test]
    fn forward_sse_stream_event_sends_delta_bytes() {
        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        forward_sse_stream_event(
            &sender,
            "chatcmpl-test",
            1,
            "m",
            &serde_json::json!({ "type": "token", "text": "hi" }),
        )
        .expect("delta path succeeds");
        let chunk = receiver.recv().expect("delta payload");
        assert!(!chunk.fail);
        assert!(chunk.ack.is_none());
        let text = String::from_utf8(chunk.bytes).expect("utf8");
        assert!(text.starts_with("data: "));
        assert!(text.contains("\"content\":\"hi\""));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn forward_sse_stream_event_no_delta_is_silent() {
        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        // Pure tool turn is withheld mid-stream — no empty probe bytes.
        forward_sse_stream_event(
            &sender,
            "chatcmpl-test",
            1,
            "m",
            &serde_json::json!({
                "type": "tool_calls",
                "calls": [{ "name": "read_file", "arguments": {} }]
            }),
        )
        .expect("no-delta path succeeds");
        assert!(
            receiver.try_recv().is_err(),
            "no-delta mid-stream must not enqueue a chunk"
        );
    }

    #[test]
    fn forward_sse_stream_event_dropped_receiver_is_cancelled_on_delta_path() {
        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        drop(receiver);
        let err = forward_sse_stream_event(
            &sender,
            "chatcmpl-test",
            1,
            "m",
            &serde_json::json!({ "type": "token", "text": "x" }),
        )
        .expect_err("delta send must fail when receiver dropped");
        assert!(
            matches!(err, hipfire_client::ClientError::Cancelled),
            "delta path: {err:?}"
        );

        // No-delta path does not touch the sender, so a dropped receiver is Ok.
        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        drop(receiver);
        forward_sse_stream_event(
            &sender,
            "chatcmpl-test",
            1,
            "m",
            &serde_json::json!({ "type": "tool_calls", "calls": [] }),
        )
        .expect("no-delta path is silent even if receiver already dropped");
    }

    #[test]
    fn channel_reader_skips_empty_non_fail_chunks() {
        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        let reader_thread = thread::spawn(move || {
            let mut reader = ChannelReader::new(receiver);
            let mut buf = [0u8; 64];
            let n = reader.read(&mut buf).expect("read");
            (n, buf)
        });
        sender
            .send(ResponseChunk::plain(Vec::new()))
            .expect("empty probe");
        sender
            .send(ResponseChunk::plain(Vec::new()))
            .expect("second empty probe");
        sender
            .send(ResponseChunk::plain(b"data: hi\n\n".to_vec()))
            .expect("real bytes");
        drop(sender);
        let (n, buf) = reader_thread.join().expect("reader joins");
        assert_eq!(n, b"data: hi\n\n".len());
        assert_eq!(&buf[..n], b"data: hi\n\n");
    }

    #[test]
    fn channel_reader_acks_only_after_full_chunk_consumption() {
        use std::sync::mpsc::TryRecvError;
        use std::time::Duration;

        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        let (ack_tx, ack_rx) = mpsc::channel::<Result<(), ()>>();
        // Chunk larger than the read buffer so the first/intermediate reads are partial.
        let chunk = b"abcdefghij".to_vec(); // 10 bytes
        sender
            .send(ResponseChunk {
                bytes: chunk.clone(),
                ack: Some(ack_tx),
                fail: false,
            })
            .expect("send acknowledged chunk");
        drop(sender);

        let mut reader = ChannelReader::new(receiver);
        let mut buf = [0u8; 3];

        // First partial read — chunk not fully consumed; no ack yet.
        let n = reader.read(&mut buf).expect("first partial");
        assert_eq!(n, 3);
        assert_eq!(&buf[..n], b"abc");
        assert!(
            matches!(ack_rx.try_recv(), Err(TryRecvError::Empty)),
            "first partial read must not acknowledge"
        );

        // Intermediate partial reads — still draining; no ack.
        let n = reader.read(&mut buf).expect("second partial");
        assert_eq!(n, 3);
        assert_eq!(&buf[..n], b"def");
        assert!(
            matches!(ack_rx.try_recv(), Err(TryRecvError::Empty)),
            "second partial read must not acknowledge"
        );

        let n = reader.read(&mut buf).expect("third partial");
        assert_eq!(n, 3);
        assert_eq!(&buf[..n], b"ghi");
        assert!(
            matches!(ack_rx.try_recv(), Err(TryRecvError::Empty)),
            "third partial read must not acknowledge"
        );

        // Final drain of remaining byte — still no ack until a *later* read.
        let n = reader.read(&mut buf).expect("final drain");
        assert_eq!(n, 1);
        assert_eq!(&buf[..n], b"j");
        assert!(
            matches!(ack_rx.try_recv(), Err(TryRecvError::Empty)),
            "full drain of current chunk still defers ack to next read"
        );

        // Fifth read after full consumption fires the progress ack, then EOF.
        let n = reader.read(&mut buf).expect("post-drain read");
        assert_eq!(n, 0);
        let ack = ack_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("ack after full consumption");
        assert_eq!(ack, Ok(()));
    }

    #[test]
    fn forward_think_fragments_preserves_cancelled_callback_error() {
        let mut content = String::new();
        let mut reasoning = String::new();
        let err = forward_think_fragments(
            vec![ThinkFragment::Content("x".into())],
            &mut content,
            &mut reasoning,
            &mut |_| Err(hipfire_client::ClientError::Cancelled),
        )
        .expect_err("callback Cancelled must surface typed");
        assert!(matches!(err, hipfire_client::ClientError::Cancelled));
        // Fragment still applied before callback failure (accumulation is local).
        assert_eq!(content, "x");
    }

    #[test]
    fn finish_sse_stream_cancelled_emits_neither_error_nor_done() {
        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        finish_sse_stream(
            sender,
            Err(anyhow::Error::from(hipfire_client::ClientError::Cancelled)),
        );
        let trailing: Vec<ResponseChunk> = receiver.try_iter().collect();
        assert!(
            trailing.is_empty(),
            "Cancelled must drop sender without frames: {trailing:?}"
        );
    }

    #[test]
    fn finish_sse_stream_success_emits_no_post_commit_bytes() {
        let (sender, receiver) = mpsc::channel::<ResponseChunk>();
        let completion = sample_completion("ok", Vec::new(), "stop");
        finish_sse_stream(sender, Ok(completion));
        let frames: Vec<ResponseChunk> = receiver.try_iter().collect();
        assert!(
            frames.is_empty(),
            "success terminal already delivered at commit_ready: {frames:?}"
        );
    }

    // =========================================================================
    // Task 11 — no-GPU fake-daemon HTTP acceptance through real serve lowering
    // =========================================================================

    /// Unix-only JSONL fake daemon used by the Task 11 HTTP matrix.
    /// Scenario selection is driven by generate request prompt/model fixture tags.
    #[cfg(unix)]
    fn write_task11_fake_daemon(root: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let daemon = root.join("task11-fake-daemon.py");
        // Python keeps correlated id/attempt_id, full reset ack, and commit handshake.
        let script = r#"#!/usr/bin/env python3
import json, os, sys

state_epoch = 0
generate_count = 0
LAST_SCENARIO = ""
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests.log")
MODEL_PATH = ""

def log_req(req):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(req, separators=(",", ":")) + "\n")
    except Exception:
        pass

def out(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def echo_ids(req):
    return req.get("id"), req.get("attempt_id")

def eligible_from_model():
    # Task 15: "ineligible" in model path/name => retry_reset_eligible false.
    blob = (MODEL_PATH or "").lower()
    return "ineligible" not in blob

def scenario_from(req):
    model = str(req.get("model") or "")
    prompt = str(req.get("prompt") or "")
    messages = req.get("messages") or []
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c:
                    prompt = c
                    break
    blob = (model + " " + prompt).lower()
    tags = (
        "t15-transient-once",
        "t15-transient-always",
        "t15-visible-token",
        "t15-visible-reasoning",
        "t15-commit-ready-error",
        "t15-class-malformed",
        "t15-class-validation",
        "t15-class-context",
        "t15-class-unsupported",
        "t15-class-internal",
        "t15-class-adaptive",
        "t15-class-mismatch",
        "t15-class-cancel",
        "t15-transient-not-retryable",
        "t15-mismatch-attempt",
        "t15-eof",
        "t15-invalid-json",
        "t15-stale-event",
        "t15-tool-then-transient",
        "t15-reset-fail-rolled",
        "t15-reset-fail-seq",
        "t15-reset-fail-epoch",
        "t15-reset-fail-attempt",
        "t11-premature-eof",
        "t11-capability-denial",
        "t11-dirty-markers",
        "t11-length-withhold",
        "t11-mixed-tool",
        "t11-two-tools",
        "t11-pure-tool",
        "t11-stop-text",
        "t11-usage",
    )
    for tag in tags:
        if tag in blob:
            return tag
    return "t11-stop-text"

def emit_correlated(ev, rid, aid):
    if rid is not None:
        ev["id"] = rid
    if aid is not None:
        ev["attempt_id"] = aid
    out(ev)

def emit_typed_error(rid, aid, message, cls="transient", retryable=True, rolled_back=False, force_aid=None):
    out({
        "type": "error",
        "id": rid,
        "message": message,
        "class": cls,
        "retryable": retryable,
        "rolled_back": rolled_back,
        "attempt_id": force_aid if force_aid is not None else (aid if aid is not None else 0),
    })

def wait_commit(rid, aid, allow_abort=False):
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except Exception:
            continue
        log_req(msg)
        ty = msg.get("type")
        if ty == "commit":
            if msg.get("id") != rid or msg.get("attempt_id") != aid:
                emit_typed_error(rid, aid, "commit correlation mismatch", cls="internal", retryable=False)
                return "error"
            return "commit"
        if ty == "abort" and allow_abort:
            return "abort"
        if ty == "unload":
            out({"type": "unloaded"})
            sys.exit(0)

def success_stop(rid, aid, text="hello from fake daemon"):
    emit_correlated({"type": "token", "text": text}, rid, aid)
    emit_correlated({
        "type": "commit_ready",
        "finish_reason": "stop",
        "prompt_tokens": 3,
        "tokens": 4,
        "tok_s": 12.0,
    }, rid, aid)
    if wait_commit(rid, aid) != "commit":
        return
    emit_correlated({
        "type": "done",
        "finish_reason": "stop",
        "prompt_tokens": 3,
        "tokens": 4,
        "tok_s": 12.0,
    }, rid, aid)

def handle_generate(req):
    global generate_count, LAST_SCENARIO
    generate_count += 1
    rid, aid = echo_ids(req)
    scenario = scenario_from(req)
    LAST_SCENARIO = scenario

    if scenario == "t11-capability-denial":
        out({
            "type": "error",
            "id": rid,
            "message": "tools not supported by this endpoint capability",
            "class": "unsupported",
            "retryable": False,
            "rolled_back": True,
            "attempt_id": aid if aid is not None else 0,
        })
        return

    # All success / premature / t15 paths start with correlated v2 gen_start
    # except pure typed pre-start errors above.
    emit_correlated({
        "type": "gen_start",
        "contract_version": 2,
    }, rid, aid)

    # --- Task 15 scenarios ---
    if scenario == "t15-transient-once":
        if generate_count == 1:
            emit_typed_error(rid, aid, "transient prefill glitch")
            return
        success_stop(rid, aid, text="retry-recovered-content")
        return

    if scenario == "t15-transient-always":
        emit_typed_error(rid, aid, "persistent transient fault")
        return

    if scenario == "t15-visible-token":
        emit_correlated({"type": "token", "text": "visible-before-fail"}, rid, aid)
        emit_typed_error(rid, aid, "transient after visible token")
        return

    if scenario == "t15-visible-reasoning":
        emit_correlated({"type": "reasoning", "text": "think-before-fail"}, rid, aid)
        emit_typed_error(rid, aid, "transient after visible reasoning")
        return

    if scenario == "t15-commit-ready-error":
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "tokens": 1,
            "tok_s": 1.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_typed_error(rid, aid, "transient after commit_ready", cls="transient", retryable=True)
        return

    class_map = {
        "t15-class-malformed": ("malformed", False, "malformed payload"),
        "t15-class-validation": ("validation", False, "validation failed"),
        "t15-class-context": ("context_length", False, "context too long"),
        "t15-class-unsupported": ("unsupported", False, "unsupported op"),
        "t15-class-internal": ("internal", False, "internal fault"),
        "t15-class-adaptive": ("adaptive_poison", False, "adaptive poison"),
        "t15-class-mismatch": ("deterministic_mismatch", False, "deterministic mismatch"),
        "t15-class-cancel": ("cancel", False, "cancelled"),
        "t15-transient-not-retryable": ("transient", False, "transient but not retryable"),
    }
    if scenario in class_map:
        cls, retryable, msg = class_map[scenario]
        emit_typed_error(rid, aid, msg, cls=cls, retryable=retryable)
        return

    if scenario == "t15-mismatch-attempt":
        bad = (aid + 999) if isinstance(aid, int) else 999999
        emit_typed_error(rid, aid, "stale attempt error", force_aid=bad)
        return

    if scenario == "t15-eof":
        # Exit after gen_start with no done — engine sees Closed.
        sys.exit(0)

    if scenario == "t15-invalid-json":
        sys.stdout.write("{not-json\n")
        sys.stdout.flush()
        return

    if scenario == "t15-stale-event":
        # Correlated gen_start already emitted; now a stale-attempt token.
        stale_aid = (aid - 1) if isinstance(aid, int) and aid else 0
        emit_correlated({"type": "token", "text": "stale"}, rid, stale_aid)
        return

    if scenario == "t15-tool-then-transient":
        if generate_count == 1:
            emit_correlated({
                "type": "tool_calls",
                "calls": [{"name": "read_file", "arguments": {"path": "stale.rs"}}],
            }, rid, aid)
            emit_typed_error(rid, aid, "transient after buffered tools")
            return
        success_stop(rid, aid, text="fold-cleared-content")
        return

    # reset-fail: first generate is typed transient so server force-resets for attempt 2.
    if scenario.startswith("t15-reset-fail"):
        emit_typed_error(rid, aid, "transient before reset-fail")
        return

    if scenario == "t11-premature-eof":
        emit_correlated({"type": "token", "text": "partial-before-eof"}, rid, aid)
        sys.exit(0)

    if scenario == "t11-stop-text":
        success_stop(rid, aid)
        return

    if scenario == "t11-pure-tool":
        pure_calls = [{"name": "read_file", "arguments": {"path": "a.rs"}}]
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 9.0,
            "calls": pure_calls,
        }, rid, aid)
        rc = wait_commit(rid, aid, allow_abort=True)
        if rc == "abort":
            emit_correlated({"type": "aborted", "reason": "client_cancelled"}, rid, aid)
            emit_correlated({"type": "done", "finish_reason": "aborted"}, rid, aid)
            return
        if rc != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 9.0,
            "calls": pure_calls,
        }, rid, aid)
        return

    if scenario == "t11-mixed-tool":
        mixed_calls = [{"name": "read_file", "arguments": {"path": "mixed.rs"}}]
        emit_correlated({"type": "token", "text": "I'll look that up."}, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.5,
            "calls": mixed_calls,
        }, rid, aid)
        rc = wait_commit(rid, aid, allow_abort=True)
        if rc == "abort":
            emit_correlated({"type": "aborted", "reason": "client_cancelled"}, rid, aid)
            emit_correlated({"type": "done", "finish_reason": "aborted"}, rid, aid)
            return
        if rc != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.5,
            "calls": mixed_calls,
        }, rid, aid)
        return

    if scenario == "t11-two-tools":
        two_calls = [
            {"name": "read_file", "arguments": {"path": "a.rs"}},
            {"name": "write_file", "arguments": {"path": "b.rs", "data": "x"}},
        ]
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.0,
            "calls": two_calls,
        }, rid, aid)
        rc = wait_commit(rid, aid, allow_abort=True)
        if rc == "abort":
            emit_correlated({"type": "aborted", "reason": "client_cancelled"}, rid, aid)
            emit_correlated({"type": "done", "finish_reason": "aborted"}, rid, aid)
            return
        if rc != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "tool_calls",
            "prompt_tokens": 2,
            "tokens": 2,
            "tok_s": 8.0,
            "calls": two_calls,
        }, rid, aid)
        return

    if scenario == "t11-length-withhold":
        emit_correlated({"type": "token", "text": "partial-length"}, rid, aid)
        emit_correlated({
            "type": "tool_calls",
            "calls": [{"name": "read_file", "arguments": {"path": "x"}}],
        }, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "length",
            "prompt_tokens": 2,
            "tokens": 3,
            "tok_s": 7.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "length",
            "prompt_tokens": 2,
            "tokens": 3,
            "tok_s": 7.0,
        }, rid, aid)
        return

    if scenario == "t11-dirty-markers":
        dirty = (
            '<tool_call>{"name":"evil","arguments":{}}</tool_call>'
            '<think>secret</think></think><|im_end|>'
        )
        emit_correlated({"type": "token", "text": dirty}, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 6.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "tokens": 1,
            "tok_s": 6.0,
        }, rid, aid)
        return

    if scenario == "t11-usage":
        emit_correlated({"type": "token", "text": "usage-path"}, rid, aid)
        emit_correlated({
            "type": "commit_ready",
            "finish_reason": "stop",
            "prompt_tokens": 11,
            "tokens": 5,
            "cached_tokens": 2,
            "tok_s": 10.0,
        }, rid, aid)
        if wait_commit(rid, aid) != "commit":
            return
        emit_correlated({
            "type": "done",
            "finish_reason": "stop",
            "prompt_tokens": 11,
            "tokens": 5,
            "cached_tokens": 2,
            "tok_s": 10.0,
        }, rid, aid)
        return

    # Default stop text
    success_stop(rid, aid)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    log_req(req)
    ty = req.get("type")
    if ty == "configure":
        out({"type": "configured"})
    elif ty == "ping":
        out({"type": "pong"})
    elif ty == "load":
        MODEL_PATH = str(req.get("model") or "")
        out({
            "type": "loaded",
            "arch": "fake",
            "dim": 1,
            "layers": 1,
            "vocab": 1,
            "vl": False,
            # cache_capable true so only force_reset (retry attempt) issues reset
            "cache_capable": True,
            "retry_reset_eligible": eligible_from_model(),
            "max_seq": 4096,
        })
    elif ty == "reset":
        aid = req.get("attempt_id")
        sc = (LAST_SCENARIO or "") + " " + (MODEL_PATH or "")
        sc = sc.lower()
        if "t15-reset-fail-rolled" in sc:
            out({
                "type": "reset",
                "rolled_back": False,
                "state_epoch": state_epoch + 1,
                "seq_pos": 0,
                "conversation_len": 0,
                "attempt_id": aid,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        if "t15-reset-fail-seq" in sc:
            out({
                "type": "reset",
                "rolled_back": True,
                "state_epoch": state_epoch + 1,
                "seq_pos": 1,
                "conversation_len": 0,
                "attempt_id": aid,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        if "t15-reset-fail-epoch" in sc:
            out({
                "type": "reset",
                "rolled_back": True,
                "state_epoch": state_epoch if state_epoch > 0 else 0,
                "seq_pos": 0,
                "conversation_len": 0,
                "attempt_id": aid,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        if "t15-reset-fail-attempt" in sc:
            out({
                "type": "reset",
                "rolled_back": True,
                "state_epoch": state_epoch + 1,
                "seq_pos": 0,
                "conversation_len": 0,
                "attempt_id": (aid + 1) if isinstance(aid, int) else 0,
                "retry_reset_eligible": eligible_from_model(),
            })
            continue
        state_epoch += 1
        out({
            "type": "reset",
            "rolled_back": True,
            "state_epoch": state_epoch,
            "seq_pos": 0,
            "conversation_len": 0,
            "attempt_id": aid,
            "retry_reset_eligible": eligible_from_model(),
        })
    elif ty == "generate":
        handle_generate(req)
    elif ty == "unload":
        out({"type": "unloaded"})
        sys.exit(0)
    elif ty == "commit":
        pass
    else:
        out({
            "type": "error",
            "message": f"unsupported op {ty}",
            "class": "validation",
            "retryable": False,
            "rolled_back": False,
            "id": "req-0",
            "attempt_id": 0,
        })
"#;
        fs::write(&daemon, script).unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
        daemon
    }

    /// In-process tiny_http harness: Engine::spawn_configured → handle_http.
    /// Does not touch HIPFIRE_DAEMON_BIN.
    #[cfg(unix)]
    struct Task11HttpHarness {
        paths: Paths,
        port: u16,
        model_name: String,
        shared: Arc<ServeShared>,
        _server: Arc<Server>,
        _join: Option<thread::JoinHandle<()>>,
        stop: Arc<AtomicBool>,
    }

    #[cfg(unix)]
    impl Task11HttpHarness {
        fn spawn(label: &str) -> Self {
            Self::spawn_inner(label, false, Duration::from_millis(0))
        }

        /// Retry-enabled variant for the Task 15 one-retry scenarios.
        fn spawn_with_retry(label: &str, retry_backoff: Duration) -> Self {
            Self::spawn_inner(label, true, retry_backoff)
        }

        fn spawn_inner(label: &str, retry_enabled: bool, retry_backoff: Duration) -> Self {
            let paths = test_paths(label);
            fs::create_dir_all(&paths.models).unwrap();
            fs::create_dir_all(&paths.root).unwrap();

            let model_name = format!("t11-fixture-{label}.hfq");
            let model_path = paths.models.join(&model_name);
            fs::write(&model_path, b"task11-dummy-model").unwrap();

            let daemon = write_task11_fake_daemon(&paths.root);
            let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
            let process_config = hipfire_config::ProcessConfig::from_resolved(&resolved).unwrap();

            // Bounded ETXTBSY retry like hipfire-client fake daemons.
            const ETXTBSY: i32 = 26;
            let mut engine = None;
            let mut last = None;
            for attempt in 0..8 {
                match Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config) {
                    Ok(e) => {
                        engine = Some(e);
                        break;
                    }
                    Err(hipfire_client::ClientError::Spawn { source, path })
                        if source.raw_os_error() == Some(ETXTBSY) =>
                    {
                        last = Some(format!("spawn {path:?}: {source}"));
                        thread::sleep(Duration::from_millis(
                            5u64.saturating_mul(1 + attempt as u64),
                        ));
                    }
                    Err(err) => panic!("Task11HttpHarness spawn non-retryable: {err}"),
                }
            }
            let mut engine = engine.unwrap_or_else(|| {
                panic!(
                    "Task11HttpHarness exhausted ETXTBSY retries: {}",
                    last.unwrap_or_default()
                )
            });
            engine.ping().expect("fake daemon ping");

            let registry = hipfire_registry::bundled().unwrap();
            let shared = Arc::new(ServeShared {
                // Test harness drives the single-daemon path.
                slot_engine: None,
                runtime: Mutex::new(ServeRuntime {
                    engine,
                    paths: paths.clone(),
                    registry,
                    current_path: None,
                    current_arch: None,
                    continuous_batch_capable: false,
                    current_max_seq: 0,
                    cache_capable: false,
                    kv_override: None,
                    kv_backend_override: None,
                    tp: None,
                    continuous_batch_size: 1,
                }),
                meta: Mutex::new(ServeMeta {
                    current_model: None,
                    loading_model: None,
                    instance_token: serve_instance_token(),
                    requests_served: 0,
                    retries_attempted: 0,
                    retries_succeeded: 0,
                    recent_tok_s: None,
                    started: Instant::now(),
                    last_activity: Instant::now(),
                }),
                max_request_bytes: 8 * 1024 * 1024,
                admission: Arc::new(Admission::new(4, Duration::from_secs(5))),
                idle_timeout: Duration::from_secs(0),
                retry_enabled,
                retry_backoff,
                backoff_hook: Mutex::new(None),
            });

            let server = Arc::new(Server::http("127.0.0.1:0").expect("bind ephemeral serve port"));
            let port = server.server_addr().to_ip().expect("ip listen addr").port();

            let stop = Arc::new(AtomicBool::new(false));
            let stop_flag = Arc::clone(&stop);
            let shared_loop = Arc::clone(&shared);
            let server_loop = Arc::clone(&server);
            let join = thread::spawn(move || {
                while !stop_flag.load(Ordering::Relaxed) {
                    match server_loop.recv_timeout(Duration::from_millis(50)) {
                        Ok(Some(request)) => {
                            let shared = Arc::clone(&shared_loop);
                            // handle_http owns the request; keep sequential so the
                            // single-engine fake daemon never races generate.
                            if let Err(error) = handle_http(request, shared) {
                                eprintln!("[task11-harness] HTTP request failed: {error:#}");
                            }
                        }
                        Ok(None) => {}
                        Err(_) => break,
                    }
                }
            });

            // Health probe — proves handle_http path is live.
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if hipfire_client::service_ready("127.0.0.1", port, Duration::from_millis(200)) {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            assert!(
                hipfire_client::service_ready("127.0.0.1", port, Duration::from_millis(500)),
                "task11 harness never became ready on port {port}"
            );

            Self {
                paths,
                port,
                model_name: model_path.display().to_string(),
                shared,
                _server: server,
                _join: Some(join),
                stop,
            }
        }

        fn port(&self) -> u16 {
            self.port
        }

        fn model(&self) -> &str {
            &self.model_name
        }

        fn base_body(&self, scenario_tag: &str, stream: bool) -> serde_json::Value {
            // Encode scenario in both model (direct path still resolves file) and
            // user prompt so the fake daemon can select without external deps.
            serde_json::json!({
                "model": self.model(),
                "stream": stream,
                "messages": [{
                    "role": "user",
                    "content": format!("{scenario_tag} please")
                }],
            })
        }

        fn tools_body(&self, scenario_tag: &str, stream: bool) -> serde_json::Value {
            let mut body = self.base_body(scenario_tag, stream);
            body["tools"] = serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read a file",
                    "parameters": {
                        "type": "object",
                        "properties": { "path": { "type": "string" } }
                    }
                }
            }]);
            body
        }

        fn requests_log_path(&self) -> PathBuf {
            self.paths.root.join("requests.log")
        }

        fn read_requests_log(&self) -> Vec<serde_json::Value> {
            let raw = fs::read_to_string(self.requests_log_path()).unwrap_or_default();
            raw.lines()
                .filter(|line| !line.trim().is_empty())
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect()
        }

        fn meta_retries(&self) -> (u64, u64) {
            let meta = self
                .shared
                .meta
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            (meta.retries_attempted, meta.retries_succeeded)
        }

        fn set_backoff_hook<F>(&self, hook: F)
        where
            F: Fn(Duration) + Send + Sync + 'static,
        {
            let mut slot = self
                .shared
                .backoff_hook
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            *slot = Some(Arc::new(hook));
        }

        fn ops_of_type<'a>(log: &'a [serde_json::Value], ty: &str) -> Vec<&'a serde_json::Value> {
            log.iter()
                .filter(|row| row.get("type").and_then(|v| v.as_str()) == Some(ty))
                .collect()
        }
    }

    #[cfg(unix)]
    impl Drop for Task11HttpHarness {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            self._server.unblock();
            if let Some(join) = self._join.take() {
                let _ = join.join();
            }
            // Dropping Engine (inside ServeShared via Arc) kills the fake child.
            // ServeShared is held only by the server thread which has exited.
            let _ = fs::remove_dir_all(&self.paths.root);
        }
    }

    #[cfg(unix)]
    #[derive(Debug, Default)]
    struct StreamCapture {
        content: String,
        reasoning: String,
        /// Individual content deltas (for per-chunk leak assertions).
        content_deltas: Vec<String>,
        /// Individual reasoning deltas (for per-chunk leak assertions).
        reasoning_deltas: Vec<String>,
        tool_calls: Vec<(u32, Option<String>, Option<String>, Option<String>)>,
        finish: Option<String>,
        usage: Option<serde_json::Value>,
        saw_done: bool,
        saw_role: bool,
    }

    /// Tool-call protocol markers that must never appear in valid-path content/reasoning.
    #[cfg(unix)]
    const TASK11_TOOL_PROTOCOL_MARKERS: &[&str] = &[
        "<tool_call>",
        "</tool_call>",
        "<tool_calls>",
        "</tool_calls>",
        "<|tool_call|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "call tool",
        "invoke tool",
    ];

    /// Assert visible text from a *valid structured-call* path has zero protocol
    /// markers and zero JSON argument fragments belonging to structured calls.
    #[cfg(unix)]
    fn assert_valid_path_text_clean(label: &str, text: &str, forbidden_arg_frags: &[&str]) {
        for marker in TASK11_TOOL_PROTOCOL_MARKERS {
            assert!(
                !text.contains(marker),
                "{label}: content/reasoning leaked tool protocol marker {marker:?} in {text:?}"
            );
        }
        for frag in forbidden_arg_frags {
            if frag.is_empty() {
                continue;
            }
            assert!(
                !text.contains(frag),
                "{label}: content/reasoning leaked structured-call argument fragment {frag:?} in {text:?}"
            );
        }
    }

    #[cfg(unix)]
    fn assert_nonstream_valid_structured_clean(
        label: &str,
        json: &serde_json::Value,
        forbidden_arg_frags: &[&str],
    ) {
        let message = &json["choices"][0]["message"];
        match message.get("content") {
            None | Some(serde_json::Value::Null) => {}
            Some(serde_json::Value::String(content)) => {
                assert_valid_path_text_clean(
                    &format!("{label}/nonstream.content"),
                    content,
                    forbidden_arg_frags,
                );
            }
            Some(other) => panic!("{label}: unexpected content shape {other}"),
        }
        if let Some(reasoning) = message
            .get("reasoning_content")
            .and_then(serde_json::Value::as_str)
        {
            assert_valid_path_text_clean(
                &format!("{label}/nonstream.reasoning"),
                reasoning,
                forbidden_arg_frags,
            );
        }
        // Calls/arguments may appear only under message.tool_calls.
        if let Some(calls) = message
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
        {
            assert!(
                !calls.is_empty(),
                "{label}: empty tool_calls array is not a structured release"
            );
            for call in calls {
                assert!(
                    call.get("function").and_then(|f| f.get("name")).is_some(),
                    "{label}: structured tool_calls entry missing function.name"
                );
            }
        }
    }

    #[cfg(unix)]
    fn assert_stream_valid_structured_clean(
        label: &str,
        cap: &StreamCapture,
        forbidden_arg_frags: &[&str],
    ) {
        assert_valid_path_text_clean(
            &format!("{label}/stream.content"),
            &cap.content,
            forbidden_arg_frags,
        );
        assert_valid_path_text_clean(
            &format!("{label}/stream.reasoning"),
            &cap.reasoning,
            forbidden_arg_frags,
        );
        for (i, delta) in cap.content_deltas.iter().enumerate() {
            assert_valid_path_text_clean(
                &format!("{label}/stream.content_delta[{i}]"),
                delta,
                forbidden_arg_frags,
            );
        }
        for (i, delta) in cap.reasoning_deltas.iter().enumerate() {
            assert_valid_path_text_clean(
                &format!("{label}/stream.reasoning_delta[{i}]"),
                delta,
                forbidden_arg_frags,
            );
        }
    }

    #[cfg(unix)]
    fn capture_stream(
        port: u16,
        body: serde_json::Value,
    ) -> std::result::Result<StreamCapture, hipfire_client::ClientError> {
        let mut cap = StreamCapture::default();
        stream_openai_chat(
            "127.0.0.1",
            port,
            body,
            Duration::from_secs(10),
            |event| {
                match event {
                    OpenAiSseEvent::Role { .. } => cap.saw_role = true,
                    OpenAiSseEvent::Content { text } => {
                        cap.content_deltas.push(text.clone());
                        cap.content.push_str(&text);
                    }
                    OpenAiSseEvent::Reasoning { text } => {
                        cap.reasoning_deltas.push(text.clone());
                        cap.reasoning.push_str(&text);
                    }
                    OpenAiSseEvent::ToolCall {
                        index,
                        id,
                        name,
                        arguments,
                    } => cap.tool_calls.push((index, id, name, arguments)),
                    OpenAiSseEvent::Finish { reason, .. } => cap.finish = Some(reason),
                    OpenAiSseEvent::Usage { usage } => cap.usage = Some(usage),
                    OpenAiSseEvent::Done => cap.saw_done = true,
                }
                Ok(())
            },
            || false,
        )?;
        Ok(cap)
    }

    #[cfg(unix)]
    fn complete_nonstream(
        port: u16,
        body: serde_json::Value,
    ) -> std::result::Result<serde_json::Value, hipfire_client::ClientError> {
        complete_openai_chat("127.0.0.1", port, body, Duration::from_secs(10))
    }

    /// Paired stream/nonstream matrix rows for stop, pure-tool, mixed-tool,
    /// two-tools, length-withhold, and usage ordering.
    ///
    /// Valid structured-call rows hard-fail on any tool-protocol marker or
    /// structured-argument JSON fragment leaking into content/reasoning.
    /// Invalid-producer dirty-marker diagnostics live in a separate test and
    /// must not weaken these valid-path assertions.
    #[cfg(unix)]
    #[test]
    fn task11_http_acceptance_matrix_stream_and_nonstream_parity() {
        let harness = Task11HttpHarness::spawn("matrix");
        let port = harness.port();

        // --- stop text parity ---
        {
            let ns = complete_nonstream(port, harness.base_body("t11-stop-text", false))
                .expect("stop nonstream");
            assert_eq!(ns["choices"][0]["finish_reason"], "stop");
            assert_eq!(
                ns["choices"][0]["message"]["content"],
                "hello from fake daemon"
            );
            assert!(ns["choices"][0]["message"].get("tool_calls").is_none());

            let st = capture_stream(port, harness.base_body("t11-stop-text", true))
                .expect("stop stream");
            assert!(st.saw_done, "every successful stream ends [DONE]");
            assert_eq!(st.finish.as_deref(), Some("stop"));
            assert_eq!(st.content, "hello from fake daemon");
            assert!(st.tool_calls.is_empty());
            assert_eq!(
                ns["choices"][0]["message"]["content"].as_str().unwrap(),
                st.content
            );
        }

        // --- pure tool call → content:null, call_0; no marker/arg leak ---
        {
            let pure_args = [r#"{"path":"a.rs"}"#, r#""path":"a.rs""#, "a.rs"];
            // "a.rs" alone is too short/common for content prose; use JSON frags only.
            let pure_frags = [r#"{"path":"a.rs"}"#, r#""path":"a.rs""#];
            let ns = complete_nonstream(port, harness.tools_body("t11-pure-tool", false))
                .expect("pure tool nonstream");
            assert_eq!(ns["choices"][0]["finish_reason"], "tool_calls");
            assert!(ns["choices"][0]["message"]["content"].is_null());
            assert_eq!(ns["choices"][0]["message"]["tool_calls"][0]["id"], "call_0");
            assert_eq!(
                ns["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "read_file"
            );
            assert_eq!(
                ns["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                pure_args[0]
            );
            assert_nonstream_valid_structured_clean("pure-tool", &ns, &pure_frags);

            let st = capture_stream(port, harness.tools_body("t11-pure-tool", true))
                .expect("pure tool stream");
            assert!(st.saw_done);
            assert_eq!(st.finish.as_deref(), Some("tool_calls"));
            assert!(st.content.is_empty(), "pure tool has no content deltas");
            assert!(st.reasoning.is_empty());
            assert_eq!(st.tool_calls.len(), 1);
            assert_eq!(st.tool_calls[0].0, 0);
            assert_eq!(st.tool_calls[0].1.as_deref(), Some("call_0"));
            assert_eq!(st.tool_calls[0].2.as_deref(), Some("read_file"));
            assert_eq!(st.tool_calls[0].3.as_deref(), Some(pure_args[0]));
            assert_stream_valid_structured_clean("pure-tool", &st, &pure_frags);
        }

        // --- mixed content + structured call: prose clean, args only in tool_calls ---
        {
            let mixed_frags = [r#"{"path":"mixed.rs"}"#, r#""path":"mixed.rs""#];
            let ns = complete_nonstream(port, harness.tools_body("t11-mixed-tool", false))
                .expect("mixed tool nonstream");
            assert_eq!(ns["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(ns["choices"][0]["message"]["content"], "I'll look that up.");
            assert_eq!(ns["choices"][0]["message"]["tool_calls"][0]["id"], "call_0");
            assert_eq!(
                ns["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "read_file"
            );
            assert_eq!(
                ns["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                r#"{"path":"mixed.rs"}"#
            );
            assert_nonstream_valid_structured_clean("mixed-tool", &ns, &mixed_frags);

            let st = capture_stream(port, harness.tools_body("t11-mixed-tool", true))
                .expect("mixed tool stream");
            assert!(st.saw_done);
            assert_eq!(st.finish.as_deref(), Some("tool_calls"));
            assert_eq!(st.content, "I'll look that up.");
            assert_eq!(st.tool_calls.len(), 1);
            assert_eq!(st.tool_calls[0].0, 0);
            assert_eq!(st.tool_calls[0].1.as_deref(), Some("call_0"));
            assert_eq!(st.tool_calls[0].2.as_deref(), Some("read_file"));
            assert_eq!(
                st.tool_calls[0].3.as_deref(),
                Some(r#"{"path":"mixed.rs"}"#)
            );
            assert_stream_valid_structured_clean("mixed-tool", &st, &mixed_frags);
        }

        // --- two calls: stable call_0/call_1 and stream indices 0/1 ---
        {
            let two_frags = [
                r#"{"path":"a.rs"}"#,
                r#""path":"a.rs""#,
                r#"{"path":"b.rs","data":"x"}"#,
                r#""path":"b.rs""#,
                r#""data":"x""#,
            ];
            let ns = complete_nonstream(port, harness.tools_body("t11-two-tools", false))
                .expect("two tools nonstream");
            assert_eq!(ns["choices"][0]["finish_reason"], "tool_calls");
            assert!(ns["choices"][0]["message"]["content"].is_null());
            let calls = ns["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool_calls array");
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0]["id"], "call_0");
            assert_eq!(calls[1]["id"], "call_1");
            assert_eq!(calls[0]["function"]["name"], "read_file");
            assert_eq!(calls[1]["function"]["name"], "write_file");
            assert_nonstream_valid_structured_clean("two-tools", &ns, &two_frags);

            let st = capture_stream(port, harness.tools_body("t11-two-tools", true))
                .expect("two tools stream");
            assert!(st.saw_done);
            assert_eq!(st.finish.as_deref(), Some("tool_calls"));
            assert!(st.content.is_empty());
            assert_eq!(st.tool_calls.len(), 2);
            assert_eq!(st.tool_calls[0].0, 0);
            assert_eq!(st.tool_calls[0].1.as_deref(), Some("call_0"));
            assert_eq!(st.tool_calls[1].0, 1);
            assert_eq!(st.tool_calls[1].1.as_deref(), Some("call_1"));
            assert_stream_valid_structured_clean("two-tools", &st, &two_frags);
        }

        // --- length withholds structured call buffered before terminal ---
        {
            // Even though calls are withheld, content must still be free of
            // protocol markers and of the buffered call's argument JSON.
            let length_frags = [r#"{"path":"x"}"#, r#""path":"x""#];
            let ns = complete_nonstream(port, harness.tools_body("t11-length-withhold", false))
                .expect("length nonstream");
            assert_eq!(ns["choices"][0]["finish_reason"], "length");
            assert!(
                ns["choices"][0]["message"].get("tool_calls").is_none(),
                "length must withhold tool_calls"
            );
            assert_eq!(ns["choices"][0]["message"]["content"], "partial-length");
            assert_nonstream_valid_structured_clean("length-withhold", &ns, &length_frags);

            let st = capture_stream(port, harness.tools_body("t11-length-withhold", true))
                .expect("length stream");
            assert!(st.saw_done);
            assert_eq!(st.finish.as_deref(), Some("length"));
            assert_eq!(st.content, "partial-length");
            assert!(
                st.tool_calls.is_empty(),
                "length stream must not release tool deltas"
            );
            assert_stream_valid_structured_clean("length-withhold", &st, &length_frags);
        }

        // --- include_usage: separate choices:[] chunk after terminal, before [DONE] ---
        {
            let mut body = harness.base_body("t11-usage", true);
            body["stream_options"] = serde_json::json!({ "include_usage": true });
            let st = capture_stream(port, body).expect("usage stream");
            assert!(st.saw_done, "successful stream ends [DONE]");
            assert_eq!(st.finish.as_deref(), Some("stop"));
            assert_eq!(st.content, "usage-path");
            let usage = st.usage.expect("include_usage must produce Usage event");
            assert_eq!(usage["prompt_tokens"], 11);
            assert_eq!(usage["completion_tokens"], 5);
            // Nonstream still has embedded usage object.
            let ns = complete_nonstream(port, harness.base_body("t11-usage", false))
                .expect("usage nonstream");
            assert_eq!(ns["usage"]["prompt_tokens"], 11);
            assert_eq!(ns["usage"]["completion_tokens"], 5);
        }
    }

    /// Invalid-producer diagnostic (authority violation): dirty marker token text
    /// stays byte-verbatim content and never becomes structured tool_calls.
    /// Kept separate so it cannot weaken valid structured-call leak assertions.
    #[cfg(unix)]
    #[test]
    fn task11_http_invalid_producer_dirty_marker_text_stays_verbatim() {
        let harness = Task11HttpHarness::spawn("dirty-markers");
        let port = harness.port();
        let dirty = concat!(
            r#"<tool_call>{"name":"evil","arguments":{}}</tool_call>"#,
            "<think>secret</think></think><|im_end|>"
        );

        let ns = complete_nonstream(port, harness.base_body("t11-dirty-markers", false))
            .expect("dirty nonstream");
        assert_eq!(ns["choices"][0]["finish_reason"], "stop");
        assert_eq!(ns["choices"][0]["message"]["content"], dirty);
        assert!(
            ns["choices"][0]["message"].get("tool_calls").is_none(),
            "dirty markers must not become structured tool_calls"
        );

        let st = capture_stream(port, harness.base_body("t11-dirty-markers", true))
            .expect("dirty stream");
        assert!(st.saw_done);
        assert_eq!(st.finish.as_deref(), Some("stop"));
        assert_eq!(st.content, dirty);
        assert!(st.tool_calls.is_empty());
        // Stream content deltas are also verbatim marker text (invalid producer).
        assert_eq!(st.content_deltas.concat(), dirty);
    }

    /// Premature daemon EOF after gen_start/token without done → client/HTTP failure.
    #[cfg(unix)]
    #[test]
    fn task11_http_premature_daemon_eof_is_failure_not_completion() {
        let harness = Task11HttpHarness::spawn("premature-eof");
        let port = harness.port();

        let ns_err = complete_nonstream(port, harness.base_body("t11-premature-eof", false))
            .expect_err("premature EOF must not succeed nonstream");
        let ns_msg = ns_err.to_string();
        assert!(
            !ns_msg.contains("\"finish_reason\""),
            "must not look like a completion payload: {ns_msg}"
        );

        let st_err = capture_stream(port, harness.base_body("t11-premature-eof", true))
            .expect_err("premature EOF must not succeed stream");
        // Stream path may surface PrematureEof (body cut mid-SSE) or Http/server_error.
        let st_msg = st_err.to_string();
        assert!(
            matches!(
                st_err,
                hipfire_client::ClientError::PrematureEof(_)
                    | hipfire_client::ClientError::Http(_)
                    | hipfire_client::ClientError::Closed { .. }
            ) || st_msg.contains("closed")
                || st_msg.contains("EOF")
                || st_msg.contains("error")
                || st_msg.contains("HTTP"),
            "unexpected stream error shape: {st_err:?}"
        );
    }

    /// Capability denial: daemon typed error on tools request → no completion/tool payload.
    #[cfg(unix)]
    #[test]
    fn task11_http_capability_denial_returns_error_without_tool_payload() {
        let harness = Task11HttpHarness::spawn("capability-denial");
        let port = harness.port();

        let ns_err = complete_nonstream(port, harness.tools_body("t11-capability-denial", false))
            .expect_err("capability denial must fail nonstream");
        let ns_msg = ns_err.to_string().to_ascii_lowercase();
        assert!(
            ns_msg.contains("not supported")
                || ns_msg.contains("unsupported")
                || ns_msg.contains("capability")
                || ns_msg.contains("http"),
            "expected capability/typed error, got: {ns_err}"
        );
        assert!(
            !ns_msg.contains("call_0") && !ns_msg.contains("tool_calls"),
            "must not return tool payload on denial: {ns_err}"
        );

        let st_err = capture_stream(port, harness.tools_body("t11-capability-denial", true))
            .expect_err("capability denial must fail stream");
        let st_msg = st_err.to_string().to_ascii_lowercase();
        assert!(
            !st_msg.contains("call_0"),
            "stream denial must not expose tool ids: {st_err}"
        );
    }

    // --- Task 15: server-owned one-retry (disabled-by-default) ---

    #[cfg(unix)]
    #[test]
    fn task15_http_transient_once_retries_and_succeeds() {
        let harness = Task11HttpHarness::spawn_with_retry("t15-once", Duration::from_millis(5));
        let port = harness.port();
        let body = harness.base_body("t15-transient-once", false);
        let completion = complete_nonstream(port, body).expect("retry must recover");
        let content = completion
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            content.contains("retry-recovered-content"),
            "unexpected content: {completion}"
        );
        let wire = completion.to_string();
        for banned in [
            "retries_attempted",
            "retry_enabled",
            "attempt_id",
            "retry_reset",
            "retries_succeeded",
        ] {
            assert!(
                !wire.contains(banned),
                "OpenAI wire must not expose {banned}: {wire}"
            );
        }
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (1, 1));

        let log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&log, "generate");
        let resets = Task11HttpHarness::ops_of_type(&log, "reset");
        assert_eq!(generates.len(), 2, "exactly two generates: {log:?}");
        assert_eq!(resets.len(), 1, "one force-reset between attempts: {log:?}");
        let a0 = generates[0]
            .get("attempt_id")
            .and_then(|v| v.as_u64())
            .expect("attempt 0");
        let a1 = generates[1]
            .get("attempt_id")
            .and_then(|v| v.as_u64())
            .expect("attempt 1");
        assert_ne!(a0, a1, "distinct attempt ids");
        assert!(a1 > a0, "monotonic attempt ids");
        let r_aid = resets[0]
            .get("attempt_id")
            .and_then(|v| v.as_u64())
            .expect("reset attempt");
        assert_eq!(r_aid, a1, "force-reset uses attempt-2 id");
        assert!(completion.get("id").and_then(|v| v.as_str()).is_some());
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_default_off_does_not_retry() {
        let harness = Task11HttpHarness::spawn("t15-gate-off");
        let port = harness.port();
        let err = complete_nonstream(port, harness.base_body("t15-transient-once", false))
            .expect_err("gate off must surface first failure");
        let _ = err;
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (0, 0));
        let log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&log, "generate");
        assert_eq!(
            generates.len(),
            1,
            "exactly one generate when disabled: {log:?}"
        );
        let resets = Task11HttpHarness::ops_of_type(&log, "reset");
        assert!(
            resets.is_empty(),
            "no force-reset when retry disabled: {log:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_visible_token_denies_retry() {
        let harness = Task11HttpHarness::spawn_with_retry("t15-vis", Duration::from_millis(5));
        let port = harness.port();
        let err = complete_nonstream(port, harness.base_body("t15-visible-token", false))
            .expect_err("visible token must deny retry");
        let _ = err;
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (0, 0));
        let log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&log, "generate");
        assert_eq!(generates.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_one_retry_max_then_fail() {
        let harness = Task11HttpHarness::spawn_with_retry("t15-always", Duration::from_millis(5));
        let port = harness.port();
        let err = complete_nonstream(port, harness.base_body("t15-transient-always", false))
            .expect_err("persistent transient must fail after one retry");
        let _ = err;
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (1, 0));
        let log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&log, "generate");
        assert_eq!(generates.len(), 2, "one retry only: {log:?}");
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_reset_failure_poisons_and_surfaces() {
        let harness =
            Task11HttpHarness::spawn_with_retry("t15-reset-fail-rolled", Duration::from_millis(5));
        let port = harness.port();
        let err = complete_nonstream(port, harness.base_body("t15-reset-fail-rolled", false))
            .expect_err("failed force-reset must surface");
        let msg = err.to_string().to_ascii_lowercase();
        assert!(
            msg.contains("reset")
                || msg.contains("roll")
                || msg.contains("daemon")
                || msg.contains("http")
                || msg.contains("error"),
            "expected reset-context error, got: {err}"
        );
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!(attempted, 1);
        assert_eq!(succeeded, 0);

        let ok = complete_nonstream(port, harness.base_body("t11-stop-text", false))
            .expect("post-poison request must reload and succeed");
        let _ = ok;
        let log = harness.read_requests_log();
        let loads = Task11HttpHarness::ops_of_type(&log, "load");
        assert!(
            loads.len() >= 2,
            "poison must force a second load: loads={loads:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_fold_clears_stale_tool_calls_on_retry() {
        let harness = Task11HttpHarness::spawn_with_retry("t15-fold", Duration::from_millis(5));
        let port = harness.port();
        let completion =
            complete_nonstream(port, harness.tools_body("t15-tool-then-transient", false))
                .expect("retry after buffered tools must succeed");
        let content = completion
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            content.contains("fold-cleared-content"),
            "unexpected content: {completion}"
        );
        let msg = completion
            .pointer("/choices/0/message")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        assert!(
            msg.get("tool_calls").is_none()
                || msg
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(|a| a.is_empty())
                    .unwrap_or(false),
            "stale attempt-1 tool_calls must not survive fold clear: {msg}"
        );
        assert!(!completion.to_string().contains("stale.rs"));
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (1, 1));
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_backoff_releases_admission_and_runtime_locks() {
        let harness = Task11HttpHarness::spawn_with_retry("t15-backoff", Duration::from_millis(50));
        let shared = Arc::clone(&harness.shared);
        let saw_free = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&saw_free);
        harness.set_backoff_hook(move |_dur| {
            let inflight = shared.admission.inflight();
            let runtime_free = shared.runtime.try_lock().is_ok();
            if inflight == 0 && runtime_free {
                flag.store(true, Ordering::SeqCst);
            }
            thread::sleep(Duration::from_millis(10));
        });
        let port = harness.port();
        let _ = complete_nonstream(port, harness.base_body("t15-transient-once", false))
            .expect("backoff path should still succeed");
        assert!(
            saw_free.load(Ordering::SeqCst),
            "admission and runtime must be free during retry backoff"
        );
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (1, 1));
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_ineligible_attestation_denies_retry() {
        let harness =
            Task11HttpHarness::spawn_with_retry("t15-ineligible-model", Duration::from_millis(5));
        let port = harness.port();
        let err = complete_nonstream(port, harness.base_body("t15-transient-once", false))
            .expect_err("ineligible attestation must deny retry");
        let _ = err;
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (0, 0));
        let log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&log, "generate");
        assert_eq!(generates.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn task15_http_commit_ready_denies_retry() {
        let harness =
            Task11HttpHarness::spawn_with_retry("t15-commit-deny", Duration::from_millis(5));
        let port = harness.port();
        // Terminal is staged at commit_ready; post-handshake daemon error does not
        // unwind the already-committed success, and must not open a retry.
        let completion =
            complete_nonstream(port, harness.base_body("t15-commit-ready-error", false))
                .expect("staged commit_ready success must surface without retry");
        let _ = completion;
        let (attempted, succeeded) = harness.meta_retries();
        assert_eq!((attempted, succeeded), (0, 0));
        let log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&log, "generate");
        assert_eq!(generates.len(), 1);
    }

    #[test]
    fn bench_generate_request_includes_numeric_first_attempt() {
        let req = bench_generate_request("bench prompt", 37);
        assert_eq!(req.get("type").and_then(|v| v.as_str()), Some("generate"));
        assert_eq!(req.get("attempt_id").and_then(|v| v.as_u64()), Some(1));
        let id = req.get("id").and_then(|v| v.as_str()).unwrap_or("");
        assert!(!id.is_empty(), "id must be a non-empty string");
        assert_eq!(
            req.get("prompt").and_then(|v| v.as_str()),
            Some("bench prompt")
        );
        assert_eq!(req.get("max_tokens").and_then(|v| v.as_u64()), Some(37));
    }

    /// Every benchmark generate must ask for answer mode.
    ///
    /// A reasoning model opens `<think>` in its first tokens and cannot close
    /// it inside a benchmark's fixed budget (16 tokens for the warmup, 128 for
    /// the measured runs). The daemon classifies an unclosed think span at
    /// finish as a non-retryable validation terminal *ahead of* the length cap
    /// — `QwenArTerminalCause::resolve` and `qwen_dflash_wire_terminal` in the
    /// daemon both order it that way — so a thinking benchmark aborts on the
    /// warmup, before it records a single sample.
    #[test]
    fn bench_generate_request_is_answer_mode_by_default() {
        let req = bench_generate_request("bench prompt", 128);
        assert_eq!(
            req.get("max_think_tokens").and_then(|v| v.as_u64()),
            Some(1),
            "benchmark generates must cap thinking"
        );
        assert_eq!(
            req.get("assistant_prefix").and_then(|v| v.as_str()),
            Some("closed_think"),
            "benchmark generates must start in answer mode"
        );
        assert_eq!(
            req.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("none")
        );
    }

    /// `--reasoning-on` restores the thinking turn for anyone who wants to
    /// benchmark that path, and must produce a request carrying none of the
    /// answer-mode fields.
    #[test]
    fn bench_reasoning_on_opts_back_into_thinking() {
        let req = bench_generate_request_reasoning("bench prompt", 128, true);
        assert!(req.get("max_think_tokens").is_none());
        assert!(req.get("assistant_prefix").is_none());
        assert!(req.get("reasoning_effort").is_none());
        // Still an ordinary benchmark generate otherwise.
        assert_eq!(req.get("max_tokens").and_then(|v| v.as_u64()), Some(128));
    }

    #[test]
    fn qwen_mq4r_decode_prewarm_is_fail_closed_to_the_exact_route() {
        let qwen = serde_json::json!({ "arch": "qwen3_5_moe" });
        let qwen_dense = serde_json::json!({ "arch": "qwen3_5" });
        let deepseek = serde_json::json!({ "arch": "deepseek4" });

        assert!(should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4r"),
            &qwen,
            None,
        ));
        assert!(should_prewarm_qwen_mq4r_decode(
            Path::new("QWEN3.6-9B.MQ4R"),
            &qwen_dense,
            Some(1),
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4"),
            &qwen,
            None,
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("deepseek-v4-flash.mq4r"),
            &deepseek,
            None,
        ));
        assert!(!should_prewarm_qwen_mq4r_decode(
            Path::new("qwen3.6-35b-a3b.mq4r"),
            &qwen,
            Some(2),
        ));
    }
}
