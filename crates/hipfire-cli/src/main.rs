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
    OpenAiSseEvent,
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
use hipfire_runtime::prompt_frame::ToolCall;
use saddle_core::caps::ReasoningContract;
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
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

mod bench_concurrency;
mod serve;
mod setup;
use crate::serve::complete::next_attempt_id;
use crate::serve::http::request_id;
use crate::serve::{detach_serve, parse_host_port, parse_pid_record, ServePidRecord};
use setup::setup_command;

pub(crate) const MODEL_SUFFIXES: &[&str] = &[
    ".hf4",
    ".hf6",
    ".hfq",
    ".mq2",
    ".mq2lloyd",
    // qt51 `MQ2G256LloydU` — the UNROTATED sibling of `.mq2lloyd` (qt19, which
    // is FWHT-rotated). Deliberately NOT folded into `.mq2lloyd`: loading a
    // rotated container as unrotated (or the reverse) does not fail loudly, it
    // produces silent garbage, so the two must stay nameable apart.
    ".mq2lloydu",
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
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
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
    /// Explicit device compiler (hipcc/amdclang++) when it lives in a different
    /// prefix than the runtime. Also set via HIPFIRE_HIPCC.
    #[arg(long, value_name = "PATH")]
    hipcc: Option<PathBuf>,
    /// Disable cross-root compiler fallback; require the compiler under the
    /// selected root. Also set via HIPFIRE_ROCM_STRICT=1.
    #[arg(long)]
    strict_rocm: bool,
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
pub(crate) struct PullArgs {
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

#[derive(Args, Debug, Clone)]
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
    #[arg(long)]
    /// Select a published lm_head variant (see the registry's `heads`), e.g.
    /// `--head q4k`. The overlay shadows the model's own head at load time;
    /// omitting this uses the head baked into the model file.
    head: Option<String>,
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
pub(crate) struct BenchArgs {
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
    /// Which backend to drive: slots (multi-slot engine), noslots (sequential
    /// daemon baseline), batch (beta continuous batching), or both.
    #[arg(long, value_parser = ["slots", "noslots", "batch", "both"], default_value = "both")]
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
pub(crate) struct ServeArgs {
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
    /// Tensor/expert-parallel degree.
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
pub(crate) struct StopArgs {
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
pub(crate) struct Paths {
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
        Some(Commands::Serve(args)) => crate::serve::serve_command(&paths, args),
        Some(Commands::Stop(args)) => crate::serve::stop_command(&paths, args),
        Some(Commands::Restart(args)) => {
            let port = args.positionals.iter().find_map(|value| {
                value.parse::<u16>().ok().or_else(|| {
                    crate::serve::parse_host_port(value)
                        .ok()
                        .flatten()
                        .map(|(_, port)| port)
                })
            });
            let _ = crate::serve::stop_command(
                &paths,
                StopArgs {
                    port,
                    force: true,
                    all: false,
                },
            );
            crate::serve::serve_command(&paths, args)
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

pub(crate) fn resolved_global(
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

pub(crate) fn list_local_models(paths: &Paths, registry: &RegistryV1) -> Result<Vec<LocalModel>> {
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

pub(crate) fn local_model_paths(paths: &Paths) -> Result<Vec<PathBuf>> {
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

pub(crate) fn pull_command(paths: &Paths, args: PullArgs) -> Result<()> {
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
    let needs_base = if args.force {
        true
    } else if destination.exists() {
        if existing_artifact_valid(&destination, entry.sha256.as_deref(), entry.size_bytes) {
            eprintln!("Already downloaded: {}", destination.display());
            false
        } else {
            eprintln!("Refreshing stale artifact: {}", destination.display());
            true
        }
    } else {
        true
    };
    if needs_base {
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
        if destination.exists() && !args.force {
            if existing_artifact_valid(&destination, sidecar.sha256.as_deref(), sidecar.size_bytes)
            {
                eprintln!("  {label} sidecar already present: {}", sidecar.file);
                continue;
            }
            eprintln!("  {label} sidecar stale, refreshing: {}", sidecar.file);
        } else if destination.exists() && args.force {
            // force always refreshes
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
    for (name, sidecar) in &entry.heads {
        let destination = paths.models.join(&sidecar.file);
        if destination.exists() && !args.force {
            if existing_artifact_valid(&destination, sidecar.sha256.as_deref(), sidecar.size_bytes)
            {
                eprintln!("  head {name} already present: {}", sidecar.file);
                continue;
            }
            eprintln!("  head {name} stale, refreshing: {}", sidecar.file);
        }
        eprintln!("  Fetching head {name}: {}", sidecar.file);
        download_verified(
            &artifact_url(entry, &sidecar.file),
            &destination,
            sidecar.sha256.as_deref(),
            sidecar.size_bytes,
            true,
        )?;
    }
    println!("{}", paths.models.join(&entry.file).display());
    Ok(())
}

pub(crate) fn artifact_url(entry: &ModelEntry, file: &str) -> String {
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

pub(crate) fn download_verified(
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

pub(crate) fn existing_artifact_valid(
    path: &Path,
    expected_sha256: Option<&str>,
    expected_size: Option<u64>,
) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if let Some(expected) = expected_size {
        if metadata.len() != expected {
            return false;
        }
    }
    if let Some(expected) = expected_sha256 {
        let Ok(digest) = sha256_path(path) else {
            return false;
        };
        if !digest.eq_ignore_ascii_case(expected) {
            return false;
        }
    }
    true
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
        targets.extend(
            entry
                .heads
                .values()
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
    let force_local = run_should_force_local(&args);
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
        anyhow!("daemon binary not found; build `cargo build --release -p hipfire-daemon`")
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
        args.head.as_deref(),
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
    let contract = loaded
        .get("reasoning_contract")
        .and_then(serde_json::Value::as_str)
        .and_then(ReasoningContract::from_wire_name)
        .unwrap_or(ReasoningContract::Unsupported);
    let effort_native = loaded
        .get("reasoning_effort_native")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let supported_efforts = loaded
        .get("reasoning_efforts")
        .and_then(serde_json::Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(|value| value.as_str().map(|string| string.to_owned()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let _ = apply_http_reasoning_request(
        &serde_json::json!({}),
        &resolved,
        &mut request,
        contract,
        effort_native,
        &supported_efforts,
    )?;

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

pub(crate) fn run_should_force_local(args: &RunArgs) -> bool {
    process_truthy("HIPFIRE_LOCAL")
        || args.image.is_some()
        || args.kv_mode.is_some()
        || args.kv_backend.is_some()
        || args.head.is_some()
        || args.speculation.is_some()
        || args.model_draft.is_some()
        || args.draft_max.is_some()
        || args.dspark_conf_threshold.is_some()
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
        let mut assistant_reasoning = String::new();
        let mut assistant_content = String::new();
        let result = stream_openai_chat(
            client_host,
            port,
            body,
            Duration::from_secs(60 * 60),
            |event| {
                match event {
                    OpenAiSseEvent::Reasoning { text } => {
                        assistant_reasoning.push_str(&text);
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                    OpenAiSseEvent::Content { text } => {
                        assistant_content.push_str(&text);
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
        let mut assistant_msg =
            serde_json::json!({ "role": "assistant", "content": assistant_content });
        if !assistant_reasoning.is_empty() {
            assistant_msg["reasoning_content"] = serde_json::Value::String(assistant_reasoning);
        }
        messages.push(assistant_msg);
    }
    let _ = args.no_color;
    Ok(())
}

pub(crate) fn resolved_for_model(
    paths: &Paths,
    model_name: &str,
    tag: Option<&str>,
    entry: Option<&ModelEntry>,
) -> Result<hipfire_config::ResolvedConfig> {
    let loaded = load_global(&paths.config)?;
    let mut layers = Vec::new();
    if let (Some(tag), Some(entry)) = (tag, entry) {
        layers.push(NamedLayer {
            source: ConfigSource::RegistryModel {
                tag: tag.to_owned(),
                revision: "v1".into(),
            },
            layer: hipfire_registry::config_layer_for_tag(tag, entry)
                .map_err(|error| anyhow!("invalid registry model defaults: {error}"))?,
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

/// How [`scan_local_models`] compares an input against on-disk filenames.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchMode {
    /// Compare the lowercased strings as written.
    Literal,
    /// Additionally strip `-`, `.` and `_` from both sides, so spellings that
    /// differ only in separators compare equal (`ornith-1.5` ↔ `ornith1.5`).
    IgnoreSeparators,
}

fn scan_local_models(local: &[PathBuf], search: &str, mode: MatchMode) -> Vec<PathBuf> {
    let normalize = |value: &str| -> String {
        let lower = value.to_ascii_lowercase();
        match mode {
            MatchMode::Literal => lower,
            MatchMode::IgnoreSeparators => lower
                .chars()
                .filter(|c| !matches!(c, '-' | '.' | '_'))
                .collect(),
        }
    };
    let needle = normalize(search);
    // A needle that normalizes to nothing (an input of only separators) would
    // make `contains` true for every file and hand back an arbitrary model.
    if needle.is_empty() {
        return Vec::new();
    }
    local
        .iter()
        .filter(|path| {
            let name = normalize(
                path.file_name()
                    .and_then(|file| file.to_str())
                    .unwrap_or_default(),
            );
            name == needle || name.contains(&needle)
        })
        .cloned()
        .collect()
}
pub(crate) fn find_model_path(
    paths: &Paths,
    registry: &RegistryV1,
    model: &str,
) -> Option<PathBuf> {
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
    // An exact on-disk spelling outranks a registry alias that would rewrite it.
    // The registry maps a model NAME to its canonical FILE, which is what makes
    // `qwen3.8:27b` resolve to `qwen3.8-27b.mq4`. But when a legacy alias points at
    // a RENAMED artifact and both spellings are present, that rewrite silently
    // serves weights the user did not name -- `ornith1.5:35b-a3b` reaching
    // `ornith-1.5-35b-a3b.mq4` while `ornith1.5-35b-a3b.mq4` sits right there.
    //
    // The comparison is on the exact stem, deliberately NOT the looser `contains`
    // used by scan_local_models: `qwen3.8:27b` would `contains`-match
    // `qwen3.8-27b.mq4r` and `qwen3.8-27b.mq4-xt` as well, and the quant sort below
    // ranks `.mq4r` ABOVE `.mq4`, so a looser rule here would silently move users
    // onto a different tier. An exact stem can only ever match the one file the
    // user actually spelled.
    let exact_stem = model.replace(':', "-").to_ascii_lowercase();
    if let Ok(local) = local_model_paths(paths) {
        if let Some(hit) = local.iter().find(|path| {
            let name = path
                .file_name()
                .and_then(|file| file.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            MODEL_SUFFIXES
                .iter()
                .any(|suffix| name == format!("{exact_stem}{suffix}"))
        }) {
            return Some(hit.clone());
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
    let local = local_model_paths(paths).ok()?;
    // Two passes. The first matches the literal spelling and is what has always
    // run. The second retries with `-`, `.` and `_` stripped from both sides, so
    // an input and an on-disk file that differ only in separators still meet:
    // `ornith-1.5:35b-a3b` finds the `ornith1.5-35b-a3b.mq4` left on disk by
    // anyone who downloaded before the artifacts were renamed.
    //
    // The fallback is strictly second — it runs only when the literal pass found
    // nothing, so it can add a match where today there is none but can never
    // change one that already resolves. That ordering is the safety argument:
    // separator-stripping is a looser `contains`, and running it first would let
    // it outrank an exact hit.
    let mut candidates = scan_local_models(&local, &search, MatchMode::Literal);
    if candidates.is_empty() {
        candidates = scan_local_models(&local, &search, MatchMode::IgnoreSeparators);
    }
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

pub(crate) fn load_params(
    resolved: &hipfire_config::ResolvedConfig,
    entry: Option<&ModelEntry>,
    model_path: &Path,
    max_tokens: u64,
    kv_override: Option<&str>,
    kv_backend_override: Option<&str>,
    head_override: Option<&str>,
) -> Result<serde_json::Value> {
    let configured_max_seq = config_u64(resolved, "memory.max_seq")?;
    let max_seq = configured_max_seq.max(max_tokens.saturating_add(1024));
    let configured_kv = config_string(resolved, "memory.kv_cache")?;
    let kv_mode = kv_override
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
        .unwrap_or(configured_kv);
    // Validate through the shared schema. `auto` is preserved so architecture
    // (maple vs qwen) can select BF16 vs Q8; do not substitute q8 here.
    field("memory.kv_cache")
        .expect("schema field")
        .parse_cli(&kv_mode)?;
    let configured_backend = config_string(resolved, "memory.kv_backend")?;
    let kv_backend = kv_backend_override
        .map(str::to_owned)
        .filter(|value| !value.is_empty())
        .unwrap_or(configured_backend)
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
    // Resolve --head <name> against the registry's `heads` map. The overlay
    // lives beside the model file, exactly like the triattn sidecar. Refuse
    // rather than fall back: a silent fall-back would serve the base's head
    // and answer a different question than the operator asked.
    let head_file = match head_override.filter(|s| !s.is_empty()) {
        None => String::new(),
        // A direct path is accepted as well as a registry name: loading a
        // model BY PATH has no registry entry, so names cannot resolve there
        // and only a path can work.
        Some(name) if Path::new(name).is_file() => name.to_string(),
        Some(name) => {
            let heads = entry.map(|e| &e.heads);
            let sidecar = heads.and_then(|h| h.get(name)).ok_or_else(|| {
                let known: Vec<&str> = heads
                    .map(|h| h.keys().map(String::as_str).collect())
                    .unwrap_or_default();
                anyhow!(
                    "--head {name}: not a file, and this model has no such head variant{}",
                    if known.is_empty() {
                        " (it publishes none)".to_string()
                    } else {
                        format!(" (available: {})", known.join(", "))
                    }
                )
            })?;
            let candidate = model_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&sidecar.file);
            if !candidate.is_file() {
                bail!(
                    "--head {name}: overlay {} not found — fetch it with \
                     `hipfire pull` or place it beside the model",
                    candidate.display()
                );
            }
            candidate.display().to_string()
        }
    };
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
        "head": head_file,
        "cask": config_bool(resolved, "memory.cask.enabled")?,
        "cask_budget": config_u64(resolved, "memory.cask.budget")?,
        "cask_beta": config_u64(resolved, "memory.cask.beta")?,
        "cask_handoff_tokens": config_u64(resolved, "memory.cask.handoff_tokens")?,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReasoningResolution {
    pub effective_mode: String,
    pub effective_effort: Option<String>,
    pub effective_cap: Option<u64>,
    pub cap_source: String,
    pub contract: ReasoningContract,
    pub warnings: Vec<String>,
}

pub(crate) fn apply_http_reasoning_request(
    body: &serde_json::Value,
    resolved: &hipfire_config::ResolvedConfig,
    request: &mut serde_json::Value,
    contract: ReasoningContract,
    effort_native: bool,
    supported_efforts: &[String],
) -> Result<ReasoningResolution> {
    let mut warnings: Vec<String> = Vec::new();
    let mut push_warn = |msg: String| {
        eprintln!("[WARN: INVALID CONFIG] {}", msg);
        warnings.push(msg);
    };
    if let Some(object) = request.as_object_mut() {
        for key in [
            "thinking_enabled",
            "assistant_prefix",
            "reasoning_effort",
            "max_think_tokens",
        ] {
            object.remove(key);
        }
    }
    if body.get("enable_thinking").is_some() {
        if let Some(value) = body.get("enable_thinking") {
            if !value.is_boolean() && !value.is_null() {
                bail!("enable_thinking must be a boolean");
            }
        }
    }
    let top_enable = body
        .get("enable_thinking")
        .and_then(serde_json::Value::as_bool);
    if body
        .pointer("/chat_template_kwargs/enable_thinking")
        .is_some()
    {
        if let Some(value) = body.pointer("/chat_template_kwargs/enable_thinking") {
            if !value.is_boolean() && !value.is_null() {
                bail!("chat_template_kwargs.enable_thinking must be a boolean");
            }
        }
    }
    let kwargs_enable = body
        .pointer("/chat_template_kwargs/enable_thinking")
        .and_then(serde_json::Value::as_bool);
    let thinking_type_raw = body
        .pointer("/thinking/type")
        .or_else(|| body.get("thinking").and_then(|value| value.get("type")));
    let mut thinking_type_str: Option<&str> = None;
    if let Some(raw) = thinking_type_raw {
        if !raw.is_string() {
            bail!("thinking.type must be enabled or disabled");
        } else {
            let s = raw.as_str().unwrap();
            if s == "enabled" || s == "disabled" {
                thinking_type_str = Some(s);
            } else {
                push_warn(format!(
                    "thinking.type '{}' dropped: expected enabled|disabled",
                    s
                ));
                thinking_type_str = None;
            }
        }
    }
    let effort_raw = body
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            body.pointer("/reasoning/effort")
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            body.pointer("/chat_template_kwargs/reasoning_effort")
                .and_then(serde_json::Value::as_str)
        });
    let effort_present = body.get("reasoning_effort").is_some()
        || body.pointer("/reasoning/effort").is_some()
        || body
            .pointer("/chat_template_kwargs/reasoning_effort")
            .is_some();
    if effort_present && effort_raw.is_none() {
        bail!("reasoning_effort must be a string");
    }
    let body_budget_present = body.get("thinking_budget").is_some();
    let body_budget_str = body
        .get("thinking_budget")
        .and_then(serde_json::Value::as_str);
    if body_budget_present && body_budget_str.is_none() {
        bail!("thinking_budget must be a string preset");
    }
    let body_top_max_present = body.get("max_think_tokens").is_some();
    let body_nested_max_present = body.pointer("/reasoning/max_tokens").is_some();
    let parse_body_think_cap = |value: &serde_json::Value, field: &str| -> Result<u64> {
        match value {
            serde_json::Value::Number(number) => {
                if let Some(parsed) = number.as_u64() {
                    if parsed > 393_216 {
                        bail!("{field} must be between 0 and 393216");
                    }
                    Ok(parsed)
                } else {
                    bail!("{field} must be between 0 and 393216");
                }
            }
            _ => bail!("{field} must be between 0 and 393216"),
        }
    };
    let top_max_opt = if body_top_max_present {
        Some(parse_body_think_cap(
            body.get("max_think_tokens").unwrap(),
            "max_think_tokens",
        )?)
    } else {
        None
    };
    let nested_max_opt = if body_nested_max_present {
        Some(parse_body_think_cap(
            body.pointer("/reasoning/max_tokens").unwrap(),
            "reasoning.max_tokens",
        )?)
    } else {
        None
    };
    let (max_opt, body_max_source) = match (top_max_opt, nested_max_opt) {
        (Some(top), Some(nested)) => {
            if top != nested {
                push_warn(
                    "reasoning.max_tokens dropped because explicit max_think_tokens takes precedence"
                        .to_string(),
                );
            }
            (Some(top), "explicit:body:max_think_tokens")
        }
        (Some(top), None) => (Some(top), "explicit:body:max_think_tokens"),
        (None, Some(nested)) => (Some(nested), "explicit:body:reasoning.max_tokens"),
        (None, None) => (None, ""),
    };
    let has_explicit_body_max = body_top_max_present || body_nested_max_present;
    let config_max_entry = resolved.get("reasoning.max_tokens").filter(|value| {
        !matches!(value.source, hipfire_config::ConfigSource::BuiltIn)
            && !matches!(value.value, hipfire_config::ConfigValue::Null)
    });
    let mut config_max_opt: Option<u64> = None;
    if let Some(entry) = config_max_entry {
        match entry.value {
            hipfire_config::ConfigValue::Integer(value) if value >= 0 => {
                config_max_opt = Some(value as u64);
            }
            _ => bail!("reasoning.max_tokens resolved to a non-negative integer"),
        }
    }
    let has_explicit_config_max = config_max_opt.is_some();
    let config_budget_entry = resolved
        .get("reasoning.budget")
        .filter(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let has_explicit_config_budget = config_budget_entry.is_some();
    let config_budget_str: Option<String> = if has_explicit_config_budget {
        Some(config_string(resolved, "reasoning.budget")?)
    } else {
        None
    };
    let has_explicit_effort = effort_raw.is_some();
    let has_explicit_toggle =
        top_enable.is_some() || kwargs_enable.is_some() || thinking_type_str.is_some();
    let is_effort_native = match contract {
        ReasoningContract::QwenJinja => effort_native,
        ReasoningContract::DeepSeek4 => true,
        ReasoningContract::MuseGlimmer => true,
        _ => false,
    };
    if matches!(contract, ReasoningContract::Unsupported) {
        if has_explicit_toggle {
            push_warn(
                "reasoning controls dropped for unsupported contract: thinking toggle ignored"
                    .to_string(),
            );
        }
        if has_explicit_effort {
            push_warn(format!(
                "reasoning_effort '{}' dropped for unsupported contract",
                effort_raw.unwrap_or("unknown")
            ));
        }
        if has_explicit_body_max
            || body_budget_present
            || has_explicit_config_max
            || has_explicit_config_budget
        {
            push_warn("max_think_tokens/budget dropped for unsupported contract".to_string());
        }
        let resolution = ReasoningResolution {
            effective_mode: "disabled".to_string(),
            effective_effort: None,
            effective_cap: None,
            cap_source: "none".to_string(),
            contract,
            warnings: warnings.clone(),
        };
        return Ok(resolution);
    }
    if matches!(contract, ReasoningContract::GemmaBoolean) {
        if has_explicit_effort {
            push_warn(format!(
                "reasoning_effort '{}' dropped for gemma_boolean: use thinking toggle only",
                effort_raw.unwrap()
            ));
        }
        if body_budget_present {
            push_warn(format!(
                "thinking_budget '{}' dropped for gemma_boolean",
                body_budget_str.unwrap()
            ));
        }
        if has_explicit_body_max {
            push_warn(format!(
                "{} {} dropped for gemma_boolean: use thinking toggle only",
                if body_top_max_present {
                    "max_think_tokens"
                } else {
                    "reasoning.max_tokens"
                },
                max_opt.unwrap()
            ));
        }
        if has_explicit_config_max {
            push_warn(format!(
                "reasoning.max_tokens {} dropped for gemma_boolean: use thinking toggle only",
                config_max_opt.unwrap()
            ));
        }
        if has_explicit_config_budget {
            push_warn(format!(
                "thinking_budget '{}' dropped for gemma_boolean",
                config_budget_str.clone().unwrap()
            ));
        }
    }
    let thinking_type_toggle = thinking_type_str.map(|value| value == "enabled");
    let toggle_values = [top_enable, kwargs_enable, thinking_type_toggle];
    let saw_enabled = toggle_values.iter().flatten().any(|value| *value);
    let saw_disabled = toggle_values.iter().flatten().any(|value| !*value);
    if saw_enabled && saw_disabled {
        push_warn("conflicting thinking toggles normalized: disabled wins".to_string());
    }
    let mut toggle_opt = toggle_values
        .into_iter()
        .flatten()
        .reduce(|previous, value| previous && value);
    let is_off_effort = matches!(
        contract,
        ReasoningContract::QwenJinja | ReasoningContract::DeepSeek4
    ) && matches!(effort_raw, Some("none") | Some("off") | Some("chat"));
    if is_off_effort {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with off/none effort; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    let is_off_budget = !is_effort_native
        && !matches!(contract, ReasoningContract::GemmaBoolean)
        && body_budget_str == Some("off");
    if is_off_budget {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with legacy budget off; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    // `reasoning_budget_tokens == 0` is the wire spelling hag and other OpenAI
    // clients use for "do not think". Absent from master but genuine: without
    // it a small max_tokens budget is burned inside <think> and the client gets
    // empty content. Route through toggle reduction so it inherits contract
    // gating and disabled-wins warnings, like is_off_budget.
    let is_off_reasoning_budget_tokens = !is_effort_native
        && !matches!(contract, ReasoningContract::GemmaBoolean)
        && body
            .get("reasoning_budget_tokens")
            .and_then(serde_json::Value::as_u64)
            == Some(0);
    if is_off_reasoning_budget_tokens {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with reasoning_budget_tokens 0; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    // `max_think_tokens == 1` is the engine's own lowered encoding of off
    // (alongside assistant_prefix=closed_think). Incoming 1 must not be a
    // contract-blind short-circuit: a client legitimately requesting a 1-token
    // cap would be silently reinterpreted. Gate on contract and participate in
    // disabled-wins, so every off-path is contract-aware.
    let is_off_max_think_one = matches!(contract, ReasoningContract::QwenJinja)
        && body
            .get("max_think_tokens")
            .and_then(serde_json::Value::as_u64)
            == Some(1);
    if is_off_max_think_one {
        if toggle_opt == Some(true) {
            push_warn(
                "thinking enabled conflicts with max_think_tokens 1; thinking disabled wins"
                    .to_string(),
            );
        }
        toggle_opt = Some(false);
    }
    let config_mode = config_string(resolved, "reasoning.mode").unwrap_or_else(|_| "on".into());
    let config_mode_is_explicit = resolved
        .get("reasoning.mode")
        .is_some_and(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let config_effort =
        config_string(resolved, "reasoning.effort").unwrap_or_else(|_| "auto".into());
    let config_effort_is_explicit = resolved
        .get("reasoning.effort")
        .is_some_and(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let config_budget =
        config_string(resolved, "reasoning.budget").unwrap_or_else(|_| "uncapped".into());
    let config_budget_is_explicit = resolved
        .get("reasoning.budget")
        .is_some_and(|value| !matches!(value.source, hipfire_config::ConfigSource::BuiltIn));
    let configured_off = (config_mode_is_explicit && config_mode == "off")
        || (config_effort_is_explicit && config_effort == "none")
        || (!is_effort_native
            && !matches!(contract, ReasoningContract::GemmaBoolean)
            && config_budget_is_explicit
            && config_budget == "off");
    let family_default = !matches!(contract, ReasoningContract::GemmaBoolean);
    let mut thinking_enabled = toggle_opt.unwrap_or_else(|| {
        if configured_off {
            false
        } else if config_mode_is_explicit {
            config_mode != "off"
        } else {
            family_default
        }
    });
    if matches!(contract, ReasoningContract::MuseGlimmer) && !thinking_enabled {
        push_warn("reasoning off dropped for muse_glimmer: always-on reasoning".to_string());
        thinking_enabled = true;
    }
    request["thinking_enabled"] = serde_json::json!(thinking_enabled);
    let prefix = match contract {
        ReasoningContract::QwenJinja | ReasoningContract::DeepSeek4 => {
            if thinking_enabled {
                "open_think"
            } else {
                "closed_think"
            }
        }
        ReasoningContract::GemmaBoolean | ReasoningContract::MuseGlimmer => "plain",
        ReasoningContract::Unsupported => "plain",
    };
    request["assistant_prefix"] = serde_json::json!(prefix);
    if !thinking_enabled {
        if has_explicit_effort {
            push_warn(format!(
                "reasoning_effort '{}' dropped: thinking disabled",
                effort_raw.unwrap()
            ));
        }
        if config_effort_is_explicit && config_effort != "auto" && config_effort != "none" {
            push_warn(format!(
                "reasoning.effort '{}' dropped: thinking disabled",
                config_effort
            ));
        }
        let cap_present = has_explicit_body_max
            || body_budget_present
            || has_explicit_config_max
            || has_explicit_config_budget;
        if cap_present {
            push_warn("max_think_tokens/budget dropped: thinking disabled".to_string());
        }
        let resolution = ReasoningResolution {
            effective_mode: "disabled".to_string(),
            effective_effort: None,
            effective_cap: None,
            cap_source: "none".to_string(),
            contract,
            warnings: warnings.clone(),
        };
        return Ok(resolution);
    }
    if matches!(contract, ReasoningContract::QwenJinja) && is_effort_native && body_budget_present {
        push_warn(format!(
            "thinking_budget '{}' ignored for effort-native contract {}: use explicit max_think_tokens for cap",
            body_budget_str.unwrap(),
            contract.wire_name()
        ));
    }
    if matches!(contract, ReasoningContract::QwenJinja)
        && is_effort_native
        && has_explicit_config_budget
    {
        push_warn(format!(
            "thinking_budget '{}' ignored for effort-native contract {}: use explicit max_think_tokens for cap",
            config_budget_str.clone().unwrap(),
            contract.wire_name()
        ));
    }

    let (mut effective_cap, mut cap_source) = if matches!(contract, ReasoningContract::GemmaBoolean)
    {
        (None, "none".to_string())
    } else if matches!(
        contract,
        ReasoningContract::DeepSeek4 | ReasoningContract::MuseGlimmer
    ) && (has_explicit_body_max
        || has_explicit_config_max
        || body_budget_present
        || has_explicit_config_budget)
    {
        if has_explicit_body_max {
            push_warn(format!(
                "{} {} dropped for {}: use reasoning_effort only",
                if body_top_max_present {
                    "max_think_tokens"
                } else {
                    "reasoning.max_tokens"
                },
                max_opt.unwrap(),
                contract.wire_name()
            ));
        }
        if has_explicit_config_max {
            push_warn(format!(
                "reasoning.max_tokens {} dropped for {}: use reasoning_effort only",
                config_max_opt.unwrap(),
                contract.wire_name()
            ));
        }
        if body_budget_present {
            push_warn(format!(
                "thinking_budget '{}' dropped for {}: use reasoning_effort only",
                body_budget_str.unwrap(),
                contract.wire_name()
            ));
        }
        if has_explicit_config_budget {
            push_warn(format!(
                "thinking_budget '{}' dropped for {}: use reasoning_effort only",
                config_budget_str.clone().unwrap(),
                contract.wire_name()
            ));
        }
        (None, "none".to_string())
    } else if has_explicit_body_max {
        if body_budget_present {
            push_warn(
                "thinking_budget dropped because explicit max_think_tokens takes precedence"
                    .to_string(),
            );
        }
        (max_opt, body_max_source.to_string())
    } else if !is_effort_native && body_budget_present {
        let budget_str = body_budget_str.unwrap();
        let mapped = match budget_str {
            "off" => None,
            "low" => Some(512),
            "med" => Some(2048),
            "high" => Some(8192),
            "xhigh" => Some(24576),
            "max" => Some(32768),
            "uncapped" => Some(0),
            other => {
                push_warn(format!(
                    "thinking_budget '{}' dropped: unknown preset",
                    other
                ));
                None
            }
        };
        if mapped.is_none() && budget_str != "off" && {
            let known = ["low", "med", "high", "xhigh", "max", "uncapped"];
            !known.contains(&budget_str)
        } {
            (None, "none".to_string())
        } else {
            (mapped, "explicit:body:thinking_budget".to_string())
        }
    } else if has_explicit_config_max {
        (config_max_opt, "config:reasoning.max_tokens".to_string())
    } else if !is_effort_native && has_explicit_config_budget {
        let budget_str = config_budget_str.as_deref().unwrap();
        let mapped = match budget_str {
            "off" => None,
            "low" => Some(512),
            "med" => Some(2048),
            "high" => Some(8192),
            "xhigh" => Some(24576),
            "max" => Some(32768),
            "uncapped" => Some(0),
            other => {
                push_warn(format!(
                    "thinking_budget '{}' dropped: unknown preset",
                    other
                ));
                None
            }
        };
        if mapped.is_none() && budget_str != "off" && {
            let known = ["low", "med", "high", "xhigh", "max", "uncapped"];
            !known.contains(&budget_str)
        } {
            (None, "none".to_string())
        } else {
            (mapped, "config:reasoning.budget".to_string())
        }
    } else {
        (None, "none".to_string())
    };
    if effective_cap == Some(0) {
        effective_cap = None;
        cap_source = "none".to_string();
    }
    if let Some(value) = effective_cap {
        request["max_think_tokens"] = serde_json::json!(value);
    }
    let mut effective_effort: Option<String> = None;
    match contract {
        ReasoningContract::QwenJinja => {
            if !effort_native {
                if has_explicit_effort {
                    push_warn(format!(
                        "reasoning_effort '{}' dropped: template does not natively support effort (Qwen3.6); use thinking_budget or max_think_tokens for cap",
                        effort_raw.unwrap()
                    ));
                }
                if config_effort_is_explicit && config_effort != "auto" {
                    push_warn(format!(
                        "reasoning.effort '{}' dropped: template does not natively support effort",
                        config_effort
                    ));
                }
                effective_effort = None;
            } else if has_explicit_effort || config_effort_is_explicit {
                let raw = if let Some(value) = effort_raw {
                    value.to_owned()
                } else {
                    config_effort.clone()
                };
                if raw == "auto" {
                    effective_effort = Some("xhigh".to_string());
                    request["reasoning_effort"] = serde_json::json!("xhigh");
                } else {
                    match raw.as_str() {
                        "low" | "medium" | "xhigh" => {
                            if !supported_efforts.is_empty() && !supported_efforts.contains(&raw) {
                                push_warn(format!(
                                    "reasoning_effort '{}' dropped: not in supported {:?}",
                                    raw, supported_efforts
                                ));
                                effective_effort = Some("xhigh".to_string());
                                request["reasoning_effort"] = serde_json::json!("xhigh");
                            } else {
                                effective_effort = Some(raw.clone());
                                request["reasoning_effort"] = serde_json::json!(raw);
                            }
                        }
                        "high" | "max" | "minimal" | "med" => {
                            push_warn(format!(
                                "reasoning_effort '{}' dropped for qwen_jinja: expected low|medium|xhigh",
                                raw
                            ));
                            effective_effort = Some("xhigh".to_string());
                            request["reasoning_effort"] = serde_json::json!("xhigh");
                        }
                        other => {
                            push_warn(format!(
                                "reasoning_effort '{}' normalized to qwen_jinja default xhigh",
                                other
                            ));
                            effective_effort = Some("xhigh".to_string());
                            request["reasoning_effort"] = serde_json::json!("xhigh");
                        }
                    }
                }
            } else {
                let default = "xhigh".to_string();
                effective_effort = Some(default.clone());
                request["reasoning_effort"] = serde_json::json!(default);
            }
        }
        ReasoningContract::DeepSeek4 => {
            let raw = if let Some(value) = effort_raw {
                value.to_owned()
            } else {
                let cfg = config_string(resolved, "reasoning.effort")
                    .unwrap_or_else(|_| "auto".to_string());
                if cfg != "auto" {
                    cfg
                } else {
                    "high".to_string()
                }
            };
            let normalized = match raw.as_str() {
                "minimal" => "low",
                "low" => "low",
                "medium" | "med" => "high",
                "xhigh" => "high",
                "high" => "high",
                "max" => "max",
                other => {
                    push_warn(format!(
                        "reasoning_effort '{}' normalized to deepseek4 default high",
                        other
                    ));
                    "high"
                }
            };
            effective_effort = Some(normalized.to_string());
            request["reasoning_effort"] = serde_json::json!(normalized);
        }
        ReasoningContract::MuseGlimmer => {
            let raw = if let Some(value) = effort_raw {
                value.to_owned()
            } else {
                let cfg = config_string(resolved, "reasoning.effort")
                    .unwrap_or_else(|_| "auto".to_string());
                if cfg != "auto" {
                    match cfg.as_str() {
                        "low" | "medium" | "high" | "xhigh" | "max" => cfg,
                        other => {
                            push_warn(format!(
                                "reasoning.effort '{}' normalized to muse_glimmer default high",
                                other
                            ));
                            "high".to_string()
                        }
                    }
                } else {
                    "high".to_string()
                }
            };
            let normalized = match raw.as_str() {
                "low" => "low",
                "medium" | "med" => "medium",
                "high" => "high",
                "xhigh" => "xhigh",
                "max" => "xhigh",
                other => {
                    push_warn(format!(
                        "reasoning_effort '{}' normalized to muse_glimmer default high",
                        other
                    ));
                    "high"
                }
            };
            effective_effort = Some(normalized.to_string());
            request["reasoning_effort"] = serde_json::json!(normalized);
        }
        ReasoningContract::GemmaBoolean => {
            effective_effort = None;
        }
        ReasoningContract::Unsupported => {
            effective_effort = None;
        }
    }
    let resolution = ReasoningResolution {
        effective_mode: if thinking_enabled {
            "enabled".to_string()
        } else {
            "disabled".to_string()
        },
        effective_effort,
        effective_cap,
        cap_source,
        contract,
        warnings: warnings.clone(),
    };
    Ok(resolution)
}

pub(crate) fn config_value<'a>(
    resolved: &'a hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<&'a hipfire_config::ConfigValue> {
    resolved
        .get(key)
        .map(|value| &value.value)
        .ok_or_else(|| anyhow!("missing resolved configuration key {key}"))
}

pub(crate) fn config_string(
    resolved: &hipfire_config::ResolvedConfig,
    key: &str,
) -> Result<String> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::String(value) => Ok(value.clone()),
        value => bail!("{key} resolved as {}, expected string", value.kind()),
    }
}

pub(crate) fn config_bool(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<bool> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Bool(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected bool", value.kind()),
    }
}

pub(crate) fn config_i64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<i64> {
    match config_value(resolved, key)? {
        hipfire_config::ConfigValue::Integer(value) => Ok(*value),
        value => bail!("{key} resolved as {}, expected integer", value.kind()),
    }
}

pub(crate) fn config_u64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<u64> {
    let value = config_i64(resolved, key)?;
    u64::try_from(value).map_err(|_| anyhow!("{key} cannot be negative"))
}

pub(crate) fn config_optional_u64(
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

pub(crate) fn config_f64(resolved: &hipfire_config::ResolvedConfig, key: &str) -> Result<f64> {
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

pub(crate) fn http_get_json(host: &str, port: u16, path: &str) -> Option<serde_json::Value> {
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
        DaemonDriver, Point, SequentialDriver, SlotDriver, WorkloadSel,
    };

    let points = parse_concurrency(spec)?;
    let max_k = *points.iter().max().expect("non-empty");
    let backend_sel = match args.backend.as_str() {
        "slots" => BackendSel::Slots,
        "noslots" => BackendSel::Sequential,
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

    // No-slots baseline: k requests one after another through the ordinary
    // daemon path. Runs after the slots engine has been dropped, so only one
    // copy of the weights is ever resident.
    if matches!(backend_sel, BackendSel::Sequential | BackendSel::Both) {
        preflight_headroom_for_model(paths, &args.model)?;
        let mut seq_args = args.clone();
        seq_args.concurrency = None;
        let (engine, _, _, _) = open_bench_engine(paths, &seq_args, None)?;
        let mut d = SequentialDriver::start(engine, max_k)?;
        eprintln!("  noslots backend up (sequential daemon path)");
        let r = sweep_backend(
            &mut d as &mut dyn ConcurrencyBackend,
            &arms,
            &points,
            args.runs,
            args.max_tokens as u64,
            &mut out,
        );
        drop(d);
        r?;
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
        // No --head on this path yet; the model's own head is used.
        None,
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
        recorded.hipcc.as_deref(),
        recorded.strict_rocm,
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
    hipcc: Option<&Path>,
    strict_rocm: bool,
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
    if let Some(hipcc) = hipcc
        .map(|p| p.to_string_lossy().into_owned())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
    {
        args.push("--hipcc".to_owned());
        args.push(hipcc);
    }
    if strict_rocm {
        args.push("--strict-rocm".to_owned());
    }
    args
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct RecordedInstallMetadata {
    rocm_root: Option<PathBuf>,
    gpu_arch: Option<String>,
    hipcc: Option<PathBuf>,
    strict_rocm: bool,
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
    let hipcc = value
        .get("hipcc")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    let strict_rocm = match value.get("strict_rocm") {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::String(s)) => {
            let s = s.trim();
            s == "1" || s.eq_ignore_ascii_case("true")
        }
        Some(serde_json::Value::Number(n)) => n.as_u64().is_some_and(|v| v != 0),
        _ => false,
    };
    RecordedInstallMetadata {
        rocm_root,
        gpu_arch,
        hipcc,
        strict_rocm,
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

pub(crate) fn unix_timestamp() -> u64 {
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
                    .find_map(|name| hipfire_config::rocm::tool_from_selected_root(root, name)),
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

pub(crate) fn find_daemon(paths: &Paths) -> Option<PathBuf> {
    if let Some(path) = env::var_os("HIPFIRE_DAEMON_BIN").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    find_daemon_in(paths, &workspace, cfg!(windows))
}

/// Daemon binary name candidates, most preferred first.
///
/// Windows ships the daemon as `daemon.exe`; ELF platforms ship an
/// extensionless `daemon`. The bare spelling is kept as a fallback so a
/// future extensionless shim still wins. `windows` is a pure parameter
/// (mirroring `hipfire_config::rocm::tool_filename_candidates`) so the
/// policy is unit-testable on any host without process-global env.
fn daemon_bin_names(windows: bool) -> &'static [&'static str] {
    if windows {
        &["daemon.exe", "daemon"]
    } else {
        &["daemon"]
    }
}

/// Candidate lookup shared by [`find_daemon`] and its platform-shaped tests:
/// probe the install root (`~/.hipfire/bin/`) and the source-tree target dir
/// (`release/`, then `debug/`), in that order, for each candidate name.
fn find_daemon_in(paths: &Paths, workspace: &std::path::Path, windows: bool) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for name in daemon_bin_names(windows) {
        candidates.push(paths.root.join("bin").join(name));
        candidates.push(workspace.join("release").join(name));
        candidates.push(workspace.join("debug").join(name));
    }
    candidates.into_iter().find(|path| path.is_file())
}

pub(crate) fn request_f64(
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

pub(crate) fn request_u64(
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

pub(crate) fn request_string(
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

pub(crate) fn request_config_value<'a>(
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

pub(crate) fn config_value_f64(value: &hipfire_config::ConfigValue, key: &str) -> Result<f64> {
    match value {
        hipfire_config::ConfigValue::Float(value) => Ok(*value),
        hipfire_config::ConfigValue::Integer(value) => Ok(*value as f64),
        _ => bail!("configuration key '{key}' did not resolve to a number"),
    }
}

pub(crate) fn config_value_u64(value: &hipfire_config::ConfigValue, key: &str) -> Result<u64> {
    match value {
        hipfire_config::ConfigValue::Integer(value) => u64::try_from(*value)
            .map_err(|_| anyhow!("configuration key '{key}' cannot be negative")),
        value => bail!(
            "configuration key '{key}' resolved as {}, expected integer",
            value.kind()
        ),
    }
}

pub(crate) fn insert_optional_f64(target: &mut serde_json::Value, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        target[key] = serde_json::json!(value);
    }
}

pub(crate) fn insert_optional_u64(target: &mut serde_json::Value, key: &str, value: Option<u64>) {
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
    use crate::serve::complete::{
        forward_think_fragments, include_reasoning_content, inject_default_system_message,
        normalize_openai_messages, Completion, ThinkFragment,
    };

    use crate::serve::{serve_instance_token, Admission, ServeMeta, ServeRuntime, ServeShared};
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
    fn model_suffix_filter_covers_current_formats() {
        assert!(is_model_file("qwen3.6-35b-a3b.mq4r"));
        assert!(is_model_file("deepseek.mq2lloyd"));
        assert!(is_model_file("deepseek-v4-flash-0731.mq2r"));
        assert!(is_model_file("deepseek-v4-flash-0731.mq2rxt"));
        assert!(is_model_file("draft.hfq"));
        assert!(!is_model_file("model.triattn.bin"));
        assert!(!is_model_file("README.md"));
    }

    /// The Ornith artifacts shipped briefly as `ornith1.5-*` before being
    /// renamed to `ornith-1.5-*`. Anyone who downloaded during that window has
    /// the old filename on disk, and the registry now points at the new one —
    /// so the canonical tag must still find their file rather than silently
    /// re-downloading 19 GB.
    #[test]
    fn separator_spellings_find_an_already_downloaded_file() {
        let paths = test_paths("separator-spellings");
        let legacy = paths.models.join("ornith1.5-35b-a3b.mq4");
        fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        fs::write(&legacy, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        // The new canonical spelling reaches the old file only via the
        // separator-insensitive fallback — this is the case the rename broke.
        assert_eq!(
            find_model_path(&paths, &registry, "ornith-1.5:35b-a3b"),
            Some(legacy.clone()),
            "hyphenated tag must find the unhyphenated file"
        );
        // The old spelling already worked through the literal pass; pin it so
        // the fallback cannot regress it.
        assert_eq!(
            find_model_path(&paths, &registry, "ornith1.5:35b-a3b"),
            Some(legacy.clone()),
            "legacy tag must keep working"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// The fallback must not outrank a literal hit. With both spellings present
    /// on disk, each input resolves to its own file, not to whichever the
    /// looser comparison happened to reach first.
    #[test]
    fn literal_match_wins_over_the_separator_fallback() {
        let paths = test_paths("separator-precedence");
        let legacy = paths.models.join("ornith1.5-35b-a3b.mq4");
        let renamed = paths.models.join("ornith-1.5-35b-a3b.mq4");
        fs::create_dir_all(paths.models.as_path()).unwrap();
        fs::write(&legacy, b"fixture").unwrap();
        fs::write(&renamed, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        assert_eq!(
            find_model_path(&paths, &registry, "ornith-1.5:35b-a3b"),
            Some(renamed),
            "exact spelling must win when it exists"
        );
        assert_eq!(
            find_model_path(&paths, &registry, "ornith1.5:35b-a3b"),
            Some(legacy),
            "exact spelling must win when it exists"
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    /// An input that normalizes to the empty string must not match everything.
    #[test]
    fn separator_only_input_matches_nothing() {
        let paths = test_paths("separator-empty");
        let model = paths.models.join("qwen3.6-35b-a3b.mq4");
        fs::create_dir_all(model.parent().unwrap()).unwrap();
        fs::write(&model, b"fixture").unwrap();
        let registry = hipfire_registry::bundled().unwrap();

        assert_eq!(find_model_path(&paths, &registry, "-.-"), None);
        fs::remove_dir_all(&paths.root).unwrap();
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
    fn daemon_discovery_prefers_windows_exe_spelling() {
        // Windows-shaped policy (runs on any host, like the rocm.rs HIPCC
        // suffix tests): daemon.exe is probed before the bare name so an
        // install or source-tree build is found on Windows.
        assert_eq!(daemon_bin_names(true), &["daemon.exe", "daemon"]);
        assert_eq!(daemon_bin_names(false), &["daemon"]);
    }

    #[test]
    fn find_daemon_discovers_daemon_exe_under_windows_shaped_policy() {
        // Only the .exe spelling exists — exactly the Windows install layout.
        let paths = test_paths("daemon-exe");
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("daemon.exe"), b"").unwrap();
        let workspace = paths.root.join("target");
        fs::create_dir_all(&workspace).unwrap();
        assert_eq!(
            find_daemon_in(&paths, &workspace, true),
            Some(bin.join("daemon.exe"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn find_daemon_windows_policy_accepts_extensionless_shim() {
        // The bare spelling stays a fallback on Windows for a future shim.
        let paths = test_paths("daemon-shim");
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("daemon"), b"").unwrap();
        let workspace = paths.root.join("target");
        fs::create_dir_all(&workspace).unwrap();
        assert_eq!(
            find_daemon_in(&paths, &workspace, true),
            Some(bin.join("daemon"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn find_daemon_prefers_install_dir_over_source_tree() {
        // Install root (~/.hipfire/bin) wins over the source-tree target dir
        // even when both carry a candidate (real host: Windows + dev build).
        let paths = test_paths("daemon-install-vs-target");
        let bin = paths.root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("daemon.exe"), b"install").unwrap();
        let workspace = paths.root.join("target");
        fs::create_dir_all(workspace.join("release")).unwrap();
        fs::write(workspace.join("release").join("daemon.exe"), b"dev").unwrap();
        assert_eq!(
            find_daemon_in(&paths, &workspace, true),
            Some(bin.join("daemon.exe"))
        );
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn find_daemon_falls_back_to_bare_spelling_for_unix_shaped_policy() {
        // Unix-shaped policy: only the extensionless daemon is probed.
        let paths = test_paths("daemon-bare");
        let release = paths.root.join("target").join("release");
        fs::create_dir_all(&release).unwrap();
        fs::write(release.join("daemon"), b"").unwrap();
        let workspace = paths.root.join("target");
        assert_eq!(
            find_daemon_in(&paths, &workspace, false),
            Some(release.join("daemon"))
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
        let params =
            load_params(&defaults, Some(entry), &model_path, 64, None, None, None).unwrap();
        assert_eq!(params["cask"], false);
        assert_eq!(params["cask_handoff_tokens"], 0);
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
        let params = load_params(&enabled, Some(entry), &model_path, 64, None, None, None).unwrap();
        assert_eq!(params["cask"], false);
        assert_eq!(params["cask_sidecar"], sidecar_path.display().to_string());
        assert_eq!(params["prefill_compression"], "off");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_forwards_explicit_vmm_backend() {
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(
            &defaults,
            None,
            &model_path,
            64,
            Some("q8"),
            Some("vmm"),
            None,
        )
        .unwrap();
        assert_eq!(params["kv_backend"], "vmm");
    }

    #[test]
    pub(crate) fn load_params_defaults_to_schema_contiguous_backend() {
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(&defaults, None, &model_path, 64, Some("q8"), None, None).unwrap();
        assert_eq!(params["kv_backend"], "contiguous");
        assert_eq!(params["max_seq"], 32768);
    }

    #[test]
    pub(crate) fn resolved_for_model_applies_qwen_tag_policy_and_excludes_original_and_sidecars() {
        let paths = test_paths("registry-qwen-tag-policy");
        fs::create_dir_all(&paths.root).unwrap();
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "qwen3.5:4b":{"repo":"x","file":"qwen3.5-4b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3.6:35b-a3b":{"repo":"x","file":"qwen3.6-35b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3.8:27b-fast":{"repo":"x","file":"qwen3.8-27b.mq4r","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3:8b":{"repo":"x","file":"qwen3-8b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"q8"},
                "qwen3.5:9b-draft":{"repo":"x","file":"qwen35-9b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "qwen3.6:27b-dflash":{"repo":"x","file":"qwen36-27b-dflash.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}
            },
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        // Exact Qwen families get VMM + 262144 + 81920
        for tag in [
            "qwen3.5:4b",
            "qwen3.6:35b-a3b",
            "qwen3.8:27b",
            "qwen3.8:27b-fast",
        ] {
            let (_, entry) = registry.model(tag).unwrap();
            let resolved = resolved_for_model(&paths, tag, Some(tag), Some(entry)).unwrap();
            assert_eq!(
                config_string(&resolved, "memory.kv_backend").unwrap(),
                "vmm",
                "{tag}"
            );
            assert_eq!(
                config_u64(&resolved, "memory.max_seq").unwrap(),
                262144,
                "{tag}"
            );
            assert_eq!(
                config_u64(&resolved, "generation.max_tokens").unwrap(),
                81920,
                "{tag}"
            );
        }

        // Original qwen3:* stays contiguous (no automatic policy) — original Qwen3 uses default schema.
        let (_, entry) = registry.model("qwen3:8b").unwrap();
        let resolved =
            resolved_for_model(&paths, "qwen3:8b", Some("qwen3:8b"), Some(entry)).unwrap();
        assert_eq!(
            config_string(&resolved, "memory.kv_backend").unwrap(),
            "contiguous",
            "original qwen3 must keep the built-in contiguous backend"
        );
        assert_eq!(config_u64(&resolved, "memory.max_seq").unwrap(), 32768);
        assert_eq!(
            config_u64(&resolved, "generation.max_tokens").unwrap(),
            4096
        );
        // More directly, check the helper layer itself has no policy.
        let direct = hipfire_registry::config_layer_for_tag("qwen3:8b", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());

        // Draft/dflash sidecars do not get the Qwen policy even though family matches.
        for tag in ["qwen3.5:9b-draft", "qwen3.6:27b-dflash"] {
            let (_, entry) = registry.model(tag).unwrap();
            let direct = hipfire_registry::config_layer_for_tag(tag, entry).unwrap();
            assert!(
                direct.get("memory.kv_backend").is_none(),
                "{tag} sidecar must not get vmm"
            );
            assert!(direct.get("memory.max_seq").is_none());
            assert!(direct.get("generation.max_tokens").is_none());
        }
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn resolved_for_model_applies_glimmer_and_deepseek_targets() {
        let paths = test_paths("registry-glimmer-deepseek-tag-policy");
        fs::create_dir_all(&paths.root).unwrap();
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{
                "muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:fast":{"repo":"x","file":"muse-glimmer-30b.mq4r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "muse-glimmer:draft":{"repo":"x","file":"muse-glimmer-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash":{"repo":"x","file":"deepseek-v4-flash-0731.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:mq2lloyd":{"repo":"x","file":"deepseek-v4-flash-0731.mq2lloyd","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash-preview":{"repo":"x","file":"deepseek-v4-flash-preview.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "deepseek-v4-flash:draft":{"repo":"x","file":"deepseek-v4-flash-draft.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"},
                "other:model":{"repo":"x","file":"other.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}
            },
            "aliases":{
                "deepseek4":"deepseek-v4-flash",
                "ds4":"deepseek-v4-flash",
                "deepseek4:preview":"deepseek-v4-flash-preview",
                "muse-glimmer:quality":"muse-glimmer"
            }
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();

        // Muse Glimmer quality and fast targets get VMM + native 131072, no invented max_tokens.
        for tag in ["muse-glimmer", "muse-glimmer:fast"] {
            let (_, entry) = registry.model(tag).unwrap();
            let resolved = resolved_for_model(&paths, tag, Some(tag), Some(entry)).unwrap();
            assert_eq!(
                config_string(&resolved, "memory.kv_backend").unwrap(),
                "vmm",
                "{tag}"
            );
            assert_eq!(
                config_u64(&resolved, "memory.max_seq").unwrap(),
                131072,
                "{tag}"
            );
            let direct = hipfire_registry::config_layer_for_tag(tag, entry).unwrap();
            assert_eq!(
                direct.get("memory.kv_backend"),
                Some(&hipfire_config::ConfigValue::String("vmm".into()))
            );
            assert_eq!(
                direct.get("memory.max_seq"),
                Some(&hipfire_config::ConfigValue::Integer(131072)),
                "{tag} should get 131072"
            );
            assert!(
                direct.get("generation.max_tokens").is_none(),
                "{tag} must not get max_tokens"
            );
        }
        // quality alias lands on trunk policy.
        let (resolved_tag, entry) = registry.model("muse-glimmer:quality").unwrap();
        assert_eq!(resolved_tag, "muse-glimmer");
        let direct = hipfire_registry::config_layer_for_tag(resolved_tag, entry).unwrap();
        assert_eq!(
            direct.get("memory.max_seq"),
            Some(&hipfire_config::ConfigValue::Integer(131072))
        );

        // Muse Glimmer draft receives none.
        let (_, entry) = registry.model("muse-glimmer:draft").unwrap();
        let direct = hipfire_registry::config_layer_for_tag("muse-glimmer:draft", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());
        let resolved = resolved_for_model(
            &paths,
            "muse-glimmer:draft",
            Some("muse-glimmer:draft"),
            Some(entry),
        )
        .unwrap();
        assert!(
            resolved.get("memory.kv_backend").is_none()
                || config_string(&resolved, "memory.kv_backend").unwrap() != "vmm"
        );

        // DeepSeek official / MQ2Lloyd / preview targets get VMM + 1M + 384Ki.
        for tag in [
            "deepseek-v4-flash",
            "deepseek-v4-flash:mq2lloyd",
            "deepseek-v4-flash-preview",
        ] {
            let (resolved_tag, entry) = registry.model(tag).unwrap();
            let resolved =
                resolved_for_model(&paths, resolved_tag, Some(resolved_tag), Some(entry)).unwrap();
            assert_eq!(
                config_string(&resolved, "memory.kv_backend").unwrap(),
                "vmm",
                "{tag}"
            );
            assert_eq!(
                config_u64(&resolved, "memory.max_seq").unwrap(),
                1048576,
                "{tag}"
            );
            assert_eq!(
                config_u64(&resolved, "generation.max_tokens").unwrap(),
                393216,
                "{tag}"
            );
            let direct = hipfire_registry::config_layer_for_tag(resolved_tag, entry).unwrap();
            assert_eq!(
                direct.get("memory.kv_backend"),
                Some(&hipfire_config::ConfigValue::String("vmm".into()))
            );
            assert_eq!(
                direct.get("memory.max_seq"),
                Some(&hipfire_config::ConfigValue::Integer(1048576))
            );
            assert_eq!(
                direct.get("generation.max_tokens"),
                Some(&hipfire_config::ConfigValue::Integer(393216))
            );
        }
        for alias in ["deepseek4", "ds4", "deepseek4:preview"] {
            let (resolved_tag, entry) = registry.model(alias).unwrap();
            let direct = hipfire_registry::config_layer_for_tag(resolved_tag, entry).unwrap();
            assert_eq!(
                direct.get("memory.max_seq"),
                Some(&hipfire_config::ConfigValue::Integer(1048576)),
                "{alias}->{resolved_tag}"
            );
            assert_eq!(
                direct.get("generation.max_tokens"),
                Some(&hipfire_config::ConfigValue::Integer(393216)),
                "{alias}->{resolved_tag}"
            );
        }
        // DeepSeek draft sidecar receives none.
        let (_, entry) = registry.model("deepseek-v4-flash:draft").unwrap();
        let direct =
            hipfire_registry::config_layer_for_tag("deepseek-v4-flash:draft", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());

        // Absent policy: unrelated model gets no automatic policy.
        let (_, entry) = registry.model("other:model").unwrap();
        let direct = hipfire_registry::config_layer_for_tag("other:model", entry).unwrap();
        assert!(direct.get("memory.kv_backend").is_none());
        assert!(direct.get("memory.max_seq").is_none());
        assert!(direct.get("generation.max_tokens").is_none());

        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn resolved_for_model_tag_policy_is_overridable_by_user() {
        let paths = test_paths("registry-tag-policy-override");
        fs::create_dir_all(&paths.root).unwrap();
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"qwen3.8:27b":{"repo":"x","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (tag, entry) = registry.model("qwen3.8:27b").unwrap();
        let resolved = resolved_for_model(&paths, tag, Some(tag), Some(entry)).unwrap();
        assert_eq!(
            config_string(&resolved, "memory.kv_backend").unwrap(),
            "vmm"
        );
        assert_eq!(config_u64(&resolved, "memory.max_seq").unwrap(), 262144);
        assert_eq!(
            config_u64(&resolved, "generation.max_tokens").unwrap(),
            81920
        );

        // Global user override wins over registry tag policy (registry below global).
        let mut user_layer = ConfigLayer::default();
        user_layer
            .set_cli("memory.kv_backend", "contiguous")
            .unwrap();
        user_layer.set_cli("memory.max_seq", "32768").unwrap();
        user_layer.set_cli("generation.max_tokens", "1024").unwrap();
        let overridden = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: hipfire_registry::config_layer_for_tag(tag, entry).unwrap(),
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test.toml"),
                },
                layer: user_layer,
            },
        ])
        .unwrap();
        assert_eq!(
            config_string(&overridden, "memory.kv_backend").unwrap(),
            "contiguous"
        );
        assert_eq!(config_u64(&overridden, "memory.max_seq").unwrap(), 32768);
        assert_eq!(
            config_u64(&overridden, "generation.max_tokens").unwrap(),
            1024
        );

        // Also verify load_params respects explicit kv_backend override over configured vmm.
        let model_path = PathBuf::from("/tmp/test-model.mq4");
        let params = load_params(
            &resolved,
            Some(entry),
            &model_path,
            64,
            Some("q8"),
            Some("contiguous"),
            None,
        )
        .unwrap();
        assert_eq!(params["kv_backend"], "contiguous");
        // Without explicit override, load_params uses the resolved vmm.
        let params2 = load_params(
            &resolved,
            Some(entry),
            &model_path,
            64,
            Some("q8"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(params2["kv_backend"], "vmm");
        assert_eq!(params2["max_seq"], 262144);

        // Glimmer target likewise overridable (backend + max_seq).
        let raw2 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"muse-glimmer":{"repo":"x","file":"muse-glimmer-30b.mq4","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry2 = RegistryV1::parse(raw2, "test").unwrap();
        let (g_tag, g_entry) = registry2.model("muse-glimmer").unwrap();
        let g_layer = hipfire_registry::config_layer_for_tag(g_tag, g_entry).unwrap();
        assert_eq!(
            g_layer.get("memory.kv_backend"),
            Some(&hipfire_config::ConfigValue::String("vmm".into()))
        );
        assert_eq!(
            g_layer.get("memory.max_seq"),
            Some(&hipfire_config::ConfigValue::Integer(131072))
        );
        assert!(g_layer.get("generation.max_tokens").is_none());
        let mut g_user = ConfigLayer::default();
        g_user.set_cli("memory.kv_backend", "contiguous").unwrap();
        g_user.set_cli("memory.max_seq", "8192").unwrap();
        let g_resolved = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: g_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: g_layer,
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test2.toml"),
                },
                layer: g_user,
            },
        ])
        .unwrap();
        assert_eq!(
            config_string(&g_resolved, "memory.kv_backend").unwrap(),
            "contiguous"
        );
        assert_eq!(config_u64(&g_resolved, "memory.max_seq").unwrap(), 8192);

        // DeepSeek target override wins over 1M/384Ki policy.
        let raw3 = r#"{
            "schema_version":1,
            "generated_at":"2026-09-01T00:00:00Z",
            "models":{"deepseek-v4-flash":{"repo":"x","file":"ds4.mq2r","size_gb":1,"min_vram_gb":1,"desc":"x"}},
            "aliases":{}
        }"#;
        let registry3 = RegistryV1::parse(raw3, "test").unwrap();
        let (d_tag, d_entry) = registry3.model("deepseek-v4-flash").unwrap();
        let d_resolved = resolved_for_model(&paths, d_tag, Some(d_tag), Some(d_entry)).unwrap();
        assert_eq!(
            config_string(&d_resolved, "memory.kv_backend").unwrap(),
            "vmm"
        );
        assert_eq!(config_u64(&d_resolved, "memory.max_seq").unwrap(), 1048576);
        assert_eq!(
            config_u64(&d_resolved, "generation.max_tokens").unwrap(),
            393216
        );
        let mut d_user = ConfigLayer::default();
        d_user.set_cli("memory.max_seq", "65536").unwrap();
        d_user.set_cli("generation.max_tokens", "2048").unwrap();
        let d_overridden = hipfire_config::resolve(vec![
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::RegistryModel {
                    tag: d_tag.to_owned(),
                    revision: "v1".into(),
                },
                layer: hipfire_registry::config_layer_for_tag(d_tag, d_entry).unwrap(),
            },
            hipfire_config::NamedLayer {
                source: hipfire_config::ConfigSource::GlobalUser {
                    path: std::path::PathBuf::from("/tmp/test3.toml"),
                },
                layer: d_user,
            },
        ])
        .unwrap();
        assert_eq!(config_u64(&d_overridden, "memory.max_seq").unwrap(), 65536);
        assert_eq!(
            config_u64(&d_overridden, "generation.max_tokens").unwrap(),
            2048
        );

        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    pub(crate) fn load_params_only_forwards_explicit_deepseek4_expert_fanout() {
        let model_path = PathBuf::from("/tmp/test-model.mq2r");
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        let params = load_params(&defaults, None, &model_path, 64, Some("q8"), None, None).unwrap();
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
        let params = load_params(&resolved, None, &model_path, 64, Some("q8"), None, None).unwrap();
        assert_eq!(params["deepseek4_experts_per_token"], 4);
    }

    #[test]
    pub(crate) fn load_params_forwards_typed_deepseek4_compute_placement() {
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
            None,
        )
        .unwrap();
        assert_eq!(params["deepseek4_compute_placement"], raw);
    }

    #[test]
    pub(crate) fn load_params_forwards_dflash_draft_from_environment() {
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

        let params = load_params(&resolved, None, &model_path, 64, Some("q8"), None, None).unwrap();
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
        let mut params =
            load_params(&resolved, None, &model_path, 64, Some("q8"), None, None).unwrap();
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
    pub(crate) fn artifact_urls_honor_endpoint_precedence() {
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
        let suffix = "hipfire-models/qwen3.6-35b-a3b/resolve/main/qwen3.6-35b-a3b.mq4r";

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
            recorded.hipcc.as_deref(),
            recorded.strict_rocm,
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
        assert!(empty.hipcc.is_none());
        assert!(!empty.strict_rocm);
        let bare = installer_handoff_args(
            &RevisionSelector {
                value: "deadbeef".into(),
                kind: RevisionKind::Commit,
            },
            None,
            None,
            None,
            false,
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
            None,
            false,
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
    fn update_handoff_forwards_hipcc_and_strict_with_backward_compat() {
        let home = env::temp_dir().join(format!(
            "hipfire-update-hipcc-{}-{}",
            std::process::id(),
            unix_timestamp()
        ));
        fs::create_dir_all(&home).unwrap();
        // New format with hipcc and strict_rocm.
        fs::write(
            home.join("install.json"),
            r#"{"rocm_root":"/opt/rocm","hipcc":"/usr/bin/hipcc","strict_rocm":true,"gpu_arch":"gfx1201"}"#,
        )
        .unwrap();
        let recorded = recorded_install_metadata(&home);
        assert_eq!(recorded.hipcc.as_deref(), Some(Path::new("/usr/bin/hipcc")));
        assert!(recorded.strict_rocm);
        assert_eq!(recorded.rocm_root.as_deref(), Some(Path::new("/opt/rocm")));
        let args = installer_handoff_args(
            &RevisionSelector {
                value: "master".into(),
                kind: RevisionKind::Auto,
            },
            recorded.rocm_root.as_deref(),
            recorded.gpu_arch.as_deref(),
            recorded.hipcc.as_deref(),
            recorded.strict_rocm,
        );
        assert!(args.contains(&"--hipcc".to_owned()));
        assert!(args.contains(&"/usr/bin/hipcc".to_owned()));
        assert!(args.contains(&"--strict-rocm".to_owned()));
        assert!(args.contains(&"--rocm-root".to_owned()));
        // Empty/whitespace hipcc is treated as None, like rocm_root.
        fs::write(
            home.join("install.json"),
            r#"{"hipcc":"  ","strict_rocm":false}"#,
        )
        .unwrap();
        let empty = recorded_install_metadata(&home);
        assert!(empty.hipcc.is_none());
        assert!(!empty.strict_rocm);
        let bare = installer_handoff_args(
            &RevisionSelector {
                value: "beta".into(),
                kind: RevisionKind::Branch,
            },
            None,
            None,
            empty.hipcc.as_deref(),
            empty.strict_rocm,
        );
        assert!(!bare.contains(&"--hipcc".to_owned()));
        assert!(!bare.contains(&"--strict-rocm".to_owned()));
        // Older file without hipcc key loads without error (backward compat).
        fs::write(
            home.join("install.json"),
            r#"{"rocm_root":"/opt/rocm","gpu_arch":"gfx1100"}"#,
        )
        .unwrap();
        let old = recorded_install_metadata(&home);
        assert_eq!(old.rocm_root.as_deref(), Some(Path::new("/opt/rocm")));
        assert!(old.hipcc.is_none());
        assert!(!old.strict_rocm);
        // Strict can be stored as string \"1\" or number 1 for compat.
        fs::write(home.join("install.json"), r#"{"strict_rocm":"1"}"#).unwrap();
        assert!(recorded_install_metadata(&home).strict_rocm);
        fs::write(home.join("install.json"), r#"{"strict_rocm":1}"#).unwrap();
        assert!(recorded_install_metadata(&home).strict_rocm);
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
    fn registry_system_prompt_is_injected_only_when_client_omits_one() {
        let mut messages = normalize_openai_messages(
            Some(&serde_json::json!([
                { "role": "user", "content": "hello" }
            ])),
            false,
        );
        inject_default_system_message(&mut messages, Some("registry identity"));
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "registry identity");

        let mut messages = normalize_openai_messages(
            Some(&serde_json::json!([
                { "role": "developer", "content": "client policy" },
                { "role": "user", "content": "hello" }
            ])),
            false,
        );
        inject_default_system_message(&mut messages, Some("registry identity"));
        assert_eq!(messages.as_array().unwrap().len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "client policy");
    }

    #[test]
    fn normalize_reasoning_sources_with_flag_on_and_off() {
        // reasoning field takes precedence over reasoning_content and inline think
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "<think>inline</think>\nvisible",
                "reasoning": "explicit reasoning",
                "reasoning_content": "secondary"
            }]
        });
        let off = normalize_openai_messages(body.get("messages"), false);
        assert_eq!(off[0]["content"], "visible");
        assert_eq!(off[0]["tool_plan"], "explicit reasoning");
        assert!(off[0].get("reasoning_content").is_none());
        let on = normalize_openai_messages(body.get("messages"), true);
        assert_eq!(on[0]["content"], "visible");
        assert_eq!(on[0]["tool_plan"], "explicit reasoning");
        assert_eq!(on[0]["reasoning_content"], "explicit reasoning");
        assert_eq!(on[0]["reasoning_content"], on[0]["tool_plan"]);

        // reasoning_content when reasoning absent
        let body2 = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "visible only",
                "reasoning_content": "from content field"
            }]
        });
        let off2 = normalize_openai_messages(body2.get("messages"), false);
        assert_eq!(off2[0]["tool_plan"], "from content field");
        assert!(off2[0].get("reasoning_content").is_none());
        let on2 = normalize_openai_messages(body2.get("messages"), true);
        assert_eq!(on2[0]["reasoning_content"], "from content field");
        assert_eq!(on2[0]["tool_plan"], "from content field");
        assert_eq!(on2[0]["content"], "visible only");

        // inline <think> when neither reasoning field present
        let body3 = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "<think>inline think</think>\n\nvisible answer"
            }]
        });
        let off3 = normalize_openai_messages(body3.get("messages"), false);
        assert_eq!(off3[0]["content"], "visible answer");
        assert_eq!(off3[0]["tool_plan"], "inline think");
        assert!(off3[0].get("reasoning_content").is_none());
        let on3 = normalize_openai_messages(body3.get("messages"), true);
        assert_eq!(on3[0]["content"], "visible answer");
        assert_eq!(on3[0]["tool_plan"], "inline think");
        assert_eq!(on3[0]["reasoning_content"], "inline think");
        assert_eq!(on3[0]["reasoning_content"], on3[0]["tool_plan"]);
    }

    #[test]
    fn include_reasoning_content_arch_predicate() {
        assert!(include_reasoning_content(Some("muse_glimmer")));
        assert!(include_reasoning_content(Some("qwen35")));
        assert!(include_reasoning_content(Some("qwen35-vl")));
        assert!(include_reasoning_content(Some("qwen36")));
        assert!(include_reasoning_content(Some("Qwen3.6-A3B")));
        assert!(include_reasoning_content(Some("qwen3.5")));
        assert!(!include_reasoning_content(Some("llama")));
        assert!(!include_reasoning_content(Some("gemma4")));
        assert!(!include_reasoning_content(Some("qwen2")));
        assert!(!include_reasoning_content(None));
    }

    #[test]
    fn normalize_reasoning_emitted_for_qwen_include_flag() {
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "visible",
                "reasoning_content": "qwen chain of thought"
            }]
        });
        let include = include_reasoning_content(Some("qwen35"));
        assert!(include);
        let on = normalize_openai_messages(body.get("messages"), include);
        assert_eq!(on[0]["reasoning_content"], "qwen chain of thought");
        assert_eq!(on[0]["tool_plan"], "qwen chain of thought");
        assert_eq!(on[0]["content"], "visible");

        let off = normalize_openai_messages(
            body.get("messages"),
            include_reasoning_content(Some("llama")),
        );
        assert!(off[0].get("reasoning_content").is_none());
        assert_eq!(off[0]["tool_plan"], "qwen chain of thought");
    }

    #[test]
    fn normalize_tool_call_id_and_tool_result_name_survive() {
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [{
                        "id": "call_42",
                        "type": "function",
                        "function": { "name": "my_tool", "arguments": "{}" }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_42",
                    "name": "my_tool",
                    "content": "result"
                }
            ]
        });
        for flag in [false, true] {
            let normalized = normalize_openai_messages(body.get("messages"), flag);
            assert_eq!(normalized[0]["tool_calls"][0]["id"], "call_42");
            assert_eq!(normalized[0]["tool_calls"][0]["name"], "my_tool");
            assert_eq!(normalized[1]["tool_call_id"], "call_42");
            assert_eq!(normalized[1]["name"], "my_tool");
            assert_eq!(normalized[1]["content"], "result");
        }
    }

    #[test]
    fn normalize_glimmer_flag_rejects_non_object_arguments_string() {
        let body = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "x",
                "tool_calls": [{
                    "function": { "name": "t", "arguments": "not-json" }
                }]
            }]
        });
        let off = normalize_openai_messages(body.get("messages"), false);
        assert_eq!(
            off[0]["tool_calls"][0]["arguments"],
            serde_json::json!({ "_raw": "not-json" })
        );
        let on = normalize_openai_messages(body.get("messages"), true);
        assert_eq!(
            on[0]["tool_calls"][0]["arguments"],
            serde_json::Value::String("not-json".into())
        );
        // JSON string that parses to non-object (array) also surfaces as string under glimmer
        let body_arr = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": "x",
                "tool_calls": [{
                    "function": { "name": "t", "arguments": "[1,2]" }
                }]
            }]
        });
        let on_arr = normalize_openai_messages(body_arr.get("messages"), true);
        assert_eq!(
            on_arr[0]["tool_calls"][0]["arguments"],
            serde_json::Value::String("[1,2]".into())
        );
        let off_arr = normalize_openai_messages(body_arr.get("messages"), false);
        // non-glimmer keeps parsed array (today's behaviour is to keep whatever parsed)
        assert_eq!(
            off_arr[0]["tool_calls"][0]["arguments"],
            serde_json::json!([1, 2])
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
            logprobs: None,
            reasoning: None,
        }
    }

    fn sample_tc(name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: None,
            name: name.into(),
            arguments,
            rendered_body: None,
        }
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
            logprobs: None,
            reasoning: None,
        }
    }

    fn sample_tool_call(name: &str) -> serde_json::Value {
        serde_json::json!({
            "name": name,
            "arguments": { "path": "README.md" }
        })
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
    fn task15_serve_retry_config_defaults_off() {
        let resolved = resolve(Vec::<NamedLayer>::new()).expect("resolve empty layers");
        let enabled = config_bool(&resolved, "serve.retry_enabled").expect("retry_enabled");
        let backoff = config_u64(&resolved, "serve.retry_backoff_ms").expect("retry_backoff_ms");
        assert!(!enabled, "serve.retry_enabled must default false");
        assert_eq!(backoff, 50);
    }

    // --- StreamContractGate / complete_request framing (fix round 2) ---

    // ── Task 6: canonical OpenAI tool-call adapter + endpoint registry ──

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
        let script = include_str!("serve/fake_daemon.py");
        fs::write(&daemon, script).unwrap();
        fs::set_permissions(&daemon, fs::Permissions::from_mode(0o755)).unwrap();
        daemon
    }

    /// In-process Hyper harness: Engine::spawn_configured → serve_listener_until.
    /// Does not touch HIPFIRE_DAEMON_BIN.
    #[cfg(unix)]
    struct Task11HttpHarness {
        paths: Paths,
        port: u16,
        model_name: String,
        shared: Arc<ServeShared>,
        shutdown: tokio_util::sync::CancellationToken,
        _join: Option<thread::JoinHandle<()>>,
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
                metrics: crate::serve::metrics::Metrics::default(),
                runtime: Mutex::new(ServeRuntime {
                    engine,
                    paths: paths.clone(),
                    registry,
                    current_path: None,
                    current_arch: None,
                    current_reasoning_contract: ReasoningContract::Unsupported,
                    current_reasoning_effort_native: false,
                    current_reasoning_efforts: Vec::new(),
                    continuous_batch_capable: false,
                    current_max_seq: 0,
                    cache_capable: false,
                    kv_override: None,
                    kv_backend_override: None,
                    tp: None,
                    continuous_batch_size: 1,
                    multi_slot_enabled: false,
                    multi_slot_slots: 4,
                    multi_slot_ctx: 8192,
                    multi_slot_prefill_chunk: 1024,
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

            let std_listener =
                std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral serve port");
            std_listener
                .set_nonblocking(true)
                .expect("set nonblocking listener");
            let port = std_listener
                .local_addr()
                .expect("listener local addr")
                .port();

            let shutdown = tokio_util::sync::CancellationToken::new();
            let shutdown_loop = shutdown.clone();
            let shared_loop = Arc::clone(&shared);
            let join = thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("task11 runtime");
                rt.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(std_listener)
                        .expect("tokio listener from std");
                    if let Err(error) = crate::serve::http::serve_listener_until(
                        listener,
                        shared_loop,
                        shutdown_loop,
                    )
                    .await
                    {
                        eprintln!("[task11-harness] serve failed: {error:#}");
                    }
                });
            });

            // Health probe — proves production Hyper service path is live.
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
                shutdown,
                _join: Some(join),
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
            self.shutdown.cancel();
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

    /// Raw non-stream POST; returns HTTP status and body bytes (no client parse).
    #[cfg(unix)]
    fn raw_nonstream_post(port: u16, body: &serde_json::Value) -> (u16, Vec<u8>) {
        use std::net::TcpStream;
        let payload = body.to_string();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {payload}",
            payload.len(),
        );
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("write timeout");
        stream.write_all(request.as_bytes()).expect("write request");
        let mut buf = Vec::new();
        let _ = stream.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        let status = text
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .unwrap_or(0);
        let body_start = text
            .find("\r\n\r\n")
            .map(|idx| idx + 4)
            .or_else(|| text.find("\n\n").map(|idx| idx + 2))
            .unwrap_or(buf.len());
        let raw_body = buf.get(body_start..).unwrap_or(&[]);
        let chunked = text[..body_start]
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked");
        if !chunked {
            return (status, raw_body.to_vec());
        }
        let mut decoded = Vec::new();
        let mut remaining = raw_body;
        loop {
            let Some(line_end) = remaining.windows(2).position(|bytes| bytes == b"\r\n") else {
                panic!("malformed chunk header");
            };
            let size = usize::from_str_radix(
                std::str::from_utf8(&remaining[..line_end])
                    .expect("chunk size utf8")
                    .split(';')
                    .next()
                    .expect("chunk size"),
                16,
            )
            .expect("chunk size hex");
            remaining = &remaining[line_end + 2..];
            if size == 0 {
                break;
            }
            assert!(remaining.len() >= size + 2, "truncated chunk body");
            decoded.extend_from_slice(&remaining[..size]);
            assert_eq!(&remaining[size..size + 2], b"\r\n");
            remaining = &remaining[size + 2..];
        }
        (status, decoded)
    }

    /// Write a non-stream request and return the live TCP stream without reading status.
    #[cfg(unix)]
    fn open_nonstream_request(port: u16, body: &serde_json::Value) -> std::net::TcpStream {
        use std::net::TcpStream;
        let payload = body.to_string();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: 127.0.0.1:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {payload}",
            payload.len(),
        );
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect serve");
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .expect("write timeout");
        // Short read timeout so accidental reads fail fast; tests close without status.
        stream
            .set_read_timeout(Some(Duration::from_millis(50)))
            .expect("read timeout");
        stream.write_all(request.as_bytes()).expect("write request");
        let _ = stream.flush();
        stream
    }

    #[cfg(unix)]
    fn wait_fixture_ready(harness: &Task11HttpHarness, deadline: Instant) -> serde_json::Value {
        let ready_path = harness.paths.root.join("fixture-ready.log");
        while Instant::now() < deadline {
            if let Ok(raw) = fs::read_to_string(&ready_path) {
                for line in raw.lines().filter(|l| !l.trim().is_empty()) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                        if val.get("type").and_then(|v| v.as_str()) == Some("fixture_ready") {
                            return val;
                        }
                    }
                }
            }
            // Fallback: requests.log also records fixture_ready.
            if let Some(row) = harness
                .read_requests_log()
                .into_iter()
                .find(|row| row.get("type").and_then(|v| v.as_str()) == Some("fixture_ready"))
            {
                return row;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!(
            "fixture-ready never appeared; log={:?}",
            harness.read_requests_log()
        );
    }

    #[cfg(unix)]
    #[test]
    fn oversized_declared_body_is_rejected_before_body_read() {
        let harness = Task11HttpHarness::spawn("oversized-content-length");
        let mut stream =
            std::net::TcpStream::connect(("127.0.0.1", harness.port())).expect("connect serve");
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        stream
            .write_all(
                b"POST /v1/chat/completions HTTP/1.1\r\n\
                  Host: localhost\r\n\
                  Content-Type: application/json\r\n\
                  Content-Length: 8388609\r\n\
                  Connection: close\r\n\r\n",
            )
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(
            response.starts_with("HTTP/1.1 413"),
            "oversized declaration was not rejected immediately: {response:?}"
        );
    }

    /// Silent non-stream client disconnect aborts the correlated daemon txn,
    /// drains done/aborted before Admission releases, then admits a follow-up.
    #[cfg(unix)]
    #[test]
    fn nonstream_client_disconnect_aborts_and_releases_admission() {
        let harness = Task11HttpHarness::spawn("ns-disconnect-abort");
        let port = harness.port();
        let body = harness.base_body("t11-long-nonstream", false);

        assert_eq!(
            harness.shared.admission.inflight(),
            0,
            "precondition: admission idle"
        );

        let stream = open_nonstream_request(port, &body);
        let ready_deadline = Instant::now() + Duration::from_secs(5);
        let ready = wait_fixture_ready(&harness, ready_deadline);
        let gen_id = ready
            .get("id")
            .cloned()
            .expect("fixture_ready carries generate id");
        let gen_aid = ready
            .get("attempt_id")
            .and_then(|v| v.as_u64())
            .expect("fixture_ready carries attempt_id");

        // Confirm generate is in the wire log with the same correlation pair.
        let request_log = harness.read_requests_log();
        let generates = Task11HttpHarness::ops_of_type(&request_log, "generate");
        assert!(
            generates.iter().any(|g| {
                g.get("id") == Some(&gen_id)
                    && g.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid)
            }),
            "generate missing for fixture pair: {:?}",
            harness.read_requests_log()
        );

        // Client disconnect without ever reading an HTTP status line.
        drop(stream);

        let poll_deadline = Instant::now() + Duration::from_secs(5);
        let mut saw_abort = false;
        let mut saw_done_aborted = false;
        let mut inflight_hit_zero_before_done = false;

        while Instant::now() < poll_deadline {
            let log = harness.read_requests_log();
            if !saw_abort {
                saw_abort = log.iter().any(|row| {
                    row.get("type").and_then(|v| v.as_str()) == Some("abort")
                        && row.get("id") == Some(&gen_id)
                        && row.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid)
                });
            }
            if !saw_done_aborted {
                saw_done_aborted = log.iter().any(|row| {
                    row.get("type").and_then(|v| v.as_str()) == Some("daemon_done_aborted")
                        && row.get("id") == Some(&gen_id)
                        && row.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid)
                });
            }
            let inflight = harness.shared.admission.inflight();
            if inflight == 0 && !saw_done_aborted {
                inflight_hit_zero_before_done = true;
            }
            if saw_abort && saw_done_aborted && inflight == 0 {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }

        assert!(
            saw_abort,
            "correlated abort{{id,attempt_id}} missing within poll bound; log={:?}",
            harness.read_requests_log()
        );
        assert!(
            saw_done_aborted,
            "daemon done/aborted marker missing; log={:?}",
            harness.read_requests_log()
        );
        assert!(
            !inflight_hit_zero_before_done,
            "Admission inflight reached 0 before done/aborted drained"
        );
        assert_eq!(
            harness.shared.admission.inflight(),
            0,
            "admission must be fully released after abort drain"
        );

        // No commit may have been sent for the cancelled attempt.
        let log = harness.read_requests_log();
        let commits = Task11HttpHarness::ops_of_type(&log, "commit");
        assert!(
            commits.iter().all(|c| {
                !(c.get("id") == Some(&gen_id)
                    && c.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid))
            }),
            "cancelled attempt must not commit: {commits:?}"
        );

        // Immediate follow-up must admit and return a single JSON success.
        let follow = complete_nonstream(port, harness.base_body("t11-stop-text", false))
            .expect("follow-up after cancel must return 200 JSON");
        assert_eq!(follow["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            follow["choices"][0]["message"]["content"],
            "hello from fake daemon"
        );
        let release_deadline = Instant::now() + Duration::from_millis(500);
        while harness.shared.admission.inflight() != 0 && Instant::now() < release_deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            harness.shared.admission.inflight(),
            0,
            "follow-up admission guard did not release after flushed response"
        );
    }

    /// Connected pre-terminal daemon error keeps non-200 OpenAI error JSON.
    #[cfg(unix)]
    #[test]
    fn nonstream_connected_error_preserves_status() {
        let harness = Task11HttpHarness::spawn("ns-connected-err");
        let port = harness.port();
        let body = harness.base_body("t15-class-validation", false);
        let (status, raw_body) = raw_nonstream_post(port, &body);
        assert!(
            status >= 400 && status != 200,
            "preterminal error must keep non-200 status, got {status}"
        );
        let text = String::from_utf8_lossy(&raw_body);
        // Strip optional chunked framing / trailing whitespace for JSON parse.
        let json_start = text.find('{').expect("OpenAI error JSON object");
        let json_text = &text[json_start..];
        let err: serde_json::Value =
            serde_json::from_str(json_text.trim_end()).expect("OpenAI error JSON");
        let message = err
            .pointer("/error/message")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            !message.is_empty(),
            "error.message required in OpenAI error body: {err}"
        );
        assert!(
            err.pointer("/choices").is_none(),
            "error response must not look like a completion: {err}"
        );
        // Exactly one top-level JSON value.
        let mut it = serde_json::Deserializer::from_str(json_text.trim_end())
            .into_iter::<serde_json::Value>();
        let _first = it.next().expect("one value").expect("valid JSON");
        assert!(
            it.next().is_none(),
            "error body must be exactly one JSON value"
        );
    }

    /// Connected non-stream success is exactly one JSON value (no SSE/trailer).
    #[cfg(unix)]
    #[test]
    fn nonstream_success_is_exactly_one_json() {
        let harness = Task11HttpHarness::spawn("ns-one-json");
        let port = harness.port();
        let body = harness.base_body("t11-stop-text", false);
        let (status, raw_body) = raw_nonstream_post(port, &body);
        assert_eq!(status, 200, "success status");
        let text = String::from_utf8_lossy(&raw_body);
        let json_start = text.find('{').expect("JSON object in body");
        let json_text = text[json_start..].trim_end();
        let mut it = serde_json::Deserializer::from_str(json_text).into_iter::<serde_json::Value>();
        let value = it
            .next()
            .expect("exactly one JSON value")
            .expect("valid JSON");
        assert!(
            it.next().is_none(),
            "non-stream success must not append a second JSON/SSE value; body={text:?}"
        );
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            value["choices"][0]["message"]["content"],
            "hello from fake daemon"
        );
        assert!(
            !text.contains("data:"),
            "non-stream body must not carry SSE framing"
        );
    }

    /// Close-before terminal → abort and no commit; close-after full JSON → commit, no abort.
    #[cfg(unix)]
    #[test]
    fn nonstream_close_before_vs_after_terminal_commit_race() {
        let harness = Task11HttpHarness::spawn("ns-commit-race");
        let port = harness.port();

        // --- before terminal: disconnect during long silent generation ---
        {
            let body = harness.base_body("t11-long-nonstream", false);
            let stream = open_nonstream_request(port, &body);
            let ready = wait_fixture_ready(&harness, Instant::now() + Duration::from_secs(5));
            let gen_id = ready.get("id").cloned().expect("id");
            let gen_aid = ready
                .get("attempt_id")
                .and_then(|v| v.as_u64())
                .expect("attempt_id");
            drop(stream);

            let deadline = Instant::now() + Duration::from_secs(5);
            let mut saw_abort = false;
            while Instant::now() < deadline {
                let log = harness.read_requests_log();
                saw_abort = log.iter().any(|row| {
                    row.get("type").and_then(|v| v.as_str()) == Some("abort")
                        && row.get("id") == Some(&gen_id)
                        && row.get("attempt_id").and_then(|v| v.as_u64()) == Some(gen_aid)
                });
                let done = log.iter().any(|row| {
                    row.get("type").and_then(|v| v.as_str()) == Some("daemon_done_aborted")
                        && row.get("id") == Some(&gen_id)
                });
                if saw_abort && done && harness.shared.admission.inflight() == 0 {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            assert!(saw_abort, "close-before-terminal must abort");
            let log = harness.read_requests_log();
            assert!(
                Task11HttpHarness::ops_of_type(&log, "commit")
                    .iter()
                    .all(|c| c.get("id") != Some(&gen_id)),
                "close-before-terminal must not commit: {log:?}"
            );
        }

        // --- after terminal: full success response consumed; commit, no abort ---
        {
            let before_aborts =
                Task11HttpHarness::ops_of_type(&harness.read_requests_log(), "abort").len();
            let body = harness.base_body("t11-stop-text", false);
            let (status, raw_body) = raw_nonstream_post(port, &body);
            assert_eq!(status, 200);
            let text = String::from_utf8_lossy(&raw_body);
            let json_start = text.find('{').expect("json");
            let value: serde_json::Value =
                serde_json::from_str(text[json_start..].trim_end()).expect("json body");
            assert_eq!(value["choices"][0]["finish_reason"], "stop");

            let deadline = Instant::now() + Duration::from_secs(3);
            let mut saw_commit = false;
            while Instant::now() < deadline {
                let log = harness.read_requests_log();
                // Latest generate should have a matching commit.
                let generates = Task11HttpHarness::ops_of_type(&log, "generate");
                let last_gen = generates.last().expect("generate for success path");
                let gid = last_gen.get("id").cloned();
                let gaid = last_gen.get("attempt_id").and_then(|v| v.as_u64());
                saw_commit = Task11HttpHarness::ops_of_type(&log, "commit")
                    .iter()
                    .any(|c| {
                        c.get("id") == gid.as_ref()
                            && c.get("attempt_id").and_then(|v| v.as_u64()) == gaid
                    });
                if saw_commit {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
            assert!(saw_commit, "close-after-terminal must commit");
            let after_aborts =
                Task11HttpHarness::ops_of_type(&harness.read_requests_log(), "abort").len();
            assert_eq!(
                after_aborts, before_aborts,
                "close-after full JSON must not emit a new abort"
            );
        }
    }

    /// Paired stream/nonstream matrix rows for stop, pure-tool, mixed-tool,
    /// two-tools, length-withhold, and usage ordering.
    ///
    /// Valid structured-call rows hard-fail on any tool-protocol marker or
    /// structured-argument JSON fragment leaking into content/reasoning.
    /// Invalid-producer dirty-marker diagnostics live in a separate test and
    /// must not weaken these valid-path assertions.
    /// (Matrix body lives with the broader Task11 suite; streaming helpers above
    /// remain the stable surface for parity checks.)
    #[cfg(unix)]

    /// Invalid-producer diagnostic (authority violation): dirty marker token text
    /// stays byte-verbatim content and never becomes structured tool_calls.
    /// Kept separate so it cannot weaken valid structured-call leak assertions.
    #[cfg(unix)]

    /// Premature daemon EOF after gen_start/token without done → client/HTTP failure.
    #[cfg(unix)]

    /// Capability denial: daemon typed error on tools request → no completion/tool payload.
    #[cfg(unix)]
    // --- Task 15: server-owned one-retry (disabled-by-default) ---
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
    fn http_reasoning_nested_max_tokens_alias_resolves_cap_source() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "low",
                "reasoning": { "max_tokens": 2048 }
            }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req["max_think_tokens"], 2048);
        assert_eq!(res.effective_cap, Some(2048));
        assert_eq!(res.cap_source, "explicit:body:reasoning.max_tokens");
        assert!(res.warnings.is_empty());
    }

    #[test]
    fn http_reasoning_top_level_max_think_tokens_precedes_nested_alias() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        let mut conflicting = serde_json::json!({});
        let res_conflict = apply_http_reasoning_request(
            &serde_json::json!({
                "max_think_tokens": 4096,
                "reasoning": { "max_tokens": 2048 }
            }),
            &resolved,
            &mut conflicting,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(conflicting["max_think_tokens"], 4096);
        assert_eq!(res_conflict.effective_cap, Some(4096));
        assert_eq!(res_conflict.cap_source, "explicit:body:max_think_tokens");
        assert!(res_conflict.warnings.iter().any(|warning| {
            warning.contains("reasoning.max_tokens")
                && warning.contains("max_think_tokens")
                && warning.contains("precedence")
        }));

        let mut equal = serde_json::json!({});
        let res_equal = apply_http_reasoning_request(
            &serde_json::json!({
                "max_think_tokens": 2048,
                "reasoning": { "max_tokens": 2048 }
            }),
            &resolved,
            &mut equal,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(equal["max_think_tokens"], 2048);
        assert_eq!(res_equal.effective_cap, Some(2048));
        assert_eq!(res_equal.cap_source, "explicit:body:max_think_tokens");
        assert!(
            res_equal
                .warnings
                .iter()
                .all(|warning| !warning.contains("reasoning.max_tokens")),
            "equal duplicate caps must not warn"
        );
    }

    #[test]
    fn http_reasoning_malformed_nested_max_tokens_is_hard_error() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        for bad in [
            serde_json::json!({ "reasoning": { "max_tokens": -1 } }),
            serde_json::json!({ "reasoning": { "max_tokens": 1.5 } }),
            serde_json::json!({ "reasoning": { "max_tokens": "2048" } }),
            serde_json::json!({ "reasoning": { "max_tokens": 393217 } }),
        ] {
            let mut req = serde_json::json!({});
            let err = apply_http_reasoning_request(
                &bad,
                &resolved,
                &mut req,
                ReasoningContract::QwenJinja,
                true,
                &supported,
            )
            .expect_err("malformed nested reasoning.max_tokens must hard-error");
            let message = format!("{err:#}");
            assert!(
                message.contains("reasoning.max_tokens")
                    && message.contains("must be between 0 and 393216"),
                "unexpected error wording: {message}"
            );
        }
    }

    #[test]
    fn http_reasoning_three_toggle_sources_disabled_wins_once() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        // pairwise: top true vs kwargs false
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "chat_template_kwargs": { "enable_thinking": false }
            }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req["thinking_enabled"], false);
        assert_eq!(res.effective_mode, "disabled");
        let warns: Vec<_> = res
            .warnings
            .iter()
            .filter(|w| w.contains("conflicting thinking toggles"))
            .collect();
        assert_eq!(warns.len(), 1, "pairwise conflict must warn once");

        // pairwise: kwargs true vs thinking.type disabled
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "thinking": { "type": "disabled" }
            }),
            &resolved,
            &mut req2,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req2["thinking_enabled"], false);
        assert_eq!(res2.effective_mode, "disabled");
        assert_eq!(
            res2.warnings
                .iter()
                .filter(|w| w.contains("conflicting thinking toggles"))
                .count(),
            1
        );

        // all three: top true, kwargs true, thinking disabled => disabled wins, single warning
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "chat_template_kwargs": { "enable_thinking": true },
                "thinking": { "type": "disabled" }
            }),
            &resolved,
            &mut req3,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req3["thinking_enabled"], false);
        assert_eq!(res3.effective_mode, "disabled");
        assert_eq!(
            res3.warnings
                .iter()
                .filter(|w| w.contains("conflicting thinking toggles"))
                .count(),
            1
        );

        // all three agree true => no conflict warning, enabled
        let mut req4 = serde_json::json!({});
        let res4 = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "chat_template_kwargs": { "enable_thinking": true },
                "thinking": { "type": "enabled" }
            }),
            &resolved,
            &mut req4,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert_eq!(req4["thinking_enabled"], true);
        assert_eq!(res4.effective_mode, "enabled");
        assert_eq!(
            res4.warnings
                .iter()
                .filter(|w| w.contains("conflicting thinking toggles"))
                .count(),
            0
        );
    }

    #[test]
    fn http_reasoning_gemma_enabled_with_cap_and_budget_dropped() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        // Gemma explicit enable true must remain enabled and report no cap, even with caps present
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "max_think_tokens": 1234,
                "thinking_budget": "high",
                "reasoning_effort": "low"
            }),
            &resolved,
            &mut req,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req["thinking_enabled"], true);
        assert_eq!(res.effective_mode, "enabled");
        assert!(
            req.get("max_think_tokens").is_none(),
            "Gemma must not send cap"
        );
        assert!(res.effective_cap.is_none());
        assert_eq!(res.cap_source, "none");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("reasoning_effort")));
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("max_think_tokens")));
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("thinking_budget")));

        // Config caps also dropped for Gemma when thinking enabled
        let mut layer = ConfigLayer::default();
        layer.set_cli("reasoning.max_tokens", "4096").unwrap();
        layer.set_cli("reasoning.budget", "low").unwrap();
        let cfg_resolved = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "test".into(),
            },
            layer,
        }])
        .unwrap();
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "enable_thinking": true }),
            &cfg_resolved,
            &mut req2,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req2["thinking_enabled"], true);
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("reasoning.max_tokens")));
    }

    #[test]
    fn http_reasoning_gemma_budget_off_does_not_disable() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        // Gemma with enable true + budget off must stay enabled (budget off ignored for Gemma)
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "thinking_budget": "off"
            }),
            &resolved,
            &mut req,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req["thinking_enabled"], true);
        assert_eq!(res.effective_mode, "enabled");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("gemma_boolean") && w.contains("thinking_budget")));
        // Gemma default disabled without explicit enable, budget off should not change that (still disabled via default, not via budget)
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking_budget": "off" }),
            &resolved,
            &mut req2,
            ReasoningContract::GemmaBoolean,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(res2.effective_mode, "disabled");
        // budget off for non-Gemma Qwen should disable
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({
                "enable_thinking": true,
                "thinking_budget": "off"
            }),
            &resolved,
            &mut req3,
            ReasoningContract::QwenJinja,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(req3["thinking_enabled"], false);
        assert_eq!(res3.effective_mode, "disabled");
    }

    #[test]
    fn http_reasoning_invalid_enum_warns_not_hard_error_and_malformed_hard_errors() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let supported = vec!["low".to_string(), "medium".to_string(), "xhigh".to_string()];
        // unknown thinking.type string -> warn+drop, not error, results in default enabled for Qwen
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({ "thinking": { "type": "maybe" } }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap();
        assert!(!res.warnings.is_empty());
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("thinking.type") && w.contains("dropped")));
        assert_eq!(res.effective_mode, "enabled"); // default for Qwen

        // unknown non-native thinking_budget -> warn+drop, not error
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking_budget": "turbo" }),
            &resolved,
            &mut req2,
            ReasoningContract::QwenJinja,
            false,
            &supported,
        )
        .unwrap();
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("thinking_budget") && w.contains("dropped")));

        // wrong JSON type for enable_thinking -> hard error
        let mut req3 = serde_json::json!({});
        let err = apply_http_reasoning_request(
            &serde_json::json!({ "enable_thinking": "true" }),
            &resolved,
            &mut req3,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("enable_thinking must be a boolean"));

        // wrong JSON type for thinking.type -> hard error
        let mut req4 = serde_json::json!({});
        let err2 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking": { "type": 123 } }),
            &resolved,
            &mut req4,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap_err();
        assert!(format!("{err2}").contains("thinking.type must be enabled or disabled"));

        // cap range violation -> hard error (body)
        let mut req5 = serde_json::json!({});
        let err3 = apply_http_reasoning_request(
            &serde_json::json!({ "max_think_tokens": 999999 }),
            &resolved,
            &mut req5,
            ReasoningContract::QwenJinja,
            true,
            &supported,
        )
        .unwrap_err();
        assert!(format!("{err3}").contains("must be between 0 and 393216"));
    }

    #[test]
    fn http_reasoning_nested_max_tokens_and_qwen_deepseek_glimmer_contracts_intact() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        // Nested reasoning.max_tokens still works
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning": { "max_tokens": 2048 } }),
            &resolved,
            &mut req,
            ReasoningContract::QwenJinja,
            true,
            &["low".to_string(), "medium".to_string(), "xhigh".to_string()],
        )
        .unwrap();
        assert_eq!(req["max_think_tokens"], 2048);
        assert_eq!(res.effective_cap, Some(2048));
        assert_eq!(res.cap_source, "explicit:body:reasoning.max_tokens");

        // Qwen non-native still drops effort
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "low" }),
            &resolved,
            &mut req2,
            ReasoningContract::QwenJinja,
            false,
            &[],
        )
        .unwrap();
        assert!(req2.get("reasoning_effort").is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("does not natively support effort")));

        // DeepSeek effort mapping intact
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning_effort": "medium" }),
            &resolved,
            &mut req3,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(req3["reasoning_effort"], "high");
        assert_eq!(res3.effective_effort.as_deref(), Some("high"));

        // Glimmer always-on intact
        let mut req4 = serde_json::json!({});
        let res4 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking": { "type": "disabled" } }),
            &resolved,
            &mut req4,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert_eq!(req4["thinking_enabled"], true);
        assert_eq!(res4.effective_mode, "enabled");
    }

    #[test]
    fn http_reasoning_deepseek_explicit_caps_dropped() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "high",
                "max_think_tokens": 4096
            }),
            &resolved,
            &mut req,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req.get("max_think_tokens").is_none());
        assert!(res.effective_cap.is_none());
        assert_eq!(res.cap_source, "none");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("max_think_tokens") && w.contains("deepseek4")));
        assert_eq!(req["reasoning_effort"], "high");
        assert_eq!(req["thinking_enabled"], true);
        assert_eq!(res.effective_mode, "enabled");
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning": { "max_tokens": 2048 } }),
            &resolved,
            &mut req2,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert!(res2
            .warnings
            .iter()
            .any(|w| w.contains("reasoning.max_tokens")));
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({ "thinking_budget": "high" }),
            &resolved,
            &mut req3,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req3.get("max_think_tokens").is_none());
        assert!(res3
            .warnings
            .iter()
            .any(|w| w.contains("thinking_budget") && w.contains("deepseek4")));
        let mut layer = ConfigLayer::default();
        layer.set_cli("reasoning.max_tokens", "8192").unwrap();
        let cfg = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "test".into(),
            },
            layer,
        }])
        .unwrap();
        let mut req4 = serde_json::json!({});
        let res4 = apply_http_reasoning_request(
            &serde_json::json!({}),
            &cfg,
            &mut req4,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap();
        assert!(req4.get("max_think_tokens").is_none());
        assert!(res4.effective_cap.is_none());
        assert!(res4
            .warnings
            .iter()
            .any(|w| w.contains("reasoning.max_tokens")));
        let mut bad = serde_json::json!({});
        let err = apply_http_reasoning_request(
            &serde_json::json!({ "max_think_tokens": "bad" }),
            &resolved,
            &mut bad,
            ReasoningContract::DeepSeek4,
            true,
            &[],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must be between 0 and 393216"));
    }

    #[test]
    fn http_reasoning_glimmer_explicit_caps_dropped() {
        let resolved = resolve(Vec::<NamedLayer>::new()).unwrap();
        let mut req = serde_json::json!({});
        let res = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning_effort": "low",
                "max_think_tokens": 512
            }),
            &resolved,
            &mut req,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert!(req.get("max_think_tokens").is_none());
        assert!(res.effective_cap.is_none());
        assert_eq!(res.cap_source, "none");
        assert!(res
            .warnings
            .iter()
            .any(|w| w.contains("max_think_tokens") && w.contains("muse_glimmer")));
        assert_eq!(req["reasoning_effort"], "low");
        let mut req2 = serde_json::json!({});
        let res2 = apply_http_reasoning_request(
            &serde_json::json!({
                "reasoning": { "max_tokens": 1024 },
                "thinking_budget": "low"
            }),
            &resolved,
            &mut req2,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert!(req2.get("max_think_tokens").is_none());
        assert!(res2.effective_cap.is_none());
        assert_eq!(res2.cap_source, "none");
        assert!(res2.warnings.iter().any(|w| w.contains("muse_glimmer")));
        let mut layer = ConfigLayer::default();
        layer.set_cli("reasoning.budget", "high").unwrap();
        let cfg = resolve([NamedLayer {
            source: ConfigSource::OneShot {
                argument: "test".into(),
            },
            layer,
        }])
        .unwrap();
        let mut req3 = serde_json::json!({});
        let res3 = apply_http_reasoning_request(
            &serde_json::json!({}),
            &cfg,
            &mut req3,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap();
        assert!(req3.get("max_think_tokens").is_none());
        assert!(res3.effective_cap.is_none());
        assert!(res3
            .warnings
            .iter()
            .any(|w| w.contains("muse_glimmer") || w.contains("thinking_budget")));
        let mut bad = serde_json::json!({});
        let err = apply_http_reasoning_request(
            &serde_json::json!({ "reasoning": { "max_tokens": 500000 } }),
            &resolved,
            &mut bad,
            ReasoningContract::MuseGlimmer,
            true,
            &[],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("must be between 0 and 393216"));
    }

    #[test]
    fn head_forces_local_even_when_service_would_be_ready() {
        let with_head = RunArgs {
            model: "maple-preview".into(),
            prompt: vec![],
            temp: None,
            top_p: None,
            repeat_penalty: None,
            max_tokens: None,
            kv_mode: None,
            head: Some("q4k".into()),
            kv_backend: None,
            speculation: None,
            model_draft: None,
            draft_max: None,
            dspark_conf_threshold: None,
            system: None,
            image: None,
            json: false,
            no_stream: false,
        };
        let without_head = RunArgs {
            head: None,
            ..with_head.clone()
        };
        // --head must force local; without head should not force local by itself
        assert!(
            run_should_force_local(&with_head),
            "--head must force local load path"
        );
        assert!(
            !run_should_force_local(&without_head),
            "without head and no other flags should not force local"
        );
        // Verify the helper is used by run_command: an HTTP service would be bypassed.
        // No network needed; the flag alone is the contract.
    }

    #[test]
    fn existing_artifact_valid_detects_fresh_and_stale() {
        use sha2::{Digest, Sha256};
        let dir = env::temp_dir().join(format!(
            "hipfire-artifact-valid-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.mq4");
        let content = b"fresh content";
        fs::write(&path, content).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(content);
        let sha = format!("{:x}", hasher.finalize());
        let size = content.len() as u64;
        assert!(existing_artifact_valid(&path, Some(&sha), Some(size)));
        assert!(!existing_artifact_valid(&path, Some(&sha), Some(size + 1)));
        assert!(!existing_artifact_valid(
            &path,
            Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
            Some(size)
        ));
        // No expectations means existence alone is valid
        assert!(existing_artifact_valid(&path, None, None));
        // Missing file is invalid
        assert!(!existing_artifact_valid(
            &dir.join("missing"),
            Some(&sha),
            Some(size)
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_params_preserves_auto_for_direct_path_and_registry() {
        // Direct-path load: no registry entry, config is auto -> must stay auto.
        let defaults = resolve(Vec::<NamedLayer>::new()).unwrap();
        assert_eq!(config_string(&defaults, "memory.kv_cache").unwrap(), "auto");
        let direct_path = PathBuf::from("/tmp/direct-model.mq4");
        let params = load_params(&defaults, None, &direct_path, 64, None, None, None).unwrap();
        assert_eq!(
            params["kv_mode"], "auto",
            "direct-path auto must survive to architecture"
        );
        // Registry path with default_kv_mode=bf16 must also preserve auto when
        // no explicit --kv-mode is given; architecture picks BF16.
        let raw = r#"{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{
                "maple-preview":{"repo":"x","file":"maple-preview.mq2lloydu","size_gb":1,"min_vram_gb":1,"desc":"x","default_kv_mode":"bf16"}
            },
            "aliases":{}
        }"#;
        let registry = RegistryV1::parse(raw, "test").unwrap();
        let (_, entry) = registry.model("maple-preview").unwrap();
        let params2 =
            load_params(&defaults, Some(entry), &direct_path, 64, None, None, None).unwrap();
        assert_eq!(
            params2["kv_mode"], "auto",
            "registry auto must survive even when entry has bf16 default"
        );
        // Explicit override still wins
        let params3 = load_params(
            &defaults,
            Some(entry),
            &direct_path,
            64,
            Some("q8"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(params3["kv_mode"], "q8");
    }

    fn write_test_registry_cache(paths: &Paths, raw: &str) {
        let registry = RegistryV1::parse(raw, "test-cache").unwrap();
        let url = env::var("HIPFIRE_REGISTRY_URL")
            .unwrap_or_else(|_| "https://example.com/test.json".into());
        let cache = serde_json::json!({
            "fetched_at": unix_timestamp() * 1000,
            "url": url,
            "registry": registry
        });
        fs::create_dir_all(paths.registry.cache.parent().unwrap()).unwrap();
        fs::write(
            &paths.registry.cache,
            serde_json::to_string(&cache).unwrap(),
        )
        .unwrap();
    }

    fn tiny_http_server(
        files: std::collections::HashMap<String, Vec<u8>>,
    ) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://127.0.0.1:{}", addr.port());
        let expected = files.len();
        let handle = std::thread::spawn(move || {
            let mut served = 0usize;
            listener.set_nonblocking(false).unwrap();
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut buf = [0u8; 4096];
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                // Extract file name from request path: /{repo}/resolve/main/{file}
                let mut body: Option<Vec<u8>> = None;
                for (name, data) in &files {
                    if req.contains(name) {
                        body = Some(data.clone());
                        break;
                    }
                }
                if let Some(data) = body {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        data.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&data);
                } else {
                    let header =
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(header.as_bytes());
                }
                let _ = stream.flush();
                served += 1;
                if served >= expected {
                    break;
                }
            }
        });
        // small pause to let listener start
        std::thread::sleep(Duration::from_millis(50));
        (base, handle)
    }

    #[test]
    fn pull_fresh_downloads_heads_with_hash_verification() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("pull-fresh-heads");
        // Build tiny artifacts
        let base_content = b"base-model-content";
        let head_q4k_content = b"head-q4k-content";
        let head_bf16_content = b"head-bf16-content";
        use sha2::{Digest, Sha256};
        let sha = |data: &[u8]| {
            let mut h = Sha256::new();
            h.update(data);
            format!("{:x}", h.finalize())
        };
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "test-model":{{
                    "repo":"test/repo",
                    "file":"test-model.mq4",
                    "size_gb":0.001,
                    "min_vram_gb":1,
                    "desc":"x",
                    "default_kv_mode":"bf16",
                    "heads":{{
                        "q4k":{{"file":"test-model-head-q4k.hfq","sha256":"{}","size_bytes":{}}},
                        "bf16":{{"file":"test-model-head-bf16.hfq","sha256":"{}","size_bytes":{}}}
                    }},
                    "sha256":"{}",
                    "size_bytes":{}
                }}
            }},
            "aliases":{{}}
        }}"#,
            sha(head_q4k_content),
            head_q4k_content.len(),
            sha(head_bf16_content),
            head_bf16_content.len(),
            sha(base_content),
            base_content.len()
        );
        write_test_registry_cache(&paths, &raw);
        let mut files = std::collections::HashMap::new();
        files.insert("test-model.mq4".to_string(), base_content.to_vec());
        files.insert(
            "test-model-head-q4k.hfq".to_string(),
            head_q4k_content.to_vec(),
        );
        files.insert(
            "test-model-head-bf16.hfq".to_string(),
            head_bf16_content.to_vec(),
        );
        let (base, handle) = tiny_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        // Pull should download base + both heads
        pull_command(
            &paths,
            PullArgs {
                model: "test-model".into(),
                force: false,
            },
        )
        .unwrap();
        assert_eq!(
            fs::read(paths.models.join("test-model.mq4")).unwrap(),
            base_content
        );
        assert_eq!(
            fs::read(paths.models.join("test-model-head-q4k.hfq")).unwrap(),
            head_q4k_content
        );
        assert_eq!(
            fs::read(paths.models.join("test-model-head-bf16.hfq")).unwrap(),
            head_bf16_content
        );
        let _ = handle.join();
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn pull_stale_same_name_artifact_refreshes_atomically() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("pull-stale-refresh");
        let fresh = b"fresh-content";
        let stale = b"stale-old-content";
        use sha2::{Digest, Sha256};
        let sha = |data: &[u8]| {
            let mut h = Sha256::new();
            h.update(data);
            format!("{:x}", h.finalize())
        };
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "test-model":{{
                    "repo":"test/repo",
                    "file":"test-model.mq4",
                    "size_gb":0.001,
                    "min_vram_gb":1,
                    "desc":"x",
                    "sha256":"{}",
                    "size_bytes":{}
                }}
            }},
            "aliases":{{}}
        }}"#,
            sha(fresh),
            fresh.len()
        );
        write_test_registry_cache(&paths, &raw);
        fs::create_dir_all(&paths.models).unwrap();
        // Place stale artifact with same name but wrong hash/size
        fs::write(paths.models.join("test-model.mq4"), stale).unwrap();
        assert!(!existing_artifact_valid(
            &paths.models.join("test-model.mq4"),
            Some(&sha(fresh)),
            Some(fresh.len() as u64)
        ));
        let mut files = std::collections::HashMap::new();
        files.insert("test-model.mq4".to_string(), fresh.to_vec());
        let (base, handle) = tiny_http_server(files);
        let _hf = EnvGuard::set("HIPFIRE_HF_BASE", &base);
        // Without --force, stale should still be detected and refreshed
        pull_command(
            &paths,
            PullArgs {
                model: "test-model".into(),
                force: false,
            },
        )
        .unwrap();
        assert_eq!(
            fs::read(paths.models.join("test-model.mq4")).unwrap(),
            fresh
        );
        let _ = handle.join();
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn rm_removes_heads_alongside_base() {
        let _env_lock = TEST_ENV_LOCK.lock().unwrap();
        let _reg = EnvGuard::set("HIPFIRE_REGISTRY_URL", "https://example.com/test.json");
        let paths = test_paths("rm-heads");
        let valid_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let raw = format!(
            r#"{{
            "schema_version":1,
            "generated_at":"2099-01-01T00:00:00Z",
            "models":{{
                "test-model":{{
                    "repo":"test/repo",
                    "file":"test-model.mq4",
                    "size_gb":0.001,
                    "min_vram_gb":1,
                    "desc":"x",
                    "heads":{{
                        "q4k":{{"file":"test-model-head-q4k.hfq","sha256":"{sha}","size_bytes":3}},
                        "bf16":{{"file":"test-model-head-bf16.hfq","sha256":"{sha}","size_bytes":3}}
                    }},
                    "sha256":"{sha}",
                    "size_bytes":3
                }}
            }},
            "aliases":{{}}
        }}"#,
            sha = valid_sha
        );
        write_test_registry_cache(&paths, &raw);
        fs::create_dir_all(&paths.models).unwrap();
        fs::write(paths.models.join("test-model.mq4"), b"base").unwrap();
        fs::write(paths.models.join("test-model-head-q4k.hfq"), b"q4k").unwrap();
        fs::write(paths.models.join("test-model-head-bf16.hfq"), b"bf16").unwrap();
        assert!(paths.models.join("test-model-head-q4k.hfq").is_file());
        rm_command(
            &paths,
            RmArgs {
                model: "test-model".into(),
                yes: true,
            },
        )
        .unwrap();
        assert!(!paths.models.join("test-model.mq4").exists());
        assert!(!paths.models.join("test-model-head-q4k.hfq").exists());
        assert!(!paths.models.join("test-model-head-bf16.hfq").exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvGuard {
        key: String,
        prev: Option<std::ffi::OsString>,
    }
    impl EnvGuard {
        fn set(key: &str, val: &str) -> Self {
            let prev = env::var_os(key);
            env::set_var(key, val);
            Self {
                key: key.to_string(),
                prev,
            }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(v) = &self.prev {
                env::set_var(&self.key, v);
            } else {
                env::remove_var(&self.key);
            }
        }
    }
}
