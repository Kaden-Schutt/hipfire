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
    /// Reshuffle a canonical .hfq into an arch-optimal layout (<model>.<arch>.hfq)
    Repack(commands::forward::RepackArgs),
    /// GPU resource lock for multi-agent coordination (acquire/release/status)
    #[command(alias = "gpu-lock")]
    Lock(commands::lock::LockArgs),
    /// Import and inspect diffusion models stored as .hfq artifacts
    ///
    /// Runtime note: runnable `.hfq` diffusion artifacts still perform CLIP
    /// tokenization as host-side setup. `txt2img`, `img2img`, and `smoke` can
    /// opt into `--rocm-device-id` to route currently GPU-backed generation
    /// boundaries through ROCm.
    ///
    /// `hipfire serve` exposes the same hybrid path through the Stable
    /// Diffusion API extension fields `rocm_device_id` or
    /// `hipfire_rocm_device_id` on `/sdapi/v1/txt2img` and
    /// `/sdapi/v1/img2img` requests, through the same keys in
    /// `override_settings`, or through the persisted `/sdapi/v1/options` value
    /// `hipfire_rocm_device_id`.
    ///
    /// `/sdapi/v1/progress` tracks active SDAPI sampling steps and returns the
    /// final generated PNG in `current_image` after a successful HFQ diffusion
    /// request completes. Live per-step latent preview decoding is not
    /// implemented yet.
    ///
    /// SDAPI img2img and inpaint resize init and mask images to the requested
    /// output dimensions before VAE encoding. `resize_mode` supports WebUI
    /// modes 0 (stretch), 1 (crop and resize), and 2 (resize and fill);
    /// mode 3 latent upscale is rejected unless no resize is needed. Masked
    /// img2img also honors WebUI's `inpainting_mask_invert`, `mask_blur`,
    /// `mask_blur_x`, `mask_blur_y`, and `mask_round` options. Txt2img
    /// high-res generation is implemented as a batched first-pass txt2img
    /// generation followed by a second-pass img2img generation at the high-res
    /// target dimensions. SDAPI high-res requests accept `enable_hr`,
    /// `firstphase_width`,
    /// `firstphase_height`, `hr_scale`, `hr_upscaler`, `hr_resize_x`,
    /// `hr_resize_y`, `hr_second_pass_steps`, `hr_checkpoint_name`,
    /// `hr_prompt`, `hr_negative_prompt`, `hr_sampler_name`, and
    /// `hr_scheduler`.
    ///
    /// The runtime accepts Q4F16_G64, f16, bf16, f32, Q8F16, Q4_K,
    /// HFQ4G128, HFQ4G256, and HFQ6G256 tensor payloads. Other packed payloads
    /// require a matching diffusion dequantizer/runtime implementation.
    Diffusion(commands::diffusion::DiffusionArgs),
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
    /// Regenerate the model-support matrix artifacts (the generated tables in
    /// crates/hipfire-model + the chart in MODEL-SUPPORT.md) from
    /// docs/model-support.toml. Hidden: a maintenance command; run via
    /// `cargo run -p hipfire-cli -- gen-model-support`.
    #[command(hide = true)]
    GenModelSupport(commands::gen_model_support::GenModelSupportArgs),
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
        Command::Chat(args) => commands::chat::run(args, loaded_config).await,
        Command::List => {
            commands::list::run();
            Ok(())
        }
        Command::Eval(args) => commands::forward::run_eval(args),
        Command::HostProfile(args) => commands::forward::run_host_profile(args),
        Command::CollectArtifacts(args) => commands::forward::run_collect_artifacts(args),
        Command::Repack(args) => commands::forward::run_repack(args),
        Command::Lock(args) => commands::lock::run(args),
        Command::Diffusion(args) => commands::diffusion::run(args),
        Command::Admin(args) => commands::admin::run(args, config).await,
        Command::GenDocs(args) => commands::gen_docs::run(args),
        Command::GenConfigSchema(args) => commands::gen_config_schema::run(args),
        Command::GenEnvDocs(args) => commands::gen_env_docs::run(args),
        Command::GenModelSupport(args) => commands::gen_model_support::run(args),
    }
}
