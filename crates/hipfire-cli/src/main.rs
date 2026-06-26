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
    /// `hipfire_rocm_device_id`. Persisted `/sdapi/v1/options` values for
    /// `send_images`, `save_images`, `outdir_samples`, `outdir_txt2img_samples`,
    /// and `outdir_img2img_samples` act as generation defaults unless the
    /// request or `override_settings` supplies a more specific value.
    ///
    /// `/sdapi/v1/progress` tracks active SDAPI sampling steps and updates
    /// `current_image` with live PNG previews decoded from intermediate
    /// latents, then leaves the final generated PNG there after a successful
    /// HFQ diffusion request completes; WebUI's `skip_current_image=true`
    /// progress query suppresses only that response's preview payload.
    /// `/sdapi/v1/memory` returns WebUI-shaped host RAM stats and marks CUDA
    /// memory stats unavailable because Hipfire uses HIP/ROCm. WebUI's
    /// create/train embedding and hypernetwork endpoints are registered for
    /// client compatibility and return an `info` response explaining that
    /// native training is not implemented by the SDAPI layer.
    ///
    /// SDAPI sampler fields follow WebUI's split controls: full scheduler names
    /// such as `DDIM`, `DPM++ 2M`, and `DPM++ 3M` are accepted directly, while schedule
    /// modifiers such as `Automatic` and `Karras` combine with `sampler_name`
    /// or `sampler_index` (for example `Euler` + `Karras` becomes
    /// `Euler Karras`).
    ///
    /// SDAPI img2img and inpaint support WebUI resize modes 0 (stretch), 1
    /// (crop and resize), 2 (resize and fill), and 3 (latent upscale). Modes
    /// 0-2 resize init and mask images before VAE encoding; mode 3 keeps the
    /// init image at its source dimensions, VAE-encodes it, then resizes the
    /// latent tensor to the requested output shape;
    /// `/sdapi/v1/latent-upscale-modes` advertises Hipfire's nearest-neighbor
    /// latent resize aliases. Hipfire also accepts common WebUI generation
    /// fields such as `styles`, `restore_faces`, `tiling`, `do_not_save_samples`,
    /// `do_not_save_grid`, `seed_resize_from_w`, `seed_resize_from_h`, `eta`,
    /// `s_churn`, `s_tmin`, `s_tmax`, `s_noise`,
    /// `override_settings_restore_afterwards`, `disable_extra_networks`, and
    /// `comments`; fields that do not affect the native runtime are returned in
    /// response `parameters` and listed in `info.ignored_fields` when active.
    /// `do_not_save_samples` suppresses disk writes even when `save_images` is
    /// true. Masked img2img also honors
    /// WebUI's `inpainting_mask_invert`, `mask_blur`,
    /// `mask_blur_x`, `mask_blur_y`, `mask_round`, and `inpainting_fill`
    /// options; default fill (0) is applied in image space before VAE encode,
    /// original (1) leaves init pixels unchanged, and latent noise (2) /
    /// latent nothing (3) additionally alter masked latents. WebUI's
    /// `inpaint_full_res` and
    /// `inpaint_full_res_padding` crop masked regions for processing and
    /// composite the generated crop back onto the init image. SDAPI requests
    /// can also import common WebUI `infotext` fields when those fields are not
    /// explicitly set in JSON. Non-empty `script_name` and `script_args`
    /// payloads are rejected because Hipfire exposes no SDAPI selectable
    /// scripts. `alwayson_scripts` accepts empty or disabled default extension
    /// payloads, but active script payloads are rejected. Txt2img high-res
    /// generation is implemented as a batched first-pass txt2img
    /// generation followed by a second-pass img2img generation at the high-res
    /// target dimensions. SDAPI high-res requests accept `enable_hr`,
    /// `firstphase_width`,
    /// `firstphase_height`, `hr_scale`, `hr_upscaler`, `hr_resize_x`,
    /// `hr_resize_y`, `hr_second_pass_steps`, `hr_checkpoint_name`,
    /// `hr_prompt`, `hr_negative_prompt`, `hr_sampler_name`, and
    /// `hr_scheduler`; `hr_checkpoint_name` may point to another resolvable
    /// diffusion HFQ artifact for the second pass.
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
