// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::Engine;
use clap::{Args, Subcommand, ValueEnum};
use hipfire_diffusion::DiffusionHipRuntimeOptions;
// GGUF-style split: the diffusers/checkpoint importer (pickle + zip parsing)
// now lives in the offline hipfire-diffusion-coexist crate, out of the
// server-linked hipfire-diffusion.
use hipfire_diffusion::{
    calibrate_diffusion_hfq, inspect_hfq_with_runtime_support, quantize_diffusion_hfq,
    resize_rgb_batch_to_cover_nearest, DiffusionBatchRequest, DiffusionError,
    DiffusionGenerationRuntimeOptions, DiffusionHfqInspection, DiffusionImg2ImgRequest,
    DiffusionPipeline, DiffusionProgress, DiffusionPrompt, DiffusionQuantFormat, DiffusionResult,
    RefineSigmaSchedule, RgbImageBatch,
};
use hipfire_diffusion_coexist::{import_diffusers_to_hfq, DiffusersImportOptions};
use serde::Serialize;

use crate::model::find_model;

/// Build a denoise progress callback that logs per-step wall-clock timing to
/// stderr (step index, per-step delta, cumulative elapsed, throughput, and ETA)
/// so batch generation speed is observable in real time. `images` is the batch
/// size, reported so the per-step rate can be read as images/step-second.
fn step_timing_progress(
    label: &str,
    images: usize,
) -> impl FnMut(DiffusionProgress) -> DiffusionResult<()> {
    let label = label.to_string();
    // Start the clock now (before generation) so the first reported step includes
    // model setup and text-encode latency rather than reading as zero.
    let start = Instant::now();
    let last = Cell::new(None::<Instant>);
    move |progress: DiffusionProgress| {
        let now = Instant::now();
        let step_dt = last.get().map_or_else(
            || now.duration_since(start).as_secs_f64(),
            |t| now.duration_since(t).as_secs_f64(),
        );
        last.set(Some(now));
        let elapsed = now.duration_since(start).as_secs_f64();
        let done = progress.completed_steps;
        let total = progress.total_steps.max(1);
        let rate = if elapsed > 0.0 {
            done as f64 / elapsed
        } else {
            0.0
        };
        let eta = if rate > 0.0 {
            (total.saturating_sub(done)) as f64 / rate
        } else {
            0.0
        };
        eprintln!(
            "[{label}] step {done}/{total} (batch {images}) +{step_dt:.2}s elapsed {elapsed:.1}s {rate:.3} steps/s ETA {eta:.0}s",
        );
        Ok(())
    }
}

#[derive(Debug, Args)]
pub struct DiffusionArgs {
    #[command(subcommand)]
    pub command: DiffusionCommand,
}

#[derive(Debug, Subcommand)]
pub enum DiffusionCommand {
    /// Convert a Diffusers snapshot or single-file checkpoint into a Hipfire .hfq artifact.
    ///
    /// The importer extracts tensors from common Diffusers single-file and
    /// sharded safetensors layouts first, then falls back to legacy PyTorch .bin
    /// archives or opaque source weight entries when a component cannot be
    /// indexed yet.
    Import(DiffusionImportArgs),
    /// Inspect a diffusion .hfq artifact and print its server-facing summary
    Inspect(DiffusionInspectArgs),
    /// Plan HIP diffusion buffers and optionally run a ROCm device preflight
    ///
    /// The preflight command prints a deterministic memory plan for the
    /// requested resolution, batch, scheduler, and prompt set. When a
    /// `--device-id` is given it also initializes the selected HIP device,
    /// allocates the planned buffer classes, runs a host/device roundtrip
    /// probe, and launches the diffusion kernel probes against CPU references.
    Preflight(DiffusionPreflightArgs),
    /// Generate PNG images directly from a diffusion .hfq artifact
    ///
    /// With `--enable-hr`, the command first generates the requested base
    /// batch, decodes those PNGs as init images, then runs an img2img second
    /// pass at `--hr-scale` or the `--hr-resize-x`/`--hr-resize-y` target.
    #[command(name = "txt2img", alias = "txt2-img")]
    Txt2Img(DiffusionTxt2ImgArgs),
    /// Generate PNG images from init images with a diffusion .hfq artifact
    #[command(name = "img2img", alias = "img2-img")]
    Img2Img(DiffusionImg2ImgArgs),
    /// Run an end-to-end diffusion admission smoke and validate output PNGs
    Smoke(DiffusionSmokeArgs),
    /// Re-encode the weight tensors of a source .hfq into a packed quant format
    ///
    /// Reads an existing diffusion .hfq (weights stored as f32/f16/bf16 source),
    /// re-encodes the large 2D+ `.weight` tensors into the requested format, and
    /// copies every other entry (biases, norms, configs, tokenizers) verbatim.
    /// Decoding is per-tensor by quant_type, so the output loads unchanged.
    Quantize(DiffusionQuantizeArgs),
    /// Run an activation-calibration pass and write a .calib.hfq sidecar
    ///
    /// Generates a few CPU-reference denoise steps over sample prompts, capturing
    /// per-weight activation statistics (imatrix + per-linear Hessian). The
    /// resulting .calib.hfq feeds `quantize --format oq4++ --calib`.
    Calibrate(DiffusionCalibrateArgs),
}

#[derive(Debug, Args)]
pub struct DiffusionCalibrateArgs {
    /// Source diffusion .hfq artifact to calibrate
    pub model: PathBuf,
    /// Output .calib.hfq sidecar path
    #[arg(long, short)]
    pub output: PathBuf,
    /// Calibration prompts (repeatable); defaults to a small built-in set
    #[arg(long = "prompt", short)]
    pub prompts: Vec<String>,
    /// Denoise steps per prompt
    #[arg(long, default_value_t = 4)]
    pub steps: u32,
    #[arg(long, default_value_t = 256)]
    pub width: u32,
    #[arg(long, default_value_t = 256)]
    pub height: u32,
    /// CFG scale (>1 captures both conditional and unconditional activations)
    #[arg(long, default_value_t = 7.5)]
    pub cfg_scale: f32,
    /// Max linear input dim K to capture a full [K,K] Hessian for (else imatrix only)
    #[arg(long, default_value_t = 2048)]
    pub hessian_max_k: usize,
}

#[derive(Debug, Args)]
pub struct DiffusionQuantizeArgs {
    /// Source diffusion .hfq artifact (typically `weight_format: source`)
    pub source: PathBuf,
    /// Output quantized .hfq artifact path
    #[arg(long, short)]
    pub output: PathBuf,
    /// Quant format: q8, q4, q4k, q4+ (data-free) or oq4/oq4++/oq8 (Opus, calibrated)
    #[arg(long, default_value = "q8")]
    pub format: String,
    /// Optional .calib.hfq sidecar (from `diffusion calibrate`); enables oq4++ LDLQ
    #[arg(long)]
    pub calib: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct DiffusionImportArgs {
    /// Diffusers snapshot directory containing model_index.json, or a .safetensors/.ckpt checkpoint
    pub source: PathBuf,
    /// Output .hfq artifact path
    #[arg(long, short)]
    pub output: PathBuf,
    /// Model name to store in the diffusion metadata; defaults to the source directory name
    #[arg(long)]
    pub model_name: Option<String>,
    /// Maximum batch size declared by the artifact. Runtime kernels may cap this lower initially.
    #[arg(long, default_value_t = 1)]
    pub max_batch: u32,
    /// Import configs/tokenizers only and skip weight indexing for fast planning/inspection.
    #[arg(long)]
    pub metadata_only: bool,
}

#[derive(Debug, Args)]
pub struct DiffusionInspectArgs {
    /// Diffusion .hfq artifact to inspect by name, shorthand, alias, or path
    pub model: PathBuf,
}

#[derive(Debug, Args)]
pub struct DiffusionPreflightArgs {
    /// Diffusion .hfq artifact to inspect by name, shorthand, alias, or path
    #[arg(long, short)]
    pub model: PathBuf,
    /// Prompt text. Repeat for batched planning, or use --batch-size with one prompt.
    #[arg(long, short, default_value = "hipfire diffusion preflight")]
    pub prompt: Vec<String>,
    /// Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt.
    #[arg(long)]
    pub negative_prompt: Vec<String>,
    /// Output image width in pixels
    #[arg(long, default_value_t = 512)]
    pub width: u32,
    /// Output image height in pixels
    #[arg(long, default_value_t = 512)]
    pub height: u32,
    /// Denoising steps
    #[arg(long, default_value_t = 20)]
    pub steps: u32,
    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 7.0)]
    pub cfg_scale: f32,
    /// Guidance-distilled model scale, separate from classifier-free guidance
    #[arg(long)]
    pub distilled_guidance_scale: Option<f32>,
    /// Scheduler/sampler name
    #[arg(long, default_value = "Automatic")]
    pub scheduler: String,
    /// Seed. Omit for zero, pass once to reuse, or repeat per prompt.
    #[arg(long)]
    pub seed: Vec<i64>,
    /// Optional subseed. Pass once to reuse or repeat per prompt.
    #[arg(long)]
    pub subseed: Vec<i64>,
    /// Blend strength for subseed latents
    #[arg(long, default_value_t = 0.0)]
    pub subseed_strength: f32,
    /// Batch size when a single prompt is supplied
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,
    /// ROCm device id to initialize and preflight (omit for plan-only)
    #[arg(long, default_value_t = 0)]
    pub device_id: i32,
}

#[derive(Debug, Args)]
pub struct DiffusionTxt2ImgArgs {
    /// Diffusion .hfq artifact to run by name, shorthand, alias, or path
    #[arg(long, short)]
    pub model: PathBuf,
    /// Prompt text. Repeat for batched generation, or use --batch-size with one prompt.
    #[arg(long, short, required = true)]
    pub prompt: Vec<String>,
    /// Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt.
    #[arg(long)]
    pub negative_prompt: Vec<String>,
    /// Output PNG file for one image, or output directory for batches
    #[arg(long, short)]
    pub output: PathBuf,
    /// Directory to write a per-step preview PNG (step_00.png, step_01.png, ...)
    /// by decoding the intermediate latent after each denoise pass. Useful for a
    /// webui progress strip; adds one VAE decode per step. Single-image runs only.
    #[arg(long)]
    pub preview_dir: Option<PathBuf>,
    /// Output image width in pixels
    #[arg(long, default_value_t = 512)]
    pub width: u32,
    /// Output image height in pixels
    #[arg(long, default_value_t = 512)]
    pub height: u32,
    /// First-pass high-res width before upscale; preserves --width/--height aspect when used alone
    #[arg(long)]
    pub firstphase_width: Option<u32>,
    /// First-pass high-res height before upscale; preserves --width/--height aspect when used alone
    #[arg(long)]
    pub firstphase_height: Option<u32>,
    /// Denoising steps
    #[arg(long, default_value_t = 20)]
    pub steps: u32,
    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 7.0)]
    pub cfg_scale: f32,
    /// Guidance-distilled model scale, separate from classifier-free guidance
    #[arg(long)]
    pub distilled_guidance_scale: Option<f32>,
    /// Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, DPM++ 2M Karras, or DPM++ 3M Karras
    #[arg(long, default_value = "Automatic")]
    pub scheduler: String,
    /// Seed. Omit for zero, pass once to reuse, or repeat per prompt.
    #[arg(long)]
    pub seed: Vec<i64>,
    /// Optional subseed. Pass once to reuse or repeat per prompt.
    #[arg(long)]
    pub subseed: Vec<i64>,
    /// Blend strength for subseed latents
    #[arg(long, default_value_t = 0.0)]
    pub subseed_strength: f32,
    /// Batch size when a single prompt is supplied
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,
    /// Run a high-res second pass by feeding first-pass txt2img results through img2img
    #[arg(long)]
    pub enable_hr: bool,
    /// High-res scale when --hr-resize-x/--hr-resize-y are both omitted or zero
    #[arg(long, default_value_t = 2.0)]
    pub hr_scale: f64,
    /// Exact high-res target width, or aspect-preserving width when used alone
    #[arg(long)]
    pub hr_resize_x: Option<u32>,
    /// Exact high-res target height, or aspect-preserving height when used alone
    #[arg(long)]
    pub hr_resize_y: Option<u32>,
    /// Denoising steps for the high-res second pass; defaults to --steps
    #[arg(long)]
    pub hr_second_pass_steps: Option<u32>,
    /// Img2img denoising strength for the high-res second pass
    #[arg(long, default_value_t = 0.75)]
    pub hr_denoising_strength: f32,
    /// Use ROCm for currently GPU-routed generation stages on this device id
    /// ROCm device to generate on. Omit to auto-detect (a single GPU is used
    /// silently; the first of several with a warning). The CPU reference oracle
    /// is opt-in via the HIPFIRE_DIFFUSION_CPU_REFERENCE environment variable.
    #[arg(long)]
    pub rocm_device_id: Option<i32>,
    /// Enable MrFlow staged sampling: a fast low-resolution pass, pixel-space
    /// super-resolution, re-encode, and a short direct-sigma refine. --width and
    /// --height are the final resolution; the low-res pass runs at those divided
    /// by the upscale factor. Flow-match backbones only (FLUX / Qwen-Image /
    /// Z-Image / Krea-2). Overrides --enable-hr.
    #[arg(long, value_enum)]
    pub mrflow: Option<MrFlowPreset>,
    /// Override the MrFlow refine start sigma (preset default). Larger values
    /// (0.16-0.20) can improve text-heavy generations.
    #[arg(long)]
    pub mrflow_refine_sigma: Option<f32>,
    /// Override the MrFlow pixel-space upscale factor (preset default 2.0).
    #[arg(long)]
    pub mrflow_upscale: Option<f64>,
    /// Use the flow-match shifted interior refine schedule (only affects refine
    /// passes with more than one step).
    #[arg(long)]
    pub mrflow_shifted: bool,
    /// RealESRGAN RRDBNet super-resolution .hfq (from `hipfire-coexistence`) for
    /// the MrFlow Stage-2 upscale. Without it, Stage 2 falls back to a plain
    /// cover-resize (much softer output).
    #[arg(long)]
    pub mrflow_sr: Option<PathBuf>,
}

/// MrFlow staged-sampling presets. The numbers follow the reference MrFlow /
/// Rebels ports: `stageN + 1` denotes N low-resolution steps and one
/// high-resolution direct-sigma refine step.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MrFlowPreset {
    /// Z-Image Turbo, 9 low-res + 1 refine, sigma 0.11, no CFG (paper demo).
    #[value(name = "zit-9plus1")]
    Zit9Plus1,
    /// Krea-2 base, 12 low-res + 1 refine, sigma 0.12, cfg 4.0.
    #[value(name = "krea2-12plus1")]
    Krea2Base12Plus1,
    /// Krea-2 base, 20 low-res + 1 refine, sigma 0.15, cfg 4.0.
    #[value(name = "krea2-20plus1")]
    Krea2Base20Plus1,
    /// Krea-2 Turbo, 8 low-res + 1 refine, sigma 0.11, no CFG.
    #[value(name = "krea2-turbo-8plus1")]
    Krea2Turbo8Plus1,
}

struct MrFlowPresetParams {
    stage1_steps: u32,
    refine_steps: u32,
    refine_sigma: f32,
    cfg_scale: f32,
    upscale_factor: f64,
}

impl MrFlowPreset {
    fn params(self) -> MrFlowPresetParams {
        match self {
            MrFlowPreset::Zit9Plus1 => MrFlowPresetParams {
                stage1_steps: 9,
                refine_steps: 1,
                refine_sigma: 0.11,
                cfg_scale: 1.0,
                upscale_factor: 2.0,
            },
            MrFlowPreset::Krea2Base12Plus1 => MrFlowPresetParams {
                stage1_steps: 12,
                refine_steps: 1,
                refine_sigma: 0.12,
                cfg_scale: 4.0,
                upscale_factor: 2.0,
            },
            MrFlowPreset::Krea2Base20Plus1 => MrFlowPresetParams {
                stage1_steps: 20,
                refine_steps: 1,
                refine_sigma: 0.15,
                cfg_scale: 4.0,
                upscale_factor: 2.0,
            },
            MrFlowPreset::Krea2Turbo8Plus1 => MrFlowPresetParams {
                stage1_steps: 8,
                refine_steps: 1,
                refine_sigma: 0.11,
                cfg_scale: 1.0,
                upscale_factor: 2.0,
            },
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            MrFlowPreset::Zit9Plus1 => "zit-9plus1",
            MrFlowPreset::Krea2Base12Plus1 => "krea2-12plus1",
            MrFlowPreset::Krea2Base20Plus1 => "krea2-20plus1",
            MrFlowPreset::Krea2Turbo8Plus1 => "krea2-turbo-8plus1",
        }
    }
}

/// Round a low-resolution dimension to the nearest multiple of 16 (min 16), so
/// the stage-1 latent grid is valid. Mirrors the Rebels preset node.
fn mrflow_round16(value: f64) -> u32 {
    let snapped = (value / 16.0).round() * 16.0;
    (snapped as u32).max(16)
}

#[derive(Debug, Args)]
pub struct DiffusionImg2ImgArgs {
    /// Diffusion .hfq artifact to run by name, shorthand, alias, or path
    #[arg(long, short)]
    pub model: PathBuf,
    /// Prompt text. Repeat for batched generation, or use --batch-size with one prompt.
    #[arg(long, short, required = true)]
    pub prompt: Vec<String>,
    /// Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt.
    #[arg(long)]
    pub negative_prompt: Vec<String>,
    /// Input image path. Repeat for an image batch, or pass once to reuse across prompts.
    #[arg(long, required = true)]
    pub init_image: Vec<PathBuf>,
    /// Optional mask image path for inpaint-capable artifacts.
    #[arg(long)]
    pub mask: Option<PathBuf>,
    /// Output PNG file for one image, or output directory for batches
    #[arg(long, short)]
    pub output: PathBuf,
    /// Output image width in pixels. Defaults to the init image width.
    #[arg(long)]
    pub width: Option<u32>,
    /// Output image height in pixels. Defaults to the init image height.
    #[arg(long)]
    pub height: Option<u32>,
    /// Denoising steps
    #[arg(long, default_value_t = 20)]
    pub steps: u32,
    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 7.0)]
    pub cfg_scale: f32,
    /// Guidance-distilled model scale, separate from classifier-free guidance
    #[arg(long)]
    pub distilled_guidance_scale: Option<f32>,
    /// Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, DPM++ 2M Karras, or DPM++ 3M Karras
    #[arg(long, default_value = "Automatic")]
    pub scheduler: String,
    /// Seed. Omit for zero, pass once to reuse, or repeat per prompt.
    #[arg(long)]
    pub seed: Vec<i64>,
    /// Optional subseed. Pass once to reuse or repeat per prompt.
    #[arg(long)]
    pub subseed: Vec<i64>,
    /// Blend strength for subseed latents
    #[arg(long, default_value_t = 0.0)]
    pub subseed_strength: f32,
    /// Batch size when a single prompt is supplied
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,
    /// Img2img denoising strength in [0, 1]
    #[arg(long, default_value_t = 0.75)]
    pub denoising_strength: f32,
    /// Use ROCm for currently GPU-routed generation stages on this device id
    /// ROCm device to generate on. Omit to auto-detect (a single GPU is used
    /// silently; the first of several with a warning). The CPU reference oracle
    /// is opt-in via the HIPFIRE_DIFFUSION_CPU_REFERENCE environment variable.
    #[arg(long)]
    pub rocm_device_id: Option<i32>,
}

#[derive(Debug, Args)]
pub struct DiffusionSmokeArgs {
    /// Diffusion .hfq artifact to run by name, shorthand, alias, or path
    #[arg(long, short)]
    pub model: PathBuf,
    /// Prompt text for the smoke run
    #[arg(long, short, default_value = "hipfire diffusion smoke test")]
    pub prompt: String,
    /// Negative prompt text
    #[arg(long, default_value = "")]
    pub negative_prompt: String,
    /// Output directory for smoke PNGs
    #[arg(long, default_value = "/tmp/hipfire-diffusion-smoke")]
    pub output_dir: PathBuf,
    /// Output image width in pixels
    #[arg(long, default_value_t = 64)]
    pub width: u32,
    /// Output image height in pixels
    #[arg(long, default_value_t = 64)]
    pub height: u32,
    /// Denoising steps
    #[arg(long, default_value_t = 1)]
    pub steps: u32,
    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 1.0)]
    pub cfg_scale: f32,
    /// Guidance-distilled model scale, separate from classifier-free guidance
    #[arg(long)]
    pub distilled_guidance_scale: Option<f32>,
    /// Scheduler/sampler name
    #[arg(long, default_value = "Euler")]
    pub scheduler: String,
    /// Seed
    #[arg(long, default_value_t = 0)]
    pub seed: i64,
    /// Batch size for each smoke leg
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,
    /// Img2img denoising strength
    #[arg(long, default_value_t = 0.5)]
    pub denoising_strength: f32,
    /// Use ROCm for currently GPU-routed generation stages on this device id
    /// ROCm device to generate on. Omit to auto-detect (a single GPU is used
    /// silently; the first of several with a warning). The CPU reference oracle
    /// is opt-in via the HIPFIRE_DIFFUSION_CPU_REFERENCE environment variable.
    #[arg(long)]
    pub rocm_device_id: Option<i32>,
    /// Only run txt2img; skip the img2img leg
    #[arg(long)]
    pub txt2img_only: bool,
    /// Skip the masked img2img leg
    #[arg(long)]
    pub skip_masked_img2img: bool,
}

pub fn run(args: DiffusionArgs) -> anyhow::Result<()> {
    match args.command {
        DiffusionCommand::Import(args) => {
            let summary = import_diffusers_to_hfq(DiffusersImportOptions {
                source: args.source,
                output: args.output,
                model_name: args.model_name,
                max_batch: args.max_batch,
                metadata_only: args.metadata_only,
            })?;
            let inspection = inspect_hfq_with_runtime_support(summary.path)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&inspection_json(inspection))?
            );
            Ok(())
        }
        DiffusionCommand::Inspect(args) => {
            let inspection = inspect_hfq_with_runtime_support(resolve_model_path(args.model))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&inspection_json(inspection))?
            );
            Ok(())
        }
        DiffusionCommand::Preflight(args) => run_preflight(args),
        DiffusionCommand::Txt2Img(args) => run_txt2img(args),
        DiffusionCommand::Img2Img(args) => run_img2img(args),
        DiffusionCommand::Smoke(args) => run_smoke(args),
        DiffusionCommand::Quantize(args) => run_quantize(args),
        DiffusionCommand::Calibrate(args) => run_calibrate(args),
    }
}

fn run_calibrate(args: DiffusionCalibrateArgs) -> anyhow::Result<()> {
    let prompts = if args.prompts.is_empty() {
        vec![
            "a photograph of an astronaut riding a horse".to_string(),
            "portrait photo of a man, detailed face, studio lighting".to_string(),
            "a landscape painting of mountains at sunset".to_string(),
        ]
    } else {
        args.prompts.clone()
    };
    let summary = calibrate_diffusion_hfq(
        &args.model,
        &args.output,
        &prompts,
        args.steps,
        args.width,
        args.height,
        args.cfg_scale,
        args.hessian_max_k,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model": args.model,
            "output": args.output,
            "prompts": prompts.len(),
            "steps": args.steps,
            "observed_tensors": summary.observed_tensors,
            "hessians": summary.hessians,
            "imatrices": summary.imatrices,
        }))?
    );
    Ok(())
}

fn run_quantize(args: DiffusionQuantizeArgs) -> anyhow::Result<()> {
    let format = DiffusionQuantFormat::parse(&args.format).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown quant format {:?}; expected one of: q8, q4, q4k, q4+, oq4, oq4++, oq8",
            args.format
        )
    })?;
    let calib = match &args.calib {
        Some(path) => Some(
            hipfire_diffusion::open_calib_sidecar(path)
                .map_err(|e| anyhow::anyhow!("open calib {path:?}: {e}"))?,
        ),
        None => None,
    };
    let summary = quantize_diffusion_hfq(&args.source, &args.output, format, calib.as_ref())?;
    let ratio = if summary.output_bytes > 0 {
        summary.source_bytes as f64 / summary.output_bytes as f64
    } else {
        0.0
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "source": args.source,
            "output": args.output,
            "format": args.format,
            "quantized_tensors": summary.quantized_tensors,
            "copied_tensors": summary.copied_tensors,
            "ldlq_tensors": summary.ldlq_tensors,
            "source_bytes": summary.source_bytes,
            "output_bytes": summary.output_bytes,
            "compression_ratio": (ratio * 100.0).round() / 100.0,
        }))?
    );
    Ok(())
}

fn inspection_json(inspection: DiffusionHfqInspection) -> serde_json::Value {
    let summary = inspection.summary;
    serde_json::json!({
        "path": summary.path,
        "title": summary.title,
        "model_name": summary.model_name,
        "pipeline_class": summary.pipeline_class,
        "max_batch": summary.max_batch,
        "weight_format": summary.weight_format,
        "runtime_support": {
            "metadata_supported": inspection.runtime_support.supported,
            "runtime": inspection.runtime_support.runtime_kind.map(|kind| kind.as_str().to_string()),
            "reason": inspection.runtime_support.reason,
        },
    })
}

fn run_preflight(args: DiffusionPreflightArgs) -> anyhow::Result<()> {
    let prompts = build_diffusion_prompts(
        &args.prompt,
        &args.negative_prompt,
        &args.seed,
        &args.subseed,
        args.batch_size,
    )?;
    let request = DiffusionBatchRequest {
        prompts,
        conditioning: None,
        width: args.width,
        height: args.height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: args.steps,
        cfg_scale: args.cfg_scale,
        distilled_guidance_scale: args.distilled_guidance_scale,
        scheduler: args.scheduler.clone(),
        subseed_strength: args.subseed_strength,
        send_images: false,
        save_images: false,
    };
    let model = resolve_model_path(args.model);
    let pipeline = DiffusionPipeline::open_hfq(&model)?;
    let memory_plan = pipeline.hip_memory_plan(&request)?;
    let rocm = match pipeline.preflight_hip_runtime(
        &request,
        DiffusionHipRuntimeOptions {
            device_id: args.device_id,
        },
    ) {
        Ok(preflight) => serde_json::json!({
            "available": true,
            "preflight": preflight,
        }),
        Err(error) => serde_json::json!({
            "available": false,
            "reason": error.to_string(),
        }),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "pass",
            "model": pipeline.summary().model_name,
            "pipeline": pipeline.summary().pipeline_class,
            "device_id": args.device_id,
            "memory_plan": memory_plan,
            "rocm": rocm,
        }))?
    );
    Ok(())
}

fn run_txt2img(args: DiffusionTxt2ImgArgs) -> anyhow::Result<()> {
    let prompts = build_diffusion_prompts(
        &args.prompt,
        &args.negative_prompt,
        &args.seed,
        &args.subseed,
        args.batch_size,
    )?;
    let request = DiffusionBatchRequest {
        prompts,
        conditioning: None,
        width: args.width,
        height: args.height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: args.steps,
        cfg_scale: args.cfg_scale,
        distilled_guidance_scale: args.distilled_guidance_scale,
        scheduler: args.scheduler.clone(),
        subseed_strength: args.subseed_strength,
        send_images: true,
        save_images: false,
    };
    let model = resolve_model_path(args.model.clone());
    let pipeline = DiffusionPipeline::open_hfq(&model)?;
    let runtime_options = resolve_runtime_options(args.rocm_device_id)?;
    let batch_images = request.prompts.len();
    let wall_start = Instant::now();
    let output = if let Some(preset) = args.mrflow {
        generate_mrflow_txt2img(&pipeline, request, &args, preset, runtime_options)?
    } else if args.enable_hr {
        generate_highres_txt2img(&pipeline, request, &args, runtime_options)?
    } else if let Some(preview_dir) = args.preview_dir.clone() {
        // Per-pass previews: decode preview_latents after each denoise step and
        // write step_NN.png. Mirrors the webui hook (DiffusionProgress carries
        // the intermediate latent; the pipeline decodes it to a PNG). Batches
        // would collide on one file per step, so restrict to single-image runs.
        if batch_images != 1 {
            anyhow::bail!("--preview-dir is only supported for single-image runs (batch size 1)");
        }
        fs::create_dir_all(&preview_dir)?;
        let mut timing = step_timing_progress("txt2img", batch_images);
        let mut progress = |progress: DiffusionProgress| -> DiffusionResult<()> {
            if let Some(latents) = progress.preview_latents.as_ref() {
                let step = progress.completed_steps;
                let b64 = pipeline.decode_preview_latents_png_base64_with_runtime_options(
                    latents,
                    runtime_options,
                )?;
                let bytes = decode_base64_png(&b64).map_err(|e| {
                    DiffusionError::InvalidRequest(format!("preview PNG decode failed: {e}"))
                })?;
                let path = preview_dir.join(format!("step_{step:02}.png"));
                fs::write(&path, bytes).map_err(|e| {
                    DiffusionError::InvalidRequest(format!(
                        "failed to write preview {}: {e}",
                        path.display()
                    ))
                })?;
            }
            timing(progress)
        };
        pipeline.generate_batch_with_progress_and_runtime_options(
            request,
            runtime_options,
            &mut progress,
        )?
    } else {
        let mut progress = step_timing_progress("txt2img", batch_images);
        pipeline.generate_batch_with_progress_and_runtime_options(
            request,
            runtime_options,
            &mut progress,
        )?
    };
    let wall = wall_start.elapsed().as_secs_f64();
    eprintln!(
        "[txt2img] generated {batch_images} image(s) in {wall:.1}s ({:.2}s/image, includes text-encode + VAE decode)",
        wall / batch_images.max(1) as f64,
    );
    let files = write_png_images(&output.images, &args.output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model": pipeline.summary().model_name,
            "pipeline": pipeline.summary().pipeline_class,
            "images": files,
            "info": output.info,
        }))?
    );
    Ok(())
}

/// MrFlow staged sampling: fast low-resolution generate, pixel-space
/// super-resolution to the target size, re-encode, then a short direct-sigma
/// refine. Reuses the same decode/upscale/img2img plumbing as the high-res
/// path, but the second pass runs the flow-match direct-sigma refine schedule
/// instead of a strength-based img2img.
///
/// The pixel-space super-resolution here is a placeholder cover-resize; a
/// native SR model (RealESRGAN-class) is the intended Stage-2 upgrade. The
/// staging, re-encode, noise injection, and refine schedule are otherwise the
/// production path.
fn generate_mrflow_txt2img(
    pipeline: &DiffusionPipeline,
    mut first_pass_request: DiffusionBatchRequest,
    args: &DiffusionTxt2ImgArgs,
    preset: MrFlowPreset,
    runtime_options: DiffusionGenerationRuntimeOptions,
) -> anyhow::Result<hipfire_diffusion::DiffusionBatchOutput> {
    let params = preset.params();
    let target_width = args.width;
    let target_height = args.height;
    if target_width == 0 || target_height == 0 {
        anyhow::bail!("MrFlow requires non-zero --width and --height (the final resolution)");
    }
    let upscale = args.mrflow_upscale.unwrap_or(params.upscale_factor);
    if !upscale.is_finite() || upscale <= 1.0 {
        anyhow::bail!("--mrflow-upscale {upscale} must be greater than 1");
    }
    let refine_sigma = args.mrflow_refine_sigma.unwrap_or(params.refine_sigma);
    if !refine_sigma.is_finite() || !(0.0 < refine_sigma && refine_sigma < 1.0) {
        anyhow::bail!("--mrflow-refine-sigma {refine_sigma} must be in (0, 1)");
    }
    let low_width = mrflow_round16(target_width as f64 / upscale);
    let low_height = mrflow_round16(target_height as f64 / upscale);
    let batch_images = first_pass_request.prompts.len();

    // Stage 1: fast low-resolution generate with the preset step count and CFG.
    first_pass_request.width = low_width;
    first_pass_request.height = low_height;
    first_pass_request.steps = params.stage1_steps;
    first_pass_request.cfg_scale = params.cfg_scale;
    first_pass_request.send_images = true;
    let mut stage1_progress = step_timing_progress("mrflow-stage1", batch_images);
    let first_pass = pipeline.generate_batch_with_progress_and_runtime_options(
        first_pass_request.clone(),
        runtime_options,
        &mut stage1_progress,
    )?;

    // Stage 2: decode + pixel-space upscale to the target resolution. With a
    // RealESRGAN model, run it (by its native factor) then cover-resize to the
    // exact target; otherwise a plain cover-resize placeholder.
    let decoded = decode_png_images_to_rgb_batch(&first_pass.images)?;
    let (upscaled, sr_label) = if let Some(sr_path) = args.mrflow_sr.as_ref() {
        let sr_model = hipfire_diffusion::DiffusionSuperResModel::open_hfq(&resolve_model_path(
            sr_path.clone(),
        ))?;
        let sr_start = Instant::now();
        let sr_upscaled = sr_model.upscale_rgb_batch(&decoded, args.rocm_device_id)?;
        eprintln!(
            "[mrflow-superres] RealESRGAN x{} on {batch_images} image(s) in {:.1}s",
            sr_model.scale(),
            sr_start.elapsed().as_secs_f64(),
        );
        let resized = resize_rgb_batch_to_cover_nearest(&sr_upscaled, target_width, target_height)?;
        (resized, format!("realesrgan-x{}", sr_model.scale()))
    } else {
        (
            resize_rgb_batch_to_cover_nearest(&decoded, target_width, target_height)?,
            "nearest-cover (placeholder)".to_string(),
        )
    };

    // Stage 3+4: re-encode the upscaled image and run the direct-sigma refine.
    let mut refine_batch = first_pass_request;
    refine_batch.width = target_width;
    refine_batch.height = target_height;
    refine_batch.steps = params.refine_steps.max(1);
    refine_batch.send_images = true;
    let mut refine_progress = step_timing_progress("mrflow-refine", batch_images);
    let mut output = pipeline.generate_img2img_batch_with_progress_and_runtime_options(
        DiffusionImg2ImgRequest {
            batch: refine_batch,
            init_image: upscaled,
            mask: None,
            inpainting_fill: None,
            resize_mode: Default::default(),
            // Ignored when refine_sigma is set; the direct-sigma schedule drives
            // the refine.
            denoising_strength: 1.0,
            refine_sigma: Some(RefineSigmaSchedule {
                first_sigma: refine_sigma,
                steps: params.refine_steps.max(1),
                shifted: args.mrflow_shifted,
            }),
        },
        runtime_options,
        &mut refine_progress,
    )?;
    if let Some(map) = output.info.as_object_mut() {
        map.insert("mode".to_string(), serde_json::json!("txt2img-mrflow"));
        map.insert(
            "mrflow_preset".to_string(),
            serde_json::json!(preset.as_str()),
        );
        map.insert("stage1_width".to_string(), serde_json::json!(low_width));
        map.insert("stage1_height".to_string(), serde_json::json!(low_height));
        map.insert(
            "stage1_steps".to_string(),
            serde_json::json!(params.stage1_steps),
        );
        map.insert("refine_sigma".to_string(), serde_json::json!(refine_sigma));
        map.insert(
            "refine_steps".to_string(),
            serde_json::json!(params.refine_steps.max(1)),
        );
        map.insert("upscale_factor".to_string(), serde_json::json!(upscale));
        map.insert("target_width".to_string(), serde_json::json!(target_width));
        map.insert(
            "target_height".to_string(),
            serde_json::json!(target_height),
        );
        // Records the Stage-2 super-resolution path: the RealESRGAN model (by
        // native factor) or the cover-resize placeholder.
        map.insert("super_resolution".to_string(), serde_json::json!(sr_label));
    }
    Ok(output)
}

fn generate_highres_txt2img(
    pipeline: &DiffusionPipeline,
    mut first_pass_request: DiffusionBatchRequest,
    args: &DiffusionTxt2ImgArgs,
    runtime_options: DiffusionGenerationRuntimeOptions,
) -> anyhow::Result<hipfire_diffusion::DiffusionBatchOutput> {
    if !args.hr_denoising_strength.is_finite() || !(0.0..=1.0).contains(&args.hr_denoising_strength)
    {
        anyhow::bail!(
            "--hr-denoising-strength {} must be between 0 and 1",
            args.hr_denoising_strength
        );
    }
    let (firstpass_width, firstpass_height) = highres_first_pass_dimensions(
        args.width,
        args.height,
        args.firstphase_width,
        args.firstphase_height,
    )?;
    first_pass_request.width = firstpass_width;
    first_pass_request.height = firstpass_height;
    first_pass_request.send_images = true;
    let first_pass = pipeline
        .generate_batch_with_runtime_options(first_pass_request.clone(), runtime_options)?;
    let init_image = decode_png_images_to_rgb_batch(&first_pass.images)?;
    let (target_width, target_height) = highres_target_dimensions(
        first_pass_request.width,
        first_pass_request.height,
        args.hr_scale,
        args.hr_resize_x,
        args.hr_resize_y,
    )?;
    let init_image = highres_second_pass_init_image(
        init_image,
        target_width,
        target_height,
        args.hr_resize_x,
        args.hr_resize_y,
    )?;
    let mut second_pass_batch = first_pass_request;
    second_pass_batch.width = target_width;
    second_pass_batch.height = target_height;
    second_pass_batch.steps = args
        .hr_second_pass_steps
        .unwrap_or(second_pass_batch.steps)
        .max(1);
    second_pass_batch.send_images = true;
    let mut output = pipeline.generate_img2img_batch_with_runtime_options(
        DiffusionImg2ImgRequest {
            batch: second_pass_batch,
            init_image,
            mask: None,
            inpainting_fill: None,
            resize_mode: Default::default(),
            denoising_strength: args.hr_denoising_strength,
            refine_sigma: None,
        },
        runtime_options,
    )?;
    if let Some(map) = output.info.as_object_mut() {
        map.insert("mode".to_string(), serde_json::json!("txt2img-hires"));
        map.insert("highres".to_string(), serde_json::json!(true));
        map.insert(
            "firstpass_width".to_string(),
            serde_json::json!(firstpass_width),
        );
        map.insert(
            "firstpass_height".to_string(),
            serde_json::json!(firstpass_height),
        );
        map.insert("hr_width".to_string(), serde_json::json!(target_width));
        map.insert("hr_height".to_string(), serde_json::json!(target_height));
        map.insert(
            "hr_second_pass_steps".to_string(),
            serde_json::json!(args.hr_second_pass_steps.unwrap_or(args.steps).max(1)),
        );
    }
    Ok(output)
}

fn highres_second_pass_init_image(
    init_image: RgbImageBatch,
    target_width: u32,
    target_height: u32,
    hr_resize_x: Option<u32>,
    hr_resize_y: Option<u32>,
) -> anyhow::Result<RgbImageBatch> {
    if hr_resize_x.unwrap_or(0) > 0 && hr_resize_y.unwrap_or(0) > 0 {
        Ok(resize_rgb_batch_to_cover_nearest(
            &init_image,
            target_width,
            target_height,
        )?)
    } else {
        Ok(init_image)
    }
}

fn highres_first_pass_dimensions(
    base_width: u32,
    base_height: u32,
    firstphase_width: Option<u32>,
    firstphase_height: Option<u32>,
) -> anyhow::Result<(u32, u32)> {
    if base_width == 0 || base_height == 0 {
        anyhow::bail!("high-res txt2img requires non-zero base width and height");
    }
    match (
        firstphase_width.unwrap_or(0),
        firstphase_height.unwrap_or(0),
    ) {
        (0, 0) => Ok((base_width, base_height)),
        (width, height) if width > 0 && height > 0 => Ok((width, height)),
        (width, 0) => Ok((
            width,
            aspect_scaled_dimension(width, base_height, base_width, "first-pass height")?,
        )),
        (0, height) => Ok((
            aspect_scaled_dimension(height, base_width, base_height, "first-pass width")?,
            height,
        )),
        _ => unreachable!("zero firstphase dimensions are handled by earlier match arms"),
    }
}

fn highres_target_dimensions(
    base_width: u32,
    base_height: u32,
    hr_scale: f64,
    hr_resize_x: Option<u32>,
    hr_resize_y: Option<u32>,
) -> anyhow::Result<(u32, u32)> {
    if base_width == 0 || base_height == 0 {
        anyhow::bail!("high-res txt2img requires non-zero base width and height");
    }
    let resize_x = hr_resize_x.unwrap_or(0);
    let resize_y = hr_resize_y.unwrap_or(0);
    match (resize_x, resize_y) {
        (0, 0) => {
            if !hr_scale.is_finite() || hr_scale <= 0.0 {
                anyhow::bail!("--hr-scale must be positive and finite");
            }
            Ok((
                scaled_highres_dimension(base_width, hr_scale, "width")?,
                scaled_highres_dimension(base_height, hr_scale, "height")?,
            ))
        }
        (width, 0) => Ok((
            width,
            aspect_scaled_dimension(width, base_height, base_width, "height")?,
        )),
        (0, height) => Ok((
            aspect_scaled_dimension(height, base_width, base_height, "width")?,
            height,
        )),
        (width, height) => Ok((width, height)),
    }
}

fn scaled_highres_dimension(dimension: u32, scale: f64, label: &str) -> anyhow::Result<u32> {
    let scaled = (dimension as f64 * scale).round();
    if scaled < 1.0 || scaled > u32::MAX as f64 {
        anyhow::bail!("high-res target {label} is out of range");
    }
    Ok(scaled as u32)
}

fn aspect_scaled_dimension(
    fixed_dimension: u32,
    scaled_dimension: u32,
    base_dimension: u32,
    label: &str,
) -> anyhow::Result<u32> {
    let value = (fixed_dimension as u64)
        .checked_mul(scaled_dimension as u64)
        .ok_or_else(|| anyhow::anyhow!("high-res target {label} is out of range"))?
        .checked_div(base_dimension as u64)
        .unwrap_or(0)
        .max(1);
    u32::try_from(value).map_err(|_| anyhow::anyhow!("high-res target {label} is out of range"))
}

fn run_img2img(args: DiffusionImg2ImgArgs) -> anyhow::Result<()> {
    let init_image = load_rgb_image_batch(&args.init_image)?;
    let prompt_batch_size = args.batch_size.max(init_image.batch);
    let prompts = build_diffusion_prompts(
        &args.prompt,
        &args.negative_prompt,
        &args.seed,
        &args.subseed,
        prompt_batch_size,
    )?;
    let width = args
        .width
        .unwrap_or_else(|| u32::try_from(init_image.width).unwrap_or(u32::MAX));
    let height = args
        .height
        .unwrap_or_else(|| u32::try_from(init_image.height).unwrap_or(u32::MAX));
    let mask = args
        .mask
        .as_ref()
        .map(|path| load_rgb_image(path))
        .transpose()?;
    let request = DiffusionImg2ImgRequest {
        batch: DiffusionBatchRequest {
            prompts,
            conditioning: None,
            width,
            height,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            seed_resize_from_width: None,
            seed_resize_from_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: args.steps,
            cfg_scale: args.cfg_scale,
            distilled_guidance_scale: args.distilled_guidance_scale,
            scheduler: args.scheduler,
            subseed_strength: args.subseed_strength,
            send_images: true,
            save_images: false,
        },
        init_image,
        mask,
        inpainting_fill: None,
        resize_mode: Default::default(),
        denoising_strength: args.denoising_strength,
        refine_sigma: None,
    };
    let model = resolve_model_path(args.model);
    let pipeline = DiffusionPipeline::open_hfq(&model)?;
    let batch_images = request.batch.prompts.len();
    let runtime_options = resolve_runtime_options(args.rocm_device_id)?;
    let mut progress = step_timing_progress("img2img", batch_images);
    let output = pipeline.generate_img2img_batch_with_progress_and_runtime_options(
        request,
        runtime_options,
        &mut progress,
    )?;
    let files = write_png_images(&output.images, &args.output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model": pipeline.summary().model_name,
            "pipeline": pipeline.summary().pipeline_class,
            "images": files,
            "info": output.info,
        }))?
    );
    Ok(())
}

/// Resolve generation runtime options. hipfire is HIP/ROCm-first: the GPU is the
/// default and is resolved here (explicit `--rocm-device-id` wins; otherwise the
/// single visible device is used silently, or the first of several with a
/// warning). The CPU reference oracle is opt-in via `HIPFIRE_DIFFUSION_CPU_REFERENCE`.
fn resolve_runtime_options(
    explicit_device: Option<i32>,
) -> anyhow::Result<DiffusionGenerationRuntimeOptions> {
    if DiffusionGenerationRuntimeOptions::cpu_reference_requested() {
        return Ok(DiffusionGenerationRuntimeOptions::cpu_reference());
    }
    let device =
        hipfire_runtime::multi_gpu::resolve_primary_device(explicit_device).map_err(|error| {
            anyhow::anyhow!(
                "failed to resolve a ROCm device: {error}; pass --rocm-device-id, or set \
                 HIPFIRE_DIFFUSION_CPU_REFERENCE=1 to use the slow CPU reference oracle"
            )
        })?;
    Ok(DiffusionGenerationRuntimeOptions::rocm_hybrid(device))
}

fn resolve_model_path(path: PathBuf) -> PathBuf {
    if path.exists() {
        return path;
    }
    path.to_str().and_then(find_model).unwrap_or(path)
}

fn run_smoke(args: DiffusionSmokeArgs) -> anyhow::Result<()> {
    let model = resolve_model_path(args.model.clone());
    let inspection = inspect_hfq_with_runtime_support(&model)?;
    if !inspection.runtime_support.supported {
        let reason = inspection
            .runtime_support
            .reason
            .unwrap_or_else(|| "runtime support unavailable".to_string());
        anyhow::bail!("diffusion smoke requires a runnable artifact: {reason}");
    }
    fs::create_dir_all(&args.output_dir)?;
    let pipeline = DiffusionPipeline::open_hfq(&model)?;
    let runtime_options = resolve_runtime_options(args.rocm_device_id)?;
    let txt2img_request = smoke_batch_request(&args, args.seed);
    let txt2img_output =
        pipeline.generate_batch_with_runtime_options(txt2img_request, runtime_options)?;
    let txt2img_files = write_png_images(&txt2img_output.images, &args.output_dir.join("txt2img"))?;
    let txt2img_validation = validate_png_files(&txt2img_files, args.width, args.height)?;

    let (img2img_report, masked_img2img_report) = if args.txt2img_only {
        (None, None)
    } else {
        let init_image = load_rgb_image_batch(&txt2img_files)?;
        let img2img_request = DiffusionImg2ImgRequest {
            batch: smoke_batch_request(&args, args.seed.saturating_add(1)),
            init_image: init_image.clone(),
            mask: None,
            inpainting_fill: None,
            resize_mode: Default::default(),
            denoising_strength: args.denoising_strength,
            refine_sigma: None,
        };
        let img2img_output = pipeline
            .generate_img2img_batch_with_runtime_options(img2img_request, runtime_options)?;
        let img2img_files =
            write_png_images(&img2img_output.images, &args.output_dir.join("img2img"))?;
        let img2img_validation = validate_png_files(&img2img_files, args.width, args.height)?;
        let img2img_report = serde_json::json!({
            "images": img2img_files,
            "validated": img2img_validation,
            "info": img2img_output.info,
        });
        let masked_report = if args.skip_masked_img2img {
            None
        } else {
            let mask_path = args.output_dir.join("mask.png");
            let mask = write_smoke_mask_png(&mask_path, init_image.width, init_image.height)?;
            let masked_request = DiffusionImg2ImgRequest {
                batch: smoke_batch_request(&args, args.seed.saturating_add(2)),
                init_image,
                mask: Some(mask),
                inpainting_fill: None,
                resize_mode: Default::default(),
                denoising_strength: args.denoising_strength,
                refine_sigma: None,
            };
            let masked_output = pipeline
                .generate_img2img_batch_with_runtime_options(masked_request, runtime_options)?;
            let masked_files = write_png_images(
                &masked_output.images,
                &args.output_dir.join("masked-img2img"),
            )?;
            let masked_validation = validate_png_files(&masked_files, args.width, args.height)?;
            Some(serde_json::json!({
                "mask": mask_path,
                "images": masked_files,
                "validated": masked_validation,
                "info": masked_output.info,
            }))
        };
        (Some(img2img_report), masked_report)
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "status": "pass",
            "model": pipeline.summary().model_name,
            "pipeline": pipeline.summary().pipeline_class,
            "runtime": txt2img_output.info.get("runtime").cloned(),
            "metadata_runtime": inspection.runtime_support.runtime_kind.map(|kind| kind.as_str().to_string()),
            "runtime_options": {
                "rocm_device_id": runtime_options.rocm_device_id,
            },
            "txt2img": {
                "images": txt2img_files,
                "validated": txt2img_validation,
                "info": txt2img_output.info,
            },
            "img2img": img2img_report,
            "masked_img2img": masked_img2img_report,
        }))?
    );
    Ok(())
}

fn smoke_batch_request(args: &DiffusionSmokeArgs, seed: i64) -> DiffusionBatchRequest {
    let batch_size = args.batch_size.max(1);
    DiffusionBatchRequest {
        conditioning: None,

        prompts: (0..batch_size)
            .map(|idx| DiffusionPrompt {
                prompt: args.prompt.clone(),
                negative_prompt: args.negative_prompt.clone(),
                seed: seed.saturating_add(i64::try_from(idx).unwrap_or(i64::MAX)),
                subseed: None,
            })
            .collect(),
        width: args.width,
        height: args.height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        seed_resize_from_width: None,
        seed_resize_from_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: args.steps,
        cfg_scale: args.cfg_scale,
        distilled_guidance_scale: args.distilled_guidance_scale,
        scheduler: args.scheduler.clone(),
        subseed_strength: 0.0,
        send_images: true,
        save_images: false,
    }
}

fn build_diffusion_prompts(
    prompt: &[String],
    negative_prompt: &[String],
    seed: &[i64],
    subseed: &[i64],
    batch_size: usize,
) -> anyhow::Result<Vec<DiffusionPrompt>> {
    let batch_len = if prompt.len() == 1 {
        batch_size.max(1)
    } else {
        if batch_size != 1 && batch_size != prompt.len() {
            anyhow::bail!(
                "--batch-size {} does not match {} repeated --prompt values",
                batch_size,
                prompt.len()
            );
        }
        prompt.len()
    };
    let prompts = expand_strings(prompt, batch_len, "--prompt")?;
    let negative_prompts = expand_strings(negative_prompt, batch_len, "--negative-prompt")?;
    let seeds = expand_i64s(seed, batch_len, "--seed", 0)?;
    let subseeds = expand_optional_i64s(subseed, batch_len, "--subseed")?;
    Ok((0..batch_len)
        .map(|idx| DiffusionPrompt {
            prompt: prompts[idx].clone(),
            negative_prompt: negative_prompts[idx].clone(),
            seed: seeds[idx],
            subseed: subseeds[idx],
        })
        .collect())
}

fn expand_strings(values: &[String], batch_len: usize, flag: &str) -> anyhow::Result<Vec<String>> {
    match values.len() {
        0 => Ok(vec![String::new(); batch_len]),
        1 => Ok(vec![values[0].clone(); batch_len]),
        len if len == batch_len => Ok(values.to_vec()),
        len => anyhow::bail!("{flag} was provided {len} times but batch size is {batch_len}"),
    }
}

fn expand_i64s(
    values: &[i64],
    batch_len: usize,
    flag: &str,
    default: i64,
) -> anyhow::Result<Vec<i64>> {
    match values.len() {
        0 => Ok(vec![default; batch_len]),
        1 => Ok(vec![values[0]; batch_len]),
        len if len == batch_len => Ok(values.to_vec()),
        len => anyhow::bail!("{flag} was provided {len} times but batch size is {batch_len}"),
    }
}

fn expand_optional_i64s(
    values: &[i64],
    batch_len: usize,
    flag: &str,
) -> anyhow::Result<Vec<Option<i64>>> {
    match values.len() {
        0 => Ok(vec![None; batch_len]),
        1 => Ok(vec![Some(values[0]); batch_len]),
        len if len == batch_len => Ok(values.iter().copied().map(Some).collect()),
        len => anyhow::bail!("{flag} was provided {len} times but batch size is {batch_len}"),
    }
}

fn load_rgb_image_batch(paths: &[PathBuf]) -> anyhow::Result<RgbImageBatch> {
    if paths.is_empty() {
        anyhow::bail!("img2img requires at least one --init-image");
    }
    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        images.push(load_rgb_image(path)?);
    }
    let width = images[0].width;
    let height = images[0].height;
    let bytes_per_image = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("init image dimensions overflow"))?;
    let mut data = Vec::with_capacity(bytes_per_image * images.len());
    for (idx, image) in images.into_iter().enumerate() {
        if image.width != width || image.height != height {
            anyhow::bail!(
                "init image {idx} dimensions {}x{} do not match first init image {width}x{height}",
                image.width,
                image.height
            );
        }
        data.extend_from_slice(&image.data);
    }
    Ok(RgbImageBatch {
        batch: paths.len(),
        width,
        height,
        data,
    })
}

fn load_rgb_image(path: &Path) -> anyhow::Result<RgbImageBatch> {
    let bytes = fs::read(path)?;
    rgb_image_batch_from_bytes(&bytes, &format!("{path:?}"))
}

fn decode_png_images_to_rgb_batch(images: &[String]) -> anyhow::Result<RgbImageBatch> {
    if images.is_empty() {
        anyhow::bail!("high-res txt2img first pass returned no images");
    }
    let mut decoded = Vec::with_capacity(images.len());
    for (idx, image) in images.iter().enumerate() {
        let bytes = decode_base64_png(image)?;
        decoded.push(rgb_image_batch_from_bytes(
            &bytes,
            &format!("first-pass image {idx}"),
        )?);
    }
    let width = decoded[0].width;
    let height = decoded[0].height;
    let bytes_per_image = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("first-pass image dimensions overflow"))?;
    let mut data = Vec::with_capacity(bytes_per_image * decoded.len());
    for (idx, image) in decoded.into_iter().enumerate() {
        if image.width != width || image.height != height {
            anyhow::bail!(
                "first-pass image {idx} dimensions {}x{} do not match first image {width}x{height}",
                image.width,
                image.height
            );
        }
        data.extend_from_slice(&image.data);
    }
    Ok(RgbImageBatch {
        batch: images.len(),
        width,
        height,
        data,
    })
}

fn rgb_image_batch_from_bytes(bytes: &[u8], label: &str) -> anyhow::Result<RgbImageBatch> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| anyhow::anyhow!("invalid image {label}: {error}"))?
        .to_rgb8();
    let width = usize::try_from(image.width())?;
    let height = usize::try_from(image.height())?;
    Ok(RgbImageBatch {
        batch: 1,
        width,
        height,
        data: image.into_raw(),
    })
}

fn write_smoke_mask_png(path: &Path, width: usize, height: usize) -> anyhow::Result<RgbImageBatch> {
    let bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("smoke mask dimensions overflow"))?;
    let mut data = vec![0u8; bytes];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let value = if x >= width / 2 { 255 } else { 0 };
            data[idx] = value;
            data[idx + 1] = value;
            data[idx + 2] = value;
        }
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let image = image::RgbImage::from_raw(width as u32, height as u32, data.clone())
        .ok_or_else(|| anyhow::anyhow!("failed to build smoke mask image"))?;
    image.save(path)?;
    Ok(RgbImageBatch {
        batch: 1,
        width,
        height,
        data,
    })
}

fn write_png_images(images: &[String], output: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if images.is_empty() {
        anyhow::bail!("diffusion request returned no images; ensure send_images is enabled");
    }
    let output_is_file = images.len() == 1 && output.extension().is_some();
    if output_is_file {
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let bytes = decode_base64_png(&images[0])?;
        fs::write(output, bytes)?;
        return Ok(vec![output.to_path_buf()]);
    }
    fs::create_dir_all(output)?;
    let mut files = Vec::with_capacity(images.len());
    for (idx, image) in images.iter().enumerate() {
        let path = output.join(format!("{idx:05}.png"));
        let bytes = decode_base64_png(image)?;
        fs::write(&path, bytes)?;
        files.push(path);
    }
    Ok(files)
}

fn decode_base64_png(image: &str) -> anyhow::Result<Vec<u8>> {
    let payload = image
        .split_once(',')
        .filter(|(prefix, _)| prefix.starts_with("data:image/"))
        .map(|(_, payload)| payload)
        .unwrap_or(image);
    let bytes = base64::engine::general_purpose::STANDARD.decode(payload)?;
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        anyhow::bail!("diffusion output is not a PNG image");
    }
    Ok(bytes)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PngValidation {
    width: u32,
    height: u32,
    dimensions: String,
    unique_rgb_values: usize,
    min_rgb: u8,
    max_rgb: u8,
    luma_min: u8,
    luma_max: u8,
    luma_range: u8,
}

fn validate_png_files(
    files: &[PathBuf],
    width: u32,
    height: u32,
) -> anyhow::Result<Vec<PngValidation>> {
    files
        .iter()
        .map(|path| validate_png_file(path, width, height))
        .collect()
}

fn validate_png_file(path: &Path, width: u32, height: u32) -> anyhow::Result<PngValidation> {
    let bytes = fs::read(path)?;
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        anyhow::bail!("{path:?} is not a PNG image");
    }
    let image = image::load_from_memory(&bytes)
        .map_err(|error| anyhow::anyhow!("invalid generated PNG {:?}: {error}", path))?
        .to_rgb8();
    if image.width() != width || image.height() != height {
        anyhow::bail!(
            "{path:?} dimensions {}x{} do not match expected {width}x{height}",
            image.width(),
            image.height()
        );
    }
    let mut unique = std::collections::BTreeSet::new();
    let mut min_rgb = u8::MAX;
    let mut max_rgb = u8::MIN;
    let mut luma_min = u8::MAX;
    let mut luma_max = u8::MIN;
    for pixel in image.pixels() {
        let [r, g, b] = pixel.0;
        unique.insert([r, g, b]);
        min_rgb = min_rgb.min(r).min(g).min(b);
        max_rgb = max_rgb.max(r).max(g).max(b);
        let luma = ((u16::from(r) * 77 + u16::from(g) * 150 + u16::from(b) * 29) >> 8) as u8;
        luma_min = luma_min.min(luma);
        luma_max = luma_max.max(luma);
    }
    let luma_range = luma_max.saturating_sub(luma_min);
    if unique.len() < 2 || luma_range < 2 {
        anyhow::bail!(
            "{path:?} is visually degenerate: unique_rgb_values={}, luma_range={luma_range}",
            unique.len()
        );
    }
    Ok(PngValidation {
        width: image.width(),
        height: image.height(),
        dimensions: format!("{}x{}", image.width(), image.height()),
        unique_rgb_values: unique.len(),
        min_rgb,
        max_rgb,
        luma_min,
        luma_max,
        luma_range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;

    fn txt2img_args() -> DiffusionTxt2ImgArgs {
        DiffusionTxt2ImgArgs {
            model: PathBuf::from("model.hfq"),
            prompt: vec!["a cat".to_string()],
            negative_prompt: Vec::new(),
            output: PathBuf::from("out.png"),
            width: 64,
            height: 64,
            firstphase_width: None,
            firstphase_height: None,
            steps: 1,
            cfg_scale: 1.0,
            distilled_guidance_scale: None,
            scheduler: "Automatic".to_string(),
            seed: Vec::new(),
            subseed: Vec::new(),
            subseed_strength: 0.0,
            batch_size: 2,
            enable_hr: false,
            hr_scale: 2.0,
            hr_resize_x: None,
            hr_resize_y: None,
            hr_second_pass_steps: None,
            hr_denoising_strength: 0.75,
            rocm_device_id: None,
            preview_dir: None,
            mrflow: None,
            mrflow_refine_sigma: None,
            mrflow_upscale: None,
            mrflow_shifted: false,
            mrflow_sr: None,
        }
    }

    #[test]
    fn mrflow_round16_snaps_to_multiple_of_16_with_floor() {
        assert_eq!(mrflow_round16(512.0), 512);
        assert_eq!(mrflow_round16(1024.0 / 2.0), 512);
        // Rounds to nearest multiple of 16.
        assert_eq!(mrflow_round16(520.0), 528);
        assert_eq!(mrflow_round16(519.0), 512);
        // Never below 16 even for tiny targets.
        assert_eq!(mrflow_round16(1.0), 16);
    }

    #[test]
    fn mrflow_presets_match_reference_numbers() {
        let turbo = MrFlowPreset::Krea2Turbo8Plus1.params();
        assert_eq!(turbo.stage1_steps, 8);
        assert_eq!(turbo.refine_steps, 1);
        assert!((turbo.refine_sigma - 0.11).abs() < 1e-6);
        assert!((turbo.cfg_scale - 1.0).abs() < 1e-6);

        let base = MrFlowPreset::Krea2Base12Plus1.params();
        assert_eq!(base.stage1_steps, 12);
        assert!((base.refine_sigma - 0.12).abs() < 1e-6);
        assert!((base.cfg_scale - 4.0).abs() < 1e-6);

        assert_eq!(MrFlowPreset::Zit9Plus1.params().stage1_steps, 9);
        assert_eq!(MrFlowPreset::Krea2Base20Plus1.params().stage1_steps, 20);
    }

    fn smoke_args() -> DiffusionSmokeArgs {
        DiffusionSmokeArgs {
            model: PathBuf::from("model.hfq"),
            prompt: "a cat".to_string(),
            negative_prompt: "blur".to_string(),
            output_dir: PathBuf::from("out"),
            width: 64,
            height: 64,
            steps: 1,
            cfg_scale: 1.0,
            distilled_guidance_scale: None,
            scheduler: "Euler".to_string(),
            seed: 40,
            batch_size: 3,
            denoising_strength: 0.5,
            rocm_device_id: None,
            txt2img_only: false,
            skip_masked_img2img: false,
        }
    }

    #[test]
    fn smoke_batch_request_uses_batch_size_and_sequential_seeds() {
        let args = smoke_args();

        let request = smoke_batch_request(&args, args.seed);

        assert_eq!(request.prompts.len(), 3);
        assert_eq!(request.prompts[0].seed, 40);
        assert_eq!(request.prompts[1].seed, 41);
        assert_eq!(request.prompts[2].seed, 42);
        assert!(request
            .prompts
            .iter()
            .all(|prompt| prompt.prompt == "a cat"));
        assert!(request
            .prompts
            .iter()
            .all(|prompt| prompt.negative_prompt == "blur"));
        assert!(request.send_images);
    }

    #[test]
    fn txt2img_prompt_builder_repeats_single_prompt_for_batch() {
        let mut args = txt2img_args();
        args.negative_prompt = vec!["blur".to_string()];
        args.seed = vec![42];
        args.subseed = vec![7];

        let prompts = build_diffusion_prompts(
            &args.prompt,
            &args.negative_prompt,
            &args.seed,
            &args.subseed,
            args.batch_size,
        )
        .unwrap();

        assert_eq!(prompts.len(), 2);
        assert_eq!(prompts[0].prompt, "a cat");
        assert_eq!(prompts[1].negative_prompt, "blur");
        assert_eq!(prompts[0].seed, 42);
        assert_eq!(prompts[1].subseed, Some(7));
    }

    #[test]
    fn txt2img_prompt_builder_rejects_mismatched_repeated_fields() {
        let mut args = txt2img_args();
        args.prompt = vec!["a".to_string(), "b".to_string()];
        args.negative_prompt = vec!["x".to_string(), "y".to_string(), "z".to_string()];

        let error = build_diffusion_prompts(
            &args.prompt,
            &args.negative_prompt,
            &args.seed,
            &args.subseed,
            args.batch_size,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("--negative-prompt"));
    }

    #[test]
    fn rocm_hybrid_targets_requested_device() {
        // Device resolution (CPU env vs ROCm + GPU detection) is covered by the
        // diffusion and runtime crates; here we only pin the GPU mapping.
        assert_eq!(
            DiffusionGenerationRuntimeOptions::rocm_hybrid(2),
            DiffusionGenerationRuntimeOptions {
                rocm_device_id: Some(2)
            }
        );
    }

    #[test]
    fn decode_base64_png_accepts_plain_and_data_url_payloads() {
        let png = b"\x89PNG\r\n\x1a\npayload";
        let encoded = base64::engine::general_purpose::STANDARD.encode(png);
        assert_eq!(decode_base64_png(&encoded).unwrap(), png);
        assert_eq!(
            decode_base64_png(&format!("data:image/png;base64,{encoded}")).unwrap(),
            png
        );
    }

    #[test]
    fn highres_target_dimensions_support_scale_and_resize_modes() {
        assert_eq!(
            highres_target_dimensions(2, 3, 2.0, None, None).unwrap(),
            (4, 6)
        );
        assert_eq!(
            highres_target_dimensions(2, 3, 2.0, Some(8), None).unwrap(),
            (8, 12)
        );
        assert_eq!(
            highres_target_dimensions(2, 3, 2.0, None, Some(9)).unwrap(),
            (6, 9)
        );
        assert_eq!(
            highres_target_dimensions(2, 3, 2.0, Some(7), Some(5)).unwrap(),
            (7, 5)
        );
        assert!(highres_target_dimensions(2, 3, 0.0, None, None).is_err());
    }

    #[test]
    fn highres_first_pass_dimensions_support_firstphase_modes() {
        assert_eq!(
            highres_first_pass_dimensions(4, 2, None, None).unwrap(),
            (4, 2)
        );
        assert_eq!(
            highres_first_pass_dimensions(4, 2, Some(2), Some(2)).unwrap(),
            (2, 2)
        );
        assert_eq!(
            highres_first_pass_dimensions(4, 2, Some(8), None).unwrap(),
            (8, 4)
        );
        assert_eq!(
            highres_first_pass_dimensions(4, 2, None, Some(3)).unwrap(),
            (6, 3)
        );
        assert!(highres_first_pass_dimensions(0, 2, Some(8), None).is_err());
    }

    #[test]
    fn highres_second_pass_init_image_cover_crops_exact_resize() {
        let image = RgbImageBatch {
            batch: 1,
            width: 2,
            height: 4,
            data: vec![
                10, 10, 10, 10, 10, 10, //
                20, 20, 20, 20, 20, 20, //
                30, 30, 30, 30, 30, 30, //
                40, 40, 40, 40, 40, 40, //
            ],
        };

        let cropped =
            highres_second_pass_init_image(image.clone(), 4, 4, Some(4), Some(4)).unwrap();
        assert_eq!(cropped.width, 4);
        assert_eq!(cropped.height, 4);
        assert_eq!(&cropped.data[..12], &[20u8; 12]);
        assert_eq!(&cropped.data[24..36], &[30u8; 12]);

        let unchanged = highres_second_pass_init_image(image.clone(), 4, 8, Some(4), None).unwrap();
        assert_eq!(unchanged, image);
    }

    #[test]
    fn decode_png_images_to_rgb_batch_accepts_matching_first_pass_images() {
        let first = tiny_png_base64(2, 2, 16);
        let second = tiny_png_base64(2, 2, 128);

        let batch = decode_png_images_to_rgb_batch(&[first, second]).unwrap();

        assert_eq!(batch.batch, 2);
        assert_eq!(batch.width, 2);
        assert_eq!(batch.height, 2);
        assert_eq!(batch.data.len(), 24);
    }

    #[test]
    fn decode_png_images_to_rgb_batch_rejects_mismatched_first_pass_images() {
        let error = decode_png_images_to_rgb_batch(&[
            tiny_png_base64(2, 2, 16),
            tiny_png_base64(1, 2, 128),
        ])
        .unwrap_err()
        .to_string();

        assert!(error.contains("do not match first image"));
    }

    #[test]
    fn load_rgb_image_batch_decodes_png_and_rejects_shape_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-cli-image-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let one = dir.join("one.png");
        let two = dir.join("two.png");
        let bad = dir.join("bad.png");
        image::RgbImage::from_raw(2, 2, vec![255; 12])
            .unwrap()
            .save(&one)
            .unwrap();
        image::RgbImage::from_raw(2, 2, vec![127; 12])
            .unwrap()
            .save(&two)
            .unwrap();
        image::RgbImage::from_raw(1, 2, vec![0; 6])
            .unwrap()
            .save(&bad)
            .unwrap();

        let batch = load_rgb_image_batch(&[one.clone(), two]).unwrap();
        assert_eq!(batch.batch, 2);
        assert_eq!(batch.width, 2);
        assert_eq!(batch.height, 2);
        assert_eq!(batch.data.len(), 24);

        let error = load_rgb_image_batch(&[one, bad]).unwrap_err().to_string();
        assert!(error.contains("do not match"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_png_file_checks_dimensions() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-cli-validate-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let image = dir.join("image.png");
        image::RgbImage::from_raw(
            3,
            2,
            vec![
                0, 0, 0, 32, 32, 32, 64, 64, 64, 96, 96, 96, 128, 128, 128, 255, 255, 255,
            ],
        )
        .unwrap()
        .save(&image)
        .unwrap();

        let validation = validate_png_file(&image, 3, 2).unwrap();
        assert_eq!(validation.dimensions, "3x2");
        assert_eq!(validation.unique_rgb_values, 6);
        assert!(validation.luma_range >= 2);
        let error = validate_png_file(&image, 2, 2).unwrap_err().to_string();
        assert!(error.contains("do not match expected"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_png_file_rejects_degenerate_content() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-cli-degenerate-validate-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let image = dir.join("flat.png");
        image::RgbImage::from_raw(3, 2, vec![64; 18])
            .unwrap()
            .save(&image)
            .unwrap();

        let error = validate_png_file(&image, 3, 2).unwrap_err().to_string();
        assert!(error.contains("visually degenerate"));
        assert!(error.contains("unique_rgb_values=1"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_smoke_mask_png_creates_half_mask() {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-diffusion-cli-mask-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mask.png");

        let mask = write_smoke_mask_png(&path, 4, 2).unwrap();

        assert_eq!(mask.batch, 1);
        assert_eq!(mask.width, 4);
        assert_eq!(mask.height, 2);
        assert_eq!(&mask.data[0..6], &[0, 0, 0, 0, 0, 0]);
        assert_eq!(&mask.data[6..12], &[255, 255, 255, 255, 255, 255]);
        assert_eq!(validate_png_file(&path, 4, 2).unwrap().dimensions, "4x2");
        let _ = fs::remove_dir_all(&dir);
    }

    fn tiny_png_base64(width: u32, height: u32, value: u8) -> String {
        let bytes = (width as usize) * (height as usize) * 3;
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(
                &vec![value; bytes],
                width,
                height,
                image::ColorType::Rgb8.into(),
            )
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(png)
    }
}
