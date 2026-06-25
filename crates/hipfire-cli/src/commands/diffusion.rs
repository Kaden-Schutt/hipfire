// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use clap::{Args, Subcommand};
#[cfg(feature = "rocm")]
use hipfire_diffusion::DiffusionHipRuntimeOptions;
use hipfire_diffusion::{
    import_diffusers_to_hfq, inspect_hfq_with_runtime_support, resize_rgb_batch_to_cover_nearest,
    DiffusersImportOptions, DiffusionBatchRequest, DiffusionGenerationRuntimeOptions,
    DiffusionHfqInspection, DiffusionImg2ImgRequest, DiffusionPipeline, DiffusionPrompt,
    RgbImageBatch,
};
use serde::Serialize;

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
    /// requested resolution, batch, scheduler, and prompt set. Builds compiled
    /// with `--features rocm` also initialize the selected HIP device, allocate
    /// the planned buffer classes, run a host/device roundtrip probe, and
    /// launch currently covered diffusion kernel probes against CPU references.
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
    /// Diffusion .hfq artifact to inspect
    pub model: PathBuf,
}

#[derive(Debug, Args)]
pub struct DiffusionPreflightArgs {
    /// Diffusion .hfq artifact to inspect
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
    /// ROCm device id to preflight when built with --features rocm
    #[arg(long, default_value_t = 0)]
    pub device_id: i32,
}

#[derive(Debug, Args)]
pub struct DiffusionTxt2ImgArgs {
    /// Diffusion .hfq artifact to run
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
    /// Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, or DPM++ 2M Karras
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
    #[arg(long)]
    pub rocm_device_id: Option<i32>,
}

#[derive(Debug, Args)]
pub struct DiffusionImg2ImgArgs {
    /// Diffusion .hfq artifact to run
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
    /// Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, or DPM++ 2M Karras
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
    #[arg(long)]
    pub rocm_device_id: Option<i32>,
}

#[derive(Debug, Args)]
pub struct DiffusionSmokeArgs {
    /// Diffusion .hfq artifact to run
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
    /// Scheduler/sampler name
    #[arg(long, default_value = "Euler")]
    pub scheduler: String,
    /// Seed
    #[arg(long, default_value_t = 0)]
    pub seed: i64,
    /// Img2img denoising strength
    #[arg(long, default_value_t = 0.5)]
    pub denoising_strength: f32,
    /// Use ROCm for currently GPU-routed generation stages on this device id
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
            let inspection = inspect_hfq_with_runtime_support(args.model)?;
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
    }
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
        width: args.width,
        height: args.height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: args.steps,
        cfg_scale: args.cfg_scale,
        scheduler: args.scheduler.clone(),
        subseed_strength: args.subseed_strength,
        send_images: false,
        save_images: false,
    };
    let pipeline = DiffusionPipeline::open_hfq(&args.model)?;
    let memory_plan = pipeline.hip_memory_plan(&request)?;
    #[cfg(feature = "rocm")]
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
    #[cfg(not(feature = "rocm"))]
    let rocm = serde_json::json!({
        "available": false,
        "reason": "hipfire-cli was built without the rocm feature; rebuild with --features rocm to run device allocation preflight",
    });
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
        width: args.width,
        height: args.height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: args.steps,
        cfg_scale: args.cfg_scale,
        scheduler: args.scheduler.clone(),
        subseed_strength: args.subseed_strength,
        send_images: true,
        save_images: false,
    };
    let pipeline = DiffusionPipeline::open_hfq(&args.model)?;
    let runtime_options = generation_runtime_options(args.rocm_device_id);
    let output = if args.enable_hr {
        generate_highres_txt2img(&pipeline, request, &args, runtime_options)?
    } else {
        pipeline.generate_batch_with_runtime_options(request, runtime_options)?
    };
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
            denoising_strength: args.hr_denoising_strength,
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
            width,
            height,
            original_width: None,
            original_height: None,
            target_width: None,
            target_height: None,
            crop_x: 0,
            crop_y: 0,
            steps: args.steps,
            cfg_scale: args.cfg_scale,
            scheduler: args.scheduler,
            subseed_strength: args.subseed_strength,
            send_images: true,
            save_images: false,
        },
        init_image,
        mask,
        denoising_strength: args.denoising_strength,
    };
    let pipeline = DiffusionPipeline::open_hfq(&args.model)?;
    let output = pipeline.generate_img2img_batch_with_runtime_options(
        request,
        generation_runtime_options(args.rocm_device_id),
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

fn generation_runtime_options(rocm_device_id: Option<i32>) -> DiffusionGenerationRuntimeOptions {
    rocm_device_id.map_or_else(
        DiffusionGenerationRuntimeOptions::cpu_reference,
        DiffusionGenerationRuntimeOptions::rocm_hybrid,
    )
}

fn run_smoke(args: DiffusionSmokeArgs) -> anyhow::Result<()> {
    let inspection = inspect_hfq_with_runtime_support(&args.model)?;
    if !inspection.runtime_support.supported {
        let reason = inspection
            .runtime_support
            .reason
            .unwrap_or_else(|| "runtime support unavailable".to_string());
        anyhow::bail!("diffusion smoke requires a runnable artifact: {reason}");
    }
    fs::create_dir_all(&args.output_dir)?;
    let pipeline = DiffusionPipeline::open_hfq(&args.model)?;
    let runtime_options = generation_runtime_options(args.rocm_device_id);
    let txt2img_request = smoke_batch_request(&args, args.seed);
    let txt2img_output =
        pipeline.generate_batch_with_runtime_options(txt2img_request, runtime_options)?;
    let txt2img_files = write_png_images(&txt2img_output.images, &args.output_dir.join("txt2img"))?;
    let txt2img_validation = validate_png_files(&txt2img_files, args.width, args.height)?;

    let (img2img_report, masked_img2img_report) = if args.txt2img_only {
        (None, None)
    } else {
        let init_image = load_rgb_image_batch(&[txt2img_files[0].clone()])?;
        let img2img_request = DiffusionImg2ImgRequest {
            batch: smoke_batch_request(&args, args.seed.saturating_add(1)),
            init_image: init_image.clone(),
            mask: None,
            denoising_strength: args.denoising_strength,
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
                denoising_strength: args.denoising_strength,
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
    DiffusionBatchRequest {
        prompts: vec![DiffusionPrompt {
            prompt: args.prompt.clone(),
            negative_prompt: args.negative_prompt.clone(),
            seed,
            subseed: None,
        }],
        width: args.width,
        height: args.height,
        original_width: None,
        original_height: None,
        target_width: None,
        target_height: None,
        crop_x: 0,
        crop_y: 0,
        steps: args.steps,
        cfg_scale: args.cfg_scale,
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
        }
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
    fn generation_runtime_options_select_cpu_or_rocm() {
        assert_eq!(
            generation_runtime_options(None),
            DiffusionGenerationRuntimeOptions::cpu_reference()
        );
        assert_eq!(
            generation_runtime_options(Some(2)),
            DiffusionGenerationRuntimeOptions::rocm_hybrid(2)
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
