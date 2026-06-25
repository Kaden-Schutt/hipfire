// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use clap::{Args, Subcommand};
#[cfg(feature = "rocm")]
use hipfire_diffusion::DiffusionHipRuntimeOptions;
use hipfire_diffusion::{
    import_diffusers_to_hfq, inspect_hfq_with_runtime_support, DiffusersImportOptions,
    DiffusionBatchRequest, DiffusionGenerationRuntimeOptions, DiffusionHfqInspection,
    DiffusionImg2ImgRequest, DiffusionPipeline, DiffusionPrompt, RgbImageBatch,
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
    Preflight(DiffusionPreflightArgs),
    /// Generate PNG images directly from a diffusion .hfq artifact
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
        scheduler: args.scheduler,
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
        scheduler: args.scheduler,
        subseed_strength: args.subseed_strength,
        send_images: true,
        save_images: false,
    };
    let pipeline = DiffusionPipeline::open_hfq(&args.model)?;
    let output = pipeline.generate_batch_with_runtime_options(
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
    let txt2img_request = smoke_batch_request(&args, args.seed);
    let txt2img_output = pipeline.generate_batch(txt2img_request)?;
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
        let img2img_output = pipeline.generate_img2img_batch(img2img_request)?;
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
            let masked_output = pipeline.generate_img2img_batch(masked_request)?;
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
            "runtime": inspection.runtime_support.runtime_kind.map(|kind| kind.as_str().to_string()),
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
    let image = image::load_from_memory(&bytes)
        .map_err(|error| anyhow::anyhow!("invalid image {:?}: {error}", path))?
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

    fn txt2img_args() -> DiffusionTxt2ImgArgs {
        DiffusionTxt2ImgArgs {
            model: PathBuf::from("model.hfq"),
            prompt: vec!["a cat".to_string()],
            negative_prompt: Vec::new(),
            output: PathBuf::from("out.png"),
            width: 64,
            height: 64,
            steps: 1,
            cfg_scale: 1.0,
            scheduler: "Automatic".to_string(),
            seed: Vec::new(),
            subseed: Vec::new(),
            subseed_strength: 0.0,
            batch_size: 2,
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
}
